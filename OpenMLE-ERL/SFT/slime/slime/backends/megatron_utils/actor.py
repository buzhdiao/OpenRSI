import logging
import os
import random
import socket
from argparse import Namespace
from contextlib import nullcontext

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray.actor import ActorHandle
from torch_memory_saver import torch_memory_saver
from transformers import AutoConfig, AutoTokenizer

from slime.ray.train_actor import TrainRayActor
from slime.utils import train_dump_utils
from slime.utils.data import process_rollout_data
from slime.utils.distributed_utils import get_gloo_group, init_process_group
from slime.utils.logging_utils import init_tracking
from slime.utils.memory_utils import clear_memory, print_memory
from slime.utils.misc import Box
from slime.utils.reloadable_process_group import destroy_process_groups, monkey_patch_torch_dist, reload_process_groups
from slime.utils.routing_replay import RoutingReplay
from slime.utils.timer import Timer, inverse_timer, timer, with_defer
from slime.utils.types import RolloutBatch

from ...utils.profile_utils import TrainProfiler
from ...utils.tensor_backper import TensorBackuper
from .checkpoint import load_checkpoint
from .cp_utils import slice_log_prob_with_cp, slice_with_cp
from .data import DataIterator, get_data_iterator, log_perf_data, log_rollout_data, sync_actor_critic_data
from .initialize import init, is_megatron_main_rank
from .loss import compute_advantages_and_returns, get_log_probs_and_entropy, get_values
from .model import forward_only, initialize_model_and_optimizer, save, train
from .update_weight.common import named_params_and_buffers
from .update_weight.update_weight_from_distributed import UpdateWeightFromDistributed
from .update_weight.update_weight_from_tensor import UpdateWeightFromTensor

logging.getLogger("megatron").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class MegatronTrainRayActor(TrainRayActor):
    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
    ) -> int | None:
        # 给 torch.distributed 打补丁（支持可重载进程组等能力）
        monkey_patch_torch_dist()

        # 调用父类的通用初始化（连接 ray、设置角色等）
        super().init(args, role, with_ref, with_opd_teacher)

        # 仅调试 rollout 时不需要构建训练模型
        if args.debug_rollout_only:
            return 0

        # 初始化 Megatron 分布式环境（进程组、随机种子、tokenizer 等）
        init(args)

        # Megatron 主 rank（PP 最后阶段 TP0）初始化日志跟踪（wandb/tensorboard 等）
        if is_megatron_main_rank():
            init_tracking(args, primary=False)

        # 创建训练性能分析器
        self.prof = TrainProfiler(args)

        # read config and tokenizer serialized to prevent concurrent writing bug.
        # 串行加载 HF config/tokenizer：同一节点上各 GPU 进程轮流加载，避免并发写缓存损坏
        for i in range(args.num_gpus_per_node):
            if i == dist.get_rank() % args.num_gpus_per_node:
                self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
            # 每个 rank 加载完后同步一次
            dist.barrier(group=get_gloo_group())

        # 记录训练并行配置（数据并行大小，供 rollout 侧参考）
        self.train_parallel_config = {
            "dp_size": mpu.get_data_parallel_world_size(with_context_parallel=False),
        }
        # 全体同步，确保并行配置已就绪
        dist.barrier(group=get_gloo_group())

        if args.offload_train:
            # 配置显存余量（memory margin），配合 torch_memory_saver 控制卸载行为
            if (x := args.train_memory_margin_bytes) > 0:
                logger.info(f"Set torch_memory_saver.memory_margin_bytes to {x}")
                torch_memory_saver.memory_margin_bytes = x

        if role == "critic":
            # critic 角色复用 actor 的参数结构，但加载/保存/学习率等用 critic 自己的配置
            self.args.load = self.args.critic_load
            self.args.save = self.args.critic_save
            self.args.lr = self.args.critic_lr
            self.args.lr_warmup_iters = self.args.critic_lr_warmup_iters

        # 构建模型与优化器（会自动从 checkpoint 恢复，返回已加载的 rollout id）
        (self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id) = initialize_model_and_optimizer(
            args, role
        )

        if role == "critic":
            # critic 初始化到此为止；开启 offload 时立即休眠释放显存
            if self.args.offload_train:
                self.sleep()
            return

        # 下一次 rollout 从已加载的 checkpoint 之后开始
        start_rollout_id = loaded_rollout_id + 1

        # 权重备份器：可把当前 GPU 权重快照到 CPU，用于 actor/ref/teacher/old_actor 之间切换
        self.weights_backuper = TensorBackuper.create(
            source_getter=lambda: named_params_and_buffers(
                self.args,
                self.model,
                convert_to_global_name=args.megatron_to_hf_mode == "raw",
                translate_gpu_to_cpu=not self.args.enable_weights_backuper,
            ),
            single_tag=None if args.enable_weights_backuper else "actor",
        )
        # 当前激活的模型标签
        self._active_model_tag: str | None = "actor"
        # 备份初始 actor 权重
        self.weights_backuper.backup("actor")

        if with_ref:
            # 按需加载参考模型（ref model，用于 KL 约束）
            self.load_other_checkpoint("ref", args.ref_load)

        # Load teacher model for Megatron-based on-policy distillation
        # 按需加载教师模型（用于基于 Megatron 的在线策略蒸馏 OPD）
        if with_opd_teacher:
            self.load_other_checkpoint("teacher", args.opd_teacher_load)

        if self.args.keep_old_actor:
            # Load old_actor checkpoint
            # 加载旧策略（old actor）checkpoint，用于计算重要性采样比等
            self.load_other_checkpoint("old_actor", args.load)
            # Create rollout_actor as a copy of current actor
            # 每步都同步推理引擎权重时，额外保留一份 rollout_actor 副本
            if args.update_weights_interval == 1:
                self.weights_backuper.backup("rollout_actor")

        # 配置里未指定词表大小时，用 tokenizer 的词表大小
        if self.args.vocab_size is None:
            self.args.vocab_size = self.tokenizer.vocab_size

        # 共置（colocate）模式用张量方式同步权重，否则用分布式方式；
        # 推理引擎要的权重来自备份器中的 "actor" 快照
        update_weight_cls = UpdateWeightFromTensor if self.args.colocate else UpdateWeightFromDistributed
        self.weight_updater = update_weight_cls(
            self.args,
            self.model,
            weights_getter=lambda: self.weights_backuper.get("actor"),
            model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
        )

        # empty cache after initialization
        # 初始化完成后清空缓存，释放临时显存
        clear_memory()

        if self.args.offload_train:
            # recover to actor in the end.
            # offload 模式：切回 actor 权重后休眠，释放训练显存
            self._switch_model("actor")
            self.sleep()

        # rollout 引擎列表初始为空，等 update_weights 时再获取
        self.rollout_engines = None

        # 可选的 rollout 数据后处理钩子（通过 "模块:函数" 路径加载）
        self.rollout_data_postprocess = None
        if self.args.rollout_data_postprocess_path is not None:
            from slime.utils.misc import load_function

            self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)

        # 通知性能分析器初始化结束
        self.prof.on_init_end()

        # 返回本次训练的起始 rollout id
        return start_rollout_id

    @timer
    def sleep(self) -> None:
        # offload 模式下休眠：释放显存与进程组
        assert self.args.offload_train

        # 清理 GPU 与主机内存缓存
        clear_memory(clear_host_memory=True)
        print_memory("before offload model")
        # 销毁可重载的进程组（NCCL 通信器占显存）
        destroy_process_groups()

        # 暂停 torch_memory_saver，把模型张量从显存卸载到 CPU
        torch_memory_saver.pause()

        print_memory("after offload model")

    @timer
    def wake_up(self) -> None:
        # offload 模式下唤醒：恢复显存与进程组
        assert self.args.offload_train
        print_memory("before wake_up model")

        # 恢复 torch_memory_saver，把模型张量搬回显存
        torch_memory_saver.resume()

        clear_memory()
        # 重建进程组
        reload_process_groups()
        print_memory("after wake_up model")

    def _get_rollout_data(self, rollout_data_ref: Box) -> RolloutBatch:
        # Fetch data through ray on CPU, not sure if this will be performance bottleneck.
        # Both first pp stage and the last pp stage will receive the data.
        # 通过 ray 在 CPU 上拉取本 DP rank 对应的 rollout 数据（首/尾 PP 阶段都会收到）
        rollout_data = process_rollout_data(
            self.args,
            rollout_data_ref,
            mpu.get_data_parallel_rank(with_context_parallel=False),
            mpu.get_data_parallel_world_size(with_context_parallel=False),
        )
        # TODO: this is ugly, move to somewhere else?
        # move tokens to GPU in advance
        # 提前把 tokens 上传 GPU（long 类型）
        rollout_data["tokens"] = [
            torch.tensor(t, dtype=torch.long, device=torch.cuda.current_device()) for t in rollout_data["tokens"]
        ]
        # 提前把 loss_masks 上传 GPU（int 类型）
        rollout_data["loss_masks"] = [
            torch.tensor(t, dtype=torch.int, device=torch.cuda.current_device()) for t in rollout_data["loss_masks"]
        ]
        if "multimodal_train_inputs" in rollout_data:
            # Move multimodal training tensors to GPU in advance
            # 提前把多模态训练张量上传 GPU（保持每个样本字典结构不变）
            rollout_data["multimodal_train_inputs"] = [
                (
                    {key: tensor.to(device=torch.cuda.current_device()) for key, tensor in mm_dict.items()}
                    if mm_dict is not None
                    else None
                )
                for mm_dict in rollout_data["multimodal_train_inputs"]
            ]

        if self.args.qkv_format == "bshd":
            # TODO: micro-batch wise dynamic, possibly move to @data.py:get_data_iterator
            # bshd：取本批最大序列长度
            max_seq_len = max(rollout_data["total_lengths"])

            # pad to reduce memory fragmentation and maybe make the computation faster
            # 按 TP 大小乘以 pad 倍数向上对齐，减少碎片
            pad_size = mpu.get_tensor_model_parallel_world_size() * self.args.data_pad_size_multiplier
            max_seq_len = (max_seq_len + pad_size - 1) // pad_size * pad_size

            # 所有样本统一使用该 max_seq_len
            rollout_data["max_seq_lens"] = [max_seq_len] * len(rollout_data["tokens"])

        # rollout/teacher 的 log_prob 是完整 response 长度；CP>1 时需切成当前 rank 持有的分段
        for key in ["rollout_log_probs", "teacher_log_probs"]:
            if key not in rollout_data:
                continue
            # 对每条样本按 CP 切分 log_prob，再转成 GPU 上的 float32 张量
            rollout_data[key] = [
                torch.tensor(
                    slice_log_prob_with_cp(
                        log_prob,
                        total_length,
                        response_length,
                        self.args.qkv_format,
                        rollout_data["max_seq_lens"][i] if self.args.qkv_format == "bshd" else None,
                    ),
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
                for i, (log_prob, total_length, response_length) in enumerate(
                    zip(
                        rollout_data[key],
                        rollout_data["total_lengths"],
                        rollout_data["response_lengths"],
                        strict=False,
                    )
                )
            ]
        if "rollout_routed_experts" in rollout_data:
            # MoE 路由专家索引从 numpy 转成 torch 张量
            rollout_data["rollout_routed_experts"] = [
                torch.from_numpy(r) for r in rollout_data["rollout_routed_experts"]
            ]
        return rollout_data

    def _switch_model(self, target_tag: str) -> None:
        # 目标标签必须已经备份过
        if target_tag not in self.weights_backuper.backup_tags:
            raise ValueError(f"Cannot switch to unknown model tag: {target_tag}")
        # 从备份器恢复目标权重到 GPU 模型
        self.weights_backuper.restore(target_tag)
        # 更新当前激活的模型标签
        self._active_model_tag = target_tag

    def fill_routing_replay(self, data_iterator, num_microbatches, rollout_data):
        # 开启 rollout 路由回放时，rollout_data 必须带 rollout_routed_experts
        if "rollout_routed_experts" not in rollout_data:
            raise ValueError(
                "rollout_routed_experts is required in rollout_data when use_rollout_routing_replay is set."
            )

        from megatron.core.transformer.transformer_block import get_num_layers_to_build
        from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

        from slime.utils.routing_replay import RoutingReplay

        # 重置所有迭代器，从第一个 micro-batch 重新遍历
        for iterator in data_iterator:
            iterator.reset()

        # TP rank 与 TP 组大小（序列并行时需要按 TP 切分路由张量）
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        # MoE 路由填充函数：生成合法的伪专家索引（对专家数取模）填充到尾部
        def pad_func(experts, pad):
            _, num_layers, topk = experts.shape
            pad = (
                torch.arange(
                    pad * num_layers * topk,
                    device=experts.device,
                    dtype=experts.dtype,
                ).reshape((pad, num_layers, topk))
                % self.args.num_experts
            )
            return torch.cat([experts, pad], dim=0)

        # 逐 micro-batch 处理路由数据
        for _ in range(sum(num_microbatches)):
            batch = data_iterator[0].get_next(["rollout_routed_experts", "tokens"])
            rollout_routed_experts = batch["rollout_routed_experts"]
            tokens = batch["tokens"]
            # 样本数必须与 tokens 一致
            assert len(rollout_routed_experts) == len(tokens)
            for a, b in zip(rollout_routed_experts, tokens, strict=False):
                # 每条样本的路由数应比 token 数少 1（最后一个 token 无预测目标）
                assert a.shape[0] == b.shape[0] - 1, f"{a.shape}, {b.shape}"

            # We need to pad the experts to the last token. We won't calculate loss on this token so this should be fine.
            # TODO: fuse this padding with the following slice_with_cp to reduce memory copy.
            # 每条样本尾部补 1 个伪路由，使其与 token 数对齐（该 token 不计 loss）
            rollout_routed_experts = [pad_func(r, 1) for r in rollout_routed_experts]
            # TODO: maybe extract a common process function for here and get_batch?
            # 与 tokens 同样做 CP zigzag 切分
            rollout_routed_experts = [slice_with_cp(r, pad_func) for r in rollout_routed_experts]
            rollout_routed_experts = torch.cat(rollout_routed_experts, dim=0)
            pad_size = mpu.get_tensor_model_parallel_world_size() * self.args.data_pad_size_multiplier
            # 补齐到 pad 粒度（与 get_batch 中 tokens 的 pad 保持一致）
            pad = (pad_size - rollout_routed_experts.size(0) % pad_size) % pad_size
            if pad != 0:
                rollout_routed_experts = pad_func(rollout_routed_experts, pad)

            if self.args.sequence_parallel:
                # 序列并行：沿序列维按 TP rank 切分路由张量
                seqlen = rollout_routed_experts.size(0)
                assert seqlen % tp_size == 0
                start, end = seqlen // tp_size * tp_rank, seqlen // tp_size * (tp_rank + 1)
                rollout_routed_experts = rollout_routed_experts[start:end]

            # 逐条记录每个 MoE 层的路由：回放时直接使用 rollout 时的专家选择
            routing_replay_offset = 0
            for vp_stage, model in enumerate(self.model):
                config = model.module.config
                # 该 vp 阶段需要构建的层数与全局层偏移
                num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
                offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
                for layer_id in range(offset, offset + num_layers_to_build):
                    # skip dense layer
                    # 跳过非 MoE（稠密）层
                    if isinstance(config.moe_layer_freq, int):
                        if layer_id % config.moe_layer_freq != 0:
                            continue
                    elif isinstance(config.moe_layer_freq, list):
                        assert len(config.moe_layer_freq) == config.num_layers
                        if config.moe_layer_freq[layer_id] == 0:
                            continue
                    # 取该层对应的专家路由并记录到对应的 RoutingReplay
                    layer_routed_experts = rollout_routed_experts[:, layer_id]
                    RoutingReplay.all_routing_replays[routing_replay_offset].record(layer_routed_experts)
                    routing_replay_offset += 1
            # 断言记录的回放数量与全局回放器数量一致
            assert routing_replay_offset == len(RoutingReplay.all_routing_replays)

        # 路由数据已被回放器持有，从 rollout_data 中移除
        del rollout_data["rollout_routed_experts"]

        # 再次重置迭代器，供后续前向/训练复用
        for iterator in data_iterator:
            iterator.reset()

    def compute_log_prob(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:

        # 仅前向计算（不反向），得到 log_prob 与熵；store_prefix 区分 ref/teacher/当前策略
        with timer(f"{store_prefix}log_probs"):
            return forward_only(
                get_log_probs_and_entropy,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
            )

    def train(self, rollout_id: int, rollout_data_ref: Box) -> None:
        # 仅调试 rollout 时不执行训练
        if self.args.debug_rollout_only:
            return

        # offload 模式：训练前先唤醒（恢复显存与进程组）
        if self.args.offload_train:
            self.wake_up()

        with timer("data_preprocess"):
            # 拉取并预处理本 rank 的 rollout 数据
            rollout_data = self._get_rollout_data(rollout_data_ref)
            # 记录 rollout 阶段的指标
            log_rollout_data(rollout_id, self.args, rollout_data)

        # 按 role 分发到 critic 或 actor 的训练流程
        if self.role == "critic":
            return self.train_critic(rollout_id, rollout_data)
        else:
            return self.train_actor(rollout_id, rollout_data)

    def train_critic(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        # Create data iterator for log_probs and train.
        # 为本步训练创建数据迭代器与 micro-batch 调度
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
        # 纯前向计算 critic 的 value 估计，写入 rollout_data
        rollout_data.update(
            forward_only(
                get_values,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
            )
        )

        # 超过 critic-only 预热步数后，才需要从 actor 侧同步 log_probs
        if rollout_id >= self.args.num_critic_only_steps:
            sync_actor_critic_data(self.args, rollout_data, self._actor_critic_groups)

        # 用 value/log_prob/奖励计算优势（advantage，GAE）与回报（return）
        compute_advantages_and_returns(self.args, rollout_data)

        # 切换到 value loss 并训练 critic
        self.args.loss_type = "value_loss"
        train(
            rollout_id,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            data_iterator,
            num_microbatches,
        )

    def train_actor(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        # Create data iterator for log_probs and train.
        # 为本步创建数据迭代器与 micro-batch 调度
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)

        if self.args.use_rollout_routing_replay:
            # 预先填充 MoE 路由回放器
            self.fill_routing_replay(data_iterator, num_microbatches, rollout_data)

        with inverse_timer("train_wait"), timer("train"):
            if self.args.compute_advantages_and_returns:
                if "ref" in self.weights_backuper.backup_tags:
                    # 切换到 ref 模型计算参考 log_prob（KL 用）
                    if self.args.use_routing_replay:
                        # 路由回放直通模式（不记录不回放）
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("ref")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="ref_",
                        )
                    )

                # Forward teacher model to get teacher_log_probs for Megatron-based OPD
                if "teacher" in self.weights_backuper.backup_tags:
                    # 切换到教师模型计算 teacher_log_prob（在线策略蒸馏用）
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("teacher")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="teacher_",
                        )
                    )

                # 切回当前策略（保留 old_actor 时用 old_actor 计算重要性比）
                self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
                if not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics:
                    if self.args.use_routing_replay:
                        if self.args.use_rollout_routing_replay:
                            # 直接回放 rollout 时的路由
                            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
                        else:
                            # 记录本次前向的路由，供反向回放
                            os.environ["ROUTING_REPLAY_STAGE"] = "record"
                    # 计算当前策略的 log_prob 与熵
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="",
                        )
                    )
                    if self.args.use_rollout_routing_replay:
                        # 回放完成后清空前向缓存
                        RoutingReplay.clear_all_forward()

                if self.args.use_critic:
                    # 使用 critic 时，跨 PP 同步 values 与 log_probs
                    sync_actor_critic_data(
                        self.args,
                        rollout_data,
                        self._actor_critic_groups,
                    )
                # 确保当前激活的是 actor 权重（训练用）
                if self._active_model_tag != "actor":
                    self._switch_model("actor")

                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                # 计算优势（advantage）与回报（return）：必须在训练前完成，
                # 因为可能需要对整个 rollout 做归一化
                compute_advantages_and_returns(self.args, rollout_data)

            if self.rollout_data_postprocess is not None:
                # 执行用户自定义的 rollout 数据后处理
                self.rollout_data_postprocess(self.args)

            # 记录 rollout 指标（此时已含 log_prob/adv 等训练侧指标）
            log_rollout_data(
                rollout_id,
                self.args,
                rollout_data,
            )

            # Train
            if self.args.use_routing_replay:
                # 反向阶段直接回放正向记录的路由（省去重计算）
                os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"
            with timer("actor_train"):
                # 执行梯度更新训练
                train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    data_iterator,
                    num_microbatches,
                )

            # 通知性能分析器本步结束
            self.prof.step(rollout_id=rollout_id)

        # CI/调试模式下保存本步训练数据快照
        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        if self.args.use_routing_replay:
            # 清空全部路由回放缓存
            RoutingReplay.clear_all()

        # update the cpu actor weight to the latest model
        # 把更新后的 actor 权重重新备份（供推理引擎同步与切换）
        self.weights_backuper.backup("actor")

        # Update ref model if needed
        # 按配置的间隔定期把当前策略同步为新的 ref 模型
        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and "ref" in self.weights_backuper.backup_tags
        ):
            with timer("ref_model_update"):
                if is_megatron_main_rank():
                    logger.info(f"Updating ref model at rollout_id {rollout_id}")
                self.weights_backuper.backup("ref")

        # 记录本步性能数据
        log_perf_data(rollout_id, self.args)

    @timer
    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        # 仅调试 rollout 时不保存
        if self.args.debug_rollout_only:
            return

        # torch dist may trigger nccl communication during saving.
        # 保存过程可能触发 NCCL 通信，offload 模式下需先重建进程组
        if self.args.offload_train:
            reload_process_groups()

        if self.args.async_save:
            from megatron.training.async_utils import maybe_finalize_async_save

            # 异步保存模式：先阻塞等待上一轮保存完成，避免并发写
            maybe_finalize_async_save(blocking=True)

        # 执行 Megatron checkpoint 保存
        save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)

        if force_sync and self.args.async_save:
            # 强制同步时阻塞等待异步保存全部落盘
            maybe_finalize_async_save(blocking=True)

        if self.args.save_hf is not None and self.role == "actor":
            from slime.backends.megatron_utils.model import save_hf_model

            # 按需额外导出一份 HuggingFace 格式模型（仅 actor）
            save_hf_model(self.args, rollout_id, self.model)

        if self.args.offload_train:
            # 保存完成后再销毁进程组
            destroy_process_groups()

    @timer
    def update_weights(self) -> None:
        # 仅训练或仅 rollout 的调试模式下不同步权重
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.use_fault_tolerance:
            # 容错模式：先让 rollout 引擎恢复（rollout 侧故障可能丢过引擎）
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.recover_rollout_engines.remote())
            dist.barrier(group=get_gloo_group())

        # 从 rollout manager 获取（可能新增的）rollout 引擎列表与锁
        rollout_engines, rollout_engine_lock, num_new_engines = ray.get(
            self.rollout_manager.get_rollout_engines_and_lock.remote()
        )

        if self.args.offload_train:
            # offload 模式：同步权重前重建进程组
            reload_process_groups()

        if num_new_engines > 0:
            # 有新引擎（弹性扩容）时建立连接，并清零新增计数
            self.weight_updater.connect_rollout_engines(rollout_engines, rollout_engine_lock)
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_num_new_engines.remote())

        # 权重同步需要访问被卸载的张量，offload 模式下临时禁用 torch_memory_saver
        with torch_memory_saver.disable() if self.args.offload_train else nullcontext():
            print_memory("before update_weights")
            # 把 actor 最新权重推送到 rollout 引擎
            self.weight_updater.update_weights()
            print_memory("after update_weights")

            if self.args.ci_test and len(rollout_engines) > 0:
                # CI 测试：随机抽查一个引擎的权重版本，必须与更新器一致
                engine = random.choice(rollout_engines)
                engine_version = ray.get(engine.get_weight_version.remote())
                if str(engine_version) != str(self.weight_updater.weight_version):
                    raise RuntimeError(
                        f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
                    )

            if getattr(self.args, "keep_old_actor", False):
                if self.args.update_weights_interval == 1:
                    logger.info("updating model queue: rollout_actor -> old_actor, actor -> rollout_actor")
                    # Queue-style update: rollout_actor params -> old_actor, actor params -> rollout_actor
                    # First copy rollout_actor to old_actor
                    # 队列式滚动更新：先复制 rollout_actor -> old_actor
                    self.weights_backuper.copy(src_tag="rollout_actor", dst_tag="old_actor")
                    # Then copy current actor to rollout_actor
                    # 再把当前 actor 备份为新的 rollout_actor
                    self.weights_backuper.backup("rollout_actor")
                else:
                    # 间隔大于 1 时直接把当前 actor 备份为 old_actor
                    self.weights_backuper.backup("old_actor")

        if self.args.offload_train:
            # 同步完成后销毁进程组
            destroy_process_groups()

    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        # 临时改写加载相关参数：只加载权重，不加载优化器/RNG，且视为 finetune
        old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True

        old_ckpt_step = None
        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            # ref 模型可指定独立的 checkpoint step
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.ref_ckpt_step
        elif model_tag == "teacher" and self.args.opd_teacher_ckpt_step is not None:
            # 教师模型同理
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.opd_teacher_ckpt_step

        # 把该 checkpoint 的权重加载进模型（优化器传 None）
        _, _ = load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
        # 恢复被临时改写的参数
        self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old_args

        if old_ckpt_step is not None:
            self.args.ckpt_step = old_ckpt_step

        # 把加载的权重备份为指定标签，并把激活模型切换过去
        self.weights_backuper.backup(model_tag)
        self._active_model_tag = model_tag

    def connect_actor_critic(
        self,
        actor_handle: ActorHandle | None = None,
        master_address: str | None = None,
        master_port: int | None = None,
    ) -> None:
        if self.role == "actor":
            # actor 侧负责选地址与端口（绑定临时端口），并把连接信息传给 critic
            master_address = ray.util.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            actor_handle.connect_actor_critic.remote(master_address=master_address, master_port=master_port)

        # 建立一个 2 进程（actor=0, critic=1）的 NCCL 组，用于跨角色同步数据
        group_name = "actor_critic"
        world_size = 2
        self._actor_critic_groups = init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=0 if self.role == "actor" else 1,
            group_name=group_name,
        )
