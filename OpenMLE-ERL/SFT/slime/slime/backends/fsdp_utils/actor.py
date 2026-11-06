# 导入 logging 用于日志输出
import logging
# 导入 os 用于路径判断（ref 模型加载路径是否为目录）
import os
# 导入 random 用于 CI 测试时随机挑选推理引擎校验权重版本
import random
# 导入 Namespace 类型注解
from argparse import Namespace
# 导入 accumulate 用于计算梯度累积边界的前缀和
from itertools import accumulate

# 导入 Ray 分布式框架
import ray
# 导入 torch 主库
import torch
# 导入 torch 分布式模块
import torch.distributed as dist
# 导入 tqdm 进度条
from tqdm import tqdm
# 导入 HF 的 AutoConfig 用于读取模型配置
from transformers import AutoConfig

# 导入 slime 的训练 Ray Actor 基类（封装 Ray actor 生命周期与角色）
from slime.ray.train_actor import TrainRayActor
# 导入 slime 的日志与训练数据/指标工具模块
from slime.utils import logging_utils, train_dump_utils, train_metric_utils
# 导入数据工具：动态 micro-batch 最小切分数计算、rollout 数据按 DP rank 切分
from slime.utils.data import get_minimum_num_micro_batch_size, process_rollout_data
# 导入 Gloo 通信组获取工具（CPU 侧 barrier 用）
from slime.utils.distributed_utils import get_gloo_group
# 导入实验跟踪（wandb 等）初始化函数
from slime.utils.logging_utils import init_tracking
# 导入显存工具：清理缓存与打印显存占用
from slime.utils.memory_utils import clear_memory, print_memory
# 导入 rollout 步数计算工具
from slime.utils.metric_utils import compute_rollout_step
# 导入 Box：Ray object ref 的轻量包装
from slime.utils.misc import Box
# 导入 PPO/RL 损失相关工具：近似 KL、GSPO KL、OPSM 掩码、策略梯度损失
from slime.utils.ppo_utils import compute_approx_kl, compute_gspo_kl, compute_opsm_mask, compute_policy_loss
# 导入处理器/分词器加载工具
from slime.utils.processing_utils import load_processor, load_tokenizer
# 导入训练性能剖析器
from slime.utils.profile_utils import TrainProfiler
# 导入计时器工具：区间计时、反向计时、延迟启动装饰器
from slime.utils.timer import Timer, inverse_timer, timer, with_defer

# 导入同包的 checkpoint 保存/加载模块
from . import checkpoint
# 导入序列打包/反解工具
from .data_packing import pack_sequences, unpack_sequences
# 导入学习率调度器工厂
from .lr_scheduler import get_lr_scheduler
# 导入两种权重同步实现：张量直传（colocate 场景）与 NCCL 广播（分离部署场景）
from .update_weight_utils import UpdateWeightFromDistributed, UpdateWeightFromTensor

# 获取本模块 logger
logger = logging.getLogger(__name__)


# FSDP 后端的训练 Ray Actor：基于纯 HF Transformers + PyTorch FSDP2 实现 SFT/RL 训练侧
class FSDPTrainRayActor(TrainRayActor):
    """Simplified TrainRayActor for pure HF+FSDP training.

    Responsibilities:
      * Initialize model/tokenizer on rank0 sequentially to avoid race on cache
      * Wrap model with FSDP
      * Provide minimal train / save / update_weights hooks compatible with existing RayTrainGroup

    Weight update strategy:
      * Rank0 gathers state_dict (full) and broadcasts tensor-by-tensor.
      * For small models this is fine; for larger models consider sharded state_dict type.
    """

    # with_defer 装饰器：函数返回时自动启动 "train_wait" 计时（衡量下一次训练的等待时间）
    @with_defer(lambda: Timer().start("train_wait"))
    def init(self, args: Namespace, role: str, with_ref: bool = False, with_opd_teacher: bool = False) -> int:  # type: ignore[override]
        # FSDP 后端不支持基于 Megatron 教师模型的在线蒸馏（OPD），直接报错提示改用其他后端
        if with_opd_teacher:
            raise NotImplementedError(
                "On-policy distillation (OPD) with Megatron teacher is not supported in FSDP backend. "
                "Please use the Megatron backend for OPD, or use --opd-type=sglang with an external teacher server."
            )
        # 调用父类初始化（角色、参数、Ray 相关设置）
        super().init(args, role, with_ref, with_opd_teacher)

        # Setup device mesh for data parallelism
        # 初始化数据并行的设备网格（device mesh）
        self._setup_device_mesh()
        # 用参数中的种子设置全局随机种子（保证可复现）
        torch.manual_seed(args.seed)

        # 记录训练并行配置（目前只有数据并行规模）
        self.train_parallel_config = {
            "dp_size": self.dp_size,
        }

        # 仅调试 rollout 时无需初始化训练组件，提前返回
        if self.args.debug_rollout_only:
            return 0

        # 读取 FSDP CPU 卸载开关（默认关闭）
        self.fsdp_cpu_offload = getattr(self.args, "fsdp_cpu_offload", False)
        # Offload train and fsdp cpu offload cannot be used together, fsdp_cpu_offload is more aggressive
        # 训练间隙 offload 与 FSDP 常驻 CPU 卸载互斥：同时开启时以更激进的 fsdp_cpu_offload 为准
        if self.args.offload_train and self.fsdp_cpu_offload:
            self.args.offload_train = False

        # 按需启用 true on-policy 优化（批量不变算子 / 自研 MoE 补丁）
        self._enable_true_on_policy_optimizations(args)
        # 仅 rank 0 初始化实验跟踪（wandb 等），primary=False 表示非主控角色
        if dist.get_rank() == 0:
            init_tracking(args, primary=False)

        # 未指定起始 rollout id 时默认从 0 开始
        if getattr(self.args, "start_rollout_id", None) is None:
            self.args.start_rollout_id = 0

        # 创建训练性能剖析器
        self.prof = TrainProfiler(args)

        # 各 rank 顺序加载 config/tokenizer（processor）：用 barrier 串行化，避免并发下载/转换缓存的竞态
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                # 从 HF checkpoint 读取模型配置（信任远程代码）
                self.hf_config = AutoConfig.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
                # 加载分词器
                self.tokenizer = load_tokenizer(self.args.hf_checkpoint, trust_remote_code=True)
                # Vision models have `vision_config` in the config
                # 视觉语言模型的 config 里带 vision_config，需要额外加载 processor（图像预处理）
                if hasattr(self.hf_config, "vision_config"):
                    self.processor = load_processor(self.args.hf_checkpoint, trust_remote_code=True)
            # barrier 等待当前 rank 加载完成，下一个 rank 再开始
            dist.barrier(group=get_gloo_group())

        # 获取模型初始化上下文管理器（rank0 用 CPU，其他 rank 用 meta 设备省内存）
        init_context = self._get_init_weight_context_manager()

        # 在初始化上下文中加载模型权重
        with init_context():
            model = self.get_model_cls().from_pretrained(
                self.args.hf_checkpoint,
                trust_remote_code=True,
                attn_implementation=self.args.attn_implementation,
            )

        # 切换到训练模式（启用 dropout 等）
        model.train()

        # 先取出完整（未分片）状态字典，供后续广播到各 FSDP 分片
        full_state = model.state_dict()

        # 用 FSDP2（fully_shard）包装模型：按 DP mesh 分片，可选 CPU 卸载
        model = apply_fsdp2(model, mesh=self.dp_mesh, cpu_offload=self.fsdp_cpu_offload, args=self.args)

        # 把 rank0 的完整 state dict 广播/加载进 FSDP2 分片模型
        model = self._fsdp2_load_full_state_dict(
            model, full_state, self.dp_mesh, cpu_offload=True if self.fsdp_cpu_offload else None
        )

        # 保存模型引用
        self.model = model

        if args.gradient_checkpointing:
            # 启用梯度检查点：以重算换显存
            self.model.gradient_checkpointing_enable()

        if args.optimizer == "adam":
            # Adam 选项实际构造 AdamW 优化器，超参来自命令行
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=args.lr,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
            )
        else:
            # 其他优化器暂不支持
            raise ValueError(f"Unsupported optimizer: {args.optimizer}. Supported options: 'adam'")

        # Initialize LR scheduler
        # 创建学习率调度器
        self.lr_scheduler = get_lr_scheduler(args, self.optimizer)

        # 全局训练步与微批次步计数器清零
        self.global_step = 0
        self.micro_step = 0

        # 从 checkpoint 恢复（若配置了 load 路径），返回 payload 供 finalize 恢复 RNG/进度
        checkpoint_payload = checkpoint.load(self)

        # Create separate ref model if needed (kept in CPU until needed)
        # 按需创建独立的参考模型（ref model，用于 KL 约束），平时驻留 CPU
        self.ref_model = None
        if with_ref:
            self.ref_model = self._create_ref_model(args.ref_load)

        # 按部署方式选择权重同步器：训练与推理同卡共置（colocate）用张量直传，否则用 NCCL 广播
        self.weight_updater = (
            UpdateWeightFromTensor(self.args, self.model)
            if self.args.colocate
            else UpdateWeightFromDistributed(self.args, self.model)
        )

        # 收尾恢复：RNG 状态、全局步数、rollout 起始 id，并做分布式同步
        checkpoint.finalize_load(self, checkpoint_payload)

        # Initialize data packing parameters
        # 记录每个 GPU 打包批次的最大 token 数（来自主参数）
        self.max_tokens_per_gpu = args.max_tokens_per_gpu  # From main arguments

        if self.args.offload_train:
            # 训练间隙卸载模式下，初始化完成后先把模型/优化器挪到 CPU
            self.sleep()

        # 通知剖析器 init 阶段结束
        self.prof.on_init_end()

        # 返回起始 rollout id（Ray actor 的初始化返回值）
        return int(getattr(self.args, "start_rollout_id", 0))

    # 根据模型类型选择合适的 HF 自动模型类
    def get_model_cls(self):
        # Vision models have `vision_config` in the config
        # 视觉语言模型用 ImageTextToText（图文到图文）自动类
        if hasattr(self.hf_config, "vision_config"):
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText
        else:
            # 纯文本模型用因果语言模型自动类
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM

    # 按需启用 true on-policy（真在线策略）相关优化
    def _enable_true_on_policy_optimizations(self, args):
        if args.true_on_policy_mode:
            # true on-policy 模式：启用 sglang 的批量不变（batch-invariant）算子，保证训练前向与推理前向逐位一致
            from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

            from .models.qwen3_moe import apply_true_on_policy_patch_for_qwen3_moe

            logger.info("FSDPTrainRayActor call enable_batch_invariant_mode for true-on-policy")
            enable_batch_invariant_mode(
                # In Qwen3, rope `inv_freq_expanded.float() @ position_ids_expanded.float()` uses bmm
                # and disabling it will make it aligned
                # Qwen3 的 RoPE 内部用 bmm，禁用 bmm 不变模式可使训练与推理对齐
                enable_bmm=False,
            )

            # 用带融合算子的自研 MoE 实现替换 HF 的 MoE 块
            apply_true_on_policy_patch_for_qwen3_moe()
        else:
            # 非 true on-policy：仅打 FSDP 兼容的 MoE 补丁（逐专家循环，保专家梯度）
            from .models.qwen3_moe_hf import apply_fsdp_moe_patch

            apply_fsdp_moe_patch()

    # 初始化数据并行的设备网格
    def _setup_device_mesh(self) -> None:
        """Setup device mesh for data parallelism."""
        # 延迟导入设备网格工具
        from torch.distributed.device_mesh import init_device_mesh

        # 读取全局进程数与当前 rank
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Pure data parallelism
        # 纯数据并行：DP 规模即世界大小
        self.dp_size = world_size
        self.dp_rank = rank

        # Create 1D device mesh for data parallelism
        # 创建 1 维 CUDA 设备网格（名为 dp），并取出对应的进程组
        self.mesh = init_device_mesh("cuda", mesh_shape=(self.dp_size,), mesh_dim_names=("dp",))
        self.dp_group = self.mesh.get_group("dp")
        self.dp_mesh = self.mesh

        # 记录网格信息
        logger.info(f"[Rank {rank}] Device mesh (1D): world_size={world_size}, dp_size={self.dp_size}")

    # 获取模型加载时的上下文管理器（控制各 rank 用 CPU 还是 meta 设备初始化）
    def _get_init_weight_context_manager(self):
        """Get context manager for model initialization.

        Returns a callable that creates a context manager.
        Uses meta device (no memory allocation) for non-rank-0 processes,
        UNLESS tie_word_embeddings=True (which causes hangs with meta tensors).

        Ref: verl/utils/fsdp_utils.py::get_init_weight_context_manager
        NOTE: tie_word_embedding causes meta_tensor init to hang
        """
        # 延迟导入 accelerate 的空权重初始化上下文
        from accelerate import init_empty_weights

        # Check if model uses tied word embeddings (which doesn't work with meta tensors)
        # 词嵌入与 LM head 权重绑定（tied embeddings）时 meta 张量会挂起，需禁用 meta 初始化
        use_meta_tensor = not self.hf_config.tie_word_embeddings

        # CPU 初始化上下文的简单包装（返回恒等上下文）
        def cpu_init_weights():
            return torch.device("cpu")

        if use_meta_tensor:
            # Rank 0: CPU, others: meta device (memory efficient for large models)
            # rank0 用真实 CPU 加载，其他 rank 用 meta 设备（零内存分配），随后靠广播同步权重
            return init_empty_weights if dist.get_rank() != 0 else cpu_init_weights
        else:
            logger.info(f"[Rank {dist.get_rank()}] tie_word_embeddings=True, loading full model to CPU on all ranks")
            return cpu_init_weights

    # 把完整 state dict 加载进 FSDP2 模型：由 rank0 广播到所有 rank
    def _fsdp2_load_full_state_dict(self, model, full_state, device_mesh, cpu_offload):
        """Load full state dict into FSDP2 model with efficient broadcast from rank 0.

        This function loads weights from rank 0 and broadcasts to all other ranks,
        avoiding the need for each rank to load the full model from disk.

        Args:
            model: FSDP2-wrapped model
            full_state: State dict (only rank 0 has real weights, others have empty dict)
            device_mesh: Device mesh for FSDP
            cpu_offload: If not None, enables StateDictOptions cpu_offload

        Ref:verl/utils/fsdp_utils.py::fsdp2_load_full_state_dict
        """
        # 延迟导入 DCP 的状态字典工具
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

        # Rank 0: move with weights, others: allocate empty tensors on device
        # rank0 把带权重的模型搬到当前 GPU
        if dist.get_rank() == 0:
            model = model.to(device=torch.cuda.current_device(), non_blocking=True)
        else:
            # to_empty creates tensors on device without initializing memory
            # 其他 rank 用 to_empty 在 GPU 上分配未初始化的张量（内容随后由广播填充）
            model = model.to_empty(device=torch.cuda.current_device())

        # 判断是否启用 state dict 的 CPU 卸载选项
        is_cpu_offload = cpu_offload is not None
        # 配置：完整 state dict、可选 CPU 卸载、从 rank0 广播
        options = StateDictOptions(full_state_dict=True, cpu_offload=is_cpu_offload, broadcast_from_rank0=True)

        # 执行加载与广播（参数分片随 FSDP 布局自动处理）
        set_model_state_dict(model, full_state, options=options)

        # set_model_state_dict will not broadcast buffers, so we need to broadcast them manually.
        # set_model_state_dict 不广播 buffer（如 BN 的 running stats），需手动广播
        for _name, buf in model.named_buffers():
            dist.broadcast(buf, src=0)

        if is_cpu_offload:
            # CPU 卸载模式：把参数挪回 CPU，但 buffer 保留在 GPU 上（前向直接可用）
            model.to("cpu", non_blocking=True)
            for buf in model.buffers():
                buf.data = buf.data.to(torch.cuda.current_device())

        # 返回加载完成的 FSDP2 模型
        return model

    # 训练间隙休眠：把模型与优化器状态卸载到 CPU 释放显存
    @timer
    def sleep(self) -> None:
        """Pause CUDA memory for all tracked tensors."""
        # 未启用训练间隙卸载则无事可做
        if not self.args.offload_train:
            return

        # 打印卸载前显存
        print_memory("before offload model")

        # 模型移到 CPU
        self.model.cpu()
        # 优化器状态（动量等）移到 CPU
        move_torch_optimizer(self.optimizer, "cpu")
        # 清理缓存
        clear_memory()
        # barrier 保证所有 rank 完成卸载
        dist.barrier(group=get_gloo_group())
        # 打印卸载后显存
        print_memory("after offload model")

    # 训练前唤醒：把模型与优化器状态搬回 GPU
    @timer
    def wake_up(self) -> None:
        """Resume CUDA memory for all tracked tensors."""
        # 未启用训练间隙卸载则无事可做
        if not self.args.offload_train:
            return

        # 模型移回 GPU
        self.model.cuda()
        # 优化器状态移回 GPU
        move_torch_optimizer(self.optimizer, "cuda")
        # barrier 同步
        dist.barrier(group=get_gloo_group())
        # 打印唤醒后显存
        print_memory("after wake_up model")

    # 保存模型 checkpoint（委托给共享的 checkpoint 工具）
    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        """Delegate checkpoint saving to the shared checkpoint utilities."""
        # 仅调试 rollout 或未配置保存路径时跳过
        if self.args.debug_rollout_only or self.args.save is None:
            return

        # FSDP 后端暂不支持异步保存
        assert not self.args.async_save, "FSDPTrainRayActor does not support async_save yet."
        # 调用 checkpoint.save 完成实际保存
        checkpoint.save(self, rollout_id)

    # 为一组打包批次计算 token 级对数概率（actor 或 ref 模型）
    def _compute_log_prob(
        self,
        model_tag: str,
        packed_batches: list[dict[str, torch.Tensor]],
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:
        """Compute token log-probabilities for a list of packed batches.

        Parameters:
            model_tag: Which parameters to use, e.g. "actor" or "ref".
            packed_batches: A list of packed batch dictionaries produced by
                `pack_sequences`, each containing at least `tokens` and
                `position_ids`; may also include multimodal keys like `pixel_values`.
            store_prefix: Prefix to use for keys in outputs (e.g., "ref_").

        Returns:
            A lightweight dictionary keyed by f"{store_prefix}log_probs". The
            actual per-sequence results are written in-place into each element of
            `packed_batches` under the same key and can be read back by callers.

        Note:
            Uses separate ref model when model_tag == "ref". The ref model is
            loaded from CPU to GPU on-demand and offloaded back after use.
        """
        # Select which model to use
        # 选择使用的模型：ref 标签且有独立 ref 模型时切换到 ref
        if model_tag == "ref" and self.ref_model is not None:
            if not self.fsdp_cpu_offload:
                # 非 CPU 卸载模式时，先把 actor 模型挪出 GPU 腾显存
                self.model.cpu()
                torch.cuda.empty_cache()
                dist.barrier(group=get_gloo_group())

            # 激活 ref 模型并切到 eval 模式
            active_model = self.ref_model
            active_model.eval()
        else:
            # 默认使用 actor 模型
            active_model = self.model

        try:
            # 返回值字典：仅记录键名，实际结果就地写入 packed_batches
            rollout_data = {f"{store_prefix}log_probs": []}
            # 计时并关闭梯度（纯前向统计）
            with timer(f"{store_prefix}log_probs"), torch.no_grad():
                # 遍历打包批次（带剖析打点与进度条）
                for batch in self.prof.iterate_train_log_probs(
                    tqdm(packed_batches, desc=f"{store_prefix}log_probs", disable=dist.get_rank() != 0)
                ):
                    # 构造模型输入参数
                    model_args = self._get_model_inputs_args(batch)
                    # 前向得到 logits：squeeze 掉 batch 维并转 fp32 提升数值精度
                    logits = active_model(**model_args).logits.squeeze(0).float()
                    # 由 logits 计算目标 token 的对数概率与熵
                    log_probs_result, entropy_result = get_logprob_and_entropy(
                        logits=logits,
                        target_tokens=batch["tokens"],
                        allow_compile=not self.args.true_on_policy_mode,
                        temperature=self.args.rollout_temperature,
                    )
                    # 把对数概率写回批次（按前缀区分 ref/actor）
                    batch[f"{store_prefix}log_probs"] = log_probs_result
                    if store_prefix == "":
                        # 仅 actor 前向时额外记录熵
                        batch["entropy"] = entropy_result
            # 返回轻量结果字典
            return rollout_data

        finally:
            # Restore actor model if it was offloaded
            # 无论是否异常，结束后把 actor 模型搬回 GPU
            if model_tag == "ref" and self.ref_model is not None:
                torch.cuda.empty_cache()
                dist.barrier(group=get_gloo_group())

                if not self.fsdp_cpu_offload:
                    self.model.cuda()
                    dist.barrier(group=get_gloo_group())

    # 把 rollout 数据打包成多个稠密批次，并给出梯度累积边界
    def _packed_data(
        self, rollout_data: dict[str, list[torch.Tensor]]
    ) -> tuple[list[dict[str, torch.Tensor]], list[int]]:
        """Pack variable-length sequences for efficient processing.

        Parameters:
            rollout_data: Dictionary of lists containing sequence-level tensors
                such as `tokens`, `loss_masks`, `rewards`, `response_lengths`,
                `advantages`, `returns`, and optional `rollout_log_probs`.

        Returns:
            A pair `(packed_batches, grad_accum)` where `packed_batches` is a list
            of packed batch dictionaries and `grad_accum` lists the micro-batch
            indices at which to perform optimizer steps.
        """
        # Pack sequences efficiently
        # 取出 token 序列列表
        tokens = rollout_data["tokens"]

        # 打包批次列表与各 local batch 所需的 micro-batch 数列表
        packed_batches = []
        mbs_size_list = []
        # 本 rank 的 local batch 大小 = 全局 batch / DP 规模
        local_batch_size = self.args.global_batch_size // self.dp_size
        assert (
            self.args.global_batch_size % self.dp_size == 0
        ), f"global_batch_size {self.args.global_batch_size} is not divisible by dp_world_size {self.dp_size}"
        # Use global_batch_size for splitting when max_tokens_per_gpu is enabled
        # 动态 batch 模式：按每卡最大 token 数切分 micro-batch
        if self.args.use_dynamic_batch_size:
            max_tokens = self.args.max_tokens_per_gpu

            # 逐 local batch 计算最少需要的 micro-batch 数
            for i in range(0, len(tokens), local_batch_size):
                mbs_size_list.append(
                    get_minimum_num_micro_batch_size(
                        [len(t) for t in rollout_data["tokens"][i : i + local_batch_size]],
                        max_tokens,
                    )
                )
            # 各 DP rank 的 micro-batch 数可能不同，取全局最大值保证步调一致
            num_microbatches = torch.tensor(mbs_size_list, dtype=torch.int, device=torch.cuda.current_device())
            dist.all_reduce(num_microbatches, op=dist.ReduceOp.MAX, group=self.dp_group)
            num_microbatches = num_microbatches.tolist()
        else:
            # 静态模式：按固定 micro_batch_size 计算，每个 local batch 的切分数相同
            num_microbatches = [self.args.global_batch_size // (self.args.micro_batch_size * self.dp_size)] * (
                len(tokens) // local_batch_size
            )

        # 按切分数逐段打包序列
        start = 0
        for mbs_size in num_microbatches:
            # 当前 local batch 的区间 [start, end)
            end = start + local_batch_size
            # 把该区间内所有序列打包成 mbs_size 个稠密批次
            packed_batches.extend(
                pack_sequences(
                    rollout_data["tokens"][start:end],
                    rollout_data["loss_masks"][start:end],
                    rollout_data["rewards"][start:end],
                    rollout_data["raw_reward"][start:end],
                    rollout_data["response_lengths"][start:end],
                    rollout_data["advantages"][start:end],
                    rollout_data["returns"][start:end],
                    rollout_log_probs=(
                        rollout_data["rollout_log_probs"][start:end] if "rollout_log_probs" in rollout_data else None
                    ),
                    multimodal_train_inputs=(
                        rollout_data["multimodal_train_inputs"][start:end]
                        if "multimodal_train_inputs" in rollout_data
                        else None
                    ),
                    num_packs=mbs_size,
                )
            )
            # 前移区间起点
            start = end
        # 梯度累积边界：micro-batch 序号的前缀和（到达这些序号时执行 optimizer.step）
        grad_accum = list(accumulate(num_microbatches))

        # 返回打包批次与累积边界
        return packed_batches, grad_accum

    # 执行一次 rollout 批次的训练更新（外层：唤醒、取数据、计时与性能日志）
    def train(self, rollout_id: int, rollout_data_ref: Box) -> None:
        """Run one training update over a rollout batch.

        Parameters:
            rollout_id: Monotonic id for logging.
            rollout_data_ref: A Box handle wrapping a Ray object reference to a
                dictionary with rollout tensors and metadata (e.g., `tokens`,
                `loss_masks`, `rewards`, `response_lengths`, optional
                `rollout_log_probs`, etc.). It will be fetched and partitioned
                by `process_rollout_data` based on data-parallel rank/size.
        """
        # 训练间隙卸载模式先唤醒模型
        if self.args.offload_train:
            self.wake_up()

        # inverse_timer 结束 "train_wait" 计时；timer 开始 "train" 计时
        with inverse_timer("train_wait"), timer("train"):
            # 取回 rollout 数据并按 DP rank 切分本 rank 份额
            rollout_data = process_rollout_data(self.args, rollout_data_ref, self.dp_rank, self.dp_size)
            # 仅调试 rollout 时只取数据不训练
            if self.args.debug_rollout_only:
                return
            # 进入训练主逻辑
            self._train_core(rollout_id=rollout_id, rollout_data=rollout_data)

        # 输出本 rollout 的性能指标（耗时、FLOPs 等）
        train_metric_utils.log_perf_data_raw(
            rollout_id=rollout_id,
            args=self.args,
            is_primary_rank=dist.get_rank() == 0,
            compute_total_fwd_flops=None,
        )

    # 记录 rollout 数据的统计指标（平均奖励、各概率量与 DP 组均值等）
    def _log_rollout_data(self, rollout_id: int, rollout_data, packed_batches):
        # 待上报的日志字典
        log_dict = {}
        # rank0 且有原始奖励时计算平均 raw reward
        if "raw_reward" in rollout_data and dist.get_rank() == 0:
            raw_reward_list = rollout_data["raw_reward"]
            if raw_reward_list:
                # 简单算术平均
                log_dict["rollout/raw_reward"] = sum(raw_reward_list) / len(raw_reward_list)

        # 逐指标统计（按 loss mask 加权的均值）
        for metric_key in ["log_probs", "rollout_log_probs", "ref_log_probs", "advantages", "returns"]:
            # 批次中不存在该指标则跳过
            if metric_key not in packed_batches[0]:
                continue
            # 累加器（GPU 张量，便于 all_reduce）
            val = torch.tensor([0.0], device=torch.cuda.current_device())
            # 遍历所有打包批次，反解回序列级再累加
            for _mbs_id, batches in enumerate(packed_batches):
                # 反解打包批次
                unpacked_batches = unpack_sequences(batches)
                for unpacked_batch in unpacked_batches:
                    if isinstance(unpacked_batch[metric_key], torch.Tensor):
                        # 张量指标：与 loss mask 逐元素相乘求和，再除以 mask 总数（clamp 防零）
                        loss_masks_tensor = unpacked_batch["loss_masks"].to(device=torch.cuda.current_device())
                        metric_tensor = unpacked_batch[metric_key].to(device=torch.cuda.current_device())
                        val += (metric_tensor * loss_masks_tensor).sum() / loss_masks_tensor.sum().clamp_min(1)
                    else:
                        # 标量指标直接累加
                        val += unpacked_batch[metric_key]
            # 跨 DP 组求和
            dist.all_reduce(val, op=dist.ReduceOp.SUM, group=self.dp_group)
            # 除以全局样本数得到平均并记录
            log_dict[f"rollout/{metric_key}"] = (
                val / (self.args.n_samples_per_prompt * self.args.rollout_batch_size)
            ).item()
        # rank0 负责输出日志
        if dist.get_rank() == 0:
            logger.info(f"rollout {rollout_id}: {log_dict}")
            # 附上 rollout 全局步数并写入跟踪后端
            log_dict["rollout/step"] = compute_rollout_step(self.args, rollout_id)
            logging_utils.log(self.args, log_dict, step_key="rollout/step")

        # CI 测试且开启 true on-policy 时，训练前向与 rollout 前向的 log_probs 必须逐位一致
        if self.args.ci_test and self.args.true_on_policy_mode:
            assert log_dict["rollout/log_probs"] == log_dict["rollout/rollout_log_probs"], (
                f"CI check failed: true_on_policy_mode is enabled, but log_probs "
                f"({log_dict['rollout/log_probs']}) != rollout_log_probs "
                f"({log_dict['rollout/rollout_log_probs']})"
            )

    # 训练主逻辑：优势估计 -> 打包 -> 前向统计 -> 逐 micro-batch 训练
    def _train_core(self, rollout_id: int, rollout_data) -> None:
        # 目前仅支持 GRPO/GSPO 两类优势估计器（组相对优势，论文概念：GRPO = Group Relative Policy Optimization）
        if self.args.advantage_estimator in ["grpo", "gspo"]:
            # GRPO/GSPO：用序列级标量奖励按响应长度广播成逐 token 的优势/回报（组内归一化在奖励侧已完成）
            rollout_data["advantages"] = rollout_data["returns"] = [
                torch.tensor([rollout_data["rewards"][i]] * rollout_data["response_lengths"][i])
                for i in range(len(rollout_data["rewards"]))
            ]
        else:
            # 其他优势估计器尚未在 FSDP 后端实现
            raise NotImplementedError(f"Unsupported advantage_estimator {self.args.advantage_estimator}")

        # 打包序列并获取梯度累积边界
        packed_batches, grad_accum = self._packed_data(rollout_data)

        # 校验累积边界非空（batch 与 micro-batch 配置不当时会得到空列表）
        assert (
            len(grad_accum) > 0
        ), f"Invalid grad_accum {grad_accum} for micro_batch_size {self.args.micro_batch_size} and global_batch_size {self.args.global_batch_size}"

        if self.ref_model is not None:
            # 存在 ref 模型时先计算参考策略的 token 对数概率（用于 KL 惩罚）
            self._compute_log_prob("ref", packed_batches, store_prefix="ref_")

        # 计算 actor 当前策略的 token 对数概率与熵
        self._compute_log_prob("actor", packed_batches)
        # 输出 rollout 统计指标
        self._log_rollout_data(rollout_id, rollout_data, packed_batches)

        # 训练阶段计时
        with timer("actor_train"):
            # 累积的指标缓冲（每个 optimizer step 结束时聚合输出）
            reported_accum: dict[str, list[torch.Tensor]] = {}
            # 先清一次梯度（set_to_none 释放显存）
            self.optimizer.zero_grad(set_to_none=True)
            # 逐 micro-batch 训练（带剖析打点与进度条）
            for mbs_id, packed_batch in self.prof.iterate_train_actor(
                enumerate(tqdm(packed_batches, desc="actor_train", disable=dist.get_rank() != 0))
            ):
                # 执行单个 micro-batch 的前向+反向
                self._train_step(
                    packed_batch=packed_batch,
                    reported_accum=reported_accum,
                    mbs_id=mbs_id,
                    grad_accum=grad_accum,
                )

        # 通知剖析器本 rollout 的训练结束
        self.prof.step(rollout_id=rollout_id)

        # 按需保存调试用训练数据（如 dump 到磁盘供排查）
        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        # Update ref model if needed (copy actor weights to ref)
        # 按配置的更新周期，把 actor 权重同步给 ref 模型（ref 是 actor 的历史快照）
        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and self.ref_model is not None
        ):
            if dist.get_rank() == 0:
                logger.info(f"Updating ref model at rollout_id {rollout_id}")
            # Copy actor model state to ref model
            # 取 actor 的 state dict 加载进 ref 模型
            actor_state = self.model.state_dict()
            self.ref_model.load_state_dict(actor_state)
            # ref 模型平时放 CPU 省显存
            self.ref_model.cpu()

    # 单个 micro-batch 的前向、策略损失计算与反向传播（核心训练步）
    def _train_step(self, packed_batch, reported_accum, mbs_id, grad_accum):
        # Prepare model inputs
        # 构造模型输入并前向：squeeze batch 维、转 fp32
        model_args = self._get_model_inputs_args(packed_batch)
        logits = self.model(**model_args).logits.squeeze(0).float()

        # Compute log probs and entropy
        # 由 logits 计算当前策略的对数概率与熵
        log_probs, entropy_result = get_logprob_and_entropy(
            logits=logits,
            target_tokens=packed_batch["tokens"],
            allow_compile=not self.args.true_on_policy_mode,
            temperature=self.args.rollout_temperature,
        )
        # 记录当前对数概率与熵到批次
        packed_batch["cur_log_probs"] = log_probs
        packed_batch["entropy"] = entropy_result

        # 反解打包批次，得到序列级的张量列表
        unpacked_batches = unpack_sequences(packed_batch)

        # 旧策略对数概率的来源：优先用 rollout 时的（省一次前向），否则用训练前计算的 log_probs
        old_log_prob_key = "rollout_log_probs" if self.args.use_rollout_logprobs else "log_probs"
        # 检查每个反解批次中旧对数概率是否存在且为张量
        missing_old_log_probs = [
            idx
            for idx, batch in enumerate(unpacked_batches)
            if old_log_prob_key not in batch or not isinstance(batch[old_log_prob_key], torch.Tensor)
        ]
        if missing_old_log_probs:
            # 缺失时直接报错，避免静默训练出错误结果
            raise KeyError(
                f"{old_log_prob_key} must be provided as torch.Tensor for all microbatches when "
                f"use_rollout_logprobs is set to {self.args.use_rollout_logprobs}. Missing in batches: {missing_old_log_probs}"
            )
        # 拼接所有序列的旧对数概率、当前对数概率与优势
        old_log_probs = torch.cat([batch[old_log_prob_key] for batch in unpacked_batches], dim=0)
        log_probs = torch.cat([batch["cur_log_probs"] for batch in unpacked_batches], dim=0)
        advantages = torch.cat([batch["advantages"] for batch in unpacked_batches], dim=0)
        # 收集各序列的 loss mask 并搬到 log_probs 所在设备
        loss_masks = [batch["loss_masks"].to(device=log_probs.device) for batch in unpacked_batches]
        # 收集各序列的响应长度
        response_lengths = [batch["response_lengths"] for batch in unpacked_batches]

        # 统一设备
        advantages = advantages.to(device=log_probs.device)
        old_log_probs = old_log_probs.to(device=log_probs.device)
        # 重要性采样比的对数形式（论文概念：log importance ratio = log π_new - log π_old）
        ppo_kl = old_log_probs - log_probs

        if self.args.use_opsm:
            # OPSM（on-policy 序列掩码）：找出新旧策略概率分布形态不一致的 token 并屏蔽
            opsm_mask, opsm_clipfrac = compute_opsm_mask(
                args=self.args,
                full_log_probs=[batch["cur_log_probs"] for batch in unpacked_batches],
                full_old_log_probs=[batch[old_log_prob_key] for batch in unpacked_batches],
                advantages=[batch["advantages"] for batch in unpacked_batches],
                loss_masks=loss_masks,
            )

        if self.args.advantage_estimator == "gspo":
            # GSPO（Group Segment Policy Optimization）：以序列为粒度重算 KL 比率
            ppo_kl = compute_gspo_kl(
                full_log_probs=[batch["cur_log_probs"] for batch in unpacked_batches],
                full_old_log_probs=[batch[old_log_prob_key] for batch in unpacked_batches],
                local_log_probs=[batch["cur_log_probs"] for batch in unpacked_batches],
                loss_masks=loss_masks,
            )

        # 计算策略梯度损失（PPO 风格 clip 目标；论文概念：PPO clipped surrogate objective）
        pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, self.args.eps_clip, self.args.eps_clip_high)

        if self.args.use_opsm:
            # 应用 OPSM 掩码：只保留分布一致的 token 的损失
            pg_loss = pg_loss * opsm_mask

        # 判断某批次是否带有非空的 rollout 对数概率
        def _has_rollout_log_probs(batch) -> bool:
            rollout_tensor = batch.get("rollout_log_probs")
            return isinstance(rollout_tensor, torch.Tensor) and rollout_tensor.numel() > 0

        # 所有序列都带 rollout 对数概率时才拼接（用于对比诊断）
        has_rollout_log_probs = all(_has_rollout_log_probs(batch) for batch in unpacked_batches)
        rollout_log_probs = (
            torch.cat([batch["rollout_log_probs"] for batch in unpacked_batches], dim=0)
            if has_rollout_log_probs
            else None
        )

        if self.args.calculate_per_token_loss:
            # 按 token 归一模式：把逐 token 量按响应长度切分后直接求和（总损失除以总 token 数在别处完成）
            pg_loss = sum_of_token(pg_loss, response_lengths, loss_masks)
            pg_clipfrac = sum_of_token(pg_clipfrac, response_lengths, loss_masks)
            ppo_kl = sum_of_token(ppo_kl.abs(), response_lengths, loss_masks)
        else:
            # 按样本归一模式（默认）：每条样本取其响应段的均值再求和
            pg_loss = sum_of_sample_mean(pg_loss, response_lengths, loss_masks)
            pg_clipfrac = sum_of_sample_mean(pg_clipfrac, response_lengths, loss_masks)
            ppo_kl = sum_of_sample_mean(ppo_kl.abs(), response_lengths, loss_masks)

        # Only compare rollout vs. train log probs when they originate from different stages.
        # 训练与 rollout 分属不同前向时，统计两者对数概率的绝对差（衡量训练/推理一致性）
        train_rollout_logprob_abs_diff = None
        if not self.args.use_rollout_logprobs and rollout_log_probs is not None:
            train_rollout_logprob_abs_diff = (old_log_probs - rollout_log_probs).abs()
            # 同样按样本均值归约并 detach（仅作指标不上传梯度）
            train_rollout_logprob_abs_diff = sum_of_sample_mean(
                train_rollout_logprob_abs_diff, response_lengths, loss_masks
            ).detach()

        # 拼接熵并计算熵损失（鼓励探索的熵正则项）
        entropy = torch.cat([batch["entropy"] for batch in unpacked_batches], dim=0)
        entropy_loss = sum_of_sample_mean(entropy, response_lengths, loss_masks)

        # 总损失 = 策略梯度损失 - 熵系数 x 熵损失
        loss = pg_loss - self.args.entropy_coef * entropy_loss

        if self.args.use_kl_loss:
            # KL 损失分支：把当前策略拉向参考策略（论文概念：KL penalty / KL regularization）
            ref_log_probs = torch.cat([batch["ref_log_probs"] for batch in unpacked_batches], dim=0)
            # 重要性比率默认为 None
            importance_ratio = None
            if self.args.use_unbiased_kl:
                # 无偏 KL 估计需要重要性比率 exp(log π - log π_old)
                importance_ratio = torch.exp(log_probs - old_log_probs)
            # 计算近似 KL（k3 估计或按类型选择）
            kl = compute_approx_kl(
                log_probs,
                ref_log_probs,
                kl_loss_type=self.args.kl_loss_type,
                importance_ratio=importance_ratio,
            )
            # KL 损失按样本均值归约
            kl_loss = sum_of_sample_mean(kl, response_lengths, loss_masks)

            # 加入总损失
            loss = loss + self.args.kl_loss_coef * kl_loss

        # 组装需要上报的指标（全部 detach，仅作日志）
        reported = {
            "loss": loss.detach(),
            "pg_loss": pg_loss.detach(),
            "pg_clipfrac": pg_clipfrac.detach(),
            "ppo_kl": ppo_kl.detach(),
            "entropy_loss": entropy_loss.detach(),
        }

        if train_rollout_logprob_abs_diff is not None:
            # 附上训练/rollout 一致性差异指标
            reported["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff

        if self.args.use_kl_loss:
            # 附上 KL 损失
            reported["kl_loss"] = kl_loss.detach()

        if self.args.use_opsm:
            # 附上 OPSM 的裁剪比例
            reported["opsm_clipfrac"] = opsm_clipfrac

        # Scale loss for gradient accumulation
        # 为梯度累积缩放损失：除以全局 batch 大小、乘以 DP 规模（各 rank 各贡献一份梯度）
        loss = loss * self.dp_size / self.args.global_batch_size
        # 反向传播
        loss.backward()

        # Accumulate reported metrics (store tensors for later mean)
        # 把指标缓存到累积字典，等 optimizer step 时统一聚合
        for k, v in reported.items():
            reported_accum.setdefault(k, []).append(v)

        if (mbs_id + 1) in grad_accum:
            # 到达梯度累积边界：执行优化器更新
            # TODO: check if the grad norm is global grad norm.
            # 梯度裁剪（论文概念：gradient clipping，防止梯度爆炸）
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            # the grad norm used to be of DTensor
            # 转成 Python float（历史上曾是 DTensor）
            grad_norm = float(grad_norm)

            # 优化器步进（AdamW 参数更新）
            self.optimizer.step()
            # Update learning rate
            # 更新学习率调度器
            self.lr_scheduler.step()
            # 清零梯度准备下一轮累积
            self.optimizer.zero_grad(set_to_none=True)
            # Aggregate logs
            # 聚合本步内累积的指标：堆叠求和
            aggregated = {k: torch.stack(v).sum().item() for k, v in reported_accum.items()}
            # TODO: change this, this is slow.
            # 跨 DP 组收集各 rank 的聚合结果（对象级 all_gather）
            reduced_aggregated = [None] * self.dp_size
            dist.all_gather_object(reduced_aggregated, aggregated, group=self.dp_group)
            # 汇总各 rank 的指标并按全局 batch 归一
            aggregated = {}
            for k in reported_accum.keys():
                aggregated[k] = sum([r[k] for r in reduced_aggregated]) / (self.args.global_batch_size)
            # 清空累积缓冲
            reported_accum.clear()
            if dist.get_rank() == 0:
                # rank0 负责打日志
                log_dict = {
                    f"train/{k}": (val.item() if torch.is_tensor(val) else val) for k, val in aggregated.items()
                }
                # 附上梯度范数
                log_dict["train/grad_norm"] = grad_norm

                # Log learning rate per parameter group; use scheduler's last computed LRs
                # 记录每个参数组的当前学习率
                lr_values = self.lr_scheduler.get_last_lr()
                for gid, _group in enumerate(self.optimizer.param_groups):
                    log_dict[f"train/lr-pg_{gid}"] = lr_values[gid]

                kl_info = ""
                if self.args.use_kl_loss and "kl_loss" in aggregated:
                    # 输出 KL 损失与等效 KL 惩罚系数
                    kl_info = f", kl_loss: {aggregated['kl_loss']:.4f}, kl_penalty: {aggregated['kl_loss'] * self.args.kl_loss_coef:.4f}"
                    logger.info(kl_info)
                # 打印本训练步的完整指标
                logger.info(f"step {self.global_step}: {log_dict}")

                # 附上训练步数并写入跟踪后端
                log_dict["train/step"] = self.global_step
                logging_utils.log(self.args, log_dict, step_key="train/step")
            # 全局训练步计数加一
            self.global_step += 1

    # 把训练后的 actor 权重同步到 rollout（推理）引擎
    @timer
    def update_weights(self) -> None:  # type: ignore[override]
        """Synchronize actor weights to rollout engines.

        Handles both colocated and distributed update modes. In offload mode,
        wakes up parameters as needed to perform the update.
        """
        # 纯训练调试或纯 rollout 调试模式下不同步权重
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        # 向 rollout 管理器请求引擎列表、引擎锁与新增引擎数
        rollout_engines, rollout_engine_lock, num_new_engines = ray.get(
            self.rollout_manager.get_rollout_engines_and_lock.remote()
        )
        if num_new_engines > 0:
            # 有新引擎加入（弹性扩容）时重新建立连接
            self.weight_updater.connect_rollout_engines(rollout_engines, rollout_engine_lock)
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                # rank0 清除新引擎标记
                ray.get(self.rollout_manager.clear_num_new_engines.remote())

        # 执行实际权重同步
        self.weight_updater.update_weights()

        if self.args.ci_test and len(rollout_engines) > 0:
            # CI 测试：随机挑一个引擎校验权重版本一致
            engine = random.choice(rollout_engines)
            engine_version = ray.get(engine.get_weight_version.remote())
            if str(engine_version) != str(self.weight_updater.weight_version):
                raise RuntimeError(
                    f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
                )

        # 清理显存缓存
        clear_memory()

    # 创建独立的参考模型（FSDP2 + CPUOffloadPolicy，常驻 CPU）
    def _create_ref_model(self, ref_load_path: str | None):
        """Create and initialize a separate reference model with FSDP2 CPUOffloadPolicy.

        Parameters:
            ref_load_path: Path to a directory containing a HF checkpoint. If
                None, a ValueError is raised.

        Returns:
            FSDP2-wrapped ref model with CPU offload enabled

        Note:
            Creates a separate FSDP2 model instance for the reference model.
            ALWAYS uses CPUOffloadPolicy for the reference model to save memory,
            regardless of the actor model's CPU offload setting.
        """
        # 未提供加载路径则报错
        if ref_load_path is None:
            raise ValueError("ref_load_path must be provided when loading reference model")

        if os.path.isdir(ref_load_path):
            # 支持从 HF checkpoint 目录加载
            logger.info(f"[Rank {dist.get_rank()}] Creating separate ref model from {ref_load_path}")

            # 获取初始化上下文（rank0 CPU / 其他 meta）
            init_context = self._get_init_weight_context_manager()

            # 在上下文中加载 ref 模型
            with init_context():
                ref_model = self.get_model_cls().from_pretrained(
                    ref_load_path,
                    trust_remote_code=True,
                    attn_implementation=self.args.attn_implementation,
                )

            # 取出完整 state dict
            full_state = ref_model.state_dict()

            # Always use CPUOffloadPolicy for reference, let FSDP2 handle the offload. It is faster than model.cpu().
            # ref 模型始终启用 CPU 卸载策略（FSDP2 原生 offload 比 model.cpu() 更快）
            ref_model = apply_fsdp2(ref_model, mesh=self.dp_mesh, cpu_offload=True, args=self.args)
            # 加载并广播完整 state dict
            ref_model = self._fsdp2_load_full_state_dict(ref_model, full_state, self.dp_mesh, cpu_offload=True)

            logger.info(f"[Rank {dist.get_rank()}] Reference model created with FSDP2 CPUOffloadPolicy")
            return ref_model
        else:
            # 单文件 checkpoint 暂不支持
            raise NotImplementedError(f"Loading from checkpoint file {ref_load_path} not yet implemented")

    # 由打包批次构造 HF 模型的输入参数字典
    def _get_model_inputs_args(self, packed_sequence: dict) -> dict:
        # tokens 变为 [1, T] 的 input_ids（模型要求 batch 维）
        input_ids = packed_sequence["tokens"].unsqueeze(0)
        # 位置 id 同样加 batch 维
        position_ids = packed_sequence["position_ids"].unsqueeze(0)

        # 组装基础输入；attention_mask 传 None，配合 varlen attention（cu_seqlens 路径）
        model_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": None,
        }
        if packed_sequence.get("multimodal_train_inputs"):
            # 多模态输入（如 pixel_values）直接合并进模型参数
            model_args.update(packed_sequence["multimodal_train_inputs"])
        return model_args


# 朴素版 selective log-softmax：对目标 token 直接收集对数概率
def selective_log_softmax_raw(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Fused version of the common `log_softmax -> gather` operation.

    The fused version of this operation avoids the (potentially large) memory overhead
    of allocating a new tensor to store the full logprobs.

    Parameters:
        logits: Tensor of shape [..., V] containing model logits.
        input_ids: Tensor of shape [...] of token indices whose log-probabilities are gathered.

    Returns:
        Tensor of shape [...] containing the log-probabilities corresponding to `input_ids`.
    """
    # 沿词表维做 log_softmax
    logprobs = logits.log_softmax(dim=-1)
    # 用 gather 只取出目标 token 的对数概率
    return torch.gather(logprobs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)


# 编译版 selective log-softmax：torch.compile 动态 shape 加速
selective_log_softmax_compiled = torch.compile(dynamic=True)(selective_log_softmax_raw)


# 为打包（varlen）序列收集下一 token 的对数概率
def gather_log_probs_packed(
    shifted_logits: torch.Tensor,
    input_ids: torch.Tensor,
    allow_compile: bool,
    cu_seqlens: torch.Tensor | float | None = None,
    temperature: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather next-token log probabilities for packed sequences.

    Parameters:
        logits: Model logits of shape [B, T, V] or [T, V].
        input_ids: Token ids of shape [B, T] or [T].
        cu_seqlens: Optional cumulative sequence lengths (unused here). Present
            for API compatibility with callers.

    Returns:
        A tensor of shape [T-1] (or [B, T-1]) with log-probabilities of targets.
    """
    # Handle batch dimension - logits should be [batch_size, seq_len, vocab_size]
    # 带批次维时先 squeeze 掉（打包序列恒为单条摊平序列）
    if shifted_logits.dim() == 3:
        # Remove batch dimension for packed sequences
        shifted_logits = shifted_logits.squeeze(0)
        input_ids = input_ids.squeeze(0)

    if temperature is not None:
        # 有温度系数时先把 logits 除以温度（温度采样，temperature scaling）
        shifted_logits = shifted_logits.div(temperature)

    # 目标 token 为右移一位后的序列：位置 t 的 logits 预测位置 t+1 的 token
    targets = input_ids[1:].to(device=shifted_logits.device)

    # Gather log probs for targets
    # 允许编译时用编译版（更快），否则用朴素版
    selective_log_softmax = selective_log_softmax_compiled if allow_compile else selective_log_softmax_raw
    return selective_log_softmax(shifted_logits, targets)


# 一次性计算对数概率与熵
def get_logprob_and_entropy(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    allow_compile: bool,
    temperature: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log probabilities and entropy.

    Parameters:
        logits: Model output logits with shape [seq_len, vocab_size]
        target_tokens: Target tokens with shape [seq_len]
        allow_compile: Whether to allow compilation
        temperature: Temperature parameter (optional)

    Returns:
        log_probs: Log probabilities with shape [seq_len - 1]
        entropy: Entropy with shape [seq_len - 1]
    """
    # 左移 logits：位置 t 的 logits 预测 t+1 的 token，故去掉最后一个位置
    shifted_logits = logits[:-1, :]
    # 收集目标 token 的对数概率
    log_probs = gather_log_probs_packed(
        shifted_logits, target_tokens, allow_compile=allow_compile, temperature=temperature
    )
    # 计算完整对数概率分布（用于熵）
    log_probs_full = torch.log_softmax(shifted_logits, dim=-1)
    # 计算概率分布
    probs = torch.softmax(shifted_logits, dim=-1)
    # 熵 = -sum p * log p（论文概念：policy entropy，衡量策略随机性）
    entropy = -(probs * log_probs_full).sum(dim=-1)
    # 返回对数概率与熵
    return log_probs, entropy


# 按样本均值求和：每条样本取响应段内（按 loss mask 加权）的均值再相加
def sum_of_sample_mean(x: torch.Tensor, response_lengths: list[int], loss_masks: list[torch.Tensor]) -> torch.Tensor:
    """Compute sum of per-sample means across variable-length responses.

    Parameters:
        x: Flat tensor containing concatenated per-token values across samples.
        response_lengths: Lengths of each sample's response segment in `x`.
        loss_masks: Per-sample masks aligned with `response_lengths`.

    Returns:
        A scalar tensor equal to the sum over samples of the mean value within
        each sample's response segment.
    """
    # 按响应长度切分 x，逐样本计算 mask 加权均值后求和；clamp 防止除零
    return sum(
        [
            (x_i * loss_mask_i).sum() / torch.clamp_min(loss_mask_i.sum(), 1)
            for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks, strict=False)
        ]
    )


# 把优化器状态（动量等张量）整体搬到目标设备（CPU/GPU）
@torch.no_grad()
def move_torch_optimizer(optimizer, device):
    """ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py"""
    # 优化器尚无状态（未 step 过）则无事可做
    if not optimizer.state:
        return

    # 遍历每个参数组中的每个参数
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            # 取该参数的优化器状态字典
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    # 把状态里的张量搬到目标设备（非阻塞传输）
                    state[key] = value.to(device, non_blocking=True)

    # CUDA 同步确保搬运完成
    torch.cuda.synchronize()


# 给模型应用 FSDP2（fully_shard）：按 DP mesh 分片，配置混合精度与可选 CPU 卸载
def apply_fsdp2(model, mesh=None, cpu_offload=False, args=None):
    """Apply FSDP v2 to the model.

    Args:
        model: The model to wrap with FSDP
        mesh: Optional DeviceMesh for FSDP. If None, uses all ranks.
        cpu_offload: If True, offload parameters, gradients, and optimizer states
            to CPU. The optimizer step will run on CPU. (Default: False)
        args: Arguments containing precision settings (fp16/bf16)

    Ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py
    """
    # 延迟导入 FSDP2 相关 API
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    # 按需构造 CPU 卸载策略（None 表示不卸载）
    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    # 取出模型声明的不切分层类型（Transformer 层类名列表）
    layer_cls_to_wrap = model._no_split_modules
    assert len(layer_cls_to_wrap) > 0 and layer_cls_to_wrap[0] is not None

    # 找到所有需要分片的子模块：指定层类型，以及未绑定词嵌入时的 Embedding 层
    modules = [
        module
        for name, module in model.named_modules()
        if module.__class__.__name__ in layer_cls_to_wrap
        or (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings)
    ]

    # Determine precision policy based on args
    # 混合精度策略（论文概念：mixed precision training，参数 bf16、规约 fp32）
    param_dtype = torch.bfloat16  # Default to bf16 as before
    reduce_dtype = torch.float32

    if args.fp16:
        # 指定 fp16 时参数精度改为 fp16
        param_dtype = torch.float16

    # 记录精度策略
    logger.info(f"FSDP MixedPrecision Policy: param_dtype={param_dtype}, reduce_dtype={reduce_dtype}")

    # 组装 FSDP kwargs：精度策略、卸载策略与设备网格
    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        ),
        "offload_policy": offload_policy,
        "mesh": mesh,
    }

    # Apply FSDP to each module (offload_policy=None is equivalent to not passing it)
    # 先对每个子模块做 fully_shard（由内向外，使各层独立分片）
    for module in modules:
        fully_shard(module, **fsdp_kwargs)

    # Apply FSDP to the top-level model
    # 最后对顶层模型做 fully_shard（覆盖剩余参数）
    fully_shard(model, **fsdp_kwargs)

    # 返回包装后的模型
    return model


# 按 token 总量求和（不做样本均值）：把逐 token 量按响应长度切分后乘 mask 求和
def sum_of_token(x: torch.Tensor, response_lengths: list[int], loss_masks: list[torch.Tensor]) -> torch.Tensor:
    # 逐样本切分、乘 mask、求和后相加
    return sum(
        [
            (x_i * loss_mask_i).sum()
            for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks, strict=False)
        ]
    )

