import dataclasses
import gc
import logging
import math
import os
from argparse import Namespace
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import torch
from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.utils import get_model_config
from megatron.training.global_vars import get_args
from megatron.training.training import get_model

from slime.utils import logging_utils
from slime.utils.memory_utils import clear_memory

from .checkpoint import load_checkpoint, save_checkpoint
from .data import DataIterator, get_batch
from .loss import loss_function
from .model_provider import get_model_provider_func, wrap_model_provider_with_freeze

logger = logging.getLogger(__name__)


def get_optimizer_param_scheduler(args: Namespace, optimizer: MegatronOptimizer) -> OptimizerParamScheduler:
    """Create and configure the optimizer learning-rate/weight-decay scheduler.

    This configures iteration-based schedules derived from the global batch size
    and run-time arguments.

    Args:
        args (Namespace): Training/runtime arguments (argparse namespace).
        optimizer (MegatronOptimizer): Megatron optimizer bound to the model.

    Returns:
        OptimizerParamScheduler: Initialized scheduler bound to ``optimizer``.
    """
    # Iteration-based training.
    # 按 RL 训练的 rollout 总量折算出训练 iteration 数：样本总数 / 全局 batch
    args.train_iters = args.num_rollout * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    if args.lr_decay_iters is None:
        # 未显式指定学习率衰减长度时默认等于总训练步数
        args.lr_decay_iters = args.train_iters
    # 把 iteration 数换算成"处理的样本数"步长，供调度器使用
    lr_decay_steps = args.lr_decay_iters * args.global_batch_size
    # 权重衰减递增的总步数
    wd_incr_steps = args.train_iters * args.global_batch_size
    # wsd（warmup-stable-decay）调度风格的衰减起点，未配置则不启用
    wsd_decay_steps = None
    if args.lr_wsd_decay_iters is not None:
        wsd_decay_steps = args.lr_wsd_decay_iters * args.global_batch_size
    if args.lr_warmup_fraction is not None:
        # 以比例形式指定 warmup：占总衰减步数的比例
        lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
    else:
        # 以 iteration 数指定 warmup
        lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size

    # 构造学习率 / 权重衰减调度器
    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=args.lr_warmup_init,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=args.override_opt_param_scheduler,
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=args.lr_wsd_decay_style,
    )

    return opt_param_scheduler


def setup_model_and_optimizer(
    args: Namespace,
    role: str = "actor",
) -> tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler]:
    """Build model(s), wrap with DDP, and construct optimizer and scheduler.

    Args:
        args (Namespace): Training/runtime arguments (argparse namespace).
        role (str): Logical role of the model (e.g., "actor", "critic").
        no_wd_decay_cond (Callable[..., bool] | None): Predicate to exclude
            parameters from weight decay.
        scale_lr_cond (Callable[..., bool] | None): Predicate to scale LR for
            selected parameter groups.
        lr_mult (float): Global learning-rate multiplier for the optimizer.

    Returns:
        tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler]:
            - List of model chunks wrapped by ``DDP``.
            - The constructed ``MegatronOptimizer`` instance.
            - The learning-rate/weight-decay scheduler tied to the optimizer.
    """
    # 不支持 MoE upcycling（稀疏权重升杯）路径
    assert not args.moe_use_upcycling
    # 必须提供初始权重：HF load 路径或 Megatron 预训练 checkpoint 二选一
    assert args.load is not None or args.pretrained_checkpoint is not None

    # 构建模型：provider 外再包一层参数冻结逻辑，并声明模型类型
    model = get_model(
        wrap_model_provider_with_freeze(get_model_provider_func(args, role), args), ModelType.encoder_or_decoder
    )

    # Optimizer
    # 把命令行中与 OptimizerConfig 字段同名的参数透传给优化器配置
    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    # 训练循环自行计时，优化器内部不需要 timers
    config.timers = None

    # 用 Megatron 的优化器工厂构造优化器（含分布式优化器等选项）
    optimizer = get_megatron_optimizer(
        config=config,
        model_chunks=model,
        use_gloo_process_groups=args.enable_gloo_process_groups,
    )
    # 绑定学习率 / 权重衰减调度器
    opt_param_scheduler = get_optimizer_param_scheduler(args, optimizer)
    return model, optimizer, opt_param_scheduler


def enable_forward_pre_hook(model_chunks: Sequence[DDP]) -> None:
    """Enable forward pre-hooks for provided DDP-wrapped model chunks.

    Args:
        model_chunks (Sequence[DDP]): Sequence of DDP modules to enable hooks on.
    """
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        # 开启前向 pre-hook：在前向之前执行参数/梯度的通信
        model_chunk.enable_forward_pre_hook()


def disable_forward_pre_hook(model_chunks: Sequence[DDP], param_sync: bool = True) -> None:
    """Disable forward pre-hooks for provided DDP-wrapped model chunks.

    Args:
        model_chunks (Sequence[DDP]): Sequence of DDP modules to disable hooks on.
        param_sync (bool): Whether to synchronize parameters when disabling.
    """
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        # 关闭前向 pre-hook；param_sync 决定关闭时是否顺带同步一次参数
        model_chunk.disable_forward_pre_hook(param_sync=param_sync)


@torch.no_grad()
def forward_only(
    f: Callable[..., dict[str, list[torch.Tensor]]],
    args: Namespace,
    model: Sequence[DDP],
    data_iterator: Sequence[DataIterator],
    num_microbatches: Sequence[int],
    store_prefix: str = "",
) -> dict[str, list[torch.Tensor]]:
    """Run forward passes only and collect non-loss outputs (e.g., logprobs).

    The model is put into evaluation mode, a forward-only pipeline pass is
    executed, and relevant outputs are aggregated and returned.

    Args:
        f (Callable[..., dict[str, list[torch.Tensor]]]): Post-forward callback used to
            compute and package outputs to collect. This should accept a logits
            tensor as its first positional argument and additional keyword-only
            arguments; see ``get_log_probs_and_entropy``/``get_values`` in
            ``megatron_utils.loss`` for examples. It will be partially applied
            so that the callable returned from the internal forward step only
            requires the logits tensor.
        args (Namespace): Runtime arguments.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding batches for inference.
        num_microbatches (Sequence[int]): Number of microbatches per rollout step.
        store_prefix (str): Prefix to prepend to stored output keys.

    Returns:
        dict[str, list[torch.Tensor]]: Aggregated outputs keyed by ``store_prefix + key``.
    """

    # reset data iterator
    # 重置各数据迭代器，保证每次 forward 从头消费本轮 rollout 数据
    for iterator in data_iterator:
        iterator.reset()

    # 取第一个模型分片的 transformer 配置（pipeline 引擎需要）
    config = get_model_config(model[0])

    def forward_step(
        data_iterator: DataIterator, model: GPTModel, return_schedule_plan: bool = False
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor], dict[str, list[torch.Tensor]]]]:
        """Forward step used by Megatron's pipeline engine.

        Args:
            data_iterator (DataIterator): Input data iterator.
            model (GPTModel): The GPT model chunk to execute.

        Returns:
            tuple[torch.Tensor, Callable[[torch.Tensor], dict[str, list[torch.Tensor]]]]:
            Output tensor(s) and a callable that computes and packages results
            to be collected by the engine.
        """

        # 纯前向不需要组合式 1F1B 的调度计划
        assert not return_schedule_plan, "forward_only step should never return schedule plan"

        # Get the batch.
        # 从数据迭代器取本 micro-batch 所需字段
        batch = get_batch(
            data_iterator,
            [
                "tokens",
                "loss_masks",
                "multimodal_train_inputs",
                "total_lengths",
                "response_lengths",
                "max_seq_lens",
            ],
            args.data_pad_size_multiplier,
            args.qkv_format,
            args.allgather_cp,
        )
        # 取出各批字段：未拼接的原始 token 序列、token id、序列打包参数等
        unconcat_tokens = batch["unconcat_tokens"]
        tokens = batch["tokens"]
        packed_seq_params = batch["packed_seq_params"]
        total_lengths = batch["total_lengths"]
        response_lengths = batch["response_lengths"]
        # 纯前向：labels 传 None，由外层回调基于输出 logits 计算 logprobs 等
        output_tensor = model(
            input_ids=tokens,
            position_ids=None,
            attention_mask=None,
            labels=None,
            packed_seq_params=packed_seq_params,
            loss_mask=batch["full_loss_masks"],
            **(batch["multimodal_train_inputs"] if batch["multimodal_train_inputs"] is not None else {}),
        )

        # 用 partial 预绑定回调参数：引擎只回调时传入 logits 一个参数
        return output_tensor, partial(
            f,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=args.use_rollout_entropy,
            max_seq_lens=batch.get("max_seq_lens", None),
        )

    # Turn on evaluation mode which disables dropout.
    # 评估模式：关闭 dropout，保证 logprob 计算确定
    for model_module in model:
        model_module.eval()

    if args.custom_megatron_before_log_prob_hook_path:
        # 支持用户自定义钩子：在计算 logprob 前对模型做干预
        from slime.utils.misc import load_function

        custom_before_log_prob_hook = load_function(args.custom_megatron_before_log_prob_hook_path)
        custom_before_log_prob_hook(args, model, store_prefix)

    # 获取 Megatron 当前并行模式的 forward-backward 引擎函数
    forward_backward_func = get_forward_backward_func()
    # Don't care about timing during evaluation
    # 评估阶段不关心计时
    config.timers = None
    forward_data_store = []
    # 本 rollout 需要执行的步数（每步若干 micro-batch）
    num_steps_per_rollout = len(num_microbatches)
    for step_id in range(num_steps_per_rollout):
        # 逐步执行纯前向流水线，收集每步的输出
        forward_data_store += forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_microbatches[step_id],
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            forward_only=True,
        )

    # Move model back to the train mode.
    # 恢复训练模式（重新启用 dropout）
    for model_module in model:
        model_module.train()

    rollout_data = {}
    # Store the results on the last stage
    # 仅流水线最后一级持有完整输出，在这里汇总
    if mpu.is_pipeline_last_stage():
        keys = forward_data_store[0].keys()
        for key in keys:
            # 把各步输出的列表按序拼接
            values = []
            for value in forward_data_store:
                assert isinstance(value[key], list)
                values += value[key]

            if args.use_dynamic_batch_size:
                # TODO: This is ugly... Find a better way to make the data have the same order.
                # TODO: move this out of the loop.
                # 动态 batch 时引擎内部重排过顺序，这里按原始 micro-batch 索引还原顺序
                origin_values = [None] * len(values)
                origin_indices = sum(data_iterator[0].micro_batch_indices, [])
                for value, origin_index in zip(values, origin_indices, strict=False):
                    origin_values[origin_index] = value
                values = origin_values
            # 以（可选前缀 +）键名存入最终结果
            rollout_data[f"{store_prefix}{key}"] = values
    return rollout_data


def train_one_step(
    args: Namespace,
    rollout_id: int,
    step_id: int,
    data_iterator: Sequence[DataIterator],
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
    num_microbatches: int,
) -> tuple[dict[str, float], float]:
    """Execute a single pipeline-parallel training step.

    Runs forward/backward over ``num_microbatches``, applies optimizer step and
    one scheduler step when gradients are valid.

    Args:
        args (Namespace): Runtime arguments.
        rollout_id (int): Rollout identifier.
        step_id (int): Step index within the current rollout.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding training batches.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
        num_microbatches (int): Number of microbatches to process.

    Returns:
        tuple[dict[str, float], float]: Reduced loss dictionary (last stage only)
        and gradient norm for logging.
    """
    # 训练时统一从 Megatron 全局变量取最新 args
    args = get_args()

    # Set grad to zero.
    # 清零梯度缓冲与优化器梯度状态，准备本步的 backward
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    if args.custom_megatron_before_train_step_hook_path:
        # 支持用户自定义钩子：训练步开始前干预
        from slime.utils.misc import load_function

        custom_before_train_step_hook = load_function(args.custom_megatron_before_train_step_hook_path)
        custom_before_train_step_hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler)

    def forward_step(data_iterator: DataIterator, model: GPTModel, return_schedule_plan: bool = False) -> tuple[
        torch.Tensor,
        Callable[[torch.Tensor], tuple[torch.Tensor, int, dict[str, torch.Tensor | list[str]]]],
    ]:
        """Forward step used by Megatron's pipeline engine during training.

        Args:
            data_iterator (DataIterator): Input data iterator.
            model (GPTModel): The GPT model chunk to execute.

        Returns:
            tuple[torch.Tensor, Callable[[torch.Tensor], tuple[torch.Tensor, int, dict[str, torch.Tensor | list[str]]]]]:
            Output tensor(s) and the loss function, which returns
            (loss, num_elems, {"keys": list[str], "values": torch.Tensor}).
        """

        # Get the batch.
        # 取训练批：包含 token、损失掩码以及 RL 所需的 logprobs / 优势 / 回报等
        batch = get_batch(
            data_iterator,
            [
                "tokens",
                "multimodal_train_inputs",
                "packed_seq_params",
                "total_lengths",
                "response_lengths",
                "loss_masks",
                "log_probs",
                "ref_log_probs",
                "values",
                "advantages",
                "returns",
                "rollout_log_probs",
                "max_seq_lens",
                "teacher_log_probs",
            ],
            args.data_pad_size_multiplier,
            args.qkv_format,
            args.allgather_cp,
        )

        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            # MoE 路由重放（routing replay）：前向阶段把路由切换为重放模式，结束后恢复
            old_stage = os.environ["ROUTING_REPLAY_STAGE"]
            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"

        if return_schedule_plan:
            # 组合式 1F1B 需要先构建调度计划（不兼容 MTP 训练）
            assert not args.enable_mtp_training, "MTP training should not be enabled when using combined 1f1b"
            output_tensor = model.build_schedule_plan(
                input_ids=batch["tokens"],
                position_ids=None,
                attention_mask=None,
                labels=None,
                packed_seq_params=batch["packed_seq_params"],
                loss_mask=batch["full_loss_masks"],
            )
        else:
            # 普通前向：组装模型前向参数
            forward_kwargs = {
                "input_ids": batch["tokens"],
                "position_ids": None,
                "attention_mask": None,
                "labels": None,
                "packed_seq_params": batch["packed_seq_params"],
                "loss_mask": batch["full_loss_masks"],
            }

            if args.enable_mtp_training:
                # MTP 训练：以输入 token 作为多 token 预测的标签
                forward_kwargs["mtp_kwargs"] = {"mtp_labels": batch["tokens"]}

            if batch["multimodal_train_inputs"] is not None:
                # 多模态输入（图像特征等）合并进前向参数
                forward_kwargs.update(batch["multimodal_train_inputs"])

            output_tensor = model(**forward_kwargs)

        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            # 恢复路由重放阶段标记
            os.environ["ROUTING_REPLAY_STAGE"] = old_stage

        # 返回输出张量与预绑定的损失函数（loss_function 内部计算 RL 损失）
        return output_tensor, partial(loss_function, args, batch, num_microbatches)

    # Forward pass.
    # 获取流水线引擎并执行一次"前向 + 反向"
    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=num_microbatches,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False,
    )

    valid_step = True
    if not getattr(args, "check_for_nan_in_loss_and_grad", True):
        # 用户关闭内置 NaN 检查时，自行做梯度有效性检查
        found_inf_flag = optimizer.prepare_grads()
        if found_inf_flag:
            # 检测到 inf/nan 梯度：本步作废
            valid_step = False
        else:
            # 梯度范数异常（nan/inf）同样作废本步
            grad_norm = optimizer.get_grad_norm()
            if isinstance(grad_norm, torch.Tensor):
                valid_step = not (torch.isnan(grad_norm) or torch.isinf(grad_norm))
            else:
                valid_step = not (math.isnan(grad_norm) or math.isinf(grad_norm))

    # CI check: verify only MTP parameters have non-zero gradients when truncation happens
    # This check must happen before optimizer.step() as gradients may be modified during step
    # CI 测试：校验 MTP 截断场景下只有 MTP 参数有梯度（必须在 optimizer.step 之前）
    if args.ci_test and args.enable_mtp_training:
        from slime.backends.megatron_utils.ci_utils import check_mtp_only_grad

        check_mtp_only_grad(model, step_id)

    if valid_step:
        # Update parameters.
        # 本步有效才执行优化器更新
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()

        # Update learning rate.
        # 调度器前进一步（按全局 batch 大小递增处理样本数）
        assert update_successful
        opt_param_scheduler.step(increment=args.global_batch_size)

    # release grad
    # 释放本步梯度缓冲，避免显存占用
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        # 最后一级流水线上聚合各 micro-batch 的损失值
        keys = losses_reduced[0]["keys"]
        values = None
        for x in losses_reduced:
            if values is None:
                values = x["values"]
            else:
                # 逐 micro-batch 累加损失（首元素为样本/token 数）
                values += x["values"]
        assert len(keys) + 1 == values.numel()
        # 数据并行（含上下文并行组）上 all-reduce 求和
        torch.distributed.all_reduce(values, group=mpu.get_data_parallel_group(with_context_parallel=True))

        loss_reduced = {}
        # 转回 Python 列表方便计算
        values = values.tolist()
        # 第一个元素是样本数（或 token 数），用于求平均
        num_samples_or_tokens = values[0]
        for key, value in zip(keys, values[1:], strict=False):
            # 每个损失项除以数量得到均值（并校正上下文并行倍数）
            loss_reduced[key] = value * mpu.get_context_parallel_world_size() / num_samples_or_tokens
        return loss_reduced, grad_norm
    return {}, grad_norm


def should_disable_forward_pre_hook(args: Namespace) -> bool:
    """Block forward pre-hook for certain configurations."""
    # 分布式优化器 + 重叠参数 gather 时需要特殊管理 pre-hook
    return args.use_distributed_optimizer and args.overlap_param_gather


def train(
    rollout_id: int,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
    data_iterator: Sequence[DataIterator],
    num_microbatches: Sequence[int],
) -> None:
    """Run training over a rollout consisting of multiple steps.

    The model is switched to train mode, training hooks are configured, and
    ``train_one_step`` is invoked for each step in the rollout.

    Args:
        rollout_id (int): Rollout identifier.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding training batches.
        num_microbatches (Sequence[int]): Microbatches per step in the rollout.
    """
    args = get_args()

    # 重置数据迭代器，准备消费本 rollout 的训练数据
    for iterator in data_iterator:
        iterator.reset()

    # Turn on training mode which enables dropout.
    # 切换到训练模式（启用 dropout）
    for model_module in model:
        model_module.train()

    # Setup some training config params.
    # 配置流水线引擎运行时参数
    config = get_model_config(model[0])
    # 损失缩放函数交给优化器管理（fp16/bf16 混合精度）
    config.grad_scale_func = optimizer.scale_loss
    config.timers = None
    if isinstance(model[0], DDP) and args.overlap_grad_reduce:
        # 重叠梯度规约时不能使用自定义 no_sync
        assert config.no_sync_func is None, (
            "When overlap_grad_reduce is True, config.no_sync_func must be None; "
            "a custom no_sync_func is not supported when overlapping grad-reduce"
        )
        # 挂载各模型分片的 no_sync（延迟到窗口末再规约）
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.align_grad_reduce:
            # 对齐各分片的梯度规约时机
            config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.align_param_gather:
        # 对齐各分片的参数同步时机
        config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    # 梯度收尾函数（all-reduce / layernorm 梯度 allgather 等）
    config.finalize_model_grads_func = finalize_model_grads

    pre_hook_enabled = False

    if args.reset_optimizer_states:
        # 重置优化器状态（step 计数与动量缓存），常用于多轮 RL 中每轮重新计数
        if (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
        ):
            print("Reset optimizer states")
        for chained_optimizer in optimizer.chained_optimizers:
            for group in chained_optimizer.optimizer.param_groups:
                if "step" in group:
                    group["step"] = 0
            for state in chained_optimizer.optimizer.state.values():
                if "step" in state:
                    # 清零 step 计数（可能是张量或标量）
                    if isinstance(state["step"], torch.Tensor):
                        state["step"].zero_()
                    else:
                        state["step"] = 0
                if "exp_avg" in state:
                    # 清零一阶动量
                    state["exp_avg"].zero_()
                if "exp_avg_sq" in state:
                    # 清零二阶动量
                    state["exp_avg_sq"].zero_()

    if args.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        # 手动 GC：关闭默认垃圾回收并定时手动触发，避免各 rank GC 时机不一致造成的性能抖动
        assert args.manual_gc_interval >= 0, "Manual garbage collection interval should be larger than or equal to 0"
        gc.disable()
        gc.collect()

    # Disable forward pre-hook to start training to ensure that errors in checkpoint loading
    # or random initialization don't propagate to all ranks in first all-gather (which is a
    # no-op if things work correctly).
    # 训练开始前先关闭 pre-hook：防止 checkpoint 加载错误在首次 all-gather 时扩散到全部 rank
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model, param_sync=False)
        # Also remove param_sync_func temporarily so that sync calls made in
        # `forward_backward_func` are no-ops.
        # 暂时移除参数同步函数，使引擎内的同步调用变成空操作
        param_sync_func = config.param_sync_func
        config.param_sync_func = None
        pre_hook_enabled = False

    num_steps_per_rollout = len(num_microbatches)

    # Run training iterations till done.
    # 逐个执行本 rollout 的全部训练步
    for step_id in range(num_steps_per_rollout):

        # Run training step.
        # 执行单步训练，得到聚合损失与梯度范数
        loss_dict, grad_norm = train_one_step(
            args,
            rollout_id,
            step_id,
            data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            num_microbatches[step_id],
        )

        if step_id == 0:
            # Enable forward pre-hook after training step has successfully run. All subsequent
            # forward passes will use the forward pre-hook / `param_sync_func` in
            # `forward_backward_func`.
            # 首步成功后再恢复 pre-hook，保证后续前向走重叠参数同步路径
            if should_disable_forward_pre_hook(args):
                enable_forward_pre_hook(model)
                config.param_sync_func = param_sync_func
                pre_hook_enabled = True

        if args.enable_mtp_training:
            # MTP 训练：从 Megatron 的损失记录器取出 MTP 损失并跨 rank 归约
            from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

            # 按 micro-batch 数对 MTP 损失做平均
            mtp_loss_scale = 1 / num_microbatches[step_id]
            tracker = MTPLossLoggingHelper.tracker
            if "values" in tracker:
                values = tracker["values"]
                if tracker.get("reduce_group") is not None:
                    # 先在 reduce 组上求和
                    torch.distributed.all_reduce(values, group=tracker.get("reduce_group"))
                if tracker.get("avg_group") is not None:
                    # 再在 avg 组上求平均
                    torch.distributed.all_reduce(values, group=tracker["avg_group"], op=torch.distributed.ReduceOp.AVG)
                # here we assume only one mtp layer
                # 假设只有一个 MTP 层，缩放后取标量
                mtp_losses = (tracker["values"] * mtp_loss_scale).item()
                MTPLossLoggingHelper.clean_loss_in_tracker()

                # CI check: verify MTP loss is within expected bounds
                # CI 校验 MTP 损失在预期范围内
                if args.ci_test:
                    from slime.backends.megatron_utils.ci_utils import check_mtp_loss

                    check_mtp_loss(mtp_losses)

        # per train step log.
        # 仅在 DP0/TP0 且最后一级流水线的 rank 上打日志
        if (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
        ):
            # 折算全局累计步数
            accumulated_step_id = rollout_id * num_steps_per_rollout + step_id
            # 从模型上取角色名（actor/critic 等）
            role = getattr(model[0], "role", "actor")
            role_tag = "" if role == "actor" else f"{role}-"
            # 组装日志字典：张量取均值，标量原样
            log_dict = {
                f"train/{role_tag}{key}": val.mean().item() if isinstance(val, torch.Tensor) else val
                for key, val in loss_dict.items()
            }
            log_dict[f"train/{role_tag}grad_norm"] = grad_norm
            if args.enable_mtp_training:
                log_dict[f"train/{role_tag}mtp_loss"] = mtp_losses

            # 记录每个参数组当前学习率
            for param_group_id, param_group in enumerate(optimizer.param_groups):
                log_dict[f"train/{role_tag}lr-pg_{param_group_id}"] = opt_param_scheduler.get_lr(param_group)

            log_dict["train/step"] = accumulated_step_id
            # 写入日志后端（tensorboard/wandb 等）
            logging_utils.log(args, log_dict, step_key="train/step")

            if args.ci_test and not args.ci_disable_kl_checker:
                # CI 检查：首步的 PPO-KL 与裁剪比例应为 0（策略尚未更新）
                if step_id == 0 and "train/ppo_kl" in log_dict and "train/pg_clipfrac" in log_dict:
                    if args.multi_latent_attention:
                        # TODO: mla currently have non-zero kl, need further investigation
                        # MLA 架构目前存在非零 KL 数值噪声，放宽阈值
                        assert log_dict["train/ppo_kl"] < 1e-8, f"{log_dict=}"
                    else:
                        assert log_dict["train/ppo_kl"] == 0.0 and log_dict["train/pg_clipfrac"] == 0.0, f"{log_dict=}"
                if accumulated_step_id == 0 and "train/kl_loss" in log_dict:
                    # 全局第 0 步的 KL 损失应为 0
                    assert log_dict["train/kl_loss"] == 0.0, f"{log_dict=}"

            # 终端输出本步日志
            logger.info(f"{role_tag}step {accumulated_step_id}: {log_dict}")

            if args.ci_save_grad_norm is not None:
                # CI 模式：把梯度范数保存到文件供后续比对
                ci_save_grad_norm_path = args.ci_save_grad_norm.format(
                    role=role,
                    rollout_id=rollout_id,
                    step_id=step_id,
                )
                torch.save(grad_norm, ci_save_grad_norm_path)
            elif args.ci_load_grad_norm is not None:
                # CI 模式：加载参考梯度范数并做相对/绝对容差校验
                ci_load_grad_norm_path = args.ci_load_grad_norm.format(
                    role=role,
                    rollout_id=rollout_id,
                    step_id=step_id,
                )
                expected_grad_norm = torch.load(ci_load_grad_norm_path)
                assert math.isclose(
                    grad_norm,
                    expected_grad_norm,
                    rel_tol=0.01,
                    abs_tol=0.01,
                ), f"grad norm mismatch: {grad_norm} != {expected_grad_norm}"
    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    # 训练结束恢复关闭 pre-hook 状态
    if pre_hook_enabled:
        disable_forward_pre_hook(model)


def save(
    iteration: int, model: Sequence[DDP], optimizer: MegatronOptimizer, opt_param_scheduler: OptimizerParamScheduler
) -> None:
    """Persist a training checkpoint safely with forward hooks disabled.

    Args:
        iteration (int): Current global iteration number.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
    """
    args = get_args()
    # 保存 checkpoint 前关闭 pre-hook，避免通信干扰
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model)
    # 写入 Megatron 格式 checkpoint（模型 + 优化器 + 调度器）
    save_checkpoint(
        iteration,
        model,
        optimizer,
        opt_param_scheduler,
        num_floating_point_operations_so_far=0,
        checkpointing_context=None,
        train_data_iterator=None,
        preprocess_common_state_dict_fn=None,
    )
    if should_disable_forward_pre_hook(args):
        # 保存完成后再恢复 pre-hook
        enable_forward_pre_hook(model)


def save_hf_model(args, rollout_id: int, model: Sequence[DDP]) -> None:
    """Save Megatron model in HuggingFace format.

    Args:
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        rollout_id (int): Rollout ID for path formatting.
    """
    # 只有 DP0 + TP0 的 rank 负责打日志
    should_log = (
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
    )

    try:
        # 通过 megatron.bridge 把 Megatron 权重导出为 HF 格式
        from megatron.bridge import AutoBridge
        from slime.utils.megatron_bridge_utils import patch_megatron_model

        # 按 rollout id 格式化输出目录
        path = Path(args.save_hf.format(rollout_id=rollout_id))

        if should_log:
            logger.info(f"Saving model in HuggingFace format to {path}")

        # 从 HF checkpoint 建桥接器（用于读取模型配置）
        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)

        # 创建输出目录
        path.mkdir(parents=True, exist_ok=True)

        with patch_megatron_model(model):
            # 在补丁上下文中导出 HF 格式权重
            bridge.save_hf_pretrained(
                model,
                path=path,
            )

        if should_log:
            logger.info(f"Successfully saved HuggingFace model to {path}")
    except Exception as e:
        # 导出失败仅记录错误，不影响主训练流程
        if should_log:
            logger.error(f"Failed to save HuggingFace format: {e}")


def initialize_model_and_optimizer(
    args: Namespace, role: str = "actor"
) -> tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler, int]:
    """Initialize model(s), optimizer, scheduler, and load from checkpoint.

    Args:
        args (Namespace): Runtime arguments.
        role (str): Logical role of the model (e.g., "actor", "critic").

    Returns:
        tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler, int]:
            DDP-wrapped model chunks, optimizer, scheduler, and iteration index.
    """

    if torch.version.hip:
        # ROCm/HIP 环境替换 Megatron 的异步文件写入器为 ROCm 兼容实现
        import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module
        from slime.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync

        filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
        print("[ROCm] Applied FileSystemWriterAsync patch for HIP compatibility")

    # 构建模型 + 优化器 + 调度器
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(args, role)
    # 在模型对象上记录角色，便于日志区分
    model[0].role = role
    # 释放构建过程中产生的临时显存
    clear_memory()
    # 从 checkpoint 加载模型 / 优化器 / 调度器状态
    iteration, _ = load_checkpoint(
        model,
        optimizer,
        opt_param_scheduler,
        checkpointing_context={},
        skip_load_to_model_and_opt=False,
    )
    clear_memory()

    # 调度器快进到当前 iteration 对应的位置
    opt_param_scheduler.step(increment=iteration * args.global_batch_size)

    return model, optimizer, opt_param_scheduler, iteration
