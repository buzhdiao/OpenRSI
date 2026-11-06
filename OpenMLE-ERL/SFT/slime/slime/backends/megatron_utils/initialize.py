import logging
import random

import numpy as np
import torch
from megatron.core import mpu, tensor_parallel
from megatron.core.config import set_experimental_flag
from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator
from megatron.training.global_vars import _build_tokenizer, set_args

logger = logging.getLogger(__name__)


def _set_random_seed(
    seed_: int,
    data_parallel_random_init: bool = False,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
    """Set random seed for reproducability."""
    # Ensure that different pipeline MP stages get different seeds.
    # 不同流水线并行（pipeline parallel）阶段加上不同偏移，避免各阶段随机数序列相同
    seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())
    # Ensure different data parallel ranks get different seeds
    # 允许数据并行各 rank 用不同随机初始化（仅当显式开启时）
    if data_parallel_random_init:
        seed = seed + (10 * mpu.get_data_parallel_rank(with_context_parallel=False))
    # 依次设置 Python / NumPy / PyTorch 的全局随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 设置张量并行的 CUDA 随机数生成器（保证 TP 组内 dropout 等的一致或差异化行为）
    tensor_parallel.model_parallel_cuda_manual_seed(seed, te_rng_tracker, inference_rng_tracker, use_cudagraphable_rng)


def _initialize_distributed(args, get_embedding_ranks=None, get_position_embedding_ranks=None):
    """Initialize torch.distributed and core model parallel."""
    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    # 按参数初始化各维模型并行通信组（张量并行 TP、流水线并行 PP、数据并行 DP、上下文并行 CP、专家并行 EP 等）
    # 进程组排列顺序：默认 tp-cp-ep-dp-pp，开启 tp-pp-dp 映射时用另一种顺序
    # 指定哪些 rank 承载 embedding / position embedding（默认首尾流水线阶段）
    mpu.initialize_model_parallel(
        args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size,
        args.virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_comm_backend=args.pipeline_model_parallel_comm_backend,
        context_parallel_size=args.context_parallel_size,
        hierarchical_context_parallel_sizes=args.hierarchical_context_parallel_sizes,
        expert_model_parallel_size=args.expert_model_parallel_size,
        num_distributed_optimizer_instances=args.num_distributed_optimizer_instances,
        expert_tensor_parallel_size=args.expert_tensor_parallel_size,
        distributed_timeout_minutes=args.distributed_timeout_minutes,
        nccl_communicator_config_path=args.nccl_communicator_config_path,
        order="tp-cp-ep-dp-pp" if not args.use_tp_pp_dp_mapping else "tp-cp-ep-pp-dp",
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
        create_gloo_process_groups=args.enable_gloo_process_groups,
    )


def init(args):
    # 把解析好的 args 注册为 Megatron 全局参数
    set_args(args)
    # 可选开启 Megatron 的实验性特性开关
    if args.enable_experimental:
        logger.info("Enable megatron experimental")
        set_experimental_flag(True)

    # Pytorch distributed.
    # 初始化分布式与各维模型并行通信组
    _initialize_distributed(args)

    # https://github.com/NVIDIA/Megatron-LM/issues/1563
    # Megatron 目前只兼容 numpy 1.x，遇到 2.x 直接报错
    assert np.__version__.startswith("1."), "Megatron does not support numpy 2.x"

    # Random seeds for reproducibility.
    # 按参数设置各处随机种子，保证可复现性
    if args.rank == 0:
        logger.info(f"> setting random seeds to {args.seed} ...")
    _set_random_seed(
        args.seed,
        args.data_parallel_random_init,
        args.te_rng_tracker,
        args.inference_rng_tracker,
    )
    # 构建（并缓存）tokenizer
    _build_tokenizer(args)
    # We won't use this. initialize to pass some validation in megatron.
    # slime 自行计算 microbatch 数，这里初始化计算器只是为了通过 Megatron 内部的校验
    init_num_microbatches_calculator(
        args.rank,
        args.rampup_batch_size,
        args.global_batch_size,
        args.micro_batch_size,
        args.data_parallel_size,
        args.decrease_batch_size_if_needed,
    )

    # 确定性模式：关闭 cudnn 非确定性优化并启用确定性算法，便于调试复现
    if args.deterministic_mode:
        if args.rank == 0:
            logger.info("> running in deterministic mode")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=False)

    # 开启 TP 通信与 GEMM 重叠时，初始化张量并行通信器
    if args.tp_comm_overlap:
        from megatron.training.initialize import _initialize_tp_communicators

        _initialize_tp_communicators()

    # 允许用户通过自定义路径注入额外的 Megatron 初始化逻辑
    if getattr(args, "custom_megatron_init_path", None):
        from slime.utils.misc import load_function

        # 动态加载 "模块:函数" 形式的自定义初始化函数并调用
        custom_init = load_function(args.custom_megatron_init_path)
        custom_init(args)


# TODO shall we use a simpler method to determine which rank to init wandb?
def is_megatron_main_rank():
    # 判断当前 rank 是否为 Megatron 的"主 rank"：DP（含 CP）为 0、TP 为 0、且位于最后一个流水线阶段
    # （最后一个 PP 阶段持有输出层与 loss，适合负责日志/wandb 上报）
    return (
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0
        and mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
    )
