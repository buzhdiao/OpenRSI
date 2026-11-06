# 导入 gc，用于手动触发垃圾回收、释放显存
import gc
# 导入 os，用于读取环境变量
import os
# 导入 shutil，用于移动 checkpoint 目录
import shutil

# 导入 PyTorch 主库
import torch
# 导入 torch 分布式模块，用于初始化进程组
import torch.distributed as dist
# 导入 Megatron 模型类型枚举
from megatron.core.enums import ModelType
# 导入 Megatron 训练参数解析与校验函数
from megatron.training.arguments import parse_args, validate_args
# 导入 Megatron checkpoint 工具：目录命名、tracker 文件、保存
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, save_checkpoint
# 导入 Megatron 模型构建函数
from megatron.training.training import get_model

# 导入 slime_plugins.mbridge 以注册 bridge 支持的模型（副作用导入）
import slime_plugins.mbridge  # noqa: F401
# 导入 mbridge 的 AutoBridge，用于把 HF 权重加载进 Megatron 模型
from mbridge import AutoBridge
# 导入 slime 的 Megatron 默认参数设置工具
from slime.backends.megatron_utils.arguments import set_default_megatron_args
# 导入 slime 的 Megatron 分布式初始化函数
from slime.backends.megatron_utils.initialize import init
# 导入 slime 的模型 provider 构造函数
from slime.backends.megatron_utils.model_provider import get_model_provider_func
# 导入日志配置工具
from slime.utils.logging_utils import configure_logger
# 导入显存打印工具
from slime.utils.memory_utils import print_memory


# 向参数解析器添加权重转换专用参数
def add_convertion_args(parser):
    """Add conversion arguments to the parser"""
    # HF 模型路径（必填）
    parser.add_argument("--hf-checkpoint", type=str, required=True, help="HuggingFace model path")
    # Megatron -> HF 权重转换模式选择（raw 或 bridge）
    parser.add_argument(
        "--megatron-to-hf-mode",
        choices=["raw", "bridge"],
        default="raw",
        help="The method to convert megatron weights to hugging face weights for SGLang.",
    )
    # 词表填充大小参数：旧版 Megatron 不认识该参数时忽略异常
    try:
        parser.add_argument("--padded-vocab-size", type=int, default=None)
    except Exception:
        # 参数已存在等异常时静默跳过
        pass
    # 返回追加过参数的解析器
    return parser


# 解析并校验转换所需的全部参数
def get_args():
    # 用 Megatron 的 parse_args 解析参数并附加转换子参数
    args = parse_args(add_convertion_args)
    # 填充 Megatron 运行所需的默认参数
    args = set_default_megatron_args(args)

    # set to pass megatron validate_args
    # 以下设置仅为通过 Megatron 的参数校验而做的占位
    args.save_interval = 1
    args.micro_batch_size = 1
    # 读取全局进程数（GPU 数）
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    # 全局 batch 大小与进程数保持一致以通过校验
    args.global_batch_size = int(os.environ.get("WORLD_SIZE", "1"))

    # 校验：GPU 数不能超过模型层数（每层至少分到一张卡）
    assert world_size <= args.num_layers, (
        f"World size {world_size} must be less than or equal to number of layers {args.num_layers}. "
        "You are using too many GPUs for this conversion."
    )

    # 定义向上取整除法的辅助函数
    def ceildiv(a, b):
        return -(a // -b)

    # 单流水线阶段但多卡时，需要把卡划入多个 PP 阶段
    if args.pipeline_model_parallel_size == 1 and world_size > 1:
        # 先尝试用全部卡数作为 PP 大小
        pp_size = world_size
        # 循环寻找可行的 PP 划分
        while True:
            # 暂定当前 PP 大小
            args.pipeline_model_parallel_size = pp_size
            # 计算最后一个流水线阶段的层数（前面的阶段尽量均分其余层）
            args.decoder_last_pipeline_num_layers = args.num_layers - ceildiv(
                args.num_layers, args.pipeline_model_parallel_size
            ) * (args.pipeline_model_parallel_size - 1)

            # 末阶段层数为正说明划分合法，结束循环
            if args.decoder_last_pipeline_num_layers > 0:
                break

            # 无解且 pp_size 为偶数时减半重试
            if pp_size % 2 == 0:
                pp_size //= 2
            else:
                # 奇数仍无解则报错
                raise ValueError(
                    f"Cannot find a valid pipeline model parallel size for {args.num_layers} layers and {world_size} GPUs."
                )
    # 打印最终确定的流水线划分
    print(
        f"Using pipeline model parallel size: {args.pipeline_model_parallel_size}, decoder last pipeline num layers: {args.decoder_last_pipeline_num_layers}"
    )

    # 执行 Megatron 的参数校验
    validate_args(args)
    # 返回整理后的参数
    return args


# 主流程：HF 权重 -> Megatron torch 分布式 checkpoint
def main():
    # ROCm（AMD）环境下需替换异步文件写入器以兼容 HIP
    if torch.version.hip:
        # 导入 Megatron 的异步文件系统写入模块
        import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module
        # 导入 slime 提供的 ROCm 兼容写入器
        from slime.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync

        # 用 ROCm 版写入器替换默认实现（monkey patch）
        filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
        # 打印补丁提示
        print("[ROCm] Applied FileSystemWriterAsync patch for HIP compatibility")

    # 配置全局日志格式
    configure_logger()

    # Initialize distributed environment
    # 初始化分布式环境：依次读取环境变量（兼容 SLURM 调度）
    world_size = int(os.getenv("WORLD_SIZE") or os.getenv("SLURM_NTASKS") or 1)
    local_rank = int(os.getenv("LOCAL_RANK") or os.getenv("SLURM_LOCALID") or 0)
    global_rank = int(os.getenv("RANK") or os.getenv("SLURM_PROCID") or 0)

    # 绑定当前进程使用的 GPU
    torch.cuda.set_device(local_rank)
    # 把分布式信息回填到环境变量（缺失时兜底）
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("RANK", str(global_rank))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    # 默认 master 地址为本机
    os.environ.setdefault("MASTER_ADDR", "localhost")
    # 默认 master 端口
    os.environ.setdefault("MASTER_PORT", "12355")
    # 初始化 NCCL 进程组
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=global_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    # 解析并校验参数
    args = get_args()
    # 初始化 Megatron 分布式环境（TP/PP/DP 通信组等）
    init(args)

    # if using AMD gpus, we have to do the conversion in cpu
    # AMD GPU 下必须使用 CPU 初始化模型
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        assert args.use_cpu_initialization, "AMD GPU requires --use_cpu_initialization=True"

    # 按 provider 构建 Megatron 模型（不包装 DDP）
    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)

    # Load model
    # 记录 HF 模型路径
    hf_model_path = args.hf_checkpoint
    # 根据模型路径创建对应的 bridge（负责权重名映射）
    bridge = AutoBridge.from_pretrained(hf_model_path, trust_remote_code=True)
    # 以省显存方式把 HF 权重加载进 Megatron 模型
    bridge.load_weights(model, hf_model_path, memory_efficient=True)
    # 打印加载完成信息
    print(f"Model loaded: {hf_model_path}")

    # CPU 初始化模式下把模型移回 CPU，节省 GPU 显存
    if args.use_cpu_initialization:
        model[0] = model[0].cpu()

    # 打印当前显存占用
    print_memory("after loading model")
    # 同步 CUDA 操作
    torch.cuda.synchronize()
    # 手动触发垃圾回收
    gc.collect()
    # 清空 PyTorch 的显存缓存
    torch.cuda.empty_cache()

    # 保存为 Megatron torch 分布式 checkpoint（迭代号 1）
    save_checkpoint(1, model, None, None, 0)

    # 只有 rank 0 负责整理 checkpoint 元信息
    if dist.get_rank() == 0:
        # change to release ckpt
        # 获取 tracker 文件路径（记录最新 checkpoint 迭代号）
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        # 写入 "release"，把 checkpoint 标记为 release 版本
        with open(tracker_filename, "w") as f:
            f.write("release")
        # 源目录：带 PP 维度的 checkpoint 目录
        source_dir = get_checkpoint_name(args.save, 1, False, return_base_dir=True)
        # 目标目录：release 形式的统一目录（-1 表示跨 PP 合并）
        target_dir = get_checkpoint_name(args.save, -1, True, return_base_dir=True)
        # 移动目录，得到标准 release 布局
        shutil.move(source_dir, target_dir)
    # 等待所有进程完成
    dist.barrier()
    # 销毁进程组，释放分布式资源
    dist.destroy_process_group()


# 作为脚本直接运行时执行主流程
if __name__ == "__main__":
    main()
