# 导入 argparse 用于构建命令行参数解析器
import argparse
# 导入 dataclasses 模块，用于反射读取数据类字段（fields()）
import dataclasses
# 从 dataclasses 导入 dataclass 装饰器
from dataclasses import dataclass

# 导入 yaml 用于解析 YAML 配置文件
import yaml


# FSDP 训练参数的数据类定义：所有字段均带默认值，作为参数的单一事实来源
@dataclass
class FSDPArgs:
    # Optim
    # 优化器类型："adam"（实际为 AdamW）
    optimizer: str = "adam"  # Optimizer type: "adam" (AdamW)
    # 初始学习率
    lr: float = 2e-5
    # 学习率预热起始值（warmup 的初始 lr 比例）
    lr_warmup_init: float = 0.0
    # 衰减结束后的最小学习率下限
    min_lr: float = 0.0
    # 学习率衰减策略：constant（恒定）、linear、cosine 等
    lr_decay_style: str = "constant"
    # 学习率衰减的迭代步数（None 表示不衰减）
    lr_decay_iters: int | None = None
    # 学习率预热的迭代步数
    lr_warmup_iters: int = 0
    # 学习率预热占总训练步数的比例（与 iters 二选一）
    lr_warmup_fraction: float | None = None
    # WSD（Warmup-Stable-Decay）调度中稳定段之后的衰减步数
    lr_wsd_decay_iters: int | None = None
    # WSD 衰减的具体方式（如 exponential 等）
    lr_wsd_decay_style: str | None = None
    # 为 True 时从 checkpoint 恢复调度器状态
    use_checkpoint_lr_scheduler: bool = True
    # 为 True 时用命令行参数覆盖 checkpoint 中保存的调度器设置
    override_lr_scheduler: bool = False
    # 权重衰减（weight decay）系数
    weight_decay: float = 0.0
    # Adam 一阶动量衰减系数 beta1
    adam_beta1: float = 0.9
    # Adam 二阶动量衰减系数 beta2
    adam_beta2: float = 0.95
    # Adam 分母中的数值稳定项 epsilon
    adam_eps: float = 1e-8
    # 预热步数占总步数的比例（HF Trainer 风格参数）
    warmup_ratio: float = 0.03

    # 注意力实现方式：默认使用 FlashAttention-2 以节省显存与加速
    attn_implementation: str = "flash_attention_2"

    # Logging
    # wandb 实验所属项目名
    wandb_project: str = "slime-fsdp"
    # wandb 运行名（None 时由框架自动生成）
    wandb_run_name: str | None = None

    # Precision
    # 是否启用梯度检查点（gradient checkpointing）：以重算换显存
    gradient_checkpointing: bool = False
    # 是否使用 FP16 混合精度训练
    fp16: bool = False

    # FSDP configuration
    # 收集完整 state dict 时是否卸载到 CPU（权重同步/保存时省 GPU 显存）
    fsdp_state_dict_cpu_offload: bool = True  # If True, offload full state dict to CPU during collection.
    # 为 True 时把参数、梯度与优化器状态都卸载到 CPU（优化器在 CPU 上运行）
    fsdp_cpu_offload: bool = (
        False  # If True, offload parameters, gradients, and optimizer states to CPU (optimizer runs on CPU)
    )
    # CPU 卸载时使用的通信后端（如 "gloo"）；置 None 关闭混合后端
    fsdp_cpu_backend: str | None = (
        "gloo"  # CPU backend for FSDP CPU offload (e.g., "gloo"). Set to None to disable hybrid backend.
    )

    # 确定性模式（关闭非确定性算子以便复现）；命名必须与 Megatron 保持一致
    deterministic_mode: bool = False  # This name must be the same as Megatron's

    # Profile
    # 是否记录显存分配历史（用于排查显存峰值）
    record_memory_history: bool = False
    # 显存快照的保存路径
    memory_snapshot_path: str = "snapshot.pickle"
    # 是否启用 PyTorch profiler 进行性能剖析
    use_pytorch_profiler: bool = False
    # profiler 开始记录的训练步
    profile_step_start: int = 10
    # profiler 结束记录的训练步
    profile_step_end: int = 12
    # TensorBoard 日志目录（None 表示不写）
    tensorboard_dir: str | None = None

    # YAML bookkeeping
    # 记录本次使用的 YAML 配置文件路径（便于回溯）
    config: str | None = None


# 解析 FSDP 训练的命令行参数；extra_args_provider 可追加自定义参数，ignore_unknown_args 控制是否忽略未知参数
def _parse_fsdp_cli(extra_args_provider=None, ignore_unknown_args=False):
    # 创建命令行解析器
    parser = argparse.ArgumentParser("FSDP SFT Training (slime)")
    # 单独注册 --config 参数（YAML 配置路径），因为它是控制其余参数来源的入口
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    # 遍历 FSDPArgs 数据类的所有字段，为每个字段自动生成对应的命令行参数
    for f in dataclasses.fields(FSDPArgs):
        # 跳过 config 字段本身，避免重复注册
        if f.name == "config":
            continue

        # Handle union types like int | None, str | None, etc.
        # 处理 int | None、str | None 这类联合类型注解
        if hasattr(f.type, "__args__"):  # Check if it's a Union type
            # For T | None, use T as the type
            # 对于 T | None 类型，取去掉 NoneType 后的第一个真实类型作为参数类型
            non_none_types = [t for t in f.type.__args__ if t is not type(None)]
            # 若过滤后没有剩余类型则回退为 str
            arg_type = non_none_types[0] if non_none_types else str
        else:
            # 普通类型注解直接使用
            arg_type = f.type

        # 布尔字段用 store_true 开关表示（出现即为 True）
        if arg_type is bool:
            parser.add_argument(f"--{f.name.replace('_', '-')}", action="store_true")
        else:
            # 其余字段按推断出的类型注册，并带数据类中的默认值；参数名用连字符风格（--lr-decay-iters）
            parser.add_argument(f"--{f.name.replace('_', '-')}", type=arg_type, default=f.default)

    # 若调用方提供了额外的参数注册函数，则让其向 parser 追加参数
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)
    # 忽略未知参数时用 parse_known_args（便于与外层框架共用命令行）；否则严格解析
    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()
    # 返回解析后的参数命名空间
    return args


# FSDP 参数解析的对外入口：先解析命令行，再用 YAML 配置补充未显式给出的参数
def fsdp_parse_args(extra_args_provider=None, ignore_unknown_args=False):
    # 先完成命令行解析
    args = _parse_fsdp_cli(extra_args_provider, ignore_unknown_args=ignore_unknown_args)
    # 若指定了 YAML 配置文件则读取它
    if args.config:
        # 打开并解析 YAML；空文件时回退为空字典
        with open(args.config) as f:
            data = yaml.safe_load(f) or {}
        # 遍历配置项并写入 args（注意：hasattr 为真时不覆盖，保证命令行优先级更高）
        for k, v in data.items():
            if not hasattr(args, k):
                setattr(args, k, v)
    # 将 rank 固定为 0：wandb 等初始化假定主进程角色
    args.rank = 0  # Primary process rank for wandb initialization
    # 计算全局并行规模：Actor 节点数 x 每节点 GPU 数
    args.world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
    # 返回补全后的参数
    return args
