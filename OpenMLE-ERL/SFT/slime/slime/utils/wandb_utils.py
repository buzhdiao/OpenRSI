# 导入 logging 模块，用于输出运行日志
import logging
# 导入 math 模块，用于检测浮点数是否为有限值
import math
# 导入 os 模块，用于读取环境变量与创建目录
import os
# 导入 Mapping 抽象基类，用于判断对象是否为映射（字典类）类型
from collections.abc import Mapping
# 导入 deepcopy，用于深拷贝配置快照，避免记录对象与原始参数相互影响
from copy import deepcopy

# 导入 W&B（Weights & Biases）实验跟踪库
import wandb

# 获取当前模块专属的 logger 实例
logger = logging.getLogger(__name__)


# 将记录值尽量转换为可写入 W&B summary 的标量；不可转换时返回 None
def _coerce_summary_scalar(value):
    """Convert logged values into summary-safe scalars when possible."""
    # None 不是可写入的标量，直接返回 None
    if value is None:
        return None
    # 布尔/整数/字符串本身就是合法标量，原样返回
    if isinstance(value, (bool, int, str)):
        return value
    # 浮点数需校验有限性：NaN 或 Inf 会导致 summary 写入异常
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    # 字典等映射类型不属于标量，不写入 summary
    if isinstance(value, Mapping):
        return None

    # 尝试获取 value 的 item() 方法（torch 张量等数值包装对象常见）
    item = getattr(value, "item", None)
    # 若 item 可调用，则递归转换（把张量取值成 Python 标量）
    if callable(item):
        try:
            return _coerce_summary_scalar(item())
        except Exception:
            # 转换失败（如多元素张量调用 item() 抛错）则视为不可写入
            return None

    # 列表/元组/集合等容器类型不是标量，不写入 summary
    if isinstance(value, (list, tuple, set)):
        return None

    # 其余未知类型一律不写入 summary
    return None


# 把标量型指标镜像进 W&B summary，保证面板能可靠地发现这些指标
def extract_summary_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
    """Mirror scalar metrics into summary so W&B can discover them reliably."""
    # 存放筛选后可写入 summary 的键值对
    summary = {}
    # 遍历全部待记录的指标
    for key, value in metrics.items():
        # 跳过以 "/step" 结尾的步数指标：它们只作横轴，不应进 summary
        if key.endswith("/step"):
            continue
        # 将指标值转换为可安全写入的标量
        scalar = _coerce_summary_scalar(value)
        # 转换失败（None）则丢弃该指标
        if scalar is None:
            continue
        # 合法标量写入结果字典
        summary[key] = scalar
    # 返回最终的 summary 更新内容
    return summary


# 记录指标到 W&B，并按需把标量值同步镜像到 summary
def log_metrics(metrics: Mapping[str, object], *, update_summary: bool = True) -> None:
    """Log metrics to W&B and optionally mirror scalar values into summary."""
    # 先以常规方式把全部指标写入 W&B 时序日志
    wandb.log(dict(metrics))
    # 未开启 summary 同步或当前没有活跃 run 时直接结束
    if not update_summary or wandb.run is None:
        return

    # 提取本次指标中可写入 summary 的标量部分
    summary_updates = extract_summary_metrics(metrics)
    # 仅当确实有内容时才执行 summary 更新
    if summary_updates:
        wandb.run.summary.update(summary_updates)


# 判断 W&B 是否应当以离线模式运行
def _is_offline_mode(args) -> bool:
    """Detect whether W&B should run in offline mode.

    Priority order:
    1) args.wandb_mode if provided
    2) WANDB_MODE environment variable
    """
    # 优先级 1：args.wandb_mode 显式指定时以其是否为 offline 为准
    if args.wandb_mode:
        return args.wandb_mode == "offline"
    # 优先级 2：回退检查环境变量 WANDB_MODE 是否等于 offline
    return os.environ.get("WANDB_MODE") == "offline"


# 初始化主进程（primary rank）的 W&B run
def init_wandb_primary(args):
    # 未启用 W&B 时清空 run id 并直接返回
    if not args.use_wandb:
        args.wandb_run_id = None
        return

    # Set W&B mode if specified (overrides WANDB_MODE env var)
    # 若指定了 wandb_mode，则写入环境变量 WANDB_MODE 覆盖默认值
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
        # 离线模式：数据只保存在本地
        if args.wandb_mode == "offline":
            logger.info("W&B offline mode enabled. Data will be saved locally.")
        # 禁用模式：完全不记录任何数据
        elif args.wandb_mode == "disabled":
            logger.info("W&B disabled mode enabled. No data will be logged.")
        # 在线模式：数据上传到 W&B 云端
        elif args.wandb_mode == "online":
            logger.info("W&B online mode enabled. Data will be uploaded to cloud.")

    # 综合参数与环境变量判断是否离线模式
    offline = _is_offline_mode(args)

    # Only perform explicit login when NOT offline
    # 仅在非离线模式且提供了 API key 时执行显式登录
    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Prepare wandb init parameters
    # add random 6 length string with characters
    # 开启随机后缀时给 group 名追加随机 id，避免同名实验互相干扰
    if args.wandb_random_suffix:
        group = args.wandb_group + "_" + wandb.util.generate_id()
        run_name = f"{group}-RANK_{args.rank}"
    else:
        # 未开启随机后缀时直接使用配置中的 group 与 run 名
        group = args.wandb_group
        run_name = args.wandb_group

    # Prepare wandb init parameters
    # 组装 wandb.init 所需的初始化参数字典
    init_kwargs = {
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "group": group,
        "name": run_name,
        "config": _compute_config_for_logging(args),
    }

    # Configure settings based on offline/online mode
    # 按离线/在线模式配置 settings
    if offline:
        # 离线模式：本地保存数据
        init_kwargs["settings"] = wandb.Settings(mode="offline")
    else:
        # 在线模式：shared 模式并标记本进程为主进程
        init_kwargs["settings"] = wandb.Settings(mode="shared", x_primary=True)

    # Add custom directory if specified
    # 指定了自定义日志目录时
    if args.wandb_dir:
        # Ensure directory exists to avoid backend crashes
        # 先确保目录存在，避免后端写入崩溃
        os.makedirs(args.wandb_dir, exist_ok=True)
        # 写入自定义目录参数
        init_kwargs["dir"] = args.wandb_dir
        # 打印日志存储位置
        logger.info(f"W&B logs will be stored in: {args.wandb_dir}")

    # 正式初始化 W&B run
    wandb.init(**init_kwargs)

    # 注册统一的指标横轴定义
    _init_wandb_common()

    # Set wandb_run_id in args for easy access throughout the training process
    # 把 run id 存回 args，供其他进程加入同一 run
    args.wandb_run_id = wandb.run.id


# 生成用于日志记录的训练配置快照
def _compute_config_for_logging(args):
    # 深拷贝 args 字典，避免日志对象与运行时配置相互影响
    output = deepcopy(args.__dict__)

    # 允许写入日志的环境变量白名单
    whitelist_env_vars = [
        "SLURM_JOB_ID",
        # We may insert more default values here, and may also allow users to configure a whitelist
    ]
    # 仅把白名单内的环境变量写入快照，避免泄露敏感信息
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}

    # 返回配置快照
    return output


# https://docs.wandb.ai/guides/track/log/distributed-training/#track-all-processes-to-a-single-run
# 初始化非主进程的 W&B：把所有进程聚合到主进程创建的同一个 run
def init_wandb_secondary(args, router_addr=None):
    # 从 args 读取主进程保存的 run id
    wandb_run_id = getattr(args, "wandb_run_id", None)
    # 主进程未启用 W&B 时无需初始化
    if wandb_run_id is None:
        return

    # Set W&B mode if specified (same as primary)
    # 与主进程一致：指定的 wandb_mode 覆盖环境变量
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    # 判断本进程是否离线模式
    offline = _is_offline_mode(args)

    # 非离线且提供 API key 时执行登录
    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Configure settings based on offline/online mode
    # 按模式构造 settings 关键字参数
    if offline:
        # 离线模式只需设置 mode
        settings_kwargs = dict(mode="offline")
    else:
        # shared 模式：本进程非主进程，且不负责 run 的收尾状态
        settings_kwargs = dict(
            mode="shared",
            x_primary=False,
            x_update_finish_state=False,
        )

    # 启用 SGLang 指标且提供路由地址时，把推理引擎指标转发到 W&B
    if getattr(args, "sglang_enable_metrics", False) and router_addr is not None:
        logger.info(f"Forward SGLang metrics at {router_addr} to WandB.")
        # 配置 OpenMetrics 抓取端点与过滤器
        settings_kwargs |= dict(
            x_stats_open_metrics_endpoints={
                "sgl_engine": f"{router_addr}/engine_metrics",
            },
            x_stats_open_metrics_filters={
                "sgl_engine.*": {},
            },
        )

    # 组装 secondary 进程初始化参数：复用 run id 并允许 resume
    init_kwargs = {
        "id": wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "config": args.__dict__,
        "resume": "allow",
        "reinit": True,
        "settings": wandb.Settings(**settings_kwargs),
    }

    # Add custom directory if specified
    # 指定自定义目录时先确保存在再写入参数
    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    # 以同一 run id 初始化，实现多进程聚合到单一 run
    wandb.init(**init_kwargs)

    # 注册统一的指标横轴定义
    _init_wandb_common()


# 为各命名空间的指标统一注册步数横轴（step metric）
def _init_wandb_common():
    # 下面一批 define_metric 把 train/rollout/eval 等前缀的指标
    # 分别绑定到对应步数横轴，保证面板绘图横轴统一正确
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("rollout/step")
    wandb.define_metric("rollout/*", step_metric="rollout/step")
    wandb.define_metric("multi_turn/*", step_metric="rollout/step")
    wandb.define_metric("passrate/*", step_metric="rollout/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    wandb.define_metric("perf/*", step_metric="rollout/step")
    wandb.define_metric("memory/*", step_metric="rollout/step")
