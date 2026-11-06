# 导入 datetime：为缺省实验名生成时间戳
import datetime
# 导入 logging：模块级日志
import logging
# 导入 os：读取 TENSORBOARD_DIR 环境变量与创建目录
import os

# 导入 wandb：读取当前 run 名称，用于对齐 TensorBoard 与 W&B 的目录命名
import wandb
# 导入单例元类：保证每个 rank 只有一个 TensorBoard writer 实例
from slime.utils.misc import SingletonMeta

# 尝试导入 TensorBoard 的 SummaryWriter
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    # 未安装 tensorboard 时置 None，运行时按需报错
    SummaryWriter = None

# 模块对外只暴露适配器类
__all__ = ["_TensorboardAdapter"]

# 创建本模块专属 logger
logger = logging.getLogger(__name__)


def _resolve_tensorboard_dir(tb_project_name, tb_experiment_name):
    """Choose a writer directory whose basename matches the W&B run name when available."""
    # 优先读取环境变量指定的 TensorBoard 目录
    env_dir = os.environ.get("TENSORBOARD_DIR", None)
    if env_dir is None:
        # 无环境变量：使用默认的 项目/实验 两级目录
        base_dir = f"tensorboard_log/{tb_project_name}/{tb_experiment_name}"
        if wandb.run is not None and getattr(wandb.run, "name", None):
            # 有活跃 wandb run 时：目录末级改用 run 名，便于两个平台对齐同一实验
            return os.path.join(f"tensorboard_log/{tb_project_name}", wandb.run.name)
        # 否则返回默认目录
        return base_dir

    # 规范化环境变量路径（去掉多余斜杠等）
    normalized = os.path.normpath(env_dir)
    if wandb.run is None or not getattr(wandb.run, "name", None):
        # 无 wandb run：直接使用规范化后的路径
        return normalized

    # 取当前 wandb run 名
    run_name = wandb.run.name
    # 取规范化路径的末级目录名
    base_name = os.path.basename(normalized)
    if base_name == "tensorboard":
        # 末级恰好叫 tensorboard：把目录切到同级 tensorboard_runs/run_name，避免多 run 混写
        logs_root = os.path.dirname(os.path.dirname(normalized))
        return os.path.join(logs_root, "tensorboard_runs", run_name)

    # 其他情况：在环境变量目录下追加 run 名作为子目录
    return os.path.join(normalized, run_name)


class _TensorboardAdapter(metaclass=SingletonMeta):
    # 类级 writer 缓存（配合单例元类，保证全局唯一）
    _writer = None

    """
    # Usage example: This will return the same instance every rank
    # tb = _TensorboardAdapter(args)  # Initialize on first call
    # tb.log({"Loss": 0.1}, step=1)

    # In other files:
    # from tensorboard_utils import _TensorboardAdapter
    # tb = _TensorboardAdapter(args)  # No parameters needed to get existing instance
    # tb.log({"Accuracy": 0.9}, step=1)
    """

    def __init__(self, args):
        # 必须显式开启 tensorboard 才允许实例化
        assert args.use_tensorboard, f"{args.use_tensorboard=}"
        # 读取项目名与实验名配置
        tb_project_name = args.tb_project_name
        tb_experiment_name = args.tb_experiment_name
        if tb_project_name is not None or os.environ.get("TENSORBOARD_DIR", None):
            if tb_project_name is not None and tb_experiment_name is None:
                # 只给了项目名：用当前时间生成缺省实验名
                tb_experiment_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # 执行真正的初始化
            self._initialize(tb_project_name, tb_experiment_name)
        else:
            # 项目名与目录都未提供，无法确定输出位置，报错
            raise ValueError("tb_project_name and tb_experiment_name, or TENSORBOARD_DIR are required")

    def _initialize(self, tb_project_name, tb_experiment_name):
        """Actual initialization logic"""
        # 解析最终写入目录
        tensorboard_dir = _resolve_tensorboard_dir(tb_project_name, tb_experiment_name)
        # 创建输出目录（已存在则忽略）
        os.makedirs(tensorboard_dir, exist_ok=True)
        # 日志提示日志目录
        logger.info(f"Saving tensorboard log to {tensorboard_dir}.")
        # 创建 SummaryWriter 并缓存到实例
        self._writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        """Log data to tensorboard

        Args:
            data (dict): Dictionary containing metric names and values
            step (int): Current step/epoch number
        """
        # 逐指标写入标量曲线
        for key in data:
            self._writer.add_scalar(key, data[key], step)

    def finish(self):
        """Close the tensorboard writer"""
        # 关闭 writer，刷新并释放事件文件句柄
        self._writer.close()
