# 导入标准库日志模块，用于全局日志配置
import logging

# 导入 wandb 工具模块：负责 Weights & Biases 实验跟踪的初始化与指标上报
from . import wandb_utils
# 导入 TensorBoard 适配器：负责把指标写入 TensorBoard 事件文件
from .tensorboard_utils import _TensorboardAdapter

# 模块级标志：记录全局日志是否已配置过，保证 configure_logger 只生效一次
_LOGGER_CONFIGURED = False


# ref: SGLang
def configure_logger(prefix: str = ""):
    # 声明修改模块级配置标志
    global _LOGGER_CONFIGURED
    # 已配置过则直接返回，避免重复 basicConfig 影响用户自定义配置
    if _LOGGER_CONFIGURED:
        return

    # 置位标志，后续再调用本函数会被短路
    _LOGGER_CONFIGURED = True

    # 配置根 logger：INFO 级别、带时间戳与可选前缀、输出文件名和行号
    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        # force=True 会先移除已有 handler 再重新配置（Python 3.8+）
        force=True,
    )


def init_tracking(args, primary: bool = True, **kwargs):
    # primary=True 表示主任务：以主 run 身份初始化 wandb
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        # 否则以 secondary 身份初始化（如子任务/非主 rank，不抢主 run 控制权）
        wandb_utils.init_wandb_secondary(args, **kwargs)


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    # 开启了 wandb 时，把整套指标上报到 wandb
    if args.use_wandb:
        wandb_utils.log_metrics(metrics)

    # 开启了 tensorboard 时，把指标写入事件文件
    if args.use_tensorboard:
        # TensorBoard 的 step 必须单独传入，指标字典中先剔除 step 键本身
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        # 用 step 键对应的值作为全局步数写入 TensorBoard
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])
