# 导入 logging：模块级日志
import logging
# 导入 Path：构造与规范化调试数据输出路径
from pathlib import Path

# 导入 torch：查询分布式 rank 与序列化保存张量数据
import torch

# 创建本模块专属 logger
logger = logging.getLogger(__name__)


def save_debug_train_data(args, *, rollout_id, rollout_data):
    # 未配置调试数据保存模板时直接跳过（海象表达式取模板字符串）
    if (path_template := args.save_debug_train_data) is not None:
        # 查询当前进程的分布式 rank，用于生成每 rank 独立的文件
        rank = torch.distributed.get_rank()
        # 用 rollout 轮次与 rank 填充路径模板，得到实际保存路径
        path = Path(path_template.format(rollout_id=rollout_id, rank=rank))
        # 日志提示保存位置
        logger.info(f"Save debug train data to {path}")
        # 递归创建父目录（已存在则忽略）
        path.parent.mkdir(parents=True, exist_ok=True)
        # 序列化保存：轮次、rank 与整套 rollout 数据，供离线调试分析
        torch.save(
            dict(
                rollout_id=rollout_id,
                rank=rank,
                rollout_data=rollout_data,
            ),
            path,
        )
