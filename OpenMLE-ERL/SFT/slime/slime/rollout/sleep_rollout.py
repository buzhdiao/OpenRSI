# 引入日志模块
import logging
# 引入 time 模块用于休眠
import time

# 获取本模块专属 logger
logger = logging.getLogger(__name__)


# 空转的 rollout 实现：不生成任何数据，仅周期性打日志（用于挂起/调试 rollout 流程）
def sleep(args, rollout_id, data_source, evaluation=False):
    # 已休眠小时计数
    count = 0
    # 无限循环持续挂起
    while True:
        # 每次休眠 3600 秒（1 小时）
        time.sleep(3600)
        # 休眠轮次加一
        count += 1
        # 记录已休眠的小时数
        logger.info(f"rollout sleep for {count} hours")
