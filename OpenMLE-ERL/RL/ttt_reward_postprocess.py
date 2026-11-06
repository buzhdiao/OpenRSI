# 导入标准库 math，用于计算目标 KL 约束值 log(2)
import math

# 从奖励-优势工具模块导入：TTT（test-time training）熵正则优势计算函数、
# 按 group_index 把样本聚合成组的工具函数
from adaptive_reward_advantage_utils import (
    compute_ttt_entropic_advantages,
    group_samples_by_group_index,
)


def post_process_rewards(args, samples):
    # 按 group_index 把同一任务/同一批 rollout 的样本聚合成组（GRPO 风格的组内相对化前置步骤）
    grouped_samples = group_samples_by_group_index(args, samples)

    # 汇总各组原始奖励：即沙箱真实执行（execution-grounded）得到的分数
    raw_rewards: list[float] = []
    # 汇总各组后处理后的奖励（组内相对优势），与 raw_rewards 按样本顺序一一对应
    processed_rewards: list[float] = []

    # 逐组处理：对每组计算组内相对优势，替代全局标准化
    for group in grouped_samples:
        # 取出该组每个样本的原始奖励值并转为 float，作为组内基线计算输入
        group_rewards = [float(sample.get_reward_value(args)) for sample in group]
        # 计算组内熵约束优势（GRPO 类组内相对奖励）：以目标 KL=log(2) 约束组内分布偏移，
        # beta_max/iters/eps 控制迭代求解的上界、次数与数值稳定
        group_advantages = compute_ttt_entropic_advantages(
            group_rewards,
            target_kl=float(math.log(2.0)),
            beta_max=1e6,
            iters=60,
            eps=1e-12,
        )
        # 原始奖励按组顺序拼接进总列表
        raw_rewards.extend(group_rewards)
        # 处理后优势按同样顺序拼接，保证上游能按索引对齐两类奖励
        processed_rewards.extend(group_advantages)

    # 返回（原始奖励列表, 处理后奖励列表）二元组
    return raw_rewards, processed_rewards
