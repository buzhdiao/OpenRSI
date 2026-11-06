# 引入 torch，用于计算奖励张量的标准差
import torch

# 引入动态过滤器的统一输出类型
from slime.rollout.filter_hub.base_types import DynamicFilterOutput
# 引入样本类型
from slime.utils.types import Sample

# 声明本模块对外暴露的过滤器
__all__ = ["check_reward_nonzero_std"]


# 动态采样过滤器（论文 Dynamic Sampling，见 DAPO）：奖励标准差为 0 的组无学习信号，予以丢弃
def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    # 取出组内每个样本的奖励值
    rewards = [sample.get_reward_value(args) for sample in samples]
    # 判断组内奖励标准差是否大于 0（全对/全错组 std=0，优势为 0，无有效梯度）
    keep = torch.tensor(rewards, dtype=torch.float).std() > 0.0
    # 返回过滤结果：丢弃时附带 zero_std_{奖励值} 的原因标签
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )
