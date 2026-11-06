# 导入 logging：模块级日志
import logging
# 导入 math：计算 log2，用于生成 pass@k 的 k 序列
import math
# 导入类型注解工具：Any 表示任意类型，Literal 限定算法名取值
from typing import Any, Literal

# 导入 numpy：批量数值计算（重塑奖励矩阵、按组统计等）
import numpy as np

# 创建本模块专属 logger
logger = logging.getLogger(__name__)


def dict_add_prefix(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    # 给字典所有键统一加前缀，便于在上报系统中区分指标来源
    return {f"{prefix}{k}": v for k, v in d.items()}


def compute_pass_rate(
    flat_rewards: list[float],
    group_size: int,
    num_groups: int | None = None,
):
    # group_size 为 1 时每组只有一条样本，无法计算 pass@k，返回空字典
    if group_size == 1:
        return {}

    # 未显式给出组数时，按总奖励数整除组大小推算
    if num_groups is None:
        num_groups = len(flat_rewards) // group_size

    # 生成 k 序列：1,2,4,...,group_size（2 的幂），即要评估的 pass@k 中的 k
    pass_rate_name_list = [2**i for i in range(int(math.log2(group_size)) + 1)]

    # 校验样本总数 = 组数 × 组大小，否则 reshape 会出错
    assert len(flat_rewards) == num_groups * group_size, f"{len(flat_rewards)=} {num_groups=} {group_size=}"
    # 把扁平奖励列表重塑为 (组数, 组大小) 矩阵：每行对应同一题的一组采样
    rewards_of_group = np.array(flat_rewards).reshape(num_groups, group_size)

    # 汇总各 pass@k 指标
    log_dict = {}
    for k in pass_rate_name_list:
        # 按行统计每道题答对（reward==1）的次数
        num_correct = np.sum(rewards_of_group == 1, axis=1)
        # 每组的采样数恒为 group_size，填充成向量
        num_samples = np.full(num_groups, group_size)

        # 用无偏估计公式计算每道题的 pass@k
        pass_k_estimates = _estimate_pass_at_k(num_samples, num_correct, k)

        # 对所有题目取平均得到整体 pass@k 指标
        pass_k = np.mean(pass_k_estimates)
        # 以 pass@k 为键写入结果字典
        log_dict[f"pass@{k}"] = pass_k

    # 返回 {pass@1, pass@2, ...} 指标字典
    return log_dict


def _estimate_pass_at_k(num_samples, num_correct, k):
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n, c, k):
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        # 错误样本数 c(=n-c) 小于 k：任取 k 个必含正确解，pass@k=1
        if n - c < k:
            return 1.0
        # 无偏估计：1 - C(n-c, k)/C(n, k)，用连乘展开避免大数组合溢出
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    # 对每道题（每组）分别代入 (采样数, 正确数, k) 计算 pass@k
    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples, num_correct, strict=False)])


def compute_statistics(values: list[float]) -> dict[str, float]:
    # 转为 numpy 数组便于批量统计
    values = np.array(values)
    # 返回均值/中位数/最大/最小四个统计量（.item() 转回 Python 标量，便于 JSON 序列化）
    return {
        "mean": np.mean(values).item(),
        "median": np.median(values).item(),
        "max": np.max(values).item(),
        "min": np.min(values).item(),
    }


def compression_ratio(
    data: str | bytes,
    *,
    encoding: str = "utf-8",
    algorithm: Literal["zlib", "gzip", "bz2", "lzma"] = "zlib",
    level: int = 9,
) -> tuple[float, float]:
    if isinstance(data, str):
        # 字符串输入先按指定编码转字节
        raw = data.encode(encoding)
    else:
        # 字节输入直接使用
        raw = data

    # 记录原始字节数
    original = len(raw)
    if original == 0:
        # 空内容压缩率无意义，按无穷大处理、节省率为 0
        return float("inf"), 0.0

    # 根据指定算法选择压缩实现（延迟导入，按需加载）
    if algorithm == "zlib":
        import zlib

        # zlib 压缩，level 越大压缩率越高、耗时越长
        compressed = zlib.compress(raw, level)
    elif algorithm == "gzip":
        import gzip

        # gzip 压缩（带 header 的 zlib 变体）
        compressed = gzip.compress(raw, compresslevel=level)
    elif algorithm == "bz2":
        import bz2

        # bzip2 压缩
        compressed = bz2.compress(raw, compresslevel=level)
    elif algorithm == "lzma":
        import lzma

        # LZMA/xz 压缩
        compressed = lzma.compress(raw, preset=level)
    else:
        # 不支持的算法名直接报错
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # 记录压缩后字节数
    comp_len = len(compressed)
    if comp_len == 0:
        # 压缩后为空（理论罕见）：比率无穷大、节省率 100%
        return float("inf"), 100.0

    # 压缩率 = 原始大小 / 压缩后大小（>1 表示有压缩效果）
    ratio = original / comp_len
    # 节省百分比 = 1 - 压缩后/原始
    savings_pct = 100.0 * (1.0 - comp_len / original)
    # 返回（压缩率, 节省率%）
    return ratio, savings_pct


def has_repetition(text: str):
    # 长文本尾部 10000 字符压缩率 > 10 时判定为重复（重复内容高度可压缩）
    if len(text) > 10000 and compression_ratio(text[-10000:])[0] > 10:
        return True
    else:
        # 不满足条件则认为无重复
        return False


def compute_rollout_step(args, rollout_id):
    # 若开启"按训练步数上报 wandb"：把 rollout 轮次折算为等效训练步数
    if args.wandb_always_use_train_step:
        # 每轮 rollout 产出的样本数 / 全局 batch = 该轮消耗的训练步数
        return rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    # 默认直接返回 rollout 轮次编号
    return rollout_id
