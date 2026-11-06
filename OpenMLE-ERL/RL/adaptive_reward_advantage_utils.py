# 本模块实现 RL 训练中的自适应奖励与组内优势（advantage）计算工具：
# 1) 自适应奖励边界：按多种统计口径（top1_top8/top1_top16/均值/中位数/理论边界），
#    从历史分数与当前组分数中估计动态 (best, worst) 带符号边界，再经 score2reward
#    把分数映射为奖励，使奖励刻度随经验池水平自适应浮动；
# 2) 组内优势：提供 GSPO 风格的组归一化优势与 TTT 熵正则（留一重要性权重）优势；
# 3) 样本分组：按 group_index 把样本聚合为组，供组内对比类算法使用。
import math
# defaultdict：按 group_index 聚合样本时键不存在自动建空列表
from collections import defaultdict
# Iterable：标记可迭代分数序列参数的类型注解
from typing import Iterable

# numpy：向量化计算均值、标准差、中位数等统计量
import numpy as np

# 奖励工具：读取静态理论边界、分数转奖励、统一带符号分数
from reward_func_utils import get_bound_limits_signed, score2reward, signed_score


# 自适应边界模式：上界取第 1 名、下界取前 8 名的末位（第 8 名）
BOUND_MODE_TOP1_TOP8 = "top1_top8"
# 自适应边界模式：上界取第 1 名、下界取前 16 名的末位（第 16 名）
BOUND_MODE_TOP1_TOP16 = "top1_top16"
# 自适应边界模式：上界取第 1 名、下界取全部分数的均值
BOUND_MODE_TOP1_ALL_MEAN = "top1_all_mean"
# 自适应边界模式：上界取第 1 名、下界取全部分数的中位数（对离群分数更稳健）
BOUND_MODE_TOP1_ALL_MEDIAN = "top1_all_median"
# 自适应边界模式：不做统计估计，直接使用元数据中的静态理论边界
BOUND_MODE_THEORETICAL = "theoretical"

# 支持的全部自适应边界模式集合，用于入参校验
SUPPORTED_ADAPTIVE_BOUND_MODES = {
    BOUND_MODE_TOP1_TOP8,
    BOUND_MODE_TOP1_TOP16,
    BOUND_MODE_TOP1_ALL_MEAN,
    BOUND_MODE_TOP1_ALL_MEDIAN,
    BOUND_MODE_THEORETICAL,
}


# 判断值是否为有效有限数值（非 None、可转 float 且不为 inf/nan），用于过滤脏分数
def _finite(value) -> bool:
    # 尝试转 float 并检查有限性，任何异常都按无效处理
    try:
        # 非 None 且为有限浮点数才视为有效
        return value is not None and math.isfinite(float(value))
    except Exception:
        # 类型不符、无法转换等情况一律视为无效
        return False


# 清洗 (best, worst) 带符号边界对：与静态理论边界取交集、缺失时回退静态边界，
# 并保证上下界之间至少留有 min_span 宽度，避免后续归一化除零
def _sanitize_bound_pair(
    best_signed: float | None,
    worst_signed: float | None,
    *,
    metadata: dict,
    min_span: float = 1e-6,
) -> tuple[float | None, float | None]:
    # 从元数据读取静态理论边界（带符号分数口径）
    static_best, static_worst = get_bound_limits_signed(metadata)

    # 动态上界不能越过静态理论上界：二者取较小值
    if _finite(best_signed) and _finite(static_best):
        best_signed = min(float(best_signed), float(static_best))
    # 动态下界不能越过静态理论下界：二者取较大值
    if _finite(worst_signed) and _finite(static_worst):
        worst_signed = max(float(worst_signed), float(static_worst))

    # 若裁剪后仍有边界无效，尝试整体回退到静态理论边界
    if not (_finite(best_signed) and _finite(worst_signed)):
        # 静态边界齐全时直接采用静态边界
        if _finite(static_best) and _finite(static_worst):
            best_signed = float(static_best)
            worst_signed = float(static_worst)
        else:
            # 静态边界也缺失，返回 (None, None) 表示自适应边界不可用
            return None, None

    # 显式转 float，保证后续算术运算类型一致
    best_signed = float(best_signed)
    worst_signed = float(worst_signed)
    # 区间退化（上界不高于下界）时，把上界抬到下界之上至少 min_span
    if best_signed <= worst_signed:
        best_signed = worst_signed + max(float(min_span), 1e-12)
    # 返回清洗后的 (上界, 下界) 边界对
    return best_signed, worst_signed


# 计算自适应奖励边界对 (best, worst)：按所选模式从历史分数与当前组分数中统计估计
# 上下界，作为把当前分数动态映射为奖励的参照系（边界随经验池水平浮动）
def compute_adaptive_bound_pair(
    *,
    metadata: dict,
    historical_scores: Iterable[float],
    current_group_scores: Iterable[float],
    mode: str,
    min_span: float = 1e-6,
    lower_shift_ratio: float = 0.0,
) -> tuple[float | None, float | None]:
    # 规范化模式字符串：去空白并转小写，空值回退到默认的 top1_top8 模式
    mode = (mode or BOUND_MODE_TOP1_TOP8).strip().lower()
    # 校验模式合法性，不支持的口径直接报错
    if mode not in SUPPORTED_ADAPTIVE_BOUND_MODES:
        raise ValueError(f"Unsupported adaptive reward bound mode: {mode}")

    # 理论模式：跳过统计估计，直接采用元数据中的静态理论边界
    if mode == BOUND_MODE_THEORETICAL:
        static_best, static_worst = get_bound_limits_signed(metadata)
        return _sanitize_bound_pair(static_best, static_worst, metadata=metadata, min_span=min_span)

    # 汇总历史分数与当前组分数，统一转为带符号分数并过滤掉非有限值
    signed_scores = [
        signed_score(score, metadata)
        for score in list(historical_scores) + list(current_group_scores)
        if _finite(score)
    ]
    # 没有任何有效分数时，回退到静态理论边界
    if not signed_scores:
        static_best, static_worst = get_bound_limits_signed(metadata)
        return _sanitize_bound_pair(static_best, static_worst, metadata=metadata, min_span=min_span)

    # 降序排序，最前面的即当前经验池中最好的带符号分数
    signed_scores = sorted(float(s) for s in signed_scores)[::-1]
    # 上界取第 1 名（最高带符号分数）
    best_signed = signed_scores[0]

    # 按模式选择下界的统计口径
    if mode == BOUND_MODE_TOP1_TOP8:
        # top1_top8：下界取第 8 名（不足 8 个时取最后一个）
        worst_signed = signed_scores[min(7, len(signed_scores) - 1)]
    elif mode == BOUND_MODE_TOP1_TOP16:
        # top1_top16：下界取第 16 名（不足 16 个时取最后一个）
        worst_signed = signed_scores[min(15, len(signed_scores) - 1)]
    elif mode == BOUND_MODE_TOP1_ALL_MEAN:
        # top1_all_mean：下界取全部分数的均值
        worst_signed = float(np.mean(signed_scores))
    else:
        # top1_all_median：下界取全部分数的中位数（对离群分数稳健）
        assert mode == BOUND_MODE_TOP1_ALL_MEDIAN
        worst_signed = float(np.median(signed_scores))

    # 读取下界下移比例（负值按 0 处理，即默认不下移）
    shift_ratio = max(float(lower_shift_ratio), 0.0)
    # 需要下移时，把下界按 (上界-下界) 的比例向下平移，留出更多低分奖励空间
    if shift_ratio > 0.0:
        worst_signed = float(worst_signed) - shift_ratio * max(float(best_signed) - float(worst_signed), 0.0)

    # 交给清洗函数与静态边界取交集并保证区间非退化
    return _sanitize_bound_pair(best_signed, worst_signed, metadata=metadata, min_span=min_span)


# 把单个分数映射为组内自适应奖励：先按历史+当前组分数估计动态边界，
# 再调用 score2reward 完成分数→奖励转换；返回 (奖励值, 诊断上下文) 二元组
def score_to_group_adaptive_reward(
    *,
    score: float | None,
    metadata: dict,
    historical_scores: Iterable[float],
    current_group_scores: Iterable[float],
    mode: str,
    reward_mapping_mode: str,
    use_score2reward: bool,
    default_reward: float = 0.0,
    min_span: float = 1e-6,
    lower_shift_ratio: float = 0.0,
    single_finite_reward_one: bool = False,
) -> tuple[float, dict[str, float | str | bool | None]]:
    # 初始化诊断上下文：是否启用、边界模式、上下界等字段待后续填写
    context: dict[str, float | str | bool | None] = {
        "enabled": False,
        "bound_mode": mode,
        "best_signed": None,
        "worst_signed": None,
        "lower_shift_ratio": float(max(lower_shift_ratio, 0.0)),
        "single_finite_reward_one_applied": False,
    }
    # 当前分数无效（None/inf/nan）时直接返回默认奖励
    if not _finite(score):
        return float(default_reward), context
    # 未开启分数→奖励映射时，原样返回分数本身
    if not use_score2reward:
        return float(score), context

    # 收集历史分数与当前组分数对应的带符号有效分数，作为边界估计样本
    available_signed_scores = [
        signed_score(value, metadata)
        for value in list(historical_scores) + list(current_group_scores)
        if _finite(value)
    ]
    # 特例：仅有 1 个有效参照分数时，若开启 single_finite_reward_one 则直接给奖励 1.0
    if single_finite_reward_one and len(available_signed_scores) == 1:
        only_signed = float(available_signed_scores[0])
        context["enabled"] = True
        context["best_signed"] = only_signed
        context["worst_signed"] = only_signed
        context["single_finite_reward_one_applied"] = True
        return 1.0, context

    # 按指定模式计算自适应上下界
    best_signed, worst_signed = compute_adaptive_bound_pair(
        metadata=metadata,
        historical_scores=historical_scores,
        current_group_scores=current_group_scores,
        mode=mode,
        min_span=min_span,
        lower_shift_ratio=lower_shift_ratio,
    )
    # 把算得的边界写入上下文，便于上层日志/调试
    context["best_signed"] = best_signed
    context["worst_signed"] = worst_signed
    # 只有上下界都有效才认为自适应映射真正启用
    context["enabled"] = _finite(best_signed) and _finite(worst_signed)

    # 复制元数据，准备把动态边界注入给 score2reward
    reward_metadata = dict(metadata)
    # 自适应边界有效时，把动态上下界写入元数据以覆盖静态边界
    if context["enabled"]:
        reward_metadata["dynamic_bound_best_signed"] = best_signed
        reward_metadata["dynamic_bound_worst_signed"] = worst_signed
    # 按指定映射模式把分数转换为奖励
    reward = score2reward(float(score), reward_metadata, mode=reward_mapping_mode)
    # 返回奖励值与诊断上下文
    return float(reward), context


# 计算 GSPO（组序列策略优化）风格的组内优势：组内奖励减去均值作为组内基线
# （可选再做标准差归一化），同一 prompt 的组内样本互为对照，无需价值网络
def compute_gspo_group_advantages(
    rewards: list[float],
    *,
    std_normalization: bool = True,
    eps: float = 1e-6,
) -> list[float]:
    # 空奖励列表直接返回空结果
    if not rewards:
        return []
    # 转为 float64 数组便于向量化运算
    arr = np.asarray(rewards, dtype=np.float64)
    # 减去组内均值，实现组内相对基线（组中心化）
    arr = arr - float(arr.mean())
    # 可选：再做标准差归一化（稳定梯度尺度）
    if std_normalization:
        # 计算样本标准差（只有一个样本时用总体口径 ddof=0，避免 NaN）
        std = float(arr.std(ddof=1 if arr.size > 1 else 0))
        # 分母加 eps 防止除零
        arr = arr / (std + float(eps))
    # 转回 Python 浮点列表返回
    return arr.astype(np.float64).tolist()


# 二分搜索熵正则温度 beta：使 softmax(beta*rewards) 分布相对均匀分布的
# KL 估计量达到 target_kl，从而控制优势权重的尖锐程度
def _solve_entropic_beta(
    rewards: np.ndarray,
    *,
    target_kl: float,
    beta_max: float,
    iters: int,
) -> float:
    # 组内样本数少于 2 时无分布可言，温度取 0
    k = int(rewards.size)
    if k < 2:
        return 0.0

    # 预计算 log(k)，作为 KL 估计中的熵修正项
    log_k = math.log(k)
    # 确保 float64 类型
    rewards = rewards.astype(np.float64)
    # 减去最大值做数值稳定（防 exp 溢出），不改变 softmax 结果
    centered = rewards - float(rewards.max())

    # 定义 KL 估计函数：给定 beta，计算 q=softmax(beta*centered) 相对均匀分布 1/k 的 KL
    def kl_hat(beta_value: float) -> float:
        # 缩放后的 logits
        logits = beta_value * centered
        # 用 logaddexp 稳定计算对数配分函数
        log_denom = float(np.logaddexp.reduce(logits))
        # 对数概率（softmax 归一化）
        log_q = logits - log_denom
        # 概率分布 q
        q = np.exp(log_q)
        # 返回 KL 估计：sum q*(log q + log k) = log k - H(q)
        return float(np.sum(q * (log_q + log_k)))

    # 初始二分区间 [0, 1]
    lo, hi = 0.0, 1.0
    # 若上界 1 对应的 KL 仍低于目标，说明 beta 需要更大，先倍增扩大区间
    if kl_hat(hi) < target_kl:
        # 倍增 hi 直到 KL 达标或到达 beta_max 上限
        while hi < beta_max and kl_hat(hi) < target_kl:
            hi *= 2.0
        # 扩到 beta_max 仍不达标，直接返回 beta_max（已尽量接近目标）
        if kl_hat(hi) < target_kl:
            return float(hi)

    # 标准二分迭代：KL 随 beta 单调递增，收缩区间直到达到迭代次数
    for _ in range(int(iters)):
        # 取区间中点
        mid = 0.5 * (lo + hi)
        # 中点 KL 低于目标则答案在右半区间
        if kl_hat(mid) < target_kl:
            lo = mid
        # 否则答案在左半区间
        else:
            hi = mid
    # 返回满足目标 KL 的温度上界
    return float(hi)


# 计算 TTT 风格的熵正则组优势：以自归一化重要性权重 w_i ∝ exp(beta*r_i) 的
# 留一（LOO）配分作基线，优势 = w_i - 1；beta 由目标 KL 约束自动求解
def compute_ttt_entropic_advantages(
    rewards: list[float],
    *,
    target_kl: float = math.log(2.0),
    beta_max: float = 1e6,
    iters: int = 60,
    eps: float = 1e-12,
) -> list[float]:
    # 空列表返回空结果
    if not rewards:
        return []
    # 转 float64 数组
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    # 单样本组无法组内对比，优势全 0
    if rewards_arr.size < 2:
        return [0.0 for _ in rewards]
    # 组内奖励全部相同（无差异信号）时，优势全 0
    if np.allclose(rewards_arr, rewards_arr[0]):
        return [0.0 for _ in rewards]

    # 先解出满足目标 KL 约束的温度 beta
    beta = _solve_entropic_beta(rewards_arr, target_kl=target_kl, beta_max=beta_max, iters=iters)
    # 减最大值做数值稳定的居中
    centered = rewards_arr - float(rewards_arr.max())
    # 计算未归一化权重 exp(beta*(r_i - r_max))
    exp_vals = np.exp(beta * centered)
    # 计算留一（LOO）配分：去掉自身后其余样本权重的均值，构成自归一化基线
    if rewards_arr.size == 1:
        loo_z = exp_vals
    else:
        loo_z = (exp_vals.sum() - exp_vals) / float(rewards_arr.size - 1)
    # 重要性权重 = 自身权重 / 留一配分（分母加 eps 防除零）
    weights = exp_vals / (loo_z + float(eps))
    # 优势 = 权重 - 1：高于留一均值为正、低于为负
    advantages = weights - 1.0
    # 转回 Python 浮点列表返回
    return advantages.astype(np.float64).tolist()


# 把样本按组聚合为 list[list]：优先使用样本自带的 group_index 属性分桶，
# 逐级回退（已是分组列表 / 单样本组 / 按 n_samples_per_prompt 顺序切片）
def group_samples_by_group_index(args, samples) -> list[list]:
    # 传入的已经是分组好的列表（首元素是列表）时，原样返回
    if samples and isinstance(samples[0], list):
        return samples

    # 用 defaultdict 按 group_index 分桶聚合
    grouped: dict[int, list] = defaultdict(list)
    # 没有 group_index 的样本放入兜底组（每个样本单独一组）
    fallback_groups: list[list] = []
    # 遍历全部样本，按 group_index 归组
    for idx, sample in enumerate(samples):
        # 读取样本的 group_index 属性（可能不存在）
        group_index = getattr(sample, "group_index", None)
        # 缺失 group_index 时放入兜底组
        if group_index is None:
            fallback_groups.append([sample])
            continue
        # 有 group_index 的样本追加到对应组
        grouped[int(group_index)].append(sample)

    # 存在按 index 分好的组时，按组号升序返回
    if grouped:
        return [grouped[key] for key in sorted(grouped)]
    # 只有兜底组时返回兜底组
    if fallback_groups:
        return fallback_groups

    # 完全无分组信息时，按每 prompt 采样数 n 顺序切片分组（n 至少为 1）
    n = max(int(getattr(args, "n_samples_per_prompt", 1) or 1), 1)
    return [samples[i : i + n] for i in range(0, len(samples), n)]
