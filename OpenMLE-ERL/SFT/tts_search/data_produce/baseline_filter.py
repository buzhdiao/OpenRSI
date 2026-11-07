"""Shared baseline token and validation/test gap filtering."""

from __future__ import annotations

# 导入 dataclass 装饰器与字段工厂，定义不可变配置/决策对象
from dataclasses import dataclass, field
# 导入类型注解工具
from typing import Any, Mapping

# 导入通用工具：安全浮点转换
from tts_search.data_produce.common import maybe_float
# 导入差距过滤模块：差距配置、成功反馈判断、相对差距计算
from tts_search.data_produce.gap_filter import (
    GapFilterConfig,
    feedback_is_success,
    relative_gap_from_feedback,
)
# 导入 token 过滤模块：长度上限判断
from tts_search.data_produce.token_filter import token_count_within_limit


# 基线 token 差距过滤配置：生成时过滤与后处理过滤共用
@dataclass(frozen=True)
class BaselineTokenGapConfig:
    """Configuration shared by generation-time and postprocess filtering."""

    # 单条样本允许的最大 token 数（SLIME 聊天模板口径）
    max_total_tokens: int = 32768
    # 嵌套的相对差距过滤配置（默认构造）
    gap_config: GapFilterConfig = field(default_factory=GapFilterConfig)
    # 恰好等于上限的样本是否保留
    keep_equal: bool = True


# 过滤决策对象：记录是否接受、原因、token 长度与差距详情
@dataclass(frozen=True)
class BaselineTokenGapDecision:
    """Decision returned by the shared baseline token/gap filter."""

    # 是否接受该样本
    accepted: bool
    # 决策原因（机器可读的短字符串）
    reason: str
    # 样本 token 长度（无法获取时为 None）
    token_length: float | None = None
    # 差距评估的附加信息字典
    gap_info: dict[str, Any] = field(default_factory=dict)


def evaluate_baseline_token_gap(
    *,
    slime_message_tokens: Any,
    feedback_text: str | None,
    metadata: Mapping[str, Any] | None,
    config: BaselineTokenGapConfig | None = None,
    task_score_scale: float | None = None,
) -> BaselineTokenGapDecision:
    """Apply the baseline token and validation/test gap filters.

    ``slime_message_tokens`` is intentionally the only accepted token count.
    API usage fields such as ``generation_total_tokens`` are not equivalent to
    SLIME chat-template tokens and must be computed before calling this helper.
    """

    # 未传配置时使用默认配置
    config = config or BaselineTokenGapConfig()
    # 把 token 计数安全转为有限浮点数
    token_value = maybe_float(slime_message_tokens)
    if token_value is None:
        # 缺少有效 token 计数时直接拒绝
        return BaselineTokenGapDecision(False, "missing_slime_message_tokens")
    # 第一道过滤：token 长度不能超过上限
    if not token_count_within_limit(
        token_value,
        max_total_tokens=config.max_total_tokens,
        keep_equal=config.keep_equal,
    ):
        # 超长样本拒绝，并记录其长度
        return BaselineTokenGapDecision(
            False,
            f"token_length>{config.max_total_tokens}",
            token_length=token_value,
        )

    # 取出嵌套的差距过滤配置
    gap_config = config.gap_config
    # 反馈文本为空时按空串处理
    feedback = feedback_text or ""
    # 第二道过滤：要求反馈表示任务成功（如训练正常完成）
    if gap_config.require_feedback_success and not feedback_is_success(feedback):
        return BaselineTokenGapDecision(
            False,
            "feedback_not_success",
            token_length=token_value,
        )

    # 第三道过滤：依据反馈与元数据计算相对差距（validation 相对 baseline 的泛化差距）
    relative_gap, gap_info = relative_gap_from_feedback(
        feedback,
        dict(metadata or {}),
        gap_config,
        task_score_scale=task_score_scale,
    )
    # 差距阈值：优先用评估信息中的实际阈值，缺失时退回配置值
    gap_threshold = float(gap_info.get("gap_threshold") or gap_config.max_relative_gap)
    if relative_gap is None:
        # 无法得到可比较的差距：若要求必须可比较则拒绝
        if gap_config.require_comparable:
            return BaselineTokenGapDecision(
                False,
                "missing_valid_test_gap",
                token_length=token_value,
                gap_info=gap_info,
            )
    elif relative_gap > gap_threshold:
        # 相对差距超过阈值则拒绝（样本泛化质量不达标）
        return BaselineTokenGapDecision(
            False,
            f"relative_gap>{gap_threshold}",
            token_length=token_value,
            gap_info=gap_info,
        )

    # 全部过滤通过：接受该样本
    return BaselineTokenGapDecision(
        True,
        "baseline_token_gap_ok",
        token_length=token_value,
        gap_info=gap_info,
    )
