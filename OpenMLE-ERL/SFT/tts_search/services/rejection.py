"""Rejection sampling policies for generated-code selection.

The generation loop only needs one question from this module: should an
evaluated generation be kept for the final dataset?  Keeping that decision
behind a policy makes it easy to swap filtering methods without changing
generation or sandbox evaluation code.
"""
# 拒绝采样（rejection sampling）策略模块：回答唯一的问题——
# 一条已评估的生成样本是否保留进最终数据集；把该决策抽象为可替换的策略。

from __future__ import annotations

# 导入 json：解析参考分数文件（JSON/JSONL）
import json
# 导入 math：NaN/inf 判断
import math
# 导入 dataclass：声明决策结构
from dataclasses import dataclass
# 导入 Path：文件路径处理（反馈文件、榜单文件）
from pathlib import Path
# 导入类型注解：Any/Mapping/Protocol
from typing import Any, Mapping, Protocol

# 导入基线过滤配置与评估函数（GLM 风格 token/有效测试 gap 过滤）
from tts_search.data_produce.baseline_filter import (
    BaselineTokenGapConfig,
    evaluate_baseline_token_gap,
)
# 导入 gap 过滤配置
from tts_search.data_produce.gap_filter import (
    GapFilterConfig,
)

# 可接受的奖牌标签集合：金/银/铜
MEDAL_LABELS = frozenset({"gold", "silver", "bronze"})


@dataclass(frozen=True)
class RejectionDecision:
    """Decision returned by a rejection policy."""

    # accepted=是否接受；reason=决策原因标签（便于日志与调试）
    accepted: bool
    reason: str


class RejectionPolicy(Protocol):
    """Interface for deciding whether an evaluated sample is accepted."""

    # 策略名称
    name: str

    def accepts_result(self, result: Any) -> RejectionDecision:
        """Return whether an EvaluationResult-like object should be accepted."""

    def accepts_record(
        self,
        record: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> RejectionDecision:
        """Return whether a serialized eval_results.jsonl record is accepted."""


class AcceptAllPolicy:
    """Accept every evaluated sample, including failed sandbox runs."""

    name = "all"

    def accepts_result(self, result: Any) -> RejectionDecision:
        # 无条件接受任何评估结果
        return RejectionDecision(True, "all")

    def accepts_record(
        self,
        record: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> RejectionDecision:
        # 序列化记录同样无条件接受
        return RejectionDecision(True, "all")


class AcceptScoredPolicy:
    """Accept every sample that has a numeric sandbox score."""

    name = "accept_scored"

    def accepts_result(self, result: Any) -> RejectionDecision:
        # 结果对象带分数即接受
        if getattr(result, "score", None) is not None:
            return RejectionDecision(True, "has_score")
        # 缺分数则拒绝
        return RejectionDecision(False, "missing_score")

    def accepts_record(
        self,
        record: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> RejectionDecision:
        # 序列化记录带分数即接受
        if record.get("score") is not None:
            return RejectionDecision(True, "has_score")
        return RejectionDecision(False, "missing_score")


class MedalPolicy:
    """Accept samples that earned one of the requested medal labels."""

    name = "medal"

    def __init__(self, accepted_medals: set[str] | None = None):
        # 未指定时默认接受全部奖牌（金银铜）
        medals = accepted_medals or set(MEDAL_LABELS)
        # 统一转小写存储，便于比较
        self._accepted_medals = {str(medal).lower() for medal in medals}

    def accepts_result(self, result: Any) -> RejectionDecision:
        # 取结果的奖牌标签并转小写
        medal = str(getattr(result, "submit_medal", "")).lower()
        # 命中可接受奖牌集合则接受
        if medal in self._accepted_medals:
            return RejectionDecision(True, f"medal:{medal}")
        # 否则拒绝（缺失时标记 missing）
        return RejectionDecision(False, f"medal:{medal or 'missing'}")

    def accepts_record(
        self,
        record: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> RejectionDecision:
        # 序列化记录的奖牌判断逻辑同上
        medal = str(record.get("submit_medal", "")).lower()
        if medal in self._accepted_medals:
            return RejectionDecision(True, f"medal:{medal}")
        return RejectionDecision(False, f"medal:{medal or 'missing'}")


class ScoreThresholdPolicy:
    """Accept samples whose sandbox score is at least a fixed threshold."""

    name = "score_threshold"

    def __init__(self, threshold: float):
        # 保存分数阈值
        self._threshold = float(threshold)

    def accepts_result(self, result: Any) -> RejectionDecision:
        # 从结果对象取分数做阈值判断
        return self._decide(getattr(result, "score", None))

    def accepts_record(
        self,
        record: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> RejectionDecision:
        # 从序列化记录取分数做阈值判断
        return self._decide(record.get("score"))

    def _decide(self, score: Any) -> RejectionDecision:
        # 缺分数：拒绝
        if score is None:
            return RejectionDecision(False, "missing_score")
        # 转浮点后与阈值比较
        score_float = float(score)
        # 达到阈值：接受
        if score_float >= self._threshold:
            return RejectionDecision(True, f"score>={self._threshold}")
        # 低于阈值：拒绝
        return RejectionDecision(False, f"score<{self._threshold}")


class RewardThresholdPolicy:
    """Accept samples whose reward is at least a fixed threshold."""

    name = "reward_threshold"

    def __init__(self, threshold: float):
        # 保存奖励阈值
        self._threshold = float(threshold)

    def accepts_result(self, result: Any) -> RejectionDecision:
        # 从结果对象取奖励做阈值判断
        return self._decide(getattr(result, "reward", None))

    def accepts_record(
        self,
        record: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> RejectionDecision:
        # 从序列化记录取奖励做阈值判断
        return self._decide(record.get("reward"))

    def _decide(self, reward: Any) -> RejectionDecision:
        # 缺奖励：拒绝
        if reward is None:
            return RejectionDecision(False, "missing_reward")
        # 转浮点后与阈值比较
        reward_float = float(reward)
        # 达到阈值：接受
        if reward_float >= self._threshold:
            return RejectionDecision(True, f"reward>={self._threshold}")
        # 低于阈值：拒绝
        return RejectionDecision(False, f"reward<{self._threshold}")


class BetterThanReferencePolicy:
    """Accept samples whose score beats a reference model score for the task."""

    name = "better_than_reference"

    def __init__(self, reference_scores: Mapping[str, float]):
        # 保存参考分数表（键统一转为字符串，值转浮点）
        self._reference_scores = {
            str(task_id): float(score) for task_id, score in reference_scores.items()
        }

    def accepts_result(self, result: Any) -> RejectionDecision:
        # 从结果对象提取任务标识、分数与元信息后统一决策
        return self._decide(
            task_id=getattr(result, "task_id", None),
            task_name=getattr(result, "task_name", None),
            score=getattr(result, "score", None),
            metadata=getattr(result, "metadata", None),
        )

    def accepts_record(
        self,
        record: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> RejectionDecision:
        # 从序列化记录提取同样字段后统一决策
        return self._decide(
            task_id=record.get("task_id"),
            task_name=record.get("task_name"),
            score=record.get("score"),
            metadata=metadata,
        )

    def _decide(
        self,
        *,
        task_id: Any,
        task_name: Any,
        score: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> RejectionDecision:
        # 缺分数：拒绝
        if score is None:
            return RejectionDecision(False, "missing_score")

        # 依次尝试用 task_id 与 task_name 匹配参考分数
        key_candidates = [str(task_id), str(task_name)]
        reference = None
        for key in key_candidates:
            if key in self._reference_scores:
                reference = self._reference_scores[key]
                break
        # 找不到参考分数：拒绝
        if reference is None:
            return RejectionDecision(False, "missing_reference_score")

        # 分数超过参考值（按指标方向）即接受
        score_float = float(score)
        if _score_beats_reference(score_float, reference, metadata):
            op = "<" if _lower_is_better(metadata) else ">"
            return RejectionDecision(True, f"score{op}{reference}")
        # 未超过参考值：拒绝
        op = ">=" if _lower_is_better(metadata) else "<="
        return RejectionDecision(False, f"score{op}{reference}")


class MixedLeaderboardBaselinePolicy:
    """Use medal rejection for leaderboard tasks and baseline-best otherwise."""

    name = "mixed"

    def __init__(
        self,
        reference_scores: Mapping[str, float],
        leaderboard_target: int = 2,
        no_leaderboard_target: int = 4,
        accepted_medals: set[str] | None = None,
    ):
        # 内部组合两个子策略：参考分数策略与奖牌策略
        self._reference_policy = BetterThanReferencePolicy(reference_scores)
        medals = accepted_medals or set(MEDAL_LABELS)
        self._medal_policy = MedalPolicy({str(medal).lower() for medal in medals})
        # 有榜单任务的接受目标数
        self._leaderboard_target = int(leaderboard_target)
        # 无榜单任务的接受目标数
        self._no_leaderboard_target = int(no_leaderboard_target)

    def has_leaderboard(self, metadata: Mapping[str, Any] | None) -> bool:
        # 判断任务是否带榜单（委托给辅助函数）
        return _metadata_has_leaderboard(metadata)

    def target_for_metadata(self, metadata: Mapping[str, Any] | None) -> int:
        # 按是否有榜单返回对应的接受目标数
        return (
            self._leaderboard_target
            if self.has_leaderboard(metadata)
            else self._no_leaderboard_target
        )

    def accepts_result(self, result: Any) -> RejectionDecision:
        # 有榜单任务用奖牌策略，否则用参考分数策略
        metadata = getattr(result, "metadata", None)
        if self.has_leaderboard(metadata):
            return self._medal_policy.accepts_result(result)
        return self._reference_policy._decide(
            task_id=getattr(result, "task_id", None),
            task_name=getattr(result, "task_name", None),
            score=getattr(result, "score", None),
            metadata=metadata,
        )

    def accepts_record(
        self,
        record: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> RejectionDecision:
        # 序列化记录同样按有无榜单分流
        if self.has_leaderboard(metadata):
            return self._medal_policy.accepts_record(record, metadata)
        return self._reference_policy._decide(
            task_id=record.get("task_id"),
            task_name=record.get("task_name"),
            score=record.get("score"),
            metadata=metadata,
        )


class BaselinePostprocessPolicy:
    """Apply GLM-4.7-style token and valid-test gap filters before a policy."""

    def __init__(
        self,
        inner: RejectionPolicy,
        *,
        max_total_tokens: int = 32768,
        max_relative_gap: float = 0.12,
        require_comparable_gap: bool = True,
    ):
        # 装饰器模式：先做基线过滤，通过后再交给内部策略
        self._inner = inner
        # 组装基线过滤配置：最大 token 数与相对 gap 上限
        self._baseline_config = BaselineTokenGapConfig(
            max_total_tokens=int(max_total_tokens),
            gap_config=GapFilterConfig(
                max_relative_gap=float(max_relative_gap),
                require_comparable=bool(require_comparable_gap),
            ),
        )
        # 策略名带内部策略名，便于追溯
        self.name = f"baseline_{inner.name}"

    def accepts_result(self, result: Any) -> RejectionDecision:
        # 先做基线（token/gap）过滤
        baseline = self._baseline_decision(
            slime_message_tokens=getattr(result, "slime_message_tokens", None),
            feedback=getattr(result, "feedback", None),
            metadata=getattr(result, "metadata", None),
        )
        if not baseline.accepted:
            # 基线未过直接拒绝
            return baseline
        # 基线通过后交给内部策略
        return self._inner.accepts_result(result)

    def accepts_record(
        self,
        record: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> RejectionDecision:
        # 序列化记录路径：先基线过滤（反馈文本从记录/文件还原）
        baseline = self._baseline_decision(
            slime_message_tokens=record.get("slime_message_tokens"),
            feedback=_feedback_from_record(record),
            metadata=metadata,
        )
        if not baseline.accepted:
            return baseline
        # 再交给内部策略
        return self._inner.accepts_record(record, metadata)

    def target_for_metadata(self, metadata: Mapping[str, Any] | None) -> int | None:
        # 内部策略若支持目标数查询则透传，否则返回 None
        target_fn = getattr(self._inner, "target_for_metadata", None)
        if target_fn is None:
            return None
        return int(target_fn(metadata))

    def _baseline_decision(
        self,
        *,
        slime_message_tokens: Any,
        feedback: str | None,
        metadata: Mapping[str, Any] | None,
    ) -> RejectionDecision:
        # 调用基线评估函数做 token/gap 过滤
        decision = evaluate_baseline_token_gap(
            slime_message_tokens=slime_message_tokens,
            feedback_text=feedback,
            metadata=metadata,
            config=self._baseline_config,
        )
        # 转换为本模块的决策类型
        return RejectionDecision(decision.accepted, decision.reason)


def _maybe_float(value: Any) -> float | None:
    # 安全转浮点：None/转换失败/NaN/inf 一律返回 None
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _range_size(low: Any, high: Any) -> float | None:
    # 计算区间宽度：任一端无效返回 None；宽度为 0 也视为无效
    low_float = _maybe_float(low)
    high_float = _maybe_float(high)
    if low_float is None or high_float is None:
        return None
    value = abs(high_float - low_float)
    return value if value > 0 else None


def _feedback_from_record(record: Mapping[str, Any]) -> str | None:
    # 从序列化记录还原反馈文本：优先内联字段
    feedback = record.get("feedback")
    if feedback is not None:
        return str(feedback)
    # 回退：读取 feedback_path 指向的文件
    feedback_path = record.get("feedback_path")
    if feedback_path:
        path = Path(str(feedback_path))
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")
    return None


def _lower_is_better(metadata: Mapping[str, Any] | None) -> bool:
    # 判断指标是否"越小越好"：无元信息默认越大越好
    if metadata is None:
        return False
    higher = metadata.get("higher_is_better")
    if isinstance(higher, str):
        # 字符串形式 "false"/"0"/"no" 视为越小越好
        return higher.strip().lower() in {"false", "0", "no"}
    # 布尔 False 表示越小越好
    return higher is False


def _score_beats_reference(
    score: float,
    reference: float,
    metadata: Mapping[str, Any] | None,
) -> bool:
    # 按指标方向判断分数是否优于参考值：越小越好用 <，否则用 >
    return score < reference if _lower_is_better(metadata) else score > reference


def _metadata_has_leaderboard(metadata: Mapping[str, Any] | None) -> bool:
    # 判断任务元信息是否带公开榜单
    if metadata is None:
        return False
    # 有榜单区间（min/max 有效且跨度>0）即视为有榜单
    if _range_size(metadata.get("leaderboard_min"), metadata.get("leaderboard_max")):
        return True

    # 否则检查磁盘上是否存在榜单文件（候选路径逐个尝试）
    leaderboard_dir = metadata.get("leaderboard_dir")
    task_name = metadata.get("task_name")
    data_dir = metadata.get("data_dir")
    candidates: list[Path] = []
    if leaderboard_dir and task_name:
        task_root = Path(str(leaderboard_dir)) / str(task_name)
        candidates.extend(
            [
                Path(str(leaderboard_dir)) / f"{task_name}.csv",
                task_root / "info" / "public_leaderboard.csv",
                task_root / "public_leaderboard.csv",
            ]
        )
    if data_dir:
        candidates.append(Path(str(data_dir)) / "info" / "public_leaderboard.csv")
    # 任一候选文件存在即认为有榜单
    return any(path.exists() for path in candidates)


def load_reference_scores(path: str | Path) -> dict[str, float]:
    """Load reference scores from JSON/JSONL.

    JSON may be either ``{"task_id": score}`` or a list of records.  JSONL
    records should contain ``task_id`` or ``task_name`` and ``score``.
    """

    # 构造参考分数文件路径
    reference_path = Path(path)
    if reference_path.suffix == ".jsonl":
        # JSONL：逐行解析，取 task_id/task_name 作为键
        scores: dict[str, float] = {}
        with open(reference_path, "r", encoding="utf-8") as f:
            for line in f:
                # 跳过空行
                if not line.strip():
                    continue
                row = json.loads(line)
                # 键优先 task_id，其次 task_name；必须有 score 才收录
                key = row.get("task_id") or row.get("task_name")
                if key is not None and row.get("score") is not None:
                    scores[str(key)] = float(row["score"])
        return scores

    # JSON：整体解析
    data = json.loads(reference_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # 字典形式：{"task_id": score} 直接转换
        return {str(key): float(value) for key, value in data.items()}
    if isinstance(data, list):
        # 列表形式：与 JSONL 相同的记录规则
        scores = {}
        for row in data:
            if not isinstance(row, dict):
                continue
            key = row.get("task_id") or row.get("task_name")
            if key is not None and row.get("score") is not None:
                scores[str(key)] = float(row["score"])
        return scores
    # 其他格式：报错
    raise ValueError(f"Unsupported reference score format: {reference_path}")


def build_rejection_policy(
    *,
    name: str | None,
    score_threshold: float | None = None,
    reward_threshold: float | None = None,
    reference_scores_path: str | Path | None = None,
    accepted_medals: list[str] | tuple[str, ...] | set[str] | None = None,
    apply_baseline_filters: bool = False,
    baseline_token_limit: int = 32768,
    baseline_relative_gap_limit: float = 0.12,
    mixed_leaderboard_target: int = 2,
    mixed_no_leaderboard_target: int = 4,
) -> RejectionPolicy:
    """Build a rejection policy from config values."""

    # 策略名规整为小写（缺省用 accept_scored）
    policy_name = (name or "accept_scored").lower()
    if policy_name in {"all", "accept_all", "none"}:
        # 全接受策略
        return AcceptAllPolicy()

    policy: RejectionPolicy
    if policy_name in {"success", "scored", "accept_scored", "all_scored"}:
        # 有分数即接受
        policy = AcceptScoredPolicy()
    elif policy_name in {"medal", "has_medal"}:
        # 奖牌策略：规整可接受奖牌集合
        medals = (
            {str(medal).lower() for medal in accepted_medals}
            if accepted_medals
            else None
        )
        policy = MedalPolicy(accepted_medals=medals)
    elif policy_name in {"score_threshold", "threshold"}:
        # 分数阈值策略：必须提供阈值
        if score_threshold is None:
            raise ValueError("score_threshold rejection policy requires a threshold")
        policy = ScoreThresholdPolicy(score_threshold)
    elif policy_name in {"reward_threshold", "reward"}:
        # 奖励阈值策略：未提供 reward_threshold 时回退用 score_threshold
        threshold = (
            reward_threshold if reward_threshold is not None else score_threshold
        )
        if threshold is None:
            raise ValueError("reward_threshold rejection policy requires a threshold")
        policy = RewardThresholdPolicy(threshold)
    elif policy_name in {"better_than_reference", "compare_reference"}:
        # 参考分数对比策略：必须提供参考分数文件
        if reference_scores_path is None:
            raise ValueError(
                "better_than_reference rejection policy requires reference_scores_path"
            )
        policy = BetterThanReferencePolicy(load_reference_scores(reference_scores_path))
    elif policy_name in {"mixed", "mixed_leaderboard_baseline"}:
        # 混合策略：榜单任务用奖牌，无榜单任务用参考分数
        medals = (
            {str(medal).lower() for medal in accepted_medals}
            if accepted_medals
            else None
        )
        reference_scores = (
            load_reference_scores(reference_scores_path)
            if reference_scores_path is not None
            else {}
        )
        policy = MixedLeaderboardBaselinePolicy(
            reference_scores,
            leaderboard_target=mixed_leaderboard_target,
            no_leaderboard_target=mixed_no_leaderboard_target,
            accepted_medals=medals,
        )
    else:
        # 未知策略名：报错
        raise ValueError(f"Unknown rejection policy: {name}")

    if apply_baseline_filters:
        # 开启基线过滤时用后处理策略包一层
        return BaselinePostprocessPolicy(
            policy,
            max_total_tokens=baseline_token_limit,
            max_relative_gap=baseline_relative_gap_limit,
        )
    return policy
