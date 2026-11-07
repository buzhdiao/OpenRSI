"""Validation-test gap extraction and filtering."""

# 让类型注解延迟求值，从而支持新版联合类型等注解语法
from __future__ import annotations

# 标准库导入：CSV 清单解析、JSON 协议解析、统计取整、正则、数据类、路径、类型注解
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# pandas 用于读取任务/指标元数据 parquet
import pandas as pd

# 公共工具：宽松数值解析、读文本、描述统计
from tts_search.data_produce.common import maybe_float, read_text, summary_stats

# 匹配反馈文本中的 “**Score**:” 行（提交/测试分数）
SCORE_RE = re.compile(r"^\*\*Score\*\*:\s*([-+eE0-9.]+)\s*$", re.MULTILINE)
# 匹配 “Final Score:” 行（另一种最终分数格式）
FINAL_SCORE_RE = re.compile(r"^Final Score:\s*([-+eE0-9.]+)\s*$", re.MULTILINE)
# 匹配 “##SCORE##” 前缀格式（早期 prompt 约定）
PREFIX_SCORE_RE = re.compile(r"##SCORE##([-+eE0-9.]+)")
# 匹配 “Final Validation Score:” 行（验证集分数）
VAL_SCORE_RE = re.compile(r"Final Validation Score:\s*([-+eE0-9.]+)")
# 匹配 “**Status**:” 行（沙箱执行状态）
STATUS_RE = re.compile(r"^\*\*Status\*\*:\s*(.+?)\s*$", re.MULTILINE)
# 匹配 “**Result**:” 行（沙箱执行结果标签）
RESULT_RE = re.compile(r"^\*\*Result\*\*:\s*(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class TaskMeta:
    """Task score-range metadata used by validation/test gap filtering.

    Args:
        task_name: Task name used for lookup.
        higher_is_better: Metric direction when known.
        theoretical_min: Theoretical minimum metric value.
        theoretical_max: Theoretical maximum metric value.
        leaderboard_min: Observed leaderboard minimum.
        leaderboard_max: Observed leaderboard maximum.
        task_id: Optional task UUID used for exact parquet metadata lookup.
        metric_*: Optional fields usually populated from the sibling
            ``*_metricprompt`` parquet.

    Returns:
        Immutable metadata object.
    """

    # 任务名（查找主键之一）
    task_name: str
    # 指标方向：True 表示越大越好；未知时为 None
    higher_is_better: bool | None
    # 理论最小值（用于构造分数区间）
    theoretical_min: float | None
    # 理论最大值
    theoretical_max: float | None
    # 排行榜观测到的最小值
    leaderboard_min: float | None
    # 排行榜观测到的最大值
    leaderboard_max: float | None
    # 任务 UUID（精确查找键）
    task_id: str | None = None
    # 指标短标签（如 rmsle），通常来自 metricprompt parquet
    metric_label: str | None = None
    # 指标全名
    metric_name: str | None = None
    # 指标方向描述（文本形式）
    metric_direction: str | None = None
    # 指标归一化类别（决定 gap 分母的选择规则）
    metric_class: str | None = None
    # 验证策略说明
    validation_strategy: str | None = None
    # 指标信息来源
    metric_source: str | None = None
    # 命中的原文片段（便于审计）
    matched_text: str | None = None


@dataclass(frozen=True)
class GapFilterConfig:
    """Configuration for relative validation/test gap filtering.

    Args:
        max_relative_gap: Maximum accepted relative gap.
        theoretical_small_range_max: Threshold for trusting small theoretical ranges.
        big_range_threshold: Threshold where ranges are treated as too large.
        no_range_is_big: Use score magnitudes when no range metadata exists.
        require_comparable: Drop rows where relative gap cannot be computed.
        require_feedback_success: Drop rows without successful feedback.
        unitless_loss_floor: Minimum denominator for unitless lower-is-better
            losses such as RMSLE, log loss, and NLL.
        use_metric_aware_denominator: Use the new metric-aware denominator rules.

    Returns:
        Immutable gap filtering configuration.
    """

    # 最大可接受相对 gap（gap / 分母），默认 12%
    max_relative_gap: float = 0.12
    # 理论区间宽度小于等于该阈值时视为“小区间”，可信任其作为分母
    theoretical_small_range_max: float = 2.0
    # 区间宽度超过该阈值视为“过大区间”，不可作为分母
    big_range_threshold: float = 100.0
    # 无区间元数据时是否按“大区间”处理（改用分数绝对值做分母）
    no_range_is_big: bool = True
    # 是否丢弃无法计算相对 gap 的行
    require_comparable: bool = True
    # 是否丢弃沙箱反馈不成功的行
    require_feedback_success: bool = True
    # 无量纲损失（RMSLE/log loss/NLL 等）的分母下限，防止除以过小的数
    unitless_loss_floor: float = 1.0
    # 是否启用新的指标感知分母规则（False 时回退旧版区间规则）
    use_metric_aware_denominator: bool = True


def load_task_meta(path: Path) -> dict[str, TaskMeta]:
    """Load task metadata from a CSV manifest.

    Args:
        path: CSV path containing task names and score range columns.

    Returns:
        Mapping from task name to TaskMeta.
    """
    # 任务名/UUID -> TaskMeta 的映射
    meta: dict[str, TaskMeta] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        # 用 DictReader 按表头逐行解析 CSV
        reader = csv.DictReader(f)
        for row in reader:
            # 指标方向默认未知
            higher = None
            if row.get("higher_is_better"):
                # “TRUE”（忽略大小写/空白）视为越大越好
                higher = row["higher_is_better"].strip().upper() == "TRUE"
            # 组装该任务的元数据对象（缺失字段宽松解析为 None）
            item = TaskMeta(
                task_name=row["task_name"],
                higher_is_better=higher,
                theoretical_min=maybe_float(row.get("theoretical_min")),
                theoretical_max=maybe_float(row.get("theoretical_max")),
                leaderboard_min=maybe_float(row.get("leaderboard_min")),
                leaderboard_max=maybe_float(row.get("leaderboard_max")),
                task_id=str(row.get("task_id") or row.get("uuid") or "") or None,
                metric_label=row.get("metric_label") or None,
                metric_name=row.get("metric_name") or None,
                metric_direction=row.get("metric_direction") or None,
                metric_class=classify_metric(
                    row.get("metric_label"), row.get("metric_name")
                ),
                validation_strategy=row.get("validation_strategy") or None,
                metric_source=row.get("metric_source") or None,
                matched_text=row.get("matched_text") or None,
            )
            # 以任务名为键登记
            meta[item.task_name] = item
            if item.task_id:
                # 同时以任务 UUID 为键登记，便于精确查找
                meta[item.task_id] = item
    return meta


def _parse_self_valid_protocol(value: Any) -> dict[str, Any]:
    """Return a normalized self-validation protocol dictionary."""
    # 本身就是字典时直接返回
    if isinstance(value, dict):
        return value
    # 字符串且非空白时尝试 JSON 解析
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            # 解析失败按空协议处理
            return {}
        # 解析成功且是字典才返回
        return parsed if isinstance(parsed, dict) else {}
    return {}


def classify_metric(metric_label: Any, metric_name: Any = None) -> str:
    """Classify a metric into a gap-normalization family.

    Args:
        metric_label: Short metric label, for example ``rmsle``.
        metric_name: Human-readable metric name.

    Returns:
        Metric class string used by denominator selection.
    """
    # 拼接标签与指标名并转小写，便于关键词匹配
    text = f"{metric_label or ''} {metric_name or ''}".lower()
    # 把下划线/连字符统一替换成空格，消除命名风格差异
    text = text.replace("_", " ").replace("-", " ")
    # SMAPE：有界百分比误差（值域 [0,2]），越小越好
    if "smape" in text:
        return "bounded_percentage_error_lower_better"
    # ROC-AUC 类：值域 [0,1] 的概率型分数
    if any(key in text for key in ("roc auc", "auroc", "auc")):
        return "bounded_probabilistic_score_0_1"
    # accuracy/F1/precision 等比例类指标：值域 [0,1] 或百分制
    if any(
        key in text
        for key in (
            "accuracy",
            "f1",
            "precision",
            "recall",
            "jaccard",
            "dice",
            "iou",
            "map@",
            " map ",
            "ndcg",
            "mrr",
        )
    ):
        return "bounded_score_0_1_or_percent"
    # QWK/Cohen's Kappa/MCC：一致性系数，值域 [-1,1]
    if any(
        key in text
        for key in ("qwk", "quadratic weighted kappa", "cohen", "kappa", "mcc")
    ):
        return "bounded_agreement_minus1_1"
    # Spearman/Pearson/R2：相关系数类，值域大致 [-1,1]
    if any(key in text for key in ("spearman", "pearson", "correlation", "r2")):
        return "bounded_correlation_or_r2"
    # RMSLE/MSLE：对数目标误差，数值近似无量纲
    if any(key in text for key in ("rmsle", "msle")):
        return "log_target_error_lower_better"
    # log loss/NLL/交叉熵：概率损失，数值近似无量纲
    if any(key in text for key in ("log loss", "logloss", "nll", "cross entropy")):
        return "probabilistic_loss_lower_better"
    # MAPE：百分比误差（天然按 100 归一）
    if "mape" in text:
        return "percentage_error_lower_better"
    # pinball/quantile/CRPS：目标量纲的分位数/分布损失
    if any(key in text for key in ("pinball", "quantile", "crps")):
        return "target_scale_distribution_loss_lower_better"
    # RMSE/MAE/MSE 等常规误差：目标量纲误差，越小越好
    if any(
        key in text
        for key in ("rmse", "mae", "mse", "rmspe", "rmsse", "error", "loss")
    ):
        return "target_scale_error_lower_better"
    # 无法识别的指标按无界处理
    return "unknown_or_unbounded"


def _metric_fields_from_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Extract metric classification fields from a parquet metadata row."""
    # 解析（可能是 JSON 字符串的）自验证协议字段
    protocol = _parse_self_valid_protocol(row.get("self_valid_protocol"))
    # 指标标签缺失时用指标名兜底
    label = protocol.get("metric_label") or protocol.get("metric_name")
    # 指标全名
    name = protocol.get("metric_name")
    # 从协议中抽取指标分类相关字段（标签统一小写；缺失处一律置 None）
    return {
        "metric_label": str(label).lower() if label else None,
        "metric_name": str(name) if name else None,
        "metric_direction": (
            str(protocol.get("metric_direction"))
            if protocol.get("metric_direction") is not None
            else None
        ),
        "metric_class": classify_metric(label, name) if label or name else None,
        "validation_strategy": (
            str(protocol.get("validation_strategy"))
            if protocol.get("validation_strategy") is not None
            else None
        ),
        "metric_source": (
            str(protocol.get("source")) if protocol.get("source") is not None else None
        ),
        "matched_text": (
            str(protocol.get("matched_text"))
            if protocol.get("matched_text") is not None
            else None
        ),
    }


def infer_metricprompt_parquet(path: Path) -> Path | None:
    """Infer a sibling ``*_metricprompt/train.parquet`` path when it exists."""
    # 只对名为 train.parquet 的源文件做兄弟目录推断
    if path.name != "train.parquet":
        return None
    # 源文件所在目录
    parent = path.parent
    if parent.name.endswith("_metricprompt"):
        # 本身已在 metricprompt 目录内：确认存在即返回自身
        return path if path.exists() else None
    # 否则推断兄弟目录 <parent>_metricprompt 下的同名文件
    candidate = parent.with_name(f"{parent.name}_metricprompt") / path.name
    # 候选文件存在才返回
    return candidate if candidate.exists() else None


def _load_metric_fields_from_parquet(path: Path) -> dict[str, dict[str, Any]]:
    """Load metric protocol fields keyed by task UUID and task name."""
    # UUID/任务名 -> 指标字段字典
    metric_fields: dict[str, dict[str, Any]] = {}
    # 文件不存在时返回空映射
    if not path.exists():
        return metric_fields
    # 只读 metadata 列，降低 IO 开销
    df = pd.read_parquet(path, columns=["metadata"])
    for value in df["metadata"]:
        # 每行 metadata 转成字典
        row = dict(value)
        # 抽取该任务的指标协议字段
        fields = _metric_fields_from_metadata(row)
        if not any(fields.values()):
            # 全部字段为空时跳过该行
            continue
        # 任务 UUID（task_id 兜底）
        task_id = str(row.get("uuid") or row.get("task_id") or "") or None
        # 任务名
        task_name = str(row.get("task_name") or "") or None
        if task_id:
            # UUID 作主键（可覆盖）
            metric_fields[task_id] = fields
        if task_name:
            # 任务名作次级键（不覆盖已有映射）
            metric_fields.setdefault(task_name, fields)
    return metric_fields


def enrich_metadata_with_metric_protocol(
    metadata: dict[str, Any],
    metric_fields: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Attach metric protocol fields to one task metadata dictionary."""
    # 任务 UUID（task_id 兜底）
    task_id = str(metadata.get("uuid") or metadata.get("task_id") or "")
    # 任务名
    task_name = str(metadata.get("task_name") or "")
    # 优先按 UUID 匹配指标字段，其次按任务名
    fields = metric_fields.get(task_id) or metric_fields.get(task_name)
    if fields is None:
        # 找不到对应指标协议时原样返回
        return metadata
    # 用 setdefault 把 7 个指标协议字段补充进元数据（已有值不被覆盖）
    metadata.setdefault("metric_label", fields.get("metric_label"))
    metadata.setdefault("metric_name", fields.get("metric_name"))
    metadata.setdefault("metric_direction", fields.get("metric_direction"))
    metadata.setdefault("metric_class", fields.get("metric_class"))
    metadata.setdefault("validation_strategy", fields.get("validation_strategy"))
    metadata.setdefault("metric_source", fields.get("metric_source"))
    metadata.setdefault("matched_text", fields.get("matched_text"))
    if "self_valid_protocol" not in metadata:
        # 元数据中没有完整协议时，重组一份 self_valid_protocol 字典写入
        metadata["self_valid_protocol"] = {
            "metric_label": fields.get("metric_label"),
            "metric_name": fields.get("metric_name"),
            "metric_direction": fields.get("metric_direction"),
            "validation_strategy": fields.get("validation_strategy"),
            "source": fields.get("metric_source"),
            "matched_text": fields.get("matched_text"),
        }
    return metadata


def enrich_tasks_with_metric_protocol(
    tasks: list[dict[str, Any]],
    *,
    source_parquet: Path | None = None,
    metric_metadata_parquet: Path | None = None,
    metadata_key: str = "metadata",
) -> int:
    """Enrich loaded task records with metric protocol from metricprompt parquet.

    Args:
        tasks: Task records loaded from the generation parquet.
        source_parquet: Official source parquet path. Used to infer sibling
            metricprompt path when ``metric_metadata_parquet`` is not provided.
        metric_metadata_parquet: Explicit metricprompt parquet path.
        metadata_key: Metadata key in each task record.

    Returns:
        Number of task records that received metric fields.
    """
    # 显式给定的 metricprompt 路径优先
    metric_path = metric_metadata_parquet
    if metric_path is None and source_parquet is not None:
        # 未指定时按兄弟目录约定推断
        metric_path = infer_metricprompt_parquet(source_parquet)
    if metric_path is None:
        # 找不到指标元数据源时无事可做
        return 0
    # 一次性加载全部任务的指标字段映射
    metric_fields = _load_metric_fields_from_parquet(metric_path)
    # 成功附加字段的任务计数
    enriched = 0
    for task in tasks:
        # 取该任务记录的元数据
        metadata = task.get(metadata_key)
        if not isinstance(metadata, dict):
            # 元数据缺失或类型不对则跳过
            continue
        # 记录增强前的 metric_class
        before = metadata.get("metric_class")
        # 原地补充指标协议字段
        enrich_metadata_with_metric_protocol(metadata, metric_fields)
        # 记录增强后的 metric_class
        after = metadata.get("metric_class")
        if before is None and after is not None:
            # 从无到有才计入增强数
            enriched += 1
    return enriched


def load_task_meta_from_parquet(
    path: Path,
    *,
    metadata_col: str = "metadata",
    metric_metadata_parquet: Path | None = None,
    enrich_metric_protocol: bool = True,
) -> dict[str, TaskMeta]:
    """Load task score-range metadata from a source parquet.

    Args:
        path: Parquet path containing a metadata column with task UUID and ranges.
        metadata_col: Name of the metadata column.
        metric_metadata_parquet: Optional metricprompt parquet containing
            ``self_valid_protocol``. When omitted, a sibling ``*_metricprompt``
            parquet is used if it exists.
        enrich_metric_protocol: Attach metric fields from the metricprompt
            parquet when available.

    Returns:
        Mapping keyed primarily by task UUID, with task-name fallback keys.
    """
    # UUID/任务名 -> TaskMeta 的映射
    meta: dict[str, TaskMeta] = {}
    # 指标协议字段映射（可选加载）
    metric_fields: dict[str, dict[str, Any]] = {}
    if enrich_metric_protocol:
        # 显式路径优先，否则按兄弟目录约定推断
        metric_path = metric_metadata_parquet or infer_metricprompt_parquet(path)
        if metric_path is not None:
            metric_fields = _load_metric_fields_from_parquet(metric_path)
    # 只读元数据列
    df = pd.read_parquet(path, columns=[metadata_col])
    for value in df[metadata_col]:
        # 每行 metadata 转字典
        row = dict(value)
        # 任务 UUID（task_id 兜底）
        task_id = str(row.get("uuid") or row.get("task_id") or "") or None
        # 任务名
        task_name = str(row.get("task_name") or "")
        # 先尝试从本行元数据解析指标字段
        fields = _metric_fields_from_metadata(row)
        if not any(fields.values()):
            # 本行缺失时回退到 metricprompt parquet 的映射
            fields = metric_fields.get(task_id or "") or metric_fields.get(task_name) or {}
        # 解析指标方向：字符串按常见真值识别
        higher = row.get("higher_is_better")
        if isinstance(higher, str):
            higher_bool = higher.strip().lower() in {"true", "1", "yes"}
        elif higher is None:
            # 缺失为未知方向
            higher_bool = None
        else:
            # 非字符串按布尔转换
            higher_bool = bool(higher)
        # 组装该任务的元数据对象
        item = TaskMeta(
            task_name=task_name,
            higher_is_better=higher_bool,
            theoretical_min=maybe_float(row.get("theoretical_min")),
            theoretical_max=maybe_float(row.get("theoretical_max")),
            leaderboard_min=maybe_float(row.get("leaderboard_min")),
            leaderboard_max=maybe_float(row.get("leaderboard_max")),
            task_id=task_id,
            metric_label=fields.get("metric_label"),
            metric_name=fields.get("metric_name"),
            metric_direction=fields.get("metric_direction"),
            metric_class=fields.get("metric_class"),
            validation_strategy=fields.get("validation_strategy"),
            metric_source=fields.get("metric_source"),
            matched_text=fields.get("matched_text"),
        )
        if task_id:
            # UUID 作主键（优先）
            meta[task_id] = item
        if task_name and task_name not in meta:
            # 任务名作次级键（不覆盖已有条目）
            meta[task_name] = item
    return meta


def lookup_task_meta(
    task_name: str,
    meta_map: dict[str, TaskMeta],
    *,
    task_id: str | None = None,
) -> TaskMeta | None:
    """Look up task metadata with optional version suffix stripping.

    Args:
        task_name: Task name from a rollout row.
        meta_map: Metadata mapping produced by ``load_task_meta``.
        task_id: Optional task UUID. This is preferred when present.

    Returns:
        Matching TaskMeta or None.
    """
    # 有 UUID 且命中时优先按 UUID 精确查找
    if task_id and task_id in meta_map:
        return meta_map[task_id]
    # 其次按任务名直接查找
    if task_name in meta_map:
        return meta_map[task_name]
    # 去掉末尾的版本后缀（如 task@3 中的 @3）再查一次
    stripped = re.sub(r"@\d+$", "", task_name)
    return meta_map.get(stripped)


def extract_submission_score(feedback_text: str) -> float | None:
    """Extract final submission/test score from feedback text.

    Args:
        feedback_text: Feedback log text.

    Returns:
        Parsed score or None when no score is found.
    """
    # 依次尝试三种分数格式（**Score** / Final Score / ##SCORE##）
    for pattern in (SCORE_RE, FINAL_SCORE_RE, PREFIX_SCORE_RE):
        match = pattern.search(feedback_text)
        if match:
            # 命中即解析数值并返回（宽松解析，失败为 None）
            return maybe_float(match.group(1))
    # 都未命中视为没有测试分数
    return None


def extract_validation_score(feedback_text: str) -> float | None:
    """Extract validation score from feedback text.

    Args:
        feedback_text: Feedback log text.

    Returns:
        Parsed validation score or None.
    """
    # 按验证分数正则查找
    match = VAL_SCORE_RE.search(feedback_text)
    # 命中则解析数值，否则返回 None
    return maybe_float(match.group(1)) if match else None


def extract_feedback_status(feedback_text: str) -> str:
    """Extract sandbox status from formatted feedback."""
    # 查找 **Status** 行
    match = STATUS_RE.search(feedback_text)
    # 命中则归一化为小写并去空白，未命中记 unknown
    return match.group(1).strip().lower() if match else "unknown"


def extract_feedback_result(feedback_text: str) -> str:
    """Extract sandbox result label from formatted feedback."""
    # 查找 **Result** 行
    match = RESULT_RE.search(feedback_text)
    # 命中则归一化为小写，未命中记 unknown
    return match.group(1).strip().lower() if match else "unknown"


def feedback_is_success(
    feedback_text: str,
    *,
    status_values: set[str] | None = None,
    result_values: set[str] | None = None,
) -> bool:
    """Check whether feedback represents a successful run."""
    # 状态白名单默认 completed/success
    status_values = status_values or {"completed", "success"}
    # 结果白名单默认 success
    result_values = result_values or {"success"}
    # 状态与结果需同时在白名单内才算成功运行
    return (
        extract_feedback_status(feedback_text) in status_values
        and extract_feedback_result(feedback_text) in result_values
    )


def _range_size(low: float | None, high: float | None) -> float | None:
    """Compute a positive score range width.

    Args:
        low: Lower bound.
        high: Upper bound.

    Returns:
        Positive absolute range size, or None when unavailable.
    """
    # 任一边界缺失则无法得到区间宽度
    if low is None or high is None:
        return None
    # 取差的绝对值
    value = abs(high - low)
    # 宽度必须为正（0 视为无效）
    return value if value > 0 else None


def _metadata_range_denominator(
    *,
    metadata: TaskMeta | None,
    config: GapFilterConfig,
) -> tuple[float | None, str]:
    """Return the legacy range-based denominator candidate."""
    # 理论分数区间宽度（无元数据时为 None）
    theoretical = (
        None
        if metadata is None
        else _range_size(metadata.theoretical_min, metadata.theoretical_max)
    )
    # 排行榜观测区间宽度
    leaderboard = (
        None
        if metadata is None
        else _range_size(metadata.leaderboard_min, metadata.leaderboard_max)
    )
    if theoretical is not None and theoretical <= config.theoretical_small_range_max:
        # 理论区间足够小：直接信任它作为分母
        return theoretical, "theoretical_range_le_small_threshold"
    # 否则优先排行榜区间，其次理论区间
    candidate = leaderboard or theoretical
    if candidate is None:
        # 两种区间都缺失
        return None, "missing_range"
    if candidate > config.big_range_threshold:
        # 区间过大：不可信，拒绝作为分母
        return None, "range_gt_big_threshold"
    if leaderboard is not None:
        # 排行榜区间可用
        return leaderboard, "leaderboard_range"
    # 理论区间大于小阈值但未超上限时也可用
    return theoretical, "theoretical_range_gt_small_threshold"


def _score_scale(
    validation_score: float | None,
    test_score: float | None,
    *,
    use_abs: bool = True,
) -> float | None:
    """Return a positive per-row score scale."""
    # 收集两个分数中的非空值
    values = [value for value in (validation_score, test_score) if value is not None]
    if not values:
        # 两个分数都缺失
        return None
    if use_abs:
        # 取绝对值后的最大者（适配负向损失等场景）
        scale = max(abs(value) for value in values)
    else:
        # 直接取最大者
        scale = max(values)
    # 尺度必须为正才有效
    return scale if scale > 0 else None


def _score_looks_percent_scale(
    validation_score: float | None,
    test_score: float | None,
) -> bool:
    """Heuristically detect 0-100 metric scores."""
    # 用行内最大分数绝对值做尺度
    scale = _score_scale(validation_score, test_score, use_abs=True)
    # 尺度明显大于 2 时按百分制（0-100）判定
    return bool(scale is not None and scale > 2.0)


def _metric_class(metadata: TaskMeta | None) -> str:
    # 无元数据时按无界/未知处理
    if metadata is None:
        return "unknown_or_unbounded"
    if metadata.metric_class:
        # 已有预计算的类别直接使用
        return metadata.metric_class
    # 否则现场根据指标标签/名称分类
    return classify_metric(metadata.metric_label, metadata.metric_name)


def metric_gap_threshold(
    metadata: TaskMeta | None,
    config: GapFilterConfig,
) -> float:
    """Return the relative-gap threshold for a task metric."""
    # 取指标类别
    metric_class = _metric_class(metadata)
    if metric_class in {"bounded_agreement_minus1_1", "bounded_correlation_or_r2"}:
        # With a [-1, 1] denominator of 2, this keeps the raw difference bound
        # at roughly 0.12 instead of allowing a 0.24 score swing.
        # [-1,1] 类指标分母为 2：把阈值减半，保证原始分数差仍约束在约 0.12
        return min(config.max_relative_gap, config.max_relative_gap / 2.0)
    # 其余类别沿用统一阈值
    return config.max_relative_gap


def choose_legacy_gap_denominator(
    *,
    metadata: TaskMeta | None,
    validation_score: float | None,
    test_score: float | None,
    config: GapFilterConfig,
) -> tuple[float | None, str]:
    """Choose denominator using the previous final-run range rules.

    Args:
        metadata: Optional score-range metadata for the task.
        validation_score: Parsed validation score.
        test_score: Parsed submission/test score.
        config: Gap filtering configuration.

    Returns:
        Denominator value and a string explaining the selected source.
    """
    # 先走区间元数据的分母判定
    denom, source = _metadata_range_denominator(metadata=metadata, config=config)
    if denom is not None:
        # 区间可用则直接返回
        return denom, source
    # 回退尺度：验证/测试分数（缺省按 0）中的最大者
    score_range = max(validation_score or 0.0, test_score or 0.0)
    if source == "range_gt_big_threshold":
        if score_range > 0:
            # 区间过大时改用行内分数尺度做分母
            return score_range, "max_abs_valid_test_for_big_range"
        return None, "nonpositive_score_range_for_big_range"
    if config.no_range_is_big and score_range > 0:
        # 缺失区间视同大区间：同样用行内分数尺度
        return score_range, "max_abs_valid_test_for_missing_range"
    # 无法确定分母
    return None, "missing_range"


def choose_gap_denominator(
    *,
    metadata: TaskMeta | None,
    validation_score: float | None,
    test_score: float | None,
    config: GapFilterConfig,
    task_score_scale: float | None = None,
) -> tuple[float | None, str]:
    """Choose a metric-aware denominator for validation/test gap.

    The previous range-based method is preserved in
    ``choose_legacy_gap_denominator``. This function is the default for both
    data production and generation-time rejection.
    """
    if not config.use_metric_aware_denominator:
        # 配置关闭指标感知时退回旧版区间规则
        return choose_legacy_gap_denominator(
            metadata=metadata,
            validation_score=validation_score,
            test_score=test_score,
            config=config,
        )

    # 指标类别（决定分母策略）
    metric_class = _metric_class(metadata)
    # 行内分数尺度（验证/测试分数绝对值最大者）
    row_scale = _score_scale(validation_score, test_score, use_abs=True)
    # 区间元数据分母及其来源说明
    metadata_denom, metadata_source = _metadata_range_denominator(
        metadata=metadata, config=config
    )

    if metric_class in {
        "bounded_probabilistic_score_0_1",
        "bounded_score_0_1_or_percent",
    }:
        if _score_looks_percent_scale(validation_score, test_score):
            # 分数呈百分制：用 [0,100] 区间宽度做分母
            return 100.0, f"metric_aware:{metric_class}:percent_range_100"
        # 否则按 [0,1] 区间
        return 1.0, f"metric_aware:{metric_class}:unit_range_1"

    if metric_class in {"bounded_agreement_minus1_1", "bounded_correlation_or_r2"}:
        # [-1,1] 类指标：区间宽度固定为 2
        return 2.0, f"metric_aware:{metric_class}:minus1_1_range_2"

    if metric_class == "bounded_percentage_error_lower_better":
        if _score_looks_percent_scale(validation_score, test_score):
            # SMAPE 呈百分制：按 100 归一
            return 100.0, "metric_aware:smape:percent_range_100"
        # 否则按小数比例制
        return 1.0, "metric_aware:smape:fraction_range_1"

    if metric_class in {
        "log_target_error_lower_better",
        "probabilistic_loss_lower_better",
    }:
        # 无量纲损失：分母取行内尺度与下限（默认 1.0）的较大者，避免除以过小值
        scale = max(row_scale or 0.0, float(config.unitless_loss_floor))
        return scale, f"metric_aware:{metric_class}:unitless_loss_floor"

    if metric_class == "percentage_error_lower_better":
        if _score_looks_percent_scale(validation_score, test_score):
            # MAPE 呈百分制：按 100 归一
            return 100.0, "metric_aware:mape:percent_range_100"
        # 否则按小数比例制
        return 1.0, "metric_aware:mape:fraction_range_1"

    if metric_class in {
        "target_scale_error_lower_better",
        "target_scale_distribution_loss_lower_better",
    }:
        if metadata_denom is not None:
            # 目标量纲指标：优先用区间元数据分母
            return metadata_denom, f"metric_aware:{metric_class}:{metadata_source}"
        if task_score_scale is not None and task_score_scale > 0:
            # 次选：任务级分数尺度（同类任务的高分位聚合值）
            return task_score_scale, f"metric_aware:{metric_class}:task_score_scale"
        if row_scale is not None:
            # 再次：行内分数尺度
            return row_scale, f"metric_aware:{metric_class}:row_score_scale"
        # 无可用尺度
        return None, f"metric_aware:{metric_class}:missing_scale"

    if metadata_denom is not None:
        # 未知/无界指标：仍优先区间元数据
        return metadata_denom, f"metric_aware:unknown:{metadata_source}"
    if task_score_scale is not None and task_score_scale > 0:
        # 其次任务级分数尺度
        return task_score_scale, "metric_aware:unknown:task_score_scale"
    if row_scale is not None and config.no_range_is_big:
        # 允许把缺失区间视作大区间时可用行内尺度
        return row_scale, "metric_aware:unknown:row_score_scale"
    # 无法确定分母
    return None, "metric_aware:unknown:missing_scale"


def relative_gap_from_scores(
    *,
    validation_score: float | None,
    test_score: float | None,
    metadata: TaskMeta | None,
    config: GapFilterConfig | None = None,
    task_score_scale: float | None = None,
) -> tuple[float | None, float | None, float | None, str, float]:
    """Compute abs gap, relative gap, denominator source, and threshold."""
    # 未给配置时使用默认配置
    config = config or GapFilterConfig()
    # 绝对 gap = |测试分数 - 验证分数|（两者齐全才可计算）
    abs_gap = (
        abs(test_score - validation_score)
        if test_score is not None and validation_score is not None
        else None
    )
    # 按指标类别选择归一分母及来源说明
    denom, denom_source = choose_gap_denominator(
        metadata=metadata,
        validation_score=validation_score,
        test_score=test_score,
        config=config,
        task_score_scale=task_score_scale,
    )
    # 相对 gap = 绝对 gap / 分母（分母必须为正）
    relative_gap = (
        abs_gap / denom
        if abs_gap is not None and denom is not None and denom > 0
        else None
    )
    # 连同该指标允许的 gap 阈值一起返回
    return abs_gap, relative_gap, denom, denom_source, metric_gap_threshold(metadata, config)


def relative_gap_from_feedback(
    feedback_text: str,
    metadata: TaskMeta | dict[str, Any] | None,
    config: GapFilterConfig | None = None,
    *,
    task_score_scale: float | None = None,
) -> tuple[float | None, dict[str, Any]]:
    """Compute metric-aware relative gap from feedback text and metadata."""
    # 未给配置时使用默认配置
    config = config or GapFilterConfig()
    # 元数据可能是 dict（生成时场景），统一转成 TaskMeta
    task_meta = metadata_to_task_meta(metadata)
    # 从反馈文本解析测试（提交）分数
    test_score = extract_submission_score(feedback_text)
    # 从反馈文本解析验证分数
    validation_score = extract_validation_score(feedback_text)
    # 计算绝对/相对 gap、分母与阈值
    abs_gap, relative_gap, denom, source, threshold = relative_gap_from_scores(
        validation_score=validation_score,
        test_score=test_score,
        metadata=task_meta,
        config=config,
        task_score_scale=task_score_scale,
    )
    # 返回相对 gap 及一组诊断注解字段（供审计与统计）
    return relative_gap, {
        "validation_score": validation_score,
        "test_score": test_score,
        "abs_gap": abs_gap,
        "relative_gap": relative_gap,
        "gap_denominator": denom,
        "gap_denominator_source": source,
        "gap_threshold": threshold,
        "metric_class": None if task_meta is None else task_meta.metric_class,
        "metric_label": None if task_meta is None else task_meta.metric_label,
        "metric_name": None if task_meta is None else task_meta.metric_name,
    }


def metadata_to_task_meta(metadata: TaskMeta | dict[str, Any] | None) -> TaskMeta | None:
    """Convert generation-time metadata dicts to ``TaskMeta``."""
    # 已经是 TaskMeta 或 None 时无需转换
    if metadata is None or isinstance(metadata, TaskMeta):
        return metadata
    # 先从元数据里的 self_valid_protocol 抽取指标字段
    fields = _metric_fields_from_metadata(metadata)
    # 指标标签：元数据优先，协议字段兜底
    metric_label = metadata.get("metric_label") or fields.get("metric_label")
    # 指标全名同理
    metric_name = metadata.get("metric_name") or fields.get("metric_name")
    # 指标类别：显式值 -> 协议字段 -> 现场分类
    metric_class = (
        metadata.get("metric_class")
        or fields.get("metric_class")
        or classify_metric(metric_label, metric_name)
    )
    # 解析指标方向：字符串按常见真值识别
    higher = metadata.get("higher_is_better")
    if isinstance(higher, str):
        higher_bool = higher.strip().lower() in {"true", "1", "yes"}
    elif higher is None:
        # 缺失为未知
        higher_bool = None
    else:
        # 非字符串按布尔转换
        higher_bool = bool(higher)
    # 组装 TaskMeta（字典中缺失的区间字段宽松解析为 None）
    return TaskMeta(
        task_name=str(metadata.get("task_name") or ""),
        higher_is_better=higher_bool,
        theoretical_min=maybe_float(metadata.get("theoretical_min")),
        theoretical_max=maybe_float(metadata.get("theoretical_max")),
        leaderboard_min=maybe_float(metadata.get("leaderboard_min")),
        leaderboard_max=maybe_float(metadata.get("leaderboard_max")),
        task_id=str(metadata.get("uuid") or metadata.get("task_id") or "") or None,
        metric_label=str(metric_label).lower() if metric_label else None,
        metric_name=str(metric_name) if metric_name else None,
        metric_direction=metadata.get("metric_direction") or fields.get("metric_direction"),
        metric_class=metric_class,
        validation_strategy=metadata.get("validation_strategy")
        or fields.get("validation_strategy"),
        metric_source=metadata.get("metric_source") or fields.get("metric_source"),
        matched_text=metadata.get("matched_text") or fields.get("matched_text"),
    )


def annotate_gap_rows(
    rows: list[dict[str, Any]],
    task_meta: dict[str, TaskMeta],
    config: GapFilterConfig | None = None,
) -> list[dict[str, Any]]:
    """Annotate rows with validation/test gap fields.

    Args:
        rows: Candidate SFT rows with feedback paths.
        task_meta: Task metadata mapping for score-range normalization.
        config: Optional gap filtering configuration.

    Returns:
        New rows with score, gap, denominator, and feedback annotations.
    """
    # 未给配置时使用默认配置
    config = config or GapFilterConfig()
    # 第一遍：缓存每行解析出的 (测试分数, 验证分数)
    parsed_scores: list[tuple[float | None, float | None]] = []
    # 任务级分数尺度（75 分位数），供 gap 分母回退使用
    task_scales: dict[str, float] = {}
    # 每个任务收集到的行内分数尺度列表
    scale_values_by_task: dict[str, list[float]] = {}
    for row in rows:
        # 读取反馈文本
        feedback_text = read_text(Path(str(row.get("feedback_path", ""))))
        # 解析测试（提交）分数
        test_score = extract_submission_score(feedback_text)
        # 解析验证分数
        validation_score = extract_validation_score(feedback_text)
        # 缓存解析结果（顺序与 rows 对应）
        parsed_scores.append((test_score, validation_score))
        # 该行的分数尺度（绝对值最大者）
        row_scale = _score_scale(validation_score, test_score, use_abs=True)
        if row_scale is None:
            # 分数全缺失时无法参与任务尺度统计
            continue
        # 任务键（UUID 优先，其次任务名）
        task_key = str(row.get("task_id") or row.get("task_name") or "")
        if task_key:
            # 归入对应任务的名下
            scale_values_by_task.setdefault(task_key, []).append(row_scale)
    for task_key, values in scale_values_by_task.items():
        # 升序排序后取约 75 分位数作为该任务的分数尺度（稳健、抗离群）
        sorted_values = sorted(values)
        # 分位下标：向上取整乘 0.75 再减一，并夹在合法范围内
        idx = min(len(sorted_values) - 1, int(math.ceil(0.75 * len(sorted_values))) - 1)
        task_scales[task_key] = sorted_values[max(0, idx)]

    # 第二遍：逐行做 gap 注解
    annotated_rows: list[dict[str, Any]] = []
    for row, (test_score, validation_score) in zip(rows, parsed_scores, strict=True):
        # 反馈文件路径
        feedback_path = Path(str(row.get("feedback_path", "")))
        # 重新读取反馈文本（用于状态/结果注解）
        feedback_text = read_text(feedback_path)
        # 查找该任务的分数区间元数据（UUID 优先）
        meta = lookup_task_meta(
            str(row.get("task_name", "")),
            task_meta,
            task_id=str(row.get("task_id") or "") or None,
        )
        # 任务键（与第一遍一致）
        task_key = str(row.get("task_id") or row.get("task_name") or "")
        # 计算 gap、分母与阈值（带任务级尺度回退）
        abs_gap, relative_gap, denom, denom_source, threshold = relative_gap_from_scores(
            metadata=meta,
            validation_score=validation_score,
            test_score=test_score,
            config=config,
            task_score_scale=task_scales.get(task_key),
        )
        # 拷贝原行避免就地修改
        annotated = dict(row)
        # 合并全部注解字段：状态/成功标记/分数/gap/分母来源/阈值与任务元数据
        annotated.update(
            {
                "feedback_status": extract_feedback_status(feedback_text),
                "feedback_result": extract_feedback_result(feedback_text),
                "feedback_success": feedback_is_success(feedback_text),
                "validation_score": validation_score,
                "test_score": test_score,
                "abs_gap": abs_gap,
                "relative_gap": relative_gap,
                "gap_denominator": denom,
                "gap_denominator_source": denom_source,
                "gap_threshold": threshold,
                "task_score_scale": task_scales.get(task_key),
                "gap_metadata_task_id": None if meta is None else meta.task_id,
                "gap_metadata_task_name": None if meta is None else meta.task_name,
                "higher_is_better": None if meta is None else meta.higher_is_better,
                "theoretical_min": None if meta is None else meta.theoretical_min,
                "theoretical_max": None if meta is None else meta.theoretical_max,
                "leaderboard_min": None if meta is None else meta.leaderboard_min,
                "leaderboard_max": None if meta is None else meta.leaderboard_max,
                "metric_label": None if meta is None else meta.metric_label,
                "metric_name": None if meta is None else meta.metric_name,
                "metric_class": None if meta is None else meta.metric_class,
                "metric_direction": None if meta is None else meta.metric_direction,
                "validation_strategy": (
                    None if meta is None else meta.validation_strategy
                ),
                "metric_source": None if meta is None else meta.metric_source,
            }
        )
        # 收集注解完成的行
        annotated_rows.append(annotated)
    return annotated_rows


def filter_by_relative_gap(
    rows: list[dict[str, Any]],
    config: GapFilterConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Split annotated rows by relative gap rules.

    Args:
        rows: Rows produced by ``annotate_gap_rows``.
        config: Gap filtering configuration.

    Returns:
        Kept rows, dropped rows, and summary statistics.
    """
    # 保留与丢弃两个桶
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for row in rows:
        # relative_gap 可计算才视为“可比”
        comparable = row.get("relative_gap") is not None
        # 使用行内阈值（按指标类别可能减半），缺失时回退全局阈值
        threshold = float(row.get("gap_threshold") or config.max_relative_gap)
        # 默认保留，以下三条规则逐条否决
        keep = True
        if config.require_comparable and not comparable:
            # 要求可比但 gap 不可计算：丢弃
            keep = False
        if config.require_feedback_success and not row.get("feedback_success", False):
            # 沙箱反馈不成功：丢弃
            keep = False
        if comparable and float(row["relative_gap"]) > threshold:
            # 相对 gap 超过阈值（验证集过拟合风险）：丢弃
            keep = False
        # 拷贝原行并记录过滤决策
        annotated = dict(row)
        annotated["kept_gap_filter"] = bool(keep)
        if keep:
            kept.append(annotated)
        else:
            dropped.append(annotated)

    # 汇总过滤统计：前后行数、可比行数及 gap 的描述统计
    stats = {
        "max_relative_gap": config.max_relative_gap,
        "before": len(rows),
        "after": len(kept),
        "dropped": len(dropped),
        "comparable_rows": sum(
            1 for row in rows if row.get("relative_gap") is not None
        ),
        "relative_gap": summary_stats(
            [
                float(row["relative_gap"])
                for row in rows
                if row.get("relative_gap") is not None
            ]
        ),
        "abs_gap": summary_stats(
            [float(row["abs_gap"]) for row in rows if row.get("abs_gap") is not None]
        ),
    }
    # 返回保留行、丢弃行与统计信息
    return kept, dropped, stats
