"""Reward deduplication, top-k selection, and old/new dataset merge helpers."""

# 让类型注解延迟求值，从而支持新版联合类型等注解语法
from __future__ import annotations

# 类型注解工具：TYPE_CHECKING 仅在静态检查时导入，避免运行时循环依赖
from typing import TYPE_CHECKING, Any

# pandas 用于按任务分组、去重与 top-k 截取
import pandas as pd

# 仅类型检查时导入拒绝策略类型（不参与运行时导入图）
if TYPE_CHECKING:
    from tts_search.services.rejection import RejectionPolicy


def _build_rejection_policy(**kwargs):
    """Lazily import rejection policy builder to avoid package import cycles."""
    # 函数体内延迟导入，规避包级别的循环导入
    from tts_search.services.rejection import build_rejection_policy

    # 透传参数构建拒绝策略对象（决定哪些样本可进入 SFT 数据）
    return build_rejection_policy(**kwargs)


def _sort_rows_for_selection(
    df: pd.DataFrame,
    *,
    task_key: str,
    rank_fields: tuple[str, ...],
) -> pd.DataFrame:
    """Sort rows deterministically before per-task selection.

    Args:
        df: Candidate row DataFrame.
        task_key: Column grouping rows by task.
        rank_fields: Descending score/reward fields used for ranking.

    Returns:
        Stable sorted DataFrame.
    """
    # 排序列的第一优先级是任务键，保证按任务分组聚集在一起
    sort_cols = [task_key]
    # 任务键升序；后续排名字段为降序（值越大越好）
    ascending = [True]
    # 依次加入排名字段（如 reward、score），按降序排
    for field in rank_fields:
        # 只对真实存在的列排序，避免 KeyError
        if field in df.columns:
            sort_cols.append(field)
            ascending.append(False)
    # 再补充确定性次级排序字段（gap 越小越好、token 越短越好，其余升序），保证排序完全可复现
    for field in ("relative_gap", "token_length", "step_index", "id"):
        # 列存在且尚未加入排序键时才追加
        if field in df.columns and field not in sort_cols:
            sort_cols.append(field)
            ascending.append(True)
    # mergesort 是稳定排序：相同键值的行保持原始相对顺序，保证选择结果确定
    return df.sort_values(sort_cols, ascending=ascending, kind="mergesort")


def select_top_per_task(
    rows: list[dict[str, Any]],
    *,
    top_k: int = 4,
    task_key: str = "task_id",
    dedupe_field: str | None = "reward",
    rank_fields: tuple[str, ...] = ("reward", "score"),
    rejection_policy: "RejectionPolicy" | str | None = "accept_scored",
    rejection_score_threshold: float | None = None,
    rejection_reward_threshold: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply rejection policy, dedupe, then keep top-k rows per task.

    Args:
        rows: Candidate SFT row dictionaries.
        top_k: Maximum number of rows retained per task.
        task_key: Field used to group rows by task.
        dedupe_field: Optional field used to drop duplicate per-task values.
        rank_fields: Descending fields used to rank rows before selection.
        rejection_policy: Policy name or object for accepting rows.
        rejection_score_threshold: Optional score threshold for score policies.
        rejection_reward_threshold: Optional reward threshold for reward policies.

    Returns:
        Selected rows and selection statistics.
    """

    # top-k 必须为正数，否则每任务截取逻辑无意义
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    # 拒绝策略：字符串/None 时现场构建策略对象，否则直接使用传入的策略实例
    policy = (
        _build_rejection_policy(
            name=rejection_policy,
            score_threshold=rejection_score_threshold,
            reward_threshold=rejection_reward_threshold,
        )
        if isinstance(rejection_policy, str) or rejection_policy is None
        else rejection_policy
    )
    # 先按策略过滤：只保留被拒绝策略接受的行（即“可计入训练数据”的样本）
    policy_rows = [row for row in rows if policy.accepts_record(row).accepted]
    if not policy_rows:
        # 无任何行通过策略时直接返回空结果与零值统计
        return [], {
            "input_rows": len(rows),
            "policy_rows": 0,
            "selected_rows": 0,
            "selected_tasks": 0,
        }

    # 通过筛选的行装载为 DataFrame，便于分组与去重
    df = pd.DataFrame(policy_rows)
    # 任务分组键缺失则无法按任务选择，直接报错
    if task_key not in df.columns:
        raise KeyError(f"Missing task key column: {task_key}")
    # 先做确定性排序，让每个任务内最优的行排在最前
    sorted_df = _sort_rows_for_selection(
        df,
        task_key=task_key,
        rank_fields=rank_fields,
    )

    if dedupe_field is not None and dedupe_field in sorted_df.columns:
        # 按 (任务键, 去重字段) 去重：同任务内奖励值相同的行只保留排序最靠前的一条
        deduped = sorted_df.drop_duplicates(
            subset=[task_key, dedupe_field],
            keep="first",
        )
    else:
        # 未指定去重字段则跳过去重
        deduped = sorted_df
    # 每个任务内取排序后的前 top_k 行（group_keys=False 保持行索引连续）
    selected = deduped.groupby(task_key, group_keys=False).head(top_k)
    # 重置索引，使输出行的索引从 0 连续编号
    selected = selected.reset_index(drop=True)

    # 汇总选择统计：输入/通过策略/去重/选中的行数与任务数，以及每任务选中条数分布
    stats = {
        "input_rows": len(rows),
        "policy_rows": len(policy_rows),
        "rows_after_dedupe": int(len(deduped)),
        "selected_rows": int(len(selected)),
        "input_tasks": int(df[task_key].nunique()),
        "selected_tasks": int(selected[task_key].nunique()),
        "dedupe_field": dedupe_field,
        "top_k": top_k,
        "per_task_selected_count_distribution": {
            str(k): int(v)
            for k, v in selected.groupby(task_key)
            .size()
            .value_counts()
            .sort_index()
            .items()
        },
    }
    # 以字典列表形式返回选中行及统计信息
    return selected.to_dict("records"), stats


def merge_new_with_old_data(
    *,
    new_train: pd.DataFrame,
    new_manifest: pd.DataFrame,
    old_train: pd.DataFrame,
    old_manifest: pd.DataFrame,
    task_key: str = "task_id",
    drop_new_when_new_le: int | None = 1,
    drop_new_when_old_between: tuple[int, int] | None = (3, 4),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Merge new SFT rows with old rows while handling task overlap.

    Args:
        new_train: New training DataFrame with ``id`` and messages.
        new_manifest: New row manifest containing task ids.
        old_train: Existing training DataFrame.
        old_manifest: Existing manifest containing task ids.
        task_key: Manifest field identifying tasks.
        drop_new_when_new_le: Drop weak new overlap when new count is at most this.
        drop_new_when_old_between: Old-count interval for dropping weak new overlap.

    Returns:
        Merged training DataFrame, merged manifest, and merge statistics.
    """

    # 深拷贝四个输入表，避免在调用方的原始数据上做就地修改
    new_manifest = new_manifest.copy()
    old_manifest = old_manifest.copy()
    new_train = new_train.copy()
    old_train = old_train.copy()
    # 记录合并前的原始行数，供最终统计对照
    original_new_rows = len(new_train)
    original_old_rows = len(old_train)
    # 统一把任务键转成字符串，避免新旧数据中 int/str 类型不一致导致对不上
    new_manifest[task_key] = new_manifest[task_key].astype(str)
    old_manifest[task_key] = old_manifest[task_key].astype(str)

    # 分别统计新旧数据中每个任务的样本条数
    new_counts = new_manifest.groupby(task_key).size()
    old_counts = old_manifest.groupby(task_key).size()
    # 求交集：新旧数据同时覆盖到的重叠任务集合
    overlap_task_ids = set(new_counts.index) & set(old_counts.index)

    # 需要整体丢弃的新任务集合（弱覆盖任务让位给旧数据）
    drop_new_task_ids: set[str] = set()
    # 两个丢弃规则参数都配置时才启用该启发式
    if drop_new_when_new_le is not None and drop_new_when_old_between is not None:
        # 解包旧数据行数区间 [low, high]
        low, high = drop_new_when_old_between
        # 重叠任务中：新数据覆盖极少（<= 阈值）且旧数据已有 3~4 条时，丢弃这条弱的新任务
        drop_new_task_ids = {
            task_id
            for task_id in overlap_task_ids
            if new_counts[task_id] <= drop_new_when_new_le
            and low <= old_counts[task_id] <= high
        }

    if drop_new_task_ids:
        # 从新清单中剔除被丢弃的任务
        new_manifest = new_manifest.loc[~new_manifest[task_key].isin(drop_new_task_ids)]
        # 训练表只保留仍出现在筛选后清单中的行（按 id 对齐）
        new_train = new_train.loc[new_train["id"].isin(set(new_manifest["id"]))]

    # 保留到最终的新任务集合
    final_new_task_ids = set(new_manifest[task_key].astype(str))
    # 旧数据中凡是新数据已覆盖的任务都要剔除，避免同任务重复
    old_keep_mask = ~old_manifest[task_key].isin(final_new_task_ids)
    # 旧清单按掩码保留
    old_manifest_kept = old_manifest.loc[old_keep_mask].copy()
    # 旧训练表按保留清单的 id 对齐过滤
    old_train_kept = old_train.loc[old_train["id"].isin(set(old_manifest_kept["id"]))]

    # 纵向拼接新旧行：ignore_index 重排索引
    merged_train = pd.concat(
        [
            new_train.assign(dataset_source="new"),
            old_train_kept.assign(dataset_source="old"),
        ],
        ignore_index=True,
    )
    # 清单同样拼接，sort=False 保持原有列顺序
    merged_manifest = pd.concat(
        [
            new_manifest.assign(dataset_source="new"),
            old_manifest_kept.assign(dataset_source="old"),
        ],
        ignore_index=True,
        sort=False,
    )

    # 汇总合并各环节的行数/任务数统计
    stats = {
        "new_input_rows": int(original_new_rows),
        "new_input_tasks": int(new_counts.size),
        "old_input_rows": int(original_old_rows),
        "old_input_tasks": int(old_counts.size),
        "overlap_tasks_before_rule": int(len(overlap_task_ids)),
        "new_tasks_dropped_by_overlap_rule": int(len(drop_new_task_ids)),
        "new_rows_after_overlap_rule": int(len(new_train)),
        "old_rows_removed_by_new_task_id": int(original_old_rows - len(old_train_kept)),
        "old_rows_remaining": int(len(old_train_kept)),
        "merged_rows": int(len(merged_train)),
        "merged_tasks": int(merged_manifest[task_key].nunique()),
    }
    # 返回合并后的训练表、清单表与统计信息
    return merged_train, merged_manifest, stats
