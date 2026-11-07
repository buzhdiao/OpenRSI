"""Collect evaluated rollouts and rebuild SFT messages."""

from __future__ import annotations

# 导入正则模块，用于从反馈文本中提取状态与结果
import re
# 导入 Counter，用于统计各状态/丢弃原因的次数
from collections import Counter
# 导入 Path，用于文件路径操作
from pathlib import Path
# 导入类型注解工具；TYPE_CHECKING 仅在静态检查时导入类型
from typing import TYPE_CHECKING, Any

# 导入 pandas，用于读取源提示词 parquet
import pandas as pd

# 导入通用工具：安全浮点转换、JSONL 读取、文本读取、任务目录名清洗
from tts_search.data_produce.common import (
    maybe_float,
    read_jsonl,
    read_text,
    safe_task_output_name,
)
# 导入 Pass@k 草稿系统提示词构造函数（用于重建 SFT 的 system 消息）
from tts_search.prompt_builder import build_pass_k_draft_system_prompt

if TYPE_CHECKING:
    # 仅类型检查时导入拒绝策略类型，避免运行时循环导入
    from tts_search.services.rejection import RejectionPolicy

# 匹配反馈文本中 "**Status**: xxx" 行的正则（多行模式）
STATUS_RE = re.compile(r"^\*\*Status\*\*:\s*(.+?)\s*$", re.MULTILINE)
# 匹配反馈文本中 "**Result**: xxx" 行的正则（多行模式）
RESULT_RE = re.compile(r"^\*\*Result\*\*:\s*(.+?)\s*$", re.MULTILINE)


def extract_feedback_status(feedback_text: str) -> str:
    """Extract sandbox status from formatted feedback.

    Args:
        feedback_text: Contents of a feedback.txt file.

    Returns:
        Lowercase status string, or ``unknown`` when absent.
    """
    # 在反馈文本中查找状态行
    match = STATUS_RE.search(feedback_text)
    # 命中则返回小写状态串，未命中返回 unknown
    return match.group(1).strip().lower() if match else "unknown"


def extract_feedback_result(feedback_text: str) -> str:
    """Extract sandbox result label from formatted feedback.

    Args:
        feedback_text: Contents of a feedback.txt file.

    Returns:
        Lowercase result string, or ``unknown`` when absent.
    """
    # 在反馈文本中查找结果行
    match = RESULT_RE.search(feedback_text)
    # 命中则返回小写结果串，未命中返回 unknown
    return match.group(1).strip().lower() if match else "unknown"


def feedback_is_success(
    feedback_text: str,
    *,
    status_values: set[str] | None = None,
    result_values: set[str] | None = None,
) -> bool:
    """Check whether feedback represents a successful run.

    Args:
        feedback_text: Contents of a feedback.txt file.
        status_values: Accepted status labels.
        result_values: Accepted result labels.

    Returns:
        True when both status and result match accepted values.
    """
    # 未提供时使用默认的通过标签集合
    status_values = status_values or {"completed", "success"}
    result_values = result_values or {"success"}
    # 状态与结果都落在通过集合内才视为成功运行
    return (
        extract_feedback_status(feedback_text) in status_values
        and extract_feedback_result(feedback_text) in result_values
    )


def _build_rejection_policy(name: str | None):
    """Lazily import rejection policy builder to avoid package import cycles."""
    # 延迟到函数内再导入，避免模块级循环导入
    from tts_search.services.rejection import build_rejection_policy

    # 按名称构建拒绝策略对象
    return build_rejection_policy(name=name)


def build_assistant_content(reasoning: str, code: str, response: str) -> str:
    """Build the assistant message for an SFT row.

    Args:
        reasoning: Optional model reasoning text.
        code: Extracted Python code.
        response: Raw model response fallback.

    Returns:
        Assistant content with reasoning and/or a Python code block.
    """
    # 同时有推理与代码：用 <think> 标签包住推理，代码放入 python 代码块
    if reasoning.strip():
        if code.strip():
            return (
                f"<think>\n{reasoning.strip()}\n</think>\n\n"
                f"```python\n{code.rstrip()}\n```"
            )
        # 只有推理：think 块后接原始响应文本
        return f"<think>\n{reasoning.strip()}\n</think>\n\n{response.strip()}"
    # 无推理但有代码：仅输出 python 代码块
    if code.strip():
        return f"```python\n{code.rstrip()}\n```"
    # 两者都缺：退化为原始响应文本
    return response.strip()


def load_prompt_source(
    source_parquet: Path,
    *,
    metadata_col: str = "metadata",
    prompt_col: str = "prompt",
) -> tuple[
    dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
    dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
]:
    """Load source prompts indexed by task id and task name.

    Args:
        source_parquet: Parquet file containing metadata and prompt columns.
        metadata_col: Column containing task metadata dictionaries.
        prompt_col: Column containing chat prompt messages.

    Returns:
        Two maps: task_id -> (metadata, prompt) and task_name -> (metadata, prompt).
    """
    # 只读取元数据与提示词两列，减小内存占用
    df = pd.read_parquet(source_parquet, columns=[metadata_col, prompt_col])
    # 按任务 ID 索引的提示词映射
    by_task_id: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    # 按任务名索引的提示词映射
    by_task_name: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    # 逐行解析并登记
    for _, row in df.iterrows():
        # 元数据列转为普通字典
        metadata = dict(row[metadata_col])
        # 提示词列可能是 numpy 数组，优先用 tolist 转为 Python 列表
        prompt_value = row[prompt_col]
        prompt = (
            prompt_value.tolist()
            if hasattr(prompt_value, "tolist")
            else list(prompt_value)
        )
        # 取任务 UUID 与任务名作为两种键
        task_id = str(metadata.get("uuid", ""))
        task_name = str(metadata.get("task_name", ""))
        # 有 ID 时登记到 ID 索引
        if task_id:
            by_task_id[task_id] = (metadata, prompt)
        # 有名称时登记到名称索引
        if task_name:
            by_task_name[task_name] = (metadata, prompt)
    # 返回两份索引映射
    return by_task_id, by_task_name


def _step_key(row: dict[str, Any]) -> tuple[str, int]:
    """Build the join key for gen/eval step rows.

    Args:
        row: Serialized generation or evaluation result row.

    Returns:
        Tuple of task id/name and integer step index.
    """
    # 优先用任务 ID 作为键的第一部分
    task_id = row.get("task_id")
    if task_id:
        # 键 = (任务ID, 步序号)
        return str(task_id), int(row.get("step_index", 0))
    # 缺少 ID 时退回任务名
    return str(row.get("task_name", "")), int(row.get("step_index", 0))


def _rank_eval_row(row: dict[str, Any]) -> tuple[int, float, str]:
    """Rank duplicate eval rows for deterministic collapse.

    Args:
        row: Evaluation result row.

    Returns:
        Sort key preferring successful, higher-scoring, earlier rows.
    """
    # 状态优先级：成功最优，其余按可诊断性排序
    status_order = {
        "success": 0,
        "scoring_failed": 1,
        "submission_missing": 2,
        "unknown": 3,
        "timeout": 4,
        "code_execution_error": 5,
    }
    # 取状态字符串，缺失按 unknown 处理
    status = str(row.get("status") or "unknown")
    # 分数安全转换
    score = maybe_float(row.get("score"))
    # 分数取负参与升序排序：分数越高排序键越小（无分数视为极差）
    score_rank = -(score if score is not None else -1e18)
    # 返回排序键：状态优先级 -> 分数（降序）-> 时间戳（早者优先，保证确定性）
    return (status_order.get(status, 99), score_rank, str(row.get("timestamp", "")))


def load_eval_results(
    path: Path,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    """Load and deduplicate eval_results.jsonl.

    Args:
        path: Path to eval_results.jsonl.

    Returns:
        Mapping from step key to selected eval row, plus load statistics.
    """
    # 读取全部评估结果行
    rows = read_jsonl(path)
    # 按步骤键去重后的结果映射
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    # 重复行计数
    duplicate_rows = 0
    # 各状态的行数统计
    status_counts: Counter[str] = Counter()
    # 逐行规范化并去重
    for raw in rows:
        row = dict(raw)
        # 分数与奖励统一安全转为浮点数
        row["score"] = maybe_float(row.get("score"))
        row["reward"] = maybe_float(row.get("reward"))
        # 步序号统一为整数
        row["step_index"] = int(row.get("step_index", 0))
        # 计算步骤键（任务 + 步序）
        key = _step_key(row)
        # 累加状态计数
        status_counts[str(row.get("status") or "unknown")] += 1
        if key in by_key:
            # 键重复时按排序键择优保留（成功、高分、更早优先）
            duplicate_rows += 1
            if _rank_eval_row(row) < _rank_eval_row(by_key[key]):
                by_key[key] = row
        else:
            # 首次出现的键直接登记
            by_key[key] = row
    # 返回去重结果与装载统计
    return by_key, {
        "raw_rows": len(rows),
        "unique_rows": len(by_key),
        "duplicate_rows_collapsed": duplicate_rows,
        "status_counts": dict(status_counts),
    }


def load_gen_results(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Load latest generation row for each task step.

    Args:
        path: Path to gen_results.jsonl.

    Returns:
        Mapping from step key to generation row.
    """
    # 按步骤键存放最新生成行的映射
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    # 逐行读取生成结果
    for raw in read_jsonl(path):
        row = dict(raw)
        # 步序号统一为整数
        row["step_index"] = int(row.get("step_index", 0))
        # 计算步骤键
        key = _step_key(row)
        # 取出同键的已有行
        previous = by_key.get(key)
        if previous is None or str(row.get("timestamp", "")) >= str(
            previous.get("timestamp", "")
        ):
            # 时间戳更新（或首次出现）时保留当前行
            by_key[key] = row
    # 返回每个任务步骤最新的生成行
    return by_key


def load_gen_results_by_request_id(path: Path) -> dict[str, dict[str, Any]]:
    """Load generation rows by request id, preserving same-step rollouts.

    Args:
        path: Path to gen_results.jsonl.

    Returns:
        Mapping from request_id to the latest generation row with that id.
    """
    # 按请求 ID 存放最新生成行的映射（保留同一步骤的多次 rollout）
    by_request_id: dict[str, dict[str, Any]] = {}
    # 逐行读取生成结果
    for raw in read_jsonl(path):
        row = dict(raw)
        # 步序号统一为整数
        row["step_index"] = int(row.get("step_index", 0))
        # 取请求 ID，缺失则跳过
        request_id = str(row.get("request_id") or "")
        if not request_id:
            continue
        # 取出同请求 ID 的已有行
        previous = by_request_id.get(request_id)
        if previous is None or str(row.get("timestamp", "")) >= str(
            previous.get("timestamp", "")
        ):
            # 时间戳更新（或首次出现）时保留当前行
            by_request_id[request_id] = row
    # 返回请求 ID 到最新生成行的映射
    return by_request_id


def find_step_dir(
    run_dir: Path,
    *,
    task_name: str,
    task_id: str | None,
    step_index: int,
) -> Path:
    """Find the artifact directory for one generated step.

    Args:
        run_dir: Root evaluation run directory.
        task_name: Task name from the result row.
        task_id: Optional task UUID from the result row.
        step_index: Step index to locate.

    Returns:
        Existing step directory when found, otherwise the expected preferred path.
    """
    # 任务工件根目录
    tasks_dir = run_dir / "tasks"
    # 候选任务目录：优先"任务名_任务ID"精确匹配，其次仅任务名
    candidates = []
    if task_id:
        candidates.append(tasks_dir / safe_task_output_name(task_name, task_id))
    candidates.append(tasks_dir / safe_task_output_name(task_name))
    # 依次检查候选目录下是否存在目标步骤子目录
    for task_dir in candidates:
        step_dir = task_dir / f"step_{step_index}"
        if step_dir.exists():
            # 命中即返回实际存在的步骤目录
            return step_dir
    # 都不存在时返回首选路径（供调用方按预期位置处理）
    return candidates[0] / f"step_{step_index}"


def step_dir_from_eval_row(
    eval_row: dict[str, Any],
    run_dir: Path,
    *,
    task_name: str,
    task_id: str,
    step_index: int,
) -> Path:
    """Resolve the artifact directory for an eval row.

    Args:
        eval_row: Serialized eval_results row.
        run_dir: Current run directory.
        task_name: Task display name.
        task_id: Task UUID.
        step_index: Rollout step index.

    Returns:
        The directory containing feedback/code artifacts, preferring the
        explicit feedback_path in eval_results when present.
    """
    # 优先使用评估结果中记录的反馈文件路径
    feedback_path = eval_row.get("feedback_path")
    if feedback_path:
        path = Path(str(feedback_path))
        if not path.is_absolute():
            # 相对路径基于当前工作目录解析
            path = Path.cwd() / path
        # 反馈文件所在目录即步骤工件目录
        return path.parent
    # 无记录时按目录命名规则推断步骤目录
    return find_step_dir(
        run_dir,
        task_name=task_name,
        task_id=task_id or None,
        step_index=step_index,
    )


def _source_prompt_for_row(
    *,
    task_id: str,
    task_name: str,
    by_task_id: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
    by_task_name: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
) -> tuple[str, str] | None:
    """Return source system/user prompt text for a result row.

    Args:
        task_id: Task UUID.
        task_name: Human-readable task name.
        by_task_id: Prompt source map keyed by task UUID.
        by_task_name: Prompt source map keyed by task name.

    Returns:
        ``(system_content, user_content)`` or None when source prompt is unavailable.
    """
    # 先按任务 ID 查找，失败再按任务名查找
    source = by_task_id.get(task_id) or by_task_name.get(task_name)
    if source is None:
        # 源提示词不可用
        return None
    # 取出消息列表
    _, prompt = source
    # 消息不足两条（缺 system/user）视为不可用
    if len(prompt) < 2:
        return None
    # 返回 (system 内容, user 内容)
    return str(prompt[0].get("content") or ""), str(prompt[1].get("content") or "")


def collect_sft_rows(
    *,
    run_dir: Path,
    source_parquet: Path,
    rejection_policy: "RejectionPolicy" | str | None = "accept_scored",
    min_reward: float | None = 0.0,
    eval_statuses: set[str] | None = None,
    require_feedback_success: bool = True,
    id_prefix: str = "sft_candidate",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect evaluated rollouts as SFT training rows.

    Args:
        run_dir: Pass@k output directory containing gen/eval results and task artifacts.
        source_parquet: Source prompt parquet used to reconstruct system/user messages.
        rejection_policy: Policy name or object deciding which eval rows are accepted.
        min_reward: Minimum reward threshold; None disables this filter.
        eval_statuses: Allowed sandbox statuses.
        require_feedback_success: Whether feedback.txt must report success.
        id_prefix: Prefix for generated training row ids.

    Returns:
        Tuple of collected SFT rows and summary counters/statistics.
    """

    # 读取全部评估结果行
    eval_rows = read_jsonl(run_dir / "eval_results.jsonl")
    # 记录原始行数与各状态计数
    eval_stats = {
        "raw_rows": len(eval_rows),
        "status_counts": dict(
            Counter(str(row.get("status") or "unknown") for row in eval_rows)
        ),
    }
    # 装载生成结果：按步骤键与按请求 ID 两套索引
    gen_by_key = load_gen_results(run_dir / "gen_results.jsonl")
    gen_by_request_id = load_gen_results_by_request_id(run_dir / "gen_results.jsonl")
    # 装载源提示词索引（任务 ID / 任务名两套键）
    by_task_id, by_task_name = load_prompt_source(source_parquet)
    # 拒绝策略：字符串或 None 时按名称构建，否则直接使用传入对象
    policy = (
        _build_rejection_policy(name=rejection_policy)
        if isinstance(rejection_policy, str) or rejection_policy is None
        else rejection_policy
    )
    # 允许的沙箱状态默认只接受 success
    eval_statuses = eval_statuses or {"success"}

    # 收集到的 SFT 训练行
    rows: list[dict[str, Any]] = []
    # 各类丢弃原因的计数器
    counters: Counter[str] = Counter()
    # 逐条评估行执行过滤与重组
    for eval_row in eval_rows:
        # 计算步骤键
        key = _step_key(eval_row)
        # 过滤 1：沙箱状态不在允许集合内则丢弃
        if str(eval_row.get("status")) not in eval_statuses:
            counters["dropped_eval_status"] += 1
            continue
        # 奖励值缺失按 0 处理
        reward = maybe_float(eval_row.get("reward")) or 0.0
        # 过滤 2：奖励不超过下限（如奖励为 0）则丢弃
        if min_reward is not None and reward <= min_reward:
            counters["dropped_reward"] += 1
            continue
        # 过滤 3：未通过拒绝策略则丢弃
        if not policy.accepts_record(eval_row).accepted:
            counters["dropped_rejection_policy"] += 1
            continue

        # 取请求 ID；优先用其索引定位生成行（同一步骤的多次 rollout 各自对应）
        request_id = str(eval_row.get("request_id") or "")
        gen_row = gen_by_request_id.get(request_id) or gen_by_key.get(key, {})
        # 任务名/任务 ID/步序号：评估行优先，缺失时回退生成行或键
        task_name = str(eval_row.get("task_name") or gen_row.get("task_name") or key[0])
        task_id = str(eval_row.get("task_id") or gen_row.get("task_id") or "")
        step_index = int(eval_row.get("step_index", gen_row.get("step_index", key[1])))
        # 定位该步骤的工件目录（feedback/code 等）
        step_dir = step_dir_from_eval_row(
            eval_row,
            run_dir,
            task_name=task_name,
            task_id=task_id,
            step_index=step_index,
        )
        # 读取反馈文本
        feedback = read_text(step_dir / "feedback.txt")
        # 过滤 4：要求反馈成功而反馈不达标则丢弃
        if require_feedback_success and not feedback_is_success(feedback):
            counters["dropped_feedback_not_success"] += 1
            continue

        # 查找源提示词（system/user 原文）
        source_prompt = _source_prompt_for_row(
            task_id=task_id,
            task_name=task_name,
            by_task_id=by_task_id,
            by_task_name=by_task_name,
        )
        # 源 system 提示词，缺失则置空
        source_system = source_prompt[0] if source_prompt else ""
        # 优先使用步骤目录中留存的 user 提示词
        stored_user = read_text(step_dir / "user_prompt.md")
        if not stored_user and source_prompt:
            # 留存文件缺失时回退到源提示词的 user 部分
            stored_user = source_prompt[1]

        # 用 Pass@k 草稿模板重建 system 消息
        system_prompt = (
            build_pass_k_draft_system_prompt(source_system) if source_system else ""
        )
        # 读取推理、代码与原始响应三类工件
        reasoning = read_text(step_dir / "reasoning.md")
        code = read_text(step_dir / "valid_code.py")
        response = read_text(step_dir / "response.md")
        # 组装 assistant 消息（推理 + 代码块）
        assistant = build_assistant_content(reasoning, code, response)

        # 组装并登记 SFT 训练行：三段消息（system/user/assistant）+ 全量溯源元数据
        # （id 顺序生成；token/奖励/耗时取自评估与生成行；*_path 记录工件位置便于追溯）
        rows.append(
            {
                "id": f"{id_prefix}-{len(rows)}",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": stored_user},
                    {"role": "assistant", "content": assistant},
                ],
                "task_id": task_id,
                "task_name": task_name,
                "request_id": eval_row.get("request_id") or gen_row.get("request_id"),
                "step_index": step_index,
                "step_dir": str(step_dir),
                "source_status": eval_row.get("status"),
                "feedback_status": extract_feedback_status(feedback),
                "feedback_result": extract_feedback_result(feedback),
                "score": maybe_float(eval_row.get("score")),
                "reward": maybe_float(eval_row.get("reward")),
                "submit_medal": eval_row.get("submit_medal"),
                "queue_time": maybe_float(eval_row.get("queue_time")),
                "run_time": maybe_float(eval_row.get("run_time")),
                "api_prompt_tokens": maybe_float(gen_row.get("prompt_tokens")),
                "api_completion_tokens": maybe_float(gen_row.get("completion_tokens")),
                "api_total_tokens": maybe_float(gen_row.get("total_tokens")),
                "feedback_path": str(step_dir / "feedback.txt"),
                "code_path": str(step_dir / "valid_code.py"),
                "reasoning_path": str(step_dir / "reasoning.md"),
                "response_path": str(step_dir / "response.md"),
                "prompt_path": str(step_dir / "user_prompt.md"),
            }
        )

    # 汇总统计：评估装载统计 + 收集规模（行数/任务数）+ 各类丢弃计数
    stats = {
        **eval_stats,
        "collected_rows": len(rows),
        "collected_tasks": len({row["task_id"] or row["task_name"] for row in rows}),
        "drop_counts": dict(counters),
    }
    # 返回 SFT 行集合与统计信息
    return rows, stats
