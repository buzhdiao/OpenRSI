#!/usr/bin/env python3
"""Select high-scoring candidates from parallel rollout results."""

# 启用延迟注解求值
from __future__ import annotations

# argparse：命令行参数解析
import argparse
# json：读写 JSONL 与 summary
import json
# math：判断分数是否为有限数值
import math
# Counter 统计家族/每任务条数分布，defaultdict 按任务分组
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    # 逐行读取 JSONL
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            # 跳过空白行
            text = line.strip()
            if not text:
                continue
            # 解析失败时报出文件与行号
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            # 每行必须是 JSON 对象
            if not isinstance(row, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            # 生成器逐行产出
            yield row


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    # 确保输出目录存在
    path.parent.mkdir(parents=True, exist_ok=True)
    # 计数写出行数
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            # 键排序写出，保证输出稳定
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    # 返回写出总行数
    return count


def numeric_score(row: dict[str, Any], field: str) -> float:
    # 取候选的分数字段
    value = row.get(field)
    try:
        # 尝试转 float；失败说明该候选没有可用分数
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"candidate has no numeric {field!r}: {row.get('request_id')}") from exc
    # NaN/Inf 同样视为非法分数
    if not math.isfinite(score):
        raise ValueError(f"candidate has a non-finite {field!r}: {row.get('request_id')}")
    # 返回有限分数
    return score


def task_value(row: dict[str, Any], field: str) -> str:
    # 取任务字段并转字符串
    value = str(row.get(field) or "")
    # 任务字段为空会导致无法分组，直接报错
    if not value:
        raise ValueError(f"candidate has no {field!r}: {row.get('request_id')}")
    # 返回任务标识
    return value


def stable_id(row: dict[str, Any], field: str) -> str:
    # 取稳定 ID 字段（用于同分排序的确定性 tie-break）
    value = str(row.get(field) or "")
    # 缺失则报错
    if not value:
        raise ValueError(f"candidate has no {field!r}")
    # 返回 ID
    return value


def generator_family(row: dict[str, Any], field: str) -> str:
    # 先看显式家族字段（小写化处理）
    explicit = str(row.get(field) or "").strip().lower()
    # 只接受 glm / qwen 两个已知家族
    if explicit in {"glm", "qwen"}:
        return explicit
    # 未显式标注时，拼接几个来源字段做关键词推断
    source_text = " ".join(
        str(row.get(key) or "").lower()
        for key in ("model", "model_name", "run_key", "source_name")
    )
    # 文本里含 glm 判为 glm 家族
    if "glm" in source_text:
        return "glm"
    # 文本里含 qwen 判为 qwen 家族
    if "qwen" in source_text:
        return "qwen"
    # 推断不出则归为 unknown（joint 策略会丢弃这类行）
    return "unknown"


def select_glm_top4_unique_score(
    rows: list[dict[str, Any]],
    *,
    task_field: str,
    score_field: str,
    id_field: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deduplicate equal scores per task, then retain the highest four.

    This preserves the historical first-batch order of operations: duplicate
    reward values are removed before the per-task Top-4 is taken.
    """

    # 按任务分组候选
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        # 任务字段缺失/为空会在 task_value 内报错
        grouped[task_value(row, task_field)].append(row)

    # 全局选中的候选列表
    selected: list[dict[str, Any]] = []
    # 因分数重复被剔除的行数
    duplicate_score_rows = 0
    # 逐任务处理（任务名排序保证输出稳定）
    for task in sorted(grouped):
        # 任务内排序：分数降序，同分按稳定 ID 升序
        ordered = sorted(
            grouped[task],
            key=lambda row: (-numeric_score(row, score_field), stable_id(row, id_field)),
        )
        # 去重后的候选（同分只保留排序靠前的第一个）
        unique_scores: list[dict[str, Any]] = []
        # 已见过的分数集合
        seen_scores: set[float] = set()
        for row in ordered:
            # 读该候选分数
            score = numeric_score(row, score_field)
            # 分数重复：剔除并计数
            if score in seen_scores:
                duplicate_score_rows += 1
                continue
            # 首次出现：登记分数并保留
            seen_scores.add(score)
            unique_scores.append(row)
        # 取去重后分数最高的前 4 条，附上名次与策略标记
        for rank, row in enumerate(unique_scores[:4], 1):
            # 拷贝原行避免污染输入
            output = dict(row)
            # 记录任务内名次
            output["selection_rank"] = rank
            # 记录选取策略
            output["selection_policy"] = "per_task_score_dedup_then_top4"
            selected.append(output)

    # 汇总统计：输入行数、任务数、重复剔除数、选取数与每任务条数分布
    summary = {
        "policy": "glm-top4-unique-score",
        "input_rows": len(rows),
        "task_count": len(grouped),
        "duplicate_score_rows_removed": duplicate_score_rows,
        "rows_after_score_dedup": len(rows) - duplicate_score_rows,
        "selected_rows": len(selected),
        "selected_count_per_task": dict(
            Counter(
                str(count)
                for count in Counter(task_value(row, task_field) for row in selected).values()
            )
        ),
    }
    # 返回（选中列表, 统计摘要）
    return selected, summary


def select_joint_top4(
    rows: list[dict[str, Any]],
    *,
    task_field: str,
    score_field: str,
    id_field: str,
    family_field: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the historical mixed-family joint Top-4 policy."""

    # 按任务分组的混合家族候选
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    # 家族无法识别而被丢弃的行数
    unknown_rows = 0
    # 输入行按家族的计数
    input_families: Counter[str] = Counter()
    for original in rows:
        # 拷贝行，避免改写调用方的数据
        row = dict(original)
        # 判定生成器家族
        family = generator_family(row, family_field)
        # 统计输入家族分布
        input_families[family] += 1
        if family == "unknown":
            # 未知家族直接丢弃并计数
            unknown_rows += 1
            continue
        # 把归一化后的家族写回行内
        row[family_field] = family
        # 按任务分组
        grouped[task_value(row, task_field)].append(row)

    # 全局选中列表
    selected: list[dict[str, Any]] = []
    # 选中行按家族计数
    selected_families: Counter[str] = Counter()
    # qwen 因联合名次 > 1 被丢弃的分布
    qwen_dropped_by_rank: Counter[str] = Counter()
    # 逐任务处理
    for task in sorted(grouped):
        # 任务内排序：分数降序，同分按稳定 ID 升序
        ordered = sorted(
            grouped[task],
            key=lambda row: (-numeric_score(row, score_field), stable_id(row, id_field)),
        )
        # 只看联合 Top-4，再按家族规则二次筛选
        for rank, row in enumerate(ordered[:4], 1):
            # 读该行的家族
            family = str(row[family_field])
            # 保留规则：glm 全保留；qwen 只有联合名次第 1 才保留
            keep = family == "glm" or (family == "qwen" and rank == 1)
            if family == "qwen" and rank > 1:
                # 记录 qwen 被丢弃时的联合名次
                qwen_dropped_by_rank[f"rank_{rank}"] += 1
            if not keep:
                # 不满足保留规则：跳过
                continue
            # 拷贝行并附名次、TopK 观测值与策略标记
            output = dict(row)
            output["selection_rank"] = rank
            output["selection_observed_topk"] = 4
            output["selection_policy"] = "joint_glm_qwen_top4_qwen_only_rank1"
            # 加入选中列表
            selected.append(output)
            # 累计选中家族计数
            selected_families[family] += 1

    # 汇总统计：输入家族分布、丢弃数、任务数、选中分布及每任务最大条数
    summary = {
        "policy": "joint-top4",
        "input_rows": len(rows),
        "input_family_counts": dict(input_families),
        "unknown_family_rows_dropped": unknown_rows,
        "task_count": len(grouped),
        "selected_rows": len(selected),
        "selected_family_counts": dict(selected_families),
        "qwen_dropped_by_joint_rank": dict(qwen_dropped_by_rank),
        "maximum_rows_per_task": max(
            Counter(task_value(row, task_field) for row in selected).values(),
            default=0,
        ),
    }
    # 返回（选中列表, 统计摘要）
    return selected, summary


def parse_args() -> argparse.Namespace:
    # 以模块 docstring 作为帮助说明
    parser = argparse.ArgumentParser(description=__doc__)
    # 候选输入文件
    parser.add_argument("--input", type=Path, required=True, help="Candidate JSONL file.")
    # 选中结果输出文件
    parser.add_argument("--output", type=Path, required=True, help="Selected JSONL file.")
    # 必选策略：glm 去重 Top-4 或 glm/qwen 联合 Top-4
    parser.add_argument(
        "--policy",
        choices=("glm-top4-unique-score", "joint-top4"),
        required=True,
        help="Selection policy to apply.",
    )
    # 各字段名可按数据 schema 配置
    parser.add_argument("--task-field", default="task_name")
    parser.add_argument("--score-field", default="score")
    parser.add_argument("--id-field", default="request_id")
    parser.add_argument("--family-field", default="generator_family")
    # 可选 summary 路径，缺省 <output>.summary.json
    parser.add_argument(
        "--summary",
        type=Path,
        help="Summary JSON path. Defaults to <output>.summary.json.",
    )
    # 解析并返回参数
    return parser.parse_args()


def main() -> None:
    # 解析命令行参数
    args = parse_args()
    # 读入全部候选行
    rows = list(read_jsonl(args.input))
    # 按策略分派选取函数
    if args.policy == "glm-top4-unique-score":
        # 策略一：glm 每任务分数去重后取 Top-4
        selected, summary = select_glm_top4_unique_score(
            rows,
            task_field=args.task_field,
            score_field=args.score_field,
            id_field=args.id_field,
        )
    else:
        # 策略二：glm/qwen 联合 Top-4（qwen 只保留名次第 1）
        selected, summary = select_joint_top4(
            rows,
            task_field=args.task_field,
            score_field=args.score_field,
            id_field=args.id_field,
            family_field=args.family_field,
        )

    # 写出选中结果
    write_jsonl(args.output, selected)
    # summary 补充输入输出路径
    summary.update({"input": str(args.input), "output": str(args.output)})
    # summary 缺省路径规则
    summary_path = args.summary or Path(str(args.output) + ".summary.json")
    # 确保 summary 目录存在
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    # 落盘 summary.json
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # 同时打印 summary
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


# 作为脚本直接运行时进入主流程
if __name__ == "__main__":
    main()
