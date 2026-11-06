#!/usr/bin/env python3
"""Recover and select useful steps from evolutionary rollout trajectories."""

# 启用延迟注解求值
from __future__ import annotations

# argparse：命令行参数解析
import argparse
# ast：literal_eval 兜底解析形如 Python 字面量的 step 列表字符串
import ast
# json：读取 stat.json / 注释文件、写出 JSONL
import json
# math：判断分数是否为有限数值
import math
# Counter 统计段类型分布，defaultdict 建父子关系，deque 做 BFS 队列
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


# “四算子”中的三个可作段根的算子（Evo-Max 进化搜索）：draft 起草 / improve 改进 / crossover 交叉
ROOT_OPERATORS = {"draft", "improve", "crossover"}
# 第四个算子：debug 调试——它不单独成段，只作为根算子段内的附加步骤
DEBUG_OPERATOR = "debug"
# 奖牌等级 -> 排名分：bronze=1 / silver=2 / gold=3，无奖牌（含各种空值写法）为 0
MEDAL_RANK = {
    "": 0,
    "none": 0,
    "null": 0,
    "nan": 0,
    "n/a": 0,
    "na": 0,
    "bronze": 1,
    "silver": 2,
    "gold": 3,
}


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
            # 生成器逐行产出，节省内存
            yield row


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    # 确保输出目录存在
    path.parent.mkdir(parents=True, exist_ok=True)
    # 计数写出行数
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            # 键排序写出，保证输出稳定可 diff
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    # 返回写出总行数
    return count


def normalize_operator(value: Any) -> str:
    # 空值兜底、去空白、转小写、连字符转下划线，统一算子名写法
    text = str(value or "").strip().lower().replace("-", "_")
    # 常见别名归一：closeover / cross_over 都视为 crossover
    return {"closeover": "crossover", "cross_over": "crossover"}.get(text, text)


def numeric_field(obj: dict[str, Any] | None, key: str) -> float | None:
    # 对象为空（None 或空 dict）时视为字段缺失
    if not obj:
        return None
    # 取字段值
    value = obj.get(key)
    # 缺失返回 None
    if value is None:
        return None
    # 尝试转 float，失败视为缺失
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    # NaN/Inf 等非有限值同样视为缺失
    return number if math.isfinite(number) else None


def strict_score(obj: dict[str, Any] | None) -> float | None:
    # 严格取 score 字段的有限数值（execution-grounded 的沙箱执行得分）
    return numeric_field(obj, "score")


def medal_rank(value: Any) -> int:
    # 查奖牌等级表；未知写法一律按无奖牌（0）处理
    return MEDAL_RANK.get(str(value or "").strip().lower(), 0)


def load_task_stat(
    stat_path: Path,
) -> tuple[dict[int, dict[str, Any]], dict[int, list[int]]]:
    # 读取任务轨迹统计文件（容错替换非法编码字符）
    stat = json.loads(stat_path.read_text(encoding="utf-8", errors="replace"))
    # steps 数组：每步一个 dict
    steps = stat.get("steps") or []
    # 建立“步号 -> 步记录”索引（跳过格式非法的记录）
    steps_by_index = {
        int(step["step"]): step
        for step in steps
        if isinstance(step, dict) and "step" in step
    }
    # 建立“父步号 -> 子步号列表”的树结构
    children: dict[int, list[int]] = defaultdict(list)
    for step in steps:
        # 跳过非法记录
        if not isinstance(step, dict) or "step" not in step:
            continue
        # 当前步作为其父步的孩子
        child = int(step["step"])
        for parent in step.get("parent_steps") or []:
            try:
                # 父步号非法时忽略该边
                children[int(parent)].append(child)
            except (TypeError, ValueError):
                continue
    # 返回（步索引表, 父子关系表）
    return steps_by_index, children


def load_higher_is_better(task_dir: Path) -> bool:
    # 读取任务目录下 aira_evo/dojo_config.json 的指标方向配置
    config_path = task_dir / "aira_evo" / "dojo_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8", errors="replace"))
    # 取 task.higher_is_better 布尔值
    value = config.get("task", {}).get("higher_is_better")
    # 缺失或类型不对则报错（方向决定分数门的比较方式，不能猜）
    if not isinstance(value, bool):
        raise RuntimeError(f"missing boolean higher_is_better in {config_path}")
    # 返回指标方向
    return value


def debug_descendants(
    root_step: int,
    children: dict[int, list[int]],
    steps_by_index: dict[int, dict[str, Any]],
) -> list[int]:
    # 收集段根之后连续 debug 子树中的所有 debug 步
    result: list[int] = []
    # BFS 队列初始化为根的直接孩子
    queue: deque[int] = deque(children.get(root_step, []))
    while queue:
        # 取出队首步号
        step_index = queue.popleft()
        # 查不到记录则跳过
        step = steps_by_index.get(step_index)
        if not step:
            continue
        # 非 debug 步：剪枝，不进入该子树
        if normalize_operator(step.get("operator") or step.get("mode")) != DEBUG_OPERATOR:
            continue
        # debug 步：收集进结果
        result.append(step_index)
        # 继续向下扩展其孩子
        queue.extend(children.get(step_index, []))
    # 去重并按步号升序返回
    return sorted(set(result))


def parent_scores_from_root(
    root: dict[str, Any],
    steps_by_index: dict[int, dict[str, Any]],
) -> tuple[list[int], list[float], list[int]]:
    # 段根父步号列表
    parent_indices: list[int] = []
    # 能取到有效分数的父步分数列表
    parent_scores: list[float] = []
    # 取不到分数的父步号列表（用于缺失分数时的宽容策略）
    missing_scores: list[int] = []
    # 遍历段根记录的 parent_steps
    for raw_index in root.get("parent_steps") or []:
        try:
            # 步号非法则跳过
            parent_index = int(raw_index)
        except (TypeError, ValueError):
            continue
        # 登记父步号
        parent_indices.append(parent_index)
        # 严格读取父步分数
        score = strict_score(steps_by_index.get(parent_index))
        if score is None:
            # 分数缺失：记入缺失列表
            missing_scores.append(parent_index)
        else:
            # 分数有效：记入分数列表
            parent_scores.append(score)
    # 返回（父步号, 父步分数, 缺失分数的父步号）
    return parent_indices, parent_scores, missing_scores


def score_better_than_parents(
    *,
    endpoint_score: float | None,
    parent_scores: list[float],
    higher_is_better: bool,
    allow_missing_parent_by_positive_endpoint: bool,
) -> bool:
    # 端点无分数：不通过
    if endpoint_score is None:
        return False
    # 没有任何父步分数（如 draft 根或父分数全缺失）：
    if not parent_scores:
        # 仅当允许宽容且端点分数为正时通过
        return bool(allow_missing_parent_by_positive_endpoint and endpoint_score > 0)
    # 有父分数：按指标方向取父步的“最好”分数作为基线
    baseline = max(parent_scores) if higher_is_better else min(parent_scores)
    # 端点分数需严格优于基线才通过
    return endpoint_score > baseline if higher_is_better else endpoint_score < baseline


def class_flags(
    *,
    segment_type: str,
    endpoint_score: float | None,
    endpoint_medal_rank: int,
    parent_scores: list[float],
    missing_parent_score_steps: list[int],
    higher_is_better: bool,
) -> tuple[bool, bool, str]:
    # 按段类型判定分数门与奖牌门，返回（是否过奖牌门, 是否过分数门, 判定原因）
    if segment_type == "draft":
        # draft 段没有父步可比较，只看端点分数是否为正
        if endpoint_score is None:
            return False, False, "draft_endpoint_score_missing"
        # 分数门：端点分数 > 0
        score_gate = endpoint_score > 0
        # 记录判定原因
        reason = "draft_score_gt0" if score_gate else "draft_score_not_gt0"
    elif segment_type in {"improve", "crossover"}:
        # improve / crossover 段需要端点分数优于父步最好分数
        if endpoint_score is None:
            return False, False, f"{segment_type}_endpoint_score_missing"
        # 调用统一的“优于父步”判定；父分数有缺失时允许以正端点分数宽容通过
        score_gate = score_better_than_parents(
            endpoint_score=endpoint_score,
            parent_scores=parent_scores,
            higher_is_better=higher_is_better,
            allow_missing_parent_by_positive_endpoint=bool(missing_parent_score_steps),
        )
        # 记录判定原因
        reason = (
            f"{segment_type}_endpoint_score_better_than_parent"
            if score_gate
            else f"{segment_type}_endpoint_score_not_better_than_parent"
        )
    else:
        # 其他类型（如 debug）不支持作为段类型
        return False, False, "unsupported_segment_type"
    # 奖牌门：分数门通过且端点拿到过奖牌（奖牌排名 > 0）
    medal_gate = score_gate and endpoint_medal_rank > 0
    # 返回双门判定结果与原因
    return medal_gate, score_gate, reason


def build_segments(stat_root: Path) -> list[dict[str, Any]]:
    # 汇总所有任务的段记录
    records: list[dict[str, Any]] = []
    # 遍历 <stat_root>/<task>/stat.json
    for stat_path in sorted(stat_root.glob("*/stat.json")):
        # 任务目录与任务名
        task_dir = stat_path.parent
        task_name = task_dir.name
        try:
            # 读取指标方向
            higher_is_better = load_higher_is_better(task_dir)
            # 方向信息可得
            missing_metric_direction = False
        except Exception:
            # 读不到方向信息时兜底为“越大越好”，并打上缺失标记
            higher_is_better = True
            missing_metric_direction = True

        # 载入该任务的步索引与父子关系
        steps_by_index, children = load_task_stat(stat_path)
        # 按步号升序遍历所有步，尝试以其为段根
        for root_index, root in sorted(steps_by_index.items()):
            # 归一化该步的算子名（兼容 operator / mode 字段）
            segment_type = normalize_operator(root.get("operator") or root.get("mode"))
            # 只有四算子中的三个根算子才能开段（debug 只能挂在段内）
            if segment_type not in ROOT_OPERATORS:
                continue
            # 段步骤 = 根步 + 其连续 debug 后代
            segment_steps = [root_index] + debug_descendants(
                root_index, children, steps_by_index
            )
            # 端点取段内最大步号（该段最终结果步）
            endpoint_index = max(segment_steps)
            # 查端点步记录，缺失则放弃该段
            endpoint = steps_by_index.get(endpoint_index)
            if endpoint is None:
                continue
            # 端点严格分数（execution-grounded 沙箱得分）
            endpoint_score = strict_score(endpoint)
            # 端点奖牌（兼容 submit_medal / medal 字段）
            endpoint_medal = endpoint.get("submit_medal") or endpoint.get("medal")
            # 收集段根父步的分数信息
            parent_indices, parent_scores, missing_parent_scores = (
                parent_scores_from_root(root, steps_by_index)
            )
            # 判定分数门与奖牌门
            medal_gate, score_gate, reason = class_flags(
                segment_type=segment_type,
                endpoint_score=endpoint_score,
                endpoint_medal_rank=medal_rank(endpoint_medal),
                parent_scores=parent_scores,
                missing_parent_score_steps=missing_parent_scores,
                higher_is_better=higher_is_better,
            )
            # 指标方向缺失时，improve/crossover 无法可靠比较，双门一律不通过
            if missing_metric_direction and segment_type in {"improve", "crossover"}:
                score_gate = False
                medal_gate = False
                reason = "missing_metric_direction"
            # 落一条段记录（含段定位、分数、门判定与原因等完整审计字段）
            records.append(
                {
                    "segment_id": f"aira::{task_name}::root_{root_index}",
                    "task_name": task_name,
                    "segment_type": segment_type,
                    "segment_root_step": root_index,
                    "segment_endpoint_step": endpoint_index,
                    "segment_step_indices": segment_steps,
                    "segment_length": len(segment_steps),
                    "higher_is_better": higher_is_better,
                    "parent_step_indices": parent_indices,
                    "parent_scores": parent_scores,
                    "missing_parent_score_steps": missing_parent_scores,
                    "endpoint_score": endpoint_score,
                    "endpoint_submit_medal": endpoint_medal,
                    "endpoint_medal_rank": medal_rank(endpoint_medal),
                    "passes_score_gate": score_gate,
                    "passes_medal_gate": medal_gate,
                    "score_gate_reason": reason,
                    "stat_path": str(stat_path.relative_to(stat_root)),
                    "endpoint_step_dir": str(
                        (task_dir / f"step_{endpoint_index}").relative_to(stat_root)
                    ),
                }
            )
    # 返回全部段记录
    return records


def parse_index_list(value: Any) -> list[int]:
    # 已经是列表：逐项转 int
    if isinstance(value, list):
        return [int(item) for item in value]
    # 非字符串（None 等）返回空列表
    if not isinstance(value, str):
        return []
    try:
        # 先按 JSON 解析（如 "[1, 2]"）
        parsed = json.loads(value)
    except json.JSONDecodeError:
        # 失败再按 Python 字面量解析（如 "{'a': 1}" 风格的单引号列表）
        parsed = ast.literal_eval(value)
    # 解析结果是列表才转 int，否则视为无效
    return [int(item) for item in parsed] if isinstance(parsed, list) else []


def load_annotations(path: Path | None) -> dict[str, list[int]]:
    # 未提供注释文件：返回空映射（调用方按缺失注释处理）
    if path is None:
        return {}
    # segment_id -> 需保留的步号列表
    result: dict[str, list[int]] = {}
    # 逐行读取注释 JSONL
    for row in read_jsonl(path):
        # 取 metadata 字段（可能缺失）
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        # 段 id 兼容三种字段：segment_id / metadata.segment_id / custom_id
        segment_id = (
            row.get("segment_id")
            or metadata.get("segment_id")
            or row.get("custom_id")
        )
        # annotation 字段存在时摊平进行内，统一读取
        annotation = row.get("annotation")
        if isinstance(annotation, dict):
            row = {**row, **annotation}
        # 首选 keep_indices 字段
        indices = parse_index_list(row.get("keep_indices"))
        # 没有 keep_indices 时回退：从 steps 列表里挑 keep_for_sft 为真的步
        if not indices and isinstance(row.get("steps"), list):
            indices = [
                int(step["step_index"])
                for step in row["steps"]
                if isinstance(step, dict)
                and step.get("keep_for_sft") is True
                and isinstance(step.get("step_index"), int)
            ]
        # 段 id 是字符串才登记
        if isinstance(segment_id, str):
            result[segment_id] = indices
    # 返回段 id 到步号列表的映射
    return result


def select_steps(
    segments: list[dict[str, Any]],
    annotations: dict[str, list[int]],
    stat_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # 最终选中的步记录
    selected: list[dict[str, Any]] = []
    # 已见过的步目录（相对路径）集合，用于跨段去重
    seen_step_dirs: set[str] = set()
    # 多步段缺注释的计数
    missing_annotations = 0
    # 逐段筛选
    for segment in segments:
        # 只保留通过奖牌门的段（分数门是奖牌门的必要条件）
        if not segment["passes_medal_gate"]:
            continue
        # 段内合法步号集合
        valid_steps = set(int(index) for index in segment["segment_step_indices"])
        if segment["segment_length"] == 1:
            # 单步段：无需注释，直接保留该步
            keep_steps = list(valid_steps)
            source = "single_step_medal_segment"
        else:
            # 多步段：必须有因果继承（causal inheritance）人工注释
            if segment["segment_id"] not in annotations:
                # 缺注释计数并跳过
                missing_annotations += 1
                continue
            # 按注释取保留步，且过滤到段内合法步号
            keep_steps = [
                index
                for index in annotations[segment["segment_id"]]
                if index in valid_steps
            ]
            source = "causal_inheritance_annotation"
        # 由 stat 相对路径还原任务目录
        task_dir = (stat_root / str(segment["stat_path"])).parent
        # 逐个落选中的步
        for step_index in keep_steps:
            # 步目录与其相对路径
            step_dir = task_dir / f"step_{step_index}"
            relative_step_dir = str(step_dir.relative_to(stat_root))
            # 同一步目录已被其他段选中：去重跳过
            if relative_step_dir in seen_step_dirs:
                continue
            # 登记已见目录
            seen_step_dirs.add(relative_step_dir)
            # 追加选中步记录（含任务、步号、目录、段信息与选取来源）
            selected.append(
                {
                    "id": f"aira::{segment['task_name']}::step_{step_index}",
                    "task_name": segment["task_name"],
                    "step_index": step_index,
                    "step_dir": relative_step_dir,
                    "segment_id": segment["segment_id"],
                    "segment_type": segment["segment_type"],
                    "segment_length": segment["segment_length"],
                    "selection_source": source,
                }
            )
    # 按任务名与步号排序，输出顺序稳定
    selected.sort(key=lambda row: (row["task_name"], row["step_index"]))
    # 返回（选中步列表, 选取统计摘要）
    return selected, {
        "selected_steps": len(selected),
        "unique_step_dirs": len(seen_step_dirs),
        "missing_multi_step_annotations": missing_annotations,
    }


def parse_args() -> argparse.Namespace:
    # 以模块 docstring 作为帮助说明
    parser = argparse.ArgumentParser(description=__doc__)
    # 轨迹统计根目录
    parser.add_argument(
        "--stat-root",
        type=Path,
        required=True,
        help="Directory containing <task>/stat.json and <task>/aira_evo/dojo_config.json.",
    )
    # 可选的因果继承注释 JSONL（多步段必需）
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Optional causal-inheritance annotation JSONL for multi-step segments.",
    )
    # 输出目录
    parser.add_argument("--output-dir", type=Path, required=True)
    # 严格模式：被选中的多步段缺注释时直接失败
    parser.add_argument(
        "--require-complete-annotations",
        action="store_true",
        help="Fail when a selected multi-step segment has no annotation.",
    )
    # 解析并返回参数
    return parser.parse_args()


def main() -> None:
    # 解析命令行参数
    args = parse_args()
    # 统计根目录必须存在
    if not args.stat_root.exists():
        raise FileNotFoundError(f"stat root does not exist: {args.stat_root}")
    # 第一步：扫描所有任务，构建段记录（含分数/奖牌双门判定）
    segments = build_segments(args.stat_root)
    # 第二步：载入因果继承注释
    annotations = load_annotations(args.annotations)
    # 第三步：执行步级选取
    selected_steps, selection_summary = select_steps(
        segments, annotations, args.stat_root
    )
    # 严格模式下，多步段缺注释则整体失败，防止静默丢样本
    if (
        args.require_complete_annotations
        and selection_summary["missing_multi_step_annotations"]
    ):
        raise RuntimeError(
            "selected multi-step segments are missing causal-inheritance annotations: "
            f"{selection_summary['missing_multi_step_annotations']}"
        )

    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # 写出全部段记录
    write_jsonl(args.output_dir / "segments.jsonl", segments)
    # 写出选中的步
    write_jsonl(args.output_dir / "selected_steps.jsonl", selected_steps)
    # 汇总统计：段数量、类型分布、双门通过数与选取摘要
    summary = {
        "input_stat_root": str(args.stat_root),
        "annotations": str(args.annotations) if args.annotations else None,
        "segment_count": len(segments),
        "segment_type_counts": dict(
            Counter(segment["segment_type"] for segment in segments)
        ),
        "score_gate_segments": sum(
            bool(segment["passes_score_gate"]) for segment in segments
        ),
        "medal_gate_segments": sum(
            bool(segment["passes_medal_gate"]) for segment in segments
        ),
        **selection_summary,
    }
    # 落盘 summary.json
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # 同时打印 summary
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


# 作为脚本直接运行时进入主流程
if __name__ == "__main__":
    main()
