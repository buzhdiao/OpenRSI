#!/usr/bin/env python3
"""Exclude training rows whose task description matches a reserved task."""

# 启用延迟注解求值，兼容新式类型标注语法
from __future__ import annotations

# argparse：命令行参数解析
import argparse
# hashlib：计算任务描述的 SHA256 摘要用于精确匹配
import hashlib
# json：逐行读写 JSONL
import json
# Counter 统计提取原因分布，defaultdict 构建哈希倒排索引
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def normalize_value(value: Any) -> Any:
    # 递归归一化：dict 的键统一转为字符串（兼容 parquet 读出的非字符串键）
    if isinstance(value, dict):
        return {str(key): normalize_value(item) for key, item in value.items()}
    # list/tuple 统一转为 list 并递归处理元素
    if isinstance(value, (list, tuple)):
        return [normalize_value(item) for item in value]
    # numpy 等带 tolist 的对象转成原生 Python 类型，保证可 JSON 序列化
    if hasattr(value, "tolist"):
        return normalize_value(value.tolist())
    # 标量原样返回
    return value


def canonicalize_task_description(text: str) -> str:
    # 空值兜底为空串；统一把 \r\n / \r 换行归一为 \n
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    # 去掉每行行尾空白（最小归一化，不做其他改动）
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    # 去掉整体首尾空白后返回规范化文本
    return normalized.strip()


def sha256_text(text: str) -> str:
    # 计算文本的 UTF-8 SHA256 十六进制摘要，作为任务描述的唯一指纹
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_user_message(messages: Any) -> str:
    # 先归一化消息结构
    normalized = normalize_value(messages)
    # 非列表结构视为无效，返回空串
    if not isinstance(normalized, list):
        return ""
    # 线性查找第一条 role=user 的消息并返回其文本内容
    for item in normalized:
        if isinstance(item, dict) and item.get("role") == "user":
            return str(item.get("content") or "")
    # 没有用户消息时返回空串
    return ""


def extract_task_description_from_user(user_text: str) -> tuple[str, str]:
    # 任务描述以固定标记 "Task Description:" 开始
    marker = "Task Description:"
    # 找到标记位置
    marker_index = user_text.find(marker)
    # 找不到标记：返回空描述与缺失原因（调用方决定是否容错）
    if marker_index < 0:
        return "", "missing_task_description_marker"
    # 截取标记之后的正文
    task_body = user_text[marker_index + len(marker) :]
    # 任务描述可能的结束边界：数据说明 / 数据预览 / 输出说明等追加段
    end_patterns = [
        "\n\nData Description:\n\n```",
        "\nData Description:\n\n```",
        "\n\nData Preview:\n```",
        "\nData Preview:\n```",
        "\n\nOutput Instructions:",
        "\nOutput Instructions:",
    ]
    # 计算各结束模式在正文中的位置（只保留找得到的）
    end_positions = [
        task_body.find(pattern)
        for pattern in end_patterns
        if task_body.find(pattern) >= 0
    ]
    # 存在结束边界：截取最近的边界之前的纯任务描述
    if end_positions:
        return task_body[: min(end_positions)].strip(), "ok_cut_before_appended_context"
    # 没有任何结束边界：把标记后的全部文本都当作任务描述
    return task_body.strip(), "ok_no_cut_marker"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    # 逐行读取 JSONL 文件
    with path.open("r", encoding="utf-8") as handle:
        # enumerate 从 1 计行号，便于报错定位
        for line_number, line in enumerate(handle, 1):
            # 跳过空白行
            text = line.strip()
            if not text:
                continue
            # 解析 JSON，失败时带文件与行号抛错
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            # 每行必须是 JSON 对象
            if not isinstance(row, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            # 以生成器逐行产出，避免整文件驻留内存
            yield row


def read_rows(path: Path) -> tuple[list[dict[str, Any]], Any | None]:
    # 按扩展名分流：parquet 走 pyarrow，其余按 JSONL 处理
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        # 整表读入并转为 Python 行列表；同时返回原表对象供后续按行号取回
        table = pq.read_table(path)
        return table.to_pylist(), table
    # JSONL：读成行列表，无源表对象
    return list(read_jsonl(path)), None


def reserved_metadata(row: dict[str, Any]) -> dict[str, Any]:
    # 优先从行的 metadata 字段取保留集元数据
    metadata = normalize_value(row.get("metadata"))
    # 是 dict 就直接用
    if isinstance(metadata, dict):
        return metadata
    # 否则行本身就是元数据（平铺格式）
    return row


def build_reserved_hash_index(
    prompt_rows: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    # 描述哈希 -> 该描述对应的元数据记录列表（同描述可对应多个 uuid）
    index: dict[str, list[dict[str, str]]] = defaultdict(list)
    # 按原顺序保留的元数据记录，用于与 manifest 校验对齐
    ordered_metadata: list[dict[str, str]] = []
    # 逐行处理保留集提示
    for position, row in enumerate(prompt_rows):
        # 取出该行元数据
        metadata = reserved_metadata(row)
        # 规范化任务描述
        description = canonicalize_task_description(
            str(metadata.get("task_description") or "")
        )
        # 保留集行缺少任务描述属于数据错误，直接报错
        if not description:
            raise ValueError(f"reserved prompt row {position} has no task_description")
        # 记录 uuid 与 task_name 作为审计字段
        record = {
            "uuid": str(metadata.get("uuid") or ""),
            "task_name": str(metadata.get("task_name") or ""),
        }
        # 追加到顺序表
        ordered_metadata.append(record)
        # 以描述哈希为键建倒排索引
        index[sha256_text(description)].append(record)
    # 返回（哈希索引, 顺序元数据）
    return index, ordered_metadata


def validate_manifest_order(
    manifest_path: Path,
    ordered_metadata: list[dict[str, str]],
) -> None:
    # 延迟导入 csv（仅校验时需要）
    import csv

    # 读取 CSV 清单为字典行列表
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # 行数必须与提示文件一致，否则顺序校验无意义
    if len(rows) != len(ordered_metadata):
        raise ValueError(
            "reserved manifest and prompt file have different row counts: "
            f"{len(rows)} != {len(ordered_metadata)}"
        )
    # 逐行校验 uuid 与 task_name 是否一一对应
    for position, (manifest, metadata) in enumerate(zip(rows, ordered_metadata)):
        for key in ("uuid", "task_name"):
            # manifest 里给了该列且值不一致时才报错（缺列则跳过）
            if key in manifest and str(manifest.get(key) or "") != metadata[key]:
                raise ValueError(
                    f"reserved manifest {key} mismatch at row {position}: "
                    f"{manifest.get(key)!r} != {metadata[key]!r}"
                )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    # 确保输出目录存在
    path.parent.mkdir(parents=True, exist_ok=True)
    # 逐行写出，先做值归一化保证可序列化；ensure_ascii=False 保留中文
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(normalize_value(row), ensure_ascii=False) + "\n")


def write_kept_data(
    input_path: Path,
    output_path: Path,
    kept_rows: list[dict[str, Any]],
    kept_indices: list[int],
    source_table: Any | None,
) -> None:
    # 输出格式跟随输入扩展名
    if output_path.suffix.lower() == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        # 有源表时按保留行号 take，完整保留原 schema 与列
        if source_table is not None:
            table = source_table.take(pa.array(kept_indices, type=pa.int64()))
        else:
            # 无源表（理论少见）则从行列表建表
            table = pa.Table.from_pylist(kept_rows)
        # 写出 parquet；row_group_size 设为行数（单行组）
        pq.write_table(table, output_path, row_group_size=max(1, len(kept_rows)))
        return
    # JSONL 输出
    write_jsonl(output_path, kept_rows)


def parse_args() -> argparse.Namespace:
    # 以模块 docstring 作为帮助说明
    parser = argparse.ArgumentParser(description=__doc__)
    # 训练数据文件（JSONL 或 Parquet）
    parser.add_argument("--input", type=Path, required=True, help="Training JSONL or Parquet.")
    # 保留任务提示文件（用于筛掉重叠样本）
    parser.add_argument(
        "--reserved-prompts",
        type=Path,
        required=True,
        help="Reserved prompt JSONL or Parquet.",
    )
    # 可选 CSV 清单，用于校验保留集 uuid/task_name 顺序
    parser.add_argument(
        "--reserved-manifest",
        type=Path,
        help="Optional CSV used to validate UUID and task-name order.",
    )
    # 输出目录
    parser.add_argument("--output-dir", type=Path, required=True)
    # 允许写入非空目录
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    # 容忍缺少 Task Description 标记的行（默认直接报错，因为无法判定是否重叠）
    parser.add_argument(
        "--allow-missing-task-description",
        action="store_true",
        help=(
            "Keep rows without a Task Description marker. By default they are "
            "rejected because overlap cannot be determined safely."
        ),
    )
    # 解析并返回参数
    return parser.parse_args()


def main() -> None:
    # 解析命令行参数
    args = parse_args()
    # 输出目录非空且未指定 --overwrite 时拒绝执行，防止误覆盖
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(
            f"output directory is not empty: {args.output_dir}; use --overwrite to proceed"
        )
    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 读取保留集提示并构建“任务描述哈希 -> 元数据”索引
    prompt_rows, _ = read_rows(args.reserved_prompts)
    reserved_hashes, ordered_metadata = build_reserved_hash_index(prompt_rows)
    # 提供了 CSV 清单时先校验顺序一致性
    if args.reserved_manifest:
        validate_manifest_order(args.reserved_manifest, ordered_metadata)

    # 读取训练数据
    training_rows, source_table = read_rows(args.input)
    # 保留的行与行号
    kept_rows: list[dict[str, Any]] = []
    kept_indices: list[int] = []
    # 保留/剔除两类审计清单
    kept_manifest: list[dict[str, Any]] = []
    removed_manifest: list[dict[str, Any]] = []
    # 各提取原因计数
    extraction_reasons: Counter[str] = Counter()
    # 命中过的保留描述哈希集合
    matched_hashes: set[str] = set()

    # 逐行检查训练样本是否与保留任务重叠
    for index, row in enumerate(training_rows):
        # 缺 messages 字段属于格式错误，直接报错
        if "messages" not in row:
            raise ValueError(f"training row {index} has no messages field")
        # 从首条 user 消息中提取任务描述正文与提取原因
        raw_description, reason = extract_task_description_from_user(
            get_user_message(row["messages"])
        )
        # 缺标记且未开启容错：无法安全判定重叠，直接报错
        if (
            reason == "missing_task_description_marker"
            and not args.allow_missing_task_description
        ):
            raise ValueError(
                "cannot safely check reserved-task overlap because a training row "
                f"has no Task Description marker: row_index={index}, id={row.get('id')!r}"
            )
        # 规范化任务描述并计算指纹（空描述的指纹为空串）
        description = canonicalize_task_description(raw_description)
        digest = sha256_text(description) if description else ""
        # 查哈希索引得到匹配的保留记录
        matches = reserved_hashes.get(digest, [])
        # 累计提取原因
        extraction_reasons[reason] += 1
        # 组装审计记录
        audit = {
            "row_index": index,
            "id": row.get("id"),
            "task_description_sha256": digest,
            "task_description_chars": len(description),
            "extract_reason": reason,
        }
        # 命中保留集：剔除该行
        if matches:
            # 记录已命中的哈希
            matched_hashes.add(digest)
            # 写入剔除清单，注明剔除原因
            removed_manifest.append(
                {
                    **audit,
                    "remove_reason": "task_description_sha256_exact_match",
                }
            )
            # 跳过保留逻辑
            continue
        # 未命中：保留该行
        kept_indices.append(index)
        kept_rows.append(row)
        kept_manifest.append(audit)

    # 输出文件扩展名跟随输入（parquet 或 jsonl）
    output_suffix = ".parquet" if args.input.suffix.lower() == ".parquet" else ".jsonl"
    # 输出数据文件路径
    output_data = args.output_dir / f"train{output_suffix}"
    # 写出过滤后的训练数据
    write_kept_data(
        args.input,
        output_data,
        kept_rows,
        kept_indices,
        source_table,
    )
    # 写出被剔除行的审计清单
    write_jsonl(args.output_dir / "removed_overlap.jsonl", removed_manifest)
    # 写出保留行的审计清单
    write_jsonl(args.output_dir / "kept_manifest.jsonl", kept_manifest)
    # 汇总统计信息
    summary = {
        "input": str(args.input),
        "reserved_prompts": str(args.reserved_prompts),
        "reserved_manifest": (
            str(args.reserved_manifest) if args.reserved_manifest else None
        ),
        "output": str(output_data),
        "input_rows": len(training_rows),
        "kept_rows": len(kept_rows),
        "removed_rows": len(removed_manifest),
        "reserved_rows": len(prompt_rows),
        "reserved_unique_description_hashes": len(reserved_hashes),
        "matched_reserved_unique_description_hashes": len(matched_hashes),
        "extraction_reason_counts": dict(extraction_reasons),
        "rule": (
            "exact SHA256 match after minimal newline and trailing-space normalization; "
            "UUID and task name are audit fields only"
        ),
    }
    # 落盘 summary.json（缩进、键排序）
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # 同时把 summary 打印到标准输出
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


# 作为脚本直接运行时进入主流程
if __name__ == "__main__":
    main()
