#!/usr/bin/env python3
"""Deduplicate SFT messages and apply a full-message token-length gate."""

# 启用延迟注解求值
from __future__ import annotations

# argparse：子命令式参数解析（dedup / token-filter）
import argparse
# hashlib：计算消息归一化 JSON 的 SHA1 摘要用于去重
import hashlib
# json：读写 JSONL 与归一化序列化
import json
from pathlib import Path
from typing import Any


def normalize_for_json(value: Any) -> Any:
    # 递归归一化：dict 键统一转字符串
    if isinstance(value, dict):
        return {str(key): normalize_for_json(item) for key, item in value.items()}
    # list/tuple 统一转 list 并递归处理
    if isinstance(value, (list, tuple)):
        return [normalize_for_json(item) for item in value]
    # numpy 等对象经 tolist 转为原生类型，保证可序列化
    if hasattr(value, "tolist"):
        return normalize_for_json(value.tolist())
    # 标量原样返回
    return value


def normalized_message_text(messages: Any) -> str:
    # 把 messages 归一化后做“键排序 + 紧凑分隔符”的确定性序列化，
    # 使内容相同但键序/空白不同的消息得到同一规范文本
    return json.dumps(
        normalize_for_json(messages),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def message_hash(messages: Any) -> str:
    # 计算规范文本的 SHA1 指纹，作为整条消息的去重键
    return hashlib.sha1(normalized_message_text(messages).encode("utf-8")).hexdigest()


def read_rows(path: Path) -> tuple[list[dict[str, Any]], Any | None]:
    # 按扩展名分流：parquet 走 pyarrow，其余按 JSONL
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        # 整表读入并转为行列表；返回源表供后续按行号 take
        table = pq.read_table(path)
        return table.to_pylist(), table
    # JSONL：逐行解析
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            # 跳过空白行
            text = line.strip()
            if not text:
                continue
            # JSON 解析失败时报出文件与行号
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            # 每行必须是 JSON 对象
            if not isinstance(row, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            rows.append(row)
    # 返回行列表；JSONL 无源表对象
    return rows, None


def write_rows(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    source_table: Any | None,
    indices: list[int],
) -> None:
    # 确保输出目录存在
    path.parent.mkdir(parents=True, exist_ok=True)
    # 输出格式跟随扩展名
    if path.suffix.lower() == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        # 有源表时按保留行号 take，保住原 schema
        if source_table is not None:
            output_table = source_table.take(pa.array(indices, type=pa.int64()))
        else:
            # 无源表时从行列表建表
            output_table = pa.Table.from_pylist(rows)
        # 写出 parquet（单行组）
        pq.write_table(output_table, path, row_group_size=max(1, len(rows)))
        return
    # JSONL 输出：逐行写出（先归一化保证可序列化）
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(normalize_for_json(row), ensure_ascii=False) + "\n")


def deduplicate(rows: list[dict[str, Any]]) -> tuple[list[int], list[dict[str, Any]]]:
    # 指纹 -> 首次出现的行号
    seen: dict[str, int] = {}
    # 保留行的下标列表
    keep_indices: list[int] = []
    # 被剔除的重复行审计记录
    dropped: list[dict[str, Any]] = []
    # 按输入顺序逐行扫描，同指纹只保留首次出现
    for index, row in enumerate(rows):
        # 缺 messages 字段属于格式错误
        if "messages" not in row:
            raise ValueError(f"row {index} has no messages field")
        # 计算该行消息指纹
        digest = message_hash(row["messages"])
        # 已见过：记为重复并跳过
        if digest in seen:
            dropped.append(
                {
                    "row_index": index,
                    "id": row.get("id"),
                    "first_row_index": seen[digest],
                    "message_hash": digest,
                    "drop_reason": "duplicate_normalized_messages",
                }
            )
            continue
        # 首次出现：登记指纹与行号并保留
        seen[digest] = index
        keep_indices.append(index)
    # 返回（保留行号, 重复行清单）
    return keep_indices, dropped


def full_message_tokens(tokenizer: Any, messages: Any) -> tuple[int, bool]:
    # 先归一化消息结构
    normalized = normalize_for_json(messages)
    try:
        # 首选：用 tokenizer 的聊天模板切分并计数（与训练时序列化方式一致）
        token_ids = tokenizer.apply_chat_template(
            normalized,
            tokenize=True,
            add_generation_prompt=False,
        )
        # 返回 token 数与“未走回退路径”标志
        return len(token_ids), False
    except Exception:
        # 模板不可用等异常时回退：直接对规范 JSON 文本编码计数
        text = normalized_message_text(normalized)
        # 返回回退路径的 token 数与“已回退”标志
        return len(tokenizer.encode(text, add_special_tokens=False)), True


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    # 确保目录存在
    path.parent.mkdir(parents=True, exist_ok=True)
    # 逐行写出审计/清单文件
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(normalize_for_json(row), ensure_ascii=False) + "\n")


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    # 两个子命令共用的输入/输出参数
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    # 可选 summary 路径，缺省为 <output>.summary.json
    parser.add_argument(
        "--summary",
        type=Path,
        help="Summary JSON path. Defaults to <output>.summary.json.",
    )


def parse_args() -> argparse.Namespace:
    # 以模块 docstring 作为帮助说明
    parser = argparse.ArgumentParser(description=__doc__)
    # 声明必选子命令
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 子命令 dedup：按归一化消息去重
    dedup_parser = subparsers.add_parser(
        "dedup", help="Keep the first occurrence of each normalized messages value."
    )
    # 复用共享参数
    add_shared_arguments(dedup_parser)
    # 可选的重复行清单路径
    dedup_parser.add_argument(
        "--dropped",
        type=Path,
        help="Dropped-row JSONL path. Defaults to <output>.dropped.jsonl.",
    )

    # 子命令 token-filter：按整条消息 token 长度过滤
    token_parser = subparsers.add_parser(
        "token-filter", help="Filter rows by full chat-template token length."
    )
    # 复用共享参数
    add_shared_arguments(token_parser)
    # 分词器模型路径与 token 上限
    token_parser.add_argument("--model-path", required=True)
    token_parser.add_argument("--max-tokens", type=int, default=32768)
    # 可选的剔除行清单路径
    token_parser.add_argument(
        "--dropped",
        type=Path,
        help="Dropped-row JSONL path. Defaults to <output>.dropped.jsonl.",
    )
    # 是否允许执行模型仓库自定义代码
    token_parser.add_argument("--trust-remote-code", action="store_true")
    # 解析并返回参数
    return parser.parse_args()


def main() -> None:
    # 解析命令行参数
    args = parse_args()
    # 读入全部行与（可能的）源表
    rows, source_table = read_rows(args.input)
    # 剔除行清单与 summary 的缺省路径规则：<output> 后缀拼接
    dropped_path = args.dropped or Path(str(args.output) + ".dropped.jsonl")
    summary_path = args.summary or Path(str(args.output) + ".summary.json")

    # 分支一：精确消息去重
    if args.command == "dedup":
        # 执行去重，得到保留行号与重复行清单
        keep_indices, dropped = deduplicate(rows)
        # 组装 summary 统计
        summary = {
            "operation": "exact_message_deduplication",
            "input_rows": len(rows),
            "kept_rows": len(keep_indices),
            "dropped_rows": len(dropped),
            "policy": (
                "SHA1 of normalized messages JSON; keep first occurrence in input order"
            ),
        }
    else:
        # 分支二：整条消息 token 长度过滤（按需加载 transformers 分词器）
        from transformers import AutoTokenizer

        # 加载分词器；trust_remote_code 透传
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=args.trust_remote_code,
        )
        # 保留行号、剔除清单与回退计数初始化
        keep_indices = []
        dropped = []
        fallback_count = 0
        # 逐行计算整条消息 token 数并做门槛过滤
        for index, row in enumerate(rows):
            # 缺 messages 字段属于格式错误
            if "messages" not in row:
                raise ValueError(f"row {index} has no messages field")
            # 计算 token 数（含是否走了回退路径）
            token_count, fallback = full_message_tokens(tokenizer, row["messages"])
            # 累计回退行数
            fallback_count += int(fallback)
            # 未超上限：保留
            if token_count <= args.max_tokens:
                keep_indices.append(index)
            else:
                # 超上限：记入剔除清单并注明原因
                dropped.append(
                    {
                        "row_index": index,
                        "id": row.get("id"),
                        "full_messages_tokens": token_count,
                        "drop_reason": "full_messages_token_limit",
                    }
                )
        # 组装 token 过滤分支的 summary 统计
        summary = {
            "operation": "full_message_token_filter",
            "input_rows": len(rows),
            "kept_rows": len(keep_indices),
            "dropped_rows": len(dropped),
            "max_tokens": args.max_tokens,
            "model_path": args.model_path,
            "tokenizer_fallback_rows": fallback_count,
        }

    # 按保留行号取出最终保留的行
    kept_rows = [rows[index] for index in keep_indices]
    # 写出过滤/去重后的数据（parquet 或 jsonl 跟随输入）
    write_rows(
        args.output,
        kept_rows,
        source_table=source_table,
        indices=keep_indices,
    )
    # 写出剔除行清单
    write_jsonl(dropped_path, dropped)
    # 补充 summary 中的输入输出路径信息
    summary.update(
        {
            "input": str(args.input),
            "output": str(args.output),
            "dropped_manifest": str(dropped_path),
        }
    )
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
