"""Writers for SFT alignment and training datasets."""

from __future__ import annotations

# 导入 json，用于写出采样报告
import json
# 导入 Path，用于构造输出文件路径
from pathlib import Path
# 导入类型注解工具
from typing import Any

# 导入 pandas，用于构建训练数据框并写出 parquet/CSV
import pandas as pd

# 导入通用工具：JSON 安全化转换与 JSONL 写出
from tts_search.data_produce.common import json_safe, write_jsonl


def to_training_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Return the minimal training format expected by SFT jobs.

    Args:
        rows: SFT rows containing ``id`` and ``messages`` fields.

    Returns:
        DataFrame with one row per training conversation.
    """

    # 只保留 SFT 训练必需的两列：样本 id 与对话消息列表
    return pd.DataFrame(
        [{"id": row["id"], "messages": row["messages"]} for row in rows]
    )


def write_alignment_dataset(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    name: str,
    write_manifest: bool = True,
) -> dict[str, str]:
    """Write JSONL, parquet, and optional manifest for SFT rows.

    Args:
        rows: Selected SFT rows to write.
        output_dir: Directory for dataset files.
        name: Base filename for written artifacts.
        write_manifest: Whether to write metadata CSV excluding messages.

    Returns:
        Mapping of artifact type to written path.
    """

    # 创建输出目录（含父目录，已存在则跳过）
    output_dir.mkdir(parents=True, exist_ok=True)
    # 把选中行转换为最小训练格式
    train_df = to_training_frame(rows)
    # 拼出 parquet 与 jsonl 两个产物的输出路径
    parquet_path = output_dir / f"{name}.parquet"
    jsonl_path = output_dir / f"{name}.jsonl"
    # 写出 parquet 格式（列式存储，训练加载高效）
    train_df.to_parquet(parquet_path, index=False)
    # 同步写出 jsonl 格式（逐行 JSON，便于流式读取）
    write_jsonl(jsonl_path, train_df.to_dict("records"))

    # 记录已写出的产物路径映射
    paths = {
        "parquet": str(parquet_path),
        "jsonl": str(jsonl_path),
    }
    # 可选写出元数据清单（剔除 messages 大字段，只留描述信息）
    if write_manifest:
        manifest_rows = [
            {key: value for key, value in row.items() if key != "messages"}
            for row in rows
        ]
        # 拼出清单文件路径并写出 CSV
        manifest_path = output_dir / f"{name}_manifest.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
        # 把清单路径也加入产物映射
        paths["manifest"] = str(manifest_path)
    # 返回产物类型到路径的映射
    return paths


def sample_tasks(
    rows: list[dict[str, Any]],
    *,
    fractions: tuple[float, ...] = (0.5, 0.25),
    seed: int = 20260502,
    task_key: str = "task_id",
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Sample task ids and keep all rows for selected tasks.

    Args:
        rows: Candidate SFT rows.
        fractions: Fractions of unique tasks to sample.
        seed: Random seed for deterministic task sampling.
        task_key: Row field identifying the task.

    Returns:
        Mapping from sample label to sampled rows, plus sampling summary.
    """

    # 构建仅含任务标识与样本 id 的轻量数据框
    df = pd.DataFrame([{task_key: row[task_key], "id": row["id"]} for row in rows])
    # 取全部唯一任务 ID：转字符串、去重、排序、重置索引，保证顺序确定
    task_ids = (
        df[task_key]
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    # 以固定随机种子整体打乱任务 ID，实现可复现的随机抽样
    shuffled = task_ids.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # 各采样比例对应的行集合
    samples: dict[str, list[dict[str, Any]]] = {}
    # 采样摘要：记录种子、采样单元、全集规模与各比例结果
    summary: dict[str, Any] = {
        "seed": seed,
        "sampling_unit": task_key,
        "full": {
            "tasks": int(task_ids.size),
            "rows": int(len(rows)),
        },
        "samples": {},
    }
    # 逐个比例执行抽样：以任务为整体（同一任务的行要么全选要么全不选）
    for fraction in fractions:
        # 按比例计算要抽的任务数量（向下取整）
        count = int(task_ids.size * fraction)
        # 从打乱后的列表中取前 count 个任务作为选中集合
        selected_tasks = set(shuffled.iloc[:count].astype(str))
        # 生成形如 "50pct" 的样本标签
        label = f"{int(fraction * 100)}pct"
        # 保留所有属于选中任务的行
        selected_rows = [
            row for row in rows if str(row.get(task_key)) in selected_tasks
        ]
        # 记录该比例的行集合
        samples[label] = selected_rows
        # 在摘要中登记该比例的任务数与行数
        summary["samples"][label] = {
            "fraction": fraction,
            "tasks": len(selected_tasks),
            "rows": len(selected_rows),
        }
    # 返回样本集合与采样摘要
    return samples, summary


def write_sampling_report(path: Path, summary: dict[str, Any]) -> None:
    """Write a JSON sampling summary.

    Args:
        path: Destination report path.
        summary: Sampling summary dictionary.

    Returns:
        None. Creates parent directories and overwrites the report.
    """
    # 先创建报告所在目录
    path.parent.mkdir(parents=True, exist_ok=True)
    # 把摘要做 JSON 安全化后以 UTF-8、缩进 2 格写出（覆盖旧报告）
    path.write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
