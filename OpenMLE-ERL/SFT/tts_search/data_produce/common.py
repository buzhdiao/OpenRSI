"""Common helpers for data production modules."""

from __future__ import annotations

# 导入 json，用于 JSONL 读写
import json
# 导入 math，用于判断 NaN/Inf
import math
# 导入 Path，用于文件路径操作
from pathlib import Path
# 导入类型注解工具
from typing import Any, Iterable

# 导入 numpy，用于汇总统计计算
import numpy as np


# 把任意标量安全转为有限浮点数；转换失败或为 NaN/Inf 时返回 None
def maybe_float(value: Any) -> float | None:
    """Convert a value to a finite float.

    Args:
        value: Raw scalar value from JSON/CSV/pandas.

    Returns:
        Finite float, or None when conversion fails or the value is NaN/Inf.
    """
    # 空值直接返回 None
    if value is None:
        return None
    try:
        # 尝试转为浮点数
        number = float(value)
    except (TypeError, ValueError):
        # 不可转换（如字符串、对象）时返回 None
        return None
    # NaN 或无穷视为无效值
    if math.isnan(number) or math.isinf(number):
        return None
    # 返回有效的有限浮点数
    return number


def json_safe(value: Any) -> Any:
    """Convert nested values to JSON-serializable Python objects.

    Args:
        value: Arbitrary nested value that may contain numpy scalars or arrays.

    Returns:
        JSON-safe value with non-finite floats converted to None.
    """
    # 字典：键统一转字符串，值递归处理
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    # 列表/元组：逐元素递归处理
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    # numpy 数组：先转 Python 列表再递归
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    # numpy 整数：转为 Python int
    if isinstance(value, np.integer):
        return int(value)
    # numpy 浮点：转为 Python float
    if isinstance(value, np.floating):
        value = float(value)
    # 浮点数：NaN/Inf 转为 None（JSON 不支持非有限数）
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    # 其余标量类型原样返回
    return value


def read_text(path: Path) -> str:
    """Read text from disk if present.

    Args:
        path: File path to read.

    Returns:
        File contents, or an empty string when the file is missing.
    """
    # 文件不存在时返回空串，调用方无需处理异常
    if not path.exists():
        return ""
    # 以 UTF-8 读取文本，非法字节忽略
    return path.read_text(encoding="utf-8", errors="ignore")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON records.

    Args:
        path: JSONL file path.

    Returns:
        List of decoded JSON object rows.
    """
    # 初始化行列表
    rows: list[dict[str, Any]] = []
    # 逐行读取 JSONL 文件
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            # 跳过空行，解析每行的 JSON 对象
            if line.strip():
                rows.append(json.loads(line))
    # 返回全部解码后的行
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write rows as newline-delimited JSON.

    Args:
        path: Destination JSONL path.
        rows: Iterable of dictionary records.

    Returns:
        None. Creates parent directories and overwrites the target file.
    """
    # 先创建输出目录
    path.parent.mkdir(parents=True, exist_ok=True)
    # 以覆盖模式逐行写出 JSONL
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            # 每行做 JSON 安全化后写出（保留非 ASCII 字符）
            f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def safe_task_output_name(task_name: str, task_id: str | None = None) -> str:
    """Build a filesystem-safe task directory name.

    Args:
        task_name: Human-readable task name.
        task_id: Optional task UUID to make duplicate names unique.

    Returns:
        Sanitized task directory name.
    """
    # 任务名中的路径分隔符替换为下划线，避免产生非法目录名
    safe_task_name = str(task_name).replace("/", "_").replace("\\", "_")
    # 未提供任务 ID 时只返回清洗后的任务名
    if task_id is None:
        return safe_task_name
    # 任务 ID 同样清洗，用于区分重名任务
    safe_task_id = str(task_id).replace("/", "_").replace("\\", "_")
    # 拼接"任务名_任务ID"作为唯一目录名
    return f"{safe_task_name}_{safe_task_id}"


def summary_stats(values: Iterable[float]) -> dict[str, float | int | None]:
    """Summarize numeric values for reports.

    Args:
        values: Iterable of raw numeric values.

    Returns:
        Count, mean, std, quantiles, min, and max; empty stats when no values are finite.
    """
    # 先清洗输入：只保留可转为有限浮点数的值
    clean_values = []
    for value in values:
        number = maybe_float(value)
        if number is not None:
            clean_values.append(number)
    # 转为浮点数组便于统计计算
    clean = np.array(clean_values, dtype=float)
    # 无有效数据时返回全空的统计结果
    if clean.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    # 计算数量、均值、标准差、最值与 5/25/50/75/95 分位数
    return {
        "count": int(clean.size),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
        "min": float(clean.min()),
        "p05": float(np.quantile(clean, 0.05)),
        "p25": float(np.quantile(clean, 0.25)),
        "p50": float(np.quantile(clean, 0.50)),
        "p75": float(np.quantile(clean, 0.75)),
        "p95": float(np.quantile(clean, 0.95)),
        "max": float(clean.max()),
    }
