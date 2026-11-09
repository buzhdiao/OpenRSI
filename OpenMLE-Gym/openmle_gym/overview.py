# 中文说明:OpenMLE-Gym 的任务总览(overview)生成器——为每个任务目录产出
# 一行 CSV 元数据(名称/模态/任务类型/原始与最终大小/算力需求/评估指标等)。
# 在系统中的位置:openmle_gym 包内,是"构建(build)后、评估前"的元信息盘点步骤;
# 每个任务都在独立子进程里分析(overview worker),主进程按确定性分块并行汇总,
# 单任务失败只落 #TASK_RUNTIME_ERROR 行,不影响兄弟任务(批次独立性)。
# 关键设计:CSV 固定列序 + BOM 头(Excel 兼容);LLM 相关字段在无密钥时可跳过
# (--skip-llm),或要求环境变量齐全;指标评估支持 in-process 与 isolated 两种模式。

# 延迟注解:类型标注运行期不求值
from __future__ import annotations

# csv:总览表的读写
import csv
# io:内存缓冲的 CSV 写出
import io
# math:分块大小向上取整
import math
# os:目录树遍历(计算大小)
import os
# sys:转发子进程 stderr
import sys
# 线程池:确定性分块并行
from concurrent.futures import ThreadPoolExecutor, as_completed
# Path:路径对象
from pathlib import Path
# 类型注解
from typing import Any, Callable, TypeVar

# tqdm:进度条
from tqdm import tqdm

# 公共工具:原子写/任务目录收集/父目录确保/任务名校验
from .common import (
    atomic_write_text,
    collect_task_dirs,
    ensure_parent,
    validate_task_name,
)
# 指标校验:进程内与隔离子进程两种样例提交评估
from .metric_validation import (
    evaluate_sample_submission,
    evaluate_sample_submission_process,
)
# 评估 LLM 配置(要求密钥齐全)
from .llm_config import eval_llm_config
# 任务子进程运行器(隔离执行 worker)
from .process_runner import run_task_process

# 泛型参数:输入项/结果
T = TypeVar("T")
R = TypeVar("R")


def _directory_size(path: Path) -> int:
    """递归统计目录总字节数(不存在返回 0,单个文件失败跳过)。"""
    total = 0
    # 目录不存在按 0 处理
    if not path.exists():
        return 0
    # 遍历目录树下所有文件累加大小
    for root, _, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            try:
                total += file_path.stat().st_size
            except OSError:
                # 取不到大小(竞态/权限):跳过
                continue
    return total


def _format_size(num_bytes: int) -> str:
    """字节数格式化为人类可读串(保留两位小数,最大到 GB)。"""
    value = float(num_bytes)
    # 逐级换算,不足 1024 即输出当前单位
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} GB"


# 总览 CSV 的固定列序
FIELDNAMES = [
    "Name",
    "Modality",
    "Task",
    "Raw Size",
    "Final Size",
    "CPU/GPU",
    "Source",
    "Metric",
    "NOTE",
    "Server Location",
]


def _require_llm_env() -> None:
    """要求评估 LLM 环境变量齐全,缺失时提示可用 --skip-llm 离线。"""
    try:
        eval_llm_config().require("Evaluation")
    except RuntimeError as exc:
        raise RuntimeError(f"{exc}. Use --skip-llm for offline metadata.") from exc


def _chunked_map(
    items: list[T],
    worker_count: int,
    func: Callable[[T], R],
    desc: str,
) -> list[R]:
    """Run work in deterministic chunks and return results in input order."""
    # 空输入直接返回
    if not items:
        return []
    # 并发数收敛到 [1, len(items)]
    worker_count = max(1, min(worker_count, len(items)))
    if worker_count == 1:
        # 单线程:顺序执行并展示进度
        return [func(item) for item in tqdm(items, desc=desc)]

    # 计算分块大小并切出连续块(块内顺序固定 => 结果确定)
    chunk_size = max(1, math.ceil(len(items) / worker_count))
    indexed_items = list(enumerate(items))
    chunks = [
        indexed_items[index:index + chunk_size]
        for index in range(0, len(indexed_items), chunk_size)
    ]
    # 结果槽位表(按输入顺序回填)
    results: list[R | None] = [None] * len(items)

    # 单块处理:返回 (索引, 结果) 对
    def process_chunk(chunk: list[tuple[int, T]]) -> list[tuple[int, R]]:
        return [(index, func(item)) for index, item in chunk]

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            for index, result in future.result():
                results[index] = result

    # 任何槽位缺失都视为内部错误(不应发生)
    if any(result is None for result in results):
        raise RuntimeError("Internal error: missing chunked metadata result")
    return [result for result in results if result is not None]


def _read_overview_rows(path: Path) -> list[dict[str, Any]]:
    """读取总览 CSV 为行字典列表(容忍 BOM 头)。"""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _write_overview_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """按固定列序原子写出总览 CSV(带 BOM,Excel 打开不乱码)。"""
    # 先在内存里拼好完整 CSV
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDNAMES)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in FIELDNAMES})
    # 原子写(临时文件+替换),前缀 BOM
    atomic_write_text(path, "\ufeff" + output.getvalue())


def _run_metadata_pipeline(
    task_dirs: list[Path],
    output_csv: Path,
    workers: int,
    execution_mode: str = "process",
    task_timeout: float = 600,
) -> None:
    """Update the base CSV with metadata pipeline LLM and metric steps."""
    # 元数据管线必须用到 LLM:先校验环境
    _require_llm_env()
    # 并发数收敛
    worker_count = max(1, min(workers, len(task_dirs)))
    # 读现有 CSV 并做一致性校验
    rows = _read_overview_rows(output_csv)
    row_by_name = {row.get("Name", ""): row for row in rows}
    task_names = [task_dir.name for task_dir in task_dirs]
    if len(row_by_name) != len(rows):
        raise ValueError("Overview CSV contains duplicate task names")
    if set(row_by_name) != set(task_names):
        raise ValueError("Overview CSV task names do not match task directories")
    # 单任务分析体:总在子进程执行(LLM + 指标都在 worker 内完成)
    def analyze_in_process(task_dir: Path) -> dict[str, Any]:
        outcome = run_task_process(
            "overview",
            {
                "task_dir": str(task_dir),
                "skip_llm": False,
                "metric_execution_mode": execution_mode,
                "metric_timeout": task_timeout,
            },
            timeout=task_timeout,
            execution_mode="process",
        )
        if not outcome.ok or not isinstance(outcome.result, dict):
            raise RuntimeError(outcome.error or "Metadata worker returned an invalid result")
        return outcome.result

    # 分块并行执行并回填
    updated = _chunked_map(task_dirs, worker_count, analyze_in_process, "metadata")
    updated_by_name = {row["Name"]: row for row in updated}
    if set(updated_by_name) != set(task_names):
        raise RuntimeError("Metadata results do not match task directories")
    # 按原任务顺序写回 CSV
    _write_overview_rows(output_csv, [updated_by_name[name] for name in task_names])


def analyze_task(
    task_dir: Path,
    skip_llm: bool = False,
    metric_execution_mode: str = "process",
    metric_timeout: float = 120,
) -> dict[str, Any]:
    """为单个任务生成总览行:尺寸/指标评估立即可得,LLM 字段置占位符。"""
    # 公开数据目录
    public_dir = task_dir / "data" / "public"
    # 原始数据:目录优先,退到 raw.txt 占位文件
    raw_path = task_dir / "raw"
    if not raw_path.exists():
        raw_path = task_dir / "raw.txt"

    # 基础行:LLM 相关列先给占位标记(跳过/待填)
    row: dict[str, Any] = {
        "Name": task_dir.name,
        "Modality": "#SKIPPED_LLM" if skip_llm else "#PENDING_LLM",
        "Task": "#SKIPPED_LLM" if skip_llm else "#PENDING_LLM",
        # 原始大小:目录递归统计,文件取 stat,都没有按 0
        "Raw Size": _format_size(_directory_size(raw_path)) if raw_path.is_dir() else (
            _format_size(raw_path.stat().st_size) if raw_path.exists() else "0.00 B"
        ),
        # 最终处理集大小
        "Final Size": _format_size(_directory_size(task_dir / "data")),
        "CPU/GPU": "#SKIPPED_LLM" if skip_llm else "#PENDING_LLM",
        "Source": "OpenMLE",
        "Metric": "",
        "NOTE": "",
        "Server Location": str(task_dir),
    }

    # 缺少任务描述:备注标记
    desc = public_dir / "description.txt"
    if not desc.exists():
        row["NOTE"] = "#DESCRIPTION_NOT_FOUND"

    try:
        # 指标评估:isolated 模式走隔离子进程,否则进程内
        if metric_execution_mode == "isolated":
            result = evaluate_sample_submission_process(
                task_dir,
                execution_mode="isolated",
                task_timeout=metric_timeout,
            )
        else:
            result = evaluate_sample_submission(task_dir)
        row["Metric"] = f"@Evaluation result: {{'score': {result['score']}}}"
    except Exception as exc:
        # isolated 失败要上抛(worker 内整体判失败);进程内失败记错误串
        if metric_execution_mode == "isolated":
            raise
        row["Metric"] = f"#MEVA_RUNTIME_ERROR: {type(exc).__name__}: {exc}"
    return row


def _analyze_task_full(
    task_dir: Path,
    skip_llm: bool = False,
    metric_execution_mode: str = "process",
    metric_timeout: float = 120,
) -> dict[str, Any]:
    """Analyze one task entirely inside its task worker process."""
    # 先做非 LLM 的基础分析
    row = analyze_task(
        task_dir,
        skip_llm=skip_llm,
        metric_execution_mode=metric_execution_mode,
        metric_timeout=metric_timeout,
    )
    # 跳过 LLM:基础行即为结果
    if skip_llm:
        return row
    _require_llm_env()
    # 延迟导入元数据管线的三个 LLM 组件
    from metadata_pipeline import (
        compute_requirement_classifier,
        sample_submission_validator,
        task_categorizer,
    )

    # LLM 任务分类:"模态@任务"格式
    category = task_categorizer.extract_choice_with_llm(str(task_dir))
    parts = category.split("@")
    row["Modality"] = parts[0].strip() if parts else category
    row["Task"] = parts[1].strip() if len(parts) > 1 else category
    # LLM 算力需求判定(结合最终数据大小)
    row["CPU/GPU"] = compute_requirement_classifier.extract_compute_requirement_with_llm(
        str(task_dir),
        row.get("Final Size", "Unknown"),
    )
    # LLM 样例提交校验
    metric_result, _ = sample_submission_validator.validate_sample_submission(task_dir)
    row["Metric"] = metric_result
    return row


def _failed_overview_row(task_dir: Path, reason: str, skip_llm: bool) -> dict[str, Any]:
    """构造失败任务的总览行:错误原因写入 Metric 与 NOTE(带标记前缀)。"""
    marker = "#SKIPPED_LLM" if skip_llm else "#PENDING_LLM"
    return {
        "Name": task_dir.name,
        "Modality": marker,
        "Task": marker,
        "Raw Size": "0.00 B",
        "Final Size": "0.00 B",
        "CPU/GPU": marker,
        "Source": "OpenMLE",
        "Metric": f"#MEVA_RUNTIME_ERROR: {reason}",
        "NOTE": f"#TASK_RUNTIME_ERROR: {reason}",
        "Server Location": str(task_dir),
    }


def overview_has_task_failures(path: str | Path) -> bool:
    """检查总览 CSV 中是否存在 #TASK_RUNTIME_ERROR 行。"""
    return any(
        row.get("NOTE", "").startswith("#TASK_RUNTIME_ERROR:")
        for row in _read_overview_rows(Path(path))
    )


def generate_overview(
    tasks_root: str | Path,
    output_csv: str | Path,
    workers: int = 1,
    skip_llm: bool = False,
    execution_mode: str = "process",
    task_timeout: float = 600,
) -> Path:
    """生成全量任务总览:每个任务在独立子进程分析,失败行不拖垮整批。"""
    # 解析任务根目录并收集任务
    tasks_root_path = Path(tasks_root).resolve()
    task_dirs = collect_task_dirs(tasks_root_path)
    if not task_dirs:
        raise FileNotFoundError(f"No task directories containing data/ found under {tasks_root}")

    # 并发数收敛
    worker_count = max(1, min(workers, len(task_dirs)))
    # 单任务实现:先做名称/路径校验,再交给子进程 worker
    def run_one_impl(task_dir: Path) -> dict[str, Any]:
        try:
            # 任务名合法且确实位于任务根之下(防路径逃逸)
            validate_task_name(task_dir.name)
            task_dir.resolve().relative_to(tasks_root_path)
        except ValueError as exc:
            return _failed_overview_row(task_dir, str(exc), skip_llm)
        # 子进程执行 overview worker(任务目录只读)
        outcome = run_task_process(
            "overview",
            {
                "task_dir": str(task_dir),
                "skip_llm": skip_llm,
                "metric_execution_mode": execution_mode,
                "metric_timeout": task_timeout,
            },
            timeout=task_timeout,
            execution_mode="process",
            readonly_paths=(task_dir,),
        )
        # 回放子进程输出
        if outcome.stdout:
            print(outcome.stdout, end="")
        if outcome.stderr:
            print(outcome.stderr, end="", file=sys.stderr)
        if (
            not outcome.ok
            or not isinstance(outcome.result, dict)
            or outcome.result.get("Name") != task_dir.name
        ):
            return _failed_overview_row(
                task_dir,
                outcome.error or "Task worker returned an invalid task result",
                skip_llm,
            )
        return outcome.result

    # 包装一层兜底:任何异常都转成失败行(批次独立性)
    def run_one(task_dir: Path) -> dict[str, Any]:
        try:
            return run_one_impl(task_dir)
        except Exception as exc:
            return _failed_overview_row(
                task_dir,
                f"{type(exc).__name__}: {exc}",
                skip_llm,
            )

    # 分块并行执行全部任务
    rows = _chunked_map(task_dirs, worker_count, run_one, "overview")

    # 按任务名排序后原子写盘
    target = ensure_parent(output_csv)
    _write_overview_rows(target, sorted(rows, key=lambda item: item["Name"]))
    return target
