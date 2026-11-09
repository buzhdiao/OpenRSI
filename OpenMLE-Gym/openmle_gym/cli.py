# 中文说明:OpenMLE-Gym 的统一命令行入口(openmle-task)。
# 在系统中的位置:包的最外层用户界面,聚合五个子命令:
# prepare-input(准备 Evo/SFT/RL 的任务输入 parquet)、build(从 Kaggle slug 构建任务包)、
# overview(生成任务总览 CSV)、metric-check(校验 metric.py)、evaluate(结构与质量评估)、
# leaderboard(下载 Kaggle 公开排行榜)。
# 关键设计:所有结果以 JSON 打印;默认 dry-run(--execute 才真执行);
# 批量命令默认"有任何失败即退出码 1",可用 --allow-partial-success 放行为 0;
# 任务执行统一支持 process/isolated 两种模式与单任务超时。

# 延迟注解:类型标注运行期不求值
from __future__ import annotations

# argparse:子命令解析
import argparse
# json:结果 JSON 打印
import json
# Path:路径对象(类型提示用)
from pathlib import Path
# 类型注解
from typing import Any

# 各子命令的实现:构建/排行榜/评估/指标校验/总览
from .build import build_tasks
from .leaderboard import download_leaderboards
from .local_evaluator import evaluation_has_task_failures, evaluate_tasks
from .metric_validation import evaluate_sample_submission_process
from .overview import generate_overview, overview_has_task_failures


def _load_env_file() -> None:
    """尽力加载 .env(dotenv 未安装时静默跳过)。"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _print_json(data: Any) -> None:
    """以两空格缩进打印 JSON(保留非 ASCII 字符)。"""
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _add_execute_flag(parser: argparse.ArgumentParser) -> None:
    """给子命令挂上 --execute 开关(默认 dry-run,不真正调用外部服务)。"""
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run external-service or account-mutating operations. Default is dry-run.",
    )


def _add_partial_success_flag(parser: argparse.ArgumentParser) -> None:
    """给批量子命令挂上 --allow-partial-success(失败项存在时仍返回 0)。"""
    parser.add_argument(
        "--allow-partial-success",
        action="store_true",
        help="Return exit code 0 even when one or more batch items failed.",
    )


def _add_execution_flags(parser: argparse.ArgumentParser, timeout: float) -> None:
    """挂上执行模式与单任务超时两个通用旗标。"""
    parser.add_argument(
        "--execution-mode",
        choices=("process", "isolated"),
        default="process",
        help="Run each task in a child process, or in a configured isolated container.",
    )
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=timeout,
        help="Maximum seconds allowed for one task worker.",
    )


def _has_batch_failures(data: Any) -> bool:
    """递归判断批量结果里是否存在失败项(summary.failed/success=False/status=failed)。"""
    if isinstance(data, dict):
        # 优先看汇总计数
        summary = data.get("summary")
        if isinstance(summary, dict) and int(summary.get("failed") or 0) > 0:
            return True
        # 兼容顶层失败标记,并递归检查 results 列表
        return (
            data.get("success") is False
            or data.get("status") == "failed"
            or _has_batch_failures(data.get("results", []))
        )
    if isinstance(data, list):
        # 列表:任一元素失败即失败
        return any(_has_batch_failures(item) for item in data)
    return False


def _print_batch_json(data: Any, allow_partial_success: bool) -> int:
    """打印批量结果 JSON 并决定退出码(默认有失败即 1)。"""
    _print_json(data)
    if not allow_partial_success and _has_batch_failures(data):
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 解析器:定义五个子命令及其全部参数。"""
    parser = argparse.ArgumentParser(
        prog="openmle-task",
        description="Build, inspect, and evaluate standardized Kaggle-style ML task packages.",
    )
    # 子命令必选
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 子命令 1:准备训练/评估输入 parquet
    prepare = subparsers.add_parser(
        "prepare-input", help="Prepare the existing Evo/SFT-rollout/RL task input Parquet."
    )
    prepare.add_argument("--task-dir", required=True, help="Downloaded task-package root.")
    prepare.add_argument("--sandbox-task-dir", required=True, help="Absolute task-package path visible to the sandbox.")
    prepare.add_argument("--output", required=True, help="Output Parquet file.")
    prepare.add_argument("--task-uuid", help="Reuse an existing input UUID; otherwise preserve the package UUID or assign a new one.")

    # 子命令 2:从 Kaggle slug 构建任务包
    build = subparsers.add_parser("build", help="Build task packages from Kaggle competition slugs.")
    build.add_argument("--slugs-file", required=True, help="Text file with one Kaggle slug or URL per line.")
    build.add_argument("--output-root", default="artifacts/builds", help="Directory where build batches are written.")
    build.add_argument("--info-csv", default="builder_core/info.csv", help="Meta Kaggle info CSV used by the legacy builder.")
    build.add_argument("--batch-name", default=None, help="Optional run/batch name.")
    build.add_argument("--retry", type=int, default=2, help="Per-competition retry count.")
    build.add_argument("--max-concurrency", type=int, default=1, help="Maximum concurrent competition builds.")
    build.add_argument("--limit", type=int, default=None, help="Optional maximum number of slugs to process.")
    build.add_argument("--include-mlebench", action="store_true", help="Do not skip known MLE-Bench slugs.")
    build.add_argument(
        "--delete-raw",
        action="store_true",
        help="Delete each task raw/ directory after build and write raw.txt from forge/fileinfo.txt.",
    )
    # 构建要真下载,默认 dry-run;支持部分成功与执行模式
    _add_execute_flag(build)
    _add_partial_success_flag(build)
    _add_execution_flags(build, 3600)

    # 子命令 3:生成任务总览 CSV
    overview = subparsers.add_parser("overview", help="Generate local task package overview CSV.")
    overview.add_argument("--tasks-root", required=True, help="Directory containing task packages.")
    overview.add_argument("--output-csv", required=True, help="CSV path to write.")
    overview.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent chunk workers for package inspection and metadata stages.",
    )
    _add_execution_flags(overview, 600)
    overview.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM-only category and CPU/GPU classification.",
    )

    # 子命令 4:单任务 metric.py 校验
    metric = subparsers.add_parser("metric-check", help="Evaluate sample_submission.csv with utils/metric.py.")
    metric.add_argument("--task-dir", required=True, help="Single task package directory.")
    _add_execution_flags(metric, 120)

    # 子命令 5:任务结构与元数据管线评估
    evaluator = subparsers.add_parser("evaluate", help="Run task-structure and metadata pipeline evaluation.")
    evaluator.add_argument("--root-dir", required=True, help="Root directory containing task packages.")
    evaluator.add_argument("--task-list", required=True, help="Text file with task directory names.")
    evaluator.add_argument("--output-dir", required=True, help="Directory for JSON and CSV results.")
    evaluator.add_argument("--overview-csv", default=None, help="Optional existing overview CSV to reuse.")
    evaluator.add_argument("--workers", type=int, default=1, help="Number of concurrent chunk workers.")
    evaluator.add_argument(
        "--skip-llm",
        action="store_true",
        help=(
            "Skip LLM metadata and quality evaluation; emit deterministic "
            "validation without quality scores."
        ),
    )
    evaluator.add_argument(
        "--local-only",
        action="store_true",
        help=(
            "Run deterministic structure and metric checks without producing "
            "quality scores (compatibility alias for --skip-llm)."
        ),
    )
    _add_execution_flags(evaluator, 600)

    # 子命令 6:下载 Kaggle 公开排行榜
    leaderboard = subparsers.add_parser("leaderboard", help="Download Kaggle public leaderboards.")
    leaderboard.add_argument("--slugs-file", required=True, help="Text file with one Kaggle slug or URL per line.")
    leaderboard.add_argument("--out-dir", default="artifacts/leaderboards", help="Output directory for CSV files.")
    leaderboard.add_argument("--kaggle-json", default=None, help="Optional Kaggle JSON credential file.")
    leaderboard.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds.")
    leaderboard.add_argument("--sleep-seconds", type=float, default=1.0, help="Delay between requests.")
    _add_execute_flag(leaderboard)
    _add_partial_success_flag(leaderboard)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口:分发子命令并返回进程退出码(0 成功/1 有失败/2 参数错误)。"""
    # 加载 .env 并解析参数
    _load_env_file()
    parser = build_parser()
    args = parser.parse_args(argv)

    # prepare-input:延迟导入实现,打印结果
    if args.command == "prepare-input":
        from .prepare_input import prepare_input

        _print_json(prepare_input(
            task_dir=args.task_dir,
            sandbox_task_dir=args.sandbox_task_dir,
            output=args.output,
            task_uuid=args.task_uuid,
        ))
        return 0

    # build:批量构建,退出码按失败情况决定
    if args.command == "build":
        return _print_batch_json(
            build_tasks(
                slugs_file=args.slugs_file,
                output_root=args.output_root,
                info_csv=args.info_csv,
                batch_name=args.batch_name,
                retry=args.retry,
                max_concurrency=args.max_concurrency,
                limit=args.limit,
                skip_mlebench=not args.include_mlebench,
                delete_raw=args.delete_raw,
                execute=args.execute,
                execution_mode=args.execution_mode,
                task_timeout=args.task_timeout,
            ),
            allow_partial_success=args.allow_partial_success,
        )

    # overview:生成总览;存在失败任务行则 partial_success + 退出码 1
    if args.command == "overview":
        output = generate_overview(
            tasks_root=args.tasks_root,
            output_csv=args.output_csv,
            workers=args.workers,
            skip_llm=args.skip_llm,
            execution_mode=args.execution_mode,
            task_timeout=args.task_timeout,
        )
        has_failures = overview_has_task_failures(output)
        _print_json(
            {
                "status": "partial_success" if has_failures else "ok",
                "output_csv": str(output),
            }
        )
        return 1 if has_failures else 0

    # metric-check:单任务指标校验(隔离子进程),恒返回 0
    if args.command == "metric-check":
        _print_json(
            evaluate_sample_submission_process(
                args.task_dir,
                execution_mode=args.execution_mode,
                task_timeout=args.task_timeout,
            )
        )
        return 0

    # evaluate:批量评估;存在任务失败则 partial_success + 退出码 1
    if args.command == "evaluate":
        paths = evaluate_tasks(
            root_dir=args.root_dir,
            task_list=args.task_list,
            output_dir=args.output_dir,
            local_only=args.local_only,
            skip_llm=args.skip_llm,
            overview_csv=args.overview_csv,
            workers=args.workers,
            execution_mode=args.execution_mode,
            task_timeout=args.task_timeout,
        )
        has_failures = evaluation_has_task_failures(paths)
        _print_json(
            {
                "status": "partial_success" if has_failures else "ok",
                **{key: str(path) for key, path in paths.items()},
            }
        )
        return 1 if has_failures else 0

    # leaderboard:批量下载排行榜
    if args.command == "leaderboard":
        return _print_batch_json(
            download_leaderboards(
                slugs_file=args.slugs_file,
                out_dir=args.out_dir,
                kaggle_json=args.kaggle_json,
                timeout=args.timeout,
                sleep_seconds=args.sleep_seconds,
                execute=args.execute,
            ),
            allow_partial_success=args.allow_partial_success,
        )

    # 未知命令:报错并返回 2
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
