# =====================================================================
# 中文注释导读（OpenRSI / OpenMLE-Gym 学习文档）：
# 本文件是 OpenMLE-Gym 的"任务构建"编排入口：把一批比赛 slug（如 Kaggle
# competition 名）批量构建成可执行、可评估的 MLE 任务包。
# 关键设计：
#   1) MLEBENCH_SLUGS 白名单：默认跳过 MLE-Bench 已收录的比赛（避免评测污染，
#      保证训练数据与评测基准去重，对应论文"duplicate against all evaluation
#      benchmarks"的去重原则）；
#   2) 进程隔离构建：每个任务在独立子进程中执行（run_task_process），互不污染，
#      并按 只读/可写 路径白名单限制文件访问；
#   3) 异步并发 + 信号量限流：max_concurrency 控制同时构建的任务数；
#   4) Preflight 预检：任务名合法性、重名检测、输出路径越界检测在分派前完成。
# 构建产出的任务包（data/forge/downloads）之后由 local_evaluator.py 做质量评估。
# =====================================================================
from __future__ import annotations

import asyncio  # 并发编排：信号量限流 + gather
import sys  # 子进程 stderr 转发
from datetime import datetime  # 批次名默认用时间戳
from pathlib import Path  # 路径与"越界"检查
from typing import Any

from .common import json_safe, read_slug_entries, read_slugs_file, validate_task_name  # slug 读取与校验工具
from .process_runner import run_task_process  # 进程隔离的任务执行器


# MLE-Bench 收录的比赛 slug 集合：默认跳过它们，防止构建出与评测基准同源的任务包
# （论文强调训练数据须对所有评测基准去重，这是去重在构建阶段的落点）
MLEBENCH_SLUGS = {
    "3d-object-detection-for-autonomous-vehicles",
    "AI4Code",
    "aerial-cactus-identification",
    "alaska2-image-steganalysis",
    "aptos2019-blindness-detection",
    "billion-word-imputation",
    "bms-molecular-translation",
    "cassava-leaf-disease-classification",
    "cdiscount-image-classification-challenge",
    "chaii-hindi-and-tamil-question-answering",
    "champs-scalar-coupling",
    "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "facebook-recruiting-iii-keyword-extraction",
    "freesound-audio-tagging-2019",
    "google-quest-challenge",
    "google-research-identify-contrails-reduce-global-warming",
    "h-and-m-personalized-fashion-recommendations",
    "herbarium-2020-fgvc7",
    "herbarium-2021-fgvc8",
    "herbarium-2022-fgvc9",
    "histopathologic-cancer-detection",
    "hms-harmful-brain-activity-classification",
    "hotel-id-2021-fgvc8",
    "hubmap-kidney-segmentation",
    "icecube-neutrinos-in-deep-ice",
    "imet-2020-fgvc7",
    "inaturalist-2019-fgvc6",
    "iwildcam-2019-fgvc6",
    "iwildcam-2020-fgvc7",
    "jigsaw-toxic-comment-classification-challenge",
    "jigsaw-unintended-bias-in-toxicity-classification",
    "kuzushiji-recognition",
    "leaf-classification",
    "learning-agency-lab-automated-essay-scoring-2",
    "lmsys-chatbot-arena",
    "mlsp-2013-birds",
    "multi-modal-gesture-recognition",
    "new-york-city-taxi-fare-prediction",
    "nfl-player-contact-detection",
    "nomad2018-predict-transparent-conductors",
    "osic-pulmonary-fibrosis-progression",
    "petfinder-pawpularity-score",
    "plant-pathology-2020-fgvc7",
    "plant-pathology-2021-fgvc8",
    "predict-volcanic-eruptions-ingv-oe",
    "random-acts-of-pizza",
    "ranzcr-clip-catheter-line-classification",
    "rsna-2022-cervical-spine-fracture-detection",
    "rsna-breast-cancer-detection",
    "rsna-miccai-brain-tumor-radiogenomic-classification",
    "seti-breakthrough-listen",
    "siim-covid19-detection",
    "siim-isic-melanoma-classification",
    "smartphone-decimeter-2022",
    "spooky-author-identification",
    "stanford-covid-vaccine",
    "statoil-iceberg-classifier-challenge",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022",
    "tensorflow-speech-recognition-challenge",
    "tensorflow2-question-answering",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "tgs-salt-identification-challenge",
    "the-icml-2013-whale-challenge-right-whale-redux",
    "tweet-sentiment-extraction",
    "us-patent-phrase-to-phrase-matching",
    "uw-madison-gi-tract-image-segmentation",
    "ventilator-pressure-prediction",
    "vesuvius-challenge-ink-detection",
    "vinbigdata-chest-xray-abnormalities-detection",
    "whale-categorization-playground",
}


def _json_safe(value: Any) -> Any:
    """Convert legacy pipeline return objects into JSON-serializable data."""
    # 委托 common.json_safe：把不可序列化对象（Path/异常等）转成 JSON 友好结构
    return json_safe(value)


def _summarize_results(results: list[Any]) -> dict[str, Any]:
    # 汇总一批构建结果：成功/失败计数与批次整体状态
    failed = 0
    succeeded = 0
    for result in results:
        if isinstance(result, dict):
            if result.get("success") is False or result.get("status") == "failed":
                failed += 1
            else:
                succeeded += 1
        else:
            # 非 dict 结果一律按失败计
            failed += 1
    total = len(results)
    if total == 0:
        status = "no_tasks"
    elif failed == 0:
        status = "success"
    elif succeeded == 0:
        status = "failed"
    else:
        status = "partial_success"
    return {
        "batch_status": status,
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
    }


def _dry_run(
    slugs: list[str],
    output_root: Path,
    batch_name: str,
    skip_mlebench: bool,
    delete_raw: bool,
) -> dict[str, Any]:
    # 演练模式（execute=False）：只报告"会构建哪些、会跳过哪些"，不实际下载/生成
    selected = [slug for slug in slugs if not (skip_mlebench and slug in MLEBENCH_SLUGS)]
    skipped = [slug for slug in slugs if skip_mlebench and slug in MLEBENCH_SLUGS]
    return {
        "status": "dry-run",
        "batch_name": batch_name,
        "output_root": str(output_root),
        "delete_raw": delete_raw,
        "selected": selected,
        "skipped_mlebench": skipped,
    }


async def _run_one_local(
    builder_root: str | Path,
    batch_name: str,
    slug: str,
    retry: int,
    output_root: str | Path,
    info_csv: str | Path,
    delete_raw: bool,
    code_execution_mode: str = "process",
    code_timeout: float = 600,
) -> Any:
    # 单任务"进程内"构建路径（不走子进程隔离）：直接实例化 builder_core 的 LangGraph
    # 流水线并异步执行。仅用于调试/受控场景，批量构建默认走 _run_all 的进程隔离。
    from builder_core.design import Graph  # 构建流水线的图定义（下载->感知->描述->准备->指标）
    from builder_core.utils.state import AgentState  # 图运行的全局状态
    from builder_core.utils.task import Task  # 单个比赛任务的阶段标志位

    builder_path = Path(builder_root)
    output_path = Path(output_root)
    info_path = Path(info_csv)
    agent = AgentState({"messages": []})
    # 所有阶段标志置 False：由 Graph 内部按需推进各阶段
    task = Task(
        name=slug,
        retry=retry,
        success=False,
        download=False,
        copy=False,
        web_info=False,
        perceive=False,
        describe=False,
        gen_prep=False,
        metric=False,
        prepare=False,
        prepares=[],
        errors=[],
    )
    app = Graph(
        batch_name=batch_name,
        competition=task,
        base_dir=output_path,
        info_csv=info_path,
        sample_dir=builder_path / "samples",
        delete_raw=delete_raw,
        code_execution_mode=code_execution_mode,
        code_timeout=code_timeout,
    ).compile()
    result = await app.ainvoke(agent)
    return result["task_result"]


async def _run_all(
    builder_root: Path,
    batch_name: str,
    slugs: list[str],
    retry: int,
    output_root: Path,
    info_csv: Path,
    max_concurrency: int,
    delete_raw: bool,
    execution_mode: str,
    task_timeout: float,
) -> list[Any]:
    # 批量构建主路径：每个任务派发到独立子进程（进程隔离 + 文件权限白名单）
    sem = asyncio.Semaphore(max(1, max_concurrency))  # 并发上限（至少 1）

    async def worker(slug: str) -> Any:
        async with sem:
            # run_task_process：fork 一个干净的子进程运行"build"入口，
            # 只允许读 builder_root/info_csv，只允许写 output_root
            outcome = await asyncio.to_thread(
                run_task_process,
                "build",
                {
                    "builder_root": str(builder_root),
                    "batch_name": batch_name,
                    "slug": slug,
                    "retry": retry,
                    "output_root": str(output_root),
                    "info_csv": str(info_csv),
                    "delete_raw": delete_raw,
                    "code_execution_mode": execution_mode,
                    "code_timeout": task_timeout,
                },
                timeout=task_timeout,
                execution_mode="process",
                readonly_paths=(builder_root, info_csv),
                writable_paths=(output_root,),
            )
            # 子进程 stdout/stderr 实时转发到父进程日志
            if outcome.stdout:
                print(outcome.stdout, end="")
            if outcome.stderr:
                print(outcome.stderr, end="", file=sys.stderr)
            if (
                outcome.ok
                and isinstance(outcome.result, dict)
                and isinstance(outcome.result.get("success"), bool)
            ):
                # 子进程成功且返回结构合法
                return outcome.result
            return {
                "name": slug,
                "status": "failed",
                "success": False,
                "errors": [
                    outcome.error or "Task worker returned an invalid result"
                ],
            }

    # 并发执行全部任务；单个任务抛异常不影响其他任务
    results = await asyncio.gather(*(worker(slug) for slug in slugs), return_exceptions=True)
    normalized = []
    for slug, result in zip(slugs, results, strict=True):
        if isinstance(result, BaseException):
            # 异常统一归一化为失败记录（带异常类型名，便于排查）
            normalized.append(
                {
                    "name": slug,
                    "status": "failed",
                    "success": False,
                    "errors": [f"{type(result).__name__}: {result}"],
                }
            )
        else:
            normalized.append(result)
    return normalized


def build_tasks(
    slugs_file: str | Path,
    output_root: str | Path = "artifacts/builds",
    info_csv: str | Path = "builder_core/info.csv",
    batch_name: str | None = None,
    retry: int = 2,
    max_concurrency: int = 1,
    limit: int | None = None,
    skip_mlebench: bool = True,
    delete_raw: bool = False,
    execute: bool = False,
    execution_mode: str = "process",
    task_timeout: float = 3600,
) -> dict[str, Any]:
    # 【任务构建总入口】读取 slug 清单 -> 预检 -> 并发派发构建 -> 汇总报告。
    # CLI（cli.py 的 build 子命令）直接调用本函数。
    slugs = read_slugs_file(slugs_file)
    if limit is not None:
        slugs = slugs[:limit]
    output_path = Path(output_root).resolve()
    info_path = Path(info_csv).resolve()
    # 批次名缺省为时间戳：保证多次构建互不覆盖
    name = batch_name or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-openmle"
    batch_path = (output_path / name).resolve()
    try:
        # 防目录穿越：batch 名不得让批次目录逃出输出根目录
        batch_path.relative_to(output_path)
    except ValueError as exc:
        raise ValueError(f"Batch path escapes output root: {name!r}") from exc

    if not execute:
        return _dry_run(slugs, output_path, name, skip_mlebench, delete_raw)

    builder_root = Path(__file__).resolve().parent.parent / "builder_core"
    if not builder_root.exists():
        raise FileNotFoundError(f"Builder core directory not found: {builder_root}")
    if not info_path.exists():
        raise FileNotFoundError(f"info.csv not found: {info_path}")

    # ---- Preflight 预检：把"必然失败"的任务在派发前拦截，附上明确原因 ----
    raw_entries = read_slug_entries(slugs_file)
    selected: list[tuple[int, str]] = []
    preflight_failures: dict[int, dict[str, Any]] = {}
    seen: set[str] = set()
    for index, slug in enumerate(raw_entries):
        if limit is not None and index >= limit:
            break
        try:
            # 1) 任务名合法性（防止路径注入字符）
            validate_task_name(slug)
        except ValueError as exc:
            preflight_failures[index] = {
                "name": slug,
                "status": "failed",
                "success": False,
                "errors": [str(exc)],
            }
            continue
        if slug in seen:
            # 2) 重名检测：同批任务名唯一
            preflight_failures[index] = {
                "name": slug,
                "status": "failed",
                "success": False,
                "errors": [f"Duplicate task name: {slug}"],
            }
            continue
        seen.add(slug)
        try:
            # 3) 三个产出路径（数据/锻造/下载包）都不得逃出批次根目录
            for relative in (
                Path("data") / slug,
                Path("forge") / slug,
                Path("downloads") / f"{slug}.zip",
            ):
                (batch_path / relative).resolve().relative_to(batch_path)
        except ValueError:
            preflight_failures[index] = {
                "name": slug,
                "status": "failed",
                "success": False,
                "errors": [f"Task output path escapes batch root: {slug}"],
            }
            continue
        if not (skip_mlebench and slug in MLEBENCH_SLUGS):
            # 4) MLE-Bench 去重：skip_mlebench=True 时跳过评测基准同源比赛
            selected.append((index, slug))
    # 并发执行构建（每个任务一个隔离子进程）
    results = asyncio.run(
        _run_all(
            builder_root=builder_root,
            batch_name=name,
            slugs=[slug for _, slug in selected],
            retry=retry,
            output_root=output_path,
            info_csv=info_path,
            max_concurrency=max_concurrency,
            delete_raw=delete_raw,
            execution_mode=execution_mode,
            task_timeout=task_timeout,
        )
    )
    # 按原始行号对齐结果：预检失败与构建结果合并成完整清单
    indexed_results = {
        index: result
        for (index, _), result in zip(selected, results, strict=True)
    }
    indexed_results.update(preflight_failures)
    safe_results = [
        _json_safe(indexed_results[index])
        for index in sorted(indexed_results)
    ]
    return {
        "status": "executed",
        "batch_name": name,
        "output_root": str(output_path),
        "delete_raw": delete_raw,
        "summary": _summarize_results(safe_results),
        "results": safe_results,
    }
