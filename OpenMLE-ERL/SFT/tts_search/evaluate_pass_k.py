"""
Pass@k Evaluation Script with Decoupled Generator/Evaluator Services.

This script implements a pass@k evaluation system where:
1. k code samples are generated per task
2. All samples are evaluated on the test set (no validation set)
3. Rejection policy decides which evaluated samples count toward the loop target
4. Generation stops when the target is met or the max generation budget is reached

Supports three modes:
- Full mode: Generate and evaluate (default)
- Generation only: Set enable_sandbox_eval=False
- Evaluation only: Set evaluate_only=True (evaluates existing gen_results.jsonl)

Checkpoint resume:
- Automatically resumes from gen_results.jsonl and eval_results.jsonl
- Skips already-generated samples
- Evaluates all unevaluated samples (including those from previous runs)

Usage:
    # Full generation + evaluation
    python -m tts_search.evaluate_pass_k --config-name experiment/parallel_glm47

    # Generation only (no evaluation)
    python -m tts_search.evaluate_pass_k --config-name experiment/parallel_glm47 enable_sandbox_eval=False

    # Evaluation only (evaluate existing generations)
    python -m tts_search.evaluate_pass_k --config-name experiment/parallel_glm47 evaluate_only=True resume_from=<OUTPUT_DIR>
"""

# 让类型注解延迟求值，从而支持新版联合类型等注解语法
from __future__ import annotations

# 标准库导入：异步调度、JSON、日志、环境变量、随机数、文件移动、路径对象、类型注解
import asyncio
import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any

# 先配置运行时存储（默认 run_id），必须在 hydra 导入前完成以便其读取正确的工作目录
from tts_search.runtime_storage import configure_runtime_storage

# 执行运行时存储配置（副作用导入）
configure_runtime_storage(default_run_id="evaluate_pass_k")

# 第三方依赖：hydra 配置框架、pandas 读 parquet、dotenv 加载环境变量、litellm 路由、omegaconf 配置对象
import hydra
import pandas as pd
from dotenv import load_dotenv
from hydra.utils import instantiate
from litellm.router import Router
from omegaconf import DictConfig, OmegaConf

# from tts_search import eval_utils
# 任务元数据增强：从 metricprompt parquet 附加 self_valid_protocol（指标感知 gap 过滤用）
from tts_search.data_produce.gap_filter import enrich_tasks_with_metric_protocol
# pass@k 草稿 prompt 构建器
from tts_search.prompt_builder import build_pass_k_draft_messages
# 解耦生成/评测的调度器及其配置
from tts_search.services import Scheduler, SchedulerConfig
# 断点续跑相关工具：从 progress/gen/eval 结果重建任务状态
from tts_search.services.scheduler import (
    build_task_states_from_progress,
    load_checkpoint,
    load_progress_snapshot,
    progress_snapshot_has_medal_counts,
    rebuild_task_states_from_results,
)

# 仓库根目录（本文件位置的上一级）
REPO_ROOT = Path(__file__).resolve().parents[1]
# tts_search 包目录
TTS_SEARCH_DIR = REPO_ROOT / "tts_search"
# 模块级 logger
logger = logging.getLogger(__name__)

# Load .env before Hydra parses config (so oc.env can resolve)
# 在 Hydra 解析配置前加载 .env，保证 oc.env 插值能取到环境变量
load_dotenv(REPO_ROOT / ".env")


def _is_minimax_config(cfg: DictConfig) -> bool:
    """Check if config uses MiniMax provider."""
    try:
        # 取模型列表第一项的 model 参数并转字符串
        model = str(cfg.litellm.model_list[0].litellm_params.get("model", ""))
        # 模型名包含 minimax 即认定为 MiniMax 配置
        return "minimax" in model.lower()
    except (AttributeError, IndexError, KeyError):
        # 配置结构不符合预期时按非 MiniMax 处理
        return False


def _parse_minimax_api_keys(cfg: DictConfig | None = None) -> list[str]:
    """Parse MiniMax API keys only if config uses MiniMax."""
    # 非 MiniMax 配置直接返回空列表（沿用配置文件中的 key）
    if cfg is not None and not _is_minimax_config(cfg):
        return []

    # 读取多 key 环境变量并去首尾空白
    keys_env = os.getenv("MINIMAX_API_KEYS", "").strip()
    if keys_env:
        try:
            if keys_env.startswith("["):
                # JSON 列表形式：解析为列表
                parsed = json.loads(keys_env)
                # 解析结果必须是列表
                if not isinstance(parsed, list):
                    raise ValueError("MINIMAX_API_KEYS must be a JSON list")
                # 逐项转字符串并过滤空项
                keys = [str(k).strip() for k in parsed if str(k).strip()]
            else:
                # 逗号分隔形式：拆分并过滤空项
                keys = [k.strip() for k in keys_env.split(",") if k.strip()]
        except Exception as exc:
            # 包装解析异常并提示合法格式
            raise ValueError(
                "Failed to parse MINIMAX_API_KEYS. Use JSON list or comma-separated."
            ) from exc
    else:
        # 回退到单 key 环境变量
        single_key = os.getenv("MINIMAX_API_KEY", "").strip()
        keys = [single_key] if single_key else []

    # 一个 key 都没有时直接报错
    if not keys:
        raise ValueError("No API keys found in MINIMAX_API_KEYS or MINIMAX_API_KEY")

    return keys


def _split_tasks_evenly(
    tasks: list[dict[str, Any]], n_shards: int
) -> list[list[dict[str, Any]]]:
    """Split task records across shards.

    Args:
        tasks: Task dictionaries loaded from the eval data source.
        n_shards: Number of output shards to create.

    Returns:
        A list of task lists, assigned round-robin for balanced shard sizes.
    """
    # 预分配 n_shards 个空分片
    shards: list[list[dict[str, Any]]] = [[] for _ in range(n_shards)]
    # 轮询（round-robin）分配：任务 i 进入第 i % n 分片，保证各分片规模均衡
    for idx, task in enumerate(tasks):
        shards[idx % n_shards].append(task)
    return shards


def _build_router_with_key(cfg: DictConfig, api_key: str) -> Router:
    """Build a LiteLLM router for one API key.

    Args:
        cfg: Hydra config containing the LiteLLM model list.
        api_key: Provider API key for this shard.

    Returns:
        A Router instance with the first model's API key overridden.
    """
    # 深拷贝 litellm 配置（不解析插值），避免污染原配置
    litellm_cfg = OmegaConf.create(OmegaConf.to_container(cfg.litellm, resolve=False))
    # 把第一个模型的 api_key 覆盖为分片专属 key
    litellm_cfg.model_list[0].litellm_params.api_key = api_key
    # 用 Hydra instantiate 实例化出 Router 对象
    return instantiate(litellm_cfg)


def _merge_dirs(src: Path, dst: Path) -> None:
    """Merge src directory into dst without overwriting existing files."""
    # 递归遍历源目录的每个条目
    for item in src.iterdir():
        # 计算目标路径
        target = dst / item.name
        if item.is_dir():
            # 子目录：先确保目标目录存在，再递归合并
            target.mkdir(parents=True, exist_ok=True)
            _merge_dirs(item, target)
        else:
            if target.exists():
                # 目标文件已存在时跳过，绝不覆盖旧结果
                logger.warning("Skip existing file during merge: %s", target)
                continue
            # 移动文件到目标位置
            shutil.move(str(item), str(target))


def _migrate_task_dirs(output_dir: Path, task_subdir: str = "tasks") -> None:
    """Move task folders under output_dir into task_subdir."""
    # 目标 tasks 子目录（旧版本任务目录直接平铺在 output_dir 下）
    tasks_dir = output_dir / task_subdir
    tasks_dir.mkdir(parents=True, exist_ok=True)

    # 遍历输出目录的一级条目
    for entry in output_dir.iterdir():
        # 只处理目录
        if not entry.is_dir():
            continue
        # 跳过 tasks 目录自身
        if entry.name == task_subdir:
            continue

        # 在 tasks/ 下建同名目标目录并递归合并内容
        target_dir = tasks_dir / entry.name
        target_dir.mkdir(parents=True, exist_ok=True)
        _merge_dirs(entry, target_dir)
        try:
            # 合并完成后尝试删除空的旧目录
            entry.rmdir()
        except OSError:
            # 目录非空（有未迁移文件）时仅告警，不中断
            logger.warning("Directory not empty after merge: %s", entry)


async def run_pass_k_evaluation(
    cfg: DictConfig,
    router: Router,
    tasks: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """
    Run pass@k evaluation for all tasks.

    Args:
        cfg: Hydra configuration
        router: LiteLLM router
        tasks: List of task configurations
        output_dir: Output directory

    Returns:
        Global statistics
    """
    # 先把历史平铺的任务目录迁移到 tasks/ 子目录，统一目录布局
    _migrate_task_dirs(output_dir, task_subdir="tasks")

    # Resume from previous run if specified
    # 读取断点续跑目录配置
    resume_from = cfg.get("resume_from", None)
    print(f"resume_from: {resume_from}")
    # 已完成任务名集合（续跑时填充）
    completed_tasks: set[str] = set()
    # 从历史结果合并回来的任务级统计
    resumed_results: dict[str, dict[str, Any]] = {}

    if resume_from is not None:
        # 续跑目录转 Path
        resume_dir = Path(resume_from)
        logger.info(f"Attempting to resume from: {resume_dir}")

        if bool(cfg.get("loop_enabled", False)):
            # 循环（loop）模式续跑：先构建任务清单
            resume_task_list = []
            for task in tasks:
                # 取每条任务记录的元数据列
                metadata = task[cfg.data.metadata_key]
                resume_task_list.append(
                    {
                        "task_id": metadata["uuid"],
                        "task_name": metadata["task_name"],
                        "metadata": metadata,
                    }
                )

            # 循环目标增量：默认 = max_steps * 每任务样本数（一轮 pass@k 的样本量）
            loop_increment = int(
                cfg.get(
                    "loop_target_increment",
                    int(cfg.max_steps) * int(cfg.get("n_samples_per_task", 1)),
                )
            )
            # 循环成功目标：达到该成功次数即认为任务完成（默认 1）
            loop_success_target = int(cfg.get("loop_success_target", 1))
            # 循环成功的度量方式：success（有成功样本）或 medal（拿到奖牌）
            loop_success_metric = str(cfg.get("loop_success_metric", "success"))
            # 奖牌目标：未配置时退化为成功目标
            loop_medal_target = (
                int(cfg.get("loop_medal_target"))
                if cfg.get("loop_medal_target") is not None
                else loop_success_target
            )
            # 循环目标上限：未配置时为 None（不设上限）
            loop_max_target = (
                int(cfg.get("loop_max_target"))
                if cfg.get("loop_max_target") is not None
                else None
            )

            # 读取上次运行的 progress.json 快照
            progress_snapshot = load_progress_snapshot(resume_dir)
            if bool(cfg.get("resume_rebuild_task_states", False)):
                # 配置要求忽略 progress.json，改由 gen/eval 结果重建
                logger.info(
                    "Ignoring progress.json and rebuilding task states from gen/eval "
                    "results because resume_rebuild_task_states=true"
                )
                progress_snapshot = None
            # 优先用进度快照构建各任务的循环状态（已完成次数/目标等）
            task_states = build_task_states_from_progress(
                progress_snapshot=progress_snapshot,
                tasks=resume_task_list,
                loop_increment=loop_increment,
                loop_max_target=loop_max_target,
                rejection_policy_name=str(
                    cfg.get("rejection_policy", loop_success_metric)
                ).lower(),
            )
            if (
                loop_success_metric == "medal"
                and task_states
                and not progress_snapshot_has_medal_counts(progress_snapshot)
            ):
                # 奖牌模式下快照缺少奖牌计数：状态不可信，置空待重建
                task_states = {}

            # Fallback: rebuild task_states from gen/eval results if missing/empty
            # 快照缺失或无效时，回退到从 gen/eval 结果文件重建任务状态
            if not task_states:
                # 加载断点（任务状态、生成结果、评测结果）
                _, _, gen_results_data, eval_results_data = load_checkpoint(resume_dir)
                task_states = rebuild_task_states_from_results(
                    tasks=resume_task_list,
                    gen_results_data=gen_results_data,
                    eval_results_data=eval_results_data,
                    loop_increment=loop_increment,
                    loop_success_target=loop_success_target,
                    loop_success_metric=loop_success_metric,
                    loop_medal_target=loop_medal_target,
                    loop_max_target=loop_max_target,
                    rejection_target=cfg.get("rejection_target"),
                )
                logger.info(
                    f"Rebuilt task_states from gen/eval results: {len(task_states)} tasks"
                )

            for state in task_states.values():
                # 收集所有已达标（done）的任务名
                if state.done and state.task_name:
                    completed_tasks.add(state.task_name)

            if completed_tasks:
                # 汇报从续跑数据恢复的已完成任务数
                logger.info(
                    f"Loaded {len(completed_tasks)} completed tasks from resume data"
                )

        logger.info(f"Resumed {len(completed_tasks)} completed tasks from {resume_dir}")

    # Create scheduler configuration with decoupled pipeline for max efficiency
    logger.info(f"sandbox_base_url: {cfg.sandbox.base_url}")
    logger.info(
        f"sandbox_cpu_base_url: {cfg.sandbox.get('cpu_base_url', cfg.sandbox.base_url)}"
    )
    logger.info(
        f"sandbox_resource_type_override: {cfg.sandbox.get('resource_type_override')}"
    )
    # 构建解耦生成/评测流水线的调度器配置（逐项从 Hydra 配置映射到 SchedulerConfig 字段）：
    # 生成侧 max_steps（即 pass@k 的 k）、每任务样本数、LLM 并发与重试；
    # 沙箱侧 URL/资源类型/并发/超时/轮询参数与 score2reward 开关；
    # 评测侧批大小、任务并发、评测开关与队列批处理超时；
    # 循环评测的成功目标/度量/奖牌目标/目标增量与上限；
    # 拒绝采样策略、阈值、参考分数、奖牌集合与基线过滤器（token 上限/分词器/相对 gap）；
    # 树搜索开关、算法与 ProgramDatabase 参数；greedy 的草稿数与 debug/draft 概率。
    scheduler_config = SchedulerConfig(
        max_steps=int(cfg.max_steps),
        n_samples_per_task=int(cfg.get("n_samples_per_task", 1)),
        llm_concurrency=int(cfg.llm_concurrency),
        llm_max_retries_per_step=int(cfg.get("llm_max_retries_per_step", 8)),
        sandbox_base_url=str(cfg.sandbox.base_url),
        sandbox_cpu_base_url=str(cfg.sandbox.get("cpu_base_url", cfg.sandbox.base_url)),
        sandbox_resource_type_override=cfg.sandbox.get("resource_type_override"),
        sandbox_concurrency=int(cfg.sandbox.concurrency),
        job_timeout=int(cfg.sandbox.job_timeout),
        wait_timeout=int(cfg.sandbox.wait_timeout),
        poll_interval=int(cfg.sandbox.poll_interval),
        use_score2reward=bool(cfg.sandbox.get("use_score2reward", True)),
        batch_size=int(cfg.get("batch_size", 16)),
        task_concurrency=int(cfg.get("task_concurrency", 1)),
        enable_sandbox_eval=bool(cfg.get("enable_sandbox_eval", True)),
        skip_eval_for_done_tasks=bool(
            cfg.get(
                "skip_eval_for_done_tasks",
                SchedulerConfig.skip_eval_for_done_tasks,
            )
        ),
        # Decoupled pipeline settings
        eval_queue_maxsize=int(cfg.get("eval_queue_maxsize", 0)),
        eval_batch_timeout=float(cfg.get("eval_batch_timeout", 1.0)),
        task_subdir="tasks",
        # Looping evaluation settings
        loop_enabled=bool(cfg.get("loop_enabled", False)),
        loop_success_target=int(cfg.get("loop_success_target", 1)),
        loop_success_metric=str(cfg.get("loop_success_metric", "success")),
        loop_medal_target=(
            int(cfg.get("loop_medal_target"))
            if cfg.get("loop_medal_target") is not None
            else None
        ),
        loop_target_increment=int(
            cfg.get(
                "loop_target_increment",
                int(cfg.max_steps) * int(cfg.get("n_samples_per_task", 1)),
            )
        ),
        loop_max_target=(
            int(cfg.get("loop_max_target"))
            if cfg.get("loop_max_target") is not None
            else int(
                cfg.get(
                    "loop_target_increment",
                    int(cfg.max_steps) * int(cfg.get("n_samples_per_task", 1)),
                )
            )
        ),
        resume_rebuild_task_states=bool(
            cfg.get(
                "resume_rebuild_task_states",
                SchedulerConfig.resume_rebuild_task_states,
            )
        ),
        rejection_policy=(
            str(cfg.get("rejection_policy"))
            if cfg.get("rejection_policy") is not None
            else None
        ),
        rejection_target=(
            int(cfg.get("rejection_target"))
            if cfg.get("rejection_target") is not None
            else None
        ),
        rejection_score_threshold=(
            float(cfg.get("rejection_score_threshold"))
            if cfg.get("rejection_score_threshold") is not None
            else None
        ),
        rejection_reward_threshold=(
            float(cfg.get("rejection_reward_threshold"))
            if cfg.get("rejection_reward_threshold") is not None
            else None
        ),
        rejection_reference_scores_path=(
            str(cfg.get("rejection_reference_scores_path"))
            if cfg.get("rejection_reference_scores_path") is not None
            else None
        ),
        rejection_accepted_medals=(
            tuple(cfg.get("rejection_accepted_medals"))
            if cfg.get("rejection_accepted_medals") is not None
            else None
        ),
        rejection_apply_baseline_filters=bool(
            cfg.get(
                "rejection_apply_baseline_filters",
                SchedulerConfig.rejection_apply_baseline_filters,
            )
        ),
        rejection_baseline_token_limit=int(
            cfg.get(
                "rejection_baseline_token_limit",
                SchedulerConfig.rejection_baseline_token_limit,
            )
        ),
        rejection_baseline_tokenizer_model=(
            str(cfg.get("rejection_baseline_tokenizer_model"))
            if cfg.get("rejection_baseline_tokenizer_model") is not None
            else SchedulerConfig.rejection_baseline_tokenizer_model
        ),
        rejection_baseline_relative_gap_limit=float(
            cfg.get(
                "rejection_baseline_relative_gap_limit",
                SchedulerConfig.rejection_baseline_relative_gap_limit,
            )
        ),
        rejection_mixed_leaderboard_target=int(
            cfg.get(
                "rejection_mixed_leaderboard_target",
                SchedulerConfig.rejection_mixed_leaderboard_target,
            )
        ),
        rejection_mixed_no_leaderboard_target=int(
            cfg.get(
                "rejection_mixed_no_leaderboard_target",
                SchedulerConfig.rejection_mixed_no_leaderboard_target,
            )
        ),
        tree_search_enabled=bool(cfg.get("tree_search_enabled", False)),
        tree_search_algorithm=str(cfg.get("tree_search_algorithm", "greedy")),
        tree_search_max_pending_per_task=int(
            cfg.get("tree_search_max_pending_per_task", 1)
        ),
        tree_search_db_path=(
            str(cfg.get("tree_search_db_path"))
            if cfg.get("tree_search_db_path") is not None
            else None
        ),
        tree_search_db_max_per_task=(
            int(cfg.get("tree_search_db_max_per_task"))
            if cfg.get("tree_search_db_max_per_task") is not None
            else None
        ),
        greedy_num_drafts=int(cfg.get("greedy_num_drafts", 5)),
        greedy_debug_prob=float(cfg.get("greedy_debug_prob", 0.5)),
        greedy_draft_prob=float(cfg.get("greedy_draft_prob", 0.0)),
        gen_retry_max=(
            int(cfg.get("gen_retry_max"))
            if cfg.get("gen_retry_max") is not None
            else SchedulerConfig.gen_retry_max
        ),
    )

    # Create scheduler. Tree-search mode initializes its ProgramDatabase lazily.
    # 创建调度器实例（树搜索模式下 ProgramDatabase 会在需要时惰性初始化）
    scheduler = Scheduler(
        router=router,
        config=scheduler_config,
        database=None,
        output_dir=output_dir,  # Initialize gen/eval writers immediately
        litellm_model_list=cfg.litellm.model_list,
    )

    # Prepare task inputs (with optional limit)
    # 读取可选的任务数上限
    max_tasks = cfg.get("max_tasks", None)
    if max_tasks is not None:
        # 转成整数
        max_tasks = int(max_tasks)
        # 截取前 N 个任务
        tasks = tasks[:max_tasks]
        logger.info(f"Limited to first {max_tasks} tasks")

    # 是否启用 pass@k 草稿 prompt 构建器（在消息中注入时限等说明）
    use_draft_prompt_builder = bool(cfg.get("use_draft_prompt_builder", False))
    # 草稿 prompt 中是否写明时间限制
    draft_time_limit = bool(cfg.get("draft_time_limit", True))
    # 时间限制数值：未显式配置时回退到沙箱作业超时
    draft_time_limit_value = cfg.get(
        "draft_time_limit_value",
        cfg.sandbox.get("job_timeout", None) if "sandbox" in cfg else None,
    )
    # 记录草稿 prompt 相关开关
    logger.info(
        "use_draft_prompt_builder=%s draft_time_limit=%s draft_time_limit_value=%s",
        use_draft_prompt_builder,
        draft_time_limit,
        draft_time_limit_value,
    )

    # 组装调度器输入
    task_inputs: list[dict[str, Any]] = []
    for task in tasks:
        # 元数据列（含任务名/uuid/数据目录等）
        metadata = task[cfg.data.metadata_key]
        # 原始对话消息（pandas 存储的列表转回 Python list）
        raw_messages: list[dict[str, str]] = task[cfg.data.input_key].tolist()
        # 需要时用草稿构建器重写消息，否则沿用原始消息
        messages = (
            build_pass_k_draft_messages(
                metadata,
                raw_messages,
                time_limit=draft_time_limit,
                time_limit_value=draft_time_limit_value,
            )
            if use_draft_prompt_builder
            else raw_messages
        )
        # 取任务名
        task_name = metadata["task_name"]
        # 取任务 UUID
        task_id = metadata["uuid"]
        # 取任务数据目录
        data_dir = metadata["data_dir"]

        # 打包为调度器统一输入格式
        task_inputs.append(
            {
                "task_id": task_id,
                "task_name": task_name,
                "messages": messages,
                "metadata": metadata,
                "data_dir": data_dir,
            }
        )

    # Get model name
    # 取模型名（模型列表第一项）
    model_name = str(cfg.litellm.model_list[0].model_name)

    # 记录评测启动信息：k 值、任务数、已完成数与使用模型
    logger.info(f"Starting pass@{scheduler_config.max_steps} evaluation")
    logger.info(
        f"Total tasks: {len(tasks)}, Completed: {len(completed_tasks)}, Remaining: {len(task_inputs)}"
    )
    logger.info(f"Using model: {model_name}")

    # Progress file for external monitoring
    # 外部监控用的进度文件路径
    progress_file = output_dir / "progress.json"

    # Start progress monitor task
    # 周期性监控并记录进度的协程（生成/评测分开统计）
    async def monitor_progress():
        """Monitor and log progress every 30 seconds with separate gen/eval tracking."""
        while True:
            # 每 30 秒轮询一次进度
            await asyncio.sleep(30)
            # 向调度器查询当前进度
            progress = scheduler.get_progress()
            # 任务数为 0 说明尚未开始计数，跳过本轮
            if progress["total_tasks"] == 0:
                continue

            # 任务完成百分比
            task_pct = progress["completed_tasks"] / progress["total_tasks"] * 100

            # Get generation progress
            # 读取生成侧进度：总数/已完成/进行中/速率及百分比
            gen_progress = progress.get("generation", {})
            gen_total = gen_progress.get("total", 0)
            gen_completed = gen_progress.get("completed", 0)
            gen_in_progress = gen_progress.get("in_progress", 0)
            gen_rate = gen_progress.get("rate_per_second", 0)
            # 生成完成率（总数为 0 时记 0）
            gen_pct = gen_completed / gen_total * 100 if gen_total > 0 else 0

            # Get evaluation progress
            # 读取评测侧进度：总数/已完成/进行中/队列等待/速率及百分比
            eval_progress = progress.get("evaluation", {})
            eval_total = eval_progress.get("total", 0)
            eval_completed = eval_progress.get("completed", 0)
            eval_in_progress = eval_progress.get("in_progress", 0)
            eval_pending = eval_progress.get("pending_in_queue", 0)
            eval_rate = eval_progress.get("rate_per_second", 0)
            # 评测完成率（总数为 0 时记 0）
            eval_pct = eval_completed / eval_total * 100 if eval_total > 0 else 0

            logger.info(
                f"📊 Progress Update: "
                f"Tasks {progress['completed_tasks']}/{progress['total_tasks']} ({task_pct:.1f}%)"
            )
            logger.info(
                f"   Generation: {gen_completed}/{gen_total} ({gen_pct:.1f}%) | "
                f"In-progress: {gen_in_progress} | Rate: {gen_rate:.2f}/s"
            )
            if scheduler_config.enable_sandbox_eval:
                # 评测开启时额外输出沙箱槽位占用情况
                active_slots = eval_progress.get("active_sandbox_slots", 0)
                max_slots = eval_progress.get("sandbox_concurrency", 1)
                utilization = eval_progress.get("sandbox_utilization", 0)
                # 记录评测进度与队列等待
                logger.info(
                    f"   Evaluation: {eval_completed}/{eval_total} ({eval_pct:.1f}%) | "
                    f"In-progress: {eval_in_progress} | Queue: {eval_pending} | Rate: {eval_rate:.2f}/s"
                )
                # 记录沙箱并发利用率
                logger.info(
                    f"   Sandbox: {active_slots}/{max_slots} slots ({utilization:.0%} utilization)"
                )
            # 记录流式写出的样本数
            logger.info(
                f"   Streamed: {progress.get('streaming_samples_written', 0)} samples"
            )

            # Save progress to file for external monitoring
            # 把进度快照持久化到 progress.json（default=str 兜底不可序列化对象）
            progress_file.write_text(
                json.dumps(progress, indent=2, default=str), encoding="utf-8"
            )

            # Stop monitoring when all tasks are completed
            # 全部任务完成时退出监控循环
            if progress["completed_tasks"] >= progress["total_tasks"]:
                break

    # Check for evaluation-only mode
    # 读取“仅评测”模式开关（不重新生成，只评测已有 gen_results.jsonl）
    evaluate_only = bool(cfg.get("evaluate_only", False))

    if evaluate_only:
        # Evaluation-only mode: evaluate existing gen_results.jsonl
        # 打印仅评测模式的醒目分隔日志
        logger.info("=" * 60)
        logger.info("EVALUATION ONLY MODE")
        logger.info("=" * 60)

        # Build task list for metadata lookup
        # 构建仅含 id 与元数据的任务清单，供评测时查表
        all_task_inputs = []
        for task in tasks:
            metadata = task[cfg.data.metadata_key]
            task_id = metadata["uuid"]
            all_task_inputs.append(
                {
                    "task_id": task_id,
                    "metadata": metadata,
                }
            )

        # Start progress monitor for eval-only mode
        # 启动进度监控后台任务
        monitor_task = asyncio.create_task(monitor_progress())

        try:
            # 执行仅评测流程：评测所有未评测样本
            eval_results = await scheduler.run_evaluation_only(
                output_dir=output_dir,
                tasks=all_task_inputs,
            )

            logger.info(f"Evaluated {len(eval_results)} samples")
        finally:
            # Cancel monitor task
            # 无论成败都取消监控任务
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

        results = []  # No TaskResult objects in eval-only mode
    else:
        # Normal mode: run generation and evaluation
        # 正常模式：生成 + 评测一体的流水线
        results = []
        if task_inputs:  # Only run if there are tasks to process
            # 启动进度监控后台任务
            monitor_task = asyncio.create_task(monitor_progress())

            try:
                # 运行调度主流程（resume=True 启用断点续跑）
                results = await scheduler.run_tasks(
                    tasks=task_inputs,
                    model_name=model_name,
                    output_dir=output_dir,
                    resume=True,  # Enable checkpoint resume
                )
            finally:
                # Cancel monitor task
                # 无论成败都取消监控任务
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
        else:
            # 没有待处理任务（全部已完成）时直接跳过
            logger.info("All tasks already completed, skipping evaluation")

    # Aggregate results (merge new and resumed)
    # 汇总结果：任务名 -> 任务级统计（合并本次新跑与续跑恢复的数据）
    task_results_map: dict[str, dict[str, Any]] = {}

    # Add new results
    # 先写入本次运行的新结果
    for result in results:
        task_results_map[result.task_name] = {
            "task_name": result.task_name,
            "task_id": result.task_id,
            "best_score": result.best_score,
            "best_reward": result.best_reward,
            "num_samples": result.num_samples,
            "total_cost": result.total_cost,
            "total_tokens": result.total_tokens,
            "status_counts": result.status_counts,
            # 从状态计数中提取成功次数
            "success_count": result.status_counts.get("success", 0),
            "medal_count": result.medal_count,
            "medal_counts": result.medal_counts,
            "test_time": result.test_time,
        }

    # Merge resumed results
    # 再合并续跑恢复的结果：仅补充本次未覆盖的任务
    for task_name, resumed_stat in resumed_results.items():
        if task_name not in task_results_map:
            task_results_map[task_name] = resumed_stat
            logger.info(f"Merged resumed result for task: {task_name}")

    # Calculate global statistics
    # 汇总全部任务的费用
    total_cost = sum(r["total_cost"] for r in task_results_map.values())
    # 汇总全部任务的 token 用量
    total_tokens = sum(r["total_tokens"] for r in task_results_map.values())
    # 汇总全部任务的样本数
    total_samples = sum(r["num_samples"] for r in task_results_map.values())

    # 组装全局统计（k 值即 pass@k 中的 k）
    global_stats = {
        "experiment_name": str(cfg.experiment_name),
        "k": scheduler_config.max_steps,
        "num_tasks": len(task_results_map),
        "total_samples": total_samples,
        "total_cost": total_cost,
        "total_tokens": total_tokens,
        # 附加调度器内部统计
        "scheduler_stats": scheduler.get_stats(),
    }

    # Save global statistics
    # 写出全局统计文件 global_stat.json
    (output_dir / "global_stat.json").write_text(
        json.dumps(global_stats, indent=2, default=str), encoding="utf-8"
    )

    # Save detailed task results
    # 写出任务级明细文件 task_results.json
    (output_dir / "task_results.json").write_text(
        json.dumps(task_results_map, indent=2), encoding="utf-8"
    )

    return global_stats


@hydra.main(
    config_path=f"{TTS_SEARCH_DIR}/configs",
    config_name="experiment/parallel_glm47",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """CLI entry for pass@k generation/evaluation.

    Args:
        cfg: Hydra config resolved from ``tts_search/configs`` and CLI overrides.

    Returns:
        None. Writes generation, evaluation, progress, and summary files to disk.
    """
    # 日志级别统一设为 INFO
    logging.basicConfig(level=logging.INFO)
    # 按配置固定随机种子，保证可复现
    random.seed(cfg.seed)

    # 解析 MiniMax API key（非 MiniMax 配置返回空列表）
    api_keys = _parse_minimax_api_keys(cfg)
    if api_keys:
        # 有 key 时按 key 数量分片并发
        logger.info("Using %d MINIMAX API key(s)", len(api_keys))
    else:
        # Not MiniMax config, use single key from config
        # 非 MiniMax：占位 None，直接使用配置文件中的 key
        api_keys = [None]  # Will use key from config
        logger.info("Not MiniMax config, using API key from config")

    # 评测数据源路径
    data_path = Path(cfg.data.eval_data)
    # JSON 直接读取；否则按 parquet 读入并转成记录列表
    tasks = (
        json.loads(data_path.read_text(encoding="utf-8"))
        if data_path.suffix == ".json"
        else pd.read_parquet(data_path).to_dict(orient="records")
    )

    # 排行榜目录配置（未配置时为 None，回退到各任务数据目录）
    leaderboard_dir = (
        str(cfg.data.leaderboard_dir) if cfg.data.leaderboard_dir is not None else None
    )
    # 把排行榜目录写入每条任务的元数据，供后续加载排行榜用
    for task in tasks:
        metadata = task[cfg.data.metadata_key]
        metadata["leaderboard_dir"] = leaderboard_dir
    # 可选的 metricprompt 元数据 parquet（用于指标感知 gap 过滤）
    metric_metadata_parquet = cfg.data.get("metric_metadata_parquet", None)
    # 为任务元数据附加 self_valid_protocol 指标协议字段
    enriched_metric_count = enrich_tasks_with_metric_protocol(
        tasks,
        source_parquet=data_path if data_path.suffix != ".json" else None,
        metric_metadata_parquet=(
            Path(str(metric_metadata_parquet))
            if metric_metadata_parquet is not None
            else None
        ),
        metadata_key=str(cfg.data.metadata_key),
    )
    if enriched_metric_count:
        # 汇报成功附加协议字段的任务记录数
        logger.info(
            "Enriched %d task metadata records with self_valid_protocol",
            enriched_metric_count,
        )

    logger.info("Loaded %d tasks from %s", len(tasks), data_path)

    # Filter tasks by task list file if specified
    # 读取可选的任务过滤清单文件（每行一个任务名）
    task_filter_file = cfg.get("task_filter_file", None)
    if task_filter_file is not None:
        task_filter_path = Path(task_filter_file)
        if task_filter_path.exists():
            # 读入允许的任务名集合（忽略空行）
            with open(task_filter_path, "r") as f:
                allowed_task_names = {line.strip() for line in f if line.strip()}

            # 记录过滤前任务数
            original_count = len(tasks)
            # 只保留任务名在白名单里的任务
            tasks = [
                task
                for task in tasks
                if task[cfg.data.metadata_key]["task_name"] in allowed_task_names
            ]
            logger.info(
                f"Task filter applied: {original_count} -> {len(tasks)} tasks "
                f"(from {task_filter_path})"
            )
        else:
            # 过滤文件不存在时仅告警，继续全量任务
            logger.warning(f"Task filter file not found: {task_filter_path}")

    # Use resume_from as output_dir if specified
    # 配置了续跑目录时，把输出直接写回该目录；否则使用配置的输出目录
    resume_from = cfg.get("resume_from", None)
    if resume_from is not None:
        base_output_dir = Path(resume_from)
        logger.info(f"Using resume directory as output: {base_output_dir}")
    else:
        base_output_dir = Path(cfg.output_dir)

    # 确保输出目录存在
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    # 保存本次运行的完整配置快照到 config.yaml
    config_output = base_output_dir / "config.yaml"
    with open(config_output, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Run evaluation
    # 按 API key 数量分三种执行路径：配置内置 key / 单个 MiniMax key / 多 key 分片
    if len(api_keys) == 1 and api_keys[0] is None:
        # Not MiniMax, use router from config directly
        # 非 MiniMax：直接从配置实例化 Router
        router: Router = instantiate(cfg.litellm)
        output_dir = base_output_dir
        # 运行 pass@k 评测主流程
        global_stats = asyncio.run(
            run_pass_k_evaluation(
                cfg=cfg,
                router=router,
                tasks=tasks,
                output_dir=output_dir,
            )
        )
    elif len(api_keys) == 1:
        # 单个 MiniMax key：构建带该 key 的 Router
        router = _build_router_with_key(cfg, api_keys[0])
        output_dir = base_output_dir
        global_stats = asyncio.run(
            run_pass_k_evaluation(
                cfg=cfg,
                router=router,
                tasks=tasks,
                output_dir=output_dir,
            )
        )
    else:
        # 多 key 分片模式：任务轮询均分到各 key，独立并发跑
        task_shards = _split_tasks_evenly(tasks, len(api_keys))
        # 各分片的输出目录
        shard_dirs: list[Path] = []
        for idx, shard_tasks in enumerate(task_shards, start=1):
            # 创建 shard_XX_of_N 目录
            shard_dir = base_output_dir / f"shard_{idx:02d}_of_{len(api_keys)}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_dirs.append(shard_dir)
            # 记录分片分配情况
            logger.info(
                "Shard %d/%d: %d task(s) -> %s",
                idx,
                len(api_keys),
                len(shard_tasks),
                shard_dir,
            )

        # 每个 API key 分片运行一个 pass@k 评测协程，gather 并发执行
        async def run_sharded_evals() -> list[dict[str, Any]]:
            """Run one pass@k evaluation coroutine per API-key shard.

            Args:
                None; closes over shard tasks, shard dirs, API keys, and config.

            Returns:
                Per-shard global statistics returned by ``run_pass_k_evaluation``.
            """
            # 读取总沙箱并发数
            sandbox_concurrency = int(cfg.sandbox.concurrency)
            # 按分片数均分沙箱并发（至少每分片 1）
            per_shard_sandbox = max(1, sandbox_concurrency // len(api_keys))
            if per_shard_sandbox * len(api_keys) != sandbox_concurrency:
                # 除不尽时记录实际分配情况
                logger.info(
                    "Sandbox concurrency %d split across %d keys -> %d per shard",
                    sandbox_concurrency,
                    len(api_keys),
                    per_shard_sandbox,
                )
            # 收集各分片的评测协程
            coros = []
            # strict=True 保证 key/任务分片/目录三者数量一致
            for api_key, shard_tasks, shard_dir in zip(
                api_keys, task_shards, shard_dirs, strict=True
            ):
                # 为该分片构建专属 Router（注入其 API key）
                router = _build_router_with_key(cfg, api_key)
                # 深拷贝整体配置作为分片配置
                shard_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
                # 覆盖为分片专属沙箱并发
                shard_cfg.sandbox.concurrency = per_shard_sandbox
                if resume_from is not None:
                    # 续跑模式下每个分片只续跑自己目录里的数据
                    shard_cfg.resume_from = str(shard_dir)
                coros.append(
                    run_pass_k_evaluation(
                        cfg=shard_cfg,
                        router=router,
                        tasks=shard_tasks,
                        output_dir=shard_dir,
                    )
                )
            # 并发执行所有分片并收集各自的全局统计
            return await asyncio.gather(*coros)

        # 运行分片评测
        shard_stats = asyncio.run(run_sharded_evals())
        # 汇总各分片统计为一份合并全局统计
        global_stats = {
            "experiment_name": str(cfg.experiment_name),
            "k": int(cfg.max_steps),
            "num_tasks": sum(s.get("num_tasks", 0) for s in shard_stats),
            "total_samples": sum(s.get("total_samples", 0) for s in shard_stats),
            "total_cost": sum(s.get("total_cost", 0) for s in shard_stats),
            "total_tokens": sum(s.get("total_tokens", 0) for s in shard_stats),
            "avg_grade": None,
        }

        # 写出合并后的全局统计文件
        (base_output_dir / "global_stat_combined.json").write_text(
            json.dumps(global_stats, indent=2, default=str), encoding="utf-8"
        )
        # 输出目录指向总目录
        output_dir = base_output_dir

    # Log summary
    # 打印评测完成的汇总日志
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    # 以下一组日志汇总：任务数、总样本数、总费用与输出目录
    logger.info("=" * 60)
    logger.info(f"Tasks: {global_stats['num_tasks']}")
    logger.info(f"Total samples: {global_stats['total_samples']}")
    logger.info(f"Total cost: ${global_stats['total_cost']:.4f}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)


# 以 python -m tts_search.evaluate_pass_k 方式运行时进入主入口
if __name__ == "__main__":
    main()
