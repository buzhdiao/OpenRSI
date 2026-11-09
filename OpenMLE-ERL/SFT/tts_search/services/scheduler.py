"""
Scheduler - Coordinates generation and evaluation services.

This scheduler manages the workflow between the generator and evaluator
services, similar to how SGLang coordinates prefill and decode operations.
It supports pass@k evaluation where k samples are generated and all
evaluated on the test set.

Key Features:
- Decoupled generation and evaluation pipelines
- Generation runs independently without waiting for evaluation
- Evaluation is optional and consumes from async queue
- Separate progress tracking for generation and evaluation
- Streaming output to JSONL files for durability
"""
# 中文说明:调度器模块——协调"生成服务"与"评估服务"的总控中枢。
# 在系统中的位置:OpenMLE-ERL/SFT/tts_search 服务层核心,类似 SGLang 协调
# prefill/decode 的角色:生成管线独立跑(生产者),评估管线从异步队列消费(消费者),
# 两条管线解耦并发,互不阻塞;支持 pass@k(每任务生成 k 份样本全部送测)
# 与循环式(loop)评估(每轮目标 x 个成功/奖牌,不达标按增量 y 继续生成)。
# 关键设计:
# - 进度(生成/评估)分开追踪,支持实时快照 progress.json + 趋势图;
# - 全部结果流式追加写 JSONL(gen_results/eval_results),断点续跑依赖 checkpoint 重放;
# - 可选树搜索(greedy)模式:生成时按搜索算法选父程序,评估后回填程序库;
# - 拒绝策略(rejection policy)决定样本是否计入"接受数",达标即停。

# 延迟注解:类型标注运行期不求值
from __future__ import annotations

# asyncio:异步队列/任务/事件/锁(整条管线的并发骨架)
import asyncio
# hashlib:代码 SHA256(搜索事件记录用)
import hashlib
# json:checkpoint/进度/结果记录的序列化
import json
# logging:模块日志
import logging
# asynccontextmanager:空评估器上下文(关闭沙箱评估时用)
from contextlib import asynccontextmanager
# dataclass/field:配置与进度结构声明
from dataclasses import dataclass, field
# datetime/timedelta/timezone:时间戳、耗时估算与 UTC 归一
from datetime import datetime, timedelta, timezone
# Path:输出目录/检查点文件路径
from pathlib import Path
# 类型注解(含回调签名)
from typing import Any, Awaitable, Callable, Mapping

# LiteLLM 路由器:模型请求的统一入口(多模型调度)
from litellm.router import Router

# 可选依赖:matplotlib 用于进度趋势图(无头环境强制 Agg 后端)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 标记 matplotlib 是否可用(不可用时跳过画图)
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

# 评估汇总工具(排行榜/奖牌重建)
from tts_search import eval_utils
# token 统计工具:加载分词器并按 chat 模板计数(SFT 基线过滤用)
from tts_search.data_produce.token_filter import (
    count_chat_template_tokens,
    load_tokenizer,
)
# 贪心搜索算法与其配置(树搜索模式)
from tts_search.greedy import AlgoConfig, GreedySearch
# 程序库与程序记录(树搜索持久化)
from tts_search.program_database import Program, ProgramDatabase
# 评估服务:请求/结果类型与服务实现
from tts_search.services.evaluator import (
    EvaluationRequest,
    EvaluationResult,
    EvaluatorService,
)
# 生成循环控制器:按"目标数-已生成数"推进每轮批量
from tts_search.services.generation_loop import (
    GenerationLoopConfig,
    GenerationLoopController,
)
# 生成服务:请求/结果类型与服务实现
from tts_search.services.generator import (
    GenerationRequest,
    GenerationResult,
    GeneratorService,
)
# 拒绝策略:奖牌标签、策略构建(决定样本是否被"接受")
from tts_search.services.rejection import (
    MEDAL_LABELS,
    RejectionPolicy,
    build_rejection_policy,
)
# 流式 JSONL 写入器:结果逐条落盘(断点续跑基础)
from tts_search.services.result_persistence import StreamingJSONLWriter
# 树搜索事件/状态:节点生成/评估事件与任务搜索状态快照
from tts_search.services.tree_search_state import (
    NODE_EVALUATED,
    NODE_GENERATED,
    SearchEvent,
    SearchState,
    append_search_event,
    atomic_write_search_state,
)

# 模块 logger
logger = logging.getLogger(__name__)


@dataclass
class SchedulerConfig:
    """调度器配置:生成/评估/管线/循环/树搜索/进度历史等全部开关与参数。"""

    # ===== 生成设置 =====
    # 搜索最大步数
    max_steps: int = 64
    # 每任务采样数(pass@k 的 k)
    n_samples_per_task: int = 1
    # LLM 请求并发
    llm_concurrency: int = 64
    # 单步生成最大重试次数
    llm_max_retries_per_step: int = 10

    # ===== 评估设置 =====
    # 沙箱 GPU 服务地址
    sandbox_base_url: str = ""
    # 沙箱 CPU 服务地址
    sandbox_cpu_base_url: str = ""
    # 强制覆盖资源类型(可选)
    sandbox_resource_type_override: str | None = None
    # 沙箱并发(在飞评估上限)
    sandbox_concurrency: int = 32
    # 单作业超时(秒)
    job_timeout: int = 3600
    # 等待作业结果总超时(秒)
    wait_timeout: int = 86400
    # 作业状态轮询间隔(秒)
    poll_interval: int = 5
    # 启用 score->reward 映射
    use_score2reward: bool = True

    # ===== 管线设置 =====
    # 评估收集批量大小
    batch_size: int = 16  # Batch size for evaluation collection
    # 并发任务数
    task_concurrency: int = 1  # Number of tasks to run concurrently
    # 是否启用沙箱评估(关闭则只生成不评估)
    enable_sandbox_eval: bool = True  # Enable/disable sandbox evaluation
    # 任务完成后跳过其在队列中的评估
    skip_eval_for_done_tasks: bool = False  # Skip queued evals after a task is done

    # ===== 解耦管线设置 =====
    # 评估队列容量(0 = 不限)
    eval_queue_maxsize: int = 0  # 0 = unlimited queue size
    # 收集评估批次的超时(秒)
    eval_batch_timeout: float = 1.0  # Timeout for collecting evaluation batch

    # ===== 任务输出设置 =====
    # 任务输出的可选子目录
    task_subdir: str | None = None  # Optional subdirectory for per-task outputs

    # ===== 循环式任务评估设置 =====
    # 是否启用循环评估
    loop_enabled: bool = False
    # 每轮所需成功数 x
    loop_success_target: int = 1  # x: required success count
    # 成功口径:success(成功数)或 medal(奖牌数)
    loop_success_metric: str = "success"  # success or medal
    # 所需奖牌数(缺省取成功目标)
    loop_medal_target: int | None = None  # required medal count; defaults to success target
    # 每轮目标增量 y
    loop_target_increment: int = 1  # y: target increment per round
    # 目标上限 z(达到即停)
    loop_max_target: int | None = None  # z: stop when target reaches this
    # 续跑时忽略 progress 快照、从结果记录重建任务状态
    resume_rebuild_task_states: bool = False
    # 拒绝策略名(缺省取成功口径)
    rejection_policy: str | None = None
    # 接受目标数(达标即停)
    rejection_target: int | None = None
    # 拒绝策略:分数阈值
    rejection_score_threshold: float | None = None
    # 拒绝策略:奖励阈值
    rejection_reward_threshold: float | None = None
    # 参考分数文件(相对差距过滤)
    rejection_reference_scores_path: str | None = None
    # 可接受的奖牌集合
    rejection_accepted_medals: tuple[str, ...] | None = None
    # 是否应用基线过滤(token 数/相对差距)
    rejection_apply_baseline_filters: bool = True
    # 基线 token 上限
    rejection_baseline_token_limit: int = 32768
    # 基线分词器模型(统计消息 token 数)
    rejection_baseline_tokenizer_model: str | None = None
    # 基线相对差距上限
    rejection_baseline_relative_gap_limit: float = 0.12
    # mixed 策略:有排行榜任务的接受目标
    rejection_mixed_leaderboard_target: int = 2
    # mixed 策略:无排行榜任务的接受目标
    rejection_mixed_no_leaderboard_target: int = 4

    # ===== 树搜索生成设置(默认关闭以保持 pass@k 语义) =====
    # 是否启用树搜索
    tree_search_enabled: bool = False
    # 搜索算法(目前仅 greedy)
    tree_search_algorithm: str = "greedy"
    # 每任务在飞生成上限(v1 固定 1)
    tree_search_max_pending_per_task: int = 1
    # 程序库 SQLite 路径(缺省落在输出目录)
    tree_search_db_path: str | None = None
    # 每任务保留程序数(缺省不剪枝)
    tree_search_db_max_per_task: int | None = None
    # greedy:每次草稿数
    greedy_num_drafts: int = 5
    # greedy:debug 模式概率
    greedy_debug_prob: float = 0.5
    # greedy:draft 模式概率
    greedy_draft_prob: float = 0.0

    # 生成重试:失败后最多重试 gen_retry_max 次(None = 无限)
    gen_retry_max: int | None = 10

    # 进度历史:每 N 秒记录生成/评估统计与完成任务数,写 JSONL + 趋势图
    progress_history_interval_seconds: float = 600.0  # 10 minutes
    # 趋势图宽度范围(英寸):超上限时横向压缩显示完整时间跨度
    progress_trend_fig_width_min: float = 10.0  # minimum figure width (inches)
    progress_trend_fig_width_max: float = 40.0  # maximum figure width; beyond this, x-axis is compressed
    # 每小时时间跨度对应的图宽(英寸,直到触顶)
    progress_trend_inches_per_hour: float = 2.0  # width per hour of time span (until hitting max)


@dataclass
class GenerationProgress:
    """生成管线的进度追踪。"""

    # 目标总数
    total: int = 0
    # 成功数
    success: int = 0
    # 进行中数
    in_progress: int = 0
    # 失败数
    failed: int = 0
    # 开始时间
    start_time: datetime | None = None
    # 结束时间
    end_time: datetime | None = None

    @property
    def pending(self) -> int:
        """待生成数 = 总数 - 成功 - 进行中 - 失败。"""
        return self.total - self.success - self.in_progress - self.failed


@dataclass
class EvaluationProgress:
    """评估管线的进度追踪。"""

    # 目标总数
    total: int = 0
    # 成功数
    success: int = 0
    # 进行中数
    in_progress: int = 0
    # 仍在队列中等待的样本数
    pending_in_queue: int = 0
    # 失败数
    failed: int = 0
    # 当前占用的沙箱评估槽位数
    active_sandbox_slots: int = 0  # Number of sandbox evaluations currently running
    # 开始时间
    start_time: datetime | None = None
    # 结束时间
    end_time: datetime | None = None


@dataclass
class TaskResult:
    """单个任务在 pass@k 评估后的最终结果。"""

    # 任务 ID
    task_id: str
    # 任务名
    task_name: str
    # 全部样本(Program 列表)
    samples: list[Program]
    # 最优程序(按奖励)
    best_program: Program
    # 最优分数
    best_score: float | None
    # 最优奖励
    best_reward: float
    # 总费用
    total_cost: float
    # 总 token 数
    total_tokens: int
    # 样本数
    num_samples: int
    # 状态计数
    status_counts: dict[str, int]
    # 奖牌总数
    medal_count: int
    # 各类奖牌计数
    medal_counts: dict[str, int]
    # 验证阶段耗时
    val_time: float
    # 测试阶段耗时
    test_time: float
    # 结果生成时间戳
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TaskLoopState:
    """循环评估下的单任务状态(目标推进与达标判定)。"""

    # 任务 ID
    task_id: str
    # 任务名
    task_name: str
    # 成功数
    success_count: int = 0
    # 奖牌数
    medal_count: int = 0
    # 通过拒绝策略的接受数
    accepted_count: int = 0
    # 已完成(生成或评估闭环)数
    completed_count: int = 0
    # 已生成数
    generated_count: int = 0
    # 在飞生成数
    pending_gen: int = 0
    # 当前轮目标数
    target_count: int = 0
    # 目标上限(可选)
    max_target: int | None = None
    # 接受目标数(可选)
    accepted_target: int | None = None
    # 下一个 step 序号
    next_step_index: int = 0
    # 是否已完成
    done: bool = False


def load_checkpoint(
    output_dir: Path,
) -> tuple[set[str], set[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """从输出目录加载断点续跑状态(读 gen/eval 两个 JSONL 文件)。

    Returns:
        (已完成生成的 request_id 集合, 已完成评估的 request_id 集合,
         生成结果记录列表, 评估结果记录列表)
    """
    # 已完成生成的请求 ID 集合
    completed_gen_ids: set[str] = set()
    # 已完成评估的请求 ID 集合
    completed_eval_ids: set[str] = set()
    # 生成结果记录列表
    gen_results_data: list[dict[str, Any]] = []
    # 评估结果记录列表
    eval_results_data: list[dict[str, Any]] = []

    # 生成结果文件与评估结果文件路径
    gen_file = output_dir / "gen_results.jsonl"
    eval_file = output_dir / "eval_results.jsonl"

    # 读取生成结果(逐行 JSON;坏行跳过)
    if gen_file.exists():
        with open(gen_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        request_id = data.get("request_id")
                        if request_id:
                            completed_gen_ids.add(request_id)
                            gen_results_data.append(data)
                    except json.JSONDecodeError:
                        continue

    # 读取评估结果(同时收集 ID 与完整记录)
    if eval_file.exists():
        with open(eval_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                request_id = data.get("request_id")
                if request_id:
                    completed_eval_ids.add(request_id)
                eval_results_data.append(data)

    # request_id -> 生成记录(用于给评估记录回填缺失字段)
    gen_by_request = {
        str(data.get("request_id")): data
        for data in gen_results_data
        if data.get("request_id") is not None
    }
    # 遍历评估记录,回填生成侧信息
    for data in eval_results_data:
        request_id = data.get("request_id")
        gen_data = gen_by_request.get(str(request_id)) if request_id is not None else None
        if gen_data is not None:
            # 三组 token 数:评估记录缺省时从生成记录回填
            for gen_key, eval_key in (
                ("prompt_tokens", "generation_prompt_tokens"),
                ("completion_tokens", "generation_completion_tokens"),
                ("total_tokens", "generation_total_tokens"),
            ):
                if data.get(eval_key) is None and gen_data.get(gen_key) is not None:
                    data[eval_key] = gen_data[gen_key]
            # 任务名/ID/步序号:评估记录优先,生成记录兜底
            task_name = data.get("task_name") or gen_data.get("task_name")
            task_id = data.get("task_id") or gen_data.get("task_id")
            step_index = data.get("step_index", gen_data.get("step_index"))
        else:
            # 无对应生成记录:只用评估记录自身字段
            task_name = data.get("task_name")
            task_id = data.get("task_id")
            step_index = data.get("step_index")

        # feedback 路径缺失时按候选路径重建(兼容新旧目录布局)
        if data.get("feedback_path") is None and task_name and step_index is not None:
            safe_name = safe_task_output_name(str(task_name), task_id)
            candidates = [
                output_dir / "tasks" / safe_name / f"step_{step_index}" / "feedback.txt",
                output_dir / safe_name / f"step_{step_index}" / "feedback.txt",
            ]
            for candidate in candidates:
                if candidate.exists():
                    data["feedback_path"] = str(candidate)
                    break

    # 返回四元组
    return completed_gen_ids, completed_eval_ids, gen_results_data, eval_results_data


def load_progress_snapshot(output_dir: Path) -> dict[str, Any] | None:
    """加载 progress.json 快照(若存在)用于续跑;坏文件返回 None。"""
    # 快照路径
    progress_path = output_dir / "progress.json"
    # 不存在则无快照
    if not progress_path.exists():
        return None
    try:
        # 读入并反序列化
        return json.loads(progress_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # 解析失败:视为无快照(续跑会走重建路径)
        return None


def safe_task_output_name(task_name: str, task_id: str | None) -> str:
    """生成任务输出目录名:任务名+任务 ID,保证重名任务目录唯一且无路径分隔符。"""
    # 任务名清洗(斜杠替换)
    safe_task_name = str(task_name).replace("/", "_").replace("\\", "_")
    # 任务 ID 清洗(缺失记 missing_task_id)
    safe_task_id = str(task_id or "missing_task_id").replace("/", "_").replace("\\", "_")
    # 拼接为唯一目录名
    return f"{safe_task_name}_{safe_task_id}"


def _eval_result_has_medal(
    data: dict[str, Any], metadata: dict[str, Any] | None = None
) -> bool:
    """判断一条评估记录是否拿到奖牌:优先看记录内 submit_medal,缺失时按排行榜重建。"""
    # 先看记录里是否已带奖牌字段
    submit_medal = data.get("submit_medal")
    if isinstance(submit_medal, str) and submit_medal.lower() in MEDAL_LABELS:
        return True

    # 无元数据或无分数:无法重建,判无奖牌
    if metadata is None or data.get("score") is None:
        return False
    try:
        # 按元数据加载排行榜
        leaderboard = eval_utils.load_leaderboard(metadata)
        if leaderboard is None:
            return False
        # 按分数重算奖牌
        medal = eval_utils.get_medal_for_score(float(data["score"]), leaderboard)
    except Exception as exc:
        # 重建失败:告警并判无奖牌(不阻断续跑)
        logger.warning(
            "Failed to rebuild medal status for task=%s score=%s: %s",
            data.get("task_name") or data.get("task_id"),
            data.get("score"),
            exc,
        )
        return False
    # 重算结果在奖牌集合内即有奖牌
    return medal in MEDAL_LABELS


def _loop_target_met(
    state: TaskLoopState,
    rejection_target: int,
) -> bool:
    """判断任务是否达标:接受数 >= 目标。"""
    return state.accepted_count >= rejection_target


def progress_snapshot_has_medal_counts(progress_snapshot: dict[str, Any] | None) -> bool:
    """检查快照中每个任务状态都带 medal 字段(决定快照能否用于 medal 口径续跑)。"""
    # 取快照中的任务状态字典
    snapshot_states = (progress_snapshot or {}).get("task_states") or {}
    # 非空且全部带 medal 才算可用
    return bool(snapshot_states) and all(
        "medal" in state_data for state_data in snapshot_states.values()
    )


def build_task_states_from_progress(
    progress_snapshot: dict[str, Any] | None,
    tasks: list[dict[str, Any]],
    loop_increment: int,
    loop_max_target: int | None,
    rejection_policy_name: str = "accept_scored",
) -> dict[str, TaskLoopState]:
    """优先从 progress.json 快照重建任务状态;无快照返回空字典。"""
    # 无快照直接返回空
    if not progress_snapshot:
        return {}
    # 取快照里的任务状态
    snapshot_states = progress_snapshot.get("task_states") or {}
    if not snapshot_states:
        return {}

    # 重建的任务状态表
    task_states: dict[str, TaskLoopState] = {}
    # 遍历任务清单,逐个恢复状态
    for task in tasks:
        task_id = task["task_id"]
        task_name = task["task_name"]
        # 快照中的该任务状态数据(缺省为空)
        state_data = snapshot_states.get(task_id, {})
        # 基础状态:目标与上限
        state = TaskLoopState(
            task_id=task_id,
            task_name=task_name,
            max_target=state_data.get("max_target", loop_max_target),
            target_count=int(state_data.get("target", loop_increment)),
        )
        # 恢复成功数与奖牌数
        state.success_count = int(state_data.get("success", 0))
        state.medal_count = int(state_data.get("medal", 0))
        # 接受数:快照里有就用;否则按口径从奖牌数/成功数推
        if "accepted" in state_data:
            state.accepted_count = int(state_data["accepted"])
        elif rejection_policy_name in {"medal", "has_medal"}:
            state.accepted_count = state.medal_count
        else:
            state.accepted_count = state.success_count
        # 恢复完成数与生成数
        state.completed_count = int(state_data.get("completed", 0))
        state.generated_count = int(state_data.get("generated", 0))
        # 恢复接受目标
        if state_data.get("accepted_target") is not None:
            state.accepted_target = int(state_data["accepted_target"])
        # 恢复完成标记
        state.done = bool(state_data.get("done", False))
        # 写入状态表
        task_states[task_id] = state
    # 返回状态表
    return task_states


def rebuild_task_states_from_results(
    tasks: list[dict[str, Any]],
    gen_results_data: list[dict[str, Any]],
    eval_results_data: list[dict[str, Any]],
    loop_increment: int,
    loop_success_target: int,
    loop_success_metric: str = "success",
    loop_medal_target: int | None = None,
    loop_max_target: int | None = None,
    rejection_policy: RejectionPolicy | None = None,
    rejection_target: int | None = None,
    accepted_targets: Mapping[str, int] | None = None,
    count_successful_generation_as_completed: bool = False,
) -> dict[str, TaskLoopState]:
    """从生成/评估结果记录重建任务状态(无快照时的续跑路径)。"""
    # 拒绝策略:外部传入或按成功口径构建
    policy = rejection_policy or build_rejection_policy(name=loop_success_metric)
    # 接受目标:显式 rejection_target > medal 口径的奖牌目标 > 成功目标
    target_accept_count = int(
        rejection_target
        if rejection_target is not None
        else (
            loop_medal_target
            if loop_success_metric == "medal" and loop_medal_target is not None
            else loop_success_target
        )
    )
    # 任务状态表
    task_states: dict[str, TaskLoopState] = {}
    # 任务 ID -> 元数据(奖牌重建用)
    task_metadata_map: dict[str, dict[str, Any]] = {}
    # 先为每个任务建初始状态
    for task in tasks:
        task_states[task["task_id"]] = TaskLoopState(
            task_id=task["task_id"],
            task_name=task["task_name"],
            max_target=loop_max_target,
            target_count=loop_increment,
            accepted_target=(
                int(accepted_targets[task["task_id"]])
                if accepted_targets and task["task_id"] in accepted_targets
                else target_accept_count
            ),
        )
        # 记录元数据
        if task.get("metadata") is not None:
            task_metadata_map[task["task_id"]] = task["metadata"]

    # 第一遍:从生成记录恢复(步序号/生成数/完成数)
    for data in gen_results_data:
        task_id = data.get("task_id")
        # 非本批任务跳过
        if task_id not in task_states:
            continue
        state = task_states[task_id]
        # 步序号推进到最大已见 +1
        step_index = int(data.get("step_index", 0))
        if step_index >= state.next_step_index:
            state.next_step_index = step_index + 1
        # 生成数 +1
        state.generated_count += 1
        # 关闭沙箱评估时成功生成即算完成;失败生成总是算完成
        if count_successful_generation_as_completed or not data.get("success", True):
            state.completed_count += 1

    # 第二遍:从评估记录恢复(完成/成功/奖牌/接受)
    for data in eval_results_data:
        task_id = data.get("task_id")
        if task_id not in task_states:
            continue
        state = task_states[task_id]
        # 完成 +1
        state.completed_count += 1
        # 成功判定
        if data.get("success", False):
            state.success_count += 1
        # 奖牌判定(必要时重建)
        if _eval_result_has_medal(data, task_metadata_map.get(task_id)):
            state.medal_count += 1
        # 拒绝策略判定
        if policy.accepts_record(data, task_metadata_map.get(task_id)).accepted:
            state.accepted_count += 1

    # 第三遍:按目标推进逻辑确定 done 与下一轮 target
    for state in task_states.values():
        # 任务级接受目标(显式覆盖优先)
        state_accept_target = (
            int(accepted_targets[state.task_id])
            if accepted_targets and state.task_id in accepted_targets
            else target_accept_count
        )
        state.accepted_target = state_accept_target
        # 已达标:标记完成,目标定格在当前完成数
        if _loop_target_met(state, state_accept_target):
            state.done = True
            state.target_count = max(state.completed_count, loop_increment)
            continue
        # 未达标:按口径推算当前轮目标
        if loop_success_metric == "medal":
            # medal 口径:按已完成轮数推(每轮 loop_increment 个)
            completed_rounds = max(
                1,
                (
                    max(state.generated_count, state.completed_count)
                    + loop_increment
                    - 1
                )
                // loop_increment,
            )
            target = loop_increment * completed_rounds
        else:
            # success 口径:按生成数推下一轮
            target = loop_increment * (state.generated_count // loop_increment + 1)
        # 目标不超过上限
        if loop_max_target is not None:
            target = min(target, loop_max_target)
        state.target_count = max(target, loop_increment)
        # 循环推进:完成数已覆盖目标时尝试升级目标或判完成
        while state.completed_count >= state.target_count:
            # 已达标或目标已到上限:结束
            if _loop_target_met(
                state,
                state_accept_target,
            ) or (
                loop_max_target is not None and state.target_count >= loop_max_target
            ):
                state.done = True
                break
            # success 口径不再自动加轮,直接退出
            if loop_success_metric != "medal":
                break
            # medal 口径:目标加一个增量
            next_target = state.target_count + loop_increment
            if loop_max_target is not None:
                next_target = min(next_target, loop_max_target)
            # 增量无法推进(被上限卡住):判完成
            if next_target <= state.target_count:
                state.done = True
                break
            # 推进目标继续下一轮
            state.target_count = next_target

    # 返回重建的状态表
    return task_states


class Scheduler:
    """生成与评估的协调调度器。

    实现解耦管线架构:
    - 生成器独立产出代码样本(生产者)
    - 评估器从队列消费并打分(消费者)
    - 两条管线通过异步队列并发运行
    - 各自独立追踪进度
    """

    def __init__(
        self,
        router: Router,
        config: SchedulerConfig,
        database: ProgramDatabase | None = None,
        output_dir: Path | str | None = None,
        litellm_model_list: list[Any] | None = None,
    ):
        """初始化调度器。

        Args:
            router: LiteLLM 路由器(模型请求入口)
            config: 调度器配置
            database: 可选的程序数据库(树搜索持久化)
            output_dir: 可选的输出目录(gen/eval 结果文件位置)
        """
        # 保存路由器
        self._router = router
        # 保存配置
        self._config = config
        # 程序数据库(可选)
        self._database = database
        # 输出目录(可选)
        self._output_dir = Path(output_dir) if output_dir else None

        # 初始化生成服务(并发/重试/模型列表透传)
        self._generator = GeneratorService(
            router=router,
            concurrency=config.llm_concurrency,
            max_retries=config.llm_max_retries_per_step,
            model_list=litellm_model_list,
        )

        # 评估器在 run_tasks 中以上下文管理器方式创建
        self._evaluator: EvaluatorService | None = None

        # 解耦管线的评估队列(None 为终止哨兵)
        self._eval_queue: asyncio.Queue[GenerationResult | None] = asyncio.Queue(
            maxsize=config.eval_queue_maxsize
        )

        # 进度追踪(生成/评估各自独立)
        self._gen_progress = GenerationProgress()
        self._eval_progress = EvaluationProgress()
        # 进度锁(保护进度统计的并发更新)
        self._progress_lock = asyncio.Lock()

        # 任务级追踪
        self._total_tasks = 0
        self._completed_tasks = 0
        # 任务状态表(loop 模式)
        self._task_states: dict[str, TaskLoopState] | None = None

        # 开始/结束时间
        self._start_time: datetime | None = None
        self._end_time: datetime | None = None

        # 生成结果缓存(评估后用于构建 Program)
        self._gen_results: dict[str, GenerationResult] = {}
        self._eval_results: dict[str, EvaluationResult] = {}
        # 结果锁(保护结果字典并发访问)
        self._results_lock = asyncio.Lock()
        # 评估结果钩子(loop 模式下挂接状态推进)
        self._eval_result_hook: Callable[[EvaluationResult], Awaitable[None]] | None = (
            None
        )
        # 评估记录元数据钩子(写 JSONL 时补搜索元数据)
        self._eval_record_metadata_hook: Callable[[EvaluationResult], dict[str, Any]] | None = None
        # 基线分词器(懒加载,SFT 基线 token 过滤用)
        self._baseline_tokenizer: Any | None = None

        # 流式写入器(在 run_tasks 里懒初始化)
        self._gen_writer: StreamingJSONLWriter | None = None
        self._eval_writer: StreamingJSONLWriter | None = None
        # 搜索算法实例(树搜索关闭时为 None)
        self._search_algorithm = None
        # 启用树搜索时的初始化与参数校验
        if self._config.tree_search_enabled:
            # 算法名统一小写
            algorithm = str(self._config.tree_search_algorithm).lower()
            # v1 只支持 greedy
            if algorithm != "greedy":
                raise ValueError(
                    "Scheduler tree_search currently supports only algorithm='greedy'"
                )
            # 每任务在飞上限必须为 1(保证父程序已评估)
            if int(self._config.tree_search_max_pending_per_task) != 1:
                raise ValueError(
                    "tree_search_max_pending_per_task must be 1 in the v1 scheduler"
                )
            # 未显式传数据库时:按配置路径或输出目录默认路径创建 SQLite 程序库
            if self._database is None:
                db_path = (
                    Path(self._config.tree_search_db_path)
                    if self._config.tree_search_db_path
                    else (self._output_dir or Path(".")) / "tree_search_programs.db"
                )
                self._database = ProgramDatabase(
                    str(db_path),
                    max_per_task=self._config.tree_search_db_max_per_task,
                )
            # 构建贪心搜索算法
            self._search_algorithm = GreedySearch(
                AlgoConfig(
                    num_drafts=self._config.greedy_num_drafts,
                    debug_prob=self._config.greedy_debug_prob,
                    draft_prob=self._config.greedy_draft_prob,
                )
            )

    async def run_evaluation_only(
        self,
        output_dir: Path,
        tasks: list[dict[str, Any]] | None = None,
    ) -> list[EvaluationResult]:
        """仅评估模式:对已有生成结果(gen_results.jsonl)补跑评估。

        找出未评估的成功样本,用 fire-and-forget 模式(与正常评估一致)送沙箱。

        适用场景:
        1. 纯评估模式(生成早已单独完成)
        2. 评估被中断后的续跑

        Args:
            output_dir: 含 gen_results.jsonl 的目录
            tasks: 可选任务清单(用于元数据查询)

        Returns:
            本次新产生的评估结果列表
        """
        # 读进度快照(用于恢复耗时起点)
        progress_snapshot = load_progress_snapshot(output_dir)
        # 记录开始时间
        self._start_time = datetime.now()

        # 初始化任务级追踪:统计任务清单中的去重任务数
        unique_task_names = set()
        if tasks:
            unique_task_names = {
                task.get("metadata", {}).get("task_name") for task in tasks
            }
            # 去掉 None(个别任务缺 task_name)
            unique_task_names.discard(None)
        # 任务总数(清单为空则 0)
        self._total_tasks = len(unique_task_names) if unique_task_names else 0
        # 完成数在评估结束后统计
        self._completed_tasks = 0  # Will be updated as evaluation completes

        # 加载检查点
        completed_gen_ids, completed_eval_ids, gen_results_data, _ = load_checkpoint(
            output_dir
        )

        # 找出未评估的成功样本(失败生成不补评)
        unevaluated = [
            data
            for data in gen_results_data
            if data["request_id"] not in completed_eval_ids
            and data.get("success", True)
        ]

        # 全部已评估:直接返回空列表
        if not unevaluated:
            logger.info("No unevaluated samples found")
            return []

        # 打印待评估/已评估统计
        logger.info(
            f"Found {len(unevaluated)} unevaluated samples out of {len(gen_results_data)} total"
        )
        logger.info(f"Already evaluated: {len(completed_eval_ids)} samples")

        # 按任务 ID 建元数据映射(奖牌重建/目录定位用)
        task_metadata_map: dict[str, dict[str, Any]] = {}
        if tasks:
            for task in tasks:
                metadata = task.get("metadata", {})
                # 优先用元数据里的 uuid
                task_id = metadata.get("uuid") or task.get("task_id")
                if task_id:
                    task_metadata_map[task_id] = metadata

        # 初始化评估进度:总数=全部生成数,已完成=已评估数;耗时按快照回推起点
        eval_elapsed = (
            progress_snapshot.get("evaluation", {}).get("elapsed_seconds")
            if progress_snapshot
            else None
        )
        self._eval_progress = EvaluationProgress(
            total=len(gen_results_data),
            success=len(completed_eval_ids),
            start_time=datetime.now() - timedelta(seconds=float(eval_elapsed)),
        )

        # 生成进度按"全部已完成"初始化(生成早已结束)
        self._gen_progress = GenerationProgress()
        self._gen_progress.total = len(gen_results_data)
        self._gen_progress.success = len(gen_results_data)
        self._gen_progress.start_time = self._start_time
        self._gen_progress.end_time = self._start_time

        # 评估写入器以追加模式打开(续跑不覆盖旧记录)
        self._eval_writer = StreamingJSONLWriter(
            output_dir / "eval_results.jsonl", append=True
        )

        # 构建评估器上下文(配置透传)
        evaluator_ctx = EvaluatorService(
            base_url=self._config.sandbox_base_url,
            cpu_base_url=self._config.sandbox_cpu_base_url,
            resource_type_override=self._config.sandbox_resource_type_override,
            concurrency=self._config.sandbox_concurrency,
            job_timeout=self._config.job_timeout,
            wait_timeout=self._config.wait_timeout,
            poll_interval=self._config.poll_interval,
        )

        # 汇总结果容器
        all_eval_results: list[EvaluationResult] = []

        # 持有评估器上下文执行整段评估
        async with evaluator_ctx as evaluator:
            self._evaluator = evaluator

            # 把全部待评估样本入队
            logger.info(f"Queueing {len(unevaluated)} samples for evaluation")
            for data in unevaluated:
                task_id = data["task_id"]
                task_name = data["task_name"]
                # 该任务的元数据(缺省空)
                metadata = task_metadata_map.get(task_id, {})

                # 构建任务输出目录(与 run_evaluation 保持一致)
                base_task_root = (
                    output_dir / self._config.task_subdir
                    if getattr(self._config, "task_subdir", None)
                    else output_dir
                )
                base_task_root.mkdir(parents=True, exist_ok=True)
                task_output_dir = base_task_root / safe_task_output_name(task_name, task_id)
                task_output_dir.mkdir(parents=True, exist_ok=True)

                # 从记录重建 GenerationResult(评估管线只关心这些字段)
                gen_result = GenerationResult(
                    request_id=data["request_id"],
                    task_id=task_id,
                    task_name=task_name,
                    code=data.get("code", ""),
                    raw_text=data.get("raw_text", ""),
                    reasoning_content=data.get("reasoning_content"),
                    prompt_tokens=data.get("prompt_tokens", 0),
                    completion_tokens=data.get("completion_tokens", 0),
                    total_tokens=data.get("total_tokens", 0),
                    cost=data.get("cost", 0.0),
                    model_name=data.get("model_name", ""),
                    messages=[],
                    metadata=metadata,
                    step_index=data.get("step_index", 0),
                    data_dir=metadata.get("data_dir", ""),
                    output_dir=task_output_dir,
                )
                # 入评估队列
                await self._eval_queue.put(gen_result)
                async with self._progress_lock:
                    self._eval_progress.pending_in_queue += 1

            # 队列终止哨兵
            await self._eval_queue.put(None)

            # 以 fire-and-forget 模式跑评估循环
            # 纯评估模式无需在内存保留结果(store_results=False)
            all_eval_results = await self._run_eval_loop(
                output_dir, store_results=False
            )

        # 记录结束时间并释放评估器
        self._end_time = datetime.now()
        self._evaluator = None

        # 更新完成任务数:统计"全部样本都已评估"的任务
        task_samples_count: dict[str, tuple[int, int]] = (
            {}
        )  # task_name -> (total, evaluated)
        # 第一遍:统计每个任务的样本总数
        for data in gen_results_data:
            task_name = data.get("task_name")
            if task_name:
                if task_name not in task_samples_count:
                    task_samples_count[task_name] = (0, 0)
                total, evaluated = task_samples_count[task_name]
                task_samples_count[task_name] = (total + 1, evaluated)

        # 第二遍:统计每个任务已评估样本数(含本次新评估的)
        all_completed_eval_ids = completed_eval_ids.union(
            {r.request_id for r in all_eval_results}
        )
        for request_id in all_completed_eval_ids:
            for data in gen_results_data:
                if data.get("request_id") == request_id:
                    task_name = data.get("task_name")
                    if task_name and task_name in task_samples_count:
                        total, evaluated = task_samples_count[task_name]
                        task_samples_count[task_name] = (total, evaluated + 1)
                    break

        # 完成任务数 = 样本全部评估完的任务个数
        self._completed_tasks = sum(
            1 for total, evaluated in task_samples_count.values() if total == evaluated
        )

        # 打印最终统计
        logger.info(
            f"Evaluation only completed: {len(all_eval_results)} samples evaluated"
        )
        logger.info(
            f"Tasks with all samples evaluated: {self._completed_tasks}/{self._total_tasks}"
        )

        # 返回本次新评估结果
        return all_eval_results

    async def run_tasks(
        self,
        tasks: list[dict[str, Any]],
        model_name: str,
        output_dir: Path | None = None,
        resume: bool = True,
    ) -> list[TaskResult]:
        """对多个任务执行循环式评估(pass@k 的循环扩展)。

        Args:
            tasks: 任务配置列表
            model_name: 生成所用模型
            output_dir: 产物输出目录(None 时用 __init__ 里的)
            resume: True 时从检查点续跑(跳过已完成的生成/评估)

        Returns:
            TaskResult 列表
        """
        # 输出目录:显式传入优先,否则用初始化时的;两者都没有则报错
        if output_dir is None:
            if self._output_dir is None:
                raise ValueError(
                    "output_dir must be provided either in __init__ or run_tasks"
                )
            output_dir = self._output_dir
        else:
            output_dir = Path(output_dir)

        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)

        # 记录开始时间
        self._start_time = datetime.now()

        # loop 模式走循环评估;未启用则报错(v1 只有 loop 入口)
        if self._config.loop_enabled:
            return await self._run_looping_tasks(
                tasks=tasks,
                model_name=model_name,
                output_dir=output_dir,
                resume=resume,
            )
        else:
            raise ValueError("Loop is not enabled")

    async def _run_looping_tasks(
        self,
        tasks: list[dict[str, Any]],
        model_name: str,
        output_dir: Path,
        resume: bool = True,
    ) -> list[TaskResult]:
        """循环评估主实现:按任务维护成功/目标计数,不达标持续生成-评估。"""
        # 目标增量必须为正
        if self._config.loop_target_increment <= 0:
            raise ValueError("loop_target_increment must be > 0")

        # 每轮批量(目标增量)
        loop_increment = int(self._config.loop_target_increment)
        # 树搜索模式:每轮固定 1 个(保证父程序链)
        if self._tree_search_enabled():
            if int(self._config.tree_search_max_pending_per_task) != 1:
                raise ValueError(
                    "tree_search_max_pending_per_task must be 1 in the v1 scheduler"
                )
            loop_increment = 1
        # 成功目标数 x
        loop_success_target = int(self._config.loop_success_target)
        # 成功口径(success/medal)
        loop_success_metric = str(self._config.loop_success_metric).lower()
        if loop_success_metric not in {"success", "medal"}:
            raise ValueError("loop_success_metric must be 'success' or 'medal'")
        # 奖牌目标(缺省取成功目标)
        loop_medal_target = (
            int(self._config.loop_medal_target)
            if self._config.loop_medal_target is not None
            else loop_success_target
        )
        # 拒绝策略名(缺省取成功口径)
        rejection_policy_name = str(
            self._config.rejection_policy or loop_success_metric
        ).lower()
        # 构建拒绝策略(阈值/参考分数/奖牌集合/基线过滤全部透传)
        rejection_policy = build_rejection_policy(
            name=rejection_policy_name,
            score_threshold=self._config.rejection_score_threshold,
            reward_threshold=self._config.rejection_reward_threshold,
            reference_scores_path=self._config.rejection_reference_scores_path,
            accepted_medals=list(self._config.rejection_accepted_medals or ()),
            apply_baseline_filters=(
                self._config.rejection_apply_baseline_filters
                and rejection_policy_name not in {"all", "accept_all", "none"}
            ),
            baseline_token_limit=self._config.rejection_baseline_token_limit,
            baseline_relative_gap_limit=(
                self._config.rejection_baseline_relative_gap_limit
            ),
            mixed_leaderboard_target=self._config.rejection_mixed_leaderboard_target,
            mixed_no_leaderboard_target=(
                self._config.rejection_mixed_no_leaderboard_target
            ),
        )
        # 接受目标:显式配置 > medal 口径的奖牌目标 > 成功目标
        rejection_target = int(
            self._config.rejection_target
            if self._config.rejection_target is not None
            else (
                loop_medal_target
                if rejection_policy_name in {"medal", "has_medal"}
                else loop_success_target
            )
        )
        # 目标上限(可选)
        loop_max_target = (
            int(self._config.loop_max_target)
            if self._config.loop_max_target is not None
            else None
        )
        # 构建生成循环控制器(批量=增量,目标=接受数,上限=最大生成)
        loop_controller = GenerationLoopController(
            GenerationLoopConfig(
                batch_size=loop_increment,
                accepted_target=rejection_target,
                max_generated=loop_max_target,
            )
        )

        # 续跑时尝试加载进度快照
        progress_snapshot = load_progress_snapshot(output_dir) if resume else None
        # 树搜索续跑不信任快照(状态由程序库重建),忽略之
        if progress_snapshot is not None and self._tree_search_enabled():
            logger.info(
                "Ignoring progress.json for tree-search resume; rebuilding from gen/eval records"
            )
            progress_snapshot = None
        # 显式要求重建时同样忽略快照
        if progress_snapshot is not None and self._config.resume_rebuild_task_states:
            logger.info(
                "Ignoring progress.json and rebuilding task states from gen/eval "
                "results because resume_rebuild_task_states=true"
            )
            progress_snapshot = None
        # 任务总数与完成数初始化
        self._total_tasks = len(tasks)
        self._completed_tasks = 0

        # 加载检查点数据
        completed_eval_ids: set[str] = set()
        gen_results_data: list[dict[str, Any]] = []
        eval_results_data: list[dict[str, Any]] = []
        if resume:
            _, completed_eval_ids, gen_results_data, eval_results_data = (
                load_checkpoint(output_dir)
            )

        # 任务 ID -> 元数据映射
        task_metadata_map: dict[str, dict[str, Any]] = {}
        for task in tasks:
            task_metadata_map[task["task_id"]] = task["metadata"]
        # 计算任务级接受目标(mixed 策略会按任务元数据差异化)
        accepted_targets = self._build_accepted_targets(
            tasks=tasks,
            policy=rejection_policy,
            default_target=rejection_target,
        )

        # 初始化任务状态:优先快照,失败/不可用则从结果记录重建
        task_states = build_task_states_from_progress(
            progress_snapshot=progress_snapshot,
            tasks=tasks,
            loop_increment=loop_increment,
            loop_max_target=loop_max_target,
            rejection_policy_name=rejection_policy_name,
        )
        # medal 口径但快照缺 medal 计数:快照不完整,弃用
        if (
            rejection_policy_name in {"medal", "has_medal"}
            and task_states
            and not progress_snapshot_has_medal_counts(progress_snapshot)
        ):
            task_states = {}
        # 无快照:从 gen/eval 记录重建
        if not task_states:
            task_states = rebuild_task_states_from_results(
                tasks=tasks,
                gen_results_data=gen_results_data,
                eval_results_data=eval_results_data,
                loop_increment=loop_increment,
                loop_success_target=loop_success_target,
                loop_success_metric=loop_success_metric,
                loop_medal_target=loop_medal_target,
                loop_max_target=loop_max_target,
                rejection_policy=rejection_policy,
                rejection_target=rejection_target,
                accepted_targets=accepted_targets,
                count_successful_generation_as_completed=not self._config.enable_sandbox_eval,
            )
        else:
            # 有快照:仍需把步序号推进到检查点之后(避免覆盖旧生成)
            for data in gen_results_data:
                task_id = data.get("task_id")
                if task_id not in task_states:
                    continue
                state = task_states[task_id]
                step_index = int(data.get("step_index", 0))
                if step_index >= state.next_step_index:
                    state.next_step_index = step_index + 1

        # 应用任务级接受目标覆盖
        for state in task_states.values():
            if state.task_id in accepted_targets:
                state.accepted_target = accepted_targets[state.task_id]

        # 挂载任务状态表并统计初始完成数
        self._task_states = task_states
        self._completed_tasks = sum(1 for s in task_states.values() if s.done)

        # 从检查点/快照初始化两管线的进度与耗时起点
        gen_elapsed = (
            progress_snapshot.get("generation", {}).get("elapsed_seconds")
            if progress_snapshot
            else None
        )
        eval_elapsed = (
            progress_snapshot.get("evaluation", {}).get("elapsed_seconds")
            if progress_snapshot
            else None
        )
        # 生成成功/失败统计
        gen_success = sum(1 for data in gen_results_data if data.get("success", True))
        gen_failed = len(gen_results_data) - gen_success
        # 评估成功/失败统计(生成失败也计入评估侧失败,保持闭环)
        eval_success = sum(
            1 for data in eval_results_data if data.get("success", False)
        )
        eval_failed = len(eval_results_data) - eval_success + gen_failed

        # 重建生成进度(耗时起点按快照回推)
        self._gen_progress = GenerationProgress(
            total=0,
            success=gen_success,
            failed=gen_failed,
            start_time=(
                datetime.now() - timedelta(seconds=float(gen_elapsed))
                if gen_elapsed
                else datetime.now()
            ),
        )
        # 重建评估进度
        self._eval_progress = EvaluationProgress(
            total=0,
            success=eval_success,
            failed=eval_failed,
            start_time=(
                datetime.now() - timedelta(seconds=float(eval_elapsed))
                if eval_elapsed
                else datetime.now()
            ),
        )
        # 按任务目标刷新两管线的 total
        self._update_loop_progress_totals(task_states)

        # 建立流式输出文件(续跑时追加)
        self._gen_writer = StreamingJSONLWriter(
            output_dir / "gen_results.jsonl", append=resume
        )
        self._eval_writer = StreamingJSONLWriter(
            output_dir / "eval_results.jsonl", append=resume
        )

        # 检查点里"已生成未评估"的样本(续跑时补评估)
        pending_eval_data = [
            data
            for data in gen_results_data
            if self._config.enable_sandbox_eval
            and data.get("success", True)
            and data.get("request_id") not in completed_eval_ids
            and data.get("task_id") in task_states
        ]

        # 评估器上下文:启用时用真实评估器;关闭时用空上下文(yield None)
        if self._config.enable_sandbox_eval:
            evaluator_ctx = EvaluatorService(
                base_url=self._config.sandbox_base_url,
                cpu_base_url=self._config.sandbox_cpu_base_url,
                resource_type_override=self._config.sandbox_resource_type_override,
                concurrency=self._config.sandbox_concurrency,
                job_timeout=self._config.job_timeout,
                wait_timeout=self._config.wait_timeout,
                poll_interval=self._config.poll_interval,
            )
        else:
            @asynccontextmanager
            async def null_evaluator_ctx():
                yield None
            evaluator_ctx = null_evaluator_ctx()

        # 全量生成/评估结果收集
        all_gen_results: list[GenerationResult] = []
        all_eval_results: list[EvaluationResult] = []
        # "全接受"类策略:无需评估即可判定接受
        accept_without_eval = rejection_policy_name in {"all", "accept_all", "none"}
        # 接受判定缓存:request_id -> (accepted, reason)
        acceptance_decisions: dict[str, tuple[bool, str]] = {}

        def _decision_for_eval(eval_result: EvaluationResult) -> tuple[bool, str]:
            # 带缓存的接受判定(同一请求只判一次)
            cached = acceptance_decisions.get(eval_result.request_id)
            if cached is not None:
                return cached
            # 补齐 SFT 基线 token 计数(策略可能需要)
            self._attach_slime_message_token_count(eval_result)
            # 交给拒绝策略判定
            decision = rejection_policy.accepts_result(eval_result)
            cached = (bool(decision.accepted), str(decision.reason))
            acceptance_decisions[eval_result.request_id] = cached
            return cached

        def _eval_record_search_metadata(eval_result: EvaluationResult) -> dict[str, Any]:
            # 为评估记录补搜索元数据(写 JSONL 时合并)
            gen_result = self._gen_results.get(eval_result.request_id)
            metadata = dict(gen_result.search_metadata or {}) if gen_result else {}
            if self._tree_search_enabled():
                # 树搜索:补接受判定与父程序链信息
                accepted, reason = _decision_for_eval(eval_result)
                metadata.update(
                    {
                        "search_algorithm": metadata.get("search_algorithm", "greedy"),
                        "search_step": metadata.get("search_step", eval_result.step_index),
                        "program_id": metadata.get("program_id", eval_result.request_id),
                        "parent_id": metadata.get("parent_id"),
                        "secondary_parent_id": (
                            metadata.get("parent_ids", [None, None])[1]
                            if len(metadata.get("parent_ids", [])) > 1
                            else None
                        ),
                        "parent_ids": metadata.get("parent_ids", []),
                        "generation_mode": metadata.get("generation_mode", "draft"),
                        # 适应度=奖励(评估后才有)
                        "fitness": eval_result.reward,
                        "accepted_by_rejection_policy": accepted,
                        "rejection_reason": reason,
                        "code_path": self._step_code_path(
                            eval_result.output_dir, eval_result.step_index
                        ),
                    }
                )
            return metadata

        # 进入评估器上下文,启动生成/评估双管线
        async with evaluator_ctx as evaluator:
            self._evaluator = evaluator
            # 任务状态锁(保护 task_states 并发更新)
            task_state_lock = asyncio.Lock()
            # 全部完成事件(用于协调退出)
            done_event = asyncio.Event()
            # 生成请求队列(None 为终止哨兵)
            gen_queue: asyncio.Queue[GenerationRequest | None] = asyncio.Queue()

            # 任务 ID -> 任务运行时信息映射
            task_info_map: dict[str, dict[str, Any]] = {}
            for task in tasks:
                task_id = task["task_id"]
                task_name = task["task_name"]
                # 任务输出根目录(支持 task_subdir 子目录)
                base_task_root = (
                    output_dir / self._config.task_subdir
                    if getattr(self._config, "task_subdir", None)
                    else output_dir
                )
                base_task_root.mkdir(parents=True, exist_ok=True)
                task_output_dir = base_task_root / safe_task_output_name(task_name, task_id)
                task_output_dir.mkdir(parents=True, exist_ok=True)
                # 原始消息列表
                messages = task["messages"]
                # 提取 system 消息(树搜索重建 prompt 时要用)
                system_messages = [
                    message
                    for message in messages
                    if str(message.get("role", "")).lower() == "system"
                ]
                # 组装任务运行时信息
                task_info_map[task_id] = {
                    "task_name": task_name,
                    "messages": messages,
                    "system_message": (
                        system_messages[0]
                        if system_messages
                        else {"role": "system", "content": ""}
                    ),
                    "description": (
                        str(messages[-1].get("content", "")) if messages else ""
                    ),
                    "metadata": task["metadata"],
                    "data_dir": task["data_dir"],
                    "output_dir": task_output_dir,
                }

            # 树搜索模式:从检查点记录预加载程序库(父程序链可续)
            self._preload_tree_database_from_checkpoint(
                gen_results_data=gen_results_data,
                eval_results_data=eval_results_data,
                task_metadata_map=task_metadata_map,
            )

            # request_id -> 已重试次数(0 = 首次尝试)
            gen_retry_counts: dict[str, int] = {}
            # 生成重试上限(None = 无限重试)
            gen_retry_max = getattr(
                self._config, "gen_retry_max", 3
            )  # None = infinite retry

            async def _refresh_completion_locked() -> None:
                # 持锁刷新完成任务数与进度总量;全部完成时点亮 done_event
                self._completed_tasks = sum(1 for s in task_states.values() if s.done)
                self._update_loop_progress_totals(task_states)
                if self._completed_tasks >= self._total_tasks:
                    done_event.set()

            async def _create_requests(
                task_id: str, count: int
            ) -> list[GenerationRequest]:
                """为任务创建 count 个生成请求(含树搜索父程序选择)。"""
                # 数量非正直接返回
                if count <= 0:
                    return []
                # 持任务状态锁做目标核算
                async with task_state_lock:
                    state = task_states[task_id]
                    # 任务已完成:不再生成
                    if state.done:
                        return []
                    # 还需要生成数 = 目标 - 已生成 - 在飞
                    needed = max(
                        0,
                        state.target_count - state.generated_count - state.pending_gen,
                    )
                    count = min(count, needed)
                    if count <= 0:
                        return []
                    # 预订在飞名额
                    state.pending_gen += count

                    info = task_info_map[task_id]
                    requests: list[GenerationRequest] = []
                    for _ in range(count):
                        # 请求 ID 与步序号
                        request_id = f"{task_id}_{state.next_step_index}"
                        step_index = state.next_step_index
                        state.next_step_index += 1
                        # 默认用任务原始消息与主模型
                        request_messages = info["messages"]
                        request_model_name = model_name
                        search_metadata: dict[str, Any] = {}
                        # 树搜索模式:由搜索算法决定 prompt/父程序/模式/模型
                        if self._tree_search_enabled():
                            assert self._database is not None
                            prompt, parent_program, mode, selected_model = (
                                self._search_algorithm.select(
                                    database=self._database,
                                    task_name=info["task_name"],
                                    description=info["description"],
                                    data_dir=info["data_dir"],
                                    max_steps=self._config.max_steps,
                                )
                            )
                            # 父程序信息提取
                            parent_program_id = None
                            parent_ids: list[str] = []
                            parent_db_id = None
                            parent_code = ""
                            if parent_program is not None:
                                parent_db_id = parent_program.id
                                parent_program_id = str(
                                    parent_program.metadata.get("program_id")
                                    or parent_program.id
                                )
                                parent_ids = [parent_program_id]
                                parent_code = parent_program.code
                            # 用系统消息 + 搜索算法构造的用户 prompt 替换原始消息
                            request_messages = [
                                dict(info["system_message"]),
                                {"role": "user", "content": str(prompt or "")},
                            ]
                            # 搜索算法可指定不同模型
                            request_model_name = selected_model or model_name
                            # 记录搜索谱系元数据
                            search_metadata = {
                                "search_algorithm": str(
                                    self._config.tree_search_algorithm
                                ).lower(),
                                "search_step": step_index,
                                "program_id": request_id,
                                "parent_id": parent_program_id,
                                "parent_ids": parent_ids,
                                "parent_db_id": parent_db_id,
                                "parent_code": parent_code,
                                "generation_mode": mode,
                            }
                        # 组装生成请求
                        requests.append(
                            GenerationRequest(
                                request_id=request_id,
                                task_id=task_id,
                                task_name=info["task_name"],
                                messages=request_messages,
                                model_name=request_model_name,
                                data_dir=info["data_dir"],
                                metadata=info["metadata"],
                                step_index=step_index,
                                output_dir=info["output_dir"],
                                search_metadata=search_metadata,
                            )
                        )
                    return requests

            async def _enqueue_requests(task_id: str, count: int) -> None:
                # 创建请求并逐个入生成队列
                requests = await _create_requests(task_id, count)
                for req in requests:
                    await gen_queue.put(req)

            async def _on_gen_result(gen_result: GenerationResult) -> None:
                """生成完成回调:更新任务状态与进度,记录搜索事件,必要时追加生成。"""
                all_gen_results.append(gen_result)
                # 需要追加的生成数
                to_enqueue = 0
                # 事件记录用状态快照
                task_state_for_event: TaskLoopState | None = None
                async with task_state_lock:
                    state = task_states.get(gen_result.task_id)
                    if state is None:
                        return
                    # 生成数 +1,在飞 -1
                    state.generated_count += 1
                    state.pending_gen = max(0, state.pending_gen - 1)
                    # 关闭沙箱评估或生成失败:直接算完成(无评估闭环)
                    if not self._config.enable_sandbox_eval or not gen_result.success:
                        state.completed_count += 1
                        if accept_without_eval:
                            state.accepted_count += 1
                    # 循环控制器决定是否追加生成
                    to_enqueue = loop_controller.advance_after_completion(state)
                    task_state_for_event = state
                    await _refresh_completion_locked()

                # 树搜索:记录"节点生成"事件(谱系与代码指纹)
                if (
                    self._tree_search_enabled()
                    and gen_result.success
                    and task_state_for_event is not None
                    and gen_result.output_dir is not None
                ):
                    search_metadata = dict(gen_result.search_metadata or {})
                    event_log_path, _ = self._task_search_paths(gen_result.output_dir)
                    append_search_event(
                        event_log_path,
                        SearchEvent(
                            event_type=NODE_GENERATED,
                            task_id=gen_result.task_id,
                            task_name=gen_result.task_name,
                            search_algorithm=str(
                                search_metadata.get("search_algorithm", "greedy")
                            ),
                            search_step=int(
                                search_metadata.get(
                                    "search_step", gen_result.step_index
                                )
                            ),
                            program_id=str(
                                search_metadata.get(
                                    "program_id", gen_result.request_id
                                )
                            ),
                            parent_ids=[
                                str(parent_id)
                                for parent_id in search_metadata.get("parent_ids", [])
                            ],
                            generation_mode=str(
                                search_metadata.get("generation_mode", "draft")
                            ),
                            code_path=self._step_code_path(
                                gen_result.output_dir, gen_result.step_index
                            ),
                            code_sha256=self._code_sha256(gen_result.code),
                            generated_count=task_state_for_event.generated_count,
                            stop_requested=task_state_for_event.done,
                        ),
                    )

                # 需要追加生成则入队
                if to_enqueue > 0:
                    await _enqueue_requests(gen_result.task_id, to_enqueue)

            async def _on_eval_result(eval_result: EvaluationResult) -> None:
                """评估完成回调:更新任务状态、回填程序库、记录搜索事件、推进目标。"""
                all_eval_results.append(eval_result)
                to_enqueue = 0
                task_state_for_event: TaskLoopState | None = None
                # 接受判定(带缓存)
                accepted, rejection_reason = _decision_for_eval(eval_result)
                # 对应的生成结果(构建 Program 用)
                gen_result_for_eval = self._gen_results.get(eval_result.request_id)
                async with task_state_lock:
                    state = task_states.get(eval_result.task_id)
                    if state is None:
                        return
                    # 完成 +1
                    state.completed_count += 1
                    # 成功判定
                    if eval_result.success:
                        state.success_count += 1
                    # 奖牌判定
                    if str(eval_result.submit_medal).lower() in MEDAL_LABELS:
                        state.medal_count += 1
                    # 接受判定
                    if accepted:
                        state.accepted_count += 1
                    # 树搜索:把评估后的程序写入程序库(谱系+分数)
                    if (
                        self._tree_search_enabled()
                        and self._database is not None
                        and gen_result_for_eval is not None
                    ):
                        program = self._build_tree_program_from_eval(
                            gen_result=gen_result_for_eval,
                            eval_result=eval_result,
                            accepted=accepted,
                            rejection_reason=rejection_reason,
                        )
                        db_program_id = self._database.add(program)
                        program.id = db_program_id
                        program.metadata["db_program_id"] = db_program_id
                    # 循环控制器决定是否追加生成
                    to_enqueue = loop_controller.advance_after_completion(state)
                    task_state_for_event = state
                    await _refresh_completion_locked()

                # 树搜索:记录"节点评估"事件并快照搜索状态
                if (
                    self._tree_search_enabled()
                    and gen_result_for_eval is not None
                    and eval_result.output_dir is not None
                    and task_state_for_event is not None
                ):
                    search_metadata = dict(gen_result_for_eval.search_metadata or {})
                    event_log_path, _ = self._task_search_paths(eval_result.output_dir)
                    append_search_event(
                        event_log_path,
                        SearchEvent(
                            event_type=NODE_EVALUATED,
                            task_id=eval_result.task_id,
                            task_name=eval_result.task_name,
                            search_algorithm=str(
                                search_metadata.get("search_algorithm", "greedy")
                            ),
                            search_step=int(
                                search_metadata.get(
                                    "search_step", eval_result.step_index
                                )
                            ),
                            program_id=str(
                                search_metadata.get(
                                    "program_id", eval_result.request_id
                                )
                            ),
                            parent_ids=[
                                str(parent_id)
                                for parent_id in search_metadata.get("parent_ids", [])
                            ],
                            generation_mode=str(
                                search_metadata.get("generation_mode", "draft")
                            ),
                            code_path=self._step_code_path(
                                eval_result.output_dir, eval_result.step_index
                            ),
                            code_sha256=self._code_sha256(eval_result.code),
                            score=eval_result.score,
                            reward=eval_result.reward,
                            fitness=eval_result.reward,
                            feedback_path=self._relative_to_task_dir(
                                (
                                    eval_result.output_dir
                                    / f"step_{eval_result.step_index}"
                                    / "feedback.txt"
                                ),
                                eval_result.output_dir,
                            ),
                            raw_log_path=self._relative_to_task_dir(
                                (
                                    eval_result.output_dir
                                    / f"step_{eval_result.step_index}"
                                    / "raw_run_log.txt"
                                ),
                                eval_result.output_dir,
                            ),
                            accepted_by_rejection_policy=accepted,
                            rejection_reason=rejection_reason,
                            generated_count=task_state_for_event.generated_count,
                            completed_count=task_state_for_event.completed_count,
                            accepted_count=task_state_for_event.accepted_count,
                            stop_requested=task_state_for_event.done,
                        ),
                    )
                    # 原子写出任务搜索状态快照(可恢复)
                    self._write_tree_search_state(
                        task_state=task_state_for_event,
                        task_output_dir=eval_result.output_dir,
                        accepted_target=task_state_for_event.accepted_target,
                    )

                # 需要追加生成则入队
                if to_enqueue > 0:
                    await _enqueue_requests(eval_result.task_id, to_enqueue)

            # 挂接评估结果钩子与记录元数据钩子
            self._eval_result_hook = _on_eval_result
            self._eval_record_metadata_hook = _eval_record_search_metadata

            # 把检查点里未评估的样本补入评估队列
            if pending_eval_data:
                for data in pending_eval_data:
                    task_id = data["task_id"]
                    task_name = data["task_name"]
                    metadata = task_metadata_map.get(task_id, {})
                    # 任务输出目录
                    base_task_root = (
                        output_dir / self._config.task_subdir
                        if getattr(self._config, "task_subdir", None)
                        else output_dir
                    )
                    base_task_root.mkdir(parents=True, exist_ok=True)
                    task_output_dir = base_task_root / safe_task_output_name(task_name, task_id)
                    task_output_dir.mkdir(parents=True, exist_ok=True)

                    # 从记录重建 GenerationResult(保留搜索谱系元数据)
                    gen_result = GenerationResult(
                        request_id=data["request_id"],
                        task_id=task_id,
                        task_name=task_name,
                        code=data.get("code", ""),
                        raw_text=data.get("raw_text", ""),
                        reasoning_content=data.get("reasoning_content"),
                        prompt_tokens=data.get("prompt_tokens", 0),
                        completion_tokens=data.get("completion_tokens", 0),
                        total_tokens=data.get("total_tokens", 0),
                        cost=data.get("cost", 0.0),
                        model_name=data.get("model_name", ""),
                        messages=[],
                        metadata=metadata,
                        step_index=data.get("step_index", 0),
                        data_dir=data.get("data_dir", metadata.get("data_dir", "")),
                        output_dir=task_output_dir,
                        search_metadata={
                            key: data[key]
                            for key in (
                                "search_algorithm",
                                "search_step",
                                "program_id",
                                "parent_id",
                                "parent_ids",
                                "parent_db_id",
                                "generation_mode",
                            )
                            if key in data
                        },
                    )
                    # 缓存生成结果并补入评估队列
                    self._gen_results[gen_result.request_id] = gen_result
                    await self._eval_queue.put(gen_result)
                    async with self._progress_lock:
                        self._eval_progress.pending_in_queue += 1

            # 为每个任务投放首轮生成请求(目标-已生成-在飞)
            for task_id, state in task_states.items():
                needed = max(
                    0, state.target_count - state.generated_count - state.pending_gen
                )
                if needed > 0:
                    await _enqueue_requests(task_id, needed)

            async def _gen_worker() -> None:
                """生成工作协程:从队列取请求 -> 调生成服务 -> 失败重试 -> 落盘/入评估队列。"""
                while True:
                    # 取请求(哨兵则退出)
                    req = await gen_queue.get()
                    if req is None:
                        gen_queue.task_done()
                        break
                    # 生成前:标记"进行中"(锁获取 5 秒超时防死锁,超时仅告警不阻塞生成)
                    try:
                        await asyncio.wait_for(
                            self._progress_lock.acquire(), timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        logger.error(
                            "progress_lock timeout before generate: request_id=%s task_id=%s step=%d locked=%s",
                            req.request_id,
                            req.task_id,
                            req.step_index,
                            self._progress_lock.locked(),
                        )
                    else:
                        try:
                            self._gen_progress.in_progress += 1
                        finally:
                            self._progress_lock.release()
                    # 真正调用生成服务
                    gen_result = await self._generator.generate(req)
                    logger.info(f"Generation completed: {gen_result.request_id}")

                    # 失败时决定是否重试
                    will_retry = False
                    if not gen_result.success:
                        retries_done = gen_retry_counts.get(gen_result.request_id, 0)
                        if gen_retry_max is None or retries_done < gen_retry_max:
                            will_retry = True
                            gen_retry_counts[gen_result.request_id] = retries_done + 1
                            logger.info(
                                "Generation failed, retrying (%s/%s): request_id=%s",
                                retries_done + 1,
                                "inf" if gen_retry_max is None else gen_retry_max,
                                gen_result.request_id,
                            )

                    # 生成后:更新进度统计(同样带 5 秒锁超时保护)
                    try:
                        await asyncio.wait_for(
                            self._progress_lock.acquire(), timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        logger.error(
                            "progress_lock timeout after generate: request_id=%s task_id=%s step=%d locked=%s",
                            gen_result.request_id,
                            gen_result.task_id,
                            gen_result.step_index,
                            self._progress_lock.locked(),
                        )
                    else:
                        try:
                            # 进行中 -1
                            self._gen_progress.in_progress -= 1
                            # 不重试时才计入成功/失败(重试算回到队列)
                            if not will_retry:
                                if gen_result.success:
                                    self._gen_progress.success += 1
                                else:
                                    self._gen_progress.failed += 1
                                    # 生成失败意味着该样本不会进入评估,评估侧同样 +1
                                    self._eval_progress.failed += 1
                        except Exception:
                            logger.exception(
                                "Progress update failed: request_id=%s task_id=%s step=%d gen_progress=%s eval_progress=%s",
                                gen_result.request_id,
                                gen_result.task_id,
                                gen_result.step_index,
                                self._gen_progress,
                                self._eval_progress,
                            )
                        finally:
                            self._progress_lock.release()

                    # 缓存生成结果(评估回调/结果组装要用)
                    async with self._results_lock:
                        self._gen_results[gen_result.request_id] = gen_result

                    if will_retry:
                        # 中文说明:重试走"同一逻辑请求重新入队"——不写盘、不触发 _on_gen_result,
                        # 保证检查点里只出现最终成功/放弃的记录。
                        info = task_info_map[gen_result.task_id]
                        retry_req = GenerationRequest(
                            request_id=gen_result.request_id,
                            task_id=gen_result.task_id,
                            task_name=info["task_name"],
                            messages=gen_result.messages,
                            model_name=gen_result.model_name,
                            data_dir=info["data_dir"],
                            metadata=info["metadata"],
                            step_index=gen_result.step_index,
                            output_dir=info["output_dir"],
                            search_metadata=dict(gen_result.search_metadata or {}),
                        )
                        await gen_queue.put(retry_req)
                    else:
                        # 最终结果:流式写盘(含搜索元数据合并)
                        if self._gen_writer:
                            gen_record = {
                                    "request_id": gen_result.request_id,
                                    "task_id": gen_result.task_id,
                                    "task_name": gen_result.task_name,
                                    "code": gen_result.code,
                                    "raw_text": gen_result.raw_text,
                                    "reasoning_content": gen_result.reasoning_content,
                                    "prompt_tokens": gen_result.prompt_tokens,
                                    "completion_tokens": gen_result.completion_tokens,
                                    "total_tokens": gen_result.total_tokens,
                                    "cost": gen_result.cost,
                                    "model_name": gen_result.model_name,
                                    "step_index": gen_result.step_index,
                                    "data_dir": gen_result.data_dir,
                                    "success": gen_result.success,
                                    "error": gen_result.error,
                                    "timestamp": gen_result.timestamp.isoformat(),
                            }
                            gen_record.update(dict(gen_result.search_metadata or {}))
                            await self._gen_writer.write(gen_record)
                        # 触发生成完成回调(状态推进)
                        await _on_gen_result(gen_result)
                        # 成功且启用评估:入评估队列
                        if self._config.enable_sandbox_eval and gen_result.success:
                            await self._eval_queue.put(gen_result)
                            async with self._progress_lock:
                                self._eval_progress.pending_in_queue += 1

                    # 标记队列任务完成
                    gen_queue.task_done()

            # 评估循环任务(关闭评估时为 None)
            eval_task = (
                asyncio.create_task(
                    self._run_eval_loop(output_dir, store_results=False)
                )
                if self._config.enable_sandbox_eval
                else None
            )

            # 进度快照间隔(至少 1 秒)
            interval = max(1.0, float(self._config.progress_history_interval_seconds))

            async def _progress_history_loop() -> None:
                """进度历史协程:定期写快照与趋势图;结束时补写一次。"""
                await self._write_progress_snapshot_and_chart(output_dir)
                while not done_event.is_set():
                    try:
                        # 等待完成事件,带超时以便周期性写快照
                        await asyncio.wait_for(
                            done_event.wait(), timeout=interval
                        )
                    except asyncio.TimeoutError:
                        pass
                    if not done_event.is_set():
                        await self._write_progress_snapshot_and_chart(output_dir)
                await self._write_progress_snapshot_and_chart(output_dir)

            # 启动进度历史任务
            progress_history_task = asyncio.create_task(_progress_history_loop())

            # 生成 worker 数 = LLM 并发
            worker_count = max(1, int(self._config.llm_concurrency))
            gen_workers = [
                asyncio.create_task(_gen_worker()) for _ in range(worker_count)
            ]

            # 恢复后可能全部任务已完成:直接点亮 done_event
            if self._completed_tasks >= self._total_tasks:
                done_event.set()

            # 等全部任务完成 -> 等生成队列清空 -> 通知 worker 退出并回收
            await done_event.wait()
            await gen_queue.join()
            for _ in range(worker_count):
                await gen_queue.put(None)
            await asyncio.gather(*gen_workers)
            if eval_task is not None:
                # 评估队列发终止哨兵并等待评估循环结束
                await self._eval_queue.put(None)
                await eval_task
            # 等进度历史任务收尾(最后一次快照)
            await progress_history_task

        # 记录结束时间并清理评估器引用
        self._end_time = datetime.now()
        self._evaluator = None
        self._gen_progress.end_time = datetime.now()
        self._eval_progress.end_time = datetime.now()

        # 组装任务级结果
        task_results: list[TaskResult] = []
        # 任务 -> 生成结果列表
        task_gen_map: dict[str, list[GenerationResult]] = {
            task["task_id"]: [] for task in tasks
        }
        # 任务 -> 评估结果列表
        task_eval_map: dict[str, list[EvaluationResult]] = {
            task["task_id"]: [] for task in tasks
        }

        # 按任务归集生成结果
        for gen_result in all_gen_results:
            if gen_result.task_id in task_gen_map:
                task_gen_map[gen_result.task_id].append(gen_result)

        # 按任务归集评估结果
        for eval_result in all_eval_results:
            if eval_result.task_id in task_eval_map:
                task_eval_map[eval_result.task_id].append(eval_result)

        # 逐任务创建最终结果对象
        for task in tasks:
            task_id = task["task_id"]
            task_name = task["task_name"]
            result = self._create_task_result(
                task_id=task_id,
                task_name=task_name,
                generation_results=task_gen_map.get(task_id, []),
                evaluation_results=task_eval_map.get(task_id, []),
                metadata=task_metadata_map.get(task_id, {}),
            )
            task_results.append(result)

        # 卸载钩子,避免下次运行复用旧闭包
        self._eval_result_hook = None
        self._eval_record_metadata_hook = None
        # 返回任务结果列表
        return task_results

    def _build_accepted_targets(
        self,
        *,
        tasks: list[dict[str, Any]],
        policy: RejectionPolicy,
        default_target: int,
    ) -> dict[str, int]:
        """计算每个任务的接受目标:mixed 等策略可按任务元数据差异化目标。"""
        # 策略是否提供按元数据取目标的方法
        target_for_metadata = getattr(policy, "target_for_metadata", None)
        # 任务 -> 目标
        targets: dict[str, int] = {}
        for task in tasks:
            # 默认用全局目标
            target = default_target
            if target_for_metadata is not None:
                # 策略按元数据计算目标
                policy_target = target_for_metadata(task.get("metadata"))
                if policy_target is not None:
                    target = int(policy_target)
            targets[task["task_id"]] = target
        return targets

    def _tree_search_enabled(self) -> bool:
        """树搜索是否生效(配置开启且算法实例存在)。"""
        return bool(self._config.tree_search_enabled and self._search_algorithm)

    def _task_search_paths(
        self, task_output_dir: Path
    ) -> tuple[Path, Path]:
        """返回任务级搜索事件日志与状态快照的路径。"""
        return task_output_dir / "search_events.jsonl", task_output_dir / "search_state.json"

    def _relative_to_task_dir(self, path: Path | None, task_output_dir: Path | None) -> str | None:
        """把路径表示为相对任务目录的形式(失败/无目录时回退绝对路径)。"""
        if path is None:
            return None
        if task_output_dir is None:
            return str(path)
        try:
            return str(path.relative_to(task_output_dir))
        except ValueError:
            return str(path)

    def _step_code_path(self, output_dir: Path | None, step_index: int) -> str | None:
        """返回某一步代码文件的相对路径约定(step_N/valid_code.py)。"""
        if output_dir is None:
            return None
        return str(Path(f"step_{step_index}") / "valid_code.py")

    def _code_sha256(self, code: str) -> str:
        """计算代码的 SHA256 指纹(搜索事件记录用)。"""
        return hashlib.sha256(code.encode("utf-8")).hexdigest()

    def _build_sft_assistant_content(self, gen_result: GenerationResult) -> str:
        """构建 SFT 风格的 assistant 回复:优先 <think>+代码块,按内容可用性降级。"""
        reasoning = str(gen_result.reasoning_content or "")
        code = str(gen_result.code or "")
        response = str(gen_result.raw_text or "")
        if reasoning.strip():
            if code.strip():
                # 思考+代码:<think> 包裹推理,代码进 python 围栏
                return (
                    f"<think>\n{reasoning.strip()}\n</think>\n\n"
                    f"```python\n{code.rstrip()}\n```"
                )
            # 只有思考:包裹推理后接原始回复
            return f"<think>\n{reasoning.strip()}\n</think>\n\n{response.strip()}"
        if code.strip():
            # 只有代码:python 围栏
            return f"```python\n{code.rstrip()}\n```"
        # 都没有:原始回复
        return response.strip()

    def _attach_slime_message_token_count(self, eval_result: EvaluationResult) -> None:
        """为评估结果补 SFT 基线的消息 token 数(策略的 token 过滤需要)。"""
        # 已有 token 数:不重复计算
        if getattr(eval_result, "slime_message_tokens", None) is not None:
            return
        # 未配置基线分词器:跳过
        tokenizer_model = self._config.rejection_baseline_tokenizer_model
        if not tokenizer_model:
            return
        # 找不到对应生成结果或消息为空:跳过
        gen_result = self._gen_results.get(eval_result.request_id)
        if gen_result is None or not gen_result.messages:
            return
        # 基线分词器懒加载(只读本地文件)
        if self._baseline_tokenizer is None:
            self._baseline_tokenizer = load_tokenizer(
                tokenizer_model,
                local_files_only=True,
            )
        # 复制消息并追加构造的 assistant 回复
        messages = [dict(message) for message in gen_result.messages]
        messages.append(
            {
                "role": "assistant",
                "content": self._build_sft_assistant_content(gen_result),
            }
        )
        # 按 chat 模板统计 token 数
        token_count = count_chat_template_tokens(messages, self._baseline_tokenizer)
        # 挂到评估结果上(动态属性)
        setattr(eval_result, "slime_message_tokens", int(token_count))
        setattr(eval_result, "slime_message_tokenizer_model", str(tokenizer_model))

    def _write_tree_search_state(
        self,
        *,
        task_state: TaskLoopState,
        task_output_dir: Path,
        accepted_target: int | None,
    ) -> None:
        """把任务级搜索状态(程序清单/谱系/计数)原子写盘,供恢复与外部观测。"""
        # 无程序库:跳过
        if self._database is None:
            return
        # 取该任务全部程序
        programs = self._database.list_by_task(task_state.task_name)
        program_ids: list[str] = []
        program_scores: dict[str, float | None] = {}
        program_fitness: dict[str, float | None] = {}
        program_code_paths: dict[str, str] = {}
        parent_map: dict[str, list[str]] = {}
        generation_modes: dict[str, str] = {}
        # 逐程序提取展示字段
        for program in programs:
            program_id = str(program.metadata.get("program_id") or program.id)
            program_ids.append(program_id)
            program_scores[program_id] = program.score
            program_fitness[program_id] = program.fitness
            if program.metadata.get("code_path"):
                program_code_paths[program_id] = str(program.metadata["code_path"])
            parent_ids = program.metadata.get("parent_ids") or []
            if parent_ids:
                parent_map[program_id] = [str(parent_id) for parent_id in parent_ids]
            generation_modes[program_id] = str(program.generation_mode)

        # 状态快照路径
        _, state_path = self._task_search_paths(task_output_dir)
        # 原子写出搜索状态
        atomic_write_search_state(
            state_path,
            SearchState(
                task_id=task_state.task_id,
                task_name=task_state.task_name,
                search_algorithm=str(self._config.tree_search_algorithm).lower(),
                next_search_step=task_state.next_step_index,
                generated_count=task_state.generated_count,
                completed_count=task_state.completed_count,
                accepted_count=task_state.accepted_count,
                accepted_target=accepted_target,
                max_generated=task_state.max_target,
                stop_requested=task_state.done,
                program_ids=program_ids,
                program_scores=program_scores,
                program_fitness=program_fitness,
                program_code_paths=program_code_paths,
                parent_map=parent_map,
                generation_modes=generation_modes,
            ),
        )

    def _build_tree_program_from_eval(
        self,
        *,
        gen_result: GenerationResult,
        eval_result: EvaluationResult,
        accepted: bool,
        rejection_reason: str,
    ) -> Program:
        """由生成+评估结果构建树搜索的 Program 记录(谱系/分数/日志齐全)。"""
        # 搜索谱系元数据
        search_metadata = dict(gen_result.search_metadata or {})
        parent_db_id = search_metadata.get("parent_db_id")
        parent_code = str(search_metadata.get("parent_code") or "")
        # 程序元数据:任务元数据 + 搜索谱系 + 运行时信息
        program_metadata = dict(gen_result.metadata)
        program_metadata.update(search_metadata)
        program_metadata.update(
            {
                "request_messages": gen_result.messages,
                "status_code": eval_result.status_code,
                "status": eval_result.status,
                "queue_time": eval_result.queue_time,
                "run_time": eval_result.run_time,
                "model_name": gen_result.model_name,
                "cost": gen_result.cost,
                "prompt_tokens": gen_result.prompt_tokens,
                "completion_tokens": gen_result.completion_tokens,
                "total_tokens": gen_result.total_tokens,
                "reasoning_content": gen_result.reasoning_content,
                "sandbox_job_id": eval_result.job_id,
                "step_index": gen_result.step_index,
                "raw_run_log": eval_result.raw_run_log,
                "clear_run_log": eval_result.clear_run_log,
                "submit_grade": eval_result.submit_grade,
                "submit_medal": eval_result.submit_medal,
                "accepted_by_rejection_policy": accepted,
                "rejection_reason": rejection_reason,
                "code_path": self._step_code_path(gen_result.output_dir, gen_result.step_index),
            }
        )
        # 构建 Program(fitness=reward,父指针指向程序库行)
        return Program(
            task_id=eval_result.task_id,
            task_name=eval_result.task_name,
            code=gen_result.code,
            score=eval_result.score,
            reward=eval_result.reward,
            fitness=eval_result.reward,
            base_reward=eval_result.reward,
            run_log=str(eval_result.clear_run_log or ""),
            feedback=eval_result.feedback,
            parent_id=int(parent_db_id) if parent_db_id is not None else None,
            parent_code=parent_code,
            generation_mode=str(search_metadata.get("generation_mode") or "draft"),
            raw_text=gen_result.raw_text,
            metadata=program_metadata,
        )

    def _preload_tree_database_from_checkpoint(
        self,
        *,
        gen_results_data: list[dict[str, Any]],
        eval_results_data: list[dict[str, Any]],
        task_metadata_map: Mapping[str, dict[str, Any]],
    ) -> None:
        """树搜索续跑:按评估记录(按步序排序)重放程序库,重建父指针。"""
        # 未启用树搜索或无程序库:跳过
        if not self._tree_search_enabled() or self._database is None:
            return
        # request_id -> 生成记录
        gen_by_request = {str(row.get("request_id")): row for row in gen_results_data}
        # program_id -> 数据库行 ID(建立父子映射)
        db_id_by_program_id: dict[str, int] = {}
        # 按步序号排序重放,保证父程序先入库
        for row in sorted(eval_results_data, key=lambda item: int(item.get("step_index", 0))):
            request_id = str(row.get("request_id"))
            gen_row = gen_by_request.get(request_id)
            # 无对应生成记录(如生成失败):跳过
            if not gen_row:
                continue
            # 程序 ID:评估记录优先,生成记录次之,request_id 兜底
            program_id = str(row.get("program_id") or gen_row.get("program_id") or request_id)
            # 已重放过:跳过
            if program_id in db_id_by_program_id:
                continue
            task_id = str(row.get("task_id") or gen_row.get("task_id"))
            metadata = dict(task_metadata_map.get(task_id, {}))
            # 父程序 ID 列表(评估记录优先)
            parent_ids = row.get("parent_ids") or gen_row.get("parent_ids") or []
            parent_program_id = str(parent_ids[0]) if parent_ids else None
            # 映射到数据库行 ID
            parent_db_id = (
                db_id_by_program_id.get(parent_program_id)
                if parent_program_id is not None
                else None
            )
            # 元数据补齐搜索谱系与接受信息
            metadata.update(
                {
                    "program_id": program_id,
                    "parent_ids": parent_ids,
                    "search_algorithm": row.get("search_algorithm") or "greedy",
                    "search_step": int(row.get("search_step", row.get("step_index", 0))),
                    "generation_mode": row.get("generation_mode") or "draft",
                    "accepted_by_rejection_policy": row.get(
                        "accepted_by_rejection_policy"
                    ),
                    "rejection_reason": row.get("rejection_reason"),
                    "code_path": row.get("code_path"),
                }
            )
            # 重建 Program 并入库
            program = Program(
                task_id=task_id,
                task_name=str(row.get("task_name") or gen_row.get("task_name")),
                code=str(gen_row.get("code") or ""),
                score=row.get("score"),
                reward=float(row.get("reward") or 0.0),
                fitness=float(row.get("fitness", row.get("reward") or 0.0)),
                base_reward=float(row.get("reward") or 0.0),
                run_log="",
                feedback="",
                parent_id=parent_db_id,
                parent_code="",
                generation_mode=str(row.get("generation_mode") or "draft"),
                raw_text=str(gen_row.get("raw_text") or ""),
                metadata=metadata,
            )
            db_id_by_program_id[program_id] = self._database.add(program)

    def _update_loop_progress_totals(
        self, task_states: dict[str, TaskLoopState]
    ) -> None:
        """按任务目标总和刷新两管线的 total(循环模式目标会动态增长)。"""
        # 所有任务当前目标之和
        total_target = sum(state.target_count for state in task_states.values())
        self._gen_progress.total = total_target
        self._eval_progress.total = total_target

    def _update_progress_trend_chart(
        self, output_dir: Path, history_path: Path
    ) -> None:
        """读取 progress_history.jsonl 并绘制 progress_trend.png(生成/评估/完成任务随时间曲线)。"""
        # 无 matplotlib 或无历史文件:跳过
        if not _HAS_MATPLOTLIB or not history_path.exists():
            print(f"No matplotlib or history_path: {history_path}")
            return
        # 历史记录列表
        records: list[dict[str, Any]] = []
        with open(history_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        # 无记录:不画图
        if not records:
            return
        # 各序列数据
        timestamps = []
        gen_success = []
        gen_completed = []
        gen_failed = []
        eval_success = []
        eval_completed = []
        eval_failed = []
        completed_tasks = []
        # 解析每条历史记录
        for r in records:
            try:
                # 时间戳统一为无时区 UTC(便于 matplotlib 处理)
                ts = datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00"))
                if ts.tzinfo is not None:
                    ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
            except (KeyError, ValueError):
                continue
            timestamps.append(ts)
            g = r.get("generation", {})
            gen_success.append(g.get("success", 0))
            gen_completed.append(g.get("completed", 0))
            gen_failed.append(g.get("failed", 0))
            e = r.get("evaluation", {})
            eval_success.append(e.get("success", 0))
            eval_completed.append(e.get("completed", 0))
            eval_failed.append(e.get("failed", 0))
            completed_tasks.append(r.get("completed_tasks", 0))
        if not timestamps:
            return
        try:
            # 中文说明:图宽在 [w_min, w_max] 内按时间跨度自适应;完整时间轴始终画出,
            # 超过 w_max 后横向压缩(x 轴不再按比例展开)。
            time_span_hours = (
                timestamps[-1] - timestamps[0]
            ).total_seconds() / 3600.0
            w_min = getattr(
                self._config, "progress_trend_fig_width_min", 10.0
            )
            w_max = getattr(
                self._config, "progress_trend_fig_width_max", 40.0
            )
            inches_per_hour = getattr(
                self._config, "progress_trend_inches_per_hour", 2.0
            )
            width = min(w_max, max(w_min, time_span_hours * inches_per_hour))
            # 3 行子图共享 x 轴
            fig, axes = plt.subplots(
                3, 1, figsize=(width, 8), sharex=True
            )
            ax_gen, ax_eval, ax_tasks = axes

            # 生成面板:成功/完成/失败三条曲线
            ax_gen.plot(timestamps, gen_success, label="success", color="C0")
            ax_gen.plot(timestamps, gen_completed, label="completed", color="C1")
            ax_gen.plot(timestamps, gen_failed, label="failed", color="C2")
            ax_gen.set_ylabel("Generation")
            ax_gen.legend(loc="upper left")
            ax_gen.grid(True, alpha=0.3)

            # 评估面板:同样三条曲线
            ax_eval.plot(timestamps, eval_success, label="success", color="C0")
            ax_eval.plot(timestamps, eval_completed, label="completed", color="C1")
            ax_eval.plot(timestamps, eval_failed, label="failed", color="C2")
            ax_eval.set_ylabel("Evaluation")
            ax_eval.legend(loc="upper left")
            ax_eval.grid(True, alpha=0.3)

            # 任务面板:完成任务数曲线
            ax_tasks.plot(timestamps, completed_tasks, label="completed_tasks", color="C3")
            ax_tasks.set_ylabel("Completed tasks")
            ax_tasks.set_xlabel("Time")
            ax_tasks.legend(loc="upper left")
            ax_tasks.grid(True, alpha=0.3)

            # 总标题并保存
            fig.suptitle("Progress trend (gen / eval / completed_tasks)")
            fig.tight_layout()
            trend_path = output_dir / "progress_trend.png"
            fig.savefig(trend_path, dpi=120)
            plt.close(fig)
            logger.info("Updated progress trend chart: %s", trend_path)
        except Exception as exc:
            # 画图失败只告警,不影响主流程
            logger.warning("Failed to update progress trend chart: %s", exc)

    async def _write_progress_snapshot_and_chart(self, output_dir: Path) -> None:
        """持锁取进度快照 -> 追加写 progress_history.jsonl -> 更新趋势图。"""
        async with self._progress_lock:
            # 任务级奖牌/接受数汇总
            medal_count = 0
            accepted_count = 0
            if self._task_states is not None:
                medal_count = sum(state.medal_count for state in self._task_states.values())
                accepted_count = sum(
                    state.accepted_count for state in self._task_states.values()
                )
            # 组装快照(完成 = 成功 + 失败)
            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "generation": {
                    "success": self._gen_progress.success,
                    "completed": self._gen_progress.success + self._gen_progress.failed,
                    "failed": self._gen_progress.failed,
                },
                "evaluation": {
                    "success": self._eval_progress.success,
                    "completed": self._eval_progress.success + self._eval_progress.failed,
                    "failed": self._eval_progress.failed,
                },
                "completed_tasks": self._completed_tasks,
                "medal_count": medal_count,
                "accepted_count": accepted_count,
            }
        # 追加写历史文件(锁外写,减少持锁时间)
        history_path = output_dir / "progress_history.jsonl"
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
        # 刷新趋势图
        self._update_progress_trend_chart(output_dir, history_path)

    async def _run_eval_loop(
        self, output_dir: Path, store_results: bool = True
    ) -> list[EvaluationResult]:
        """评估主循环(fire-and-forget 模式)。

        从评估队列消费样本,以最大并发提交沙箱评估;
        一个评估完成立即补位下一个,保持沙箱并发饱和。

        Args:
            output_dir: 评估产物输出目录
            store_results: 是否把结果存入内存(默认 True)

        Returns:
            全部评估结果列表
        """
        # 全量结果收集
        all_eval_results: list[EvaluationResult] = []
        # 待提交的生成结果缓冲
        pending_gen_results: list[GenerationResult] = []
        # 并发上限 = 沙箱并发
        batch_size = self._config.sandbox_concurrency

        # fire-and-forget:在飞评估任务集合(尽快提交,保持并发饱和)
        pending_eval_tasks: set[asyncio.Task] = set()
        # 生成侧是否已结束(收到哨兵)
        generation_done = False

        async def _drain_eval_queue(room: int) -> None:
            # 非阻塞排空队列(最多 room 个);遇哨兵置 generation_done
            nonlocal generation_done
            if room <= 0:
                return
            for _ in range(room):
                try:
                    item = self._eval_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    generation_done = True
                    break
                pending_gen_results.append(item)
                async with self._progress_lock:
                    self._eval_progress.pending_in_queue -= 1

        def _submit_pending_requests() -> None:
            # 把缓冲中的样本逐个提交为评估批次任务(直到并发满)
            while len(pending_eval_tasks) < batch_size and pending_gen_results:
                batch = pending_gen_results[:1]
                del pending_gen_results[:1]
                task = asyncio.create_task(
                    self._process_eval_batch(
                        batch, output_dir, store_results=store_results
                    )
                )
                pending_eval_tasks.add(task)

        def _collect_completed_eval_tasks(
            done_tasks: set[asyncio.Task],
        ) -> None:
            # 收集已完成评估任务的结果;任务崩溃只记录异常
            for task in done_tasks:
                try:
                    batch_results = task.result()
                except Exception:
                    logger.exception("Evaluation batch task crashed unexpectedly")
                    continue
                if store_results:
                    all_eval_results.extend(batch_results)

        # 主循环:直到生成结束且无在飞/待提交任务
        while not generation_done or pending_eval_tasks or pending_gen_results:
            # 计算可吸收的剩余并发额度
            room = batch_size - len(pending_eval_tasks) - len(pending_gen_results)
            await _drain_eval_queue(room)
            _submit_pending_requests()

            # 全部结束:退出
            if generation_done and not pending_eval_tasks and not pending_gen_results:
                break

            # 并发已满:等任一任务完成再补位
            if len(pending_eval_tasks) >= batch_size:
                done, pending = await asyncio.wait(
                    pending_eval_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                _collect_completed_eval_tasks(done)
                pending_eval_tasks = set(pending)
                continue

            # 未满:同时等"队列新样本"与"任一任务完成"(避免空转)
            queue_get_task = asyncio.create_task(self._eval_queue.get())
            wait_targets = set(pending_eval_tasks)
            wait_targets.add(queue_get_task)

            done, pending = await asyncio.wait(
                wait_targets,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if queue_get_task in done:
                # 队列先到:处理新样本
                item = queue_get_task.result()
                if item is None:
                    generation_done = True
                else:
                    pending_gen_results.append(item)
                    async with self._progress_lock:
                        self._eval_progress.pending_in_queue -= 1
            else:
                # 任务先完成:取消排队中的 get(防悬挂)
                queue_get_task.cancel()
                try:
                    await queue_get_task
                except asyncio.CancelledError:
                    pass

            # 收集本轮已完成的评估任务
            completed_eval_tasks = {task for task in done if task is not queue_get_task}
            if completed_eval_tasks:
                _collect_completed_eval_tasks(completed_eval_tasks)

            # 更新在飞任务集合(排除已取消的 get 任务)
            pending_eval_tasks = {
                task for task in pending if task is not queue_get_task
            }

        # 收尾:等待所有剩余评估任务完成
        if pending_eval_tasks:
            results = await asyncio.gather(*pending_eval_tasks)
            for batch_results in results:
                if store_results:
                    all_eval_results.extend(batch_results)

        # 记录评估管线结束时间
        async with self._progress_lock:
            self._eval_progress.end_time = datetime.now()

        logger.info("Evaluation pipeline completed")
        return all_eval_results

    async def _process_eval_batch(
        self,
        gen_results: list[GenerationResult],
        output_dir: Path,
        store_results: bool = True,
    ) -> list[EvaluationResult]:
        """把一批生成结果送入评估服务,处理并落盘评估结果。

        Args:
            gen_results: 待评估的生成结果
            output_dir: 评估产物输出目录
            store_results: 是否写入 self._eval_results(正常模式用)

        Returns:
            评估结果列表
        """
        # 空批次或无评估器:直接返回
        if not gen_results or self._evaluator is None:
            return []

        # 构建评估请求列表
        eval_requests = []
        for gen_result in gen_results:
            # 任务已完成时可选跳过其队列中的评估(省沙箱资源)
            if self._config.skip_eval_for_done_tasks and self._task_states is not None:
                task_state = self._task_states.get(gen_result.task_id)
                if task_state is not None and task_state.done:
                    logger.info(
                        "Skipping eval for done task: request_id=%s task_id=%s",
                        gen_result.request_id,
                        gen_result.task_id,
                    )
                    continue

            # 确定任务输出目录
            base_task_root = (
                output_dir / self._config.task_subdir
                if getattr(self._config, "task_subdir", None)
                else output_dir
            )
            base_task_root.mkdir(parents=True, exist_ok=True)
            task_output_dir = base_task_root / safe_task_output_name(
                gen_result.task_name,
                gen_result.task_id,
            )
            task_output_dir.mkdir(parents=True, exist_ok=True)

            # 组装评估请求(带生成侧 token 统计)
            eval_request = EvaluationRequest(
                request_id=gen_result.request_id,
                task_id=gen_result.task_id,
                task_name=gen_result.task_name,
                code=gen_result.code,
                data_dir=gen_result.data_dir,
                metadata=gen_result.metadata,
                step_index=gen_result.step_index,
                output_dir=task_output_dir,
                use_score2reward=self._config.use_score2reward,
                generation_prompt_tokens=gen_result.prompt_tokens,
                generation_completion_tokens=gen_result.completion_tokens,
                generation_total_tokens=gen_result.total_tokens,
            )
            eval_requests.append(eval_request)

        # 全部被跳过:无请求
        if not eval_requests:
            return []

        # 更新进度:计入在飞与占用沙箱槽位
        async with self._progress_lock:
            self._eval_progress.in_progress += len(eval_requests)
            self._eval_progress.active_sandbox_slots += len(eval_requests)

        # 执行批量评估(无论成败最后都释放槽位)
        try:
            eval_results = await self._evaluator.evaluate_batch(eval_requests)
        finally:
            # 释放沙箱槽位
            async with self._progress_lock:
                self._eval_progress.active_sandbox_slots -= len(eval_requests)

        # 逐条处理评估结果
        for eval_result in eval_results:
            # 更新进度
            async with self._progress_lock:
                self._eval_progress.in_progress -= 1
                if not eval_result.success:
                    self._eval_progress.failed += 1
                else:
                    self._eval_progress.success += 1

            # 缓存结果(仅正常模式)
            if store_results:
                async with self._results_lock:
                    self._eval_results[eval_result.request_id] = eval_result

            # 流式写评估记录
            if self._eval_writer:
                eval_record = {
                        "request_id": eval_result.request_id,
                        "task_id": eval_result.task_id,
                        "task_name": eval_result.task_name,
                        "status_code": eval_result.status_code,
                        "status": eval_result.status,
                        "score": eval_result.score,
                        "reward": eval_result.reward,
                        "submit_grade": eval_result.submit_grade,
                        "submit_medal": eval_result.submit_medal,
                        "job_id": eval_result.job_id,
                        "queue_time": eval_result.queue_time,
                        "run_time": eval_result.run_time,
                        "step_index": eval_result.step_index,
                        "success": eval_result.success,
                        "generation_prompt_tokens": (
                            eval_result.generation_prompt_tokens
                        ),
                        "generation_completion_tokens": (
                            eval_result.generation_completion_tokens
                        ),
                        "generation_total_tokens": eval_result.generation_total_tokens,
                        "slime_message_tokens": getattr(
                            eval_result, "slime_message_tokens", None
                        ),
                        "slime_message_tokenizer_model": getattr(
                            eval_result, "slime_message_tokenizer_model", None
                        ),
                        "feedback_path": (
                            str(
                                eval_result.output_dir
                                / f"step_{eval_result.step_index}"
                                / "feedback.txt"
                            )
                            if eval_result.output_dir is not None
                            else None
                        ),
                        "error": eval_result.error,
                        "timestamp": eval_result.timestamp.isoformat(),
                    }
                # 钩子补搜索元数据(树搜索谱系)
                if self._eval_record_metadata_hook is not None:
                    eval_record.update(self._eval_record_metadata_hook(eval_result))
                await self._eval_writer.write(eval_record)
            # 触发评估结果钩子(状态推进);钩子异常只告警不中断
            if self._eval_result_hook is not None:
                try:
                    await self._eval_result_hook(eval_result)
                except Exception as exc:
                    logger.warning("Eval result hook failed: %s", exc)

        return eval_results

    def _create_task_result(
        self,
        task_id: str,
        task_name: str,
        generation_results: list[GenerationResult],
        evaluation_results: list[EvaluationResult],
        metadata: dict[str, Any],
    ) -> TaskResult:
        """由生成与评估结果组装单个任务的 TaskResult(含最优程序与统计)。"""
        # 样本 Program 列表
        programs: list[Program] = []
        # request_id -> 评估结果映射
        eval_map = {r.request_id: r for r in evaluation_results}

        # 费用/token/状态/奖牌/耗时累计器
        total_cost = 0.0
        total_tokens = 0
        status_counts: dict[str, int] = {}
        medal_counts: dict[str, int] = {medal: 0 for medal in ("gold", "silver", "bronze")}
        test_time = 0.0

        # 遍历该任务全部生成结果,配对评估结果
        for gen_result in generation_results:
            eval_result = eval_map.get(gen_result.request_id)

            if eval_result:
                print("logging evaled program")
                # 程序元数据:任务元数据 + 评估运行时信息
                program_metadata = dict(metadata)
                program_metadata["request_messages"] = gen_result.messages
                program_metadata["status_code"] = eval_result.status_code
                program_metadata["status"] = eval_result.status
                program_metadata["queue_time"] = eval_result.queue_time
                program_metadata["run_time"] = eval_result.run_time
                program_metadata["model_name"] = gen_result.model_name
                program_metadata["cost"] = gen_result.cost
                program_metadata["prompt_tokens"] = gen_result.prompt_tokens
                program_metadata["completion_tokens"] = gen_result.completion_tokens
                program_metadata["total_tokens"] = gen_result.total_tokens
                program_metadata["reasoning_content"] = gen_result.reasoning_content
                program_metadata["sandbox_job_id"] = eval_result.job_id
                program_metadata["step_index"] = gen_result.step_index
                program_metadata["raw_run_log"] = eval_result.raw_run_log
                program_metadata["clear_run_log"] = eval_result.clear_run_log
                program_metadata["submit_grade"] = eval_result.submit_grade
                program_metadata["submit_medal"] = eval_result.submit_medal

                # 构建 Program(评估过的样本)
                program = Program(
                    task_id=task_id,
                    task_name=task_name,
                    code=gen_result.code,
                    score=eval_result.score,
                    reward=eval_result.reward,
                    fitness=eval_result.reward,
                    base_reward=eval_result.reward,
                    run_log=str(eval_result.clear_run_log or ""),
                    feedback=eval_result.feedback,
                    parent_id=None,
                    parent_code="",
                    generation_mode="draft",
                    raw_text=gen_result.raw_text,
                    metadata=program_metadata,
                )
                programs.append(program)

                # 统计状态/奖牌/测试耗时
                status = eval_result.status
                status_counts[status] = status_counts.get(status, 0) + 1
                submit_medal = str(eval_result.submit_medal).lower()
                if submit_medal in medal_counts:
                    medal_counts[submit_medal] += 1
                if eval_result.run_time is not None:
                    test_time += eval_result.run_time
            elif gen_result.success and not self._config.enable_sandbox_eval:
                print("logging unevaled program")
                # 关闭沙箱评估:成功的生成样本按"未评估"收录(分数为空)
                program_metadata = dict(metadata)
                program_metadata["request_messages"] = gen_result.messages
                program_metadata["model_name"] = gen_result.model_name
                program_metadata["cost"] = gen_result.cost
                program_metadata["prompt_tokens"] = gen_result.prompt_tokens
                program_metadata["completion_tokens"] = gen_result.completion_tokens
                program_metadata["total_tokens"] = gen_result.total_tokens
                program_metadata["reasoning_content"] = gen_result.reasoning_content
                program_metadata["step_index"] = gen_result.step_index
                program_metadata["status"] = "unevaluated"
                program_metadata["evaluated"] = False

                program = Program(
                    task_id=task_id,
                    task_name=task_name,
                    code=gen_result.code,
                    score=None,
                    reward=0.0,
                    fitness=0.0,
                    base_reward=0.0,
                    run_log="",
                    feedback="",
                    parent_id=None,
                    parent_code="",
                    generation_mode="draft",
                    raw_text=gen_result.raw_text,
                    metadata=program_metadata,
                )
                programs.append(program)
                status_counts["unevaluated"] = status_counts.get("unevaluated", 0) + 1

            # 费用与 token 累计(无论是否评估)
            total_cost += gen_result.cost
            total_tokens += gen_result.total_tokens

        # 最优程序 = 奖励最高者(无样本为 None)
        best_program = max(programs, key=lambda p: p.reward) if programs else None

        if best_program is None:
            # 无任何样本:构造空占位程序
            best_program = Program(
                task_id=task_id,
                task_name=task_name,
                code="",
                score=None,
                reward=0.0,
                fitness=0.0,
                base_reward=0.0,
            )

        # 配置了程序库:把样本入库并回填行 ID
        if self._database and programs:
            for program in programs:
                program_id = self._database.add(program)
                program.id = program_id

        # 旧版任务级选择逻辑(已禁用,注释保留)
        # if self._config.per_task_selection and programs:
        #     self._perform_task_selection(task_name, programs)

        # 组装 TaskResult
        return TaskResult(
            task_id=task_id,
            task_name=task_name,
            samples=programs,
            best_program=best_program,
            best_score=best_program.score,
            best_reward=best_program.reward,
            total_cost=total_cost,
            total_tokens=total_tokens,
            num_samples=len(programs),
            status_counts=status_counts,
            medal_count=sum(medal_counts.values()),
            medal_counts=medal_counts,
            val_time=0.0,
            test_time=test_time,
        )

    def get_stats(self) -> dict[str, Any]:
        """获取调度器统计(生成/评估服务统计 + 时间线 + 进度)。"""
        # 生成服务统计
        gen_stats = self._generator.get_stats()
        # 评估服务统计(无评估器为空)
        eval_stats = self._evaluator.get_stats() if self._evaluator else {}

        return {
            "generator": gen_stats,
            "evaluator": eval_stats,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "end_time": self._end_time.isoformat() if self._end_time else None,
            "duration_seconds": (
                (self._end_time - self._start_time).total_seconds()
                if self._start_time and self._end_time
                else None
            ),
            "progress": self.get_progress(),
        }

    def get_progress(self) -> dict[str, Any]:
        """获取两条管线的实时进度(生成/评估分开统计)。"""
        # 生成已耗时(秒)
        gen_elapsed = None
        if self._gen_progress.start_time:
            end = self._gen_progress.end_time or datetime.now()
            gen_elapsed = (end - self._gen_progress.start_time).total_seconds()

        # 评估已耗时(秒)
        eval_elapsed = None
        if self._eval_progress.start_time:
            end = self._eval_progress.end_time or datetime.now()
            eval_elapsed = (end - self._eval_progress.start_time).total_seconds()

        # 两条管线的吞吐率(完成数/秒)
        gen_rate = (
            (self._gen_progress.success + self._gen_progress.failed) / gen_elapsed
            if gen_elapsed and gen_elapsed > 0
            else 0
        )
        eval_rate = (
            (self._eval_progress.success + self._eval_progress.failed) / eval_elapsed
            if eval_elapsed and eval_elapsed > 0
            else 0
        )

        # 任务状态快照(loop 模式)
        task_states_snapshot = None
        if self._task_states is not None:
            task_states_snapshot = {
                task_id: {
                    "task_name": state.task_name,
                    "success": state.success_count,
                    "medal": state.medal_count,
                    "accepted": state.accepted_count,
                    "completed": state.completed_count,
                    "target": state.target_count,
                    "accepted_target": state.accepted_target,
                    "max_target": state.max_target,
                    "generated": state.generated_count,
                    "done": state.done,
                }
                for task_id, state in self._task_states.items()
            }

        # 组装进度报告
        return {
            "total_tasks": self._total_tasks,
            "completed_tasks": self._completed_tasks,
            "task_states": task_states_snapshot,
            "generation": {
                "total": self._gen_progress.total,
                "success": self._gen_progress.success,
                "completed": self._gen_progress.success + self._gen_progress.failed,
                "in_progress": self._gen_progress.in_progress,
                "pending": self._gen_progress.pending,
                "failed": self._gen_progress.failed,
                "elapsed_seconds": gen_elapsed,
                "rate_per_second": gen_rate,
            },
            "evaluation": {
                "total": self._eval_progress.total,
                "success": self._eval_progress.success,
                "completed": self._eval_progress.success + self._eval_progress.failed,
                "in_progress": self._eval_progress.in_progress,
                "pending_in_queue": self._eval_progress.pending_in_queue,
                "active_sandbox_slots": self._eval_progress.active_sandbox_slots,
                "sandbox_concurrency": self._config.sandbox_concurrency,
                # 沙箱并发利用率 = 在飞槽位 / 并发上限
                "sandbox_utilization": (
                    self._eval_progress.active_sandbox_slots
                    / self._config.sandbox_concurrency
                    if self._config.sandbox_concurrency > 0
                    else 0
                ),
                "failed": self._eval_progress.failed,
                "elapsed_seconds": eval_elapsed,
                "rate_per_second": eval_rate,
            },
            # 旧字段:向后兼容外部脚本
            "total_samples": self._gen_progress.total,
            "generated_samples": self._gen_progress.success,
            "evaluated_samples": self._eval_progress.success,
            "generating_samples": self._gen_progress.in_progress,
            "evaluating_samples": self._eval_progress.in_progress,
            "streaming_samples_written": 0,
        }
