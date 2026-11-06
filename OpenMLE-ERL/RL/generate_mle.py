# Single-turn operators generation for MLE agent with remote sandbox
# 中文说明:本模块是 MLE(Machine Learning Engineering)智能体的单轮生成与奖励入口,
# 实现"四算子"(draft/improve/debug/crossover)驱动的程序进化搜索(Evo-Max 风格),
# 代码好坏以远程沙箱的真实执行结果为准(execution-grounded),评分最终映射为 RL 训练奖励。

# 异步原语:协程锁、Future、超时控制等
import asyncio
# JSON 编解码(序列化指标、解析 LLM 输出等)
import json
# 异步 HTTP 客户端,用于访问沙箱服务与 LLM 推理服务
import httpx
# 标准日志库
import logging
# 数学工具(isfinite 等,校验分数是否为有限值)
import math
# 读取环境变量配置
import os
# 统计工具,稍后注入给 program_database 模块使用
import statistics
# 线程锁,保护跨事件循环共享的原语注册表
import threading
# 时间戳(prompt dump 目录命名等)
import time
# 弱引用字典:事件循环销毁后其专属锁/信号量自动回收
import weakref
# 路径拼接(prompt dump 输出目录)
from pathlib import Path

# SLIME rollout 框架导入:生成状态(含 tokenizer)
from slime.rollout.sglang_rollout import GenerateState
# SLIME 的异步 POST 工具(向 SGLang router 请求生成)
from slime.utils.http_utils import post
# SLIME 的训练样本类型(tokens/response/loss_mask 等字段载体)
from slime.utils.types import Sample
# 奖励工具集:沙箱调用、代码抽取、分数-奖励映射、静态边界、作弊检测、验证-测试差距惩罚等
from reward_func_utils import (
    get_sandbox_result,
    get_proxy_sandbox_result,
    extract_code,
    score2reward,
    score2reward_with_static_priority,
    has_static_bounds_with_priority,
    get_clear_log,
    hack_check_async,
    has_final_validation_score_print,
    extract_validation_score,
    validation_test_gap_info,
    apply_validation_test_gap_penalty,
    VALIDATION_TEST_GAP_PENALTY_ENABLED,
    VALIDATION_TEST_GAP_PENALTY_PIECEWISE_ENABLED,
    VALIDATION_TEST_GAP_PENALTY_COEF,
    VALIDATION_TEST_GAP_PENALTY_HIGH_COEF,
    VALIDATION_TEST_GAP_PENALTY_TOLERANCE,
)
# 组级自适应奖励边界工具:按"组内分数+历史分数"动态确定奖励映射上下界(GRPO 组内优势的奖励整形)
from adaptive_reward_advantage_utils import (
    BOUND_MODE_THEORETICAL,
    SUPPORTED_ADAPTIVE_BOUND_MODES,
    score_to_group_adaptive_reward,
)

# 程序数据库与搜索算法:进化搜索的数据库、程序对象与三种搜索策略(greedy/evo/airaevo)
from program_database import (
    ProgramDatabase,
    Program,
    AIRAGreedySearch,
    AIRAEvoSearch,
    AIRAInferenceEvoSearch,
)
# 日志工具:初始化/获取全局训练数据记录器
from logging_utils import init_logger, get_logger
# 以模块对象形式引用 logging_utils,便于跨模块读写其全局状态(如 _global_logger)
import logging_utils as _logging_utils_module
# 以模块对象形式引用 program_database,用于下面的依赖注入
import program_database as _program_database_module
# AIRA 经验卡片构建与 JSON 解析工具(airaevo 搜索用)
from airaevo_experience import build_experience_card, parse_json_object
# AIRA 丰富记忆摘要的 prompt 构建器(airaevo 搜索用)
from prompt_builder import build_airaevo_rich_memory_summary_prompt

# 依赖注入:program_database 模块内部依赖 statistics 但未直接导入,这里手动注入
_program_database_module.statistics = statistics

# Configure logging format with timestamp at module level
# 获取本模块专用 logger(模块级配置一次)
logger = logging.getLogger("mle_agent")
if not logger.handlers:
    # Set up console handler with timestamp format
    # 尚无 handler 时才配置:创建输出到控制台的 handler
    console_handler = logging.StreamHandler()
    # 设置带时间戳/模块名/级别的日志格式
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    # 将 handler 挂到 logger 上
    logger.addHandler(console_handler)

# Get n_samples_per_prompt from environment variable or default to 8
# 从环境变量读取每个 prompt 的采样数(组大小),默认 8;GRPO 类算法按组计算组内优势
N_SAMPLES_PER_PROMPT = int(os.environ.get("N_SAMPLES_PER_PROMPT", "8"))
# 视为"生成中断"的代码类别集合(整组生成被中止,需重排重试)
GENERATION_ABORT_CODE_CATEGORIES = {"generation_abort", "generation_abort_group"}
# 生成中断相关的临时 metadata 键名:请求生成前先清空,避免上一轮状态残留
GENERATION_ABORT_TRANSIENT_METADATA_KEYS = (
    "generation_aborted",
    "generation_abort_reason",
    "generation_abort_finish_reason",
    "group_generation_aborted",
)

# GPU 沙箱服务的基础 URL(公开版本无默认端点,必须显式配置)
GPU_BASE_URL = os.getenv("GPU_BASE_URL", "").strip()
# CPU 沙箱服务的基础 URL(保留配置项)
CPU_BASE_URL = os.getenv("CPU_BASE_URL", "").strip()
try:
    # 沙箱并发上限(同时执行的沙箱任务数),默认 64
    SANDBOX_CONCURRENCY = int(os.getenv("SANDBOX_CONCURRENCY", "64"))
except (TypeError, ValueError):
    # 环境变量非法时回退默认并发 64
    SANDBOX_CONCURRENCY = 64
if SANDBOX_CONCURRENCY < 0:
    # 负数同样回退默认并发 64
    SANDBOX_CONCURRENCY = 64

# Toggle score2reward transformation
# 是否启用 score->reward 的映射变换(默认开启;关闭则奖励等于原始分数)
USE_SCORE2REWARD = os.getenv("USE_SCORE2REWARD", "1") not in ("0")
# 动态奖励映射模式(power_clip 等):决定分数到奖励的曲线形状
REWARD_MAPPING_MODE = os.getenv("REWARD_MAPPING_MODE", "power_clip").strip().lower()
# 静态奖励映射模式:使用固定上下界时的映射曲线形状
STATIC_REWARD_MAPPING_MODE = os.getenv("STATIC_REWARD_MAPPING_MODE", "power_clip").strip().lower()
# 是否要求代码自验证(必须打印 Final Validation Score),默认关闭
ENABLE_SELF_VALIDATION = os.getenv("ENABLE_SELF_VALIDATION", "0").strip().lower() not in (
    "0",
    "false",
    "no",
    "",
)

# Reward shaping experiments (configured via env vars)
# - IMPROVE_REWARD_STRATEGY:
#     - base: (default) improve reward = effective_base_reward (subject to basic failure checks)
#     - gate_parent: if effective_base_reward >= parent_reward -> effective_base_reward else 0
#     - diff_parent: max(effective_base_reward - parent_reward, 0)
# 改进型奖励(improve/debug/crossover 相对父代)的策略选择,取值 base/gate_parent/diff_parent
IMPROVE_REWARD_STRATEGY = os.getenv("IMPROVE_REWARD_STRATEGY", "base").strip().lower()
# 改进增量(delta)的额外奖励系数:final = shaped_base + coef * delta
IMPROVE_DELTA_BONUS_COEF = float(os.getenv("IMPROVE_DELTA_BONUS_COEF", "0.5"))
# 是否启用动态分数边界:按"当前组内分数+历史分数"自适应确定奖励映射范围
ENABLE_DYNAMIC_SCORE_BOUNDS = os.getenv("ENABLE_DYNAMIC_SCORE_BOUNDS", "0").strip().lower() not in (
    "0",
    "false",
    "no",
    "",
)
# 动态边界的选取模式(如 top1_top8:组内/历史最优与 top8 分位)
ADAPTIVE_REWARD_BOUND_MODE = os.getenv("ADAPTIVE_REWARD_BOUND_MODE", "top1_top8").strip().lower()
if ADAPTIVE_REWARD_BOUND_MODE not in SUPPORTED_ADAPTIVE_BOUND_MODES:
    # 配置了不支持的边界模式:记录告警并回退到理论边界模式
    logger.warning(
        "Unsupported ADAPTIVE_REWARD_BOUND_MODE=%s. Falling back to %s.",
        ADAPTIVE_REWARD_BOUND_MODE,
        BOUND_MODE_THEORETICAL,
    )
    ADAPTIVE_REWARD_BOUND_MODE = BOUND_MODE_THEORETICAL
# 动态边界的最小跨度,防止上下界过近导致奖励放大失真
DYNAMIC_SCORE_BOUND_MIN_SPAN = float(os.getenv("DYNAMIC_SCORE_BOUND_MIN_SPAN", "1e-6"))
try:
    # 动态下界向下平移比例(把下界再往下压,缓解负奖励过强)
    ADAPTIVE_REWARD_BOUND_LOWER_SHIFT_RATIO = float(os.getenv("ADAPTIVE_REWARD_BOUND_LOWER_SHIFT_RATIO", "0.0"))
except (TypeError, ValueError):
    # 非法值回退为 0(不平移)
    ADAPTIVE_REWARD_BOUND_LOWER_SHIFT_RATIO = 0.0
if not math.isfinite(ADAPTIVE_REWARD_BOUND_LOWER_SHIFT_RATIO):
    # NaN/Inf 等非有限值回退为 0
    ADAPTIVE_REWARD_BOUND_LOWER_SHIFT_RATIO = 0.0
# 平移比例不允许为负,截断到 0
ADAPTIVE_REWARD_BOUND_LOWER_SHIFT_RATIO = max(ADAPTIVE_REWARD_BOUND_LOWER_SHIFT_RATIO, 0.0)
# 组内只有一个有限分数时,是否直接把该分数映射为奖励 1(而不是走静态回退)
ADAPTIVE_BOUND_SINGLE_FINITE_REWARD_ONE = os.getenv("ADAPTIVE_BOUND_SINGLE_FINITE_REWARD_ONE", "0").strip().lower() not in (
    "0",
    "false",
    "no",
    "",
)
# 静态边界缺失时是否用常量兜底(best/worst),否则视为无边界
METRIC_STATIC_BOUND_USE_CONST_FALLBACK = os.getenv("METRIC_STATIC_BOUND_USE_CONST_FALLBACK", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "",
)
# 静态兜底的上界常量(leaderboard best 缺失时使用)
METRIC_STATIC_FALLBACK_SIGNED_BEST = float(os.getenv("METRIC_STATIC_FALLBACK_SIGNED_BEST", "100.0"))
# 静态兜底的下界常量(leaderboard worst 缺失时使用)
METRIC_STATIC_FALLBACK_SIGNED_WORST = float(os.getenv("METRIC_STATIC_FALLBACK_SIGNED_WORST", "-100.0"))

# MLE 生成/搜索核心配置:沙箱并发、程序数据库路径与容量、四算子(draft/improve/debug/crossover)
# 触发概率、搜索算法(greedy/evo/airaevo)及 airaevo 的岛屿进化参数
MLE_CONFIGS = {
    # Sandbox execution concurrency limit
    "sandbox_concurrency": SANDBOX_CONCURRENCY,
    # Program database settings
    "db_path": os.environ.get("PROGRAM_DB_PATH", "program_database.db"),
    "max_programs_per_task": int(os.environ.get("MAX_PROGRAMS_PER_TASK", "0")),
    "draft_probability": float(os.environ.get("DRAFT_PROBABILITY", "0.5")),  # 2:1:1 => draft=0.5
    "improve_probability": float(os.environ.get("IMPROVE_PROBABILITY", "0.25")),
    "debug_probability": float(os.environ.get("DEBUG_PROBABILITY", "0.25")),  # 2:1:1 => debug=0.25
    "crossover_probability": float(os.environ.get("CROSSOVER_PROBABILITY", "0.5")),
    "search_algorithm": os.environ.get("SEARCH_ALGORITHM", "evo"),  # 'greedy' | 'evo' | 'airaevo'
    "airaevo_policy": os.environ.get("AIRAEVO_POLICY", "rl_mixed"),
    "airaevo_num_islands": int(os.environ.get("AIRAEVO_NUM_ISLANDS", "1")),
    "airaevo_max_island_size": int(os.environ.get("AIRAEVO_MAX_ISLAND_SIZE", "500")),
    "airaevo_crossover_after_generation": int(os.environ.get("AIRAEVO_CROSSOVER_AFTER_GENERATION", "2")),
    "airaevo_execution_timeout": int(os.environ.get("AIRAEVO_EXECUTION_TIMEOUT", os.environ.get("JOB_TIMEOUT", "3600"))),
}



# Async primitives are bound lazily to the event loop that first waits on them.
# SLIME uses separate loops for train rollout and eval rollout in the same
# process, so keep these primitives loop-local.
# 全局注册表总锁:保护下面三个弱引用字典的读写
_LOOP_PRIMITIVES_LOCK = threading.Lock()
# 每个事件循环专属的 asyncio.Lock 集合(loop -> {名字: Lock})
_LOOP_LOCKS = weakref.WeakKeyDictionary()
# 每个事件循环专属的 asyncio.Semaphore 集合(loop -> {名字: Semaphore})
_LOOP_SEMAPHORES = weakref.WeakKeyDictionary()
# 每个任务(task_key)专属的动态边界注册锁,避免并发写组奖励屏障缓存
_DYNAMIC_SCORE_BOUND_TASK_LOCKS = weakref.WeakKeyDictionary()


def _get_loop_lock(name: str) -> asyncio.Lock:
    # 取当前事件循环内指定名字的 asyncio.Lock(按 loop 维度惰性创建/复用)
    loop = asyncio.get_running_loop()
    # 加全局同步锁保护注册表读写(临界区很短)
    with _LOOP_PRIMITIVES_LOCK:
        # 查询该 loop 的锁集合,不存在则初始化
        loop_locks = _LOOP_LOCKS.get(loop)
        if loop_locks is None:
            loop_locks = {}
            _LOOP_LOCKS[loop] = loop_locks
        # 查询指定名字的锁,不存在则新建并登记
        lock = loop_locks.get(name)
        if lock is None:
            lock = asyncio.Lock()
            loop_locks[name] = lock
        # 返回该 loop 内全局唯一的名叫 name 的锁
        return lock


def _get_loop_semaphore(name: str, value: int) -> asyncio.Semaphore:
    # 取当前事件循环内指定名字的信号量(按 loop 维度惰性创建/复用)
    loop = asyncio.get_running_loop()
    with _LOOP_PRIMITIVES_LOCK:
        # 查询该 loop 的信号量集合,不存在则初始化
        loop_semaphores = _LOOP_SEMAPHORES.get(loop)
        if loop_semaphores is None:
            loop_semaphores = {}
            _LOOP_SEMAPHORES[loop] = loop_semaphores
        # 查询指定名字的信号量,不存在则按 value 新建并登记
        semaphore = loop_semaphores.get(name)
        if semaphore is None:
            semaphore = asyncio.Semaphore(value)
            loop_semaphores[name] = semaphore
        # 返回该 loop 内全局唯一的信号量
        return semaphore


def _get_sandbox_semaphore() -> asyncio.Semaphore:
    # 获取沙箱执行限流信号量(并发上限取自 MLE_CONFIGS)
    return _get_loop_semaphore("sandbox", MLE_CONFIGS["sandbox_concurrency"])


# Counter to track concurrent sandbox executions
# 当前正在执行沙箱任务的协程计数(用于日志展示并发水位)
SANDBOX_CONCURRENT_COUNT = 0


# 训练期沙箱作业超时(秒)
JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT", "300"))
# 训练期轮询沙箱结果的最长等待(秒)
WAIT_TIMEOUT = int(os.getenv("WAIT_TIMEOUT", "1800"))
# 评估期的沙箱作业超时(默认复用训练值)
EVAL_JOB_TIMEOUT = int(os.getenv("EVAL_JOB_TIMEOUT", str(JOB_TIMEOUT)))
# 评估期的结果等待超时(默认复用训练值)
EVAL_WAIT_TIMEOUT = int(os.getenv("EVAL_WAIT_TIMEOUT", str(WAIT_TIMEOUT)))
# 评估期生成被 abort 后的最大重试次数
EVAL_GENERATION_ABORT_MAX_RETRIES = int(os.getenv("EVAL_GENERATION_ABORT_MAX_RETRIES", "60"))
# 评估期 abort 重试之间的休眠秒数
EVAL_GENERATION_ABORT_RETRY_SLEEP = float(os.getenv("EVAL_GENERATION_ABORT_RETRY_SLEEP", "2.0"))
# 是否经代理访问沙箱(否则直连)
USE_PROXY_SANDBOX = os.getenv("USE_PROXY_SANDBOX", "0").strip().lower() not in ("0", "false", "no", "")
# 是否把发给 LLM 的完整 prompt 落盘为文本文件(调试观察四算子 prompt 用)
ENABLE_PROMPT_DUMP = os.getenv("ENABLE_PROMPT_DUMP", "0").strip().lower() not in ("0", "false", "no", "")
# prompt 落盘的根目录(默认为本文件同级的 test 目录)
PROMPT_DUMP_BASE_DIR = os.getenv(
    "PROMPT_DUMP_BASE_DIR",
    str((Path(__file__).resolve().parent / "test").resolve()),
)
# 本次运行在落盘根目录下的子目录名(留空则按时间戳自动生成)
PROMPT_DUMP_RUN_SUBDIR = os.getenv("PROMPT_DUMP_RUN_SUBDIR", "").strip()
# 是否为 airaevo 启用"丰富记忆"摘要(LLM 总结方法与父代对比经验)
AIRAEVO_RICH_MEMORY_ENABLED = os.getenv("AIRAEVO_RICH_MEMORY_ENABLED", "1").strip().lower() not in ("0", "false", "no", "")
# 丰富记忆摘要生成允许的最大新 token 数
AIRAEVO_RICH_MEMORY_MAX_TOKENS = int(os.getenv("AIRAEVO_RICH_MEMORY_MAX_TOKENS", "512"))
# 丰富记忆摘要请求的超时秒数
AIRAEVO_RICH_MEMORY_TIMEOUT = float(os.getenv("AIRAEVO_RICH_MEMORY_TIMEOUT", "120"))

# Global program database instance
# 全局程序数据库(单例,存放所有已执行程序及其分数/奖励,进化搜索的记忆库)
_program_database = None

# Global search algorithm instance
# 全局搜索算法实例(单例:greedy/evo/airaevo 之一,决定四算子如何调度)
_search_algorithm = None

# Global cache for parent selection results per group
# Key: group_index, Value: {
#   'data': (input_messages, parent_program, mode),
#   'access_count': int,  # How many times this cache has been accessed
#   'total_samples': int  # Expected total samples in this group (n_samples_per_prompt)
# }
_parent_selection_cache = {}

# 各任务历史最优静态基础奖励缓存(task_name -> 历史最优值)
_history_static_base_reward_cache: dict[str, float] = {}
# prompt 落盘目录缓存(惰性创建后复用)
_prompt_dump_dir: Path | None = None
# 组级自适应奖励屏障缓存:组键 -> {期望成员数, 各成员分数与 Future}
_adaptive_group_reward_cache: dict[tuple[str, str], dict[str, object]] = {}


async def _ensure_prompt_dump_dir() -> Path:
    # 惰性创建并缓存本次运行的 prompt 落盘目录(双检锁,避免重复创建)
    global _prompt_dump_dir
    # 已创建过则直接复用
    if _prompt_dump_dir is not None:
        return _prompt_dump_dir
    # 加事件循环级锁,保证并发协程只创建一次
    async with _get_loop_lock("prompt_dump"):
        # 双检:拿到锁后可能已被其他协程创建
        if _prompt_dump_dir is not None:
            return _prompt_dump_dir
        # 运行子目录名:优先用配置,否则按当前时间戳生成
        run_name = PROMPT_DUMP_RUN_SUBDIR or f"prompt_dump_{int(time.time())}"
        # 拼出完整输出目录路径
        out_dir = Path(PROMPT_DUMP_BASE_DIR) / run_name
        # 递归创建目录(已存在则忽略)
        out_dir.mkdir(parents=True, exist_ok=True)
        # 缓存到全局,后续调用直接复用
        _prompt_dump_dir = out_dir
        # 记录落盘目录路径
        logger.info("[PROMPT_DUMP] output_dir=%s", str(out_dir))
        # 返回该目录
        return _prompt_dump_dir


async def _dump_prompt_to_txt(
    *,
    task_name: str,
    task_id: str,
    group_index: int | None,
    mode: str,
    parent_program: Program | None,
    secondary_parent_program: Program | None,
    system_prompt: str,
    user_prompt: str,
) -> None:
    # 将本次请求的 system/user prompt 及元信息落盘为 txt(调试观察四算子 prompt 用)
    if not ENABLE_PROMPT_DUMP:
        # 未开启落盘开关则直接返回
        return
    # 确保落盘目录存在
    out_dir = await _ensure_prompt_dump_dir()
    # 以毫秒时间戳作为文件名前缀,避免同名覆盖
    ts_ms = int(time.time() * 1000)
    # 清洗任务名:仅保留字母数字与 -_,其余替换为下划线,并截断到 64 字符
    safe_task = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (task_name or "unknown"))[:64]
    # 组索引为空时用 "none" 占位
    gid = "none" if group_index is None else str(group_index)
    # 生成唯一文件名:时间戳_任务_模式_组号
    file_path = out_dir / f"{ts_ms}_{safe_task}_{mode}_g{gid}.txt"
    # 组装落盘内容:元信息头 + system prompt + user prompt
    content = [
        f"task_name: {task_name}",
        f"task_id: {task_id}",
        f"group_index: {group_index}",
        f"generation_mode: {mode}",
        f"parent_id: {getattr(parent_program, 'id', None)}",
        f"secondary_parent_id: {getattr(secondary_parent_program, 'id', None)}",
        "",
        "===== SYSTEM PROMPT =====",
        system_prompt,
        "",
        "===== USER PROMPT =====",
        user_prompt,
        "",
    ]
    file_path.write_text("\n".join(content), encoding="utf-8")


def _is_finite_number(value) -> bool:
    # 判断 value 是否为有限数值(非 None 且可转 float 且非 NaN/Inf)
    try:
        return value is not None and math.isfinite(float(value))
    except Exception:
        # 转换失败(如非法类型)一律视为非有限
        return False


def _dynamic_bound_task_key(metadata: dict) -> str:
    # 从 metadata 依优先级取任务标识,作为动态边界/组缓存的键
    for key in ("task_id", "uuid", "data_dir", "task_name"):
        # 依次尝试各候选键,取第一个非空值
        value = metadata.get(key)
        if value:
            return str(value)
    # 全部缺失时返回占位标识
    return "unknown_task"


def _dynamic_bound_task_identity(metadata: dict) -> tuple[str, str, str]:
    # 返回任务三元组:(缓存键, 任务ID, 任务名),用于查询历史分数
    task_key = _dynamic_bound_task_key(metadata)
    # 任务 ID 优先取 task_id,缺失时退回 uuid
    task_id = str(metadata.get("task_id") or metadata.get("uuid") or "")
    # 任务名缺失时用 "unknown" 占位
    task_name = str(metadata.get("task_name") or "unknown")
    return task_key, task_id, task_name


async def _get_dynamic_bound_task_lock(task_key: str) -> asyncio.Lock:
    # 获取某任务专属的注册锁:同一任务的组奖励屏障串行化,不同任务互不阻塞
    loop = asyncio.get_running_loop()
    # 先拿注册表级异步锁,避免并发创建同一 task_key 的锁
    async with _get_loop_lock("dynamic_score_bound_registry"):
        # 在全局同步锁内读取该 loop 的任务锁表
        with _LOOP_PRIMITIVES_LOCK:
            task_locks = _DYNAMIC_SCORE_BOUND_TASK_LOCKS.get(loop)
            if task_locks is None:
                # 该 loop 尚无任务锁表则初始化
                task_locks = {}
                _DYNAMIC_SCORE_BOUND_TASK_LOCKS[loop] = task_locks
            # 查询该任务的锁(可能为 None)
            lock = task_locks.get(task_key)
        if lock is None:
            # 该任务还没有锁:新建(短暂持有全局锁写入)
            lock = asyncio.Lock()
            with _LOOP_PRIMITIVES_LOCK:
                task_locks = _DYNAMIC_SCORE_BOUND_TASK_LOCKS.get(loop)
                if task_locks is None:
                    # 双检:极小概率表被清空,重建
                    task_locks = {}
                    _DYNAMIC_SCORE_BOUND_TASK_LOCKS[loop] = task_locks
                # 登记任务锁
                task_locks[task_key] = lock
        # 返回任务专属锁
        return lock


def _group_cache_key(task_key: str, group_index: int | None) -> tuple[str, str] | None:
    # 构造组缓存键 (任务键, 组索引);无组索引时返回 None 表示不参与组缓存
    if group_index is None:
        return None
    return (task_key, str(group_index))


def _use_group_adaptive_reward_bounds() -> bool:
    # 是否走组级自适应奖励边界:需同时开启动态边界、score2reward 且映射模式为 power_clip
    return ENABLE_DYNAMIC_SCORE_BOUNDS and USE_SCORE2REWARD and REWARD_MAPPING_MODE == "power_clip"


def _build_adaptive_bound_metadata_context(
    *,
    task_key: str,
    group_index: int | None,
    history_count: int,
    group_count: int,
    best_signed: float | None,
    worst_signed: float | None,
    group_generation_aborted: bool = False,
) -> dict[str, float | int | str | bool | None]:
    # 汇总动态边界上下文(写入样本 metadata,便于训练侧审计奖励来源与边界依据)
    return {
        "enabled": _is_finite_number(best_signed) and _is_finite_number(worst_signed),
        "task_key": task_key,
        "group_index": group_index,
        "bound_mode": ADAPTIVE_REWARD_BOUND_MODE,
        "lower_shift_ratio": ADAPTIVE_REWARD_BOUND_LOWER_SHIFT_RATIO,
        "single_finite_reward_one": ADAPTIVE_BOUND_SINGLE_FINITE_REWARD_ONE,
        "history_score_count_before": int(history_count),
        "group_score_count": int(group_count),
        "history_score_count_after_including_group": int(history_count) + int(group_count),
        "best_signed": float(best_signed) if _is_finite_number(best_signed) else None,
        "worst_signed": float(worst_signed) if _is_finite_number(worst_signed) else None,
        "group_generation_aborted": bool(group_generation_aborted),
    }


async def _resolve_group_reward_mapping(
    metadata: dict,
    *,
    group_index: int | None,
    group_size: int,
    sample_key: str,
    score: float | None,
    base_reward: float,
) -> dict[str, object]:
    # 组级奖励映射屏障:同组所有样本到齐后,统一用"组内分数+历史分数"确定动态边界并映射奖励,
    # 保证 GRPO 组内优势计算基于一致的奖励上下界(先凑齐 execution-grounded 分数再统一换算)。
    task_key, task_id, task_name = _dynamic_bound_task_identity(metadata)
    # 本样本是否为生成中断(整组需重排重试)
    generation_aborted = bool(metadata.get("generation_aborted"))
    # 预构造"未启用动态映射"时的默认上下文(计数按当前单个样本计)
    default_context = _build_adaptive_bound_metadata_context(
        task_key=task_key,
        group_index=group_index,
        history_count=0,
        group_count=1 if _is_finite_number(score) else 0,
        best_signed=None,
        worst_signed=None,
        group_generation_aborted=generation_aborted,
    )
    # 默认结果:直接沿用传入的 base_reward,不做组级映射
    default_result: dict[str, object] = {
        "base_reward": float(base_reward),
        "context": default_context,
    }

    if not _use_group_adaptive_reward_bounds():
        # 未启用组级动态边界:若有有效分数则用普通 score2reward 映射
        if USE_SCORE2REWARD and _is_finite_number(score):
            mapped_reward = score2reward(float(score), dict(metadata), mode=REWARD_MAPPING_MODE)
            return {"base_reward": float(mapped_reward), "context": default_context}
        # 否则原样返回默认结果
        return default_result

    # 分数有效才允许参与组内动态映射
    allow_mapping = _is_finite_number(score)
    # 获取全局程序数据库(查询该任务的历史分数)
    db = await get_program_database()

    # 元数据中的显式奖励组 ID(评估期可能由数据源预先分组)
    reward_group_id = metadata.get("reward_group_id")
    # 该奖励组期望的成员数量
    reward_group_expected_count = metadata.get("reward_group_expected_count")
    # 本样本在奖励组内的成员序号
    reward_group_member_index = metadata.get("reward_group_member_index")

    if group_index is None and not reward_group_id:
        # 无组信息(如单样本评估):直接用历史分数做自适应边界映射
        historical_scores = db.get_task_scores(task_id=task_id, task_name=task_name)
        mapped_reward, context = score_to_group_adaptive_reward(
            score=score,
            metadata=metadata,
            historical_scores=historical_scores,
            current_group_scores=[score] if allow_mapping else [],
            mode=ADAPTIVE_REWARD_BOUND_MODE,
            reward_mapping_mode=REWARD_MAPPING_MODE,
            use_score2reward=allow_mapping,
            default_reward=base_reward,
            min_span=DYNAMIC_SCORE_BOUND_MIN_SPAN,
            lower_shift_ratio=ADAPTIVE_REWARD_BOUND_LOWER_SHIFT_RATIO,
            single_finite_reward_one=ADAPTIVE_BOUND_SINGLE_FINITE_REWARD_ONE,
        )
        context = _build_adaptive_bound_metadata_context(
            task_key=task_key,
            group_index=group_index,
            history_count=len(historical_scores),
            group_count=1 if allow_mapping else 0,
            best_signed=context.get("best_signed"),
            worst_signed=context.get("worst_signed"),
            group_generation_aborted=generation_aborted,
        )
        # 返回映射后的奖励与补全后的上下文
        return {"base_reward": float(mapped_reward), "context": context}

    if reward_group_id:
        # 显式奖励组模式:缓存键与锁键都以组 ID 为准
        group_key = ("reward_group", str(reward_group_id))
        group_lock_key = f"reward_group:{reward_group_id}"
        try:
            # 期望成员数取元数据声明值,非法时回退为组大小
            expected_count = max(int(reward_group_expected_count or group_size or 1), 1)
        except Exception:
            # 转换失败同样回退为组大小(至少为 1)
            expected_count = max(int(group_size or 1), 1)
        if reward_group_member_index is not None:
            # 有成员序号时用 组ID:序号 作为样本键,保证组内唯一
            sample_key = f"{reward_group_id}:{reward_group_member_index}"
    else:
        # 常规模式:缓存键按(任务键,组索引),锁按任务键粒度
        group_key = _group_cache_key(task_key, group_index)
        group_lock_key = task_key
        expected_count = max(int(group_size or 1), 1)

    if group_key is None:
        # 无有效组键(无组索引)时直接返回默认结果
        return default_result

    # 获取该任务/组的注册锁,串行化对组屏障缓存的修改
    task_lock = await _get_dynamic_bound_task_lock(group_lock_key)
    # 为本样本创建 Future:组到齐后由"释放者"填充映射结果
    loop = asyncio.get_running_loop()
    waiter = loop.create_future()

    async with task_lock:
        # 查询该组的屏障缓存条目
        entry = _adaptive_group_reward_cache.get(group_key)
        if entry is None:
            # 首个到达的样本:创建条目并登记期望成员数
            entry = {
                "expected_count": expected_count,
                "items": {},
                "seen_count": 0,
            }
            _adaptive_group_reward_cache[group_key] = entry

        # 查询同 key 样本是否已登记(防重复)
        existing_item = entry["items"].get(sample_key)
        if existing_item is not None:
            # 重复样本:记录告警并复用已登记样本的 Future(不重复参与计数)
            logger.warning(
                "[GROUP REWARD BARRIER] duplicate sample ignored group_key=%s sample_key=%s member=%s seen=%s/%s",
                group_key,
                sample_key,
                reward_group_member_index,
                entry.get("seen_count", 0),
                entry.get("expected_count"),
            )
            waiter = existing_item["future"]
        else:
            # 首次登记该样本:保存分数/基础奖励/是否可映射/元数据/中断标志及其 Future
            entry["items"][sample_key] = {
                "score": score,
                "base_reward": float(base_reward),
                "allow_mapping": bool(allow_mapping),
                "metadata": dict(metadata),
                "generation_aborted": generation_aborted,
                "future": waiter,
            }
            # 更新已到样本计数
            entry["seen_count"] = len(entry["items"])
            logger.info(
                "[GROUP REWARD BARRIER] enter group_key=%s sample_key=%s member=%s seen=%s/%s task_key=%s group_index=%s score=%s code_category=%s",
                group_key,
                sample_key,
                reward_group_member_index,
                entry.get("seen_count", 0),
                entry.get("expected_count"),
                task_key,
                group_index,
                score,
                metadata.get("code_category"),
            )

        if int(entry["seen_count"]) >= int(entry["expected_count"]):
            # 组内样本已到齐:由最后一个到达者统一计算全组奖励并释放所有等待者
            try:
                # 拉取该任务的历史分数(参与动态边界计算)
                historical_scores = db.get_task_scores(task_id=task_id, task_name=task_name)
                # 收集组内允许映射且分数有效的样本分数
                current_group_scores = [
                    float(item["score"])
                    for item in entry["items"].values()
                    if bool(item.get("allow_mapping")) and _is_finite_number(item.get("score"))
                ]
                # 组内任一样本发生生成中断则整组标记为中断
                group_generation_aborted = any(
                    bool(item.get("generation_aborted")) for item in entry["items"].values()
                )
                # 无历史分数且组内有效分数<=1 时退回静态边界映射
                use_static_fallback = len(historical_scores) == 0 and len(current_group_scores) <= 1
                if ADAPTIVE_BOUND_SINGLE_FINITE_REWARD_ONE and len(current_group_scores) == 1:
                    # 配置了"单有限分数映射为1"时不做静态回退
                    use_static_fallback = False

                # 全组统一的动态边界(以最后一个有效计算为准)
                resolved_best_signed = None
                resolved_worst_signed = None
                resolved_any_dynamic_bound = False
                # 收集(样本条目, 映射后奖励),稍后统一回填 Future
                resolved_items: list[tuple[dict[str, object], float]] = []
                for item in entry["items"].values():
                    # 逐个样本计算映射奖励
                    item_metadata = item.get("metadata") or metadata
                    if use_static_fallback and bool(item.get("allow_mapping")) and _is_finite_number(
                        item.get("score")
                    ):
                        # 静态回退路径:用固定上下界映射该样本分数
                        mapped_reward = compute_static_score_reward(float(item["score"]), item_metadata)
                        bound_context = {"best_signed": None, "worst_signed": None}
                    else:
                        # 动态路径:基于历史+组内分数计算自适应边界并映射
                        mapped_reward, bound_context = score_to_group_adaptive_reward(
                            score=item.get("score"),
                            metadata=item_metadata,
                            historical_scores=historical_scores,
                            current_group_scores=current_group_scores,
                            mode=ADAPTIVE_REWARD_BOUND_MODE,
                            reward_mapping_mode=REWARD_MAPPING_MODE,
                            use_score2reward=bool(item.get("allow_mapping")),
                            default_reward=float(item.get("base_reward", 0.0)),
                            min_span=DYNAMIC_SCORE_BOUND_MIN_SPAN,
                            lower_shift_ratio=ADAPTIVE_REWARD_BOUND_LOWER_SHIFT_RATIO,
                            single_finite_reward_one=ADAPTIVE_BOUND_SINGLE_FINITE_REWARD_ONE,
                        )
                    best_signed = bound_context.get("best_signed")
                    worst_signed = bound_context.get("worst_signed")
                    if _is_finite_number(best_signed) and _is_finite_number(worst_signed):
                        # 记录有效的动态边界,供全组上下文使用
                        resolved_best_signed = float(best_signed)
                        resolved_worst_signed = float(worst_signed)
                        resolved_any_dynamic_bound = True
                    # 暂存该样本的映射结果
                    resolved_items.append((item, float(mapped_reward)))
                # 构造全组共享的上下文模板(写入各样本 metadata)
                context_template = _build_adaptive_bound_metadata_context(
                    task_key=task_key,
                    group_index=group_index,
                    history_count=len(historical_scores),
                    group_count=len(current_group_scores),
                    best_signed=resolved_best_signed if resolved_any_dynamic_bound else None,
                    worst_signed=resolved_worst_signed if resolved_any_dynamic_bound else None,
                    group_generation_aborted=group_generation_aborted,
                )
                logger.info(
                    "[GROUP REWARD BARRIER] release group_key=%s unique_items=%s expected=%s finite_scores=%s aborted=%s",
                    group_key,
                    len(entry["items"]),
                    entry.get("expected_count"),
                    len(current_group_scores),
                    group_generation_aborted,
                )
                for item, mapped_reward in resolved_items:
                    # 逐个唤醒等待中的样本协程,回填其映射结果
                    item["future"].set_result({"base_reward": mapped_reward, "context": dict(context_template)})
            except Exception as exc:
                logger.exception("[GROUP REWARD BARRIER] release failed group_key=%s", group_key)
                # 只处理异常路径：让同组所有等待者收到同一个错误，避免永久等待。
                for item in entry["items"].values():
                    future = item["future"]
                    if not future.done():
                        future.set_exception(exc)
            finally:
                # 正常和异常路径都必须清理组缓存，避免失败组永久残留。
                _adaptive_group_reward_cache.pop(group_key, None)

    # 阻塞等待本样本的映射结果(组到齐后被释放;重复样本复用同一 Future)
    return await waiter


def _apply_dynamic_bound_context_metadata(sample: Sample, context: dict, expected_group_count: int) -> None:
    # 把动态边界上下文逐项写入样本 metadata(供训练侧与 CSV 审计奖励来源)
    # 是否成功启用动态边界
    sample.metadata["dynamic_score_bounds_enabled"] = bool(context.get("enabled"))
    # 任务缓存键
    sample.metadata["dynamic_score_bounds_task_key"] = context.get("task_key")
    # 组索引
    sample.metadata["dynamic_score_bounds_group_index"] = context.get("group_index")
    # 组期望成员数
    sample.metadata["dynamic_score_bounds_group_expected_count"] = expected_group_count
    # 动态边界模式(top1_top8/theoretical 等)
    sample.metadata["adaptive_reward_bound_mode"] = context.get("bound_mode")
    # 参与边界计算的映射前历史分数数量
    sample.metadata["adaptive_reward_history_score_count_before"] = context.get("history_score_count_before")
    # 组内参与映射的分数数量
    sample.metadata["adaptive_reward_group_score_count"] = context.get("group_score_count")
    # 映射后的历史+组内总分数数量
    sample.metadata["adaptive_reward_history_score_count_after"] = context.get("history_score_count_after_including_group")
    # 本组确定的奖励上界(带符号分数)
    sample.metadata["dynamic_bound_best_signed"] = context.get("best_signed")
    # 本组确定的奖励下界(带符号分数)
    sample.metadata["dynamic_bound_worst_signed"] = context.get("worst_signed")
    # 本组是否发生了生成中断
    sample.metadata["group_generation_aborted"] = bool(context.get("group_generation_aborted"))


def _return_generation_abort_for_retry(sample: Sample, *, task_name: str, group_index: int | None) -> float:
    # 标记样本为"生成中断"并返回占位奖励 -1,交由框架重排该组(不进数据库/CSV)
    # 记录中断标志,等待框架重排
    sample.metadata["generation_aborted"] = True
    # 缺省中断原因为整组中止
    sample.metadata.setdefault("generation_abort_reason", "group_generation_aborted")
    # 同时标记整组中断
    sample.metadata["group_generation_aborted"] = True
    # 代码类别记为 generation_abort
    sample.metadata["code_category"] = "generation_abort"
    # 以下奖励字段统一置占位值 -1(不参与真实训练)
    sample.metadata["base_reward"] = -1.0
    sample.metadata["final_reward"] = -1.0
    sample.metadata["dynamic_raw_reward"] = -1.0
    sample.metadata["static_raw_reward"] = -1.0
    # 分数置空、运行时间清零
    sample.metadata["score"] = None
    sample.metadata["running_time"] = 0.0
    # 状态置为 ABORTED,请求框架重排
    sample.status = Sample.Status.ABORTED
    logger.warning(
        "[GENERATION ABORT] Requeueing group before DB/CSV logging: task=%s group_index=%s reason=%s",
        task_name,
        group_index,
        sample.metadata.get("generation_abort_reason"),
    )
    # 返回占位奖励 -1
    return -1.0


def compute_mode_rewards(
    *,
    generation_mode: str,
    effective_base_reward: float,
    parent_base_reward: float | None,
    parent_reward: float | None,
    parent_code: str,
    code: str,
    crossover_parent_base_reward: float | None = None,
    crossover_parent_reward: float | None = None,
    crossover_parent_code: str = "",
    improve_reward_strategy: str = "base",
    improve_delta_bonus_coef: float = 0.5,
) -> tuple[float, float, float, float]:
    """
    Compute shaped base reward and final reward for draft/improve/debug/crossover.

    Returns:
        (shaped_base_reward, final_reward, delta, improve_bonus)
    """
    # 参考父代奖励:优先用父代的 base_reward,缺失时退回父代 final reward
    reference_parent_reward = parent_base_reward if parent_base_reward is not None else parent_reward
    # 交叉父代(crossover 的第二父代)参考奖励,同样优先 base_reward
    reference_crossover_parent_reward = (
        crossover_parent_base_reward if crossover_parent_base_reward is not None else crossover_parent_reward
    )

    # 改进型算子(improve/debug/crossover,四算子中的三个)且有父代参考时做相对奖励整形
    if generation_mode in ("improve", "debug", "crossover") and reference_parent_reward is not None:
        # 基础奖励为 0(执行失败/无产出)时全部归零
        if effective_base_reward == 0.0:
            return 0.0, 0.0, 0.0, 0.0
        # 与父代(或交叉父代)代码完全相同视为无贡献,不奖励"复制粘贴"
        if parent_code == code or (generation_mode == "crossover" and crossover_parent_code == code):
            return 0.0, 0.0, 0.0, 0.0

        # 取父代参考奖励为浮点数
        ref_parent = float(reference_parent_reward)
        if generation_mode == "crossover" and reference_crossover_parent_reward is not None:
            # 交叉模式取两个父代中更优者作参照,鼓励超越较强的父代
            ref_parent = max(ref_parent, float(reference_crossover_parent_reward))

        # 读取改进奖励策略(base/gate_parent/diff_parent)
        strategy = (improve_reward_strategy or "base").strip().lower()
        if strategy == "gate_parent":
            # 门控策略:只有不差于父代才给分,否则 0
            shaped_base_reward = effective_base_reward if effective_base_reward >= ref_parent else 0.0
        elif strategy == "diff_parent":
            # 差分策略:只奖励超出父代的部分
            shaped_base_reward = max(effective_base_reward - ref_parent, 0.0)
        else:
            # 默认 base 策略:直接使用有效基础奖励
            shaped_base_reward = effective_base_reward

        # 相对父代的改进量(下限 0)
        delta = max(effective_base_reward - ref_parent, 0.0)
        # 改进加成 = 系数 × 改进量
        improve_bonus = float(improve_delta_bonus_coef) * delta
        # 最终奖励 = 整形后基础奖励 + 改进加成
        final_reward = shaped_base_reward + improve_bonus
        # 返回四元组(基础奖励, 最终奖励, 改进量, 加成)
        return float(shaped_base_reward), float(final_reward), float(delta), float(improve_bonus)

    # draft mode or no parent reference
    # draft 算子(或无父代参考):奖励即基础奖励,无改进项
    shaped_base_reward = float(effective_base_reward)
    return shaped_base_reward, shaped_base_reward, 0.0, 0.0


def compute_static_score_reward(score: float | None, metadata: dict) -> float:
    """Map score to the original static bounds reward, independent of dynamic reward mode."""
    # 无分数(未执行/失败)映射为 0
    if score is None:
        return 0.0
    try:
        # 转换为浮点分数
        score_f = float(score)
    except (TypeError, ValueError):
        # 无法转换则按 0 处理
        return 0.0
    if not math.isfinite(score_f):
        # NaN/Inf 视为无效,按 0 处理
        return 0.0

    if not USE_SCORE2REWARD:
        # 未开启映射变换时奖励即分数本身
        return score_f

    # 复制 metadata 并剔除动态边界字段,确保只用静态边界
    reward_metadata = dict(metadata)
    reward_metadata.pop("dynamic_bound_best_signed", None)
    reward_metadata.pop("dynamic_bound_worst_signed", None)
    # 静态上界兜底常量(按配置决定是否启用)
    fallback_best = METRIC_STATIC_FALLBACK_SIGNED_BEST if METRIC_STATIC_BOUND_USE_CONST_FALLBACK else None
    # 静态下界兜底常量
    fallback_worst = METRIC_STATIC_FALLBACK_SIGNED_WORST if METRIC_STATIC_BOUND_USE_CONST_FALLBACK else None
    # 优先使用 leaderboard(榜单)边界的静态映射结果
    return float(
        score2reward_with_static_priority(
            score_f,
            reward_metadata,
            mode=STATIC_REWARD_MAPPING_MODE,
            priority="leaderboard",
            fallback_best=fallback_best,
            fallback_worst=fallback_worst,
        )
    )


def compute_metric_static_base_reward(score: float | None, metadata: dict) -> float | None:
    """
    Metric-only static base reward:
    - no dynamic bound
    - no mode shaping
    - leaderboard bounds preferred, theoretical bounds as fallback
    - mercy semantics: penalties are handled outside and should map to 0
    """
    # 静态上界兜底常量
    fallback_best = METRIC_STATIC_FALLBACK_SIGNED_BEST if METRIC_STATIC_BOUND_USE_CONST_FALLBACK else None
    # 静态下界兜底常量
    fallback_worst = METRIC_STATIC_FALLBACK_SIGNED_WORST if METRIC_STATIC_BOUND_USE_CONST_FALLBACK else None
    if not has_static_bounds_with_priority(
        metadata,
        priority="leaderboard",
        fallback_best=fallback_best,
        fallback_worst=fallback_worst,
    ):
        # 无任何可用静态边界时返回 None(调用方按缺省处理)
        return None
    # 无分数按 0 处理
    if score is None:
        return 0.0
    try:
        # 转换为浮点分数
        score_f = float(score)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score_f):
        # NaN/Inf 视为无效
        return 0.0

    if not USE_SCORE2REWARD:
        # 未开启映射变换时奖励即分数本身
        return score_f

    # 复制 metadata 并剔除动态边界字段,只用静态边界
    reward_metadata = dict(metadata)
    reward_metadata.pop("dynamic_bound_best_signed", None)
    reward_metadata.pop("dynamic_bound_worst_signed", None)
    # 按静态边界映射分数(优先 leaderboard 边界)
    metric_reward = score2reward_with_static_priority(
        score_f,
        reward_metadata,
        mode=STATIC_REWARD_MAPPING_MODE,
        priority="leaderboard",
        fallback_best=fallback_best,
        fallback_worst=fallback_worst,
    )
    try:
        # 校验映射结果为有限浮点数
        metric_reward_f = float(metric_reward)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(metric_reward_f):
        # 非有限结果视为映射失败
        return None
    return metric_reward_f


def _get_program_reward_views(program: Program | None) -> dict[str, float | None]:
    # 读取程序的动态/静态两类奖励视图(父代奖励传播给子代参照用)
    if program is None:
        # 无父代程序:四个视图均为 None
        return {
            "dynamic_raw_reward": None,
            "dynamic_base_reward": None,
            "static_raw_reward": None,
            "static_base_reward": None,
        }

    # 取程序 metadata(非字典时按空字典处理)
    metadata = program.metadata if isinstance(program.metadata, dict) else {}
    # 动态奖励缺失时回退到程序主字段,静态奖励仅来自 metadata
    return {
        "dynamic_raw_reward": metadata.get("dynamic_raw_reward", program.reward),
        "dynamic_base_reward": metadata.get("dynamic_base_reward", program.base_reward),
        "static_raw_reward": metadata.get("static_raw_reward"),
        "static_base_reward": metadata.get("static_base_reward"),
    }


def _compute_task_reward_frontier(db: ProgramDatabase, *, task_name: str, metadata_fallback: dict) -> dict[str, float]:
    # 计算该任务的奖励前沿:动态/静态/指标静态三类"更新前"历史最优,用于判断本次是否刷新纪录
    # 取出该任务的所有历史程序
    programs = db.get_all(task_name)
    # 三类最优值初始化为 0(奖励非负语义)
    best_dynamic = 0.0
    best_static = 0.0
    best_metric_static_base = 0.0

    for program in programs:
        # 逐个程序扫描统计
        # metadata 非字典时按空字典处理
        program_meta = program.metadata if isinstance(program.metadata, dict) else {}
        # 动态原始奖励缺失时回退到程序主 reward 字段
        dynamic_reward = program_meta.get("dynamic_raw_reward", program.reward)
        try:
            dynamic_reward_f = float(dynamic_reward)
        except (TypeError, ValueError):
            dynamic_reward_f = 0.0
        # 有效值参与动态最优更新
        if math.isfinite(dynamic_reward_f):
            best_dynamic = max(best_dynamic, dynamic_reward_f)

        # 静态原始奖励仅存于 metadata
        static_reward = program_meta.get("static_raw_reward")
        if static_reward is None and program.score is not None:
            # 缺失时按分数重算静态奖励(兼容旧数据)
            static_reward = compute_static_score_reward(program.score, program_meta or metadata_fallback)
        try:
            static_reward_f = float(static_reward) if static_reward is not None else 0.0
        except (TypeError, ValueError):
            static_reward_f = 0.0
        # 有效值参与静态最优更新
        if math.isfinite(static_reward_f):
            best_static = max(best_static, static_reward_f)

        # 指标静态基础奖励同样仅存于 metadata
        metric_static_base = program_meta.get("metric_static_base_reward")
        if metric_static_base is None and program.score is not None:
            # 缺失时按分数重算(带 leaderboard 边界)
            metric_static_base = compute_metric_static_base_reward(program.score, program_meta or metadata_fallback)
        try:
            metric_static_base_f = float(metric_static_base) if metric_static_base is not None else 0.0
        except (TypeError, ValueError):
            metric_static_base_f = 0.0
        # 有效值参与指标静态最优更新
        if math.isfinite(metric_static_base_f):
            best_metric_static_base = max(best_metric_static_base, metric_static_base_f)

    # 返回三类"更新前"历史最优
    return {
        "best_dynamic_raw_reward_before": float(best_dynamic),
        "best_static_raw_reward_before": float(best_static),
        "best_metric_static_base_reward_before": float(best_metric_static_base),
    }


async def _update_history_best_static_base_reward(task_name: str, best_before: float, current_value: float | None) -> float:
    # 更新任务历史最优静态基础奖励,并返回所有任务历史最优的均值(作为全局参考信号)
    async with _get_loop_lock("history_static_base_reward"):
        # 加锁更新缓存,避免并发协程竞争
        existing = _history_static_base_reward_cache.get(task_name, 0.0)
        # 与传入的"更新前最优"取较大者
        merged = max(existing, float(best_before or 0.0))
        if current_value is not None:
            # 当前值有效时也并入最优
            try:
                current_f = float(current_value)
            except (TypeError, ValueError):
                current_f = None
            if current_f is not None and math.isfinite(current_f):
                merged = max(merged, current_f)
        # 写回该任务的历史最优
        _history_static_base_reward_cache[task_name] = merged
        if not _history_static_base_reward_cache:
            # 缓存为空(理论上不会发生)返回 0
            return 0.0
        # 返回全部任务历史最优的均值
        return float(sum(_history_static_base_reward_cache.values()) / len(_history_static_base_reward_cache))


async def get_program_database() -> ProgramDatabase:
    """Get or create the global program database instance."""
    # 获取(或首次创建)全局程序数据库单例
    global _program_database
    if _program_database is None:
        # 双检锁:拿到事件循环锁后再检查一次,避免重复建库
        async with _get_loop_lock("database"):
            if _program_database is None:
                # 按配置创建数据库(路径与每任务容量上限)
                _program_database = ProgramDatabase(
                    db_path=MLE_CONFIGS["db_path"], max_per_task=MLE_CONFIGS["max_programs_per_task"]
                )
    # 返回单例
    return _program_database


def get_search_algorithm():
    """Get or create the global search algorithm instance."""
    # 获取(或首次创建)全局搜索算法单例:决定每轮触发哪个四算子(draft/improve/debug/crossover)
    global _search_algorithm
    if _search_algorithm is None:
        # 读取算法类型与四算子触发概率
        algo_type = MLE_CONFIGS["search_algorithm"]
        draft_prob = MLE_CONFIGS["draft_probability"]
        improve_prob = MLE_CONFIGS["improve_probability"]
        debug_prob = MLE_CONFIGS["debug_probability"]
        crossover_prob = MLE_CONFIGS["crossover_probability"]
        if algo_type == "greedy":
            # 贪心搜索:只用 draft/debug 两种算子
            _search_algorithm = AIRAGreedySearch(draft_probability=draft_prob, debug_probability=debug_prob)
        elif algo_type == "evo":
            # 进化搜索(Evo-Max 风格):完整四算子概率调度
            _search_algorithm = AIRAEvoSearch(
                draft_probability=draft_prob,
                improve_probability=improve_prob,
                debug_probability=debug_prob,
                crossover_probability=crossover_prob,
            )
        elif algo_type == "airaevo":
            # AIRA 推理期进化:多岛屿 + 经验卡片 + 策略选择
            _search_algorithm = AIRAInferenceEvoSearch(
                policy=MLE_CONFIGS["airaevo_policy"],
                num_islands=MLE_CONFIGS["airaevo_num_islands"],
                max_island_size=MLE_CONFIGS["airaevo_max_island_size"],
                crossover_after_generation=MLE_CONFIGS["airaevo_crossover_after_generation"],
                crossover_probability=crossover_prob,
                execution_timeout=MLE_CONFIGS["airaevo_execution_timeout"],
            )
        else:
            # 未知算法直接报错
            raise ValueError(f"Unknown search algorithm: {algo_type}")
    # 返回单例
    return _search_algorithm



def _to_token_id_list(tokenized):
    """Normalize tokenizer output to a flat list[int] for Sample.tokens."""
    # 兼容 BatchEncoding:优先取 input_ids 字段
    if hasattr(tokenized, "input_ids"):
        tokenized = tokenized.input_ids
    elif isinstance(tokenized, dict) and "input_ids" in tokenized:
        tokenized = tokenized["input_ids"]

    # 张量/数组统一转成 Python 列表
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()

    if tokenized and isinstance(tokenized[0], (list, tuple)):
        # 单样本批可能嵌套一层;多于一批则属于异常
        if len(tokenized) != 1:
            raise ValueError(f"Expected a single prompt, got {len(tokenized)} tokenized prompts")
        tokenized = tokenized[0]

    # 返回扁平的 token id 列表
    return list(tokenized)

def _get_search_selection_metadata(search_algo) -> dict:
    # 提取搜索算法最近一次父代选择的附加元数据(快照拷贝,避免共享可变状态)
    metadata = getattr(search_algo, "last_selection_metadata", None)
    return dict(metadata) if isinstance(metadata, dict) else {}


def _extract_llm_text(output: dict) -> str:
    # 从 LLM 服务的多种响应格式中提取生成文本(兼容 SGLang 与 OpenAI 风格)
    if not isinstance(output, dict):
        return ""
    # 先尝试常见的顶层文本字段
    for key in ("text", "output", "response"):
        value = output.get(key)
        if isinstance(value, str):
            return value
    # 再尝试 OpenAI 风格的 choices[0] 结构
    choices = output.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(choice.get("text"), str):
                return choice["text"]
    # 无法识别的格式返回空串
    return ""


async def _maybe_update_airaevo_program_metadata(args, db: ProgramDatabase, program: Program) -> dict:
    # 仅在 airaevo 搜索下执行:为程序生成经验卡片与"丰富记忆"摘要并合入数据库
    if MLE_CONFIGS.get("search_algorithm") != "airaevo" or program.id is None:
        # 非 airaevo 或程序未入库则跳过
        return {}
    # 取该任务全部程序,排除自身后作为历史参考
    all_programs = db.get_all(program.task_name)
    previous_programs = [candidate for candidate in all_programs if candidate.id != program.id]
    # 构建结构化经验卡片(当前程序相对历史程序的方法摘要)
    card = build_experience_card(program, all_programs=previous_programs)
    # 待合入数据库的更新:先放入经验卡片
    updates = {"airaevo_experience_card": card}
    if AIRAEVO_RICH_MEMORY_ENABLED:
        # 取父代程序用于对比总结
        parent_program = db.get_by_id(int(program.parent_id)) if program.parent_id is not None else None
        # 构建丰富记忆摘要的 prompt(任务描述+当前程序+父代+经验卡片)
        prompt = build_airaevo_rich_memory_summary_prompt(
            task_description=str(program.metadata.get("task_description") or ""),
            current_program=program,
            parent_program=parent_program,
            current_card=card,
        )
        # 摘要生成使用的 LLM 地址,未配置时退用 SGLang router
        url = os.getenv("AIRAEVO_MEMORY_LLM_URL", "").strip()
        if not url:
            url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        # 请求体:温度 0 保证摘要稳定,限制最大 token
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": AIRAEVO_RICH_MEMORY_MAX_TOKENS,
            },
            "return_logprob": False,
        }
        try:
            # 带超时请求 LLM 生成摘要
            output = await asyncio.wait_for(post(url, payload), timeout=AIRAEVO_RICH_MEMORY_TIMEOUT)
            # 解析 LLM 输出中的 JSON 对象
            parsed = parse_json_object(_extract_llm_text(output))
            if isinstance(parsed, dict):
                # 提取方法概要与父代对比经验两个字段
                method_overview = str(parsed.get("method_overview") or "").strip()
                parent_exp = str(parsed.get("parent_comparison_experience") or "").strip()
                if method_overview and parent_exp:
                    # 字段齐全:写入丰富摘要并同步进经验卡片
                    updates["airaevo_rich_summary"] = {
                        "method_overview": method_overview,
                        "parent_comparison_experience": parent_exp,
                    }
                    card["rich_summary"] = updates["airaevo_rich_summary"]
                    updates["airaevo_experience_card"] = card
                else:
                    # 缺字段:记录错误类型
                    updates["airaevo_rich_summary_error"] = "missing_required_fields"
            else:
                # 输出不是合法 JSON 对象
                updates["airaevo_rich_summary_error"] = "invalid_json"
        except Exception as exc:
            # 摘要失败不影响主流程,仅记录错误类型
            logger.warning("[AIRAEVO MEMORY] rich summary failed for program_id=%s: %s", program.id, exc)
            updates["airaevo_rich_summary_error"] = type(exc).__name__
    # 将卡片/摘要合并进程序数据库的 metadata
    db.merge_program_metadata(int(program.id), updates)
    # 返回更新内容(调用方会同步进样本 metadata)
    return updates


def _normalize_token_ids(token_ids):
    """Normalize tokenizer/SGLang token containers to a plain list[int]."""
    # Newer transformers versions may return BatchEncoding from
    # apply_chat_template(tokenize=True). Extract input_ids before concatenation.
    # 兼容 BatchEncoding/字典容器:先取出 input_ids
    if hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids
    elif isinstance(token_ids, dict) and "input_ids" in token_ids:
        token_ids = token_ids["input_ids"]

    # torch.Tensor and numpy.ndarray both expose tolist; normalize before unwrapping.
    # 张量/数组统一转成 Python 列表
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    # A one-sample batch may be nested; rollout Sample.tokens requires one dimension.
    # 解包元组与单样本嵌套,保证一维列表
    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)
    if token_ids and isinstance(token_ids[0], (list, tuple)):
        # 多于一条序列说明输入异常,直接报错
        if len(token_ids) != 1:
            raise ValueError(f"Expected single prompt token sequence, got {len(token_ids)} sequences")
        # 解出唯一一条序列
        token_ids = list(token_ids[0])

    # 返回扁平的 list[int]
    return list(token_ids)



async def generate(args, sample: Sample, sampling_params, evaluation) -> Sample:
    """
    Single-turn generation with draft/improve mode support.

    Draft mode: Generate code from scratch based on task description.
    Improve mode: Improve existing code from database based on parent's score and execution.

    Optimization: Samples with the same group_index share the same parent selection result.

    Args:
        args: Arguments object
        sample: Sample to generate for
        sampling_params: Sampling parameters for generation
        evaluation: If True, force draft mode (draft_probability=1.0)
    """
    # 本函数完成一次单轮生成:四算子选择父代并改写 prompt -> 请求 LLM 生成 -> 回填样本字段
    assert not args.partial_rollout, f"Partial rollout is not supported for this function at the moment."

    assert evaluation is not None, "evaluation must be provided"
    # 打印训练/评估标志
    print(f"[GENERATE] evaluation: {evaluation}")

    # 初始化 rollout 状态(含 tokenizer)
    state = GenerateState(args)
    # SGLang router 的生成接口地址
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Get program database and search algorithm
    # 获取全局程序数据库(历史程序池)
    db = await get_program_database()
    # 获取全局搜索算法(决定本轮触发哪个算子并挑选父代)
    search_algo = get_search_algorithm()

    # Extract task information from sample metadata
    # 任务名(缺省 unknown)
    task_name = sample.metadata.get("task_name", "unknown")
    task_id = sample.metadata.get("uuid", "")  # task_id is the uuid
    # 数据目录(沙箱运行时挂载的数据路径)
    data_dir = sample.metadata.get("data_dir", "")
    # 组索引:同组样本共享父代选择结果
    group_index = getattr(sample, "group_index", None)

    # Get original prompt content from sample (messages with exactly one system and one user)
    # 函数内延迟导入 numpy(仅用于类型判断)
    import numpy as np

    # If sample.prompt is a numpy array, convert to list
    # numpy 数组先转为列表,统一后续处理
    if isinstance(sample.prompt, np.ndarray):
        sample.prompt = sample.prompt.tolist()

    # 校验 prompt 必须是消息字典列表
    assert isinstance(sample.prompt, list), "sample.prompt must be a list of message dicts"
    # 拷贝出可改写的输入消息列表(避免污染原样本)
    input_messages: list[dict[str, str]] = []
    for message in sample.prompt:
        # 逐条校验消息结构(dict[str, str] 且含 role/content)后拷贝
        assert isinstance(message, dict), "each message must be a dict"
        assert all(
            isinstance(k, str) and isinstance(v, str) for k, v in message.items()
        ), "each message must be dict[str, str]"
        assert "role" in message and "content" in message, "each message must contain role and content"
        input_messages.append(dict(message))
    # 统计消息角色,供后续定位 system/user
    roles = [message["role"] for message in input_messages]
    assert (
        roles.count("system") == 1 and roles.count("user") == 1
    ), "sample.prompt must contain exactly one system and one user message"
    # 记录 system/user 消息位置
    system_index = roles.index("system")
    user_index = roles.index("user")
    # 提取原始 system prompt 文本
    system_prompt = input_messages[system_index]["content"]
    # 提取原始 user prompt 文本
    user_prompt = input_messages[user_index]["content"]
    # 任务描述文本(可能为空)
    task_description = sample.metadata.get("task_description") or ""
    # 数据描述文本(可能为空)
    data_description = sample.metadata.get("data_description") or ""

    # # Mixed draft/improve mode in a group
    # prompt, parent_program, mode = search_algo.select(
    #     database=db,
    #     task_name=task_name,
    #     description=description,
    #     data_dir=data_dir,
    #     max_steps=1
    # )

    # Check if we already have a cached parent selection for this group
    # 检查本组是否已有缓存的父代选择结果(同组样本共享一次 select 调用)
    global _parent_selection_cache
    # 评估期不缓存(cache_key=None,每个评估样本独立选择)
    cache_key = None if evaluation else group_index
    # 交叉算子的第二父代,默认无
    secondary_parent_program = None
    # 父代选择的附加元数据(如选岛/策略信息)
    selection_metadata = {}

    # Get n_samples_per_prompt from args
    # 组大小:优先取 args,缺省用环境变量默认值
    n_samples_per_prompt = getattr(args, "n_samples_per_prompt", None)
    if n_samples_per_prompt is None:
        n_samples_per_prompt = N_SAMPLES_PER_PROMPT

    # cache_key is not None and
    # 缓存命中路径:组内非首个样本直接复用选择结果
    if cache_key in _parent_selection_cache:
        # Reuse cached result for this group
        # 加锁读取组缓存,防止并发读写
        async with _get_loop_lock("parent_selection_cache"):
            # 取该组的缓存条目(可能已被清理)
            cache_entry = _parent_selection_cache.get(cache_key)
            if cache_entry is not None:
                # 命中:解包缓存的输入消息与父代选择结果
                cached_data = cache_entry["data"]
                if len(cached_data) == 4:
                    # 新版四元组含交叉第二父代
                    input_messages, parent_program, mode, secondary_parent_program = cached_data
                else:
                    # 兼容旧三元组格式
                    input_messages, parent_program, mode = cached_data
                    secondary_parent_program = None
                # 恢复父代选择的附加元数据
                selection_metadata = dict(cache_entry.get("selection_metadata") or {})
                # 累加本条目的访问次数
                cache_entry["access_count"] += 1
                access_count = cache_entry["access_count"]
                # 期望的组内样本总数
                total_samples = cache_entry["total_samples"]

                # logger.info(f"[CACHE HIT] group_index={group_index}, access={access_count}/{total_samples}")

                # Auto-cleanup: Remove cache entry when all samples have accessed it
                # 所有样本都已访问过:自动清理缓存条目
                if access_count >= total_samples:
                    del _parent_selection_cache[cache_key]
                    # logger.info(
                    #     f"[CACHE AUTO-CLEANUP] Removed group_index={group_index} after {access_count} accesses"
                    # )
            else:
                # Race condition: cache was deleted, regenerate
                # 竞态:缓存刚被取走,本协程重新执行一次父代选择
                (
                    (augmented_system_prompt, augmented_user_prompt),
                    parent_program,
                    mode,
                    secondary_parent_program,
                ) = search_algo.select(
                    database=db,
                    task_id=task_id,
                    task_description=task_description,
                    data_description=data_description,
                    public_system_prompt=system_prompt,
                    public_user_prompt=user_prompt,
                    data_dir=data_dir,
                    max_steps=1,
                    task_name=task_name,
                    evaluation=evaluation,
                )
                selection_metadata = _get_search_selection_metadata(search_algo)
                # 用增强后的 prompt 覆盖原 system/user 内容
                input_messages[system_index]["content"] = augmented_system_prompt
                input_messages[user_index]["content"] = augmented_user_prompt
                # logger.info(f"[CACHE RACE] Regenerated for group_index={group_index}")
    else:
        # Perform parent selection (first sample in the group or no group_index)
        # 缓存未命中路径:组内首个样本(或无组索引)执行真正的父代选择
        async with _get_loop_lock("parent_selection_cache"):
            # Double-check after acquiring lock
            # 拿到锁后双检:可能其他协程刚建好缓存
            if cache_key is not None and cache_key in _parent_selection_cache:
                # 命中:解包缓存的选择结果
                cache_entry = _parent_selection_cache[cache_key]
                cached_data = cache_entry["data"]
                if len(cached_data) == 4:
                    # 新版四元组含交叉第二父代
                    input_messages, parent_program, mode, secondary_parent_program = cached_data
                else:
                    # 兼容旧三元组格式
                    input_messages, parent_program, mode = cached_data
                    secondary_parent_program = None
                # 恢复父代选择的附加元数据
                selection_metadata = dict(cache_entry.get("selection_metadata") or {})
                # 累加本条目的访问次数
                cache_entry["access_count"] += 1
                access_count = cache_entry["access_count"]
                # 期望的组内样本总数
                total_samples = cache_entry["total_samples"]

                # logger.info(f"[CACHE HIT AFTER LOCK] group_index={group_index}, access={access_count}/{total_samples}")

                # Auto-cleanup: Remove cache entry when all samples have accessed it
                # 所有样本都已访问过:自动清理缓存条目
                if access_count >= total_samples:
                    del _parent_selection_cache[cache_key]
                    # logger.info(
                    #     f"[CACHE AUTO-CLEANUP] Removed group_index={group_index} after {access_count} accesses"
                    # )
            else:
                # Use search algorithm to decide mode, select parent, and build prompt
                # 调用搜索算法:决定算子模式、挑父代(及交叉第二父代)、构建增强 prompt
                (
                    (augmented_system_prompt, augmented_user_prompt),
                    parent_program,
                    mode,
                    secondary_parent_program,
                ) = search_algo.select(
                    database=db,
                    task_id=task_id,
                    task_description=task_description,
                    data_description=data_description,
                    public_system_prompt=system_prompt,
                    public_user_prompt=user_prompt,
                    data_dir=data_dir,
                    max_steps=1,
                    task_name=task_name,
                    evaluation=evaluation,
                )
                selection_metadata = _get_search_selection_metadata(search_algo)
                # 用增强后的 prompt 覆盖原 system/user 内容
                input_messages[system_index]["content"] = augmented_system_prompt
                input_messages[user_index]["content"] = augmented_user_prompt

                # Cache the result if we have a valid group_index
                if cache_key is not None:
                    # 有组索引时缓存选择结果,供组内其余样本复用
                    _parent_selection_cache[cache_key] = {
                        "data": (input_messages, parent_program, mode, secondary_parent_program),
                        "access_count": 1,  # First access
                        "total_samples": n_samples_per_prompt,
                        "selection_metadata": selection_metadata,
                    }
                    # eval_tag = " [EVALUATION]" if evaluation else ""
                    # logger.info(
                    #     f"[CACHE MISS] Created cache for group_index={group_index}, mode={mode}, total_samples={n_samples_per_prompt}{eval_tag}"
                    # )

    # 取(可能被增强过的)system/user prompt,并确保消息列表与其一致
    augmented_system_prompt = input_messages[system_index]["content"]
    augmented_user_prompt = input_messages[user_index]["content"]
    input_messages[system_index]["content"] = augmented_system_prompt
    input_messages[user_index]["content"] = augmented_user_prompt

    if evaluation:
        # 评估期允许数据源显式指定 prompt 类型(算子模式)
        eval_prompt_type = str(sample.metadata.get("prompt_type") or mode or "draft").strip().lower()
        if eval_prompt_type in {"draft", "improve", "debug", "crossover"}:
            # 合法类型则覆盖搜索算法选出的模式
            mode = eval_prompt_type

    # Store mode and parent info in sample metadata for use in reward_func
    # These will be used to create the Program object in reward_func
    # 将算子模式写入 metadata,供奖励函数区分四算子
    sample.metadata["generation_mode"] = mode
    # 标记训练/评估
    sample.metadata["evaluation"] = bool(evaluation)
    # 任务 ID
    sample.metadata["task_id"] = task_id
    if selection_metadata:
        # 附加父代选择的元数据(如岛屿/策略信息)
        sample.metadata.update(selection_metadata)
    # 防御:确保 secondary_parent_program 变量一定已定义
    secondary_parent_program = locals().get("secondary_parent_program", None)
    if parent_program:
        # 有父代:读取其动态/静态奖励视图
        parent_views = _get_program_reward_views(parent_program)
        # 以下把父代身份与各奖励视图写入 metadata,供奖励函数做相对改进计算
        sample.metadata["parent_id"] = parent_program.id
        sample.metadata["parent_reward"] = parent_program.reward
        sample.metadata["parent_base_reward"] = parent_program.base_reward
        sample.metadata["parent_dynamic_raw_reward"] = parent_views["dynamic_raw_reward"]
        sample.metadata["parent_dynamic_base_reward"] = parent_views["dynamic_base_reward"]
        sample.metadata["parent_static_raw_reward"] = parent_views["static_raw_reward"]
        sample.metadata["parent_static_base_reward"] = parent_views["static_base_reward"]
        sample.metadata["parent_score"] = parent_program.score
        sample.metadata["parent_code"] = parent_program.code
    else:
        # 无父代(draft):父代相关字段全部置空占位
        sample.metadata["parent_id"] = None
        sample.metadata["parent_reward"] = None
        sample.metadata["parent_base_reward"] = None
        sample.metadata["parent_dynamic_raw_reward"] = None
        sample.metadata["parent_dynamic_base_reward"] = None
        sample.metadata["parent_static_raw_reward"] = None
        sample.metadata["parent_static_base_reward"] = None
        sample.metadata["parent_score"] = None
        sample.metadata["parent_code"] = ""
    if secondary_parent_program:
        # 有交叉第二父代(crossover 算子):记录其身份与奖励视图
        secondary_parent_views = _get_program_reward_views(secondary_parent_program)
        sample.metadata["crossover_parent_id"] = secondary_parent_program.id
        sample.metadata["crossover_parent_reward"] = secondary_parent_program.reward
        sample.metadata["crossover_parent_base_reward"] = secondary_parent_program.base_reward
        sample.metadata["crossover_parent_dynamic_raw_reward"] = secondary_parent_views["dynamic_raw_reward"]
        sample.metadata["crossover_parent_dynamic_base_reward"] = secondary_parent_views["dynamic_base_reward"]
        sample.metadata["crossover_parent_static_raw_reward"] = secondary_parent_views["static_raw_reward"]
        sample.metadata["crossover_parent_static_base_reward"] = secondary_parent_views["static_base_reward"]
        sample.metadata["crossover_parent_code"] = secondary_parent_program.code
        # 记录交叉使用的两个父代 ID
        sample.metadata["crossover_parent_ids"] = [parent_program.id, secondary_parent_program.id]
    else:
        # 无交叉父代:相关字段置空/占位
        sample.metadata["crossover_parent_id"] = None
        sample.metadata["crossover_parent_reward"] = None
        sample.metadata["crossover_parent_base_reward"] = None
        sample.metadata["crossover_parent_dynamic_raw_reward"] = None
        sample.metadata["crossover_parent_dynamic_base_reward"] = None
        sample.metadata["crossover_parent_static_raw_reward"] = None
        sample.metadata["crossover_parent_static_base_reward"] = None
        sample.metadata["crossover_parent_code"] = ""
        sample.metadata["crossover_parent_ids"] = [parent_program.id] if parent_program else []

    # 用 chat 模板渲染最终输入文本(tokenize=False 拿字符串)
    input_text = state.tokenizer.apply_chat_template(
        input_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # 按开关把 prompt 落盘(调试)
    await _dump_prompt_to_txt(
        task_name=task_name,
        task_id=task_id,
        group_index=group_index,
        mode=mode,
        parent_program=parent_program,
        secondary_parent_program=secondary_parent_program,
        system_prompt=augmented_system_prompt,
        user_prompt=augmented_user_prompt,
    )
    # 再渲染一次拿到 prompt 的 token id 序列(训练需要)
    prompt_token_ids = _to_token_id_list(
        state.tokenizer.apply_chat_template(
            input_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    # 归一化为扁平 list[int]
    prompt_token_ids = _normalize_token_ids(prompt_token_ids)

    # Single-turn generation
    # print(f"[GENERATE] prompt: {prompt}")
    payload = {
        "text": input_text,
        "sampling_params": sampling_params,
        "return_logprob": True,  # Request log probabilities for training
    }
    for key in GENERATION_ABORT_TRANSIENT_METADATA_KEYS:
        # 请求前清空上一轮可能残留的中断标记
        sample.metadata.pop(key, None)
    if sample.metadata.get("code_category") in GENERATION_ABORT_CODE_CATEGORIES:
        # 中断类别的 code_category 一并清空,重试后重新判定
        sample.metadata.pop("code_category", None)
    # LLM 输出占位
    output = None
    # 评估期 abort 重试计数
    eval_abort_attempts = 0
    while True:
        # 发起生成请求
        output = await post(url, payload)
        if output["meta_info"]["finish_reason"]["type"] != "abort":
            # 正常结束(长度截断或 stop)即退出
            break
        if not evaluation:
            # 训练期不重试,直接交给后续 abort 处理逻辑
            break
        # 评估期遇到 abort:计数并重试
        eval_abort_attempts += 1
        sample.metadata["eval_generation_abort_retry_count"] = eval_abort_attempts
        if eval_abort_attempts > EVAL_GENERATION_ABORT_MAX_RETRIES:
            # 超过最大重试次数:抛错让上层感知评估异常
            raise RuntimeError(
                "Eval generation kept returning finish_reason=abort after "
                f"{EVAL_GENERATION_ABORT_MAX_RETRIES} retries for task={task_name} group_index={group_index}"
            )
        logger.warning(
            "[EVAL GENERATION ABORT] Retrying generation: task=%s group_index=%s attempt=%s/%s",
            task_name,
            group_index,
            eval_abort_attempts,
            EVAL_GENERATION_ABORT_MAX_RETRIES,
        )
        if EVAL_GENERATION_ABORT_RETRY_SLEEP > 0:
            # 按配置休眠后再重试
            await asyncio.sleep(EVAL_GENERATION_ABORT_RETRY_SLEEP)

    # Handle abort: do NOT return early — set an empty response so reward_func is still
    # called.  This is necessary to unblock the group-level barrier in
    # _resolve_group_reward_mapping (ENABLE_DYNAMIC_SCORE_BOUNDS=1).  If we return early
    # here the other n_samples_per_prompt-1 coroutines in the same group will wait forever
    # on their waiter future and permanently occupy an active_tasks slot.
    if output["meta_info"]["finish_reason"]["type"] == "abort":
        # 生成被中止:构造空响应样本,仍交由 reward_func 走组屏障逻辑,避免同组其他协程永久等待
        # 记录原始 finish_reason
        finish_reason = output["meta_info"].get("finish_reason")
        # 标记生成中断及原因
        sample.metadata["generation_aborted"] = True
        sample.metadata["generation_abort_reason"] = "sglang_finish_reason_abort"
        sample.metadata["generation_abort_finish_reason"] = finish_reason
        sample.metadata["code_category"] = "generation_abort"
        sample.tokens = prompt_token_ids  # no response tokens
        sample.response_length = 0
        sample.response = ""
        sample.loss_mask = []
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        # 状态置 COMPLETED 以保证样本仍进入后续奖励流程(中断信息记录在 metadata 中)
        sample.status = Sample.Status.COMPLETED
        return sample

    # Process response
    # 取出生成文本
    response = output["text"]
    # response_token_ids = state.tokenizer(response, add_special_tokens=False)["input_ids"]

    if "output_token_logprobs" in output["meta_info"]:
        # 从 logprob 列表同时取响应 token id 与对应对数概率
        response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        # 累加本轮响应的 logprob(策略梯度训练用)
        sample.rollout_log_probs += cur_log_probs
    else:
        # 缺少 logprob 属于异常情况
        assert False
        response_token_ids = state.tokenizer(response, add_special_tokens=False)["input_ids"]
    # 归一化响应 token id 为扁平列表
    response_token_ids = _normalize_token_ids(response_token_ids)

    # 构造 loss mask:响应部分全 1(仅对生成的 token 计损失)
    loss_mask = [1] * len(response_token_ids)

    # Set sample fields
    # 回填样本字段:完整 token 序列 = prompt + response
    sample.tokens = prompt_token_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_mask

    # Set status based on finish reason
    # 按结束原因设置样本状态
    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            # 达到长度上限:截断
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            # 被中止
            sample.status = Sample.Status.ABORTED
        case "stop":
            # 正常结束
            sample.status = Sample.Status.COMPLETED

    return sample


async def reward_func(args, sample, **kwargs):
    """
    The reward function for code generation tasks.

    For draft mode: reward is calculated from score as before.
    For improve mode: reward is the difference between current reward and parent reward.

    Args:
        args: the arguments
        sample: the sample to evaluate
    """
    # 奖励函数:在沙箱中执行生成的代码(execution-grounded 评分)并计算四算子各自的训练奖励
    global SANDBOX_CONCURRENT_COUNT
    if not isinstance(sample, Sample):
        # 类型防御:必须传入 Sample 实例
        raise TypeError("Sample must be an instance of Sample class.")
    if not USE_PROXY_SANDBOX and not GPU_BASE_URL:
        # 直连模式下必须配置 GPU 沙箱地址
        raise RuntimeError("GPU_BASE_URL is required; the public release has no default sandbox endpoint")

    # Initialize logger if not already done, or re-initialize if n_samples_per_prompt changed.
    # get_logger() auto-creates with default n_samples_per_prompt=8, so we must initialize
    # explicitly here before it's first called to get the correct group size (e.g. 16 for 8x16).
    # 组大小:优先取 args,缺省用环境变量默认值
    n_samples_per_prompt = getattr(args, "n_samples_per_prompt", None)
    if n_samples_per_prompt is None:
        n_samples_per_prompt = int(os.environ.get("N_SAMPLES_PER_PROMPT", "8"))

    if _logging_utils_module._global_logger is None or _logging_utils_module._global_logger.n_samples_per_prompt != n_samples_per_prompt:
        # 日志器未初始化或组大小变化时,按当前组大小重新初始化
        init_logger(n_samples_per_prompt=n_samples_per_prompt)
    # 取全局日志器实例
    logger_instance = get_logger()

    # 恢复生成阶段写入的上下文:原始回复、任务信息与评估标志
    raw_text = sample.response or ""  # Store raw response text
    data_dir = sample.metadata.get("data_dir")
    task_id = sample.metadata.get("task_id", "")
    task_name = sample.metadata.get("task_name", "unknown")
    evaluation = bool(sample.metadata.get("evaluation", False))
    # 按训练/评估选择沙箱超时配置
    sandbox_job_timeout = EVAL_JOB_TIMEOUT if evaluation else JOB_TIMEOUT
    sandbox_wait_timeout = EVAL_WAIT_TIMEOUT if evaluation else WAIT_TIMEOUT
    # Runtime safeguard: allow the sandbox job up to 3600 seconds and reserve
    # 300 additional seconds for polling and network jitter.
    # 兜底上限:作业最长 3600 秒,轮询等待最长 3900 秒(留 300 秒网络余量)
    sandbox_max_job_timeout = int(os.getenv("SANDBOX_MAX_JOB_TIMEOUT", "3600"))
    sandbox_max_wait_timeout = int(os.getenv("SANDBOX_MAX_WAIT_TIMEOUT", "3900"))
    if sandbox_max_job_timeout > 0:
        # 作业超时不得超过兜底上限
        sandbox_job_timeout = min(int(sandbox_job_timeout), sandbox_max_job_timeout)
    if sandbox_max_wait_timeout > 0:
        # 等待超时不得超过兜底上限
        sandbox_wait_timeout = min(int(sandbox_wait_timeout), sandbox_max_wait_timeout)
    group_index = getattr(sample, "group_index", None)
    generation_mode = sample.metadata.get("generation_mode", "draft")
    # 父代(improve/debug 的参照)各项奖励与代码
    parent_id = sample.metadata.get("parent_id")
    parent_reward = sample.metadata.get("parent_reward")
    parent_base_reward = sample.metadata.get("parent_base_reward")
    parent_dynamic_raw_reward = sample.metadata.get("parent_dynamic_raw_reward", parent_reward)
    parent_dynamic_base_reward = sample.metadata.get("parent_dynamic_base_reward", parent_base_reward)
    parent_static_raw_reward = sample.metadata.get("parent_static_raw_reward")
    parent_static_base_reward = sample.metadata.get("parent_static_base_reward")
    parent_code = sample.metadata.get("parent_code", "")
    # 交叉父代(crossover 的第二参照)各项奖励与代码
    crossover_parent_reward = sample.metadata.get("crossover_parent_reward")
    crossover_parent_base_reward = sample.metadata.get("crossover_parent_base_reward")
    crossover_parent_dynamic_raw_reward = sample.metadata.get("crossover_parent_dynamic_raw_reward", crossover_parent_reward)
    crossover_parent_dynamic_base_reward = sample.metadata.get(
        "crossover_parent_dynamic_base_reward", crossover_parent_base_reward
    )
    crossover_parent_static_raw_reward = sample.metadata.get("crossover_parent_static_raw_reward")
    crossover_parent_static_base_reward = sample.metadata.get("crossover_parent_static_base_reward")
    crossover_parent_code = sample.metadata.get("crossover_parent_code", "")
    # 沙箱资源类型(默认 gpu)
    resource_type = sample.metadata.get("cpu_gpu", "gpu")  # default to 'file' if not specified
    # print("resource_type:", resource_type)

    # logger.info(f"data_dir: {data_dir}")
    # logger.info(f"generation_mode: {generation_mode}, parent_id: {parent_id}, parent_reward: {parent_reward}")

    if bool(sample.metadata.get("generation_aborted")):
        # 该样本生成阶段已被中止:仍需以占位分数参与组屏障,随后整组重排
        sample.metadata["score"] = None
        sample.metadata["running_time"] = 0.0
        # 记录两种奖励映射模式,便于审计
        sample.metadata["reward_mapping_mode"] = REWARD_MAPPING_MODE
        sample.metadata["static_reward_mapping_mode"] = STATIC_REWARD_MAPPING_MODE
        # 以 score=None、base_reward=-1 进入组级奖励映射屏障
        reward_mapping_result = await _resolve_group_reward_mapping(
            sample.metadata,
            group_index=group_index,
            group_size=n_samples_per_prompt,
            sample_key=f"{task_id}:{group_index}:{id(sample)}",
            score=None,
            base_reward=-1.0,
        )
        # 把屏障上下文写回样本 metadata
        _apply_dynamic_bound_context_metadata(
            sample,
            reward_mapping_result.get("context") or {},
            n_samples_per_prompt,
        )
        # 标记中断并返回 -1,请求框架重排该组
        return _return_generation_abort_for_retry(sample, task_name=task_name, group_index=group_index)

    # extract code from sample.response
    # 从模型原始回复中抽取代码块
    code = extract_code(raw_text)

    # Determine code category for CSV logging (not stored in DB)
    # 代码类别先默认 valid,后续按检查结果覆盖
    code_category = "valid"  # Default to valid
    mercy_raw_reward = None  # Only set for hack/empty cases (no extra sandbox call needed)
    static_raw_reward = 0.0
    static_mercy_raw_reward = 0.0
    dynamic_raw_reward = 0.0
    dynamic_mercy_raw_reward = 0.0
    metric_static_base_reward = 0.0
    # 指标静态映射的边界兜底常量
    metric_fallback_best = METRIC_STATIC_FALLBACK_SIGNED_BEST if METRIC_STATIC_BOUND_USE_CONST_FALLBACK else None
    metric_fallback_worst = METRIC_STATIC_FALLBACK_SIGNED_WORST if METRIC_STATIC_BOUND_USE_CONST_FALLBACK else None
    # 该任务是否有可用的静态边界(0/1)
    metric_static_base_reward_has_bounds = int(
        has_static_bounds_with_priority(
            sample.metadata,
            priority="leaderboard",
            fallback_best=metric_fallback_best,
            fallback_worst=metric_fallback_worst,
        )
    )
    # 作弊检测的类别与原因占位
    hack_category = "valid"
    hack_reason = ""
    # 记录自验证开关状态
    sample.metadata["self_validation_enabled"] = bool(ENABLE_SELF_VALIDATION)

    # Check if code is empty or has no valid code
    if not code or len(code.strip()) == 0:
        # 空代码:类别记 empty
        code_category = "empty"
        logger.info("[CODE CATEGORY] empty")
    elif ENABLE_SELF_VALIDATION and not has_final_validation_score_print(code):
        # 自验证开启但代码缺少最终验证分数输出:记 no_verify
        code_category = "no_verify"
        logger.info("[CODE CATEGORY] no_verify")
    else:
        # 是否要求留出集(holdout)自验证
        require_holdout_validation = bool(ENABLE_SELF_VALIDATION)
        # Check for hacking/cheating before sandbox execution.
        logger.info("Running hack_check on generated code...")
        # Keep hack checking ahead of sandbox execution. The asynchronous path
        # yields while waiting for the judge and caps it with HACK_CHECK_CONCURRENCY.
        # 在进沙箱前先做作弊/取巧检测(hack check),作弊代码不浪费沙箱资源
        is_valid_ml, hack_category, hack_reason = await hack_check_async(
            code,
            require_holdout_validation=require_holdout_validation,
        )
        if not is_valid_ml:
            # 检测未通过:采用检测器给出的类别与原因
            code_category = hack_category
            logger.info(f"[CODE CATEGORY] {code_category}, reason: {hack_reason}")
        else:
            logger.info("[CODE CATEGORY] valid")

    # hack/hack_verify 两类视为作弊
    is_hack = code_category in {"hack", "hack_verify"}

    # Persist hack as 0/1 for downstream metric aggregation.
    # 作弊标记与代码类别写入 metadata
    sample.metadata["hack"] = 1 if is_hack else 0
    sample.metadata["code_category"] = code_category

    # 沙箱返回的原始分数占位
    score = None
    base_reward = 0.0  # Raw base reward (score->reward mapping) before any shaping/baseline
    effective_base_reward = 0.0  # Base reward after optional baseline adjustment
    shaped_base_reward = 0.0  # Stored base_reward: reward before extra improvement bonus
    final_reward = 0.0  # Final reward returned for training
    mercy_raw_reward = 0.0
    # 执行结果信息占位
    result_info = None
    # run_log = ""
    status_code = None
    payload = None
    # 动态边界上下文占位(非空时表示组级映射已发生)
    dynamic_bound_context: dict[str, float | int | str | bool | None] | None = None
    running_time = 0.0
    run_log = ""
    # 验证集分数占位
    validation_score = None
    # 验证-测试差距信息(默认无差距)
    validation_gap_info = validation_test_gap_info(None, None, sample.metadata)
    gap_penalty = 0.0
    gap_multiplier = 1.0

    # Handle different code categories
    if code_category == "empty":
        # 空代码:不进沙箱,直接给惩罚性奖励 -1
        logger.warning("[EMPTY CODE] Response contains no code")
        score = None
        base_reward = -1
        effective_base_reward = -1
        shaped_base_reward = -1
        final_reward = -1
        # 结果标记 empty_code
        result_info = "empty_code"
        status_code = -2  # Special status code for empty code
        payload = {"error": "empty_code", "reason": "No code generated", "running_time": 0.0}
        # Mercy raw reward: if not penalized, would be 0
        # 仁慈(mercy)口径:不惩罚时的假设奖励为 0
        mercy_raw_reward = 0.0
        # 静态/动态各口径的原始奖励同步为惩罚值与仁慈值
        static_raw_reward = final_reward
        static_mercy_raw_reward = mercy_raw_reward
        dynamic_raw_reward = final_reward
        dynamic_mercy_raw_reward = mercy_raw_reward
        metric_static_base_reward = 0.0

    elif code_category == "no_verify":
        # 缺少自验证输出:给 -0.5 的轻惩罚
        logger.warning("[VERIFY CHECK FAILED] Missing Final Validation Score print in generated code")
        score = None
        base_reward = -0.5
        effective_base_reward = -0.5
        shaped_base_reward = -0.5
        final_reward = -0.5
        result_info = "no_verify"
        # 专用状态码 -3
        status_code = -3
        payload = {
            "error": "no_verify",
            "reason": "Missing print(f'Final Validation Score: {score}') style validation output",
        }
        mercy_raw_reward = 0.0
        static_raw_reward = final_reward
        static_mercy_raw_reward = mercy_raw_reward
        dynamic_raw_reward = final_reward
        dynamic_mercy_raw_reward = mercy_raw_reward
        metric_static_base_reward = 0.0

    elif code_category == "hack":
        # For hack, give penalty but record mercy reward (0 instead of -1)
        # 作弊(hack):惩罚 -0.5,同时记录仁慈奖励 0(不惩罚时的假设值)
        logger.warning(f"[HACK DETECTED] Code is cheating/guessing: {hack_reason}")
        score = None
        base_reward = -0.5
        effective_base_reward = -0.5
        shaped_base_reward = -0.5
        final_reward = -0.5
        result_info = f"hack_detected:{hack_reason}"
        status_code = -1  # Special status code for hack detection
        payload = {"error": "hack_detected", "reason": hack_reason, "running_time": 0.0}
        # Mercy raw reward: if not penalized, would be 0
        mercy_raw_reward = 0.0
        static_raw_reward = final_reward
        static_mercy_raw_reward = mercy_raw_reward
        dynamic_raw_reward = final_reward
        dynamic_mercy_raw_reward = mercy_raw_reward
        metric_static_base_reward = 0.0
        logger.info(f"[MERCY MODE] mercy_raw_reward: {mercy_raw_reward} (if not penalized)")

    elif code_category == "hack_verify":
        # 验证造假(hack_verify):验证分数并非来自真实留出集验证
        hack_reason_detail = hack_reason
        logger.warning(f"[HACK VERIFY DETECTED] Validation score is not from a real hold-out validation: {hack_reason_detail}")
        score = None
        base_reward = -0.5
        effective_base_reward = -0.5
        shaped_base_reward = -0.5
        final_reward = -0.5
        result_info = f"hack_verify:{hack_reason_detail}"
        # 专用状态码 -4
        status_code = -4
        payload = {"error": "hack_verify", "reason": hack_reason_detail}
        mercy_raw_reward = 0.0
        static_raw_reward = final_reward
        static_mercy_raw_reward = mercy_raw_reward
        dynamic_raw_reward = final_reward
        dynamic_mercy_raw_reward = mercy_raw_reward
        metric_static_base_reward = 0.0
        logger.info(f"[MERCY MODE] mercy_raw_reward: {mercy_raw_reward} (if not penalized)")

    else:
        # Valid code - execute sandbox normally

        # Use semaphore to limit concurrent sandbox executions
        # 有效代码:用信号量限流后提交远程沙箱执行(execution-grounded 评分)
        async with _get_sandbox_semaphore():
            # Increment counter when entering
            # 进入执行:并发计数 +1(加锁保证原子)
            async with _get_loop_lock("sandbox_concurrent_count"):
                SANDBOX_CONCURRENT_COUNT += 1
                current_count = SANDBOX_CONCURRENT_COUNT

            logger.info(
                f"🚀 [SANDBOX] Entering get_sandbox_result | Current concurrent: {current_count}/{MLE_CONFIGS['sandbox_concurrency']}"
            )
            # 打印本次沙箱调用的超时配置
            logger.info(
                "[SANDBOX TIMEOUT] evaluation=%s job_timeout=%s wait_timeout=%s",
                evaluation,
                sandbox_job_timeout,
                sandbox_wait_timeout,
            )

            try:
                # Create and reuse one AsyncClient.
                # 限制连接池大小与保活时间,避免大量并发时打满连接
                limits = httpx.Limits(max_connections=16, max_keepalive_connections=16, keepalive_expiry=30.0)
                # 细分超时:连接 10s/读 100s/写 30s/池 30s
                timeout = httpx.Timeout(connect=10, read=100, write=30, pool=30)
                if resource_type == "GPU":
                    # GPU 资源:创建异步客户端并选择直连/代理沙箱执行
                    async with httpx.AsyncClient(base_url=GPU_BASE_URL, limits=limits, timeout=timeout) as client:
                        sandbox_fn = get_proxy_sandbox_result if USE_PROXY_SANDBOX else get_sandbox_result
                        # 提交代码执行,拿到状态码与结果负载
                        status_code, payload = await sandbox_fn(
                            client=client,
                            code_str=code,
                            data_dir=data_dir,
                            job_timeout=sandbox_job_timeout,
                            wait_timeout=sandbox_wait_timeout,
                            resource_type="gpu",
                        )
                else:
                    # 非 GPU 分支当前同样走 GPU 沙箱(保留结构以便扩展)
                    async with httpx.AsyncClient(base_url=GPU_BASE_URL, limits=limits, timeout=timeout) as client:
                        sandbox_fn = get_proxy_sandbox_result if USE_PROXY_SANDBOX else get_sandbox_result
                        status_code, payload = await sandbox_fn(
                            client=client,
                            code_str=code,
                            data_dir=data_dir,
                            job_timeout=sandbox_job_timeout,
                            wait_timeout=sandbox_wait_timeout,
                            resource_type="gpu",
                        )
            finally:

                # Decrement counter when exiting
                # 退出执行:并发计数 -1(finally 保证异常时也归还)
                async with _get_loop_lock("sandbox_concurrent_count"):
                    SANDBOX_CONCURRENT_COUNT -= 1
                    current_count = SANDBOX_CONCURRENT_COUNT

                logger.info(
                    f"✅ [SANDBOX] Exiting get_sandbox_result | Current concurrent: {current_count}/{MLE_CONFIGS['sandbox_concurrency']}"
                )

        # 汇总执行结果
        logger.info("\n== Run Result Summary ==")
        if status_code == 200:  # Finished.
            # 执行成功:解析结果负载
            logger.info(f"job_status: {payload.get('status')}")
            result = payload.get("result") or {}
            running_time = payload.get("running_time", 0.0) or 0.0
            run_log = get_clear_log(result.get("run_log"))
            # logger.info(f"run_log: {run_log}")
            # 从清洗后的运行日志中提取验证集分数
            validation_score = extract_validation_score(run_log)
            logger.info(f"run_result_status: {result.get('result')}")
            result_info = result.get("result")
            logger.info(f"running_time: {running_time}")
            logger.info(f"validation_score: {validation_score}")
            # logger.info("score:", result.get("score"))
            score = result.get("score")
            # Map score to base reward
            if score is None:
                base_reward = 0.0
                logger.info(f"score is None, base_reward: {base_reward}")
            else:
                if USE_SCORE2REWARD:
                    # 开启映射时先挂起:待组级自适应边界到齐后统一做 score->reward
                    base_reward = 0.0
                    logger.info(
                        "score: %s, deferring score2reward until group-level adaptive bound is resolved",
                        score,
                    )
                else:
                    # 未开启映射:奖励即原始分数
                    base_reward = score
                    logger.info(f"score: {score}, base_reward: {base_reward} (reward equals score)")

        elif status_code == 503:  # Connection failed.
            # 连接失败:记录错误类型与详情,奖励按 0 处理
            logger.info(f"NOT 200, status_code: {status_code}")
            error_type = payload.get("type", "unknown")
            error_detail = payload.get("detail", "Connection failed")
            running_time = payload.get("running_time", 0.0) or 0.0
            logger.info(f"Connection error ({error_type}): {error_detail}")
            result_info = f"connection_error:{error_type}"
            run_log = f"Connection error: {error_type}"
            base_reward = 0.0

        else:  # Still running or another non-terminal state.
            # 其他非终态/错误:记录详细错误信息,奖励按 0 处理
            logger.info(f"NOT 200, status_code: {status_code}")
            error_msg = payload.get("error", "unknown error")
            detail = payload.get("detail", {})
            running_time = payload.get("running_time", 0.0) or 0.0
            logger.info(f"Error status code: {status_code}")
            logger.info(f"Error: {error_msg}")
            if isinstance(detail, dict):
                # 结构化详情:逐项记录并取消息作为运行日志
                logger.info(f"job_id: {detail.get('job_id')}")
                logger.info(f"run_status: {detail.get('status')}")
                if detail.get("message"):
                    logger.info(f"message: {detail['message']}")
                run_log = detail.get("message", str(detail))
            else:
                logger.info(f"detail: {detail}")
                run_log = str(detail)
            result_info = detail
            base_reward = 0.0

        # 回写分数/耗时与两种映射模式
        sample.metadata["score"] = score
        sample.metadata["running_time"] = running_time
        sample.metadata["reward_mapping_mode"] = REWARD_MAPPING_MODE
        sample.metadata["static_reward_mapping_mode"] = STATIC_REWARD_MAPPING_MODE
        # 进入组级奖励映射屏障:组到齐后按动态边界把分数映射为奖励
        reward_mapping_result = await _resolve_group_reward_mapping(
            sample.metadata,
            group_index=group_index,
            group_size=n_samples_per_prompt,
            sample_key=f"{task_id}:{group_index}:{id(sample)}",
            score=score,
            base_reward=base_reward,
        )
        # 覆盖为屏障统一映射后的基础奖励
        base_reward = float(reward_mapping_result["base_reward"])
        adaptive_context = reward_mapping_result.get("context") or {}
        # 上下文写回样本 metadata
        _apply_dynamic_bound_context_metadata(sample, adaptive_context, n_samples_per_prompt)
        if bool(adaptive_context.get("group_generation_aborted")):
            # 组内发生生成中断:整组放弃本次结果并重排
            sample.metadata["generation_abort_reason"] = "group_generation_aborted"
            return _return_generation_abort_for_retry(sample, task_name=task_name, group_index=group_index)
        if USE_SCORE2REWARD:
            # 打印动态边界与映射后的奖励,便于审计
            logger.info(
                "score: %s, adaptive_bound_mode=%s, best_signed=%s, worst_signed=%s, mapped_base_reward=%s",
                score,
                adaptive_context.get("bound_mode"),
                adaptive_context.get("best_signed"),
                adaptive_context.get("worst_signed"),
                base_reward,
            )

        # 静态口径的基础奖励(固定边界,不随组变化)
        static_base_reward_raw = compute_static_score_reward(score, sample.metadata)
        # 指标静态基础奖励(仅指标视角,可能为 None)
        metric_static_base_reward = compute_metric_static_base_reward(score, sample.metadata)
        # 重新确认静态边界可用性(执行后 metadata 可能更新)
        metric_static_base_reward_has_bounds = int(
            has_static_bounds_with_priority(
                sample.metadata,
                priority="leaderboard",
                fallback_best=metric_fallback_best,
                fallback_worst=metric_fallback_worst,
            )
        )
        # 计算验证集与测试集分数的差距信息(防"验证集过拟合"式刷分)
        validation_gap_info = validation_test_gap_info(validation_score, score, sample.metadata)

        # 记录惩罚前的三类奖励,便于审计
        dynamic_base_reward_before_gap = float(base_reward)
        static_base_reward_before_gap = float(static_base_reward_raw)
        metric_static_base_reward_before_gap = metric_static_base_reward

        # 依配置对验证-测试差距施加惩罚(乘性衰减)
        base_reward, gap_penalty, gap_multiplier = apply_validation_test_gap_penalty(
            base_reward,
            validation_gap_info.get("relative_gap"),
            VALIDATION_TEST_GAP_PENALTY_COEF,
        )
        # 静态口径同样施加差距惩罚
        static_base_reward_raw, _, _ = apply_validation_test_gap_penalty(
            static_base_reward_raw,
            validation_gap_info.get("relative_gap"),
            VALIDATION_TEST_GAP_PENALTY_COEF,
        )
        if metric_static_base_reward is not None:
            # 指标静态口径存在时同样惩罚
            metric_static_base_reward, _, _ = apply_validation_test_gap_penalty(
                metric_static_base_reward,
                validation_gap_info.get("relative_gap"),
                VALIDATION_TEST_GAP_PENALTY_COEF,
            )
        # 打印差距与惩罚详情
        logger.info(
            "validation/test gap: validation=%s, test=%s, relative_gap=%s, penalty=%s, multiplier=%s",
            validation_gap_info.get("validation_score"),
            validation_gap_info.get("test_score"),
            validation_gap_info.get("relative_gap"),
            gap_penalty,
            gap_multiplier,
        )

        # 基线调整当前未启用(占位)
        baseline_value = None
        baseline_count = 0
        # 有效基础奖励 = 差距惩罚后的动态奖励
        effective_base_reward = float(base_reward)
        static_effective_base_reward = float(static_base_reward_raw)
        # 以下把验证-测试差距与各类奖励中间量写入 metadata(训练侧审计用)
        sample.metadata["validation_score"] = validation_gap_info.get("validation_score")
        sample.metadata["validation_test_abs_gap"] = validation_gap_info.get("abs_gap")
        sample.metadata["validation_test_relative_gap"] = validation_gap_info.get("relative_gap")
        sample.metadata["validation_test_gap_denominator"] = validation_gap_info.get("gap_denominator")
        sample.metadata["validation_test_gap_denominator_source"] = validation_gap_info.get("gap_denominator_source")
        sample.metadata["validation_test_gap_penalty_enabled"] = VALIDATION_TEST_GAP_PENALTY_ENABLED
        sample.metadata["validation_test_gap_penalty_piecewise_enabled"] = VALIDATION_TEST_GAP_PENALTY_PIECEWISE_ENABLED
        sample.metadata["validation_test_gap_penalty_coef"] = VALIDATION_TEST_GAP_PENALTY_COEF
        sample.metadata["validation_test_gap_penalty_high_coef"] = VALIDATION_TEST_GAP_PENALTY_HIGH_COEF
        sample.metadata["validation_test_gap_penalty_tolerance"] = VALIDATION_TEST_GAP_PENALTY_TOLERANCE
        sample.metadata["validation_test_gap_penalty"] = gap_penalty
        sample.metadata["validation_test_gap_multiplier"] = gap_multiplier
        sample.metadata["dynamic_base_reward_before_gap_penalty"] = dynamic_base_reward_before_gap
        sample.metadata["static_base_reward_before_gap_penalty"] = static_base_reward_before_gap
        sample.metadata["metric_static_base_reward_before_gap_penalty"] = metric_static_base_reward_before_gap
        sample.metadata["base_reward_raw"] = base_reward
        sample.metadata["base_reward_baseline_spec"] = "none"
        sample.metadata["base_reward_baseline_value"] = baseline_value
        sample.metadata["base_reward_baseline_count"] = baseline_count
        sample.metadata["base_reward_effective"] = effective_base_reward
        sample.metadata["dynamic_base_reward_raw"] = base_reward
        sample.metadata["dynamic_base_reward_effective"] = effective_base_reward
        sample.metadata["static_base_reward_raw"] = static_base_reward_raw
        sample.metadata["static_base_reward_effective"] = static_effective_base_reward
        sample.metadata["metric_static_base_reward"] = metric_static_base_reward
        sample.metadata["metric_static_base_reward_has_bounds"] = metric_static_base_reward_has_bounds

        # Calculate final reward based on mode
        # 依算子模式做奖励整形(improve/debug/crossover 相对父代,draft 直接取值)
        shaped_base_reward, final_reward, delta, improve_bonus = compute_mode_rewards(
            generation_mode=generation_mode,
            effective_base_reward=effective_base_reward,
            parent_base_reward=parent_dynamic_base_reward,
            parent_reward=parent_dynamic_raw_reward,
            parent_code=parent_code,
            code=code,
            crossover_parent_base_reward=crossover_parent_dynamic_base_reward,
            crossover_parent_reward=crossover_parent_dynamic_raw_reward,
            crossover_parent_code=crossover_parent_code,
            improve_reward_strategy=IMPROVE_REWARD_STRATEGY,
            improve_delta_bonus_coef=IMPROVE_DELTA_BONUS_COEF,
        )
        # 静态口径做同样的模式整形(用于对比审计)
        static_shaped_base_reward, static_final_reward, _, _ = compute_mode_rewards(
            generation_mode=generation_mode,
            effective_base_reward=static_effective_base_reward,
            parent_base_reward=parent_static_base_reward,
            parent_reward=parent_static_raw_reward,
            parent_code=parent_code,
            code=code,
            crossover_parent_base_reward=crossover_parent_static_base_reward,
            crossover_parent_reward=crossover_parent_static_raw_reward,
            crossover_parent_code=crossover_parent_code,
            improve_reward_strategy=IMPROVE_REWARD_STRATEGY,
            improve_delta_bonus_coef=IMPROVE_DELTA_BONUS_COEF,
        )
        if generation_mode in ("improve", "debug", "crossover"):
            # 改进型算子:打印基础奖励/改进量/加成
            logger.info(
                f"[{generation_mode.upper()} MODE] base_reward={shaped_base_reward:.6f} delta={delta:.6f} bonus={improve_bonus:.6f} final_reward={final_reward:.6f}"
            )
        else:
            # draft 算子:打印最终奖励
            logger.info(f"[DRAFT MODE] base_reward/final_reward: {final_reward}")
        # 记录改进量与加成
        sample.metadata["improve_delta"] = delta
        sample.metadata["improve_bonus"] = improve_bonus

        # 以下把整形后的各类奖励写回 metadata,并同步局部变量供入库
        sample.metadata["base_reward"] = shaped_base_reward
        sample.metadata["final_reward"] = final_reward
        sample.metadata["dynamic_base_reward"] = shaped_base_reward
        sample.metadata["static_base_reward"] = static_shaped_base_reward
        dynamic_raw_reward = final_reward
        dynamic_mercy_raw_reward = final_reward
        static_raw_reward = static_final_reward
        static_mercy_raw_reward = static_final_reward
        mercy_raw_reward = dynamic_mercy_raw_reward
        sample.metadata["dynamic_raw_reward"] = dynamic_raw_reward
        sample.metadata["dynamic_mercy_raw_reward"] = dynamic_mercy_raw_reward
        sample.metadata["static_raw_reward"] = static_raw_reward
        sample.metadata["static_mercy_raw_reward"] = static_mercy_raw_reward
        sample.metadata["mercy_raw_reward"] = mercy_raw_reward

    # 统一回写运行时间与分数(所有类别分支)
    sample.metadata["running_time"] = running_time
    sample.metadata["score"] = score
    # 验证-测试差距相关字段:用 setdefault 避免覆盖 valid 分支已写入的值
    sample.metadata.setdefault("validation_score", validation_gap_info.get("validation_score"))
    sample.metadata.setdefault("validation_test_abs_gap", validation_gap_info.get("abs_gap"))
    sample.metadata.setdefault("validation_test_relative_gap", validation_gap_info.get("relative_gap"))
    sample.metadata.setdefault("validation_test_gap_denominator", validation_gap_info.get("gap_denominator"))
    sample.metadata.setdefault(
        "validation_test_gap_denominator_source",
        validation_gap_info.get("gap_denominator_source"),
    )
    sample.metadata.setdefault("validation_test_gap_penalty_enabled", VALIDATION_TEST_GAP_PENALTY_ENABLED)
    sample.metadata.setdefault(
        "validation_test_gap_penalty_piecewise_enabled",
        VALIDATION_TEST_GAP_PENALTY_PIECEWISE_ENABLED,
    )
    sample.metadata.setdefault("validation_test_gap_penalty_coef", VALIDATION_TEST_GAP_PENALTY_COEF)
    sample.metadata.setdefault("validation_test_gap_penalty_high_coef", VALIDATION_TEST_GAP_PENALTY_HIGH_COEF)
    sample.metadata.setdefault("validation_test_gap_penalty_tolerance", VALIDATION_TEST_GAP_PENALTY_TOLERANCE)
    sample.metadata.setdefault("validation_test_gap_penalty", gap_penalty)
    sample.metadata.setdefault("validation_test_gap_multiplier", gap_multiplier)
    if code_category != "valid":
        # 非有效代码(hack/no_verify 等):补齐奖励字段并走组屏障
        sample.metadata["base_reward_raw"] = base_reward
        sample.metadata["base_reward_effective"] = effective_base_reward
        sample.metadata["base_reward"] = shaped_base_reward
        sample.metadata["final_reward"] = final_reward
        sample.metadata["mercy_raw_reward"] = mercy_raw_reward
        sample.metadata["dynamic_base_reward_raw"] = base_reward
        sample.metadata["dynamic_base_reward_effective"] = effective_base_reward
        sample.metadata["dynamic_base_reward"] = shaped_base_reward
        sample.metadata["dynamic_raw_reward"] = dynamic_raw_reward
        sample.metadata["dynamic_mercy_raw_reward"] = dynamic_mercy_raw_reward
        sample.metadata["static_base_reward_raw"] = base_reward
        sample.metadata["static_base_reward_effective"] = effective_base_reward
        sample.metadata["static_base_reward"] = shaped_base_reward
        sample.metadata["static_raw_reward"] = static_raw_reward
        sample.metadata["static_mercy_raw_reward"] = static_mercy_raw_reward
        sample.metadata["metric_static_base_reward"] = metric_static_base_reward
        sample.metadata["metric_static_base_reward_has_bounds"] = metric_static_base_reward_has_bounds
        # 仍要进入组级奖励映射屏障,保证组内所有样本同时释放(避免组屏障死等)
        reward_mapping_result = await _resolve_group_reward_mapping(
            sample.metadata,
            group_index=group_index,
            group_size=n_samples_per_prompt,
            sample_key=f"{task_id}:{group_index}:{id(sample)}",
            score=score,
            base_reward=base_reward,
        )
        dynamic_bound_context = reward_mapping_result.get("context") or {}
        _apply_dynamic_bound_context_metadata(sample, dynamic_bound_context, n_samples_per_prompt)
        if bool(dynamic_bound_context.get("group_generation_aborted")):
            # 组级中断:整组重排
            sample.metadata["generation_abort_reason"] = "group_generation_aborted"
            return _return_generation_abort_for_retry(sample, task_name=task_name, group_index=group_index)

    # 计算任务奖励前沿(本次更新前的历史最优),用于判断是否刷新纪录
    task_reward_frontier = _compute_task_reward_frontier(db=await get_program_database(), task_name=task_name, metadata_fallback=sample.metadata)
    # 记录更新前的动态/静态历史最优
    sample.metadata["task_best_dynamic_raw_reward_before"] = task_reward_frontier["best_dynamic_raw_reward_before"]
    sample.metadata["task_best_static_raw_reward_before"] = task_reward_frontier["best_static_raw_reward_before"]
    # 与历史最优的差距(负数表示已超越旧纪录)
    sample.metadata["dynamic_raw_reward_gap_to_task_best_before"] = dynamic_raw_reward - task_reward_frontier[
        "best_dynamic_raw_reward_before"
    ]
    sample.metadata["static_raw_reward_gap_to_task_best_before"] = static_raw_reward - task_reward_frontier[
        "best_static_raw_reward_before"
    ]
    # 是否刷新动态/静态历史最优(带 1e-12 容差)
    sample.metadata["dynamic_raw_reward_is_task_best_improvement"] = int(
        dynamic_raw_reward > task_reward_frontier["best_dynamic_raw_reward_before"] + 1e-12
    )
    sample.metadata["static_raw_reward_is_task_best_improvement"] = int(
        static_raw_reward > task_reward_frontier["best_static_raw_reward_before"] + 1e-12
    )
    sample.metadata["task_best_metric_static_base_reward_before"] = task_reward_frontier[
        "best_metric_static_base_reward_before"
    ]
    # 指标静态口径是否达到历史最优(需有边界且不低于历史最优)
    sample.metadata["metric_static_base_reward_is_task_best_improvement"] = int(
        metric_static_base_reward_has_bounds
        and metric_static_base_reward >= task_reward_frontier["best_metric_static_base_reward_before"] - 1e-12
    )
    # 更新历史最优并取全局均值
    sample.metadata["history_best_static_base_reward"] = await _update_history_best_static_base_reward(
        task_name,
        task_reward_frontier["best_metric_static_base_reward_before"],
        metric_static_base_reward,
    )

    # Store status_code in sample.metadata for logging
    if "status_codes" not in sample.metadata:
        sample.metadata["status_codes"] = []
    sample.metadata["status_codes"].append(status_code)

    # Add program to database
    db = await get_program_database()
    program_id = None
    # 构造程序对象入库:包含代码、执行分数、奖励、父代与算子模式(进化搜索的记忆载体)
    program = Program(
        task_id=task_id,
        task_name=task_name,
        status_code=status_code,
        payload=payload,
        code=code,
        score=score,
        running_time=running_time,
        reward=final_reward,  # Training reward (includes improve bonus when applicable)
        base_reward=shaped_base_reward,  # Original reward before extra improve/debug/crossover bonus
        parent_id=parent_id,
        parent_code=parent_code,
        generation_mode=generation_mode,
        raw_text=raw_text,
        hack=sample.metadata["hack"],  # Add hack check result
        metadata=sample.metadata,  # Keep metadata for additional info
    )
    # Store successful executions plus explicitly categorized invalid code we want to keep for analysis.
    # Connection/runtime failures and empty code are still skipped.
    # 入库条件:执行成功,或属于需要保留分析的作弊/无验证类别(连接失败与空代码不入库)
    should_store_program = status_code == 200 or code_category in {"hack", "no_verify", "hack_verify"}
    if should_store_program:
        # 写入数据库并取回分配的程序 ID
        program_id = db.add(program)
        logger.info(f"[DATABASE] Added program id={program_id} to database for task {task_name}")
        saved_program = db.get_by_id(program_id)
        # 读取数据库回填的探索/利用/冷却系数(airaevo 用)
        exploit_coefficient = saved_program.exploit_coefficient if saved_program is not None else None
        explore_coefficient = saved_program.explore_coefficient if saved_program is not None else None
        cooling_coefficient = saved_program.cooling_coefficient if saved_program is not None else None
        if saved_program is not None:
            # airaevo 模式下生成经验卡片/丰富摘要并合入 metadata
            airaevo_updates = await _maybe_update_airaevo_program_metadata(args, db, saved_program)
            if airaevo_updates:
                sample.metadata.update(airaevo_updates)
    else:
        # 不满足入库条件:记录跳过原因
        logger.info(
            f"[DATABASE] Skipped program insertion for task {task_name}: status_code={status_code}, code_category={code_category}"
        )
        exploit_coefficient = None
        explore_coefficient = None
        cooling_coefficient = None

    # 汇总组级动态边界统计信息到 metadata
    sample.metadata["dynamic_score_bounds_group_seen_count"] = sample.metadata.get("adaptive_reward_group_score_count")
    sample.metadata["dynamic_score_bounds_group_updated"] = True
    sample.metadata["dynamic_score_bounds_best_after"] = sample.metadata.get("dynamic_bound_best_signed")
    sample.metadata["dynamic_score_bounds_worst_after"] = sample.metadata.get("dynamic_bound_worst_signed")

    # Log training data for group tracking (CSV includes code_category and mercy_raw_reward)
    # 按组缓冲记录训练数据,凑满一组后写 CSV
    logger_instance.log_sample(
        sample=sample,
        score=score,
        reward=final_reward,  # Log the final reward used for training
        static_base_reward=metric_static_base_reward,
        base_reward=base_reward,  # Log base reward separately
        result_info=result_info,
        code=code,
        raw_text=raw_text,  # Log raw model response
        parent_id=parent_id,
        parent_code=parent_code,  # Log parent code
        mode=generation_mode,
        hack=sample.metadata["hack"],
        code_category=code_category,  # Log code category (CSV only)
        static_raw_reward=static_raw_reward,
        static_mercy_raw_reward=static_mercy_raw_reward,
        dynamic_raw_reward=dynamic_raw_reward,
        exploit_coefficient=exploit_coefficient,
        explore_coefficient=explore_coefficient,
        cooling_coefficient=cooling_coefficient,
    )

    # 返回训练用最终奖励
    return final_reward


async def save_db_snapshot(snapshot_path: str):
    """
    Save a snapshot of the program database.

    Args:
        snapshot_path: Path where the snapshot should be saved
    """
    # 将程序数据库快照持久化到指定路径(训练结束/定期保存进化成果)
    db = await get_program_database()
    db.save_snapshot(snapshot_path)
