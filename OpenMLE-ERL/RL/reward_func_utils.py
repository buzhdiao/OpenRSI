# 导入 time：重试退避 sleep 与耗时统计
import time
# 导入 httpx：同步/异步 HTTP 客户端（沙箱执行与 GPT 判卷接口）
import httpx
# 导入 asyncio：异步并发（rollout 中异步提交沙箱与判卷）
import asyncio
# 导入 re：正则（代码块提取、验证分打印检查）
import re
# 导入 logging：统一日志
import logging
# 导入 random：重试抖动（jitter）与代理沙箱的随机分数
import random
# 导入 black：对模型生成的代码做格式化
import black
# 导入 math：sigmoid、isfinite 等数值工具
import math
# 导入 json：解析 GPT 判卷返回的 JSON
import json
# 导入 pandas：读取公共排行榜 CSV
import pandas as pd
# 导入 Path：路径处理
from pathlib import Path
# 导入 os：读取环境变量
import os
# 导入 threading：判卷运行时注册表的进程内锁
import threading
# 导入 weakref：按事件循环弱引用缓存判卷运行时
import weakref
# 导入 argparse：命令行入口参数解析
import argparse

# Configure logging format with timestamp at module level
# 模块级 logger：mle_agent；已有 handler 时不重复添加
logger = logging.getLogger("mle_agent")
if not logger.handlers:
    # Set up console handler with timestamp format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

# 沙箱服务 API key（execution-grounded 奖励的代码执行后端）
SANDBOX_API_KEY = os.getenv("SANDBOX_API_KEY", "").strip()

# GPT API configuration for hack_check
# 尝试导入 OpenAI 客户端（同步与异步）；未安装时置 None 以便优雅跳过判卷
try:
    from openai import AsyncOpenAI, OpenAI
except ImportError:
    AsyncOpenAI = None
    OpenAI = None

# GPT 判卷（hack_check，反作弊评审）服务密钥
GPT_API_KEY = os.getenv("GPT_API_KEY", "").strip()
# GPT 判卷服务 base_url
GPT_BASE_URL = os.getenv("GPT_BASE_URL", "").strip()
# 沙箱服务地址（GPU 资源池）
GPU_BASE_URL = os.getenv("GPU_BASE_URL", "").strip()
# 沙箱服务地址（CPU 资源池）
CPU_BASE_URL = os.getenv("CPU_BASE_URL", "").strip()
# HuggingFace 镜像端点（沙箱内下载模型/数据集用），缺省 hf-mirror
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com").strip() or "https://hf-mirror.com"

# Sync hack_check blocks the rollout event loop. Parallel async path lets other
# coroutines proceed; HACK_CHECK_CONCURRENCY caps in-flight GPT judge calls.
# 判卷并发上限：同步版会阻塞 rollout 事件循环，异步版用它限制在途 GPT 判卷请求数
try:
    HACK_CHECK_CONCURRENCY = int(os.getenv("HACK_CHECK_CONCURRENCY", "64"))
except (TypeError, ValueError):
    # 环境变量非法时回落默认 64
    HACK_CHECK_CONCURRENCY = 64
# 并发上限至少为 1
HACK_CHECK_CONCURRENCY = max(1, HACK_CHECK_CONCURRENCY)

# 判卷运行时注册表的进程内锁
_HACK_CHECK_RUNTIME_LOCK = threading.Lock()
# 每个事件循环各自的（客户端, 信号量）运行时缓存；弱引用使循环销毁后缓存自动回收
_HACK_CHECK_RUNTIMES = weakref.WeakKeyDictionary()

# power_clip 奖励模式的幂次 alpha（score2reward 用），可由环境覆盖
try:
    POWER_CLIP_ALPHA = float(os.getenv("POWER_CLIP_ALPHA", "2.0"))
except (TypeError, ValueError):
    # 非法值回落默认 2.0
    POWER_CLIP_ALPHA = 2.0
# NaN/Inf 同样回落默认
if not math.isfinite(POWER_CLIP_ALPHA):
    POWER_CLIP_ALPHA = 2.0
# 幂次不允许为负
POWER_CLIP_ALPHA = max(POWER_CLIP_ALPHA, 0.0)

# 匹配整行 "print(f"Final Validation Score: {...}")" 语句的正则（检查生成代码是否按要求打印验证分）
FINAL_VALIDATION_SCORE_PRINT_RE = re.compile(
    r"""^[ \t]*print\s*\(\s*f(?P<quote>['"])Final Validation Score:\s*\{\s*[^{}\n]+\s*\}(?P=quote)\s*\)\s*$""",
    re.MULTILINE,
)
# 从运行输出文本中提取 "Final Validation Score: <数值>" 的正则
FINAL_VALIDATION_SCORE_VALUE_RE = re.compile(r"Final Validation Score:\s*([-+eE0-9.]+)")


def _env_flag(name: str, default: bool) -> bool:
    # 读取布尔型环境变量
    raw = os.getenv(name)
    # 未设置时返回默认值
    if raw is None:
        return default
    # "0/false/no/off/空串" 视为 False，其余任何值视为 True
    return str(raw).strip().lower() not in ("0", "false", "no", "off", "")


# 是否启用“验证分-测试分差距惩罚”（抑制对验证集过拟合的解，属于 execution-grounded 奖励的防作弊项）
VALIDATION_TEST_GAP_PENALTY_ENABLED = _env_flag("VALIDATION_TEST_GAP_PENALTY_ENABLED", True)
# 是否启用分段惩罚（超出容差的部分用更高系数加罚）
VALIDATION_TEST_GAP_PENALTY_PIECEWISE_ENABLED = _env_flag("VALIDATION_TEST_GAP_PENALTY_PIECEWISE_ENABLED", False)
# 基础惩罚系数：相对差距每 1 倍容差扣减奖励的比例
try:
    VALIDATION_TEST_GAP_PENALTY_COEF = float(os.getenv("VALIDATION_TEST_GAP_PENALTY_COEF", "0.05"))
except (TypeError, ValueError):
    # 非法值回落默认 0.05
    VALIDATION_TEST_GAP_PENALTY_COEF = 0.05
# 非有限值回落默认
if not math.isfinite(VALIDATION_TEST_GAP_PENALTY_COEF):
    VALIDATION_TEST_GAP_PENALTY_COEF = 0.05
# 基础系数不允许为负
VALIDATION_TEST_GAP_PENALTY_COEF = max(VALIDATION_TEST_GAP_PENALTY_COEF, 0.0)
# 高段惩罚系数：超出容差部分的加罚系数
try:
    VALIDATION_TEST_GAP_PENALTY_HIGH_COEF = float(os.getenv("VALIDATION_TEST_GAP_PENALTY_HIGH_COEF", "0.25"))
except (TypeError, ValueError):
    # 非法值回落默认 0.25
    VALIDATION_TEST_GAP_PENALTY_HIGH_COEF = 0.25
# 非有限值回落默认
if not math.isfinite(VALIDATION_TEST_GAP_PENALTY_HIGH_COEF):
    VALIDATION_TEST_GAP_PENALTY_HIGH_COEF = 0.25
# 高段系数不允许为负
VALIDATION_TEST_GAP_PENALTY_HIGH_COEF = max(VALIDATION_TEST_GAP_PENALTY_HIGH_COEF, 0.0)
# 差距容差：相对差距在此范围内不惩罚
try:
    VALIDATION_TEST_GAP_PENALTY_TOLERANCE = float(os.getenv("VALIDATION_TEST_GAP_PENALTY_TOLERANCE", "0.12"))
except (TypeError, ValueError):
    # 非法值回落默认 0.12
    VALIDATION_TEST_GAP_PENALTY_TOLERANCE = 0.12
# 容差必须是正的有限值
if not math.isfinite(VALIDATION_TEST_GAP_PENALTY_TOLERANCE) or VALIDATION_TEST_GAP_PENALTY_TOLERANCE <= 0.0:
    VALIDATION_TEST_GAP_PENALTY_TOLERANCE = 0.12


def extract_json(text: str) -> str:
    """Extract JSON content from text, removing markdown code blocks if present."""
    # 去首尾空白
    text = text.strip()
    # Remove markdown code blocks
    # 以 ``` 开头说明包着 markdown 围栏，需要剥掉
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    # 返回纯 JSON 文本
    return text


def has_final_validation_score_print(code: str) -> bool:
    """Check whether the code prints a Final Validation Score via an f-string."""
    # 用正则检查代码里是否存在规定格式的验证分打印语句
    return bool(FINAL_VALIDATION_SCORE_PRINT_RE.search(code))


def extract_validation_score(text: str | None) -> float | None:
    """Extract the final validation score printed by generated code."""
    # 空文本直接返回 None
    if not text:
        return None
    # 在运行输出中查找验证分数值
    match = FINAL_VALIDATION_SCORE_VALUE_RE.search(str(text))
    # 找不到返回 None
    if not match:
        return None
    try:
        # 捕获组转 float
        value = float(match.group(1))
    except (TypeError, ValueError):
        # 转换失败视为未提取到
        return None
    # 非有限值（NaN/Inf）视为无效
    return value if math.isfinite(value) else None


def _build_hack_check_messages(code: str, require_holdout_validation: bool) -> list[dict[str, str]]:
    """Shared judge prompts for sync and async hack_check paths."""
    # 基础判卷规则：列出六类常见作弊模式（乱猜、常数输出、硬编码规则、空/平凡预测、不用数据、不训练）
    base_rules = (
        "You are an expert ML engineer reviewer. You will be shown Python code "
        "that is supposed to solve a machine learning task. Your job is to determine "
        "if the code is genuinely applying machine learning techniques, or if it's "
        "trying to cheat by:\n"
        "1. Simply guessing/returning random values\n"
        "2. Returning constant values without any learning\n"
        "3. Using hard-coded rules instead of learning from data\n"
        "4. Outputting empty or trivial predictions\n"
        "5. Not loading or using the provided data\n"
        "6. Not training any model at all\n"
    )
    if require_holdout_validation:
        # 严格模式：额外要求验证分必须真的在留出验证集上算出（防 hack_verify 类造假验证）
        system_prompt = (
            base_rules
            + "7. Printing a fake validation metric without actually computing it on a hold-out validation set\n"
            + "8. Printing a metric from the training set, leaked labels, cross-validation only, or a hard-coded/proxy value while pretending it is hold-out validation\n\n"
            + "The code is ONLY valid if its final printed validation metric is genuinely computed on a hold-out validation set. "
            + "If it prints `Final Validation Score` but does not create/use a real hold-out validation split and evaluate on it, "
            + "mark it invalid with category `hack_verify`.\n\n"
            + "Respond with valid JSON matching the schema:\n"
            + "{"
            + '"is_valid": <true|false>, '
            + '"category": <"valid"|"hack"|"hack_verify">, '
            + '"reason": <string explaining your decision>'
            + "}."
        )
    else:
        # 普通模式：只判是否作弊，不校验验证集流程
        system_prompt = (
            base_rules
            + "\nRespond with valid JSON matching the schema:\n"
            + "{"
            + '"is_valid": <true|false>, '
            + '"category": <"valid"|"hack">, '
            + '"reason": <string explaining your decision>'
            + "}."
        )
    # 用户消息：附上待判卷的代码
    user_prompt = (
        "Please analyze the following Python code and determine if it's a legitimate "
        "machine learning implementation or if it's cheating/guessing.\n\n"
        "```python\n"
        f"{code}\n"
        "```"
    )
    # 返回标准 chat 消息列表
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _is_retryable_hack_check_error(exc: Exception) -> bool:
    # 网络超时/连接类错误可重试
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    try:
        import ssl

        # SSL 层错误可重试
        if isinstance(exc, ssl.SSLError):
            return True
    except Exception:
        # ssl 模块不可用时跳过该检查
        pass
    # 错误消息转小写做关键词匹配
    msg = str(exc).lower()
    # 可重试错误的关键词集合（超时、连接中断、TLS/SSL、握手等）
    retryable_substrings = (
        "read timed out",
        "timed out",
        "timeout",
        "connection reset",
        "connection aborted",
        "connection refused",
        "tls",
        "ssl",
        "eof occurred",
        "handshake",
    )
    # 任一关键词命中即视为瞬态可重试错误
    return any(item in msg for item in retryable_substrings)


def _parse_hack_check_completion(completion, require_holdout_validation: bool) -> tuple[bool, str, str]:
    # 取出判卷模型返回的文本
    content = completion.choices[0].message.content
    # 空响应：保守放行并说明原因
    if content is None:
        logger.error("Empty response from GPT API in hack_check")
        return True, "valid", "GPT API returned empty response"
    # 剥掉可能的 markdown 围栏
    content = extract_json(content)
    try:
        # 解析判卷 JSON
        parsed = json.loads(content)
        # 读取 is_valid（缺省 True）
        is_valid = parsed.get("is_valid", True)
        # 读取判卷理由
        reason = parsed.get("reason", "No reason provided")
        # 读取类别；缺失时按 is_valid 推断
        category = parsed.get("category", "valid" if is_valid else "hack")
        if is_valid:
            # 判定有效
            return True, "valid", reason
        # 严格模式下允许 hack_verify 类别，否则只允许 hack
        allowed_categories = {"hack", "hack_verify"} if require_holdout_validation else {"hack"}
        # 类别不在允许范围时归一为 hack
        if category not in allowed_categories:
            category = "hack"
        # 返回无效判定（类别, 理由）
        return False, category, reason
    except json.JSONDecodeError as exc:
        # JSON 解析失败：保守放行（判卷故障不阻塞训练，宁可漏判）
        logger.error("Failed to parse GPT response in hack_check: %s", content)
        return True, "valid", f"Failed to parse GPT response: {str(exc)}"


def _get_async_hack_check_runtime():
    """Per-event-loop AsyncOpenAI client + concurrency semaphore."""
    # 未安装 openai 包：返回空运行时
    if AsyncOpenAI is None:
        return None, None
    # 缺少 API key 配置则报错
    if not GPT_API_KEY:
        raise RuntimeError("Missing GPT_API_KEY")
    # 缺少 base_url 配置则报错
    if not GPT_BASE_URL:
        raise RuntimeError("Missing GPT_BASE_URL")
    # 取当前运行中的事件循环
    loop = asyncio.get_running_loop()
    # 加锁检查/创建该循环专属的运行时（避免并发重复初始化）
    with _HACK_CHECK_RUNTIME_LOCK:
        runtime = _HACK_CHECK_RUNTIMES.get(loop)
        if runtime is None:
            # 首次使用：建客户端（连接10s/读30s/写30s/池10s 超时）与并发信号量
            timeout = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0)
            runtime = (
                AsyncOpenAI(api_key=GPT_API_KEY, base_url=GPT_BASE_URL, timeout=timeout),
                asyncio.Semaphore(HACK_CHECK_CONCURRENCY),
            )
            # 缓存到弱引用表：事件循环销毁后自动回收
            _HACK_CHECK_RUNTIMES[loop] = runtime
            logger.info("[HACK_CHECK] initialized async runtime: concurrency=%s", HACK_CHECK_CONCURRENCY)
    # 返回（客户端, 信号量）
    return runtime


def hack_check(
    code: str,
    model: str = "o3-mini",
    temperature: float = 0.0,
    require_holdout_validation: bool = False,
) -> tuple[bool, str, str]:
    """
    Sync GPT judge for cheating/guessing code. Prefer hack_check_async in rollout.
    """
    # 未安装 openai 包：跳过判卷并放行
    if OpenAI is None:
        logger.warning("OpenAI package not installed, skipping hack_check")
        return True, "valid", "OpenAI package not available"
    # 缺少 API key 则报错
    if not GPT_API_KEY:
        raise RuntimeError("Missing GPT_API_KEY")
    # 缺少 base_url 则报错
    if not GPT_BASE_URL:
        raise RuntimeError("Missing GPT_BASE_URL")

    try:
        # 重试参数：最多 5 次，指数退避 1s 起、上限 12s
        max_retries = 5
        base_delay_s = 1.0
        max_delay_s = 12.0
        # HTTP 超时配置：连接 10s / 读 30s / 写 30s / 连接池 10s
        timeout = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0)
        # 创建同步 OpenAI 客户端
        client = OpenAI(api_key=GPT_API_KEY, base_url=GPT_BASE_URL, timeout=timeout)
        # 组装判卷消息（系统规则 + 待检代码）
        messages = _build_hack_check_messages(code, require_holdout_validation)

        completion = None
        last_exc: Exception | None = None
        # 重试循环：仅对瞬态错误退避重试
        for attempt in range(1, max_retries + 1):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=messages,
                )
                # 成功则清空异常记录并跳出
                last_exc = None
                break
            except Exception as exc:
                # 记录最近一次异常
                last_exc = exc
                # 达到重试上限或错误不可重试：直接抛出
                if (attempt >= max_retries) or (not _is_retryable_hack_check_error(exc)):
                    raise
                # 指数退避：min(上限, 基础延迟 * 2^(attempt-1))
                delay = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
                # 叠加随机抖动，避免多协程同刻重试
                delay = delay + random.uniform(0.0, min(0.5, 0.1 * delay))
                logger.warning(
                    "Transient GPT API error in hack_check (attempt %s/%s): %s: %s; retrying in %.2fs",
                    attempt,
                    max_retries,
                    type(exc).__name__,
                    str(exc),
                    delay,
                )
                # 同步 sleep 等待后重试
                time.sleep(delay)

        if completion is None:
            # 防御分支：循环结束仍无结果
            if last_exc is not None:
                logger.error("hack_check failed without completion: %s: %s", type(last_exc).__name__, str(last_exc))
            # 无完成结果：保守放行
            return True, "valid", "GPT API call failed (no completion)"

        # 解析判卷结果（是否有效, 类别, 理由）
        return _parse_hack_check_completion(completion, require_holdout_validation)

    except Exception as e:
        # 任何异常都保守放行：判卷故障不得阻塞训练主流程
        logger.error(f"Error in hack_check: {type(e).__name__}: {str(e)}")
        return True, "valid", f"Error during check: {type(e).__name__}"


async def hack_check_async(
    code: str,
    model: str = "o3-mini",
    temperature: float = 0.0,
    require_holdout_validation: bool = False,
) -> tuple[bool, str, str]:
    """
    Async concurrent hack_check for rollout. Cap via HACK_CHECK_CONCURRENCY (default 64).
    """
    # 未安装异步客户端：跳过判卷并放行
    if AsyncOpenAI is None:
        logger.warning("AsyncOpenAI package not installed, skipping async hack_check")
        return True, "valid", "AsyncOpenAI package not available"
    # 缺少 API key 则报错
    if not GPT_API_KEY:
        raise RuntimeError("Missing GPT_API_KEY")
    # 缺少 base_url 则报错
    if not GPT_BASE_URL:
        raise RuntimeError("Missing GPT_BASE_URL")

    # 计时：总耗时起点
    total_started = time.perf_counter()
    # 信号量排队等待时长
    queue_wait_s = 0.0
    # API 调用耗时
    api_elapsed_s = 0.0
    # 结果状态（供 finally 的耗时日志使用）
    result_status = "error"
    try:
        # 获取（或初始化）当前事件循环的判卷运行时
        client, semaphore = _get_async_hack_check_runtime()
        if client is None or semaphore is None:
            # 运行时不可用：跳过判卷
            logger.warning("Async hack_check runtime unavailable, skipping check")
            result_status = "runtime_unavailable"
            return True, "valid", "Async hack_check runtime unavailable"

        # 记录排队起点
        queue_started = time.perf_counter()
        # 信号量限流：同一事件循环最多 HACK_CHECK_CONCURRENCY 个在途判卷请求
        async with semaphore:
            # 计算排队等待时长
            queue_wait_s = time.perf_counter() - queue_started
            # 记录 API 调用起点
            api_started = time.perf_counter()
            # 重试参数：最多 5 次，指数退避 1s 起、上限 12s
            max_retries = 5
            base_delay_s = 1.0
            max_delay_s = 12.0
            # 组装判卷消息
            messages = _build_hack_check_messages(code, require_holdout_validation)
            completion = None
            last_exc: Exception | None = None
            # 重试循环：仅对瞬态错误退避重试
            for attempt in range(1, max_retries + 1):
                try:
                    completion = await client.chat.completions.create(
                        model=model,
                        temperature=temperature,
                        messages=messages,
                    )
                    # 成功则清空异常记录并跳出
                    last_exc = None
                    break
                except Exception as exc:
                    # 记录最近一次异常
                    last_exc = exc
                    # 达到上限或错误不可重试：抛出
                    if (attempt >= max_retries) or (not _is_retryable_hack_check_error(exc)):
                        raise
                    # 指数退避
                    delay = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
                    # 叠加随机抖动
                    delay = delay + random.uniform(0.0, min(0.5, 0.1 * delay))
                    logger.warning(
                        "Transient GPT API error in async hack_check (attempt %s/%s): %s: %s; retrying in %.2fs",
                        attempt,
                        max_retries,
                        type(exc).__name__,
                        str(exc),
                        delay,
                    )
                    # 异步 sleep 等待后重试（不阻塞事件循环）
                    await asyncio.sleep(delay)

            # 统计 API 调用耗时
            api_elapsed_s = time.perf_counter() - api_started
            if completion is None:
                # 防御分支：无完成结果时保守放行
                if last_exc is not None:
                    logger.error(
                        "async hack_check failed without completion: %s: %s",
                        type(last_exc).__name__,
                        str(last_exc),
                    )
                result_status = "no_completion"
                return True, "valid", "GPT API call failed (no completion)"

            # 正常完成
            result_status = "completed"
            # 解析判卷结果
            return _parse_hack_check_completion(completion, require_holdout_validation)
    except Exception as exc:
        # 任何异常保守放行：判卷故障不阻塞 rollout
        logger.error("Error in async hack_check: %s: %s", type(exc).__name__, str(exc))
        return True, "valid", f"Error during check: {type(exc).__name__}"
    finally:
        # 无论成败都输出判卷耗时日志（区分排队与 API 耗时，便于调并发上限）
        total_elapsed_s = time.perf_counter() - total_started
        logger.info(
            "[HACK_CHECK_TIMING] status=%s cap=%s queue_wait_s=%.3f api_elapsed_s=%.3f total_elapsed_s=%.3f",
            result_status,
            HACK_CHECK_CONCURRENCY,
            queue_wait_s,
            api_elapsed_s,
            total_elapsed_s,
        )


def get_clear_log(run_log: str | None) -> str:
    """
    Extract core output from run_log using the wrapper markers:
      - --- OUTPUT START --- ~~~ --- OUTPUT END ---
      - --- SANDBOX STDOUT START --- ~~~ --- SANDBOX STDOUT END ---
    Remove all log content enclosed by [HB] ... [HB].
    """
    # 空日志直接返回空串
    if not run_log:
        return ""

    # 两对包装标记：整体输出段与沙箱标准输出段
    markers = [
        ("--- OUTPUT START ---", "--- OUTPUT END ---"),
        ("--- SANDBOX STDOUT START ---", "--- SANDBOX STDOUT END ---"),
    ]

    # 心跳标记及其长度（心跳块内的日志要整体剔除）
    hb_marker = "[HB]"
    hb_len = len(hb_marker)

    # 收集的各段核心输出
    sections: list[str] = []

    for start_marker, end_marker in markers:
        # 每对标记都从头搜索（支持日志中多次出现）
        search_pos = 0
        while True:
            # 查找下一段的起始标记
            start_idx = run_log.find(start_marker, search_pos)
            if start_idx == -1:
                # 没有更多段：该对标记扫描结束
                break

            # 内容从起始标记之后开始；若紧跟换行则跳过该换行
            content_start = start_idx + len(start_marker)
            if content_start < len(run_log) and run_log[content_start] == "\n":
                content_start += 1

            # 查找结束标记；找不到就取到日志末尾
            end_idx = run_log.find(end_marker, content_start)
            if end_idx == -1:
                content = run_log[content_start:].strip()
            else:
                content = run_log[content_start:end_idx].strip()

            # Remove every [HB] ... [HB] block with string search.
            # 用字符串搜索剔除所有 [HB] ... [HB] 心跳块
            if content:
                parts: list[str] = []
                pos = 0
                while True:
                    # 找心跳块的起始标记
                    s = content.find(hb_marker, pos)
                    if s == -1:
                        # 没有更多块：收下剩余内容
                        parts.append(content[pos:])
                        break
                    # 收下块之前的内容
                    parts.append(content[pos:s])
                    # 找心跳块的结束标记
                    e = content.find(hb_marker, s + hb_len)
                    if e == -1:
                        # Discard an unclosed block from the opening marker onward.
                        break
                    # 跳过整个块，从结束标记之后继续扫描
                    pos = e + hb_len

                # 拼回剔除心跳后的内容并去首尾空白
                content = "".join(parts).strip()

            if content:
                # 非空段落才收集（前面补换行做分隔）
                sections.append(f"\n{content}")

            if end_idx == -1:
                # 结束标记缺失：该对标记扫描结束
                break
            # 从结束标记之后继续找下一段
            search_pos = end_idx + len(end_marker)

    # 用空行连接所有段落得到干净的核心输出
    return "\n\n".join(sections)


async def get_sandbox_result(
    client: httpx.AsyncClient,  # Reuse the caller-provided client.
    code_str: str,
    data_dir: str,
    *,
    resource_type: str = "gpu",
    priority: int = 1,
    job_timeout: int = 3600,
    wait_timeout: int = 7200,
    # wait_timeout: int = 100,
    poll_interval: int = 5,
):
    # 缺少沙箱 API key 则报错
    if not SANDBOX_API_KEY:
        raise RuntimeError("Missing SANDBOX_API_KEY")
    # 缺少沙箱服务地址则报错
    if not GPU_BASE_URL:
        raise RuntimeError("Missing GPU_BASE_URL")
    # 资源类型固定为 gpu（当前后端约定）
    normalized_resource = "gpu"
    # 优先级只允许 1/2，其他值归一为 1
    normalized_priority = priority if priority in (1, 2) else 1
    # 组装提交作业载荷：代码、数据目录、超时、资源类型与环境变量（execution-grounded 执行入口）
    payload = {
        "name": data_dir,
        "code": code_str,
        "data_dir": data_dir,
        "timeout": job_timeout,
        "resource_type": normalized_resource,
        "priority": normalized_priority,
        "environment": {
            "EXECUTION_MODE": "shell",
            "DATA_DIR": data_dir,
            "SANDBOX_DATA_DIR": data_dir,
            "HF_ENDPOINT": HF_ENDPOINT,
        },
    }
    # 认证请求头
    headers = {"X-API-Key": SANDBOX_API_KEY}

    def safe_json(resp: httpx.Response):
        # 尝试解析响应 JSON；空响应或解析异常都返回空 dict
        try:
            data = resp.json()
            return {} if data is None else data
        except Exception:
            return {}

    def format_httpx_error(exc: Exception, fallback_url: str) -> str:
        # 组装易读的错误描述：类型 + 方法/URL + 消息 + 根因
        parts = [f"type={type(exc).__name__}"]
        # 尝试从异常对象上取原始请求
        req = getattr(exc, "request", None)
        if req is not None:
            # 有请求对象：记录请求方法与 URL
            method = getattr(req, "method", "UNKNOWN")
            url = getattr(req, "url", fallback_url)
            parts.append(f"method={method}")
            parts.append(f"url={url}")
        else:
            # 无请求对象：记录兜底 URL
            parts.append(f"url={fallback_url}")

        # 异常消息；为空时退回 repr
        message = str(exc).strip()
        if message:
            parts.append(f"message={message}")
        else:
            parts.append(f"repr={exc!r}")

        # 记录异常链根因
        cause = getattr(exc, "__cause__", None)
        if cause is not None:
            cause_message = str(cause).strip()
            if cause_message:
                parts.append(f"cause={type(cause).__name__}: {cause_message}")
            else:
                parts.append(f"cause={cause!r}")
        # 用分号拼接各部分
        return "; ".join(str(p) for p in parts)

    # 1) Submit the job without waiting for completion.
    # 提交阶段的重试计数
    submit_connect_retries = 0
    # 提交阶段最大重试次数
    max_submit_connect_retries = 20
    # 提交响应
    submit_resp = None
    # 提交循环：对瞬态错误重试，直到成功或重试耗尽
    while True:
        try:
            submit_resp = await client.post(
                "/api/v1/jobs",  # Use a relative path with the client's base_url.
                json=payload,
                headers=headers,
            )
            # 网关类(502/503/504)与限流(429)状态码视为瞬态错误
            if submit_resp is None or submit_resp.status_code in (502, 503, 504, 429):
                if submit_connect_retries < max_submit_connect_retries:
                    # 还有重试额度：计数、等待 1 秒后重试
                    submit_connect_retries += 1
                    await asyncio.sleep(float(1))
                    print(
                        f"WARNING: transient error {submit_resp.status_code} when submitting job, retrying...",
                        flush=True,
                    )
                    continue
            # 正常响应（或重试耗尽）：跳出提交循环
            break
        except Exception as e:
            # 连接类异常：格式化错误详情
            error_detail = format_httpx_error(e, GPU_BASE_URL)
            if submit_connect_retries < max_submit_connect_retries:
                # 还有重试额度：线性递增延迟加抖动后重试
                submit_connect_retries += 1
                retry_delay = 1.0 * submit_connect_retries + random.uniform(0.0, 1.0)
                print(
                    f"WARNING: transient submit connect error to {GPU_BASE_URL}: {error_detail}; "
                    f"retry {submit_connect_retries}/{submit_connect_retries} in {retry_delay:.2f}s",
                    flush=True,
                )
                await asyncio.sleep(retry_delay)
                continue
            # 重试耗尽：返回连接失败
            error_msg = f"Failed to connect to sandbox API at {GPU_BASE_URL}: {error_detail}"
            print(f"ERROR: {error_msg}")
            return 503, {"error": "connection_failed", "detail": error_msg, "type": type(e).__name__}
        # except httpx.TimeoutException as e:
        #     error_msg = f"Failed to submit job to sandbox API at {GPU_BASE_URL}: {format_httpx_error(e, GPU_BASE_URL)}"
        #     print(f"ERROR: {error_msg}")
        #     return 503, {"error": "submit_timeout", "detail": error_msg, "type": type(e).__name__}
        # except httpx.TransportError as e:
        #     error_msg = f"Failed to submit job to sandbox API at {GPU_BASE_URL}: {format_httpx_error(e, GPU_BASE_URL)}"
        #     print(f"ERROR: {error_msg}")
        #     return 503, {"error": "submit_transport_failed", "detail": error_msg, "type": type(e).__name__}

    if submit_resp is None:
        # 极端防御：从未拿到任何响应
        return 503, {
            "error": "connection_failed",
            "detail": f"Failed to connect to sandbox API at xxx",
            "type": "ConnectError",
        }
    # 解析提交响应体
    submit_data = safe_json(submit_resp)
    print("submit_data:", submit_data)
    # 4xx/5xx：提交失败，返回状态码与响应数据
    if submit_resp.status_code >= 400:
        return submit_resp.status_code, {"error": "submit failed", "data": submit_data}

    # 取出作业 id
    job_id = submit_data.get("job_id")
    if not job_id:
        # 服务端未返回作业 id：视为服务端错误
        return 500, {"error": "no job_id returned", "data": submit_data}

    # 2) Poll until the job reaches a terminal state or the deadline expires.
    # 仍在进行中的状态集合
    wait_status = ["running", "queued"]
    # 终态集合
    finished_status = ["success", "failed", "timeout", "canceled", "completed"]
    # 轮询截止时间（单调时钟，避免系统时间跳变）
    deadline = time.monotonic() + wait_timeout
    # 轮询错误重试计数
    poll_error_retries = 0
    # 轮询最大错误重试次数
    max_poll_error_retries = 20
    # 累计处于 running 状态的时长
    total_running_time = 0.0
    # 上一次轮询到的状态
    last_status = None
    # 上一次轮询时间戳
    last_status_poll_at = None
    # 轮询主循环：截止时间前持续查询
    while time.monotonic() < deadline:
        try:
            # 查询作业状态
            r = await client.get(f"/api/v1/jobs/{job_id}", headers=headers)
            # 网关/限流状态码视为瞬态错误
            if r is None or r.status_code in (502, 503, 504, 429):
                if poll_error_retries < max_poll_error_retries:
                    # 还有重试额度：退避重试
                    poll_error_retries += 1
                    retry_delay = min(5.0, 1.0 + 0.25 * poll_error_retries) + random.uniform(0.0, 0.5)
                    print(
                        f"WARNING: transient poll status {r.status_code} for {job_id}, "
                        f"retry {poll_error_retries}/{max_poll_error_retries} in {retry_delay:.2f}s",
                        flush=True,
                    )
                    await asyncio.sleep(retry_delay)
                    continue
        except Exception as e:
            # 传输类异常：格式化详情
            error_detail = format_httpx_error(e, f"{GPU_BASE_URL}/api/v1/jobs/{job_id}")
            if poll_error_retries < max_poll_error_retries:
                # 还有重试额度：退避重试
                poll_error_retries += 1
                retry_delay = min(5.0, 1.0 + 0.25 * poll_error_retries) + random.uniform(0.0, 0.5)
                print(
                    f"WARNING: transient poll transport error for {job_id}: {error_detail}; "
                    f"retry {poll_error_retries}/{max_poll_error_retries} in {retry_delay:.2f}s",
                    flush=True,
                )
                await asyncio.sleep(retry_delay)
                continue
            # 重试耗尽：返回轮询失败
            error_msg = f"Failed to poll job status: {error_detail}"
            print(f"ERROR: {error_msg}")
            return 503, {"error": "poll_failed", "detail": error_msg, "job_id": job_id, "type": type(e).__name__}
        # 本次轮询成功：清零错误重试计数
        poll_error_retries = 0
        # 解析状态数据
        data = safe_json(r)
        # 当前作业状态
        status = data.get("status")
        # 本次轮询时间戳
        current_poll_at = time.monotonic()
        if last_status == "running" and last_status_poll_at is not None:
            # 上一区间也处于 running：累加运行时长
            total_running_time += max(0.0, current_poll_at - last_status_poll_at)
        # 更新状态与时间戳
        last_status = status
        last_status_poll_at = current_poll_at
        print(f"job_id:{job_id}, status({time.monotonic()}):{status}")
        if status in wait_status:
            # 仍在运行/排队：按轮询间隔（加抖动防惊群）等待后再查
            await asyncio.sleep(poll_interval + random.uniform(0.0, 0.5))
            continue
        # 作业离开等待状态：把累计运行时长写回数据
        data["running_time"] = total_running_time
        if status in finished_status:
            # 终态：返回 200 与完整结果
            return 200, data
        else:
            # 未知状态：同样返回数据，交由上层处理
            return 200, data

    # 超过等待时限：返回 504 超时
    return 504, {"error": "wait_timeout exceeded", "job_id": job_id, "running_time": total_running_time}


async def get_proxy_sandbox_result(
    client: httpx.AsyncClient,
    code_str: str,
    data_dir: str,
    *,
    resource_type: str = "all",
    job_timeout: int = 3600,
    wait_timeout: int = 3800,
    poll_interval: int = 5,
):
    """
    Proxy sandbox result for fast RL plumbing tests.
    It mirrors get_sandbox_result signature but does not execute code.
    """
    # 丢弃全部参数：本函数不执行任何代码，仅模拟沙箱返回（联调用）
    _ = (client, code_str, data_dir, resource_type, job_timeout, wait_timeout, poll_interval)
    # 以 40% 概率模拟代理瞬态失败（用于测试上层重试路径）
    if random.random() < 0.4:
        return 503, {
            "error": "proxy_connection_failed",
            "type": "ProxyConnectionError",
            "detail": "Simulated transient proxy failure (40% probability)",
            "running_time": 0.0,
        }

    # 随机生成 [0,1) 分数作为假奖励
    score = random.random()
    # 组装与真实沙箱一致的返回结构（status/running_time/result/score/run_log）
    payload = {
        "status": "success",
        "running_time": 0.0,
        "result": {
            "result": "proxy_success",
            "score": score,
            "run_log": f"[PROXY SANDBOX] generated random score={score:.6f}",
        },
    }
    # 返回 200 与模拟数据
    return 200, payload


def is_valid_python_script(script):
    """Check if a script is a valid Python script."""
    try:
        # 只做语法编译检查（不执行），验证是否为合法 Python 脚本
        compile(script, "<string>", "exec")
        return True
    except Exception as e:
        # 语法错误：记录日志并返回 False
        logger.error(f"Error in is_valid_python_script: {type(e).__name__}: {str(e)}")
        return False


def format_code(code) -> str:
    """Format Python code using Black."""
    try:
        # 用 Black 统一代码风格
        return black.format_str(code, mode=black.FileMode())
    except Exception as e:
        # 格式化失败（多为语法问题）：原样返回
        logger.error(f"Error in format_code: {type(e).__name__}: {str(e)}")
        return code


def _iter_code_candidates(code):
    # 候选一：原始代码
    yield code
    # 候选二：整段没有真实换行却含多个字面 "\n"（模型把转义符写进代码）时，替换为真实换行再试
    if "\\n" in code and "\n" not in code and code.count("\\n") >= 2:
        yield code.replace("\\n", "\n")


def extract_code(text):
    """Extract python code blocks from the text."""
    # logger.info(f"raw text: {text}")
    parsed_codes = []

    # When code is in a text or python block
    # 场景一：文本中存在 ```python ... ``` 围栏代码块
    matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        # 取围栏内的代码体
        code_block = match[1]
        parsed_codes.append(code_block)

    # When the entire text is code or backticks of the code block is missing
    # 场景二：没有围栏（或整段文本就是代码）时，按整文匹配
    if len(parsed_codes) == 0:
        matches = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            # 取匹配组中的代码体
            code_block = matches[0][2]
            parsed_codes.append(code_block)

    # validate the parsed codes
    # 校验每个候选块：取第一个能编译通过的候选并格式化后收下
    valid_code_blocks = []
    for code in parsed_codes:
        for candidate in _iter_code_candidates(code):
            if is_valid_python_script(candidate):
                valid_code_blocks.append(format_code(candidate))
                break
    # logger.info(f"valid_code_blocks: {valid_code_blocks}")
    # 拼接所有有效块并整体再格式化一次
    return format_code("\n\n".join(valid_code_blocks))


# online_percentile 模式的近期分数缓存：任务键 -> 最近分数队列
_RECENT = {}
# 近期分数队列的最大长度（滚动窗口）
_RECENT_MAXLEN = 20
# 排行榜根目录的缺省空元组（可被 LEADERBOARD_ROOTS 覆盖）
DEFAULT_LEADERBOARD_ROOTS: tuple[str, ...] = ()
# 单根目录版环境变量（兼容旧配置）
LEADERBOARD_ROOT = os.getenv("LEADERBOARD_ROOT", "").strip()
# 公共排行榜 DataFrame 缓存：((根目录元组, 任务名) -> DataFrame/None)
_PUBLIC_LEADERBOARD_CACHE = {}


def _stable_sigmoid(x: float) -> float:
    # 数值稳定的 sigmoid：非正数走等价变换，避免 exp 溢出
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _signed(score: float, meta: dict) -> float:
    # 读取指标方向（缺省“越大越好”）
    higher = meta.get("higher_is_better", True)
    s = float(score)
    # 统一到“越大越好”的符号空间：越小越好时取负
    return s if higher else -s


def signed_score(score: float, meta: dict) -> float:
    """Public helper for converting raw score to signed score."""
    # 对外暴露的分数符号化转换
    return _signed(score, meta)


def _finite(x) -> bool:
    # 判断数值是否存在且有限（None/NaN/Inf/不可转换均为无效）
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False


def _range_size(low, high) -> float | None:
    # 两端都有效才能计算区间宽度
    if not (_finite(low) and _finite(high)):
        return None
    # 区间宽度取两端点绝对差
    value = abs(float(high) - float(low))
    # 宽度非正（零宽区间）视为无效
    return value if value > 0 else None


def choose_validation_test_gap_denominator(
    metadata: dict,
    validation_score,
    test_score,
    *,
    theoretical_small_range_max: float = 2.0,
    big_range_threshold: float = 100.0,
    no_range_is_big: bool = True,
) -> tuple[float | None, str]:
    """Choose the score scale used to normalize validation/test gap."""
    # 理论区间宽度（theoretical_min/max）
    theoretical = _range_size(metadata.get("theoretical_min"), metadata.get("theoretical_max"))
    # 榜单区间宽度（leaderboard_min/max）
    leaderboard = _range_size(metadata.get("leaderboard_min"), metadata.get("leaderboard_max"))

    # 用验证/测试分中较大的绝对值估计实际分数尺度
    score_range_candidates = [abs(float(v)) for v in (validation_score, test_score) if _finite(v)]
    score_range = max(score_range_candidates) if score_range_candidates else 0.0

    # 理论区间足够小（如准确率类 0~1）：直接用理论区间做归一分母
    if theoretical is not None and theoretical <= theoretical_small_range_max:
        return theoretical, "theoretical_range_le_small_threshold"

    # 候选区间：优先榜单，缺省回落理论
    candidate = leaderboard if leaderboard is not None else theoretical
    if candidate is None:
        # 完全没有区间信息时：
        if no_range_is_big and score_range > 0:
            # 视同大区间场景，用分数尺度兜底
            return score_range, "max_abs_valid_test_for_missing_range"
        # 无分母可用
        return None, "missing_range"

    if candidate > big_range_threshold:
        # 区间过大（如 MAE 类无界指标）：退回用分数尺度
        if score_range > 0:
            return score_range, "max_abs_valid_test_for_big_range"
        # 分数尺度也不可用
        return None, "nonpositive_score_range_for_big_range"
    if leaderboard is not None:
        # 正常情况：用榜单区间
        return leaderboard, "leaderboard_range"
    # 只有理论区间：用理论区间
    return theoretical, "theoretical_range_gt_small_threshold"


def validation_test_gap_info(validation_score, test_score, metadata: dict) -> dict[str, float | str | None]:
    """Return absolute and relative validation/test gap information."""
    # 初始化差距信息结构（验证分/测试分/绝对差距/相对差距/分母及来源）
    info: dict[str, float | str | None] = {
        "validation_score": float(validation_score) if _finite(validation_score) else None,
        "test_score": float(test_score) if _finite(test_score) else None,
        "abs_gap": None,
        "relative_gap": None,
        "gap_denominator": None,
        "gap_denominator_source": "unavailable",
    }
    if not (_finite(validation_score) and _finite(test_score)):
        # 任一分数缺失：记录原因并直接返回
        info["gap_denominator_source"] = "missing_validation_or_test_score"
        return info

    # 绝对差距 = |测试分 - 验证分|
    abs_gap = abs(float(test_score) - float(validation_score))
    # 选择归一分母及其来源
    denominator, source = choose_validation_test_gap_denominator(metadata, validation_score, test_score)
    # 写入绝对差距与分母信息
    info["abs_gap"] = abs_gap
    info["gap_denominator"] = denominator
    info["gap_denominator_source"] = source
    if denominator is not None and denominator > 0:
        # 相对差距 = 绝对差距 / 分母
        info["relative_gap"] = abs_gap / denominator
    # 返回差距信息
    return info


def apply_validation_test_gap_penalty(
    reward,
    relative_gap,
    penalty_coef: float | None = None,
    gap_tolerance: float | None = None,
    high_penalty_coef: float | None = None,
    enabled: bool | None = None,
    piecewise_enabled: bool | None = None,
) -> tuple[float, float, float]:
    """
    Penalize positive rewards by validation/test relative gap.

    Returns:
        (penalized_reward, penalty, multiplier)
    """
    try:
        # 奖励转 float
        reward_f = float(reward)
    except (TypeError, ValueError):
        # 转换失败按 0 奖励处理
        return 0.0, 0.0, 1.0
    # 惩罚开关：未显式传入时用全局配置
    penalty_enabled = VALIDATION_TEST_GAP_PENALTY_ENABLED if enabled is None else bool(enabled)
    # 仅对正的有限奖励且相对差距有效时惩罚；其余原样返回（负奖励不惩罚）
    if (not penalty_enabled) or not math.isfinite(reward_f) or reward_f <= 0.0 or not _finite(relative_gap):
        return reward_f, 0.0, 1.0

    # 基础系数解析与兜底
    coef = VALIDATION_TEST_GAP_PENALTY_COEF if penalty_coef is None else float(penalty_coef)
    if not math.isfinite(coef):
        coef = VALIDATION_TEST_GAP_PENALTY_COEF
    coef = max(coef, 0.0)

    # 高段系数解析与兜底
    high_coef = VALIDATION_TEST_GAP_PENALTY_HIGH_COEF if high_penalty_coef is None else float(high_penalty_coef)
    if not math.isfinite(high_coef):
        high_coef = VALIDATION_TEST_GAP_PENALTY_HIGH_COEF
    high_coef = max(high_coef, 0.0)

    # 容差解析与兜底（必须为正）
    tolerance = VALIDATION_TEST_GAP_PENALTY_TOLERANCE if gap_tolerance is None else float(gap_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        tolerance = VALIDATION_TEST_GAP_PENALTY_TOLERANCE

    # 归一化差距 = 相对差距 / 容差（负差距按 0 处理）
    normalized_gap = max(float(relative_gap), 0.0) / tolerance
    # 分段开关：未显式传入时用全局配置
    use_piecewise = VALIDATION_TEST_GAP_PENALTY_PIECEWISE_ENABLED if piecewise_enabled is None else bool(piecewise_enabled)
    if use_piecewise:
        # 分段惩罚：容差内用基础系数，超出部分叠加高段系数
        penalty = coef * min(normalized_gap, 1.0) + high_coef * max(normalized_gap - 1.0, 0.0)
    else:
        # 线性惩罚：系数 * 归一化差距
        penalty = coef * normalized_gap
    # 乘数下限 0：奖励按 (1 - 惩罚) 缩放
    multiplier = max(0.0, 1.0 - penalty)
    # 返回（惩罚后奖励, 惩罚量, 乘数）
    return reward_f * multiplier, penalty, multiplier


def _task_name_from_metadata(meta: dict) -> str | None:
    # 优先取 task_name / task 字段
    task_name = meta.get("task_name") or meta.get("task")
    if task_name:
        return str(task_name)
    # 兜底：从数据目录名推断任务名
    data_dir = meta.get("data_dir")
    if data_dir:
        return Path(str(data_dir)).name
    # 都没有则返回 None
    return None


def _leaderboard_roots() -> list[Path]:
    if os.getenv("LEADERBOARD_ROOTS"):
        # 多根目录版：按平台路径分隔符拆分环境变量
        raw_roots = os.getenv("LEADERBOARD_ROOTS", "")
        roots = [item for item in raw_roots.split(os.pathsep) if item]
    else:
        # 单根目录版（兼容旧配置）
        roots = [LEADERBOARD_ROOT] if LEADERBOARD_ROOT else []

    # 去重并保持原有顺序
    output = []
    seen = set()
    for root in roots:
        path = Path(root)
        key = str(path)
        if key not in seen:
            output.append(path)
            seen.add(key)
    # 返回去重后的根目录列表
    return output


def _public_leaderboard_candidates(root: Path, task_name: str) -> tuple[Path, Path]:
    # 榜单文件的两种候选布局：<root>/<task>/info/public_leaderboard.csv 与 <root>/<task>.csv
    return (
        root / task_name / "info" / "public_leaderboard.csv",
        root / f"{task_name}.csv",
    )


def _normalize_leaderboard_columns(leaderboard: pd.DataFrame) -> pd.DataFrame:
    # 拷贝一份，避免修改调用方的 DataFrame
    leaderboard = leaderboard.copy()
    # 列名清洗：去掉 BOM 字符与首尾空白
    leaderboard.columns = [str(column).lstrip("\ufeff").strip() for column in leaderboard.columns]
    # 建立“小写列名 -> 原列名”映射
    column_by_lower = {str(column).lower(): column for column in leaderboard.columns}
    rename = {}
    if "score" in column_by_lower:
        # score 列统一命名为 "score"
        rename[column_by_lower["score"]] = "score"
    if "rank" in column_by_lower:
        # rank 列统一命名为 "rank"
        rename[column_by_lower["rank"]] = "rank"
    # 返回重命名后的榜单
    return leaderboard.rename(columns=rename)


def _load_public_leaderboard(meta: dict) -> pd.DataFrame | None:
    # 从元数据解析任务名
    task_name = _task_name_from_metadata(meta)
    if not task_name:
        # 无任务名则无法定位榜单
        return None

    # 根目录元组（转字符串作为缓存键的一部分）
    roots = tuple(str(root) for root in _leaderboard_roots())
    cache_key = (roots, task_name)
    # 命中缓存直接返回（含“确认不存在”的负缓存）
    if cache_key in _PUBLIC_LEADERBOARD_CACHE:
        return _PUBLIC_LEADERBOARD_CACHE[cache_key]

    # 在各根目录的候选路径中查找榜单文件
    path = None
    for root in _leaderboard_roots():
        for candidate in _public_leaderboard_candidates(root, task_name):
            if candidate.exists():
                path = candidate
                break
        if path is not None:
            break
    if path is None:
        # 找不到榜单：缓存 None 并返回
        _PUBLIC_LEADERBOARD_CACHE[cache_key] = None
        return None
    try:
        # 读取榜单 CSV
        leaderboard = pd.read_csv(path)
    except Exception as exc:
        # 读取失败：记录警告并负缓存
        logger.warning("Failed to read public leaderboard %s: %s", path, exc)
        _PUBLIC_LEADERBOARD_CACHE[cache_key] = None
        return None

    # 规范化列名
    leaderboard = _normalize_leaderboard_columns(leaderboard)
    if "score" not in leaderboard.columns:
        # 无 score 列则榜单不可用
        logger.warning("Public leaderboard %s has no score column", path)
        _PUBLIC_LEADERBOARD_CACHE[cache_key] = None
        return None

    # score 列强转数值，无法解析的变 NaN
    leaderboard["score"] = pd.to_numeric(leaderboard["score"], errors="coerce")
    # 剔除 NaN 分数行并重置索引
    leaderboard = leaderboard[leaderboard["score"].notna()].reset_index(drop=True)
    if leaderboard.empty:
        # 剔除后为空：榜单不可用
        _PUBLIC_LEADERBOARD_CACHE[cache_key] = None
        return None

    if "rank" in leaderboard.columns:
        # 有 rank 列则按 rank 稳定排序（第 1 名在前）
        leaderboard = leaderboard.sort_values("rank", kind="stable").reset_index(drop=True)
    # 写入缓存并返回
    _PUBLIC_LEADERBOARD_CACHE[cache_key] = leaderboard
    return leaderboard


def _leaderboard_lower_is_better(leaderboard: pd.DataFrame) -> bool:
    # 依排序后的首尾分数判断方向：首行更小即“越小越好”
    scores = leaderboard["score"]
    return bool(float(scores.iloc[0]) < float(scores.iloc[-1]))


def _leaderboard_medal_for_score(score: float, leaderboard: pd.DataFrame) -> str:
    # 分数无效：无奖牌
    if not _finite(score):
        return "None"

    # 榜单分数序列（按名次排好，重置索引）
    scores = leaderboard["score"].reset_index(drop=True)
    # 参赛队伍数
    num_teams = len(scores)

    def score_at_position(position: int) -> float:
        # 取第 position 名的分数（1-based，越界夹紧到 [1, 队伍数]）
        idx = min(max(position, 1), num_teams) - 1
        return float(scores.iloc[idx])

    if num_teams < 100:
        # 小榜单：按比例切奖牌阈值（金前 10%、银前 20%、铜前 40%）
        gold_threshold = score_at_position(max(1, int(num_teams * 0.1)))
        silver_threshold = score_at_position(max(1, int(num_teams * 0.2)))
        bronze_threshold = score_at_position(max(1, int(num_teams * 0.4)))
    elif num_teams < 250:
        # 中小榜单：金牌固定取前 10 名分数线，银/铜按比例
        gold_threshold = score_at_position(10)
        silver_threshold = score_at_position(max(1, int(num_teams * 0.2)))
        bronze_threshold = score_at_position(max(1, int(num_teams * 0.4)))
    elif num_teams < 1000:
        # 大榜单：金前 10+0.2% 名，银前 50 名，铜前 100 名
        gold_threshold = score_at_position(10 + int(num_teams * 0.002))
        silver_threshold = score_at_position(50)
        bronze_threshold = score_at_position(100)
    else:
        # 超大榜单：金前 10+0.2%，银前 5%，铜前 10%
        gold_threshold = score_at_position(10 + int(num_teams * 0.002))
        silver_threshold = score_at_position(max(1, int(num_teams * 0.05)))
        bronze_threshold = score_at_position(max(1, int(num_teams * 0.1)))

    # 按指标方向与阈值判定奖牌等级
    score = float(score)
    if _leaderboard_lower_is_better(leaderboard):
        # 越小越好：分数 ≤ 对应阈值即达标
        if score <= gold_threshold:
            return "Gold"
        if score <= silver_threshold:
            return "Silver"
        if score <= bronze_threshold:
            return "Bronze"
    else:
        # 越大越好：分数 ≥ 对应阈值即达标
        if score >= gold_threshold:
            return "Gold"
        if score >= silver_threshold:
            return "Silver"
        if score >= bronze_threshold:
            return "Bronze"
    # 未达任何奖牌阈值
    return "None"


def leaderboard_medal_binary_reward(score, metadata: dict) -> float:
    # 加载任务公共榜单
    leaderboard = _load_public_leaderboard(metadata)
    if leaderboard is None or not _finite(score):
        # 榜单缺失或分数无效：奖励 0
        return 0.0
    # 拿到任意奖牌记 1 分，否则 0 分（二值奖牌奖励）
    return 1.0 if _leaderboard_medal_for_score(float(score), leaderboard) != "None" else 0.0


def leaderboard_rank_reward(score, metadata: dict) -> float:
    # 加载任务公共榜单
    leaderboard = _load_public_leaderboard(metadata)
    if leaderboard is None or not _finite(score):
        # 榜单缺失或分数无效：奖励 0
        return 0.0

    # 榜单全部分数
    scores = leaderboard["score"].astype(float).tolist()
    # 指标方向
    lower_better = _leaderboard_lower_is_better(leaderboard)
    # 统一转换到“越大越好”空间，便于统一比较
    signed_scores = [(-value if lower_better else value) for value in scores]
    signed_score = -float(score) if lower_better else float(score)

    # 队伍数
    num_teams = len(signed_scores)
    if num_teams <= 1:
        # 榜单只有自己（或为空）：默认满分
        return 1.0

    # 统计严格优于该分数的队伍数
    better_count = sum(value > signed_score for value in signed_scores)
    # 排名奖励：第 1 名 1 分、最后 1 名 0 分，线性插值
    reward = 1.0 - (better_count / (num_teams - 1))
    # 夹紧到 [0,1] 后返回
    return max(0.0, min(1.0, float(reward)))


def _static_bound_limits_signed(meta: dict):
    # 读取指标方向
    higher = meta.get("higher_is_better", True)

    def to_signed(v):
        # 无效值返回 None
        if not _finite(v):
            return None
        v = float(v)
        # 统一到“越大越好”的符号空间
        return v if higher else -v

    # 理论区间与榜单区间的四端点（符号空间）
    tmin = to_signed(meta.get("theoretical_min"))
    tmax = to_signed(meta.get("theoretical_max"))
    lmin = to_signed(meta.get("leaderboard_min"))
    lmax = to_signed(meta.get("leaderboard_max"))

    if higher:
        # 越大越好：最好限取两个上界的较大者，最差限取两个下界的较小者
        best_cands = [tmax, lmax]
        worst_cands = [tmin, lmin]
    else:
        # 越小越好（已取负）：最好限取两个下界（负得少者），最差限取两个上界
        best_cands = [tmin, lmin]
        worst_cands = [tmax, lmax]

    # 过滤掉缺失的候选端点
    best_cands = [x for x in best_cands if x is not None]
    worst_cands = [x for x in worst_cands if x is not None]

    # 最好/最差限；候选全空则返回 None
    best = max(best_cands) if best_cands else None
    worst = min(worst_cands) if worst_cands else None
    # 返回（最好限, 最差限）
    return best, worst


def get_bound_limits_signed(meta: dict):
    """Return one-sided signed-space metadata limits: (best_limit, worst_limit)."""
    # 对外暴露的静态边界查询
    return _static_bound_limits_signed(meta)


def _apply_max_range(best, worst, fallback_best=None, fallback_worst=None):
    # 任一端无效：不做区间宽度限制
    if not (_finite(best) and _finite(worst)):
        return best, worst
    best = float(best)
    worst = float(worst)

    # 最大允许区间宽度
    MAX_RANGE = 1e6
    if best - worst > MAX_RANGE:
        if _finite(fallback_best) and _finite(fallback_worst) and (float(fallback_best) - float(fallback_worst)) <= MAX_RANGE:
            # 回退区间宽度合法：改用回退区间
            best, worst = float(fallback_best), float(fallback_worst)
        else:
            # 否则截断区间宽度到上限（最差端向最好端收拢）
            worst = best - MAX_RANGE

    # 返回修正后的区间
    return best, worst


def _bounds_signed(meta: dict):
    # 静态边界（元数据声明）
    static_best, static_worst = _static_bound_limits_signed(meta)

    # 动态边界（运行中统计出的更紧边界）
    dynamic_best = meta.get("dynamic_bound_best_signed")
    dynamic_worst = meta.get("dynamic_bound_worst_signed")
    has_dynamic = _finite(dynamic_best) and _finite(dynamic_worst)
    if has_dynamic:
        # 动态边界转 float
        dynamic_best = float(dynamic_best)
        dynamic_worst = float(dynamic_worst)

        if _finite(static_best):
            # 动态最好界不得宽于静态最好界
            dynamic_best = min(dynamic_best, float(static_best))
        if _finite(static_worst):
            # 动态最差界不得宽于静态最差界
            dynamic_worst = max(dynamic_worst, float(static_worst))

        if dynamic_best > dynamic_worst:
            # 动态区间有效：应用最大宽度限制（回退到静态区间）
            return _apply_max_range(dynamic_best, dynamic_worst, static_best, static_worst)

    if not (_finite(static_best) and _finite(static_worst)):
        # 静态区间也无效：无边界可用
        return None, None

    # 使用静态区间
    return _apply_max_range(static_best, static_worst)


def _preferred_static_bounds_signed(
    meta: dict,
    priority: str = "leaderboard",
    fallback_best: float | None = None,
    fallback_worst: float | None = None,
):
    # 指标方向
    higher = meta.get("higher_is_better", True)

    def to_signed(v):
        # 无效值返回 None
        if not _finite(v):
            return None
        v = float(v)
        # 统一到“越大越好”的符号空间
        return v if higher else -v

    # 理论区间与榜单区间的四端点（符号空间）
    tmin = to_signed(meta.get("theoretical_min"))
    tmax = to_signed(meta.get("theoretical_max"))
    lmin = to_signed(meta.get("leaderboard_min"))
    lmax = to_signed(meta.get("leaderboard_max"))

    # 优先级字符串归一化
    priority = (priority or "leaderboard").strip().lower()

    if priority == "theoretical":
        # 理论优先：理论区间为主，榜单区间为备
        primary_best, secondary_best = (tmax, lmax) if higher else (tmin, lmin)
        primary_worst, secondary_worst = (tmin, lmin) if higher else (tmax, lmax)
    else:
        # 榜单优先（默认）：榜单区间为主，理论区间为备
        primary_best, secondary_best = (lmax, tmax) if higher else (lmin, tmin)
        primary_worst, secondary_worst = (lmin, tmin) if higher else (lmax, tmax)

    # 主候选缺失时改用备选端点
    best = primary_best if primary_best is not None else secondary_best
    worst = primary_worst if primary_worst is not None else secondary_worst

    if not (_finite(best) and _finite(worst)):
        # 双端仍不完整：尝试调用方提供的兜底区间
        if _finite(fallback_best) and _finite(fallback_worst):
            return _apply_max_range(float(fallback_best), float(fallback_worst))
        # 无可用区间
        return None, None

    # 应用最大宽度限制后返回
    return _apply_max_range(best, worst)


def has_static_bounds_with_priority(
    metadata: dict,
    priority: str = "leaderboard",
    fallback_best: float | None = None,
    fallback_worst: float | None = None,
) -> bool:
    # 按指定优先级解析边界
    best, worst = _preferred_static_bounds_signed(
        metadata,
        priority=priority,
        fallback_best=fallback_best,
        fallback_worst=fallback_worst,
    )
    # 双端有效即认为存在可用静态边界
    return _finite(best) and _finite(worst)


def score2reward_with_static_priority(
    score,
    metadata,
    mode="logistic",
    priority="leaderboard",
    fallback_best: float | None = None,
    fallback_worst: float | None = None,
    power_alpha: float | None = None,
):
    """
    Map score to reward using static bounds with an explicit source priority.

    priority="leaderboard" means:
    - prefer leaderboard_min/max
    - fallback to theoretical_min/max
    """
    # 非有限分数：奖励 0
    if not _finite(score):
        return 0.0

    # 二值奖牌模式：直接委托
    if mode == "leaderboard_medal_binary":
        return leaderboard_medal_binary_reward(score, metadata)

    # 排名模式：直接委托
    if mode == "leaderboard_rank":
        return leaderboard_rank_reward(score, metadata)

    # 分数转 float 并符号化（统一到越大越好空间）
    score = float(score)
    signed = _signed(score, metadata)

    if mode == "power_clip":
        # 幂裁剪模式：按优先级解析边界
        best, worst = _preferred_static_bounds_signed(
            metadata,
            priority=priority,
            fallback_best=fallback_best,
            fallback_worst=fallback_worst,
        )
        if best is None:
            # 无可用边界：奖励 0
            return 0.0

        # 区间宽度（下限 1e-9 防除零）
        rng = max(best - worst, 1e-9)
        # 归一化到 [0,1]
        s_prime = (signed - worst) / rng
        s_prime = 0.0 if s_prime < 0 else (1.0 if s_prime > 1 else s_prime)
        # 幂次 alpha（未指定时用全局配置）
        alpha = POWER_CLIP_ALPHA if power_alpha is None else max(float(power_alpha), 0.0)
        # 幂次锐化：拉大好差分数的奖励差距
        return s_prime**alpha

    if mode == "margin_tanh":
        # tanh 边距模式：按优先级解析边界
        best, worst = _preferred_static_bounds_signed(
            metadata,
            priority=priority,
            fallback_best=fallback_best,
            fallback_worst=fallback_worst,
        )
        if best is None:
            # 无可用边界：奖励 0
            return 0.0

        # 区间中心与半宽（下限 1e-9 防除零）
        center = 0.5 * (best + worst)
        scale = max(0.5 * (best - worst), 1e-9)
        # 归一化边距
        m = (signed - center) / scale
        # tanh 映射到 (0,1)，再平方锐化
        r = 0.5 + 0.5 * math.tanh(m / 0.50)
        return r**2

    # 默认 logistic 模式：读取方向与区间端点
    higher_is_better = metadata.get("higher_is_better")
    theoretical_min = metadata.get("theoretical_min")
    theoretical_max = metadata.get("theoretical_max")
    leaderboard_min = metadata.get("leaderboard_min")
    leaderboard_max = metadata.get("leaderboard_max")

    if priority == "theoretical":
        # 理论优先：理论端点在前，榜单端点为备
        first_min, second_min = theoretical_min, leaderboard_min
        first_max, second_max = theoretical_max, leaderboard_max
    else:
        # 榜单优先：榜单端点在前，理论端点为备
        first_min, second_min = leaderboard_min, theoretical_min
        first_max, second_max = leaderboard_max, theoretical_max

    if higher_is_better:
        # 越大越好：最好分取上界，最差分取下界
        score_range_best_score = first_max if first_max is not None else second_max
        score_range_worst_score = first_min if first_min is not None else second_min
    else:
        # 越小越好：最好分取下界，最差分取上界
        score_range_best_score = first_min if first_min is not None else second_min
        score_range_worst_score = first_max if first_max is not None else second_max

    if score_range_best_score is None or score_range_worst_score is None:
        # 端点不完整：奖励 0
        return 0.0

    # 区间宽度
    score_range = score_range_best_score - score_range_worst_score
    if score_range == 0 or math.isnan(score_range):
        # 宽度为 0 或非法：奖励 0
        return 0.0

    # 归一化：x=0 对应最差分，x=2 对应最好分
    x = 2.0 * (score - score_range_worst_score) / score_range
    if x >= 0:
        # 数值稳定的 sigmoid
        reward = 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        reward = exp_x / (1.0 + exp_x)

    # 返回 sigmoid 奖励
    return reward


def score2reward(score, metadata, mode="logistic", power_alpha: float | None = None):
    """
    Map a metric score to a reward.
    """
    # 读取方向与理论/榜单四端点
    higher_is_better = metadata.get("higher_is_better")
    theoretical_min = metadata.get("theoretical_min")
    theoretical_max = metadata.get("theoretical_max")
    leaderboard_min = metadata.get("leaderboard_min")
    leaderboard_max = metadata.get("leaderboard_max")

    def safe_max(*vals):
        # 忽略 None 后取最大；全空返回 None
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None

    def safe_min(*vals):
        # 忽略 None 后取最小；全空返回 None
        vals = [v for v in vals if v is not None]
        return min(vals) if vals else None

    # assert mode=="linear_sign" , f"Unsupported mode: {mode}"

    if mode == "linear_sign":
        # 线性符号模式：分数直接作为（带方向的）奖励
        if abs(float(score)) < 1e-7:
            # 分数近乎为 0：视为未产生有效结果，给强惩罚
            return -100.0

        if higher_is_better is False:
            # 越小越好：取负统一方向
            return -float(score)
        else:
            # 越大越好：原样返回
            return float(score)

    if mode == "leaderboard_medal_binary":
        # 二值奖牌模式：委托
        return leaderboard_medal_binary_reward(score, metadata)

    if mode == "leaderboard_rank":
        # 排名模式：委托
        return leaderboard_rank_reward(score, metadata)

    # 符号化分数（统一到越大越好空间）
    s = _signed(score, metadata)

    # A: Power Clip on MinMax
    if mode == "power_sigmoid":
        # 旧模式名已废弃：指引用户改用 power_clip
        raise NotImplementedError('mode="power_sigmoid" is not implemented; use mode="power_clip" instead.')

    if mode == "power_clip":
        # 幂裁剪：解析（含动态收紧的）边界
        best, worst = _bounds_signed(metadata)
        if best is None:
            # 无边界：退回普通 sigmoid
            return _stable_sigmoid(s)

        # 区间宽度（下限 1e-9 防除零）
        rng = max(best - worst, 1e-9)
        # 归一化到 [0,1] 并夹紧
        s_prime = (s - worst) / rng
        s_prime = 0.0 if s_prime < 0 else (1.0 if s_prime > 1 else s_prime)
        # 幂次 alpha（未指定时用全局配置）
        alpha = POWER_CLIP_ALPHA if power_alpha is None else max(float(power_alpha), 0.0)
        # 幂次锐化后返回
        return s_prime**alpha

    # B: Tanh margin shaping
    if mode == "margin_tanh":
        # tanh 边距：解析边界
        best, worst = _bounds_signed(metadata)
        if best is None:
            return 0.5 + 0.5 * math.tanh(s)  # Fallback when no bounds are available.

        # 区间中心与半宽
        center = 0.5 * (best + worst)
        scale = max(0.5 * (best - worst), 1e-9)

        # 归一化边距
        m = (s - center) / scale

        # 温度参数
        T = 0.50
        y = math.tanh(m / max(T, 1e-9))  # (-1,1)
        r = 0.5 + 0.5 * y  # (0,1)

        # gamma 平方锐化：拉大好差分数的奖励差距
        gamma = 2.0
        return r**gamma

    # C: Online percentile
    if mode == "online_percentile":
        # 以任务为键取近期分数队列（在线分位基准）
        key = str(metadata.get("data_dir", "unknown_task"))
        dq = _RECENT.get(key)
        if dq is None:
            # 首次出现该任务：建固定长度滚动队列
            from collections import deque

            dq = deque(maxlen=_RECENT_MAXLEN)
            _RECENT[key] = dq

        if len(dq) < 10:
            # 历史样本不足 10：先累积，用 sigmoid 给过渡奖励
            dq.append(s)
            return _stable_sigmoid(s)

        # 排序历史分数
        sorted_vals = sorted(dq)
        # rank = count(vals <= s)
        import bisect

        # 用二分查找计算当前分数的历史分位
        rank = bisect.bisect_right(sorted_vals, s)
        p = rank / len(sorted_vals)  # [0,1]

        # 当前分数入队（滚动窗口更新基准）
        dq.append(s)

        # 分位平方锐化后返回
        gamma = 2.0
        return p**gamma

    # initial sigmoid reward
    # 默认模式：基于理论/榜单区间的 logistic 归一化
    if higher_is_better:
        # best score
        score_range_best_score = theoretical_max if theoretical_max is not None else leaderboard_max

        # worst score
        score_range_worst_score = safe_max(leaderboard_min, theoretical_min, -100)

    else:
        # best score
        score_range_best_score = theoretical_min if theoretical_min is not None else leaderboard_min

        # worst score
        score_range_worst_score = safe_min(leaderboard_max, theoretical_max, 100)

    # Return a neutral reward when the score interval is still unknown.
    if score_range_best_score is None or score_range_worst_score is None:
        # 区间未知：给中性奖励 0.5
        return 0.5

    # 区间宽度
    score_range = score_range_best_score - score_range_worst_score
    if score_range == 0 or math.isnan(score_range):
        # 宽度非法：中性奖励
        return 0.5

    # 归一化：x=0 对应最差分，x=2 对应最好分
    x = 2.0 * (score - score_range_worst_score) / score_range
    if x >= 0:
        # 数值稳定的 sigmoid
        reward = 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        reward = exp_x / (1.0 + exp_x)

    # 返回 sigmoid 奖励
    return reward


def test_score2reward(base_dir: str):
    """
    Test score2reward function for all tasks in Selected_Dojo.

    For each task, displays score and reward at:
    - Rank 1 (best)
    - 20th percentile
    - 40th percentile
    - 60th percentile
    - 80th percentile
    - Last rank (worst)

    Args:
        base_dir: Base directory containing task folders
    """
    # 基目录转 Path
    base_path = Path(base_dir)
    if not base_path.exists():
        # 目录不存在：报错
        raise ValueError(f"Base directory does not exist: {base_dir}")

    # Find all task directories
    # 收集有效任务目录（须同时含 task_metadata.json 与 public_leaderboard.csv）
    task_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            metadata_path = item / "info" / "task_metadata.json"
            leaderboard_path = item / "info" / "public_leaderboard.csv"
            if metadata_path.exists() and leaderboard_path.exists():
                task_dirs.append(item)

    if not task_dirs:
        # 没找到有效任务：提示并返回
        print(f"No valid tasks found in {base_dir}")
        return

    print(f"Found {len(task_dirs)} tasks to test\n")
    print("=" * 100)

    for task_dir in sorted(task_dirs):
        # 任务名与元数据/榜单文件路径
        task_name = task_dir.name
        metadata_path = task_dir / "info" / "task_metadata.json"
        leaderboard_path = task_dir / "info" / "public_leaderboard.csv"

        # Load metadata
        # 读取任务元数据
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            # 读取失败：跳过该任务
            print(f"Error loading metadata for {task_name}: {e}")
            continue

        # Load leaderboard
        # 读取榜单分数
        try:
            df = pd.read_csv(leaderboard_path)
            if "score" not in df.columns:
                # 无 score 列：跳过
                print(f"Warning: {task_name} leaderboard does not have 'score' column")
                continue
            scores = df["score"].dropna().astype(float).tolist()
            if not scores:
                # 无有效分数：跳过
                print(f"Warning: {task_name} leaderboard has no valid scores")
                continue
        except Exception as e:
            # 读取失败：跳过
            print(f"Error loading leaderboard for {task_name}: {e}")
            continue

        # Sort scores based on higher_is_better
        # 按方向排序：最优在前
        higher_is_better = metadata.get("higher_is_better", True)
        if higher_is_better:
            scores_sorted = sorted(scores, reverse=True)  # Best to worst
        else:
            scores_sorted = sorted(scores)  # Best (lowest) to worst (highest)

        total_entries = len(scores_sorted)
        if total_entries == 0:
            continue

        # Calculate indices for percentiles
        # Use floor to get the index, ensuring we get the entry at or below the percentile
        def percentile_index(pct: float) -> int:
            """Calculate index for a given percentile (0.0 to 1.0)."""
            # 分位转下标：向下取整并夹紧到合法范围
            idx = int((total_entries - 1) * pct)
            return max(0, min(idx, total_entries - 1))

        # 采样位置：第 1 名、20%、40%、60%、80% 分位与最后一名
        indices = [
            0,  # Rank 1 (best)
            percentile_index(0.2),  # 20%
            percentile_index(0.4),  # 40%
            percentile_index(0.6),  # 60%
            percentile_index(0.8),  # 80%
            total_entries - 1,  # Last (worst)
        ]

        # Get scores at these positions
        # 取这些位置的分数与（1-based）名次
        selected_scores = [scores_sorted[i] for i in indices]
        selected_ranks = [i + 1 for i in indices]  # Rank is 1-indexed

        # Calculate rewards
        # 逐个分数计算映射后的奖励
        rewards = []
        for score in selected_scores:
            try:
                reward = score2reward(score, metadata)
                rewards.append(reward)
            except Exception as e:
                # 单点失败记 NaN，不中断整个测试
                print(f"  Warning: Failed to calculate reward for score {score:.6f}: {e}")
                rewards.append(float("nan"))

        # Display results
        # 打印任务概览与分位表（以下连续打印为纯输出行）
        print(f"\nTask: {task_name}")
        print(f"  UUID: {metadata.get('uuid', 'N/A')}")
        print(f"  Higher is better: {higher_is_better}")
        print(f"  Theoretical min: {metadata.get('theoretical_min', 'N/A')}")
        print(f"  Theoretical max: {metadata.get('theoretical_max', 'N/A')}")
        print(f"  Total entries: {total_entries}")
        print(f"  Score range: [{min(scores):.6f}, {max(scores):.6f}]")
        print(f"\n  {'Position':<12} {'Rank':<8} {'Score':<15} {'Reward':<10}")
        print(f"  {'-' * 12} {'-' * 8} {'-' * 15} {'-' * 10}")

        labels = ["1st (best)", "20%", "40%", "60%", "80%", "Last (worst)"]
        # 逐行打印分位表；NaN 显示为 N/A
        for label, rank, score, reward in zip(labels, selected_ranks, selected_scores, rewards):
            if math.isnan(reward):
                # NaN：显示占位符
                reward_str = "N/A"
            else:
                # 正常格式化到 6 位小数
                reward_str = f"{reward:.6f}"
            print(f"  {label:<12} {rank:<8} {score:<15.6f} {reward_str:<10}")

        print("=" * 100)


def test_hack_check(code_list: list[str], labels: list[str] = None):
    """
    Test the hack_check function on a list of code samples.

    Args:
        code_list: List of code strings to test
        labels: Optional list of descriptive labels for each code sample
    """
    # 未提供标签：自动生成默认标签
    if labels is None:
        labels = [f"Code {i+1}" for i in range(len(code_list))]

    if len(labels) != len(code_list):
        # 标签数与代码数不一致：报错
        raise ValueError(f"Length of labels ({len(labels)}) must match length of code_list ({len(code_list)})")

    print("=" * 100)
    print("Testing hack_check function")
    print("=" * 100)

    # 各代码样本的测试结果
    results = []
    for i, (code, label) in enumerate(zip(code_list, labels)):
        # 打印测试进度
        print(f"\n[{i+1}/{len(code_list)}] Testing: {label}")
        print("-" * 100)

        # 调用同步 GPT 判卷
        is_valid, category, reason = hack_check(code)

        # 汇总单条结果（含 200 字符代码预览）
        result = {
            "label": label,
            "is_valid": is_valid,
            "category": category,
            "reason": reason,
            "code_preview": code[:200] + "..." if len(code) > 200 else code,
        }
        results.append(result)

        # 打印判定详情
        status = "✅ VALID" if is_valid else "❌ INVALID (HACK DETECTED)"
        print(f"Status: {status}")
        print(f"Category: {category}")
        print(f"Reason: {reason}")
        print(f"Code preview: {result['code_preview']}")

    # Summary
    # 汇总统计有效/无效比例
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    valid_count = sum(1 for r in results if r["is_valid"])
    invalid_count = len(results) - valid_count
    print(f"Total: {len(results)}")
    print(f"Valid: {valid_count} ({valid_count/len(results)*100:.1f}%)")
    print(f"Invalid: {invalid_count} ({invalid_count/len(results)*100:.1f}%)")

    print("\nDetailed Results:")
    # 逐条打印判定理由
    for i, r in enumerate(results):
        status = "✅" if r["is_valid"] else "❌"
        print(f"{status} {r['label']}: {r['reason']}")

    # 返回全部测试结果
    return results


if __name__ == "__main__":
    # 命令行入口：对指定目录下的所有任务做 score->reward 映射自检
    parser = argparse.ArgumentParser(description="Validate score-to-reward conversion on a leaderboard directory.")
    parser.add_argument("base_dir", help="Directory containing the task folders to inspect.")
    args = parser.parse_args()
    # 执行测试
    test_score2reward(args.base_dir)
