# 奖励函数与沙箱工具模块：负责向沙箱提交/轮询代码执行任务、从模型回复中抽取并校验代码、
# 将指标分数映射为奖励（score→reward，对应 RL 中的奖励塑形 reward shaping）、
# 清洗运行日志并格式化给模型看的反馈文本。
# 导入 time：提供单调时钟计时（轮询截止时间与运行时长统计）
import time
# 导入 os：读取环境变量（沙箱 API Key 等）
import os
# 导入 httpx：异步 HTTP 客户端
import httpx
# 导入 asyncio：异步睡眠与并发原语
import asyncio
# 导入 re：从模型回复中正则抽取代码块
import re
# 导入 logging：模块级日志
import logging
# 导入 black：统一格式化抽取出的代码
import black
# 导入 math：exp/tanh/isfinite 等数值函数
import math
# 导入 json：序列化 payload
import json
# 导入 random：为重试间隔加随机抖动
import random
# 导入 pandas：DataFrame 相关依赖
import pandas as pd
# 导入 Path：路径处理
from pathlib import Path

# Configure logging format with timestamp at module level
# 在模块级配置名为 mle_agent 的日志器（带时间戳格式）
logger = logging.getLogger("mle_agent")
# 仅在尚未配置 handler 时初始化，避免重复添加
if not logger.handlers:
    # Set up console handler with timestamp format
    # 创建输出到控制台的 handler
    console_handler = logging.StreamHandler()
    # 设置"时间-名称-级别-消息"日志格式
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    # 把 handler 挂载到 logger
    logger.addHandler(console_handler)

def normalize_sandbox_resource_type(
    resource_type: str | None, default: str = "gpu"
) -> str:
    """Normalize sandbox resource type to cpu/gpu with a safe default."""
    # 先规整默认值：转小写、去首尾空白
    normalized_default = str(default or "gpu").strip().lower()
    # 默认值不合法时兜底为 "gpu"
    if normalized_default not in {"cpu", "gpu"}:
        normalized_default = "gpu"

    # 未传入资源类型时直接返回（已规整的）默认值
    if resource_type is None:
        return normalized_default

    # 规整传入的资源类型字符串
    normalized_resource = str(resource_type).strip().lower()
    # 合法（cpu/gpu）则直接返回
    if normalized_resource in {"cpu", "gpu"}:
        return normalized_resource
    # 不合法则回退到默认值
    return normalized_default


def resolve_sandbox_resource_type(
    metadata: dict | None,
    override: str | None = None,
    default: str = "gpu",
) -> str:
    """Resolve effective sandbox resource type from override or task metadata."""
    # 显式覆盖（override）优先级最高
    if override is not None:
        return normalize_sandbox_resource_type(override, default=default)
    # 否则从任务 metadata 的 cpu_gpu 字段解析
    task_resource = (metadata or {}).get("cpu_gpu")
    return normalize_sandbox_resource_type(task_resource, default=default)


async def get_sandbox_result(
    client: httpx.AsyncClient,
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
    # 异步提交代码到沙箱并轮询直至完成，返回 (状态码, 结果负载) 二元组
    # 规整资源类型（cpu/gpu）
    normalized_resource = normalize_sandbox_resource_type(resource_type, default="gpu")
    # 优先级仅允许 1/2，非法值回退为 1
    normalized_priority = priority if priority in (1, 2) else 1
    # 读取客户端 base_url（去掉末尾斜杠）用于错误信息展示
    client_base_url = str(getattr(client, "base_url", "")).rstrip("/")
    # base_url 缺失时直接报错
    if not client_base_url:
        raise ValueError("The sandbox HTTP client must define a base URL.")
    # 读取资源类型专属 API Key，缺失时回退通用 Key
    sandbox_api_key = (
        os.getenv(f"SANDBOX_{normalized_resource.upper()}_API_KEY")
        or os.getenv("SANDBOX_API_KEY")
    )
    # 未配置任何 API Key 则报错
    if not sandbox_api_key:
        raise ValueError(
            f"SANDBOX_{normalized_resource.upper()}_API_KEY or SANDBOX_API_KEY "
            "is required."
        )
    # 组装提交负载：任务名/代码/数据目录/超时/资源类型/优先级及运行环境变量
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
        },
    }
    # 请求头携带 API Key 鉴权
    headers = {"X-API-Key": sandbox_api_key}

    def safe_json(resp: httpx.Response):
        # 容错解析响应 JSON：失败或空响应统一返回空字典
        try:
            data = resp.json()
            return {} if data is None else data
        except Exception:
            return {}

    def format_httpx_error(exc: Exception, fallback_url: str) -> str:
        # 把 httpx 异常格式化为可读的详细错误串（类型/方法/URL/消息/根因）
        parts = [f"type={type(exc).__name__}"]
        # 尝试取出关联的请求对象
        req = getattr(exc, "request", None)
        if req is not None:
            # 有请求对象：记录 HTTP 方法与 URL（缺失时用兜底 URL）
            method = getattr(req, "method", "UNKNOWN")
            url = getattr(req, "url", fallback_url)
            parts.append(f"method={method}")
            parts.append(f"url={url}")
        else:
            # 无请求对象：仅记录兜底 URL
            parts.append(f"url={fallback_url}")

        # 异常消息非空记录消息，否则记录 repr
        message = str(exc).strip()
        if message:
            parts.append(f"message={message}")
        else:
            parts.append(f"repr={exc!r}")

        # 若存在底层根因异常（__cause__），追加记录
        cause = getattr(exc, "__cause__", None)
        if cause is not None:
            cause_message = str(cause).strip()
            if cause_message:
                parts.append(f"cause={type(cause).__name__}: {cause_message}")
            else:
                parts.append(f"cause={cause!r}")
        # 用分号拼接各部分返回
        return "; ".join(str(p) for p in parts)

    # Submit the job without waiting for completion.
    # 第一步：提交任务（POST /jobs），对瞬时故障做有限次重试
    submit_connect_retries = 0
    # 提交阶段最大重试次数
    max_submit_connect_retries = 20
    submit_resp = None
    # 循环直至提交成功或重试额度耗尽
    while True:
        try:
            # 发送提交请求
            submit_resp = await client.post(
                "/api/v1/jobs",
                json=payload,
                headers=headers,
            )
            # 网关类错误（502/503/504）与限流（429）视为瞬时故障
            if submit_resp is None or submit_resp.status_code in (502, 503, 504, 429):
                # 仍有重试额度：等待 1 秒后重试
                if submit_connect_retries < max_submit_connect_retries:
                    submit_connect_retries += 1
                    await asyncio.sleep(float(1))
                    status_code = (
                        submit_resp.status_code if submit_resp is not None else "none"
                    )
                    print(
                        f"WARNING: transient error {status_code} when submitting job, retrying...",
                        flush=True,
                    )
                    continue
            # 成功或额度耗尽：跳出重试循环
            break
        except Exception as e:
            # 网络异常：格式化错误详情
            error_detail = format_httpx_error(e, client_base_url)
            # 仍有重试额度：按递增退避 + 随机抖动后重试
            if submit_connect_retries < max_submit_connect_retries:
                submit_connect_retries += 1
                retry_delay = 1.0 * submit_connect_retries + random.uniform(0.0, 1.0)
                print(
                    f"WARNING: transient submit connect error to {client_base_url}: {error_detail}; "
                    f"retry {submit_connect_retries}/{submit_connect_retries} in {retry_delay:.2f}s",
                    flush=True,
                )
                await asyncio.sleep(retry_delay)
                continue
            # 重试耗尽：返回 503 连接失败结果
            error_msg = (
                f"Failed to connect to sandbox API at {client_base_url}: {error_detail}"
            )
            print(f"ERROR: {error_msg}")
            return 503, {"error": "connection_failed", "detail": error_msg, "type": type(e).__name__}
        # except httpx.TimeoutException as e:
        #     error_msg = f"Failed to submit job to sandbox API at {client_base_url}: {format_httpx_error(e, client_base_url)}"
        #     print(f"ERROR: {error_msg}")
        #     return 503, {"error": "submit_timeout", "detail": error_msg, "type": type(e).__name__}
        # except httpx.TransportError as e:
        #     error_msg = f"Failed to submit job to sandbox API at {client_base_url}: {format_httpx_error(e, client_base_url)}"
        #     print(f"ERROR: {error_msg}")
        #     return 503, {"error": "submit_transport_failed", "detail": error_msg, "type": type(e).__name__}

    # 提交响应缺失：返回 503 连接失败
    if submit_resp is None:
        return 503, {
            "error": "connection_failed",
            "detail": f"Failed to connect to sandbox API at xxx",
            "type": "ConnectError",
        }
    # 容错解析提交响应
    submit_data = safe_json(submit_resp)
    print("submit_data:", submit_data)
    # 提交返回 4xx/5xx：直接把状态码与错误数据返回给调用方
    if submit_resp.status_code >= 400:
        return submit_resp.status_code, {"error": "submit failed", "data": submit_data}

    # 取出任务 ID
    job_id = submit_data.get("job_id")
    # 服务端未返回 job_id：返回 500
    if not job_id:
        return 500, {"error": "no job_id returned", "data": submit_data}

    # Poll until the job completes or the overall timeout expires.
    # 第二步：轮询任务状态直至完成或总等待超时
    # 仍在进行中的状态集合
    wait_status = ["running", "queued"]
    # 已终结的状态集合
    finished_status = ["success", "failed", "timeout", "canceled", "completed"]
    # 用单调时钟计算轮询截止时刻（避免系统时间跳变影响）
    deadline = time.monotonic() + wait_timeout
    poll_error_retries = 0
    # 轮询阶段最大连续错误重试次数
    max_poll_error_retries = 20
    # 累计运行时长（秒）
    total_running_time = 0.0
    last_status = None
    last_status_poll_at = None
    # 在截止时间前持续轮询
    while time.monotonic() < deadline:
        try:
            # 查询任务状态
            r = await client.get(f"/api/v1/jobs/{job_id}", headers=headers)
            # 网关类错误与限流视为瞬时故障
            if r is None or r.status_code in (502, 503, 504, 429):
                # 仍有重试额度：退避等待后重试
                if poll_error_retries < max_poll_error_retries:
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
            # 轮询网络异常：格式化错误详情
            error_detail = format_httpx_error(
                e, f"{client_base_url}/api/v1/jobs/{job_id}"
            )
            # 仍有重试额度：退避等待后重试
            if poll_error_retries < max_poll_error_retries:
                poll_error_retries += 1
                retry_delay = min(5.0, 1.0 + 0.25 * poll_error_retries) + random.uniform(0.0, 0.5)
                print(
                    f"WARNING: transient poll transport error for {job_id}: {error_detail}; "
                    f"retry {poll_error_retries}/{max_poll_error_retries} in {retry_delay:.2f}s",
                    flush=True,
                )
                await asyncio.sleep(retry_delay)
                continue
            # 重试耗尽：返回 503 轮询失败
            error_msg = f"Failed to poll job status: {error_detail}"
            print(f"ERROR: {error_msg}")
            return 503, {"error": "poll_failed", "detail": error_msg, "job_id": job_id, "type": type(e).__name__}
        # 一次成功轮询即清零错误计数
        poll_error_retries = 0
        data = safe_json(r)
        # 读取当前任务状态
        status = data.get("status")
        # 记录本次轮询时刻
        current_poll_at = time.monotonic()
        # 上一轮状态也是 running 时，把两次轮询间隔累加进运行时长
        if last_status == "running" and last_status_poll_at is not None:
            total_running_time += max(0.0, current_poll_at - last_status_poll_at)
        # 更新"上一次状态/上一次轮询时刻"
        last_status = status
        last_status_poll_at = current_poll_at
        print(f"job_id:{job_id}, status({time.monotonic()}):{status}")
        # 仍在排队/运行：按轮询间隔 + 随机抖动继续等待
        if status in wait_status:
            await asyncio.sleep(poll_interval + random.uniform(0.0, 0.5))
            continue
        # 把累计运行时长写入结果负载
        data["running_time"] = total_running_time
        # 进入已知终态：返回 200 与结果
        if status in finished_status:
            return 200, data
        else:
            # 未知状态：同样返回当前数据交由上层判断
            return 200, data

    # 整体等待超时：返回 504 与已累计的运行时长
    return 504, {"error": "wait_timeout exceeded", "job_id": job_id, "running_time": total_running_time}



def is_valid_python_script(script):
    """Check if a script is a valid Python script."""
    # 尝试编译脚本以验证语法合法性
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        # 出现语法错误则视为无效脚本
        return False


def format_code(code) -> str:
    """Format Python code using Black."""
    # 用 Black 统一格式化代码
    try:
        return black.format_str(code, mode=black.FileMode())
    except black.parsing.InvalidInput:  # type: ignore
        # 无法解析（如含非法语法）时原样返回
        return code


def extract_code(text):
    """Extract python code blocks from the text."""
    # logger.info(f"raw text: {text}")
    # 存放抽取出的候选代码块
    parsed_codes = []

    # When code is in a text or python block
    # 情形一：从 ```python / ``` 围栏代码块中抽取（re.DOTALL 允许跨行匹配）
    matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        # 取围栏内的代码体（第 2 个捕获组）
        code_block = match[1]
        parsed_codes.append(code_block)

    # When the entire text is code or backticks of the code block is missing
    # 情形二：没有匹配到围栏时，把整段文本当作代码
    if len(parsed_codes) == 0:
        matches = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            # 取第 3 个捕获组（剥离可选围栏与首尾空行）
            code_block = matches[0][2]
            parsed_codes.append(code_block)

    # validate the parsed codes
    # 校验语法合法性后逐块格式化，仅保留有效代码块
    valid_code_blocks = [
        format_code(c) for c in parsed_codes if is_valid_python_script(c)
    ]
    # logger.info(f"valid_code_blocks: {valid_code_blocks}")
    # 合并所有有效代码块并整体格式化后返回
    return format_code("\n\n".join(valid_code_blocks))


# 在线百分位模式的缓存：按任务 key 保存最近样本分数
_RECENT = {}
# 最近样本队列的最大长度
_RECENT_MAXLEN = 20


def _stable_sigmoid(x: float) -> float:
    # 数值稳定的 sigmoid（避免 exp 溢出），用于奖励归一化
    # x>=0 时用标准式
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    # x<0 时用等价变换式，防止 exp(-x) 上溢
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _signed(score: float, meta: dict) -> float:
    # 按"是否越大越好"统一转换：确保返回值越大代表结果越好
    higher = meta.get("higher_is_better", True)
    s = float(score)
    return s if higher else -s


def _finite(x) -> bool:
    # 判断值是否为有限数值（非 None/NaN/inf）
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        # 转换失败一律视为非有限
        return False


def _bounds_signed(meta: dict):
    # 汇总理论/榜单上下界，返回符号化后的 (最好值, 最差值) 区间
    higher = meta.get("higher_is_better", True)

    def to_signed(v):
        # 单个边界值的符号化（非有限值返回 None）
        if not _finite(v):
            return None
        v = float(v)
        return v if higher else -v

    # 理论与榜单的上下界分别做符号化
    tmin = to_signed(meta.get("theoretical_min"))
    tmax = to_signed(meta.get("theoretical_max"))
    lmin = to_signed(meta.get("leaderboard_min"))
    lmax = to_signed(meta.get("leaderboard_max"))

    # 汇总所有可用边界候选
    cands = [tmin, tmax, lmin, lmax]
    cands = [x for x in cands if x is not None]
    # 少于 2 个边界无法构成区间
    if len(cands) < 2:
        return None, None

    # 区间上端取最大（最好），下端取最小（最差）
    best = max(cands)
    worst = min(cands)

    # 区间跨度过大（>1e6）视为异常，需要收敛
    MAX_RANGE = 1e6
    if best - worst > MAX_RANGE:
        # 优先退回跨度可控的榜单区间
        if lmax is not None and lmin is not None and (lmax - lmin) <= MAX_RANGE:
            best, worst = max(lmax, lmin), min(lmax, lmin)
        else:
            # 否则把下端截断到 best - MAX_RANGE
            worst = best - MAX_RANGE

    return best, worst


def score2reward(score, metadata, mode="logistic"):
    """Map a metric score to a normalized reward."""
    # 读取指标元信息：方向性与理论/榜单边界（用于奖励归一化）
    higher_is_better = metadata.get("higher_is_better")
    theoretical_min = metadata.get("theoretical_min")
    theoretical_max = metadata.get("theoretical_max")
    leaderboard_min = metadata.get("leaderboard_min")
    leaderboard_max = metadata.get("leaderboard_max")

    def safe_max(*vals):
        # 忽略 None 后取最大（全为空返回 None）
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None

    def safe_min(*vals):
        # 忽略 None 后取最小（全为空返回 None）
        vals = [v for v in vals if v is not None]
        return min(vals) if vals else None

    # assert mode=="linear_sign" , f"Unsupported mode: {mode}"

    # 模式 linear_sign：保号线性映射（符号即奖励，用于二元"达标/不达标"信号）
    if mode == "linear_sign":
        # 分数绝对值近似 0：视为未产出有效提交，给强负奖励
        if abs(float(score)) < 1e-7:
            return -100.0

        # 指标越小越好时取相反数，统一为越大越好
        if higher_is_better is False:
            return -float(score)
        else:
            return float(score)

    # 统一转换为"越大越好"的带符号分数
    s = _signed(score, metadata)

    # A: Power sigmoid on min-max bounds.
    # 模式 power_sigmoid：基于边界的幂次 sigmoid 奖励塑形
    if mode == "power_sigmoid":
        # 取符号化后的 (最好, 最差) 区间
        best, worst = _bounds_signed(metadata)
        # 无区间信息时退化为原始 sigmoid
        if best is None:
            return _stable_sigmoid(s)

        # 区间宽度（防 0）
        rng = max(best - worst, 1e-9)
        # p in [0,1]
        # 计算分数在区间中的归一化位置 p∈[0,1]
        p = (s - worst) / rng
        # 越界时截断到 [0,1]
        p = 0.0 if p < 0 else (1.0 if p > 1 else p)

        # 温度 T 控制 sigmoid 过渡带宽度
        T = 0.50
        # 以 0.5 为中心、T 为尺度做标准化
        x = (p - 0.5) / max(T, 1e-9)
        r = _stable_sigmoid(x)  # (0,1)

        # 幂次 alpha>1 拉大优劣样本的奖励差距
        alpha = 2.0
        return r**alpha

    # B: Tanh margin shaping.
    # 模式 margin_tanh：以区间中心为基准的 tanh 边际塑形
    if mode == "margin_tanh":
        # 取符号化后的 (最好, 最差) 区间
        best, worst = _bounds_signed(metadata)
        # 无区间时退化为 0.5+0.5*tanh(s)
        if best is None:
            return 0.5 + 0.5 * math.tanh(s)

        # 区间中心作为基准点
        center = 0.5 * (best + worst)
        # 半区间宽度作为尺度（防 0）
        scale = max(0.5 * (best - worst), 1e-9)

        # 计算相对中心的边际 m
        m = (s - center) / scale

        # 温度 T 控制 tanh 过渡带宽度
        T = 0.50
        y = math.tanh(m / max(T, 1e-9))  # (-1,1)
        r = 0.5 + 0.5 * y  # (0,1)

        # 幂次 gamma>1 强化头部样本优势
        gamma = 2.0
        return r**gamma

    # C: Online percentile.
    # 模式 online_percentile：按最近样本的在线百分位给奖励（相对排名信号）
    if mode == "online_percentile":
        # 以任务数据目录为 key，取该任务的最近分数队列
        key = str(metadata.get("data_dir", "unknown_task"))
        dq = _RECENT.get(key)
        if dq is None:
            # 首次遇到该任务：初始化定长双端队列
            from collections import deque

            dq = deque(maxlen=_RECENT_MAXLEN)
            _RECENT[key] = dq

        # 样本不足 10 个：先累积样本，用 sigmoid 给基础奖励
        if len(dq) < 10:
            dq.append(s)
            return _stable_sigmoid(s)

        # 对历史分数排序
        sorted_vals = sorted(dq)
        # rank = count(vals <= s)
        # 用二分查找计算当前分数的排名（<=s 的个数）
        import bisect

        rank = bisect.bisect_right(sorted_vals, s)
        p = rank / len(sorted_vals)  # [0,1]

        # 当前分数入队（定长队列自动淘汰最旧样本）
        dq.append(s)

        # 幂次放大高百分位样本的优势
        gamma = 2.0
        return p**gamma

    # initial sigmoid reward
    # 默认 logistic 模式：用理论/榜单边界做 min-max 归一化后过 sigmoid
    if higher_is_better:
        # best score
        # 越大越好：最好值优先取理论上界，缺失时用榜单上界
        score_range_best_score = (
            theoretical_max if theoretical_max is not None else leaderboard_max
        )

        # worst score
        # 最差值取榜单下界/理论下界与 -100 的最大者（兜底防区间爆炸）
        score_range_worst_score = safe_max(leaderboard_min, theoretical_min, -100)

    else:
        # best score
        # 越小越好：最好值优先取理论下界，缺失时用榜单下界
        score_range_best_score = (
            theoretical_min if theoretical_min is not None else leaderboard_min
        )

        # worst score
        # 最差值取榜单上界/理论上界与 100 的最小者
        score_range_worst_score = safe_min(leaderboard_max, theoretical_max, 100)

    # Return a neutral reward when the score range remains unknown.
    # 区间未知：返回中性奖励 0.5
    if score_range_best_score is None or score_range_worst_score is None:
        return 0.5

    # 计算区间宽度
    score_range = score_range_best_score - score_range_worst_score
    # 宽度为 0 或 NaN：无法归一化，返回中性奖励
    if score_range == 0 or math.isnan(score_range):
        return 0.5

    # 把分数线性映射到 [-2,2]（相对最差值的位置 ×2）
    x = 2.0 * (score - score_range_worst_score) / score_range
    # 数值稳定地计算 sigmoid 奖励
    if x >= 0:
        reward = 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        reward = exp_x / (1.0 + exp_x)

    return reward


def get_clear_log(run_log: str | None) -> str:
    """Extract marked output sections and remove heartbeat log blocks."""
    # 空日志直接返回空串
    if not run_log:
        return ""

    # 支持两组输出标记（通用输出段 / 沙箱 stdout 段）
    markers = [
        ("--- OUTPUT START ---", "--- OUTPUT END ---"),
        ("--- SANDBOX STDOUT START ---", "--- SANDBOX STDOUT END ---"),
    ]

    # 心跳日志标记及其长度
    hb_marker = "[HB]"
    hb_len = len(hb_marker)

    # 收集清洗后的段落
    sections: list[str] = []

    # 逐对标记扫描日志文本
    for start_marker, end_marker in markers:
        # 从上次结束位置继续搜索
        search_pos = 0
        while True:
            # 定位下一个起始标记
            start_idx = run_log.find(start_marker, search_pos)
            if start_idx == -1:
                # 该标记对已无更多段落，处理下一对
                break

            # 内容从起始标记之后开始
            content_start = start_idx + len(start_marker)
            # 跳过标记后紧跟的换行符
            if content_start < len(run_log) and run_log[content_start] == "\n":
                content_start += 1

            # 定位结束标记
            end_idx = run_log.find(end_marker, content_start)
            if end_idx == -1:
                # 无结束标记：内容取到文本末尾
                content = run_log[content_start:].strip()
            else:
                # 有结束标记：取两标记之间的内容
                content = run_log[content_start:end_idx].strip()

            # Remove every heartbeat block with index searches.
            # 用索引扫描删除所有 [HB]...[HB] 心跳块
            if content:
                parts: list[str] = []
                pos = 0
                while True:
                    # 找心跳块起点
                    s = content.find(hb_marker, pos)
                    if s == -1:
                        # 之后无心跳内容，剩余部分全部保留
                        parts.append(content[pos:])
                        break
                    # 保留心跳块之前的内容
                    parts.append(content[pos:s])
                    # 找心跳块终点（下一个 [HB] 标记）
                    e = content.find(hb_marker, s + hb_len)
                    if e == -1:
                        # Drop an unterminated heartbeat block through end of text.
                        # 未闭合的心跳块：丢弃直到文本末尾
                        break
                    # 跳到该心跳块之后继续扫描
                    pos = e + hb_len

                # 重新拼接并去掉首尾空白
                content = "".join(parts).strip()

            # 非空段落才收集（前置换行便于后续按空行拼接）
            if content:
                sections.append(f"\n{content}")

            if end_idx == -1:
                # 无结束标记说明已扫到末尾
                break
            # 从结束标记之后继续搜索下一段
            search_pos = end_idx + len(end_marker)

    # 用空行拼接所有清洗后的段落
    return "\n\n".join(sections)


def format_sandbox_feedback(status_code: int, payload: dict) -> str:
    """Format sandbox execution results into feedback message"""
    # 分支一：HTTP 200，正常拿到沙箱执行结果
    if status_code == 200:
        result = payload.get("result") or {}
        status = payload.get("status", "unknown")
        run_log = get_clear_log(result.get("run_log"))
        run_result = result.get("result", "")
        score = result.get("score")

        # 连续拼接 Markdown 反馈：标题与执行状态
        feedback = f"## Execution Result\n"
        feedback += f"**Status**: {status}\n\n"

        # 有得分时附上得分行
        if score is not None:
            feedback += f"**Score**: {score}\n\n"

        # 有结果摘要时附上结果行
        if run_result:
            feedback += f"**Result**: {run_result}\n\n"

        if run_log:
            # Truncate log if too long
            # 日志过长（>2000 字符）时截断：保留头尾各 1000 字符
            if len(run_log) > 2000:
                run_log = run_log[:1000] + "\n... (truncated) ...\n" + run_log[-1000:]
            feedback += f"**Execution Log**:\n```\n{run_log}\n```\n\n"

    elif status_code == 503:
        # 分支二：连接沙箱失败（503）
        error_type = payload.get("type", "unknown")
        error_detail = payload.get("detail", "Connection failed")
        # 连续拼接错误类型/详情与重试建议
        feedback = f"## Connection Error\n"
        feedback += f"**Type**: {error_type}\n"
        feedback += f"**Detail**: {error_detail}\n\n"
        feedback += "Please check your code and try again."
    else:
        # 分支三：其他执行错误
        error_msg = payload.get("error", "unknown error")
        detail = payload.get("detail", {})
        # 拼接错误标题与错误信息
        feedback = f"## Execution Error\n"
        feedback += f"**Error**: {error_msg}\n"
        # detail 为字典时补充状态与消息字段
        if isinstance(detail, dict):
            if detail.get("status"):
                feedback += f"**Status**: {detail.get('status')}\n"
            if detail.get("message"):
                feedback += f"**Message**: {detail.get('message')}\n"
        feedback += "\nPlease fix the errors and try again."

    return feedback


if __name__ == "__main__":
    # Test score2reward for all tasks
    # 直接运行本模块时：测试 score2reward（test_score2reward 由外部环境提供）
    test_score2reward()
