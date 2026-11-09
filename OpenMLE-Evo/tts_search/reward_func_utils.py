# 中文说明:tts_search 的奖励函数工具库——"远程沙箱执行 -> 分数抽取 -> 奖励映射"的公共底座。
# 在系统中的位置:OpenMLE-Evo/tts_search 下,被生成脚本(generate_mle 等)与评估脚本引用;
# 职责包括:
# 1) 沙箱作业的提交与轮询(get_sandbox_result,幂等键+重试+超时保护;旧版 deprecated 保留);
# 2) LLM 输出的代码抽取与格式化(extract_code/format_code);
# 3) 分数到奖励的整形(score2reward:logistic/power_sigmoid/margin_tanh/online_percentile);
# 4) 运行日志清洗(get_clear_log 去心跳噪声)、最终验证分解析、反馈消息排版。
# 关键设计:对可重试状态码(429/5xx)做指数退避+抖动;奖励映射统一把分数
# 按 higher_is_better 翻转成"越大越好"的有符号量后再整形。

# 延迟注解:类型标注运行期不求值
from __future__ import annotations

# os:读取沙箱服务地址/密钥等环境变量
import os
# time:monotonic 时钟(轮询 deadline)
import time
# httpx:异步 HTTP 客户端(沙箱 API)
import httpx
# asyncio:协程睡眠/异步控制
import asyncio
# random:重试退避的随机抖动(防雪崩)
import random
# re:代码块/最终验证分等正则抽取
import re
# logging:模块日志
import logging
# black:代码格式化(抽取后统一风格)
import black
# math:sigmoid/tanh/有限性判断
import math
# json:读取任务元数据
import json
# pandas:排行榜 CSV 解析(测试函数用)
import pandas as pd
# Path:路径对象
from pathlib import Path
# uuid:生成幂等键/trace_id
import uuid
# warnings:标记弃用 API
import warnings

# 模块级配置带时间戳的日志格式
logger = logging.getLogger('mle_agent')
# 尚无 handler 时才挂载,避免重复输出
if not logger.handlers:
    # 创建控制台 handler 并设置带时间戳格式
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(console_handler)

# 沙箱服务地址(环境变量注入,可为空走默认)
BASE_URL = os.environ.get("SANDBOX_URL", "")
# 沙箱 GPU 队列的 API 密钥
API_KEY = os.environ.get("SANDBOX_GPU_API_KEY", "")

# 视为"可重试瞬时错误"的 HTTP 状态码集合(限流/网关错误)
RETRYABLE_SANDBOX_STATUS_CODES = {429, 502, 503, 504}


def _build_retryable_sandbox_payload(
    *,
    error: str,
    detail: str,
    status_code: int | None = None,
    job_id: str | None = None,
    data: dict | None = None,
    error_type: str | None = None,
) -> dict:
    """构造"可重试失败"的标准化载荷(retryable=True,调用方据此决定重排重试)。"""
    # 基础字段:错误码/详情/可重试标记
    payload = {
        "error": error,
        "detail": detail,
        "retryable": True,
    }
    # 有状态码则带上
    if status_code is not None:
        payload["status_code"] = int(status_code)
    # 有作业 ID 则带上(轮询失败场景)
    if job_id is not None:
        payload["job_id"] = job_id
    # 有响应体则带上(便于排查)
    if data is not None:
        payload["data"] = data
    # 有异常类型则带上
    if error_type is not None:
        payload["type"] = error_type
    return payload

async def get_sandbox_result_deprecated(
    client: httpx.AsyncClient,         # ✅ 复用的客户端从外部传入
    code_str: str,
    data_dir: str,
    *,
    resource_type: str,
    eval_split=None,
    priority: int = 1,
    job_timeout: int = 3600,
    wait_timeout: int = 86400,
    # wait_timeout: int = 100,
    poll_interval: int = 5,
):
    """旧版沙箱客户端入口(v0.9.32 前):提交+轮询一体,但无幂等键,新代码请用 get_sandbox_result。"""
    # 发出弃用警告(指向新接口)
    warnings.warn(
        "get_sandbox_result_deprecated() uses the pre-v0.9.32 sandbox client. "
        "Use get_sandbox_result() for idempotent submit retry and safer polling.",
        DeprecationWarning,
        stacklevel=2,
    )
    # 资源类型统一小写
    normalized_resource = str(resource_type).lower()
    # 组装作业载荷:任务名/代码/数据目录/超时/资源与优先级/执行环境
    payload = {
        "name": data_dir,
        "code": code_str,
        "data_dir": data_dir,
        "timeout": job_timeout,
        "resource_type": normalized_resource,
        "priority": int(priority),  # 1 (high) | 2 (low)
        "environment": {
            "EXECUTION_MODE": "shell",
            "DATA_DIR": data_dir,
            "SANDBOX_DATA_DIR": data_dir,
        },
    }
    # 请求头带 API 密钥
    headers = {"X-API-Key": API_KEY}

    # 安全 JSON 解析:任何异常都返回空字典
    def safe_json(resp: httpx.Response):
        try:
            data = resp.json()
            return {} if data is None else data
        except Exception:
            return {}

    # 1) 只提交，不等待（带重试）
    submit_error_retries = 0
    while True:
        try:
            # 提交作业(相对路径,复用 client.base_url)
            submit_resp = await client.post(
                "/api/v1/jobs",              # ✅ 使用 base_url，path 写相对地址
                json=payload,
                headers=headers,
            )
            # 可重试状态码:最多重试 5 次(固定 1 秒间隔)
            if submit_resp.status_code in RETRYABLE_SANDBOX_STATUS_CODES:
                if submit_error_retries < 5:
                    submit_error_retries += 1
                    await asyncio.sleep(float(1))
                    logger.info(f"WARNING: transient error {submit_resp.status_code} when submitting job, retrying...")
                    continue
                # 重试耗尽:返回标准化的可重试失败载荷
                detail = (
                    f"Sandbox API returned retryable status {submit_resp.status_code} "
                    "when submitting job after retries"
                )
                logger.info(f"ERROR: {detail}")
                return submit_resp.status_code, _build_retryable_sandbox_payload(
                    error="submit_transient_failed",
                    detail=detail,
                    status_code=submit_resp.status_code,
                    data=safe_json(submit_resp),
                )
            # 非可重试状态:跳出提交循环
            break
        except Exception as e:
            # 连接异常:同样最多重试 5 次
            if submit_error_retries < 5:
                submit_error_retries += 1
                await asyncio.sleep(float(1))
                logger.info(f"WARNING: exception {type(e).__name__} when submitting job, retrying...")
                continue
            # 重试耗尽:返回连接失败(503)
            error_msg = f"Failed to connect to sandbox API at {client.base_url}: {type(e).__name__}: {str(e)}"
            logger.info(f"ERROR: {error_msg}")
            return 503, _build_retryable_sandbox_payload(
                error="connection_failed",
                detail=error_msg,
                status_code=503,
                error_type=type(e).__name__,
            )
    # 解析提交响应
    submit_data = safe_json(submit_resp)
    logger.info("submit_data: %s", submit_data)
    # 提交仍失败(>=400):原样返回状态码
    if submit_resp.status_code >= 400:
        return submit_resp.status_code, {"error": "submit failed", "data": submit_data}

    # 取出作业 ID
    job_id = submit_data.get("job_id")
    if not job_id:
        # 无作业 ID:服务端异常,返回 500
        return 500, {"error": "no job_id returned", "data": submit_data}

    # 2) 轮询直到结束或完全超时
    # 进行中的状态集合
    wait_status = ["running", "queued"]
    # 轮询死线(monotonic 时钟,不受系统改时影响)
    deadline = time.monotonic() + wait_timeout
    # 轮询错误重试计数
    poll_error_retries = 0
    while time.monotonic() < deadline:
        try:
            # 查询作业状态
            r = await client.get(f"/api/v1/jobs/{job_id}", headers=headers)
            # 可重试状态码:最多重试 5 次
            if r.status_code in RETRYABLE_SANDBOX_STATUS_CODES:
                if poll_error_retries < 5:
                    poll_error_retries += 1
                    await asyncio.sleep(float(1))
                    logger.info(f"WARNING: transient error {r.status_code} when polling job status, retrying...")
                    continue
                # 重试耗尽:返回可重试失败载荷
                detail = (
                    f"Sandbox API returned retryable status {r.status_code} "
                    "when polling job status after retries"
                )
                logger.info(f"ERROR: {detail}")
                return r.status_code, _build_retryable_sandbox_payload(
                    error="poll_transient_failed",
                    detail=detail,
                    status_code=r.status_code,
                    job_id=job_id,
                    data=safe_json(r),
                )
        except Exception as e:
            # 轮询异常:不重试,直接返回 503(作业仍在服务端运行)
            error_msg = f"Failed to poll job status: {type(e).__name__}: {str(e)}"
            logger.info(f"ERROR: {error_msg}")
            return 503, _build_retryable_sandbox_payload(
                error="poll_failed",
                detail=error_msg,
                status_code=503,
                job_id=job_id,
                error_type=type(e).__name__,
            )
        # 4xx/5xx 等非可重试失败:原样返回
        if r.status_code >= 400:
            return r.status_code, {
                "error": "poll failed",
                "detail": safe_json(r),
                "job_id": job_id,
            }
        # 解析状态
        data = safe_json(r)
        status = data.get("status")
        logger.info(f"job_id:{job_id}, status({time.monotonic()}):{status}")
        # 按固定间隔轮询
        await asyncio.sleep(poll_interval)
        if status in wait_status:
            # 仍在运行/排队:继续轮询
            continue
        else:
            # 到达终态:返回 200 与完整数据
            return 200, data

    # 超出等待时限:返回 504
    return 504, {"error": "wait_timeout exceeded", "job_id": job_id}


async def get_sandbox_result(
    client: httpx.AsyncClient,
    code_str: str,
    data_dir: str,
    *,
    resource_type: str = "gpu",
    priority: int = 1,
    job_timeout: int = 3600,
    wait_timeout: int = 86400,
    poll_interval: int = 10,
    eval_split: str | None = None,
):
    """提交代码到远程沙箱并轮询到终态;幂等键+指数退避重试,返回 (状态码, 结果载荷)。"""
    # 资源类型归一(空值回退 cpu)
    normalized_resource = (resource_type or "cpu").lower()
    # 优先级只允许 1/2,非法回退 1(高优)
    normalized_priority = priority if priority in (1, 2) else 1
    # 本次请求的幂等键/追踪 ID
    trace_id = uuid.uuid4().hex
    # 规整服务地址(去尾部斜杠)
    base_url = str(client.base_url).rstrip("/")
    # 作业载荷:名称/代码/数据目录/超时/资源/优先级/幂等键/执行环境
    payload = {
        "name": data_dir,
        "code": code_str,
        "data_dir": data_dir,
        "timeout": job_timeout,
        "resource_type": normalized_resource,
        "priority": normalized_priority,
        "idempotency_key": trace_id,
        "environment": {
            "EXECUTION_MODE": "shell",
            "DATA_DIR": data_dir,
            "SANDBOX_DATA_DIR": data_dir,
        },
    }
    # 指定评估切分时:载荷与环境变量双重注入
    if eval_split:
        normalized_eval_split = str(eval_split).strip().lower()
        payload["eval_split"] = normalized_eval_split
        payload["environment"]["EVAL_SPLIT"] = normalized_eval_split
    # 请求头:API 密钥 + 追踪 ID
    headers = {"X-API-Key": API_KEY, "X-Trace-ID": trace_id}

    # 安全 JSON 解析:异常/None 都回退空字典
    def safe_json(resp: httpx.Response):
        try:
            data = resp.json()
            return {} if data is None else data
        except Exception:
            return {}

    # 把 httpx 异常格式化成可读的一行(类型/方法/URL/消息/根因)
    def format_httpx_error(exc: Exception, fallback_url: str) -> str:
        parts = [f"type={type(exc).__name__}"]
        # 优先取异常自带的请求信息
        req = getattr(exc, "request", None)
        if req is not None:
            method = getattr(req, "method", "UNKNOWN")
            url = getattr(req, "url", fallback_url)
            parts.append(f"method={method}")
            parts.append(f"url={url}")
        else:
            parts.append(f"url={fallback_url}")

        # 消息为空则用 repr
        message = str(exc).strip()
        parts.append(f"message={message}" if message else f"repr={exc!r}")

        # 附带异常根因(raise ... from)
        cause = getattr(exc, "__cause__", None)
        if cause is not None:
            cause_message = str(cause).strip()
            if cause_message:
                parts.append(f"cause={type(cause).__name__}: {cause_message}")
            else:
                parts.append(f"cause={cause!r}")
        return "; ".join(str(part) for part in parts)

    # 读取服务端的 Retry-After 建议间隔(截断到 [0,30] 秒)
    def retry_after_seconds(resp: httpx.Response, fallback: float) -> float:
        value = resp.headers.get("Retry-After")
        if value:
            try:
                return max(0.0, min(float(value), 30.0))
            except ValueError:
                pass
        return fallback

    # 提交阶段最多重试 50 次(指数退避,应对沙箱服务抖动)
    submit_connect_retries = 50
    # 提交阶段视为瞬时的状态码
    transient_submit_statuses = {429, 500, 502, 503, 504}
    submit_resp = None
    # 逐次尝试提交
    for submit_attempt in range(1, submit_connect_retries + 1):
        try:
            # 发起提交请求
            submit_resp = await client.post(
                "/api/v1/jobs",
                json=payload,
                headers=headers,
            )
            if (
                submit_resp.status_code in transient_submit_statuses
                and submit_attempt < submit_connect_retries
            ):
                # 瞬时失败:按 Retry-After 或指数退避+抖动等待后重试
                retry_delay = retry_after_seconds(
                    submit_resp,
                    min(30.0, 1.5**submit_attempt) + random.uniform(0.0, 1.0),
                )
                logger.info(
                    "WARNING: transient submit status %s trace_id=%s; retry %d/%d in %.2fs",
                    submit_resp.status_code,
                    trace_id,
                    submit_attempt,
                    submit_connect_retries,
                    retry_delay,
                )
                await asyncio.sleep(retry_delay)
                continue
            # 成功或不可重试状态:跳出
            break
        except (
            httpx.ConnectTimeout,
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
            httpx.TransportError,
        ) as exc:
            # 传输层异常:格式化错误详情
            error_detail = format_httpx_error(exc, base_url)
            if submit_attempt < submit_connect_retries:
                # 未耗尽重试:指数退避+抖动
                retry_delay = min(30.0, 1.5**submit_attempt) + random.uniform(0.0, 1.0)
                logger.info(
                    "WARNING: transient submit error to %s: %s; trace_id=%s; retry %d/%d in %.2fs",
                    base_url,
                    error_detail,
                    trace_id,
                    submit_attempt,
                    submit_connect_retries,
                    retry_delay,
                )
                await asyncio.sleep(retry_delay)
                continue
            # 重试耗尽:返回 503 与错误详情
            error_msg = f"Failed to submit job to sandbox API at {base_url}: {error_detail}"
            logger.info("ERROR: %s", error_msg)
            return 503, {
                "error": "submit_failed_after_retries",
                "detail": error_msg,
                "type": type(exc).__name__,
                "trace_id": trace_id,
            }

    # 循环异常退出(理论上不可达):兜底连接失败
    if submit_resp is None:
        return 503, {
            "error": "connection_failed",
            "detail": f"Failed to connect to sandbox API at {base_url}",
            "type": "ConnectError",
            "trace_id": trace_id,
        }

    # 解析提交结果
    submit_data = safe_json(submit_resp)
    logger.info("submit_data: %s", submit_data)
    # 提交仍 >=400:原样返回
    if submit_resp.status_code >= 400:
        return submit_resp.status_code, {
            "error": "submit failed",
            "data": submit_data,
            "trace_id": trace_id,
        }

    # 取作业 ID;缺失视为服务端异常
    job_id = submit_data.get("job_id")
    if not job_id:
        return 500, {
            "error": "no job_id returned",
            "data": submit_data,
            "trace_id": trace_id,
        }

    # 进行中状态集合
    wait_status = ["running", "queued"]
    # 终态集合
    finished_status = ["completed", "failed", "cancelled"]
    # 轮询死线
    deadline = time.monotonic() + wait_timeout
    # 瞬时错误重试计数与上限
    poll_error_retries = 0
    max_poll_error_retries = 20
    # 异常响应体重试计数与上限
    malformed_poll_retries = 0
    max_malformed_poll_retries = 3
    # 主轮询循环
    while time.monotonic() < deadline:
        try:
            # 查询作业状态
            response = await client.get(f"/api/v1/jobs/{job_id}", headers=headers)
            if response.status_code in (502, 503, 504, 429):
                # 瞬时错误:线性+抖动退避,最多 20 次
                if poll_error_retries < max_poll_error_retries:
                    poll_error_retries += 1
                    retry_delay = (
                        min(5.0, 1.0 + 0.25 * poll_error_retries)
                        + random.uniform(0.0, 0.5)
                    )
                    logger.info(
                        "WARNING: transient poll status %s for %s, retry %d/%d in %.2fs",
                        response.status_code,
                        job_id,
                        poll_error_retries,
                        max_poll_error_retries,
                        retry_delay,
                    )
                    await asyncio.sleep(retry_delay)
                    continue
            # 4xx:不可重试,直接返回失败
            if 400 <= response.status_code < 500:
                return response.status_code, {
                    "error": "poll_failed",
                    "job_id": job_id,
                    "trace_id": trace_id,
                    "data": safe_json(response),
                }
        except httpx.TransportError as exc:
            # 传输层异常:同样退避重试
            error_detail = format_httpx_error(exc, f"{base_url}/api/v1/jobs/{job_id}")
            if poll_error_retries < max_poll_error_retries:
                poll_error_retries += 1
                retry_delay = (
                    min(5.0, 1.0 + 0.25 * poll_error_retries)
                    + random.uniform(0.0, 0.5)
                )
                logger.info(
                    "WARNING: transient poll transport error for %s: %s; retry %d/%d in %.2fs",
                    job_id,
                    error_detail,
                    poll_error_retries,
                    max_poll_error_retries,
                    retry_delay,
                )
                await asyncio.sleep(retry_delay)
                continue
            # 重试耗尽:返回 503
            error_msg = f"Failed to poll job status: {error_detail}"
            logger.info("ERROR: %s", error_msg)
            return 503, {
                "error": "poll_failed",
                "detail": error_msg,
                "job_id": job_id,
                "trace_id": trace_id,
                "type": type(exc).__name__,
            }

        # 本轮成功:错误计数清零
        poll_error_retries = 0
        data = safe_json(response)
        status = data.get("status")
        logger.info("job_id:%s, status(%s):%s", job_id, time.monotonic(), status)
        if status in wait_status:
            # 仍在运行:间隔+抖动后继续
            await asyncio.sleep(poll_interval + random.uniform(0.0, 1.0))
            continue
        if status in finished_status:
            # 终态:补上 trace_id 后返回 200
            data.setdefault("trace_id", trace_id)
            return 200, data

        # 未知状态(异常响应体):有限次容忍后放弃
        malformed_poll_retries += 1
        if malformed_poll_retries <= max_malformed_poll_retries:
            retry_delay = (
                min(5.0, 1.0 + malformed_poll_retries)
                + random.uniform(0.0, 0.5)
            )
            logger.info(
                "WARNING: unexpected poll body for %s: status=%s; retry %d/%d in %.2fs",
                job_id,
                status,
                malformed_poll_retries,
                max_malformed_poll_retries,
                retry_delay,
            )
            await asyncio.sleep(retry_delay)
            continue
        # 容忍耗尽:返回 502 与原始数据
        return 502, {
            "error": "unexpected_poll_response",
            "job_id": job_id,
            "trace_id": trace_id,
            "data": data,
        }

    # 超出等待时限
    return 504, {
        "error": "wait_timeout exceeded",
        "job_id": job_id,
        "trace_id": trace_id,
    }

def is_valid_python_script(script):
    """检查字符串是否为合法 Python 脚本(只做编译校验,不执行)。"""
    try:
        # 语法编译通过即合法
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        # 语法错误:非法
        return False

def format_code(code) -> str:
    """用 Black 统一格式化代码;解析失败则原样返回(不抛错)。"""
    try:
        return black.format_str(code, mode=black.FileMode())
    except black.parsing.InvalidInput:  # type: ignore
        # 无法解析的输入保持原样
        return code

def extract_code(text):
    """从 LLM 输出中抽取 Python 代码块,校验语法后合并为一个格式化脚本。"""
    # logger.info(f"raw text: {text}")
    # 收集解析出的代码块
    parsed_codes = []

    # 常规路径:代码在 ``` 或 ```python 围栏内
    matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        code_block = match[1]
        parsed_codes.append(code_block)

    # 回退路径:整段文本就是代码,或围栏缺失
    if len(parsed_codes) == 0:
        matches = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            code_block = matches[0][2]
            parsed_codes.append(code_block)

    # 校验并格式化:只保留语法合法的块
    valid_code_blocks = [
        format_code(c) for c in parsed_codes if is_valid_python_script(c)
    ]
    # logger.info(f"valid_code_blocks: {valid_code_blocks}")
    # 用空行拼接后整体格式化
    return format_code("\n\n".join(valid_code_blocks))


# online_percentile 模式的历史分数缓存:任务路径 -> 分数队列
_RECENT = {}
# 每个任务最多缓存的历史分数条数
_RECENT_MAXLEN = 20

def _stable_sigmoid(x: float) -> float:
    """数值稳定的 sigmoid(按符号分支避免 exp 上溢)。"""
    if x >= 0:
        # 正数分支
        return 1.0 / (1.0 + math.exp(-x))
    # 负数分支
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _signed(score: float, meta: dict) -> float:
    """把分数统一转成"越大越好"的有符号量(越小越好则取负)。"""
    higher = meta.get("higher_is_better", True)
    s = float(score)
    return s if higher else -s


def _finite(x) -> bool:
    """判断值是否为有限数(非 None/非 NaN/非 inf)。"""
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False

def _bounds_signed(meta: dict):
    """计算有符号的 (最优界, 最差界);候选不足两个则返回 (None, None)。"""
    higher = meta.get("higher_is_better", True)

    # 内部工具:转有限有符号值,非法返回 None
    def to_signed(v):
        if not _finite(v):
            return None
        v = float(v)
        return v if higher else -v

    # 四个候选界:理论界与排行榜界
    tmin = to_signed(meta.get("theoretical_min"))
    tmax = to_signed(meta.get("theoretical_max"))
    lmin = to_signed(meta.get("leaderboard_min"))
    lmax = to_signed(meta.get("leaderboard_max"))

    cands = [tmin, tmax, lmin, lmax]
    # 过滤非法候选
    cands = [x for x in cands if x is not None]
    # 至少要两个才能构成区间
    if len(cands) < 2:
        return None, None

    best = max(cands)
    worst = min(cands)

    # 区间过宽(可疑界)时收敛:优先回退排行榜界,否则压缩区间宽度
    MAX_RANGE = 1e6
    if best - worst > MAX_RANGE:
        if lmax is not None and lmin is not None and (lmax - lmin) <= MAX_RANGE:
            best, worst = max(lmax, lmin), min(lmax, lmin)
        else:
            worst = best - MAX_RANGE

    return best, worst

def score2reward(score, metadata, mode="logistic"):
    """
    将 metric 的 score 映射为 reward。
    """
    # 读取任务元数据:方向与四类参考界
    higher_is_better = metadata.get("higher_is_better")
    theoretical_min = metadata.get("theoretical_min")
    theoretical_max = metadata.get("theoretical_max")
    leaderboard_min = metadata.get("leaderboard_min")
    leaderboard_max = metadata.get("leaderboard_max")

    # 忽略 None 的 max(全 None 返回 None)
    def safe_max(*vals):
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None

    # 忽略 None 的 min(全 None 返回 None)
    def safe_min(*vals):
        vals = [v for v in vals if v is not None]
        return min(vals) if vals else None

    # assert mode=="linear_sign" , f"Unsupported mode: {mode}"

    # 模式 0:仅符号线性(越小越好取负),最朴素的奖励整形
    if mode == "linear_sign":
        if higher_is_better is False:
            return -float(score)
        else:
            return float(score)


    # 统一成"越大越好"的有符号分数
    s = _signed(score, metadata)


    # 方案 A：在 MinMax 区间上的幂次 sigmoid
    if mode == "power_sigmoid":
        # 取有符号 (最优界, 最差界)
        best, worst = _bounds_signed(metadata)
        if best is None:
            # 无界可用:退化为裸 sigmoid
            return _stable_sigmoid(s)

        # 归一化到 [0,1](区间宽做下限保护)
        rng = max(best - worst, 1e-9)
        # p in [0,1]
        p = (s - worst) / rng
        p = 0.0 if p < 0 else (1.0 if p > 1 else p)

        # 以 0.5 为中心的温度缩放
        T = 0.50
        x = (p - 0.5) / max(T, 1e-9)
        r = _stable_sigmoid(x)  # (0,1)

        # 幂次 alpha=2 压低中段奖励,拉开头部差距
        alpha = 2.0
        return r ** alpha

    # 方案 B：Tanh 边距整形(按区间中心/半宽归一)
    if mode == "margin_tanh":
        best, worst = _bounds_signed(metadata)
        if best is None:
            # 无界可用:兜底裸 tanh
            return 0.5 + 0.5 * math.tanh(s)  # 兜底

        # 区间中心与半宽
        center = 0.5 * (best + worst)
        scale = max(0.5 * (best - worst), 1e-9)

        # 相对中心的边距(无量纲)
        m = (s - center) / scale

        # 温度缩放后压到 (0,1)
        T = 0.50
        y = math.tanh(m / max(T, 1e-9))  # (-1,1)
        r = 0.5 + 0.5 * y               # (0,1)

        # 幂次 gamma=2 强化区分度
        gamma = 2.0
        return r ** gamma

    # 方案 C：在线分位奖励(以历史分数队列的排名为奖励)
    if mode == "online_percentile":
        # 以任务路径为键隔离历史
        key = str(metadata.get("data_dir", "unknown_task"))
        dq = _RECENT.get(key)
        if dq is None:
            from collections import deque
            dq = deque(maxlen=_RECENT_MAXLEN)
            _RECENT[key] = dq

        # 样本不足 10 个:用裸 sigmoid 起步(排名不可靠)
        if len(dq) < 10:
            dq.append(s)
            return _stable_sigmoid(s)

        # 排名分位:count(历史值 <= s) / 总数
        sorted_vals = sorted(dq)
        # rank = count(vals <= s)
        import bisect
        rank = bisect.bisect_right(sorted_vals, s)
        p = rank / len(sorted_vals)  # [0,1]

        # 当前分数入队(供后续轮次)
        dq.append(s)

        # 幂次 gamma=2 强化头部
        gamma = 2.0
        return p ** gamma

    # 默认 sigmoid 奖励:按方向确定区间端点
    if higher_is_better:
        # best score
        score_range_best_score = (
            theoretical_max
            if theoretical_max is not None
            else leaderboard_max
        )

        # worst score
        # 最差端取各候选下界的较大者,并叠加 -100 兜底
        score_range_worst_score = safe_max(
            leaderboard_min,
            theoretical_min,
            -100
        )

    else:
        # best score
        score_range_best_score = (
            theoretical_min
            if theoretical_min is not None
            else leaderboard_min
        )

        # worst score
        # 反向任务对称处理,叠加 +100 兜底
        score_range_worst_score = safe_min(
            leaderboard_max,
            theoretical_max,
            100
        )

    # 若仍然无法确定区间，直接返回中性奖励
    if score_range_best_score is None or score_range_worst_score is None:
        return 0.5

    # 区间宽度为 0 或 NaN:同样给中性奖励
    score_range = score_range_best_score - score_range_worst_score
    if score_range == 0 or math.isnan(score_range):
        return 0.5

    # 相对位置缩放到 x∈[-2,2] 区间后做稳定 sigmoid
    x = 2.0 * (score - score_range_worst_score) / score_range
    if x >= 0:
        reward = 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        reward = exp_x / (1.0 + exp_x)

    return reward



def test_score2reward(base_dir: str = "./Selected_Dojo"):
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
    # 根目录必须存在
    base_path = Path(base_dir)
    if not base_path.exists():
        raise ValueError(f"Base directory does not exist: {base_dir}")

    # 找出全部任务目录(含 metadata 与排行榜两个文件)
    task_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            metadata_path = item / "info" / "task_metadata.json"
            leaderboard_path = item / "info" / "public_leaderboard.csv"
            if metadata_path.exists() and leaderboard_path.exists():
                task_dirs.append(item)

    # 没有可用任务:提示后返回
    if not task_dirs:
        print(f"No valid tasks found in {base_dir}")
        return

    print(f"Found {len(task_dirs)} tasks to test\n")
    print("=" * 100)

    # 逐任务(按名称排序)打印分位-奖励对照
    for task_dir in sorted(task_dirs):
        task_name = task_dir.name
        metadata_path = task_dir / "info" / "task_metadata.json"
        leaderboard_path = task_dir / "info" / "public_leaderboard.csv"

        # 读取元数据
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Error loading metadata for {task_name}: {e}")
            continue

        # 读取排行榜分数
        try:
            df = pd.read_csv(leaderboard_path)
            if "score" not in df.columns:
                print(f"Warning: {task_name} leaderboard does not have 'score' column")
                continue
            scores = df["score"].dropna().astype(float).tolist()
            if not scores:
                print(f"Warning: {task_name} leaderboard has no valid scores")
                continue
        except Exception as e:
            print(f"Error loading leaderboard for {task_name}: {e}")
            continue

        # 按方向排序分数
        higher_is_better = metadata.get("higher_is_better", True)
        if higher_is_better:
            scores_sorted = sorted(scores, reverse=True)  # Best to worst
        else:
            scores_sorted = sorted(scores)  # Best (lowest) to worst (highest)

        total_entries = len(scores_sorted)
        if total_entries == 0:
            continue

        # 计算各分位的索引
        # 用 floor 取整,保证取到不高于该分位的记录
        def percentile_index(pct: float) -> int:
            """Calculate index for a given percentile (0.0 to 1.0)."""
            idx = int((total_entries - 1) * pct)
            return max(0, min(idx, total_entries - 1))

        # 六个采样位置:第一名/20%/40%/60%/80%/最后一名
        indices = [
            0,  # Rank 1 (best)
            percentile_index(0.2),  # 20%
            percentile_index(0.4),  # 40%
            percentile_index(0.6),  # 60%
            percentile_index(0.8),  # 80%
            total_entries - 1,  # Last (worst)
        ]

        # 取出这些位置上的分数与名次
        selected_scores = [scores_sorted[i] for i in indices]
        selected_ranks = [i + 1 for i in indices]  # Rank is 1-indexed

        # 计算对应奖励
        rewards = []
        for score in selected_scores:
            try:
                reward = score2reward(score, metadata)
                rewards.append(reward)
            except Exception as e:
                # 单点失败记 NaN,不中断整体
                print(f"  Warning: Failed to calculate reward for score {score:.6f}: {e}")
                rewards.append(float('nan'))

        # 展示任务概况
        print(f"\nTask: {task_name}")
        print(f"  UUID: {metadata.get('uuid', 'N/A')}")
        print(f"  Higher is better: {higher_is_better}")
        print(f"  Theoretical min: {metadata.get('theoretical_min', 'N/A')}")
        print(f"  Theoretical max: {metadata.get('theoretical_max', 'N/A')}")
        print(f"  Total entries: {total_entries}")
        print(f"  Score range: [{min(scores):.6f}, {max(scores):.6f}]")
        # 表头与分隔线
        print(f"\n  {'Position':<12} {'Rank':<8} {'Score':<15} {'Reward':<10}")
        print(f"  {'-' * 12} {'-' * 8} {'-' * 15} {'-' * 10}")

        # 逐行打印分位-名次-分数-奖励
        labels = ["1st (best)", "20%", "40%", "60%", "80%", "Last (worst)"]
        for label, rank, score, reward in zip(labels, selected_ranks, selected_scores, rewards):
            if math.isnan(reward):
                reward_str = "N/A"
            else:
                reward_str = f"{reward:.6f}"
            print(f"  {label:<12} {rank:<8} {score:<15.6f} {reward_str:<10}")

        print("=" * 100)

def get_clear_log(run_log: str | None) -> str:
    """
    从 run_log 中提取核心输出，基于新的包裹标记：
      - --- OUTPUT START --- ~~~ --- OUTPUT END ---
      - --- SANDBOX STDOUT START --- ~~~ --- SANDBOX STDOUT END ---
    并删除所有 [HB] ... [HB] 包裹的日志内容。
    """
    if not run_log:
        return ""

    markers = [
        ("--- OUTPUT START ---", "--- OUTPUT END ---"),
        ("--- SANDBOX STDOUT START ---", "--- SANDBOX STDOUT END ---"),
    ]

    hb_marker = "[HB]"
    hb_len = len(hb_marker)

    # 收集到的有效输出小节
    sections: list[str] = []

    # 逐标记对扫描全部出现位置
    for start_marker, end_marker in markers:
        search_pos = 0
        while True:
            # 找到下一个起始标记
            start_idx = run_log.find(start_marker, search_pos)
            if start_idx == -1:
                break

            # 内容起点跳过标记本身(以及紧随的换行)
            content_start = start_idx + len(start_marker)
            if content_start < len(run_log) and run_log[content_start] == "\n":
                content_start += 1

            # 找结束标记;缺失则取到日志末尾
            end_idx = run_log.find(end_marker, content_start)
            if end_idx == -1:
                content = run_log[content_start:].strip()
            else:
                content = run_log[content_start:end_idx].strip()

            # === 删除所有 [HB] ... [HB] 块（用 find，不用 re） ===
            if content:
                parts: list[str] = []
                pos = 0
                while True:
                    s = content.find(hb_marker, pos)
                    if s == -1:
                        # 无更多 [HB]:收尾剩余部分
                        parts.append(content[pos:])
                        break
                    # 保留 [HB] 之前的内容
                    parts.append(content[pos:s])
                    e = content.find(hb_marker, s + hb_len)
                    if e == -1:
                        # 未闭合：从 s 到末尾丢弃
                        break
                    # 跳过成对 [HB] 包裹的内容
                    pos = e + hb_len

                content = "".join(parts).strip()

            # 非空小节才收录
            if content:
                sections.append(f"\n{content}")

            # 无结束标记:本标记对扫描完毕
            if end_idx == -1:
                break
            # 从结束标记之后继续找下一对
            search_pos = end_idx + len(end_marker)

    # 小节间以空行拼接
    return "\n\n".join(sections)


def parse_final_validation_score(run_log: str | None) -> float | None:
    """从运行日志解析最后一次"Final Validation Score:"的浮点值(无则 None)。"""
    # 空日志直接返回
    if not run_log:
        return None

    # 匹配任意合法浮点格式(含符号/科学计数)
    matches = re.findall(
        r"Final Validation Score:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
        run_log,
    )
    if not matches:
        return None

    try:
        # 取最后一次出现(最终版本优先)
        return float(matches[-1])
    except (TypeError, ValueError):
        return None

def format_sandbox_feedback(status_code: int, payload: dict) -> str:
    """Format sandbox execution results into feedback message"""
    if status_code == 200:
        # 成功:拆出状态/清洗日志/结果/分数
        result = payload.get("result") or {}
        status = payload.get("status", "unknown")
        run_log = get_clear_log(result.get("run_log"))
        run_result = result.get("result", "")
        score = result.get("score")

        # 组装 markdown 反馈:标题与状态
        feedback = f"## Execution Result\n"
        feedback += f"**Status**: {status}\n\n"

        # 有分数才写
        if score is not None:
            feedback += f"**Score**: {score}\n\n"

        # 有结果摘要才写
        if run_result:
            feedback += f"**Result**: {run_result}\n\n"

        if run_log:
            # Truncate log if too long
            # 日志过长时保留头尾各 1000 字符
            if len(run_log) > 2000:
                run_log = run_log[:1000] + "\n... (truncated) ...\n" + run_log[-1000:]
            feedback += f"**Execution Log**:\n```\n{run_log}\n```\n\n"

    elif status_code == 503:
        # 连接错误:类型+详情,提示重试
        error_type = payload.get("type", "unknown")
        error_detail = payload.get("detail", "Connection failed")
        feedback = f"## Connection Error\n"
        feedback += f"**Type**: {error_type}\n"
        feedback += f"**Detail**: {error_detail}\n\n"
        feedback += "Please check your code and try again."
    else:
        # 其他执行错误:错误码+结构化详情
        error_msg = payload.get("error", "unknown error")
        detail = payload.get("detail", {})
        feedback = f"## Execution Error\n"
        feedback += f"**Error**: {error_msg}\n"
        # dict 型详情才展开状态/消息
        if isinstance(detail, dict):
            if detail.get("status"):
                feedback += f"**Status**: {detail.get('status')}\n"
            if detail.get("message"):
                feedback += f"**Message**: {detail.get('message')}\n"
        feedback += "\nPlease fix the errors and try again."

    return feedback

if __name__ == "__main__":
    # Test score2reward for all tasks
    # 直接运行时对全部任务做分位-奖励对照测试
    test_score2reward()
