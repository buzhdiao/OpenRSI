"""
Generator Service - Handles code generation using LLM.

This service is responsible for generating code solutions for ML tasks.
It operates independently from the evaluation service and communicates
through async queues managed by the scheduler.
"""
# 生成服务模块：调用 LLM 为机器学习任务生成代码方案，与评估服务解耦，
# 由调度器通过异步队列协同；含重试、token 用量与成本统计、产物落盘。

from __future__ import annotations

# 导入 asyncio：信号量并发控制、重试退避睡眠、批量并发
import asyncio
# 导入 json：序列化响应产物
import json
# 导入 logging：模块日志
import logging
# 导入 dataclass/field：声明请求/结果数据结构
from dataclasses import dataclass, field
# 导入 datetime：结果时间戳
from datetime import datetime
# 导入 Path：产物目录处理
from pathlib import Path
# 导入类型注解 Any
from typing import Any

# 导入 HTTPStatusError：HTTP 状态异常（用于网关错误重试）
from httpx import HTTPStatusError
# 导入 LiteLLM 的模型响应类型与单 token 成本计算
from litellm import ModelResponse, cost_per_token
# 导入 LiteLLM 各类异常：连接失败/请求错误/限流/服务不可用
from litellm.exceptions import (
    APIConnectionError,
    BadRequestError,
    RateLimitError,
    ServiceUnavailableError,
)
# 导入 LiteLLM Router：多模型路由调用
from litellm.router import Router
# 导入 OmegaConf：解析 DictConfig 配置对象
from omegaconf import DictConfig, OmegaConf

# 导入代码抽取工具：从模型回复中提取合法 Python 代码
from tts_search.reward_func_utils import extract_code

# 兼容导入 MidStreamFallbackError：旧版 LiteLLM 无此异常时用本地替身
try:
    from litellm.exceptions import MidStreamFallbackError
except ImportError:

    class MidStreamFallbackError(Exception):
        """Compatibility exception for LiteLLM versions without this type."""

        # 兼容构造器：保留 provider/model/response 上下文字段
        def __init__(
            self,
            message: str,
            *,
            llm_provider: str | None = None,
            model: str | None = None,
            response: Any = None,
        ) -> None:
            super().__init__(message)
            self.llm_provider = llm_provider
            self.model = model
            self.response = response

# 模块级 logger 配置：INFO 级别 + 控制台输出
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


@dataclass
class GenerationRequest:
    """Request for code generation."""

    # 字段说明：request_id=请求唯一标识；task_id/task_name=任务标识；
    # messages=对话消息列表；model_name=使用的模型；data_dir=任务数据目录；
    # metadata=任务元信息；step_index=当前搜索步数；output_dir=产物目录（可选）；
    # search_metadata=搜索策略附加信息（可选）
    request_id: str
    task_id: str
    task_name: str
    messages: list[dict[str, str]]
    model_name: str
    data_dir: str
    metadata: dict[str, Any]
    step_index: int
    output_dir: Path | None = None
    search_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Result from code generation."""

    # 字段说明：request_id/task_id/task_name=请求与任务标识；
    # code=抽取出的可执行代码；raw_text=模型原始回复；reasoning_content=思考链内容；
    # prompt_tokens/completion_tokens/total_tokens=token 用量；cost=本次调用成本（美元）；
    # model_name=模型名；messages=对话消息；metadata=任务元信息；step_index=搜索步数；
    # data_dir=数据目录；output_dir=产物目录；search_metadata=搜索附加信息；
    # error=错误信息（None 表示成功）；timestamp=结果创建时间
    request_id: str
    task_id: str
    task_name: str
    code: str
    raw_text: str
    reasoning_content: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    model_name: str
    messages: list[dict[str, str]]
    metadata: dict[str, Any]
    step_index: int
    data_dir: str
    output_dir: Path | None = None
    search_metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def success(self) -> bool:
        # 无错误且抽到了非空代码才算生成成功
        return self.error is None and self.code != ""


class GeneratorService:
    """
    Service for code generation using LLM.

    This service handles:
    - Async code generation with retry logic
    - Token usage tracking
    - Cost calculation
    - Output artifact persistence
    """

    def __init__(
        self,
        router: Router,
        concurrency: int = 64,
        max_retries: int = 10,
        model_list: list[Any] | None = None,
    ):
        """
        Initialize the generator service.

        Args:
            router: LiteLLM router for model requests
            concurrency: Maximum concurrent generation requests
            max_retries: Maximum retries for failed requests
        """
        # 保存 LiteLLM 路由器（负责实际模型调用）
        self._router = router
        # 用信号量限制最大并发生成数
        self._semaphore = asyncio.Semaphore(concurrency)
        # 单请求最大重试次数
        self._max_retries = max_retries
        # 保存模型列表配置（用于查询流式开关等参数）
        self._model_list = model_list or []
        # 请求队列（预留的内部队列）
        self._request_queue: asyncio.Queue[GenerationRequest] = asyncio.Queue()
        # 结果队列（预留的内部队列）
        self._result_queue: asyncio.Queue[GenerationResult] = asyncio.Queue()
        # 服务运行标志
        self._running = False
        # 工作协程列表
        self._workers: list[asyncio.Task] = []

        # Statistics
        # 运行统计：请求数、成功/失败数、token 用量与成本累计
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._total_tokens = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cost = 0.0
        self._total_prompt_cost = 0.0
        self._total_response_cost = 0.0

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate code for a single request.

        Args:
            request: Generation request with task details

        Returns:
            GenerationResult with generated code and metadata
        """
        # 初始化：原始回复/思考内容/重试计数/响应对象/用量/错误信息
        raw_text = None
        reasoning_content: str | None = None
        retry_count = 0
        response: ModelResponse | None = None
        usage: dict[str, Any] = {}
        error_msg: str | None = None
        # 读取该模型的流式配置：是否强制流式、是否请求用量统计
        force_stream, include_usage = self._get_streaming_config(request.model_name)

        # 循环重试直至同时拿到回复文本与思考内容（或重试耗尽）
        while raw_text is None or reasoning_content is None:
            # 重试额度用尽：返回空生成结果
            if retry_count >= self._max_retries:
                logger.error(
                    "LLM retries exhausted; returning empty generation. task_id=%s task_name=%s step=%d model=%s",
                    request.task_id,
                    request.task_name,
                    request.step_index,
                    request.model_name,
                )
                raw_text = ""
                reasoning_content = None
                usage = {}
                error_msg = (
                    f"LLM retries exhausted after {self._max_retries} attempts"
                )
                break

            # 每轮重试前清空本轮状态
            reasoning_content = None
            usage = {}
            response = None
            try:
                # 信号量限流后发起模型调用
                async with self._semaphore:
                    if force_stream:
                        # 流式模式：分块收集回复与思考内容
                        logger.info("Using streaming mode")
                        raw_text, reasoning_content, usage = (
                            await self._acompletion_stream_collect(
                                model_name=request.model_name,
                                messages=request.messages,
                                include_usage=include_usage,
                            )
                        )
                    else:
                        # 非流式模式：一次性等待完整响应
                        logger.info("Using non-streaming mode")
                        response = await self._router.acompletion(
                            model=request.model_name,
                            messages=request.messages,  # type: ignore
                        )  # type: ignore
            except RateLimitError as exc:
                # 限流：重试计数 +1 并按次数线性退避
                retry_count += 1
                logger.warning(
                    "Rate limit hit; retrying. retry=%d task_id=%s task_name=%s step=%d model=%s error=%s",
                    retry_count,
                    request.task_id,
                    request.task_name,
                    request.step_index,
                    request.model_name,
                    exc,
                )
                await asyncio.sleep(1.0 * retry_count)  # Exponential backoff
                continue
            except (ServiceUnavailableError, APIConnectionError) as exc:
                # 服务不可用/连接失败：同样退避重试
                retry_count += 1
                logger.warning(
                    "Provider connection/service error; retrying. retry=%d task_id=%s task_name=%s step=%d model=%s error=%s",
                    retry_count,
                    request.task_id,
                    request.task_name,
                    request.step_index,
                    request.model_name,
                    exc,
                )
                await asyncio.sleep(1.0 * retry_count)
                continue
            except MidStreamFallbackError as exc:
                # 流式传输中途被打断：记录现场后重试
                retry_count += 1
                logger.warning(
                    "Streaming interrupted (mid-stream). retry=%d task_id=%s task_name=%s step=%d model=%s error=%s",
                    retry_count,
                    request.task_id,
                    request.task_name,
                    request.step_index,
                    request.model_name,
                    exc,
                )
                # 指定了输出目录时保存出错时的 prompt 与错误现场
                if request.output_dir:
                    step_dir = request.output_dir / f"step_{request.step_index}"
                    step_dir.mkdir(parents=True, exist_ok=True)
                    user_content = (
                        str(request.messages[-1].get("content", ""))
                        if request.messages
                        else ""
                    )
                    (step_dir / "user_prompt.md").write_text(
                        user_content, encoding="utf-8"
                    )
                    (step_dir / "stream_error.txt").write_text(
                        str(exc), encoding="utf-8"
                    )
                await asyncio.sleep(1.0 * retry_count)
                continue
            except HTTPStatusError as exc:
                # HTTP 状态错误：网关类错误（502/503/504）退避重试，其他直接抛出
                retry_count += 1
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in {502, 503, 504}:
                    logger.warning(
                        "Upstream gateway error; retrying. retry=%d task_id=%s task_name=%s step=%d model=%s status=%s",
                        retry_count,
                        request.task_id,
                        request.task_name,
                        request.step_index,
                        request.model_name,
                        status,
                    )
                    await asyncio.sleep(1.0 * retry_count)
                    continue
                raise
            except BadRequestError as exc:
                # 请求错误：非流式下遇到"切换流式/生成超时"提示时改用流式重试
                msg = str(exc)
                if (not force_stream) and (
                    ("Please switch to streaming mode" in msg)
                    or ("generation timed out" in msg)
                ):
                    retry_count += 1
                    logger.warning(
                        "Non-stream timed out; switching to streaming. retry=%d task_id=%s task_name=%s step=%d model=%s error=%s",
                        retry_count,
                        request.task_id,
                        request.task_name,
                        request.step_index,
                        request.model_name,
                        exc,
                    )
                    # 立即改用流式收集一次
                    async with self._semaphore:
                        raw_text, reasoning_content, usage = (
                            await self._acompletion_stream_collect(
                                model_name=request.model_name,
                                messages=request.messages,
                                include_usage=include_usage,
                            )
                        )
                    response = None
                    # 后续轮次固定走流式
                    force_stream = True
                else:
                    raise
            except Exception as exc:
                # 兜底：其他未知异常统一退避 1 秒重试
                retry_count += 1
                logger.warning(
                    "Generation error; retrying. retry=%d task_id=%s error=%s",
                    retry_count,
                    request.task_id,
                    exc,
                )
                await asyncio.sleep(1.0)
                continue

            if not force_stream:
                # 非流式：从响应对象解析回复与思考内容
                assert response is not None
                message = response.choices[0].message  # type: ignore
                content = message.content  # type: ignore

                if isinstance(content, list):
                    # Anthropic-style content blocks: split text and thinking.
                    # Anthropic 风格内容块：拆分文本与思考块
                    logger.debug("Anthropic-style content blocks")
                    text_parts: list[str] = []
                    thinking_parts: list[str] = []
                    for block in content:
                        # thinking 块归入思考内容，text 块归入回复文本
                        if block.type == "thinking":
                            thinking_parts.append(block.thinking)
                        elif block.type == "text":
                            text_parts.append(block.text)
                    # 有文本块才设置 raw_text，否则保持 None 以触发重试
                    raw_text = "".join(text_parts) if text_parts else None
                    reasoning_content = (
                        "".join(thinking_parts) if thinking_parts else None
                    )
                    logger.debug(
                        "Anthropic-style reasoning content: %s",
                        (reasoning_content or "")[:100],
                    )
                else:
                    # 普通 string 内容：直接作为回复文本
                    raw_text = str(content) if content is not None else None

                if reasoning_content is None:
                    # OpenAI/LiteLLM reasoning content may live in a separate field.
                    logger.debug("OpenAI/LiteLLM content blocks")
                    # OpenAI/LiteLLM 风格：思考内容可能存放在独立字段

                    # Resolve reasoning fields in provider-neutral priority order.
                    # 按 provider 无关的优先级顺序尝试常见思考字段
                    for attr in ("reasoning_content", "reasoning", "thinking"):
                        if hasattr(message, attr):
                            logger.debug(
                                "Found reasoning-like attribute: %s", attr
                            )
                            reasoning_content = getattr(message, attr)
                            break

                    # 回退一：reasoning_details 字段（取首个元素的 text）
                    if reasoning_content is None and hasattr(
                        message, "reasoning_details"
                    ):
                        logger.debug("Found reasoning_details attribute")
                        reasoning_content = message.reasoning_details[0]["text"]  # type: ignore

                    # 回退二：provider_specific_fields 中的 reasoning_details
                    if reasoning_content is None and hasattr(
                        message, "provider_specific_fields"
                    ):
                        logger.debug("Found provider_specific_fields attribute")
                        psf = message.provider_specific_fields
                        if isinstance(psf, dict) and "reasoning_details" in psf:
                            logger.debug(
                                "Found reasoning_details in provider_specific_fields"
                            )
                            reasoning_details = psf["reasoning_details"]
                            if (
                                isinstance(reasoning_details, list)
                                and len(reasoning_details) > 0
                            ):
                                # Extract text from all reasoning blocks and concatenate
                                # 拼接所有思考块中的 text
                                reasoning_texts = []
                                for detail in reasoning_details:
                                    if isinstance(detail, dict) and "text" in detail:
                                        reasoning_texts.append(detail["text"])
                                if reasoning_texts:
                                    reasoning_content = "".join(reasoning_texts)


                # 从响应对象读取 token 用量
                usage = response.get("usage", {}) or {}

            if raw_text is None:
                # 未拿到回复文本：计数重试后重来
                retry_count += 1
                logger.warning(
                    "Empty LLM response; retrying. retry=%d task_id=%s task_name=%s step=%d model=%s",
                    retry_count,
                    request.task_id,
                    request.task_name,
                    request.step_index,
                    request.model_name,
                )
                await asyncio.sleep(1.0 * retry_count)
                continue
            if reasoning_content is None:
                # 未拿到思考内容：同样计数重试后重来
                retry_count += 1
                logger.warning(
                    "Reasoning content is None; retrying. retry=%d task_id=%s task_name=%s step=%d model=%s response=%s",
                    retry_count,
                    request.task_id,
                    request.task_name,
                    request.step_index,
                    request.model_name,
                    str(response),
                )
                await asyncio.sleep(1.0 * retry_count)
                continue
        # if raw_text is None:
        #     error_msg = f"Failed to generate after {retry_count} retries"
        #     logger.error(
        #         "Generation failed: task_id=%s task_name=%s error=%s",
        #         request.task_id,
        #         request.task_name,
        #         error_msg,
        #     )
        #     return GenerationResult(
        #         request_id=request.request_id,
        #         task_id=request.task_id,
        #         task_name=request.task_name,
        #         code="",
        #         raw_text="",
        #         reasoning_content=None,
        #         prompt_tokens=0,
        #         completion_tokens=0,
        #         total_tokens=0,
        #         cost=0.0,
        #         model_name=request.model_name,
        #         messages=request.messages,
        #         metadata=request.metadata,
        #         step_index=request.step_index,
        #         data_dir=request.data_dir,
        #         output_dir=request.output_dir,
        #         error=error_msg,
        #     )

        # Extract code from response
        # 从回复文本中抽取合法 Python 代码
        code = extract_code(raw_text)

        # Calculate token usage and cost
        # 非流式路径下再补读一次用量（流式路径已在上文收集）
        if response is not None:
            usage = response.get("usage", {}) or {}
        # 各类 token 计数（总数缺失时用 prompt+completion 兜底）
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))

        # Calculate costs
        # 成本初始化：prompt 成本与补全成本
        prompt_cost_usd = 0.0
        response_cost_usd = 0.0

        # 读取 LiteLLM 隐藏参数（可能内置了成本信息）
        hidden_params = getattr(response, "_hidden_params", None)

        # Try to get prompt_cost and response_cost from hidden_params
        # 优先使用 hidden_params 中的成本
        has_prompt_cost = False
        has_response_cost = False
        if isinstance(hidden_params, dict):
            if "response_cost" in hidden_params:
                logger.debug("hidden params has response cost")
                response_cost_usd = float(hidden_params["response_cost"])
                has_response_cost = True

        # Calculate missing costs using cost_per_token
        # 缺失的成本用 cost_per_token 按单价计算
        if not has_prompt_cost or not has_response_cost:
            try:
                calculated_prompt_cost, calculated_completion_cost = cost_per_token(
                    model=request.model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                if not has_prompt_cost:
                    prompt_cost_usd = float(calculated_prompt_cost)
                if not has_response_cost:
                    response_cost_usd = float(calculated_completion_cost)
            except Exception as exc:
                # 计算失败仅告警，缺失成本按 0 处理
                logger.warning(
                    "Could not calculate cost for model=%s: %s. Using 0.0 for missing costs.",
                    request.model_name,
                    exc,
                )
        logger.debug(
            f"prompt_cost_usd: {prompt_cost_usd}, response_cost_usd: {response_cost_usd}"
        )

        # Calculate total cost (prompt + response)
        # 总成本 = prompt 成本 + 补全成本
        total_cost_usd = prompt_cost_usd + response_cost_usd

        # Save artifacts if output directory is specified
        # 指定了输出目录时保存生成产物
        if request.output_dir:
            # 按步数建立子目录
            step_dir = request.output_dir / f"step_{request.step_index}"
            step_dir.mkdir(parents=True, exist_ok=True)
            # 取最后一条 user 消息作为 prompt 产物
            user_content = (
                str(request.messages[-1].get("content", "")) if request.messages else ""
            )
            # 组装响应负载（原始文本/思考内容/用量）
            response_payload: dict[str, Any] = {
                "raw_text": raw_text,
                "reasoning_content": reasoning_content,
                "usage": usage,
            }
            if response is not None:
                # 尽力序列化完整响应对象（兼容多种 API 风格）
                if hasattr(response, "model_dump"):
                    response_payload["response"] = response.model_dump()
                elif hasattr(response, "dict"):
                    response_payload["response"] = response.dict()
                else:
                    try:
                        response_payload["response"] = dict(response)
                    except Exception:
                        response_payload["response"] = str(response)
            logger.debug(f"response_payload: {str(response_payload)[:50]}")
            # 保存 prompt 与原始回复
            (step_dir / "user_prompt.md").write_text(user_content, encoding="utf-8")
            (step_dir / "response.md").write_text(raw_text, encoding="utf-8")
            try:
                # 序列化响应负载为 JSON（datetime 等对象自动转 ISO 字符串）
                response_json = json.dumps(
                    response_payload,
                    indent=2,
                    default=lambda o: o.isoformat() if hasattr(o, "isoformat") else str(o),
                )
            except Exception as e:
                # 序列化失败：退化为最小负载并记录错误
                logger.warning(
                    "Failed to serialize response_payload to JSON: %s. Writing minimal payload.",
                    e,
                )
                response_json = json.dumps(
                    {
                        "raw_text": raw_text,
                        "reasoning_content": reasoning_content,
                        "usage": {k: str(v) for k, v in (usage or {}).items()},
                        "response_serialization_error": str(e),
                    },
                    indent=2,
                )
            # 保存响应 JSON、可执行代码，以及思考内容（若有）
            (step_dir / "response.json").write_text(response_json, encoding="utf-8")
            (step_dir / "valid_code.py").write_text(code, encoding="utf-8")
            if reasoning_content:
                (step_dir / "reasoning.md").write_text(
                    str(reasoning_content), encoding="utf-8"
                )

        # Update statistics
        # 更新统计：请求总数与成功/失败判定
        self._total_requests += 1
        if error_msg is None and code != "":
            self._successful_requests += 1
        else:
            self._failed_requests += 1
        # 累计 token 用量与成本
        self._total_tokens += total_tokens
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        self._total_cost += total_cost_usd
        self._total_prompt_cost += prompt_cost_usd
        self._total_response_cost += response_cost_usd

        # 组装并返回生成结果
        return GenerationResult(
            request_id=request.request_id,
            task_id=request.task_id,
            task_name=request.task_name,
            code=code,
            raw_text=raw_text,
            reasoning_content=reasoning_content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=total_cost_usd,
            model_name=request.model_name,
            messages=request.messages,
            metadata=request.metadata,
            step_index=request.step_index,
            data_dir=request.data_dir,
            output_dir=request.output_dir,
            search_metadata=dict(request.search_metadata or {}),
            error=error_msg,
        )

    def _to_plain_dict(self, value: Any) -> dict[str, Any]:
        # 把配置对象转成普通 dict：None → 空字典；DictConfig → 递归解析；dict → 浅拷贝
        if value is None:
            return {}
        if isinstance(value, DictConfig):
            return OmegaConf.to_container(value, resolve=True)  # type: ignore
        if isinstance(value, dict):
            return dict(value)
        return {}

    def _get_model_litellm_params(self, model_name: str) -> dict[str, Any]:
        # 在模型列表中查找指定模型的 litellm_params 配置
        for model_entry in self._model_list:
            entry = self._to_plain_dict(model_entry)
            entry_name = str(entry.get("model_name", ""))
            if entry_name == str(model_name):
                return self._to_plain_dict(entry.get("litellm_params"))
        return {}

    def _get_streaming_config(self, model_name: str) -> tuple[bool, bool]:
        # logger.info(f"model_list: {self._model_list}")
        # 读取该模型的流式相关参数
        params = self._get_model_litellm_params(model_name)
        # logger.info(f"params: {params}")
        # 是否启用流式（默认关闭）
        use_stream = bool(params.get("stream", False))
        # 是否请求用量统计（默认开启）
        include_usage = bool(params.get("include_usage", True))
        return use_stream, include_usage

    async def _acompletion_stream_collect(
        self,
        *,
        model_name: str,
        messages: list[dict[str, str]],
        include_usage: bool = True,
    ) -> tuple[str | None, str | None, dict[str, Any]]:
        # 流式调用并分块收集回复与思考内容，返回 (raw_text, reasoning_content, usage)
        # 需要用量统计时设置 stream_options
        stream_options = {"include_usage": True} if include_usage else None
        # 回复文本与思考内容的分块缓存
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: dict[str, Any] = {}

        try:
            # 发起流式补全请求
            stream = await self._router.acompletion(
                model=model_name,
                messages=messages,  # type: ignore
                stream=True,
                stream_options=stream_options,
            )  # type: ignore

            # 逐块消费流
            async for chunk in stream:
                # 用量信息通常出现在单独的 dict 块中
                if isinstance(chunk, dict) and chunk.get("usage"):
                    usage = chunk.get("usage") or {}

                try:
                    choice0 = chunk["choices"][0]
                except Exception:
                    # 无 choices 的块（如纯 usage 块）直接跳过
                    continue

                # 取增量 delta 并累积文本内容
                delta = choice0.get("delta") or {}
                content = delta.get("content")
                if content:
                    text_parts.append(str(content))

                # 累积思考内容（兼容多种字段名）
                rc = (
                    delta.get("reasoning_content")
                    or delta.get("reasoning")
                    or delta.get("thinking")
                )
                if rc:
                    reasoning_parts.append(str(rc))

                # 兼容某些 provider 把完整 message 放在 delta 里的情况
                msg = delta.get("message") if isinstance(delta, dict) else None
                if isinstance(msg, dict):
                    rc2 = msg.get("reasoning_content") or msg.get("reasoning")
                    if rc2:
                        reasoning_parts.append(str(rc2))

        except MidStreamFallbackError as exc:
            # 流中途回退：带上已收集长度信息重新抛出，便于上层诊断
            partial_text = "".join(text_parts) if text_parts else None
            raise MidStreamFallbackError(
                message=f"{exc}\n[partial_text_len={len(partial_text or '')}]",
                llm_provider=getattr(exc, "llm_provider", None),
                model=getattr(exc, "model", None),
                response=getattr(exc, "response", None),
            ) from exc

        # 拼接最终文本（空集合时返回 None 以触发上层重试）
        raw_text = "".join(text_parts) if text_parts else None
        reasoning_content = "".join(reasoning_parts) if reasoning_parts else None
        logger.info(
            "stream collected: raw_text_len=%d reasoning_content_len=%d raw_head=%r reasoning_head=%r",
            len(raw_text or ""),
            len(reasoning_content or ""),
            (raw_text or "")[:100],
            (reasoning_content or "")[:100],
        )
        return raw_text, reasoning_content, usage

    async def generate_batch(
        self, requests: list[GenerationRequest]
    ) -> list[GenerationResult]:
        """
        Generate code for multiple requests concurrently.

        Args:
            requests: List of generation requests

        Returns:
            List of generation results
        """
        # 为每个请求创建协程并并发执行
        tasks = [self.generate(req) for req in requests]
        return await asyncio.gather(*tasks)

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        # Calculate per-token costs
        # 计算单位 token 成本（总量为 0 时按 0 处理）
        total_cost_per_token = (
            self._total_cost / self._total_tokens if self._total_tokens > 0 else 0.0
        )
        total_prompt_cost_per_token = (
            self._total_prompt_cost / self._total_prompt_tokens
            if self._total_prompt_tokens > 0
            else 0.0
        )
        total_response_cost_per_token = (
            self._total_response_cost / self._total_completion_tokens
            if self._total_completion_tokens > 0
            else 0.0
        )

        # 返回统计字典：请求数/成功失败/token 用量/成本/单位成本
        return {
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "total_tokens": self._total_tokens,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_cost": self._total_cost,
            "total_prompt_cost": self._total_prompt_cost,
            "total_response_cost": self._total_response_cost,
            "total_cost_per_token": total_cost_per_token,
            "total_prompt_cost_per_token": total_prompt_cost_per_token,
            "total_response_cost_per_token": total_response_cost_per_token,
        }
