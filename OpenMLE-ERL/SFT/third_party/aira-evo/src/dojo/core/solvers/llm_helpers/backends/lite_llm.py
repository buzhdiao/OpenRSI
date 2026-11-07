# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 json 模块，用于序列化/反序列化函数调用参数等 JSON 数据
import json
# 导入 logging 模块，用于记录后端运行日志
import logging
# 导入 os 模块，用于读取环境变量（API Key、开关、超时等配置）
import os
# 导入 dataclass 装饰器，用于定义函数规格（FunctionSpec）数据类
from dataclasses import dataclass
# 导入类型注解工具，用于函数签名标注
from typing import Any, Dict, List, Optional, Tuple, Union

# 导入 jsonschema，用于校验函数参数的 JSON Schema 合法性
import jsonschema
# 导入 litellm 统一网关库，可通过同一接口调用多家 LLM 提供商
import litellm
# 导入 time 模块，用于统计请求耗时（延迟测量）
import time

# 导入 DataClassJsonMixin，让数据类支持与 JSON 字符串互转
from dataclasses_json import DataClassJsonMixin
# 从 litellm 导入 completion 调用函数，作为实际发起 LLM 请求的入口
from litellm import completion as completion_fn

# 指定 Azure OpenAI 的 API 版本（litellm 走 azure 后端时需要）
litellm.api_version = "2024-12-01-preview"
# 关闭 litellm 的冗长调试输出，避免刷屏
litellm.set_verbose = False

# 默认请求重试次数
NUM_RETRIES = 10
# 默认请求超时时间（秒）
TIMEOUT = 1500
# 环境变量名：是否启用流式输出
STREAM_ENV = "AIRA_LITELLM_STREAM"
# 环境变量名：请求超时时间
TIMEOUT_ENV = "AIRA_LITELLM_TIMEOUT"
# 环境变量名：请求重试次数
NUM_RETRIES_ENV = "AIRA_LITELLM_NUM_RETRIES"


# Configure logging
# 获取名为 "Backend" 的日志器，用于本后端模块统一输出日志
logger = logging.getLogger("Backend")
# 设置日志级别为 INFO，只输出 INFO 及以上级别的消息
logger.setLevel(logging.INFO)
# 创建一个输出到标准错误流的日志处理器
handler = logging.StreamHandler()
# 定义日志格式：[级别] 名称: 消息
formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
# 将格式器绑定到处理器上
handler.setFormatter(formatter)
# 把处理器挂到日志器上，完成日志配置
logger.addHandler(handler)


def _env_bool(name: str, default: bool) -> bool:
    # 从环境变量读取布尔配置；未设置时返回默认值
    value = os.getenv(name)
    # 环境变量不存在则使用调用方给定的默认值
    if value is None:
        return default
    # 只要值（去空格转小写后）不属于常见的"假值"集合，就视为真
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    # 从环境变量读取整数配置；未设置时返回默认值
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    # 从环境变量读取浮点数配置；未设置时返回默认值
    value = os.getenv(name)
    return default if value is None else float(value)


def _to_dict(value: Any) -> dict[str, Any]:
    # 把任意对象尽量转换成字典（兼容 dict、dataclass、pydantic 模型等）
    if value is None:
        # 空值直接映射为空字典
        return {}
    if isinstance(value, dict):
        # 本身就是字典则原样返回
        return value
    # 尝试获取对象上的 to_dict 方法（dataclasses_json 风格）
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        # 存在 to_dict 方法则调用并返回结果
        return to_dict()
    # 再尝试获取 pydantic v2 的 model_dump 方法
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        # 存在 model_dump 方法则调用并返回结果
        return model_dump()
    # 都不行时退化为空字典，保证调用方拿到 dict 类型
    return {}


@dataclass
class FunctionSpec(DataClassJsonMixin):
    # 函数调用规格：描述一个可被 LLM 调用的函数（名称、参数 JSON Schema、描述）
    name: str
    json_schema: Dict[str, Any]  # JSON schema
    description: str

    def __post_init__(self):
        # Validate the JSON schema
        # 数据类初始化后校验 JSON Schema 是否符合 Draft7 规范，尽早发现格式错误
        jsonschema.Draft7Validator.check_schema(self.json_schema)

    @property
    def as_openai_tool_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI's function format."""
        # 转换为 OpenAI function calling 所需的请求字段格式
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.json_schema,
        }

    @property
    def openai_tool_choice_dict(self) -> Dict[str, Any]:
        # 构造 OpenAI 的 tool_choice：强制模型只调用本函数
        return {
            "type": "function",
            "function": {"name": self.name},
        }

    @property
    def as_anthropic_tool_dict(self):
        """Convert to Anthropic's tool format."""
        # 转换为 Anthropic 工具格式
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.json_schema,  # Anthropic uses input_schema instead of parameters
        }

    @property
    def anthropic_tool_choice_dict(self):
        """Convert to Anthropic's tool choice format."""
        # 构造 Anthropic 的工具选择：强制模型调用本工具
        return {
            "type": "tool",  # Anthropic uses "tool" instead of "function"
            "name": self.name,
        }


class LiteLLMClient:
    # 基于 litellm 的统一 LLM 客户端：封装请求、函数调用、流式解析与用量统计
    PromptType = Union[str, Dict[str, Any], List[Any]]
    FunctionCallType = Dict[str, Any]
    OutputType = Union[str, FunctionCallType]

    def __init__(self, client_cfg):
        """
        Initialize the OpenAI client with any desired default arguments or configuration.
        """
        # 记录要使用的模型 ID（如 gpt-4-turbo）
        self.model = client_cfg.model_id
        # 记录 API 基础地址（兼容自建/代理服务）
        self.base_url = client_cfg.base_url
        # 优先读取按模型名定制的 API Key 环境变量（如 PRIMARY_KEY_GPT4）
        api_key = os.getenv("PRIMARY_KEY_" + self.model.replace("-", "_").upper(), "")
        if api_key:
            # 找到模型专属 Key 则直接使用
            self.api_key = api_key
        else:
            # 否则回退到通用的 PRIMARY_KEY 环境变量
            self.api_key = os.getenv("PRIMARY_KEY", "")
        # 是否使用 Azure 客户端
        self.use_azure_client = client_cfg.use_azure_client
        # 记录提供商类型（openai/anthropic/selfhosted 等）
        self.provider = client_cfg.provider
        if self.use_azure_client:
            # Azure 后端要求模型名带 azure/ 前缀
            self.model_prefix = "azure/"
        elif "/" in self.model:
            # 模型名已含提供商前缀（如 openai/gpt-4）则不再追加
            self.model_prefix = ""
        else:
            # 默认按 openai/ 前缀处理
            self.model_prefix = "openai/"

        # 拼上提供商前缀，得到 litellm 实际使用的完整模型标识
        self.model = self.model_prefix + self.model

        # 将 httpx 的日志级别降为 WARNING，避免每次请求都打印连接日志
        logging.getLogger("httpx").setLevel(logging.WARNING)

    @property
    def client_content_key(self):
        # 消息体中承载正文的字段名（OpenAI 风格为 "content"）
        return "content"

    def _calculate_cost(self, prompt_tokens, completion_tokens):
        """Calculate the API cost for a request based on token usage and provider-specific pricing."""
        # 费用初始化为 0
        cost = 0.0
        # Example cost calculation for different providers/models
        if self.provider.lower() == "openai":
            # Define cost per 1K tokens for some known OpenAI models (in USD)
            if "gpt-3.5" in self.model.lower():
                # GPT-3.5 系列的提示/补全单价（每千 token，美元）
                prompt_cost_per_1k = 0.0
                completion_cost_per_1k = 0.0
            elif "gpt-4" in self.model.lower():
                # GPT-4 系列的单价
                prompt_cost_per_1k = 0.0
                completion_cost_per_1k = 0.0
            else:
                # Default rates for other OpenAI models (if any)
                # 其他 OpenAI 模型使用默认单价
                prompt_cost_per_1k = 0.0
                completion_cost_per_1k = 0.0
            # Calculate cost proportionally to the number of tokens (token counts are divided by 1000 for per-1K pricing)
            # 按 token 数量比例计算总费用（token 数除以 1000 再乘每千 token 单价）
            cost = (prompt_tokens / 1000.0) * prompt_cost_per_1k + (
                completion_tokens / 1000.0
            ) * completion_cost_per_1k
        elif self.provider.lower() == "anthropic":
            prompt_cost_per_1k = 0.0  # example cost per 1K tokens for prompts on Anthropic
            completion_cost_per_1k = 0.0  # example cost per 1K tokens for completions on Anthropic
            # Anthropic 提供商按同样的公式计费
            cost = (prompt_tokens / 1000.0) * prompt_cost_per_1k + (
                completion_tokens / 1000.0
            ) * completion_cost_per_1k
        else:
            # Other providers or default case
            # If costs are not known, leave as 0 or implement accordingly
            # 未知提供商或未配置价格时费用记 0
            cost = 0.0
        return round(cost, 6)  # rounding to a reasonable number of decimal places for currency

    def count_tokens(self, text):
        """Utility method to count tokens in a given text string."""
        # In a real scenario, this should use the model's tokenizer for accuracy.
        # Here, we'll use a simple whitespace split as a placeholder.
        # 文本为空则 token 数记 0
        if text is None:
            return 0
        # 简易估算：按空白分词的词数近似 token 数（真实场景应使用模型分词器）
        return len(text.split())

    def _should_stream(self, filtered_kwargs: dict[str, Any], func_spec: Optional[FunctionSpec]) -> bool:
        # 判断本次请求是否应使用流式输出
        if func_spec is not None:
            # 函数调用场景不支持流式，直接返回 False
            return False
        if "stream" in filtered_kwargs:
            # 调用方显式指定了 stream 参数则尊重该设置
            return bool(filtered_kwargs["stream"])
        # 将基础地址转小写便于判断是否本地服务
        base_url = str(self.base_url or "").lower()
        # 提供商为 selfhosted 且地址指向本地回环时，视为本地自托管服务
        is_local_selfhosted = self.provider.lower() == "selfhosted" and (
            "localhost" in base_url or "127.0.0.1" in base_url
        )
        # 默认：本地自托管服务走流式，其余由环境变量 STREAM_ENV 决定
        return _env_bool(STREAM_ENV, is_local_selfhosted)

    def _consume_stream(self, stream: Any) -> tuple[str, dict[str, Any]]:
        # 消费流式响应：把分块拼接为完整文本，并收集用量/结束原因/推理内容
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage_stats: dict[str, Any] = {}
        finish_reason = None

        # 逐块遍历流式返回
        for chunk in stream:
            # 把当前块统一转成字典便于取值
            chunk_payload = _to_dict(chunk)
            # 提取本块携带的用量统计（通常最后一个块带 usage）
            usage = chunk_payload.get("usage")
            if usage:
                # 有用量信息则合并进统计字典
                usage_stats.update(_to_dict(usage))

            # 遍历本块的候选选择（一般只有一个）
            for choice in chunk_payload.get("choices") or []:
                # 将选择项转为字典
                choice_payload = _to_dict(choice)
                # 流式块用 delta 增量字段；兼容非流式风格的 message 字段
                delta = _to_dict(choice_payload.get("delta") or choice_payload.get("message"))

                # 取出增量正文
                content = delta.get("content")
                if content:
                    # 有正文增量则追加到文本片段列表
                    text_parts.append(str(content))

                # 兼容多种字段名提取"思考/推理"内容
                reasoning = (
                    delta.get("reasoning_content")
                    or delta.get("reasoning")
                    or delta.get("thinking")
                )
                if reasoning:
                    # 有推理增量则追加到推理片段列表
                    reasoning_parts.append(str(reasoning))

                if choice_payload.get("finish_reason") is not None:
                    # 记录流结束原因（stop/length 等）
                    finish_reason = choice_payload.get("finish_reason")

        if finish_reason is not None:
            # 把结束原因写入用量统计
            usage_stats["finish_reason"] = finish_reason
        if reasoning_parts:
            # 把所有推理片段拼接后写入统计
            usage_stats["reasoning_content"] = "".join(reasoning_parts)
        # 标记本次响应来自流式通道
        usage_stats["streamed"] = True

        # 返回拼接后的完整文本与用量统计
        return "".join(text_parts), usage_stats

    def _query_client(
        self,
        messages: List[Dict[str, str]],
        model_kwargs: Optional[Dict[str, Any]] = None,
        json_schema: Optional[str] = None,
        function_name: Optional[str] = None,
        function_description: Optional[str] = None,
    ) -> Tuple[OutputType, Dict[str, Any]]:
        # Prepare function specifications if provided
        # 函数规格默认为空
        func_spec = None
        if json_schema and function_name and function_description:
            # 三要素齐备时构造 FunctionSpec（JSON Schema 以字符串传入需先解析）
            func_spec = FunctionSpec(function_name, json.loads(json_schema), function_description)

        # Always include necessary model parameters
        # 复制一份模型参数，避免修改调用方传入的字典
        model_kwargs = dict(model_kwargs or {})
        # 强制注入模型名
        model_kwargs["model"] = self.model
        # 注入基础地址
        model_kwargs["base_url"] = self.base_url
        # 注入 API Key
        model_kwargs["api_key"] = self.api_key
        # 过滤掉值为 None 的参数，避免传给 API 引发错误
        filtered_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

        # Attach function specifications if provided
        if func_spec is not None:
            # 把函数规格放入 functions 参数
            filtered_kwargs["functions"] = [func_spec.as_openai_tool_dict]
            # 强制要求模型调用指定函数
            filtered_kwargs["function_call"] = {"name": func_spec.name}

        if "minimax" in self.model.lower() and not any(
            message.get("role") == "user" for message in messages
        ):
            # MiniMax rejects system-only chats, which AIRA operators use by design.
            # MiniMax 不接受纯 system 消息的对话，需合并为一条 user 消息
            merged_content = "\n\n".join(
                str(message.get(self.client_content_key, "")) for message in messages
            ).strip()
            # 用合并后的内容构造单条 user 消息替换原消息列表
            messages = [{"role": "user", self.client_content_key: merged_content}]

        # 判断本次是否走流式输出
        use_stream = self._should_stream(filtered_kwargs, func_spec)
        # 取出是否在流式中包含用量统计的开关（默认 True）
        include_usage = bool(filtered_kwargs.pop("include_usage", True))
        if use_stream:
            # 开启流式
            filtered_kwargs["stream"] = True
            if include_usage:
                # 需要 usage 时设置 stream_options 让最后一块携带用量信息
                filtered_kwargs.setdefault("stream_options", {"include_usage": True})
        else:
            # 非流式时移除所有流式相关参数
            filtered_kwargs.pop("stream", None)
            filtered_kwargs.pop("stream_options", None)

        # 读取重试次数配置（环境变量可覆盖默认值）
        num_retries = _env_int(NUM_RETRIES_ENV, NUM_RETRIES)
        # 读取超时配置并保持浮点类型
        request_timeout = _env_float(TIMEOUT_ENV, TIMEOUT)
        # 同时设置 max_retries 与 num_retries，兼容 litellm 不同版本的参数名
        filtered_kwargs["max_retries"] = num_retries
        filtered_kwargs["num_retries"] = num_retries
        # 注入超时时间（秒）
        filtered_kwargs["timeout"] = float(request_timeout)

        # Record start time for latency measurement
        # 记录起始时间（单调时钟，不受系统时间回拨影响），用于计算延迟
        start_time = time.monotonic()

        # Execute the LLM call, with fallback for function calling errors
        # 响应对象初始化为 None
        completion = None
        # 输出初始化为 None
        output = None
        usage_stats: dict[str, Any] = {}
        try:
            # 发起 LLM 调用
            completion = completion_fn(messages=messages, **filtered_kwargs)
            if use_stream:
                # 流式模式下消费流并解析出文本与用量
                output, usage_stats = self._consume_stream(completion)
        except litellm.BadRequestError as e:
            if "function calling" in str(e).lower() or "functions" in str(e).lower():
                # 若报错与函数调用相关，则降级为纯文本生成
                logger.warning(
                    "Function calling was attempted but is not supported by this model. "
                    "Falling back to plain text generation."
                )
                # Remove function calling parameters and retry
                # 移除函数调用相关参数后重试
                filtered_kwargs.pop("functions", None)
                filtered_kwargs.pop("function_call", None)
                use_stream = self._should_stream(filtered_kwargs, None)
                if use_stream:
                    # 重新按降级后的配置决定流式开关
                    filtered_kwargs["stream"] = True
                    if include_usage:
                        filtered_kwargs.setdefault("stream_options", {"include_usage": True})
                # 以纯文本模式重新发起请求
                completion = completion_fn(messages=messages, **filtered_kwargs)
                if use_stream:
                    # 流式模式下再次消费流
                    output, usage_stats = self._consume_stream(completion)
            else:
                # 与函数调用无关的错误直接抛出
                raise

        # Calculate latency
        # 计算本次请求的总延迟
        latency = time.monotonic() - start_time

        # Extract usage stats from the LLM response (if available)
        # 首个选择项缓存，稍后解析输出时使用
        choice = None
        if not use_stream and completion is not None:
            # 非流式时取第一个选择项
            choice = completion.choices[0]
            # 从响应中提取用量统计（无则给空字典）
            usage_stats = completion.to_dict().get("usage", {}) or {}
            # 取出消息对象
            message = choice.message
            # 兼容多字段名提取推理内容
            reasoning_content = (
                getattr(message, "reasoning_content", None)
                or getattr(message, "reasoning", None)
                or getattr(message, "thinking", None)
            )
            if reasoning_content is None:
                # 主字段缺失时再从提供商专有字段中找推理内容
                provider_specific_fields = getattr(
                    message, "provider_specific_fields", None
                )
                # 统一转成字典
                provider_specific_fields = _to_dict(provider_specific_fields)
                reasoning_content = (
                    provider_specific_fields.get("reasoning_content")
                    or provider_specific_fields.get("reasoning")
                    or provider_specific_fields.get("thinking")
                )
                reasoning_details = provider_specific_fields.get("reasoning_details")
                if reasoning_content is None and isinstance(reasoning_details, list):
                    # 仍未找到时尝试解析 reasoning_details 列表（逐项取 text 拼接）
                    reasoning_parts = [
                        str(detail.get("text"))
                        for detail in reasoning_details
                        if isinstance(detail, dict) and detail.get("text")
                    ]
                    if reasoning_parts:
                        # 拼接所有推理片段
                        reasoning_content = "".join(reasoning_parts)
            if reasoning_content:
                # 把推理内容写入用量统计
                usage_stats["reasoning_content"] = str(reasoning_content)

        # Add latency and success status to the stats
        # 记录延迟
        usage_stats["latency"] = latency
        # 标记请求成功
        usage_stats["success"] = True

        # If token counts are not available from the response, estimate them.
        if "prompt_tokens" not in usage_stats:
            # 响应未给出提示 token 数时，把所有消息内容拼接后做简易估算
            prompt_text = " ".join([str(m.get(self.client_content_key, "")) for m in messages])
            usage_stats["prompt_tokens"] = self.count_tokens(prompt_text)
        if "completion_tokens" not in usage_stats:
            # 响应未给出补全 token 数时，对输出文本做简易估算
            completion_text = output if isinstance(output, str) else choice.message.content
            usage_stats["completion_tokens"] = self.count_tokens(completion_text)
        # 汇总总 token 数
        usage_stats["total_tokens"] = usage_stats["prompt_tokens"] + usage_stats["completion_tokens"]

        # Calculate cost using a helper (this method can adjust for different backends)
        # 按单价表计算本次请求费用并写入统计
        usage_stats["cost"] = self._calculate_cost(usage_stats["prompt_tokens"], usage_stats["completion_tokens"])

        # Parse the response as before
        if use_stream:
            # 流式模式下输出已是文本；为空则给空字符串兜底
            output = output or ""
        elif func_spec is None or "functions" not in filtered_kwargs:
            # 普通文本请求：直接取消息正文
            output = choice.message.content
        else:
            # 函数调用请求：解析 function_call
            try:
                # 尝试取出函数调用对象
                function_call = choice.message.function_call
            except:
                # 消息上没有 function_call 属性时置空
                function_call = None
            if not function_call:
                # 模型未按约定发起函数调用，退化为文本输出
                logger.warning(
                    "No function call was used despite function spec. Fallback to text.\n"
                    f"Message content: {choice.message.content}"
                )
                output = choice.message.content
            else:
                if not str(function_call.name).strip() == str(func_spec.name).strip():
                    # 函数名与规格不符，视为无效调用，退化为文本输出
                    logger.warning(
                        f"Function name mismatch: expected {func_spec.name}, "
                        f"got {function_call.name}. Fallback to text."
                    )
                    output = choice.message.content
                else:
                    try:
                        # 把函数参数（JSON 字符串）解析成字典作为输出
                        output = json.loads(function_call.arguments)
                    except json.JSONDecodeError as ex:
                        # 参数不是合法 JSON 时记录错误并抛出
                        logger.error(f"Error decoding function arguments:\n{function_call.arguments}")
                        raise ex

        # 返回解析后的输出与用量统计
        return output, usage_stats

    def query(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[str] = None,
        function_name: Optional[str] = None,
        function_description: Optional[str] = None,
        **model_kwargs,
    ) -> OutputType:
        """
        General LLM query for various backends with a single system and user message.
        Supports function calling for some backends.

        Args:
            system_message (PromptType | None): Uncompiled system message.
            user_message (PromptType | None): Uncompiled user message.
            model (str): Identifier for the model to use (e.g., "gpt-4-turbo").
            temperature (float | None, optional): Sampling temperature.
            max_tokens (int | None, optional): Maximum number of tokens to generate.
            func_spec (FunctionSpec | None, optional): Optional FunctionSpec for function calling.
            **model_kwargs: Additional keyword arguments for the model.

        Returns:
            OutputType: A string completion or a dict with function call details.
        """

        if self.model == "azure/o1-preview" or self.model == "azure/o3-mini":
            # 这两个 Azure 推理模型不支持 system 角色与 temperature 参数，需做兼容处理
            messages = [{"role": "user", self.client_content_key: m[self.client_content_key]} for m in messages]
            if "temperature" in model_kwargs:
                # 移除不支持的 temperature 参数
                model_kwargs.pop("temperature")

        # 调用内部方法执行真正的请求并拿到输出与用量
        output, usage_stats = self._query_client(
            messages=messages,
            model_kwargs=model_kwargs,
            json_schema=json_schema,
            function_name=function_name,
            function_description=function_description,
        )

        # 返回输出与用量统计给上层调用者
        return output, usage_stats
