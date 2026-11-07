# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 json 模块，用于解析函数调用参数中的 JSON 字符串
import json
# 导入 logging 模块，用于记录后端运行日志
import logging
# 导入 os 模块，用于读取 API Key 等环境变量
import os
# 导入 dataclass 装饰器，用于定义函数规格（FunctionSpec）数据类
from dataclasses import dataclass
# 导入 time 模块，用于统计请求耗时（延迟测量）
import time
# 导入类型注解工具，用于函数签名标注
from typing import Any, Dict, List, Optional, Tuple, Union

# 导入 jsonschema，用于校验函数参数的 JSON Schema 合法性
import jsonschema
# 导入 openai 官方 SDK
import openai

# import weave
# 导入 DataClassJsonMixin，让数据类支持与 JSON 字符串互转
from dataclasses_json import DataClassJsonMixin

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


class OpenAIClient:
    # 基于 openai SDK 的 LLM 客户端：封装聊天补全请求、函数调用与用量统计
    PromptType = Union[str, Dict[str, Any], List[Any]]
    FunctionCallType = Dict[str, Any]
    OutputType = Union[str, FunctionCallType]

    def __init__(self, client_cfg):
        """
        Initialize the OpenAI client with any desired default arguments or configuration.
        """
        # 记录要使用的模型 ID（如 gpt-4-turbo）
        self.model = client_cfg.model_id
        # 记录 API 基础地址（兼容代理/自建服务）
        self.base_url = client_cfg.base_url
        # 从环境变量读取 API Key
        self.api_key = os.getenv("PRIMARY_KEY", "")
        # 是否使用 Azure OpenAI 客户端
        self.use_azure_client = client_cfg.use_azure_client

        if self.use_azure_client:
            # Azure 通道需要专门的 AzureOpenAI 客户端（带 API 版本与 Azure 端点）
            self._client = openai.AzureOpenAI(
                max_retries=3, api_key=self.api_key, api_version="2024-10-21", azure_endpoint=self.base_url
            )
        else:
            # 标准 OpenAI 兼容客户端（含基础地址与 Key）
            self._client = openai.OpenAI(max_retries=3, base_url=self.base_url, api_key=self.api_key)

        # 将 httpx 的日志级别降为 WARNING，避免每次请求都打印连接日志
        logging.getLogger("httpx").setLevel(logging.WARNING)

    @property
    def client_content_key(self):
        # 消息体中承载正文的字段名（OpenAI 风格为 "content"）
        return "content"

    def _query_client(
        self,
        messages: List[Dict[str, str]],
        model_kwargs: Dict[str, Any] = {},
        json_schema: Optional[str] = None,
        function_name: Optional[str] = None,
        function_description: Optional[str] = None,
    ) -> Tuple[OutputType, float, int, int, Dict[str, Any]]:
        # 强制注入模型名到请求参数中
        model_kwargs["model"] = self.model
        # 过滤掉值为 None 的参数，避免传给 API 引发错误
        filtered_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

        # 函数规格默认为空
        func_spec = None
        if json_schema and function_name and function_description:
            # 三要素齐备时构造 FunctionSpec（JSON Schema 以字符串传入需先解析）
            func_spec = FunctionSpec(function_name, json.loads(json_schema), function_description)

        # Attach function specifications if provided
        if func_spec is not None:
            # 把函数规格放入 functions 参数
            filtered_kwargs["functions"] = [func_spec.as_openai_tool_dict]
            # 强制要求模型调用指定函数
            filtered_kwargs["function_call"] = {"name": function_name}

        # 响应对象初始化为 None
        completion = None
        # 记录起始时间（单调时钟），用于计算延迟
        start_time = time.monotonic()
        try:
            # 发起聊天补全请求
            completion = self._client.chat.completions.create(messages=messages, **filtered_kwargs)
        except openai.BadRequestError as e:
            # Check if function calling is not supported
            if "function calling" in str(e).lower() or "functions" in str(e).lower():
                # 若报错与函数调用相关，则降级为纯文本生成
                logger.warning(
                    "Function calling was attempted but is not supported by this model. "
                    "Falling back to plain text generation."
                )
                # Remove function-calling parameters and retry
                # 移除函数调用相关参数
                filtered_kwargs.pop("functions", None)
                filtered_kwargs.pop("function_call", None)

                # Retry without function calling
                # 以纯文本模式重新发起请求
                completion = self._client.chat.completions.create(messages=messages, **filtered_kwargs)
            else:
                # Re-raise other exceptions
                # 与函数调用无关的错误直接抛出
                raise

        # Calculate latency
        # 计算本次请求的总延迟
        latency = time.monotonic() - start_time

        # 取第一个选择项用于后续解析
        choice = completion.choices[0]
        if completion is not None:
            # 从响应中提取用量统计（token 数等）
            usage_stats = completion.to_dict()["usage"]
        else:
            # 无响应时给空字典兜底
            usage_stats = {}
        # 把延迟写入用量统计
        usage_stats["latency"] = latency

        # Parse the response
        if func_spec is None or "functions" not in filtered_kwargs:
            # No function calling was used
            # 普通文本请求：直接取消息正文
            output = choice.message.content
        else:
            # Attempt to extract function call
            # 函数调用请求：提取 function_call 对象
            function_call = choice.message.function_call
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

        if self.model == "o1-preview":
            # o1-preview 不支持 system 角色，需要把所有消息改为 user 角色
            messages = [{"role": "user", self.client_content_key: m[self.client_content_key]} for m in messages]
            if "temperature" in model_kwargs:
                # o1 系列不支持 temperature 参数，移除之
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
