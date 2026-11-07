# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are MIT-licensed
# Copyright (c) 2024 Weco AI Ltd
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/WecoAI/aideml/blob/main/LICENSE

# JSON 解析：解析函数调用参数与 JSON Schema
import json
# 标准日志模块
import logging
# 读取环境变量（API 密钥）
import os
# dataclass 装饰器
from dataclasses import dataclass
# 时间工具：测量请求延迟
import time
# 类型注解工具
from typing import Any, Dict, List, Optional, Tuple, Union
# Google Gemini 官方 SDK
from google import genai
# Gemini SDK 的类型定义（Content/Schema 等）
from google.genai import types
# JSON Schema 校验库
import jsonschema
# OmegaConf 配置库（预留）
from omegaconf import OmegaConf
# OpenAI SDK（预留，本文件未直接调用）
import openai

# import weave
# dataclass 的 JSON 序列化混入
from dataclasses_json import DataClassJsonMixin

# Configure logging
# 配置本模块日志：INFO 级别、输出到标准错误流
logger = logging.getLogger("Backend")
# 设置 INFO 级别
logger.setLevel(logging.INFO)
# 流式输出处理器
handler = logging.StreamHandler()
# 日志格式：级别 名字 消息
formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
# 绑定格式
handler.setFormatter(formatter)
# 挂载处理器
logger.addHandler(handler)


# 函数调用（function calling）规格：函数名、参数 JSON Schema 与描述
@dataclass
class FunctionSpec(DataClassJsonMixin):
    # 函数名
    name: str
    # 参数的 JSON Schema 定义
    json_schema: Dict[str, Any]  # JSON schema
    # 函数功能描述
    description: str

    def __post_init__(self):
        # Validate the JSON schema
        # 构造后立即校验 JSON Schema 的合法性（Draft 7 规范）
        jsonschema.Draft7Validator.check_schema(self.json_schema)

    @property
    def as_openai_tool_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI's function format."""
        # 转换为 OpenAI 工具（tool）格式的字典
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.json_schema,
        }

    @property
    def openai_tool_choice_dict(self) -> Dict[str, Any]:
        # 强制模型调用指定函数的 tool_choice 结构
        return {
            "type": "function",
            "function": {"name": self.name},
        }


# Gemini（GDM）模型客户端后端：把统一的消息/函数调用接口适配到 Gemini SDK
class GDMClient:
    # 提示词类型：字符串、字典或列表
    PromptType = Union[str, Dict[str, Any], List[Any]]
    # 函数调用结果类型
    FunctionCallType = Dict[str, Any]
    # 输出类型：文本或函数调用字典
    OutputType = Union[str, FunctionCallType]

    def __init__(self, client_cfg):
        # 从客户端配置读取模型 ID
        self.model = client_cfg.model_id
        # 从环境变量读取 Gemini API 密钥
        self.api_key = os.getenv("GEMINI_API_KEY", "")

        # 创建 Gemini SDK 客户端
        self._client = genai.Client(api_key=self.api_key)

    @property
    def client_content_key(self):
        # 内容字段的键名（与其他后端接口对齐）
        return "content"

    def generate_response(self, messages: list, functions: list = None, **model_kwargs):
        """
        Call the Gemini model with a conversation and optional function definitions.
        Returns a tuple: (output_text, usage_stats).
        """
        # Separate system instruction if present
        # 若存在系统消息则单独抽出（Gemini 要求 system instruction 独立传递）
        system_instruction = None
        # 用户/助手消息列表
        content_list = []
        # 遍历 OpenAI 风格的消息
        for msg in messages:
            # 取消息角色
            role = msg.get("role")
            # 取消息文本
            text = msg.get("content", "")
            # 系统消息：作为独立系统指令
            if role == "system":
                system_instruction = text  # use config for system instructions
            # 用户消息：转为 Gemini 的 user 内容
            elif role == "user":
                content_list.append(types.Content(role="user", parts=[types.Part.from_text(text=text)]))
            # 助手消息：Gemini 中该角色叫 "model"，做名称映射
            elif role in ("assistant", "model"):
                # Treat assistant messages as 'model' role in Gemini
                content_list.append(types.Content(role="model", parts=[types.Part.from_text(text=text)]))

        # Prepare function tools if any functions are provided
        # 若提供了函数定义则准备工具（tools）
        tools = None
        if functions:
            # Gemini 函数声明列表
            func_decls = []
            # 逐个转换函数定义
            for func in functions:
                # Convert OpenAI function spec to Gemini FunctionDeclaration
                # 取函数名、描述与参数 Schema
                name = func.get("name")
                desc = func.get("description", "")
                params = func.get("parameters", {})

                # Convert parameters JSON schema to types.Schema recursively
                # 内部函数：把参数 JSON Schema 递归转换为 Gemini 的 types.Schema
                def to_schema(schema):
                    # 取类型名
                    schema_type = schema.get("type", "")
                    # OpenAI uses lowercase types; Gemini expects uppercase (OBJECT, STRING, etc.)
                    # OpenAI 用小写类型名，Gemini 要求大写：构造 Schema 时转大写
                    schema_obj = types.Schema(type=schema_type.upper() if schema_type else None)
                    # 有描述则附加
                    if "description" in schema:
                        schema_obj.description = schema["description"]
                    # 有子属性则逐个递归转换
                    if "properties" in schema:
                        schema_obj.properties = {key: to_schema(val) for key, val in schema["properties"].items()}
                    # 有必填列表则附加
                    if "required" in schema:
                        schema_obj.required = schema["required"]
                    # 返回 Gemini Schema 对象
                    return schema_obj

                # 转换参数 Schema 并构造函数声明
                param_schema = to_schema(params)
                func_decls.append(types.FunctionDeclaration(name=name, description=desc, parameters=param_schema))
            # 存在声明才打包为工具列表
            if func_decls:
                tools = [types.Tool(function_declarations=func_decls)]

        # Map model_kwargs to GenerateContentConfig
        # 把通用生成参数映射到 Gemini 的 GenerateContentConfig
        config_args = {}
        # Map known parameter names to Gemini config fields
        # 有系统指令则传入
        if system_instruction:
            config_args["system_instruction"] = system_instruction
        # 采样温度直接同名映射
        if "temperature" in model_kwargs:
            config_args["temperature"] = model_kwargs["temperature"]
        # 核采样参数直接同名映射
        if "top_p" in model_kwargs:
            config_args["top_p"] = model_kwargs["top_p"]
        # top-k 采样参数直接同名映射
        if "top_k" in model_kwargs:
            config_args["top_k"] = model_kwargs["top_k"]
        # max_tokens for output
        # OpenAI 风格的 max_tokens 映射为 Gemini 的 max_output_tokens
        if "max_tokens" in model_kwargs:
            config_args["max_output_tokens"] = model_kwargs["max_tokens"]
        # Gemini 原生参数名直接映射
        if "max_output_tokens" in model_kwargs:
            config_args["max_output_tokens"] = model_kwargs["max_output_tokens"]
        # 存在性惩罚直接映射
        if "presence_penalty" in model_kwargs:
            config_args["presence_penalty"] = model_kwargs["presence_penalty"]
        # 频率惩罚直接映射
        if "frequency_penalty" in model_kwargs:
            config_args["frequency_penalty"] = model_kwargs["frequency_penalty"]
        # 停止序列：统一为列表
        if "stop" in model_kwargs:
            # `stop` could be a single string or list of strings
            stops = model_kwargs["stop"]
            config_args["stop_sequences"] = stops if isinstance(stops, list) else [stops]
        # Include function calling config if tools are present
        # 有工具时附加函数调用配置（ANY 模式：强制模型必须发起函数调用）
        if tools:
            config_args["tool_config"] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            )
            config_args["tools"] = tools

        # Create config object
        # 用收集的参数构造生成配置对象
        config = types.GenerateContentConfig(**config_args)

        # Call the Gemini model
        # Record start time for latency measurement
        # 记录起始时间以测量延迟
        start_time = time.monotonic()
        # 发起生成请求（多条消息传列表；单条直接传元素；无消息传空串）
        response = self._client.models.generate_content(
            model=self.model,
            contents=content_list if len(content_list) > 1 else (content_list[0] if content_list else ""),
            # If only one content element, can pass it directly (string or Content)
            config=config,
        )
        # Calculate latency
        # 单调时钟差值即请求耗时
        latency = time.monotonic() - start_time

        # Parse the response
        # 解析响应
        if tools is None:
            # No function calling was used
            # 未使用函数调用：直接取文本输出
            output = response.text
        else:
            # Attempt to extract function call
            # 使用了函数调用：尝试提取模型发起的调用
            function_call = response.function_calls
            # 模型没有发起调用
            if not function_call:
                # 记录告警并回退到文本输出
                logger.warning(
                    "No function call was used despite function spec. Fallback to text.\n"
                    f"Message content: {response.text}"
                )
                output = response.text
            else:
                # 取第一个函数调用
                function_call = function_call[0]
                try:
                    # 参数是 JSON 字符串则解析为字典
                    if isinstance(function_call.args, str):
                        output = json.load(function_call.args)
                    # 参数已是字典则直接使用
                    elif isinstance(function_call.args, dict):
                        output = function_call.args
                except json.JSONDecodeError as ex:
                    # 参数不是合法 JSON：记录错误详情并重新抛出
                    logger.error(f"Error decoding function arguments:\n{str(function_call.args)}")
                    raise ex

        # Gather token usage stats
        # 汇总 token 用量统计
        usage = response.usage_metadata  # usage stats object
        # 提示词 token 数
        prompt_tokens = getattr(usage, "prompt_token_count", None)
        # Gemini may provide list of candidate token counts
        # 补全 token 数先置 None
        completion_tokens = None
        # 存在候选计数则取其作为补全 token 数
        if hasattr(usage, "candidates_token_count") and usage.candidates_token_count:
            # Use first candidate's token count as completion tokens
            completion_tokens = usage.candidates_token_count
        # 总 token 数
        total_tokens = getattr(usage, "total_token_count", None)
        # Fallback: if completion not provided, compute if possible
        # 兜底：未提供补全数时用总数减去提示数推算
        if prompt_tokens is not None and total_tokens is not None and completion_tokens is None:
            completion_tokens = total_tokens - prompt_tokens
        # 组装用量统计字典（含请求延迟）
        usage_stats = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency": latency,
        }
        # 返回输出与用量统计
        return output, usage_stats

    def _query_client(
        self,
        messages: List[Dict[str, str]],
        model_kwargs: Dict[str, Any] = {},
        json_schema: Optional[str] = None,
        function_name: Optional[str] = None,
        function_description: Optional[str] = None,
    ) -> Tuple[OutputType, float, int, int, Dict[str, Any]]:
        # 内部查询：组装函数规格并调用 generate_response，含函数调用不支持时的降级重试
        # 把模型 ID 注入参数
        model_kwargs["model"] = self.model
        # 过滤掉值为 None 的参数
        filtered_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

        # 函数规格先置 None
        func_spec = None
        # Schema、函数名、描述三要素齐全才构造规格
        if json_schema and function_name and function_description:
            func_spec = FunctionSpec(function_name, json.loads(json_schema), function_description)

        # Attach function specifications if provided
        # 若提供了函数规格则转为 OpenAI 工具格式列表
        functions = None
        if func_spec is not None:
            functions = [func_spec.as_openai_tool_dict]

        try:
            # 正常发起调用
            output, usage_stats = self.generate_response(messages, functions, **filtered_kwargs)
        except Exception as e:
            # Check if function calling is not supported
            # 调用失败：检查是否为"不支持函数调用"类错误
            if "function calling" in str(e).lower() or "functions" in str(e).lower():
                # 记录告警
                logger.warning(
                    "Function calling was attempted but is not supported by this model. "
                    "Falling back to plain text generation."
                )

                # Retry without function calling
                # 去掉函数定义后重试（纯文本生成）
                output, usage_stats = self.generate_response(messages=messages, functions=None, **filtered_kwargs)
            else:
                # Re-raise other exceptions
                # 其他异常不属于可降级场景，原样重新抛出
                raise

        # 返回输出与用量统计
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

        # 通用查询入口：委托内部查询方法
        output, usage_stats = self._query_client(
            messages=messages,
            model_kwargs=model_kwargs,
            json_schema=json_schema,
            function_name=function_name,
            function_description=function_description,
        )

        # 返回输出文本/函数调用与用量统计
        return output, usage_stats
