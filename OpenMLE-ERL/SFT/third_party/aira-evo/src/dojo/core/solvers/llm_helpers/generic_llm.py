# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入类型注解工具，用于函数签名标注
from typing import Any, Dict, List, Optional

# 导入 hydra（此处保留导入以维持依赖关系）
import hydra
# 导入 OmegaConf 配置库
from omegaconf import OmegaConf

# 导入客户端工厂函数，按配置创建具体 LLM 后端
from dojo.core.solvers.llm_helpers.backends.utils import get_client
# 导入统一日志器与日志事件枚举
from dojo.utils.logger import get_logger, LogEvent
# 导入 Jinja 提示词模板类
from dojo.core.solvers.llm_helpers.prompt_template import JinjaPrompt

# 导入算子配置数据类
from dojo.config_dataclasses.operators.base import OperatorConfig
# 导入 LLM 客户端配置数据类
from dojo.config_dataclasses.client.base import ClientConfig
# 导入 LLM 并发槽位获取函数，用于限制对 LLM 的并发调用
from tts_search.airaevo_concurrency import acquire_llm_slot

# 导入 logging 模块
import logging

# 获取本模块专属日志器
log = logging.getLogger(__name__)

# 将 openai SDK 日志级别设为 DEBUG，便于排查请求细节
logging.getLogger("openai").setLevel(logging.DEBUG)
# 将 httpx 日志级别设为 DEBUG
logging.getLogger("httpx").setLevel(logging.DEBUG)


class GenericLLM:
    # 通用 LLM 封装层：组合提示词模板与后端客户端，对外提供统一调用接口

    def __init__(self, cfg: OperatorConfig) -> None:
        """
        Initializes the GenericLLM with the provided configuration.

        Args:
            cfg: Arbitrary keyword arguments representing the configuration.
        """
        # Convert the configuration dictionary into an OmegaConf object for easier access
        # 保存算子配置，供后续构造提示词使用
        self.cfg = cfg

        # Initialize the LLM client using the provided client configuration
        # 通过工厂函数创建具体的 LLM 客户端（openai/litellm/gdm）
        self.client = get_client(self.cfg.llm.client)

        # LLM Generation Arguments
        # 保存生成参数（temperature、max_tokens 等），调用时透传给后端
        self.generation_kwargs = self.cfg.llm.generation_kwargs

        # Set up the prompt templates based on the configuration
        # 根据配置初始化各类提示词模板
        self._set_up_prompts()

        # 获取全局运行日志器
        self.logger = get_logger()

        # 调用计数器：累计本实例发起的 LLM 调用次数
        self.call_tracker = 0

    @property
    def client_content_key(self):
        # 透传后端的消息正文字段名（如 "content"）
        return self.client.client_content_key

    def _set_up_prompts(self) -> None:
        """
        Sets up the prompt templates by instantiating them using Hydra.
        This method checks for the presence of various prompt templates in the configuration
        and instantiates them accordingly.
        """
        # Check and instantiate the system message prompt template if it exists
        # 初始化系统消息（system message）提示词模板
        self.system_message_prompt_template = JinjaPrompt(self.cfg.system_message_prompt_template)
        # 初始化首轮用户消息提示词模板
        self.init_user_message_prompt_template = JinjaPrompt(self.cfg.init_user_message_prompt_template)
        # 初始化后续轮次用户消息提示词模板
        self.user_message_prompt_template = JinjaPrompt(self.cfg.user_message_prompt_template)

    def _query_client_with_limit(
        self,
        messages: List[Dict[str, str]],
        *,
        json_schema: Optional[str] = None,
        function_name: Optional[str] = None,
        function_description: Optional[str] = None,
    ) -> Any:
        # 先获取 LLM 并发槽位（限流），再发起真正的后端调用
        with acquire_llm_slot():
            # 把消息、函数调用规格与生成参数一并传给后端客户端
            return self.client.query(
                messages,
                json_schema=json_schema,
                function_name=function_name,
                function_description=function_description,
                **self.generation_kwargs,
            )

    def __call__(
        self,
        query_data: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        no_user_message: bool = False,
        json_schema: Optional[str] = None,
        function_name: Optional[str] = None,
        function_description: Optional[str] = None,
    ) -> Any:
        """
        Makes a call to the LLM client with the provided query data or messages.

        Args:
            query_data: A dictionary containing data for formatting prompts.
            messages: A list of message dictionaries to send to the LLM.
            no_user_message: If True, does not append a user message to the prompts.
            json_schema: An optional JSON schema string.

        Returns:
            The response from the LLM client.

        Raises:
            AssertionError: If both query_data and messages are None.
        """
        # 记录一条警告级日志，标记即将发起 LLM 查询
        log.warning("sending query to llm")
        # 调用计数加一
        self.call_tracker += 1

        # Ensure that at least one of query_data or messages is provided
        # 断言 query_data 与 messages 不能同时为空，否则无法构造请求
        assert not (query_data is None and messages is None), (
            "Neither the query_data nor the messages object were specified."
        )

        # If query_data is not provided, directly query the client with the provided messages
        if query_data is None:
            # 调用方已备好完整消息列表：直接限流调用后端
            output, usage_stats = self._query_client_with_limit(
                messages,
                json_schema=json_schema,
                function_name=function_name,
                function_description=function_description,
            )
            # 在用量统计中记录累计调用次数
            usage_stats["cumulative_num_llm_calls"] = self.call_tracker
            # 返回输出及包含用量、提示消息、补全文本的元信息字典
            return output, {"usage": usage_stats, "prompt_messages": messages, "completion_text": str(output)}

        # If messages are not provided, initialize them with a system message using the query_data
        if messages is None:
            # 未提供消息列表时，先用 query_data 渲染系统消息作为对话开头
            messages = [
                {
                    "role": "system",
                    self.client.client_content_key: self.system_message_prompt_template.format(**query_data),
                }
            ]

        # If no_user_message is True, directly query the client without adding a user message
        if no_user_message:
            # 不追加用户消息的场景：以现有消息直接限流调用后端
            output, usage_stats = self._query_client_with_limit(
                messages,
                json_schema=json_schema,
                function_name=function_name,
                function_description=function_description,
            )
            # 记录累计调用次数
            usage_stats["cumulative_num_llm_calls"] = self.call_tracker

            # 返回输出与元信息
            return output, {"usage": usage_stats, "prompt_messages": messages, "completion_text": str(output)}

        # Append a user message based on whether it's the first message or a subsequent one
        if len(messages) == 1:
            # First user message uses the initial user message prompt template
            # 对话只有系统消息时，这是首轮提问，用首轮用户消息模板渲染
            user_message = {
                "role": "user",
                self.client.client_content_key: self.init_user_message_prompt_template.format(**query_data),
            }
        else:
            # Subsequent user messages use the standard user message prompt template
            # 已有多轮消息时，用标准用户消息模板渲染
            user_message = {
                "role": "user",
                self.client.client_content_key: self.user_message_prompt_template.format(**query_data),
            }
        # 把用户消息追加到消息列表末尾
        messages.append(user_message)

        # Query the client with the updated messages
        # 携带完整对话消息限流调用后端
        output, usage_stats = self._query_client_with_limit(
            messages,
            json_schema=json_schema,
            function_name=function_name,
            function_description=function_description,
        )
        # 记录累计调用次数
        usage_stats["cumulative_num_llm_calls"] = self.call_tracker

        # 记录日志标记已收到 LLM 响应
        log.warning("got response from llm")
        # 返回输出与元信息（用量、提示消息、补全文本）
        return output, {"usage": usage_stats, "prompt_messages": messages, "completion_text": str(output)}
