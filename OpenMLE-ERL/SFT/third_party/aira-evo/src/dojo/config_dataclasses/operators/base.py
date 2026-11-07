# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 dataclass 装饰器与 field 工厂
from dataclasses import dataclass, field

# OmegaConf 配置库：SI 为字符串插值，MISSING 为必填占位符
from omegaconf import SI, MISSING

# 通用配置基类
from aira_core.config.base import BaseConfig
# 通用 LLM 配置
from dojo.config_dataclasses.llm.generic_llm import GenericLLMConfig
# Jinja 提示词模板配置
from dojo.config_dataclasses.llm.jinjaprompt import JinjaPromptConfig


# 算子（Operator）配置：单个 LLM 调用步骤所需的模型配置与三类提示词模板；
# 算子是求解器中执行某一功能（如生成/评估方案）的基本单元
@dataclass
class OperatorConfig(BaseConfig):
    # 该算子使用的 LLM 配置（客户端与生成参数）
    llm: GenericLLMConfig = field(
        default_factory=GenericLLMConfig,
        metadata={
            "description": "LLM configuration for the operator.",
            "example": {},
        },
    )

    # 系统消息（system message）的提示词模板
    system_message_prompt_template: JinjaPromptConfig = field(default_factory=JinjaPromptConfig)
    # 首条用户消息的提示词模板（初始化对话用）
    init_user_message_prompt_template: JinjaPromptConfig = field(default_factory=JinjaPromptConfig)
    # 后续每轮用户消息的提示词模板
    user_message_prompt_template: JinjaPromptConfig = field(default_factory=JinjaPromptConfig)

    # 配置校验入口
    def validate(self) -> None:
        # 复用父类校验逻辑，本类无额外约束
        super().validate()
