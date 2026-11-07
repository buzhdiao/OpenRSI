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
# LLM 客户端（连接参数）配置
from dojo.config_dataclasses.client.base import ClientConfig


# 通用 LLM 配置：组合客户端连接配置与文本生成参数
@dataclass
class GenericLLMConfig(BaseConfig):
    # 求解器所用的 LLM 客户端配置（模型、API 地址等）
    client: ClientConfig = field(
        default_factory=ClientConfig,
        metadata={
            "description": "Client configuration for the solver.",
            "example": {},
        },
    )

    # 传给 LLM 的生成参数（如 temperature、max_tokens 等），默认空字典即用客户端默认值
    generation_kwargs: dict = field(
        default_factory=dict,
        metadata={"help": "LLM generation arguments."},
    )

    # 配置校验入口
    def validate(self) -> None:
        # 复用父类校验逻辑，本类无额外约束
        super().validate()
