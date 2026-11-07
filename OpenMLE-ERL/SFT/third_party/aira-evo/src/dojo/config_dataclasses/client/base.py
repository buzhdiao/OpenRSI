# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 dataclass 装饰器与 field：声明数据类字段及其元数据
from dataclasses import dataclass, field

# 导入 SI 与 MISSING：OmegaConf 的插值符号与必填哨兵（本文件实际仅用到默认值字段）
from omegaconf import SI, MISSING

# 导入配置基类：提供哈希/校验等通用能力
from aira_core.config.base import BaseConfig


# @dataclass：自动生成构造/比较等方法
@dataclass
# LLM 客户端配置类：描述模型接入方式（接口/模型/端点/提供方）
class ClientConfig(BaseConfig):
    # API 后端类型字段，默认 litellm；exclude_from_hash 使其不参与配置哈希
    api: str = field(
        default="litellm",
        metadata={
            "description": "API endpoint for the LLM client.",
            "example": "litellm",
            "exclude_from_hash": True,
        },
    )

    # 模型标识字段：要调用的具体模型 ID
    model_id: str = field(
        default="gpt-4o",
        metadata={
            "description": "The identifier for the model to use.",
            "example": "deepseek-ai/DeepSeek-R1",
        },
    )

    # API 基础端点 URL 字段；同样排除出哈希（端点变化不代表配置语义变化）
    base_url: str = field(
        default="https://azure-services-endpoint-here.azure-api.net",
        metadata={
            "description": "Base URL for the API endpoint.",
            "example": "http://self-hosted-endpoint:20000/v1",
            "exclude_from_hash": True,
        },
    )

    # 是否使用 Azure 客户端发起请求的开关字段
    use_azure_client: bool = field(
        default=True,
        metadata={
            "description": "Whether to use Azure client for API calls.",
            "exclude_from_hash": True,
        },
    )

    # LLM 服务提供方字段（如 openai/selfhosted 等）
    provider: str = field(
        default="openai",
        metadata={
            "description": "The provider of the LLM service.",
            "example": "selfhosted",
            "exclude_from_hash": True,
        },
    )

    # 校验逻辑：直接复用基类的递归校验
    def validate(self) -> None:
        super().validate()
