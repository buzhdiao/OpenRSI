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


# 记忆（Memory）算子配置：控制智能体如何维护历史信息（上下文记忆策略），
# 决定哪些历史内容可被后续 LLM 调用引用
@dataclass
class MemoryOpConfig(BaseConfig):
    # 使用的记忆处理器：simple_memory（保留记忆）或 no_memory（不使用记忆）
    memory_processor: str = field(
        default="simple_memory",
        metadata={
            "help": "The memory processor to use. Options: simple_memory, no_memory.",
            "choices": ["simple_memory", "no_memory"],
        },
    )
    # 传给记忆处理器的额外关键字参数（例如限制最大记忆条数）
    memory_op_kwargs: dict = field(
        default_factory=dict,
        metadata={
            "help": "Additional arguments for the memory processor.",
            "example": {"max_memory_size": 100},
        },
    )

    # 配置校验入口
    def validate(self) -> None:
        # 复用父类校验逻辑，本类无额外约束
        super().validate()
