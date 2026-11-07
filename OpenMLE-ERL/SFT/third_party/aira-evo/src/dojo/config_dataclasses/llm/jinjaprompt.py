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


# Jinja 提示词模板配置：用 Jinja2 模板语言渲染 LLM 的 prompt（提示词）
@dataclass
class JinjaPromptConfig(BaseConfig):
    # prompt 模板正文，支持 {{变量}} 等 Jinja 语法
    template: str = field(
        default="",
        metadata={"help": "The Jinja template to be used for the prompt."},
    )

    # 模板渲染时需要传入的变量名列表
    input_variables: list[str] = field(
        default_factory=list,
        metadata={"help": "List of input variables for the Jinja template."},
    )

    # 预先固定的"部分变量"（partial），渲染时与输入变量合并
    partial_variables: dict = field(
        default_factory=dict,
        metadata={"help": "Dictionary of partial variables for the Jinja template."},
    )

    # 配置校验入口
    def validate(self) -> None:
        # 复用父类校验逻辑，本类无额外约束
        super().validate()
