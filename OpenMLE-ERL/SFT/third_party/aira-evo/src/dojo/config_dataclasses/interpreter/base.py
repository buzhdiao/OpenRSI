# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 dataclass 装饰器与 field：声明数据类字段及其元数据
from dataclasses import dataclass, field

# 导入 SI：OmegaConf 的插值符号，允许字段默认值引用其他配置节点
from omegaconf import SI

# 导入配置基类：提供哈希/校验等通用能力
from aira_core.config.base import BaseConfig


# @dataclass：自动生成构造/比较等方法
@dataclass
# 解释器基类配置：智能体执行代码的沙箱环境参数
class InterpreterConfig(BaseConfig):
    # 工作目录字段：默认插值引用 logger.output_dir 下的 workspace_agent/
    working_dir: str = field(
        default=SI("${logger.output_dir}/workspace_agent/"),
        metadata={
            "help": "Working directory for the interpreter.",
            "exclude_from_hash": True,
        },
    )
    # 执行超时字段：默认插值引用 solver.execution_timeout，保持与求解器一致
    timeout: int = field(
        default=SI("${solver.execution_timeout}"),
        metadata={"help": "Timeout for the interpreter."},
    )

    # 校验逻辑：直接复用基类的递归校验
    def validate(self) -> None:
        super().validate()
