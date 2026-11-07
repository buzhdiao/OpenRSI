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


# 任务（Task）基类配置：描述智能体要解决的机器学习任务本身（名称、基准、数据位置）
@dataclass
class TaskConfig(BaseConfig):
    # 任务名称，通常对应具体竞赛/数据集名
    name: str = field(
        default=MISSING,
        metadata={
            "help": "Task name, e.g., 'random-acts-of-pizza'.",
        },
    )
    # 任务所属的总体评测基准（benchmark）名称
    benchmark: str = field(
        default=MISSING,
        metadata={
            "help": "Overaching benchmark for the task, e.g., 'mlebench', 'AIRS-H'.",
        },
    )
    # 任务数据存放目录：默认插值为 Hydra 运行时工作目录下的 data 子目录
    data_dir: str = field(
        default=SI("${hydra:runtime.cwd}/data"),
        metadata={
            "help": "The directory where the data is stored.",
            "exclude_from_hash": True,
        },
    )

    # 配置校验入口
    def validate(self) -> None:
        # 复用父类校验逻辑，本类无额外约束
        super().validate()
