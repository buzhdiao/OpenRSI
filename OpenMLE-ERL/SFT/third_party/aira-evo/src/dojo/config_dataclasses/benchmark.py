# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 dataclass 装饰器与 field：声明数据类字段及其元数据
from dataclasses import dataclass, field

# 导入 Any：字段类型标注中的任意类型
from typing import Any
# 导入 SI 与 MISSING：OmegaConf 的插值符号与"必填缺失"哨兵值
from omegaconf import SI, MISSING

# 导入配置基类：提供哈希/校验等通用能力
from aira_core.config.base import BaseConfig

# 导入配置构建工具（本文件中未直接使用该导入）
from dojo.utils.config import build
# 导入任务配置映射 TASK_MAP（本文件中未直接使用该导入）
from dojo.config_dataclasses.task import TASK_MAP
# 导入 MLEBench 任务配置类：用于把基准任务名展开成具体任务配置
from dojo.config_dataclasses.task.mlebench import MLEBenchTaskConfig


# @dataclass：自动生成构造/比较等方法
@dataclass
# 基准（benchmark）配置类：描述在哪个基准上跑哪些任务及覆盖参数
class BenchmarkConfig(BaseConfig):
    # 基准名称字段：MISSING 表示必填（未提供则校验失败）
    name: str = field(
        default=MISSING,
        metadata={
            "help": "Benchmark name, e.g., 'mlebench', 'AIRS-H'.",
        },
    )

    # 任务名列表字段：指定该基准下要运行的任务集合
    tasks: list[str] = field(
        default=MISSING,
        metadata={
            "help": "List of task name, e.g., ['random-acts-of-pizza'].",
        },
    )

    # 全局覆盖参数：对所有任务统一生效的配置覆盖
    overrides: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Overrides for all tasks.",
        },
    )

    # 按任务覆盖参数：仅对特定任务生效的配置覆盖（键为任务名）
    task_overrides: dict[str, dict] = field(
        default_factory=dict,
        metadata={
            "help": "Overrides for specific tasks, allowing customization of task configurations.",
        },
    )

    # 校验逻辑：直接复用基类的递归校验
    def validate(self) -> None:
        super().validate()

    # 把基准配置展开成逐任务的配置对象列表，便于上层迭代执行
    def to_cfg_list(self):
        """
        Convert the benchmark configuration to an iterator of task configurations.
        This allows for easy iteration over tasks with their respective configurations.
        """
        # 存放展开后的任务配置列表
        task_cfgs = []
        # 遍历基准下的每个任务名
        for task_name in self.tasks:
            # 构造 MLEBench 任务配置：任务名、所属基准、全局覆盖与该任务专属覆盖
            task_cfg = MLEBenchTaskConfig(
                name=task_name,
                benchmark=self.name,
                **self.overrides,
                **self.task_overrides[task_name] if task_name in self.task_overrides else {},
            )

            # 收集该任务配置
            task_cfgs.append(task_cfg)
        # 返回全部任务配置列表
        return task_cfgs
