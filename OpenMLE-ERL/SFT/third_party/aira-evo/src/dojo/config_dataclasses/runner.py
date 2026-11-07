# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# JSON 序列化：保存配置
import json
# 深拷贝配置对象
import copy
# 导入 dataclass 装饰器与 field 工厂
from dataclasses import dataclass, field

# OmegaConf：II 插值、MISSING 必填占位符与主入口
from omegaconf import II, MISSING, OmegaConf

# 通用配置基类
from aira_core.config.base import BaseConfig
# 日志配置
from dojo.config_dataclasses.logger import LoggerConfig
# 元数据配置
from dojo.config_dataclasses.metadata import MetadataConfig
# 评测基准（benchmark）配置
from dojo.config_dataclasses.benchmark import BenchmarkConfig
# 求解器配置
from dojo.config_dataclasses.solver.base import SolverConfig
# 代码执行解释器配置
from dojo.config_dataclasses.interpreter.base import InterpreterConfig
# 作业启动器（本地/Slurm）配置
from dojo.config_dataclasses.launcher.base import LauncherConfig


# Runner（批处理调度器）配置：聚合各子配置，并支持对若干变量做网格扫参（sweep）
@dataclass
class RunnerConfig(BaseConfig):
    # 日志记录器配置
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    # 实验元数据配置
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    # 求解器配置
    solver: SolverConfig = field(default_factory=SolverConfig)
    # 代码执行解释器配置
    interpreter: InterpreterConfig = field(default_factory=InterpreterConfig)
    # 评测基准配置
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    # 作业启动器配置
    launcher: LauncherConfig = field(default_factory=LauncherConfig)
    # 扫参变量表：键为配置字段路径，值为要遍历的候选列表
    vars: dict[str, list] = field(
        default_factory=dict,
        metadata={
            "help": "Override variables to sweep over.",
            "exclude_from_hash": True,
        },
    )

    # 配置校验入口
    def validate(self) -> None:
        # 先执行父类的通用校验
        super().validate()

        # Check if there is any duplicated values in the vars dictionary
        # 逐个检查扫参变量
        for key, values in self.vars.items():
            # 候选列表为空（None 或 []）时无法扫参
            if not values:
                raise ValueError(f"Value for {key} is None or empty")
            # 列表长度与去重后集合长度不等说明有重复候选值
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicated values found in vars for key: {key}")

    def save(self) -> None:
        """
        For distributed training, be careful to only call this function on the main process.
        """
        # 把配置转为普通字典容器（枚举转字符串，保证 JSON 可序列化）
        config_dict = OmegaConf.to_container(
            OmegaConf.structured(self),
            enum_to_str=True,
        )
        # 写入输出目录下的 dojo_config.json（留档）
        with open(f"{self.logger.output_dir}/dojo_config.json", "w") as file:
            json.dump(config_dict, file)

    @property
    def id(self) -> str:
        """
        Generate a unique ID for the runner based on an unresolved config.

        This is not a robust hash at all, it will be missing some fields
        due to the fact that some fields are unresolved, but it is a
        simple way to aggregate the runs with the same config together
        at run time because we know that the fields overwritten in `vars`
        will be present here.
        """
        # 深拷贝自身，避免污染原始配置
        config = copy.deepcopy(self)
        # 取出所有不参与哈希的字段
        fields_to_exclude = RunnerConfig.fields_to_exclude_from_hash()
        # Accessing fields of config will resolve them
        # 逐个把排除字段重置为默认值（访问字段会触发插值解析）
        for field, field_type in fields_to_exclude:
            # 数值字段重置为 0（动态赋值以支持点分嵌套路径）
            if field_type in (int, float):
                exec(f"config.{field} = 0")
            # 布尔字段重置为 False
            elif field_type == bool:
                exec(f"config.{field} = False")
            # 其余类型重置为空字符串
            else:
                exec(f'config.{field} = ""')

        # 返回基于重置后配置计算的哈希作为 Runner ID
        return config.hash()
