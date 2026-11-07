# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# JSON 序列化/反序列化：保存与加载配置
import json
# 导入 dataclass 装饰器与 field 工厂
from dataclasses import dataclass, field

# OmegaConf：II 插值、MISSING 必填占位符与主入口
from omegaconf import II, MISSING, OmegaConf

# 通用配置基类
from aira_core.config.base import BaseConfig
# 从字典递归还原 dataclass 的工具函数
from dojo.config_dataclasses.utils import dataclass_from_dict
# 日志配置
from dojo.config_dataclasses.logger import LoggerConfig
# 实验元数据配置
from dojo.config_dataclasses.metadata import MetadataConfig
# 任务配置
from dojo.config_dataclasses.task.base import TaskConfig
# 求解器配置
from dojo.config_dataclasses.solver.base import SolverConfig
# 代码执行解释器配置
from dojo.config_dataclasses.interpreter.base import InterpreterConfig


# 单次实验（Run）的完整配置：聚合日志、元数据、任务、求解器与解释器五大部分
@dataclass
class RunConfig(BaseConfig):
    # 本次实验的唯一 ID：通过 generate_id 解析器根据配置内容自动生成（ID 本身不参与哈希以免循环依赖）
    id: str = field(
        default=II("generate_id: "),
        metadata={"help": "Unique ID to use for an experiment.", "exclude_from_hash": True},
    )
    # 本 Run 所属 Runner 作业的唯一 ID：归到同一 meta_id 下的所有 Run 聚合即一次完整实验的全部结果
    meta_id: str = field(
        default="",
        metadata={
            "help": """
            Unique ID of the Runner job that this Run is part of. Aggregating all runs
            under a single meta_id allows to get the full results of a single experiment.
            """,
            "exclude_from_hash": True,
        },
    )
    # 日志记录器配置（输出目录、wandb 等）
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    # 实验元数据（用户、种子、commit 等）
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    # 要执行的任务配置
    task: TaskConfig = field(default_factory=TaskConfig)
    # 求解该任务的智能体策略配置
    solver: SolverConfig = field(default_factory=SolverConfig)
    # 代码执行解释器配置
    interpreter: InterpreterConfig = field(default_factory=InterpreterConfig)

    def save(self) -> None:
        """
        For distributed training, be careful to only call this function on the main process.
        """
        # 把配置对象转为普通字典容器（枚举转字符串，保证 JSON 可序列化）
        config_dict = OmegaConf.to_container(
            OmegaConf.structured(self),
            enum_to_str=True,
        )
        # 写入输出目录下的 dojo_config.json（留档便于复现）
        with open(f"{self.logger.output_dir}/dojo_config.json", "w") as file:
            json.dump(config_dict, file)

    @classmethod
    def load_from_json(cls, json_path: str) -> "RunConfig":
        """
        Load a RunConfig from a JSON file.
        """
        # 读取 JSON 配置文件并反序列化为字典
        with open(json_path, "r") as file:
            config_data = json.load(file)
        # 递归还原为 RunConfig 对象
        return RunConfig.from_dict(config_data)

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        # 借助工具函数把字典按字段类型递归重建 dataclass
        return dataclass_from_dict(cls, data)
