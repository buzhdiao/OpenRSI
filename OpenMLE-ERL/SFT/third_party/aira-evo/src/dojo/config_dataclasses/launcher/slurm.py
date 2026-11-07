# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 dataclass 装饰器与 field 工厂，用于声明数据类配置
from dataclasses import dataclass, field

# OmegaConf 配置库：SI 为字符串插值，MISSING 为必填占位符
from omegaconf import SI, MISSING

# 启动器（Launcher）配置基类
from dojo.config_dataclasses.launcher.base import LauncherConfig
# Slurm 环境探测工具：获取默认分区/账户/QoS
from dojo.utils.environment import (
    get_default_slurm_partition,
    get_default_slurm_account,
    get_default_slurm_qos
)


# Slurm 启动器配置：描述作业在 Slurm 集群上运行所需的资源与环境参数
@dataclass
class SlurmConfig(LauncherConfig):
    # Slurm 计费账户名，默认从当前环境探测
    account: str = field(
        default=get_default_slurm_account(),
        metadata={
            "help": "Slurm account name",
            "exclude_from_hash": True,
        },
    )
    # Slurm 的 QoS（服务质量）级别，控制作业优先级与限额
    qos: str = field(
        default=get_default_slurm_qos(),
        metadata={
            "help": "Quality of service level",
            "exclude_from_hash": True,
        },
    )
    # 作业申请分配的计算节点数量
    nodes: int = field(
        default=1,
        metadata={
            "help": "Number of nodes to allocate",
        },
    )
    # 每个节点上启动的 Slurm 任务数
    ntasks_per_node: int = field(
        default=1,
        metadata={
            "help": "Number of tasks per node",
            "exclude_from_hash": True,
        },
    )
    # 每个节点申请的 GPU 数量
    gpus_per_node: int = field(
        default=1,
        metadata={
            "help": "Number of GPUs per node",
        },
    )
    # 每个任务分配的 CPU 核心数
    cpus_per_task: int = field(
        default=24,
        metadata={
            "help": "Number of CPU cores per task",
            "exclude_from_hash": True,
        },
    )
    # 作业提交到的 Slurm 分区（队列）名
    partition: str = field(
        default=get_default_slurm_partition(),
        metadata={
            "help": "Slurm partition to use",
            "exclude_from_hash": True,
        },
    )
    # 作业最大运行时限（格式 HH:MM:SS），超时会被 Slurm 杀掉
    time: str = field(
        default="24:00:00",
        metadata={
            "help": "Maximum job runtime in format HH:MM:SS",
            "exclude_from_hash": True,
        },
    )
    # 每个节点申请的内存量（Slurm 格式字符串）
    mem: str = field(
        default="200G",
        metadata={
            "help": "Memory allocation per node",
            "exclude_from_hash": True,
        },
    )
    # requeue: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "Whether to requeue the job if it fails.",
    #         "exclude_from_hash": False,
    #     },
    # ) # Not used anywhere

    # 作业数组（job array）的最大并行度；None 表示不限制
    array_parallelism: int | None = field(
        default=None,
        metadata={
            "help": "Maximum number of parallel jobs to run.",
            "exclude_from_hash": True,
        },
    )

    # 配置校验入口
    def validate(self) -> None:
        # 直接复用父类 LauncherConfig 的校验逻辑，本类没有额外约束需要检查
        super().validate()
