# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 dataclass 装饰器与 field 工厂
from dataclasses import dataclass, field

# OmegaConf 插值语法：II("resolver:") 在配置解析时调用注册的解析器生成值
from omegaconf import II

# 通用配置基类
from aira_core.config.base import BaseConfig


# 实验元数据配置：记录启动脚本、issue、随机种子及运行环境的溯源信息
@dataclass
class MetadataConfig(BaseConfig):
    # 启动本实验所用脚本的标识 ID
    script_id: str = field(
        default="",
        metadata={"help": "ID of the script used to launch the experiment."},
    )
    # 用于实验归档追踪的 GitHub issue 编号，一个 issue 通常对应一组相关实验（如一次超参搜索）
    git_issue_id: str = field(
        default="",
        metadata={
            "help": "GitHub issue ID for experiment tracking. An issue ID typically corresponds to multiple related experiments (e.g. a hyperparameter sweep)"
        },
    )
    # 对本实验的文字描述
    description: str = field(
        default="",
        metadata={"help": "Description of the experiment."},
    )
    # 实验随机种子，默认 42，保证可复现性
    seed: int = field(default=42, metadata={"help": "Random seed for the experiment."})

    # Tracking information (generated automatically through OmegaConf interpolations)
    # 实验启动时间：get_current_time 解析器在解析配置时取当前时间
    launch_time: str = field(
        default=II("get_current_time:"),
        metadata={
            "help": "Date and time when the experiment was launched.",
            "exclude_from_hash": True,
        },
    )
    # 运行实验所用代码的根目录：读取环境变量 PWD（当前工作目录）
    base_path: str = field(
        default=II("oc.env:PWD"),
        metadata={"help": "Base path for code used to run the experiment.", "exclude_from_hash": True},
    )
    # 启动实验的用户名：读取环境变量 USER
    user: str = field(
        default=II("oc.env:USER"),
        metadata={
            "help": "User who launched the experiment.",
            "exclude_from_hash": True,
        },
    )

    # Useful debugging information (generated automatically through OmegaConf interpolations)
    # 本次运行对应的（最近一次）git commit 哈希，便于定位代码版本
    git_commit_id: str = field(
        default=II("get_git_commit_id:"),
        metadata={
            "help": "(Last) Commit id for this run. For Arrival experiments, this is generated using omegaconf interpolations.",
            "exclude_from_hash": True,
        },
    )
    # 实验运行所用的 PyTorch 版本号
    torch_version: str = field(
        default=II("get_torch_version:"),
        metadata={
            "help": "Version of PyTorch that the experiment was run with. For Arrival experiments, this is generated using omegaconf interpolations."
        },
    )
    # 本实验对应的 Slurm 作业 ID（集群调度器分配），非 Slurm 环境下为空
    slurm_id: str = field(
        default="",
        metadata={
            "help": "Slurm id for the experiment.",
            "exclude_from_hash": True,
        },
    )
