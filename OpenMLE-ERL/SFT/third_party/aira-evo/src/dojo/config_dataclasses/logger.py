# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 dataclass 装饰器与 field 工厂
from dataclasses import dataclass, field

# 跨平台路径拼接
import os
# OmegaConf 字符串插值 SI：默认值中可引用其他配置项
from omegaconf import SI

# 通用配置基类
from aira_core.config.base import BaseConfig
# 获取日志根目录的工具函数
from dojo.utils.environment import get_log_dir


# 日志记录器配置：控制实验产物（日志、指标、checkpoint）写到哪里、以什么方式记录
@dataclass
class LoggerConfig(BaseConfig):
    # 实验产物（如日志、checkpoint）输出目录：日志根目录下按 用户/issue/实验 id 分层，SI 插值在解析时展开
    output_dir: str = field(
        default=os.path.join(
            get_log_dir(), SI("aira-dojo/user_${metadata.user}_issue_${metadata.git_issue_id}/${id}")
        ),
        metadata={
            "help": "The output directory where experiment artifacts (e.g. logs, checkpoints) will be written.",
            "exclude_from_hash": True,
        },
    )
    # 是否把日志同时输出到标准输出（控制台）
    use_console: bool = field(
        default=True,
        metadata={
            "help": "Whether to log to stdout.",
            "exclude_from_hash": True,
        },
    )
    # 是否把指标上报到 wandb.ai 在线实验追踪平台
    use_wandb: bool = field(
        default=True,
        metadata={
            "help": "Whether to log to wandb.ai.",
            "exclude_from_hash": True,
        },
    )
    # 是否在本地以 JSON 格式保存日志/结果文件
    use_json: bool = field(
        default=True,
        metadata={
            "help": "Whether to save files locally in JSON format",
            "exclude_from_hash": True,
        },
    )
    # wandb 上的实体（团队/用户）名
    wandb_entity: str | None = field(
        default="aira-dojo",
        metadata={
            "help": "entity name in wandb.ai",
            "exclude_from_hash": True,
        },
    )
    # wandb 上的项目名，用于归集同一批实验
    wandb_project_name: str | None = field(
        default="hillclimbing-mlebench",
        metadata={
            "help": "Project name in wandb.ai.",
            "exclude_from_hash": True,
        },
    )

    # 附加到实验上的标签，便于 wandb 中筛选分组
    tags: list[str] = field(
        default_factory=lambda: [],
        metadata={
            "help": "Tags to add to the experiment.",
            "exclude_from_hash": True,
        },
    )

    # 是否记录细粒度统计（mean/std/min/max）；统计量过多会刷屏 wandb 面板，故做成可选
    detailed_logging: bool = field(
        default=False,
        metadata={
            "help": "Having mean/std/min/max can clutter wandb so we make it optional.",
            "exclude_from_hash": True,
        },
    )

    # 是否在启动时把完整配置打印到标准输出（便于复现）
    print_config: bool = field(
        default=True,
        metadata={
            "help": "Whether to print the config to stdout.",
            "exclude_from_hash": True,
        },
    )
    # 是否把环境变量一并写入配置文件（利于排查环境问题）
    write_env_vars: bool = field(
        default=True,
        metadata={
            "help": "Whether to write the environment variables to the config file.",
            "exclude_from_hash": True,
        },
    )

    # 配置校验入口
    def validate(self) -> None:
        # 复用父类校验逻辑，本类无额外约束
        super().validate()
