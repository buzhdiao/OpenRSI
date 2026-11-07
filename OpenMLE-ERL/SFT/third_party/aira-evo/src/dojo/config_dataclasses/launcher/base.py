# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 dataclass 装饰器与 field：声明数据类字段及其元数据
from dataclasses import dataclass, field

# 导入 SI 与 MISSING：OmegaConf 的插值符号与必填哨兵（本文件仅使用默认值字段）
from omegaconf import SI, MISSING

# 导入配置基类：提供哈希/校验等通用能力
from aira_core.config.base import BaseConfig


# @dataclass：自动生成构造/比较等方法
@dataclass
# 作业启动器配置类：控制任务提交与监控行为的运行期选项
class LauncherConfig(BaseConfig):
    # 是否等待作业完成：False 时提交后立即返回（异步提交）
    await_completion: bool = field(
        default=True,
        metadata={
            "help": "Whether to await completion of the job. If False, the job will be submitted and the function will return immediately.",
            "exclude_from_hash": True,
            "exclude_from_executor": True,
        },
    )

    # 调试模式开关：True 时不真正提交作业，仅打印将要执行的命令
    debug: bool = field(
        default=False,
        metadata={
            "help": "Whether to run in debug mode. If True, the job will not be submitted and the commands will be printed instead.",
            "exclude_from_hash": True,
            "exclude_from_executor": True,
        },
    )

    # 是否监控作业运行状态的开关
    monitor_jobs: bool = field(
        default=True,
        metadata={
            "help": "Whether to monitor the jobs.",
            "exclude_from_hash": True,
            "exclude_from_executor": True,
        },
    )

    # 校验逻辑：直接复用基类的递归校验
    def validate(self) -> None:
        super().validate()
