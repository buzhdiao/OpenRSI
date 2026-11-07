# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 标准日志模块
import logging

# OmegaConf：把字典包装为可属性访问的配置对象
from omegaconf import OmegaConf

# 底层作业提交工具函数
from dojo.core.runners.slurm.slurm_utils import submit_job

# 模块级 logger
log = logging.getLogger(__name__)


# 基于 submitit 的 Slurm 作业启动器：按配置把命令提交到集群，可选等待完成
class SlurmSubmitit:
    def __init__(self, **cfg) -> None:
        # 把关键字参数包装为 OmegaConf 配置
        self.cfg = OmegaConf.create(cfg)
        # 已提交作业计数器，用于生成唯一作业名
        self._counter = 0

    async def launch(self, cmd, job_metadata=None):
        # 异步提交一个作业；cfg.await_completion 为真时阻塞等待作业结束
        cfg = self.cfg

        # 提交作业（命令、工作目录、日志目录、唯一作业名及其余 Slurm 参数）
        job = submit_job(
            command=cmd,
            working_dir=cfg.working_dir,
            log_dir=cfg.log_dir,
            slurm_job_name=f"{cfg.exp_name}_{self._counter}",
            **cfg.params,
        )
        # 提交后计数器自增
        self._counter += 1

        # 配置为需要等待完成
        if self.cfg.await_completion:
            # 记录"已提交，等待中"
            log.info(f"[Run {self._counter}] Submitted -- waiting")
            # 异步阻塞直到作业执行结束
            await job.awaitable().result()
        else:
            # 仅记录"已提交"
            log.info(f"[Run {self._counter}] Submitted.")

        # 返回作业句柄与元数据
        return job, job_metadata
