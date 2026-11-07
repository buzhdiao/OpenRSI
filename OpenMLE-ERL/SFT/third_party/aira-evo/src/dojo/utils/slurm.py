# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 os，用于读取 SLURM 相关环境变量
import os
# 导入 submitit 的作业环境对象，用于解析 SLURM 作业信息
from submitit import JobEnvironment


# 获取当前 SLURM 作业 ID（用于日志与结果目录命名等）
def get_slurm_id() -> str:
    # 普通作业直接取环境变量 SLURM_JOB_ID；数组作业（array job）则用 JobEnvironment 取子任务 ID
    return (
        os.environ.get("SLURM_JOB_ID", "")
        if os.environ.get("SLURM_ARRAY_JOB_ID", None) is None
        else str(JobEnvironment().job_id)
    )
