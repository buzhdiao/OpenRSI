# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 异步 IO：并发观察多个 Slurm 作业
import asyncio
# dataclass 工具：声明 JobResult
import dataclasses
# 标准日志模块
import logging
# 操作系统接口：设置环境变量
import os
# 子进程：在作业中执行 shell 命令
import subprocess
# 枚举类型：定义作业状态
from enum import Enum
# 类型注解工具
from typing import Callable, Optional, Tuple

# Submitit 库：向 Slurm 提交 Python 函数作业
import submitit
# Jinja2 模板：渲染作业命令行
from jinja2 import Template

# 模块级 logger
log = logging.getLogger(__name__)


# 作业状态枚举：对 Slurm/Submitit 状态的简化归类
class JobStatus(Enum):
    # 成功完成
    COMPLETED = "COMPLETED"
    # 失败
    FAILED = "FAILED"
    # 被取消
    CANCELLED = "CANCELLED"
    # 未知状态
    UNKNOWN = "UNKNOWN"


# 已结束作业的聚合信息（ID、元数据、状态、日志）
@dataclasses.dataclass
class JobResult:
    """Aggregate information about the finished job."""

    # Slurm 作业 ID
    job_id: str
    # 提交方关联到该作业的自定义元数据
    metadata: dict[str, str | int | float, bool]
    # 归一化后的作业状态
    status: JobStatus
    # 各子任务的标准输出日志
    log_out: list[str]
    # 各子任务的标准错误日志
    log_err: list[str]


# 维护一个 asyncio 任务池，用于异步观察 submitit 作业状态
class JobObserver:
    """Manages a pool of asyncio tasks for observing submitit job status."""

    # 类级单例引用
    shared = None  # Singleton

    def __init__(self):
        # We keep a list of tasks created by 'observe'
        # 保存 observe() 创建的所有观察协程任务
        self._observing_tasks: list[asyncio.Task] = []

    def observe(
        self,
        job: submitit.Job,
        metadata: Optional[dict[str, str | int | float, bool]] = None,
        callback: Optional[Callable[[JobResult], None]] = None,
        focus_rank: Optional[int] = None,
        poll_interval: int = 10,
    ) -> None:
        """
        Observe the status of a submitit job, and execute a callback when finished.

        Args:
            job (submitit.Job): The Submitit job to watch.
            metadata (dict): Some data you want to associate with this job.
            callback (Callable[[JobResult], None]): Called after job finishes with the JobResult.
            focus_rank (Optional[int]): If set, only fetch logs from that subtask index.
            poll_interval (int): How often (seconds) to poll the job status.
        """
        # 创建独立协程任务（不阻塞调用方），观察作业直到结束
        task = asyncio.create_task(
            self._observe_job(
                job=job, poll_interval=poll_interval, focus_rank=focus_rank, callback=callback, metadata=metadata
            )
        )
        # 记录任务，供 wait() 统一等待
        self._observing_tasks.append(task)

    async def wait(self) -> None:
        """Returns only when all observed jobs and their callbacks are complete."""
        # 存在观察任务时才并发等待所有任务结束
        if self._observing_tasks:
            await asyncio.gather(*self._observing_tasks)

    async def _observe_job(
        self,
        job: submitit.Job,
        poll_interval=30,
        focus_rank: Optional[int] = None,
        callback: Optional[Callable[[JobResult], None]] = None,
        metadata: Optional[dict[str, str | int | float, bool]] = None,
    ) -> None:
        """
        Loop that periodically checks the job until it's done,
        then calls the user-specified callback with a JobResult.
        """
        # 作业未结束则异步休眠一个轮询间隔
        while not job.done():
            await asyncio.sleep(poll_interval)

        # Once we exit the loop, the job is done from Submitit's perspective,
        # meaning it's not in [PENDING, RUNNING, REQUEUED, ...].
        # 取 Slurm 原始状态并统一大写（无状态则视为 UNKNOWN）
        slurm_state = job.state.upper() if job.state else "UNKNOWN"
        # 映射为简化 JobStatus
        status = self._map_slurm_state_to_job_status(slurm_state, job)

        # Gather logs (stdout & stderr). Focus rank is optional.
        # 抓取标准输出与标准错误日志
        log_out, log_err = get_logs(job, focus_rank)

        # Create the result object
        # 构造结果对象
        result = JobResult(
            job_id=job.job_id,
            metadata=metadata,
            status=status,
            log_out=log_out,
            log_err=log_err,
        )

        # 提供了回调才把结果交给回调处理
        if callback is not None:
            callback(result)

    def _map_slurm_state_to_job_status(self, slurm_state: str, job: submitit.Job) -> JobStatus:
        """
        Convert a Slurm or Submitit job state to simpler JobStatus states
        """
        # 正常完成
        if slurm_state == "COMPLETED":
            return JobStatus.COMPLETED

        # 被取消
        elif slurm_state in ("CANCELLED"):
            return JobStatus.CANCELLED

        # 作业抛出了异常 -> 失败
        elif job.exception() is not None:
            return JobStatus.FAILED

        # 失败/节点故障/超时 -> 失败
        elif slurm_state in ("FAILED", "NODE_FAIL", "TIMEOUT"):
            return JobStatus.FAILED

        # 其余状态一律归为未知
        return JobStatus.UNKNOWN


# 初始化模块级单例
JobObserver.shared = JobObserver()  # Initialize the singleton


def get_logs(job: submitit.Job, focus_rank: Optional[int] = None) -> Tuple[list[str], list[str]]:
    """
    Gathers job stdout/stderr logs as lists of strings.

    If focus_rank is given, only logs for that subtask index.
    Otherwise, all tasks in an array job (or the single task in a normal job).
    """
    # If a particular rank is requested, only retrieve logs from that subtask
    # 指定了 rank 时只抓取该子任务的日志
    if focus_rank is not None:
        # 取指定索引的子任务
        subjob = job.task(focus_rank)
        # 取标准输出（空则用空串兜底）
        out = subjob.stdout() or ""
        # 取标准错误（空则用空串兜底）
        err = subjob.stderr() or ""
        # 以单元素列表返回
        return [out], [err]

    # If the job has no subtasks (single job), gather from itself
    # 无子任务（普通单作业）时直接从自身收集
    if not job._sub_jobs:
        return [job.stdout() or ""], [job.stderr() or ""]

    # Otherwise, gather from each subtask in the array job
    # 否则逐个收集数组作业的每个子任务日志
    outs, errs = [], []
    # 遍历所有子任务
    for sub in job._sub_jobs:
        # 子任务标准输出
        out = sub.stdout() or ""
        # 子任务标准错误
        err = sub.stderr() or ""
        # 追加输出
        outs.append(out)
        # 追加错误
        errs.append(err)
    # 返回全部子任务日志
    return outs, errs


def submit_job(
    command: str,
    slurm_job_name: str,
    working_dir: str = ".",
    log_dir="slurm_logs",
    env_vars: Optional[dict[str, str]] = None,
    **slurm_kwargs,
):
    """
    Launches a SLURM job using submitit with the specified arguments.

    Args:
        command (str): Path to the Python file that serves as the job entry point.
        num_nodes (int): Number of nodes for the SLURM job.
        cpus_per_task (int): Number of CPUs per task.
        gpus_per_node (int): Number of GPUs per node.
        tasks_per_node (int): Number of tasks per node.
        timeout_min (int): Timeout for the job in minutes.
        job_name (str): Name of the SLURM job.
        account (str): SLURM account to use for the job.
        qos (str): Quality of Service (QoS) for the job.

    Returns:
        str: Job ID of the submitted SLURM job.
    """

    # Create a Submitit executor
    # 创建 Submitit 执行器，日志写入 log_dir
    executor = submitit.AutoExecutor(folder=log_dir)

    # 其余 Slurm 参数透传为附加参数
    slurm_parameters = slurm_kwargs
    # 指定作业启动时切换到的工作目录
    slurm_parameters["chdir"] = working_dir

    # SLURM configuration
    # 设置作业名与附加 Slurm 参数
    executor.update_parameters(slurm_job_name=slurm_job_name, slurm_additional_parameters=slurm_parameters)

    def job_function():
        # 需要注入环境变量时逐个写入进程环境
        if env_vars:
            for k, v in env_vars.items():
                os.environ[k] = str(v)

        # 打印即将执行的命令
        print(f"Running command: {command}")
        # 以 shell 执行命令，非零退出码抛异常
        subprocess.run(command, shell=True, check=True)

    # Submit the job
    # 提交函数到 Slurm 队列
    job = executor.submit(job_function)

    # 记录提交成功的作业 ID
    log.info(f"Job submitted with ID: {job.job_id}, {job}")
    # 返回作业句柄供后续观察
    return job


async def main():
    # 异步演示入口：批量提交按模板渲染的作业并收集结果（测试用）
    ENV_VARS = {}

    # 演示用的 Slurm 资源参数（账户/QoS/分区需替换为自己的）
    slurm_parameters = {
        "account": "<<<YOUR_SLURM_ACCOUNT>>>",  # Replace with your SLURM account
        "qos": "<<<YOUR_SLURM_QOS>>>",  # Replace with your SLURM QoS
        "nodes": 1,
        "ntasks_per_node": 1,
        "gpus-per-node": 0,
        "cpus-per-task": 24,
        "partition": "<<<YOUR_SLURM_PARTITION>>>",
        "time": "00:05:00",
    }

    # Test 1
    # job = submit_job(
    #     # command='python test_slurm.py -i test_input/text_desc.txt',
    #     command='python test_slurm_2.py -x 1 -y 2 -j 1',
    #     working_dir='.',
    #     log_dir='submitit_logs/%j', # %x = job name, %j = job ID
    #     slurm_job_name="ts23",
    #     **slurm_parameters
    # )

    # await job.awaitable().result()
    # log_out, log_err = get_logs(job, focus_rank=None)

    # print("~~~Stdout~~~")
    # print(log_out[0])

    # print("~~~Stderr~~~")
    # print(log_err[0])

    # Test 2
    # 测试 2：用 Jinja2 模板渲染参数化命令，并发提交多个作业
    # 带占位符的命令模板
    template_str = """python test_slurm_2.py -x {{ x }} -y {{ y }} -j {{ j }}"""
    # 编译模板
    template = Template(template_str)

    # 随机数：生成演示参数
    import random

    # 固定种子保证可复现
    random.seed(12345)

    # 收集异步等待任务
    tasks = []
    # 提交 4 个作业
    for j in range(1, 5):
        # 随机参数 x
        x = random.randint(1, 4)
        # 随机参数 y
        y = random.randint(1, 4)
        # 该作业的参数字典
        job_data_dict = {"x": x, "y": y, "j": j}
        # 渲染出实际命令行
        cmd = template.render(**job_data_dict)
        # 提交作业
        job = submit_job(
            # command='python test_slurm.py -i test_input/text_desc.txt',
            command=cmd,
            working_dir=".",
            log_dir="slurm_logs/%j",  # %x = job name, %j = job ID
            slurm_job_name="ts23",
            **slurm_parameters,
        )

        # Create an asyncio task from the job awaitable.
        # 包装协程：等待作业完成并带回其参数
        async def _wrapper(job, data_dict):
            # 阻塞等待至作业结束
            await job.awaitable().result()
            # 返回作业句柄与参数
            return job, data_dict

        # 创建并发等待任务
        task = asyncio.create_task(_wrapper(job, job_data_dict))
        # 加入任务列表
        tasks.append(task)

    # Process tasks as they complete.
    # 按完成顺序处理各任务
    for completed in asyncio.as_completed(tasks):
        # 取出先完成的作业与参数
        job, job_data = await completed

        # 打印作业参数
        print("~~~Job data~~~")
        print(job_data)

        # 抓取全部日志
        log_out, log_err = get_logs(job, focus_rank=None)

        # 打印标准输出
        print("~~~Stdout~~~")
        print(log_out[0])

        # 打印标准错误
        print("~~~Stderr~~~")
        print(log_err[0])


if __name__ == "__main__":
    # 以脚本运行时启动异步演示
    asyncio.run(main())
