# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 asyncio：异步协调作业提交与返回
import asyncio
# 导入 itertools：笛卡尔积，用于枚举扫参组合
import itertools
# 导入 logging：标准日志模块
import logging
# 导入 os：切换工作目录、设置环境变量
import os
# 导入 copy：深拷贝配置对象，隔离各扫参组合
import copy
# 导入 sys：系统相关模块（保留以备扩展）
import sys
# 导入 importlib：动态导入模块（保留以备扩展）
import importlib
# 导入 inspect：获取模块源码路径
import inspect
# 导入 glob：按模式匹配文件（快照时包含 mlebench 相关文件）
import glob
# 导入 subprocess：调用外部命令（复制 aira-core 代码到快照）
import subprocess
# 导入 Path：路径对象操作
from pathlib import Path
# 导入 asdict：把数据类实例转为字典
from dataclasses import asdict
# 导入 datetime：生成快照目录的时间戳
from datetime import datetime
# 导入 time：统计快照制作耗时
import time

# 导入 RsyncSnapshot：用 rsync 把仓库快照到独立目录，隔离作业运行代码与可变原目录
from submitit.helpers import RsyncSnapshot

# 导入 hydra：命令行配置框架入口
import hydra
# 导入 jinja2：模板引擎（保留以备扩展）
import jinja2
# 导入 submitit：向 SLURM 等集群调度器提交作业的库
import submitit
# 导入 monitor_jobs：阻塞监控已提交作业直至结束
from submitit.helpers import monitor_jobs
# 导入 load_dotenv：加载 .env 环境变量
from dotenv import load_dotenv
# 导入 OmegaConf 配置类型与工具
from omegaconf import DictConfig, OmegaConf
# 导入自定义 OmegaConf 插值解析器的注册函数
from dojo.config_dataclasses.omegaconf.resolvers import register_new_resolvers

# 导入 rich 美化输出工具
from dojo.utils import rich_utils
# 导入日志根目录获取函数
from dojo.utils.environment import get_log_dir
# 导入单次运行(Run)配置数据类
from dojo.config_dataclasses.run import RunConfig

# 导入单次运行主流程 _main：它将作为作业入口函数被序列化后提交到集群执行
from dojo.main_run import _main as main_run
# 导入启动器基础配置类型
from dojo.config_dataclasses.launcher.base import LauncherConfig
# 导入 SLURM 启动器配置类型
from dojo.config_dataclasses.launcher.slurm import SlurmConfig
# 导入批量运行器(Runner)配置数据类
from dojo.config_dataclasses.runner import RunnerConfig
# 导入 git 快照工具：制作浅快照、获取仓库根目录
from dojo.utils.git import make_snapshot_shallow_git, get_git_top_level

# 加载 .env 环境变量
load_dotenv()
# 创建模块级 logger
log = logging.getLogger(__name__)

# 注册自定义 OmegaConf 插值解析器
register_new_resolvers()


# 提交一批作业：先制作代码快照（保证作业运行代码不受后续改动影响），再批量提交到集群
def launch_jobs(config_list: list[RunConfig], launcher_cfg: LauncherConfig):
    # Go to the git root directory
    # 记录当前工作目录，结束时恢复
    og_path = os.getcwd()
    # 获取 git 仓库根目录
    git_root = Path(get_git_top_level())
    # 切换到 git 根目录（rsync 与后续相对路径均以此为基准）
    os.chdir(git_root)

    # Make snapshot for the batch of jobs
    # 以当前时间生成快照目录的唯一字符串标识
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # 拼出快照目录路径：日志根目录/aira-dojo/snapshots/时间戳
    snapshot_path = Path(get_log_dir()) / "aira-dojo" / "snapshots" / f"{date_str}"
    # 创建快照目录（不允许已存在，避免覆盖历史快照）
    snapshot_path.mkdir(parents=True, exist_ok=False)
    # 调试日志：输出快照目标路径
    log.debug(f"Snapshotting to {snapshot_path}")

    # 记录快照开始时间
    start_time = time.time()
    # 用 RsyncSnapshot 上下文管理器把整个仓库（含子模块）复制到快照目录，进入后工作目录即变为快照目录
    with RsyncSnapshot(
        snapshot_dir=snapshot_path,
        with_submodules=True,
        exclude=[
            "*__pycache__",
            "*.mypy_cache",
        ],
        include=glob.glob(f"./src/dojo/tasks/mlebench/**", recursive=True),
    ):
        # Note 1: This copies all files of git repo at the `snapshot_path,
        # then `cd` into `snapshot_path` so os.getcwd() will return
        # `snapshot_path` and `snapshot_path` is the root of git

        # Note 2: Setting the PYTHONPATH to include the snapshot src folder
        # This is needed to actually use the code from the snapshot
        # otherwise it will use the code from the change-exposed folder
        # because of the editable installation of aira-dojo, aira-core and mle-bench

        # Note 3: PYTHONPATH parses multiple paths given with `:` char
        # So here we are actually bypassing our three editable packages
        # with snapshot/dojo, snapshot/aira_core and mlebench

        # 调试日志：输出快照制作耗时
        log.debug(f"Making snapshot took {time.time() - start_time} seconds")

        # Copy the aira-core code to the snapshot path
        # 延迟导入 aira_core 以定位其实际安装路径
        import aira_core

        # 定位 aira-core 包的 __init__.py 文件路径
        aira_core_init_file = Path(
            inspect.getsourcefile(aira_core)
        )  # This points to the __init__.py file of aira-core
        # 向上两级得到 aira-core 包的根目录
        aira_core_path = aira_core_init_file.parents[
            1
        ] 

        # Copy the aira-core code to the snapshot path
        # 把 aira-core 源码整目录复制进快照，使作业使用快照内代码
        subprocess.run(["cp", "-r", str(aira_core_path), str(snapshot_path)])

        # 重设 PYTHONPATH 指向快照内的源码路径，覆盖可编辑安装(editable install)的同名包
        os.environ["PYTHONPATH"] = (
            f"{snapshot_path}/src:{snapshot_path}/aira-core/src:{snapshot_path}/src/dojo/tasks/mlebench/mle-bench"  # This will override the packages that are installed in editable mode such that it uses the code from the snapshot
        )

        #####################
        # Create an Executor
        # 当前仅支持 SLURM 启动器配置
        if isinstance(launcher_cfg, SlurmConfig):
            # SLURM 日志目录：%j 为作业 ID 占位符，submitit 会自动替换
            slurm_folder = Path(get_log_dir()) / "slurm_logs" / "%j"
            # 创建 SLURM 执行器（作业提交器）
            executor = submitit.SlurmExecutor(folder=slurm_folder)
            # 把启动器配置转为字典，过滤掉空值与标记为不传给执行器的字段
            executor_kwargs = {
                key: val
                for key, val in asdict(launcher_cfg).items()
                if val and not launcher_cfg.__dataclass_fields__[key].metadata.get("exclude_from_executor", False)
            }
            # 将过滤后的参数应用到 SLURM 执行器（如分区、时限、节点数）
            executor.update_parameters(**executor_kwargs)
        else:
            # 不支持的启动器类型直接报错
            raise ValueError("Unsupported launcher configuration type")

        #####################
        ## Submit the jobs
        # 作业句柄列表
        jobs = []
        # 批量提交模式：同批作业共享 SLURM 作业数组(job array)前缀
        with executor.batch():
            # 遍历每份运行配置
            for run_cfg in config_list:
                # 将单次运行主流程与对应配置作为一个作业提交
                job = executor.submit(main_run, run_cfg)
                # 收集作业句柄
                jobs.append(job)

    # Move back to the original path
    # 恢复进入 with 前的原工作目录
    os.chdir(og_path)

    # 返回全部作业句柄
    return jobs


# 递归设置配置项：支持点号嵌套键（如 "metadata.seed"）
def override_config(config, key, value):
    # 键中含点号：说明是嵌套键
    if "." in key:
        # Handle nested keys like 'metadata.seed'
        # 按点号拆分键路径
        parts = key.split(".")
        # 若第一层属性存在
        if hasattr(config, parts[0]):
            # 递归进入下一层继续覆盖
            override_config(getattr(config, parts[0]), ".".join(parts[1:]), value)
        else:
            # Key doesn't exist
            # 键不存在则报错
            raise ValueError(f"Key '{key}' not found in config.")
    elif hasattr(config, key):
        # Handle top-level keys
        # 顶层键：用反射设置属性值
        setattr(config, key, value)
    else:
        # Key doesn't exist and isn't nested
        # 顶层键也不存在则报错
        raise ValueError(f"Key '{key}' not found in config.")


# 递归读取配置项：与 override_config 对应，同样支持点号嵌套键
def fetch_config(config, key):
    # 键中含点号：说明是嵌套键
    if "." in key:
        # Handle nested keys like 'metadata.seed'
        # 拆分键路径
        parts = key.split(".")
        # 若第一层属性存在
        if hasattr(config, parts[0]):
            # 递归读取剩余路径的值
            return fetch_config(getattr(config, parts[0]), ".".join(parts[1:]))
        else:
            # Key doesn't exist
            # 键不存在则报错
            raise ValueError(f"Key '{key}' not found in config.")
    elif hasattr(config, key):
        # Handle top-level keys
        # 顶层键：反射读取属性值
        return getattr(config, key)
    else:
        # Key doesn't exist and isn't nested
        # 均不存在则报错
        raise ValueError(f"Key '{key}' not found in config.")


# 异步主流程：把每份 Runner 配置展开为多份 Run 配置，再提交作业
async def _main(runner_configs: list[RunnerConfig]):
    # 汇总所有展开后的运行配置
    run_configs = []
    # 遍历每份 Runner 配置
    for runner_cfg in runner_configs:
        # Runner 中可含多个基准任务，逐个展开为任务配置
        for task_cfg in runner_cfg.benchmark.to_cfg_list():
            # 组装单次运行配置：任务来自展开结果，其余部分沿用 Runner 配置
            run_cfg = RunConfig(
                meta_id=runner_cfg.id,
                logger=runner_cfg.logger,
                metadata=runner_cfg.metadata,
                task=task_cfg,
                solver=runner_cfg.solver,
                interpreter=runner_cfg.interpreter,
            )

            # Resolve interpolations (e.g. experiment ID, time, etc.)
            # 转为 OmegaConf 结构以便解析插值
            run_cfg = OmegaConf.structured(run_cfg)
            # 解析插值引用（实验 ID、时间等）
            OmegaConf.resolve(run_cfg)
            # 转回数据类对象
            run_cfg = OmegaConf.to_object(run_cfg)
            # 校验该份运行配置的合法性
            run_cfg.validate()  # Make sure everything is valid

            # Add the run config to the list
            # 加入待提交列表
            run_configs.append(run_cfg)

    # 调试模式：只打印作业摘要而不真正提交（干跑/dry run）
    if runner_cfg.launcher.debug:
        # 调试模式下没有实际作业
        jobs = []

        # 汇总被扫参的键，外加任务名，用于摘要打印
        swept_keys = list(runner_cfg.vars.keys()) + ["task.name"]
        # Print a summary of the jobs that would be launched
        # 调试日志：干跑模式打印作业摘要
        log.debug("Dry run mode: printing job summary")
        # 遍历每份展开后的配置
        for cfg in run_configs:
            # 逐键输出键名与取值
            for key in swept_keys:
                # 打印键与值对齐后的行
                print(f"{key}{' ' * (30 - len(key))}{fetch_config(cfg, key)}")
            # 打印分隔线
            print("============" * 5)
    else:
        # 调试日志：开始提交作业
        log.debug("Launching jobs...")
        # 真正提交全部作业
        jobs = launch_jobs(run_configs, runner_cfg.launcher)

    # 返回作业句柄列表
    return jobs


# Hydra 入口装饰器：默认读取 configs/default_runner.yaml 作为配置基线
@hydra.main(version_base="1.3.2", config_path="configs", config_name="default_runner")
def main(_cfg: DictConfig):
    ## Validate and setup config
    # 1) Check structure
    # 按数据类结构实例化原始配置
    og_cfg: RunnerConfig = hydra.utils.instantiate(_cfg)

    ## Create a list of Runner configs, given the list of override variables
    # 存放每套扫参组合对应的 Runner 配置
    runner_configs = []
    # 读取命令行给出的扫参变量字典（变量名到候选值列表）
    cmd_vars = dict(og_cfg.vars)
    # 扫参变量名列表
    keys = list(cmd_vars.keys())
    # 对所有变量的候选值做笛卡尔积，枚举每一种组合
    for idx, values in enumerate(itertools.product(*(cmd_vars[key] for key in keys))):
        # 将组合值与变量名配对成字典
        single_vars_comb = dict(zip(keys, values))

        # 1) Apply override variables to the run config
        # 深拷贝原始配置，避免各组合之间相互污染
        runner_cfg: RunnerConfig = copy.deepcopy(og_cfg)
        # 逐个应用本组合的覆盖变量
        for k, v in single_vars_comb.items():
            # 按键路径覆盖配置值
            override_config(runner_cfg, k, v)

        # runner_cfg.validate()

        # 收集本组合的配置
        runner_configs.append(runner_cfg)

    # 运行异步主流程，提交全部作业
    jobs = asyncio.run(_main(runner_configs))
    # 过滤提交失败的作业句柄（未通过干跑测试的）
    jobs = [
        j for j in jobs if j is not None
    ]  # Filter out the jobs that failed to launch (i.e., didn't pass the dry run test)

    # 若配置要求监控作业
    if og_cfg.launcher.monitor_jobs:
        # 日志：开始监控
        log.info("Monitoring jobs...")
        # Monitor the jobs
        # 阻塞式监控所有作业直至全部结束
        monitor_jobs(jobs)
    else:
        # 日志：作业已提交但不监控
        log.info("Jobs launched successfully, but not monitored.")
        # 提取各作业的作业数组(job array)前缀，去重排序后拼接
        job_arrays = ", ".join(sorted(set(str(job.job_id).split("_", 1)[0] for job in jobs)))
        # 日志：汇报作业数量与所属作业数组
        log.info(f"Monitoring {len(jobs)} jobs from job arrays {job_arrays} \n")


if __name__ == "__main__":
    main()
