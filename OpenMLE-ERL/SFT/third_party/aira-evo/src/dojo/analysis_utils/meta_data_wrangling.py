# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 defaultdict：按竞赛 ID 聚合统计时提供默认值容器
from collections import defaultdict
# 导入 concurrent.futures：进程池并行处理多个实验/元实验
import concurrent.futures
# 导入 copy：对象深拷贝（本文件中保留的通用导入）
import copy
# 导入 itertools：chain 等迭代工具，用于拍平嵌套列表
import itertools
# 导入 json：读写 journal.json 等 JSON 数据
import json
# 导入 math：isclose 等浮点比较工具
import math
# 导入 os：os.path.relpath 等路径工具
import os
# 导入 shutil：递归删除误生成的目录
import shutil
# 导入 yaml：读取 .hydra/config.yaml 实验配置
import yaml
# 导入 numpy：数组运算与 NaN 处理
import numpy as np
# 导入 pandas：构建与清洗实验数据 DataFrame
import pandas as pd
# 导入 Path：面向对象的路径操作
from pathlib import Path
# 导入类型标注工具：可调用/字典/列表/序列/元组/联合/可选/任意/集合
from typing import Callable, Dict, List, Sequence, Tuple, Union, Optional, Any, Set
# 导入 pickle：反序列化 submitit 提交时保存的任务 pickle
import pickle
# 导入 subprocess：执行 sacct 等 shell 命令
import subprocess
# 导入 re：正则表达式，解析日志与路径模式
import re
# 导入 mmap：内存映射大日志文件，高效查找关键行
import mmap
# 导入 datetime/timedelta：时间戳解析与耗时计算
from datetime import datetime, timedelta
# 导入 deque：定长双端队列，高效读取文件末尾 N 行
from collections import deque
# 导入 OmegaConf：本文件中预留的配置工具（未直接使用）
from omegaconf import OmegaConf

# 导入 mlebench 注册表：获取竞赛对象、排行榜与评分方向（lower is better 判定）
from mlebench.registry import Competition, Registry, registry

# 导入运行配置类：读取各实验的 dojo_config.json
from dojo.config_dataclasses.run import RunConfig
# 导入实验目录判断工具：识别实验目录与元实验目录
from dojo.utils.experiment_logs import is_experiment, is_meta_experiment
# 导入环境工具：获取 mlebench 数据根目录
from dojo.utils.environment import get_mlebench_data_dir
# 导入日志转 JSON 的函数：JOURNAL.jsonl -> journal.json
from dojo.analysis_utils.journal_to_tree import save_journal_log_as_json


# Section: Submitit Wrangling
#######################################


# 从元实验目录中提取全部实验的 slurm 任务 ID 列表
def metaexp2ids(metaexp_path: Path) -> List[str]:
    """
    Extract job IDs from a meta-experiment directory.

    Args:
        metaexp_path: Path to the meta-experiment directory

    Returns:
        List of job IDs
    """
    # 对元实验目录下每个有效实验子目录，读取其配置中的 slurm_id
    slurm_ids = [
        RunConfig.load_from_json(subdir / "dojo_config.json").metadata.slurm_id
        for subdir in metaexp_path.iterdir()
        if is_experiment(subdir)
    ]

    # 返回任务 ID 列表
    return slurm_ids


# 执行一条 bash 命令并返回其标准输出；失败时抛出带 stderr 的异常
def run_bash_command(command: str) -> str:
    """
    Run a bash command and return the output.

    Args:
        command: Bash command to run

    Returns:
        Command output

    Raises:
        Exception: If command fails
    """
    # 以 shell 方式运行命令，捕获 stdout/stderr 并以文本形式返回
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # 返回码为 0 表示命令成功，返回标准输出
    if result.returncode == 0:
        return result.stdout
    else:
        # 非 0 返回码视为失败，抛出携带错误输出的异常
        raise Exception(f"Error: {result.stderr}")


# 由作业数据渲染 stdout/stderr 的真实输出路径（展开 %a/%A 占位符）
def render_output_paths(job: Dict[str, Any]) -> Tuple[str, str]:
    """
    Render the output paths for stdout and stderr based on job data.

    Args:
        job: Dictionary containing job data

    Returns:
        Tuple of (stdout_path, stderr_path)
    """

    # 内部辅助：判断路径模板中的 %a/%A 模式是否有效（只含合法的数组作业占位符）
    def has_valid_pattern(pat):
        # 把模式字符列表转成集合
        set_pat = set(pat)
        # 数组作业路径模板允许出现的占位符集合：%a（任务号）与 %A（主作业号）
        required_patterns = {"%a", "%A"}
        # 模式必须是 %a/%A 的子集且其中的字符都属于这两个占位符
        return set_pat.issubset(required_patterns) and set_pat.issubset(pat)

    # 取出作业的 stdout 路径模板
    std_out = job.get("stdout")
    # 取出作业的 stderr 路径模板
    stderr = job.get("stderr")
    # 用正则提取 stdout 模板中的全部 %x 占位符
    pattern_stdout = re.findall(r"%[a-zA-Z]", std_out)
    # 用正则提取 stderr 模板中的全部 %x 占位符
    pattern_stderr = re.findall(r"%[a-zA-Z]", stderr)
    # 两个模板都有效时，自行展开占位符渲染真实路径
    if has_valid_pattern(pattern_stdout) and has_valid_pattern(pattern_stderr):
        # 数组作业的主作业 ID（%A 对应）
        base_job_id = job["array"]["job_id"]
        # 数组任务编号（%a 对应）
        task_id = job["array"]["task_id"]["number"]
        # 把 %A/%a 替换为具体 ID，得到 stdout 实际路径
        std_out_path = std_out.replace("%A", str(base_job_id)).replace("%a", str(task_id))
        # 同样方式渲染 stderr 实际路径
        std_err_path = stderr.replace("%A", str(base_job_id)).replace("%a", str(task_id))
    else:
        # Fallback to the old way of getting stdout and stderr paths
        # 模板无效时退回使用 slurm 预展开的路径字段
        std_out_path = job.get("stdout_expanded")
        std_err_path = job.get("stderr_expanded")
    # 返回 (stdout 路径, stderr 路径) 二元组
    return std_out_path, std_err_path


# 调用 sacct 查询指定任务 ID 的详细作业数据（JSON 解析后整理为字典）
def _get_job_data(job_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Get job data from Slurm for specified job IDs.

    Args:
        job_ids: List of job IDs

    Returns:
        Dictionary mapping job IDs to job data

    Raises:
        ValueError: If job data cannot be retrieved
    """
    # Build the comma-separated string of job IDs
    # 把任务 ID 列表拼成逗号分隔的字符串，供 sacct 的 -j 参数使用
    job_ids_str = ",".join(str(jid) for jid in job_ids)

    # 组装 sacct 命令：查询这些任务并输出全部字段（JSON 格式）
    cmd = ["sacct", "-j", job_ids_str, "--format=ALL", "--json"]

    # Convert the command list to a string (if run_bash_command expects a string)
    # 命令列表展平为字符串（run_bash_command 以 shell 字符串方式执行）
    cmd_str = " ".join(cmd)

    try:
        # Execute the sacct command
        # 执行 sacct 命令并解析其 JSON 输出
        result = run_bash_command(cmd_str)
        data = json.loads(result)
    except Exception as e:
        # 命令执行或解析失败时抛出带原因的 ValueError
        raise ValueError(f"Error retrieving job data: {e}")

    # Check that we received job information
    # 校验返回数据中确实包含作业信息
    jobs = data.get("jobs")
    if not jobs:
        raise ValueError("No job information found in the JSON response.")

    # 结果字典：完整任务 ID -> 任务数据
    results = {}

    for job in jobs:
        # Extract the basic fields from each job entry
        # 提取作业名称
        job_name = job.get("name")
        # 数组作业的主作业 ID
        base_job_id = job["array"]["job_id"]
        # 数组任务编号
        task_id = job["array"]["task_id"]["number"]
        # 拼出 "主作业ID_任务号" 形式的完整任务 ID 作为键
        full_job_id = f"{base_job_id}_{task_id}"

        # Extract time information directly from JSON
        # 直接从 JSON 中提取各时间字段（epoch 秒）
        submit_time_epoch = job.get("time", {}).get("submission")
        start_time_epoch = job.get("time", {}).get("start")
        end_time_epoch = job.get("time", {}).get("end")
        elapsed_seconds = job.get("time", {}).get("elapsed")

        # 提交时间转 ISO 字符串（缺失时记 Unknown）
        submit_str = datetime.fromtimestamp(submit_time_epoch).isoformat() if submit_time_epoch else "Unknown"
        # 开始时间转 ISO 字符串（缺失时记 Unknown）
        start_str = datetime.fromtimestamp(start_time_epoch).isoformat() if start_time_epoch else "Unknown"

        # 运行时长秒数转 "时:分:秒" 字符串（缺失时记 Unknown）
        elapsed_str = str(timedelta(seconds=elapsed_seconds)) if elapsed_seconds is not None else "Unknown"
        # job.get("stdout_expanded") and job.get("stderr_expanded") was not always giving me a valid path
        # So I will try, if possible to render the path myself. The reason is because there exist 2 ids:
        # job["array"]["job_id"] and job["job_id"]
        # These 2 don't always match and we use job["array"]["job_id"] for our stdout and stderr paths
        # but somehow slurm expands stdout_expanded and stderr_expanded with job["job_id"]
        # However, this is very specific to the current loggin path structure. So will attempt to find the pattern
        # and if not, I will resort to the old way of getting the stdout and stderr paths
        # 渲染 stdout/stderr 路径（优先自行展开占位符，失败则回退）
        std_out_path, std_err_path = render_output_paths(job)

        # Calculate WaitingTime
        # 计算排队等待时长（开始时间 - 提交时间）
        waiting_time_str = None
        if submit_time_epoch and start_time_epoch:
            try:
                # 提交时间转为 datetime
                dt_submit = datetime.fromtimestamp(submit_time_epoch)
                # 开始时间转为 datetime
                dt_start = datetime.fromtimestamp(start_time_epoch)
                # 二者差即等待时长
                waiting_delta = dt_start - dt_submit
                # 转为字符串形式保存
                waiting_time_str = str(waiting_delta)
            except ValueError:  # Should not happen with timestamps, but good practice
                pass

        # 汇总该任务的全部字段写入结果字典
        results[full_job_id] = {
            "job_name": job_name,
            "state": job.get("state"),
            "exit_code": job.get("derived_exit_code"),
            "stdout_path": std_out_path,  # See comment above where render_output_paths is called
            "stderr_path": std_err_path,  # See comment above where render_output_paths is called
            "JobID": full_job_id,  # Ensure JobID is included, matching previous _get_sacct_times output
            "Submit": submit_str,
            "Start": start_str,
            "Elapsed": elapsed_str,
            "WaitingTime": waiting_time_str,
        }

    # 返回 任务ID -> 任务数据 的字典
    return results


# 分批查询 slurm 作业数据并汇总（k 为每批任务数上限）
def get_slurm_data(job_ids: Union[str, List[str]], k: int = 10) -> Dict[str, Dict[str, Any]]:
    """
    Get comprehensive job data from Slurm for the specified job IDs.

    Args:
        job_ids: Job ID or list of job IDs
        k: Batch size for querying Slurm

    Returns:
        Dictionary mapping job IDs to job data
    """
    # 汇总结果的字典
    results = {}

    # 允许传入单个 ID 字符串，统一包装成列表
    if isinstance(job_ids, str):
        job_ids = [job_ids]

    # 按批大小 k 分批查询 slurm，避免单次查询任务数过多
    for i in range(0, len(job_ids), k):
        # 取出当前批次的任务 ID
        batch = job_ids[i : i + k]
        # 查询该批次任务数据
        batch_results = _get_job_data(batch)
        # 合并进总结果
        results.update(batch_results)

    # 返回汇总后的任务数据
    return results


# 从日志文件中提取 "Output dir:" 指向的实验输出目录
def extract_path(file_path: Path) -> Optional[str]:
    """
    Extract output directory path from a log file.

    Args:
        file_path: Path to log file

    Returns:
        Extracted output directory path, or None if not found
    """
    # 收集包含 "Output dir:" 的行
    matching_lines = []
    with open(file_path, "r") as f:
        # 用内存映射避免把大文件整体载入内存
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            # 逐行迭代映射内容
            for line in iter(mm.readline, b""):
                # 命中输出目录标记行则收集（去掉换行并解码）
                if b"- Output dir:" in line:
                    matching_lines.append(line.rstrip(b"\n").decode("utf-8"))

    # 匹配 "Output dir:" 后的绝对路径
    pattern = re.compile(r"Output dir:\s+(\/.*)")
    # 存放解析出的输出目录
    output_dirs = []

    for line in matching_lines:
        # 在每条命中行中搜索路径模式
        match = re.search(pattern, line)
        if match:
            # 提取捕获组中的路径
            path = match.group(1)
            # Strip ANSI color codes if present
            # 若路径带 ANSI 颜色码，截断至转义符前
            output_dirs.append(path.split("\x1b")[0])

    # 一条都没找到时告警并返回 None
    if len(output_dirs) == 0:
        print(f"WARNING: Output dir was not available in {str(file_path)}")
        return None

    # 返回最后一次出现的输出目录（最接近当前状态）
    return output_dirs[-1]


# 建立 submitit 任务 ID 到实验目录（相对路径）的映射
def get_submitit_logs2exp_mapping(mexp_dir: Union[str, Path]) -> Dict[str, str]:
    """
    Map submitit job IDs to experiment directories.

    Args:
        mexp_dir: Meta-experiment directory

    Returns:
        Dictionary mapping job IDs to experiment directories
    """
    # 字典推导：对每个有效实验子目录，取配置中的 slurm_id，映射到其相对路径
    job_id2experiment_id = {
        RunConfig.load_from_json(subdir / "dojo_config.json").metadata.slurm_id: subdir.relative_to(mexp_dir)
        for subdir in mexp_dir.iterdir()
        if is_experiment(subdir)
    }
    # 返回 任务ID -> 实验目录(相对路径) 的映射
    return job_id2experiment_id


# 从运行命令（RunConfig）中提取实验名
def get_exp_name_from_cmd(cmd: RunConfig) -> str:
    """
    Extract experiment name from a command string.

    Args:
        cmd: Command string

    Returns:
        Experiment name

    Raises:
        Exception: If experiment name not found
    """

    # 实验名即运行配置的 id 字段
    return cmd.id


# 从运行命令（RunConfig）中提取竞赛 ID
def get_competition_id_from_cmd(cmd: RunConfig) -> str:
    """
    Extract competition ID from a command string.

    Args:
        cmd: Command string

    Returns:
        Competition ID

    Raises:
        Exception: If competition ID not found
    """

    # 竞赛 ID 即任务配置中的 name 字段
    return cmd.task.name


# 从运行命令（RunConfig）中提取随机种子
def get_seed_from_cmd(cmd: RunConfig) -> str:
    """
    Extract seed from a command string.

    Args:
        cmd: Command string

    Returns:
        Seed value

    Raises:
        Exception: If seed not found
    """

    # 随机种子存放在运行配置的 metadata.seed 中
    return cmd.metadata.seed


# 获取指定任务在 slurm_logs 下的 stdout/stderr 日志文件相对路径
def get_submitit_stdout_stderr_paths(mexp_dir: Union[str, Path], job_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Get paths to stdout and stderr files for a job.

    Args:
        mexp_dir: Meta-experiment directory
        job_id: Job ID

    Returns:
        Tuple of (stdout_path, stderr_path), each may be None if not found
    """
    # 把元实验目录解析为绝对路径
    mexp_dir = Path(mexp_dir).resolve()
    # 初始化 stdout 路径为 None
    std_out = None
    # 初始化 stderr 路径为 None
    err_out = None

    # slurm 日志目录位于元实验目录上两级的 slurm_logs/<任务ID> 下
    slurm_logs_dir = mexp_dir.parent.parent / "slurm_logs" / job_id
    # 匹配该目录下的 .out 标准输出文件
    std_files = list(slurm_logs_dir.glob("*.out"))
    # 匹配该目录下的 .err 标准错误文件
    err_files = list(slurm_logs_dir.glob("*.err"))

    # used os instead of pathlib as they are not compatible with the relative_to method anymore
    # 找到 .out 文件时计算其相对元实验目录的路径
    if len(std_files) > 0:
        std_out = os.path.relpath(std_files[0], start=mexp_dir)
    # 找到 .err 文件时同样计算相对路径
    if len(err_files) > 0:
        err_out = os.path.relpath(err_files[0], start=mexp_dir)

    # 返回 (stdout 相对路径, stderr 相对路径)，未找到则为 None
    return std_out, err_out


# 通过 submitit 的提交 pickle 把任务 ID 关联到实验目录并提取命令/竞赛信息
def link_jobs_to_experiments(mexp_dir: Union[str, Path]) -> Dict[str, Tuple[str, str, Optional[str], str]]:
    """
    Link Slurm jobs to experiment directories and extract relevant information.

    Args:
        mexp_dir: Meta-experiment directory

    Returns:
        Dictionary mapping job IDs to tuples of (cmd, exp_name, exp_dir, competition_id)
    """
    # 解析元实验目录为绝对路径
    mexp_dir = Path(mexp_dir).resolve()
    # submitit 日志根目录：元实验目录上两级
    submitit_logs_path = mexp_dir.parent.parent / "slurm_logs"
    # 先建立 任务ID -> 实验目录 的映射
    job_id2exp_mapping = get_submitit_logs2exp_mapping(mexp_dir)

    # 递归查找所有 submitit 提交 pickle 文件
    pkl_files = list(submitit_logs_path.rglob("*submitted.pkl"))
    # 结果字典：任务ID -> (命令, 实验名, 实验目录, 竞赛ID)
    job_id2cmd__exp_name__exp_dir__competition_id = {}

    for pkl_path in pkl_files:
        # pickle 文件所在目录名即任务 ID
        job_id = pkl_path.parent.name
        # 该任务 ID 不在实验映射中时跳过（无法关联到实验目录）
        if job_id not in job_id2exp_mapping.keys():
            continue

        # 反序列化 submitit 提交对象
        with pkl_path.open("rb") as file:
            data = pickle.load(file)

        # Get the matching experiment directory
        # 提交函数的第一个参数即运行命令（RunConfig）
        cmd = data.args[0]
        # 从命令中提取竞赛 ID
        competition_id = get_competition_id_from_cmd(cmd)
        # 从命令中提取实验名
        exp_name = get_exp_name_from_cmd(cmd)
        # could have gotten it from cmd as well
        # 由任务 ID 映射出对应实验目录
        exp_dir = job_id2exp_mapping[job_id]

        # 把四元组信息按任务 ID 存入结果
        job_id2cmd__exp_name__exp_dir__competition_id[job_id] = (cmd, exp_name, exp_dir, competition_id)

    # 返回任务 ID 到实验信息的映射
    return job_id2cmd__exp_name__exp_dir__competition_id


# 从 stderr 日志末尾提取干净的错误消息（定位错误起点，找不到则退回末尾 k 行）
def try_get_clean_err_messsage(err_file: Path, k: int) -> Union[str, List[str]]:
    """
    Extract a clean error message from stderr logs.

    Args:
        err_file: Path to stderr file
        k: Maximum number of lines to analyze

    Returns:
        Clean error message or list of last lines if clean message not found
    """
    # 打开错误日志，只保留末尾 k 行（deque 定长队列自动丢弃更早的行）
    with err_file.open("r") as file:
        last_lines = deque(file, maxlen=k)

    # 收集错误消息相关行
    results = []
    # 标记是否已找到错误消息起点
    found = False
    for line in last_lines:
        # 首先识别 hydra 风格的错误起始标记行
        if not found and line.startswith("Error executing job with overrides:"):
            found = True

        # 找到起点后逐行收集，直到 submitit 的错误头出现为止
        if found:
            results.append(line)
            if line.startswith("submitit ERROR ("):
                break

        # 兜底：没有 hydra 标记时直接以 submitit 错误头为起点
        if not found and line.startswith("submitit ERROR ("):
            found = True

    # 一条都没匹配到时告警，退回返回原始末尾 k 行
    if len(results) == 0:
        print(f"WARNING: The start of the error message was not found in {str(err_file)}.")
        return last_lines

    # 拼接收集到的错误行并返回
    return "".join(results)


# 用 slurm 数据构建元实验的任务级 DataFrame（状态/耗时/日志路径/竞赛与命令）
def prepare_meta_exp_slurm_dataframe(metaexp_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Prepare a DataFrame with meta-experiment data.

    Args:
        metaexp_dir: Meta-experiment directory

    Returns:
        DataFrame with meta-experiment data
    """
    # 解析元实验目录为绝对路径
    metaexp_dir = Path(metaexp_dir).resolve()

    # Get job IDs
    # 提取全部任务 ID
    ids = metaexp2ids(metaexp_dir)
    # 打印处理的数组任务数量及主作业 ID
    print(f"Processing {len(ids)} array jobs associated with job id {ids[0].split('_')[0]}")

    # Get job data
    # 查询 slurm 拿到任务数据
    data = get_slurm_data(ids)
    # 转成 DataFrame：键为索引（任务ID），转置后一行一个任务
    df = pd.DataFrame(data).T

    # 重置索引并把任务号拆出来作为新索引
    df = df.reset_index()
    df["taskID"] = df["index"].str.split("_").str[1]
    df = df.set_index("taskID")
    df.drop(columns="index", inplace=True)

    # Validate job IDs and names
    # 校验所有任务共享同一主作业 ID（属于同一次数组提交）
    job_ids = df["JobID"].apply(lambda x: x.split("_")[0]).unique()
    assert len(job_ids) == 1, f"Not all jobs have the same base job id: {str(job_ids)}"
    print("Base job ID:", job_ids[0])

    # 校验所有任务共享同一作业名（即元实验名）
    job_names = df["job_name"].unique()
    assert len(job_names) == 1, f"Not all jobs have the same job name: {str(job_names)}"
    mexp_name = job_names[0]
    print("Meta exp name:", mexp_name)
    df.drop(columns="job_name", inplace=True)

    # Clean up DataFrame
    # 复制一份作为清理后的 DataFrame
    df_clean = df.copy()

    # Add submitit stdout/stderr paths
    # 为每个任务补充 submitit 的 stdout/stderr 相对路径
    std_outs = []
    err_outs = []

    for job_id in df_clean["JobID"]:
        # 查找该任务的日志文件相对路径
        std_out, err_out = get_submitit_stdout_stderr_paths(metaexp_dir, job_id)
        # 收集 stdout 路径
        std_outs.append(std_out)
        # 收集 stderr 路径
        err_outs.append(err_out)

    # 写入新列
    df_clean["SubmititStdOut"] = std_outs
    df_clean["SubmititStdErr"] = err_outs

    # Parse state
    # 从状态字典中取当前状态的第一个值（如 COMPLETED/FAILED）
    df_clean["State"] = df["state"].apply(lambda x: x["current"][0])
    # 重命名原始状态列
    df_clean.rename(columns={"state": "state_dict"}, inplace=True)
    df_clean.rename(columns={"WaitingTime": "Waiting"}, inplace=True)

    # 只保留需要展示的列
    columns_to_show = ["JobID", "State", "Elapsed", "Waiting", "SubmititStdOut", "SubmititStdErr"]
    df_clean = df_clean[columns_to_show]
    # 索引转为数值类型以便按任务号排序
    df_clean.index = pd.to_numeric(df_clean.index)
    df_clean.sort_index(inplace=True)

    # Link jobs to experiments
    # 建立任务 ID 到实验信息的映射
    job_id2info = link_jobs_to_experiments(metaexp_dir)

    # 按任务 ID 逐列补充竞赛 ID、实验目录名与命令
    df_clean["CompID"] = df_clean["JobID"].apply(lambda job_id: job_id2info[job_id][3])
    df_clean["ExpDir"] = df_clean["JobID"].apply(lambda job_id: Path(job_id2info[job_id][2]).resolve().stem)
    df_clean["Cmd"] = df_clean["JobID"].apply(lambda job_id: job_id2info[job_id][0])

    # 返回清理后的 DataFrame
    return df_clean


# 分析失败任务：附加 stderr/stdout 日志尾部并细化状态（区分 TIMEOUT）
def analyze_failures(
    df_clean: pd.DataFrame, mexp_dir: Union[str, Path], std_err_k: int = 300, std_out_k: int = 50
) -> pd.DataFrame:
    """
    Analyze failures in the meta-experiment.

    Args:
        df_clean: DataFrame with meta-experiment data
        mexp_dir: Meta-experiment directory
        std_err_k: Maximum number of stderr lines to analyze
        std_out_k: Maximum number of stdout lines to analyze

    Returns:
        DataFrame with failure analysis added
    """
    # 解析元实验目录为绝对路径
    mexp_dir = Path(mexp_dir).resolve()
    # 复制原始 DataFrame 以免修改调用方数据
    df_with_logs = df_clean.copy()

    # 三列新数据的收集器：stdout 尾部、stderr 尾部、结合日志判定的状态
    std_outs, std_errs, state_logs = [], [], []

    for index, row in df_with_logs.iterrows():
        # 非 FAILED 状态的任务无需读取日志，用占位符填充
        if row["State"] != "FAILED":
            std_outs.append("/")
            std_errs.append("/")
            state_logs.append(row["State"])
            continue

        # 取出失败任务的 ID
        job_id = row["JobID"]
        # 取出其竞赛 ID
        comp_id = row["CompID"]

        # Retrieve the first .err and .out files from the slurm logs directory
        # 由相对路径拼出该任务的 .err 与 .out 日志文件路径
        err_file = mexp_dir / row["SubmititStdErr"]
        std_file = mexp_dir / row["SubmititStdOut"]

        # 提取整理后的错误消息（末尾 k 行内定位）
        std_err_message = try_get_clean_err_messsage(err_file, std_err_k)
        # 收集 stderr 尾部
        std_errs.append(std_err_message)

        # Get the standard output details
        # 读取 stdout 末尾 k 行作为输出尾部
        with std_file.open("r") as file:
            last_stdout_lines = deque(file, maxlen=std_out_k)
        std_outs.append(last_stdout_lines)

        # 错误消息中包含超时未被重新排队的信息时，把状态细化为 TIMEOUT
        if (
            isinstance(std_err_message, str)
            and "UncompletedJobError: Job not requeued because: timed-out" in std_err_message
        ):
            state_logs.append("TIMEOUT")
        else:
            # 其他情况沿用 slurm 原始状态
            state_logs.append(row["State"])

    # 把三列新数据写回 DataFrame
    df_with_logs["StdOut_tail"] = std_outs
    df_with_logs["StdErr_tail"] = std_errs
    df_with_logs["StateFromLogs"] = state_logs

    # 返回带日志信息的 DataFrame
    return df_with_logs


# 打印按竞赛 ID 分组的状态分布摘要（交叉表）
def print_failure_summary(df_with_logs: pd.DataFrame) -> None:
    """
    Print a summary of failures in the meta-experiment.

    Args:
        df_with_logs: DataFrame with failure analysis
    """
    # 打印标题
    print("State distribution by competition ID:")
    # 用交叉表展示 竞赛ID x 日志状态 的分布
    print(pd.crosstab(df_with_logs["CompID"], df_with_logs["StateFromLogs"]))


# 打印失败任务的详细日志（按开关展示 stderr/stdout）
def print_detailed_failures(df_with_logs: pd.DataFrame, show_std_err: bool = True, show_std_out: bool = False) -> None:
    """
    Print detailed information about failed jobs.

    Args:
        df_with_logs: DataFrame with failure analysis
        show_std_err: Whether to show stderr logs
        show_std_out: Whether to show stdout logs
    """
    # 筛选出失败任务并复制
    df_failed = df_with_logs[df_with_logs["StateFromLogs"] == "FAILED"].copy()
    # 按竞赛 ID 与任务号排序，便于阅读
    df_failed.sort_values(by=["CompID", "taskID"], inplace=True)

    # 打印失败任务总数
    print("Number of failed jobs:", len(df_failed))
    for index, row in df_failed.iterrows():
        # 取出任务 ID
        job_id = row["JobID"]
        # 取出竞赛 ID
        comp_id = row["CompID"]

        # Create a header for the job log output
        # 构造任务标题行与分隔线
        header = f"Job ID: {job_id} | Competition ID: {comp_id}"
        separator = "=" * len(header)
        print(separator)
        print(header)
        print(separator)

        # If enabled, print the standard error details with clear formatting
        # 按开关打印 stderr 日志（带独立分隔样式）
        if show_std_err:
            print("\n")
            std_err_header = "~~~~ Standard Error ~~~~"
            separator = "~" * len(std_err_header)
            print(separator)
            print(std_err_header)
            print(separator)
            print("\n")

            # 输出该任务的错误消息尾部
            print(row["StdErr_tail"])

        # If enabled, print the standard output details with clear formatting
        # 按开关打印 stdout 日志
        if show_std_out:
            print("\n~~~~ Standard Output ~~~~")
            separator = "~" * len("~~~~ Standard Output ~~~~")
            print(separator)
            # 输出该任务的输出尾部
            print(row["StdOut_tail"])

        # Add spacing after each job's logs for clarity
        # 每个任务日志后加空行分隔
        print("\n" + "-" * len(header) + "\n")


# 对单个元实验执行完整的 slurm 层分析（构建 DataFrame + 失败分析 + 摘要打印）
def analyze_slurm_meta_exp(
    metaexp_dir: Union[str, Path], show_failures: bool = True, show_std_err: bool = True, show_std_out: bool = False
) -> pd.DataFrame:
    """
    Analyze a meta-experiment.

    Args:
        metaexp_dir: Meta-experiment directory
        show_failures: Whether to show failure summary
        show_std_err: Whether to show stderr logs for failed jobs
        show_std_out: Whether to show stdout logs for failed jobs

    Returns:
        DataFrame with meta-experiment analysis based on slurm.
    """
    # 解析元实验目录为绝对路径
    metaexp_dir = Path(metaexp_dir).resolve()

    # Prepare DataFrame
    # 第一步：准备 slurm 数据 DataFrame
    df_clean = prepare_meta_exp_slurm_dataframe(metaexp_dir)

    # Analyze failures
    # 第二步：分析失败任务并附加日志尾部
    df_with_logs = analyze_failures(df_clean, metaexp_dir)

    # Print summary
    # 打印状态分布摘要
    print("\nSummary:")
    print_failure_summary(df_with_logs)

    # Print detailed failures if requested
    # 按需打印失败任务的详细日志
    if show_failures:
        print("\nDetailed failures:")
        print_detailed_failures(df_with_logs, show_std_err, show_std_out)

    # 返回带日志分析的 DataFrame
    return df_with_logs


# 并行分析多个元实验并汇总为单一 DataFrame
def analyze_all_slurm_meta_exps(meta_experiment_paths: List[Union[str, Path]]) -> pd.DataFrame:
    """
    Analyze multiple meta-experiments based on slurm.

    Args:
         metaexp_dirs: List of meta-experiment directories
    """

    # 收集各元实验的 DataFrame
    dfs = []
    # Resolve paths first
    # 先把所有元实验路径解析为绝对路径
    resolved_paths = [Path(meta_exp).resolve() for meta_exp in meta_experiment_paths]

    # Use ProcessPoolExecutor to parallelize the analysis
    # 用进程池并行分析各元实验
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit tasks for each meta experiment path
        # 为每个路径提交分析任务（关闭详细打印以减少输出）
        future_to_path = {
            executor.submit(
                analyze_slurm_meta_exp, path, show_failures=False, show_std_err=False, show_std_out=False
            ): path
            for path in resolved_paths
        }

        # Collect results as they complete
        # 按完成顺序收集各分析结果
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                # 等待该元实验的分析结果
                df = future.result()
                # 结果非空 DataFrame 时收集
                if df is not None and not df.empty:
                    dfs.append(df)
            except Exception as exc:
                # 单个元实验失败不影响整体
                print(f"Error processing {path}: {exc}")

    # 没有任何有效结果时返回空 DataFrame
    if not dfs:
        print("Warning: No valid DataFrames were generated from the meta-experiments.")
        return pd.DataFrame()  # Return an empty DataFrame if no results

    # 纵向拼接所有结果并重置索引
    final_df = pd.concat(dfs, ignore_index=True)
    final_df.reset_index(drop=True, inplace=True)
    # Ensure index is numeric before trying to sort. Check if index is already numeric.
    # 排序前确保索引是数值类型（taskID）
    if not pd.api.types.is_numeric_dtype(final_df.index):
        final_df.index = pd.to_numeric(final_df.index, errors="coerce")
        # Handle potential NaNs introduced by coercion if necessary, e.g., drop or fill them
        # 转换失败的行（NaN 索引）直接丢弃
        final_df.dropna(axis=0, subset=[final_df.index.name], inplace=True)

    # 按任务号排序后返回
    final_df.sort_index(inplace=True)
    return final_df


# Section: Extraction of data into dataframe
################################################


# 处理单个实验目录：把 JOURNAL.jsonl 转换为 journal.json 并返回其路径
def process_experiment_folder_journal_log(
    experiment_path: Path, regenerate_tree: bool, seconds_cutoff: float = None
) -> Union[Path, None]:
    """
    Processes a single experiment folder:
        1. converts JOURNAL.jsonl -> journal_tree.json
        2. returns the path to the new file if successful, else None.
    """
    # 原始日志 JSONL 的路径
    jsonl_path = experiment_path / "json" / "JOURNAL.jsonl"
    # 目标 JSON 文件路径
    json_path = experiment_path / "journal.json"
    if jsonl_path.exists():
        # 目标不存在或要求重新生成时才执行转换
        if (not json_path.exists()) or regenerate_tree:
            print(f"Processing {jsonl_path} -> {json_path}")
            save_journal_log_as_json(jsonl_path, experiment_path, "journal.json", seconds_cutoff)
        # 返回 journal.json 路径（无论刚生成还是已存在）
        return json_path
    else:
        # 找不到原始日志时提示并返回 None
        print(f"JOURNAL.jsonl not found in {experiment_path}")
        return None


# 从 .hydra/config.yaml 读取竞赛 ID
def load_competition_id(config_file: Union[str, Path]) -> str:
    """
    Reads the .hydra/config.yaml file and returns the competition_id
    under the 'task' field.
    """
    # 读取 YAML 配置文件
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # 返回 task 节点下的 competition_id（缺失时为 None）
    return config.get("task", {}).get("competition_id", None)


# 从 .hydra/config.yaml 读取实验种子
def load_seed(config_file: Union[str, Path]) -> str:
    """
    Reads the .hydra/config.yaml file and returns the seed of the experiment.
    """
    # 读取 YAML 配置文件
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # 返回实验种子（缺失时默认 -1）
    return config.get("seed", -1)


# 从 路径->方法名 映射中查询元实验对应的方法（算法）名
def load_method_name(meta_experiment_path: Path, path_to_method_name: Dict[str, str]) -> str:
    """
    Returns the method name for a given meta_experiment_path
    from the path_to_method_name mapping.
    """
    # 直接从 路径->方法名 映射字典中查找该元实验对应的方法名
    return path_to_method_name[str(meta_experiment_path)]


# 并行加载单个实验目录的数据（方法名/竞赛 ID/实验名/种子/节点列表）
def _load_experiment_data(
    meta_path: Path, experiment_folder: Path, path_to_method_name: Dict[str, str]
) -> Dict[str, Any]:
    """
    Loads data for a single experiment folder in parallel:
      - method_name (from path_to_method_name)
      - competition_id (from .hydra/config.yaml if present)
      - experiment_name (folder name)
      - nodes (from journal.json)
    Returns a dict of the form:
      {
        "method": ...,
        "competition_id": ...,
        "experiment_name": ...,
        "nodes": [...]
      }
    or None if journal.json doesn't exist or folder is invalid.
    """
    # Identify method name
    # 由元实验路径查出对应的方法（算法）名
    method_name = load_method_name(meta_path, path_to_method_name)

    # Attempt to load competition_id and seed
    # 尝试从实验配置中读取竞赛 ID 与种子
    config_file = experiment_folder / "dojo_config.json"
    competition_id = None
    seed = None
    if config_file.is_file():
        # 读取运行配置
        cfg = RunConfig.load_from_json(config_file)
        # 竞赛 ID 取任务名
        competition_id = cfg.task.name
        # 种子取 metadata.seed
        seed = cfg.metadata.seed

    # Load journal.json
    # journal.json 路径（Phase A 的转换产物）
    tree_json_path = experiment_folder / "journal.json"
    if not tree_json_path.is_file():
        return None  # skip if missing

    # 读取 journal.json 全部数据
    with open(tree_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 'nodes' should be a list of step dictionaries
    # 取出节点列表（每个元素是一个步骤/节点的字典）
    nodes = data.get("nodes", [])

    # 汇总该实验的全部数据并返回
    return {
        "method": method_name,
        "competition_id": competition_id,
        "experiment_name": experiment_folder.name,
        "nodes": nodes,
        "seed": seed,
        "ExpDir": str(experiment_folder.stem),
    }


# Phase A：并行把所有实验的 JOURNAL.jsonl 转换为 journal.json
def process_all_meta_experiments(
    meta_experiment_paths: List[Path],
    regenerate_trees: bool,
    max_processes: int | None = None,
    seconds_cutoff: float = None,
):
    """
    Phase A:
        1. Get all experiment folders in all provided metaexperiment folders.
        2. Creates one ProcessPoolExecutor for all experiment folders to
        converting JOURNAL.jsonl -> journal.json in parallel.
    """
    # 收集所有待处理的实验目录
    all_exp_dirs = []
    for meta_exp in meta_experiment_paths:
        # 转为 Path 对象
        meta_path = Path(meta_exp)
        # 非法元实验目录直接跳过
        if not is_meta_experiment(meta_path):
            print(f"Skipping invalid meta path: {meta_path}")
            continue
        # Collect all experiment dirs in this meta path
        # 遍历该元实验目录下的所有子目录
        for exp_dir in meta_path.iterdir():
            # 非实验目录跳过
            if not is_experiment(exp_dir):
                continue
            # Check if there has been a mistake and a dir was created.
            # 修复历史错误：journal.json 若被误建为目录则递归删除
            if Path(exp_dir / "journal.json").is_dir():
                shutil.rmtree(Path(exp_dir / "journal.json"))

            # 强制重新生成或目标尚不存在时才需要处理
            check_to_generate = regenerate_trees or (not (exp_dir / "journal.json").exists())
            if check_to_generate:
                all_exp_dirs.append(exp_dir)

    # Now use a single pool for all experiment folders
    # 用单个进程池并行转换所有实验的日志
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
        # 提交转换任务，记录 future 到目录的映射
        futures = {
            executor.submit(process_experiment_folder_journal_log, d, regenerate_trees, seconds_cutoff): d
            for d in all_exp_dirs
        }
        # 按完成顺序收集结果
        for future in concurrent.futures.as_completed(futures):
            exp_dir = futures[future]
            try:
                # 拿到转换产物路径
                result = future.result()
                if result:
                    print(f"Created: {result}")
            except Exception as exc:
                # 单个实验失败不影响其余
                print(f"{exp_dir} generated an exception: {exc}")


# Phase B：并行收集所有实验的树数据并聚合（实验列表/字段全集/最大步数）
def gather_all_meta_experiment_data(
    meta_experiment_paths: List[Union[str, Path]],
    path_to_method_name: dict,
    max_steps_cap: int = 500,
    max_processes: int | None = None,
):
    """
    Phase B: Iterates through each meta experiment folder and each experiment inside,
    but uses a single ProcessPoolExecutor to load them in parallel.
    Collects:
       - method_name,
       - competition_id,
       - experiment_name,
       - 'nodes' (list of step dictionaries).
    Returns:
       all_experiments, all_keys, overall_max_steps
    """
    # 1. Collect all (meta_path, experiment_folder) pairs
    # 第一步：收集所有 (元实验路径, 实验目录) 组合
    all_folders = []
    for meta_exp in meta_experiment_paths:
        # 转为 Path 对象
        meta_path = Path(meta_exp)
        # 非目录路径跳过
        if not meta_path.is_dir():
            print(f"Skipping invalid meta experiment path: {meta_path}")
            continue

        # 实验目录即元实验目录本身
        experiments_dir = meta_path
        if not experiments_dir.is_dir():
            print(f"Warning: no 'experiments' folder found in {meta_path}")
            continue

        # 收集所有有效实验目录
        for experiment_folder in experiments_dir.iterdir():
            if is_experiment(experiment_folder):
                all_folders.append((meta_path, experiment_folder))
    # 2. Parallel load each experiment folder
    # 第二步：并行加载每个实验目录的数据
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
        # 为每个 (元路径, 实验目录) 组合提交加载任务
        future_to_folder = {
            executor.submit(_load_experiment_data, meta_path, folder, path_to_method_name): (meta_path, folder)
            for (meta_path, folder) in all_folders
        }

        # 按完成顺序收集加载结果
        for future in concurrent.futures.as_completed(future_to_folder):
            meta_path, folder = future_to_folder[future]
            try:
                # 拿到单实验数据（可能为 None）
                exp_data = future.result()
                if exp_data is not None:
                    results.append(exp_data)
            except Exception as exc:
                # 加载失败只记录不影响其他实验
                print(f"Error loading {folder}: {exc}")

    # 3. Aggregate results -> all_experiments, all_keys, overall_max_steps
    # 第三步：聚合为实验列表、节点字段全集与最大步数
    all_experiments = []
    all_keys = set()
    overall_max_steps = 0

    for exp_data in results:
        # 取出该实验的节点列表
        nodes = exp_data["nodes"]
        # 步数即节点数
        step_count = len(nodes)
        # 维护全局最大步数
        if step_count > overall_max_steps:
            overall_max_steps = step_count

        # Collect all node keys
        # 收集所有节点出现过的字段名（并集）
        for node_dict in nodes:
            all_keys.update(node_dict.keys())

        # Append to final experiments list
        # 加入最终实验列表
        all_experiments.append(exp_data)

    # 4. Cap the maximum steps if it exceeds max_steps_cap
    # 最大步数不超过上限 max_steps_cap
    if overall_max_steps > max_steps_cap:
        overall_max_steps = max_steps_cap

    # 返回三元组：实验列表、字段全集、最大步数
    return all_experiments, all_keys, overall_max_steps


# 把实验列表构建成一行一个实验的 DataFrame（节点字段按 max_steps 对齐填充）
def build_dataframe_one_row_per_experiment(all_experiments: List[dict], all_keys: set, max_steps: int) -> pd.DataFrame:
    """
    Given:
      - all_experiments: list of experiment dicts (method, competition_id, experiment_name, nodes)
      - all_keys: union of all fields present in any node
      - max_steps: the maximum number of steps to store per experiment (lists padded to this length)

    Returns a DataFrame with:
      - columns for method, competition_id, experiment_name
      - columns for each node field in all_keys, whose values are lists of length max_steps
        (padded with NaN if an experiment has fewer than max_steps).
    """
    # 收集每行（每个实验）的数据字典
    rows = []

    for exp in all_experiments:
        # 取出该实验的方法名
        method_name = exp["method"]
        # 取出竞赛 ID
        competition_id = exp["competition_id"]
        # 取出随机种子（seed：控制随机性的可复现性参数）
        seed = exp["seed"]
        # 取出实验名
        experiment_name = exp["experiment_name"]
        # 取出节点列表
        nodes = exp["nodes"]
        # 取出实验目录名
        experiment_dir = exp["ExpDir"]

        # Prepare the row dict, starting with metadata
        # 行字典先放入标量元数据
        row_data = {
            "method": method_name,
            "competition_id": competition_id,
            "experiment_name": experiment_name,
            "seed": seed,
            "ExpDir": experiment_dir,
        }

        # For each possible node field, create a list of length max_steps
        # 为每个可能的节点字段建立长度为 max_steps 的列表，缺省填 NaN
        row_data.update({key: [np.nan] * max_steps for key in all_keys})

        # Fill in node field values for each step, up to max_steps
        # 把每个节点的字段值填入对应步的位置（超出上限截断）
        for i, node_dict in enumerate(nodes):
            if i >= max_steps:
                break
            for key, val in node_dict.items():
                # None 值统一写为 NaN
                row_data[key][i] = val if val is not None else np.nan

        # Optionally convert numeric lists to arrays
        # 纯数值列表转换为 numpy 数组，便于后续向量化计算
        for key in row_data:
            val = row_data[key]
            if isinstance(val, list) and val and isinstance(val[0], (int, float, bool, np.number)):
                row_data[key] = np.asarray(val)
            else:
                row_data[key] = val

        # Get padding masks
        # 生成填充掩码（padding mask）：标记真实步与填充位
        used_steps = min(len(nodes), max_steps)
        padding_mask = np.zeros(max_steps, dtype=bool)
        padding_mask[used_steps:] = True
        # 填充位为 True 的掩码
        row_data["padding_mask"] = padding_mask
        # 取反得到真实步掩码
        row_data["non_padding_mask"] = np.logical_not(padding_mask)

        # 收集该行
        rows.append(row_data)

    # Build the final DataFrame
    # 汇总为最终 DataFrame（一行一个实验）
    df = pd.DataFrame(rows)
    return df


# 端到端流程：转换树 -> 收集数据 -> 构建 DataFrame 并按 ExpDir 合并 slurm 信息
def collect_all_meta_experiments_in_one_df(
    meta_experiment_paths: List[Union[str, Path]],
    path_to_method_name: dict,
    max_steps_cap: int = 500,
    regenerate_trees: bool = False,
    seconds_cutoff: float = None,
    max_processes: int | None = None,
) -> pd.DataFrame:
    """
    Performs Phase A: Processing meta experiments to generate trees.
    Performs Phase B: Gathers meta experiments data.
    Performs Phase C: Processing each experiment to pad the series and
        build data frame with one row per experiment.
    """
    # Phase A: Process all the files to generate trees
    # 阶段 A：并行把所有 JOURNAL.jsonl 转换为 journal.json
    process_all_meta_experiments(meta_experiment_paths, regenerate_trees, max_processes, seconds_cutoff)

    # Phase B: Gather all the tree data
    # 阶段 B：并行收集全部实验的树数据
    all_experiments, all_keys, overall_max_steps = gather_all_meta_experiment_data(
        meta_experiment_paths,
        path_to_method_name,
        max_steps_cap=max_steps_cap,
        max_processes=max_processes,
    )

    # Build the final DataFrame with one row per experiment
    # 构建一行一个实验的主 DataFrame
    final_df = build_dataframe_one_row_per_experiment(all_experiments, all_keys, max_steps_cap)

    # Get df with slurm data
    # 解析全部元实验路径并获取 slurm 层数据
    meta_experiment_paths = [Path(path).resolve() for path in meta_experiment_paths]
    slurm_df = analyze_all_slurm_meta_exps(meta_experiment_paths)
    # # Join the data frame rows based on ExpDir
    # 按 ExpDir 外连接合并（校验一一对应，防止键重复）
    final_df = final_df.merge(slurm_df, on="ExpDir", how="outer", validate="one_to_one")

    # 返回合并后的最终 DataFrame
    return final_df


# 依据 slurm 信息过滤无效行（极早失败与 FAILED/节点故障状态）
def filter_dataframe_based_on_slurm(df: pd.DataFrame):
    # 复制 DataFrame，避免改动调用方数据
    df = df.copy()

    print("Original number of rows:", len(df))

    # Filter all rows where the method is nan AKA the experiment failed extremely early
    # and the journal.json file was not created.
    # 过滤 method 为 NaN 的行：对应极早失败、journal.json 未生成的实验
    before_method_filter = len(df)
    df = df[~pd.isna(df["method"])]
    after_method_filter = len(df)
    print(f"Filtered {before_method_filter - after_method_filter} rows based on early failure.")
    # Filter all rows where the state is FAILED
    # 过滤 slurm 状态为各类失败（FAILED/节点故障）的行
    before_state_filter = len(df)
    df = df[df["StateFromLogs"] != "FAILED"]
    df = df[df["StateFromLogs"] != "NODE_FAIL"]
    df = df[df["StateFromLogs"] != "NODE_FAILED"]
    after_state_filter = len(df)
    print(f"Filtered {before_state_filter - after_state_filter} rows based on state failure.")

    # 打印过滤后的剩余行数
    print("Number of rows after filtering based on slurm:", len(df))

    # 返回过滤后的 DataFrame
    return df


# 依据首节点到末节点的实际耗时过滤运行过快的行
def filter_dataframe_based_on_node_to_node_elapsed(
    df: pd.DataFrame,
    elapsed_time: str = "0-23:00:00",
    *,
    length_cutoff: int | None = None,
    show_filtered_times: bool = True,
    max_examples: int = 5,
) -> pd.DataFrame:
    """
    Drops rows whose first node to last node duration < `elapsed_time`.

    Examples that survive either because they are long enough
    or because they ran long enough are retained.
    """
    # 复制 DataFrame 避免副作用
    df = df.copy()
    # 计算每行的实际步数（length）：step 列最大值加一
    df["length"] = df["step"].apply(lambda x: int(pd.Series(x).max() + 1))

    # 把 "0-23:00:00" 形式的时长字符串规范为 Timedelta
    elapsed_td = pd.to_timedelta(elapsed_time.replace("-", " days "))

    # 计算首节点到末节点实际耗时（用真实步掩码过滤填充位）的内部函数
    def get_timestamp_diff(row):
        # 取出非填充位对应的时间戳
        timestamp_list = list(itertools.compress(row["timestamp"], row["non_padding_mask"]))
        # 首节点时间
        start = pd.to_datetime(timestamp_list[0])
        # 末节点时间
        end = pd.to_datetime(timestamp_list[-1])
        # 返回二者差值
        return end - start

    # 逐行计算首末节点耗时写入新列
    df["first_node_to_last_node_elapsed"] = df.apply(lambda x: get_timestamp_diff(x), axis=1)
    # 未给步数上限时：只要耗时不足即丢弃
    if length_cutoff is None:
        mask_to_drop = df["first_node_to_last_node_elapsed"] < elapsed_td
    else:
        # 给定步数上限时：仅在步数偏少且耗时不足时丢弃
        short_enough = df["length"] < length_cutoff
        ran_too_quick = df["first_node_to_last_node_elapsed"] < elapsed_td
        mask_to_drop = short_enough & ran_too_quick

    # 统计将被丢弃的行数
    n_dropped = mask_to_drop.sum()

    # 可选：打印若干被丢弃行的示例信息
    if show_filtered_times and n_dropped:
        dropped_examples = (
            df.loc[
                mask_to_drop,
                ["length", "Elapsed", "first_node_to_last_node_elapsed", "method", "seed", "competition_id"],
            ]
            .head(max_examples)
            .apply(
                lambda r: f"length={r.length}, slurm_elapsed={r.Elapsed}, first_node_to_last_node_elapsed={r.first_node_to_last_node_elapsed}, method={r.method}, seed={r.seed}, 'comp={r.competition_id}",
                axis=1,
            )
            .tolist()
        )
        example_str = ";  ".join(dropped_examples)
        more = " …" if n_dropped > max_examples else ""
        print(f"Dropped rows first_node_to_last_node_elapsed < {elapsed_td}): {example_str}{more}")

    # 丢弃目标行
    df = df[~mask_to_drop]
    # 打印过滤统计
    print(f"Filtered {n_dropped} rows")
    print("Remaining rows:", len(df))
    # 返回过滤后的 DataFrame
    return df


# 依据 slurm 的 Elapsed 时长过滤运行过快的行
def filter_dataframe_based_on_slurm_elapsed(
    df: pd.DataFrame,
    elapsed_time: str = "0-23:00:00",
    *,
    length_cutoff: int | None = None,  # NEW — e.g. 200
    show_filtered_times: bool = True,
    max_examples: int = 5,
) -> pd.DataFrame:
    """
    • Drops rows where 'method' is NaN.
    • Drops rows whose 'length' < `length_cutoff`
      *and* whose 'Elapsed' duration < `elapsed_time`.

    Examples that survive either because they are long enough
    or because they ran long enough are retained.
    """
    # 复制 DataFrame 避免副作用
    df = df.copy()
    print("Original rows:", len(df))

    # -- 1. basic validity check ------------------------------------------------
    # 基础有效性检查：剔除 method 为 NaN 的行
    df = df[~df["method"].isna()]
    print("After method NaN filter:", len(df))

    # -- 2. calculate 'length' ---------------------------------------------------
    # 计算实际步数：step 列最大值加一
    df["length"] = df["step"].apply(lambda x: int(pd.Series(x).max() + 1))

    # -- 3. normalise 'Elapsed' to timedelta ------------------------------------
    # 把目标时长字符串规范为 Timedelta
    elapsed_td = pd.to_timedelta(elapsed_time.replace("-", " days "))
    # slurm 的 Elapsed 列按数值（秒）或字符串两种形式分别转换
    if pd.api.types.is_numeric_dtype(df["Elapsed"]):
        df["Elapsed"] = pd.to_timedelta(df["Elapsed"], unit="s")
    else:
        df["Elapsed"] = pd.to_timedelta(df["Elapsed"].astype(str).str.replace("-", " days ", regex=False))

    # -- 4. build mask -----------------------------------------------------------
    # 构建丢弃掩码
    if length_cutoff is None:
        # 未给步数上限：slurm 耗时不足即丢弃
        mask_to_drop = df["Elapsed"] < elapsed_td
    else:
        # 给定步数上限：步数偏少且 slurm 耗时不足才丢弃
        short_enough = df["length"] < length_cutoff
        ran_too_quick = df["Elapsed"] < elapsed_td
        mask_to_drop = short_enough & ran_too_quick

    # 统计将被丢弃的行数
    n_dropped = mask_to_drop.sum()

    # -- 5. (optional) show examples --------------------------------------------
    # 可选：打印被丢弃行的示例
    if show_filtered_times and n_dropped:
        dropped_examples = (
            df.loc[mask_to_drop, ["length", "Elapsed", "method", "seed", "competition_id"]]
            .head(max_examples)
            .apply(
                lambda r: f"length={r.length}, elapsed={r.Elapsed}, {r.method}, {r.competition_id}, {r.seed}", axis=1
            )
            .tolist()
        )
        example_str = ";  ".join(dropped_examples)
        more = " …" if n_dropped > max_examples else ""
        print(f"Dropped rows (length < {length_cutoff} & elapsed < {elapsed_td}): {example_str}{more}")

    # -- 6. drop and report ------------------------------------------------------
    # 执行丢弃并报告统计
    df = df[~mask_to_drop]
    print(f"Filtered {n_dropped} rows with combined condition")
    print("Remaining rows:", len(df))
    # 返回过滤后的 DataFrame
    return df


# 返回代表"最差分数"的哨兵常量（按优化方向取正负号）
def _sentinel(lower_is_better: bool) -> float:
    """
    Returns a numeric sentinel:  +1e12   if lower scores are better,
                                -1e12   if higher scores are better.
    Using a single constant makes post‑hoc filtering easy.
    """
    # 越小越优时用巨大正数表示最差分；越大越优时用巨大负数
    return 1e12 if lower_is_better else -1e12


# 数据有效性过滤：为无有效节点的组写入最差哨兵分，并剔除步数过短的实验
def filter_dataframe_based_on_data_validity(df: pd.DataFrame, min_length: int = 5) -> pd.DataFrame:
    """
    Ensures that every (competition, method, seed) group contains at least one valid node.
    If **all** nodes in a group are invalid, their `metric` and `metric_info/score`
    arrays are overwritten with sentinel values so that the group is always treated
    as the *worst possible* submission in downstream ranking code.

    Parameters
    ----------
    df : pd.DataFrame
        Expected columns (same as in the original code):
            * step – list/array of ints
            * metric – list/array (validation score per node)
            * metric_info/score – list/array (test score per node)
            * is_buggy – list/array (bool or 0/1 per node)
            * competition_id, method, seed – scalars
    Returns
    -------
    pd.DataFrame
        A **copy** of the input with extra statistics columns and sentinel‑fixed rows.
    """

    # 复制原始 DataFrame，保持调用方数据不变
    df = df.copy()  # keep original intact
    print("Original number of rows:", len(df))
    # 计算每行实际步数
    df["length"] = df["step"].apply(lambda x: int(pd.Series(x).max() + 1))
    # 统计验证分数非 NaN 的节点数（有效验证节点数）
    df["num_validation_nodes"] = df["metric"].apply(lambda x: np.count_nonzero(~np.isnan(np.asarray(x, dtype=float))))
    # 统计测试分数非 NaN 的节点数（有效测试节点数）
    df["num_test_nodes"] = df["metric_info/score"].apply(
        lambda x: np.count_nonzero(~np.isnan(np.asarray(x, dtype=float)))
    )
    # 统计缺陷（buggy）节点数：NaN 按 0 计
    df["num_buggy_nodes"] = df["is_buggy"].apply(
        lambda x: int(np.nan_to_num(np.asarray(x, dtype=float), nan=0.0).sum())
    )
    # 正常节点数 = 总步数 - 缺陷节点数
    df["num_good_nodes"] = df["length"] - df["num_buggy_nodes"]
    # 正常节点占比
    df["good_node_proportion"] = df["num_good_nodes"] / df["length"]
    # 测试分数最大值（NaN 视为 -1e12 不参与最优）
    df["max_test_score"] = df["metric_info/score"].apply(
        lambda x: np.nan_to_num(np.asarray(x, dtype=float), nan=-1e12).max()
    )
    # 测试分数最小值（NaN 视为 +1e12 不参与最差）
    df["min_test_score"] = df["metric_info/score"].apply(
        lambda x: np.nan_to_num(np.asarray(x, dtype=float), nan=+1e12).min()
    )
    # 验证分数最大值
    df["max_validation_score"] = df["metric"].apply(
        lambda x: np.nan_to_num(np.asarray(x, dtype=float), nan=-1e12).max()
    )
    # 验证分数最小值
    df["min_validation_score"] = df["metric"].apply(
        lambda x: np.nan_to_num(np.asarray(x, dtype=float), nan=+1e12).min()
    )

    # 判断每行是否存在至少一个合法可选择的节点
    df["any_valid_node"] = df.apply(lambda row: bool(np.any(get_selection_mask(row))), axis=1)

    # print the number of rows with no valid nodes
    # 打印完全没有合法节点的行数
    num_no_valid_nodes = len(df[df["any_valid_node"] == False])
    print(f"Number of rows with no valid nodes: {num_no_valid_nodes}")

    # Overwrite groups with no valid node
    # 重建 mlebench 注册表（指向数据目录），用于查询竞赛信息
    new_registry = registry.set_data_dir(Path(get_mlebench_data_dir()))

    # Iterate over (competition, method, seed) groups
    # 把没有任何合法节点的组改写为本竞赛的最差分数
    print("Overwriting rows with no valid nodes to have worst score in competition...")
    # 分组键：竞赛、方法、种子
    group_cols = ["competition_id", "method", "seed"]
    for keys, group_idx in df.groupby(group_cols).groups.items():
        # 解包组键
        comp_id, method_name, seed_val = keys
        # 取出该组的全部行
        group_rows = df.loc[group_idx]

        # Quick exit if group already has ≥1 valid node
        # 组内已有合法节点时无需改写，直接跳过
        if group_rows["any_valid_node"].any():
            continue

        # Determine optimisation direction for this competition
        # 查询竞赛对象并判定优化方向（lower is better：分数越低越好）
        comp_obj: Any = new_registry.get_competition(comp_id)
        lower_is_better: bool = is_lower_better(comp_obj)
        # 取对应的哨兵值（sentinel：代表最差分数的固定常数）
        sentinel = _sentinel(lower_is_better)

        # ------------------------------------------------------------------
        # Actually overwrite the columns for every row in this group
        # ------------------------------------------------------------------
        # Set validation and test scores to the sentinel value
        # 把该组所有行的验证分数与测试分数整体覆写为哨兵值
        df.loc[group_idx, "metric"] = df.loc[group_idx, "metric"].apply(
            lambda arr: np.full_like(arr, sentinel, dtype=float)
        )
        df.loc[group_idx, "metric_info/score"] = df.loc[group_idx, "metric_info/score"].apply(
            lambda arr: np.full_like(arr, sentinel, dtype=float)
        )
        # Set the medal flags to 0
        # 把所有奖牌相关标志位清零（无奖牌/未超中位数）
        df.loc[group_idx, "metric_info/above_median"] = df.loc[group_idx, "metric_info/above_median"].apply(
            lambda arr: np.full_like(arr, 0, dtype=float)
        )
        df.loc[group_idx, "metric_info/any_medal"] = df.loc[group_idx, "metric_info/any_medal"].apply(
            lambda arr: np.full_like(arr, 0, dtype=float)
        )
        df.loc[group_idx, "metric_info/bronze_medal"] = df.loc[group_idx, "metric_info/bronze_medal"].apply(
            lambda arr: np.full_like(arr, 0, dtype=float)
        )
        df.loc[group_idx, "metric_info/silver_medal"] = df.loc[group_idx, "metric_info/silver_medal"].apply(
            lambda arr: np.full_like(arr, 0, dtype=float)
        )
        df.loc[group_idx, "metric_info/gold_medal"] = df.loc[group_idx, "metric_info/gold_medal"].apply(
            lambda arr: np.full_like(arr, 0, dtype=float)
        )

        # Re‑compute the min/max columns we created earlier so they stay in sync
        # 同步更新此前生成的最值列，保持一致
        df.loc[group_idx, "max_test_score"] = sentinel
        df.loc[group_idx, "min_test_score"] = sentinel
        df.loc[group_idx, "max_validation_score"] = sentinel
        df.loc[group_idx, "min_validation_score"] = sentinel

    # Filter all rows where the length is less than 5
    # 过滤步数少于下限的行（实验应至少有若干节点才有效）
    print(f"Filtering rows with length < {min_length}...")
    before_length_filter = len(df)
    df = df[df["length"] >= min_length]
    after_length_filter = len(df)

    print(
        f"Filtered {before_length_filter - after_length_filter} rows based on length. There should always be at least 5 nodes in the experiment."
    )
    print("Number of rows after filtering:", len(df))

    return df


# 判断值是否为序列类型（list/ndarray/Series）
def _is_sequence_like(x: Any) -> bool:
    """True for list / np.ndarray / pd.Series; False otherwise."""
    # 判断值是否为逐节点序列类型（列表/数组/Series）
    return isinstance(x, (list, np.ndarray, pd.Series))


# 把序列截断到前 max_len 个元素（保留原容器类型）
def _truncate_to_max(seq: Any, max_len: int) -> Any:
    """Return `seq` truncated to the first `max_len` elements if needed."""
    # 非序列类型原样返回
    if not _is_sequence_like(seq):
        return seq

    # Keep the **type** of the original container
    # numpy 数组用切片截断（保留 ndarray 类型）
    if isinstance(seq, np.ndarray):
        return seq[:max_len]
    # Series 用 iloc 截断（保留索引结构）
    if isinstance(seq, pd.Series):
        return seq.iloc[:max_len]
    return seq[:max_len]  # plain list


# 把每行所有序列列截断到最多 max_num_nodes 个节点
def filter_dataframe_to_have_limited_nodes(
    df: pd.DataFrame,
    max_num_nodes: int,
) -> pd.DataFrame:
    """
    Truncates every *sequence‑like* column so that each row contains
    at most `max_num_nodes` nodes/entries.

    Parameters
    ----------
    df : pd.DataFrame
        May contain a mix of scalar and per‑node sequence columns.
    max_num_nodes : int
        Hard upper limit on how many nodes are kept per row/column.

    Returns
    -------
    pd.DataFrame
        A **copy** of the input with all sequence columns safely truncated.
    """
    # 复制 DataFrame，保持调用方数据不变
    df = df.copy()  # keep caller’s DataFrame intact

    # ------------------------------------------------------------------
    # 1) Find all sequence‑like columns by inspecting the first
    #    non‑null value in each column (cheap and reliable).
    # ------------------------------------------------------------------
    # 第一步：通过检查每列第一个非空值识别所有序列型列（廉价且可靠）
    seq_cols = []
    for col in df.columns:
        first_non_na = df[col].dropna().iloc[0] if df[col].notna().any() else None
        # 第一个非空值是序列类型则把该列记为序列列
        if _is_sequence_like(first_non_na):
            seq_cols.append(col)

    # ------------------------------------------------------------------
    # 2) Truncate them row‑wise
    # ------------------------------------------------------------------
    # 第二步：逐行把每个序列列截断到上限
    for col in seq_cols:
        df[col] = df[col].apply(_truncate_to_max, max_len=max_num_nodes)

    # ------------------------------------------------------------------
    # 3) Bring the `length` column (if it exists) back in sync
    # ------------------------------------------------------------------
    # 第三步：若存在 length 列则同步压缩到上限
    if "length" in df.columns:
        df["length"] = df["length"].clip(upper=max_num_nodes)

    # 返回截断后的副本
    return df


# 新增一列：每个节点相对首个有效节点的时间差（忽略填充位）
def add_node_elapsed_from_first(
    df: pd.DataFrame,
    *,
    output_col: str = "seconds_from_first_node",
    unit: str = "seconds",  # "seconds"  (float)  or "timedelta"
) -> pd.DataFrame:
    """
    For every row, compute how much time has elapsed between each node’s
    timestamp and the *first *valid* node’s* timestamp.

    • Uses ``non_padding_mask`` to ignore padding.
    • Produces a sequence the same length as ``timestamp``.
      – First valid position → 0 (or Timedelta('0 days')).
      – Padding positions → np.nan.
    • Appends the result under ``output_col`` and returns a **copy**.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``timestamp`` (list‑like of str/datetime) and
        ``non_padding_mask`` (list/array of bools).
    output_col : str, default "seconds_from_first_node"
        Name of the new column to create.
    unit : {"seconds", "timedelta"}, default "seconds"
        Whether to return floats (seconds) or raw ``pd.Timedelta`` objects.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the extra column added.
    """
    # 复制 DataFrame 避免副作用
    df = df.copy()

    # 逐行计算相对首个有效节点的时间差的内部函数
    def _row_elapsed(row) -> List[Any]:
        # 该行的时间戳序列
        ts_raw: Sequence[Any] = row["timestamp"]
        # 该行的真实步掩码
        mask: Sequence[bool] = row["non_padding_mask"]

        # Guard – if *every* element is padding, return all‑NaNs
        # 全为填充位时返回全 NaN 列表
        if not any(mask):
            return [np.nan] * len(ts_raw)

        # Find the first *valid* timestamp
        # 定位第一个真实节点的时间戳作为基准
        first_valid_idx = next(idx for idx, m in enumerate(mask) if m)
        first_ts = pd.to_datetime(ts_raw[first_valid_idx], errors="coerce")

        # 收集每个位置的时间差
        elapsed_values: list[Any] = []
        for t, m in zip(ts_raw, mask):
            if not m:
                # 填充位写 NaN
                elapsed_values.append(np.nan)
                continue

            # 计算当前节点与首节点的时差
            td = pd.to_datetime(t, errors="coerce") - first_ts
            # 按单位参数决定返回秒数还是 Timedelta 对象
            elapsed_values.append(td.total_seconds() if unit == "seconds" else td)

        # 返回该行的时间差序列
        return elapsed_values

    # 逐行应用并写入输出列
    df[output_col] = df.apply(_row_elapsed, axis=1)
    # 返回带新列的副本
    return df


# 把序列切到指定长度（保留原容器类型）
def _slice_to_len(seq: Any, new_len: int) -> Any:
    """Return *seq* sliced to ``new_len`` while preserving its type."""
    # 非序列类型原样返回
    if not _is_sequence_like(seq):
        return seq
    # ndarray 用切片截断
    if isinstance(seq, np.ndarray):
        return seq[:new_len]
    # Series 用 iloc 截断
    if isinstance(seq, pd.Series):
        return seq.iloc[:new_len]
    return seq[:new_len]  # plain list


# 按时间上限截断：丢弃超阈值的节点并对齐全部序列列、同步 length
def truncate_dataframe_based_on_elapsed(
    df: pd.DataFrame,
    max_elapsed: str | int | float | pd.Timedelta,
    *,
    elapsed_col: str = "seconds_from_first_node",
) -> pd.DataFrame:
    """
    Remove every node that lies *after* ``max_elapsed`` from the first node.

    • Relies on an existing per‑row sequence column (default
      ``seconds_from_first_node``) produced by ``add_node_elapsed_from_first``.
    • Works on a **copy** of *df* – the original is untouched.
    • Truncates *every* list / ndarray / Series column row‑wise so that
      **all sequence columns stay perfectly aligned**.
    • Updates ``length`` (if present) to the new, shorter size.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *elapsed_col* with one scalar per node (float seconds
        or ``pd.Timedelta`` objects). Missing nodes should be ``np.nan``.
    max_elapsed : str | int | float | pd.Timedelta
        Threshold after which nodes are discarded.

        ── If *elapsed_col* holds **floats (seconds)**:
             – pass a number (seconds) **or** a timedelta‑like string
               e.g. ``"2:30:00"``.
        ── If *elapsed_col* holds **Timedeltas**:
             – pass a ``pd.Timedelta``, or a timedelta‑like string.

    elapsed_col : str, default "seconds_from_first_node"
        Name of the per‑node elapsed‑time column to consult.

    Returns
    -------
    pd.DataFrame
        A copy with all sequence columns truncated row‑wise.
    """
    # 检查依赖的耗时列是否存在
    if elapsed_col not in df.columns:
        raise KeyError(f"Column '{elapsed_col}' not found in the DataFrame.")

    # 复制 DataFrame，保持原数据不变
    df = df.copy()

    # ------------------------------------------------------------------
    # 1) Normalise the threshold to the same *unit* as elapsed_col
    # ------------------------------------------------------------------
    sample_val = df[elapsed_col].dropna().iloc[0][0]  # first scalar in the first non‑null row
    if isinstance(sample_val, (int, float)):
        # elapsed is in *seconds*  → convert threshold to float seconds
        # 该列以秒（float）为单位：阈值统一转成秒
        if isinstance(max_elapsed, (int, float)):
            threshold = float(max_elapsed)
        else:  # str or Timedelta
            # 字符串或 Timedelta 先转时长再取总秒数
            threshold = pd.to_timedelta(max_elapsed).total_seconds()
    else:
        # elapsed is Timedelta                                  → ensure threshold is Timedelta
        # 该列以 Timedelta 为单位：阈值同样转为 Timedelta
        threshold = max_elapsed if isinstance(max_elapsed, pd.Timedelta) else pd.to_timedelta(max_elapsed)

    # ------------------------------------------------------------------
    # 2) Work out the *new* length for every row
    # ------------------------------------------------------------------
    def _new_len(elapsed_seq: Sequence[Any]) -> int:
        """
        Return the number of nodes to keep so that every retained
        node has elapsed ≤ threshold and is not NaN.
        """
        for idx, v in enumerate(elapsed_seq):
            if pd.isna(v):
                return idx  # stop at first padding
            if v > threshold:
                return idx  # stop at first node past limit
        return len(elapsed_seq)  # all nodes kept

    # 为每行计算截断后的新长度
    df["__new_len"] = df[elapsed_col].apply(_new_len)

    # ------------------------------------------------------------------
    # 3) Identify *all* sequence columns in one inexpensive sweep
    # ------------------------------------------------------------------
    # 第三步：一次廉价扫描识别全部序列型列
    seq_cols: list[str] = []
    for col in df.columns:
        first_val = df[col].dropna().iloc[0] if df[col].notna().any() else None
        # 第一个非空值为序列类型则记录该列
        if _is_sequence_like(first_val):
            seq_cols.append(col)

    # ------------------------------------------------------------------
    # 4) Truncate them row‑wise
    # ------------------------------------------------------------------
    # 第四步：按各自的新长度逐行截断，保证各序列列保持对齐
    for col in seq_cols:
        df[col] = df.apply(lambda r: _slice_to_len(r[col], r["__new_len"]), axis=1)

    # ------------------------------------------------------------------
    # 5) Keep ancillary metadata in sync
    # ------------------------------------------------------------------
    # 第五步：同步更新 length 列为新长度
    if "length" in df.columns:
        df["length"] = df["__new_len"]

    # 6) Clean up helper column
    # 第六步：删除辅助列
    df.drop(columns="__new_len", inplace=True)

    # 返回截断后的副本
    return df


# Section: Formatting of dataframe into ideal format
################################################


# 计算合法可选节点的掩码（有验证分 且 提交有效 且 非缺陷非填充）
def get_selection_mask(row):
    """Get the mask of nodes that could be selected."""
    # We check if the validation metric is nan.
    # 检查验证指标是否为 NaN（nan：缺失值标记）
    metric_is_nan = np.isnan(row["metric"])
    # We inverse this to get the locations which have values.
    # 取反得到"有值"的位置
    metric_is_not_nan = np.logical_not(metric_is_nan)
    # We check if submission.csv is valid for grader.
    # 检查各节点的提交文件是否通过评分器验证（NaN 按 0 即无效处理）
    is_valid_submission = np.nan_to_num(row["metric_info/valid_submission"], nan=0)
    # We then check if node not legal for selection for any other reason (see greedy code parse eval result fn)
    # 检查节点是否因其他原因不可选（缺陷节点，NaN 按 1 即缺陷处理）
    is_buggy_node = np.nan_to_num(row["is_buggy"].astype(float), nan=1)
    # We can get the good nodes from this
    # 合法好节点 = 非缺陷 且 位于真实步（非填充）
    is_good_node = np.logical_not(is_buggy_node.astype(bool)) & np.asarray(row["non_padding_mask"], dtype=bool)
    # Lastly, we and everything together
    # 三个条件按位与：有验证分数 且 提交有效 且 是好节点
    selection_mask = np.logical_and(
        np.logical_and(metric_is_not_nan.astype(bool), is_valid_submission.astype(bool)), is_good_node.astype(bool)
    )
    # This gives us a mask for all legally selectable nodes
    # 返回所有合法可选节点的掩码
    return selection_mask


# 检查指定节点的提交 csv 是否有效
def get_valid_submission(row, node_idx):
    """Simply check if a node has a valid submission csv."""
    # 返回指定节点位置的提交有效性标志（NaN 视为无效）
    return np.nan_to_num(row["metric_info/valid_submission"], nan=0)[node_idx]


# 模拟搜索返回的节点：按验证（或测试）分数在合法节点中选最优
def get_expected_return_node(row, lower_is_better: bool, use_validation_score: bool = True):
    """Process the validation metric data to see what node the search would have returned."""
    # Get Validation Scores
    # 取出每节点的分数：默认用验证分数，也可切换为测试分数
    if use_validation_score:
        v_scores = row["metric"]
    else:
        v_scores = row["metric_info/score"]
    # 全序列最大分数（NaN 填 -1e8 保证不影响最大值）
    max_v_score = np.max(np.nan_to_num(v_scores, nan=-1e8))
    # 全序列最小分数（NaN 填 1e8 保证不影响最小值）
    min_v_score = np.min(np.nan_to_num(v_scores, nan=1e8))

    # Get Selection Masks
    # 获取合法可选节点的掩码
    selection_mask = get_selection_mask(row)

    # Get the masked validation scores
    # 构造"最差填充值"：越小越优时取比最大分还差的值，反之取比最小分还差的值
    fill_value = (1 + max_v_score) if lower_is_better else (min_v_score - 1)
    v_scores = np.nan_to_num(v_scores, nan=fill_value)
    # 非法节点位置用最差填充值替换，保证不会被选中
    masked_v_scores = np.where(selection_mask, v_scores, fill_value)

    # Select the node
    # 按优化方向选出最优节点（argmin/argmax）
    if lower_is_better:
        best_node = np.argmin(masked_v_scores)
    else:
        best_node = np.argmax(masked_v_scores)

    if np.all(selection_mask == False):
        # If all nodes are invalid, we just return the last node
        # 所有节点都非法时退化为返回最后一个节点
        best_node = -1

    # 返回搜索最终应返回的节点索引
    return best_node


# 取选中节点在指定指标上的分数（含 NaN 时的诊断与兜底逻辑）
def get_metric(row, metric_key, valid_submission, expected_return_node, lower_is_better):
    # 取出指定指标在全部节点上的分数序列
    metric_scores = row[metric_key]
    # 选中"期望返回节点"处的分数
    selected_metric_score = metric_scores[expected_return_node]
    # 选中分数为 NaN 时的诊断与兜底处理
    if np.isnan(selected_metric_score):
        # 打印实验标识与诊断信息
        print(f"{row['method']} - {row['competition_id']} - {row['seed']}: selected metric ({metric_key}) is nan.")
        print("Expected return node:", expected_return_node)
        print("Valid submission:", valid_submission)
        print("Valid Node Exists?:", np.any(get_selection_mask(row)))
        print("Node is buggy:", row["is_buggy"][expected_return_node])

        # This is happening due to mlebench's grader being able to return nans even if everything supposedly works.
        # 原因：mlebench 评分器即使一切正常也可能返回 NaN
        if "medal" in metric_key:
            # 奖牌类指标 NaN 兜底为 0（无奖牌）
            selected_metric_score = 0
        elif "above_median" in metric_key:
            # 超中位数标志 NaN 兜底为 0
            selected_metric_score = 0
        else:
            # 分数类指标按优化方向兜底为最差哨兵值
            if lower_is_better:
                selected_metric_score = 1e12
            else:
                selected_metric_score = -1e12

        print(
            f"{row['method']} - {row['competition_id']} - {row['seed']}: due to nan, selected metric ({metric_key}) will be {selected_metric_score}."
        )

    # 返回兜底后的选中分数
    return selected_metric_score


# 计算分数在排行榜中的名次与百分位（视作一次新增提交，并列取平均名次）
def get_rank_and_percentile(
    score: float, leaderboard: pd.DataFrame, lower_is_better: bool
) -> Dict[str, Union[float, None]]:
    """
    Calculates the percentile rank of `score` as if it were an additional submission in the leaderboard.

    The function computes the average rank of `score` among all scores (including itself) and then maps it
    to a percentile between 0 and 1 using:

        percentile = (n - avg_rank) / (n - 1)

    where n is the total number of scores after including the new one.

    - A percentile of 1 indicates the best score.
    - A percentile of 0 indicates the worst score.

    If score is None, returns {"percentile": None, "rank": None}.
    """

    # Ensure the leaderboard has a 'score' column.
    # 前置校验：排行榜必须含 score 列
    if "score" not in leaderboard.columns:
        raise ValueError("Leaderboard must have a `score` column.")

    # Return early if score is None.
    # 分数缺失时提前返回：百分位 0、名次垫底
    if score is None or np.isnan(score):
        return {"percentile": 0, "rank": len(list(leaderboard["score"])) + 1}

    # Ensure the score is numeric.
    # 校验分数必须是数值类型
    if not isinstance(score, (int, float)):
        raise TypeError(f"Expected score to be a float or int, got {type(score)}.")

    # Combine existing scores with the new score.
    # 把新分数并入排行榜分数列表
    scores_list = list(leaderboard["score"]) + [score]
    # 并入后的总分点数 n
    n = len(scores_list)

    # If there's only one score, it is by definition the best.
    # 只有一个分数时它必然是第一名
    if n == 1:
        return {"percentile": 1.0, "rank": 1}

    # Sort scores: best first. For lower_is_better, lower numbers are better.
    # 排序：最优在前（按优化方向决定升/降序）
    if lower_is_better:
        sorted_scores = sorted(scores_list)  # ascending: best is first
    else:
        sorted_scores = sorted(scores_list, reverse=True)  # descending: best is first

    # Use a tolerance-based comparison to handle floating point imprecision.
    # Find all positions (1-indexed) where the new score is "close" to an existing score.
    # 用容差比较处理浮点误差：找出与新分数"足够接近"的所有名次（1 起始）
    tol_rel = 1e-9
    tol_abs = 1e-12
    ranks = [i + 1 for i, s in enumerate(sorted_scores) if math.isclose(s, score, rel_tol=tol_rel, abs_tol=tol_abs)]

    # In case no value is considered close (shouldn't happen because score is in scores_list), fall back to exact equality.
    # 容差比较无命中时退回精确相等（理论上不会发生）
    if not ranks:
        ranks = [i + 1 for i, s in enumerate(sorted_scores) if s == score]
    try:
        # 并列名次取平均（平均排名法）
        avg_rank = sum(ranks) / len(ranks)

        # Compute the percentile.
        # 计算百分位：(n - 平均名次) / (n - 1)，1 为最佳、0 为最差
        percentile = (n - avg_rank) / (n - 1)

        # 返回百分位与名次
        return {"percentile": percentile, "rank": avg_rank}
    except Exception as e:
        # 计算异常时打印错误信息
        print(f"Error occurred in calculating percentile and rank for Score = {score}: {e}")


# 判定竞赛的优化方向（分数是否越低越好）
def is_lower_better(competition):
    # 读取该竞赛的排行榜 CSV
    leaderboard_df = pd.read_csv(competition.leaderboard)

    try:
        # 尝试访问 score 列，缺失通常说明未拉取 Git LFS 大文件
        _ = leaderboard_df["score"]
    except:
        raise Exception("You must run GIT LFS for mlebench.")

    # 委托评分器判定优化方向（lower is better：分数越低越好）
    return competition.grader.is_lower_better(leaderboard_df)


# 计算一组数值的常用统计量（均值/最值/标准差/分位数/原始数据）
def get_statistics(x, label):
    # 样本数大于 1 时才计算统计量（单个样本无统计意义）
    if len(x) > 1:
        # 汇总均值/最值/标准差/分位数与原始数据
        data = {
            f"avg_{label}": np.mean(x),
            f"min_{label}": np.min(x),
            f"max_{label}": np.max(x),
            f"std_{label}": np.std(x),
            f"q1_{label}": np.quantile(x, 0.25),
            f"q3_{label}": np.quantile(x, 0.75),
            f"raw_data_{label}": x,
        }
        # 返回统计字典
        return data

    # 样本不足时返回空字典
    return {}


# 把逐实验数据转换为竞赛报告行（节点选择/指标提取/排名百分位/统计量）
def format_experiment_data(
    experiments_df: pd.DataFrame,
    max_num_seeds: int = 20,
    select_using_test: bool = False,
    node_selector_function: Optional[Callable] = None,
) -> pd.DataFrame:
    """Take in the dataframe containing experiment data per row and convert it into competition reports."""
    # Get the unique tasks and methods
    # 取出全部出现过的竞赛与方法
    unique_tasks = experiments_df["competition_id"].unique()
    unique_methods = experiments_df["method"].unique()

    # 打印处理的任务与方法
    print(f"Processing {unique_tasks}")
    print(f"Processing {unique_methods}")

    # You might have a registry or some object to get competition info:
    # 重建 mlebench 注册表以查询竞赛详情
    new_registry = registry.set_data_dir(Path(get_mlebench_data_dir()))

    # 收集所有实验的报告行
    all_reports = []
    # 统计量嵌套字典：统计量[任务][方法]
    statistics = defaultdict(lambda: defaultdict(float))
    for task in unique_tasks:
        # Filter rows for this particular competition_id
        # 筛选当前竞赛的所有行
        comp_mask = experiments_df["competition_id"] == task
        task_df = experiments_df[comp_mask].copy()

        # Get the competition details i.e. lower is better
        # 获取竞赛对象与优化方向（lower is better）
        competition = new_registry.get_competition(task)
        lower_is_better = is_lower_better(competition)
        # 读取竞赛排行榜，用于计算名次/百分位
        leaderboard_df = pd.read_csv(competition.leaderboard)
        for method in unique_methods:
            # Filter rows for this particular method
            # 筛选当前方法在该竞赛下的所有行（不同种子）
            method_mask = task_df["method"] == method
            method_df = task_df[method_mask].copy()
            # 该方法的总运行次数
            total_num_runs = len(method_df)

            # Now create the report dict for each row
            # 已成功生成报告的行索引集合
            reported_row_indices = []
            # 已生成报告的行数
            num_reported_rows = 0
            # 当前行在原 DataFrame 中的序号
            idx = -1
            for _, row in method_df.iterrows():
                idx += 1

                try:
                    if node_selector_function is None:
                        # 默认节点选择器：用验证分数（或测试分数）模拟搜索返回的节点
                        expected_return_node = get_expected_return_node(
                            row, lower_is_better, use_validation_score=not select_using_test
                        )
                    else:
                        # 使用自定义节点选择函数
                        expected_return_node = node_selector_function(row, lower_is_better)

                    # 检查选中节点的提交是否有效
                    valid_submission = get_valid_submission(row, expected_return_node)
                    # 记录选中节点的验证分数
                    validation_score = row["metric"][expected_return_node]

                    # 提取选中节点的测试分数及各类奖牌/中位数标志（NaN 已兜底）
                    score = get_metric(
                        row, "metric_info/score", valid_submission, expected_return_node, lower_is_better
                    )
                    any_medal = get_metric(
                        row, "metric_info/any_medal", valid_submission, expected_return_node, lower_is_better
                    )
                    gold_medal = get_metric(
                        row, "metric_info/gold_medal", valid_submission, expected_return_node, lower_is_better
                    )
                    silver_medal = get_metric(
                        row, "metric_info/silver_medal", valid_submission, expected_return_node, lower_is_better
                    )
                    bronze_medal = get_metric(
                        row, "metric_info/bronze_medal", valid_submission, expected_return_node, lower_is_better
                    )
                    above_median = get_metric(
                        row, "metric_info/above_median", valid_submission, expected_return_node, lower_is_better
                    )

                    # 计算该分数相对排行榜的名次与百分位（relative placement：相对排名）
                    relative_placement = get_rank_and_percentile(score, leaderboard_df, lower_is_better)

                    # 百分位异常（NaN 或非浮点）时打印警告
                    if np.isnan(relative_placement["percentile"]) or not isinstance(
                        relative_placement["percentile"], float
                    ):
                        print(
                            f"{row['method']} - {row['competition_id']} - {row['seed']}: Percentile is Not a float = {relative_placement['percentile']}."
                        )

                    # 分数为哨兵值（无有效节点）时，断言百分位必须为 0
                    if score == 1e12 or score == -1e12:
                        assert relative_placement["percentile"] == 0, (
                            f"{row['method']} - {row['competition_id']} - {row['seed']}: Score is sentinel value = {score}."
                        )

                    # 汇总该行（一次运行）的报告字典
                    report_row = {
                        "experiment_id": row["method"],
                        "competition_id": row["competition_id"],
                        "score": float(score),
                        "validation_score": float(validation_score.item()),
                        "any_medal": bool(any_medal),
                        "gold_medal": bool(gold_medal),
                        "silver_medal": bool(silver_medal),
                        "bronze_medal": bool(bronze_medal),
                        "above_median": bool(above_median),
                        "valid_submission": bool(valid_submission),
                        "length": row["length"],
                        "validation_scores": list(row["metric"]),
                        "test_scores": list(row["metric_info/score"]),
                        "rank": relative_placement["rank"],
                        "percentile": relative_placement["percentile"],
                        "lower_is_better": lower_is_better,
                        "seed": row["seed"],
                        "expected_return_node": expected_return_node,
                        "num_validation_nodes": row["num_validation_nodes"],
                        "num_test_nodes": row["num_test_nodes"],
                        "num_good_nodes": row["num_good_nodes"],
                        "num_buggy_nodes": row["num_buggy_nodes"],
                        "future_unused_nodes": row["length"] - expected_return_node,
                        "seconds_from_first_node": row["seconds_from_first_node"],
                        "any_medal_series": list(row["metric_info/any_medal"]),
                        "gold_medal_series": list(row["metric_info/gold_medal"]),
                        "silver_medal_series": list(row["metric_info/silver_medal"]),
                        "bronze_medal_series": list(row["metric_info/bronze_medal"]),
                        "above_median_series": list(row["metric_info/above_median"]),
                        # 逐节点测试分数对应的百分位序列（相对排行榜）
                        "percentile_series": [
                            get_rank_and_percentile(s, leaderboard_df, lower_is_better)["percentile"]
                            for s in list(row["metric_info/score"])
                        ],
                        # 逐节点验证分数对应的百分位序列
                        "val_percentile_series": [
                            get_rank_and_percentile(s, leaderboard_df, lower_is_better)["percentile"]
                            for s in list(row["metric"])
                        ],
                    }

                    # 收集报告行并更新计数
                    all_reports.append(report_row)
                    reported_row_indices.append(idx)
                    num_reported_rows += 1

                    # 达到种子数上限后停止（控制每方法报告的运行数）
                    if num_reported_rows >= max_num_seeds:
                        break
                except Exception as e:
                    # 单行处理失败只记录，不中断整体流程
                    print(f"Error in processing {row['method']} - {row['competition_id']} - {row['seed']}: {e}")

            # Get the statistics
            # 只对成功生成报告的行计算统计量
            method_df = method_df.iloc[reported_row_indices]
            # 汇总各指标的统计量与总运行数
            statistics[task][method] = {
                **get_statistics(method_df["length"], "length"),
                **get_statistics(method_df["num_validation_nodes"], "num_validation_nodes"),
                **get_statistics(method_df["num_test_nodes"], "num_test_nodes"),
                **get_statistics(method_df["num_good_nodes"], "num_good_nodes"),
                **get_statistics(method_df["num_buggy_nodes"], "num_buggy_nodes"),
                "total_num_runs": total_num_runs,
            }

    # Finally convert all_reports into a single DataFrame:
    # 把全部报告行转为最终 DataFrame
    reports_df = pd.DataFrame(all_reports)
    # 返回报告 DataFrame 与统计量字典
    return reports_df, statistics


# 把报告 DataFrame 聚合成 方法 -> 分数矩阵（种子 x 竞赛） 的字典
def parse_into_aggregate_dict(report_df, metric, algorithms=None):
    # 未显式指定算法列表时，取报告 DataFrame 中的全部实验 ID
    if algorithms is None:
        algorithms = list(report_df["experiment_id"].unique())

    # Get the unique methods
    # 待处理的方法列表
    methods = algorithms
    print(f"Processing {methods}")
    # 结果字典：方法 -> 分数矩阵
    score_dict = {}

    for method in methods:
        # Filter the dataframe for the current method
        # 筛选当前方法的全部报告行
        m_df = report_df[report_df["experiment_id"] == method].copy()
        # Create a 'seed' column based on the order of rows for each task
        # 按竞赛分组行序生成种子编号（seed 列）
        m_df["seed"] = m_df.groupby("competition_id").cumcount()
        # Pivot the dataframe: rows = seeds, columns = tasks, values = metric
        # 透视：行=种子，列=竞赛，值=指定指标
        pivot_df = m_df.pivot(index="seed", columns="competition_id", values=metric)
        # filter out the rows where the metric is NaN
        # This is to ensure we only keep rows where the metric is not NaN
        # Remove rows with NaN values
        # 剔除含 NaN 的行，只保留指标完整的种子
        pivot_df = pivot_df.dropna()
        # Convert the pivoted dataframe to a numpy matrix
        # 透视表转为浮点矩阵存入结果
        score_dict[method] = pivot_df.to_numpy().astype(float)
        # 二次保险：再次过滤任何含 NaN 的行
        score_dict[method] = score_dict[method][~np.isnan(score_dict[method]).any(axis=1)]

    # 返回 方法 -> 分数矩阵 的聚合字典
    return score_dict
