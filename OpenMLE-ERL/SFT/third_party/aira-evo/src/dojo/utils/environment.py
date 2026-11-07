# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 subprocess，用于执行硬件探测命令（nvidia-smi、python -c 等）
import subprocess
# 导入 os，用于读取环境变量
import os
# 导入 omegaconf 配置库（供配置加载场景使用）
import omegaconf
# 导入 dotenv，用于从 .env 文件加载环境变量
from dotenv import load_dotenv

# 启动时加载 .env 文件中的环境变量
load_dotenv()

# 环境变量缺失时的统一错误信息模板（用 env_var_name 填充具体变量名）
ENV_VAR_NOT_FOUND_ERR = (
    "{env_var_name} environment variable is not set or is empty."
    " Make sure to set it in your .env file or in the environment variables."
)


# 探测可用硬件（GPU 型号列表或 CPU）
def get_hardware():
    """Determine available hardware (GPU or CPU)."""
    try:
        # Check if `nvidia-smi` is available and get GPU name
        # 调用 nvidia-smi 逐行查询每块 GPU 的名称
        result = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, text=True)
        # Process output: trim spaces, remove duplicates, and format
        # 去除每行空白、去重、排序后用逗号拼接成硬件描述字符串
        hardware = ", ".join(sorted(set(line.strip() for line in result.split("\n") if line.strip())))
    except subprocess.CalledProcessError:
        # 命令失败说明没有可用 GPU，退回到 CPU 描述
        hardware = "a CPU"  # Default if no GPU is found
    # 返回硬件描述
    return hardware


def check_pytorch_gpu():
    """Check if PyTorch can use a GPU."""
    # 在子进程中检查 PyTorch 能否使用 GPU（可用则打印设备名）
    try:
        return subprocess.check_output(
            "python -c \"import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'WARNING: No GPU')\"",
            shell=True,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        # 检查命令执行失败时返回错误提示
        return "ERROR: PyTorch check failed"


def check_tensorflow_gpu():
    """Check if TensorFlow can use a GPU."""
    # 在子进程中检查 TensorFlow 能否看到 GPU 设备
    try:
        return subprocess.check_output(
            "python -c \"import tensorflow as tf; print('GPUs Available:', tf.config.list_physical_devices('GPU'))\"",
            shell=True,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        # 检查命令执行失败时返回错误提示
        return "ERROR: TensorFlow check failed"


# 把秒数转换为"小时+分钟+秒"的可读格式（用于记录运行时长）
def format_time(seconds):
    """Convert time in seconds to a human-readable format."""
    # 计算小时部分（整除 3600 秒）
    hours = seconds // 3600
    # 去掉整小时后计算分钟部分
    minutes = (seconds % 3600) // 60
    # 剩余的秒数部分
    secs = seconds % 60
    # 拼接为可读字符串返回
    return f"{hours}hrs {minutes}mins {secs}secs"


# 从环境变量读取日志目录路径；缺失则报错
def get_log_dir():
    """Get the log directory, creating it if it doesn't exist."""
    # 读取 LOGGING_DIR 环境变量（默认空串）
    log_dir = os.getenv("LOGGING_DIR", "")
    # 为空说明未配置，抛出统一错误
    if not log_dir:
        raise ValueError(ENV_VAR_NOT_FOUND_ERR.format(env_var_name="LOGGING_DIR"))
    # 返回日志目录路径
    return log_dir


# 从环境变量读取超分图像目录路径；缺失则报错
def get_superimage_dir():
    """Get the superimage directory, creating it if it doesn't exist."""
    # 读取 SUPERIMAGE_DIR 环境变量（默认空串）
    superimage_dir = os.getenv("SUPERIMAGE_DIR", "")
    # raise an error if empty string or None
    # 为空说明未配置，抛出统一错误
    if not superimage_dir:
        raise ValueError(ENV_VAR_NOT_FOUND_ERR.format(env_var_name="SUPERIMAGE_DIR"))
    # 返回目录路径
    return superimage_dir


# 从环境变量读取 MLE-Bench 数据目录路径；缺失则报错
def get_mlebench_data_dir():
    """Get the MLEBench data directory, creating it if it doesn't exist."""
    # 读取 MLE_BENCH_DATA_DIR 环境变量（默认空串）
    mlebench_data_dir = os.getenv("MLE_BENCH_DATA_DIR", "")
    # 为空说明未配置，抛出统一错误
    if not mlebench_data_dir:
        raise ValueError(ENV_VAR_NOT_FOUND_ERR.format(env_var_name="MLE_BENCH_DATA_DIR"))
    # 返回数据目录路径
    return mlebench_data_dir


# 从环境变量读取默认 Slurm 分区名；缺失则报错
def get_default_slurm_partition():
    """Get the default Slurm partition from the configuration."""
    # 读取 DEFAULT_SLURM_PARTITION 环境变量（默认空串）
    slurm_partition = os.getenv("DEFAULT_SLURM_PARTITION","")
    # 为空说明未配置，抛出统一错误
    if not slurm_partition:
        raise ValueError(ENV_VAR_NOT_FOUND_ERR.format(env_var_name="DEFAULT_SLURM_PARTITION"))
    # 返回分区名
    return slurm_partition


# 从环境变量读取默认 Slurm 账户名；缺失则报错
def get_default_slurm_account():
    """Get the default Slurm account from the configuration."""
    # 读取 DEFAULT_SLURM_ACCOUNT 环境变量（默认空串）
    slurm_account = os.getenv("DEFAULT_SLURM_ACCOUNT", "")
    # 为空说明未配置，抛出统一错误
    if not slurm_account:
        raise ValueError(ENV_VAR_NOT_FOUND_ERR.format(env_var_name="DEFAULT_SLURM_ACCOUNT"))
    # 返回账户名
    return slurm_account


# 从环境变量读取默认 Slurm QoS（服务质量等级）；缺失则报错
def get_default_slurm_qos():
    """Get the default Slurm QoS from the configuration."""
    # 读取 DEFAULT_SLURM_QOS 环境变量（默认空串）
    slurm_qos = os.getenv("DEFAULT_SLURM_QOS", "")
    # 为空说明未配置，抛出统一错误
    if not slurm_qos:
        raise ValueError(ENV_VAR_NOT_FOUND_ERR.format(env_var_name="DEFAULT_SLURM_QOS"))
    # 返回 QoS 名称
    return slurm_qos
