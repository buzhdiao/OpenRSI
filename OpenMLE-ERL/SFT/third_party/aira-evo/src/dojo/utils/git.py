# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入日志模块
import logging
# 导入 os，用于取路径基名
import os
# 导入 subprocess，用于执行 git 命令（安全扫描豁免注释保留）
import subprocess  # noqa: S404
# 导入 tempfile，用于创建临时克隆目录
import tempfile
# 导入 time，用于统计快照耗时
import time
# 导入 Path，用于路径类型标注
from pathlib import Path

# 创建本模块的日志器
logger = logging.getLogger(__name__)


# 执行一条 shell 命令并返回结果对象
def run_command(command: str, check: bool = True, verbose: bool = False) -> subprocess.CompletedProcess:
    # 冗余模式下记录将要执行的命令
    if verbose:
        logger.info(f"Running command: `{command}`")

    # 以 shell 方式运行命令；check=True 时失败即抛异常；捕获 stdout/stderr 并按文本返回
    return subprocess.run(
        command,
        shell=True,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


# 获取当前 git 仓库的名称（顶级目录的基名）
def get_repo_name() -> str:
    # 取仓库顶级路径的最后一段作为仓库名
    return os.path.basename(get_git_top_level())


# 获取当前 git 仓库的顶级目录路径
def get_git_top_level() -> str:
    # 调用 git rev-parse --show-toplevel 并去除首尾空白
    return subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode("utf-8").strip()


# 为快照目录制作"浅 git 仓库"：只保留深度为 1 的 .git 元数据，
# 使实验快照能记录代码来源与版本信息，而不复制完整工作区
def make_snapshot_shallow_git(git_root: Path, snapshot_root: Path) -> None:
    # 记录开始时间以统计耗时
    start_time = time.time()
    # 使用临时目录存放中间克隆产物
    with tempfile.TemporaryDirectory() as tmpdir:
        # 遍历要快照的仓库列表（当前仅根仓库；保留结构以便将来支持子模块）
        for repo in ["."]:  # This is needed if we ever have submodules
            logger.info(
                f"Making snapshot repo {repo if repo != '.' else 'aira-dojo'} of {snapshot_root} as shallow git ..."
            )
            run_command(
                f"git clone --depth 1 file:///{git_root}/{repo} {tmpdir}/{repo}"
            )  # We do not do --recurse-submodules here, because --depth 1 does not work with --recurse-submodules
            # 把浅克隆得到的 .git 目录拷贝进快照目录，使其成为 git 仓库
            run_command(f"cp -r {tmpdir}/{repo}/.git {snapshot_root}/{repo}")

    # 记录快照制作总耗时
    logger.info(f"Making snapshot repo of {snapshot_root} as shallow git took {time.time() - start_time} seconds")


def has_uncommitted_changes() -> bool:
    """Check if there are uncommited changes.

    Returns:
        bool: wether there are uncommiteed changes.
    """
    # 用 porcelain 模式查询工作区状态（输出稳定、便于脚本解析）
    command = "git status --porcelain"
    # 有任何输出即说明存在未提交的改动
    output = subprocess.check_output(command.split()).strip().decode("utf-8")
    # 返回是否存在未提交改动
    return len(output) > 0


def get_current_commit_id(repo: str) -> str:
    """Get current commit id.

    Returns:
        str: current commit id.
    """
    # 基础命令：查询当前 HEAD 的提交 ID
    command = "git rev-parse HEAD"
    # 属于子模块的仓库名列表
    submodule_repos = ["arrival_data", "fairseq2"]
    # 支持查询的全部仓库名列表
    repos = submodule_repos + ["arrival_ssl"]
    # 传入不支持的仓库名时直接报错
    if repo not in repos:
        raise ValueError(f"Unknown {repo=}. Supported {repos=}")
    # 子模块仓库需查询其独立提交 ID（命令格式 <HEAD>:<repo>）
    if repo in submodule_repos:
        command = f"{command}:{repo}"

    # 执行命令并取出提交 ID（去除首尾空白）
    commit_id = (
        subprocess.check_output(command.split()).strip().decode("utf-8")  # noqa: S603
    )
    # 返回提交 ID
    return commit_id
