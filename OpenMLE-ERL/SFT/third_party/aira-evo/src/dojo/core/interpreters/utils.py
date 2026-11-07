# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 日期时间：用于生成实验名的时间戳
import datetime
# 标准日志模块
import logging
# 多进程：获取当前进程名
import multiprocessing
# 操作系统接口（部分工具使用）
import os
# 高级文件操作：复制/移动/删除文件与目录
import shutil
# zip 压缩包读取与解压
import zipfile
# dataclass 装饰器
from dataclasses import dataclass
# 文件路径对象
from pathlib import Path
# 类型注解工具
from typing import Any, Dict, Hashable, Optional, Union, cast

# 随机生成可读名称（用于实验命名）
import coolname
# 终端富文本输出库
import rich
# 配置库主入口
from omegaconf import OmegaConf
# rich 的语法高亮渲染器（用于展示 YAML 配置）
from rich.syntax import Syntax

# 获取全局日志器
from dojo.utils.logger import get_logger

# ---------------------------
# Logger Configuration
# ---------------------------

# 模块级日志器实例
logger = get_logger()

# ---------------------------
# Utility Functions
# ---------------------------


def get_custom_process_name():
    """Simple function to extract id and rename process"""
    # 取当前进程对象
    current_process = multiprocessing.current_process()
    # 按连字符切分进程名并返回最后一段作为进程 ID（如 "Process-3" 的 "3"）
    return current_process.name.split("-")[-1]


def copy_contents(source: Path, destination: Path, use_symlinks: bool = True) -> None:
    """
    Copy contents from source to destination. If the source is a file, copy the file.
    If the source is a directory, merge its contents with the destination directory.

    Args:
        source (Path): Source directory or file.
        destination (Path): Destination directory.
        use_symlinks (bool): Whether to create symbolic links instead of copying.
    """
    # 目标目录不存在则先递归创建
    if not destination.exists():
        destination.mkdir(parents=True, exist_ok=True)

    # 目标必须是目录，否则抛出值错误
    if not destination.is_dir():
        raise ValueError(f"Destination {destination} must be a directory.")

    # 源是单个文件则按单文件逻辑复制
    if source.is_file():
        _copy_file(source, destination, use_symlinks)
    else:
        # 源是目录则按目录合并逻辑复制
        _copy_directory(source, destination, use_symlinks)


def _copy_file(src: Path, dst_dir: Path, use_symlinks: bool) -> None:
    """
    Copy a single file to the destination directory.

    Args:
        src (Path): Source file.
        dst_dir (Path): Destination directory.
        use_symlinks (bool): Whether to create a symbolic link.
    """
    # 目标路径 = 目标目录 + 源文件名
    destination = dst_dir / src.name
    # 目标已存在则跳过（不覆盖）
    if destination.exists():
        logger.warning(f"File {destination} already exists. Skipping copy.")
        return

    try:
        # 使用符号链接模式
        if use_symlinks:
            # 建立指向源文件的符号链接
            destination.symlink_to(src)
            logger.debug(f"Symlinked file {src} to {destination}")
        else:
            # 真实复制模式：复制文件并保留元数据
            shutil.copy2(src, destination)
            logger.debug(f"Copied file {src} to {destination}")
    except OSError as error:
        # 链接/复制失败（如跨文件系统），记录失败原因
        logger.error(f"Failed to {'symlink' if use_symlinks else 'copy'} file {src} to {destination}: {error}")
        # 复制模式下失败属于严重问题，向上抛出
        if not use_symlinks:
            raise


def _copy_directory(src: Path, dst_dir: Path, use_symlinks: bool) -> None:
    """
    Merge contents of the source directory into the destination directory.

    Args:
        src (Path): Source directory.
        dst_dir (Path): Destination directory.
        use_symlinks (bool): Whether to create symbolic links.
    """
    # 遍历源目录下的每个条目
    for item in src.iterdir():
        # 对应的目标路径
        destination = dst_dir / item.name
        # 目标已存在则跳过该项
        if destination.exists():
            logger.warning(f"Destination {destination} already exists. Skipping.")
            continue

        try:
            # 符号链接模式：直接链接整个条目
            if use_symlinks:
                destination.symlink_to(item)
                logger.debug(f"Symlinked {item} to {destination}")
            # 真实复制且是子目录：递归复制整棵子树
            elif item.is_dir():
                shutil.copytree(item, destination)
                logger.debug(f"Copied directory {item} to {destination}")
            else:
                # 真实复制且是文件：复制并保留元数据
                shutil.copy2(item, destination)
                logger.debug(f"Copied file {item} to {destination}")
        except OSError as error:
            # 复制/链接出错，记录失败原因
            logger.error(f"Failed to copy {'directory' if item.is_dir() else 'file'} {item} to {destination}: {error}")
            # 目录复制失败（如子树中部分文件已存在）：允许覆盖后强制再复制
            if item.is_dir() and not use_symlinks:
                shutil.copytree(item, destination, dirs_exist_ok=True)
                logger.debug(f"Force copied directory {item} to {destination}")


def remove_unwanted_items(path: Path, patterns: Optional[list] = None) -> None:
    """
    Remove unwanted files and directories based on specified patterns.

    Args:
        path (Path): Root directory to clean.
        patterns (Optional[list]): List of patterns to remove. Defaults to ["__MACOSX", ".DS_Store"].
    """
    # 未指定模式则用默认清理列表（macOS 打包垃圾文件）
    if patterns is None:
        patterns = ["__MACOSX", ".DS_Store"]

    # 逐个模式处理
    for pattern in patterns:
        # 递归匹配所有符合模式的路径
        for item in path.rglob(pattern):
            try:
                # 是目录则递归删除整棵树
                if item.is_dir():
                    shutil.rmtree(item)
                    logger.debug(f"Removed directory {item}")
                # 是文件则直接删除
                elif item.is_file():
                    item.unlink()
                    logger.debug(f"Removed file {item}")
            except OSError as error:
                # 删除失败则记录并继续
                logger.error(f"Failed to remove {item}: {error}")


def extract_zip_file(zip_path: Path, extract_to: Path) -> None:
    """
    Extract a single zip file to the specified directory.

    Args:
        zip_path (Path): Path to the zip file.
        extract_to (Path): Directory to extract contents to.
    """
    try:
        # 以只读方式打开 zip 并解压全部内容到目标目录
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        # 记录解压成功日志
        logger.debug(f"Extracted {zip_path} to {extract_to}")
    except zipfile.BadZipFile:
        # 文件不是合法 zip，记录错误并重新抛出
        logger.error(f"Bad zip file: {zip_path}")
        raise
    except Exception as error:
        # 其他解压错误，记录并重新抛出
        logger.error(f"Error extracting {zip_path}: {error}")
        raise


def flatten_directory_structure(zip_output_dir: Path, expected_name: str) -> None:
    """
    Flatten the directory structure if the extracted content is nested within a single directory.

    Args:
        zip_output_dir (Path): Directory where zip was extracted.
        expected_name (str): Expected name of the single directory or file.
    """
    # 列出解压目录的全部条目
    contents = list(zip_output_dir.iterdir())
    # 恰好只有一个且名字符合预期才处理
    if len(contents) == 1 and contents[0].name == expected_name:
        # 该唯一条目
        single_item = contents[0]
        # 是目录：把其内容全部上移一层
        if single_item.is_dir():
            logger.debug(f"Flattening directory structure in {zip_output_dir}")
            # 遍历子目录内容
            for item in single_item.iterdir():
                # 移动到上层目录
                shutil.move(str(item), zip_output_dir)
                logger.debug(f"Moved {item} to {zip_output_dir}")
            # 删除已空的子目录
            single_item.rmdir()
            logger.debug(f"Removed empty directory {single_item}")
        # 是文件：把它重命名为目录本身
        elif single_item.is_file():
            logger.debug(f"Renaming file {single_item} to {zip_output_dir}")
            # 临时重命名目标（避免同名冲突）
            temp_rename = zip_output_dir.with_suffix(".__tmp_rename")
            # 先把文件移到临时名
            single_item.rename(temp_rename)
            # 删除原（已空的）目录
            zip_output_dir.rmdir()
            # 再把文件改名回目录名，完成替换
            temp_rename.rename(zip_output_dir)
            logger.debug(f"Renamed {temp_rename} to {zip_output_dir}")


def extract_all_archives(path: Path) -> None:
    """
    Extract all zip archives within the specified path and clean up the directories.

    Args:
        path (Path): Root directory to search for zip files.
    """
    # 递归查找所有 zip 文件
    for zip_file in path.rglob("*.zip"):
        # 输出目录 = 去掉 .zip 后缀的同名目录
        output_dir = zip_file.with_suffix("")

        # 输出目录已存在 -> 已解压过，跳过
        if output_dir.exists():
            logger.debug(f"Skipping extraction of {zip_file} as {output_dir} already exists.")
            # 异常情况：同名目标是个带后缀的文件而非目录，直接删除 zip 视为冗余
            if output_dir.is_file() and output_dir.suffix:
                try:
                    zip_file.unlink()
                    logger.debug(f"Removed existing zip file {zip_file}")
                except OSError as error:
                    logger.error(f"Failed to remove zip file {zip_file}: {error}")
            continue

        try:
            # 记录开始解压
            logger.info(f"Extracting {zip_file} to {output_dir}")
            # 创建输出目录
            output_dir.mkdir(parents=True, exist_ok=True)
            # 执行解压
            extract_zip_file(zip_file, output_dir)

            # Clean up unwanted files
            # 清理解压出的无用文件（如 __MACOSX）
            remove_unwanted_items(output_dir)

            # Flatten directory structure if necessary
            # 必要时拍平嵌套目录结构
            flatten_directory_structure(output_dir, output_dir.name)

            # Remove the zip file after successful extraction
            # 解压成功后删除原 zip 包
            zip_file.unlink()
            logger.debug(f"Removed zip file {zip_file}")
        except Exception as error:
            # 单个 zip 失败不影响其他，记录原因
            logger.error(f"Failed to extract {zip_file}: {error}")


def get_next_log_index(directory: Path) -> int:
    """
    Determine the next available index for naming log or workspace directories.

    Args:
        directory (Path): Directory to scan for existing indices.

    Returns:
        int: Next available index.
    """
    # 收集已存在的序号
    indices = []
    # 遍历目录条目
    for item in directory.iterdir():
        # 只关心子目录
        if item.is_dir():
            try:
                # 取目录名首段并尝试解析为整数序号
                index = int(item.name.split("-")[0])
                # 记录该序号
                indices.append(index)
            except (ValueError, IndexError):
                # 首段不是数字 -> 命名不规范，记录告警并忽略
                logger.warning(f"Ignoring directory with invalid name format: {item.name}")
    # 取最大序号 +1（空目录则从 0 开始）
    next_index = max(indices, default=-1) + 1
    # 记录 debug 日志
    logger.debug(f"Next log index in {directory}: {next_index}")
    # 返回下一个可用序号
    return next_index


def prepare_configuration(cfg: OmegaConf) -> OmegaConf:
    """
    Prepare and validate the configuration.

    Args:
        cfg (OmegaConf): Initial configuration.

    Returns:
        OmegaConf: Prepared and validated configuration.

    Raises:
        ValueError: If required configuration fields are missing.
    """

    # Resolve and validate paths
    # 把各路径解析为绝对路径（输入数据目录相对 cwd 解析）
    cfg.input_data_dir = _resolve_path(cfg.input_data_dir, relative_to_parent=False)
    # 解析日志目录
    cfg.log_dir = _resolve_path(cfg.log_dir)
    # 解析工作区目录
    cfg.workspace_dir = _resolve_path(cfg.workspace_dir)

    # Ensure log and workspace directories exist
    # 创建日志目录
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    # 创建工作区目录
    cfg.workspace_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Log directory: {cfg.log_dir}")
    logger.debug(f"Workspace directory: {cfg.workspace_dir}")

    # Generate a unique experiment name
    # 用当前时间的 ISO 格式串保证唯一性
    experiment_index = datetime.datetime.now().isoformat()
    # 未指定实验名则用 coolname 随机生成三词短名
    cfg.exp_name = cfg.exp_name or coolname.generate_slug(3)
    # 前缀进程 ID 与时间戳，进一步避免并发冲突
    cfg.exp_name = f"{get_custom_process_name()}-{experiment_index}-{cfg.exp_name}"
    logger.info(f"Experiment name: {cfg.exp_name}")

    # Update log and workspace directories with experiment name
    # 在日志与工作区目录下新建实验名子目录
    cfg.log_dir = (cfg.log_dir / cfg.exp_name).resolve()
    cfg.workspace_dir = (cfg.workspace_dir / cfg.exp_name).resolve()
    # 创建实验日志目录
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    # 创建实验工作区目录
    cfg.workspace_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Updated log directory: {cfg.log_dir}")
    logger.debug(f"Updated workspace directory: {cfg.workspace_dir}")

    # 记录配置就绪
    logger.info("Configuration prepared.")

    # 类型收窄后返回配置
    return cast(OmegaConf, cfg)


def _resolve_path(path_str: Union[str, Path], relative_to_parent: bool = False) -> Path:
    """
    Resolve a string or Path to an absolute Path.

    Args:
        path_str (Union[str, Path]): The path to resolve.
        relative_to_parent (bool): If True, resolve relative to the parent directory of the script.

    Returns:
        Path: Resolved absolute path.
    """
    # 统一转成 Path 对象
    path = Path(path_str)
    # 相对路径需要解析
    if not path.is_absolute():
        # 基于本脚本上两级目录解析
        if relative_to_parent:
            # 取脚本所在目录的父目录作为基准
            base_path = Path(__file__).parent.parent
            # 拼接并解析为绝对路径
            path = (base_path / path).resolve()
        else:
            # 基于当前工作目录解析
            path = path.resolve()
    else:
        # 已是绝对路径：仅做规范化（处理 .. 与符号链接）
        path = path.resolve()
    # 记录解析结果
    logger.debug(f"Resolved path: {path}")
    # 返回绝对路径
    return path


def display_configuration(cfg: OmegaConf) -> None:
    """
    Display the configuration in a formatted YAML style using rich.

    Args:
        cfg (OmegaConf): Configuration to display.
    """
    # 把配置序列化为 YAML 文本
    yaml_content = OmegaConf.to_yaml(cfg)
    # 构造带行号的 YAML 语法高亮对象
    syntax = Syntax(yaml_content, "yaml", theme="paraiso-dark", line_numbers=True)
    # 富文本打印到终端
    rich.print(syntax)
    # 记录展示完成日志
    logger.info("Configuration displayed.")
