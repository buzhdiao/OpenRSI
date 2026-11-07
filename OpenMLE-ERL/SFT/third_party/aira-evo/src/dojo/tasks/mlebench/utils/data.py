# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入日志模块，用于创建模块级日志器
import logging
# 导入 Path，提供面向对象的路径操作
from pathlib import Path
# 导入 tarfile，用于处理 .tar/.tar.gz 压缩包
import tarfile
# 导入 zipfile，用于处理 .zip 压缩包
import zipfile
# 导入 py7zr，用于处理 .7z 压缩包
import py7zr
# 导入 os，用于遍历目录与拼接路径
import os
# 导入 shutil，作为压缩解包的兜底方案
import shutil

# 创建本模块的日志器
logger = logging.getLogger(__name__)

# 任务根目录：当前 utils 目录的上一级（mlebench 任务目录）
TASK_DIR = Path(__file__).parent.parent


# 将目录打包为 tar 归档（逐文件添加，保留相对路径结构）
def tar_directory(root_dir, output_file):
    # 记录打包的源目录与目标文件
    logger.info(f"Creating tarball from {root_dir} to {output_file}")
    # 以写模式打开输出 tar 文件
    with tarfile.open(output_file, "w") as tar:
        # 递归遍历源目录下的所有子目录与文件
        for root, dirs, files in os.walk(root_dir):
            # 逐个处理目录中的文件
            for file in files:
                # 拼出该文件的完整路径
                file_path = os.path.join(root, file)
                # 添加进归档；arcname 取相对根目录父级的路径，保证解包后目录结构一致
                tar.add(file_path, arcname=os.path.relpath(file_path, os.path.dirname(root_dir)))


# 读取指定数据划分（split）中的比赛 ID 列表
def get_competition_ids_in_split(split_id):
    # 拼出划分文件路径：mlebench/splits/<split_id>.txt
    fname = Path(__file__).parent.parent / "splits" / f"{split_id}.txt"

    # 读取文件内容并按行拆分为比赛 ID 列表
    with open(fname, "r") as f:
        competition_ids = f.read().splitlines()

    # 返回该划分包含的比赛 ID
    return competition_ids


def extract_all_from_path(
    path: Path, already_extracted: set = set(), force: bool = True, delete_compressed: bool = False
) -> None:
    """Extracts the contents of a compressed file to a destination directory."""

    # 内部函数：把单个压缩文件解压到目标目录
    def extract(file: Path, dst: Path) -> None:
        """Extracts a compressed file to the specified destination."""
        # Check if the destination directory already exists
        # 若目标名以 .json/.csv/.txt 结尾，说明其本身是文件而非目录，解压目标应改为其父目录
        if dst.name.endswith(".json") or dst.name.endswith(".csv") or dst.name.endswith(".txt"):
            logger.info(f"Adjusting destination from {dst} to {dst.parent}")
            # 将解压目标调整为父目录
            dst = dst.parent
            
        # 按压缩格式选择对应的解压库
        try:
            # .7z 格式用 py7zr 解压
            if file.suffix == ".7z":
                with py7zr.SevenZipFile(file, mode="r") as ref:
                    # 解压全部内容到目标目录
                    ref.extractall(dst)
            # .zip 格式用 zipfile 解压
            elif file.suffix == ".zip":
                with zipfile.ZipFile(file, "r") as ref:
                    # 解压全部内容到目标目录
                    ref.extractall(dst)
            # .gz 或 .tar.gz 格式用 tarfile 以 gzip 模式解压
            elif file.suffix == ".gz" or file.name.endswith(".tar.gz"):
                with tarfile.open(file, "r:gz") as ref:
                    # 解压全部内容到目标目录
                    ref.extractall(dst)
            # 其余压缩格式暂不支持，抛出未实现异常
            else:
                raise NotImplementedError(f"Unsupported compression format: `{file.suffix}`.")
        except Exception as e:
            # final attempt to extract using shutil
            # 首选解压失败时记录错误，改用 shutil 兜底
            logger.error(f"Failed to extract {file} with suffix {file.suffix}: {e}; will try with shutil.")
            try:
                # 由 shutil 自动识别归档格式并解包
                shutil.unpack_archive(file, dst)
            except Exception as e2:
                # 兜底也失败则记录错误并把异常抛出
                logger.error(f"Failed to extract {file} with shutil: {e2}")
                raise e2

    # 内部函数：判断文件是否属于本工具可处理的压缩格式
    def is_compressed(file: Path) -> bool:
        """Check if the file is a compressed format that we can handle."""
        # 后缀在支持列表中，或文件名以 .tar.gz 结尾即视为压缩文件
        return file.suffix in {".7z", ".zip", ".gz"} or file.name.endswith(".tar.gz")

    # 若传入路径本身是压缩文件，先解压它自身
    if is_compressed(path):
        # .tar.gz 需去掉 7 个字符的后缀得到目标目录名
        if path.name.endswith(".tar.gz"):
            dst = path.parent / path.name[:-7]  # Remove .tar.gz
        # 其他压缩格式用去后缀的词干作为目标目录名
        else:
            dst = path.parent / path.stem

        # 目标目录已存在、未开启强制解压且尚未登记处理过时，跳过解压
        if dst.exists() and not force and dst not in already_extracted:
            logger.info(f"Skipping extraction for {path}, already exists at {dst}.")
        # 否则执行解压
        else:
            logger.info(f"Extracting {path} to {dst}")
            extract(path, dst)

        # 若开启了解压后删除压缩包的选项
        if delete_compressed:
            path_to_delete = path
            logger.info(f"Deleting compressed file {path_to_delete} after extraction.")
            # 删除已解压完成的压缩文件
            path_to_delete.unlink()

        # 后续递归改为在解压出的目录中进行
        path = dst

    # 若路径不是目录（可能是纯数据文件），无需继续解压
    if not path.is_dir():
        return
    # 找出目录下尚未处理过、且属于压缩格式的文件集合
    to_extract = {fpath for fpath in set(path.iterdir()) - already_extracted if is_compressed(fpath)}
    # 先登记待处理集合，防止递归过程中重复处理
    already_extracted.update(to_extract)
    # 递归解压每个子压缩文件（支持嵌套压缩包）
    for fpath in to_extract:
        extract_all_from_path(
            fpath, already_extracted=already_extracted, force=force, delete_compressed=delete_compressed
        )
