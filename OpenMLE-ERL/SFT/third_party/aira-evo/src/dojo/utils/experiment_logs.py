# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 Path 与 Union，用于路径处理和类型标注
from pathlib import Path
from typing import Union


# 判断给定路径是否为一个实验目录（单次运行记录的目录）
def is_experiment(path: Union[Path, str]) -> bool:
    """
    Check if the given file path is an experiment directory.
    An experiment directory contains a 'json' subdirectory.
    """
    # Feel free to make this more robust by checking for specific files or directories.
    # 统一转换为 Path 对象
    path = Path(path)
    # 不是目录则不可能是实验目录
    if not path.is_dir():
        return False
    # 实验目录的标志：存在 dojo_config.json 配置文件
    path_to_cfg = path / "dojo_config.json"
    # 配置文件不存在则判定不是实验目录
    if not path_to_cfg.exists():
        return False
    # 以上条件都满足，判定为实验目录
    return True


# 判断给定路径是否为"元实验"目录（聚合了多个实验目录）
def is_meta_experiment(path: Union[Path, str]) -> bool:
    """
    Check if the given file path is a meta-experiment directory.
    A meta-experiment directory contains multiple experiment directories.
    """
    # 任一子目录是实验目录即判定为元实验目录
    return any(is_experiment(child) for child in Path(path).iterdir() if child.is_dir())
