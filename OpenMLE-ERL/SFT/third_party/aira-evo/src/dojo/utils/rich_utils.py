# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are Apache 2.0 licensed
# Copyright (c) 2023 Neptune Labs.
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/neptune-ai/neptune-client/blob/master/LICENSE

# 导入日志模块，用于提示配置字段缺失
import logging
# 导入 math，用于判断浮点是否为无穷/NaN
import math
# 导入 time（预留）
import time
# 导入 Path，用于保存配置树文件
from pathlib import Path
# 导入类型注解工具
from typing import Any, Mapping, MutableMapping, Sequence, Union

# 导入 rich 库，用于终端美化输出
import rich
# 导入 rich 的语法高亮组件
import rich.syntax
# 导入 rich 的树形结构组件
import rich.tree
# 导入 omegaconf 配置对象及其解析工具
from omegaconf import DictConfig, OmegaConf

# 导入运行配置数据类
from dojo.config_dataclasses.run import RunConfig

# Define the boundaries for 32-bit integers.
# 定义 32 位整数边界：超出该范围的整数在部分后端（如 wandb）无法记录
MAX_32_BIT_INT = 2147483647
MIN_32_BIT_INT = -2147483648


def is_unsupported_float(value: Any) -> bool:
    """
    Determine if the given value is a float that is either infinite or NaN.

    Args:
        value (Any): The value to check.

    Returns:
        bool: True if the value is a float and is either infinite or NaN, otherwise False.
    """
    # 仅浮点类型才可能出现无穷/NaN
    if isinstance(value, float):
        # 是无穷或 NaN 则属于不支持的浮点值
        return math.isinf(value) or math.isnan(value)
    # 其他类型均视为支持
    return False


class StringifyValue:
    """
    A wrapper class that processes and stores a value ensuring that it is
    safe for string conversion.

    The class performs the following:
      - If the value is an integer outside the 32-bit range, it converts it to a float.
      - If the value is a float that is unsupported (infinite or NaN), it converts it to a string.
    """

    def __init__(self, value: Any):
        """
        Process the input value to ensure it is suitable for stringification.

        Args:
            value (Any): The value to process.
        """
        # Convert integers outside the 32-bit range to floats.
        # 超出 32 位范围的整数转为浮点数，保证可被日志后端接受
        if isinstance(value, int) and (value > MAX_32_BIT_INT or value < MIN_32_BIT_INT):
            value = float(value)

        # Convert unsupported floats (infinity or NaN) to strings.
        # 不支持的浮点值（无穷/NaN）转为字符串保存
        if is_unsupported_float(value):
            value = str(value)

        # 保存处理后的值
        self.__value = value

    @property
    def value(self) -> Any:
        """
        Retrieve the processed value.

        Returns:
            Any: The processed value.
        """
        # 返回处理后的原始值
        return self.__value

    def __str__(self) -> str:
        """
        Return the string representation of the processed value.

        Returns:
            str: The string representation.
        """
        # 返回普通字符串形式
        return str(self.__value)

    def __repr__(self) -> str:
        """
        Return the official string representation of the processed value.

        Returns:
            str: The official string representation.
        """
        # 返回官方表示形式（repr）
        return repr(self.__value)


def stringify_unsupported(value: Any) -> Union[StringifyValue, Mapping[str, Any]]:
    """
    Recursively process a value, converting unsupported types to a StringifyValue instance.

    If the provided value is a mutable mapping (like a dictionary), the function will
    recursively process each key-value pair, ensuring that:
      - All keys are converted to strings.
      - All values are processed so that unsupported values are wrapped accordingly.

    Args:
        value (Any): A dictionary-like object or a single value.

    Returns:
        Union[StringifyValue, Mapping[str, Any]]:
            - A processed dictionary with stringified keys and values, or
            - A StringifyValue instance wrapping the processed value.
    """
    # 字典类输入：键统一转字符串，值递归处理
    if isinstance(value, MutableMapping):
        # Recursively process each item in the mapping.
        # 逐项递归转换，保证嵌套结构中所有键值都被安全化
        return {str(k): stringify_unsupported(v) for k, v in value.items()}

    # 普通值：包装为 StringifyValue（内部做越界整数/异常浮点转换）
    return StringifyValue(value=value)


def print_config_tree(
    cfg: RunConfig,
    print_order: Sequence[str] = [],
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        save_to_file (bool, optional): Whether to export config to the hydra output folder.
    """

    # 树形样式：暗淡显示
    style = "dim"
    # 创建以 CONFIG 为根节点的配置树
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    # 待打印字段的队列
    queue = []

    # add fields from `print_order` to queue
    # 先按指定顺序把存在于配置中的字段加入队列，缺失字段记录警告
    for field in print_order:
        queue.append(field) if field in cfg else logging.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    # 再把其余未指定的字段追加到队列尾部
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    # 依队列逐字段生成配置树分支
    for field in queue:
        # 为该字段添加一个分支节点
        branch = tree.add(field, style=style, guide_style=style)

        # 取出该字段的配置内容
        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            # 嵌套 DictConfig 转为 YAML 文本（可选解析引用）
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            # 普通值直接转字符串
            branch_content = str(config_group)

        # 分支内容按 YAML 语法高亮展示
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    # 在终端打印配置树
    rich.print(tree)

    # save config tree to file
    # 需要时把配置树写入输出目录的 config_tree.log
    if save_to_file:
        with open(Path(cfg.logger.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)
