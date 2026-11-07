# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 dataclass 装饰器与 field：声明数据类字段及其元数据
from dataclasses import dataclass, field

# 导入 SI：OmegaConf 的插值符号（继承字段中使用，本文件仅声明新字段）
from omegaconf import SI

# 导入解释器基类配置：Python 解释器配置在其基础上扩展
from dojo.config_dataclasses.interpreter.base import InterpreterConfig


# @dataclass：自动生成构造/比较等方法
@dataclass
# Python 解释器配置类：在基类上增加 Python 特有的执行选项
class PythonInterpreterConfig(InterpreterConfig):
    # 是否用符号链接代替文件复制的开关：对大文件可避免昂贵的拷贝
    use_symlinks: bool = field(
        default=True,
        metadata={"help": "Use symlinks to avoid copying files. This is useful for large files."},
    )

    # 校验逻辑：直接复用基类的递归校验
    def validate(self) -> None:
        super().validate()
