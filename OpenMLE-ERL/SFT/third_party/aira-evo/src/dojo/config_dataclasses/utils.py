# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 用于判断 dataclass 类型并枚举其字段
from dataclasses import is_dataclass, fields
# 类型注解工具
from typing import Any, Type


def dataclass_from_dict(dataclass_type: Type[Any], data: dict) -> Any:
    """Recursively convert a dictionary to a dataclass instance."""
    # 校验传入类型确实是 dataclass，否则抛出值错误
    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} is not a dataclass type")
    # 收集构造实例所需的关键字参数
    init_args = {}
    # 遍历该 dataclass 声明的每个字段
    for field in fields(dataclass_type):
        # 从字典中按字段名取值（缺失则为 None）
        field_value = data.get(field.name)
        # 字段类型本身是 dataclass 且值是字典 -> 需要递归还原
        if is_dataclass(field.type) and isinstance(field_value, dict):
            # Recursively convert nested dataclass fields
            # 递归构建子对象
            init_args[field.name] = dataclass_from_dict(field.type, field_value)
        else:
            # 普通字段直接使用字典中的值
            init_args[field.name] = field_value
    # 用收集好的参数实例化目标 dataclass
    return dataclass_type(**init_args)
