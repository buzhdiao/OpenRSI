# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 hashlib：用于对配置序列化后的字符串计算哈希指纹，唯一标识一份实验配置
import hashlib
# 导入 inspect：用于判断字段的类型标注是否是真正的类（区别于 typing 泛型对象）
import inspect
# 从标准库 dataclasses 导入 dataclass 装饰器与 fields 函数：前者声明数据类，后者枚举数据类的字段
from dataclasses import dataclass, fields
# 导入 Any：类型标注中表示任意类型
from typing import Any


# @dataclass 装饰器：自动生成 __init__、__repr__、__eq__ 等样板方法
@dataclass
# 所有配置类的基类：提供调试配置切换、递归校验、哈希指纹等通用能力
class BaseConfig:
    """
    BaseConfig is a base class for all configuration classes.
    """

    # 递归地将整个配置树切换为调试模式：对每个具备该方法属性的子配置逐级调用
    def configure_for_debug(self) -> None:
        """
        This function recursively calls the `configure_for_debug` function
        for all the attributes (if applicable).
        When implementing this method in the derived classes and overriding
        some fields, you may want to invoke the `configure_for_debug`
        method from the super class.
        ```
        """
        # 遍历本数据类声明的所有字段（fields 返回 Field 对象列表）
        for class_field in fields(self):
            # 通过反射 getattr 取出该字段当前的值
            field_value = getattr(self, class_field.name)
            # 若字段值本身拥有 configure_for_debug 方法（即嵌套的配置对象），则满足递归条件
            if _check_if_field_value_has_a_callable(field_value=field_value, name_of_callable="configure_for_debug"):
                # 调用子配置的 configure_for_debug，使其同样进入调试模式
                field_value.configure_for_debug()

    # 递归校验整个配置树：对每个具备 validate 方法的子配置逐级调用
    def validate(self) -> None:
        """
        This function recursively calls the `validate` function
        for all the attributes (if applicable).
        """
        # 遍历数据类的所有字段
        for class_field in fields(self):
            # 取出字段当前的值
            field_value = getattr(self, class_field.name)
            # 若字段值拥有 validate 方法（嵌套配置对象），则递归校验
            if _check_if_field_value_has_a_callable(field_value=field_value, name_of_callable="validate"):
                # 调用子配置自身的校验逻辑
                field_value.validate()

    # 类方法：收集所有需要从哈希计算中排除的字段，返回 (字段名, 字段类型) 列表
    @classmethod
    def fields_to_exclude_from_hash(cls) -> list[tuple[str, Any]]:
        # 初始化待排除字段列表，元素为 (字段名, 字段类型) 二元组
        fields_to_exclude: list[tuple[str, Any]] = []
        # 遍历本数据类的全部字段定义（__dataclass_fields__ 为字段名到 Field 对象的映射）
        for field_name, field_obj in cls.__dataclass_fields__.items():
            # 取出该字段的类型标注
            field_type = field_obj.type
            # Note: the reason we need to check if the field_type is a class
            # is that some fancy types (e.g. Optional[BaseConfig]) will be instances
            # of a typing object instead of a class. We currently do not support
            # excluding fields with types other than str, int, float from the hash.
            # 类型不是 str/int/float/BaseConfig 的真类（如 Optional[BaseConfig] 这类 typing 对象）时跳过
            if not (inspect.isclass(field_type) and issubclass(field_type, (str, int, float, BaseConfig))):
                continue

            # 若字段类型本身是 BaseConfig 的子类（嵌套配置），递归收集其内部待排除的字段
            if issubclass(field_type, BaseConfig):
                # 以 "外层字段名.内层字段名" 的点路径形式展开并合并进排除列表
                fields_to_exclude.extend(
                    (f"{field_name}.{nested_field_name}", nested_field_type)
                    for nested_field_name, nested_field_type in field_type.fields_to_exclude_from_hash()
                )
            # 否则若字段 metadata 中显式声明了 exclude_from_hash=True，则单独排除该字段
            elif field_obj.metadata.get("exclude_from_hash"):
                # 把 (字段名, 字段类型) 加入排除列表
                fields_to_exclude.append((field_name, field_type))

        # 返回汇总后的待排除字段列表
        return fields_to_exclude

    # 计算配置对象的哈希指纹（SHA-224），用于区分/比对不同的实验配置
    def hash(self) -> str:
        """
        This function hashes all fields of the dataclass
        except those explicitly with `exclude_from_hash=True`
        specified in the field metadata
        """
        # 取出所有待排除字段的名称（只保留二元组中的名字部分）
        fields_to_exclude = [field for field, _ in self.fields_to_exclude_from_hash()]
        # 初始化参与哈希的字段字典：字段名 -> 字符串化后的值
        fields_to_hash = {}
        # 遍历数据类的所有字段
        for class_field in fields(self):
            # 命中排除名单的字段直接跳过，不参与哈希
            if class_field.name in fields_to_exclude:
                continue
            # 取出字段值
            field_value = getattr(self, class_field.name)
            # 若字段值本身是 BaseConfig（嵌套配置），则递归取其哈希摘要
            if isinstance(field_value, BaseConfig):
                fields_to_hash[class_field.name] = field_value.hash()
            else:
                # 普通字段统一转为字符串，保证后续可拼接
                fields_to_hash[class_field.name] = str(field_value)

        # 将所有 "字段名_值" 项按字段名排序后用 '-' 连接（排序保证哈希结果确定）
        final_string_to_hash = "-".join(sorted(f"{key}_{value}" for key, value in fields_to_hash.items()))
        # 用 SHA-224 计算该字符串的十六进制哈希摘要并返回
        return hashlib.sha224(final_string_to_hash.encode()).hexdigest()


# 私有辅助函数：判断对象是否拥有名为 name_of_callable 且可调用的属性
def _check_if_field_value_has_a_callable(field_value: Any, name_of_callable: str):
    # hasattr 检查属性存在，callable 进一步确认其可被调用
    return hasattr(field_value, name_of_callable) and callable(getattr(field_value, name_of_callable))
