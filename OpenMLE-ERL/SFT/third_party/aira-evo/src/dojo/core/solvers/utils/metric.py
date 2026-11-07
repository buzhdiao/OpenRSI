# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are MIT-licensed
# Copyright (c) 2024 Weco AI Ltd
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/WecoAI/aideml/blob/main/LICENSE

# 导入 sys 模块，用于获取最大整数（最差指标哨兵值）
import sys
# 导入 dataclass/field，用于声明数据类字段
from dataclasses import dataclass, field
# 导入 total_ordering，只需实现 __eq__/__gt__ 即可自动补全其余比较运算
from functools import total_ordering
# 导入类型注解工具
from typing import Any

# 导入 numpy，用于支持数值与数组类型指标
import numpy as np
# 导入 DataClassJsonMixin，让数据类支持 JSON 序列化
from dataclasses_json import DataClassJsonMixin


@dataclass
@total_ordering
class MetricValue(DataClassJsonMixin):
    """
    Represents the value of a metric to be optimized, which can be compared to other metric values.
    Comparisons (and max, min) are based on which value is better, not which is larger.
    """

    # 指标数值：可为浮点/整数/numpy 数值类型或 None（None 表示无效/最差）
    value: float | int | np.number | np.floating | np.ndarray | None
    # 优化方向：True 表示越大越好（maximize），False 表示越小越好，None 表示未知
    maximize: bool | None = field(default=None, kw_only=True, compare=False)
    # 指标附属信息字典（如混淆矩阵、曲线数据等）
    info: dict = field(default_factory=lambda: dict(), kw_only=True, compare=False)

    def __post_init__(self):
        # 数据类初始化后的规整逻辑
        if self.value is not None:
            # 数值非空时断言其为基础数值类型（拒绝数组/字符串等）
            assert isinstance(self.value, (float, int, np.number, np.floating))
            # 统一转换为 float，保证后续比较与序列化一致
            self.value = float(self.value)

    def __gt__(self, other) -> bool:
        """True if self is a _better_ (not necessarily larger) metric value than other"""
        if self.value is None:
            # 自身无值（最差）时不可能优于任何对象
            return False
        if other.value is None:
            # 对方无值（最差）时自身必然更优
            return True

        # 仅在同类型且优化方向一致时才有可比性
        assert type(self) is type(other) and (self.maximize == other.maximize)

        if self.value == other.value:
            # 数值相等则不存在"严格更优"
            return False

        # 先按数值大小比较，再按优化方向决定是否取反（maximize：大者优；minimize：小者优）
        comp = self.value > other.value
        return comp if self.maximize else not comp  # type: ignore

    def __eq__(self, other: Any) -> bool:
        # 相等比较基于原始数值
        return self.value == other.value

    def __repr__(self) -> str:
        # repr 直接复用 __str__ 的输出
        return str(self)

    def __str__(self) -> str:
        if self.maximize is None:
            # 优化方向未知时用问号占位
            opt_dir = "?"
        elif self.maximize:
            # 最大化方向显示向上箭头
            opt_dir = "↑"
        else:
            # 最小化方向显示向下箭头
            opt_dir = "↓"
        # 输出形如 Metric↑(0.8765) 的可读描述
        return f"Metric{opt_dir}({self.value_npsafe:.4f})"

    @property
    def is_worst(self):
        """True if the metric value is the worst possible value."""
        # 数值为 None 即代表"最差"（无效指标）
        return self.value is None

    @property
    def value_npsafe(self):
        # 返回 numpy 安全的数值：无值时用 NaN 占位，避免数值计算报错
        return self.value if self.value is not None else float("nan")

    @property
    def get_value(self):
        # 返回可直接用于数值比较的值
        if self.value is None:
            # 无值时返回极值哨兵：最大化场景取最小整数，最小化场景取最大整数
            return -sys.maxsize if self.maximize else sys.maxsize

        # 正常情况直接返回数值
        return self.value

    def add_value(self, add_value: float):
        # 在当前值上累加增量（无值时不做任何事）
        if self.value is None:
            return
        self.value += add_value


@dataclass
class WorstMetricValue(MetricValue):
    """
    Represents an invalid metric value, e.g. when the agent creates a buggy solution.
    Always compares worse than any valid metric value.
    """

    # 该类型的 value 恒为 None（即"最差"）
    value: None = None

    def __repr__(self):
        # repr 复用父类实现
        return super().__repr__()

    def __str__(self):
        # str 复用父类实现
        return super().__str__()
