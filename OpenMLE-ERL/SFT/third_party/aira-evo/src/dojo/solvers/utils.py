# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 Enum：定义枚举类型
from enum import Enum
# 导入搜索树节点类型，仅用于函数签名注解
from dojo.core.solvers.utils.journal import Node


# 复杂度枚举：标注候选方案代码的复杂程度（低/中/高），供起草算子控制提示词
class Complexity(Enum):
    # 低复杂度：简单方案
    LOW = "simple"
    # 中复杂度：常规方案
    MEDIUM = "normal"
    # 高复杂度：复杂方案
    HIGH = "complex"

    # 使枚举成员在打印/f-string 中直接显示其字符串值
    def __str__(self):
        return self.value


# 根据子节点数量（或给定数值）判定复杂度级别；函数名中 "complextiy" 为源码原有拼写笔误
def get_complextiy_level(node: Node = None, num: int = None) -> Complexity:
    """
    Determine the complexity level based on the number of children.

    Args:
        node: The node to evaluate
        num: The number of children (optional)

    Returns:
        Complexity: The complexity level (LOW, MEDIUM, HIGH)
    """
    # 若传入了节点，则按其子节点数量判断复杂度
    if node is not None:
        # 子节点数少于 2：视为低复杂度
        if len(node.children) < 2:
            return Complexity.LOW
        # 子节点数少于 4：视为中复杂度
        elif len(node.children) < 4:
            return Complexity.MEDIUM
        else:
            # 其余情况（4 个及以上子节点）：高复杂度
            return Complexity.HIGH

    # 若未传节点但给定了数值 num，则直接按数值判断
    if num is not None:
        # 数值小于 2：低复杂度
        if num < 2:
            return Complexity.LOW
        # 数值小于 4：中复杂度
        elif num < 4:
            return Complexity.MEDIUM
        else:
            # 其余情况：高复杂度
            return Complexity.HIGH

    # 节点与数值均未提供时，抛出参数错误
    raise ValueError("Either node or num must be provided.")
