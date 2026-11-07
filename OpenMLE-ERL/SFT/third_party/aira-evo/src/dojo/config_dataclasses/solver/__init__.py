# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 延迟注解求值，允许类型注解里直接写字符串形式
from __future__ import annotations

# 映射协议所需的抽象基类
from collections.abc import Iterator, Mapping
# 任意类型注解
from typing import Any


# 懒加载的求解器注册表：只有在真正访问时才 import 各求解器实现，
# 避免包初始化阶段的循环导入与不必要的加载开销
class _LazySolverMap(Mapping[str, Any]):
    # 惰性构建映射：此时才真正导入各求解器实现类
    def _materialize(self) -> dict[str, Any]:
        # 惰性导入进化搜索（Evolutionary，演化策略）求解器
        from dojo.solvers.evo import Evolutionary
        # 惰性导入贪心（Greedy，爬山式）求解器
        from dojo.solvers.greedy import Greedy
        # 惰性导入蒙特卡洛树搜索（MCTS）求解器
        from dojo.solvers.mcts import MCTS

        # 配置类名 -> 求解器实现类 的映射
        return {
            "GreedySolverConfig": Greedy,
            "MCTSSolverConfig": MCTS,
            "EvolutionarySolverConfig": Evolutionary,
        }

    # 按键取求解器时才触发导入并构建映射
    def __getitem__(self, key: str) -> Any:
        return self._materialize()[key]

    # 迭代键时同样惰性构建
    def __iter__(self) -> Iterator[str]:
        return iter(self._materialize())

    # 求映射长度时惰性构建
    def __len__(self) -> int:
        return len(self._materialize())


# 模块级求解器映射实例（访问时才真正加载）
SOLVER_MAP: Mapping[str, Any] = _LazySolverMap()
