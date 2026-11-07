# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 启用延迟类型求值：允许类型注解写成字符串，避免运行时导入开销/循环依赖
from __future__ import annotations

# 导入 Iterator 与 Mapping：实现只读惰性映射所需的抽象接口
from collections.abc import Iterator, Mapping
# 导入 Any：类型标注中的任意类型
from typing import Any


# 惰性解释器映射类：仅在真正被访问时才导入并物化解释器注册表
class _LazyInterpreterMap(Mapping[str, Any]):
    # 物化函数：实际执行 import 并返回 名称->解释器类 的字典
    def _materialize(self) -> dict[str, Any]:
        # 延迟导入 Python 解释器类：把昂贵/带副作用的导入推迟到首次访问
        from dojo.core.interpreters.python import PythonInterpreter

        # 返回配置名到解释器实现的映射
        return {
            "PythonInterpreterConfig": PythonInterpreter,
        }

    # 按键取值：触发惰性物化后查字典
    def __getitem__(self, key: str) -> Any:
        return self._materialize()[key]

    # 迭代键名：同样通过惰性物化获得键集合
    def __iter__(self) -> Iterator[str]:
        return iter(self._materialize())

    # 返回键数量：物化后取字典长度
    def __len__(self) -> int:
        return len(self._materialize())


# 模块级导出的只读惰性映射：对使用者表现为普通字典，但不提前加载解释器
INTERPRETER_MAP: Mapping[str, Any] = _LazyInterpreterMap()
