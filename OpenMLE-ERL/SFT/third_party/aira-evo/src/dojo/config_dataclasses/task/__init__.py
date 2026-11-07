# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 延迟注解求值，注解统一按字符串处理
from __future__ import annotations

# 映射协议所需的抽象基类
from collections.abc import Iterator, Mapping
# 任意类型注解
from typing import Any


# 懒加载的任务注册表：真正访问时才 import 任务实现，避免包初始化时的循环导入
class _LazyTaskMap(Mapping[str, Any]):
    # 惰性构建映射：此时才真正导入任务实现类
    def _materialize(self) -> dict[str, Any]:
        # 惰性导入 MLE-Bench 任务实现
        from dojo.tasks.mlebench.task import MLEBenchTask

        # 配置类名 -> 任务实现类 的映射
        return {"MLEBenchTaskConfig": MLEBenchTask}

    # 按键取任务类时才触发导入并构建映射
    def __getitem__(self, key: str) -> Any:
        return self._materialize()[key]

    # 迭代键时同样惰性构建
    def __iter__(self) -> Iterator[str]:
        return iter(self._materialize())

    # 求映射长度时惰性构建
    def __len__(self) -> int:
        return len(self._materialize())


# 模块级任务映射实例（访问时才真正加载）
TASK_MAP: Mapping[str, Any] = _LazyTaskMap()
