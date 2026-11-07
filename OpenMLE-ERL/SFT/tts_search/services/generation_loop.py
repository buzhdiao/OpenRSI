"""Generation loop control for batched rejection sampling."""
# 生成循环控制模块：决定"每个任务还要继续生成多少样本、何时停止"，
# 与生成/评估服务解耦（批量拒绝采样 rejection sampling 的节奏控制）。

from __future__ import annotations

# 导入 dataclass：声明配置结构
from dataclasses import dataclass
# 导入类型注解 Any
from typing import Any


@dataclass(frozen=True)
class GenerationLoopConfig:
    """Rules for when to keep generating samples for one task."""

    # 字段说明：batch_size=每批生成数量；accepted_target=期望接受的样本数目标；
    # max_generated=单个任务最大生成上限（None 表示不设上限）
    batch_size: int = 4
    accepted_target: int = 4
    max_generated: int | None = 32

    def __post_init__(self) -> None:
        # 校验 batch_size 必须为正
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        # 校验 accepted_target 必须为正
        if self.accepted_target <= 0:
            raise ValueError("accepted_target must be > 0")
        # 校验 max_generated（若设置）必须为正
        if self.max_generated is not None and self.max_generated <= 0:
            raise ValueError("max_generated must be > 0")


class GenerationLoopController:
    """Controls per-task generation targets independently of gen/eval services."""

    def __init__(self, config: GenerationLoopConfig):
        # 保存不可变的循环配置
        self._config = config

    @property
    def batch_size(self) -> int:
        # 暴露每批生成数量
        return self._config.batch_size

    @property
    def accepted_target(self) -> int:
        # 暴露接受目标数
        return self._config.accepted_target

    def initialize_state(self, state: Any) -> None:
        """Ensure a fresh state starts with one batch target."""

        # 目标数尚未初始化（<=0）时：设为一个批次的大小
        if getattr(state, "target_count", 0) <= 0:
            state.target_count = self.batch_size
        # 生成上限尚未初始化时：写入配置的最大生成数
        if getattr(state, "max_target", None) is None:
            state.max_target = self._config.max_generated

    def generation_needed(self, state: Any) -> int:
        """Return how many new generations should be queued for this state."""

        # 任务已结束则不再排队新生成
        if getattr(state, "done", False):
            return 0
        # 需补量 = 目标数 - 已生成数 - 在途生成数（下限 0）
        return max(
            0,
            int(state.target_count)
            - int(state.generated_count)
            - int(state.pending_gen),
        )

    def advance_after_completion(self, state: Any) -> int:
        """Advance target after completed evals and return additional needed gens."""

        # 任务完成数已达到当前目标时，尝试滚动扩批
        while not state.done and state.completed_count >= state.target_count:
            # 已满足接受目标或已到生成上限：结束循环
            if self._target_met(state) or self._max_target_reached(state):
                state.done = True
                break

            # 目标数按批扩大
            next_target = state.target_count + self.batch_size
            if self._config.max_generated is not None:
                # 但不超过最大生成上限
                next_target = min(next_target, self._config.max_generated)
            if next_target <= state.target_count:
                # 目标无法再扩大（被上限钳住）：结束循环
                state.done = True
                break
            # 更新目标数，继续下一批
            state.target_count = next_target

        # 返回完成评估后还需要补排的生成数量
        return self.generation_needed(state)

    def _target_met(self, state: Any) -> bool:
        # 读取状态上自带的接受目标（没有则用全局配置）
        accepted_target = getattr(state, "accepted_target", None)
        if accepted_target is None:
            accepted_target = self.accepted_target
        # 已接受样本数达到目标即视为达标
        return int(state.accepted_count) >= int(accepted_target)

    def _max_target_reached(self, state: Any) -> bool:
        # 配置了上限且当前目标数已达到该上限
        return (
            self._config.max_generated is not None
            and int(state.target_count) >= self._config.max_generated
        )
