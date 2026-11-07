# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 dataclass 装饰器与 field 工厂
from dataclasses import dataclass, field

# OmegaConf 配置库：SI 为字符串插值，MISSING 为必填占位符
from omegaconf import SI, MISSING

# 求解器基类配置
from dojo.config_dataclasses.solver.base import SolverConfig


# MCTS 求解器配置：蒙特卡洛树搜索（Monte Carlo Tree Search）策略，
# 在候选解树上按 UCB 公式平衡探索（exploration）与利用（exploitation）
@dataclass
class MCTSSolverConfig(SolverConfig):
    # --- Search Configuration ---
    # 每个搜索步展开的子节点数量
    num_children: int = field(
        default=MISSING, metadata={"description": "Number of child nodes expanded per search step"}
    )
    # 调试分析允许的最大递归深度
    max_debug_depth: int = field(default=MISSING, metadata={"description": "Maximum depth of debugging analysis"})
    # UCT 公式的探索常数（UCB 上置信界中的 c），越大越偏向探索
    uct_c: float = field(
        default=MISSING, metadata={"description": "Upper Confidence Bound (UCB) exploration constant"}
    )
    # 调试分析允许的最大耗时
    max_debug_time: float = field(
        default=MISSING, metadata={"description": "Maximum time allowed for debugging analysis"}
    )
    # --- Agent Configuration ---
    # 是否在执行前给智能体展示数据预览
    data_preview: bool = field(
        default=MISSING,
        metadata={"description": "Whether to provide the agent with a preview of the data before execution"},
    )

    # 配置校验入口
    def validate(self) -> None:
        # 复用父类校验逻辑（MISSING 字段须由用户显式配置）
        super().validate()
