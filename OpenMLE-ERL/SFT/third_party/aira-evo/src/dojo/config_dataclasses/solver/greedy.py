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


# 贪心求解器（Greedy Solver）配置：爬山式迭代改进（hill climbing）策略，
# 每步基于当前最优解生成并挑选更优候选，不做全局种群搜索
@dataclass
class GreedySolverConfig(SolverConfig):
    # --- Search Configuration ---
    # 执行改进迭代的轮数（爬山步数）
    improvement_steps: int = field(
        default=5,
        metadata={
            "description": "Number of improvement iterations to perform.",
            "example": 3,
        },
    )

    # 是否在执行前给智能体展示数据预览
    data_preview: bool = field(
        default=False,
        metadata={
            "description": "Whether to provide the agent with a preview of the data before execution.",
            "example": True,
        },
    )

    # --- Debugging Configuration ---
    # 调试分析允许的最大递归深度
    max_debug_depth: int = field(
        default=3,
        metadata={
            "description": "Maximum depth of debugging analysis.",
            "example": 2,
        },
    )

    # 搜索过程中触发调试步骤的概率
    debug_prob: float = field(
        default=0.5,
        metadata={
            "description": "Probability of running a debug step in the process.",
            "example": 0.3,
        },
    )

    # --- Drafting Configuration ---
    # 每步生成的候选草稿（draft）数量，供后续择优
    num_drafts: int = field(
        default=5,
        metadata={
            "description": "Number of draft outputs to generate for selection.",
            "example": 3,
        },
    )

    # 配置校验入口
    def validate(self) -> None:
        # 复用父类校验逻辑，本类无额外约束
        super().validate()
