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


# 进化求解器（Evolutionary Solver）配置：基于岛屿模型的进化搜索（island-model evolution）；
# 维护候选解种群，经变异/交叉/迁移不断迭代改进
@dataclass
class EvolutionarySolverConfig(SolverConfig):
    # Hyperparameters for the evolutionary process.
    # 岛屿（子种群）数量；MISSING 表示必须由用户显式提供
    num_islands: int = field(default=MISSING, metadata={"help": "Number of islands (sub-populations) to initialize."})
    # 每个岛屿允许保留的最大样本（个体）数
    max_island_size: int = field(
        default=MISSING, metadata={"help": "Maximum number of samples allowed in each island."}
    )
    # 选择交叉（crossover）而非变异（mutation）的概率
    crossover_prob: float = field(
        default=MISSING, metadata={"help": "Probability of choosing a crossover operation over mutation."}
    )
    # 用强岛屿的优质个体刷新弱岛屿（迁移）的概率
    migration_prob: float = field(
        default=MISSING,
        metadata={"help": "Probability of refreshing weaker islands with seeds from stronger islands."},
    )
    # 探索起始温度（模拟退火式温度调度，高温利于探索）
    initial_temp: float = field(default=MISSING, metadata={"help": "Starting temperature for exploration."})
    # 结束温度：温度降低后偏向利用（exploitation）
    final_temp: float = field(
        default=MISSING, metadata={"help": "Final temperature value, shifting towards exploitation."}
    )
    # 距离允许发生迁移还需要多少代
    num_generations_till_migration: int = field(
        default=MISSING, metadata={"help": "Number of generations before migration can occur."}
    )
    # 距离允许发生交叉还需要多少代
    num_generations_till_crossover: int = field(
        default=MISSING, metadata={"help": "Number of generations before crossover can occur."}
    )

    # Few-shot prompting configuration for different operations.
    # 各操作在提示词中附带的少样本（few-shot）示例个数：默认 improve 1 个、crossover 2 个
    few_shot: dict = field(
        default_factory=lambda: {"improve": 1, "crossover": 2},
        metadata={"help": "Few-shot prompting configuration for different operations."},
    )

    # Evolution settings for the search process.
    # 总进化代数
    num_generations: int = field(default=MISSING, metadata={"help": "Total number of generations to run."})
    # 每一代包含的个体数
    individuals_per_generation: int = field(
        default=MISSING, metadata={"help": "Number of individuals per generation."}
    )

    # 调试分析允许的最大耗时
    max_debug_time: float = field(
        default=MISSING, metadata={"description": "Maximum time allowed for debugging analysis"}
    )
    # 调试分析允许的最大递归深度
    max_debug_depth: int = field(default=MISSING, metadata={"description": "Maximum depth of debugging analysis"})

    # --- Agent Configuration ---
    # 是否在执行前给智能体展示数据预览（前几行等）
    data_preview: bool = field(
        default=MISSING,
        metadata={"description": "Whether to provide the agent with a preview of the data before execution"},
    )

    # 配置校验入口
    def validate(self) -> None:
        # 复用父类校验逻辑（MISSING 字段由 OmegaConf 在解析时强制要求填写）
        super().validate()
