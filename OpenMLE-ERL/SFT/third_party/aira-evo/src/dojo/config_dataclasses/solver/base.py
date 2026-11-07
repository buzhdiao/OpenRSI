# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 dataclass 装饰器与 field 工厂
from dataclasses import dataclass, field

# OmegaConf 配置库：SI 为字符串插值，MISSING 为必填占位符
from omegaconf import SI, MISSING

# 通用配置基类
from aira_core.config.base import BaseConfig
# 算子配置（LLM 调用单元）
from dojo.config_dataclasses.operators.base import OperatorConfig

# 记忆算子配置
from dojo.config_dataclasses.operators.memory import MemoryOpConfig


# 求解器（Solver）基类配置：定义搜索过程的通用参数（步数、资源、评估方式等）；
# 求解器即驱动整个智能体搜索循环的策略模块
@dataclass
class SolverConfig(BaseConfig):
    # 搜索过程允许的最大步数（防止无限循环搜索）
    step_limit: int = field(
        default=500,
        metadata={
            "description": "Maximum number of steps allowed in the search process.",
            "example": 300,
        },
    )

    # 求解器可使用的第三方包白名单（覆盖数值计算/数据分析/机器学习/深度学习等常用工具）
    available_packages: list[str] = field(
        default_factory=lambda: [
            "numpy",  # Numerical computing
            "pandas",  # Data manipulation and analysis
            "scikit-learn",  # Machine learning library
            "statsmodels",  # Statistical modeling and econometrics
            "xgboost",  # Gradient boosting for structured data
            "lightgbm",  # Efficient gradient boosting framework
            "torch",  # PyTorch deep learning framework
            "torchvision",  # Image processing utilities for PyTorch
            "torch-geometric",  # Graph-based deep learning with PyTorch
            "bayesian-optimization",  # Bayesian optimization library
            "timm",  # PyTorch image models collection
        ],
        metadata={
            "description": "List of available packages for the solver.",
            "example": ["numpy", "pandas"],
        },
    )
    # 求解器用到的算子集合：名称 -> 算子配置（由具体求解器子类填充）
    operators: dict[str, OperatorConfig] = field(
        default_factory={},
        metadata={
            "description": "List of operator names for the solver.",
            "example": {},
        },
    )
    # 正常搜索循环使用的记忆策略配置
    memory: MemoryOpConfig = field(
        default_factory=MemoryOpConfig,
        metadata={
            "description": "Memory configuration for the solver.",
            "example": {},
        },
    )
    # 调试循环（修复报错阶段）单独使用的记忆策略配置
    debug_memory: MemoryOpConfig = field(
        default_factory=MemoryOpConfig,
        metadata={
            "description": "Memory configuration for debug cycle of the solver.",
            "example": {},
        },
    )

    # Experiment name variable used in path configuration.
    # 实验名称：用于生成输出路径，默认插值为实验唯一 ID
    exp_name: str = field(
        default=SI("${id}"),
        metadata={
            "description": "Experiment name variable used in path configuration.",
            "example": "user_...",
            "exclude_from_hash": True,
        },
    )

    # 单次代码执行（解释器运行）的超时时间，默认 14400 秒（4 小时）
    execution_timeout: int = field(
        default=14400,
        metadata={
            "help": "The timeout for the interpreter in seconds.",
        },
    )

    # 整个任务的总时间预算，默认 86400 秒（24 小时）
    time_limit_secs: int = field(
        default=86400,
        metadata={
            "help": "The time limit for the task in seconds.",
        },
    )

    # 是否在执行结束后导出搜索过程结果
    export_search_results: bool = field(
        default=True,
        metadata={
            "description": "Whether to export search results after execution",
            "example": True,
            "exclude_from_hash": True,
        },
    )

    # 检查点（checkpoint）保存路径，用于断点续跑
    checkpoint_path: str = field(
        default=SI("${logger.output_dir}/checkpoint"),
        metadata={
            "description": "Path to the checkpoint file.",
            "example": "path/to/checkpoint",
            "exclude_from_hash": True,
        },
    )

    # 评估时是否使用测试集分数（默认不用，防止对测试集过拟合）
    use_test_score: bool = field(
        default=False,
        metadata={
            "description": "Whether to use the test score for evaluation",
            "example": True,
        },
    )

    # 提示词中是否考虑方案复杂度差异（仅部分算子支持）
    use_complexity: bool = field(
        default=False,
        metadata={
            "description": "Whether to consider complexity differences in prompts - only works with certain operators",
            "example": True,
        },
    )

    # LLM API 调用失败后的最大重试次数
    max_llm_call_retries: int = field(
        default=3,
        metadata={
            "description": "Maximum number of retries for failed LLM API calls.",
            "example": 2,
            "exclude_from_hash": True,
        },
    )

    # 配置校验入口
    def validate(self) -> None:
        # 复用父类校验逻辑，本类无额外约束
        super().validate()
