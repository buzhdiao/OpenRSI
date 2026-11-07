"""
Services module for decoupled code generation and evaluation.

This module implements a decoupled architecture similar to SGLang's prefill/decode
separation, where code generation and sandbox evaluation are handled by separate
services coordinated through a scheduler.
"""
# 服务子包：把代码生成（generator）与沙箱评估（evaluator）解耦为两个独立服务，
# 由调度器（scheduler）通过异步队列协同，配合拒绝采样策略（rejection）筛选样本。

# 导入评估服务：负责沙箱执行与打分
from tts_search.services.evaluator import EvaluatorService
# 导入生成循环控制器：控制每任务"生成多少/何时停止"
from tts_search.services.generation_loop import (
    GenerationLoopConfig,
    GenerationLoopController,
)
# 导入生成服务：负责调用 LLM 生成代码
from tts_search.services.generator import GeneratorService
# 导入各类拒绝采样（rejection sampling）策略
from tts_search.services.rejection import (
    AcceptAllPolicy,
    AcceptScoredPolicy,
    BaselinePostprocessPolicy,
    BetterThanReferencePolicy,
    MedalPolicy,
    MixedLeaderboardBaselinePolicy,
    RejectionDecision,
    RewardThresholdPolicy,
    ScoreThresholdPolicy,
    build_rejection_policy,
)
# 导入调度器及其配置：协调生成/评估两个服务
from tts_search.services.scheduler import Scheduler, SchedulerConfig

# 对外导出的公开接口清单
__all__ = [
    "GeneratorService",
    "EvaluatorService",
    "GenerationLoopConfig",
    "GenerationLoopController",
    "AcceptAllPolicy",
    "AcceptScoredPolicy",
    "BaselinePostprocessPolicy",
    "BetterThanReferencePolicy",
    "MedalPolicy",
    "MixedLeaderboardBaselinePolicy",
    "RejectionDecision",
    "RewardThresholdPolicy",
    "ScoreThresholdPolicy",
    "build_rejection_policy",
    "Scheduler",
    "SchedulerConfig",
]
