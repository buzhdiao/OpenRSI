"""Reusable data production utilities for SFT dataset construction."""

# SFT 数据生产公共工具包：聚合各子模块的配置类与处理函数，统一对外导出
# 对齐模块：任务采样、训练数据框转换与对齐数据集写出
from tts_search.data_produce.alignment import (
    sample_tasks,
    to_training_frame,
    write_alignment_dataset,
)
# 基线差距过滤模块：基于基线 token 差距的评估与决策
from tts_search.data_produce.baseline_filter import (
    BaselineTokenGapConfig,
    BaselineTokenGapDecision,
    evaluate_baseline_token_gap,
)
# 收集模块：汇总生成 SFT 训练行
from tts_search.data_produce.collect import collect_sft_rows
# 差距过滤模块：相对差距的计算、标注与筛选
from tts_search.data_produce.gap_filter import (
    GapFilterConfig,
    annotate_gap_rows,
    choose_gap_denominator,
    choose_legacy_gap_denominator,
    classify_metric,
    enrich_tasks_with_metric_protocol,
    filter_by_relative_gap,
    load_task_meta_from_parquet,
    relative_gap_from_feedback,
)
# 流水线模块：数据生产总配置与 SFT 数据集构建入口
from tts_search.data_produce.pipeline import DataProductionConfig, build_sft_dataset
# 选择模块：新旧数据合并与每任务择优
from tts_search.data_produce.selection import (
    merge_new_with_old_data,
    select_top_per_task,
)
# token 过滤模块：分词器加载与消息长度过滤
from tts_search.data_produce.token_filter import (
    TokenFilterConfig,
    count_chat_template_tokens,
    count_message_tokens,
    filter_by_token_length,
    load_tokenizer,
    token_count_within_limit,
)

# 对外导出的公共名称清单（配置类、过滤函数与数据处理工具）
__all__ = [
    "DataProductionConfig",
    "GapFilterConfig",
    "TokenFilterConfig",
    "BaselineTokenGapConfig",
    "BaselineTokenGapDecision",
    "annotate_gap_rows",
    "build_sft_dataset",
    "choose_gap_denominator",
    "choose_legacy_gap_denominator",
    "classify_metric",
    "collect_sft_rows",
    "count_chat_template_tokens",
    "count_message_tokens",
    "enrich_tasks_with_metric_protocol",
    "evaluate_baseline_token_gap",
    "filter_by_relative_gap",
    "filter_by_token_length",
    "load_task_meta_from_parquet",
    "load_tokenizer",
    "merge_new_with_old_data",
    "relative_gap_from_feedback",
    "sample_tasks",
    "select_top_per_task",
    "token_count_within_limit",
    "to_training_frame",
    "write_alignment_dataset",
]
