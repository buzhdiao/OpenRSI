"""Composable SFT data production pipeline."""

# 让类型注解延迟求值，从而支持 PEP 604 联合类型语法（如 ``Path | None``）
from __future__ import annotations

# 标准库导入：不可变数据类装饰器、文件路径对象、任意类型注解
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 数据产出流水线各阶段组件：写出对齐数据集、收集 SFT 行、gap 过滤、top-k 选择、token 长度过滤
from tts_search.data_produce.alignment import write_alignment_dataset
from tts_search.data_produce.collect import collect_sft_rows
from tts_search.data_produce.gap_filter import (
    GapFilterConfig,
    annotate_gap_rows,
    filter_by_relative_gap,
    load_task_meta,
    load_task_meta_from_parquet,
)
from tts_search.data_produce.selection import select_top_per_task
from tts_search.data_produce.token_filter import (
    TokenFilterConfig,
    filter_by_token_length,
)


@dataclass(frozen=True)
class DataProductionConfig:
    """Configuration for the end-to-end SFT data production pipeline.

    Args:
        run_dir: Pass@k evaluation output directory.
        source_parquet: Source prompt parquet for reconstructing conversations.
        task_manifest: Optional legacy CSV with task score-range metadata.
        task_metadata_parquet: Optional parquet whose metadata column is used for
            valid-test gap ranges. Defaults to source_parquet.
        metric_metadata_parquet: Optional metricprompt parquet used to attach
            ``self_valid_protocol`` for metric-aware gap calculation.
        tokenizer_model: Tokenizer path or model id for length filtering.
        output_dir: Directory where selected datasets are written.
        token_limit: Maximum total chat-template tokens.
        relative_gap_limit: Maximum validation/test relative gap.
        top_k_per_task: Number of selected rows per task.
        token_workers: Number of tokenization workers.

    Returns:
        Immutable pipeline configuration.
    """

    # pass@k 评估的输出目录（收集阶段的输入来源）
    run_dir: Path
    # 源任务 prompt parquet，用于重建对话消息
    source_parquet: Path
    # 分词器的本地路径或模型 ID，用于 token 长度过滤
    tokenizer_model: Path | str
    # 选中数据集的写出目录
    output_dir: Path
    # 可选：旧版 CSV 任务清单（含分数区间元数据）；为 None 时改用 parquet 元数据
    task_manifest: Path | None = None
    # 可选：提供 valid-test gap 区间的任务元数据 parquet，缺省回退到 source_parquet
    task_metadata_parquet: Path | None = None
    # 可选：metricprompt parquet，用于附加 self_valid_protocol 以支持指标感知的 gap 计算
    metric_metadata_parquet: Path | None = None
    # 单条样本经 chat 模板后的最大总 token 数（超长样本直接过滤）
    token_limit: int = 32768
    # 验证集/测试集相对 gap 的最大容忍值（防止模型过拟合验证信号而测试掉分）
    relative_gap_limit: float = 0.12
    # 每个任务最终保留的 top-k 条数
    top_k_per_task: int = 4
    # token 计数使用的并行工作进程数
    token_workers: int = 1


def build_sft_dataset(config: DataProductionConfig) -> dict[str, Any]:
    """Run collect -> token filter -> gap filter -> top-k -> write dataset.

    Args:
        config: Data production configuration.

    Returns:
        Summary statistics and output paths for each pipeline stage.
    """

    # 确保输出目录存在（parents=True 允许自动创建多级父目录）
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # 阶段一：从 run_dir 收集 SFT 候选行，只接受已打分且沙箱反馈成功的样本
    collected, collect_stats = collect_sft_rows(
        run_dir=config.run_dir,
        source_parquet=config.source_parquet,
        rejection_policy="accept_scored",
        min_reward=0.0,
        require_feedback_success=True,
    )
    # 阶段二：按 chat 模板 token 长度过滤超长样本（keep_equal=True 表示恰好等于上限也保留）
    token_kept, token_dropped, token_stats = filter_by_token_length(
        collected,
        TokenFilterConfig(
            tokenizer_model=config.tokenizer_model,
            max_total_tokens=config.token_limit,
            workers=config.token_workers,
            keep_equal=True,
        ),
    )
    # 加载任务分数区间元数据：默认从 parquet 元数据列读取；仅当显式给出台账 CSV 时走旧版 CSV 加载
    task_meta = (
        load_task_meta_from_parquet(
            config.task_metadata_parquet or config.source_parquet,
            metric_metadata_parquet=config.metric_metadata_parquet,
        )
        if config.task_manifest is None
        else load_task_meta(config.task_manifest)
    )
    # 阶段三：为每行标注验证/测试分数、相对 gap、gap 分母来源等注解字段
    gap_rows = annotate_gap_rows(
        token_kept,
        task_meta,
        GapFilterConfig(max_relative_gap=config.relative_gap_limit),
    )
    # 按相对 gap 阈值把标注后的行切分为保留/丢弃两组
    gap_kept, gap_dropped, gap_stats = filter_by_relative_gap(
        gap_rows,
        GapFilterConfig(max_relative_gap=config.relative_gap_limit),
    )
    # 阶段四：按任务分组做奖励去重，并保留每任务 top-k 行
    selected, selection_stats = select_top_per_task(
        gap_kept,
        top_k=config.top_k_per_task,
        dedupe_field="reward",
        rejection_policy="accept_scored",
    )
    # 阶段五：把最终选中的样本写成对齐（SFT 训练）数据集
    output_paths = write_alignment_dataset(
        selected,
        output_dir=config.output_dir,
        name="sft_selected",
    )
    # 汇总各阶段统计信息与输出路径，供上层日志/审计使用
    return {
        "collect": collect_stats,
        "token_filter": token_stats,
        "token_dropped": len(token_dropped),
        "gap_filter": gap_stats,
        "gap_dropped": len(gap_dropped),
        "selection": selection_stats,
        "paths": output_paths,
    }
