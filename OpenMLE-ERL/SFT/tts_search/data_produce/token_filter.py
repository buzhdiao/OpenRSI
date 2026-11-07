"""Tokenizer-based length filtering for chat SFT rows."""

# 让类型注解延迟求值，从而支持新版联合类型等注解语法
from __future__ import annotations

# 进程池用于多进程并行计算 token 数
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# tqdm 提供进度条显示
from tqdm import tqdm

# 公共统计工具：汇总均值/分位数等描述统计
from tts_search.data_produce.common import summary_stats

# 工作进程本地的分词器缓存（每个子进程各自初始化一次）
TOKENIZER: Any = None


def _import_auto_tokenizer() -> Any:
    """Import AutoTokenizer in this local env with a known metadata mismatch."""
    # 导入包元数据模块，用于临时改写 distribution 版本查询
    import importlib.metadata as importlib_metadata

    # 保存原始 version 函数，结束后必须还原
    original_version = importlib_metadata.version

    def patched_version(distribution_name: str) -> str:
        # 统一包名风格：小写并把下划线换成连字符，便于匹配
        normalized = distribution_name.lower().replace("_", "-")
        if normalized == "tokenizers":
            # 查询真实版本号
            version = original_version(distribution_name)
            # 本环境 tokenizers 0.22.x 与 transformers 元数据校验不兼容，伪装成 0.21.4
            if version.startswith("0.22."):
                return "0.21.4"
        if normalized == "huggingface-hub":
            # 查询真实版本号
            version = original_version(distribution_name)
            # huggingface-hub 1.x 同样存在元数据不匹配，伪装成 0.36.0
            if version.startswith("1."):
                return "0.36.0"
        # 其余包原样返回真实版本
        return original_version(distribution_name)

    # 在 transformers 导入期间打上版本补丁
    importlib_metadata.version = patched_version
    try:
        from transformers import AutoTokenizer
    finally:
        # 无论导入成败都还原原始 version 函数，避免污染全局元数据
        importlib_metadata.version = original_version
    return AutoTokenizer


@dataclass(frozen=True)
class TokenFilterConfig:
    """Tokenizer length filtering configuration.

    Args:
        tokenizer_model: Local tokenizer path or model id.
        max_total_tokens: Maximum allowed chat-template token count.
        workers: Number of tokenization worker processes.
        chunksize: Multiprocessing chunk size.
        keep_equal: Keep rows exactly equal to the token limit.
        local_files_only: Load tokenizer without network access.

    Returns:
        Immutable token filtering configuration.
    """

    # 分词器本地路径或 HF 模型 ID
    tokenizer_model: str | Path
    # chat 模板后的最大允许总 token 数（SLIME 训练长度上限）
    max_total_tokens: int = 32768
    # 并行分词的工作进程数
    workers: int = 1
    # 进程池 map 的分块大小
    chunksize: int = 32
    # 是否保留恰好等于上限的样本（False 则必须严格小于）
    keep_equal: bool = False
    # 仅从本地加载分词器，禁止联网下载
    local_files_only: bool = True


def _init_worker(tokenizer_model: str, local_files_only: bool) -> None:
    """Initialize a process-local tokenizer.

    Args:
        tokenizer_model: Tokenizer path or model id.
        local_files_only: Whether transformers may access the network.

    Returns:
        None. Stores the tokenizer in the module-level worker cache.
    """
    # 声明写入模块级缓存
    global TOKENIZER
    # 拿到打过版本补丁的 AutoTokenizer 类
    AutoTokenizer = _import_auto_tokenizer()

    # 在当前工作进程中加载分词器一次，之后所有计数复用该实例
    TOKENIZER = AutoTokenizer.from_pretrained(
        tokenizer_model,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )


def load_tokenizer(tokenizer_model: str | Path, local_files_only: bool = True) -> Any:
    """Load the tokenizer used for SLIME message token accounting."""
    # 拿到打过版本补丁的 AutoTokenizer 类
    AutoTokenizer = _import_auto_tokenizer()

    # 在调用方进程里直接加载并返回分词器
    return AutoTokenizer.from_pretrained(
        str(tokenizer_model),
        trust_remote_code=True,
        local_files_only=local_files_only,
    )


def count_chat_template_tokens(messages: list[dict[str, str]], tokenizer: Any) -> int:
    """Count final SLIME chat-template tokens for one message list."""
    # 用训练时一致的 chat 模板把消息列表转成 token 序列（不加生成前缀，与 SLIME 口径一致）
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    # 返回 token 序列长度（显式转 int，防止返回其他数值类型）
    return int(len(ids))


def token_count_within_limit(
    token_count: Any,
    *,
    max_total_tokens: int = 32768,
    keep_equal: bool = True,
) -> bool:
    """Return whether a token count passes the configured SLIME limit."""
    try:
        # 把任意输入转成整数计数
        count = int(token_count)
    except (TypeError, ValueError):
        # 无法解析成数字（如 None 或脏数据）时按不通过处理
        return False
    # keep_equal=True 时允许恰好等于上限，否则要求严格小于
    return count <= max_total_tokens if keep_equal else count < max_total_tokens


def _count_one(item: tuple[int, list[dict[str, str]]]) -> tuple[int, int]:
    """Count tokens for one chat row.

    Args:
        item: Tuple of row index and chat messages.

    Returns:
        Tuple of row index and token count.
    """
    # 解包行下标与消息列表
    idx, messages = item
    # 用进程本地的 TOKENIZER 统计该行 token 数，并带回原下标以便回填
    return idx, count_chat_template_tokens(messages, TOKENIZER)


def count_message_tokens(
    rows: list[dict[str, Any]],
    config: TokenFilterConfig,
) -> list[int]:
    """Return exact chat-template token counts for each row.

    Args:
        rows: SFT rows containing ``messages``.
        config: Tokenizer and length-filter configuration.

    Returns:
        Token counts aligned with input row order.
    """

    # 构造 (行下标, 消息列表) 的输入序列，便于并行后按下标回填
    inputs = [(idx, row["messages"]) for idx, row in enumerate(rows)]
    # 预分配与输入等长的结果数组，保证输出顺序与输入一致
    token_lengths = [0] * len(rows)
    if config.workers <= 1:
        # 单进程模式：直接在当前进程初始化分词器
        _init_worker(str(config.tokenizer_model), config.local_files_only)
        # 用内置 map 逐行惰性计算 token 数
        iterator = map(_count_one, inputs)
        # 进度条逐行消费结果并回填到对应下标
        for idx, length in tqdm(
            iterator,
            total=len(inputs),
            desc="Tokenize messages",
            unit="row",
            dynamic_ncols=True,
        ):
            token_lengths[idx] = length
        return token_lengths

    # 多进程模式：创建进程池，initializer 让每个子进程各自加载一次分词器
    with ProcessPoolExecutor(
        max_workers=config.workers,
        initializer=_init_worker,
        initargs=(str(config.tokenizer_model), config.local_files_only),
    ) as executor:
        # 按块分发任务，减少进程间通信开销
        iterator = executor.map(_count_one, inputs, chunksize=config.chunksize)
        # 同样用进度条消费结果并按下标回填
        for idx, length in tqdm(
            iterator,
            total=len(inputs),
            desc="Tokenize messages",
            unit="row",
            dynamic_ncols=True,
        ):
            token_lengths[idx] = length
    return token_lengths


def filter_by_token_length(
    rows: list[dict[str, Any]],
    config: TokenFilterConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Annotate rows with token length and split into kept/dropped rows.

    Args:
        rows: Candidate SFT rows.
        config: Tokenizer and length-filter configuration.

    Returns:
        Kept rows, dropped rows, and summary statistics.
    """

    # 先批量计算每行精确的 chat 模板 token 数
    token_lengths = count_message_tokens(rows, config)
    # 保留与丢弃两个桶
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    # strict=True 保证行数与计数一一对应，长度不一致时立即报错
    for row, length in zip(rows, token_lengths, strict=True):
        # 拷贝原行，避免修改调用方数据
        annotated = dict(row)
        # 写入该行 token 长度注解
        annotated["token_length"] = int(length)
        # 判断是否超出配置的长度上限
        keep = token_count_within_limit(
            length,
            max_total_tokens=config.max_total_tokens,
            keep_equal=config.keep_equal,
        )
        # 记录过滤决策标记
        annotated["kept_token_filter"] = bool(keep)
        if keep:
            # 长度合规：进入保留桶
            kept.append(annotated)
        else:
            # 超长：进入丢弃桶
            dropped.append(annotated)

    # 汇总过滤前后行数及 token 长度的描述统计
    stats = {
        "tokenizer_model": str(config.tokenizer_model),
        "max_total_tokens": config.max_total_tokens,
        "keep_equal": config.keep_equal,
        "before": len(rows),
        "after": len(kept),
        "dropped": len(dropped),
        "token_length": summary_stats(token_lengths),
    }
    return kept, dropped, stats
