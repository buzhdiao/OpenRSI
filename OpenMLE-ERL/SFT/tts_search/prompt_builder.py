"""
Prompt building utilities for draft and improve modes.
"""
# 提示词构建模块：负责 draft（从零草稿）与 improve（改进父程序）两种模式的
# prompt 组装，以及 pass@k 流水线所需的消息重建与旧版公共 prompt 的拆分。
# 导入 os（保留兼容性占位）
import os
# 导入类型注解工具
from typing import Any, Optional, Sequence
# 导入程序数据结构 Program（improve 模式需要父程序信息）
from tts_search.program_database import Program
# 导入日志清洗工具 get_clear_log
from tts_search.reward_func_utils import get_clear_log

# moved to system prompt
# You are a top-ranked Kaggle grandmaster with extensive competition experience.
# Your objective is to solve a Kaggle competition,
# with the goal of maximizing the Position Score (Your rank in the leaderboard) in limited steps.
# You must use Machine Learning/Deep Learning/Computer Vision/NLP/etc. methods to solve the problem,
# the score of random guess or without any ML/DL/CV/NLP methods will be cancelled finally.
# You are likely to train models according to specific competition requirements.
# You have access to a GPU and several CPUs for training DL/ML models.
# Use cuda and PyTorch for faster training if needed.

# moved to user prompt
# You have a total of {max_steps} actions available.
# For a solution and then implement this solution in Python. We will now provide a description of the task.

# Task Description:
# {description}

# Output Instructions:
# The code should be a single-file python program that is self-contained and can be executed as-is.
# No parts of the code should be skipped, don't terminate the script before finishing the code.
# Your response should only contain a single code block.
# All the provided input data is stored in {virtual_public_data_dir} directory.
# **Save test predictions to `submission.csv` in ./ directory as specified in the task description.** This file is critical for evaluation and scoring.
# Your metric score (position score) depends on generating a valid `submission.csv` file at the correct location.
# Your goal is to achieve the highest score, so ensure your code produces this file correctly.


def _safe_text(text: str | None) -> str:
    # 空值安全化：把 None 或空白文本统一规整为去空格后的字符串
    return (text or "").strip()


def split_legacy_public_prompts(
    public_system_prompt: str, public_user_prompt: str
) -> tuple[str, str, str, str]:
    """Split old fully assembled public prompts into task/data/template parts."""

    # 先对旧版 user prompt 做去空白规整
    user_text = _safe_text(public_user_prompt)
    # 定义三段式结构的定位标记：任务描述 / 数据预览 / 输出说明
    task_marker = "Task Description:"
    data_marker = "Data Preview:"
    output_marker = "Output Instructions:"

    # 在文本中定位三个标记的偏移
    task_index = user_text.find(task_marker)
    data_index = user_text.find(data_marker)
    output_index = user_text.find(output_marker)
    # 若任一标记缺失或顺序不满足 task < data < output，则认为不是旧版结构
    if (
        task_index == -1
        or data_index == -1
        or output_index == -1
        or not (task_index < data_index < output_index)
    ):
        # 拆分失败：原样返回系统提示与完整 user 文本
        return "", "", _safe_text(public_system_prompt), user_text

    # 依标记切出任务描述段（task 标记之后到 data 标记之前）
    task_description = user_text[task_index + len(task_marker) : data_index].strip()
    # 切出数据描述段（data 标记之后到 output 标记之前）
    data_description = user_text[data_index + len(data_marker) : output_index].strip()
    # 剩余部分（output 标记起）作为公共 user prompt 模板
    extracted_public_user_prompt = user_text[output_index:].strip()

    # 规整旧版 system prompt
    system_text = _safe_text(public_system_prompt)
    # 旧版前缀句已并入新的 system 模板，需要剥离
    legacy_prefix = "You are a Kaggle Grandmaster attending a high-stakes competition."
    # 若以旧前缀开头，则去掉第一行（即旧前缀所在行）
    if system_text.startswith(legacy_prefix):
        _, _, system_text = system_text.partition("\n")

    # 返回四元组：任务描述、数据描述、清洗后的系统提示、公共 user prompt
    return task_description, data_description, system_text, extracted_public_user_prompt


def _format_time_limit(time_limit_value: Any) -> str:
    """Format a configured runtime limit for prompt text."""
    # 未配置时限：使用通用描述
    if time_limit_value is None:
        return "the allowed time budget"
    # 布尔值不属于有效时限，同样使用通用描述
    if isinstance(time_limit_value, bool):
        return "the allowed time budget"
    # 数值型时限：格式化为 "N seconds"（:g 去除多余小数位）
    if isinstance(time_limit_value, (int, float)):
        return f"{time_limit_value:g} seconds"
    # 其他类型先转字符串并去空白
    value = _safe_text(str(time_limit_value))
    # 空串兜底回通用描述
    return value or "the allowed time budget"


def build_pass_k_draft_system_prompt(public_system_prompt: str) -> str:
    """Build the system prompt used by the pass@k draft pipeline."""
    # 拼接 pass@k draft 的系统提示：固定角色设定 + 任务自带公共系统提示
    return (
        "You are a Kaggle Grandmaster attending a high-stakes competition. "
        "In order to win this competition, you need to come up with an excellent "
        "and creative plan for a solution and then implement this solution in Python.\n"
        f"{_safe_text(public_system_prompt)}"
    ).strip()


def build_pass_k_draft_user_prompt(
    task_description: str,
    data_description: str,
    public_user_prompt: str,
    time_limit: bool = True,
    time_limit_value: Any = None,
) -> str:
    """Build the user prompt used by the pass@k draft pipeline."""
    # 按 time_limit 开关决定是否在提示中强调运行时限
    if time_limit:
        # 先把时限配置格式化为可读文本
        formatted_time_limit = _format_time_limit(time_limit_value)
        # valid: remind model to be mindful of runtime
        # 有效的时限约束：提醒模型把限时当硬性要求，并确保产出 submission.csv
        guidance = (
            "Please focus on proposing an advanced idea addressing this task.\n\n"
            f"ML tasks have a strict execution time limit of {formatted_time_limit} seconds. "
            "Treat finishing within this exact limit as a hard requirement, not a soft preference. "
            "Design and implement a solution that reliably completes before the timeout and still writes `submission.csv` correctly.\n\n"
        )
    else:
        # valid without time limit / no-valid
        # 无时限配置：仅强调提出先进方案
        guidance = "Please focus on proposing an advanced idea addressing this task.\n\n"
    # 组装最终 user prompt：任务描述 + 数据描述 + 指引 + 公共 user 模板
    return (
        "We will now provide a description of the task.\n\n"
        "Task Description:\n\n"
        f"{_safe_text(task_description)}\n\n"
        "Data Description:\n\n"
        f"{_safe_text(data_description)}\n\n"
        "Draft Guidance:\n\n"
        f"{guidance}"
        f"{_safe_text(public_user_prompt)}"
    ).strip()


def _extract_public_prompt_components(
    messages: Sequence[dict[str, Any]] | None,
) -> tuple[str, str]:
    """Extract the first public system/user prompts from stored task messages."""
    # 初始化：公共系统提示与公共用户提示
    public_system_prompt = ""
    public_user_prompt = ""
    # 遍历历史消息，抓取第一条 system 与第一条 user 内容
    for message in messages or []:
        # 角色名规整为小写
        role = str(message.get("role", "")).strip().lower()
        # 内容做空值安全化
        content = _safe_text(str(message.get("content", "")))
        # 第一条 system 消息填入公共系统提示
        if role == "system" and not public_system_prompt:
            public_system_prompt = content
        # 第一条 user 消息填入公共用户提示
        elif role == "user" and not public_user_prompt:
            public_user_prompt = content
        # 两者都拿到即可提前结束
        if public_system_prompt and public_user_prompt:
            break
    # 返回提取到的 (system, user) 公共提示
    return public_system_prompt, public_user_prompt


def build_pass_k_draft_messages(
    metadata: dict[str, Any],
    messages: Sequence[dict[str, Any]] | None,
    time_limit: bool = True,
    time_limit_value: Any = None,
) -> list[dict[str, str]]:
    """
    Rebuild pass@k draft messages from task metadata plus public prompts.

    The parquet rows store task/data descriptions in metadata and public
    system/user instructions in the input prompt. This builder mirrors the
    reference MLE draft prompt structure by combining them into a fresh
    system/user message pair.
    """
    # 从存储的消息中提取公共系统/用户提示
    public_system_prompt, public_user_prompt = _extract_public_prompt_components(
        messages
    )
    # 用 metadata 中的任务/数据描述 + 公共提示重建标准的 system/user 消息对
    return [
        {
            "role": "system",
            "content": build_pass_k_draft_system_prompt(public_system_prompt),
        },
        {
            "role": "user",
            "content": build_pass_k_draft_user_prompt(
                task_description=str(metadata.get("task_description", "")),
                data_description=str(metadata.get("data_description", "")),
                time_limit=time_limit,
                time_limit_value=time_limit_value,
                public_user_prompt=public_user_prompt,
            ),
        },
    ]


def build_draft_prompt(description: str, virtual_data_dir: str, max_steps: int = 1) -> str:
    """
    Build prompt for draft mode (generating code from scratch).

    Args:
        description: Task description text
        virtual_data_dir: Virtual task data path (for example, <TASK_DATA_ROOT>/task_name).
        max_steps: Maximum number of steps (not used in single-turn, kept for compatibility)

    Returns:
        Description text
    """

    # draft 模式单轮对话：任务描述本身即是完整 prompt，直接返回
    return description

# moved to system prompt
# You are a top-ranked Kaggle grandmaster with extensive competition experience.
# Your objective is to solve a Kaggle competition,
# with the goal of maximizing the Position Score (Your rank in the leaderboard) in limited steps.
# You must use Machine Learning/Deep Learning/Computer Vision/NLP/etc. methods to solve the problem,
# the score of random guess or without any ML/DL/CV/NLP methods will be cancelled finally.
# You are likely to train models according to specific competition requirements.
# You have access to a GPU and several CPUs for training DL/ML models.
# Use cuda and PyTorch for faster training if needed.

# moved to user prompt
# You have a total of {max_steps} actions available.
# For a solution and then implement this solution in Python. We will now provide a description of the task.

# Task Description:
# {description}

# Output Instructions:
# The code should be a single-file python program that is self-contained and can be executed as-is.
# No parts of the code should be skipped, don't terminate the script before finishing the code.
# Your response should only contain a single code block.
# All the provided input data is stored in {virtual_public_data_dir} directory.
# **Save test predictions to `submission.csv` in ./ directory as specified in the task description.** This file is critical for evaluation and scoring.
# Your metric score (position score) depends on generating a valid `submission.csv` file at the correct location.
# Your goal is to achieve the highest score, so ensure your code produces this file correctly.

def build_improve_prompt(description: str, virtual_data_dir: str, parent_program: Program, max_steps: int = 1) -> str:
    """
    Build prompt for improve mode (improving existing code).

    Args:
        description: Task description text (already formatted with <|im_start|> and <|im_end|> tags)
        virtual_data_dir: Virtual data directory path
        parent_program: Parent program to improve upon
        payload: Payload from sandbox execution
        max_steps: Maximum number of steps (not used in single-turn, kept for compatibility)

    Returns:
        Complete prompt for improve mode with parent info inserted into user section
    """

    # Format parent program information
    # 取父程序的反馈文本（评估结论）
    feedback = parent_program.feedback
    # 取父程序的代码文本
    parent_code = parent_program.code

    # improve 模式：在任务描述后拼接"上次尝试"信息（代码 + 反馈）与改进指令，
    # 引导模型基于历史结果做针对性改进（对应搜索式智能体循环中的父代继承）
    return f"""
{description}
Previous Attempt:
A previous solution was attempted with the following results:

Code:
```python
{parent_code}
```

{feedback}

Your Task:
Based on the previous attempt, its score, and execution output, please improve the solution.

Analyze what worked well and what didn't. Consider:
1. Were there any errors or issues in the execution?
2. Can the model/approach be improved?
3. Are there better feature engineering techniques?
4. Can hyperparameters be tuned?
5. Should a different model or ensemble be tried?

Please improve your code based on this feedback. Your goal is to achieve a BETTER performance than the previous attempt, so ensure your improvements are effective.
"""


def build_prompt(mode: str, description: str, virtual_data_dir: str,
                parent_program: Optional[Program] = None, max_steps: int = 1) -> str:
    """
    Build prompt based on mode.

    Args:
        mode: Either 'draft' or 'improve'
        description: Task description text
        virtual_data_dir: Virtual data directory path
        parent_program: Parent program (required for improve mode)
        payload: Payload from sandbox execution
        max_steps: Maximum number of steps

    Returns:
        Complete prompt
    """
    # draft 模式：按任务描述直接生成
    if mode == 'draft':
        return build_draft_prompt(description, virtual_data_dir, max_steps)
    # improve 模式：必须有父程序
    elif mode == 'improve':
        if parent_program is None:
            raise ValueError("parent_program is required for improve mode")
        return build_improve_prompt(description, virtual_data_dir, parent_program, max_steps)
    # 未知模式：显式报错，避免静默走错分支
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'draft' or 'improve'")
