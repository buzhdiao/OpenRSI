"""Prompt building utilities aligned with prompt_alignment/.

This module builds the *final* (system_prompt, user_prompt) pair for the LLM.
It supports draft / improve(debug) / crossover modes.
"""
# 中文说明:tts_search 的最终 prompt 构建器——把程序库取出的上下文(任务描述、
# 数据描述、父代代码与反馈)拼装成 LLM 实际收到的 (system_prompt, user_prompt) 对。
# 在系统中的位置:生成服务(generation/services)与各搜索策略之间,是"选父代之后、
# 调 LLM 之前"的最后一层;支持 draft / improve / debug / crossover 四种模式,
# 并为"数据集自带完整 prompt"的旧格式提供兼容路径(切分与追加式扩展)。
# 关键设计:system 头部统一是"Kaggle 大师"人设句;所有注入片段先 strip 防空白污染。

# 延迟注解:类型标注运行期不求值
from __future__ import annotations

# 类型注解
from typing import Optional

# 程序对象:父代代码与执行反馈的载体
from tts_search.program_database import Program


def _safe_text(text: str | None) -> str:
    """空值安全 + 去首尾空白(所有注入片段的统一入口)。"""
    return (text or "").strip()


def split_legacy_public_prompts(
    public_system_prompt: str, public_user_prompt: str
) -> tuple[str, str, str, str]:
    """
    Split legacy fully-assembled prompts into the 4 fragments expected by the aligned templates:

    - task_description: between "Task Description:" and "Data Preview:"
    - data_description: between "Data Preview:" and "Output Instructions:"
    - public_user_prompt: starting from "Output Instructions:" (inclusive)
    - public_system_prompt: the legacy system prompt with the first line removed
      (to avoid duplicating the template's first line)

    If the legacy markers are not found, returns empty descriptions and keeps the prompts unchanged.
    """

    # 用户侧旧 prompt 统一去空白
    user_text = (public_user_prompt or "").strip()
    # 三个旧格式分段标记
    task_marker = "Task Description:"
    data_marker = "Data Preview:"
    output_marker = "Output Instructions:"

    # 定位三个标记的先后位置
    task_index = user_text.find(task_marker)
    data_index = user_text.find(data_marker)
    output_index = user_text.find(output_marker)
    if (
        task_index == -1
        or data_index == -1
        or output_index == -1
        or not (task_index < data_index < output_index)
    ):
        # 标记缺失或顺序不对:描述返回空,两段 prompt 原样保留
        return "", "", (public_system_prompt or "").strip(), user_text

    # 按标记切出任务描述 / 数据描述 / 输出指令段
    task_description = user_text[task_index + len(task_marker) : data_index].strip()
    data_description = user_text[data_index + len(data_marker) : output_index].strip()
    extracted_public_user_prompt = user_text[output_index:].strip()

    # 旧 system prompt 去重:若以模板人设句开头则删掉第一行(避免与新模板重复)
    system_text = (public_system_prompt or "").strip()
    legacy_prefix = "You are a Kaggle Grandmaster attending a high-stakes competition."
    if system_text.startswith(legacy_prefix):
        _, _, system_text = system_text.partition("\n")

    return task_description, data_description, system_text, extracted_public_user_prompt


def build_draft_system_prompt(public_system_prompt: str) -> str:
    """draft 模式系统提示:统一人设句 + 数据集自带 system 片段。"""
    return (
        "You are a Kaggle Grandmaster attending a high-stakes competition. "
        "In order to win this competition, you need to come up with an excellent "
        "and creative plan for a solution and then implement this solution in Python.\n"
        f"{_safe_text(public_system_prompt)}"
    )


def build_draft_prompt(task_description: str, data_description: str, public_user_prompt: str) -> str:
    """draft 模式用户提示:任务/数据描述 + 草稿指导 + 输出要求。"""
    return (
        "We will now provide a description of the task.\n\n"
        "Task Description:\n\n"
        f"{_safe_text(task_description)}\n\n"
        "Data Description:\n\n"
        f"{_safe_text(data_description)}\n\n"
        "Draft Guidance:\n\n"
        "Please focus on proposing an advanced idea addressing this task.\n\n"
        f"{_safe_text(public_user_prompt)}"
    )


def build_improve_system_prompt(public_system_prompt: str) -> str:
    """improve 模式系统提示:人设句多一句"要在已有解上改进"。"""
    return (
        "You are a Kaggle Grandmaster attending a high-stakes competition. "
        "In order to win this competition, you need to come up with an excellent "
        "and creative plan for a solution and then implement this solution in Python, "
        "that improves upon an existing solution to the task.\n"
        f"{_safe_text(public_system_prompt)}"
    )

def build_debug_system_prompt(public_system_prompt: str) -> str:
    """debug 模式系统提示:人设句强调"调试并修复已有解"。"""
    return (
        "You are a Kaggle Grandmaster attending a high-stakes competition. "
        "In order to win this competition, you need to come up with an excellent "
        "and creative plan for a solution and then implement this solution in Python, "
        "that debugs and fixes an existing solution to the task.\n"
        f"{_safe_text(public_system_prompt)}"
    )


def build_improve_prompt(
    task_description: str,
    data_description: str,
    parent_program: Program,
    public_user_prompt: str,
) -> str:
    """improve 模式用户提示:任务/数据描述 + 父代代码与反馈 + 改进指导。"""
    # 取父代反馈与代码
    feedback = parent_program.feedback
    parent_code = _safe_text(parent_program.code)

    return (
        "We will now provide a description of the task.\n\n"
        "Task Description:\n\n"
        f"{_safe_text(task_description)}\n\n"
        "Data Description:\n\n"
        f"{_safe_text(data_description)}\n\n"
        "A previous solution was attempted with the following results:\n\n"
        "Code:\n\n"
        f"{parent_code}\n\n"
        "Feedback:\n\n"
        f"{feedback}\n\n"
        "Improve Guidance:\n\n"
        "Based on the previous attempt, its score, and execution output, please analyze what worked well and what didn't, then improve the solution.\n\n"
        f"{_safe_text(public_user_prompt)}"
    )

def build_debug_prompt(
    task_description: str,
    data_description: str,
    parent_program: Program,
    public_user_prompt: str,
) -> str:
    """debug 模式用户提示:与 improve 类似,但指导语聚焦"找根因、先跑通再改进"。"""
    # 取父代反馈与代码
    feedback = parent_program.feedback
    parent_code = _safe_text(parent_program.code)

    return (
        "We will now provide a description of the task.\n\n"
        "Task Description:\n\n"
        f"{_safe_text(task_description)}\n\n"
        "Data Description:\n\n"
        f"{_safe_text(data_description)}\n\n"
        "A previous solution was attempted with the following results:\n\n"
        "Code:\n\n"
        f"{parent_code}\n\n"
        "Feedback:\n\n"
        f"{feedback}\n\n"
        "Debug Guidance:\n\n"
        "Based on the previous attempt, its score, and execution output, please identify the root cause of the failure and fix the code so it runs successfully and follows the output requirements. After it runs, improve the solution if possible.\n\n"
        f"{_safe_text(public_user_prompt)}"
    )


def build_crossover_system_prompt(public_system_prompt: str) -> str:
    """crossover 模式系统提示:强调"综合两个解的优势、产生超越两者的第三代解"。"""
    return (
        "You are a Kaggle Grandmaster attending a high-stakes competition. "
        "Your goal is to synthesize the strengths of two provided solutions, neutralize their respective weaknesses, "
        'and propose novel improvements that neither original solution possessed. A simple "merge" of existing code won\'t suffice to win. '
        "You must identify the hidden synergies between previous attempts and engineering a third-generation solution that transcends its predecessors.\n"
        f"{_safe_text(public_system_prompt)}"
    )


def build_crossover_prompt(
    task_description: str,
    data_description: str,
    parent_program: Program,
    secondary_parent_program: Program,
    public_user_prompt: str,
) -> str:
    """crossover 模式用户提示:两组"代码+反馈"并列呈现,要求合成更优混合解。"""
    # 取两个父代的反馈与代码
    feedback_a = parent_program.feedback
    feedback_b = secondary_parent_program.feedback
    parent_code_a = parent_program.code
    parent_code_b = secondary_parent_program.code

    return (
        "We will now provide a description of the task.\n\n"
        "Task Description:\n\n"
        f"{_safe_text(task_description)}\n\n"
        "Data Description:\n\n"
        f"{_safe_text(data_description)}\n\n"
        "Two previous solutions were attempted with the following results:\n\n"
        "Code 1:\n\n"
        f"{_safe_text(parent_code_a)}\n\n"
        "Feedback 1:\n\n"
        f"{feedback_a}\n\n"
        "Code 2:\n\n"
        f"{_safe_text(parent_code_b)}\n\n"
        "Feedback 2:\n\n"
        f"{feedback_b}\n\n"
        "Crossover Guidance:\n\n"
        "Based on the previous attempts, their scores, and execution outputs, please synthesize the winning components of both solutions into a superior hybrid while introducing novel improvements to address shared weaknesses.\n\n"
        f"{_safe_text(public_user_prompt)}"
    )


def build_prompt(
    mode: str,
    task_description: str,
    data_description: str,
    public_system_prompt: str,
    public_user_prompt: str,
    virtual_data_dir: str,
    parent_program: Optional[Program] = None,
    secondary_parent_program: Optional[Program] = None,
    max_steps: int = 1,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) based on mode.

    Note: `virtual_data_dir` and `max_steps` are kept for compatibility.
    """

    # 兼容占位:两个参数当前不参与拼接
    _ = virtual_data_dir
    _ = max_steps

    # Legacy fallback: some datasets already provide fully assembled prompts in `prompt`.
    # In that case metadata may not contain task/data descriptions; we pass through for draft,
    # and append improve/crossover guidance on top for later steps.
    # 旧格式回退:数据集已给完整 prompt(无任务/数据描述)时,draft 原样透传,
    # improve/debug/crossover 则把"父代代码+反馈+指导"追加到原 prompt 之后。
    if _safe_text(task_description) == "" and _safe_text(data_description) == "":
        system_prompt = _safe_text(public_system_prompt)
        user_prompt = _safe_text(public_user_prompt)
        # draft:完全透传
        if mode == "draft":
            return system_prompt, user_prompt
        # improve/debug:必须有父代
        if mode in ("improve", "debug"):
            if parent_program is None:
                raise ValueError("parent_program is required for improve/debug mode")
            parent_code = _safe_text(parent_program.code)
            feedback = parent_program.feedback
            # 默认改进指导(debug 时换成调试指导)
            guidance_title = "Improve Guidance"
            guidance_text = (
                "Based on the previous attempt, its score, and execution output, please analyze what worked well and what didn't, then improve the solution.\n"
            )
            if mode == "debug":
                guidance_title = "Debug Guidance"
                guidance_text = (
                    "Based on the previous attempt, its score, and execution output, please identify the root cause of the failure and fix the code so it runs successfully and follows the output requirements. After it runs, improve the solution if possible.\n"
                )
            # 追加块:父代代码 + 反馈 + 对应指导
            appended = (
                "A previous solution was attempted with the following results:\n\n"
                "Code:\n\n"
                f"{parent_code}\n\n"
                "Feedback:\n\n"
                f"{feedback}\n\n"
                f"{guidance_title}:\n\n"
                f"{guidance_text}"
            )
            return system_prompt, f"{user_prompt}\n\n{appended}"
        # crossover:必须有两个父代
        if mode == "crossover":
            if parent_program is None or secondary_parent_program is None:
                raise ValueError(
                    "Both parent_program and secondary_parent_program are required for crossover mode"
                )
            parent_code_a = _safe_text(parent_program.code)
            parent_code_b = _safe_text(secondary_parent_program.code)
            feedback_a = parent_program.feedback
            feedback_b = secondary_parent_program.feedback
            # 追加块:两组代码与反馈 + 交叉指导
            appended = (
                "Two previous solutions were attempted with the following results:\n\n"
                "Code 1:\n\n"
                f"{parent_code_a}\n\n"
                "Feedback 1:\n\n"
                f"{feedback_a}\n\n"
                "Code 2:\n\n"
                f"{parent_code_b}\n\n"
                "Feedback 2:\n\n"
                f"{feedback_b}\n\n"
                "Crossover Guidance:\n\n"
                "Based on the previous attempts, their scores, and execution outputs, please synthesize the winning components of both solutions into a superior hybrid while introducing novel improvements to address shared weaknesses.\n"
            )
            return system_prompt, f"{user_prompt}\n\n{appended}"

    # 标准路径:draft 直接组装
    if mode == "draft":
        return (
            build_draft_system_prompt(public_system_prompt),
            build_draft_prompt(task_description, data_description, public_user_prompt),
        )
    # improve/debug:必须有父代,再按模式分流
    if mode in ("improve", "debug"):
        if parent_program is None:
            raise ValueError("parent_program is required for improve/debug mode")
        if mode == "debug":
            return (
                build_debug_system_prompt(public_system_prompt),
                build_debug_prompt(
                    task_description=task_description,
                    data_description=data_description,
                    parent_program=parent_program,
                    public_user_prompt=public_user_prompt,
                ),
            )
        return (
            build_improve_system_prompt(public_system_prompt),
            build_improve_prompt(
                task_description=task_description,
                data_description=data_description,
                parent_program=parent_program,
                public_user_prompt=public_user_prompt,
            ),
        )
    # crossover:双父代合成
    if mode == "crossover":
        if parent_program is None or secondary_parent_program is None:
            raise ValueError(
                "Both parent_program and secondary_parent_program are required for crossover mode"
            )
        return (
            build_crossover_system_prompt(public_system_prompt),
            build_crossover_prompt(
                task_description=task_description,
                data_description=data_description,
                parent_program=parent_program,
                secondary_parent_program=secondary_parent_program,
                public_user_prompt=public_user_prompt,
            ),
        )
    # 未知模式:报错并提示合法取值
    raise ValueError(f"Unknown mode: {mode}. Must be one of draft/improve/debug/crossover")
