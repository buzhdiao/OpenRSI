"""Prompt building utilities for draft/improve/crossover modes."""

# 导入 Optional 类型注解，用于标记可选参数
from typing import Optional
# 导入 Program 数据类：进化搜索数据库中的程序节点（父代/次级父代）
from program_database import Program
# 导入 get_clear_log：清洗沙箱执行日志（去噪、截断）
from reward_func_utils import get_clear_log


def _safe_text(text: Optional[str]) -> str:
    # 工具函数：把 None/空白文本安全转为去首尾空白的字符串，避免拼接 prompt 时出现 "None"
    return (text or "").strip()


def format_sandbox_feedback(status_code: int, payload: dict) -> str:
    """Format sandbox execution results into feedback message"""
    if status_code == 200:
        # 沙箱执行成功：把执行结果 payload 格式化为反馈文本
        result = payload.get("result") or {}
        # 读取执行状态描述（缺省 unknown）
        status = payload.get("status", "unknown")
        # 读取并清洗运行日志
        run_log = get_clear_log(result.get("run_log"))
        # 读取运行结果摘要（缺省空字符串）
        run_result = result.get("result", "")
        # 读取得分（可能为 None 表示无分数）
        score = result.get("score")

        # 反馈文本以执行结果标题开头
        feedback = f"## Execution Result\n"
        # 追加状态行
        feedback += f"**Status**: {status}\n\n"

        if score is not None:
            # 有分数时追加得分行
            feedback += f"**Score**: {score}\n\n"

        if run_result:
            # 有运行结果时追加结果行
            feedback += f"**Result**: {run_result}\n\n"

        if run_log:
            # Truncate log if too long
            # 日志过长时只保留头尾各 1000 字符并标注截断
            if len(run_log) > 2000:
                run_log = run_log[:1000] + "\n... (truncated) ...\n" + run_log[-1000:]
            # 以代码块形式追加执行日志
            feedback += f"**Execution Log**:\n```\n{run_log}\n```\n\n"

    elif status_code == 503:
        # 连接失败：格式化连接错误反馈
        error_type = payload.get("type", "unknown")
        error_detail = payload.get("detail", "Connection failed")
        # 反馈文本以连接错误标题开头
        feedback = f"## Connection Error\n"
        # 追加错误类型行
        feedback += f"**Type**: {error_type}\n"
        # 追加错误详情行
        feedback += f"**Detail**: {error_detail}\n\n"
    else:
        # 其他错误（如超时、异常返回）：格式化通用错误反馈
        error_msg = payload.get("error", "unknown error")
        detail = payload.get("detail", {})
        # 反馈文本以其他错误标题开头
        feedback = f"## Other Error\n"
        # 追加错误信息行
        feedback += f"**Error**: {error_msg}\n"
        if isinstance(detail, dict):
            # detail 为字典时尝试提取其中的状态与消息
            if detail.get("status"):
                # 追加内部状态行
                feedback += f"**Status**: {detail.get('status')}\n"
            if detail.get("message"):
                # 追加内部消息行
                feedback += f"**Message**: {detail.get('message')}\n"

    # 返回拼装好的反馈文本
    return feedback


def build_draft_system_prompt(public_system_prompt: str) -> str:
    # 构造 draft（起草）算子的系统提示：Kaggle 大师人设 + 公共系统提示
    return (
        "You are a Kaggle Grandmaster attending a high-stakes competition. "
        "In order to win this competition, you need to come up with an excellent "
        "and creative plan for a solution and then implement this solution in Python.\n"
        f"{_safe_text(public_system_prompt)}"
    )


def build_draft_prompt(task_description: str, data_description: str, public_user_prompt: str) -> str:
    # 构造 draft（起草）算子的用户提示：任务/数据描述 + 起草指导 + 公共用户提示
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
    # 构造 improve（改进）算子的系统提示：在现有解法基础上改进的人设 + 公共系统提示
    return (
        "You are a Kaggle Grandmaster attending a high-stakes competition. "
        "In order to win this competition, you need to come up with an excellent "
        "and creative plan for a solution and then implement this solution in Python, "
        "that improves upon an existing solution to the task.\n"
        f"{_safe_text(public_system_prompt)}"
    )


def build_improve_prompt(
    task_description: str,
    data_description: str,
    parent_program: Program,
    public_user_prompt: str,
) -> str:
    # 取父代程序的沙箱执行状态码
    status_code = parent_program.status_code
    # 取父代程序的沙箱执行结果 payload
    payload = parent_program.payload
    # 把父代的执行结果格式化为反馈文本（execution-grounded 证据）
    feedback = format_sandbox_feedback(status_code, payload)
    # 取父代代码文本并做安全清洗
    parent_code = _safe_text(parent_program.code)

    # 拼装 improve 提示：任务/数据描述 + 父代代码与执行反馈 + 改进指导
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


def build_crossover_system_prompt(public_system_prompt: str) -> str:
    # 构造 crossover（交叉）算子的系统提示：融合两个解法优势、超越前代的第三代方案
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
    # 取第一个父代程序的沙箱执行状态码
    status_code_a = parent_program.status_code
    # 取第二个父代程序的沙箱执行状态码
    status_code_b = secondary_parent_program.status_code
    # 把第一个父代的执行结果格式化为反馈文本
    feedback_a = format_sandbox_feedback(status_code_a, parent_program.payload)
    # 把第二个父代的执行结果格式化为反馈文本
    feedback_b = format_sandbox_feedback(status_code_b, secondary_parent_program.payload)
    # 取第一个父代的代码文本
    parent_code_a = parent_program.code
    # 取第二个父代的代码文本
    parent_code_b = secondary_parent_program.code

    # 拼装 crossover 提示：任务/数据描述 + 两个父代的代码与反馈 + 交叉指导
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



def _airaevo_memory_text(memory: Optional[str], empty_text: str) -> str:
    # 工具函数：返回清洗后的经验记忆文本；为空时返回给定的占位文案
    memory = _safe_text(memory)
    if memory:
        # 记忆非空时直接返回
        return memory
    # 记忆为空时返回占位文案
    return empty_text


def _airaevo_memory_block(
    memory: Optional[str],
    label: str,
    empty_suffix: str,
    post_instructions: tuple[str, ...] = (),
) -> str:
    # 工具函数：把经验记忆渲染成 markdown 代码块段落，空记忆时输出占位说明
    memory = _safe_text(memory)
    if not memory:
        # 无记忆内容时返回"标签: 占位文案"的形式
        return f"{label}: {empty_suffix}\n"
    # 有记忆内容时渲染为 markdown 代码块
    block = f"{label}:\n```markdown\n{memory}\n```\n"
    if post_instructions:
        # 若提供了附加指令，则逐行拼接并追加到记忆块之后
        block += "\n".join(post_instructions) + "\n"
    # 返回完整的记忆块文本
    return block


def _airaevo_data_overview_block(data_overview: Optional[str]) -> str:
    # 工具函数：把运行时数据概览渲染成提示段落；缺失或占位文本时返回空串
    data_overview = _safe_text(data_overview)
    if not data_overview:
        # 无数据概览时返回空串
        return ""
    # 转小写以便匹配各种占位文案
    lowered = data_overview.lower()
    if lowered in {"data (missing)", "no data preview available", "(no data preview available)"}:
        # 占位文案视同无数据概览
        return ""
    # 渲染为代码块形式的运行时数据概览段落
    return f"\nRuntime Data Overview:\n```\n{data_overview}\n```\n"


def _airaevo_execution_block(execution_timeout: str | int | float | None) -> str:
    # 工具函数：构造"执行预算"提示段落，告知模型沙箱超时约束
    timeout = _safe_text(str(execution_timeout or "")) or "the sandbox timeout"
    if timeout.replace(".", "", 1).isdigit():
        # 纯数字超时值（允许一个小数点）补上秒单位后缀
        timeout = f"{timeout}s"
    # 渲染执行预算段落：强调代码将在沙箱内执行且须在限时内完成
    return (
        "Execution Budget:\n\n"
        f"Be aware of the running time of the code, it should complete within {timeout}.\n"
        f"Your code will be executed in a sandbox after generation.\n"
        f"A single sandbox run must finish within {timeout}.\n"
        "Always choose a validation/training strategy that leaves enough time to write ./submission.csv.\n"
    )


def _airaevo_submission_check() -> str:
    # 工具函数：构造"提交检查"提示段落，要求生成前自检 submission.csv 的格式合规性
    return (
        "Submission Check:\n\n"
        "Before finishing, verify that ./submission.csv exists, has the same row count and required columns/order as sample_submission.csv when available, preserves the sample id/order, and contains no NaN/inf values.\n"
        "When building features, align train/test columns explicitly and handle missing values before model fitting or prediction.\n"
    )


def _airaevo_feedback(program: Program) -> str:
    # 工具函数：把指定程序节点的沙箱执行结果格式化为反馈文本
    return format_sandbox_feedback(program.status_code, program.payload)


def build_airaevo_prompt(
    mode: str,
    task_description: str,
    data_description: str,
    public_system_prompt: str,
    public_user_prompt: str,
    virtual_data_dir: str,
    parent_program: Optional[Program] = None,
    secondary_parent_program: Optional[Program] = None,
    max_steps: int = 1,
    memory: Optional[str] = None,
    data_overview: Optional[str] = None,
    execution_timeout: str | int | float | None = None,
) -> tuple[str, str]:
    """Build prompts aligned with the latest inference AIRA-Evo operator templates."""
    # 显式丢弃仅为兼容保留的虚拟数据目录参数
    _ = virtual_data_dir
    # 显式丢弃仅为兼容保留的最大步数参数
    _ = max_steps
    # 归一化算子模式：转小写，空值按 draft 处理
    mode = str(mode or "draft").lower()
    # 清洗公共系统提示
    public_system_prompt = _safe_text(public_system_prompt)
    # 清洗公共用户提示
    public_user_prompt = _safe_text(public_user_prompt)
    # 清洗任务描述
    task_description = _safe_text(task_description)
    # 清洗数据描述
    data_description = _safe_text(data_description)
    # 预渲染运行时数据概览段落
    data_block = _airaevo_data_overview_block(data_overview)
    # 预渲染执行预算段落
    execution_block = _airaevo_execution_block(execution_timeout)
    # 预渲染提交检查段落
    submission_block = _airaevo_submission_check()

    if mode == "draft":
        # draft（起草）算子：无父代，从零提出新想法
        system_prompt = (
            "You are a Kaggle Grandmaster attending a high-stakes competition. In order to win this competition, you need to come up with an excellent and creative plan for a solution and then implement this solution in Python.\n"
            f"{public_system_prompt}"
        )
        # 渲染 draft 模式的经验记忆块（已探索想法）
        memory_block = _airaevo_memory_block(memory, "Previously explored ideas", "none yet.")
        user_prompt = (
            "We will now provide a description of the task.\n\n"
            "Task Description:\n\n"
            f"{task_description}\n\n"
            "Data Description:\n\n"
            f"{data_description}\n\n"
            "Draft Guidance:\n\n"
            "Please focus on proposing an advanced idea addressing this task.\n"
            "Propose exactly one new solution idea that is different from the previously explored ideas.\n"
            "Keep the evaluation method consistent across iterations.\n\n"
            "AIRA Evo Search Context:\n\n"
            f"{memory_block}"
            f"{data_block}\n"
            f"{execution_block}\n\n{submission_block}\n\n{public_user_prompt}"
        )
        # 返回（系统提示, 用户提示）二元组
        return system_prompt, user_prompt

    if mode == "improve":
        # improve（改进）算子：基于单个父代程序改进
        if parent_program is None:
            # 改进模式必须有父代程序
            raise ValueError("parent_program is required for AIRA improve mode")
        system_prompt = (
            "You are a Kaggle Grandmaster attending a high-stakes competition. In order to win this competition, you need to come up with an excellent and creative plan for a solution and then implement this solution in Python, that improves upon an existing solution to the task.\n"
            f"{public_system_prompt}"
        )
        # 渲染改进模式的目标记忆块，并附加纵向/横向记忆与榜单统计的使用指引
        memory_block = _airaevo_memory_block(
            memory,
            "Targeted memory for this improvement",
            "none available.",
            (
                "Use the selected parent memory to preserve what already works.",
                "Use vertical ancestor memory to understand the recent evolution path and which changes helped or hurt.",
                "Use horizontal sibling memory to avoid repeating nearby failed alternatives and to borrow useful details.",
                "Use board stats only as weak global guidance; prioritize evidence from the selected parent branch.",
            ),
        )
        user_prompt = (
            "We will now provide a description of the task.\n\n"
            "Task Description:\n\n"
            f"{task_description}\n\n"
            "Data Description:\n\n"
            f"{data_description}\n\n"
            "A previous solution was attempted with the following results:\n\n"
            "Code:\n\n"
            f"{_safe_text(parent_program.code)}\n\n"
            "Feedback:\n\n"
            f"{_airaevo_feedback(parent_program)}\n\n"
            "Improve Guidance:\n\n"
            "Based on the previous attempt, its score, and execution output, please analyze what worked well and what didn't, then improve the solution.\n"
            "Propose exactly one improvement idea that is different from the previously explored improvement ideas.\n"
            "Keep the evaluation method consistent across iterations.\n\n"
            "AIRA Evo Search Context:\n\n"
            f"{memory_block}"
            f"{data_block}\n"
            f"{execution_block}\n\n{submission_block}\n\n{public_user_prompt}"
        )
        # 返回（系统提示, 用户提示）二元组
        return system_prompt, user_prompt

    if mode == "debug":
        # debug（调试）算子：修复失败/零分程序
        if parent_program is None:
            # 调试模式必须有父代程序
            raise ValueError("parent_program is required for AIRA debug mode")
        system_prompt = (
            "You are a Kaggle Grandmaster attending a high-stakes competition. In order to win this competition, you need to come up with an excellent and creative plan for a solution and then implement this solution in Python, that debugs and fixes an existing solution to the task.\n"
            f"{public_system_prompt}"
        )
        # 渲染调试模式的目标记忆块，附加同错误记忆优先、最小改动等指引
        memory_block = _airaevo_memory_block(
            memory,
            "Targeted debug memory",
            "none available.",
            (
                "Use same-error memories first to avoid repeating known failure modes.",
                "Use recent related memories to preserve working modeling logic while fixing the bug.",
                "Make the smallest code change that resolves the failure before attempting extra improvements.",
            ),
        )
        user_prompt = (
            "We will now provide a description of the task.\n\n"
            "Task Description:\n\n"
            f"{task_description}\n\n"
            "Data Description:\n\n"
            f"{data_description}\n\n"
            "A previous solution was attempted with the following results:\n\n"
            "Code:\n\n"
            f"{_safe_text(parent_program.code)}\n\n"
            "Feedback:\n\n"
            f"{_airaevo_feedback(parent_program)}\n\n"
            "Debug Guidance:\n\n"
            "Based on the previous attempt, its score, and execution output, please identify the root cause of the failure and fix the code so it runs successfully and follows the output requirements. After it runs, improve the solution if possible.\n"
            "Preserve the current core idea unless the feedback shows that it cannot be made valid.\n\n"
            "AIRA Evo Search Context:\n\n"
            f"{memory_block}"
            f"{data_block}\n"
            f"{execution_block}"
            "If the previous attempt timed out, reduce compute first: fewer epochs/folds, smaller input size, smaller model, cached features, or simpler ensembling.\n\n"
            f"{submission_block}\n\n{public_user_prompt}"
        )
        # 返回（系统提示, 用户提示）二元组
        return system_prompt, user_prompt

    if mode == "crossover":
        # crossover（交叉）算子：融合两个父代程序的优点
        if parent_program is None or secondary_parent_program is None:
            # 交叉模式必须同时提供两个父代程序
            raise ValueError("Both parent_program and secondary_parent_program are required for AIRA crossover mode")
        system_prompt = (
            "You are a Kaggle Grandmaster attending a high-stakes competition. Your goal is to synthesize the strengths of two provided solutions, neutralize their respective weaknesses, and propose novel improvements that neither original solution possessed. A simple \"merge\" of existing code won't suffice to win. You must identify the hidden synergies between previous attempts and engineering a third-generation solution that transcends its predecessors.\n"
            f"{public_system_prompt}"
        )
        # 渲染交叉模式的目标记忆块，附加识别各分支最强可复用组件等指引
        memory_block = _airaevo_memory_block(
            memory,
            "Targeted memory for this crossover",
            "none available.",
            (
                "Use parent memories to identify each branch's strongest reusable components.",
                "Use vertical ancestor memory to understand how each branch evolved.",
                "Use horizontal sibling memory to avoid combining changes that already failed nearby.",
                "Prefer a coherent hybrid with clear synergy over mechanically concatenating both codebases.",
            ),
        )
        user_prompt = (
            "We will now provide a description of the task.\n\n"
            "Task Description:\n\n"
            f"{task_description}\n\n"
            "Data Description:\n\n"
            f"{data_description}\n\n"
            "Two previous solutions were attempted with the following results:\n\n"
            "Code 1:\n\n"
            f"{_safe_text(parent_program.code)}\n\n"
            "Feedback 1:\n\n"
            f"{_airaevo_feedback(parent_program)}\n\n"
            "Code 2:\n\n"
            f"{_safe_text(secondary_parent_program.code)}\n\n"
            "Feedback 2:\n\n"
            f"{_airaevo_feedback(secondary_parent_program)}\n\n"
            "Crossover Guidance:\n\n"
            "Based on the previous attempts, their scores, and execution outputs, please synthesize the winning components of both solutions into a superior hybrid while introducing novel improvements to address shared weaknesses.\n"
            "Propose exactly one crossover plan and keep the evaluation method consistent across iterations.\n\n"
            "AIRA Evo Search Context:\n\n"
            f"{memory_block}"
            f"{data_block}\n"
            f"{execution_block}\n\n{submission_block}\n\n{public_user_prompt}"
        )
        # 返回（系统提示, 用户提示）二元组
        return system_prompt, user_prompt

    # 未知算子模式时报错
    raise ValueError(f"Unknown AIRA mode: {mode}")


def build_airaevo_rich_memory_summary_prompt(
    task_description: str,
    current_program: Program,
    parent_program: Optional[Program],
    current_card: dict,
) -> str:
    # 取父代程序的 AIRA-Evo 经验卡片（无父代或元数据非字典时为空字典）
    parent_card = parent_program.metadata.get("airaevo_experience_card", {}) if parent_program and isinstance(parent_program.metadata, dict) else {}
    # 构造经验总结提示：让模型把当前实验节点蒸馏为可复用记忆（输出严格 JSON）
    return (
        "You are an expert machine learning experiment analyst.\n"
        "Your job is to distill one completed experiment node into reusable memory.\n\n"
        "Return exactly one valid JSON object and nothing else. Use exactly these keys:\n"
        "{\"method_overview\": string, \"parent_comparison_experience\": string}\n\n"
        "# Task Description\n"
        f"{_safe_text(task_description)}\n\n"
        "# Current Node Metadata\n"
        f"node_id: {current_program.id}\n"
        f"operator_that_created_this_node: {current_program.generation_mode}\n"
        f"method_family: {current_card.get('method_family_auto') or 'unknown'}\n"
        f"status: {current_card.get('status') or 'unknown'}\n"
        f"score: {current_card.get('fitness')}\n"
        f"delta_vs_parent: {current_card.get('delta_vs_parent')}\n"
        f"runtime_seconds: {current_card.get('sandbox_time_used')}\n"
        f"error_signature: {current_card.get('error_signature')}\n\n"
        "# Parent Node Metadata\n"
        f"parent_node_id: {parent_program.id if parent_program else 'none'}\n"
        f"parent_method_family: {parent_card.get('method_family_auto') or 'unknown'}\n"
        f"parent_score: {parent_card.get('fitness')}\n\n"
        "# Current Node Plan / Raw Response\n"
        f"{_safe_text(current_program.raw_text)[:5000]}\n\n"
        "# Current Node Code\n"
        "```python\n"
        f"{_safe_text(current_program.code)[:6000]}\n"
        "```\n\n"
        "# Current Node Execution Output\n"
        "```\n"
        f"{format_sandbox_feedback(current_program.status_code, current_program.payload)[:4000]}\n"
        "```\n\n"
        "# Parent Code\n"
        "```python\n"
        f"{_safe_text(parent_program.code)[:4000] if parent_program else ''}\n"
        "```\n\n"
        "# Instructions\n"
        "Write method_overview in 2-5 concise sentences describing the concrete method used by the current node.\n"
        "Write parent_comparison_experience in 2-5 concise sentences comparing current node vs parent and extracting reusable success or failure experience.\n"
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
    """
    Build prompt based on mode.

    Args:
        mode: One of 'draft' | 'improve' | 'debug' | 'crossover'
        task_description: Task description text
        data_description: Data description text
        public_system_prompt: Existing system prompt to append
        public_user_prompt: Existing user prompt to append
        virtual_data_dir: Virtual data directory path (kept for compatibility)
        parent_program: Parent program (required for improve mode)
        max_steps: Maximum number of steps (kept for compatibility)

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # 显式丢弃仅为兼容保留的虚拟数据目录参数
    _ = virtual_data_dir
    # 显式丢弃仅为兼容保留的最大步数参数
    _ = max_steps
    if mode == "draft":
        # draft（起草）算子：使用起草专用的系统/用户提示
        return (
            build_draft_system_prompt(public_system_prompt),
            build_draft_prompt(task_description, data_description, public_user_prompt),
        )
    elif mode in ("improve", "debug"):
        # improve/debug 算子共用改进型提示（均基于单个父代程序）
        if parent_program is None:
            # 改进/调试模式必须有父代程序
            raise ValueError("parent_program is required for improve/debug mode")
        return (
            build_improve_system_prompt(public_system_prompt),
            build_improve_prompt(
                task_description=task_description,
                data_description=data_description,
                parent_program=parent_program,
                public_user_prompt=public_user_prompt,
            ),
        )
    elif mode == "crossover":
        # crossover（交叉）算子：基于两个父代程序融合
        if parent_program is None or secondary_parent_program is None:
            # 交叉模式必须同时提供两个父代程序
            raise ValueError("Both parent_program and secondary_parent_program are required for crossover mode")
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
    else:
        # 未知模式时报错，并列出合法的四算子取值
        raise ValueError(f"Unknown mode: {mode}. Must be one of draft/improve/debug/crossover")
