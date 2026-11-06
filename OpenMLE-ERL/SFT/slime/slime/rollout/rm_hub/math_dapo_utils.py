# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

# DAPO 数学规则评分工具：\\boxed 答案抽取、答案规范化与判分（改写自 lm-evaluation-harness）

# 引入正则模块，用于 LaTeX 答案抽取与规范化
import re
# 引入 signal，用于实现判分超时保护
import signal


# 抽取字符串中最后一个 \\boxed{...} 表达式（含外层命令）
def last_boxed_only_string(string: str) -> str | None:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    # 从右往左找 "\\boxed{" 首次出现位置（即最后一个 boxed）
    idx = string.rfind("\\boxed{")
    # 找不到任何 boxed：返回 None
    if idx < 0:
        return None

    # 从 boxed 起始位置开始扫描的游标
    i = idx
    # 匹配的右花括号位置（未找到时保持 None）
    right_brace_idx = None
    # 当前尚未闭合的左花括号计数（起始 "{" 记 1 层）
    num_left_braces_open = 0

    # 向右逐字符扫描，寻找与起始 "{" 配对的右括号
    while i < len(string):
        # 遇到左花括号：层数加一
        if string[i] == "{":
            num_left_braces_open += 1
        # 遇到右花括号：层数减一
        if string[i] == "}":
            num_left_braces_open -= 1
            # 层数归零说明找到了配对的右括号
            if num_left_braces_open == 0:
                # 记录右括号位置
                right_brace_idx = i
                # 找到即退出扫描
                break
        # 游标右移一位
        i += 1

    # 找到配对右括号则返回完整 boxed 表达式，否则 None
    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


# 去掉 "\\boxed{}" 外壳，返回其内部内容
def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    # boxed 前缀常量
    left = "\\boxed{"
    # 校验输入必须以 "\\boxed{" 开头
    assert s[: len(left)] == left, f"box error: {s}"
    # 校验输入必须以 "}" 结尾
    assert s[-1] == "}", f"box error: {s}"
    # 去掉前缀与末尾右括号，返回内部内容
    return s[len(left) : -1]


# 基于 SIGALRM 的超时上下文管理器（Linux 下保护判分不被死循环卡住）
class timeout:

    # 初始化：记录超时秒数与超时报错信息
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    # 超时信号处理函数：抛出 TimeoutError
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    # 进入上下文：注册 SIGALRM 处理器并启动定时闹钟
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    # 退出上下文：取消未触发的闹钟
    def __exit__(self, type, value, traceback):
        signal.alarm(0)


# Constants for normalization
# 答案规范化时的替换规则表：(原串, 替换为)
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

# 答案规范化时需要整体删除的干扰词/单位/LaTeX 片段
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
    "<|im_end|>",
    "<|endoftext|>",
]


# 规范化定量推理题的最终答案（Minerva 风格），便于字符串级比对
def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    # 统一转成字符串
    final_answer = str(final_answer)
    # 若含等号，只取等号右侧部分作为答案
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    # 依次应用替换规则表
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    # 依次删除干扰表达式
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    # 若含 $...$ 数学段：仅保留两个 $ 之间的内容
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    # 剥掉 \\text{...} 外壳只留内容
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    # 剥掉 \\textbf{...} 外壳只留内容
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    # 剥掉 \\overline{...} 外壳只留内容
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    # 剥掉 \\boxed{...} 外壳只留内容
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    # 把简写 \\fracab 补全为标准形式 \\frac{a}{b}
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    # 把简写 \\sqrta 补全为标准形式 \\sqrt{a}
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    # 最后移除所有 $ 符号
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    # 去掉千分位逗号后是纯数字时：删除所有逗号（如 1,234 -> 1234）
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    # 去首尾空白后返回
    return final_answer.strip()


# Minerva 标准判分：按 "Answer: X" 模式抽取预测并与规范化后的标签比对
def is_correct_minerva(
    solution_str: str, gt: str, gt_need_extract: bool = False, answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)"
) -> tuple[bool, str]:
    """Check if the solution is correct according to Minerva criteria.

    Args:
        solution_str: The solution string to check
        gt: The ground truth answer
        gt_need_extract: Whether the ground truth needs extraction
        answer_pattern: Regex pattern to extract the answer

    Returns:
        Tuple of (is_correct, normalized_prediction)
    """
    # Extract answer from solution
    # 在解答文本中找出所有匹配 "Answer: ..." 的片段
    match = re.findall(answer_pattern, solution_str)
    # 取最后一次匹配作为预测答案（无匹配则标记为无效）
    extracted_answer = match[-1] if match else "[INVALID]"
    # 对预测答案做规范化
    pred = normalize_final_answer(extracted_answer)

    # Process ground truth
    # 标签也需要抽取时：先取其 boxed 内容再去外壳
    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    # 否则直接规范化标签
    else:
        gt = normalize_final_answer(gt)

    gt = str(int(float(gt)))  # in dapo, all answers are integers

    # 返回（是否正确, 规范化后的预测）
    return (pred == gt), pred


# 严格 boxed 判分：只看末尾（或 pause 段之后）的 \\boxed 答案是否与标签完全一致
def is_correct_strict_box(pred: str, gt: str, pause_tokens_index: list[int] | None = None) -> tuple[int, str | None]:
    """Check if the prediction is correct using strict boxed answer criteria.

    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract the relevant part of the prediction
    # 给定 pause token 下标时：从最后一个 pause 前约 100 字符处开始截取
    if pause_tokens_index is not None:
        # 校验 pause 下标数量必须为 4
        assert len(pause_tokens_index) == 4
        # 截取 pause 之后的相关片段
        pred = pred[pause_tokens_index[-1] - 100 :]
    # 常规情形：只看最后 100 个字符
    else:
        pred = pred[-100:]

    # Extract and check the boxed answer
    # 从截取片段中抽取最后一个 boxed 表达式
    boxed_pred = last_boxed_only_string(pred)
    # 去外壳得到内部答案（无 boxed 时为 None）
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    # 严格相等判分：正确记 1 否则 -1，并返回抽取出的预测
    return 1 if (extracted_pred == gt) else -1, extracted_pred


# 统一校验入口：根据配置选择严格 boxed 或 Minerva 判分
def verify(
    solution_str: str, answer: str, strict_box_verify: bool = False, pause_tokens_index: list[int] | None = None
) -> bool:
    """Verify if the solution is correct.

    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        True if the solution is correct, False otherwise
    """
    # 严格 boxed 模式：调用严格判分器
    if strict_box_verify:
        correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
        # 得分为 1 视为正确
        return correct == 1, pred

    # 默认 Minerva 模式判分
    correct, pred = is_correct_minerva(solution_str, answer)
    # 返回判分结果与预测
    return correct, pred


# 计算单条解答的奖励分：正确 1.0 / 错误 -1.0，并附准确率与预测
def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: list[int] | None = None,
) -> float:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        config: Configuration object containing reward model settings
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """
    # Limit solution length for efficiency
    # 只保留解答末尾 300 字符以加速判分（MATH-500 最长答案仅 159 字符）
    solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters

    # Verify the solution
    # 调用统一校验入口判分
    correct, pred = verify(solution_str, ground_truth, strict_box_verify, pause_tokens_index)

    # 按判分结果映射奖励：正确 1.0 错误 -1.0
    reward = 1.0 if correct else -1.0
    # 准确率标记即判分布尔值
    acc = correct

    # 返回分数、准确率与预测组成的字典
    return {
        "score": reward,
        "acc": acc,
        "pred": pred,
    }
