# 引入正则模块，用于答案字母抽取与文本规范化
import re
# 引入字符串常量模块，获取选项字母表
import string
# 引入 Iterable 类型标注
from collections.abc import Iterable

# 默认合法选项字母：A-H 共 8 个大写字母
DEFAULT_VALID_LETTERS = list(string.ascii_uppercase[:8])


# 去除思维链前缀：只保留 </think> 之后的正式回答部分
def _strip_chain_of_thought(text: str) -> str:
    # 空文本直接返回空串
    if not text:
        return ""

    # 含 </think> 标记时取最后一次出现之后的文本（CoT 之后才是最终答案）
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1]

    # 无思维链标记则原文返回
    return text


# 文本规范化：小写化并把所有非字母数字字符替换为空格、去首尾空白
def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


# 从模型响应中尽力抽取所选选项字母（论文 GPQA 多选题评测的答案解析）
def _extract_letter_from_response(response: str, valid_letters: Iterable[str]) -> str | None:
    """
    Best-effort extraction of the selected option letter from the model response.
    """
    # 空响应无法抽取
    if not response:
        return None

    # 先剥离思维链，只看最终回答
    text = _strip_chain_of_thought(response)
    # 按优先级排列的抽取正则：依次匹配"answer/option/choice is X"、"X is correct"、"final answer is X"
    patterns = [
        r"(?:answer|option|choice)\s*(?:is|:)?\s*([A-Z])",
        r"([A-Z])\s*(?:is\s*(?:the)?\s*correct)",
        r"final\s*(?:answer|option)\s*(?:is|:)?\s*([A-Z])",
    ]

    # 把合法字母集合统一转成大写集合
    valid_letters = {letter.upper() for letter in valid_letters}
    # 按优先级逐个模式尝试匹配
    for pattern in patterns:
        # 忽略大小写地在文本中搜索该模式
        match = re.search(pattern, text, flags=re.IGNORECASE)
        # 命中则取出捕获的字母
        if match:
            # 统一转大写
            letter = match.group(1).upper()
            # 字母在合法选项范围内才算有效抽取
            if letter in valid_letters:
                return letter

    # Fallback: last standalone capital letter that is valid.
    # 兜底策略：找出文本中所有独立的大写字母
    candidates = re.findall(r"\b([A-Z])\b", text)
    # 从后往前取（越靠后越接近最终结论）
    for letter in reversed(candidates):
        # 统一转大写
        letter = letter.upper()
        # 第一个落在合法范围内的即作为答案
        if letter in valid_letters:
            return letter

    # 所有策略都失败：返回 None
    return None


# GPQA 规则奖励：比对抽取出的选项字母与正确选项，答对给 1.0 否则 0.0
def compute_gpqa_reward(response: str, label, metadata: dict | None = None) -> float:
    """Rule-based scorer for GPQA-style multiple-choice evaluation."""
    # 响应为 None 直接判 0 分
    if response is None:
        return 0.0

    # 元数据为空时退化为空字典
    metadata = metadata or {}

    # 从元数据取选项列表
    choices = metadata.get("choices")
    # 选项是 dict 时取其值列表
    if isinstance(choices, dict):
        choices = list(choices.values())
    # 其他非空形式统一转成列表
    elif choices is not None:
        choices = list(choices)

    # 从元数据取合法选项字母
    valid_letters = metadata.get("valid_letters")
    # 已显式给出合法字母：统一转成大写字符串列表
    if valid_letters:
        valid_letters = [str(letter).upper() for letter in valid_letters]
    # 未给出但有选项数：按选项数量取字母表前缀（A、B、C…）
    elif choices:
        valid_letters = list(string.ascii_uppercase[: len(choices)])
    # 都没有：退回默认的 A-H
    else:
        valid_letters = DEFAULT_VALID_LETTERS

    # 从元数据取正确选项字母
    correct_letter = metadata.get("correct_letter")
    # 是字符串时规范化（去空白并大写）
    if isinstance(correct_letter, str):
        correct_letter = correct_letter.strip().upper()
    # 否则置为 None（后续从 label 推断）
    else:
        correct_letter = None

    # 标签文本暂为 None
    label_text = None
    # 标签是字符串的情形
    if isinstance(label, str):
        # 去除首尾空白后作为标签文本
        label_text = label.strip()
        # 标签本身是单个合法字母且尚无正确字母：直接把标签当作正确选项字母
        if len(label_text) == 1 and label_text.upper() in valid_letters and not correct_letter:
            correct_letter = label_text.upper()
    # 标签是数值的情形：解释为正确选项的下标
    elif isinstance(label, (int, float)):
        # 转成整数下标
        idx = int(label)
        # 下标在合法范围内时映射为对应字母
        if 0 <= idx < len(valid_letters):
            correct_letter = valid_letters[idx]

    # 仍无正确字母且有选项与文本标签：用规范化后的文本匹配选项内容
    if not correct_letter and choices and label_text:
        # 规范化标签文本
        normalized_label = _normalize_text(label_text)
        # 遍历各选项寻找文本一致的项
        for idx, choice in enumerate(choices):
            # 规范化后完全相等则命中
            if _normalize_text(str(choice)) == normalized_label:
                # 命中项对应的字母即正确字母
                correct_letter = valid_letters[idx]
                # 把匹配到的选项文本回写进元数据（缺省才写入，避免覆盖）
                metadata.setdefault("correct_answer", choice)
                # 命中后退出循环
                break

    # 从模型响应中抽取所选选项字母
    extracted_letter = _extract_letter_from_response(response, valid_letters)
    # 双方字母都齐全：直接比较，相等给 1.0
    if extracted_letter and correct_letter:
        return 1.0 if extracted_letter == correct_letter else 0.0

    # 字母比对不可行时退化为文本匹配：先收集正确答案的候选文本
    candidate_answers = []
    # 已知正确字母且有选项列表：取该字母对应的选项文本
    if correct_letter and choices:
        # 查找正确字母在合法字母表中的下标
        try:
            idx = valid_letters.index(correct_letter)
        # 字母不在字母表中则下标置空
        except ValueError:
            idx = None
        # 下标有效且在选项范围内：加入候选
        if idx is not None and idx < len(choices):
            candidate_answers.append(str(choices[idx]))

    # 元数据中的正确答案/答案文本字段也加入候选
    for key in ("correct_answer", "answer_text"):
        # 取字段值
        value = metadata.get(key)
        # 非空才纳入候选
        if value:
            candidate_answers.append(str(value))

    # 文本标签本身也作为候选答案
    if label_text:
        candidate_answers.append(label_text)

    # 把所有候选答案规范化后去重成集合
    normalized_targets = {_normalize_text(text) for text in candidate_answers if text}
    # 规范化剥离思维链后的模型响应
    normalized_response = _normalize_text(_strip_chain_of_thought(response))
    # 任一候选答案作为子串出现在响应中即判对
    for target in normalized_targets:
        # 非空且被响应包含：给 1.0
        if target and target in normalized_response:
            return 1.0

    # 最后兜底：抽出了字母、无正确字母但有文本标签时，与标签首字母比较
    if extracted_letter and not correct_letter and label_text:
        return 1.0 if extracted_letter == label_text.strip().upper() else 0.0

    # 所有判定路径都未命中：0 分
    return 0.0
