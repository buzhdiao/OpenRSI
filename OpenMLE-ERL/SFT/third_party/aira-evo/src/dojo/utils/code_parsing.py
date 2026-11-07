# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are MIT-licensed
# Copyright (c) 2024 Weco AI Ltd
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/WecoAI/aideml/blob/main/LICENSE

# 导入正则模块，用于从文本中提取代码块与 JSON
import re

# 导入 black 代码格式化库，用于规范化提取出的 Python 代码
import black
# 导入日志模块
import logging
# 导入 json 模块，用于解析模型输出的 JSON
import json

# 创建本模块的日志器
log = logging.getLogger(__name__)


def is_valid_python_script(script: str) -> bool:
    """
    Check if the provided script is syntactically valid Python code.

    Args:
        script (str): The Python script to validate.

    Returns:
        bool: True if the script compiles without syntax errors, False otherwise.
    """
    # 尝试把脚本编译为字节码，以验证语法是否合法
    try:
        compile(script, "<string>", "exec")
        # 编译通过，说明是合法的 Python 语法
        return True
    except SyntaxError:
        # 出现语法错误则判定为非法脚本
        return False


def format_code(code: str) -> str:
    """
    Format Python code using Black.

    Args:
        code (str): The Python code to format.

    Returns:
        str: The formatted code if successful; otherwise, the original code.
    """
    # 用 black 的默认风格格式化代码字符串
    try:
        return black.format_str(code, mode=black.FileMode())
    except black.parsing.InvalidInput:  # type: ignore
        # 无法解析（如不是 Python 代码）时原样返回
        return code


def extract_code(text: str) -> str:
    """
    Extract Python code blocks from the given text.

    Searches for code enclosed in triple backticks (optionally with "python")
    and, if none are found, treats the entire text as code.

    Args:
        text (str): The text containing potential Python code blocks.

    Returns:
        str: A single string with all valid, formatted Python code blocks.
    """
    # 收集解析出的代码块列表
    parsed_codes = []
    # 先匹配标准 markdown 代码围栏 ```python ... ```（python 标记可省略），DOTALL 允许跨行
    matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL)
    # 逐个取出围栏内的代码内容
    for match in matches:
        code_block = match[1]
        # 追加到代码块列表
        parsed_codes.append(code_block)

    # 没有标准围栏时，把整段文本当作代码处理（围栏可选）
    if not parsed_codes:
        matches = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            code_block = matches[0][2]
            # 取出第三分组（真正的代码内容）
            parsed_codes.append(code_block)

    # 只保留语法合法的代码块，并统一用 black 格式化
    valid_code_blocks = [format_code(c) for c in parsed_codes if is_valid_python_script(c)]
    # 把多个合法代码块用空行拼接后整体再格式化一次
    formatted_code = format_code("\n\n".join(valid_code_blocks))
    # 拼接结果非空且整体语法合法时，返回最终代码
    if bool(formatted_code) and is_valid_python_script(formatted_code):
        return formatted_code

    # 所有提取尝试都失败时抛出异常，表示解法不是合法 Python 代码
    raise Exception("Solution is not valid python code.")


def parse_json_output(response_text):
    """
    Attempts to extract and parse JSON from a string that might be wrapped in markdown/code blocks,
    contain a leading "json" prefix, or include trailing commas.
    """
    # 内部函数：删除对象/数组闭合符号前的尾随逗号（LLM 常见输出瑕疵）
    def _remove_trailing_commas(text: str) -> str:
        # 匹配逗号后跟空白与 } 或 ] 的位置，只去掉逗号本身
        return re.sub(r",\s*([}\]])", r"\1", text)

    # 内部函数：去掉推理模型输出中的 <think>...</think> 思考段
    def _strip_think_blocks(text: str) -> str:
        # 非贪婪匹配整段 think 块并删除，忽略大小写、允许跨行
        return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()

    # 内部函数：扫描文本，返回从最早的 "{" 起能完整解析出的第一个 JSON 对象
    def _scan_first_json_object(text: str):
        # 创建支持增量解码的 JSON 解码器
        decoder = json.JSONDecoder()
        # 从左到右逐字符扫描
        for index, char in enumerate(text):
            # 只在遇到 "{" 时尝试解码
            if char != "{":
                continue
            try:
                # 从当前位置尝试解码出完整的 JSON 值
                value, _ = decoder.raw_decode(text[index:])
                # 解码结果为字典时视为目标 JSON 对象
                if isinstance(value, dict):
                    return value
            except json.decoder.JSONDecodeError:
                # 从该 "{" 无法完整解码则继续向后扫描
                continue
        # 全文扫描不到合法 JSON 对象时返回 None
        return None

    # 字符串输入：按多级策略提取并解析 JSON
    if isinstance(response_text, str):
        original_text = response_text  # Keep the original for logging.
        # 先剥离思考块并去除首尾空白
        response_text = _strip_think_blocks(response_text).strip()

        # Try simply load it
        # 第一级：直接尝试整体解析
        try:
            return json.loads(response_text)
        except json.decoder.JSONDecodeError as e:
            # 直接解析失败则记录错误与原始响应
            log.info("JSON decode error: " + str(e) + " | Original response: " + original_text)

        # 1. Extract JSON from markdown/code block wrappers (```json ... ```).
        # 第二级：尝试从 markdown 代码围栏中取出 JSON 内容
        code_block_pattern = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```", re.IGNORECASE)
        match = code_block_pattern.search(response_text)
        if match:
            # 命中围栏则用围栏内文本替换原文本
            response_text = match.group(1)

        # 2. Remove any leading "json" prefix (case-insensitive).
        # 第三级：去掉开头可能存在的 "json" 前缀（不区分大小写）
        response_text = re.sub(r"^json\s*", "", response_text, flags=re.IGNORECASE)

        # 3. Remove trailing commas before closing objects or arrays.
        # 第四级：清除闭合符号前的尾随逗号
        response_text = _remove_trailing_commas(response_text)

        try:
            return json.loads(response_text)
        except json.decoder.JSONDecodeError as e:
            # 各级清洗后仍失败：记录错误，并对原始文本做兜底扫描
            log.info("JSON decode error: " + str(e) + " | Original response: " + original_text)
            # 对原始文本（去思考块、去尾逗号）扫描第一个可解析的 JSON 对象
            scanned = _scan_first_json_object(_remove_trailing_commas(_strip_think_blocks(original_text)))
            # 扫描成功则返回该对象
            if scanned is not None:
                return scanned
            # 彻底失败时返回空字典，保证调用方拿到可用的映射
            return {}
    else:
        # 非字符串输入（如已是 dict）：直接尝试转换为字典
        try:
            return dict(response_text)
        except Exception as e:
            # 转换失败则记录错误并返回空字典
            log.info("Error converting non-string response to dict: " + str(e))
            return {}
