# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 fcntl，用于文件锁（多进程并发写日志文件时互斥）
import fcntl
# 导入 inspect，用于解析函数签名
import inspect
# 导入 math，作为动态执行启发式函数时的依赖注入
import math
# 导入 multiprocessing，供动态执行的函数使用进程池
import multiprocessing
# 导入 random，供动态执行的函数使用随机数
import random
# 导入 re，用于正则提取代码块
import re
# 导入 Counter 计数工具
from collections import Counter
# 导入进程池与线程池执行器，供动态执行的函数使用
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
# 导入浅拷贝与深拷贝工具，供动态执行的函数使用
from copy import copy, deepcopy
# 导入 partial，供动态执行的函数固定部分参数
from functools import partial
# 导入类型注解工具
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 导入 black 代码格式化库
import black
# 导入 numpy，供动态执行的函数使用
import numpy


def is_valid_python_script(script):
    """
    Check if a given script is a valid Python script.

    This function attempts to compile the provided script. If the compilation
    is successful, it returns True, indicating that the script is syntactically
    correct Python code. If a SyntaxError is raised during compilation, it
    returns False, indicating invalid Python syntax.

    Args:
        script (str): The Python script to validate.

    Returns:
        bool: True if the script is valid Python code, False otherwise.
    """
    try:
        # Attempt to compile the script in 'exec' mode
        # 用内置 compile 做纯语法检查（exec 模式但不执行），通过即合法
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        # If a SyntaxError occurs, the script is invalid
        # 语法错误则判定为非法脚本
        return False


def format_code(code) -> str:
    """
    Format Python code using the Black code formatter.

    This function uses Black to format the provided Python code string. If the
    code is invalid and cannot be parsed by Black, it returns the original
    unformatted code.

    Args:
        code (str): The Python code to format.

    Returns:
        str: The formatted Python code if successful; otherwise, the original code.
    """
    try:
        # Format the code using Black with default settings
        # 用 black 按默认风格格式化代码
        return black.format_str(code, mode=black.FileMode())
    except black.parsing.InvalidInput:  # type: ignore
        # If Black cannot parse the code, return it unchanged
        # black 无法解析时原样返回，不让格式化失败中断流程
        return code


def extract_code(text):
    """
    Extract Python code blocks from a given text.

    This function searches the input text for Python code blocks enclosed within
    triple backticks (```python ... ```). It extracts these code blocks, formats
    them using Black, and ensures they are valid Python scripts. If no such
    blocks are found, it attempts to extract the entire text as a code block.

    Args:
        text (str): The input text containing potential Python code blocks.

    Returns:
        str: A single string containing all valid, formatted Python code blocks
             extracted from the input text, separated by two newlines.
    """
    # 收集提取到的代码块
    parsed_codes = []

    # Regex pattern to find code blocks enclosed in ```python ... ```
    # The (python)? makes the 'python' specifier optional
    # 匹配 ```...``` 或 ```python ...``` 代码围栏（DOTALL 跨行匹配）
    matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        # 取围栏内的代码主体
        code_block = match[1]
        parsed_codes.append(code_block)

    # If no code blocks were found with backticks, try extracting the entire text
    if len(parsed_codes) == 0:
        # Regex pattern to match the entire text as a code block, with optional backticks
        # 没有围栏时把整段文本当作（可能缺围栏的）代码处理
        matches = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            # 取第 3 个捕获组（代码主体）
            code_block = matches[0][2]
            parsed_codes.append(code_block)

    # Validate each extracted code block and format it
    # 逐块语法校验，仅保留合法块并用 black 统一格式化
    valid_code_blocks = [format_code(c) for c in parsed_codes if is_valid_python_script(c)]

    # Combine all valid code blocks into a single string separated by two newlines
    # 把所有合法代码块用两个换行拼接后再整体格式化，返回最终代码文本
    return format_code("\n\n".join(valid_code_blocks))


def opt_messages_to_list(
    system_message: str | None, user_message: str | None, content_key: str
) -> list[dict[str, str]]:
    """Helper function to turn optional system/user messages into a list."""
    # 消息列表初始化为空
    messages = []
    if system_message:
        # 系统消息非空时构造 role=system 的消息
        messages.append({"role": "system", content_key: str(system_message)})
    if user_message:
        # 用户消息非空时构造 role=user 的消息
        messages.append({"role": "user", content_key: str(user_message)})
    # 返回消息列表
    return messages


def define_function_from_string(
    function_string: str,
) -> Tuple[Optional[Callable], List[str]]:
    """
    Takes a string containing a function definition and returns the defined function.

    Args:
    - function_string (str): The string containing the function definition.

    Returns:
    - function: The defined function.
    """
    # 动态执行所用的命名空间
    namespace = {}
    # 预注入常用依赖（数学库、numpy、类型工具、随机、拷贝、并发池等），
    # 让 LLM 生成的启发式函数字符串无需额外 import 即可运行
    additional_globals = {
        "math": math,
        "numpy": numpy,
        "Tuple": Tuple,
        "List": List,
        "Callable": Callable,
        "Any": Any,
        "Union": Union,
        "Optional": Optional,
        "Dict": Dict,
        "random": random,
        "partial": partial,
        "copy": copy,
        "deepcopy": deepcopy,
        "multiprocessing": multiprocessing,
        "ProcessPoolExecutor": ProcessPoolExecutor,
        "ThreadPoolExecutor": ThreadPoolExecutor,
    }
    # 把预注入依赖并入命名空间
    namespace.update(additional_globals)
    # 在受限命名空间中动态执行函数定义字符串
    exec(function_string, namespace)
    # 从命名空间中取出名为 heuristic_fn 的函数（未找到则为 None）
    function = next((value for key, value in namespace.items() if key == "heuristic_fn"), None)
    # 函数存在时用 inspect 解析其参数名列表，否则给空列表
    args = inspect.getfullargspec(function).args if function else []
    # 返回（函数, 参数名列表）
    return function, args


class DataLogger:
    # 数据记录器：以追加方式把样本写入文件，并用文件锁保证多进程写入安全

    def __init__(self, log_file_path: str):
        # 保存日志文件路径
        self.log_file_path = log_file_path

    def log(self, data_sample):
        # 追加模式打开日志文件
        with open(self.log_file_path, "a", encoding="utf-8") as file:
            # 加排他锁，避免多进程并发写坏同一行
            fcntl.flock(file, fcntl.LOCK_EX)
            # 把样本转成字符串写入并换行
            file.write(str(data_sample) + "\n")
            # 释放文件锁
            fcntl.flock(file, fcntl.LOCK_UN)


def stable_softmax(x, axis=-1, temp=1.0):
    # 数值稳定的 softmax（softmax，归一化指数函数）：先减最大值防止 exp 溢出
    # 先按温度缩放 logits（temp 越小分布越尖）
    x = x / temp
    # 减去沿指定轴的最大值，保证指数计算不溢出
    x = x - numpy.max(x, axis=axis, keepdims=True)
    # 计算指数
    e_x = numpy.exp(x)
    # 除以总和完成归一化，得到概率分布
    return e_x / numpy.sum(e_x, axis=axis, keepdims=True)


def normalized(x: List[float], temp: float = 1) -> List[float]:
    # 把一组分数归一化为概率分布：转 numpy 后做稳定 softmax 再转回列表
    return list(stable_softmax(numpy.asarray(x), temp=temp))
