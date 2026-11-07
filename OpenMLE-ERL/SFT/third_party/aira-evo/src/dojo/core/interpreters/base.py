# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are MIT-licensed
# Copyright (c) 2024 Weco AI Ltd
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/WecoAI/aideml/blob/main/LICENSE

# 抽象基类与抽象方法装饰器
from abc import ABC, abstractmethod
# dataclass 装饰器
from dataclasses import dataclass
# 面向对象的文件路径类型
from pathlib import Path
# 任意类型注解
from typing import Any

# 让 dataclass 具备 JSON 序列化/反序列化能力的混入类
from dataclasses_json import DataClassJsonMixin


# 在解释器中执行一段代码的结果（含输出、耗时与异常信息），可序列化为 JSON
@dataclass
class ExecutionResult(DataClassJsonMixin):
    """
    Result of executing a code snippet in the interpreter.
    Contains the output, execution time, and exception information.
    """

    # 终端输出（stdout/stderr 逐行文本）
    term_out: list[str]
    # 执行耗时（秒）
    exec_time: float
    # 进程退出码，None 表示尚未退出或未获取
    exit_code: int | None = None
    # 评估调用（如提交评测）的返回值
    eval_return: Any | None = None
    # 是否因超时被强制终止
    timed_out: bool = False

    # 构造一个"空结果"的静态工厂方法
    @staticmethod
    def get_empty():
        # 返回所有字段均为中性默认值的 ExecutionResult
        return ExecutionResult(
            term_out=[" "],
            exec_time=0,
            exit_code=0,
            eval_return=None,
            timed_out=False,
        )


# 代码解释器的抽象基类，约定接口：创建隔离的执行环境/进程、
# 按实现执行任意语言代码、捕获并返回执行结果、清理资源
class Interpreter(ABC):
    """
    An abstract base class defining the interface for a code interpreter that:
      - Creates an isolated environment or process to run code,
      - Executes code of any language, depending on implementation.
      - Captures and returns execution results (stdout, stderr, exceptions),
      - Cleans up resources (terminates processes, closes streams, etc.).
    """

    # 初始化接口：子类必须实现
    @abstractmethod
    def __init__(
        self,
        working_dir: Path | str,
        data_dir: Path | str = None,
        timeout: int = 3600,
    ) -> None:
        """
        Initializes the interpreter with a working directory, execution timeout,
        and optional exception formatting preferences.

        Args:
            working_dir (Path | str): The working directory in which code will be executed.
            timeout (int, optional): The maximum allowed runtime for a code snippet, in seconds.
        """
        raise NotImplementedError("Subclasses must implement __init__()")

    # 执行代码接口：子类必须实现
    @abstractmethod
    def run(
        self,
        code: str,
        reset_session: bool = True,
        persist_file: bool = False,
        file_name: str = "runfile.py",
        execute_code: bool = True,
    ) -> ExecutionResult:
        """
        Execute the provided code in the environment managed by this interpreter.

        Args:
            code (str): The code to be executed.
            reset_session (bool, optional): Whether to reset the interpreter session
                before executing the code (e.g., restart the subprocess).
            persist_file (bool, optional): If True, do not delete the file with the code
                after execution. (Implementation detail may vary.)
            file_name (str, optional): Name of the file to which code is written before
                execution.
            execute_code (bool): Whether to execute the code or not. If False, this
                function simply writes the code to a file.

        Returns:
            ExecutionResult: The result of execution, including stdout, stderr, exceptions,
                             execution time, etc.
        """
        raise NotImplementedError("Subclasses must implement run()")

    # 清理会话接口：终止子进程、关闭句柄等，子类必须实现
    @abstractmethod
    def cleanup_session(self) -> None:
        """
        Cleans up resources, such as terminating the subprocess or closing open handles.
        """
        raise NotImplementedError("Subclasses must implement cleanup_session()")
