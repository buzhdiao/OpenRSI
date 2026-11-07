# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are MIT-licensed
# Copyright (c) 2024 Weco AI Ltd
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/WecoAI/aideml/blob/main/LICENSE

"""
Python interpreter for executing code snippets and capturing their output.
Supports:
- captures stdout and stderr
- captures exceptions and stack traces
- limits execution time
"""

# 操作系统接口：切换目录、发送信号、删除文件
import os
# 队列模块：用于捕获 Queue.Empty 异常
import queue
# 信号机制：向子进程发送中断/强杀信号
import signal
# 运行时系统接口：重定向 stdout/stderr、修改模块搜索路径
import sys
# 时间工具：计算执行耗时
import time
# 异常栈追踪：格式化与提取堆栈信息
import traceback
# 多进程子进程与进程间通信队列
from multiprocessing import Process, Queue
# 文件路径对象
from pathlib import Path
# 任意类型注解
from typing import Any

# 把秒数转为人类可读形式（如 "3 minutes"）
import humanize
# OmegaConf 配置库：在 main() 演示中构建配置
from omegaconf import OmegaConf

# 解释器抽象基类与执行结果类型
from dojo.core.interpreters.base import ExecutionResult, Interpreter
# 日志工具：聚合日志器、事件类型、获取全局日志器
from dojo.utils.logger import CollectiveLogger, LogEvent, get_logger
# 目录内容复制工具
from dojo.core.interpreters.utils import copy_contents

# Python 解释器配置类
from dojo.config_dataclasses.interpreter.python import PythonInterpreterConfig

# 标准日志模块
import logging

# 模块级标准 logger
log = logging.getLogger(__name__)


def exception_summary(
    e: BaseException,
    working_dir: Path,
    exec_file_name: str,
) -> tuple[str, str, dict, list[tuple]]:
    """Generates a string that summarizes an exception and its stack trace."""
    # 把异常格式化为多行栈追踪文本
    tb_lines = traceback.format_exception(e)
    # 过滤掉 dojo 内部与 importlib 的栈帧，只保留用户代码相关部分
    tb_str = "".join([line for line in tb_lines if "dojo/" not in line and "importlib" not in line])

    # replace whole path to file with just filename (to remove agent workspace dir)
    # 把文件的完整路径替换为纯文件名，避免把智能体工作区绝对路径泄漏给 LLM
    tb_str = tb_str.replace(str(working_dir / exec_file_name), exec_file_name)

    # 收集异常对象的结构化信息
    exc_info = {}
    # 有 args 属性则记录异常参数
    if hasattr(e, "args"):
        # 参数逐个转为字符串
        exc_info["args"] = [str(i) for i in e.args]
    # 常见异常附加属性（如 ImportError.name）
    for att in ["name", "msg", "obj"]:
        # 存在才记录
        if hasattr(e, att):
            # 属性值转为字符串
            exc_info[att] = str(getattr(e, att))

    # 提取结构化栈帧列表
    tb = traceback.extract_tb(e.__traceback__)
    # 每帧取（文件名、行号、函数名、代码行）
    exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]

    # 返回（栈文本、异常类名、异常信息、栈帧列表）四元组
    return tb_str, e.__class__.__name__, exc_info, exc_stack


# 一个"文件式"对象：把写入内容转发到多进程队列（用于重定向子进程输出）
class RedirectQueue:
    """A file-like object that writes to a multiprocessing Queue."""

    def __init__(self, queue: Queue, timeout: float = 5) -> None:
        # 底层多进程队列
        self.queue = queue
        # 单次写入的阻塞超时（秒）
        self.timeout = timeout

    def write(self, msg: str) -> None:
        try:
            # 把消息放入队列，最多等 timeout 秒
            self.queue.put(msg, timeout=self.timeout)
        except queue.Full:
            # 队列满说明消费端卡住，记录写入超时告警
            logging.warning("RedirectQueue write timed out")

    def flush(self) -> None:
        # 文件接口要求的 flush，这里无需真正刷新
        pass


# Python 解释器：在独立子进程中执行 LLM 生成的代码，捕获输出与异常并限制执行时间
class PythonInterpreter(Interpreter):
    # 标记该解释器在本地（当前机器）运行
    local = True
    # 标记不是工厂类
    factory = False

    def __init__(
        self,
        cfg: PythonInterpreterConfig,
        data_dir: Path | None = None,
    ) -> None:
        """
        Simulates a standalone Python REPL with an execution time limit.

        Args:
            working_dir (Path | str): Working directory of the agent
            timeout (int, optional): Timeout for each code execution step in seconds. Defaults to 3600.
        """
        # 获取全局聚合日志器
        self.logger = get_logger()
        # 解析工作目录为绝对路径
        self.working_dir = Path(cfg.working_dir).resolve()

        # 工作目录不存在则递归创建
        if not self.working_dir.exists():
            # 记录创建目录的信息日志
            log.info(f"Working directory {self.working_dir} does not exist -- creating it")
            # 递归创建目录
            self.working_dir.mkdir(parents=True, exist_ok=True)

        # 显式传入了数据目录
        if data_dir is not None:
            # 解析为绝对路径
            self.data_dir = Path(data_dir).resolve()
            # assume the data_dir is already created and ready
            # 工作区内的 data 链接/副本位置
            data_link = self.working_dir / "data"

            # 配置为使用符号链接
            if cfg.use_symlinks:
                # 已存在旧链接则先删除
                if data_link.is_symlink():
                    data_link.unlink()
                # 建立指向真实数据目录的符号链接
                data_link.symlink_to(self.data_dir)
            else:
                # 直接把数据内容复制进工作区
                copy_contents(source=self.data_dir, destination=data_link, use_symlinks=False)
        else:
            # 未传数据目录：在工作区内新建一个空 data 目录兜底
            self.data_dir = self.working_dir / "data"
            self.data_dir.mkdir(parents=True, exist_ok=True)
            # 记录告警（原代码此 f-string 缺少 f 前缀，原样保留）
            log.info("WARNING: A data directory was not passed, so it was created at {str(self.data_dir)}")

        # 从配置读取单次执行超时（秒）
        self.timeout = cfg.timeout
        # 子进程句柄，初始为 None
        self.process: Process | None = None  # type: ignore

    def child_proc_setup(self, result_outq: Queue) -> None:
        """
        Pre-execution setup in the child process:
        - Changes directory
        - Disables warnings
        - Redirects stdout/stderr to a queue
        """
        # 第三方库：一键屏蔽 Python 运行期警告
        import shutup

        # 屏蔽所有告警输出，避免污染捕获的终端输出
        shutup.mute_warnings()
        # 子进程切换到智能体工作目录
        os.chdir(str(self.working_dir))

        # this helps python find modules in the current working dir
        # 把工作目录加入模块搜索路径，便于 Python 找到当前目录下的模块
        sys.path.append(str(self.working_dir))

        # capture stdout and stderr
        # 重定向标准输出与标准错误，统一捕获进队列
        sys.stdout = sys.stderr = RedirectQueue(result_outq)

    def _run_session(
        self,
        code_inq: Queue,
        result_outq: Queue,
        event_outq: Queue,
    ) -> None:
        """
        Main loop running in the child process.
        Waits for (code, file_name, persist_file, execute_code) to arrive in `code_inq`,
        writes it to `file_name`, executes it, and reports results back.
        """
        # 先做子进程环境初始化
        self.child_proc_setup(result_outq)

        # 全局命名空间：跨多次 exec 保持会话状态（reset_session=False 时复用）
        global_scope: dict[str, Any] = {}
        # 持续循环接收任务
        while True:
            # Here, we expect a tuple: (code, file_name, persist_file).
            # 阻塞等待父进程发来下一个任务
            data = code_inq.get()
            # If you had a sentinel-based exit, you could do:
            # if data == "quit":
            #     break

            # 解包任务参数（代码、文件名、是否保留文件、是否执行）
            code, agent_file_name, persist_file, execute_code = data
            # 每次执行前都回到工作目录，防止上一次执行改了目录
            os.chdir(str(self.working_dir))

            # Write code to the chosen file name
            # 把代码写入指定文件名（便于 traceback 显示与持久化）
            with open(agent_file_name, "w") as f:
                f.write(code)

            # 需要真正执行代码
            if execute_code:
                # 通知父进程：即将开始执行
                event_outq.put(("state:ready",))
                try:
                    # 编译并在全局命名空间中执行代码
                    exec(compile(code, agent_file_name, "exec"), global_scope)
                    # We retrieve the user-supplied result (if any) from __result__
                    # 从 __result__ 变量取出代码约定的返回结果（无则为 None）
                    return_value = global_scope.get("__result__", None)

                    # 上报执行成功事件（异常字段均为 None）
                    event_outq.put(("state:finished", None, None, None, return_value))
                except BaseException as e:
                    # 汇总异常信息（栈文本、类名、信息、栈帧）
                    tb_str, e_cls_name, exc_info, exc_stack = exception_summary(
                        e,
                        self.working_dir,
                        agent_file_name,
                    )
                    # 把栈追踪文本送入输出队列
                    result_outq.put(tb_str)
                    # 父进程超时会发 SIGINT，此处把 KeyboardInterrupt 归类为超时错误
                    if e_cls_name == "KeyboardInterrupt":
                        e_cls_name = "TimeoutError"

                    # In case of an exception, we pass None for return_value
                    # 上报失败事件与异常四元组（返回值置 None）
                    event_outq.put(("state:finished", e_cls_name, exc_info, exc_stack, None))
            else:
                # 不执行代码（仅落盘）也按协议发"就绪"事件
                event_outq.put(("state:ready",))
                # 再发"完成"事件（无异常、无返回值）
                event_outq.put(("state:finished", None, None, None, None))

            # Remove the file only if it is NOT meant to persist
            # 仅当不要求保留文件时才删除刚写入的脚本
            if not persist_file:
                os.remove(agent_file_name)

            # put EOF marker to indicate that we're done capturing output
            # 放入 EOF 标记，告知父进程本轮输出捕获结束
            result_outq.put("<|EOF|>")

    def create_process(self) -> None:
        """
        Spawns the child process that will run Python code in an isolated environment.
        """
        # we use three queues to communicate with the child process:
        # - code_inq: send code to child to execute
        # - result_outq: receive stdout/stderr from child
        # - event_outq: receive events from child (e.g. state:ready, state:finished)
        # 创建代码输入队列
        self.code_inq = Queue()
        # 创建结果输出队列
        self.result_outq = Queue()
        # 创建事件队列
        self.event_outq = Queue()
        # 构造子进程，入口为 _run_session
        self.process = Process(
            target=self._run_session,
            args=(self.code_inq, self.result_outq, self.event_outq),
        )
        # 启动子进程
        self.process.start()

    def cleanup_session(self) -> None:
        """
        Terminate the child process if it's still running, with escalation (terminate -> kill -> sigkill).
        """
        # 无子进程则无事可做
        if self.process is None:
            return
        try:
            # 第一步：温和终止（SIGTERM）
            self.process.terminate()
            # 最多等 2 秒看是否退出
            self.process.join(timeout=2)

            # 退出码仍为 None 说明还没死
            if self.process.exitcode is None:
                # 记录升级终止的告警
                self.logger.warning("Process failed to terminate, killing immediately", LogEvent.INTERPRETER)
                # 第二步：立即杀死（SIGKILL 语义）
                self.process.kill()
                # 等待彻底退出
                self.process.join()

                # 仍然没死则动用原生信号
                if self.process.exitcode is None:
                    # 记录错误
                    self.logger.error("Process refuses to die, using SIGKILL", LogEvent.INTERPRETER)
                    # 第三步：直接发送 SIGKILL 强杀
                    os.kill(self.process.pid, signal.SIGKILL)
        except Exception as e:
            # 清理过程中的任何异常都不外抛，只记录
            self.logger.error(f"Error during process cleanup: {e}", LogEvent.INTERPRETER)
        finally:
            # 无论成功与否都释放进程句柄
            if self.process is not None:
                # 关闭进程对象（释放队列等底层资源）
                self.process.close()
                # 置空引用
                self.process = None

    def fetch_file(
        self,
        path: str,
    ) -> str | None:
        # 文件不存在则返回 None
        if not Path(path).exists():
            return None
        # 存在则原样返回路径（本地解释器无需真正"取回"）
        return path

    def run(
        self,
        code: str,
        reset_session: bool = True,
        persist_file: bool = False,
        file_name: str = "runfile.py",
        execute_code: bool = True,
        include_exec_time: bool = True,
    ) -> ExecutionResult:
        """
        Execute the provided Python code in a separate process and return its output.

        Parameters:
            code (str): Python code to execute.
            reset_session (bool, optional): Whether to reset the interpreter session
                before executing the code. Defaults to True.
            persist_file (bool, optional): If True, do not delete the file after execution.
                Defaults to False.
            file_name (str, optional): Name of the file into which the code is written
                before execution. Defaults to "runfile.py".
            execute_code: (bool): Whether to execute the code or not. If False, this
                function simply writes the code to a file.

        Returns:
            ExecutionResult: Object containing the output, metadata, and any exception info.
        """

        # 记录 debug 日志：开始执行及是否重置会话
        self.logger.debug(f"REPL is executing code (reset_session={reset_session})", LogEvent.INTERPRETER)
        # 记录即将执行的完整代码
        log.info(f"Executing code:\n```\npython\n{code}\n```")

        # 需要全新会话
        if reset_session:
            # 已有旧子进程
            if self.process is not None:
                # terminate and clean up previous process
                # 回收旧进程资源
                self.cleanup_session()
            # 派生新的子进程
            self.create_process()
        else:
            # reset_session must be True on first exec
            # 复用现有会话：首次执行必须先 reset_session=True 建立子进程
            assert self.process is not None, "Process not started. Try reset_session=True first."

        # 确认子进程存活可接收任务
        assert self.process.is_alive()

        # Send the tuple (code, file_name, persist_file, execute_code) to the child
        # 把任务元组投递给子进程
        self.code_inq.put((code, file_name, persist_file, execute_code))

        # wait for child to actually start execution
        # 等待子进程真正开始执行（10 秒超时）
        try:
            # 阻塞等待 "state:ready" 事件
            state = self.event_outq.get(timeout=10)
        except queue.Empty:
            # 超时未收到 -> 子进程启动失败
            msg = "REPL child process failed to start execution"
            # 记录严重错误
            self.logger.critical(msg, LogEvent.INTERPRETER)
            # 准备转储输出队列内容用于排查
            queue_dump = ""
            # 排空结果队列
            while not self.result_outq.empty():
                # 逐条取出（保留最后一条作为摘要）
                queue_dump = self.result_outq.get()
                # 记录队列内容（截断到 1000 字符）
                self.logger.error(f"REPL output queue dump: {queue_dump[:1000]}", LogEvent.INTERPRETER)
            # 清理僵死的子进程
            self.cleanup_session()
            # 返回失败结果（退出码 1）
            return ExecutionResult(term_out=[msg, queue_dump], exec_time=0, exit_code=1)
        # 确认收到的是 ready 事件
        assert state[0] == "state:ready", state
        # 记录开始时间，用于计算耗时
        start_time = time.time()

        # 标记子进程是否已超出时限（避免重复发送 SIGINT）
        child_in_overtime = False  # indicates if we've exceeded time limit

        # 轮询等待执行结束或超时
        while True:
            try:
                # wait for state:finished from child
                # 等待子进程的 finished 事件（1 秒轮询间隔）
                state = self.event_outq.get(timeout=1)
                # 确认是 finished 事件
                assert state[0] == "state:finished", state
                # 计算实际执行耗时
                exec_time = time.time() - start_time
                # 正常结束，跳出轮询
                break
            except queue.Empty:
                # no message yet, check if child is alive
                # 1 秒内没有新事件：未在加班期却已死亡 -> 崩溃
                if not child_in_overtime and not self.process.is_alive():
                    # 崩溃说明
                    msg = "REPL child process died unexpectedly"
                    # 记录严重错误
                    self.logger.critical(msg, LogEvent.INTERPRETER)
                    # 准备转储输出队列
                    queue_dump = ""
                    # 排空结果队列
                    while not self.result_outq.empty():
                        # 取出残留输出
                        queue_dump = self.result_outq.get()
                        # 记录转储内容
                        self.logger.error(f"REPL output queue dump: {queue_dump[:1000]}", LogEvent.INTERPRETER)
                    # 清理死亡进程
                    self.cleanup_session()
                    # 返回失败结果
                    return ExecutionResult(term_out=[msg, queue_dump], exec_time=0, exit_code=1)

                # child is alive, check timeout
                # 子进程还活着：未配置超时则无限等待
                if self.timeout is None:
                    continue
                # 当前已运行时长
                running_time = time.time() - start_time
                # 超过时限
                if running_time > self.timeout:
                    # 记录超时告警
                    self.logger.warning(f"Execution exceeded timeout of {self.timeout}s", LogEvent.INTERPRETER)
                    # 向子进程发 SIGINT 触发 KeyboardInterrupt（子进程把它转成 TimeoutError）
                    os.kill(self.process.pid, signal.SIGINT)
                    # 标记进入加班期
                    child_in_overtime = True

                    # terminate if we're overtime by more than 5 seconds
                    # 若超出宽限期（60 秒）仍不退出则强杀
                    if running_time > self.timeout + 60:
                        # 记录强杀告警
                        self.logger.warning("Child failed to terminate, killing it..", LogEvent.INTERPRETER)
                        # 强制清理子进程
                        self.cleanup_session()

                        # 手工构造超时"finished"事件元组
                        state = (None, "TimeoutError", {}, [], None)
                        # 耗时按配置的超时值计
                        exec_time = self.timeout
                        # 结束轮询
                        break

        # we now unpack a 5-tuple from the child
        # state: ("state:finished", exc_type, exc_info, exc_stack, eval_return)
        # 解包子进程回传的 5 元组（取异常类名、异常信息、栈帧、评估返回值）
        e_cls_name, exc_info, exc_stack, eval_return = state[1:]

        # collect all output lines from child up to the EOF marker
        # 从输出队列收集子进程的全部输出，直到读到 EOF 标记
        output: list[str] = []
        # 记录收集开始时间
        start_collect = time.time()
        # 队列非空或尚未读到 EOF 就继续收
        while not self.result_outq.empty() or not output or output[-1] != "<|EOF|>":
            try:
                # 收集超过 10 秒仍未 EOF 则放弃
                if time.time() - start_collect > 10:
                    # 记录收集超时告警
                    self.logger.warning("Output collection timed out", LogEvent.INTERPRETER)
                    break
                # 取一条输出（最多等 1 秒）
                output.append(self.result_outq.get(timeout=1))
            except queue.Empty:
                # 暂时无数据则继续等
                continue
        # remove the EOF marker if present
        # 若最后一条是 EOF 标记则移除（它不属于真实输出）
        if output and output[-1] == "<|EOF|>":
            output.pop()

        # if we timed out, show that in the output
        # 若是超时结束，则在输出末尾追加超时说明
        if e_cls_name == "TimeoutError":
            # 人类可读的超时提示
            output.append(f"TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}")
        elif include_exec_time:
            # 否则按需附带执行耗时说明
            output.append(
                f"Execution time: {humanize.naturaldelta(exec_time)} (time limit is {humanize.naturaldelta(self.timeout)})."
            )

        # 记录执行完成日志
        log.info(f"Code execution finished.")

        # 组装执行结果对象返回（无异常退出码 0，有异常为 1）
        return ExecutionResult(
            term_out=output,
            exec_time=exec_time,
            exit_code=0 if e_cls_name is None else 1,
            eval_return=eval_return,
        )


def main():
    # Instantiate our dummy logger.
    # 本函数是解释器的手动演示入口：构造简易配置与日志器
    cfg = {}
    # 日志子配置（同时赋给临时变量 logger_config）
    cfg["logger"] = logger_config = {
        "base_exp_path": "results",  # Base path for logging.
        "use_console": True,  # Whether to log to stdout.
        "use_wandb": False,  # Whether to log to wandb.ai."
    }
    # 转为 OmegaConf 配置对象
    cfg = OmegaConf.create(cfg)
    # 创建聚合日志器
    logger = CollectiveLogger(cfg)

    # Use the current working directory (must exist).
    # 解析当前目录为绝对路径作为运行目录
    working_dir = Path(".").resolve()

    # Create an instance of PythonInterpreter with a 10-second timeout.
    # 创建一个 10 秒超时的解释器实例（注意：此调用签名与当前类定义不一致，演示代码未同步）
    interpreter = PythonInterpreter(logger, working_dir, timeout=10)

    # Define a simple Python code snippet to execute.
    # 定义一段简单的待执行代码（打印问候与计数循环，并按约定把结果写入 __result__）
    code_snippet = """
print("Hello from PythonInterpreter!")
for i in range(5):
    print("Count:", i)
__result__ = "Execution Completed Successfully"
"""

    # Execute the code.
    # 执行这段代码
    result = interpreter.run(
        code=code_snippet,
        reset_session=True,
        persist_file=False,
        file_name="demo_run.py",
        execute_code=True,
    )

    # Print the captured terminal output and other execution details.
    # 打印捕获的终端输出与其他执行细节
    print("=== Terminal Output ===")
    # 遍历输出行
    for line in result.term_out:
        # 逐行打印
        print(line)
    # 打印执行耗时
    print("\nExecution Time:", result.exec_time)
    # 打印返回值
    print("Return Value:", result.eval_return)

    # Clean up the interpreter process.
    # 清理解释器子进程
    interpreter.cleanup_session()


if __name__ == "__main__":
    # 以脚本方式运行时执行演示
    main()
