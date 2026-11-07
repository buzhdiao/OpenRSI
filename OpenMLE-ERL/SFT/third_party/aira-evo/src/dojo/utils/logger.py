# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are Apache 2.0 licensed
# Copyright (c) 2024 Edan Toledo
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/EdanToledo/Stoix/blob/main/LICENSE

# 导入 abc，用于定义日志器抽象基类
import abc
# 导入 json，用于校验日志数据可序列化
import json
# 导入 logging，标准日志模块
import logging
# 导入 os，用于路径拼接
import os
# 导入 time，用于缓冲落盘计时
import time
# 导入 uuid4，生成全局唯一运行标识
from uuid import uuid4
# 导入 zipfile（预留）
import zipfile
# 导入 defaultdict，按事件命名空间缓冲日志条目
from collections import defaultdict
# 导入 datetime，生成时间戳
from datetime import datetime
# 导入 Enum，定义日志事件类型
from enum import Enum
# 导入 Path，用于日志文件路径操作
from pathlib import Path
# 导入类型注解工具
from typing import Dict, List, Optional, Union

# 导入 jsonlines，按行追加写入 JSON 日志
import jsonlines
# 导入 numpy，用于统计描述函数
import numpy as np

# import weave
# 导入 colorama 终端颜色常量，用于控制台彩色输出
from colorama import Fore, Style
# 导入 DataClassJsonMixin，识别并转换 dataclass 指标对象
from dataclasses_json import DataClassJsonMixin
# 导入 omegaconf 的 DictConfig 配置类型
from omegaconf import DictConfig
# 导入 pandas 内部的字典扁平化工具，用于展平嵌套指标字典
from pandas.io.json._normalize import _simple_json_normalize as flatten_dict

# 导入 wandb 实验跟踪平台客户端
import wandb
# 导入日志器配置数据类
from dojo.config_dataclasses.logger import LoggerConfig


# 日志事件类型枚举：区分指标来自哪个组件（智能体/求解器/任务/解释器等）
class LogEvent(Enum):
    # 智能体相关事件
    AGENT = "agent"
    # 求解器相关事件
    SOLVER = "solver"
    # 任务相关事件
    TASK = "task"
    # 解释器相关事件
    INTERPRETER = "interpreter"
    # 杂项事件
    MISC = "misc"
    # 评估相关事件
    EVAL = "eval"
    # LLM 客户端相关事件
    LLM_CLIENT = "llm_client"


# 各事件类型在控制台输出时使用的颜色映射
EVENT_COLOURS = {
    LogEvent.AGENT: Fore.MAGENTA,
    LogEvent.EVAL: Fore.GREEN,
    LogEvent.INTERPRETER: Fore.WHITE,
    LogEvent.SOLVER: Fore.CYAN,
    LogEvent.TASK: Fore.YELLOW,
    LogEvent.MISC: Fore.RED,
    LogEvent.LLM_CLIENT: Fore.BLUE,
}


# 全局集合日志器：统一管理指标/文件日志与彩色文本消息
class CollectiveLogger:
    """The main logger."""

    def __init__(self) -> None:
        # 配置占位：未配置前为 Ellipsis
        self._cfg = ...

    @property
    def cfg(self):
        # 未配置时访问配置直接报错
        if self._cfg == ...:
            raise ValueError("No config set.")
        # 返回已保存的配置
        return self._cfg

    # 依据配置创建底层多路日志器；传入空配置则进入低资源模式（不记录指标）
    def config(self, config: DictConfig) -> None:
        if config:
            # 按配置创建并挂载多路日志器（wandb/console/json 组合）
            self.logger: BaseLogger = _make_multi_logger(config)
            # 保存配置
            self._cfg = config
            # 正常资源模式，允许记录指标
            self.low_resource_use = False
        else:
            # 低资源模式：跳过指标记录
            self.low_resource_use = True
            # 配置置空
            self._cfg = None
        # 返回自身，支持链式调用
        return self

    def log(self, metrics: Dict, event: Union[LogEvent, str], step: Optional[int] = None) -> None:
        """Log a dictionary metrics at a given timestep.

        Args:
            metrics (Dict): dictionary of metrics to log.
            event (LogEvent): the event that the metrics are associated with.
            step (Optional[int]): the step at which these metrics are logged.
        """
        # 非低资源模式才真正记录指标字典
        if not self.low_resource_use:
            self.logger.log_dict(metrics, event, step)

    # 记录一个文件（同样受低资源模式约束）
    def log_file(self, file_path: str) -> None:
        if not self.low_resource_use:
            self.logger.log_file(file_path)

    # 停止底层日志器（如结束 wandb 运行、刷新缓冲）
    def stop(self) -> None:
        """Stop the logger."""
        self.logger.stop()

    # 把消息包装为带事件颜色与事件名前缀的格式化文本
    def process_event_message(self, msg: object, event: Optional[LogEvent] = None) -> None:
        # 未指定事件时默认按杂项处理
        if not event:
            event = LogEvent.MISC
        # 用事件对应颜色 + 高亮 + [事件名] 前缀包装消息，结尾重置样式
        return f"{EVENT_COLOURS[event]}{Style.BRIGHT}[{event.value.upper()}] - {msg}{Style.RESET_ALL}"

    # 以 info 级别记录带事件标记的消息
    def info(self, msg: object, event: Optional[LogEvent] = None) -> None:
        logging.info(self.process_event_message(msg, event))

    # 以 warning 级别记录
    def warning(self, msg: object, event: Optional[LogEvent] = None) -> None:
        logging.warning(self.process_event_message(msg, event))

    # 以 error 级别记录
    def error(self, msg: object, event: Optional[LogEvent] = None) -> None:
        logging.error(self.process_event_message(msg, event))

    # 以 debug 级别记录
    def debug(self, msg: object, event: Optional[LogEvent] = None) -> None:
        logging.debug(self.process_event_message(msg, event))

    # 以 critical 级别记录
    def critical(self, msg: object, event: Optional[LogEvent] = None) -> None:
        logging.critical(self.process_event_message(msg, event))


# 日志器抽象基类：定义各后端（wandb/console/json）需实现的接口
class BaseLogger(abc.ABC):
    @abc.abstractmethod
    def __init__(self, cfg: LoggerConfig, unique_token: str) -> None:
        pass

    @abc.abstractmethod
    def log_stat(self, key: str, value: float, event: LogEvent, step: Optional[int] = None) -> None:
        """Log a single metric."""
        raise NotImplementedError

    # 记录一个指标字典：先扁平化嵌套结构，再逐键写入
    def log_dict(self, data: Dict, event: Union[LogEvent, str], step: Optional[int] = None) -> None:
        """Log a dictionary of metrics."""
        # in case the dict is nested, flatten it.
        # 嵌套字典展平，键之间用 / 连接
        data = flatten_dict(data, sep="/")

        # 逐键记录指标
        for key, value in data.items():
            # dataclass 指标对象先转为普通字典
            if isinstance(value, DataClassJsonMixin):
                value = value.to_dict()
            self.log_stat(
                key,
                value,
                event,
                step,
            )

    # 记录文件：默认不支持，需子类覆盖
    def log_file(self, file_path: str) -> None:
        raise NotImplementedError

    # 停止日志器：默认无需处理
    def stop(self) -> None:
        """Stop the logger."""
        return None


# 多路日志器：把每条记录同时转发给多个底层日志器
class MultiLogger(BaseLogger):
    """Logger that can log to multiple loggers at oncce."""

    def __init__(self, loggers: List[BaseLogger]) -> None:
        # 保存底层日志器列表
        self.loggers = loggers

    # 单指标广播给所有底层日志器
    def log_stat(self, key: str, value: float, event: LogEvent, step: Optional[int] = None) -> None:
        for logger in self.loggers:
            logger.log_stat(key, value, event, step)

    # 指标字典广播给所有底层日志器
    def log_dict(self, data: Dict, event: LogEvent, step: Optional[int] = None) -> None:
        for logger in self.loggers:
            logger.log_dict(data, event, step)

    # 文件记录广播给所有底层日志器
    def log_file(self, file_path: str) -> None:
        for logger in self.loggers:
            logger.log_file(file_path)

    # 停止时逐个关闭所有底层日志器
    def stop(self) -> None:
        for logger in self.loggers:
            logger.stop()


class WandBLogger(BaseLogger):
    """Logger for wandb.ai."""

    # 初始化：读取 wandb 配置并启动一次运行（run）
    def __init__(self, cfg: LoggerConfig, unique_token: str) -> None:
        # 读取标签列表
        tags = list(cfg.logger.tags)
        # 读取 wandb 项目名
        project = cfg.logger.wandb_project_name
        # 读取 wandb 实体（团队/用户名）
        entity = cfg.logger.wandb_entity

        # 初始化 wandb 运行，上传完整配置便于复现
        wandb.init(entity=entity, project=project, tags=tags, config=cfg)

        # 是否启用详细日志
        self.detailed_logging = cfg.logger.detailed_logging
        # 保存本次运行的唯一 token
        self.unique_token = unique_token

    # 记录单个指标：键名以 "事件命名空间/指标名" 形式写入
    def log_stat(self, key: str, value: float, event: Union[LogEvent, str], step: Optional[int] = None) -> None:
        # 事件为枚举时取其字符串值作为命名空间
        if isinstance(event, LogEvent):
            namespace = event.value
        else:
            # 否则直接用事件字符串
            namespace = event
        # 组装带命名空间的键值对
        data_to_log = {f"{namespace}/{key}": value}
        # 写入 wandb，可指定步数
        wandb.log(data_to_log, step=step)

    # 把文件上传保存到 wandb
    def log_file(self, file_path: str) -> None:
        wandb.save(file_path)

    # 结束 wandb 运行
    def stop(self) -> None:
        wandb.finish()  # type: ignore


class ConsoleLogger(BaseLogger):
    """Logger for writing to stdout."""

    # 初始化控制台日志器：绑定输出流与彩色格式
    def __init__(self, cfg: LoggerConfig, unique_token: str) -> None:
        # 获取名为 Console 的日志器
        self.logger = logging.getLogger("Console")

        # 清空已有处理器，避免重复输出
        self.logger.handlers = []

        # 新建流处理器（输出到标准流）
        ch = logging.StreamHandler()
        # 消息用青色高亮显示，时间格式为 时:分:秒
        formatter = logging.Formatter(f"{Fore.CYAN}{Style.BRIGHT}%(message)s{Style.RESET_ALL}", "%H:%M:%S")
        # 绑定格式器并挂载处理器
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # Set to info to suppress debug outputs.
        # 级别设为 INFO，过滤掉 debug 输出
        self.logger.setLevel("INFO")

    # 向控制台输出单条指标
    def log_stat(self, key: str, value: float, event: LogEvent, step: Optional[int] = None) -> None:
        # 事件为枚举时取其值作为命名空间并匹配颜色
        if isinstance(event, LogEvent):
            namespace = event.value
            colour = EVENT_COLOURS[event]
        else:
            # 字符串事件统一按杂项颜色处理
            namespace = event
            colour = EVENT_COLOURS[LogEvent.MISC]

        # Replace underscores with spaces and capitalise keys.
        # 键名下划线替换为空格并首字母大写，提升可读性
        key = key.replace("_", " ").capitalize()

        # 按颜色高亮格式输出"事件 - 键: 值（保留 3 位小数）"
        self.logger.info(f"{colour}{Style.BRIGHT}{namespace.upper()} - {key}: {value:.3f}{Style.RESET_ALL}")

    # 向控制台输出一整组指标字典
    def log_dict(self, data: Dict, event: LogEvent, step: Optional[int] = None) -> None:
        # in case the dict is nested, flatten it.
        # 嵌套字典展平为单层（键间用空格连接）
        data = flatten_dict(data, sep=" ")

        # 依据事件类型确定命名空间与显示颜色
        if isinstance(event, LogEvent):
            namespace = event.value
            colour = EVENT_COLOURS[event]
        else:
            namespace = event
            colour = EVENT_COLOURS[LogEvent.MISC]
        # Replace underscores with spaces and capitalise keys.
        # 所有键名下划线转空格并首字母大写
        keys = [k.replace("_", " ").capitalize() for k in data.keys()]
        # Round values to 3 decimal places if they are floats.
        # 逐值格式化：整数原样、浮点保留 3 位、字符串加引号换行显示、其他类型标注类型名
        values = []
        for v in data.values():
            if isinstance(v, int):
                values.append(v)
            elif isinstance(v, float):
                values.append(f"{float(v):.3f}")
            elif isinstance(v, str):
                values.append(f"\n'{v}'\n")
            else:
                t = type(v)
                v = str(v)
                values.append(f"\n\n{t.__name__.capitalize()}: \n'{v}' \n")

        # 把所有"键: 值"对用竖线拼接为一行
        log_str = " | ".join([f"{k}: {v}" for k, v in zip(keys, values)])

        # 带颜色高亮输出整行日志
        self.logger.info(f"{colour}{Style.BRIGHT}{namespace.upper()} - {log_str}{Style.RESET_ALL}")

    # 控制台日志器不支持文件记录，忽略
    def log_file(self, file_path: str) -> None:
        pass


# JSON 日志器：把指标按事件命名空间缓冲后批量写入 jsonl 文件
class JsonLogger(BaseLogger):
    def __init__(self, cfg: LoggerConfig, unique_token: str, flush_interval: int = 100) -> None:
        # 定位 json 日志输出目录 <output_dir>/json
        self.json_logs_path = Path(os.path.join(cfg.logger.output_dir, "json"))
        # 目录不存在则递归创建
        self.json_logs_path.mkdir(parents=True, exist_ok=True)

        # 按命名空间缓冲日志条目：{namespace: [log_entries]}
        self.buffer = defaultdict(list)  # {namespace: [log_entries]}
        # 记录上次落盘时间（单调时钟，不受系统时间调整影响）
        self.last_flush_time = time.monotonic()
        # 两次落盘之间的时间间隔（秒）
        self.flush_interval = flush_interval  # Time in seconds between file writes

    # 记录同一 step 下的一组指标
    def log_dict(self, data: Dict, event: Union["LogEvent", str], step: Optional[int] = None) -> None:
        """Log a dictionary of metrics under the same step."""
        # Flatten nested dicts
        # 先展平嵌套字典
        data = flatten_dict(data, sep="/")  # Flatten nested dicts
        # 交给 log_stat 缓冲
        self.log_stat(data, event, step)

    # 把整组键值对作为一条日志加入缓冲
    def log_stat(self, data: Dict[str, float], event: Union["LogEvent", str], step: Optional[int] = None) -> None:
        """Logs all key-value pairs under the same step in a buffer."""
        # 事件为枚举时取其值作为命名空间
        if isinstance(event, LogEvent):
            namespace = event.value
        else:
            # 否则直接使用事件字符串
            namespace = event

        # Create a log entry with timestamp, step, and data
        # 组装日志条目：时间戳 + 步数 + 数据
        log_entry = {"timestamp": datetime.now().isoformat(), "step": step, "data": data}

        # Check if data is JSON serializable
        # 先试序列化，确保数据可写入 JSON
        try:
            json.dumps(log_entry)
            # 可序列化则加入对应命名空间的缓冲
            self.buffer[namespace].append(log_entry)
        except (TypeError, ValueError):
            # 不可序列化的数据跳过并提示
            print(f"Skipping non-serializable data: {data}")
            return

        # Check if it's time to flush
        # 距上次落盘超过设定间隔时触发批量写入
        if time.monotonic() - self.last_flush_time > self.flush_interval:
            self.flush_logs()

    # 把缓冲中的全部日志落盘（jsonlines 追加写）
    def flush_logs(self):
        """Writes all buffered logs to disk using jsonlines."""
        # 逐命名空间处理缓冲条目
        for namespace, entries in self.buffer.items():
            # 该命名空间没有条目则跳过
            if not entries:
                continue

            # 目标文件：<json目录>/<命名空间>.jsonl
            file_path = self.json_logs_path / f"{namespace}.jsonl"

            # Append to the jsonlines file
            # 以追加模式打开 jsonlines 文件逐条写入
            with jsonlines.open(file_path, mode="a") as writer:
                for entry in entries:
                    writer.write(entry)

        # Clear buffer and update last flush time
        # 清空缓冲并重置落盘计时
        self.buffer.clear()
        self.last_flush_time = time.monotonic()

    # JSON 日志器不支持文件记录，忽略
    def log_file(self, file_path: str) -> None:
        pass

    # 停止前确保缓冲全部落盘
    def stop(self) -> None:
        """Ensure all logs are written before stopping."""
        self.flush_logs()


# 依据配置组合出多路日志器（按开关启用 wandb/console/json 后端）
def _make_multi_logger(cfg: LoggerConfig) -> BaseLogger:
    """Creates a MultiLogger given a config"""

    # 待组装的底层日志器列表
    loggers: List[BaseLogger] = []
    # 生成本次运行唯一 token：时间戳 + UUID
    unique_token = datetime.now().strftime("%Y%m%d%H%M%S%f") + str(uuid4())

    # 按配置开关逐个实例化启用的日志后端
    if cfg.logger.use_wandb:
        loggers.append(WandBLogger(cfg, unique_token))
    if cfg.logger.use_console:
        loggers.append(ConsoleLogger(cfg, unique_token))
    if cfg.logger.use_json:
        loggers.append(JsonLogger(cfg, unique_token))

    # 包装为多路日志器返回
    return MultiLogger(loggers)


# 对指标数组生成汇总统计（均值、标准差、最小值、最大值）
def describe(x: np.ndarray) -> Union[Dict[str, np.ndarray], np.ndarray]:
    """Generate summary statistics for an array of metrics (mean, std, min, max)."""

    # 非数组输入原样返回
    if not isinstance(x, np.ndarray):
        return x
    elif x.size <= 1:
        # 单元素数组去掉多余维度后直接返回
        return np.squeeze(x)

    # np instead of jnp because we don't jit here
    # 返回四种统计量组成的字典
    return {"mean": np.mean(x), "std": np.std(x), "min": np.min(x), "max": np.max(x)}


# 全局日志器注册表（按名字存放）
LOGGERS = dict()


# 初始化主日志器实例
def init_logger():
    LOGGERS["main"] = CollectiveLogger()


# 获取主日志器
def get_logger():
    return LOGGERS["main"]


# 用给定参数配置主日志器（透传给 CollectiveLogger.config）
def config_logger(*args, **kwargs):
    return get_logger().config(*args, **kwargs)


# 模块导入时即完成初始化，保证全局日志器可用
init_logger()
