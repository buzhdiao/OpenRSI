# 导入 logging：模块级日志
import logging
# 导入 contextmanager：把计时器包装成 with 上下文管理器
from contextlib import contextmanager
# 导入 wraps：装饰器包装时保留原函数元信息
from functools import wraps
# 导入 time 函数：获取当前墙钟时间（秒）
from time import time

# 导入 torch.distributed：判断分布式是否初始化、查询 rank（仅 rank0 打日志）
import torch.distributed

# 导入单例元类：保证全局只有一个 Timer 实例
from .misc import SingletonMeta

# 模块对外暴露 Timer 类与 timer 装饰器
__all__ = ["Timer", "timer"]

# 创建本模块专属 logger
logger = logging.getLogger(__name__)


class Timer(metaclass=SingletonMeta):
    def __init__(self):
        # 累计耗时表：名称 -> 累计秒数
        self.timers = {}
        # 进行中的计时：名称 -> 开始时间戳
        self.start_time = {}

    def start(self, name):
        # 同名计时器不允许重复启动
        assert name not in self.start_time, f"Timer {name} already started."
        # 记录开始时间戳
        self.start_time[name] = time()
        # 分布式已初始化且为 rank0 时打印启动日志（避免多 rank 刷屏）
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(f"Timer {name} start")

    def end(self, name):
        # 结束前必须已经启动
        assert name in self.start_time, f"Timer {name} not started."
        # 计算本次耗时
        elapsed_time = time() - self.start_time[name]
        # 累加到总耗时表
        self.add(name, elapsed_time)
        # 从进行中表移除
        del self.start_time[name]
        # rank0 打印结束日志与耗时
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(f"Timer {name} end (elapsed: {elapsed_time:.1f}s)")

    def reset(self, name=None):
        if name is None:
            # 未指定名称：清空全部累计计时
            self.timers = {}
        elif name in self.timers:
            # 指定名称且存在：仅删除该名称的累计值
            del self.timers[name]

    def add(self, name, elapsed_time):
        # 累加耗时（不存在则从 0 开始累计）
        self.timers[name] = self.timers.get(name, 0) + elapsed_time

    def log_dict(self):
        # 返回全部累计耗时的字典（供上报 perf 指标）
        return self.timers

    @contextmanager
    def context(self, name):
        # 进入上下文时启动计时
        self.start(name)
        try:
            # 把控制权交给 with 语句体
            yield
        finally:
            # 无论正常退出还是异常，都结束计时
            self.end(name)


def timer(name_or_func):
    """
    Can be used either as a decorator or a context manager:

    @timer
    def func():
        ...

    or

    with timer("block_name"):
        ...
    """
    # When used as a context manager
    # 传入字符串时按上下文管理器使用
    if isinstance(name_or_func, str):
        name = name_or_func
        # 返回计时上下文管理器
        return Timer().context(name)

    # 否则按装饰器使用：name_or_func 即被装饰函数
    func = name_or_func

    @wraps(func)
    def wrapper(*args, **kwargs):
        # 以函数名作为计时名称包裹调用
        with Timer().context(func.__name__):
            return func(*args, **kwargs)

    # 返回包装后的函数
    return wrapper


@contextmanager
def inverse_timer(name):
    # 反向计时器：进入时"结束"指定计时器，退出时重新"启动"（用于把某段区间排除在外部计时之外）
    Timer().end(name)
    try:
        # 交出控制权
        yield
    finally:
        # 退出时恢复计时
        Timer().start(name)


def with_defer(deferred_func):
    # 装饰器工厂：把"延迟执行"的函数挂到目标函数上
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                # 正常执行原函数
                return fn(*args, **kwargs)
            finally:
                # 无论成功失败都执行延迟函数（类似 Go 的 defer）
                deferred_func()

        # 返回包装后的函数
        return wrapper

    # 返回装饰器
    return decorator
