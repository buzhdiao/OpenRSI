# 导入 importlib：按字符串路径动态导入模块（用于加载自定义函数）
import importlib
# 导入 subprocess：在宿主机上执行 shell 命令
import subprocess

# 导入 ray：获取当前节点的 IP 地址
import ray

# 导入端口探测工具：查找连续空闲端口时使用
from slime.utils.http_utils import is_port_available


def load_function(path):
    """
    Load a function from a module.
    :param path: The path to the function, e.g. "module.submodule.function".
    :return: The function object.
    """
    # 以最后一个点为界，把 "module.submodule.function" 拆成模块路径与函数名
    module_path, _, attr = path.rpartition(".")
    # 动态导入对应模块
    module = importlib.import_module(module_path)
    # 从模块中取出函数对象并返回
    return getattr(module, attr)


class SingletonMeta(type):
    """
    A metaclass for creating singleton classes.
    """

    # 类级字典：缓存每个单例类已创建的实例
    _instances = {}

    def __call__(cls, *args, **kwargs):
        # 该类尚未创建过实例时才真正实例化
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            # 登记到缓存，之后的调用都会复用它
            cls._instances[cls] = instance
        # 始终返回缓存的同一实例（单例语义）
        return cls._instances[cls]

    def clear_instances(cls):
        # 清空缓存（主要用于测试间隔离），下次调用会重新实例化
        cls._instances = {}


def exec_command(cmd: str, capture_output: bool = False) -> str | None:
    # 打印即将执行的命令（flush 保证实时可见）
    print(f"EXEC: {cmd}", flush=True)

    try:
        result = subprocess.run(
            # 用 bash -c 执行命令字符串，保留 shell 语法（管道/变量展开等）
            ["bash", "-c", cmd],
            shell=False,
            # 命令返回非零退出码时抛出异常
            check=True,
            # 是否捕获子进程的 stdout/stderr
            capture_output=capture_output,
            # 捕获模式下要求输出按文本返回；非捕获模式不传该参数
            **(dict(text=True) if capture_output else {}),
        )
    except subprocess.CalledProcessError as e:
        if capture_output:
            # 命令失败时把已捕获的输出打印出来，便于定位问题
            print(f"{e.stdout=} {e.stderr=}")
        # 继续向上抛出异常，由调用方决定如何处理
        raise

    if capture_output:
        # 打印捕获到的标准输出与标准错误
        print(f"Captured stdout={result.stdout} stderr={result.stderr}")
        # 返回标准输出内容
        return result.stdout


def get_current_node_ip():
    # 通过 Ray 内部服务接口查询当前节点的 IP（在 Ray 集群内最可靠）
    address = ray._private.services.get_node_ip_address()
    # strip ipv6 address
    # 去掉 IPv6 地址可能带的方括号
    address = address.strip("[]")
    # 返回干净的 IP 字符串
    return address


def get_free_port(start_port=10000, consecutive=1):
    # find the port where port, port + 1, port + 2, ... port + consecutive - 1 are all available
    # 从起始端口开始，找到连续 consecutive 个端口都空闲的位置（如 ServerClient 需要端口段）
    port = start_port
    # 只要存在任一被占用的端口就继续后移
    while not all(is_port_available(port + i) for i in range(consecutive)):
        port += 1
    # 返回满足连续空闲要求的起始端口
    return port


def should_run_periodic_action(
    rollout_id: int,
    interval: int | None,
    num_rollout_per_epoch: int | None = None,
    num_rollout: int | None = None,
) -> bool:
    """
    Return True when a periodic action (eval/save/checkpoint) should run.

    Args:
        rollout_id: The current rollout index (0-based).
        interval: Desired cadence; disables checks when None.
        num_rollout_per_epoch: Optional epoch boundary to treat as a trigger.
    """
    # 未配置周期间隔时，周期性动作整体禁用
    if interval is None:
        return False

    # 到达最后一轮 rollout 时强制触发（确保收尾动作如最终保存一定执行）
    if num_rollout is not None and rollout_id == num_rollout - 1:
        return True

    # 转为从 1 开始的步数，便于做整除判断
    step = rollout_id + 1
    # 按配置间隔触发，或到达 epoch 边界（步数是每轮 rollout 数的整数倍）触发
    return (step % interval == 0) or (num_rollout_per_epoch is not None and step % num_rollout_per_epoch == 0)


class Box:
    def __init__(self, inner):
        # 把对象包一层"盒子"，绕开某些框架对属性赋值的拦截（保持引用可变）
        self._inner = inner

    @property
    def inner(self):
        # 只读访问内部对象
        return self._inner


from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

import torch


# details: https://stackoverflow.com/questions/773/how-do-i-use-itertools-groupby
def group_by(iterable, key=None):
    """Similar to itertools.groupby, but do not require iterable to be sorted"""
    # 用默认字典收集分组结果（无需像 itertools.groupby 那样先排序）
    ret = defaultdict(list)
    # 按键函数（缺省用元素自身）把元素归入对应分组
    for item in iterable:
        ret[key(item) if key is not None else item].append(item)
    # 转为普通字典返回
    return dict(ret)


# TODO fsdp can also use this
def chunk_named_params_by_size(named_params: Iterable[tuple[str, torch.Tensor]], chunk_size: int):
    # 按张量字节数把命名参数切分成大小受控的块（分桶传输权重时使用）
    return _chunk_by_size(
        named_params,
        # 每个元素的大小 = 该参数张量的字节数
        compute_size=lambda named_weight: named_weight[1].nbytes,
        chunk_size=chunk_size,
    )


def _chunk_by_size(objects: Iterable[Any], compute_size: Callable[[Any], int], chunk_size: int):
    # 当前正在累积的分桶
    bucket: list[Any] = []
    # 当前桶的累计大小
    bucket_size = 0

    # 逐个对象装箱
    for obj in objects:
        # 计算当前对象的尺寸
        obj_size = compute_size(obj)

        # 桶非空且再加入会超限时：先交付当前桶再开新桶
        if bucket and (bucket_size + obj_size) >= chunk_size:
            yield bucket
            bucket = []
            bucket_size = 0

        # 对象入桶并累计大小
        bucket.append(obj)
        bucket_size += obj_size

    # 循环结束后把最后未满的桶也交付出去
    if bucket:
        yield bucket
