# 分布式工具模块:提供 Gloo 通信组管理、可创建多个主进程组的 init_process_group、
# 以及跨卡全局统计量的掩码白化(whitening)计算。
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)


# 模块级 Gloo 通信组缓存(懒初始化的单例)
GLOO_GROUP = None


def init_gloo_group():
    """Initialize Gloo group for distributed communication."""
    # 声明修改模块级全局变量
    global GLOO_GROUP
    if GLOO_GROUP is None:
        # 首次调用时基于 NCCL 世界新建一个 Gloo 后端的通信组(CPU 通信,常用于控制面消息)
        GLOO_GROUP = dist.new_group(backend="gloo")
    # 返回全局唯一的 Gloo 组
    return GLOO_GROUP


def get_gloo_group():
    """Get the Gloo group for distributed communication."""
    global GLOO_GROUP
    if GLOO_GROUP is None:
        # 尚未初始化就取用属于用法错误,直接报错提示
        raise RuntimeError("Gloo group has not been initialized. Call _init_gloo_group() first.")
    # 返回已初始化的 Gloo 组
    return GLOO_GROUP


# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
def init_process_group(
    backend: str | Backend = None,
    init_method: str | None = None,
    timeout: timedelta | None = None,
    world_size: int = -1,
    rank: int = -1,
    store: Store | None = None,
    group_name: str = None,
    pg_options: Any | None = None,
):
    # init_method 与 store 二者只能指定其一(它们是两种等价的 rendezvous 方式)
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        # 直接给定 store 时必须显式提供全局规模与当前 rank
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        # 两者都没给时退回默认的环境变量方式(env://)
        init_method = "env://"

    if backend:
        # 把字符串后端名转成 PyTorch 的 Backend 枚举
        backend = Backend(backend)
    else:
        # 未指定后端时使用 "undefined",由框架自行选择
        backend = Backend("undefined")

    if timeout is None:
        # 未指定超时时采用 PyTorch 默认的进程组超时时间
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        # 通过 init_method(如 tcp/env/文件)执行 rendezvous,得到 store 及规整后的 rank/world_size
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        # 给 store 设置与进程组一致的超时
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        # 用 PrefixStore 按组名加前缀隔离键空间,避免多租户 store 中的键名冲突
        store = PrefixStore(group_name, store)

    # NOTE: The pg_options parameter was renamed into backend_options in PyTorch 2.6.0
    # https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # We need to determine the appropriate parameter name based on PyTorch version
    # 按 PyTorch 版本选择 _new_process_group_helper 的关键字参数名(2.6 起改名)
    pg_options_param_name = "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    # 调用内部辅助函数真正创建进程组(不指定子组 ranks,即全局主组)
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )

    # 在全局 _world 注册表里登记该组内 rank 到本地 rank 的映射,补齐官方 API 之外的状态
    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    # 返回新建的进程组
    return pg


def distributed_masked_whiten(
    values: torch.Tensor,
    mask: torch.Tensor,
    process_group: dist.ProcessGroup | None = None,
    shift_mean: bool = True,
    epsilon: float = 1e-8,
):
    """
    Performs whitening on a tensor using global statistics from all participating GPUs.

    It calculates the global mean and variance across all ranks in the default
    process group (the WORLD) and uses these global statistics to normalize the
    local data on each rank.

    Args:
        values (torch.Tensor): The local tensor of values to whiten.
        mask (torch.Tensor): The local mask corresponding to the values.
        process_group: The process group for all_reduce.
                      If None, uses the default world group.
        shift_mean (bool): If True, the output is zero-mean. Defaults to True.
        epsilon (float): A small value for numerical stability.

    Returns:
        torch.Tensor: The locally whitened tensor using global statistics.
    """
    # Calculate local intermediate statistics
    # 计算本卡掩码内的加权和
    local_sum = (values * mask).sum()
    # 计算本卡掩码内的平方和(用于求方差)
    local_sum_sq = ((values**2) * mask).sum()
    # 计算本卡掩码元素个数(即有效样本数)
    local_mask_sum = mask.sum()

    # 把三个本地统计量拼成一个 float32 张量,便于一次通信完成聚合
    stats_tensor = torch.tensor(
        [local_sum, local_sum_sq, local_mask_sum],
        device=values.device,
        dtype=torch.float32,
    )

    # Aggregate via all_reduce within the DP group
    # 在 DP 组内做 all_reduce 求和,得到跨卡的全局统计量
    dist.all_reduce(stats_tensor, group=process_group)

    # Calculate global stats from aggregated results
    # 拆出全局和、全局平方和、全局掩码和
    global_sum, global_sum_sq, global_mask_sum = stats_tensor

    if global_mask_sum.item() == 0:
        # 所有卡的有效元素总数为 0,无法计算均值/方差
        raise ValueError("The global mask sum across all participating GPUs is zero.")

    # 全局均值 = 全局和 / 全局有效元素数
    global_mean = global_sum / global_mask_sum
    # 全局平方均值,用于推导方差(E[x^2])
    global_mean_sq = global_sum_sq / global_mask_sum
    # 全局方差 = E[x^2] - (E[x])^2
    global_var = global_mean_sq - global_mean**2

    # Bessel's correction for unbiased estimate
    # 有效元素数 >= 2 时做 Bessel 校正(n/(n-1)),得到无偏方差估计
    if global_mask_sum.item() >= 2:
        bessel_correction = global_mask_sum / (global_mask_sum - 1)
        global_var = global_var * bessel_correction

    # Whiten local data using global stats
    # 用全局均值与全局方差(加 epsilon 防止除零)白化本地数据,使其零均值单位方差
    whitened_values = (values - global_mean) * torch.rsqrt(global_var + epsilon)

    if not shift_mean:
        # 不需要零均值时,把均值加回去,仅做方差缩放
        whitened_values += global_mean

    # 返回白化后的本地张量
    return whitened_values
