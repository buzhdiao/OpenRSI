# 导入 gc：触发 Python 层垃圾回收，释放失去引用的显存张量
import gc
# 导入 logging：记录显存使用情况日志
import logging

# 导入 torch：显存同步、缓存清理与显存统计都依赖其 CUDA 接口
import torch
# 导入 torch.distributed：获取全局 rank，便于区分不同进程的日志
import torch.distributed as dist

# 创建本模块专属 logger
logger = logging.getLogger(__name__)


def clear_memory(clear_host_memory: bool = False):
    # 先同步 CUDA 流，确保所有异步计算完成后再清理，避免清理正在使用的显存
    torch.cuda.synchronize()
    # 触发 Python 垃圾回收，让已无引用的张量进入释放流程
    gc.collect()
    # 释放 PyTorch 缓存分配器持有的空闲显存块，归还给驱动
    torch.cuda.empty_cache()
    if clear_host_memory:
        # 可选：进一步释放锁页主机内存（pinned memory）缓存
        torch._C._host_emptyCache()


def available_memory():
    # 取当前进程使用的 CUDA 设备编号
    device = torch.cuda.current_device()
    # 查询该设备上驱动层面的空闲与总显存（字节）
    free, total = torch.cuda.mem_get_info(device)
    # 汇总各类显存指标并统一换算成 GB 返回
    return {
        # 设备编号（字符串形式便于日志展示）
        "gpu": str(device),
        # 总显存
        "total_GB": _byte_to_gb(total),
        # 空闲显存（驱动视角）
        "free_GB": _byte_to_gb(free),
        # 已用显存 = 总量 - 空闲
        "used_GB": _byte_to_gb(total - free),
        # PyTorch 张量实际占用的显存
        "allocated_GB": _byte_to_gb(torch.cuda.memory_allocated(device)),
        # PyTorch 缓存分配器向驱动申请的显存（含未使用的缓存块）
        "reserved_GB": _byte_to_gb(torch.cuda.memory_reserved(device)),
    }


def _byte_to_gb(n: int):
    # 字节转 GB，保留两位小数便于日志阅读
    return round(n / (1024**3), 2)


def print_memory(msg, clear_before_print: bool = False):
    if clear_before_print:
        # 打印前先做一轮显存清理，使统计反映清理后的真实水位
        clear_memory()

    # 收集当前显存统计信息
    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    # 每个 rank 都要打印：不同 rank 的显存行为可能不同（如负载不均）
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    # 把统计结果返回给调用方复用
    return memory_info
