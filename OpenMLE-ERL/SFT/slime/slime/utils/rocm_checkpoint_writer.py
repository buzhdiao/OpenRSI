# 导入 torch：检测运行平台是否为 HIP/ROCm
import torch
# 导入 Megatron 的异步文件系统 checkpoint 写入器作为基类
from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync


class ROCmFileSystemWriterAsync(FileSystemWriterAsync):
    """
    FileSystemWriterAsync wrapper for ROCm compatibility.

    On ROCm/HIP, using non_blocking=True causes tensors to be stored in pinned memory,
    which triggers segmentation faults when forking subprocesses afterward.
    """

    @staticmethod
    def preload_tensors(*args, **kwargs):
        # Change argument non_blocking to False on HIP platform
        # The tensors will be stored in pinned memory if non_blocking=True
        # Currently on the ROCm platform, forking a subprocess afterward
        # with pinned_memory=True will trigger segmentation fault
        # HIP/ROCm 平台兼容性处理：强制关闭非阻塞拷贝
        if torch.version.hip:
            # 日志提示已检测到 HIP 平台并修改参数
            print("HIP/ROCm detected: setting non_blocking=False in preload_tensors")
            if "non_blocking" in kwargs:
                # 关键字参数形式：直接改为 False，避免张量进入锁页内存
                kwargs["non_blocking"] = False
            elif len(args) > 1 and isinstance(args[-1], bool):
                # non_blocking is typically the last argument
                # 位置参数形式：non_blocking 通常是末位的布尔参数，替换为 False
                args = args[:-1] + (False,)

        # 委托原始实现完成张量预加载
        return FileSystemWriterAsync.preload_tensors(*args, **kwargs)
