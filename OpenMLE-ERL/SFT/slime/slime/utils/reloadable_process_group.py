# 导入 logging：模块级日志
import logging
# 导入 os：获取进程 pid，用于区分不同进程的组注册表
import os
# 导入 contextmanager：包装通信调用实现显存自检
from contextlib import contextmanager

# 导入 torch：分布式进程组能力的基础
import torch
# 导入 torch.distributed：进程组创建与集合通信接口
import torch.distributed as dist

# 导入显存工具：通信前自检显存、出错时打印显存状态
from slime.utils.memory_utils import available_memory, clear_memory, print_memory

# 创建本模块专属 logger
logger = logging.getLogger(__name__)

# 全局注册表：pid -> (原始 new_group 函数)，支持在权重重载后重建通信组
old_new_group_dict = {}


def monkey_patch_torch_dist():
    # 记录当前进程 pid
    pid = os.getpid()
    # 本进程已打过补丁则直接返回（保证幂等）
    if pid in old_new_group_dict:
        # 校验注册的函数未被外部改动
        assert dist.old_new_group == old_new_group_dict[pid]
        return

    # 日志提示开始给 torch.distributed 打猴子补丁
    logger.info("Applying monkey patch to torch.distributed")

    # 保存原始的 new_group 函数
    old_new_group = dist.new_group
    # 登记到注册表（供 reload 时重建组使用）
    old_new_group_dict[pid] = old_new_group
    # 暴露为 dist.old_new_group，方便其他模块访问原始实现
    dist.old_new_group = old_new_group

    # 替换版 new_group：创建组后包一层可重载代理
    def new_group(*args, **kwargs):
        # 先按原逻辑创建真实的进程组
        group = old_new_group(*args, **kwargs)
        # skip none nccl group.
        # gloo 后端（CPU 组）不需要可重载管理，直接返回
        if len(args) >= 3 and args[2] == "gloo" or "backend" in kwargs and kwargs["backend"] == "gloo":
            return group

        # Get ranks from arguments
        # 从位置参数或关键字参数中提取组成员 rank 列表
        if len(args) >= 1 and args[0] is not None:
            ranks = args[0]
        elif "ranks" in kwargs and kwargs["ranks"] is not None:
            ranks = kwargs["ranks"]
        else:
            # If no ranks specified, use all ranks in world
            # 未指定 ranks 时默认全体 rank
            ranks = list(range(dist.get_world_size()))

        # 单 rank 组无需跨机通信，不需要可重载代理
        if len(ranks) == 1:
            return group

        # 用可重载代理包装真实组后返回
        group = ReloadableProcessGroup(group, ranks)
        return group

    # 用替换版覆盖 dist.new_group
    dist.new_group = new_group

    def get_new_query_function(func):
        """Wrap query functions (get_rank, get_world_size, etc.) without memory check."""

        # 查询类函数包装：解包代理组后透传，不做显存检查
        def new_function(*args, **kwargs):
            # 位置参数中的代理组替换为内部真实组
            args = tuple([arg.group if isinstance(arg, ReloadableProcessGroup) else arg for arg in args])
            # 关键字参数同样替换
            kwargs = {k: (v.group if isinstance(v, ReloadableProcessGroup) else v) for k, v in kwargs.items()}
            # 调用原函数
            return func(*args, **kwargs)

        # 返回包装后的函数
        return new_function

    def get_new_comm_function(func):
        """Wrap communication functions with memory check."""

        # 通信类函数包装：解包代理组并附带显存自检
        def new_function(*args, **kwargs):
            # 解包代理组为真实组
            args = tuple([arg.group if isinstance(arg, ReloadableProcessGroup) else arg for arg in args])
            kwargs = {k: (v.group if isinstance(v, ReloadableProcessGroup) else v) for k, v in kwargs.items()}
            # 在显存守护上下文中执行通信
            with _wrap_low_level_call():
                return func(*args, **kwargs)

        # 返回包装后的函数
        return new_function

    # 以下把所有查询类 API 换成解包版（不检查显存）
    dist.get_rank = get_new_query_function(dist.get_rank)
    dist.get_world_size = get_new_query_function(dist.get_world_size)
    dist.get_backend = get_new_query_function(dist.get_backend)
    dist.get_global_rank = get_new_query_function(dist.get_global_rank)
    dist.get_group_rank = get_new_query_function(dist.get_group_rank)
    dist.get_process_group_ranks = get_new_query_function(dist.get_process_group_ranks)

    # 把所有集合通信 API 换成带显存自检的包装版
    dist.all_reduce = get_new_comm_function(dist.all_reduce)
    dist.all_gather = get_new_comm_function(dist.all_gather)
    dist.all_gather_into_tensor = get_new_comm_function(dist.all_gather_into_tensor)
    dist.all_gather_object = get_new_comm_function(dist.all_gather_object)
    dist.all_to_all = get_new_comm_function(dist.all_to_all)
    dist.all_to_all_single = get_new_comm_function(dist.all_to_all_single)
    dist.broadcast = get_new_comm_function(dist.broadcast)
    dist.reduce = get_new_comm_function(dist.reduce)
    dist.reduce_scatter = get_new_comm_function(dist.reduce_scatter)
    dist.reduce_scatter_tensor = get_new_comm_function(dist.reduce_scatter_tensor)
    dist.scatter = get_new_comm_function(dist.scatter)
    dist.gather = get_new_comm_function(dist.gather)
    dist.barrier = get_new_comm_function(dist.barrier)
    dist.send = get_new_comm_function(dist.send)
    dist.recv = get_new_comm_function(dist.recv)
    dist._coalescing_manager = get_new_comm_function(dist._coalescing_manager)

    # p2p
    # 点对点异步通信：保存原始 isend/irecv 供 P2POp 参数转换时使用
    old_isend = dist.isend
    old_irecv = dist.irecv

    # isend/irecv 也走显存自检包装
    dist.isend = get_new_comm_function(dist.isend)
    dist.irecv = get_new_comm_function(dist.irecv)

    def get_new_p2pop_function(func):
        # P2POp 构造相关包装：转换其中的特殊参数
        def new_function(*args, **kwargs):
            # 嵌套转换函数
            def convert(arg):
                if isinstance(arg, ReloadableProcessGroup):
                    # 代理组替换为内部真实组
                    return arg.group
                elif arg == dist.isend:
                    # dist.isend 已被补丁替换，还原为原始函数
                    arg = old_isend
                elif arg == dist.irecv:
                    # 同上还原 irecv
                    arg = old_irecv
                return arg

            # 逐个转换位置参数（生成器形式）
            args = (convert(arg) for arg in args)
            # 转换关键字参数
            kwargs = {k: convert(v) for k, v in kwargs.items()}
            # 调用原构造逻辑
            return func(*args, **kwargs)

        # 返回包装后的构造函数
        return new_function

    # 替换 P2POp 的构造入口，使 batch p2p 操作也能正确解包
    dist.P2POp.__new__ = get_new_p2pop_function(dist.P2POp.__new__)
    dist.P2POp.__init__ = get_new_p2pop_function(dist.P2POp.__init__)


class ReloadableProcessGroup(torch.distributed.ProcessGroup):
    # 进程组代理：内部真实 NCCL 组被销毁后可按 ranks 信息重建
    # 类级注册表：pid -> 本进程创建的所有代理组
    GROUPS = {}

    def __init__(self, group, ranks):
        # 用真实组的 rank/size 初始化基类
        super().__init__(
            rank=dist.get_rank(group),
            size=dist.get_world_size(group),
        )
        # 持有内部真实进程组
        self.group = group
        # 记录组的 rank 列表（重建时需要）
        self.group_info = {
            "ranks": ranks,
        }
        # 取当前进程 pid
        pid = os.getpid()
        # 按 pid 登记到类级注册表
        if pid not in ReloadableProcessGroup.GROUPS:
            ReloadableProcessGroup.GROUPS[pid] = []
        ReloadableProcessGroup.GROUPS[pid].append(self)

    def __getattr__(self, name):
        # 未显式实现的属性一律转发给内部真实组
        return getattr(self.group, name)

    @staticmethod
    def destroy_process_groups():
        # 销毁本进程所有代理组内部的 NCCL 通信器（权重重载前调用）
        pid = os.getpid()
        # 遍历本进程的全部代理组
        for reloadable_group in ReloadableProcessGroup.GROUPS.get(pid, []):
            # 内部组已为空则跳过
            if reloadable_group.group is None:
                continue
            try:
                # 调用底层销毁接口释放 NCCL 通信器
                dist.destroy_process_group(reloadable_group.group)
            except ValueError as e:
                # 组已被销毁/失效时仅告警，不中断流程
                logger.warning(
                    f"Process group already invalid/destroyed; skipping cleanup. Exception: {e}",
                    exc_info=True,
                )

            # 从实例上删除组引用
            del reloadable_group.group
            # 置空标记内部组已失效
            reloadable_group.group = None

    @staticmethod
    def reload_process_groups():
        # 按保存的 ranks 重建所有已失效的内部组（权重重载后调用）
        pid = os.getpid()
        # 取本进程全部代理组
        reloadable_groups = ReloadableProcessGroup.GROUPS.get(pid, [])
        # 日志提示重建数量
        logger.info(f"Reloading {len(reloadable_groups)} process groups in pid {pid}")
        # 取原始 new_group 函数
        old_new_group = old_new_group_dict.get(pid)
        for reloadable_group in reloadable_groups:
            # 内部组仍存活则无需重建
            if reloadable_group.group is not None:
                continue
            # 用原始接口按 NCCL 后端重建组
            group = old_new_group(ranks=reloadable_group.group_info["ranks"], backend="nccl")
            # 回填到代理对象
            reloadable_group.group = group

    def rank(self) -> int:
        # 查询本组内 rank：转发内部组
        return self.group.rank()

    def size(self) -> int:
        # 查询组大小：转发内部组
        return self.group.size()

    def name(self) -> str:
        # 查询组名：转发内部组
        return self.group.name()

    def shutdown(self) -> None:
        # 优雅关闭：内部组存在时转发
        if self.group is not None:
            self.group.shutdown()

    def abort(self) -> None:
        # 紧急中止：内部组存在时转发
        if self.group is not None:
            self.group.abort()

    def _fwd(self, method, *args, **kwargs):
        # 通用通信转发：先检查内部组，再带显存自检调用
        inner = self.group
        if inner is None:
            # 内部组尚未重建就发起通信，直接报错
            raise RuntimeError("ReloadableProcessGroup: inner PG is None, call reload() first.")
        with _wrap_low_level_call():
            # 反射调用内部组的对应方法
            return getattr(inner, method)(*args, **kwargs)

    def _fwd_query(self, method, *args, **kwargs):
        """Forward non-communication calls without memory check."""
        # 非通信转发：只检查内部组，不做显存自检
        inner = self.group
        if inner is None:
            # 内部组未就绪时报错
            raise RuntimeError("ReloadableProcessGroup: inner PG is None, call reload() first.")
        return getattr(inner, method)(*args, **kwargs)

    # 以下是一组通信方法覆盖：全部委托 _fwd 转发（通信类，带显存自检）
    def barrier(self, *a, **kw):
        return self._fwd("barrier", *a, **kw)

    def broadcast(self, *a, **kw):
        return self._fwd("broadcast", *a, **kw)

    def allreduce(self, *a, **kw):
        return self._fwd("allreduce", *a, **kw)

    def allreduce_coalesced(self, *a, **kw):
        return self._fwd("allreduce_coalesced", *a, **kw)

    def reduce(self, *a, **kw):
        return self._fwd("reduce", *a, **kw)

    def allgather(self, *a, **kw):
        return self._fwd("allgather", *a, **kw)

    def _allgather_base(self, *a, **kw):
        return self._fwd("_allgather_base", *a, **kw)

    def allgather_coalesced(self, *a, **kw):
        return self._fwd("allgather_coalesced", *a, **kw)

    def allgather_into_tensor_coalesced(self, *a, **kw):
        return self._fwd("allgather_into_tensor_coalesced", *a, **kw)

    def gather(self, *a, **kw):
        return self._fwd("gather", *a, **kw)

    def scatter(self, *a, **kw):
        return self._fwd("scatter", *a, **kw)

    def reduce_scatter(self, *a, **kw):
        return self._fwd("reduce_scatter", *a, **kw)

    def _reduce_scatter_base(self, *a, **kw):
        return self._fwd("_reduce_scatter_base", *a, **kw)

    def reduce_scatter_tensor_coalesced(self, *a, **kw):
        return self._fwd("reduce_scatter_tensor_coalesced", *a, **kw)

    def alltoall_base(self, *a, **kw):
        return self._fwd("alltoall_base", *a, **kw)

    def alltoall(self, *a, **kw):
        return self._fwd("alltoall", *a, **kw)

    def send(self, *a, **kw):
        return self._fwd("send", *a, **kw)

    def recv(self, *a, **kw):
        return self._fwd("recv", *a, **kw)

    def recv_anysource(self, *a, **kw):
        return self._fwd("recv_anysource", *a, **kw)

    # coalescing 相关：开始端为查询类，结束端属于通信收尾
    def _start_coalescing(self, *a, **kw):
        return self._fwd_query("_start_coalescing", *a, **kw)

    def _end_coalescing(self, *a, **kw):
        return self._fwd("_end_coalescing", *a, **kw)

    # 后端查询类：委托 _fwd_query（无显存检查）
    def _get_backend_name(self):
        return self._fwd_query("_get_backend_name")

    def _get_backend(self, *a, **kw):
        return self._fwd_query("_get_backend", *a, **kw)

    def _set_default_backend(self, *a, **kw):
        return self._fwd_query("_set_default_backend", *a, **kw)

    @property
    def bound_device_id(self):
        # 绑定设备属性的读取：直接转发内部组
        return self.group.bound_device_id

    @bound_device_id.setter
    def bound_device_id(self, dev):
        # 绑定设备属性的写入：直接转发内部组
        self.group.bound_device_id = dev


def destroy_process_groups():
    """Destroy all reloadable process groups."""
    # 模块级便捷入口：销毁本进程全部可重载组
    ReloadableProcessGroup.destroy_process_groups()


def reload_process_groups():
    """Reload all reloadable process groups."""
    # 模块级便捷入口：重建本进程全部可重载组
    ReloadableProcessGroup.reload_process_groups()


@contextmanager
def _wrap_low_level_call():
    try:
        # 通信前查询当前显存水位
        mem_info = available_memory()
        # 空闲显存不足 5GB 时先做一轮清理，降低通信中 OOM 概率
        if mem_info["free_GB"] < 5:
            clear_memory()
        # 放行原始通信调用
        yield
    except Exception as e:
        # 通信失败时打印显存状态辅助排障
        mem_info = print_memory("after torch distributed error")
        # 把显存信息附加到异常注解上，随异常一起上报
        e.add_note(f"{mem_info=}")
        # 继续抛出原异常
        raise
