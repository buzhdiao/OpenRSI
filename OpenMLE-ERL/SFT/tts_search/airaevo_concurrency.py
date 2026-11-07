# 启用延迟注解求值，允许类型注解使用新语法且不影响运行时
from __future__ import annotations

# 导入 base64，用于鉴权密钥的编解码
import base64
# 导入 math，用于并发数均摊时的向上取整
import math
# 导入 os，用于读取共享并发相关的环境变量
import os
# 导入 time，用于测量排队等待时间
import time
# 导入上下文管理器装饰器，封装信号量的获取/释放
from contextlib import contextmanager
# 导入多进程信号量，实现跨进程并发限制
from multiprocessing import Semaphore
# 导入 SyncManager，通过网络把信号量共享给多个客户端进程
from multiprocessing.managers import SyncManager
# 导入加密安全的随机字节生成，用于生成鉴权密钥
from secrets import token_bytes
# 导入类型注解工具
from typing import Any, Iterator


# 环境变量名：是否启用共享并发控制
ENV_ENABLED = "AIRAEVO_SHARED_CONCURRENCY"
# 环境变量名：共享信号量服务器主机地址
ENV_HOST = "AIRAEVO_SHARED_CONCURRENCY_HOST"
# 环境变量名：共享信号量服务器端口
ENV_PORT = "AIRAEVO_SHARED_CONCURRENCY_PORT"
# 环境变量名：base64 编码的鉴权密钥
ENV_AUTHKEY = "AIRAEVO_SHARED_CONCURRENCY_AUTHKEY"


# 客户端信号量管理器：连接服务端以获取共享信号量代理
class _ClientSemaphoreManager(SyncManager):
    pass


# 注册可远程调用的方法：获取 LLM 调用并发信号量
_ClientSemaphoreManager.register("get_llm_semaphore")
# 注册可远程调用的方法：获取沙箱执行并发信号量
_ClientSemaphoreManager.register("get_sandbox_semaphore")


# 共享并发控制服务端：集中管理 LLM 调用与沙箱执行的全局并发上限，
# 供多进程/多任务通过信号量排队，避免总量超限
class SharedConcurrencyServer:
    def __init__(self, *, llm_concurrency: int, sandbox_concurrency: int) -> None:
        # 创建 LLM 并发信号量（最多允许 llm_concurrency 个并行调用）
        self._llm_semaphore = Semaphore(llm_concurrency)
        # 创建沙箱并发信号量（最多允许 sandbox_concurrency 个并行执行）
        self._sandbox_semaphore = Semaphore(sandbox_concurrency)
        # 生成 16 字节随机鉴权密钥，防止未授权进程连接
        authkey = token_bytes(16)

        # 服务端信号量管理器类定义
        class _ServerSemaphoreManager(SyncManager):
            pass

        # 注册远程方法：返回 LLM 信号量
        _ServerSemaphoreManager.register(
            "get_llm_semaphore",
            callable=self._get_llm_semaphore,
        )
        # 注册远程方法：返回沙箱信号量
        _ServerSemaphoreManager.register(
            "get_sandbox_semaphore",
            callable=self._get_sandbox_semaphore,
        )

        # 绑定本机随机端口并设置鉴权密钥
        self._manager = _ServerSemaphoreManager(
            address=("127.0.0.1", 0),
            authkey=authkey,
        )
        # 启动管理器服务进程
        self._manager.start()
        # 取出实际监听的主机与端口
        host, port = self._manager.address
        # 生成需要注入子进程的环境变量（启用开关/地址/端口/密钥）
        self.env = {
            ENV_ENABLED: "1",
            ENV_HOST: str(host),
            ENV_PORT: str(port),
            ENV_AUTHKEY: base64.urlsafe_b64encode(authkey).decode("ascii"),
        }

    # 返回 LLM 并发信号量（供远程调用获取）
    def _get_llm_semaphore(self) -> Semaphore:
        return self._llm_semaphore

    # 返回沙箱并发信号量（供远程调用获取）
    def _get_sandbox_semaphore(self) -> Semaphore:
        return self._sandbox_semaphore

    # 关闭管理器及其服务进程
    def shutdown(self) -> None:
        self._manager.shutdown()


# 客户端管理器单例缓存
_client_manager: _ClientSemaphoreManager | None = None
# LLM 信号量代理单例缓存
_llm_semaphore_proxy: Any | None = None
# 沙箱信号量代理单例缓存
_sandbox_semaphore_proxy: Any | None = None


# 计算每个 epoch 的任务并发数：优先使用显式配置，否则按总并发与 epoch 数均摊
def resolve_task_concurrency_per_epoch(
    *,
    configured_task_concurrency: int | None,
    llm_concurrency: int,
    sandbox_concurrency: int,
    num_epochs: int,
) -> int:
    # 显式配置了任务并发数则直接使用
    if configured_task_concurrency is not None:
        return int(configured_task_concurrency)

    # 以 LLM 与沙箱并发上限中的较大者作为目标总并发
    target_total_concurrency = max(int(llm_concurrency), int(sandbox_concurrency))
    # 均摊到各 epoch 并向上取整，至少为 1
    return max(1, math.ceil(target_total_concurrency / int(num_epochs)))


# 建立并缓存客户端管理器连接；未启用共享并发时返回 None
def _connect_client_manager() -> _ClientSemaphoreManager | None:
    # 使用模块级单例
    global _client_manager
    # 环境变量未开启说明不在共享并发模式下
    if os.getenv(ENV_ENABLED) != "1":
        return None

    # 首次调用时解析鉴权密钥并建立连接
    if _client_manager is None:
        # 解码 base64 鉴权密钥
        authkey = base64.urlsafe_b64decode(os.environ[ENV_AUTHKEY].encode("ascii"))
        # 用环境变量中的地址与端口创建管理器
        _client_manager = _ClientSemaphoreManager(
            address=(os.environ[ENV_HOST], int(os.environ[ENV_PORT])),
            authkey=authkey,
        )
        # 连接到共享并发服务端
        _client_manager.connect()
    # 返回已连接的管理器
    return _client_manager


# 获取 LLM 信号量的远程代理（带缓存）；不可用时返回 None
def _get_llm_semaphore_proxy() -> Any | None:
    # 使用模块级单例
    global _llm_semaphore_proxy
    # 先确保客户端管理器已连接
    manager = _connect_client_manager()
    if manager is None:
        # 非共享并发模式返回 None
        return None
    # 首次调用时向服务端索取信号量代理
    if _llm_semaphore_proxy is None:
        _llm_semaphore_proxy = manager.get_llm_semaphore()
    # 返回缓存的代理
    return _llm_semaphore_proxy


# 获取沙箱信号量的远程代理（带缓存）；不可用时返回 None
def _get_sandbox_semaphore_proxy() -> Any | None:
    # 使用模块级单例
    global _sandbox_semaphore_proxy
    # 先确保客户端管理器已连接
    manager = _connect_client_manager()
    if manager is None:
        # 非共享并发模式返回 None
        return None
    # 首次调用时向服务端索取信号量代理
    if _sandbox_semaphore_proxy is None:
        _sandbox_semaphore_proxy = manager.get_sandbox_semaphore()
    # 返回缓存的代理
    return _sandbox_semaphore_proxy


# 占用一个 LLM 调用并发槽位的上下文管理器；yield 出排队等待秒数
@contextmanager
def acquire_llm_slot() -> Iterator[float]:
    # 获取共享 LLM 信号量代理
    semaphore = _get_llm_semaphore_proxy()
    if semaphore is None:
        # 未启用共享并发时不限流，直接通过且排队时间为 0
        yield 0.0
        return

    # 记录开始等待的时间
    wait_start = time.perf_counter()
    # 阻塞获取一个并发槽位
    semaphore.acquire()
    # 计算排队耗时（可作为指标反映并发拥堵程度）
    queue_time = time.perf_counter() - wait_start
    try:
        # 持有槽位执行被保护的代码块
        yield queue_time
    finally:
        # 无论成功失败都释放槽位
        semaphore.release()


# 占用一个沙箱执行并发槽位的上下文管理器；yield 出排队等待秒数
@contextmanager
def acquire_sandbox_slot() -> Iterator[float]:
    # 获取共享沙箱信号量代理
    semaphore = _get_sandbox_semaphore_proxy()
    if semaphore is None:
        # 未启用共享并发时不限流，直接通过且排队时间为 0
        yield 0.0
        return

    # 记录开始等待的时间
    wait_start = time.perf_counter()
    # 阻塞获取一个并发槽位
    semaphore.acquire()
    # 计算排队耗时
    queue_time = time.perf_counter() - wait_start
    try:
        # 持有槽位执行被保护的代码块
        yield queue_time
    finally:
        # 无论成功失败都释放槽位
        semaphore.release()
