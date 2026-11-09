# 中文说明:airaevo 的"跨进程共享并发闸门"——用 multiprocessing.SyncManager
# 把 LLM 与沙箱两个信号量做成可被同机多个任务进程共享的配额。
# 在系统中的位置:aira-evo 多进程 runner 的并发骨架:父进程起一个
# SharedConcurrencyServer(随机端口+随机 authkey),子进程通过环境变量发现并
# 连接后,用 acquire_*_slot(_async) 上下文管理器取号;未启用共享时这些函数
# 直接放行(yield 0.0),退化为各进程自治限流。
# 关键设计:
# - 信号量在服务进程内创建并由 initializer 注入,客户端只拿代理;
# - 异步侧用 to_thread 包同步 acquire + 短轮询,避免阻塞事件循环;
# - 每次取号返回排队耗时,便于并发观测与日志。

# 延迟注解:类型标注运行期不求值
from __future__ import annotations

# asyncio:异步轮询与 to_thread
import asyncio
# base64:authkey 的 URL 安全编码(放环境变量)
import base64
# math:并发数向上取整
import math
# multiprocessing:默认上下文与信号量创建
import multiprocessing
# os:读取共享并发的环境变量
import os
# threading:客户端连接与代理获取的进程内互斥
import threading
# time:排队耗时统计
import time
# contextmanager:同步上下文管理器
from contextlib import asynccontextmanager
from contextlib import contextmanager
# BaseContext:可注入的 mp 上下文(便于测试用 spawn 等)
from multiprocessing.context import BaseContext
# SyncManager:跨进程共享对象的服务基类
from multiprocessing.managers import SyncManager
# token_bytes:生成随机 authkey
from secrets import token_bytes
# 类型注解
from typing import Any, AsyncIterator, Iterator

# 环境变量名:是否启用共享并发/服务主机/端口/认证密钥
ENV_ENABLED = "AIRAEVO_SHARED_CONCURRENCY"
ENV_HOST = "AIRAEVO_SHARED_CONCURRENCY_HOST"
ENV_PORT = "AIRAEVO_SHARED_CONCURRENCY_PORT"
ENV_AUTHKEY = "AIRAEVO_SHARED_CONCURRENCY_AUTHKEY"
# 异步侧轮询信号量的间隔(秒)
ASYNC_SEMAPHORE_POLL_INTERVAL_SECONDS = 0.25


class _ClientSemaphoreManager(SyncManager):
    """客户端侧管理器:只声明远端方法,由 connect() 后代理调用。"""
    pass


class _ServerSemaphoreManager(SyncManager):
    """服务端侧管理器:注册远端方法的真实实现。"""
    pass


# 服务进程内持有的两个信号量(initializer 注入)
_server_llm_semaphore: Any | None = None
_server_sandbox_semaphore: Any | None = None


def _initialize_server_semaphores(llm_semaphore: Any, sandbox_semaphore: Any) -> None:
    """管理器服务进程的 initializer:把两个信号量装进服务端全局变量。"""
    global _server_llm_semaphore, _server_sandbox_semaphore
    _server_llm_semaphore = llm_semaphore
    _server_sandbox_semaphore = sandbox_semaphore


def _get_server_llm_semaphore() -> Any:
    """服务端取 LLM 信号量(未初始化即编程错误)。"""
    if _server_llm_semaphore is None:
        raise RuntimeError("Shared LLM semaphore is not initialized")
    return _server_llm_semaphore


def _get_server_sandbox_semaphore() -> Any:
    """服务端取沙箱信号量(未初始化即编程错误)。"""
    if _server_sandbox_semaphore is None:
        raise RuntimeError("Shared sandbox semaphore is not initialized")
    return _server_sandbox_semaphore


# 客户端注册远端方法(仅签名,真实实现来自服务端)
_ClientSemaphoreManager.register("get_llm_semaphore")
_ClientSemaphoreManager.register("get_sandbox_semaphore")
# 服务端注册远端方法与实现
_ServerSemaphoreManager.register(
    "get_llm_semaphore",
    callable=_get_server_llm_semaphore,
)
_ServerSemaphoreManager.register(
    "get_sandbox_semaphore",
    callable=_get_server_sandbox_semaphore,
)


class SharedConcurrencyServer:
    """共享并发服务:创建两个跨进程信号量,并把连接信息导出为子进程环境变量。"""

    def __init__(
        self,
        *,
        llm_concurrency: int,
        sandbox_concurrency: int,
        mp_context: BaseContext | None = None,
    ) -> None:
        # 选择 mp 上下文(默认跟随当前进程)
        context = mp_context or multiprocessing.get_context()
        # 在父进程创建两个信号量(计数=配置并发)
        self._llm_semaphore = context.Semaphore(llm_concurrency)
        self._sandbox_semaphore = context.Semaphore(sandbox_concurrency)
        # 随机 16 字节 authkey(只暴露给本批子进程)
        authkey = token_bytes(16)

        # 本机随机端口启动管理器服务
        self._manager = _ServerSemaphoreManager(
            address=("127.0.0.1", 0),
            authkey=authkey,
            ctx=context,
        )
        # 服务进程启动时注入信号量(信号量真正的宿主)
        self._manager.start(
            initializer=_initialize_server_semaphores,
            initargs=(self._llm_semaphore, self._sandbox_semaphore),
        )
        # 取实际绑定的 host/port
        host, port = self._manager.address
        # 导出给子进程的发现环境变量(启停/地址/密钥)
        self.env = {
            ENV_ENABLED: "1",
            ENV_HOST: str(host),
            ENV_PORT: str(port),
            ENV_AUTHKEY: base64.urlsafe_b64encode(authkey).decode("ascii"),
        }

    def shutdown(self) -> None:
        """关停管理器服务进程(信号量随之销毁)。"""
        self._manager.shutdown()


# 客户端侧的进程级缓存:管理器连接与两个信号量代理(懒创建,只建一次)
_client_manager: _ClientSemaphoreManager | None = None
_llm_semaphore_proxy: Any | None = None
_sandbox_semaphore_proxy: Any | None = None
_client_lock = threading.Lock()


def resolve_task_concurrency_per_epoch(
    *,
    configured_task_concurrency: int | None,
    llm_concurrency: int,
    sandbox_concurrency: int,
    num_epochs: int,
    async_workers: int | str | None = 1,
) -> int:
    """推算每轮(epoch)的任务并发:显式配置优先,否则按总配额/worker 数均摊。"""
    # 显式配置直接生效
    if configured_task_concurrency is not None:
        return int(configured_task_concurrency)

    # 延迟导入避免循环依赖
    from tts_search.airaevo_async_resources import resolve_async_worker_count

    # 目标总并发取 LLM/沙箱配额的较大者
    target_total_concurrency = max(int(llm_concurrency), int(sandbox_concurrency))
    # 任务 worker 总数 = 轮数 x 每轮异步 worker 数
    task_worker_count = int(num_epochs) * resolve_async_worker_count(async_workers)
    # 每个 worker 分到的任务并发(至少 1)
    return max(1, math.ceil(target_total_concurrency / task_worker_count))


def _connect_client_manager() -> _ClientSemaphoreManager | None:
    """按环境变量连接共享服务(未启用返回 None;连接进程内只做一次)。"""
    global _client_manager
    # 未启用共享并发:直接返回
    if os.getenv(ENV_ENABLED) != "1":
        return None

    with _client_lock:
        if _client_manager is None:
            # 解码 authkey 并连接服务端
            authkey = base64.urlsafe_b64decode(os.environ[ENV_AUTHKEY].encode("ascii"))
            manager = _ClientSemaphoreManager(
                address=(os.environ[ENV_HOST], int(os.environ[ENV_PORT])),
                authkey=authkey,
            )
            manager.connect()
            _client_manager = manager
    return _client_manager


def _get_llm_semaphore_proxy() -> Any | None:
    """取 LLM 信号量代理(未启用共享返回 None;代理进程内缓存)。"""
    global _llm_semaphore_proxy
    manager = _connect_client_manager()
    if manager is None:
        return None
    with _client_lock:
        if _llm_semaphore_proxy is None:
            _llm_semaphore_proxy = manager.get_llm_semaphore()
    return _llm_semaphore_proxy


def _get_sandbox_semaphore_proxy() -> Any | None:
    """取沙箱信号量代理(未启用共享返回 None;代理进程内缓存)。"""
    global _sandbox_semaphore_proxy
    manager = _connect_client_manager()
    if manager is None:
        return None
    with _client_lock:
        if _sandbox_semaphore_proxy is None:
            _sandbox_semaphore_proxy = manager.get_sandbox_semaphore()
    return _sandbox_semaphore_proxy


async def _acquire_semaphore_async(semaphore: Any) -> float:
    """异步取号:同步非阻塞 acquire 放线程池 + 短轮询;返回排队耗时。"""
    wait_start = time.perf_counter()
    # 非阻塞尝试拿锁;拿不到就小睡后重试(不阻塞事件循环)
    while not await asyncio.to_thread(semaphore.acquire, False):
        await asyncio.sleep(ASYNC_SEMAPHORE_POLL_INTERVAL_SECONDS)
    return time.perf_counter() - wait_start


@contextmanager
def acquire_llm_slot() -> Iterator[float]:
    """同步取 LLM 配额:未启用直接放行(排队耗时 0);用完在 finally 释放。"""
    semaphore = _get_llm_semaphore_proxy()
    if semaphore is None:
        yield 0.0
        return

    # 记录排队耗时
    wait_start = time.perf_counter()
    semaphore.acquire()
    queue_time = time.perf_counter() - wait_start
    try:
        yield queue_time
    finally:
        semaphore.release()


@contextmanager
def acquire_sandbox_slot() -> Iterator[float]:
    """同步取沙箱配额:未启用直接放行;用完在 finally 释放。"""
    semaphore = _get_sandbox_semaphore_proxy()
    if semaphore is None:
        yield 0.0
        return

    # 记录排队耗时
    wait_start = time.perf_counter()
    semaphore.acquire()
    queue_time = time.perf_counter() - wait_start
    try:
        yield queue_time
    finally:
        semaphore.release()


@asynccontextmanager
async def acquire_llm_slot_async() -> AsyncIterator[float]:
    """异步取 LLM 配额:连接探测与释放都放线程池,避免阻塞事件循环。"""
    semaphore = await asyncio.to_thread(_get_llm_semaphore_proxy)
    if semaphore is None:
        yield 0.0
        return

    # 轮询式取号(带排队耗时)
    queue_time = await _acquire_semaphore_async(semaphore)
    try:
        yield queue_time
    finally:
        await asyncio.to_thread(semaphore.release)


@asynccontextmanager
async def acquire_sandbox_slot_async() -> AsyncIterator[float]:
    """异步取沙箱配额:与 LLM 版对称。"""
    semaphore = await asyncio.to_thread(_get_sandbox_semaphore_proxy)
    if semaphore is None:
        yield 0.0
        return

    # 轮询式取号(带排队耗时)
    queue_time = await _acquire_semaphore_async(semaphore)
    try:
        yield queue_time
    finally:
        await asyncio.to_thread(semaphore.release)
