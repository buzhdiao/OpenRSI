# =====================================================================
# 中文注释导读（OpenRSI / OpenMLE-Evo 学习文档）：
# 本模块负责 Evo-Max 异步搜索（async_steady_state，见论文 OpenMLE-Evo-Max 一节）
# 的资源解析：执行模式、异步 worker 数量、沙箱端点（OpenMLE Sandbox 多控制器路由）
# 以及 worker 规格（WorkerSpec）的构造。evo.py 的 search_async_steady_state() 在
# 启动前调用这里的函数，把"多少 worker、谁用哪块 GPU、请求发往哪个沙箱"确定下来。
# 论文术语：Evo-Max 异步搜索 = benchmark 无关经验先验 + 多 worker 异步稳态搜索。
# =====================================================================
from __future__ import annotations

import os  # 读取环境变量（沙箱 API Key）
import subprocess  # 调用 nvidia-smi 探测本机 GPU
from dataclasses import dataclass  # WorkerSpec 不可变数据类
from typing import Any

# 支持的执行模式集合：
#   generation           —— 标准逐代进化搜索（串行、单 worker）
#   async_steady_state   —— Evo-Max 异步稳态搜索（多 worker 流水线）
EXECUTION_MODES = frozenset({"generation", "async_steady_state"})
# 支持的沙箱分配策略集合：目前只实现 round_robin（轮转）
SANDBOX_ASSIGNMENTS = frozenset({"round_robin"})


def resolve_execution_mode(value: str | None) -> str:
    # 把配置里的执行模式字符串规整为合法值；非法值直接报错（fail fast）
    mode = str(value or "generation").strip().lower()  # 缺省为标准 generation 模式
    if mode not in EXECUTION_MODES:
        supported = ", ".join(sorted(EXECUTION_MODES))
        raise ValueError(
            f"Unsupported execution mode: {mode}. Supported modes: {supported}"
        )
    return mode


def detect_gpu_indices() -> list[int]:
    # 探测本机可见 GPU 的索引列表（Evo-Max 把 worker 绑到 GPU 上做归因）
    try:
        # 调用 nvidia-smi 查询 GPU index，每行一个数字
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            check=False,  # 失败不抛异常，走返回码判断
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        # 没有 nvidia-smi（无 GPU 环境）：返回空列表
        return []
    if result.returncode != 0:
        # 命令执行失败同样视为无 GPU
        return []
    # 逐行解析出整数 GPU 编号
    return [
        int(line.strip())
        for line in result.stdout.splitlines()
        if line.strip().isdigit()
    ]


def resolve_async_worker_count(
    value: int | str | None, *, gpu_indices: list[int] | None = None
) -> int:
    # 解析异步 worker 数量：
    #   None   -> 1（默认单 worker，等价于退化为准串行）
    #   "auto" -> 本机 GPU 数量（每块 GPU 一个 worker）
    #   整数   -> 直接使用（至少为 1）
    if value is None:
        return 1
    if isinstance(value, str) and value.lower() == "auto":
        resolved_gpu_indices = (
            detect_gpu_indices() if gpu_indices is None else gpu_indices
        )
        # 至少 1 个 worker：无 GPU 环境也要能跑
        return max(1, len(resolved_gpu_indices))
    return max(1, int(value))


def resolve_async_sandbox_urls(
    value: Any, *, fallback_url: str | None = None
) -> list[str]:
    # 解析异步沙箱端点列表（支持三种形态）：
    #   字符串逗号分隔 / Python 列表 / None 回退到任务默认沙箱
    if value is None or value == "":
        # 未配置：使用任务侧 base_url 作为唯一沙箱
        return [str(fallback_url).strip()] if str(fallback_url or "").strip() else []
    if isinstance(value, str):
        # "url1,url2,url3" 形式：切分并去掉空白
        return [item.strip() for item in value.split(",") if item.strip()]
    urls = [str(item).strip() for item in list(value) if str(item).strip()]
    if urls:
        # 列表形式：原样返回
        return urls
    # 空列表：回退默认沙箱
    return [str(fallback_url).strip()] if str(fallback_url or "").strip() else []


@dataclass(frozen=True)
class WorkerSpec:
    """单个异步 worker 的规格快照（不可变）。

    worker_id 用于日志归因；gpu_index 记录该 worker 使用的 GPU（可为 None）；
    sandbox_url 是该 worker 专属的 OpenMLE Sandbox 端点（多控制器路由）。
    """

    worker_id: int  # worker 编号（0 起）
    gpu_index: int | None  # 绑定的 GPU 索引（无 GPU 环境为 None）
    sandbox_url: str  # 该 worker 发送执行请求的沙箱地址


class RoundRobinSandboxPool:
    """沙箱地址轮转池：把 N 个 worker 均匀摊到 M 个沙箱端点上。"""

    def __init__(self, urls: list[str]) -> None:
        self.urls = list(urls)  # 沙箱端点列表
        self._index = 0  # 下一个要发出的位置游标

    def next_url(self) -> str:
        # 取模轮转：worker0->url0, worker1->url1, ... workerM->url0 ...
        url = self.urls[self._index % len(self.urls)]
        self._index += 1
        return url


def build_worker_specs(
    *,
    worker_count: int,
    sandbox_urls: list[str],
    assignment: str = "round_robin",
) -> list[WorkerSpec]:
    # 构造全部 worker 规格：GPU 按 worker_id 取模绑定，沙箱按轮转分配
    if assignment not in SANDBOX_ASSIGNMENTS:
        supported = ", ".join(sorted(SANDBOX_ASSIGNMENTS))
        raise ValueError(
            f"Unsupported sandbox assignment: {assignment}. "
            f"Supported assignments: {supported}"
        )
    if not sandbox_urls:
        # 异步模式必须至少有一个沙箱端点，否则 worker 无处执行代码
        raise ValueError("At least one async sandbox URL is required")
    gpu_indices = detect_gpu_indices()  # 探测本机 GPU 列表
    pool = RoundRobinSandboxPool(sandbox_urls)
    return [
        WorkerSpec(
            worker_id=worker_id,
            gpu_index=gpu_indices[worker_id % len(gpu_indices)]
            if gpu_indices
            else None,  # 有 GPU 则轮转绑定，无 GPU 记 None
            sandbox_url=pool.next_url(),  # 沙箱轮转分配
        )
        for worker_id in range(worker_count)
    ]


async def fetch_router_idle_worker_count(
    router_url: str,
    *,
    api_key: str | None = None,
    timeout_seconds: float = 5.0,
) -> int | None:
    # 查询 OpenMLE Sandbox 路由器当前空闲 worker 数（观测/日志用，不参与调度决策）
    import httpx  # 延迟导入：仅在真正需要联网时加载

    # 鉴权：优先显式传入的 key，其次环境变量 SANDBOX_GPU_API_KEY
    headers = {"X-API-Key": api_key or os.environ.get("SANDBOX_GPU_API_KEY", "")}
    try:
        async with httpx.AsyncClient(
            base_url=router_url, timeout=timeout_seconds, trust_env=False  # 忽略系统代理，直连沙箱
        ) as client:
            response = await client.get("/api/v1/workers/status", headers=headers)
        if response.status_code != 200:
            return None  # 非 200 视为不可观测，静默返回 None
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        # 网络错误或 JSON 解析失败：同样静默降级
        return None

    workers = payload.get("workers_show") if isinstance(payload, dict) else None
    if not isinstance(workers, list):
        return None
    # 统计状态为 "idle" 的 worker 数量
    return sum(
        1 for worker in workers if str(worker.get("status", "")).lower() == "idle"
    )
