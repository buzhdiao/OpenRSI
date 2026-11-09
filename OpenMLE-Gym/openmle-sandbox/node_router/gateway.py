# 中文说明:openmle-sandbox 的"节点路由网关"(FastAPI 反向代理)。
# 在系统中的位置:node_router 子系统,位于客户端(node_client)与多个沙箱
# 后端(node_controller)之间;对上层暴露与后端一致的 /api/v1 接口,
# 把作业按策略路由到某个后端,并持久化"作业 -> 后端"映射(SQLite)。
# 关键设计:
# - 路由策略 idle_gpu_first:按各后端的空闲 GPU 数(扣除虚拟负载)择优,平手回退 round_robin;
# - backends.yaml 热加载(按 mtime 检测,带锁防并发重载);
# - 作业路由映射落库,网关重启后仍能找到原后端;查不到时全后端探测并回填;
# - 聚合类接口(workers/status、jobs 列表)并发拉取所有后端再归并。

# 延迟注解:类型标注运行期不求值
from __future__ import annotations

# asyncio:锁与并发 gather(多后端并行请求)
import asyncio
# os:读取网关配置/管理令牌环境变量
import os
# sqlite3:作业路由映射的持久化存储
import sqlite3
# time:monotonic 时钟(worker 状态 TTL)
import time
# dataclass:Backend/GatewayConfig 不可变配置结构
from dataclasses import dataclass
# datetime/timezone:UTC 时间戳(路由记录创建时间)
from datetime import datetime, timezone
# Path:配置/数据库路径
from pathlib import Path
# 类型注解
from typing import Any, Dict, Iterable, List, Optional, Tuple

# httpx:异步 HTTP 客户端(转发与探测)
import httpx
# yaml:解析 backends.yaml
import yaml
# FastAPI 组件:应用/请求头/异常/请求对象
from fastapi import FastAPI, Header, HTTPException, Request
# 响应类型:JSON 响应与字节流响应
from fastapi.responses import JSONResponse, Response


# 脚本所在目录(默认配置与数据库都放这里)
SCRIPT_DIR = Path(__file__).resolve().parent
# 默认后端配置文件
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "backends.yaml"
# 默认路由映射数据库
DEFAULT_DB_PATH = SCRIPT_DIR / "gateway_routes.sqlite3"


@dataclass(frozen=True)
class Backend:
    """单个沙箱后端的静态描述(id/地址/启停/轮询权重)。"""
    # 后端唯一 ID
    id: str
    # 后端服务地址
    base_url: str
    # 是否启用
    enabled: bool = True
    # round_robin 权重(重复入队次数)
    weight: int = 1


@dataclass(frozen=True)
class GatewayConfig:
    """网关整体配置:后端列表 + 路由策略 + 各类超时 + 路由库路径。"""
    # 后端列表
    backends: List[Backend]
    # 主路由策略(默认空闲 GPU 优先)
    strategy: str = "idle_gpu_first"
    # 兜底策略(默认轮询)
    fallback: str = "round_robin"
    # worker 状态缓存 TTL(秒)
    worker_status_ttl_seconds: float = 2.0
    # 转发请求超时(秒)
    request_timeout_seconds: float = 120.0
    # 健康检查超时(秒)
    health_timeout_seconds: float = 5.0
    # 路由映射 SQLite 路径
    route_store_path: Path = DEFAULT_DB_PATH


class RouteStore:
    """作业/上传 -> 后端 的映射存储:SQLite 持久化 + 异步锁串行化写读。"""

    def __init__(self, db_path: Path):
        # 记录库路径并确保父目录存在
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # 异步锁:同一时刻只允许一个协程访问数据库
        self._lock = asyncio.Lock()
        # 初始化表结构
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """新建 SQLite 连接(行工厂返回 Row,便于按列名取值)。"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """建两张映射表(作业/上传),已存在则跳过。"""
        with self._connect() as conn:
            # WAL 模式:并发读写更稳
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_routes (
                    job_id TEXT PRIMARY KEY,
                    backend_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS upload_routes (
                    upload_id TEXT PRIMARY KEY,
                    backend_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _now() -> str:
        """当前 UTC 时间 ISO 字符串。"""
        return datetime.now(timezone.utc).isoformat()

    async def set_job_backend(self, job_id: str, backend_id: str) -> None:
        """写入/更新作业 -> 后端映射(冲突时覆盖)。"""
        async with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO job_routes(job_id, backend_id, created_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(job_id) DO UPDATE SET
                        backend_id=excluded.backend_id,
                        created_at=excluded.created_at
                    """,
                    (job_id, backend_id, self._now()),
                )

    async def get_job_backend(self, job_id: str) -> Optional[str]:
        """查询作业对应的后端 ID(无记录返回 None)。"""
        async with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT backend_id FROM job_routes WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
        return str(row["backend_id"]) if row else None

    async def set_upload_backend(self, upload_id: str, backend_id: str) -> None:
        """写入/更新上传 -> 后端映射(冲突时覆盖)。"""
        async with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO upload_routes(upload_id, backend_id, created_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(upload_id) DO UPDATE SET
                        backend_id=excluded.backend_id,
                        created_at=excluded.created_at
                    """,
                    (upload_id, backend_id, self._now()),
                )

    async def get_upload_backend(self, upload_id: str) -> Optional[str]:
        """查询上传对应的后端 ID(无记录返回 None)。"""
        async with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT backend_id FROM upload_routes WHERE upload_id = ?",
                    (upload_id,),
                ).fetchone()
        return str(row["backend_id"]) if row else None


class GatewayState:
    """网关运行态:配置/后端视图/HTTP 客户端/路由库/选择与重载的并发原语。"""

    def __init__(self, config_path: Path, config: GatewayConfig):
        # 配置文件路径(热加载检测对象)
        self.config_path = config_path
        # 记录配置 mtime(纳秒)
        self.config_mtime_ns = self._config_mtime_ns()
        # 当前生效配置
        self.config = config
        # 全量后端视图(含历史上出现过、现已移出的)
        self.all_backends: Dict[str, Backend] = {}
        # 启用后端视图
        self.backends: Dict[str, Backend] = {}
        # 启用后端的稳定顺序列表
        self.backend_list: List[Backend] = []
        # 作业路由持久化存储
        self.route_store = RouteStore(config.route_store_path)
        # 共享 HTTP 客户端(懒创建)
        self._client: Optional[httpx.AsyncClient] = None
        # 轮询游标
        self._rr_index = 0
        # 各后端的虚拟负载(已选但尚未上报的作业数,防雪崩扎堆)
        self._virtual_load: Dict[str, int] = {}
        # worker 状态缓存:backend_id -> (采样时刻, 数据或 None)
        self._worker_status_cache: Dict[str, Tuple[float, Optional[Dict[str, Any]]]] = {}
        # 选路互斥锁(同一时刻只有一个选择流程)
        self._select_lock = asyncio.Lock()
        # 配置重载互斥锁
        self._reload_lock = asyncio.Lock()
        # 立即应用初始配置
        self.apply_config(config)

    def _config_mtime_ns(self) -> int:
        """读取配置文件的 mtime(纳秒)。"""
        return self.config_path.stat().st_mtime_ns

    async def client(self) -> httpx.AsyncClient:
        """懒创建共享 HTTP 客户端(连接超时上限 10 秒)。"""
        if self._client is None:
            timeout = httpx.Timeout(
                self.config.request_timeout_seconds,
                connect=min(10.0, self.config.request_timeout_seconds),
            )
            self._client = httpx.AsyncClient(timeout=timeout)
        return self._client

    async def close(self) -> None:
        """关闭共享客户端(配置超时变化或应用关停时)。"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def apply_config(self, config: GatewayConfig) -> None:
        """应用新配置:合并历史后端、刷新启用列表,并对齐负载/缓存视图。"""
        # 本次配置声明的后端
        configured = {backend.id: backend for backend in config.backends}
        # 历史上存在但本次未声明的后端(保留元数据供旧作业查询)
        historical = {
            backend_id: backend
            for backend_id, backend in self.all_backends.items()
            if backend_id not in configured
        }
        self.config = config
        # 全量视图 = 历史 + 新配置
        self.all_backends = {**historical, **configured}
        # 启用视图:只留 enabled 的
        self.backends = {
            backend.id: backend
            for backend in config.backends
            if backend.enabled
        }
        self.backend_list = list(self.backends.values())
        # 虚拟负载只保留仍启用的后端(新后端从 0 开始)
        self._virtual_load = {
            backend.id: self._virtual_load.get(backend.id, 0)
            for backend in self.backend_list
        }
        # worker 状态缓存只保留仍被追踪的后端
        self._worker_status_cache = {
            backend_id: value
            for backend_id, value in self._worker_status_cache.items()
            if backend_id in self.all_backends
        }

    async def reload_config_if_needed(self, *, force: bool = False) -> bool:
        """按 mtime 热加载配置;force=True 时无视缓存强制重载。返回是否真的重载。"""
        try:
            current_mtime_ns = self._config_mtime_ns()
        except FileNotFoundError as exc:
            # 配置丢失:force 场景报错,常规请求静默跳过
            if force:
                raise HTTPException(status_code=500, detail=f"gateway config missing: {self.config_path}") from exc
            return False

        # mtime 未变且非强制:无需重载
        if not force and current_mtime_ns == self.config_mtime_ns:
            return False

        # 加锁重载(双重检查避免并发重复加载)
        async with self._reload_lock:
            current_mtime_ns = self._config_mtime_ns()
            if not force and current_mtime_ns == self.config_mtime_ns:
                return False

            # 解析新配置
            new_config = load_config(self.config_path)
            # 超时参数是否变化(变化则需重建客户端)
            timeout_changed = (
                new_config.request_timeout_seconds != self.config.request_timeout_seconds
                or new_config.health_timeout_seconds != self.config.health_timeout_seconds
            )
            # 路由库路径变化:切换存储实例
            if new_config.route_store_path != self.route_store.db_path:
                self.route_store = RouteStore(new_config.route_store_path)
            # 应用新配置并更新 mtime
            self.apply_config(new_config)
            self.config_mtime_ns = current_mtime_ns
            # 超时变化:关闭旧客户端(下次请求按新超时重建)
            if timeout_changed:
                await self.close()
            return True

    def require_backends(self) -> None:
        """断言存在启用后端,否则 503。"""
        if not self.backend_list:
            raise HTTPException(status_code=503, detail="no enabled backends configured")

    def get_backend(self, backend_id: str) -> Optional[Backend]:
        """按 ID 查后端(含未启用的历史后端)。"""
        return self.all_backends.get(backend_id)

    async def choose_backend(self, api_key: str) -> Backend:
        """按策略选一个后端:idle_gpu_first 优先,失败/不可用回退 round_robin。"""
        # 先尝试热加载配置
        await self.reload_config_if_needed()
        self.require_backends()
        statuses = None
        if self.config.strategy == "idle_gpu_first":
            # 并发拉取所有后端的 worker 状态(异常不中断彼此)
            statuses = await asyncio.gather(
                *(self.fetch_worker_status(backend, api_key) for backend in self.backend_list),
                return_exceptions=True,
            )
        # 选路加锁(虚拟负载更新必须串行)
        async with self._select_lock:
            if self.config.strategy == "idle_gpu_first" and statuses is not None:
                # 优先按空闲 GPU 选择
                selected = self._choose_by_idle_gpu(statuses)
                if selected is not None:
                    # 选中即加一虚拟负载,防止后续请求继续扎堆
                    self._virtual_load[selected.id] = self._virtual_load.get(selected.id, 0) + 1
                    return selected
            # 回退轮询
            selected = self._choose_round_robin()
            self._virtual_load[selected.id] = self._virtual_load.get(selected.id, 0) + 1
            return selected

    def _choose_round_robin(self) -> Backend:
        """加权轮询:权重决定后端在环中重复的次数。"""
        self.require_backends()
        # 展开成权重副本序列
        weighted: List[Backend] = []
        for backend in self.backend_list:
            weighted.extend([backend] * max(1, int(backend.weight)))
        # 按游标取模轮转
        backend = weighted[self._rr_index % len(weighted)]
        self._rr_index += 1
        return backend

    def _choose_by_idle_gpu(self, statuses: List[Any]) -> Optional[Backend]:
        """按"空闲 GPU - 虚拟负载"打分选后端;总分平手再看总容量。"""
        scored: List[Tuple[int, int, Backend]] = []
        for backend, status in zip(self.backend_list, statuses):
            # 状态拉取失败/格式不对:该后端不参与本轮
            if isinstance(status, Exception) or not isinstance(status, dict):
                continue
            summary = status.get("summary") if isinstance(status.get("summary"), dict) else {}
            # 空闲与总数(兼容不同字段名)
            idle = int(summary.get("gpu_idle") or summary.get("idle") or 0)
            total = int(summary.get("gpu_total") or summary.get("total") or 0)
            # 打分:空闲数扣掉虚拟负载(在途作业)
            score = idle - self._virtual_load.get(backend.id, 0)
            # 平手时用总容量打破,再按稳定配置顺序
            scored.append((score, total, backend))
        if not scored:
            # 全部不可用:返回 None 走回退
            return None
        # 按分数降序,平手按容量降序
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored[0][2]

    async def fetch_worker_status(self, backend: Backend, api_key: str) -> Optional[Dict[str, Any]]:
        """拉取后端 worker 状态(带 TTL 缓存);失败/出错缓存 None 并清零其虚拟负载。"""
        cached = self._worker_status_cache.get(backend.id)
        now = time.monotonic()
        # 缓存未过期:直接复用
        if cached and now - cached[0] < self.config.worker_status_ttl_seconds:
            return cached[1]
        client = await self.client()
        try:
            response = await client.get(
                f"{backend.base_url}/api/v1/workers/status",
                headers={"X-API-Key": api_key},
            )
            if response.status_code >= 400:
                # 失败也缓存(避免高频打挂的后端)
                self._worker_status_cache[backend.id] = (now, None)
                return None
            data = response.json()
        except Exception:
            self._worker_status_cache[backend.id] = (now, None)
            return None
        # 成功:缓存数据,并认为此前积压的作业已被后端消化
        self._worker_status_cache[backend.id] = (now, data)
        self._virtual_load[backend.id] = 0
        return data


def load_config(path: Path) -> GatewayConfig:
    """从 YAML 解析网关配置(严格校验,错误立即抛 RuntimeError)。"""
    if not path.exists():
        raise RuntimeError(f"gateway config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise RuntimeError("gateway config must be a YAML mapping")

    # 解析后端列表(每项必须是含 id/base_url 的映射)
    backend_items = raw.get("backends") or []
    if not isinstance(backend_items, list):
        raise RuntimeError("config.backends must be a list")
    backends: List[Backend] = []
    for item in backend_items:
        if not isinstance(item, dict):
            raise RuntimeError("each backend must be a mapping")
        # ID 去空白
        backend_id = str(item.get("id") or "").strip()
        # 地址去空白与尾斜杠
        base_url = str(item.get("base_url") or "").strip().rstrip("/")
        if not backend_id or not base_url:
            raise RuntimeError("each backend requires id and base_url")
        backends.append(
            Backend(
                id=backend_id,
                base_url=base_url,
                enabled=bool(item.get("enabled", True)),
                weight=int(item.get("weight", 1)),
            )
        )

    # 三个可选段:路由策略/请求超时/存储
    routing = raw.get("routing") or {}
    if not isinstance(routing, dict):
        routing = {}
    request = raw.get("request") or {}
    if not isinstance(request, dict):
        request = {}
    storage = raw.get("storage") or {}
    if not isinstance(storage, dict):
        storage = {}

    # 路由库路径:环境变量优先,其次配置,最后默认;相对路径基于配置文件目录
    db_path_raw = os.getenv("SANDBOX_GATEWAY_DB") or storage.get("route_store_path") or str(DEFAULT_DB_PATH)
    db_path = Path(str(db_path_raw))
    if not db_path.is_absolute():
        db_path = (path.parent / db_path).resolve()

    return GatewayConfig(
        backends=backends,
        strategy=str(routing.get("strategy", "idle_gpu_first")),
        fallback=str(routing.get("fallback", "round_robin")),
        worker_status_ttl_seconds=float(routing.get("worker_status_ttl_seconds", 2.0)),
        request_timeout_seconds=float(request.get("timeout_seconds", 120.0)),
        health_timeout_seconds=float(request.get("health_timeout_seconds", 5.0)),
        route_store_path=db_path,
    )


def config_path_from_env() -> Path:
    """从环境变量取配置路径(默认脚本目录下 backends.yaml)。"""
    return Path(os.getenv("SANDBOX_GATEWAY_CONFIG", str(DEFAULT_CONFIG_PATH))).resolve()


# 模块加载即完成:定位配置 -> 加载 -> 建运行态 -> 建 FastAPI 应用
CONFIG_PATH = config_path_from_env()
config = load_config(CONFIG_PATH)
state = GatewayState(CONFIG_PATH, config)
app = FastAPI(title="Sandbox Gateway API", version="0.1.0")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """应用关停时释放共享 HTTP 客户端。"""
    await state.close()


def forward_headers(request: Request, api_key: Optional[str] = None) -> Dict[str, str]:
    """构造转发头:透传 API 密钥与 Content-Type/Accept(其余头不透传)。"""
    headers: Dict[str, str] = {}
    if api_key:
        headers["X-API-Key"] = api_key
    content_type = request.headers.get("content-type")
    if content_type:
        headers["Content-Type"] = content_type
    accept = request.headers.get("accept")
    if accept:
        headers["Accept"] = accept
    return headers


def proxy_response(response: httpx.Response) -> Response:
    """把后端响应原样转成 FastAPI 响应(状态码/字节体/去参的 media type)。"""
    content_type = response.headers.get("content-type", "application/json")
    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=content_type.split(";")[0],
    )


async def request_backend(
    backend: Backend,
    method: str,
    path: str,
    request: Request,
    api_key: Optional[str],
    *,
    content: Optional[bytes] = None,
) -> httpx.Response:
    """向指定后端转发请求(未显式给 body 的写方法自动读取原请求体)。"""
    client = await state.client()
    if content is None and method.upper() in {"POST", "PUT", "PATCH"}:
        content = await request.body()
    try:
        return await client.request(
            method,
            f"{backend.base_url}{path}",
            params=request.query_params,
            headers=forward_headers(request, api_key),
            content=content,
        )
    except httpx.HTTPError as exc:
        # 转发失败统一 502(带后端 ID 与错误详情)
        raise HTTPException(
            status_code=502,
            detail=f"backend request failed: backend={backend.id}, error={exc}",
        ) from exc


async def resolve_backend_for_job(job_id: str, api_key: str) -> Backend:
    """定位作业所在后端:先查路由库,未命中则全后端探测并回填映射。"""
    await state.reload_config_if_needed()
    # 先查持久化映射
    backend_id = await state.route_store.get_job_backend(job_id)
    if backend_id:
        backend = state.get_backend(backend_id)
        if backend:
            return backend
        # 映射指向的后端已被禁用:报 503
        raise HTTPException(status_code=503, detail=f"mapped backend is not enabled: {backend_id}")

    # 无映射(网关重启/旧作业):逐个后端探测
    client = await state.client()
    for backend in state.all_backends.values():
        try:
            response = await client.get(
                f"{backend.base_url}/api/v1/jobs/{job_id}",
                headers={"X-API-Key": api_key},
            )
        except Exception:
            continue
        if response.status_code == 200:
            # 命中:回填映射后返回
            await state.route_store.set_job_backend(job_id, backend.id)
            return backend
    raise HTTPException(status_code=404, detail="Job not found")


def summarize_worker_payloads(payloads: Iterable[Tuple[Backend, Optional[Dict[str, Any]]]]) -> Dict[str, Any]:
    """归并各后端的 worker 状态:汇总统计 + 明细排序 + 每后端健康标记。"""
    # 明细列表与每后端摘要
    workers_show: List[Dict[str, Any]] = []
    backend_summaries: Dict[str, Any] = {}
    # 汇总计数器
    total = gpu_total = cpu_total = 0
    gpu_idle = cpu_idle = gpu_quarantined = cpu_quarantined = 0

    # 逐后端累加
    for backend, payload in payloads:
        if not isinstance(payload, dict):
            # 拉取失败:该后端标记不可用
            backend_summaries[backend.id] = {"ok": False}
            continue
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        backend_summaries[backend.id] = {"ok": True, "summary": summary}
        total += int(summary.get("total") or 0)
        gpu_total += int(summary.get("gpu_total") or 0)
        cpu_total += int(summary.get("cpu_total") or 0)
        gpu_idle += int(summary.get("gpu_idle") or 0)
        cpu_idle += int(summary.get("cpu_idle") or 0)
        gpu_quarantined += int(summary.get("gpu_quarantined") or 0)
        cpu_quarantined += int(summary.get("cpu_quarantined") or 0)
        # 明细补充 backend_id 后并入列表
        for worker in payload.get("workers_show") or []:
            if isinstance(worker, dict):
                enriched = dict(worker)
                enriched["backend_id"] = backend.id
                workers_show.append(enriched)

    # 全局汇总(忙碌 = 总数 - 空闲 - 隔离,负值截 0)
    summary = {
        "total": total,
        "cpu_total": cpu_total,
        "gpu_total": gpu_total,
        "cpu_idle": cpu_idle,
        "gpu_idle": gpu_idle,
        "cpu_busy": max(0, cpu_total - cpu_idle - cpu_quarantined),
        "gpu_busy": max(0, gpu_total - gpu_idle - gpu_quarantined),
        "cpu_quarantined": cpu_quarantined,
        "gpu_quarantined": gpu_quarantined,
    }
    # 一行人类可读统计串
    stats = {
        f"total:{total}, cpu_total:{cpu_total}, gpu_total:{gpu_total}; "
        f"cpu_idle:{cpu_idle}/{cpu_total}, gpu_idle:{gpu_idle}/{gpu_total}; "
        f"cpu_quarantined:{cpu_quarantined}/{cpu_total}, gpu_quarantined:{gpu_quarantined}/{gpu_total}"
    }
    # 明细按后端+端点稳定排序
    workers_show.sort(key=lambda item: (item.get("backend_id") or "", item.get("endpoint") or ""))
    return {
        "workers_show": workers_show,
        "stats": stats,
        "summary": summary,
        "backends": backend_summaries,
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    """聚合健康检查:逐个后端探 /health,汇总整体健康状态。"""
    await state.reload_config_if_needed()
    client = await state.client()
    results: Dict[str, Any] = {}
    ok_count = 0
    # 逐后端探测(短超时)
    for backend in state.backend_list:
        try:
            response = await client.get(
                f"{backend.base_url}/health",
                timeout=state.config.health_timeout_seconds,
            )
            ok = response.status_code == 200
            body: Any
            try:
                body = response.json()
            except Exception:
                body = response.text[:500]
            results[backend.id] = {"ok": ok, "status_code": response.status_code, "body": body}
            ok_count += 1 if ok else 0
        except Exception as exc:
            results[backend.id] = {"ok": False, "error": str(exc)}
    return {
        "status": "healthy" if ok_count > 0 else "unhealthy",
        "configured_backends": len(state.all_backends),
        "enabled_backends": len(state.backend_list),
        "healthy_backends": ok_count,
        "backends": results,
    }


@app.get("/api/v1/workers/status")
async def get_workers_status(x_api_key: str = Header(...)) -> Dict[str, Any]:
    """聚合 worker 状态:并发拉取全部启用后端,非 dict 结果归一为 None 后汇总。"""
    await state.reload_config_if_needed()
    state.require_backends()
    # 并发拉取(异常也会出现在结果里)
    payloads = await asyncio.gather(
        *(state.fetch_worker_status(backend, x_api_key) for backend in state.backend_list),
        return_exceptions=True,
    )
    # 归一化成 (后端, 数据或 None) 序列
    normalized: List[Tuple[Backend, Optional[Dict[str, Any]]]] = []
    for backend, payload in zip(state.backend_list, payloads):
        normalized.append((backend, payload if isinstance(payload, dict) else None))
    return summarize_worker_payloads(normalized)


@app.post("/api/v1/jobs/submit_and_wait")
async def submit_and_wait_not_implemented(x_api_key: str = Header(...)) -> JSONResponse:
    """显式拒绝 submit_and_wait:网关不阻塞等待,客户端应自行提交+轮询。"""
    _ = x_api_key
    return JSONResponse(
        status_code=501,
        content={
            "detail": "submit_and_wait is not implemented in the gateway; use POST /api/v1/jobs and poll /api/v1/jobs/{job_id}",
        },
    )


def require_admin(x_api_key: Optional[str], x_gateway_admin_token: Optional[str]) -> None:
    """管理接口鉴权:优先校验管理令牌,未配置时回退内置 API 密钥。"""
    admin_token = os.getenv("SANDBOX_GATEWAY_ADMIN_TOKEN", "")
    if admin_token:
        if x_gateway_admin_token != admin_token:
            raise HTTPException(status_code=401, detail="invalid gateway admin token")
        return
    if x_api_key not in {"mlsandbox-router-oss-71c4bb62f1084d208e7d783589168cd4"}:
        raise HTTPException(status_code=401, detail="invalid gateway admin credential")


def backend_config_snapshot() -> Dict[str, Any]:
    """当前网关配置快照(路径/策略/后端清单),供管理接口展示。"""
    return {
        "config_path": str(state.config_path),
        "config_mtime_ns": state.config_mtime_ns,
        "route_store_path": str(state.route_store.db_path),
        "strategy": state.config.strategy,
        "fallback": state.config.fallback,
        "configured_backends": [
            {
                "id": backend.id,
                "base_url": backend.base_url,
                "enabled": backend.enabled,
                "weight": backend.weight,
            }
            for backend in state.all_backends.values()
        ],
        "enabled_backends": [backend.id for backend in state.backend_list],
    }


@app.get("/admin/config")
async def admin_config(
    x_api_key: Optional[str] = Header(None),
    x_gateway_admin_token: Optional[str] = Header(None),
) -> Dict[str, Any]:
    """查看网关配置(鉴权后返回当前快照)。"""
    require_admin(x_api_key, x_gateway_admin_token)
    await state.reload_config_if_needed()
    return backend_config_snapshot()


@app.post("/admin/reload")
async def admin_reload(
    x_api_key: Optional[str] = Header(None),
    x_gateway_admin_token: Optional[str] = Header(None),
) -> Dict[str, Any]:
    """强制重载配置(鉴权后 force 重载,返回快照并带 reloaded 标记)。"""
    require_admin(x_api_key, x_gateway_admin_token)
    reloaded = await state.reload_config_if_needed(force=True)
    snapshot = backend_config_snapshot()
    snapshot["reloaded"] = reloaded
    return snapshot


@app.post("/api/v1/uploads")
async def uploads_not_implemented(x_api_key: str = Header(...)) -> JSONResponse:
    """上传接口未实现:显式 501,提示客户端直连后端。"""
    _ = x_api_key
    return JSONResponse(
        status_code=501,
        content={"detail": "uploads are not implemented in the gateway"},
    )


@app.post("/api/v1/jobs")
async def submit_job(request: Request, x_api_key: str = Header(...)) -> Response:
    """提交作业:选路 -> 转发;成功且拿到 job_id 时记录路由映射。"""
    backend = await state.choose_backend(x_api_key)
    response = await request_backend(backend, "POST", "/api/v1/jobs", request, x_api_key)
    if response.status_code < 400:
        try:
            data = response.json()
        except Exception:
            data = None
        if isinstance(data, dict) and data.get("job_id"):
            # 作业 -> 后端映射持久化(重启后仍可定位)
            await state.route_store.set_job_backend(str(data["job_id"]), backend.id)
    return proxy_response(response)


@app.get("/api/v1/jobs")
async def list_jobs(request: Request, x_api_key: str = Header(...)) -> Dict[str, Any]:
    """跨后端聚合作业列表:拉满 offset+limit 条,全局按创建时间排序后再分页。"""
    await state.reload_config_if_needed()
    state.require_backends()
    # 客户端想要的分页参数
    requested_limit = int(request.query_params.get("limit", "50"))
    requested_offset = int(request.query_params.get("offset", "0"))
    query = dict(request.query_params)
    # 每个后端都从 0 拉到 offset+limit(保证全局分页正确)
    query["limit"] = str(max(requested_limit + requested_offset, requested_limit))
    query["offset"] = "0"
    client = await state.client()

    # 单后端拉取(失败返回 None)
    async def fetch(backend: Backend) -> Tuple[Backend, Optional[Dict[str, Any]]]:
        try:
            response = await client.get(
                f"{backend.base_url}/api/v1/jobs",
                params=query,
                headers={"X-API-Key": x_api_key},
            )
            if response.status_code >= 400:
                return backend, None
            return backend, response.json()
        except Exception:
            return backend, None

    # 并发拉取全部后端
    results = await asyncio.gather(*(fetch(backend) for backend in state.backend_list))
    jobs: List[Dict[str, Any]] = []
    backend_counts: Dict[str, int] = {}
    for backend, payload in results:
        backend_jobs = payload.get("jobs") if isinstance(payload, dict) else []
        backend_counts[backend.id] = len(backend_jobs) if isinstance(backend_jobs, list) else 0
        if isinstance(backend_jobs, list):
            for job in backend_jobs:
                if isinstance(job, dict):
                    # 明细补充 backend_id 来源标记
                    enriched = dict(job)
                    enriched["backend_id"] = backend.id
                    jobs.append(enriched)
    # 全局按创建时间倒序
    jobs.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    # 应用原分页窗口
    sliced = jobs[requested_offset : requested_offset + requested_limit]
    return {"jobs": sliced, "count": len(sliced), "fetched_count": len(jobs), "backend_counts": backend_counts}


@app.get("/api/v1/jobs/{job_id}")
async def get_job_status(job_id: str, request: Request, x_api_key: str = Header(...)) -> Response:
    """查询作业状态:定位后端后原样转发。"""
    backend = await resolve_backend_for_job(job_id, x_api_key)
    response = await request_backend(backend, "GET", f"/api/v1/jobs/{job_id}", request, x_api_key)
    return proxy_response(response)


@app.get("/api/v1/jobs/{job_id}/logs")
async def get_job_logs(job_id: str, request: Request, x_api_key: str = Header(...)) -> Response:
    """查询作业日志:定位后端后原样转发。"""
    backend = await resolve_backend_for_job(job_id, x_api_key)
    response = await request_backend(backend, "GET", f"/api/v1/jobs/{job_id}/logs", request, x_api_key)
    return proxy_response(response)


@app.delete("/api/v1/jobs/{job_id}")
async def cancel_job(job_id: str, request: Request, x_api_key: str = Header(...)) -> Response:
    """取消作业:定位后端后转发 DELETE。"""
    backend = await resolve_backend_for_job(job_id, x_api_key)
    response = await request_backend(backend, "DELETE", f"/api/v1/jobs/{job_id}", request, x_api_key)
    return proxy_response(response)
