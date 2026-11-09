# 中文说明:本模块是沙箱节点控制器(node_controller)的 API 服务端,
# 基于 FastAPI 提供"提交训练/评估作业 -> 查询状态/日志 -> 取消作业"的 HTTP 接口。
# 在 OpenRSI 系统中的位置:OpenMLE-Gym 沙箱集群的调度入口——API Server 把作业写入
# Redis 队列(cpu/gpu x 优先级 4 类队列),由 worker 节点消费执行,元数据持久化在
# PostgreSQL(jobs 表);支持幂等提交(idempotency_key + payload 哈希)、提交限流
# (进程内信号量)、可观测性埋点(OBS JSON 日志)与作业目录世界可读权限。
# 关键设计:所有路径访问被限制在 STORAGE_PATH 下;按"日期/数据集标签/作业ID"组织
# 作业目录;终态(completed/failed/cancelled)后才返回完整 result。

# 延迟注解:类型标注运行期不求值
from __future__ import annotations

# asyncio:异步端点中把阻塞 DB 查询丢进线程池执行
import asyncio
# contextlib:suppress(吞掉清理异常)
import contextlib
# contextmanager:写生成器型上下文管理器(DB 连接、提交准入槽)
from contextlib import contextmanager
# hashlib:payload 哈希(幂等冲突检测)与代码 SHA256 记录
import hashlib
# json:环境变量/结果字段与 JSON 互转、队列消息序列化
import json
# os:环境变量读取、目录遍历
import os
# re:数据集标签清洗(非法字符替换)
import re
# shutil:代码归档移动/解压/目录复制
import shutil
# threading:DB 池初始化锁、提交准入信号量
import threading
# time:各阶段耗时统计(monotonic)
import time
# uuid:作业 ID/上传 ID/trace_id 生成
import uuid
# datetime:UTC 时间戳(创建时间/取消时间)
from datetime import datetime
# Path:作业目录、存储路径构造
from pathlib import Path
# 类型注解
from typing import Dict, List, Optional, Tuple

# psycopg2:PostgreSQL 驱动(作业元数据存储)
import psycopg2
# 唯一约束冲突异常类型(幂等竞态识别)
from psycopg2 import errors as pg_errors
# 线程化连接池(进程内复用 DB 连接)
from psycopg2 import pool as pg_pool
# redis 客户端:作业队列与 worker 心跳存储
import redis
# FastAPI 组件:依赖注入、路由、文件上传、请求头、HTTP 异常
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
# 流式响应:日志/运行日志按块返回
from fastapi.responses import StreamingResponse
# pydantic 模型:请求/响应体校验
from pydantic import BaseModel, Field
# RealDictCursor:查询结果按列名取值的字典游标
from psycopg2.extras import RealDictCursor

# FastAPI 应用实例:对外暴露的沙箱训练 API
app = FastAPI(title="Sandbox ML Training API", version="1.0.0")

# ============ 配置区:全部可通过环境变量覆盖 ============
# Redis 主机与端口(作业队列/心跳)
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
# PostgreSQL 连接参数(容器默认值)
DB_HOST = os.getenv("DB_HOST", "postgres")
DB_NAME = os.getenv("DB_NAME", "sandbox")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
# DB 连接池上下限
DB_POOL_MIN = int(os.getenv("DB_POOL_MIN", "2"))
DB_POOL_MAX = int(os.getenv("DB_POOL_MAX", "10"))
# 存储根目录默认值(共享存储上的沙箱根)
DEFAULT_STORAGE_ROOT = Path("/mnt/pubdatasets2/mlsandbox")
# 实际存储根(解析符号链接后的绝对路径)
STORAGE_PATH = Path(os.getenv("STORAGE_PATH", str(DEFAULT_STORAGE_ROOT))).resolve()
# 上传根目录(默认在存储根的 uploads/ 下),启动即创建
UPLOAD_ROOT = Path(os.getenv("UPLOAD_ROOT", str(STORAGE_PATH / "uploads"))).resolve()
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
# submit_and_wait 轮询间隔(秒)
DEFAULT_POLL_INTERVAL = int(os.getenv("JOB_WAIT_POLL_INTERVAL", "3"))
# submit_and_wait 总等待超时(秒)
DEFAULT_WAIT_TIMEOUT = int(os.getenv("JOB_WAIT_TIMEOUT", "1800"))
# 作业终态集合:进入终态后可返回完整结果
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
# 客户端任务 ID 在 environment 字典中的键名
CLIENT_TASK_KEY = "CLIENT_TASK_ID"
# 可观测性埋点开关(默认开启;0/false/no/off 关闭)
ENABLE_OBS = os.getenv("ENABLE_OBS", "1").lower() not in {"0", "false", "no", "off"}
# OBS 日志落盘目录(未配置则只打印 stdout)
OBS_LOG_DIR_RAW = os.getenv("OBS_LOG_DIR", "").strip()
OBS_LOG_DIR = Path(OBS_LOG_DIR_RAW).resolve() if OBS_LOG_DIR_RAW else None
# 本进程提交并发上限(0 表示不限流)
SUBMIT_MAX_INFLIGHT_PER_PROCESS = int(os.getenv("SUBMIT_MAX_INFLIGHT_PER_PROCESS", "0"))
# 等待准入槽的最长时间(秒)
SUBMIT_SLOT_WAIT_SECONDS = float(os.getenv("SUBMIT_SLOT_WAIT_SECONDS", "2"))
# 限流被拒时建议客户端重试的间隔(至少 1 秒)
SUBMIT_RETRY_AFTER_SECONDS = max(1, int(os.getenv("SUBMIT_RETRY_AFTER_SECONDS", "2")))
# 幂等提交开关(默认开启)
ENABLE_IDEMPOTENCY = os.getenv("ENABLE_IDEMPOTENCY", "1").lower() not in {"0", "false", "no", "off"}
# 四类作业队列:资源类型(cpu/gpu)x 优先级(1 高/2 低)
JOB_QUEUE_CPU_P1 = "job_queue_cpu_p1"
JOB_QUEUE_CPU_P2 = "job_queue_cpu_p2"
JOB_QUEUE_GPU_P1 = "job_queue_gpu_p1"
JOB_QUEUE_GPU_P2 = "job_queue_gpu_p2"
# 资源类型+优先级 -> 队列名的映射
JOB_QUEUE_BY_RESOURCE_PRIORITY = {
    "cpu": {1: JOB_QUEUE_CPU_P1, 2: JOB_QUEUE_CPU_P2},
    "gpu": {1: JOB_QUEUE_GPU_P1, 2: JOB_QUEUE_GPU_P2},
}

# Redis 客户端:decode_responses=True 使返回值为 str 而非 bytes
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
# DB 连接池(延迟到 startup 事件创建)
_DB_POOL: Optional[pg_pool.ThreadedConnectionPool] = None
# 池创建锁(防止并发初始化)
_DB_POOL_LOCK = threading.Lock()
# 提交准入信号量(配置了上限才创建;BoundedSemaphore 防止超额释放)
_SUBMIT_SEMAPHORE = (
    threading.BoundedSemaphore(SUBMIT_MAX_INFLIGHT_PER_PROCESS)
    if SUBMIT_MAX_INFLIGHT_PER_PROCESS > 0
    else None
)


def _obs_log(event: str, **fields) -> None:
    """写入一条结构化观测日志:stdout 打印 + 可选落盘 JSONL(便于事后检索)。"""
    # 埋点总开关关闭则直接返回
    if not ENABLE_OBS:
        return
    # 组装日志载荷:UTC 时间戳 + 事件名 + 附加字段
    payload = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "event": event,
        **fields,
    }
    # 序列化为一行 JSON(键排序保证日志稳定可 grep)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    # 立即刷新打印(容器环境下即时可见)
    print(line, flush=True)
    # 配置了落盘目录则追加写入 JSONL 文件
    if OBS_LOG_DIR is not None:
        try:
            # 目录不存在则逐级创建
            OBS_LOG_DIR.mkdir(parents=True, exist_ok=True)
            # 追加写 api_server.jsonl
            with (OBS_LOG_DIR / "api_server.jsonl").open("a", encoding="utf-8") as fp:
                fp.write(line + "\n")
        except OSError as exc:
            # 落盘失败不影响主流程,只提示
            print(f"[obs-log] failed to write {OBS_LOG_DIR}: {exc}", flush=True)


def _stage_log(
    stage: str,
    start_time: float,
    *,
    trace_id: Optional[str] = None,
    job_id: Optional[str] = None,
    **fields,
) -> None:
    """记录一个提交阶段的耗时埋点(用于定位提交链路慢在哪一步)。"""
    _obs_log(
        # 事件名固定为 api.stage,stage 字段区分阶段
        "api.stage",
        stage=stage,
        # 阶段耗时(毫秒,保留 3 位小数)
        elapsed_ms=round((time.monotonic() - start_time) * 1000, 3),
        # 关联 trace 与作业 ID
        trace_id=trace_id,
        job_id=job_id,
        **fields,
    )


@contextmanager
def submit_admission_slot():
    """提交准入槽上下文:限流时最多等 SLOT_WAIT 秒,超时返回 503 + Retry-After。"""
    # 未配置限流:直接放行
    if _SUBMIT_SEMAPHORE is None:
        yield
        return
    # 尝试在限定时间内拿到一个槽位
    acquired = _SUBMIT_SEMAPHORE.acquire(timeout=max(0.0, SUBMIT_SLOT_WAIT_SECONDS))
    # 拿不到槽位:返回 503 server_busy,并带 Retry-After 响应头
    if not acquired:
        raise HTTPException(
            status_code=503,
            detail={"error": "server_busy", "retry_after": SUBMIT_RETRY_AFTER_SECONDS},
            headers={"Retry-After": str(SUBMIT_RETRY_AFTER_SECONDS)},
        )
    try:
        # 持有槽位执行提交逻辑
        yield
    finally:
        # 无论成败都释放槽位(Bounded 信号量防重复释放)
        _SUBMIT_SEMAPHORE.release()


def _is_relative_to(path: Path, base: Path) -> bool:
    """判断 path 是否位于 base 目录之下(解析符号链接后比较)。"""
    try:
        # 尝试取相对路径:能取到即在 base 之下
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        # relative_to 抛 ValueError 说明不在 base 下
        return False


def _normalize_dataset_label(source: Optional[str]) -> str:
    """把数据目录路径归一化为目录名标签(仅保留字母数字与 ._-),用于作业目录命名。"""
    # 无数据目录:使用 default 标签
    if not source:
        return "default"
    try:
        # 先按 Path 取最后一段目录名
        candidate = Path(source.rstrip("/")).name
    except Exception:
        # 非法路径字符等异常:置空走手工解析
        candidate = ""
    # Path 解析失败或为空:按字符串剥掉首尾斜杠后取最后一段
    if not candidate:
        stripped = source.strip("/ ")
        candidate = stripped.split("/")[-1] if stripped else ""
    # 替换所有非法字符为下划线(目录名安全)
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate)
    # 清洗后为空则回退 default
    return label or "default"


def _relocate_source_to_job(src: Path, code_root: Path) -> Path:
    """把提交的代码源(上传目录/普通目录/单文件)安置进本作业的 code 目录。

    - 上传目录内的路径:整棵 upload 移动(move)到作业目录(避免拷贝大目录);
    - 普通目录:递归复制(copytree,覆盖同名);
    - 单文件:复制文件(copy2 保留元数据)。
    """
    # 情况一:源路径在上传根目录下
    if _is_relative_to(src, UPLOAD_ROOT):
        # 计算相对路径(第一段为上传 ID)
        relative = src.relative_to(UPLOAD_ROOT)
        # 该次上传的根目录
        upload_root = UPLOAD_ROOT / relative.parts[0]
        # 确保作业 code 目录存在
        code_root.mkdir(parents=True, exist_ok=True)
        # 把整个上传目录移动(move)进作业 code 目录
        shutil.move(str(upload_root), str(code_root))
        # 移动后的上传根
        moved_root = code_root / relative.parts[0]
        # 若源是上传目录内的子路径,返回对应子路径
        if len(relative.parts) > 1:
            return moved_root.joinpath(*relative.parts[1:])
        # 否则返回移动后的根
        return moved_root

    # 情况二:源是一个普通目录,递归复制进 code 目录
    if src.is_dir():
        # 目标目录
        dest = code_root / src.name
        # 同名目标已存在则先删除(覆盖语义)
        if dest.exists():
            shutil.rmtree(dest)
        # 递归复制
        shutil.copytree(src, dest)
        # 返回目标目录
        return dest

    # 情况三:源是单个文件
    code_root.mkdir(parents=True, exist_ok=True)
    # 目标文件
    dest_file = code_root / src.name
    # copy2 保留时间戳等元数据
    shutil.copy2(src, dest_file)
    # 返回目标文件
    return dest_file


def _make_world_writable(path: Path) -> None:
    """把作业目录整体 chmod 777:沙箱 worker 以不同用户运行,需要读写作业产物。"""
    # 路径不存在直接返回
    if not path.exists():
        return

    def _chmod(target: Path) -> None:
        # 嵌套函数:尽力 chmod,失败(权限/其他 OSError)忽略
        try:
            target.chmod(0o777)
        except (PermissionError, OSError):
            pass

    # 先改根目录
    _chmod(path)
    # 目录则遍历全部子项(不跟随符号链接,避免越界改权限)
    if path.is_dir():
        for root, dirs, files in os.walk(path, followlinks=False):
            root_path = Path(root)
            # 改当前层目录
            _chmod(root_path)
            # 改该层所有子目录与文件
            for name in dirs + files:
                _chmod(root_path / name)


def _load_json(value):
    """把 DB 里以 JSON 字符串存储的字段尽量还原为对象;无法解析则原样返回。"""
    # None 原样
    if value is None:
        return None
    # 已经是 dict/list 直接返回
    if isinstance(value, (dict, list)):
        return value
    try:
        # 尝试按 JSON 解析字符串
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        # 非 JSON 字符串(如旧数据):原样返回
        return value


class JobSubmission(BaseModel):
    """提交作业的请求体模型(pydantic 校验)。"""
    # 客户端自定义任务标识(可选,回显用)
    task_id: Optional[str] = Field(None, description="客户端自定义任务标识")
    # 幂等键:同一逻辑任务重试时复用,避免重复建作业
    idempotency_key: Optional[str] = Field(
        None,
        max_length=128,
        description="客户端幂等键；同一逻辑任务重试时复用",
    )
    # 任务名称(必填)
    name: str = Field(..., description="任务名称")
    # 直接内联的训练代码内容
    code: Optional[str] = Field(None, description="训练代码内容")
    # 或共享存储上已有的代码路径(与 code 二选一)
    code_file_path: Optional[str] = Field(None, description="共享存储上的代码路径")
    # 沙盒内可访问的数据目录绝对路径
    data_dir: Optional[str] = Field(None, description="沙盒内可访问的数据目录绝对路径")
    # 需要安装的依赖包列表
    requirements: List[str] = Field(default_factory=list, description="依赖包列表")
    # 额外数据路径列表
    data_paths: List[str] = Field(default_factory=list, description="数据路径")
    # GPU 需求数量(1-8)
    gpu_count: int = Field(1, ge=1, le=8, description="所需 GPU 数量")
    # 作业超时秒数
    timeout: int = Field(3600, description="超时时间(秒)")
    # 注入沙盒的环境变量
    environment: Dict[str, str] = Field(default_factory=dict, description="环境变量")
    # 自定义 shell 命令(仅 shell 执行模式使用)
    execution_command: Optional[str] = Field(
        default=None, description="自定义 shell 命令 (仅 shell 模式使用)"
    )
    # 自定义工作目录(沙箱内部路径)
    working_dir: Optional[str] = Field(
        default=None, description="自定义工作目录 (Sandbox 内部路径)"
    )
    # 优先级:1 高 / 2 低,默认 1
    priority: int = Field(
        default=1,
        ge=1,
        le=2,
        description="任务优先级: 1(高)/2(低)，默认 1"
    )
    # 资源类型:gpu/cpu,默认 cpu
    resource_type: str = Field(
        default="cpu",
        description="资源类型: gpu/cpu (默认 cpu)"
    )


def _canonical_submit_payload(job: JobSubmission) -> Dict:
    """把提交请求展开成规范化字典(用于幂等哈希;字段做最小归一)。"""
    # 逐字段展开;priority/resource_type 做与入队一致的归一
    return {
        "name": job.name,
        "task_id": job.task_id,
        "code": job.code,
        "code_file_path": job.code_file_path,
        "data_dir": job.data_dir,
        "requirements": list(job.requirements or []),
        "data_paths": list(job.data_paths or []),
        "gpu_count": job.gpu_count,
        "timeout": job.timeout,
        "environment": dict(job.environment or {}),
        "execution_command": job.execution_command,
        "working_dir": job.working_dir,
        "priority": int(job.priority) if job.priority in (1, 2) else 1,
        "resource_type": (job.resource_type or "cpu").strip().lower(),
    }


def _submit_payload_hash(job: JobSubmission) -> str:
    """计算提交载荷的 SHA256(键排序、紧凑分隔),作为幂等键的载荷指纹。"""
    # 取规范化载荷
    canonical = _canonical_submit_payload(job)
    # 紧凑序列化(键排序,同样内容必得同哈希)
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    # 返回 SHA256 十六进制摘要
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalize_idempotency_key(value: Optional[str]) -> Optional[str]:
    """归一化幂等键:功能关闭或未提供返回 None;空白串视为未提供。"""
    # 幂等关闭或未传键:不启用
    if not ENABLE_IDEMPOTENCY or value is None:
        return None
    # 去除首尾空白
    key = value.strip()
    # 空串返回 None
    return key or None


def fetch_idempotent_job(api_key: str, idempotency_key: str):
    """按 (api_key, idempotency_key) 查最早创建的作业行(幂等命中查询)。"""
    # 借用 DB 连接上下文
    with get_db_connection() as conn:
        # 使用游标执行查询
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM jobs
                WHERE api_key = %s AND idempotency_key = %s
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (api_key, idempotency_key),
            )
            # 返回命中的行(可能为 None)
            return cursor.fetchone()


def _task_id_from_job_row(row) -> Optional[str]:
    """从作业行的 environment 字段提取客户端任务 ID(历史存储位置)。"""
    # 解析 environment(可能是 JSON 字符串)
    environment_data = _load_json(row.get("environment")) if row else None
    # 是字典才取键
    if isinstance(environment_data, dict):
        return environment_data.get(CLIENT_TASK_KEY)
    # 无法解析返回 None
    return None


def _created_at_for_response(row) -> str:
    """把行内 created_at(datetime 或字符串)统一为 ISO 字符串。"""
    # 取原始值
    created_at = row.get("created_at")
    # datetime 类型:isoformat 后补 Z(UTC)
    if hasattr(created_at, "isoformat"):
        return created_at.isoformat() + "Z"
    # 兜底直接转字符串
    return str(created_at)



class JobResponse(BaseModel):
    """提交作业的响应体:作业 ID、状态、创建时间等。"""
    # 作业 ID
    job_id: str
    # 当前状态(queued)
    status: str
    # 创建时间(ISO 字符串)
    created_at: str
    # 预计开始时间(暂未实现,恒 None)
    estimated_start: Optional[str] = None
    # 回显的客户端任务 ID
    task_id: Optional[str] = None


class JobStatus(BaseModel):
    """作业状态查询的响应体:完整状态、进度、时间线与结果。"""
    # 作业 ID
    job_id: str
    # 任务名
    name: str
    # 当前状态
    status: str
    # 进度百分比
    progress: int
    # 执行该作业的 worker
    worker_id: Optional[str]
    # 创建时间
    created_at: str
    # 开始时间
    started_at: Optional[str]
    # 最近更新时间
    updated_at: str
    # 完成时间
    completed_at: Optional[str]
    # 日志下载 URL
    logs_url: str
    # 最终结果(终态后非空)
    result: Optional[Dict]
    # 指标(终态后非空)
    metrics: Optional[Dict]
    # 客户端任务 ID
    task_id: Optional[str]


@app.on_event("startup")
def startup_event():
    """FastAPI 启动钩子:创建进程级 PostgreSQL 连接池(双重检查 + 锁防并发重复建池)。"""
    # 声明使用全局池变量
    global _DB_POOL
    # 快路径:池已存在直接返回
    if _DB_POOL is not None:
        return
    # 持锁做双检,防止多 worker 并发初始化
    with _DB_POOL_LOCK:
        # 锁内再检查一次
        if _DB_POOL is not None:
            return
        try:
            # 最小连接数至少 1
            minconn = max(1, DB_POOL_MIN)
            # 最大连接数不低于最小值
            maxconn = max(minconn, DB_POOL_MAX)
            # 创建线程化连接池(字典游标,按列名取值)
            _DB_POOL = pg_pool.ThreadedConnectionPool(
                minconn,
                maxconn,
                host=DB_HOST,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                cursor_factory=RealDictCursor,
            )
            # 打印初始化成功信息
            print(f"Database connection pool initialized ({minconn}-{maxconn}).")
        except Exception as exc:
            # 初始化失败不崩溃启动:请求时 get_db_connection 会返回 503 提示重试
            print(f"Error initializing connection pool: {exc}")


@contextmanager
def get_db_connection():
    """DB 连接上下文:从池借连接,异常回滚,归还前保证连接干净(防止事务状态残留)。"""
    # 池未初始化(如测试/未触发 startup):尝试即时初始化
    if _DB_POOL is None:
        startup_event()
    # 仍无池:返回 503 让客户端重试
    if _DB_POOL is None:
        raise HTTPException(status_code=503, detail="database pool not initialized, please retry")
    try:
        # 从池借一个连接
        conn = _DB_POOL.getconn()
    except pg_pool.PoolError as exc:
        # 池耗尽:返回 503 提示重试
        raise HTTPException(
            status_code=503,
            detail="database connection pool exhausted, please retry",
        ) from exc
    try:
        # 把连接交给 with 体内的业务代码
        yield conn
    except Exception:
        # 业务异常:回滚未提交事务后向上抛
        conn.rollback()
        raise
    finally:
        # 中文说明:psycopg2 在第一条语句就会开启事务;
        # 必须在归还前回滚,保证只读请求也把"干净"连接还给池,避免下次借到残留事务状态。
        try:
            # 连接未关闭才回滚
            if not conn.closed:
                conn.rollback()
        except Exception:
            # 回滚失败不阻断归还
            pass
        # 归还连接到池
        _DB_POOL.putconn(conn)


def verify_api_key(x_api_key: str = Header(...)):
    """API Key 校验依赖:请求头 X-API-Key 必须命中白名单,否则 401。"""
    # 有效密钥白名单(开源示例密钥)
    valid_keys = {"mlsandbox-oss-3fad186210e530e6b4cc53576cd2723e"}
    # 未命中:401
    if x_api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    # 校验通过:返回 key 供后续按租户隔离查询
    return x_api_key


def ensure_storage_path(path_str: str) -> Path:
    """校验用户提供的存储路径:必须在 STORAGE_PATH 之下且真实存在。"""
    # 解析为绝对路径
    path = Path(path_str).resolve()
    try:
        # 尝试取相对路径做越界检查
        path.relative_to(STORAGE_PATH)
    except ValueError as exc:
        # 越界(路径逃逸)直接拒绝
        raise HTTPException(
            status_code=400,
            detail=f"path must be under {STORAGE_PATH}",
        ) from exc
    # 路径必须已存在(代码/工作目录先行上传)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"path does not exist: {path}")
    # 返回校验后的路径
    return path


def fetch_job_row(job_id: str, api_key: str):
    """按作业 ID+租户 key 查完整作业行(含 result/metrics 大字段)。"""
    # 借 DB 连接
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # 精确匹配作业 ID 与租户
            cursor.execute(
                "SELECT * FROM jobs WHERE job_id = %s AND api_key = %s",
                (job_id, api_key),
            )
            # 返回行(可能 None)
            return cursor.fetchone()


def fetch_job_status_row(job_id: str, api_key: str):
    """按作业 ID+租户查状态行(轻量版:不查 result/metrics 大字段,轮询用)。"""
    # 借 DB 连接
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # 只取状态相关列;result/metrics 用 NULL 占位(减少大对象读取)
            cursor.execute(
                """
                SELECT
                    job_id, name, status, progress, worker_id, api_key,
                    created_at, started_at, updated_at, completed_at,
                    environment,
                    NULL::jsonb AS result,
                    NULL::jsonb AS metrics
                FROM jobs
                WHERE job_id = %s AND api_key = %s
                """,
                (job_id, api_key),
            )
            # 返回行(可能 None)
            return cursor.fetchone()


async def wait_for_job_completion(
    job_id: str,
    api_key: str,
    timeout: int,
    poll_interval: int,
) -> "JobStatus":
    """异步轮询作业直到终态;超时抛 202(仍在运行),404(作业不存在)。"""
    # 取当前事件循环
    loop = asyncio.get_running_loop()
    # 截止时间(monotonic 不受系统时钟调整影响)
    deadline = time.monotonic() + timeout
    # 循环轮询
    while True:
        # 阻塞 DB 查询丢进线程池执行,避免卡住事件循环
        job = await loop.run_in_executor(None, fetch_job_status_row, job_id, api_key)
        # 作业不存在:404
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        # 取当前状态
        status = job["status"]
        # 进入终态:再查一次完整行(补 result/metrics)
        if status in TERMINAL_STATUSES:
            job = await loop.run_in_executor(None, fetch_job_row, job_id, api_key)
            # 完整行也应存在,否则 404
            if not job:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            # 返回序列化后的完整状态
            return serialize_job_row(job)
        # 超过截止时间仍未终态:以 202 返回当前进度(客户端可继续轮询)
        if time.monotonic() >= deadline:
            raise HTTPException(
                status_code=202,
                detail={
                    "job_id": job_id,
                    "status": status,
                    "message": "job is still running",
                },
            )
        # 按配置间隔异步休眠后继续轮询
        await asyncio.sleep(poll_interval)


def serialize_job_row(job: Dict) -> JobStatus:
    """把 DB 作业行转换为 JobStatus 响应模型(时间统一 ISO+Z,JSON 字段还原)。"""
    # 还原 result JSON
    result_data = _load_json(job.get("result"))
    # 还原 metrics JSON
    metrics_data = _load_json(job.get("metrics"))
    # 还原 environment JSON
    environment_data = _load_json(job.get("environment"))
    # 提取客户端任务 ID
    task_id = None
    if isinstance(environment_data, dict):
        task_id = environment_data.get(CLIENT_TASK_KEY)

    # 构造响应模型(可选时间字段 None 透传)
    return JobStatus(
        job_id=job["job_id"],
        name=job["name"],
        status=job["status"],
        progress=job.get("progress", 0),
        worker_id=job.get("worker_id"),
        created_at=job["created_at"].isoformat() + "Z",
        started_at=job.get("started_at").isoformat() + "Z"
        if job.get("started_at")
        else None,
        updated_at=job["updated_at"].isoformat() + "Z",
        completed_at=job.get("completed_at").isoformat() + "Z"
        if job.get("completed_at")
        else None,
        logs_url=f"/api/v1/jobs/{job['job_id']}/logs",
        result=result_data,
        metrics=metrics_data,
        task_id=task_id,
    )


def create_job_entry(job: JobSubmission, api_key: str, trace_id: Optional[str] = None) -> Tuple[str, str, Optional[str], str]:
    """创建作业的完整落盘流程:幂等检查 -> 建目录 -> 落代码 -> 组环境变量 ->
    chmod -> 插库 -> 入 Redis 队列。返回 (job_id, created_at, task_id, status)。"""
    # 必须提供代码内容或代码路径之一
    if job.code is None and not job.code_file_path:
        raise HTTPException(
            status_code=400, detail="Must provide either 'code' or 'code_file_path'"
        )

    # trace_id:外部传入或自动生成(贯穿日志与 DB)
    trace_id = trace_id or uuid.uuid4().hex
    # 记录总耗时起点
    total_start = time.monotonic()
    # 归一化幂等键
    idempotency_key = _normalize_idempotency_key(job.idempotency_key)
    # 启用幂等时计算载荷哈希
    payload_hash = _submit_payload_hash(job) if idempotency_key else None

    # 幂等前置检查:同键已存在则直接复用旧作业
    if idempotency_key:
        existing = fetch_idempotent_job(api_key, idempotency_key)
        if existing:
            # 取已存作业的载荷哈希
            existing_hash = existing.get("idempotency_payload_hash")
            # 同键不同载荷:拒绝(防止误覆盖语义)
            if existing_hash != payload_hash:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "idempotency_key_conflict",
                        "message": "idempotency_key already exists with different payload",
                    },
                )
            # 记录幂等命中埋点
            _obs_log(
                "api.submit.idempotency_hit",
                trace_id=trace_id,
                job_id=existing.get("job_id"),
                idempotency_key=idempotency_key,
                status=existing.get("status"),
            )
            # 直接返回已存作业的信息(不重复入队)
            return (
                existing["job_id"],
                _created_at_for_response(existing),
                _task_id_from_job_row(existing),
                existing.get("status") or "queued",
            )

    # 生成作业 ID(16 位十六进制)与 UTC 创建时间
    job_id = f"job_{uuid.uuid4().hex[:16]}"
    created_at = datetime.utcnow().isoformat() + "Z"

    # 归一化资源类型并校验
    resource_type = (job.resource_type or "cpu").strip().lower()
    if resource_type not in {'cpu', 'gpu'}:
        raise HTTPException(status_code=400, detail="resource_type must be one of 'cpu','gpu'")
    # 归一化优先级(非法值回退 1)
    priority = int(job.priority) if job.priority in (1, 2) else 1
    # 合并用户环境变量
    merged_environment = dict(job.environment or {})
    # data_dir 同步写入 DATA_DIR 环境变量(沙箱约定入口)
    if job.data_dir:
        merged_environment["DATA_DIR"] = job.data_dir
    # 取最终的数据目录
    data_dir_value = merged_environment.get("DATA_DIR")
    # 兼容旧约定:同时提供 SANDBOX_DATA_DIR
    if data_dir_value:
        merged_environment.setdefault("SANDBOX_DATA_DIR", data_dir_value)
    # 注入资源类型/优先级/链路 ID(调度与追踪用)
    merged_environment['RESOURCE_TYPE'] = resource_type
    merged_environment['PRIORITY'] = str(priority)
    merged_environment["TRACE_ID"] = trace_id
    # 数据集标签(目录命名)
    dataset_label = _normalize_dataset_label(data_dir_value)

    # 作业目录:存储根/jobs/日期/数据集标签/作业ID(按日期分区便于归档)
    date_label = datetime.utcnow().strftime("%Y-%m-%d")
    job_dir = STORAGE_PATH / "jobs" / date_label / dataset_label / job_id
    # 旧版目录布局(无日期分区)注释保留备查
    # job_dir = STORAGE_PATH / "jobs" / dataset_label / job_id
    # 代码目录
    code_root = job_dir / "code"
    # 输出目录
    outputs_dir = job_dir / "outputs"
    # 阶段计时:建目录
    stage_start = time.monotonic()
    code_root.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    _stage_log("submit.mkdir", stage_start, trace_id=trace_id, job_id=job_id)

    # 工作目录源(需校验在存储根内)
    working_src = None
    if job.working_dir:
        working_src = ensure_storage_path(job.working_dir)

    # 记录提交代码的规模与哈希(审计/重复检测)
    submitted_code_chars = None
    submitted_code_sha256 = None
    # 阶段计时:写代码
    stage_start = time.monotonic()
    # 分支一:内联代码 -> 写 main.py
    if job.code is not None:
        # 空值兜底为空串
        code_text = job.code or ""
        # 记录字符数
        submitted_code_chars = len(code_text)
        # 计算代码 SHA256
        submitted_code_sha256 = hashlib.sha256(code_text.encode("utf-8")).hexdigest()
        # 目标文件 main.py
        code_path = code_root / "main.py"
        # 写出代码
        code_path.write_text(code_text, encoding="utf-8")
    else:
        # 分支二:存储上的代码路径,校验后安置进作业目录
        source_path = ensure_storage_path(job.code_file_path)
        code_path = _relocate_source_to_job(source_path, code_root)
    _stage_log("submit.write_code", stage_start, trace_id=trace_id, job_id=job_id)

    # 决定最终执行工作目录:上传目录的移动后路径 / 原目录 / 代码所在目录
    if working_src:
        if _is_relative_to(working_src, UPLOAD_ROOT):
            # 上传目录:换算为移动到 code_root 后的对应路径
            relative = working_src.relative_to(UPLOAD_ROOT)
            moved_root = code_root / relative.parts[0]
            if len(relative.parts) > 1:
                final_working_dir = moved_root.joinpath(*relative.parts[1:])
            else:
                final_working_dir = moved_root
        else:
            # 普通存储目录:原样使用
            final_working_dir = working_src
    else:
        # 未指定工作目录:退回代码所在目录
        final_working_dir = code_path.parent

    # 注入作业路径族环境变量(沙箱内定位产物)
    merged_environment.setdefault("JOB_ID", job_id)
    merged_environment.setdefault("JOB_DIR", str(job_dir))
    merged_environment.setdefault("JOB_CODE_DIR", str(code_root))
    merged_environment.setdefault("JOB_OUTPUT_DIR", str(outputs_dir))
    merged_environment["JOB_DATASET_LABEL"] = dataset_label
    merged_environment["JOB_STORAGE_DIR"] = str(job_dir)
    merged_environment["EXECUTION_WORKDIR"] = str(final_working_dir)
    # shell 模式的自定义命令
    if job.execution_command:
        merged_environment["EXECUTION_COMMAND"] = job.execution_command
    # 客户端任务 ID 写入 environment(查询时回显)
    if job.task_id:
        merged_environment[CLIENT_TASK_KEY] = job.task_id

    # 阶段计时:chmod
    stage_start = time.monotonic()
    # 作业目录整体开放读写(worker 用户与 API 用户不同)
    _make_world_writable(job_dir)
    _stage_log("submit.chmod", stage_start, trace_id=trace_id, job_id=job_id)

    # 组装入队消息体(worker 消费所需全部信息)
    job_data = {
        "job_id": job_id,
        "name": job.name,
        "code_file_path": str(code_path),
        "requirements": job.requirements,
        "data_paths": job.data_paths,
        "data_dir": data_dir_value,
        "gpu_count": job.gpu_count,
        "timeout": job.timeout,
        "environment": merged_environment,
        "status": "queued",
        "resource_type": resource_type,
        "priority": priority,
        "created_at": created_at,
        "api_key": api_key,
        "dataset_label": dataset_label,
        "job_storage_dir": str(job_dir),
        "trace_id": trace_id,
        "idempotency_key": idempotency_key,
        "idempotency_payload_hash": payload_hash,
        "working_dir": job.working_dir,
    }
    # data_dir 显式存在时确保写入消息体
    if job.data_dir:
        job_data["data_dir"] = job.data_dir
    # task_id 显式存在时写入消息体
    if job.task_id:
        job_data["task_id"] = job.task_id

    # 阶段计时:DB 插入
    stage_start = time.monotonic()
    try:
        # 插入 jobs 表(幂等键+哈希一并入库;唯一约束兜底并发幂等)
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO jobs (
                        job_id, name, code_file_path, requirements, data_paths,
                        gpu_count, timeout, environment, status, created_at, api_key,
                        idempotency_key, idempotency_payload_hash,
                        submitted_code_chars, submitted_code_sha256
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        job_id,
                        job.name,
                        str(code_path),
                        json.dumps(job.requirements),
                        json.dumps(job.data_paths),
                        job.gpu_count,
                        job.timeout,
                        json.dumps(merged_environment),
                        "queued",
                        created_at,
                        api_key,
                        idempotency_key,
                        payload_hash,
                        submitted_code_chars,
                        submitted_code_sha256,
                    ),
                )
            # 提交事务
            conn.commit()
    except pg_errors.UniqueViolation:
        # 唯一约束冲突:并发提交同一幂等键(竞态)
        if idempotency_key:
            # 查出先成功的那条
            existing = fetch_idempotent_job(api_key, idempotency_key)
            # 载荷一致:视作幂等命中,复用旧作业
            if existing and existing.get("idempotency_payload_hash") == payload_hash:
                # 清理本请求刚建的目录(尽力而为)
                with contextlib.suppress(Exception):
                    shutil.rmtree(job_dir)
                # 记录竞态命中埋点
                _obs_log(
                    "api.submit.idempotency_race_hit",
                    trace_id=trace_id,
                    job_id=existing.get("job_id"),
                    idempotency_key=idempotency_key,
                )
                # 返回先成功作业的信息
                return (
                    existing["job_id"],
                    _created_at_for_response(existing),
                    _task_id_from_job_row(existing),
                    existing.get("status") or "queued",
                )
        # 载荷不一致或其他唯一冲突:409
        raise HTTPException(status_code=409, detail={"error": "idempotency_key_conflict"})
    _stage_log("submit.db_insert", stage_start, trace_id=trace_id, job_id=job_id)

    # 按资源类型+优先级选定队列
    queue_name = JOB_QUEUE_BY_RESOURCE_PRIORITY[resource_type][priority]
    # 阶段计时:Redis 入队
    stage_start = time.monotonic()
    # 消息推入队尾(worker 从队头取,先到先得)
    redis_client.rpush(queue_name, json.dumps(job_data))
    _stage_log("submit.redis_rpush", stage_start, trace_id=trace_id, job_id=job_id, queue=queue_name)
    # 记录提交总耗时
    _stage_log("submit.total", total_start, trace_id=trace_id, job_id=job_id)

    # 返回 (作业ID, 创建时间, 客户端任务ID, 初始状态)
    return job_id, created_at, job.task_id, "queued"


@app.get("/health")
def health_check():
    """健康检查:同时探测 Redis 与 PostgreSQL 连通性,任一失败返回 unhealthy。"""
    try:
        # Redis ping
        redis_client.ping()
        # 借一次 DB 连接(能借到即 PG 可用)
        with get_db_connection():
            pass
        # 两项都正常
        return {"status": "healthy", "redis": "connected", "database": "connected"}
    except Exception as exc:
        # 任一组件异常:上报 unhealthy 与错误信息
        return {"status": "unhealthy", "error": str(exc)}


@app.post("/api/v1/jobs", response_model=JobResponse)
def submit_job(
    job: JobSubmission,
    api_key: str = Depends(verify_api_key),
    x_trace_id: Optional[str] = Header(None),
):
    """提交作业(异步模式):立即入队并返回作业 ID,不等待完成。"""
    # 链路 ID:请求头携带或自动生成
    trace_id = x_trace_id or uuid.uuid4().hex
    # 持准入槽执行创建(限流保护)
    with submit_admission_slot():
        job_id, created_at, task_id, status = create_job_entry(job, api_key, trace_id=trace_id)
    # 返回作业响应
    return JobResponse(
        job_id=job_id,
        status=status,
        created_at=created_at,
        estimated_start=None,
        task_id=task_id,
    )


@app.post("/api/v1/jobs/submit_and_wait", response_model=JobStatus)
async def submit_and_wait(
    job: JobSubmission,
    api_key: str = Depends(verify_api_key),
    x_trace_id: Optional[str] = Header(None),
    timeout: Optional[int] = None,
    poll_interval: Optional[int] = None,
):
    """提交并等待完成:入队后同步轮询到终态(或 202 超时返回)。"""
    # 链路 ID
    trace_id = x_trace_id or uuid.uuid4().hex
    # 持准入槽提交
    with submit_admission_slot():
        job_id, _, _, _ = create_job_entry(job, api_key, trace_id=trace_id)
    # 等待超时:显式参数或默认值
    wait_timeout = timeout if timeout is not None else DEFAULT_WAIT_TIMEOUT
    # 轮询间隔:显式参数或默认值
    poll = poll_interval if poll_interval is not None else DEFAULT_POLL_INTERVAL
    try:
        # 轮询到终态并返回完整状态
        return await wait_for_job_completion(job_id, api_key, wait_timeout, poll)
    except HTTPException as exc:
        # 202(仍在运行)时补全 detail 中的 job_id,便于客户端继续查
        if exc.status_code == 202 and isinstance(exc.detail, dict):
            exc.detail.setdefault("job_id", job_id)
        # 原样向上抛
        raise


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str, api_key: str = Depends(verify_api_key)):
    """查询单个作业状态;终态时补查完整行(含 result/metrics)。"""
    # 先查轻量状态行
    job = fetch_job_status_row(job_id, api_key)
    # 不存在:404
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    # 终态作业:再查完整行补齐 result/metrics
    if job["status"] in TERMINAL_STATUSES:
        job = fetch_job_row(job_id, api_key)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

    # 序列化返回
    return serialize_job_row(job)


@app.get("/api/v1/jobs/{job_id}/logs")
def get_job_logs(job_id: str, api_key: str = Depends(verify_api_key)):
    """获取作业日志:优先返回 DB result.run_log,否则回退到磁盘 logs.txt 流式输出。"""
    # 查完整作业行
    job = fetch_job_row(job_id, api_key)
    # 不存在:404
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # 还原 result 字段
    result_data = _load_json(job.get("result"))
    # result.run_log 存在且非空:直接作为文本流返回(最新口径)
    if isinstance(result_data, dict):
        run_log = result_data.get("run_log")
        if isinstance(run_log, str) and run_log.strip():
            return StreamingResponse(iter([run_log]), media_type="text/plain")

    # 还原 environment 定位磁盘日志目录
    environment_data = _load_json(job.get("environment"))
    log_dir: Optional[Path] = None
    if isinstance(environment_data, dict):
        # 优先用记录的存储目录
        storage_dir = environment_data.get("JOB_STORAGE_DIR")
        if storage_dir:
            log_dir = Path(storage_dir)
        else:
            # 兼容旧数据:按数据集标签重建目录
            dataset_label = environment_data.get("JOB_DATASET_LABEL")
            if dataset_label:
                log_dir = STORAGE_PATH / "jobs" / dataset_label / job_id
    # 均未命中:旧版扁平目录兜底
    if log_dir is None:
        log_dir = STORAGE_PATH / "jobs" / job_id

    # 磁盘日志文件路径
    log_file = log_dir / "logs.txt"
    # 文件尚未生成:返回提示
    if not log_file.exists():
        return {"logs": "No logs available yet"}

    def iter_file():
        # 生成器:逐行惰性读取日志文件(大日志不占内存)
        with log_file.open("r", encoding="utf-8") as fp:
            for line in fp:
                yield line

    # 以纯文本流式返回
    return StreamingResponse(iter_file(), media_type="text/plain")


@app.get("/api/v1/jobs")
def list_jobs(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    api_key: str = Depends(verify_api_key),
):
    """列出本租户的作业(可按状态过滤,按创建时间倒序分页)。"""
    # 借 DB 连接
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # 基础查询:按租户过滤
            query = "SELECT * FROM jobs WHERE api_key = %s"
            # 参数列表(防注入)
            params = [api_key]
            # 可选状态过滤
            if status:
                query += " AND status = %s"
                params.append(status)
            # 排序与分页
            query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            # 执行查询
            cursor.execute(query, params)
            # 取全部行
            jobs = cursor.fetchall()

    # 返回作业列表与数量
    return {"jobs": jobs, "count": len(jobs)}


@app.delete("/api/v1/jobs/{job_id}")
def cancel_job(job_id: str, api_key: str = Depends(verify_api_key)):
    """取消作业:仅 queued/running 可取消;再把消息从 Redis 队列移除并打取消标记。"""
    # 借 DB 连接做状态更新
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # 只允许取消排队中/运行中的作业;租户隔离
            cursor.execute(
                """
                UPDATE jobs
                SET status = 'cancelled', updated_at = %s
                WHERE job_id = %s AND api_key = %s AND status IN ('queued', 'running')
                """,
                (datetime.utcnow().isoformat() + "Z", job_id, api_key),
            )
            # 未更新到任何行:404(不存在或已终态)
            if cursor.rowcount == 0:
                raise HTTPException(
                    status_code=404, detail="Job not found or cannot be cancelled"
                )
            # 提交更新
        conn.commit()

    # 从 Redis 队列中移除排队消息(best effort,异常不阻断)
    removed = False
    try:
        # 遍历四类队列
        queue_keys = [JOB_QUEUE_CPU_P1, JOB_QUEUE_CPU_P2, JOB_QUEUE_GPU_P1, JOB_QUEUE_GPU_P2]
        for queue_key in queue_keys:
            # 读出整个队列
            queue_items = redis_client.lrange(queue_key, 0, -1)
            # 逐条解析找目标作业
            for raw in queue_items:
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    # 非法消息跳过
                    continue
                # 命中目标作业:按值精确移除一条并跳出
                if item.get("job_id") == job_id:
                    redis_client.lrem(queue_key, 1, raw)
                    removed = True
                    break
            # 已移除则不再遍历其他队列
            if removed:
                break
        # 写取消标记(1 小时过期),worker 收到 running 作业时可据此快速中止
        redis_client.setex(f"job:{job_id}:cancelled", 3600, "1")
    except Exception:
        # Redis 操作失败不影响已提交的 DB 取消状态
        pass

    # 返回取消结果(是否从队列移除)
    return {"message": "Job cancelled successfully", "removed_from_queue": removed}


@app.get("/api/v1/workers/status")
def get_worker_status():
    """汇总 worker 集群状态:心跳/状态/资源类型,以及各类计数摘要。"""
    # worker 信息列表
    workers = []
    # 扫描所有 worker 心跳键
    for key in redis_client.scan_iter("worker:*:heartbeat"):
        # 从键名解析 worker ID
        worker_id = key.split(":")[1]
        # 心跳时间
        heartbeat = redis_client.get(key)
        # 状态键
        status_key = f"worker:{worker_id}:status"
        # 状态(缺失记 unknown)
        status = redis_client.get(status_key) or "unknown"
        # 服务端点
        endpoint = redis_client.get(f"worker:{worker_id}:endpoint")
        # 资源类型(统一小写)
        resource_type = (redis_client.get(f"worker:{worker_id}:resource_type") or "unknown").lower()
        # 汇总该 worker 信息
        workers.append(
            {
                "worker_id": worker_id,
                "status": status,
                "endpoint": endpoint,
                "resource_type": resource_type,
                "last_heartbeat": heartbeat,
            }
        )

    # 对外展示子集:按 endpoint 排序(便于比对)
    workers_show = sorted(
        (
            {
                "endpoint": w.get("endpoint"),
                "resource_type": w.get("resource_type"),
                "status": w.get("status"),
            }
            for w in workers
        ),
        key=lambda x: x.get("endpoint") or "",
    )

    # 各类计数
    total = len(workers_show)
    # CPU/GPU 总数
    cpu_total = sum(1 for w in workers_show if w.get("resource_type") == "cpu")
    gpu_total = sum(1 for w in workers_show if w.get("resource_type") == "gpu")
    # 空闲数
    cpu_idle = sum(1 for w in workers_show if w.get("resource_type") == "cpu" and w.get("status") == "idle")
    gpu_idle = sum(1 for w in workers_show if w.get("resource_type") == "gpu" and w.get("status") == "idle")
    # 隔离数(故障/禁用节点)
    cpu_quarantined = sum(
        1 for w in workers_show if w.get("resource_type") == "cpu" and w.get("status") == "quarantined"
    )
    gpu_quarantined = sum(
        1 for w in workers_show if w.get("resource_type") == "gpu" and w.get("status") == "quarantined"
    )

    # 计数摘要字典
    summary = {
        "total": total,
        "cpu_total": cpu_total,
        "gpu_total": gpu_total,
        "cpu_idle": cpu_idle,
        "gpu_idle": gpu_idle,
        # 忙碌 = 总数 - 空闲 - 隔离(下限 0 防异常数据为负)
        "cpu_busy": max(0, cpu_total - cpu_idle - cpu_quarantined),
        "gpu_busy": max(0, gpu_total - gpu_idle - gpu_quarantined),
        "cpu_quarantined": cpu_quarantined,
        "gpu_quarantined": gpu_quarantined,
    }

    # 人类可读的一行统计串
    stats = {
        f"total:{total}, cpu_total:{cpu_total}, gpu_total:{gpu_total}; "
        f"cpu_idle:{cpu_idle}/{cpu_total}, gpu_idle:{gpu_idle}/{gpu_total}; "
        f"cpu_quarantined:{cpu_quarantined}/{cpu_total}, gpu_quarantined:{gpu_quarantined}/{gpu_total}"
    }

    # 返回三段结构
    return {
        "workers_show": workers_show,
        "stats": stats,
        "summary": summary,
    }



@app.post("/api/v1/uploads")
def upload_code_archive(
    file: UploadFile = File(...),
    extract: bool = True,
    api_key: str = Depends(verify_api_key),
):
    """上传代码归档:保存文件、可选解压、探测默认入口与文件清单。"""
    # 生成上传 ID
    upload_id = f"upload_{uuid.uuid4().hex[:16]}"
    # 上传目标目录
    dest_dir = UPLOAD_ROOT / upload_id
    # 创建目录
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 归档保存路径
    archive_path = dest_dir / file.filename
    # 流式写盘(不整块读入内存)
    with archive_path.open("wb") as fp:
        shutil.copyfileobj(file.file, fp)

    # 解压根目录(未解压时即上传目录)
    extraction_root = dest_dir
    # 是否成功解压
    extracted = False
    # extract=True 才尝试解压
    if extract:
        try:
            # shutil 按扩展名自动识别 zip/tar 等格式解压
            shutil.unpack_archive(str(archive_path), str(dest_dir))
            extracted = True
            # 解压成功后删除归档本体(missing_ok 容忍并发删除)
            archive_path.unlink(missing_ok=True)
        except (shutil.ReadError, ValueError):
            # 非归档文件/未知格式:保留原文件不解压
            pass

    # 解压成功后做"单顶层目录扁平化"探测
    if extracted:
        # 列出顶层条目(忽略 macOS 压缩附带的 __MACOSX)
        entries = [
            p for p in dest_dir.iterdir() if not p.name.startswith("__MACOSX")
        ]
        # 顶层只有一个目录:认为它是工程根
        if len(entries) == 1 and entries[0].is_dir():
            extraction_root = entries[0]

    # 探测默认入口脚本(按常见命名优先级)
    default_entry = None
    for candidate in ("main.py", "train.py", "run.py", "run.sh"):
        candidate_path = extraction_root / candidate
        # 第一个命中的即默认入口
        if candidate_path.exists():
            default_entry = candidate_path
            break

    # 采样最多 50 个相对路径,供客户端预览归档内容
    sample_files: List[str] = []
    try:
        for path in extraction_root.glob("**/*"):
            # 计算相对路径
            rel = path.relative_to(extraction_root)
            # 未满 50 条则记录
            if rel.parts and len(sample_files) < 50:
                sample_files.append(str(rel))
            # 满 50 条提前结束
            if len(sample_files) >= 50:
                break
    except OSError:
        # 遍历失败不阻断(返回部分结果)
        pass

    # 返回上传信息:ID、根路径、默认入口、文件采样
    return {
        "upload_id": upload_id,
        "base_path": str(extraction_root),
        "default_entry": str(default_entry) if default_entry else None,
        "sample_files": sample_files,
    }
