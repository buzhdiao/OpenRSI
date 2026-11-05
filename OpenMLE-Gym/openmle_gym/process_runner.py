# =====================================================================
# 中文注释导读（OpenRSI / OpenMLE-Gym 学习文档）：
# 本文件是 OpenMLE-Gym 的"进程隔离执行器"：把每个任务（构建 build / 准备 prepare /
# 指标计算 metric / 评估 evaluate）放进一次性子进程甚至容器里运行。
# 安全设计（对应论文中"可验证任务环境"的隔离要求）：
#   - process 模式：独立 Python 子进程 + request/result JSON 文件通信，
#     子进程崩溃不拖垮主进程；
#   - isolated 模式：docker/podman 容器，断网、只读根文件系统、受限
#     CPU/内存/PID、非特权用户、no-new-privileges、/tmp 无执行位；
#   - 环境变量白名单转发：API 密钥只按需透传（prepare/metric 阶段一律不给）。
# 通信协议：父进程写 request.json -> 子进程 task_worker 执行 -> 写 result.json
# -> 父进程读取信封 {ok, result, error}。全程"不抛异常"，失败统一封装为 Outcome。
# =====================================================================
from __future__ import annotations

import os  # 环境变量与进程组操作
import shutil  # 探测 docker/podman 可执行文件
import signal  # SIGTERM/SIGKILL 超时终止
import subprocess  # 子进程启动与通信
import sys  # process 模式使用当前解释器
import tempfile  # 一次性请求/结果临时目录
from dataclasses import dataclass  # Outcome 不可变结果容器
from pathlib import Path
from typing import Any, Iterable

from .common import atomic_write_json  # 原子写 JSON（避免读到半截文件）


# 允许转发进任务子进程的环境变量白名单：LLM 与 Kaggle 凭据按需透传，
# 白名单之外的一律不进子进程（减少凭据泄漏面）
FORWARDED_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_COMPAT_API_KEY",
    "ANTHROPIC_MODEL",
    "KAGGLE_CONFIG_DIR",
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "MODEL",
    "OPENMLE_BUILD_LLM_API_KEY",
    "OPENMLE_BUILD_LLM_BASE_URL",
    "OPENMLE_BUILD_LLM_MODEL",
    "OPENMLE_EVAL_LLM_API_KEY",
    "OPENMLE_EVAL_LLM_BASE_URL",
    "OPENMLE_EVAL_LLM_MODEL",
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
)


@dataclass(frozen=True)
class TaskProcessOutcome:
    """一次任务进程执行的最终结果（永不抛异常，失败信息集中在 error 字段）。"""

    ok: bool  # 是否成功
    result: Any = None  # 成功时的业务结果（来自 result.json 信封）
    error: str | None = None  # 失败原因（超时/非零退出/坏 JSON/worker 报错）
    stdout: str = ""  # 子进程标准输出（转发给上层日志）
    stderr: str = ""  # 子进程标准错误
    returncode: int | None = None  # 子进程退出码


def _load_result_envelope(path: Path) -> dict[str, Any]:
    # 读取子进程写出的结果信封；必须是 JSON 对象否则视为协议违规
    import json

    envelope = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(envelope, dict):
        raise TypeError("Task result must be a JSON object")
    return envelope


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    # 两阶段终止：先对整个进程组发 SIGTERM（优雅退出，5 秒宽限），
    # 仍不退出则 SIGKILL 强杀——保证超时任务不留孤儿进程
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()


def _container_command(
    request_path: Path,
    result_path: Path,
    operation: str,
    readonly_paths: Iterable[Path],
    writable_paths: Iterable[Path],
) -> list[str]:
    # 构造容器（isolated）模式的完整启动命令：多重沙箱加固
    runtime = shutil.which("docker") or shutil.which("podman")  # 两种容器运行时皆可
    image = os.environ.get("OPENMLE_GYM_ISOLATED_IMAGE")  # 必须显式指定隔离镜像
    if runtime is None:
        raise RuntimeError("isolated execution requires docker or podman")
    if not image:
        raise RuntimeError("isolated execution requires OPENMLE_GYM_ISOLATED_IMAGE")

    # 容器内以当前用户（或 nobody）运行，避免 root 提权
    uid = os.getuid() or 65534
    gid = os.getgid() or 65534
    command = [
        runtime,
        "run",
        "--rm",  # 退出即清理容器
        "--network",
        "none",  # 断网：任务无法外联
        "--read-only",  # 根文件系统只读
        "--pids-limit",
        "128",  # 防 fork 炸弹
        "--memory",
        os.environ.get("OPENMLE_GYM_MEMORY_LIMIT", "8g"),  # 内存上限（可用环境变量调整）
        "--cpus",
        os.environ.get("OPENMLE_GYM_CPU_LIMIT", "2"),  # CPU 上限
        "--user",
        f"{uid}:{gid}",  # 非特权用户
        "--workdir",
        "/tmp",
        "--cap-drop",
        "ALL",  # 丢弃全部 Linux capabilities
        "--security-opt",
        "no-new-privileges",  # 禁止提权
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=1g",  # /tmp 可写但不可执行、不可设 suid
    ]
    mounted: set[tuple[str, str]] = set()

    def mount(path: Path, mode: str) -> None:
        # 去重挂载：同一 (路径, 模式) 只挂一次
        resolved = path.resolve()
        key = (str(resolved), mode)
        if key in mounted:
            return
        mounted.add(key)
        command.extend(["--volume", f"{resolved}:{resolved}:{mode}"])

    mount(request_path.parent, "rw")  # 请求/结果目录必须可写
    for path in readonly_paths:
        mount(path, "ro")  # 白名单只读路径
    for path in writable_paths:
        mount(path, "rw")  # 白名单可写路径（如输出目录）
    forwarded_env = () if operation in {"prepare", "metric"} else FORWARDED_ENV
    # prepare/metric 阶段不透传任何凭据；其余阶段只转发白名单中已设置的变量
    for name in forwarded_env:
        if os.environ.get(name):
            command.extend(["--env", name])
    command.extend(
        [
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",  # 不写 .pyc（配合只读根文件系统）
            image,
            "python",
            "-m",
            "openmle_gym.task_worker",  # 与 process 模式同一个 worker 入口
            operation,
            str(request_path),
            str(result_path),
        ]
    )
    return command


def run_task_process(
    operation: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    execution_mode: str = "process",
    readonly_paths: Iterable[str | Path] = (),
    writable_paths: Iterable[str | Path] = (),
) -> TaskProcessOutcome:
    """Run one task in a disposable process and return a non-throwing outcome."""
    # 【任务进程执行总入口】process / isolated 两种模式；任何失败都折叠为 Outcome
    if execution_mode not in {"process", "isolated"}:
        return TaskProcessOutcome(ok=False, error=f"Unknown execution mode: {execution_mode}")

    source_root = Path(__file__).resolve().parent.parent
    # 一次性临时目录承载 request/result，用完即毁
    with tempfile.TemporaryDirectory(prefix="openmle-task-") as temporary_dir:
        temporary_root = Path(temporary_dir)
        request_path = temporary_root / "request.json"
        result_path = temporary_root / "result.json"
        # 原子写入请求载荷，子进程读到的一定是完整 JSON
        atomic_write_json(request_path, payload)

        try:
            if execution_mode == "isolated":
                command = _container_command(
                    request_path,
                    result_path,
                    operation,
                    [Path(path) for path in readonly_paths],
                    [Path(path) for path in writable_paths],
                )
            else:
                # process 模式：当前解释器直接跑 task_worker 模块
                command = [
                    sys.executable,
                    "-m",
                    "openmle_gym.task_worker",
                    operation,
                    str(request_path),
                    str(result_path),
                ]
        except Exception as exc:
            return TaskProcessOutcome(
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            process = subprocess.Popen(
                command,
                cwd=source_root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,  # 新会话 => 独立进程组，便于整组终止
            )
            try:
                # 带超时等待子进程结束并回收输出
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                # 超时：终止整个进程组后回收残余输出
                _terminate_process_group(process)
                stdout, stderr = process.communicate()
                return TaskProcessOutcome(
                    ok=False,
                    error=f"Task timed out after {timeout:g}s",
                    stdout=stdout,
                    stderr=stderr,
                    returncode=process.returncode,
                )
        except Exception as exc:
            return TaskProcessOutcome(
                ok=False,
                error=f"Failed to start task process: {type(exc).__name__}: {exc}",
            )

        if process.returncode != 0:
            return TaskProcessOutcome(
                ok=False,
                error=f"Task process exited with code {process.returncode}",
                stdout=stdout,
                stderr=stderr,
                returncode=process.returncode,
            )
        if not result_path.is_file():
            # 退出码为 0 却没写结果文件：协议违规
            return TaskProcessOutcome(
                ok=False,
                error="Task process did not produce a result file",
                stdout=stdout,
                stderr=stderr,
                returncode=process.returncode,
            )
        try:
            envelope = _load_result_envelope(result_path)
        except Exception as exc:
            return TaskProcessOutcome(
                ok=False,
                error=f"Invalid task result JSON: {type(exc).__name__}: {exc}",
                stdout=stdout,
                stderr=stderr,
                returncode=process.returncode,
            )
        if not isinstance(envelope, dict) or envelope.get("ok") is not True:
            # worker 自报失败：透传其 error 字段
            error = envelope.get("error") if isinstance(envelope, dict) else None
            return TaskProcessOutcome(
                ok=False,
                error=str(error or "Task worker reported failure"),
                stdout=stdout,
                stderr=stderr,
                returncode=process.returncode,
            )
        # 全部检查通过：返回业务结果
        return TaskProcessOutcome(
            ok=True,
            result=envelope.get("result"),
            stdout=stdout,
            stderr=stderr,
            returncode=process.returncode,
        )
