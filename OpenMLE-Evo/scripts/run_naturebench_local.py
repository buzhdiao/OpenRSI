#!/usr/bin/env python3
# 中文说明:本脚本是 NatureBench "纯本地"评估的一键启动器(launcher):
# 负责定位 NatureBench 仓库、解析本地运行时 python(显式解释器或 conda 环境)、
# 下载/校验任务包、探测运行时可用的科学计算包、按需拉起 SSH 模型隧道或直连
# OpenAI 兼容模型服务、启动(或复用)NatureBench 评估服务,
# 最后拼装 hydra 命令调用 scripts/evaluate_naturebench.py 完成端到端评估。
# 位置:OpenMLE-Evo/scripts 下;所有子进程失败都会快速失败并回收进程组。

# 延迟注解:类型标注运行期不求值
from __future__ import annotations

# argparse:命令行参数解析
import argparse
# json:解析健康检查/模型列表响应、构造 hydra 列表
import json
# os:环境变量读写、进程组终止
import os
# signal:SIGTERM/SIGKILL 信号
import signal
# socket:抢占空闲回环端口
import socket
# subprocess:管理下载/服务/隧道/评估等子进程
import subprocess
# sys:当前解释器路径
import sys
# time:健康检查轮询的休眠与截止时间
import time
# urllib.error:区分网络类异常
import urllib.error
# urllib.request:HTTP 健康检查与模型列表获取
import urllib.request
# 带时区的时间戳(实验目录命名)
from datetime import datetime, timezone
# Path:路径构造
from pathlib import Path

# 仓库根目录(OpenMLE-Evo)
REPO_ROOT = Path(__file__).resolve().parents[1]
# 默认任务集文件:benchmarks/naturebench_local_quick/tasks.txt(快速示例)
DEFAULT_TASK_SET = REPO_ROOT / "benchmarks" / "naturebench_local_quick" / "tasks.txt"
# 默认快速任务 ID(反事实统计任务)
DEFAULT_QUICK_TASK = "s42256-023-00611-x"
# 候选可用的本地科学计算包清单:包名(pip) -> import 模块名
LOCAL_PACKAGE_IMPORTS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "h5py": "h5py",
    "statsmodels": "statsmodels",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "anndata": "anndata",
    "scanpy": "scanpy",
    "umap-learn": "umap",
    "loompy": "loompy",
    "networkx": "networkx",
    "igraph": "igraph",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
}
# 必需包子集:缺一不可,缺失即拒绝启动
REQUIRED_LOCAL_PACKAGES = {"numpy", "scipy", "pandas", "scikit-learn"}
# 依赖探测脚本的输出标记:便于从 stdout 中定位 JSON 结果行
PACKAGE_INSPECTION_MARKER = "__NATUREBENCH_LOCAL_PACKAGES__="


def _read_task_set(path: Path) -> list[str]:
    """读取任务集文件:忽略空行与 # 注释行;空集或含重复 ID 都视为配置错误。"""
    # 逐行去空白,过滤空行与注释行
    tasks = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    # 空任务集无法评估,直接报错
    if not tasks:
        raise ValueError(f"NatureBench task set is empty: {path}")
    # 出现重复任务 ID 说明清单写错,提前报错
    if len(tasks) != len(set(tasks)):
        raise ValueError(f"NatureBench task set contains duplicate ids: {path}")
    # 返回任务 ID 列表
    return tasks


def _resolve_naturebench_repo(value: str | None) -> Path:
    """定位 NatureBench 仓库:按 CLI 参数 -> NATUREBENCH_REPO 环境变量 -> 常见路径。

    有效的仓库必须同时包含 run_naturebench.py 与 eval_service.py 两个入口脚本。
    """
    # 候选路径列表
    candidates = []
    # CLI 参数显式指定(展开 ~)
    if value:
        candidates.append(Path(value).expanduser())
    # 环境变量指定
    env_value = os.environ.get("NATUREBENCH_REPO")
    if env_value:
        candidates.append(Path(env_value).expanduser())
    # 常见约定路径:仓库同级目录与当前工作目录下的 NatureBench
    candidates.extend([REPO_ROOT.parent / "NatureBench", Path.cwd() / "NatureBench"])
    # 逐个候选校验两个入口脚本是否存在
    for candidate in candidates:
        # 解析为绝对路径
        resolved = candidate.resolve()
        # 两个入口都在才算有效仓库
        if (resolved / "run_naturebench.py").is_file() and (
            resolved / "eval_service.py"
        ).is_file():
            # 返回该仓库路径
            return resolved
    # 全部候选失败:列出检查过的路径并报错,提示两种指定方式
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not locate the NatureBench repository. Pass --naturebench-repo "
        f"or set NATUREBENCH_REPO. Checked: {rendered}"
    )


def _resolve_runtime_python(
    *,
    local_python: str | None,
    conda_env: str | None,
    conda_executable: str,
) -> Path:
    """解析评估实际使用的 python 解释器。

    优先级:显式 --local-python > conda 环境(剥离外层 venv 影响)> 当前解释器。
    """
    # 情况一:显式指定解释器路径
    if local_python:
        # 展开 ~ 并取绝对路径
        path = Path(os.path.abspath(Path(local_python).expanduser()))
        # 路径必须是真实文件
        if not path.is_file():
            raise FileNotFoundError(f"Local Python does not exist: {path}")
        # 返回该解释器
        return path
    # 情况二:未指定 conda 环境,直接用当前解释器
    if not conda_env:
        return Path(sys.executable).absolute()

    # 情况三:conda 环境——先复制环境变量
    conda_environment = os.environ.copy()
    # 弹出外层虚拟环境的 VIRTUAL_ENV,避免 conda run 时被外层 venv 激活影响
    outer_virtualenv = conda_environment.pop("VIRTUAL_ENV", None)
    # 若存在外层 venv:同步把它加入 PATH 的那段 bin 目录移除
    if outer_virtualenv:
        # 计算 venv 的 bin 目录绝对形式
        virtualenv_bin = os.path.normpath(str(Path(outer_virtualenv) / "bin"))
        # 拆分 PATH 各段
        path_entries = conda_environment.get("PATH", "").split(os.pathsep)
        # 过滤掉 venv 的 bin 段后重新拼接
        conda_environment["PATH"] = os.pathsep.join(
            entry for entry in path_entries if os.path.normpath(entry) != virtualenv_bin
        )
    # 在目标 conda 环境中运行一句 python,打印其解释器真实路径
    process = subprocess.run(
        [
            conda_executable,
            "run",
            "--no-capture-output",
            "-n",
            conda_env,
            "python",
            "-c",
            "import sys; print(sys.executable)",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=conda_environment,
    )
    # conda 命令失败:环境不存在或损坏,给出创建指引
    if process.returncode != 0:
        # 取 stderr/stdout 尾部作为细节
        detail = (process.stderr or process.stdout or "").strip()
        raise RuntimeError(
            f"Conda environment {conda_env!r} is unavailable: {detail}. "
            "Create it with: conda env create -f environments/naturebench-local.yml"
        )
    # 取 stdout 中的非空行(conda 可能有额外输出)
    lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    # 没有任何输出:无法确定解释器路径
    if not lines:
        raise RuntimeError(
            f"Could not resolve Python for Conda environment {conda_env!r}"
        )
    # 最后一行即解释器路径(避免 conda 前置横幅干扰)
    path = Path(os.path.abspath(Path(lines[-1]).expanduser()))
    # 解析出的路径必须真实存在
    if not path.is_file():
        raise FileNotFoundError(f"Resolved Conda Python does not exist: {path}")
    # 返回 conda 环境解释器
    return path


def _health_url(host: str, port: int) -> str:
    """构造评估服务的健康检查 URL。"""
    # /health 端点
    return f"http://{host}:{port}/health"


def _service_is_healthy(host: str, port: int) -> bool:
    """探测评估服务是否健康(HTTP 200 且返回 status=ok);任何异常都视为不健康。"""
    try:
        # 请求健康端点,1 秒超时
        with urllib.request.urlopen(_health_url(host, port), timeout=1) as response:
            # 解析响应 JSON
            payload = json.loads(response.read().decode("utf-8"))
        # 状态码 200 且业务状态为 ok 才算健康
        return response.status == 200 and payload.get("status") == "ok"
    except (OSError, ValueError, urllib.error.URLError):
        # 网络/解析异常一律视为服务不可用
        return False


def _eval_service_supports_control_token(naturebench_repo: Path) -> bool:
    """通过源码探测当前 NatureBench eval_service 是否支持控制令牌参数。"""
    # 读取 eval_service.py 源码(容错替换非法字节)
    source = (naturebench_repo / "eval_service.py").read_text(
        encoding="utf-8", errors="replace"
    )
    # 源码中出现 --control-token-file 即认为支持
    return "--control-token-file" in source


def _resolve_eval_control_token_file(
    *,
    naturebench_repo: Path,
    output_dir: Path,
    configured: str | None,
    reusing_service: bool,
) -> Path | None:
    """确定评估服务的控制令牌文件路径(用于管理类端点鉴权)。

    - 旧版服务(不支持令牌):显式配置了令牌则报错;否则返回 None;
    - 显式配置优先;复用已运行服务时用仓库默认令牌;新起服务时用输出目录下私生文件。
    """
    # 服务不支持控制令牌
    if not _eval_service_supports_control_token(naturebench_repo):
        # 但用户显式配置了令牌:参数与旧服务不兼容,报错
        if configured:
            raise RuntimeError(
                "--eval-control-token-file was provided, but this NatureBench "
                "eval service does not support control tokens"
            )
        # 返回 None 表示该服务不使用令牌
        return None

    # 决定令牌文件路径:显式配置 > 复用服务默认 > 新服务私生文件
    if configured:
        # 展开并解析为绝对路径
        token_path = Path(configured).expanduser().resolve()
    elif reusing_service:
        # 复用已运行服务:使用该服务启动时的仓库默认令牌路径
        token_path = naturebench_repo / "eval_logs" / "eval_control_token"
    else:
        # 由本启动器新起服务:使用输出目录下的专用令牌文件
        token_path = output_dir / ".eval_service" / "control_token"

    # 令牌文件存在但内容为空:无效令牌,报错
    if token_path.exists() and not token_path.read_text(encoding="utf-8").strip():
        raise RuntimeError(
            f"NatureBench eval control token file is empty: {token_path}"
        )
    # 复用服务但令牌文件不存在:无法对已运行服务发控制指令,报错并给出建议
    if reusing_service and not token_path.is_file():
        raise RuntimeError(
            "A current NatureBench eval service is already running, but its "
            f"control token file is unavailable: {token_path}. Pass the file "
            "used to start that service with --eval-control-token-file, or use "
            "a free --eval-port."
        )
    # 新起服务:预先创建令牌文件所在目录
    if not reusing_service:
        token_path.parent.mkdir(parents=True, exist_ok=True)
    # 返回令牌文件路径
    return token_path


def _find_free_loopback_port(host: str = "127.0.0.1") -> int:
    """向内核申请一个空闲的回环端口(绑定端口 0 后读取分配值)。"""
    # 创建 TCP socket,绑定 (host, 0) 让内核分配端口
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((host, 0))
        # 返回分配到的端口号(套接字随 with 退出关闭)
        return int(listener.getsockname()[1])


def _build_ssh_model_tunnel_command(
    *,
    ssh_host: str,
    local_host: str,
    local_port: int,
    remote_host: str,
    remote_port: int,
) -> list[str]:
    """构造 SSH 端口转发隧道命令:把远端模型服务映射到本地端口。

    BatchMode 避免交互式密码;ExitOnForwardFailure 保证转发失败即退出;
    ServerAlive 保活; -N 表示只做转发不执行远程命令。
    """
    # -L 转发规则字符串
    forward = f"{local_host}:{local_port}:{remote_host}:{remote_port}"
    # 组装完整 ssh 命令
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-N",
        "-L",
        forward,
        ssh_host,
    ]


def _fetch_model_ids(
    base_url: str,
    *,
    api_key: str,
    timeout: float = 2,
) -> list[str]:
    """向 OpenAI 兼容端点请求 /models,返回暴露的模型 ID 列表。"""
    # 有 key 才带 Authorization 头;SSH 免鉴权隧道可为空
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    # 构造 /models 请求
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/models",
        headers=headers,
    )
    # 发请求并解析响应
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    # 提取 data 数组中的 id 字段
    return [str(model["id"]) for model in payload.get("data", []) if model.get("id")]


def _select_model_id(requested: str | None, available: list[str]) -> str:
    """确定要使用的模型 ID:显式指定必须可用;未指定时仅当服务只暴露一个模型才自动选。"""
    # 用户显式指定了模型 ID
    if requested:
        # 指定的模型不在服务列表中:报错并列出可用项
        if requested not in available:
            raise RuntimeError(
                f"Requested model {requested!r} is not served. Available: {available}"
            )
        # 返回指定模型
        return requested
    # 未指定且服务只暴露一个模型:直接采用
    if len(available) == 1:
        return available[0]
    # 未指定但服务暴露多个模型:无法自动决定,报错
    raise RuntimeError(
        "The model service must expose exactly one model when --model-id is "
        f"omitted. Available: {available}"
    )


def _wait_for_model_api(
    process: subprocess.Popen[str],
    *,
    base_url: str,
    api_key: str,
    requested_model_id: str | None,
    timeout: float,
    log_path: Path,
) -> str:
    """等待模型 API 就绪并确定模型 ID;超时或隧道进程退出即报错(附日志尾部)。"""
    # 截止时间
    deadline = time.monotonic() + timeout
    # 记录最后一次错误信息
    last_error = ""
    # 轮询直到成功/超时/进程退出
    while time.monotonic() < deadline:
        try:
            # 尝试拉取模型列表
            available = _fetch_model_ids(base_url, api_key=api_key)
            # 列表非空即可确定模型 ID
            if available:
                return _select_model_id(requested_model_id, available)
        except (OSError, ValueError, urllib.error.URLError) as exc:
            # 记录错误细节,稍后随失败信息一起抛出
            last_error = f"{type(exc).__name__}: {exc}"
        # 隧道进程已退出:不再等待
        if process.poll() is not None:
            break
        # 短暂休眠后重试
        time.sleep(0.25)
    # 读取隧道日志尾部 2000 字符辅助排障
    detail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
    # 抛出带上下文的失败信息
    raise RuntimeError(
        f"Model API did not become healthy at {base_url}. "
        f"Last error: {last_error}. SSH log tail:\n{detail}"
    )


def _model_hydra_overrides(model_id: str, base_url: str) -> list[str]:
    """把模型 ID 与 base_url 转成三条 hydra 覆盖参数(litellm 列表第 0 项)。"""
    # 覆盖 model_name、litellm 适配的 openai/<model> 与 base_url(去掉尾部斜杠)
    return [
        f"litellm.model_list.0.model_name={model_id}",
        f"litellm.model_list.0.litellm_params.model=openai/{model_id}",
        f"litellm.model_list.0.litellm_params.base_url={base_url.rstrip('/')}",
    ]


def _wait_for_service(
    process: subprocess.Popen[str],
    *,
    host: str,
    port: int,
    timeout: float,
    log_path: Path,
) -> None:
    """等待评估服务健康;超时或服务进程退出即报错(附日志尾部)。"""
    # 截止时间
    deadline = time.monotonic() + timeout
    # 轮询健康检查
    while time.monotonic() < deadline:
        # 健康则返回
        if _service_is_healthy(host, port):
            return
        # 服务进程已退出:停止等待
        if process.poll() is not None:
            break
        # 短暂休眠后重试
        time.sleep(0.25)
    # 有日志文件则取尾部 2000 字符
    detail = ""
    if log_path.is_file():
        detail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
    # 抛出失败信息(含健康 URL 与日志尾部)
    raise RuntimeError(
        f"NatureBench eval service did not become healthy at "
        f"{_health_url(host, port)}. Log tail:\n{detail}"
    )


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    """终止一个进程及其整组(SIGTERM -> 5 秒宽限 -> SIGKILL);已退出则跳过。"""
    # 进程已结束:无需处理
    if process.poll() is not None:
        return
    try:
        # POSIX 上按进程组发 SIGTERM(连同其子进程一起收尾)
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        # 其他平台只能终止单进程
        else:
            process.terminate()
        # 最多等 5 秒
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        # 宽限期内未退出:升级为强杀
        if process.poll() is None:
            # POSIX 上 SIGKILL 整组
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            # 等待彻底退出
            process.wait()


def _verify_task_packages(data_root: Path, tasks: list[str]) -> None:
    """校验每个任务包的结构完整性:metadata.json、problem/data、evaluation/evaluator.py。"""
    # 若 data_root 下有 tasks 子目录,任务包在 tasks/ 内;否则任务包直接位于 data_root
    tasks_root = data_root / "tasks" if (data_root / "tasks").is_dir() else data_root
    # 每个任务必须存在的相对路径
    required = (
        Path("metadata.json"),
        Path("problem") / "data",
        Path("evaluation") / "evaluator.py",
    )
    # 收集所有缺失项(任务 x 必需路径 的笛卡尔积过滤)
    missing = [
        str(tasks_root / task / relative)
        for task in tasks
        for relative in required
        if not (tasks_root / task / relative).exists()
    ]
    # 有缺失即报错,逐行列出缺失路径
    if missing:
        raise FileNotFoundError(
            "NatureBench task packages are incomplete:\n- " + "\n- ".join(missing)
        )


def _inspect_runtime_packages(
    runtime_python: Path,
    *,
    environment: dict[str, str],
    package_imports: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """用目标解释器真实试 import 各依赖包,返回 (可用包列表, {包名: 错误信息})。

    通过子进程执行一段探测脚本,以 PACKAGE_INSPECTION_MARKER 标记定位 JSON 结果,
    避免被无关 stdout 干扰。
    """
    # 未显式传入则使用默认包清单
    imports = package_imports if package_imports is not None else LOCAL_PACKAGE_IMPORTS
    # 探测脚本:逐包 importlib 尝试导入,失败记录错误;最后打印标记+JSON
    script = (
        "import importlib, json\n"
        f"imports = {json.dumps(imports, sort_keys=True)}\n"
        "errors = {}\n"
        "for package, module in imports.items():\n"
        "    try:\n"
        "        importlib.import_module(module)\n"
        "    except Exception as exc:\n"
        "        errors[package] = f'{type(exc).__name__}: {exc}'\n"
        f"print({PACKAGE_INSPECTION_MARKER!r} + json.dumps(errors, sort_keys=True))\n"
    )
    # 用目标解释器运行探测脚本(180 秒超时,首包 import 可能较慢)
    process = subprocess.run(
        [str(runtime_python), "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    # 探测脚本自身崩溃:取输出尾部报错
    if process.returncode != 0:
        detail = ((process.stderr or "") + (process.stdout or ""))[-2000:]
        raise RuntimeError(f"Failed to inspect local runtime packages: {detail}")
    # 从 stdout 尾部向前找带标记的行,提取标记后的 JSON 串
    payload = next(
        (
            line[len(PACKAGE_INSPECTION_MARKER) :]
            for line in reversed(process.stdout.splitlines())
            if line.startswith(PACKAGE_INSPECTION_MARKER)
        ),
        None,
    )
    # 解析 JSON;解析失败说明输出被污染,报错并附 stdout 尾部
    try:
        unavailable = dict(json.loads(payload))
    except (IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Local runtime package inspection returned invalid output: "
            f"{process.stdout[-1000:]}"
        ) from exc
    # 可用包 = 清单中未被记为错误的包
    available = [package for package in imports if package not in unavailable]
    # 返回 (可用, 不可用及原因)
    return available, unavailable


def _hydra_list(values: list[str]) -> str:
    """把字符串列表序列化为 hydra 命令行可用的 JSON 数组字面量。"""
    # ensure_ascii 保证无特殊字符;紧凑分隔避免空格被 argparse 拆词
    return json.dumps(values, ensure_ascii=True, separators=(",", ":"))


def _parse_args() -> argparse.Namespace:
    """解析启动器命令行参数(含互斥组与透传的 hydra 覆盖项)。"""
    # 参数解析器:说明用途
    parser = argparse.ArgumentParser(
        description="Run AIRA-Evo NatureBench tasks entirely on the local machine."
    )
    # NatureBench 仓库路径(可省,自动探测)
    parser.add_argument("--naturebench-repo", default=None)
    # 数据根目录(默认仓库下 .naturebench/data)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / ".naturebench" / "data"))
    # 任务集文件(默认快速示例集)
    parser.add_argument("--task-set", default=str(DEFAULT_TASK_SET))
    # 直接指定任务 ID(可多次追加)
    parser.add_argument("--task", action="append", default=[])
    # 跳过下载步骤(离线复跑)
    parser.add_argument("--skip-download", action="store_true")
    # 数据集版本(可选)
    parser.add_argument("--dataset-revision", default=None)
    # 运行时二选一:显式解释器 or conda 环境
    runtime = parser.add_mutually_exclusive_group()
    runtime.add_argument("--local-python", default=None)
    runtime.add_argument("--conda-env", default="naturebench-local")
    # conda 可执行文件路径
    parser.add_argument("--conda-executable", default="conda")
    # 评估服务监听地址与端口
    parser.add_argument("--eval-host", default="127.0.0.1")
    parser.add_argument("--eval-port", type=int, default=8321)
    # 评估服务启动健康等待超时
    parser.add_argument("--eval-start-timeout", type=float, default=30)
    # 控制令牌文件:复用外部服务时需要;本启动器自起的服务用输出目录私生文件
    parser.add_argument(
        "--eval-control-token-file",
        default=None,
        help=(
            "Control-token file used by a current NatureBench eval service. "
            "For a service started by this launcher, a private file under the "
            "experiment output is used automatically."
        ),
    )
    # 模型服务暴露的模型 ID(SGLang --served-model-name)
    parser.add_argument(
        "--model-id",
        default=None,
        help="Model ID exposed by the OpenAI-compatible endpoint (the SGLang --served-model-name value).",
    )
    # 模型端点二选一:直连 URL or SSH 隧道主机
    model_endpoint = parser.add_mutually_exclusive_group()
    model_endpoint.add_argument(
        "--model-base-url",
        default=None,
        help="Direct OpenAI-compatible /v1 URL for a hosted API or an already-running local/remote service.",
    )
    model_endpoint.add_argument(
        "--model-ssh-host",
        default=None,
        help="SSH host for tunneling to an already-running remote SGLang service; this launcher does not deploy it.",
    )
    # SSH 隧道:远端模型服务 API 端口
    parser.add_argument(
        "--model-ssh-port",
        type=int,
        default=None,
        help="API port of the already-running SGLang service on --model-ssh-host.",
    )
    # SSH 服务器视角下 SGLang 监听的主机
    parser.add_argument(
        "--model-ssh-remote-host",
        default="127.0.0.1",
        help="Host visible from the SSH server where SGLang is listening.",
    )
    # 本地隧道绑定地址与端口
    parser.add_argument("--model-ssh-local-host", default="127.0.0.1")
    parser.add_argument("--model-ssh-local-port", type=int, default=None)
    # 模型 API 就绪等待超时
    parser.add_argument("--model-ssh-start-timeout", type=float, default=30)
    # hydra 实验配置名(默认本地快速示例配置)
    parser.add_argument("--config-name", default="experiment/naturebench_local_quick")
    # 实验名(缺省按时间戳生成)
    parser.add_argument("--experiment-name", default=None)
    # 输出目录(缺省按实验名落在 output/ 下)
    parser.add_argument("--output-dir", default=None)
    # 冒烟模式:单任务+单节点,端到端快速验证
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one task and one search node for an end-to-end check.",
    )
    # 其余 hydra 覆盖项:放在 '--' 之后原样透传
    parser.add_argument(
        "hydra_overrides",
        nargs=argparse.REMAINDER,
        help="Additional Hydra overrides after '--'.",
    )
    # 解析并返回
    return parser.parse_args()


def main() -> None:
    """启动器主流程:解析参数 -> 定位仓库/解释器 -> 选任务 -> 下载校验 ->
    依赖探测 -> 模型/评估服务就绪 -> 拼装并执行评估命令 -> finally 清理子进程。"""
    # 解析命令行参数
    args = _parse_args()
    # SSH 隧道必须配套远端端口
    if args.model_ssh_host and args.model_ssh_port is None:
        raise ValueError("--model-ssh-port is required with --model-ssh-host")
    # 定位 NatureBench 仓库
    naturebench_repo = _resolve_naturebench_repo(args.naturebench_repo)
    # 解析运行时解释器
    runtime_python = _resolve_runtime_python(
        local_python=args.local_python,
        conda_env=args.conda_env,
        conda_executable=args.conda_executable,
    )

    # 确定任务集合:显式 --task 去重保序 > smoke 快速任务 > 任务集文件
    if args.task:
        # dict.fromkeys 去重且保持首次出现顺序
        tasks = list(dict.fromkeys(args.task))
    elif args.smoke:
        # 冒烟模式只跑默认快速任务
        tasks = [DEFAULT_QUICK_TASK]
    else:
        # 从任务集文件读取
        tasks = _read_task_set(Path(args.task_set).expanduser().resolve())

    # 生成 UTC 时间戳,缺省实验名带上时间戳保证目录唯一
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    # 实验名:显式指定或自动生成
    experiment_name = args.experiment_name or f"naturebench_local_quick_{stamp}"
    # 输出目录:显式指定或 output/<实验名>,统一取绝对路径
    output_dir = (
        Path(args.output_dir or REPO_ROOT / "output" / experiment_name)
        .expanduser()
        .resolve()
    )
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    # 把本次实际使用的任务清单落盘(评估命令与下载命令共用,便于追溯)
    selection_path = output_dir / "task_set.txt"
    selection_path.write_text("\n".join(tasks) + "\n", encoding="utf-8")

    # 数据根目录取绝对路径
    data_root = Path(args.data_dir).expanduser().resolve()
    # 未跳过下载时:调用 run_naturebench.py 只做数据下载/刷新
    if not args.skip_download:
        # 下载命令:--download-only 模式,按任务集清单下载
        download_command = [
            str(runtime_python),
            str(naturebench_repo / "run_naturebench.py"),
            "--data-dir",
            str(data_root),
            "--tasks",
            str(selection_path),
            "--download-only",
        ]
        # 指定数据集版本则追加参数
        if args.dataset_revision:
            download_command.extend(["--dataset-revision", args.dataset_revision])
        # 提示开始下载
        print(
            f"Downloading or refreshing {len(tasks)} NatureBench task(s)...", flush=True
        )
        # 在 NatureBench 仓库目录内运行下载,失败即终止
        subprocess.run(download_command, cwd=naturebench_repo, check=True)
    # 下载(或跳过)后校验任务包结构完整性
    _verify_task_packages(data_root, tasks)

    # 准备运行时环境变量
    runtime_env = os.environ.copy()
    # 缓存根目录:数据目录的父级 cache/
    cache_root = data_root.parent / "cache"
    # matplotlib 专属缓存目录
    matplotlib_cache = cache_root / "matplotlib"
    # 创建缓存目录
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    # XDG_CACHE_HOME 指向缓存根(避免写用户主目录)
    runtime_env.setdefault("XDG_CACHE_HOME", str(cache_root))
    # MPLCONFIGDIR 指向 matplotlib 缓存(并行运行时防配置目录冲突)
    runtime_env.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    # 探测运行时可用的依赖包
    available_packages, unavailable_packages = _inspect_runtime_packages(
        runtime_python,
        environment=runtime_env,
    )
    # 必需包中缺失的部分(排序保证信息稳定)
    missing_required = sorted(REQUIRED_LOCAL_PACKAGES & unavailable_packages.keys())
    # 缺必需包:拒绝启动,附缺失细节与重建指引
    if missing_required:
        # 拼接每个缺失包的错误细节
        details = "; ".join(
            f"{package}: {unavailable_packages[package]}"
            for package in missing_required
        )
        raise RuntimeError(
            "Local NatureBench runtime is missing required packages: "
            f"{details}. Recreate it with environments/naturebench-local.yml."
        )
    # 打印可用包清单(会作为 agent 可用工具的提示注入)
    print(
        "Available local candidate packages: " + ", ".join(available_packages),
        flush=True,
    )
    # 可选包缺失只降级:从 agent prompt 中剔除,不阻断
    if unavailable_packages:
        print(
            "Omitting unavailable optional packages from the agent prompt: "
            + ", ".join(sorted(unavailable_packages)),
            flush=True,
        )

    # 准备传给评估子进程的环境变量
    child_env = runtime_env.copy()
    # SSH 免鉴权隧道场景:没有 key 时自动填 EMPTY
    if not child_env.get("PRIMARY_KEY") and args.model_ssh_host:
        child_env["PRIMARY_KEY"] = "EMPTY"
    # 其他场景缺 key 则拒绝启动(评估需要真实模型服务)
    if not child_env.get("PRIMARY_KEY"):
        raise RuntimeError(
            "Missing model API key. Set PRIMARY_KEY in the "
            "environment. SSH-tunneled models without auth use EMPTY automatically."
        )

    # 以下为需要清理的子进程与日志句柄(先声明,finally 统一回收)
    model_tunnel_process: subprocess.Popen[str] | None = None
    model_tunnel_log_handle = None
    # 模型 base_url 与模型 ID(后续按隧道/直连分支填充)
    model_base_url = args.model_base_url
    model_id = args.model_id
    # 评估服务子进程与日志句柄
    eval_process: subprocess.Popen[str] | None = None
    eval_log_handle = None
    try:
        # 分支一:SSH 隧道接入远端已运行的模型服务
        if args.model_ssh_host:
            # 本地端口:显式指定或申请空闲端口
            local_port = args.model_ssh_local_port or _find_free_loopback_port(
                args.model_ssh_local_host
            )
            # 本地隧道入口 URL
            model_base_url = f"http://{args.model_ssh_local_host}:{local_port}/v1"
            # 隧道日志文件
            tunnel_log_path = output_dir / "model_ssh_tunnel.log"
            # 追加模式打开日志句柄
            model_tunnel_log_handle = tunnel_log_path.open("a", encoding="utf-8")
            # 构造 ssh 隧道命令
            tunnel_command = _build_ssh_model_tunnel_command(
                ssh_host=args.model_ssh_host,
                local_host=args.model_ssh_local_host,
                local_port=local_port,
                remote_host=args.model_ssh_remote_host,
                remote_port=args.model_ssh_port,
            )
            # 启动隧道进程(POSIX 上独立进程组,便于整组回收)
            model_tunnel_process = subprocess.Popen(
                tunnel_command,
                stdout=model_tunnel_log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=os.name == "posix",
            )
            # 等模型 API 就绪并确定模型 ID(失败会抛错进入 finally 清理)
            model_id = _wait_for_model_api(
                model_tunnel_process,
                base_url=model_base_url,
                api_key=child_env["PRIMARY_KEY"],
                requested_model_id=model_id,
                timeout=args.model_ssh_start_timeout,
                log_path=tunnel_log_path,
            )
            # 打印隧道连接信息
            print(
                f"Connected model {model_id} through SSH tunnel "
                f"{args.model_ssh_host}:{args.model_ssh_port} -> {model_base_url}",
                flush=True,
            )
        # 分支二:直连已运行的 OpenAI 兼容服务
        elif model_base_url:
            # 拉取可用模型列表
            available = _fetch_model_ids(
                model_base_url,
                api_key=child_env["PRIMARY_KEY"],
                timeout=args.model_ssh_start_timeout,
            )
            # 校验并确定模型 ID
            model_id = _select_model_id(model_id, available)
            # 打印直连信息
            print(f"Using model {model_id} at {model_base_url}", flush=True)

        # 判断评估服务端口是否已有健康服务(可复用则不必重启)
        reusing_eval_service = _service_is_healthy(args.eval_host, args.eval_port)
        # 确定控制令牌文件(复用/新起策略不同)
        eval_control_token_file = _resolve_eval_control_token_file(
            naturebench_repo=naturebench_repo,
            output_dir=output_dir,
            configured=args.eval_control_token_file,
            reusing_service=reusing_eval_service,
        )
        # 复用已有健康服务
        if reusing_eval_service:
            print(
                f"Reusing healthy NatureBench eval service at "
                f"http://{args.eval_host}:{args.eval_port}",
                flush=True,
            )
        else:
            # 新起评估服务:日志文件与句柄
            eval_log_path = output_dir / "eval_service.log"
            eval_log_handle = eval_log_path.open("a", encoding="utf-8")
            # 服务启动命令(用解析出的运行时解释器)
            eval_command = [
                str(runtime_python),
                str(naturebench_repo / "eval_service.py"),
                "--host",
                args.eval_host,
                "--port",
                str(args.eval_port),
            ]
            # 服务支持令牌时传入控制令牌文件
            if eval_control_token_file is not None:
                eval_command.extend(
                    ["--control-token-file", str(eval_control_token_file)]
                )
            # 启动评估服务(POSIX 独立进程组)
            eval_process = subprocess.Popen(
                eval_command,
                cwd=naturebench_repo,
                stdout=eval_log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=runtime_env,
                start_new_session=os.name == "posix",
            )
            # 等待服务健康(失败抛错,进入 finally 清理)
            _wait_for_service(
                eval_process,
                host=args.eval_host,
                port=args.eval_port,
                timeout=args.eval_start_timeout,
                log_path=eval_log_path,
            )
            # 打印服务就绪信息
            print(
                f"Started local NatureBench eval service at "
                f"http://{args.eval_host}:{args.eval_port}",
                flush=True,
            )

        # 服务健康后,再次确认控制令牌文件可读(竞态防护:服务起来了但文件没就绪)
        if eval_control_token_file is not None:
            try:
                # 读取并去除首尾空白
                control_token = eval_control_token_file.read_text(
                    encoding="utf-8"
                ).strip()
            except OSError as exc:
                # 文件不可读:服务无法被管理,报错
                raise RuntimeError(
                    "NatureBench eval service became healthy without a readable "
                    f"control token file: {eval_control_token_file}"
                ) from exc
            # 文件内容为空:无效令牌,报错
            if not control_token:
                raise RuntimeError(
                    "NatureBench eval service control token file is empty: "
                    f"{eval_control_token_file}"
                )
            # 通过环境变量把令牌文件路径传给评估子进程(文件内容由子进程自行读取)
            child_env["NATUREBENCH_EVAL_CONTROL_TOKEN_FILE"] = str(
                eval_control_token_file
            )

        # 组装评估主命令:调用 scripts/evaluate_naturebench.py(hydra 入口)
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate_naturebench.py"),
            # hydra 配置名
            "--config-name",
            args.config_name,
            # 实验名与输出目录
            f"experiment_name={experiment_name}",
            f"output_dir={output_dir}",
            # NatureBench 数据根与任务根
            f"data.naturebench_root={data_root}",
            "data.tasks_root=tasks",
            # 任务列表(JSON 数组字面量)
            f"data.task_list={_hydra_list(tasks)}",
            # 评估服务地址
            f"data.eval_service_url=http://{args.eval_host}:{args.eval_port}",
            # 批次名与实验名一致
            f"data.batch_name={experiment_name}",
            # 本地执行模式
            "data.execution_mode=local",
            # 工作区根目录
            f"data.workspace_root={output_dir / 'workspaces'}",
            # 本地解释器(任务子进程用它运行)
            f"data.local_python={runtime_python}",
            # 显式解释器模式下 conda 环境置空
            "data.local_conda_env=null",
            # 把探测到的可用包注入 solver(agent prompt 用)
            "search.runner.solver.available_packages="
            f"{_hydra_list(available_packages)}",
        ]
        # 有模型端点时追加三条 hydra 覆盖(模型名+base_url)
        if model_base_url:
            # 模型 ID 必须已被解析出来
            if model_id is None:
                raise RuntimeError("Could not resolve model id")
            command.extend(_model_hydra_overrides(model_id, model_base_url))
        # 冒烟模式:覆盖为最小规模(2 步/单任务/单候选/30 分钟预算/600 秒执行超时)
        if args.smoke:
            command.extend(
                [
                    "max_steps=2",
                    "time_budget=1800",
                    "model_plus_sandbox_time_budget=1800",
                    "llm_concurrency=1",
                    "search.runner.task_concurrency=1",
                    "search.runner.solver.step_limit=2",
                    "search.runner.solver.num_generations=1",
                    "search.runner.solver.individuals_per_generation=1",
                    "search.runner.solver.execution_timeout=600",
                    "data.execution_timeout=600",
                ]
            )
        # 追加用户透传的 hydra 覆盖(过滤掉分隔符 --)
        command.extend(value for value in args.hydra_overrides if value != "--")
        # 提示开始评估
        print(
            f"Running {len(tasks)} local NatureBench task(s); output: {output_dir}",
            flush=True,
        )
        # 在仓库根目录运行评估命令,失败即抛(check=True)
        subprocess.run(command, cwd=REPO_ROOT, env=child_env, check=True)
    finally:
        # 无论成败:回收评估服务进程组
        if eval_process is not None:
            _terminate_process_group(eval_process)
        # 关闭评估日志句柄
        if eval_log_handle is not None:
            eval_log_handle.close()
        # 回收模型隧道进程组
        if model_tunnel_process is not None:
            _terminate_process_group(model_tunnel_process)
        # 关闭隧道日志句柄
        if model_tunnel_log_handle is not None:
            model_tunnel_log_handle.close()


# 直接运行脚本时进入主流程
if __name__ == "__main__":
    main()
