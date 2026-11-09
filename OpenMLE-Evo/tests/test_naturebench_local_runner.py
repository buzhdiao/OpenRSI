# 中文说明:本测试文件验证 NatureBench 本地(local)执行链路的两大组件:
# 1) third_party/aira-evo/examples/nature_bench/base_task.py 的 NatureBenchTask:
#    本地 python 命令解析(显式解释器/conda 环境)、候选代码 import 预检、
#    本地解题执行(注入 DATA_DIR)、控制令牌只发给控制类端点、超时杀整组进程;
# 2) scripts/run_naturebench_local.py 启动器的辅助函数:任务集读取、运行时
#    python 解析、任务包校验、控制令牌文件选择、真实 import 探测、SSH 模型隧道
#    与 hydra 覆盖参数;另有一个依赖真实 NatureBench 仓库的外部契约测试。
# 位置:OpenMLE-Evo/tests 下,保障 local 模式与官方评估服务的契约不漂移。

# 延迟注解:类型标注运行期不求值
from __future__ import annotations

# 按文件路径动态加载模块(被测脚本不在常规包路径上)
import importlib.util
# json:构造 metadata、解析 hydra 任务列表
import json
# os:环境变量与非 posix 平台判断
import os
# socket:为外部契约测试抢占一个空闲端口
import socket
# subprocess:启动真实评估服务子进程
import subprocess
# sys:取当前解释器路径、调整 sys.path
import sys
# time:服务健康检查的等待与截止时间
import time
# urllib.request:健康检查与(被 mock 的)服务调用
import urllib.request
# Path:路径构造
from pathlib import Path

# pytest:测试框架与近似断言(approx)
import pytest

# 仓库根目录(OpenMLE-Evo)
REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    """按文件路径加载模块,加载失败直接断言报错。"""
    # 构造模块规格
    spec = importlib.util.spec_from_file_location(name, path)
    # 规格必须存在(文件路径合法)
    assert spec is not None
    # 创建模块对象
    module = importlib.util.module_from_spec(spec)
    # loader 必须存在
    assert spec.loader is not None
    # 执行模块代码完成初始化
    spec.loader.exec_module(module)
    # 返回模块
    return module


def _base_task_module():
    """加载 NatureBench 的 base_task.py(先把它依赖的 aira-evo src 加进 sys.path)。"""
    # 把 aira-evo 源码目录插入 sys.path 首位(base_task 需要导入其中模块)
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "aira-evo" / "src"))
    # 按路径加载 base_task 模块并返回
    return _load_module(
        "naturebench_base_task_local",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "base_task.py",
    )


def _task_config(tmp_path: Path, **overrides):
    """构造一份最小可用的 NatureBenchTask 配置(local 执行模式),可按键覆盖。"""
    # 任务目录
    task_dir = tmp_path / "task"
    # 数据目录:problem/data
    data_dir = task_dir / "problem" / "data"
    # 创建目录
    data_dir.mkdir(parents=True)
    # 基础配置字典
    config = {
        # 任务名
        "task_name": "local-test",
        # 数据目录(绝对路径)
        "data_dir": str(data_dir),
        # problem 目录(数据目录的父级)
        "problem_dir": str(data_dir.parent),
        # 任务目录
        "task_dir": str(task_dir),
        # 执行模式:本地
        "execution_mode": "local",
        # 评估服务地址(本地默认端口)
        "eval_service_url": "http://127.0.0.1:8321",
        # 工作区根目录
        "workspace_root": str(tmp_path / "workspaces"),
        # 开启候选预检
        "candidate_preflight": True,
        # 预检包含 import 试运行
        "candidate_preflight_imports": True,
        # 执行超时 5 秒
        "execution_timeout": 5,
    }
    # 应用调用方的覆盖项
    config.update(overrides)
    # 返回配置
    return config


def test_local_python_command_prefers_explicit_interpreter(tmp_path):
    """配置了显式解释器路径时,本地命令应直接使用它,忽略 conda 环境设置。"""
    # 用显式 local_python 与(应被忽略的)conda 环境构造任务
    task = _base_task_module().NatureBenchTask(
        _task_config(
            tmp_path,
            local_python="/opt/naturebench/bin/python",
            local_conda_env="ignored-env",
        )
    )

    # 生成 python 命令应原样使用显式解释器并透传参数
    assert task._local_python_command("-c", "print(1)") == [
        "/opt/naturebench/bin/python",
        "-c",
        "print(1)",
    ]


def test_local_python_command_supports_conda_environment(tmp_path):
    """未给显式解释器但配置了 conda 环境时,应生成 conda run 包装命令。"""
    # 用 conda 环境+conda 可执行路径构造任务
    task = _base_task_module().NatureBenchTask(
        _task_config(
            tmp_path,
            local_conda_env="naturebench-local",
            local_conda_executable="/opt/conda/bin/conda",
        )
    )

    # 命令应为 conda run --no-capture-output -n <env> python <script> 形式
    assert task._local_python_command("run.py") == [
        "/opt/conda/bin/conda",
        "run",
        "--no-capture-output",
        "-n",
        "naturebench-local",
        "python",
        "run.py",
    ]


def test_local_import_preflight_uses_candidate_runtime(tmp_path):
    """候选预检应真实执行 import:可导入的代码通过,缺失模块被拦截并给出反馈。"""
    # 用当前解释器构造任务
    task = _base_task_module().NatureBenchTask(
        _task_config(tmp_path, local_python=sys.executable)
    )

    # 候选 1:import 标准库并使用,应通过
    passed = task._candidate_preflight("import json\nprint(json.dumps({}))")
    # 候选 2:import 一个必然不存在的模块,应被拦截
    blocked = task._candidate_preflight("import definitely_missing_naturebench_module")

    # 通过的候选 ok 为 True
    assert passed["ok"] is True
    # 拦截的候选 ok 为 False
    assert blocked["ok"] is False
    # 拦截类别应标记为"依赖不可用"
    assert blocked["category"] == "unavailable_import"
    # 反馈文本中应包含缺失模块名,便于定位
    assert "definitely_missing_naturebench_module" in blocked["feedback"]


def test_local_solution_uses_configured_python(tmp_path):
    """本地解题执行应使用配置的解释器运行,并注入 DATA_DIR 环境变量。"""
    # 用当前解释器构造任务
    task = _base_task_module().NatureBenchTask(
        _task_config(tmp_path, local_python=sys.executable)
    )
    # 开启一次 validation 阶段的尝试(创建工作区/输出目录)
    attempt = task._start_attempt(phase="validation")

    # 候选代码:打印解释器路径与 DATA_DIR 环境变量
    result = task._run_local_solution(
        "import os, sys\nprint(sys.executable)\nprint(os.environ['DATA_DIR'])",
        workspace=attempt.workspace,
        output_dir=attempt.output_dir,
    )

    # 执行状态应为 success
    assert result["status"] == "success"
    # 运行日志中应包含所用解释器路径(证明用的是配置的解释器)
    assert sys.executable in result["raw_run_log"]
    # 日志中应包含验证数据目录(证明 DATA_DIR 已注入)
    assert task.validation_data_dir in result["raw_run_log"]


def test_current_eval_contract_adds_control_header_only_to_control_endpoints(
    monkeypatch,
    tmp_path,
):
    """控制令牌头只应加在控制类端点(register),业务端点(evaluate)不得携带。"""
    # 加载 base_task 模块
    base_task = _base_task_module()
    # 控制令牌文件路径
    control_token_file = tmp_path / "eval-control-token"
    # 写入令牌内容(带换行,模拟真实文件)
    control_token_file.write_text("private-control-token\n", encoding="utf-8")
    # 构造携带令牌文件配置的任务
    task = base_task.NatureBenchTask(
        _task_config(
            tmp_path,
            eval_control_token_file=str(control_token_file),
        )
    )
    # 记录每次伪造请求的 (request, timeout)
    requests = []

    class FakeResponse:
        # 伪造的 urlopen 响应:支持上下文管理与 read
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            # 返回最小合法 JSON 响应
            return b'{"status":"ok"}'

    def fake_urlopen(request, timeout):
        # 记录请求对象与超时
        requests.append((request, timeout))
        # 返回伪造响应
        return FakeResponse()

    # 用假 urlopen 替换 base_task 内使用的 urllib.request.urlopen
    monkeypatch.setattr(base_task.urllib.request, "urlopen", fake_urlopen)

    # 调用控制类端点 register
    task._post_json("register", {"task_name": "local-test"}, timeout=10)
    # 调用业务类端点 evaluate
    task._post_json("evaluate", {"eval_token": "opaque"}, timeout=10)

    # register 请求头统一小写
    register_headers = {
        key.lower(): value for key, value in requests[0][0].header_items()
    }
    # evaluate 请求头统一小写
    evaluate_headers = {
        key.lower(): value for key, value in requests[1][0].header_items()
    }
    # 控制端点必须携带控制令牌头
    assert register_headers["x-naturebench-control-token"] == "private-control-token"
    # 业务端点不得携带(防止令牌泄漏给业务调用方)
    assert "x-naturebench-control-token" not in evaluate_headers


def test_local_timeout_terminates_process_group(tmp_path):
    """本地执行超时应终止整个进程组:孙进程也收到 SIGTERM 并自行记录被终止。"""
    # 仅 POSIX 支持进程组语义,其他平台直接返回(测试框架视为通过)
    if os.name != "posix":
        return
    # 构造任务:1 秒超时,0.1 秒终止宽限
    task = _base_task_module().NatureBenchTask(
        _task_config(
            tmp_path,
            local_python=sys.executable,
            execution_timeout=1,
            local_terminate_grace_seconds=0.1,
        )
    )
    # 开启一次尝试,取其工作区
    attempt = task._start_attempt(phase="validation")
    # 孙进程写自己 PID 的文件路径
    child_pid_path = attempt.workspace / "child.pid"
    # 孙进程被 SIGTERM 后写标记的文件路径
    child_term_path = attempt.workspace / "child.terminated"
    # 孙进程代码:注册 SIGTERM 处理器(写 terminated 标记后退出),然后长眠 60 秒
    child_code = (
        "import pathlib, signal, sys, time; "
        "signal.signal(signal.SIGTERM, lambda *_: "
        "(pathlib.Path(sys.argv[1]).write_text('terminated'), sys.exit(0))); "
        "time.sleep(60)"
    )

    # 候选代码:先 fork 一个孙进程并记录 PID,然后自己也长眠 60 秒(必然触发超时)
    result = task._run_local_solution(
        "import pathlib, subprocess, sys, time\n"
        f"pid_path = pathlib.Path({str(child_pid_path)!r})\n"
        f"term_path = {str(child_term_path)!r}\n"
        f"child_code = {child_code!r}\n"
        "child = subprocess.Popen([sys.executable, '-c', child_code, term_path])\n"
        "pid_path.write_text(str(child.pid))\n"
        "time.sleep(60)\n",
        workspace=attempt.workspace,
        output_dir=attempt.output_dir,
    )

    # 结果状态应为 timeout
    assert result["status"] == "timeout"
    # 孙进程确实启动过(PID 文件存在)
    assert child_pid_path.is_file()
    # 孙进程收到了 SIGTERM 并写了 terminated 标记(证明整组被终止)
    assert child_term_path.read_text(encoding="utf-8") == "terminated"


def test_local_launcher_helpers_validate_task_packages(tmp_path):
    """启动器辅助函数:任务集解析(忽略注释行)、解释器解析、任务包校验、hydra 列表。"""
    # 动态加载启动器脚本
    launcher = _load_module(
        "naturebench_local_launcher",
        REPO_ROOT / "scripts" / "run_naturebench_local.py",
    )
    # 快速示例任务 ID
    task_id = "s42256-023-00611-x"
    # 任务包根目录
    task_root = tmp_path / "tasks" / task_id
    # 创建 problem/data 目录
    (task_root / "problem" / "data").mkdir(parents=True)
    # 创建 evaluation 目录
    (task_root / "evaluation").mkdir()
    # 写 metadata.json(空对象占位)
    (task_root / "metadata.json").write_text("{}", encoding="utf-8")
    # 写 evaluator.py(空文件占位,存在即可)
    (task_root / "evaluation" / "evaluator.py").write_text("", encoding="utf-8")
    # 任务集文件:注释行+任务 ID
    task_set = tmp_path / "tasks.txt"
    task_set.write_text(f"# local\n{task_id}\n", encoding="utf-8")

    # 任务集解析应忽略 # 注释行,只返回任务 ID 列表
    assert launcher._read_task_set(task_set) == [task_id]
    # 显式 local_python 时解析为绝对路径
    assert (
        launcher._resolve_runtime_python(
            local_python=sys.executable,
            conda_env=None,
            conda_executable="conda",
        )
        == Path(sys.executable).absolute()
    )
    # 任务包校验应通过(目录结构齐全,不抛异常)
    launcher._verify_task_packages(tmp_path, [task_id])
    # hydra 列表格式应为 JSON 数组字符串
    assert json.loads(launcher._hydra_list([task_id])) == [task_id]


def test_local_launcher_selects_control_token_file_for_current_service(tmp_path):
    """控制令牌文件选择:新起服务时生成输出目录下的令牌;复用服务时用仓库默认令牌。"""
    # 动态加载启动器
    launcher = _load_module(
        "naturebench_local_launcher_control_token",
        REPO_ROOT / "scripts" / "run_naturebench_local.py",
    )
    # 伪造一个 NatureBench 仓库目录
    naturebench_repo = tmp_path / "NatureBench"
    naturebench_repo.mkdir()
    # eval_service.py 支持 --control-token-file 参数(说明是"当前版本"服务)
    (naturebench_repo / "eval_service.py").write_text(
        'parser.add_argument("--control-token-file")\n',
        encoding="utf-8",
    )
    # 输出目录
    output_dir = tmp_path / "output"

    # 新起服务(非复用):应生成输出目录下 .eval_service/control_token
    new_service_token = launcher._resolve_eval_control_token_file(
        naturebench_repo=naturebench_repo,
        output_dir=output_dir,
        configured=None,
        reusing_service=False,
    )
    # 断言生成路径
    assert new_service_token == output_dir / ".eval_service" / "control_token"

    # 仓库默认令牌路径(eval_logs 下)
    default_token = naturebench_repo / "eval_logs" / "eval_control_token"
    # 创建其父目录
    default_token.parent.mkdir()
    # 写入已有令牌
    default_token.write_text("existing-token\n", encoding="utf-8")
    # 复用服务:应直接采用仓库默认令牌(与服务启动时一致)
    reused_token = launcher._resolve_eval_control_token_file(
        naturebench_repo=naturebench_repo,
        output_dir=output_dir,
        configured=None,
        reusing_service=True,
    )
    # 断言复用路径
    assert reused_token == default_token


def test_local_launcher_rejects_current_service_without_readable_control_token(
    tmp_path,
):
    """复用服务但默认令牌不可读时应拒绝启动(避免发不出控制指令)。"""
    # 动态加载启动器
    launcher = _load_module(
        "naturebench_local_launcher_missing_control_token",
        REPO_ROOT / "scripts" / "run_naturebench_local.py",
    )
    # 伪造 NatureBench 仓库目录
    naturebench_repo = tmp_path / "NatureBench"
    naturebench_repo.mkdir()
    # eval_service.py 支持 --control-token-file(当前版本服务)
    (naturebench_repo / "eval_service.py").write_text(
        'parser.add_argument("--control-token-file")\n',
        encoding="utf-8",
    )

    # 复用服务但没有任何可读令牌:应抛 RuntimeError 且信息提到 control token
    with pytest.raises(RuntimeError, match="control token"):
        launcher._resolve_eval_control_token_file(
            naturebench_repo=naturebench_repo,
            output_dir=tmp_path / "output",
            configured=None,
            reusing_service=True,
        )


def test_local_quick_example_targets_only_counterfactual_task():
    """快速示例的默认任务集应只指向单个反事实任务(保持示例轻量)。"""
    # 动态加载启动器
    launcher = _load_module(
        "naturebench_local_launcher_quick_task",
        REPO_ROOT / "scripts" / "run_naturebench_local.py",
    )

    # 默认快速任务 ID 应为反事实任务
    assert launcher.DEFAULT_QUICK_TASK == "s42256-023-00611-x"
    # 默认任务集解析后只含该任务
    assert launcher._read_task_set(launcher.DEFAULT_TASK_SET) == [
        launcher.DEFAULT_QUICK_TASK
    ]
    # 默认任务集文件所在目录名应为 naturebench_local_quick
    assert launcher.DEFAULT_TASK_SET.parent.name == "naturebench_local_quick"


def test_runtime_python_resolution_ignores_outer_virtualenv(monkeypatch, tmp_path):
    """解析 conda 运行时 python 时应剥离外层 venv 的 VIRTUAL_ENV/PATH 影响。"""
    # 动态加载启动器
    launcher = _load_module(
        "naturebench_local_launcher_conda",
        REPO_ROOT / "scripts" / "run_naturebench_local.py",
    )
    # 伪造的 conda 环境 python 路径
    python_path = tmp_path / "conda" / "bin" / "python"
    # 创建其父目录
    python_path.parent.mkdir(parents=True)
    # 创建空文件
    python_path.touch()
    # 捕获 fake_run 收到的环境变量
    captured_environment = None

    def fake_run(*args, **kwargs):
        # 声明引用外部变量以便赋值
        nonlocal captured_environment
        # 记录子进程 env
        captured_environment = kwargs["env"]
        # 返回伪造的 CompletedProcess:stdout 即 python 路径
        return subprocess.CompletedProcess(args[0], 0, f"{python_path}\n", "")

    # 伪造外层虚拟环境与被污染的 PATH
    monkeypatch.setenv("VIRTUAL_ENV", "/tmp/project-venv")
    monkeypatch.setenv("PATH", "/tmp/project-venv/bin:/usr/bin:/bin")
    # 替换启动器内的 subprocess.run
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)

    # 用 conda 环境解析运行时 python
    resolved = launcher._resolve_runtime_python(
        local_python=None,
        conda_env="naturebench-local",
        conda_executable="conda",
    )

    # 解析结果应为 conda 环境内的 python(而非 venv 的)
    assert resolved == python_path
    # 环境变量必须被捕获到
    assert captured_environment is not None
    # VIRTUAL_ENV 应被剥离,防止激活外层 venv
    assert "VIRTUAL_ENV" not in captured_environment
    # PATH 中 venv 段应被移除
    assert captured_environment["PATH"] == "/usr/bin:/bin"


def test_local_launcher_inspects_real_imports(tmp_path):
    """运行时依赖探测应真实执行 import,准确区分可用与缺失的包。"""
    # 动态加载启动器
    launcher = _load_module(
        "naturebench_local_launcher_imports",
        REPO_ROOT / "scripts" / "run_naturebench_local.py",
    )

    # 用当前解释器探测两个 import(json 可用,另一个必然缺失)
    available, unavailable = launcher._inspect_runtime_packages(
        Path(sys.executable),
        environment=os.environ.copy(),
        package_imports={
            "json": "json",
            "missing": "definitely_missing_naturebench_module",
        },
    )

    # 可用清单应只含 json
    assert available == ["json"]
    # 缺失包的错误信息应包含 ModuleNotFoundError
    assert "ModuleNotFoundError" in unavailable["missing"]


def test_local_launcher_builds_ssh_model_tunnel_and_hydra_overrides():
    """SSH 模型隧道命令与 hydra 覆盖参数的构造应符合端口转发与配置命名约定。"""
    # 动态加载启动器
    launcher = _load_module(
        "naturebench_local_launcher_ssh",
        REPO_ROOT / "scripts" / "run_naturebench_local.py",
    )

    # 构造 SSH 隧道命令(本地 31010 -> 远端 30010)
    command = launcher._build_ssh_model_tunnel_command(
        ssh_host="model-host",
        local_host="127.0.0.1",
        local_port=31010,
        remote_host="127.0.0.1",
        remote_port=30010,
    )

    # 命令尾部四项:纯端口转发(不执行远程命令)-N、-L 规则、目标主机
    assert command[-4:] == [
        "-N",
        "-L",
        "127.0.0.1:31010:127.0.0.1:30010",
        "model-host",
    ]
    # 必须开启 ExitOnForwardFailure,转发失败即退出(避免假连接)
    assert "ExitOnForwardFailure=yes" in command
    # hydra 覆盖参数应正确填入模型名与 base_url(三个键)
    assert launcher._model_hydra_overrides(
        "served-model", "http://127.0.0.1:31010/v1/"
    ) == [
        "litellm.model_list.0.model_name=served-model",
        "litellm.model_list.0.litellm_params.model=openai/served-model",
        "litellm.model_list.0.litellm_params.base_url=http://127.0.0.1:31010/v1",
    ]


def test_local_adapter_against_current_naturebench_service(tmp_path):
    """外部契约测试:对真实 NatureBench eval_service 跑两轮评估,验证分数聚合与尝试目录。

    需要 NATUREBENCH_REPO 环境变量指向真实仓库;未设置则跳过。
    """
    # 读取环境变量中的 NatureBench 仓库路径
    naturebench_repo_value = os.environ.get("NATUREBENCH_REPO")
    # 未设置则跳过(纯单元环境不拉外部仓库)
    if not naturebench_repo_value:
        pytest.skip("set NATUREBENCH_REPO to run the external contract test")
    # 展开用户目录并取绝对路径
    naturebench_repo = Path(naturebench_repo_value).expanduser().resolve()
    # 评估服务脚本路径
    eval_service = naturebench_repo / "eval_service.py"
    # 服务不存在也跳过
    if not eval_service.is_file():
        pytest.skip(f"NatureBench eval service is unavailable: {eval_service}")

    # 构造任务包目录结构
    task_dir = tmp_path / "task-package"
    # 数据目录
    data_dir = task_dir / "problem" / "data"
    # 评估目录
    evaluation_dir = task_dir / "evaluation"
    # 创建目录
    data_dir.mkdir(parents=True)
    evaluation_dir.mkdir()
    # 写 metadata.json:声明 fake-instance 数据集与主指标 accuracy(方向 higher_is_better,SOTA 0.5)
    (task_dir / "metadata.json").write_text(
        json.dumps(
            {
                "performance_entries": [
                    {
                        "dataset_name": "fake-instance",
                        "metrics": [
                            {
                                "name": "accuracy",
                                "is_primary": True,
                                "metric_direction": "higher_is_better",
                                "sota_score": 0.5,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    # 写 evaluator.py:从 OUTPUT_DIR/score.json 读分数并按数据集组织返回
    (evaluation_dir / "evaluator.py").write_text(
        "import json\n"
        "import os\n"
        "from pathlib import Path\n\n"
        "def run_evaluation():\n"
        "    score_path = Path(os.environ['OUTPUT_DIR']) / 'score.json'\n"
        "    score = json.loads(score_path.read_text())['score']\n"
        "    return {'fake-instance': {'accuracy': score}}\n",
        encoding="utf-8",
    )
    # 控制令牌文件
    control_token_file = tmp_path / "control-token"
    # 先抢占一个空闲端口(绑定 0 由内核分配,再关闭监听套接字)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = int(listener.getsockname()[1])
    # 启动真实评估服务子进程(stdout 合并 stderr 便于排查)
    service = subprocess.Popen(
        [
            sys.executable,
            str(eval_service),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--control-token-file",
            str(control_token_file),
        ],
        cwd=naturebench_repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # 健康检查截止时间:10 秒
        deadline = time.monotonic() + 10
        # 轮询直到 /health 返回 200 或服务退出
        while time.monotonic() < deadline:
            try:
                # 请求健康检查端点
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=0.5
                ) as response:
                    # 健康即跳出
                    if response.status == 200:
                        break
            except OSError:
                # 服务尚未就绪,继续等
                pass
            # 服务提前退出则终止轮询
            if service.poll() is not None:
                break
            # 短暂休眠后重试
            time.sleep(0.05)
        else:
            # 循环自然结束仍未健康:失败
            pytest.fail("current NatureBench eval service did not become healthy")
        # 服务应仍在运行
        assert service.poll() is None

        # 构造接入真实服务的任务实例(60 秒时间预算)
        task = _base_task_module().NatureBenchTask(
            _task_config(
                tmp_path,
                task_name="current-contract-test",
                task_dir=str(task_dir),
                data_dir=str(data_dir),
                problem_dir=str(data_dir.parent),
                eval_service_url=f"http://127.0.0.1:{port}",
                eval_control_token_file=str(control_token_file),
                local_python=sys.executable,
                batch_name="current-contract-batch",
            ),
            time_budget=60,
        )
        # 候选代码模板:把 SCORE 占位替换为具体分数写入 score.json
        candidate = (
            "import json, os\n"
            "from pathlib import Path\n"
            "output = Path(os.environ['OUTPUT_DIR'])\n"
            "output.mkdir(parents=True, exist_ok=True)\n"
            "(output / 'score.json').write_text(json.dumps({'score': SCORE}))\n"
        )
        # 第一次评估:分数 0.8(validation 阶段)
        first = task.evaluate_code(
            candidate.replace("SCORE", "0.8"), phase="validation"
        )
        # 第二次评估:分数 0.6
        second = task.evaluate_code(
            candidate.replace("SCORE", "0.6"), phase="validation"
        )

        # 第一次聚合改进 = 0.8 - SOTA 0.5 的基准(相对基线的累计提升)
        assert first["aggregate_improvement"] == pytest.approx(0.6)
        # 第二次聚合改进应下降到 0.2(反映本次更差)
        assert second["aggregate_improvement"] == pytest.approx(0.2)
        # 历史最优聚合改进应保持 0.6
        assert second["best_aggregate_improvement"] == pytest.approx(0.6)
        # 第一次是第 1 次尝试
        assert first["attempt"] == 1
        # 第二次是第 2 次尝试
        assert second["attempt"] == 2
        # 两次输出目录都在 workspace 父目录下
        assert Path(first["output_dir"]).parent.name == "workspace"
        assert Path(second["output_dir"]).parent.name == "workspace"
        # 每次尝试使用独立输出目录(互不覆盖)
        assert first["output_dir"] != second["output_dir"]
    finally:
        # 无论成败都清理服务进程
        service.terminate()
        try:
            # 最多等 5 秒正常退出
            service.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # 超时则强杀并再等待
            service.kill()
            service.wait(timeout=5)
