# 中文说明:本测试文件是 NatureBench 与 aira-evo 求解框架集成的端到端回归测试:
# 覆盖"任务构建"(build_tasks 把 NatureBench 原始任务编译成 airaevo config.yaml)、
# "任务执行"(NatureBenchTask 的 prepare/step_task、适应度=aggregate_improvement、
# 截断分与原始分并存)、"候选预检"(禁止 pip install)、SCM 远程 GPU(独占/共享池、
# GPU 等待计时、SSH 瞬断重试)、eval-service 的 token 契约与注册一次语义、
# final_submit 默认关闭、节点选择忽略失败的高分节点、summary 不依赖 MLE 排行榜、
# 以及敏感配置(api_key/token)脱敏等契约。
# 位置:OpenMLE-Evo/tests 下;被测代码主要在 third_party/aira-evo 的 examples 与 src,
# 通过 importlib 按文件路径动态加载(避免依赖整包安装)。

# 延迟注解:类型标注运行期不求值
from __future__ import annotations

# concurrent.futures:线程池,模拟同一任务的并发尝试(attempt)
import concurrent.futures
# importlib.util:按文件路径动态加载被测脚本为模块
import importlib.util
# json:构造/解析 metadata.json 与 stat.json
import json
# shlex:校验远程 tar 命令对含空格/分号路径的引号转义
import shlex
# subprocess:TimeoutExpired 异常 + monkeypatch 替换命令执行
import subprocess
# sys:把仓库根目录与 aira-evo 源码目录注入 sys.path
import sys
# threading:Barrier 同步两个并发尝试同时进入临界点
import threading
# Path:路径构造
from pathlib import Path
# SimpleNamespace:伪造 subprocess.run 返回对象(最小属性容器)
from types import SimpleNamespace

# pytest:断言(approx/raises)与 fixture(tmp_path/monkeypatch)
import pytest
# yaml:生成 build 配置与读回任务 config
import yaml
# OmegaConf:构造 evaluate_naturebench 的层级配置对象
from omegaconf import OmegaConf

# 仓库根目录(OpenMLE-Evo),所有动态加载都相对它定位
REPO_ROOT = Path(__file__).resolve().parents[1]
# 把仓库根目录放进 sys.path(可 import tts_search 等本地包)
sys.path.insert(0, str(REPO_ROOT))
# 把 aira-evo 源码目录也放进 sys.path(可 import dojo 包)
sys.path.insert(0, str(REPO_ROOT / "third_party" / "aira-evo" / "src"))


def _load_module(name: str, path: Path):
    """按文件路径把被测脚本加载为独立模块(不依赖包安装,便于单测隔离)。"""
    # 由路径构造模块加载 spec
    spec = importlib.util.spec_from_file_location(name, path)
    # spec 构造失败直接失败(路径不存在等硬错误)
    assert spec is not None
    # 依据 spec 创建空模块对象
    module = importlib.util.module_from_spec(spec)
    # loader 必然存在,断言以满足类型约定
    assert spec.loader is not None
    # 真正执行模块顶层代码,得到可用模块
    spec.loader.exec_module(module)
    # 返回加载好的模块
    return module


def _write_fake_naturebench_task(
    root: Path,
    task_name: str = "fake-nature-task",
) -> Path:
    """在 root 下伪造一个最小 NatureBench 任务目录(problem 描述+数据+metadata)。"""
    # 任务目录:tasks/<task_name>
    task_dir = root / "tasks" / task_name
    # 题目目录:problem(描述与数据都在其下)
    problem_dir = task_dir / "problem"
    # 可见数据目录:problem/data
    data_dir = problem_dir / "data"
    # 一次性创建全部多级目录
    data_dir.mkdir(parents=True)
    # 写任务 README:告知求解者目标与输出目录约定(OUTPUT_DIR)
    (problem_dir / "README.md").write_text(
        "# Fake Nature task\n\nPredict the target and write outputs under OUTPUT_DIR.",
        encoding="utf-8",
    )
    # 写数据说明:声明可见数据位于 DATA_DIR 且是一个小 CSV
    (problem_dir / "data_description.md").write_text(
        "The visible data lives under DATA_DIR and contains one small CSV file.",
        encoding="utf-8",
    )
    # 写一个仅 1 行样本、2 个特征的极小数据集
    (data_dir / "input.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    # 写任务元数据:任务名/领域/资源/性能指标定义(accuracy 主指标,SOTA=0.8)
    (task_dir / "metadata.json").write_text(
        json.dumps(
            {
                # 任务名(与目录名一致)
                "task_name": task_name,
                # 领域标签(单元测试)
                "domain": "unit-test",
                # 默认资源类型
                "resource": "cpu",
                # 性能条目:一个数据集实例 + 一个主指标
                "performance_entries": [
                    {
                        # 实例名
                        "dataset_name": "fake-instance",
                        # 该实例的指标列表
                        "metrics": [
                            {
                                # 指标名
                                "name": "accuracy",
                                # 是主指标(参与聚合适应度)
                                "is_primary": True,
                                # 方向:越高越好
                                "metric_direction": "higher_is_better",
                                # SOTA 参考分
                                "sota_score": 0.8,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    # 返回任务目录(供配置/断言引用)
    return task_dir


def test_naturebench_build_tasks_writes_airaevo_config(tmp_path):
    """build_tasks 应把 NatureBench 任务完整翻译为 airaevo config.yaml(逐字段校验透传)。"""
    # 动态加载被测脚本:NatureBench 任务构建器
    build_tasks = _load_module(
        "naturebench_build_tasks",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "build_tasks.py",
    )
    # 伪造一个 NatureBench 根目录并放入假任务
    naturebench_root = tmp_path / "NatureBench"
    _write_fake_naturebench_task(naturebench_root)
    # 写构建配置:data 段覆盖根目录/任务清单/评估服务/批次名/本地执行参数等
    build_cfg_path = tmp_path / "build.yaml"
    build_cfg_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    # NatureBench 原始任务根目录
                    "naturebench_root": str(naturebench_root),
                    # 只构建这一个任务
                    "task_list": ["fake-nature-task"],
                    # 评估服务地址
                    "eval_service_url": "http://127.0.0.1:8321",
                    # 批次名(输出目录组织用)
                    "batch_name": "unit-batch",
                    # 本地执行模式
                    "execution_mode": "local",
                    # 执行超时(秒)
                    "execution_timeout": 123,
                    # 本地使用的 Python 解释器
                    "local_python": "/opt/naturebench/bin/python",
                    # 不指定 conda 环境
                    "local_conda_env": None,
                    # 本地进程优雅终止宽限期
                    "local_terminate_grace_seconds": 2,
                    # 候选进程可继承的环境变量白名单
                    "candidate_env_allowlist": ["XDG_CACHE_HOME"],
                }
            }
        ),
        encoding="utf-8",
    )

    # 执行构建:输出到 airaevo_tasks 目录
    output_root = tmp_path / "airaevo_tasks"
    build_tasks.build_tasks(build_cfg_path, output_root=output_root)

    # 读回生成的任务配置文件
    task_cfg = yaml.safe_load(
        (output_root / "fake-nature-task" / "config.yaml").read_text(encoding="utf-8")
    )
    # 基准名必须标记为 naturebench
    assert task_cfg["benchmark"] == "naturebench"
    # 任务名透传
    assert task_cfg["task_name"] == "fake-nature-task"
    # 指标方向:accuracy 越高越好 -> higher_is_better=True
    assert task_cfg["higher_is_better"] is True
    # 算子族标记为 naturebench
    assert task_cfg["operator_family"] == "naturebench"
    # 求解器默认配置指向 naturebench/evo.yaml
    assert task_cfg["solver_defaults"] == "naturebench/evo.yaml"
    # 执行模式透传
    assert task_cfg["execution_mode"] == "local"
    # 评估服务地址透传
    assert task_cfg["eval_service_url"] == "http://127.0.0.1:8321"
    # 批次名透传
    assert task_cfg["batch_name"] == "unit-batch"
    # 超时秒数透传
    assert task_cfg["execution_timeout"] == 123
    # 本地解释器透传
    assert task_cfg["local_python"] == "/opt/naturebench/bin/python"
    # conda 环境保持 None(未指定)
    assert task_cfg["local_conda_env"] is None
    # 优雅终止宽限期透传
    assert task_cfg["local_terminate_grace_seconds"] == 2
    # 环境变量白名单透传
    assert task_cfg["candidate_env_allowlist"] == ["XDG_CACHE_HOME"]
    # 数据目录应指向 problem/data
    assert task_cfg["data_dir"].endswith("problem/data")
    # 任务描述取自 README 内容
    assert "Predict the target" in task_cfg["task_description"]
    # 数据描述取自 data_description.md
    assert "DATA_DIR" in task_cfg["data_description"]
    # 公共用户 prompt 需包含 aggregate_improvement 关键词
    assert "aggregate_improvement" in task_cfg["public_user_prompt"]
    # 未提供 EDA 附录时可见数据分析为空串
    assert task_cfg["visible_data_analysis"] == ""
    # 任务族指导必须包含"禁止安装包"约束
    assert "Do not install packages" in task_cfg["task_family_guidance"]
    # 候选预检默认开启
    assert task_cfg["candidate_preflight"] is True
    # 预检的 import 静态检查也默认开启
    assert task_cfg["candidate_preflight_imports"] is True


def test_naturebench_build_tasks_extracts_empirical_eda_addendum(tmp_path):
    """build_tasks 应只把 EDA 文档中"Empirical EDA Addendum"小节注入配置(而非整篇文档)。"""
    # 动态加载任务构建器
    build_tasks = _load_module(
        "naturebench_build_tasks_eda",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "build_tasks.py",
    )
    # 伪造 NatureBench 根目录与假任务
    naturebench_root = tmp_path / "NatureBench"
    _write_fake_naturebench_task(naturebench_root)
    # 伪造 EDA 分析文档根目录
    eda_root = tmp_path / "eda"
    eda_root.mkdir()
    # 文档含无关段落 + "Empirical EDA Addendum"小节(只有小节内容应被注入)
    (eda_root / "fake-nature-task.md").write_text(
        "# Full document\n\n"
        "This section must not be injected.\n\n"
        "## 5. Empirical EDA Addendum\n\n"
        "Verified shape: (10, 2).\n"
        "Use the visible train.csv path.\n",
        encoding="utf-8",
    )
    # 构建配置:额外指定 visible_data_analysis_root 指向 EDA 目录
    build_cfg_path = tmp_path / "build.yaml"
    build_cfg_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    # NatureBench 根目录
                    "naturebench_root": str(naturebench_root),
                    # 任务清单
                    "task_list": ["fake-nature-task"],
                    # 本地执行
                    "execution_mode": "local",
                    # EDA 文档根目录(按 <task>.md 查找)
                    "visible_data_analysis_root": str(eda_root),
                }
            }
        ),
        encoding="utf-8",
    )

    # 执行构建
    output_root = tmp_path / "airaevo_tasks"
    build_tasks.build_tasks(build_cfg_path, output_root=output_root)

    # 读回任务配置
    task_cfg = yaml.safe_load(
        (output_root / "fake-nature-task" / "config.yaml").read_text(encoding="utf-8")
    )
    # 可见数据分析只应是 EDA 附录小节的正文(前后无关段落被剔除)
    assert task_cfg["visible_data_analysis"] == (
        "Verified shape: (10, 2).\nUse the visible train.csv path."
    )


def test_naturebench_build_tasks_uses_task_set_resource_labels(tmp_path):
    """task-set 目录下的 <资源标签>.txt 决定任务的 resource 字段(gpu_high 覆盖默认 cpu)。"""
    # 动态加载任务构建器
    build_tasks = _load_module(
        "naturebench_build_tasks_resource",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "build_tasks.py",
    )
    # 伪造 NatureBench 根目录与假任务
    naturebench_root = tmp_path / "NatureBench"
    _write_fake_naturebench_task(naturebench_root)
    # 建 task-set 目录:每行一个任务名的资源清单文件
    task_set_root = naturebench_root / "task-set"
    task_set_root.mkdir()
    # 声明 fake-nature-task 属于 gpu_high 资源组
    (task_set_root / "gpu_high.txt").write_text("fake-nature-task\n", encoding="utf-8")
    # 构建配置(最简:根目录+任务清单+本地执行)
    build_cfg_path = tmp_path / "build.yaml"
    build_cfg_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    # NatureBench 根目录
                    "naturebench_root": str(naturebench_root),
                    # 任务清单
                    "task_list": ["fake-nature-task"],
                    # 本地执行
                    "execution_mode": "local",
                }
            }
        ),
        encoding="utf-8",
    )

    # 执行构建
    output_root = tmp_path / "airaevo_tasks"
    build_tasks.build_tasks(build_cfg_path, output_root=output_root)

    # 读回任务配置
    task_cfg = yaml.safe_load(
        (output_root / "fake-nature-task" / "config.yaml").read_text(encoding="utf-8")
    )
    # 资源标签应取自 task-set 清单文件名(覆盖 metadata 里的 cpu)
    assert task_cfg["resource"] == "gpu_high"


def test_naturebench_build_tasks_uses_official_scm_resource_lines(tmp_path):
    """官方 SCM 资源行(gemini_*.txt)应把每类任务映射到正确的 GPU 配置(独占/共享/主机覆盖)。"""
    # 动态加载任务构建器
    build_tasks = _load_module(
        "naturebench_build_tasks_resource_lines",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "build_tasks.py",
    )
    # 伪造 NatureBench 根目录,并批量造 5 个覆盖各类资源行的任务
    naturebench_root = tmp_path / "NatureBench"
    for task_name in [
        "cpu-task",
        "thu-task",
        "thu-shared-task",
        "amd-task",
        "amd-shared-task",
    ]:
        _write_fake_naturebench_task(naturebench_root, task_name=task_name)
    # 建 task-set 目录存放官方资源行清单
    task_set_root = naturebench_root / "task-set"
    task_set_root.mkdir()
    # all.txt:总清单,逐行列出全部任务
    (task_set_root / "all.txt").write_text(
        "\n".join(
            [
                "cpu-task",
                "thu-task",
                "thu-shared-task",
                "amd-task",
                "amd-shared-task",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    # CPU 资源行:cpu-task 走 THU 主机的 CPU 队列
    (task_set_root / "gemini_3.5_flash_cpu_thu.txt").write_text(
        "cpu-task\n", encoding="utf-8"
    )
    # THU 独占 GPU 资源行
    (task_set_root / "gemini_3.5_flash_gpu_thu_normal.txt").write_text(
        "thu-task\n", encoding="utf-8"
    )
    # THU 共享 GPU 资源行
    (task_set_root / "gemini_3.5_flash_gpu_thu_shared.txt").write_text(
        "thu-shared-task\n", encoding="utf-8"
    )
    # AMD 独占 GPU 资源行
    (task_set_root / "gemini_3.5_flash_gpu_amd_normal.txt").write_text(
        "amd-task\n", encoding="utf-8"
    )
    # AMD 共享 GPU 资源行
    (task_set_root / "gemini_3.5_flash_gpu_amd_shared.txt").write_text(
        "amd-shared-task\n", encoding="utf-8"
    )
    # 构建配置:scm_docker 模式 + 每类资源行的细化参数(设备号/池文件/共享槽位/主机覆盖)
    build_cfg_path = tmp_path / "build.yaml"
    build_cfg_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    # NatureBench 根目录
                    "naturebench_root": str(naturebench_root),
                    # 用 task-set 文件替代 task_list
                    "task_set_path": "task-set/all.txt",
                    # SCM 远程 docker 执行模式
                    "execution_mode": "scm_docker",
                    # 默认 SCM 主机
                    "scm_host": "scm-primary",
                    # 顶层 GPU 设备(占位,应被资源行覆盖)
                    "gpu_devices": "all",
                    # 各资源行的细化配置
                    "scm_resource_lines": {
                        # THU 独占:整卡 0/1 + 独占池文件
                        "gpu_thu_normal": {
                            "gpu_devices": [0, 1],
                            "gpu_pool_file": "/tmp/thu-exclusive.json",
                        },
                        # THU 共享:2 号卡 + 4 槽位 + 共享池文件
                        "gpu_thu_shared": {
                            "shared_gpu_device": 2,
                            "shared_gpu_slots": 4,
                            "shared_gpu_pool_file": "/tmp/thu-shared.json",
                        },
                        # AMD 独占:覆盖主机为 scm-secondary
                        "gpu_amd_normal": {
                            "scm_host": "scm-secondary",
                            "gpu_devices": [0],
                            "gpu_pool_file": "/tmp/amd-exclusive.json",
                        },
                        # AMD 共享:同样覆盖主机
                        "gpu_amd_shared": {
                            "scm_host": "scm-secondary",
                            "shared_gpu_device": 1,
                            "shared_gpu_slots": 3,
                            "shared_gpu_pool_file": "/tmp/amd-shared.json",
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    # 执行构建
    output_root = tmp_path / "airaevo_tasks"
    build_tasks.build_tasks(build_cfg_path, output_root=output_root)

    # 分别读回 5 个任务的配置
    cpu_cfg = yaml.safe_load((output_root / "cpu-task" / "config.yaml").read_text())
    thu_cfg = yaml.safe_load((output_root / "thu-task" / "config.yaml").read_text())
    thu_shared_cfg = yaml.safe_load(
        (output_root / "thu-shared-task" / "config.yaml").read_text()
    )
    amd_cfg = yaml.safe_load((output_root / "amd-task" / "config.yaml").read_text())
    amd_shared_cfg = yaml.safe_load(
        (output_root / "amd-shared-task" / "config.yaml").read_text()
    )

    # CPU 任务:资源行 cpu_thu、纯 CPU、无 GPU 模式、不得出现 gpu_devices
    assert cpu_cfg["resource_line"] == "cpu_thu"
    assert cpu_cfg["resource"] == "cpu"
    assert cpu_cfg["gpu_mode"] == "none"
    assert "gpu_devices" not in cpu_cfg

    # THU 独占:资源行/资源/主机/独占模式/设备号/池文件逐项对齐
    assert thu_cfg["resource_line"] == "gpu_thu_normal"
    assert thu_cfg["resource"] == "gpu_thu_normal"
    assert thu_cfg["scm_host"] == "scm-primary"
    assert thu_cfg["gpu_mode"] == "exclusive"
    assert thu_cfg["gpu_devices"] == [0, 1]
    assert thu_cfg["gpu_pool_file"] == "/tmp/thu-exclusive.json"

    # THU 共享:共享模式 + 卡号/槽位数/共享池文件
    assert thu_shared_cfg["resource_line"] == "gpu_thu_shared"
    assert thu_shared_cfg["gpu_mode"] == "shared"
    assert thu_shared_cfg["shared_gpu_device"] == 2
    assert thu_shared_cfg["shared_gpu_slots"] == 4
    assert thu_shared_cfg["shared_gpu_pool_file"] == "/tmp/thu-shared.json"

    # AMD 独占:主机被覆盖为 scm-secondary
    assert amd_cfg["resource_line"] == "gpu_amd_normal"
    assert amd_cfg["scm_host"] == "scm-secondary"
    assert amd_cfg["gpu_mode"] == "exclusive"
    assert amd_cfg["gpu_devices"] == [0]

    # AMD 共享:主机覆盖 + 共享参数
    assert amd_shared_cfg["resource_line"] == "gpu_amd_shared"
    assert amd_shared_cfg["scm_host"] == "scm-secondary"
    assert amd_shared_cfg["gpu_mode"] == "shared"
    assert amd_shared_cfg["shared_gpu_device"] == 1
    assert amd_shared_cfg["shared_gpu_slots"] == 3


def test_naturebench_build_tasks_fails_when_scm_resource_lines_are_unreadable(
    tmp_path,
    monkeypatch,
):
    """远端资源行清单读不到(cat 失败)时,build_tasks 应显式报 RuntimeError 快速失败。"""
    # 动态加载任务构建器
    build_tasks = _load_module(
        "naturebench_build_tasks_resource_lines_failure",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "build_tasks.py",
    )
    # 伪造 NatureBench 根目录与假任务
    naturebench_root = tmp_path / "NatureBench"
    _write_fake_naturebench_task(naturebench_root)
    # 构建配置:scm_docker 模式,task-set 根指向远端不存在目录
    build_cfg_path = tmp_path / "build.yaml"
    build_cfg_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    # NatureBench 根目录
                    "naturebench_root": str(naturebench_root),
                    # 任务清单
                    "task_list": ["fake-nature-task"],
                    # SCM docker 模式(需要远端 task-set)
                    "execution_mode": "scm_docker",
                    # 通过 SSH 在该主机读取 task-set
                    "scm_task_set_host": "scm-primary",
                    # 远端不存在的 task-set 根目录
                    "scm_task_set_root": "/missing/task-set",
                }
            }
        ),
        encoding="utf-8",
    )

    # 伪造一个失败的进程返回(非零退出码 + 报错 stderr)
    class FailedProcess:
        returncode = 1
        stdout = ""
        stderr = "cat: /missing/task-set/gemini_3.5_flash_cpu_thu.txt: No such file"

    # 用 monkeypatch 替换构建器内的 subprocess.run,恒返回失败结果
    monkeypatch.setattr(
        build_tasks.subprocess, "run", lambda *args, **kwargs: FailedProcess()
    )

    # 期望构建直接抛 RuntimeError(带明确错误信息),而不是静默产出错误配置
    with pytest.raises(
        RuntimeError, match="Failed to read NatureBench resource task-set"
    ):
        build_tasks.build_tasks(build_cfg_path, output_root=tmp_path / "airaevo_tasks")


def test_naturebench_build_tasks_allows_zero_submit_repeats(tmp_path):
    """submit_repeats=0(不做最终提交)是合法配置,应原样写进任务配置。"""
    # 动态加载任务构建器
    build_tasks = _load_module(
        "naturebench_build_tasks_zero_submit",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "build_tasks.py",
    )
    # 伪造 NatureBench 根目录与假任务
    naturebench_root = tmp_path / "NatureBench"
    _write_fake_naturebench_task(naturebench_root)
    # 构建配置:显式设置 submit_repeats=0(不提交最终答案)
    build_cfg_path = tmp_path / "build.yaml"
    build_cfg_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    # NatureBench 根目录
                    "naturebench_root": str(naturebench_root),
                    # 任务清单
                    "task_list": ["fake-nature-task"],
                    # 本地执行
                    "execution_mode": "local",
                    # 提交次数为 0(纯搜索,不 final submit)
                    "submit_repeats": 0,
                }
            }
        ),
        encoding="utf-8",
    )

    # 执行构建
    output_root = tmp_path / "airaevo_tasks"
    build_tasks.build_tasks(build_cfg_path, output_root=output_root)

    # 读回任务配置
    task_cfg = yaml.safe_load(
        (output_root / "fake-nature-task" / "config.yaml").read_text(encoding="utf-8")
    )
    # 0 值不能被当成"缺省"丢弃,必须原样保留
    assert task_cfg["submit_repeats"] == 0


def test_naturebench_task_uses_current_aggregate_improvement_as_fitness(tmp_path):
    """step_task 后验证适应度应取本次(而非历史最优)的 aggregate_improvement。"""
    # 动态加载 NatureBench 任务基类
    base_task = _load_module(
        "naturebench_base_task",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "base_task.py",
    )
    # dojo 任务常量:辅助评估信息/任务描述键/验证适应度键
    from dojo.core.tasks.constants import (
        AUX_EVAL_INFO,
        TASK_DESCRIPTION,
        VALIDATION_FITNESS,
    )

    # 假任务:替身运行与评估,免真实执行环境
    class FakeNatureBenchTask(base_task.NatureBenchTask):
        def _run_solution(self, code: str, *, phase: str, attempt=None) -> dict:
            # 断言候选代码确实被原样送进来
            assert "hello naturebench" in code
            # 返回一次成功的执行记录
            return {
                "status_code": 0,
                "status": "success",
                "raw_run_log": "ran candidate",
                "clear_run_log": "ran candidate",
                "feedback": "execution succeeded",
                "run_time": 0.25,
            }

        def _post_evaluate(self, output_dir: Path) -> dict:
            # 伪造评估结果:本次聚合提升 0.37,历史最优 0.42(二者必须区分开)
            return {
                "aggregate_improvement": 0.37,
                "best_aggregate_improvement": 0.42,
                "raw_scores": {"fake-instance": {"accuracy": 0.9}},
                "per_instance_improvement": {"fake-instance": 0.125},
                "attempt": 3,
            }

    # 准备最小数据目录
    data_dir = tmp_path / "problem" / "data"
    data_dir.mkdir(parents=True)
    # 用标准任务配置实例化假任务
    task = FakeNatureBenchTask(
        {
            # 基准名
            "benchmark": "naturebench",
            # 任务名
            "task_name": "fake-nature-task",
            # 指标越高越好
            "higher_is_better": True,
            # 数据目录
            "data_dir": str(data_dir),
            # 题目目录(数据目录的上级)
            "problem_dir": str(data_dir.parent),
            # 任务包目录
            "task_dir": str(tmp_path),
            # 工作区根目录
            "workspace_root": str(tmp_path / "workspace"),
            # 评估服务地址
            "eval_service_url": "http://127.0.0.1:8321",
            # 批次名
            "batch_name": "unit-batch",
            # 本地执行
            "execution_mode": "local",
            # 任务描述
            "task_description": "Do the fake task.",
            # 数据描述
            "data_description": "Use DATA_DIR.",
            # 已验证的可见数据分析(EDA 附录)
            "visible_data_analysis": "Verified shape: (10, 2).",
        }
    )

    # 初始化任务状态,并执行一次候选解
    state, task_info = task.prepare()
    state, result = task.step_task(state, "print('hello naturebench')")

    # 越高越好 -> lower_is_better 必须为 False
    assert task_info["lower_is_better"] is False
    # 任务信息携带 EDA 附录原文
    assert task_info["visible_data_analysis"] == "Verified shape: (10, 2)."
    # 任务描述中要出现"Verified Visible-Data Analysis"小节
    assert "Verified Visible-Data Analysis" in task_info[TASK_DESCRIPTION]
    # 验证适应度 = 本次聚合提升(0.37),而非历史最优(0.42)
    assert result[VALIDATION_FITNESS] == pytest.approx(0.37)
    # 辅助信息:本次聚合提升
    assert result[AUX_EVAL_INFO]["aggregate_improvement"] == pytest.approx(0.37)
    # 辅助信息:历史最优聚合提升
    assert result[AUX_EVAL_INFO]["best_aggregate_improvement"] == pytest.approx(0.42)
    # 辅助信息:score 字段与本次提升一致
    assert result[AUX_EVAL_INFO]["score"] == pytest.approx(0.37)
    # 尝试序号自增到 1
    assert state["naturebench_attempt_index"] == 1


def test_naturebench_task_clips_selection_fitness_but_preserves_raw_aggregate(
    tmp_path,
):
    """配置 selection_score_clip 后:选择用适应度被截断,原始聚合分仍完整保留。"""
    # 动态加载 NatureBench 任务基类
    base_task = _load_module(
        "naturebench_base_task_clipped_fitness",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "base_task.py",
    )
    # dojo 任务常量:辅助评估信息/验证适应度键
    from dojo.core.tasks.constants import AUX_EVAL_INFO, VALIDATION_FITNESS

    # 假任务:只关心评估返回的超大分数(64.05)
    class FakeNatureBenchTask(base_task.NatureBenchTask):
        def _run_solution(self, code: str, *, phase: str, attempt=None) -> dict:
            # 一次成功执行
            return {
                "status_code": 0,
                "status": "success",
                "raw_run_log": "ran candidate",
                "clear_run_log": "ran candidate",
                "feedback": "execution succeeded",
                "run_time": 0.25,
            }

        def _post_evaluate(self, output_dir: Path) -> dict:
            # 伪造远超截断阈值的聚合提升
            return {
                "aggregate_improvement": 64.05,
                "best_aggregate_improvement": 64.05,
                "raw_scores": {"fake-instance": {"accuracy": 999.0}},
                "per_instance_improvement": {"fake-instance": 64.05},
                "attempt": 1,
            }

    # 准备最小数据目录
    data_dir = tmp_path / "problem" / "data"
    data_dir.mkdir(parents=True)
    # 实例化任务:开启 selection_score_clip=1.0
    task = FakeNatureBenchTask(
        {
            # 基准名
            "benchmark": "naturebench",
            # 任务名
            "task_name": "fake-nature-task",
            # 指标越高越好
            "higher_is_better": True,
            # 数据目录
            "data_dir": str(data_dir),
            # 题目目录
            "problem_dir": str(data_dir.parent),
            # 任务包目录
            "task_dir": str(tmp_path),
            # 工作区根目录
            "workspace_root": str(tmp_path / "workspace"),
            # 评估服务地址
            "eval_service_url": "http://127.0.0.1:8321",
            # 批次名
            "batch_name": "unit-batch",
            # 本地执行
            "execution_mode": "local",
            # 任务描述
            "task_description": "Do the fake task.",
            # 数据描述
            "data_description": "Use DATA_DIR.",
            # 选择分截断阈值:1.0
            "selection_score_clip": 1.0,
        }
    )

    # 初始化并执行一次候选解
    state, _ = task.prepare()
    _, result = task.step_task(state, "print('hello naturebench')")

    # 验证适应度(选择用)被截到 1.0
    assert result[VALIDATION_FITNESS] == pytest.approx(1.0)
    # 辅助信息里的选择分同样被截断
    assert result[AUX_EVAL_INFO]["selection_score"] == pytest.approx(1.0)
    # 原始聚合提升 64.05 不被截断(取证/汇总用)
    assert result[AUX_EVAL_INFO]["aggregate_improvement"] == pytest.approx(64.05)
    # score 字段保留原始值
    assert result[AUX_EVAL_INFO]["score"] == pytest.approx(64.05)


def test_naturebench_preflight_blocks_package_install_without_evaluation(tmp_path):
    """候选代码含 pip install 时,预检应直接拦截:不执行、不评估,返回 422 分类。"""
    # 动态加载 NatureBench 任务基类
    base_task = _load_module(
        "naturebench_base_task_preflight",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "base_task.py",
    )
    # dojo 任务常量:辅助评估信息键
    from dojo.core.tasks.constants import AUX_EVAL_INFO

    # 记录"执行/评估"调用次序(预检拦截则应全空)
    calls = []

    # 计数假任务:把每次运行/评估都记账
    class CountingTask(base_task.NatureBenchTask):
        def _run_solution(self, code: str, *, phase: str, attempt=None) -> dict:
            # 记录一次运行
            calls.append(("run", code, phase))
            # 返回成功执行记录
            return {
                "status_code": 0,
                "status": "success",
                "raw_run_log": "ran candidate",
                "clear_run_log": "ran candidate",
                "feedback": "ran candidate",
                "run_time": 0.25,
            }

        def _post_evaluate(self, output_dir: Path) -> dict:
            # 记录一次评估
            calls.append(("evaluate", str(output_dir)))
            # 返回聚合提升
            return {"aggregate_improvement": 0.2}

    # 准备最小数据目录
    data_dir = tmp_path / "problem" / "data"
    data_dir.mkdir(parents=True)
    # 实例化任务:开启候选预检
    task = CountingTask(
        {
            # 基准名
            "benchmark": "naturebench",
            # 任务名
            "task_name": "fake-nature-task",
            # 数据目录
            "data_dir": str(data_dir),
            # 题目目录
            "problem_dir": str(data_dir.parent),
            # 任务包目录
            "task_dir": str(tmp_path),
            # 工作区根目录
            "workspace_root": str(tmp_path / "workspace"),
            # 评估服务地址
            "eval_service_url": "http://127.0.0.1:8321",
            # 批次名
            "batch_name": "unit-batch",
            # 本地执行
            "execution_mode": "local",
            # 开启预检
            "candidate_preflight": True,
        }
    )

    # 初始化后提交一段尝试 pip install 的候选代码
    state, _ = task.prepare()
    _, result = task.step_task(
        state,
        "import subprocess\nsubprocess.run(['pip', 'install', 'andi'])",
    )

    # 取辅助评估信息
    info = result[AUX_EVAL_INFO]
    # 预检拦截:执行与评估一次都没发生
    assert calls == []
    # 状态码为 422(预检失败)
    assert info["status_code"] == 422
    # 预检分类:禁止安装包
    assert info["preflight_category"] == "prohibited_package_install"
    # 反馈文本说明不允许安装包
    assert "package installation is not allowed" in info["feedback"]


def test_naturebench_scm_gpu_wait_is_reported_separately(monkeypatch, tmp_path):
    """SCM 执行中等待 GPU 的秒数应从 stdout 标记里拆出,单独上报并从反馈中剔除。"""
    # 动态加载 NatureBench 任务基类
    base_task = _load_module(
        "naturebench_base_task_gpu_wait",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "base_task.py",
    )

    # 伪造 ssh 执行结果:stdout 首行是 GPU 等待标记(7 秒),后面才是候选输出
    class FakeCompletedProcess:
        returncode = 0
        stdout = "__AIREVO_GPU_WAIT_SECONDS__=7\ncandidate stdout\n"
        stderr = ""

    # 捕获型假任务:把 SCM 相关方法全部换成桩
    class CapturingTask(base_task.NatureBenchTask):
        def _sync_workspace_to_scm(
            self,
            workspace: Path,
            remote_workspace: str,
        ) -> None:
            # 工作区同步置空(单测不需要真实同步)
            return None

        def _resolve_scm_task_root(self) -> str:
            # 返回固定远端任务根
            return "/remote/task"

        def _ssh(self, remote_command: str, **kwargs):
            # 恒返回伪造结果
            return FakeCompletedProcess()

    # 两个单调时钟读数:起点 100s、终点 110s(总耗时 10s)
    ticks = iter([100.0, 110.0])
    monkeypatch.setattr(base_task.time, "monotonic", lambda: next(ticks))
    # 准备最小数据目录
    data_dir = tmp_path / "problem" / "data"
    data_dir.mkdir(parents=True)
    # 实例化 SCM 独占 GPU 任务
    task = CapturingTask(
        {
            # 基准名
            "benchmark": "naturebench",
            # 任务名
            "task_name": "fake-nature-task",
            # 资源行:THU 独占 GPU
            "resource": "gpu_thu_normal",
            # 独占模式
            "gpu_mode": "exclusive",
            # 使用 0 号卡
            "gpu_devices": [0],
            # 数据目录
            "data_dir": str(data_dir),
            # 题目目录
            "problem_dir": str(data_dir.parent),
            # 任务包目录
            "task_dir": str(tmp_path),
            # 工作区根目录
            "workspace_root": str(tmp_path / "workspace"),
            # SCM docker 执行模式
            "execution_mode": "scm_docker",
            # SCM 主机
            "scm_host": "scm-primary",
            # 远端工作区根
            "scm_workspace_root": "/remote/workspaces/unit",
            # 远端任务根列表
            "scm_task_roots": ["/remote/tasks"],
            # 本机可达的评估服务地址
            "eval_service_url": "http://localhost:8321",
            # SCM 主机视角的评估服务地址
            "scm_eval_service_url": "http://localhost:8321",
            # 容器内可达的评估服务地址
            "scm_container_eval_service_url": "http://host.docker.internal:8321",
            # 批次名
            "batch_name": "unit-batch",
            # 执行超时 60 秒
            "execution_timeout": 60,
        }
    )

    # 启动一次验证阶段尝试,并走 SCM docker 执行路径
    attempt = task._start_attempt(phase="validation")
    result = task._run_scm_docker_solution(
        "print('hello naturebench')",
        attempt=attempt,
    )

    # 总运行时长应为 10 秒(monotonic 差值)
    assert result["run_time"] == pytest.approx(10.0)
    # GPU 等待 7 秒被单独拆出上报
    assert result["gpu_wait_seconds"] == pytest.approx(7.0)
    # 有效运行时间 = 10 - 7 = 3 秒
    assert result["active_run_time"] == pytest.approx(3.0)
    # 内部标记不得泄漏到用户可见反馈里
    assert "__AIREVO_GPU_WAIT_SECONDS__" not in result["feedback"]


def test_naturebench_task_registers_eval_service_once_before_evaluation(tmp_path):
    """eval-service 的 register/start_timer 只在首次评估前调用一次,后续 attempt 只 evaluate。"""
    # 动态加载 NatureBench 任务基类
    base_task = _load_module(
        "naturebench_base_task_register",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "base_task.py",
    )

    # 记录每次 HTTP 端点调用(端点、负载、超时)
    calls = []

    # 假任务:拦截 _post_json 记账并伪造响应
    class RegisteringFakeNatureBenchTask(base_task.NatureBenchTask):
        def _run_solution(self, code: str, *, phase: str, attempt=None) -> dict:
            # 一次成功执行
            return {
                "status_code": 0,
                "status": "success",
                "raw_run_log": "ran candidate",
                "clear_run_log": "ran candidate",
                "feedback": "execution succeeded",
                "run_time": 0.25,
            }

        def _post_json(
            self,
            endpoint: str,
            payload: dict,
            *,
            timeout: int | float,
        ) -> dict:
            # 记账本次端点调用
            calls.append((endpoint, dict(payload), timeout))
            # register/start_timer 返回 ok
            if endpoint == "register":
                return {"status": "ok"}
            if endpoint == "start_timer":
                return {"status": "ok"}
            # evaluate 返回聚合提升(attempt 序号=已 evaluate 次数)
            return {
                "aggregate_improvement": 0.2,
                "best_aggregate_improvement": 0.2,
                "raw_scores": {},
                "per_instance_improvement": {},
                "attempt": len([call for call in calls if call[0] == "evaluate"]),
            }

    # 任务包目录(problem/data 在其下)
    task_package_dir = tmp_path / "task-package"
    data_dir = task_package_dir / "problem" / "data"
    data_dir.mkdir(parents=True)
    # 实例化任务:带 4 小时时间预算
    task = RegisteringFakeNatureBenchTask(
        {
            # 基准名
            "benchmark": "naturebench",
            # 任务名
            "task_name": "fake-nature-task",
            # 指标越高越好
            "higher_is_better": True,
            # 数据目录
            "data_dir": str(data_dir),
            # 题目目录
            "problem_dir": str(data_dir.parent),
            # 任务包目录
            "task_dir": str(task_package_dir),
            # 工作区根目录
            "workspace_root": str(tmp_path / "workspace"),
            # 评估服务地址
            "eval_service_url": "http://127.0.0.1:8321",
            # 批次名
            "batch_name": "unit-batch",
            # 本地执行
            "execution_mode": "local",
            # 任务描述
            "task_description": "Do the fake task.",
            # 数据描述
            "data_description": "Use DATA_DIR.",
            # 执行超时
            "execution_timeout": 600,
        },
        time_budget=14400,
    )

    # 初始化后连续两次 step(两次 attempt)
    state, _ = task.prepare()
    task.step_task(state, "print('first')")
    task.step_task(state, "print('second')")

    # 端点次序:注册一次 -> 启动计时一次 -> 两次评估(第二次不再 register)
    assert [endpoint for endpoint, _, _ in calls] == [
        "register",
        "start_timer",
        "evaluate",
        "evaluate",
    ]
    # 取 register 的负载
    register_payload = calls[0][1]
    # 注册负载:任务名
    assert register_payload["task_name"] == "fake-nature-task"
    # 注册负载:绝对化的数据目录
    assert register_payload["data_dir"] == str(task_package_dir.resolve())
    # 注册负载:时间预算(秒)
    assert register_payload["timeout"] == 14400
    # 注册负载:批次名
    assert register_payload["batch_name"] == "unit-batch"
    # 注册负载:输出目录形如 <batch>/<task>
    assert register_payload["out_dir"].endswith("unit-batch/fake-nature-task")
    # start_timer 只需任务名+批次名
    assert calls[1][1] == {
        "task_name": "fake-nature-task",
        "batch_name": "unit-batch",
    }


def test_naturebench_task_uses_token_contract_and_rebinds_each_attempt(tmp_path):
    """eval token 契约:token 来自控制文件、随 register 下发,每次 attempt 重新注册并绑定新输出目录。"""
    # 动态加载 NatureBench 任务基类
    base_task = _load_module(
        "naturebench_base_task_token_contract",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "base_task.py",
    )

    # 控制端 token 文件(由服务侧下发,不能透传给候选环境)
    control_token_file = tmp_path / "eval-control-token"
    control_token_file.write_text("service-control-token\n", encoding="utf-8")
    # 端点调用记账
    calls = []

    # 假任务:校验 attempt 上下文并伪造端点响应
    class TokenContractTask(base_task.NatureBenchTask):
        def _run_solution(self, code: str, *, phase: str, attempt=None) -> dict:
            # 返回执行记录,并带上本次 attempt 的输出目录
            return {
                "status_code": 0,
                "status": "success",
                "raw_run_log": "ran candidate",
                "clear_run_log": "ran candidate",
                "feedback": "execution succeeded",
                "run_time": 0.25,
                "output_dir": str(attempt.output_dir),
            }

        def _post_json(
            self,
            endpoint: str,
            payload: dict,
            *,
            timeout: int | float,
        ) -> dict:
            # 记账本次调用
            calls.append((endpoint, dict(payload), timeout))
            # register/start_timer 恒 ok
            if endpoint in {"register", "start_timer"}:
                return {"status": "ok"}
            # evaluate:序号=已评估次数,分数按序号变化以区分两次
            attempt_number = len([call for call in calls if call[0] == "evaluate"])
            return {
                "aggregate_improvement": attempt_number / 10,
                "best_aggregate_improvement": attempt_number / 10,
                "raw_scores": {},
                "per_instance_improvement": {},
                "attempt": attempt_number,
            }

    # 任务包目录
    task_package_dir = tmp_path / "task-package"
    data_dir = task_package_dir / "problem" / "data"
    data_dir.mkdir(parents=True)
    # 实例化任务:配置控制 token 文件
    task = TokenContractTask(
        {
            # 基准名
            "benchmark": "naturebench",
            # 任务名
            "task_name": "fake-nature-task",
            # 指标越高越好
            "higher_is_better": True,
            # 数据目录
            "data_dir": str(data_dir),
            # 题目目录
            "problem_dir": str(data_dir.parent),
            # 任务包目录
            "task_dir": str(task_package_dir),
            # 工作区根目录(每个 attempt 一个子工作区)
            "workspace_root": str(tmp_path / "workspaces"),
            # 评估服务地址
            "eval_service_url": "http://127.0.0.1:8321",
            # 评估控制 token 文件路径
            "eval_control_token_file": str(control_token_file),
            # 批次名
            "batch_name": "unit-batch",
            # 本地执行
            "execution_mode": "local",
            # 任务描述
            "task_description": "Do the fake task.",
            # 数据描述
            "data_description": "Use DATA_DIR.",
            # 执行超时
            "execution_timeout": 600,
        },
        time_budget=14400,
    )

    # 初始化后两次 step(两次 attempt)
    state, _ = task.prepare()
    task.step_task(state, "print('first')")
    task.step_task(state, "print('second')")

    # 端点次序:首次完整流程 + 第二次 attempt 重新 register(刷新绑定)再 evaluate
    assert [endpoint for endpoint, _, _ in calls] == [
        "register",
        "start_timer",
        "evaluate",
        "register",
        "evaluate",
    ]
    # 收集两次注册负载
    registrations = [
        payload for endpoint, payload, _ in calls if endpoint == "register"
    ]
    # 收集两次评估负载
    evaluations = [payload for endpoint, payload, _ in calls if endpoint == "evaluate"]
    # 两次注册使用同一个 eval token(来自控制文件)
    assert registrations[0]["eval_token"] == registrations[1]["eval_token"]
    # 评估负载逐字段校验:任务名/批次名/本次 attempt 的输出目录/eval token
    assert evaluations == [
        {
            # 任务名
            "task_name": "fake-nature-task",
            # 批次名
            "batch_name": "unit-batch",
            # 第一次 attempt 的输出目录
            "output_dir": str(
                tmp_path
                / "workspaces"
                / "fake-nature-task_validation_1"
                / "workspace"
                / "output"
            ),
            # 与注册 token 一致
            "eval_token": registrations[0]["eval_token"],
        },
        {
            # 任务名
            "task_name": "fake-nature-task",
            # 批次名
            "batch_name": "unit-batch",
            # 第二次 attempt 的输出目录(独立工作区)
            "output_dir": str(
                tmp_path
                / "workspaces"
                / "fake-nature-task_validation_2"
                / "workspace"
                / "output"
            ),
            # 同一 token
            "eval_token": registrations[0]["eval_token"],
        },
    ]
    # 两次注册的输出目录分别对应 validation_1 / validation_2
    assert registrations[0]["out_dir"].endswith("fake-nature-task_validation_1")
    assert registrations[1]["out_dir"].endswith("fake-nature-task_validation_2")
    # 第二次注册不是 force 重置(不覆盖首次计时)
    assert "force" not in registrations[1]
    # 候选环境里必须有 EVAL_SERVICE_URL
    assert task._solution_env(task._workspace_root())["EVAL_SERVICE_URL"]
    # 但控制 token 绝不能出现在候选进程环境里(防泄漏)
    assert (
        "service-control-token"
        not in task._solution_env(task._workspace_root()).values()
    )


def test_naturebench_scm_docker_uses_exclusive_gpu_pool_not_all_flag(tmp_path):
    """独占 GPU 走池文件选卡:不得用 --gpus all,且远程命令需带超时与繁忙卡过滤。"""
    # 动态加载 NatureBench 任务基类
    base_task = _load_module(
        "naturebench_base_task_gpu",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "base_task.py",
    )

    # 捕获容器:记录远端命令与 ssh 超时
    captured = {}

    # 伪造 ssh 返回(成功)
    class FakeCompletedProcess:
        returncode = 0
        stdout = "ok"
        stderr = ""

    # 捕获型假任务:同步置空,记录 ssh 命令
    class CapturingTask(base_task.NatureBenchTask):
        def _sync_workspace_to_scm(
            self,
            workspace: Path,
            remote_workspace: str,
        ) -> None:
            # 只记录远端工作区路径
            captured["remote_workspace"] = remote_workspace

        def _resolve_scm_task_root(self) -> str:
            # 固定远端任务根
            return "/remote/task"

        def _ssh(self, remote_command: str, **kwargs):
            # 记录远端命令与超时参数
            captured["remote_command"] = remote_command
            captured["ssh_timeout"] = kwargs.get("timeout")
            return FakeCompletedProcess()

    # 准备最小数据目录
    data_dir = tmp_path / "problem" / "data"
    data_dir.mkdir(parents=True)
    # 实例化独占 GPU 任务(卡 0/1 + 独占池文件)
    task = CapturingTask(
        {
            # 基准名
            "benchmark": "naturebench",
            # 任务名
            "task_name": "fake-nature-task",
            # 资源行:THU 独占
            "resource": "gpu_thu_normal",
            # 独占模式
            "gpu_mode": "exclusive",
            # 候选卡号列表
            "gpu_devices": [0, 1],
            # 独占池文件
            "gpu_pool_file": "/tmp/airaevo-unit-exclusive",
            # 数据目录
            "data_dir": str(data_dir),
            # 题目目录
            "problem_dir": str(data_dir.parent),
            # 任务包目录
            "task_dir": str(tmp_path),
            # 工作区根目录
            "workspace_root": str(tmp_path / "workspace"),
            # SCM docker 模式
            "execution_mode": "scm_docker",
            # SCM 主机
            "scm_host": "scm-primary",
            # 远端工作区根
            "scm_workspace_root": "/remote/workspaces/unit",
            # 远端任务根列表
            "scm_task_roots": ["/remote/tasks"],
            # 评估服务地址(三种视角)
            "eval_service_url": "http://localhost:8321",
            "scm_eval_service_url": "http://localhost:8321",
            "scm_container_eval_service_url": "http://host.docker.internal:8321",
            # 批次名
            "batch_name": "unit-batch",
            # 执行超时 60 秒
            "execution_timeout": 60,
        }
    )

    # 启动 attempt 并执行 SCM docker 解
    attempt = task._start_attempt(phase="validation")
    task._run_scm_docker_solution(
        "print('hello naturebench')",
        attempt=attempt,
    )

    # 禁止整卡通配(--gpus all / device=all)
    assert "--gpus all" not in captured["remote_command"]
    assert "device=all" not in captured["remote_command"]
    # 必须通过池选卡变量传 GPU
    assert "AIREVO_GPU_ID" in captured["remote_command"]
    # 候选卡号 0 和 1 都在命令里
    assert "0 1" in captured["remote_command"]
    # docker 以池内选中卡运行(device=$AIREVO_GPU_ID)
    assert "device=$AIREVO_GPU_ID" in captured["remote_command"]
    # docker 前有 timeout 包裹(强杀宽限 30s,总 60s)
    assert "timeout -k 30s 60s docker" in captured["remote_command"]
    # ssh 本身超时 >= 4 小时(覆盖候选执行时长)
    assert captured["ssh_timeout"] >= 14400
    # 用 nvidia-smi 做繁忙卡探测
    assert "nvidia-smi" in captured["remote_command"]
    # 忙碌判定阈值:显存占用 >= 2000MB 视为占用
    assert "AIREVO_GPU_SKIP_BUSY_MB=2000" in captured["remote_command"]
    # 忙碌判定阈值:利用率 >= 80% 视为占用
    assert "AIREVO_GPU_SKIP_BUSY_UTIL=80" in captured["remote_command"]


def test_naturebench_scm_docker_uses_shared_gpu_slot_pool(tmp_path):
    """共享 GPU 走槽位锁:固定卡号+槽位编号+存活持有者检测,保证多任务安全共卡。"""
    # 动态加载 NatureBench 任务基类
    base_task = _load_module(
        "naturebench_base_task_shared_gpu",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "base_task.py",
    )

    # 捕获容器
    captured = {}

    # 伪造 ssh 返回(成功)
    class FakeCompletedProcess:
        returncode = 0
        stdout = "ok"
        stderr = ""

    # 捕获型假任务
    class CapturingTask(base_task.NatureBenchTask):
        def _sync_workspace_to_scm(
            self,
            workspace: Path,
            remote_workspace: str,
        ) -> None:
            # 记录远端工作区
            captured["remote_workspace"] = remote_workspace

        def _resolve_scm_task_root(self) -> str:
            # 固定远端任务根
            return "/remote/task"

        def _ssh(self, remote_command: str, **kwargs):
            # 记录远端命令
            captured["remote_command"] = remote_command
            return FakeCompletedProcess()

    # 准备最小数据目录
    data_dir = tmp_path / "problem" / "data"
    data_dir.mkdir(parents=True)
    # 实例化共享 GPU 任务(9 号卡 4 槽位)
    task = CapturingTask(
        {
            # 基准名
            "benchmark": "naturebench",
            # 任务名
            "task_name": "fake-shared-task",
            # 资源行:THU 共享
            "resource": "gpu_thu_shared",
            # 共享模式
            "gpu_mode": "shared",
            # 共享卡号
            "shared_gpu_device": 9,
            # 槽位总数
            "shared_gpu_slots": 4,
            # 共享池文件
            "shared_gpu_pool_file": "/tmp/airaevo-unit-shared",
            # 数据目录
            "data_dir": str(data_dir),
            # 题目目录
            "problem_dir": str(data_dir.parent),
            # 任务包目录
            "task_dir": str(tmp_path),
            # 工作区根目录
            "workspace_root": str(tmp_path / "workspace"),
            # SCM docker 模式
            "execution_mode": "scm_docker",
            # SCM 主机
            "scm_host": "scm-primary",
            # 远端工作区根
            "scm_workspace_root": "/remote/workspaces/unit",
            # 远端任务根列表
            "scm_task_roots": ["/remote/tasks"],
            # 评估服务地址(三种视角)
            "eval_service_url": "http://localhost:8321",
            "scm_eval_service_url": "http://localhost:8321",
            "scm_container_eval_service_url": "http://host.docker.internal:8321",
            # 批次名
            "batch_name": "unit-batch",
            # 执行超时
            "execution_timeout": 60,
        }
    )

    # 启动 attempt 并执行
    attempt = task._start_attempt(phase="validation")
    task._run_scm_docker_solution(
        "print('hello naturebench')",
        attempt=attempt,
    )

    # 禁止整卡通配
    assert "--gpus all" not in captured["remote_command"]
    # 固定使用 9 号卡
    assert "AIREVO_GPU_ID=9" in captured["remote_command"]
    # 注入共享槽位变量
    assert "AIREVO_SHARED_SLOT" in captured["remote_command"]
    # 注入槽位总数
    assert "AIREVO_SHARED_GPU_SLOTS=4" in captured["remote_command"]
    # 以槽位锁文件防并发冲突
    assert "slot_${AIREVO_SLOT}.lock" in captured["remote_command"]
    # docker 固定 device=9
    assert "device=9" in captured["remote_command"]
    # nvidia-smi 做存活检测
    assert "nvidia-smi" in captured["remote_command"]
    # 检测槽位是否仍有活持有者(进程在才占槽)
    assert "airaevo_gpu_has_live_shared_holder" in captured["remote_command"]
    # 显存占用阈值
    assert "AIREVO_GPU_SKIP_BUSY_MB=2000" in captured["remote_command"]


def test_naturebench_scm_ssh_retries_transient_connection_close(
    monkeypatch,
    tmp_path,
):
    """SSH 瞬断(Connection closed, 255)应按 scm_ssh_retries 自动重试并最终成功。"""
    # 动态加载 NatureBench 任务基类
    base_task = _load_module(
        "naturebench_base_task_ssh_retry",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "base_task.py",
    )

    # 调用记账
    calls = []

    # 可参数化的伪造进程结果
    class FakeCompletedProcess:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    # 替身 subprocess.run:第一次返回瞬断错误,第二次成功
    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if len(calls) == 1:
            # 第一次:255 + Connection closed(典型瞬断)
            return FakeCompletedProcess(
                255,
                stderr="Connection closed by UNKNOWN port 65535",
            )
        # 第二次:正常成功
        return FakeCompletedProcess(0, stdout="ok")

    # 替换基类内的命令执行与重试睡眠(不真等)
    monkeypatch.setattr(base_task.subprocess, "run", fake_run)
    monkeypatch.setattr(base_task.time, "sleep", lambda delay: None)

    # 准备最小数据目录
    data_dir = tmp_path / "problem" / "data"
    data_dir.mkdir(parents=True)
    # 实例化 SCM 任务:允许 2 次 SSH 重试
    task = base_task.NatureBenchTask(
        {
            # 基准名
            "benchmark": "naturebench",
            # 任务名
            "task_name": "fake-nature-task",
            # 数据目录
            "data_dir": str(data_dir),
            # 题目目录
            "problem_dir": str(data_dir.parent),
            # 任务包目录
            "task_dir": str(tmp_path),
            # 工作区根目录
            "workspace_root": str(tmp_path / "workspace"),
            # SCM docker 模式
            "execution_mode": "scm_docker",
            # SCM 主机
            "scm_host": "scm-primary",
            # 远端工作区根
            "scm_workspace_root": "/remote/workspaces/unit",
            # 远端任务根列表
            "scm_task_roots": ["/remote/tasks"],
            # 评估服务地址(三种视角)
            "eval_service_url": "http://localhost:8321",
            "scm_eval_service_url": "http://localhost:8321",
            "scm_container_eval_service_url": "http://host.docker.internal:8321",
            # 批次名
            "batch_name": "unit-batch",
            # SSH 重试次数
            "scm_ssh_retries": 2,
        }
    )

    # 执行一次 ssh 命令(内部应自动重试)
    process = task._ssh("echo ok")

    # 最终成功:退出码 0、stdout 正确
    assert process.returncode == 0
    assert process.stdout == "ok"
    # 确实调了两次(1 次失败 + 1 次重试成功)
    assert len(calls) == 2


def test_naturebench_runner_uses_naturebench_task_and_prompt_paths():
    """runner 侧的 naturebench 适配:基准名/默认求解器/四算子与经验版 prompt 路径都要正确切换。"""
    # 动态加载单任务 runner(mle_bench 与 naturebench 共用)
    single_task_runner = _load_module(
        "airaevo_single_task_runner_for_naturebench",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "mle_bench"
        / "single_task_runner.py",
    )

    # 基准名识别:naturebench 配置返回 naturebench
    assert (
        single_task_runner._benchmark_name({"benchmark": "naturebench"})
        == "naturebench"
    )
    # 求解器默认配置相对路径
    assert single_task_runner._solver_defaults_relative_path(
        {"benchmark": "naturebench"}
    ) == Path("naturebench/evo.yaml")

    # 无经验模式:四算子 + 分析器都指向普通模板
    ordinary_paths = single_task_runner._operator_paths_for_benchmark(
        "naturebench",
        experience_enabled=False,
    )
    assert ordinary_paths["draft"] == "naturebench/aira_operators/draft.yaml"
    assert ordinary_paths["improve"] == "naturebench/aira_operators/improve.yaml"
    assert ordinary_paths["debug"] == "naturebench/aira_operators/debug.yaml"
    assert ordinary_paths["crossover"] == "naturebench/aira_operators/crossover.yaml"
    assert ordinary_paths["analyze"] == "naturebench/aira_operators/analyze.yaml"

    # 经验模式:improve/debug/crossover 换成 *_experience 模板,并启用丰富记忆摘要
    experience_paths = single_task_runner._operator_paths_for_benchmark(
        "naturebench",
        experience_enabled=True,
    )
    assert (
        experience_paths["improve"]
        == "naturebench/aira_operators/improve_experience.yaml"
    )
    assert (
        experience_paths["debug"] == "naturebench/aira_operators/debug_experience.yaml"
    )
    assert (
        experience_paths["crossover"]
        == "naturebench/aira_operators/crossover_experience.yaml"
    )
    assert experience_paths["rich_memory_summary"] == (
        "naturebench/aira_operators/rich_memory_summary.yaml"
    )


def test_naturebench_operator_templates_accept_verified_visible_data_analysis():
    """全部 naturebench 算子模板必须预留"已验证可见数据分析"占位符并声明对应输入变量。"""
    # 算子模板根目录
    operators_root = (
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "src"
        / "dojo"
        / "configs"
        / "solver"
        / "operators"
        / "naturebench"
        / "aira_operators"
    )
    # 逐个校验每个算子模板
    for filename, operator_name in [
        ("draft.yaml", "draft"),
        ("improve.yaml", "improve"),
        ("improve_experience.yaml", "improve"),
        ("debug.yaml", "debug"),
        ("debug_experience.yaml", "debug"),
        ("crossover.yaml", "crossover"),
        ("crossover_experience.yaml", "crossover"),
    ]:
        # 读入并解析模板 YAML
        payload = yaml.safe_load((operators_root / filename).read_text())
        # 取算子的首轮用户消息模板
        prompt = payload[operator_name]["init_user_message_prompt_template"]
        # 必须含"Verified Visible-Data Analysis"小节标题
        assert "Verified Visible-Data Analysis" in prompt["template"]
        # 必须含 EDA 占位符
        assert "{{visible_data_analysis}}" in prompt["template"]
        # 输入变量需声明 visible_data_analysis
        assert "visible_data_analysis" in prompt["input_variables"]

    # analyze 算子:系统模板需保留任务描述占位符
    analyze_payload = yaml.safe_load((operators_root / "analyze.yaml").read_text())
    analyze_prompt = analyze_payload["analyze"]["system_message_prompt_template"]
    assert "{{task_desc}}" in analyze_prompt["template"]


def test_naturebench_final_submit_is_disabled_by_default_and_uses_search_score():
    """naturebench 默认不做 final_submit(需显式开启),跳过时提交分回退为搜索选择分。"""
    # 动态加载单任务 runner
    single_task_runner = _load_module(
        "airaevo_single_task_runner_final_submit",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "mle_bench"
        / "single_task_runner.py",
    )

    # naturebench + 默认配置:不执行最终提交
    assert (
        single_task_runner._should_run_final_submit(
            benchmark="naturebench",
            submit_repeats=1,
            task_cfg={},
            runner_cfg={},
        )
        is False
    )
    # naturebench + runner 显式开启 final_submit:才执行
    assert (
        single_task_runner._should_run_final_submit(
            benchmark="naturebench",
            submit_repeats=1,
            task_cfg={},
            runner_cfg={"solver": {"final_submit": True}},
        )
        is True
    )
    # mlebench 基准:默认就执行最终提交
    assert (
        single_task_runner._should_run_final_submit(
            benchmark="mlebench",
            submit_repeats=1,
            task_cfg={},
            runner_cfg={},
        )
        is True
    )
    # 未做提交时,有效提交分回退为最终选择分(0.23)
    assert single_task_runner._effective_submit_score(
        benchmark="naturebench",
        submit_score=None,
        final_selection_score=0.23,
        final_submit_enabled=False,
    ) == pytest.approx(0.23)


def test_naturebench_final_selection_ignores_failed_high_score_nodes():
    """最终节点选择必须跳过"失败但分高"的节点,只能从成功节点中挑选。"""
    # 动态加载单任务 runner
    single_task_runner = _load_module(
        "airaevo_single_task_runner_selection",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "mle_bench"
        / "single_task_runner.py",
    )

    # 指标替身:携带分数与信息字典
    class Metric:
        def __init__(self, value: float, info: dict):
            self.value = value
            self.info = info

    # 节点替身:id/步数/代码/指标
    class Node:
        def __init__(self, node_id: str, value: float, info: dict):
            self.id = node_id
            self.step = len(node_id)
            self.code = f"print({node_id!r})"
            self.metric = Metric(value, info)

    # 日志替身:只暴露节点列表与根节点判定
    class Journal:
        def __init__(self, nodes: list[Node]):
            self.nodes = nodes

        def is_root_node(self, node: Node) -> bool:
            return False

    # 求解器替身:聚合日志
    class Solver:
        def __init__(self, nodes: list[Node]):
            self.journal = Journal(nodes)

    # 造一个"失败但高分"节点和一个"成功但低分"节点
    failed_high = Node(
        "failed-high",
        10.0,
        {"selection_score": 10.0, "status": "failed", "status_code": 500},
    )
    success_low = Node(
        "success-low",
        0.2,
        {"selection_score": 0.2, "status": "success", "status_code": 200},
    )

    # 执行选择
    selected = single_task_runner._select_best_available_node(
        Solver([failed_high, success_low]),
        seed=7,
        task_name="fake-task",
        sample_index=0,
        benchmark="naturebench",
    )

    # 选中的一定是成功节点(失败节点被过滤)
    assert selected is success_low
    # 选择来源标记为 validation
    assert selected._final_selection_source == "validation"


def test_naturebench_leaderboard_flags_use_official_thresholds():
    """打榜标志沿用官方阈值:0.0 达 SOTA,严格大于 0.1 才算超越 SOTA。"""
    # 动态加载单任务 runner
    single_task_runner = _load_module(
        "airaevo_single_task_runner_leaderboard_flags",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "mle_bench"
        / "single_task_runner.py",
    )

    # 负分:既没达到也没超越
    assert single_task_runner._naturebench_leaderboard_flags(-0.01) == {
        "match_sota": False,
        "surpass_sota": False,
    }
    # 恰好 0:达到 SOTA
    assert single_task_runner._naturebench_leaderboard_flags(0.0) == {
        "match_sota": True,
        "surpass_sota": False,
    }
    # 0.1:达到但不算超越(边界归 match)
    assert single_task_runner._naturebench_leaderboard_flags(0.1) == {
        "match_sota": True,
        "surpass_sota": False,
    }
    # 略超 0.1:才算真正超越
    assert single_task_runner._naturebench_leaderboard_flags(0.1001) == {
        "match_sota": True,
        "surpass_sota": True,
    }


def test_solutions_database_can_force_a_fresh_draft_after_repeated_error():
    """反复超时后可强制"新鲜草稿":跳过父代采样,且父代不足时算子自动回退。"""
    # 直接导入 evo 求解器的解数据库
    from dojo.solvers.evo.evo import SolutionsDatabase

    # 静默日志替身(info/warning/debug 全吞)
    class Logger:
        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

        def debug(self, *args, **kwargs):
            return None

    # 节点替身:id + 指标值
    class Node:
        def __init__(self, node_id: str, value: float):
            self.id = node_id
            self.metric = type("Metric", (), {"value": value})()

    # 建库:单岛、容量 4、越大越好
    database = SolutionsDatabase(
        num_islands=1,
        max_size=4,
        lower_is_better=False,
        logger=Logger(),
    )
    # 岛内放一个历史节点
    database._islands[0].nodes.append(Node("first", 0.1))
    # 声明"反复超时",请求下一轮强制全新草稿
    database.request_fresh_draft("repeated_timeout")

    # 采样:概率配置下 improve/crossover 都不应被选中
    nodes, island_id, operator = database.sample_in_context(
        {"improve": 1, "crossover": 2},
        temperature=1.0,
        crossover_prob=0.5,
        fresh_draft_prob=0.0,
    )

    # 强制新鲜草稿:无父代、岛 0、算子强制为 draft
    assert nodes == []
    assert island_id == 0
    assert operator == "draft"
    # 记录强制原因
    assert database.last_parent_selection["reason"] == "repeated_timeout"

    # 第二次采样:要求 crossover 但父代不足
    nodes, island_id, operator = database.sample_in_context(
        {"improve": 1, "crossover": 2},
        temperature=1.0,
        crossover_prob=1.0,
        fresh_draft_prob=0.0,
    )

    # 仍在岛 0
    assert island_id == 0
    # 算子回退为 improve(交叉需要 2 个父代,库存不够)
    assert operator == "improve"
    # 父代就是历史节点 first
    assert [node.id for node in nodes] == ["first"]
    # 记录回退原因:crossover 父代短缺
    assert database.last_parent_selection["operator_fallback_reason"] == (
        "crossover_parent_shortage"
    )


def test_naturebench_valid_final_score_requires_successful_node():
    """有效最终分只认成功节点:失败节点的 aggregate_improvement 再高也不可用(None)。"""
    # 动态加载单任务 runner
    single_task_runner = _load_module(
        "airaevo_single_task_runner_valid_score",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "mle_bench"
        / "single_task_runner.py",
    )

    # 指标替身:只带信息字典
    class Metric:
        def __init__(self, info: dict):
            self.info = info

    # 节点替身
    class Node:
        def __init__(self, info: dict):
            self.metric = Metric(info)

    # 失败节点(高分 0.9)
    failed_node = Node(
        {"aggregate_improvement": 0.9, "status": "failed", "status_code": 500}
    )
    # 成功节点(低分 0.2)
    successful_node = Node(
        {"aggregate_improvement": 0.2, "status": "success", "status_code": 200}
    )

    # 失败节点:提取不到有效分(None)
    assert single_task_runner._naturebench_valid_score_from_node(failed_node) is None
    # 成功节点:返回其聚合提升 0.2
    assert single_task_runner._naturebench_valid_score_from_node(
        successful_node
    ) == pytest.approx(0.2)


def test_naturebench_skipped_submit_outputs_count_as_completed(tmp_path):
    """断点续跑判定:naturebench 任务显式跳过 final_submit 且已有完整产物,视为已完成。"""
    # 动态加载批量 runner
    runner = _load_module(
        "airaevo_runner_resume_naturebench",
        REPO_ROOT / "third_party" / "aira-evo" / "examples" / "mle_bench" / "runner.py",
    )

    # 伪造任务输出目录:最优代码 + 提交代码 + 状态统计
    (tmp_path / "valid_code_final.py").write_text("print('best')", encoding="utf-8")
    (tmp_path / "submit_code.py").write_text("print('best')", encoding="utf-8")
    # stat.json:标记 final_submit 被跳过、各分齐全
    (tmp_path / "stat.json").write_text(
        json.dumps(
            {
                # 基准名
                "benchmark": "naturebench",
                # 显式跳过最终提交
                "final_submit_skipped": True,
                # 最终分
                "final_score": 0.23,
                # 提交分(回退值)
                "submit_score": 0.23,
                # 历史最优聚合提升
                "best_aggregate_improvement": 0.23,
            }
        ),
        encoding="utf-8",
    )

    # 完成判定应为 True(可安全跳过续跑)
    assert runner._has_completed_task_outputs(tmp_path) is True


def test_naturebench_operator_prompts_use_output_dir_not_submission_csv():
    """prompt 契约:NatureBench 用 OUTPUT_DIR/DATA_DIR 产出约定,不得出现 mlebench 的 submission.csv。"""
    # 算子 prompt 根目录
    prompts_root = (
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "src"
        / "dojo"
        / "configs"
        / "solver"
        / "operators"
        / "naturebench"
        / "aira_operators"
    )
    # 逐个校验 7 个算子 prompt
    for prompt_name in [
        "draft.yaml",
        "improve.yaml",
        "debug.yaml",
        "crossover.yaml",
        "improve_experience.yaml",
        "debug_experience.yaml",
        "crossover_experience.yaml",
    ]:
        # 读入 prompt 全文
        prompt_text = (prompts_root / prompt_name).read_text(encoding="utf-8")
        # 数据目录约定存在
        assert "DATA_DIR" in prompt_text
        # 输出目录约定存在
        assert "OUTPUT_DIR" in prompt_text
        # 聚合提升指标说明存在
        assert "aggregate_improvement" in prompt_text
        # 可用包占位符存在
        assert "{{packages}}" in prompt_text
        # 提示 h5py 用法
        assert "h5py" in prompt_text
        # 层拼接路径提示存在
        assert "/layers/spliced" in prompt_text
        # 绝不能出现 mlebench 的提交文件约定
        assert "submission.csv" not in prompt_text

    # analyze prompt 同样校验
    analyze_prompt = (prompts_root / "analyze.yaml").read_text(encoding="utf-8")
    assert "NatureBench" in analyze_prompt
    assert "DATA_DIR" in analyze_prompt
    assert "OUTPUT_DIR" in analyze_prompt
    assert "aggregate_improvement" in analyze_prompt
    assert "submission.csv" not in analyze_prompt

    # 校验 naturebench 求解器配置:h5py 在可用包列表里
    solver_cfg = yaml.safe_load(
        (
            REPO_ROOT
            / "third_party"
            / "aira-evo"
            / "src"
            / "dojo"
            / "configs"
            / "solver"
            / "naturebench"
            / "evo.yaml"
        ).read_text(encoding="utf-8")
    )
    assert "h5py" in solver_cfg["available_packages"]


def test_evaluate_naturebench_payloads_point_to_naturebench_builder_and_runner(
    tmp_path,
):
    """evaluate_naturebench 应把配置拆成"构建载荷"与"runner 载荷",且 api_key 不下发给 runner。"""
    # 动态加载评估编排脚本
    evaluate_naturebench = _load_module(
        "evaluate_naturebench",
        REPO_ROOT / "scripts" / "evaluate_naturebench.py",
    )
    # 构造完整层级配置:实验元信息/预算/并发/litellm 模型/数据段/搜索段
    cfg = OmegaConf.create(
        {
            # 实验名
            "experiment_name": "naturebench_smoke",
            # 输出目录
            "output_dir": str(tmp_path / "outputs"),
            # 随机种子
            "seed": 7,
            # 总时间预算
            "time_budget": 14400,
            # 模型+沙箱时间预算
            "model_plus_sandbox_time_budget": 14400,
            # 每任务采样数
            "n_samples_per_task": 1,
            # 每步候选数
            "candidates_per_step": 1,
            # 最大步数
            "max_steps": 2,
            # LLM 并发
            "llm_concurrency": 1,
            # 沙箱并发
            "sandbox": {"concurrency": 1},
            # litellm 模型列表(单模型)
            "litellm": {
                "model_list": [
                    {
                        # 模型别名
                        "model_name": "unit-model",
                        # 路由参数:密钥/地址/温度
                        "litellm_params": {
                            "api_key": "unit-key",
                            "base_url": "http://127.0.0.1:30000/v1",
                            "temperature": 0.2,
                        },
                    }
                ]
            },
            # 数据段:NatureBench 根目录与 SCM 远程参数
            "data": {
                # NatureBench 根目录
                "naturebench_root": "/tmp/NatureBench",
                # 任务清单
                "task_list": ["fake-nature-task"],
                # 评估服务地址
                "eval_service_url": "http://127.0.0.1:8321",
                # SCM docker 模式
                "execution_mode": "scm_docker",
                # 批次名
                "batch_name": "unit-batch",
                # 不做最终提交
                "submit_repeats": 0,
                # SCM 主机
                "scm_host": "scm-primary",
                # 远端工作区根
                "scm_workspace_root": "/remote/workspaces/unit",
                # 远端任务根
                "scm_task_roots": ["/remote/naturebench/tasks/part1"],
                # 主机视角评估地址
                "scm_eval_service_url": "http://localhost:8321",
                # 容器视角评估地址
                "scm_container_eval_service_url": "http://host.docker.internal:8321",
            },
            # 搜索段:runner 子配置
            "search": {
                "runner": {
                    # aira-evo 包根
                    "package_root": "third_party/aira-evo",
                    # 任务级并发
                    "task_concurrency": 1,
                    # 非严格续跑
                    "strict_resume": False,
                    # LLM 适配
                    "llm": {"api": "litellm", "provider": "selfhosted"},
                    # 求解器超时
                    "solver": {"execution_timeout": 60},
                    # 解释器用符号链接
                    "interpreter": {"use_symlinks": True},
                    # 日志打印到控制台
                    "logger": {"use_console": True},
                }
            },
        }
    )

    # 拆分出构建载荷与 runner 载荷
    prepare_payload, runner_payload = evaluate_naturebench._build_payloads(
        cfg,
        output_dir=tmp_path / "outputs",
        task_root=tmp_path / "tasks",
    )

    # 构建载荷:数据段逐字段透传
    assert prepare_payload["data"]["naturebench_root"] == "/tmp/NatureBench"
    assert prepare_payload["data"]["task_list"] == ["fake-nature-task"]
    assert prepare_payload["data"]["execution_mode"] == "scm_docker"
    assert prepare_payload["data"]["submit_repeats"] == 0
    assert prepare_payload["data"]["scm_host"] == "scm-primary"
    assert prepare_payload["data"]["scm_workspace_root"] == "/remote/workspaces/unit"
    # 远端任务根以列表透传
    assert prepare_payload["data"]["scm_task_roots"] == [
        "/remote/naturebench/tasks/part1"
    ]
    # 主机视角评估地址透传
    assert prepare_payload["data"]["scm_eval_service_url"] == "http://localhost:8321"
    # 容器视角评估地址透传
    assert (
        prepare_payload["data"]["scm_container_eval_service_url"]
        == "http://host.docker.internal:8321"
    )
    # runner 载荷:任务根目录指向构建产物
    assert runner_payload["task_root"] == str(tmp_path / "tasks")
    # LLM 用别名,而非路由配置
    assert runner_payload["llm"]["model_id"] == "unit-model"
    # 密钥不得进入 runner 载荷
    assert "api_key" not in runner_payload["llm"]
    # 步数上限来自 max_steps
    assert runner_payload["solver"]["step_limit"] == 2
    # 执行超时透传
    assert runner_payload["solver"]["execution_timeout"] == 60


def test_naturebench_default_configs_skip_submit_and_reduce_search_artifacts():
    """默认配置契约:SCM 数据配置跳过提交+开预检;搜索配置关产物导出+收敛调试开销。"""
    # 读入 SCM 全量数据配置
    data_cfg = OmegaConf.load(
        REPO_ROOT / "tts_search" / "configs" / "data" / "naturebench_scm_all.yaml"
    )
    # 读入 airaevo naturebench 搜索配置
    search_cfg = yaml.safe_load(
        (
            REPO_ROOT / "tts_search" / "configs" / "search" / "airaevo_naturebench.yaml"
        ).read_text(encoding="utf-8")
    )

    # 数据配置:不做最终提交
    assert data_cfg["submit_repeats"] == 0
    # 候选预检开启
    assert data_cfg["candidate_preflight"] is True
    # import 静态检查开启
    assert data_cfg["candidate_preflight_imports"] is True
    # 预检超时 45 秒
    assert data_cfg["candidate_preflight_timeout"] == 45
    # GPU 等待超时 1 小时
    assert data_cfg["scm_gpu_wait_timeout"] == 3600
    # 搜索配置:不执行最终提交
    assert search_cfg["runner"]["solver"]["final_submit"] is False
    # 不导出搜索结果(减少产物)
    assert search_cfg["runner"]["solver"]["export_search_results"] is False
    # 日志不打印到控制台
    assert search_cfg["runner"]["logger"]["use_console"] is False
    # 调试深度上限 2
    assert search_cfg["runner"]["solver"]["max_debug_depth"] == 2
    # 单次调试时间上限 1200 秒
    assert search_cfg["runner"]["solver"]["max_debug_time"] == 1200
    # 新鲜草稿概率 0.2(维持多样性)
    assert search_cfg["runner"]["solver"]["fresh_draft_prob"] == pytest.approx(0.2)
    # 单任务墙钟上限 6 小时
    assert search_cfg["runner"]["solver"]["max_wall_time_secs"] == 21600


def test_naturebench_summary_does_not_require_mle_leaderboard(tmp_path):
    """汇总 summary 对 naturebench 任务不依赖 MLE 排行榜:奖牌列给 N/A,score_best@k 正常。"""
    # 复用 tts_search 的汇总工具
    from tts_search import eval_utils

    # 伪造任务输出目录:program_ep_0/<task>/stat.json
    task_dir = tmp_path / "program_ep_0" / "fake-task"
    task_dir.mkdir(parents=True)
    # stat.json:naturebench 基准 + 提交分等
    (task_dir / "stat.json").write_text(
        json.dumps(
            {
                # 基准名
                "benchmark": "naturebench",
                # 任务名
                "task_name": "fake-task",
                # 提交分
                "submit_score": 0.123,
                # 提交等级
                "submit_grade": 1.0,
                # 状态计数
                "status_count": {"success": 1},
                # 总花费
                "total_cost": 0.0,
            }
        ),
        encoding="utf-8",
    )

    # 生成汇总:任务元信息刻意指向不存在的排行榜目录
    summary = eval_utils.write_summary_csv(
        tmp_path,
        {
            "fake-task": {
                # 基准名
                "benchmark": "naturebench",
                # 任务名
                "task_name": "fake-task",
                # 无 MLE 排行榜的数据目录
                "data_dir": str(tmp_path / "no-mle-leaderboard-here"),
            }
        },
        expected_task_names=["fake-task"],
    )

    # summary.csv 落盘
    assert (tmp_path / "summary.csv").exists()
    # 任务列正确
    assert summary.loc[0, "Task"] == "fake-task"
    # best@k 分取自 stat.json
    assert summary.loc[0, "score_best@k"] == pytest.approx(0.123)
    # 奖牌均值列无排行榜支撑,给 N/A
    assert summary.loc[0, "score_avg@k_medal"] == "N/A"


def test_generic_llm_converts_generation_kwargs_to_plain_containers(
    monkeypatch,
    tmp_path,
):
    """GenericLLM 需把 OmegaConf 的生成参数转成原生 dict,保证 extra_body 能被客户端正常消费。"""
    # 提供日志目录环境变量(被测代码要求)
    monkeypatch.setenv("LOGGING_DIR", str(tmp_path / "logs"))

    # 导入通用 LLM 封装
    from dojo.core.solvers.llm_helpers import generic_llm

    # 捕获传给客户端的 kwargs
    captured_kwargs = {}

    # 客户端替身:记录 kwargs 并返回固定生成结果
    class FakeClient:
        # 内容键名约定
        client_content_key = "content"

        def query(self, messages, **kwargs):
            # 记录本次查询参数
            captured_kwargs.update(kwargs)
            # 返回代码块文本与 token 统计
            return "```python\nprint(1)\n```", {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "cost": 0.0,
            }

    # 替换客户端工厂,恒返回 FakeClient
    monkeypatch.setattr(generic_llm, "get_client", lambda client_cfg: FakeClient())

    # 构造 GenericLLM 配置:客户端+生成参数+三类消息模板
    cfg = OmegaConf.create(
        {
            # LLM 段
            "llm": {
                # 客户端配置
                "client": {
                    # 走 litellm API
                    "api": "litellm",
                    # 模型别名
                    "model_id": "unit-model",
                    # 服务地址
                    "base_url": "http://127.0.0.1:30002/v1",
                    # 不用 Azure
                    "use_azure_client": False,
                    # 自托管提供方
                    "provider": "selfhosted",
                },
                # 生成参数:温度 + extra_body(嵌套 dict)
                "generation_kwargs": {
                    "temperature": 0.2,
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
                },
            },
            # 系统消息模板
            "system_message_prompt_template": {
                "template": "{{public_system_prompt}}",
                "input_variables": ["public_system_prompt"],
                "partial_variables": {},
            },
            # 首轮用户消息模板
            "init_user_message_prompt_template": {
                "template": "{{task_description}}",
                "input_variables": ["task_description"],
                "partial_variables": {},
            },
            # 后续用户消息模板(空)
            "user_message_prompt_template": {
                "template": "",
                "input_variables": [],
                "partial_variables": {},
            },
        }
    )

    # 实例化并调用一次
    llm = generic_llm.GenericLLM(cfg)
    output, _ = llm(
        query_data={
            "public_system_prompt": "",
            "task_description": "write code",
        }
    )

    # 输出保留代码块围栏
    assert output.startswith("```python")
    # extra_body 已转成原生 dict
    assert isinstance(llm.generation_kwargs["extra_body"], dict)
    # 不再是 OmegaConf 节点
    assert not OmegaConf.is_config(llm.generation_kwargs["extra_body"])
    # 传给客户端的 extra_body 同样是原生 dict
    assert isinstance(captured_kwargs["extra_body"], dict)
    # 嵌套内容原样保留
    assert captured_kwargs["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": True
    }


def test_naturebench_concurrent_attempts_keep_distinct_workspaces(tmp_path):
    """并发 attempt 的竞态安全:两个线程同时执行,输出目录与尝试序号互不串扰。"""
    # 动态加载 NatureBench 任务基类
    base_task = _load_module(
        "naturebench_base_task_concurrent_attempts",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "base_task.py",
    )
    # dojo 任务常量:辅助评估信息键
    from dojo.core.tasks.constants import AUX_EVAL_INFO

    # 双方在运行点集合,制造最大并发窗口
    barrier = threading.Barrier(2)

    # 并发假任务:运行时在 barrier 汇合并返回各自 attempt 的输出目录
    class ConcurrentTask(base_task.NatureBenchTask):
        def _run_solution(self, code: str, *, phase: str, attempt=None) -> dict:
            # 等两个 attempt 同时进入
            barrier.wait(timeout=5)
            # 输出目录优先取 attempt;缺失才回退活动目录
            output_dir = (
                attempt.output_dir if attempt is not None else self._active_output_dir
            )
            # 返回执行记录(带输出目录)
            return {
                "status_code": 0,
                "status": "success",
                "raw_run_log": code,
                "clear_run_log": code,
                "feedback": code,
                "run_time": 0.01,
                "output_dir": str(output_dir),
            }

        def _post_evaluate(self, output_dir: Path) -> dict:
            # 固定评估结果
            return {
                "aggregate_improvement": 0.1,
                "best_aggregate_improvement": 0.1,
                "raw_scores": {},
                "per_instance_improvement": {},
            }

    # 准备最小数据目录
    data_dir = tmp_path / "problem" / "data"
    data_dir.mkdir(parents=True)
    # 实例化任务:关闭预检(候选代码是合法 print)
    task = ConcurrentTask(
        {
            # 基准名
            "benchmark": "naturebench",
            # 任务名
            "task_name": "race-task",
            # 指标越高越好
            "higher_is_better": True,
            # 数据目录
            "data_dir": str(data_dir),
            # 题目目录
            "problem_dir": str(data_dir.parent),
            # 任务包目录
            "task_dir": str(tmp_path),
            # 工作区根目录
            "workspace_root": str(tmp_path / "workspace"),
            # 评估服务地址
            "eval_service_url": "http://127.0.0.1:8321",
            # 批次名
            "batch_name": "unit-batch",
            # 本地执行
            "execution_mode": "local",
            # 关闭预检
            "candidate_preflight": False,
        }
    )

    # 单个候选的执行体:全新空状态跑一次
    def run_candidate(index: int):
        state, result = task.step_task({}, f"print({index})")
        return state, result[AUX_EVAL_INFO]

    # 双线程并发执行
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run_candidate, (1, 2)))

    # 汇总两个 attempt 的输出目录与序号
    output_dirs = {payload["output_dir"] for _, payload in results}
    attempt_indices = {state["naturebench_attempt_index"] for state, _ in results}
    # 两个不同的输出目录(工作区隔离)
    assert len(output_dirs) == 2
    # 序号恰为 {1, 2}(自增无丢失)
    assert attempt_indices == {1, 2}


def test_naturebench_docker_timeout_forcibly_removes_named_container(
    monkeypatch,
    tmp_path,
):
    """docker 超时后必须按 --name 强制 rm 容器,且网络只暴露宿主网关映射。"""
    # 动态加载 NatureBench 任务基类
    base_task = _load_module(
        "naturebench_base_task_docker_timeout",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "base_task.py",
    )
    # 命令记账
    calls = []

    # 替身命令执行:docker run 恒超时,其余命令成功
    def fake_run(command, **kwargs):
        calls.append(list(command))
        if command[:2] == ["docker", "run"]:
            # 抛超时异常(带部分输出,模拟真实子进程被杀)
            raise subprocess.TimeoutExpired(
                command,
                timeout=kwargs["timeout"],
                output="partial stdout",
                stderr="partial stderr",
            )
        # 其他命令(docker rm 等)返回成功
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    # 替换命令执行
    monkeypatch.setattr(base_task.subprocess, "run", fake_run)
    # 准备数据目录与工作区/输出目录
    data_dir = tmp_path / "problem" / "data"
    workspace = tmp_path / "workspace"
    output_dir = workspace / "output"
    data_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    # 实例化 docker 模式任务(超时 10 秒)
    task = base_task.NatureBenchTask(
        {
            # 基准名
            "benchmark": "naturebench",
            # 任务名
            "task_name": "docker-timeout-task",
            # 数据目录
            "data_dir": str(data_dir),
            # 题目目录
            "problem_dir": str(data_dir.parent),
            # 任务包目录
            "task_dir": str(tmp_path),
            # 工作区根目录
            "workspace_root": str(tmp_path / "workspaces"),
            # 评估服务地址
            "eval_service_url": "http://127.0.0.1:8321",
            # 批次名
            "batch_name": "unit-batch",
            # docker 执行模式
            "execution_mode": "docker",
            # 执行超时
            "execution_timeout": 10,
        }
    )

    # 执行 docker 解
    result = task._run_docker_solution(
        "print('hello')",
        workspace=workspace,
        output_dir=output_dir,
        attempt_index=7,
    )

    # 取首条命令(docker run)并解析容器名
    run_command = calls[0]
    container_name = run_command[run_command.index("--name") + 1]
    # 不加 --network 隔离选项
    assert "--network" not in run_command
    # 用 host-gateway 映射让容器可达宿主服务
    assert "host.docker.internal:host-gateway" in run_command
    # 注入容器视角的评估服务地址
    assert "EVAL_SERVICE_URL=http://host.docker.internal:8321" in run_command
    # 超时状态码 504
    assert result["status_code"] == 504
    # 状态标记 timeout
    assert result["status"] == "timeout"
    # 第二条命令必须按容器名强制删除
    assert calls[1] == ["docker", "rm", "-f", container_name]


def test_naturebench_remote_tar_command_quotes_workspace_component():
    """远端 tar 解包命令必须对工作区路径做 shell 引号转义(防空格/分号注入)。"""
    # 动态加载 NatureBench 任务基类
    base_task = _load_module(
        "naturebench_base_task_remote_quote",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "base_task.py",
    )
    # 恶意样例路径:含空格与命令注入片段
    remote_workspace = "/remote/work dir;touch /tmp/should-not-run"

    # 生成远端解包命令
    remote_command = base_task.NatureBenchTask._remote_tar_extract_command(
        remote_workspace
    )

    # 命令必须等于"tar -C <引号包裹的路径> -xf -"(注入片段被安全转义)
    assert remote_command == ("tar -C " + shlex.quote(remote_workspace) + " -xf -")


def test_naturebench_local_candidate_env_does_not_inherit_secrets(
    monkeypatch,
    tmp_path,
):
    """本地候选进程环境按白名单构造:DATA_DIR/PATH 可见,云密钥/API 密钥不得继承。"""
    # 动态加载 NatureBench 任务基类
    base_task = _load_module(
        "naturebench_base_task_candidate_env",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "base_task.py",
    )
    # 预置两个不应泄漏的敏感环境变量
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "do-not-copy")
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-copy")
    # 准备最小数据目录
    data_dir = tmp_path / "problem" / "data"
    data_dir.mkdir(parents=True)
    # 实例化本地执行任务
    task = base_task.NatureBenchTask(
        {
            # 基准名
            "benchmark": "naturebench",
            # 任务名
            "task_name": "candidate-env-task",
            # 数据目录
            "data_dir": str(data_dir),
            # 题目目录
            "problem_dir": str(data_dir.parent),
            # 任务包目录
            "task_dir": str(tmp_path),
            # 工作区根目录
            "workspace_root": str(tmp_path / "workspaces"),
            # 评估服务地址
            "eval_service_url": "http://127.0.0.1:8321",
            # 批次名
            "batch_name": "unit-batch",
            # 本地执行
            "execution_mode": "local",
        }
    )

    # 构造候选进程环境
    env = task._solution_env(tmp_path / "output")

    # 数据目录注入正确
    assert env["DATA_DIR"] == str(data_dir)
    # PATH 可继承(运行时需要)
    assert "PATH" in env
    # AWS 密钥不得出现
    assert "AWS_SECRET_ACCESS_KEY" not in env
    # OpenAI 密钥不得出现
    assert "OPENAI_API_KEY" not in env


def test_runner_config_redaction_removes_nested_api_keys():
    """runner 落盘前的配置脱敏:嵌套 api_key/token 及 overrides 里的密钥赋值全部清除。"""
    # 动态加载批量 runner
    runner = _load_module(
        "airaevo_runner_redaction",
        REPO_ROOT / "third_party" / "aira-evo" / "examples" / "mle_bench" / "runner.py",
    )
    # 构造含敏感信息的配置:顶层 llm 密钥/嵌套 token/overrides 里的密钥字符串
    payload = {
        "llm": {"api_key": "top-secret", "model_id": "unit-model"},
        "nested": [{"token": "also-secret", "safe": 3}],
        "overrides": ["litellm.model_list.0.litellm_params.api_key=cli-secret"],
    }

    # 执行脱敏
    redacted = runner._redact_sensitive_config(payload)

    # 顶层 api_key 被置 None
    assert redacted["llm"]["api_key"] is None
    # 嵌套 token 被置 None
    assert redacted["nested"][0]["token"] is None
    # YAML 序列化后不出现任何明文密钥
    assert "top-secret" not in yaml.safe_dump(redacted)
    assert "also-secret" not in yaml.safe_dump(redacted)
    assert "cli-secret" not in yaml.safe_dump(redacted)
