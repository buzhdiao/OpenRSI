# 中文说明:本测试文件回归验证 OpenMLE-Evo 侧"文档化的既有 Gym 消费路径"。
# 即:OpenMLE-Gym 打包的输入(titanic-extended-task.json)经 cli prepare-input
# 生成 parquet 后,既可被 Evo 侧 aira-evo 的 build_tasks/base_task(SandboxMLEBenchTask)
# 消费,也可被 SFT 侧同一套 builder 消费,且两条链路产出的任务配置关键字段一致;
# 同时验证沙箱打分回传路径(tasks/submit_root 两种根目录)与分数选取逻辑。
# 位置:OpenMLE-Evo/tests 下,是 Evo-Gym-Erl 三仓对齐的契约型回归测试。

# 延迟注解:类型标注运行期不求值
from __future__ import annotations

# 按文件路径动态加载模块(build_tasks.py/base_task.py 不在常规包路径上)
import importlib.util
# JSON 读写:加载示例输入包、写 task_metadata.json
import json
# 校验真实 scorer 输出的分数是否为有限值
import math
# 复制/修改环境变量传给子进程(如 PYTHONPATH 指向 SFT)
import os
# 正则:从真实 scorer 的 stdout 提取 ##SCORE## 分数
import re
# 复制 SFT 侧 build_tasks.py 脚本到临时目录再执行
import shutil
# 启动子进程运行 openmle_gym.cli 与构建脚本
import subprocess
# 子进程需指定解释器路径
import sys
# Path:构造包目录、parquet、配置文件路径
from pathlib import Path

# parquet 读写与首行恢复校验
import pandas as pd
# pytest:测试框架、fixture、参数化、跳过条件
import pytest
# OmegaConf:构造/保存/加载 yaml 任务配置
from omegaconf import OmegaConf

# 仓库根目录:OpenMLE-Evo(tests 的上一级)
ROOT = Path(__file__).resolve().parents[1]
# 示例输入包:Gym 侧预置的 titanic-extended 任务 JSON(含 prompt 与 metadata)
EXAMPLE = (
    ROOT.parent / "OpenMLE-Gym/examples/consumer-inputs/titanic-extended-task.json"
)
# 被测任务的任务名(uuid 形式),包目录与配置目录均按此命名
TASK = "titanic-extended@1"


# autouse fixture:每个用例都把 LOGGING_DIR 重定向到临时目录,
# 避免被测代码把日志写进真实目录污染仓库
@pytest.fixture(autouse=True)
def logging_directory(tmp_path, monkeypatch):
    # 通过 monkeypatch 设置环境变量,被测模块读取后即指向临时日志目录
    monkeypatch.setenv("LOGGING_DIR", str(tmp_path / "logs"))


def module(name, path):
    """按文件路径加载一个独立模块(绕开常规 import 机制)。"""
    # 以给定模块名与文件路径构造模块规格
    spec = importlib.util.spec_from_file_location(name, path)
    # 根据规格创建空模块对象
    result = importlib.util.module_from_spec(spec)
    # 真正执行模块代码,完成初始化
    spec.loader.exec_module(result)
    # 返回可用的模块对象
    return result


def build(tmp_path, validation_dir, submit_root):
    """搭建端到端最小环境:从示例 JSON 打包 -> 生成 parquet -> 构建任务配置。

    返回 (adapter 模块, SandboxMLEBenchTask 实例, 使用的 yaml 配置路径),
    供后续用例对评估路径与分数做断言。
    """
    # 读入示例输入包原始内容(prompt + metadata),作为比对基准
    original = json.loads(EXAMPLE.read_text())
    # 构造待打包的任务目录 download/<TASK>
    package = tmp_path / "download" / TASK
    # 创建 info 子目录(存放任务元数据与数据描述)
    (package / "info").mkdir(parents=True)
    # 创建 data/public 子目录(存放描述文件等公开数据)
    (package / "data/public").mkdir(parents=True)
    # 复制一份 metadata,准备把内嵌的长文本抽到独立文件
    metadata = original["metadata"].copy()
    # 把任务描述写入 data/public/description.txt,并从 metadata 中移除该键
    (package / "data/public/description.txt").write_text(
        metadata.pop("task_description")
    )
    # 把数据描述写入 info/data_description.txt,同样从 metadata 移除
    (package / "info/data_description.txt").write_text(metadata.pop("data_description"))
    # 移除 data_dir 键:该字段在打包时由 --sandbox-task-dir 重新指定
    metadata.pop("data_dir")
    # 剩余 metadata 序列化为 info/task_metadata.json
    (package / "info/task_metadata.json").write_text(json.dumps(metadata))
    # 目标 parquet 输出路径
    parquet = tmp_path / "eval.parquet"
    # 子进程调用 openmle_gym.cli 的 prepare-input 子命令完成"输入包 -> 评估 parquet"转换
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "openmle_gym.cli",
            "prepare-input",
            "--task-dir",
            str(package),
            "--sandbox-task-dir",
            validation_dir,
            "--output",
            str(parquet),
        ],
        cwd=ROOT.parent / "OpenMLE-Gym",
        capture_output=True,
        text=True,
    )
    # 转换命令必须成功,失败时打印 stdout+stderr 便于定位
    assert result.returncode == 0, result.stdout + result.stderr
    # 读回 parquet 首行(该示例只有一行样本)
    restored = pd.read_parquet(parquet).iloc[0]
    # parquet 里恢复出的 prompt 应与原始 prompt 列表完全一致(无损往返)
    assert list(restored.prompt) == original["prompt"]
    # 预期的 metadata:原始 metadata 但 data_dir 被替换为传入的 validation_dir
    expected = original["metadata"].copy()
    expected["data_dir"] = validation_dir
    # 校验 metadata 无损往返
    assert restored.metadata == expected

    # 构造 builder 所需的配置字典:数据与沙箱两大块
    cfg = {
        "data": {
            # 评估数据指向刚生成的 parquet
            "eval_data": str(parquet),
            # prompt 存放的列名
            "input_key": "prompt",
            # metadata 存放的列名
            "metadata_key": "metadata",
            # 排行榜输出目录
            "leaderboard_dir": str(tmp_path),
            # 提交目录占位(本链路不使用)
            "submit_dir": "unused",
            # 提交数据根目录(由用例参数化,验证路径映射)
            "submit_data_dir_root": submit_root,
            # 评估协议:self_valid(自我验证划分)
            "evaluation_protocol": "self_valid",
        },
        "sandbox": {
            # 沙箱地址映射:cpu 资源指向占位 URL(测试中会被 monkeypatch 掉)
            "base_url_map": [{"resource": "cpu", "base_url": "http://unused.example"}],
            # 单次沙箱作业超时(秒)
            "job_timeout": 30,
            # 等待结果总超时(秒)
            "wait_timeout": 30,
            # 轮询间隔为 0(测试尽快返回)
            "poll_interval": 0,
            # 启用 score->reward 映射
            "use_score2reward": True,
            # 启用运行日志清洗打分
            "use_clear_run_log_score": True,
        },
    }
    # 把配置字典保存为 yaml 文件,供 build_tasks 消费
    config = tmp_path / "build.yaml"
    OmegaConf.save(OmegaConf.create(cfg), config)
    # 动态加载 Evo 侧的 build_tasks 构建脚本模块
    builder = module(
        "gym_example_builder",
        ROOT / "third_party/aira-evo/examples/mle_bench/build_tasks.py",
    )
    # 执行构建:按配置把任务物化到 built/<TASK> 目录
    builder.build_tasks(config, tmp_path / "built")
    # 读回物化后的任务 config.yaml 并转为普通字典
    task_cfg = OmegaConf.to_container(
        OmegaConf.load(tmp_path / "built" / TASK / "config.yaml")
    )
    # 动态加载 base_task 适配器模块
    adapter = module(
        "gym_example_task",
        ROOT / "third_party/aira-evo/examples/mle_bench/base_task.py",
    )
    # 返回适配器模块、任务实例与配置路径
    return adapter, adapter.SandboxMLEBenchTask(task_cfg), config


# 参数化提交数据根目录:验证 validation 与 submission 两种根路径映射都正确
@pytest.mark.parametrize("submit_root", ["/mounted/tasks", "/mounted/test"])
def test_existing_evo_phase_paths_and_scores(tmp_path, monkeypatch, submit_root):
    """验证两个评估阶段(validation/test)分别请求正确的数据目录并按默认逻辑选分。"""
    # 构建端到端环境:validation 目录固定为 /mounted/tasks/<TASK>,提交根目录参数化
    adapter, task, _ = build(tmp_path, f"/mounted/tasks/{TASK}", submit_root)
    # 记录沙箱调用实际收到的 data_dir 列表
    requested = []

    async def response(**kwargs):
        # 捕获本次请求携带的数据目录
        requested.append(kwargs["data_dir"])
        # 中文说明:故意让 stdout 里的分数与 JSON 分数不一致(0.99 vs 0.75),
        # 以锁定"默认取 JSON score 而非解析 stdout"的现有选择行为
        return 200, {
            "result": {
                "result": "success",
                "score": 0.75,
                "run_log": "Final Validation Score: 0.99",
            }
        }

    # 用假沙箱响应替换适配器的 get_sandbox_result,避免真实网络
    monkeypatch.setattr(adapter, "get_sandbox_result", response)
    # 设置仅测试用的沙箱 API key(被测代码要求环境变量存在)
    monkeypatch.setenv("SANDBOX_CPU_API_KEY", "test-only")
    # 跑 validation 阶段评估
    validation = task.evaluate_code("print('unchanged code')", phase="validation")
    # 跑 test(提交)阶段评估
    submission = task.evaluate_code("print('unchanged code')", phase="test")
    # 断言两阶段请求的数据目录:validation 用任务目录,submission 用 submit_root 拼接
    assert requested == [f"/mounted/tasks/{TASK}", f"{submit_root}/{TASK}"]
    # validation 阶段的选分应取 JSON score(0.75),而非 stdout 的 0.99
    assert validation["selection_score"] == 0.75
    # submission 阶段同样取 JSON score
    assert submission["submit_score"] == 0.75
    # build_submit_code 对无占位符的代码应原样返回(不做包装)
    assert (
        task.build_submit_code("print('unchanged code')") == "print('unchanged code')"
    )


def test_same_parquet_is_consumed_by_existing_sft_builder(tmp_path):
    """验证同一份 parquet 被 SFT 侧既有 builder 消费时,任务配置与 Evo 侧一致。"""
    # 先走同样的构建流程拿到 yaml 配置
    _, _, config = build(tmp_path, f"/mounted/tasks/{TASK}", "/mounted/tasks")
    # SFT 源码根目录
    sft = ROOT.parent / "OpenMLE-ERL/SFT"
    # SFT 构建输出目录
    output = tmp_path / "sft-built"
    # 创建输出目录
    output.mkdir()
    # 目标脚本路径:把 SFT 侧 build_tasks.py 拷贝到临时目录执行
    script = output / "build_tasks.py"
    # 复制 SFT 侧构建脚本
    shutil.copyfile(
        sft / "third_party/aira-evo/examples/mle_bench/build_tasks.py", script
    )
    # 复制当前环境并注入 PYTHONPATH 指向 SFT,使脚本可导入其依赖
    env = os.environ.copy()
    env["PYTHONPATH"] = str(sft)
    # 子进程运行 SFT 构建脚本,复用同一份 config
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    # SFT 构建必须成功
    assert result.returncode == 0, result.stderr
    # 读取 Evo 侧物化出的任务配置
    evo_config = OmegaConf.to_container(
        OmegaConf.load(tmp_path / "built" / TASK / "config.yaml")
    )
    # 读取 SFT 侧物化出的任务配置
    sft_config = OmegaConf.to_container(OmegaConf.load(output / TASK / "config.yaml"))
    # 逐个比对两边任务配置的关键字段,保证两链路完全对齐
    for key in (
        "data_dir",
        "uuid",
        "task_description",
        "data_description",
        "public_system_prompt",
        "public_user_prompt",
        "higher_is_better",
    ):
        # 任一字段不一致即失败(契约破坏)
        assert sft_config[key] == evo_config[key]


# 仅在显式设置 OPENMLE_GYM_TEST_PACKAGE 时执行(需要真实下载的任务包)
@pytest.mark.skipif(
    not os.environ.get("OPENMLE_GYM_TEST_PACKAGE"),
    reason="Set a downloaded package for real scorer verification",
)
def test_downloaded_package_matches_record_and_existing_scorer(tmp_path):
    """真实任务包验证:打包内容与 JSON 记录一致,且既有真实 scorer 能算出有效分数。"""
    # 解析环境变量指定的真实任务包路径
    package = Path(os.environ["OPENMLE_GYM_TEST_PACKAGE"]).resolve()
    # 读入示例 JSON 记录作为比对基准
    record = json.loads(EXAMPLE.read_text())
    # 任务描述文件内容应与记录中的 metadata 一致
    assert (
        record["metadata"]["task_description"]
        == (package / "data/public/description.txt").read_text()
    )
    # 数据描述文件内容应与记录一致
    assert (
        record["metadata"]["data_description"]
        == (package / "info/data_description.txt").read_text()
    )
    # 真实包的 parquet 输出路径
    prepared = tmp_path / "real-package.parquet"
    # 子进程调用 prepare-input,显式带上记录中的 uuid 以复现记录
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "openmle_gym.cli",
            "prepare-input",
            "--task-dir",
            str(package),
            "--sandbox-task-dir",
            "/mounted/tasks/" + TASK,
            "--task-uuid",
            record["metadata"]["uuid"],
            "--output",
            str(prepared),
        ],
        cwd=ROOT.parent / "OpenMLE-Gym",
        capture_output=True,
        text=True,
    )
    # 转换必须成功
    assert result.returncode == 0, result.stdout + result.stderr
    # 读回 parquet 首行
    restored = pd.read_parquet(prepared).iloc[0]
    # 记录中的 data_dir 替换为沙箱任务目录后再比对
    record["metadata"]["data_dir"] = "/mounted/tasks/" + TASK
    # prompt 与 metadata 均应与记录完全一致
    assert list(restored.prompt) == record["prompt"]
    assert restored.metadata == record["metadata"]
    # 既有真实 scorer 脚本路径(读提交文件并按 metric 打分)
    scorer = ROOT.parent / "OpenMLE-Gym/openmle-sandbox/node_workers/read_and_metric.py"
    # 子进程运行真实 scorer:传入任务包、答案文件与提交样例
    result = subprocess.run(
        [
            sys.executable,
            str(scorer),
            "-d",
            str(package),
            "-a",
            str(package / "data/private/test_answer.csv"),
            "-p",
            str(package / "data/public/sample_submission.csv"),
        ],
        capture_output=True,
        text=True,
    )
    # scorer 必须成功退出
    assert result.returncode == 0, result.stdout + result.stderr
    # 用正则从 stdout 提取 ##SCORE## 后的分数
    match = re.search(r"##SCORE##([0-9.eE+-]+)", result.stdout)
    # 必须能匹配到分数,否则打印全部 stdout 便于排查
    assert match, result.stdout
    # 分数转 float
    score = float(match.group(1))
    # 分数必须是有限值且非负(样例提交对真实答案打分的合理范围)
    assert math.isfinite(score) and score >= 0
