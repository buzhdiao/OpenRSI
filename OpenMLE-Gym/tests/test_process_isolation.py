# 中文说明:OpenMLE-Gym 任务子进程"隔离性"回归测试(process_runner):
# 验证崩溃/超时/同名动态模块的故障都被限制在单个任务的子进程内,
# 不影响同批其他任务;隔离容器模式不回退、容器命令无网络/无源码挂载/无密钥注入;
# 以及 prepare 操作有独立的结果信封契约。
# 位置:OpenMLE-Gym/tests 下,是"批次独立性 + 安全隔离"的核心守门测试。

# 延迟注解:类型标注运行期不求值
from __future__ import annotations

# importlib.util:探测可选依赖 pandas 是否存在
import importlib.util
# os:patch 环境变量(容器命令不含密钥的断言)
import os
# tempfile:每个用例独立的临时目录
import tempfile
# unittest:测试组织与条件跳过
import unittest
# 线程池:并发执行多个任务子进程
from concurrent.futures import ThreadPoolExecutor
# Path:路径构造
from pathlib import Path
# patch:替换 shutil.which 与环境变量
from unittest.mock import patch

# 被测对象:容器命令拼装/结果信封解析/任务子进程运行器
from openmle_gym.process_runner import (
    _container_command,
    _load_result_envelope,
    run_task_process,
)


# pandas 可用性(度量 worker 测试的前置条件)
HAS_PANDAS = importlib.util.find_spec("pandas") is not None


def _make_task(root: Path, name: str, metric_code: str) -> Path:
    """快速造一个最小任务目录(public 样例提交 + private 答案 + 指定 metric.py)。"""
    # 任务根与三类子目录
    task = root / name
    public = task / "data" / "public"
    private = task / "data" / "private"
    utils = task / "utils"
    # 依次创建目录
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    utils.mkdir(parents=True)
    # 写样例提交与答案(单行,匹配)
    (public / "sample_submission.csv").write_text("target\n1\n", encoding="utf-8")
    (private / "test_answer.csv").write_text("target\n1\n", encoding="utf-8")
    # 写指定的 metric.py(各用例可传不同内容)
    (utils / "metric.py").write_text(metric_code, encoding="utf-8")
    return task


# 正常指标代码:提交校验恒通过,评估恒 1.0
GOOD_METRIC = """
class GoodMetrics:
    def validate_submission(self, submission, ground_truth):
        return None

    def evaluate(self, y_true=None, y_pred=None):
        return 1.0
"""


@unittest.skipUnless(HAS_PANDAS, "pandas is required for metric worker tests")
class ProcessIsolationTests(unittest.TestCase):
    """进程隔离行为测试集(无 pandas 时整类跳过)。"""

    def test_invalid_result_json_is_rejected(self) -> None:
        """结果信封是坏 JSON 时必须报错,不能静默接受。"""
        with tempfile.TemporaryDirectory() as directory:
            # 写一个非法 JSON 结果文件
            result = Path(directory) / "result.json"
            result.write_text("{invalid", encoding="utf-8")
            # 解析应抛异常
            with self.assertRaises(Exception):
                _load_result_envelope(result)

    def test_crashed_task_does_not_break_sibling_tasks(self) -> None:
        """崩溃任务(os._exit(9))只影响自身;同批前后两个任务都正常完成。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # 造三个任务:好-坏-好(坏的直接退出进程)
            good_a = _make_task(root, "good-a", GOOD_METRIC)
            crashed = _make_task(
                root,
                "crashed",
                "import os\nos._exit(9)\n",
            )
            good_b = _make_task(root, "good-b", GOOD_METRIC)

            # 单任务执行体:跑 metric worker(任务目录只读)
            def run(task: Path):
                return run_task_process(
                    "metric",
                    {"task_dir": str(task)},
                    timeout=10,
                    readonly_paths=(task,),
                )

            # 三任务并发执行
            with ThreadPoolExecutor(max_workers=3) as executor:
                outcomes = list(executor.map(run, (good_a, crashed, good_b)))

            # 前后两个成功,中间崩溃(退出码 9 被如实上报)
            self.assertTrue(outcomes[0].ok)
            self.assertFalse(outcomes[1].ok)
            self.assertEqual(outcomes[1].returncode, 9)
            self.assertTrue(outcomes[2].ok)
            self.assertEqual(outcomes[0].result["score"], 1.0)
            self.assertEqual(outcomes[2].result["score"], 1.0)

    def test_same_dynamic_module_name_is_isolated_between_tasks(self) -> None:
        """两个任务用同名动态模块但不同实现:彼此隔离,分数互不串扰。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # 两个任务目录不同、内部模块名相同、指标实现不同(1.0 vs 2.0)
            first = _make_task(root / "first", "same-name", GOOD_METRIC)
            second = _make_task(
                root / "second",
                "same-name",
                "class OtherMetrics:\n"
                "    def evaluate(self, y_true=None, y_pred=None):\n"
                "        return 2.0\n",
            )

            # 单任务执行体
            def run(task: Path):
                return run_task_process(
                    "metric",
                    {"task_dir": str(task)},
                    timeout=10,
                    readonly_paths=(task,),
                )

            # 并发执行
            with ThreadPoolExecutor(max_workers=2) as executor:
                outcomes = list(executor.map(run, (first, second)))
            # 各自拿到自己的分数(模块注册表未互相污染)
            self.assertEqual(outcomes[0].result["score"], 1.0)
            self.assertEqual(outcomes[1].result["score"], 2.0)

    def test_timeout_is_local_to_one_task(self) -> None:
        """超时只作用于单个任务:挂死任务超时报错,好任务继续成功。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # 一个死循环任务 + 一个正常任务
            hanging = _make_task(root, "hanging", "while True:\n    pass\n")
            good = _make_task(root, "good", GOOD_METRIC)
            # 挂死任务给 0.2 秒超时
            timed_out = run_task_process(
                "metric",
                {"task_dir": str(hanging)},
                timeout=0.2,
                readonly_paths=(hanging,),
            )
            # 正常任务给足超时
            succeeded = run_task_process(
                "metric",
                {"task_dir": str(good)},
                timeout=10,
                readonly_paths=(good,),
            )
            # 超时任务失败且错误信息标明 timed out;好任务不受影响
            self.assertFalse(timed_out.ok)
            self.assertIn("timed out", timed_out.error or "")
            self.assertTrue(succeeded.ok)

    def test_isolated_mode_never_falls_back(self) -> None:
        """isolated 模式绝不回退到普通进程:要么容器成功,要么报容器相关错误。"""
        with tempfile.TemporaryDirectory() as directory:
            task = _make_task(Path(directory), "task", GOOD_METRIC)
            # 以 isolated 模式执行(环境里可能没有 docker)
            outcome = run_task_process(
                "metric",
                {"task_dir": str(task)},
                timeout=10,
                execution_mode="isolated",
                readonly_paths=(task,),
            )
            if outcome.ok:
                # 容器可用时结果应正确
                self.assertEqual(outcome.result["score"], 1.0)
            else:
                # 失败必须是容器/镜像/退出相关错误,而非静默回退
                self.assertRegex(
                    outcome.error or "",
                    r"(docker|podman|OPENMLE_GYM_ISOLATED_IMAGE|exited)",
                )

    def test_isolated_command_has_no_network_or_source_mount(self) -> None:
        """容器命令安全契约:无网络/只读根/丢能力/仅挂载声明路径,且不含任何密钥。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # 准备请求/结果文件与输入输出目录
            request = root / "request.json"
            result = root / "result.json"
            task_input = root / "input"
            task_output = root / "output"
            task_input.mkdir()
            task_output.mkdir()
            request.write_text("{}", encoding="utf-8")
            with (
                # 假装系统有 docker
                patch("openmle_gym.process_runner.shutil.which", return_value="/usr/bin/docker"),
                # 注入镜像名与若干"敏感"环境变量(应被过滤)
                patch.dict(
                    os.environ,
                    {
                        "OPENMLE_GYM_ISOLATED_IMAGE": "openmle:test",
                        "OPENMLE_BUILD_LLM_API_KEY": "build-secret",
                        "OPENMLE_EVAL_LLM_API_KEY": "eval-secret",
                        "KAGGLE_KEY": "kaggle-secret",
                    },
                    clear=False,
                ),
            ):
                # 拼装容器命令
                command = _container_command(
                    request,
                    result,
                    "metric",
                    (task_input,),
                    (task_output,),
                )
            # 渲染成整串便于子串断言
            rendered = " ".join(command)
            # 禁网络
            self.assertIn("--network none", rendered)
            # 根文件系统只读
            self.assertIn("--read-only", command)
            # 丢弃全部 Linux 能力
            self.assertIn("--cap-drop ALL", rendered)
            # 输入只读挂载、输出可写挂载
            self.assertIn(f"{task_input.resolve()}:{task_input.resolve()}:ro", command)
            self.assertIn(f"{task_output.resolve()}:{task_output.resolve()}:rw", command)
            # 源码根不得整体挂入容器
            source_root = Path(__file__).resolve().parents[1]
            self.assertNotIn(f"{source_root}:{source_root}:ro", command)
            # 任何密钥环境变量都不得出现在命令行里
            self.assertNotIn("OPENMLE_BUILD_LLM_API_KEY", command)
            self.assertNotIn("OPENMLE_EVAL_LLM_API_KEY", command)
            self.assertNotIn("KAGGLE_KEY", command)

    def test_prepare_operation_uses_its_own_result_envelope(self) -> None:
        """prepare 操作走独立信封:返回 {'executed': True} 且产物真实落盘。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # 建 raw/public/private 三目录与 prepare.py
            raw = root / "raw"
            public = root / "public"
            private = root / "private"
            raw.mkdir()
            public.mkdir()
            private.mkdir()
            prepare = root / "prepare.py"
            prepare.write_text(
                "def prepare(raw_dir, public_dir, private_dir):\n"
                "    (public_dir / 'done.txt').write_text('ok')\n",
                encoding="utf-8",
            )
            # 以 prepare 操作执行子进程
            outcome = run_task_process(
                "prepare",
                {
                    "prepare_path": str(prepare),
                    "raw_dir": str(raw),
                    "public_dir": str(public),
                    "private_dir": str(private),
                },
                timeout=10,
            )
            # 成功且结果信封符合 prepare 契约
            self.assertTrue(outcome.ok)
            self.assertEqual(outcome.result, {"executed": True})
            # prepare 副作用真实发生
            self.assertEqual((public / "done.txt").read_text(), "ok")


if __name__ == "__main__":
    unittest.main()
