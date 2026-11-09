# 中文说明:本测试文件验证 OpenMLE-Gym 批量构建/评估管线的"批次独立性"(batch independence):
# 单个任务崩溃、写盘失败或重名,都不能取消/污染同批其他任务的处理结果。
# 覆盖场景:build 阶段失败任务被收集且不取消兄弟任务、isolated 模式下外层仍是
# process 执行、overview/evaluate 阶段子进程崩溃保留兄弟行、单任务结果写盘失败
# 不影响聚合、重复任务名按单任务失败处理。
# 位置:OpenMLE-Gym/tests 下,是构建与评估稳健性的核心回归测试。

# 延迟注解:类型标注运行期不求值
from __future__ import annotations

# asyncio:以 asyncio.run 驱动 _run_all 异步构建入口
import asyncio
# csv:写/读 overview 与 summary 的 CSV 文件
import csv
# 动态探测依赖包是否存在(pandas/tqdm 可选依赖)
import importlib.util
# JSON:读回 combined 评估结果文件
import json
# tempfile:为每个用例创建隔离的临时目录
import tempfile
# unittest:测试类与跳过装饰器
import unittest
# Path:路径构造
from pathlib import Path
# patch:替换 openmle_gym 内部的进程运行器与原子写函数
from unittest.mock import patch

# 被测函数:_run_all 是批量构建的总入口
from openmle_gym.build import _run_all
# 真实的原子 JSON 写函数:作为"只让特定写失败"的兜底实现
from openmle_gym.common import atomic_write_json as real_atomic_write_json
# 任务子进程结果类型:模拟进程运行器的返回值
from openmle_gym.process_runner import TaskProcessOutcome


# 批量功能依赖 pandas 与 tqdm;二者都存在才运行这些测试,否则整类跳过
HAS_BATCH_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("pandas", "tqdm")
)


def _make_task(root: Path, name: str, metric_code: str) -> Path:
    """在 root 下快速造出一个最小可识别的任务目录(含 public/private/utils/raw)。"""
    # 任务根目录
    task = root / name
    # 公开数据目录(训练侧可见)
    public = task / "data" / "public"
    # 私有数据目录(答案所在)
    private = task / "data" / "private"
    # 工具脚本目录(prepare.py/metric.py)
    utils = task / "utils"
    # 原始数据目录
    raw = task / "raw"
    # 依次创建四个目录(public/private/utils 需多级创建)
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    utils.mkdir(parents=True)
    raw.mkdir()
    # 写任务描述文件(overview 会读取展示)
    (public / "description.txt").write_text("description", encoding="utf-8")
    # 最小训练集(feature+target 两行)
    (public / "train.csv").write_text("feature,target\n1,1\n", encoding="utf-8")
    # 最小测试集(仅 feature)
    (public / "test.csv").write_text("feature\n1\n", encoding="utf-8")
    # 提交样例
    (public / "sample_submission.csv").write_text("target\n1\n", encoding="utf-8")
    # 私有答案文件
    (private / "test_answer.csv").write_text("target\n1\n", encoding="utf-8")
    # prepare.py:空实现即可(构建只要求文件存在且可执行)
    (utils / "prepare.py").write_text("def prepare(*args):\n    pass\n", encoding="utf-8")
    # metric.py:内容由调用方注入(可传"好指标"或"故意崩溃"代码)
    (utils / "metric.py").write_text(metric_code, encoding="utf-8")
    # 返回任务路径
    return task


# 一个健康的 metric.py:类名以 Metrics 结尾,evaluate 恒返回 1.0
GOOD_METRIC = """
class GoodMetrics:
    def evaluate(self, y_true=None, y_pred=None):
        return 1.0
"""


# 依赖不齐时跳过整组测试
@unittest.skipUnless(HAS_BATCH_DEPS, "batch dependencies are required")
class BatchIndependenceTests(unittest.TestCase):
    def test_build_collects_failure_without_cancelling_siblings(self) -> None:
        """构建批次中某任务进程失败时,结果仍按序收集,且兄弟任务正常完成。"""
        def fake_runner(operation, payload, **kwargs):
            # 按任务 slug 区分行为
            slug = payload["slug"]
            # "bad" 任务模拟进程崩溃(ok=False,返回码 9)
            if slug == "bad":
                return TaskProcessOutcome(ok=False, error="crashed", returncode=9)
            # 其他任务模拟成功并回传标准 result 结构
            return TaskProcessOutcome(
                ok=True,
                result={"name": slug, "success": True, "errors": []},
                returncode=0,
            )

        # 替换真实的进程运行器,避免执行真实构建子进程
        with patch("openmle_gym.build.run_task_process", side_effect=fake_runner):
            # 以 3 并发跑三个任务,中间夹一个会失败的 "bad"
            results = asyncio.run(
                _run_all(
                    builder_root=Path("builder_core"),
                    batch_name="batch",
                    slugs=["good-a", "bad", "good-b"],
                    retry=0,
                    output_root=Path("output"),
                    info_csv=Path("info.csv"),
                    max_concurrency=3,
                    delete_raw=False,
                    execution_mode="process",
                    task_timeout=10,
                )
            )
        # 结果列表必须保持输入顺序且全部收集(失败任务不丢失也不取消兄弟)
        self.assertEqual([result["name"] for result in results], ["good-a", "bad", "good-b"])
        # 第一个任务成功
        self.assertTrue(results[0]["success"])
        # 中间的失败任务被如实记录为失败
        self.assertFalse(results[1]["success"])
        # 第三个任务不受影响,仍然成功
        self.assertTrue(results[2]["success"])

    def test_build_isolated_mode_keeps_outer_task_process(self) -> None:
        """isolated 执行模式只影响任务内代码执行方式,外层仍是 process 进程执行。"""
        # 记录每次进程运行器被调用时的入参
        calls = []

        def fake_runner(operation, payload, **kwargs):
            # 记录 (operation, payload, kwargs) 以便断言两种模式字段的传递
            calls.append((operation, payload, kwargs))
            # 一律返回成功结果
            return TaskProcessOutcome(
                ok=True,
                result={"name": payload["slug"], "success": True, "errors": []},
            )

        # 替换进程运行器
        with patch("openmle_gym.build.run_task_process", side_effect=fake_runner):
            # 以 isolated 模式跑单个任务
            asyncio.run(
                _run_all(
                    builder_root=Path("builder_core"),
                    batch_name="batch",
                    slugs=["task"],
                    retry=0,
                    output_root=Path("output"),
                    info_csv=Path("info.csv"),
                    max_concurrency=1,
                    delete_raw=False,
                    execution_mode="isolated",
                    task_timeout=10,
                )
            )
        # 传给进程运行器的 execution_mode 应仍是 "process"(外层进程执行不被降级)
        self.assertEqual(calls[0][2]["execution_mode"], "process")
        # isolated 模式作为 code_execution_mode 透传给任务内部执行
        self.assertEqual(calls[0][1]["code_execution_mode"], "isolated")

    def test_overview_crash_keeps_sibling_rows(self) -> None:
        """overview 汇总时某任务导入/运行即崩溃,仍要输出其错误行,兄弟任务行不受影响。"""
        # 延迟导入被测函数,保持文件顶部依赖最小
        from openmle_gym.overview import generate_overview

        # 每个用例独立的临时目录
        with tempfile.TemporaryDirectory() as directory:
            # 临时根目录
            root = Path(directory)
            # tasks 目录存放两个任务
            tasks = root / "tasks"
            tasks.mkdir()
            # 好任务:健康指标
            _make_task(tasks, "good", GOOD_METRIC)
            # 崩溃任务:metric.py 一执行就 os._exit(7) 直接退出进程
            _make_task(tasks, "crashed", "import os\nos._exit(7)\n")
            # overview 输出 CSV 路径
            output = root / "overview.csv"
            # 2 workers 并发生成 overview,跳过 LLM 步骤并限 10 秒超时
            generate_overview(tasks, output, workers=2, skip_llm=True, task_timeout=10)
            # 用 utf-8-sig 读回 CSV(兼容 BOM 头)
            with output.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            # 两行都应存在且按名称排序
            self.assertEqual([row["Name"] for row in rows], ["crashed", "good"])
            # 崩溃任务的 NOTE 必须以运行时错误标记开头
            self.assertTrue(rows[0]["NOTE"].startswith("#TASK_RUNTIME_ERROR:"))
            # 好任务的 NOTE 应为空(无错误)
            self.assertEqual(rows[1]["NOTE"], "")

    def test_overview_rejects_task_symlink_outside_batch_root(self) -> None:
        """overview 对逃逸出批次根目录的符号链接任务目录应拒绝并记为运行时错误。"""
        # 延迟导入被测函数
        from openmle_gym.overview import generate_overview

        # 独立临时目录
        with tempfile.TemporaryDirectory() as directory:
            # 临时根目录
            root = Path(directory)
            # 批次任务目录
            tasks = root / "tasks"
            tasks.mkdir()
            # 在批次根外造一个真实任务目录
            outside = root / "outside"
            # 先把真实任务建在根目录下(outside 的位置)
            _make_task(root, "outside", GOOD_METRIC)
            # 在 tasks 下创建指向批次根之外目录的符号链接(目录型)
            (tasks / "escaped").symlink_to(outside, target_is_directory=True)
            # overview 输出路径
            output = root / "overview.csv"
            # 单 worker 生成 overview
            generate_overview(tasks, output, workers=1, skip_llm=True)
            # 读回 CSV 首行
            with output.open(encoding="utf-8-sig", newline="") as file:
                row = next(csv.DictReader(file))
            # 该符号链接任务仍以名字 "escaped" 出现(不被静默吞掉)
            self.assertEqual(row["Name"], "escaped")
            # 其 NOTE 标记为运行时错误(拒绝逃逸目录的访问)
            self.assertTrue(row["NOTE"].startswith("#TASK_RUNTIME_ERROR:"))

    def test_evaluate_crash_keeps_sibling_results(self) -> None:
        """评估阶段某任务进程崩溃:崩溃任务标记失败,兄弟任务正常评估并汇总。"""
        # 延迟导入本地评估入口
        from openmle_gym.local_evaluator import evaluate_tasks

        # 独立临时目录
        with tempfile.TemporaryDirectory() as directory:
            # 临时根目录
            root = Path(directory)
            # 任务目录
            tasks = root / "tasks"
            tasks.mkdir()
            # 好任务 + 崩溃任务
            _make_task(tasks, "good", GOOD_METRIC)
            _make_task(tasks, "crashed", "import os\nos._exit(7)\n")
            # 任务清单文件:先 crashed 后 good,验证结果顺序与清单一致
            task_list = root / "tasks.txt"
            task_list.write_text("crashed\ngood\n", encoding="utf-8")
            # 预置 overview CSV:只列 Name 字段
            overview = root / "overview.csv"
            with overview.open("w", encoding="utf-8", newline="") as file:
                # DictWriter 按字段名写行
                writer = csv.DictWriter(file, fieldnames=["Name"])
                # 写表头
                writer.writeheader()
                # 两行任务名
                writer.writerow({"Name": "crashed"})
                writer.writerow({"Name": "good"})
            # 执行本地评估:local_only、2 workers、10 秒超时
            paths = evaluate_tasks(
                tasks,
                task_list,
                root / "evaluation",
                local_only=True,
                overview_csv=overview,
                workers=2,
                task_timeout=10,
            )
            # 读回聚合的 combined 结果 JSON
            results = json.loads(paths["combined"].read_text(encoding="utf-8"))
            # 两个结果都收集且顺序与清单一致
            self.assertEqual([result["task_name"] for result in results], ["crashed", "good"])
            # 崩溃任务带 task_process_failed 标记
            self.assertTrue(results[0]["task_process_failed"])
            # 正常任务不应带该标记(键不存在)
            self.assertNotIn("task_process_failed", results[1])
            # 正常任务因 local_only 未做质量评分,quality_scores 为空字典
            self.assertEqual(results[1]["quality_scores"], {})
            # 质量评估状态为 skipped(明确跳过而非失败)
            self.assertEqual(
                results[1]["quality_evaluation"]["status"],
                "skipped",
            )
            # 读回 summary CSV
            with paths["summary"].open(encoding="utf-8", newline="") as file:
                summary_rows = list(csv.DictReader(file))
            # 正常任务(第二行)因跳过评估,总分列为空
            self.assertEqual(summary_rows[1]["Overall Score"], "")
            # 推荐列为空
            self.assertEqual(summary_rows[1]["Recommendation"], "")
            # 验证状态为 passed(任务构建/验证本身没问题)
            self.assertEqual(summary_rows[1]["Validation Status"], "passed")
            # 评估状态为 skipped
            self.assertEqual(summary_rows[1]["Evaluation Status"], "skipped")

    def test_evaluate_item_write_failure_keeps_sibling_and_aggregates(self) -> None:
        """单任务结果 JSON 写盘失败:该任务记为失败且错误信息明确,兄弟任务照常写盘聚合。"""
        # 延迟导入本地评估入口
        from openmle_gym.local_evaluator import evaluate_tasks

        # 独立临时目录
        with tempfile.TemporaryDirectory() as directory:
            # 临时根目录
            root = Path(directory)
            # 任务目录
            tasks = root / "tasks"
            tasks.mkdir()
            # 一个"不可写"任务 + 一个好任务
            _make_task(tasks, "unwritable", GOOD_METRIC)
            _make_task(tasks, "good", GOOD_METRIC)
            # 任务清单:先 unwritable 后 good
            task_list = root / "tasks.txt"
            task_list.write_text("unwritable\ngood\n", encoding="utf-8")
            # 预置 overview CSV
            overview = root / "overview.csv"
            with overview.open("w", encoding="utf-8", newline="") as file:
                # 只含 Name 字段的写器
                writer = csv.DictWriter(file, fieldnames=["Name"])
                # 表头
                writer.writeheader()
                # 两行任务名
                writer.writerow({"Name": "unwritable"})
                writer.writerow({"Name": "good"})

            def flaky_write(path, value):
                # 中文说明:仅当写 unwritable.json 时抛 OSError,模拟单文件写失败
                if Path(path).name == "unwritable.json":
                    raise OSError("simulated item write failure")
                # 其他文件走真实原子写
                return real_atomic_write_json(path, value)

            # 替换 local_evaluator 内引用的原子写函数为 flaky 版本
            with patch(
                "openmle_gym.local_evaluator.atomic_write_json",
                side_effect=flaky_write,
            ):
                # 执行评估
                paths = evaluate_tasks(
                    tasks,
                    task_list,
                    root / "evaluation",
                    local_only=True,
                    overview_csv=overview,
                    workers=2,
                    task_timeout=10,
                )
            # 读回聚合结果
            results = json.loads(paths["combined"].read_text(encoding="utf-8"))
            # 写失败的任务被标记为进程失败(降级处理)
            self.assertTrue(results[0]["task_process_failed"])
            # 其错误信息应包含"写任务结果失败"的提示
            self.assertIn("Failed to write task result", results[0]["errors"][0])
            # 兄弟任务不带失败标记
            self.assertNotIn("task_process_failed", results[1])
            # 兄弟任务的结果文件确实成功落盘
            self.assertTrue((root / "evaluation" / "good.json").is_file())

    def test_evaluate_duplicate_task_is_an_item_failure(self) -> None:
        """清单里重复出现同一任务:重复条目按"该任务失败"处理,不整体崩溃。"""
        # 延迟导入本地评估入口
        from openmle_gym.local_evaluator import evaluate_tasks

        # 独立临时目录
        with tempfile.TemporaryDirectory() as directory:
            # 临时根目录
            root = Path(directory)
            # 任务目录
            tasks = root / "tasks"
            tasks.mkdir()
            # 只有一个真实任务 "same"
            _make_task(tasks, "same", GOOD_METRIC)
            # 清单里把 "same" 写了两次
            task_list = root / "tasks.txt"
            task_list.write_text("same\nsame\n", encoding="utf-8")
            # 预置 overview CSV(只一行)
            overview = root / "overview.csv"
            with overview.open("w", encoding="utf-8", newline="") as file:
                # 只含 Name 字段
                writer = csv.DictWriter(file, fieldnames=["Name"])
                # 表头
                writer.writeheader()
                # 一行任务名
                writer.writerow({"Name": "same"})
            # 执行评估
            paths = evaluate_tasks(
                tasks,
                task_list,
                root / "evaluation",
                local_only=True,
                overview_csv=overview,
                workers=2,
            )
            # 读回聚合结果
            results = json.loads(paths["combined"].read_text(encoding="utf-8"))
            # 首次出现的条目正常,不带失败标记
            self.assertNotIn("task_process_failed", results[0])
            # 重复条目被标记为失败
            self.assertTrue(results[1]["task_process_failed"])
            # 错误信息应明确指出是重复任务名
            self.assertIn("Duplicate task name", results[1]["errors"][0])


# 直接运行本文件时走 unittest 主入口
if __name__ == "__main__":
    unittest.main()
