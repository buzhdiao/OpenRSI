# 中文说明:本测试文件针对 OpenMLE-Gym 的本地评估器(local_evaluator)做单元回归:
# 覆盖 CSV 逻辑行统计(含多行字段)、任务结构校验(行列守恒/对齐)、确定性结构失败、
# 提交-答案 schema 与标识符对齐、在线硬门槛短路(不调用 LLM 评审)、local_only 不做
# 质量评分、LLM 调用失败不算"不推荐"、AI 质量结果被服务端校验并重算、
# major_issues 必须携带结构化证据,以及评审 prompt 中锁定的关键契约文案。
# 位置:OpenMLE-Gym/tests 下,是评估器确定性规则与 prompt 契约的守门测试。

# 延迟注解:类型标注运行期不求值
from __future__ import annotations

# json:构造/解析评估结果 JSON
import json
# tempfile:为每个用例创建隔离临时目录
import tempfile
# unittest:测试类组织方式
import unittest
# Path:构造任务目录与文件路径
from pathlib import Path
# patch:替换评估器内部的 AI 评审函数与环境变量
from unittest.mock import patch

# 被测对象:单任务评估入口、AI 评审入口、CSV 取证、AI 质量校验、
# 任务结构分析、任务失败判定
from openmle_gym.local_evaluator import (
    _evaluate_task,
    _evaluate_with_ai,
    _inspect_csv,
    _validate_ai_quality,
    analyze_task_structure,
    evaluation_has_task_failures,
)


# 测试夹具的 metric.py 内容:类名以 Metrics 结尾,evaluate 恒返回 0.5
METRIC_CODE = """
class FixtureMetrics:
    def evaluate(self, y_true=None, y_pred=None):
        return 0.5
"""


def _make_task(root: Path, name: str = "fixture") -> Path:
    """构造一个结构完整的最小任务目录(含多行 CSV 字段的边缘样例)。"""
    # 任务根目录
    task = root / name
    # 公开数据目录
    public = task / "data" / "public"
    # 私有答案目录
    private = task / "data" / "private"
    # 工具脚本目录
    utils = task / "utils"
    # 原始数据目录
    raw = task / "raw"
    # 依次创建四个目录
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    utils.mkdir(parents=True)
    raw.mkdir(parents=True)
    # 写任务描述
    (public / "description.txt").write_text("A fixture task.", encoding="utf-8")
    # 训练集:2 行逻辑记录,其中一条 text 含引号+换行的多行字段(考验 CSV 解析)
    (public / "train.csv").write_text(
        'id,text,target\n1,"first line\\nsecond line",0\n2,plain,1\n',
        encoding="utf-8",
    )
    # 测试集:2 行逻辑记录,同样含多行字段
    (public / "test.csv").write_text(
        'id,text\n3,"test line\\ncontinued"\n4,plain\n',
        encoding="utf-8",
    )
    # 提交样例:与测试集行对齐(id 3/4)
    (public / "sample_submission.csv").write_text(
        "id,target\n3,0\n4,1\n",
        encoding="utf-8",
    )
    # 私有答案:2 行,标签与提交样例不同(避免复制关系)
    (private / "test_answer.csv").write_text(
        "id,target\n3,1\n4,0\n",
        encoding="utf-8",
    )
    # 原始训练数据:4 行带标签(用于守恒/利用率取证)
    (raw / "train.csv").write_text(
        "id,text,target\n1,a,0\n2,b,1\n3,c,1\n4,d,0\n",
        encoding="utf-8",
    )
    # 原始测试数据:2 行无标签(约定上有意排除在处理集之外)
    (raw / "test.csv").write_text(
        "id,text\n5,e\n6,f\n",
        encoding="utf-8",
    )
    # prepare.py:空实现占位
    (utils / "prepare.py").write_text(
        "def prepare(raw, public, private):\n    pass\n",
        encoding="utf-8",
    )
    # metric.py:写入夹具指标类
    (utils / "metric.py").write_text(METRIC_CODE, encoding="utf-8")
    # 返回任务路径
    return task


def _judge_result() -> dict:
    """构造一份合格的 AI 评审结果(五个维度打分+理由,无 major issue)。"""
    # 返回固定结构的评审字典
    return {
        # 任务有效性 5 分及理由
        "task_validity": 5,
        "task_validity_reason": "The task is meaningful.",
        # 数据充分性 4 分及理由
        "data_sufficiency": 4,
        "data_sufficiency_reason": "The fixture is sufficient for this test.",
        # 原始数据利用 4 分及理由
        "raw_data_usage": 4,
        "raw_data_usage_reason": "Preparation is defined.",
        # 任务复杂度 4 分及理由
        "task_complexity": 4,
        "task_complexity_reason": "The task is non-trivial.",
        # 数据质量 4 分及理由
        "data_quality": 4,
        "data_quality_reason": "The package is aligned.",
        # 无重大问题
        "major_issues": [],
    }


class LocalEvaluatorTests(unittest.TestCase):
    def test_csv_inspection_counts_logical_records_and_labels_preview(self) -> None:
        """CSV 取证按逻辑记录计数(多行字段算一行),预览行数受 max_preview_rows 限制。"""
        with tempfile.TemporaryDirectory() as directory:
            # 预览 CSV 路径
            path = Path(directory) / "multiline.csv"
            # 写入 2 条逻辑记录,第一条含换行的多行字段
            path.write_text(
                'id,text\n1,"line one\\nline two"\n2,plain\n',
                encoding="utf-8",
            )
            # 只预览 1 行
            evidence = _inspect_csv(path, max_preview_rows=1)
        # 逻辑行数应为 2(而非物理行数 3)
        self.assertEqual(evidence["row_count"], 2)
        # 预览行数为 1
        self.assertEqual(evidence["preview_row_count"], 1)
        # 标记预览被截断
        self.assertTrue(evidence["preview_is_truncated"])
        # 预览第二列(索引 1)应保留字面量换行转义,即多行字段未被截断破坏
        self.assertEqual(evidence["preview_rows"][0][1], "line one\\nline two")

    def test_structure_validation_uses_csv_records_and_alignment(self) -> None:
        """结构分析基于 CSV 逻辑记录计数,并通过行列守恒与对齐校验。"""
        with tempfile.TemporaryDirectory() as directory:
            # 构造夹具任务
            task = _make_task(Path(directory))
            # 运行结构分析
            result = analyze_task_structure(task)
        # 训练集逻辑行数为 2
        self.assertEqual(result["train_rows"], 2)
        # 测试集逻辑行数为 2
        self.assertEqual(result["test_rows"], 2)
        # 校验状态应为通过
        self.assertEqual(result["validation"]["status"], "passed")
        # 取答案 CSV 的预览证据
        preview = result["csv_previews"]["test_answer.csv"]
        # 总行数 2
        self.assertEqual(preview["total_rows"], 2)
        # 预览行数 2(未截断)
        self.assertEqual(preview["preview_row_count"], 2)
        # 取原始数据使用证据
        usage = result["raw_usage_evidence"]
        # 行数守恒应成立
        self.assertTrue(usage["row_conservation"])
        # 带标签行利用率为 100%
        self.assertEqual(usage["labeled_row_utilization_ratio"], 1.0)
        # 原始测试集角色应被记为"有意排除的无标签原始测试集"
        self.assertEqual(
            usage["raw_test_role"],
            "unlabeled_original_test_intentionally_excluded",
        )

    def test_structural_mismatch_is_a_deterministic_failure(self) -> None:
        """提交样例行数与测试集不一致时,结构校验确定性地失败并给出 finding ID。"""
        with tempfile.TemporaryDirectory() as directory:
            # 构造夹具任务
            task = _make_task(Path(directory))
            # 把提交样例改写成只有 1 行(与 2 行测试集不匹配)
            (task / "data" / "public" / "sample_submission.csv").write_text(
                "id,target\n3,0\n",
                encoding="utf-8",
            )
            # 重新做结构分析
            result = analyze_task_structure(task)
        # 校验状态应为失败
        self.assertEqual(result["validation"]["status"], "failed")
        # 收集所有 finding 的 ID
        finding_ids = {
            finding["id"] for finding in result["validation"]["findings"]
        }
        # 必须包含"提交样例行数不匹配"这一确定性 finding
        self.assertIn("row_count_mismatch_sample_submission_csv", finding_ids)

    def test_metric_owns_submission_answer_schema_and_identifier_alignment(self) -> None:
        """指标类自带校验时,提交/答案可为不同 schema,只要指标自身声明能处理对齐。"""
        with tempfile.TemporaryDirectory() as directory:
            # 构造夹具任务
            task = _make_task(Path(directory))
            # 提交样例换成多概率列 schema(id + 两个概率列)
            (task / "data" / "public" / "sample_submission.csv").write_text(
                "id,prob_12h,prob_24h\n3,0.1,0.2\n4,0.3,0.4\n",
                encoding="utf-8",
            )
            # 答案换成完全不同的 schema 与 ID(event_id 等)
            (task / "data" / "private" / "test_answer.csv").write_text(
                "event_id,time_to_hit_hours,event\n30,12,1\n40,72,0\n",
                encoding="utf-8",
            )
            # 结构分析(会顺带跑 metric_check 与 schema 对齐检查)
            result = analyze_task_structure(task)
        # 指标自检状态应为 ok(校验提交后仍能给出分数)
        self.assertEqual(result["metric_check"]["status"], "ok")
        # 结构校验整体通过
        self.assertEqual(result["validation"]["status"], "passed")
        # 收集 finding ID
        finding_ids = {
            finding["id"] for finding in result["validation"]["findings"]
        }
        # 不应报"提交-答案 schema 不匹配"
        self.assertNotIn("submission_answer_schema_mismatch", finding_ids)
        # 也不应报"提交-答案标识符不匹配"
        self.assertNotIn("submission_answer_identifier_mismatch", finding_ids)

    def test_online_hard_gate_does_not_call_quality_judge(self) -> None:
        """在线模式下结构硬门槛失败即短路:总分 0、不推荐,且不再调用 LLM 评审。"""
        with tempfile.TemporaryDirectory() as directory:
            # 构造夹具任务
            task = _make_task(Path(directory))
            # 把提交样例改成 1 行,制造结构不匹配(触发硬门槛)
            (task / "data" / "public" / "sample_submission.csv").write_text(
                "id,target\n3,0\n",
                encoding="utf-8",
            )
            # 监控 AI 评审函数是否被调用(应为 0 次)
            with patch(
                "openmle_gym.local_evaluator._evaluate_with_ai"
            ) as evaluator:
                # 在线模式(非 local_only、不跳过 LLM)跑单任务评估
                result = _evaluate_task(
                    root_dir=str(task.parent),
                    task=task.name,
                    overview={},
                    local_only=False,
                    skip_llm=False,
                )
        # 确定性硬门槛失败后,LLM 评审不应被调用
        evaluator.assert_not_called()
        # 校验状态为失败
        self.assertEqual(result["validation"]["status"], "failed")
        # 质量总分记 0
        self.assertEqual(result["quality_scores"]["overall_score"], 0)
        # 推荐结论为不推荐
        self.assertEqual(
            result["quality_scores"]["recommendation"],
            "not_recommended",
        )
        # 质量评估来源应标注为确定性硬门槛(而非 AI)
        self.assertEqual(
            result["quality_evaluation"]["source"],
            "deterministic_hard_gate",
        )

    def test_local_only_performs_no_quality_scoring(self) -> None:
        """local_only 模式只做结构校验,完全不触发质量评分。"""
        with tempfile.TemporaryDirectory() as directory:
            # 构造夹具任务(结构合法)
            task = _make_task(Path(directory))
            # local_only 模式评估
            result = _evaluate_task(
                root_dir=str(task.parent),
                task=task.name,
                overview={},
                local_only=True,
                skip_llm=False,
            )
        # 结构校验通过
        self.assertEqual(result["validation"]["status"], "passed")
        # 质量分数为空字典(没有评分)
        self.assertEqual(result["quality_scores"], {})
        # 质量评估状态为 skipped
        self.assertEqual(result["quality_evaluation"]["status"], "skipped")

    def test_llm_failure_is_not_reported_as_not_recommended(self) -> None:
        """LLM 调用失败(超时)不应被误报为"不推荐":质量状态记 failed 且无推荐结论。"""
        with tempfile.TemporaryDirectory() as directory:
            # 构造夹具任务
            task = _make_task(Path(directory))
            # 让 AI 评审抛出超时异常,模拟服务方故障
            with patch(
                "openmle_gym.local_evaluator._evaluate_with_ai",
                side_effect=TimeoutError("provider timeout"),
            ):
                # 在线模式评估
                result = _evaluate_task(
                    root_dir=str(task.parent),
                    task=task.name,
                    overview={},
                    local_only=False,
                    skip_llm=False,
                )
        # 结构校验不受 LLM 故障影响,仍通过
        self.assertEqual(result["validation"]["status"], "passed")
        # 质量分数为空(没有可信评分)
        self.assertEqual(result["quality_scores"], {})
        # 质量评估状态为 failed(区别于"被评不推荐")
        self.assertEqual(result["quality_evaluation"]["status"], "failed")
        # 分数里不应出现推荐字段(避免把故障当成负面结论)
        self.assertNotIn("recommendation", result["quality_scores"])

        # 第二段:验证带 failed 质量评估的结果会被判为"任务失败"
        with tempfile.TemporaryDirectory() as directory:
            # 聚合结果文件路径
            combined = Path(directory) / "all_results.json"
            # 把上面得到的结果写入 JSON
            combined.write_text(json.dumps([result]), encoding="utf-8")
            # 失败判定应返回 True(质量评估 failed 视作任务失败)
            self.assertTrue(
                evaluation_has_task_failures({"combined": combined})
            )

    def test_quality_result_is_validated_and_recomputed(self) -> None:
        """AI 返回的总分/推荐若与维度分不一致,服务端必须重算覆盖。"""
        # 取一份合格的评审结果
        raw = _judge_result()
        # 恶意/错误地把总分写成 0
        raw["overall_score"] = 0
        # 并把推荐写成不推荐
        raw["recommendation"] = "not_recommended"
        # 走校验函数:应基于维度分重算
        result = _validate_ai_quality(raw)
        # 重算后的总分应为 4.2((5+4+4+4+4)/5 = 4.2)
        self.assertEqual(result["overall_score"], 4.2)
        # 推荐结论应被纠正为 recommended
        self.assertEqual(result["recommendation"], "recommended")

    def test_major_issue_requires_structured_evidence(self) -> None:
        """major_issues 里的每个条目必须是结构化证据对象,纯字符串声明应被拒绝。"""
        # 取合格评审结果
        raw = _judge_result()
        # 把 major_issues 塞进一个裸字符串(不合规)
        raw["major_issues"] = ["unsupported claim"]
        # 校验应抛 ValueError,错误信息提到 evidence object
        with self.assertRaisesRegex(ValueError, "evidence object"):
            _validate_ai_quality(raw)

    def test_prompt_documents_random_submission_and_holdout_contract(self) -> None:
        """评审 prompt 必须锁定关键契约文案:随机合法提交、holdout 期望、行数取证权威性等。"""
        # 构造传给评审器的任务信息
        task_info = {
            # 任务名
            "task_name": "fixture",
            # 缺失的必需文件为空
            "missing_required": [],
            # 指标自检通过且分数 0.1
            "metric_check": {"status": "ok", "score": 0.1},
            # CSV 证据为空占位
            "csv_evidence": {},
            # 训练行数
            "train_rows": 8,
            # 测试行数
            "test_rows": 2,
            # 文件类型清单
            "file_types": [],
            # 原始数据大小(MB)
            "raw_size_mb": 1,
            # 处理后数据大小(MB)
            "data_size_mb": 1,
        }
        # 同时伪造环境变量(API key/模型名)与 _evaluate_with_openai 的返回
        with (
            patch.dict(
                "os.environ",
                {
                    "OPENMLE_EVAL_LLM_API_KEY": "key",
                    "OPENMLE_EVAL_LLM_MODEL": "model",
                },
                clear=True,
            ),
            patch(
                "openmle_gym.local_evaluator._evaluate_with_openai",
                return_value=_judge_result(),
            ) as evaluator,
        ):
            # 触发 AI 评审(内部会组装 prompt 并调用被 mock 的 openai 函数)
            _evaluate_with_ai(task_info)
        # 取出本次调用实际使用的 prompt 文本
        prompt = evaluator.call_args.args[0]
        # 契约 1:提交样例只用于格式参考,模型可随机生成 schema 合法的预测
        self.assertIn("random schema-valid predictions", prompt)
        # 契约 2:从带标签原始训练集留出 holdout 做评估是预期行为
        self.assertIn("holdout from the labeled raw training set is expected", prompt)
        # 契约 3:raw_usage_evidence 是行使用情况的权威口径
        self.assertIn(
            "raw_usage_evidence is the authoritative row-usage accounting",
            prompt,
        )
        # 契约 4:不得把该口径称为不一致/误导/不匹配
        self.assertIn(
            "Never call it inconsistent, misleading, or mismatched",
            prompt,
        )
        # 契约 5:不得引入对原始竞赛的外部知识
        self.assertIn(
            "Do not use outside knowledge about the original competition",
            prompt,
        )
        # 契约 6:行数只与处理后 CSV 证据比较,绝不与原始 CSV 证据比较
        self.assertIn(
            "Compare its row counts only with processed_csv_evidence, "
            "never with raw_csv_evidence",
            prompt,
        )
        # 契约 7:validate_submission 与 evaluate 视为同一条打分管线
        self.assertIn(
            "Treat metric.validate_submission() and metric.evaluate() as one "
            "scoring pipeline",
            prompt,
        )
        # 契约 8:多余/缺失/重复的预测行均为非法
        self.assertIn(
            "Extra, missing, or duplicate prediction rows are invalid",
            prompt,
        )
        # 契约 9:只有证据里出现真实标识符问题时才可上报
        self.assertIn(
            "Report an identifier issue only when the evidence contains an actual",
            prompt,
        )
        # 缺 metric.py 时 prompt 应内嵌固定占位文案
        self.assertIn('"metric_code": "No metric.py file"', prompt)


# 直接运行本文件时走 unittest 主入口
if __name__ == "__main__":
    unittest.main()
