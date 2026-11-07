# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 os 模块，用于展开指令文本中的环境变量
import os
# 导入 Path，提供面向对象的文件路径操作
from pathlib import Path
# 导入类型注解工具，用于函数签名标注
from typing import Any, Dict, Optional, Tuple

# 从 mlebench 基准库导入提交文件校验函数（检查 submission.csv 是否符合比赛格式）
from mlebench.grade import validate_submission
# 导入 mlebench 的比赛注册表，用于按名称获取比赛对象
from mlebench.registry import registry

# 导入 MLE-Bench 评估模块，提供指标方向判断与提交评分能力
import dojo.tasks.mlebench.evaluate as evaluate
# 导入解释器基类：ExecutionResult 表示一次代码执行的结果，Interpreter 为执行器接口
from dojo.core.interpreters.base import ExecutionResult, Interpreter
# 导入任务基类 Task，本文件的 MLEBenchTask 继承自它
from dojo.core.tasks.base import Task
from dojo.core.tasks.constants import (
    EXECUTION_OUTPUT,
    TASK_DESCRIPTION,
    TEST_FITNESS,
    VALID_SOLUTION_FEEDBACK,
    VALIDATION_FITNESS,
    AUX_EVAL_INFO,
    VALID_SOLUTION,
)
# 导入从模型输出文本中提取代码块的工具函数
from dojo.utils.code_parsing import extract_code

# 导入 MLE-Bench 任务的数据类配置（含 eval_script、数据目录等字段）
from dojo.config_dataclasses.task.mlebench import MLEBenchTaskConfig


# 将评估报告统一为可序列化/可比较的形式：数值型一律转为 float，其余原样保留
def parse_report(report: Dict[str, Any]):
    # 初始化解析后的报告字典
    parsed_report = {}
    # 逐项遍历原始报告的键值对
    for key, value in report.items():
        # 若值为布尔或整数/浮点数，统一转成 float，便于后续统一处理与记录
        if isinstance(value, (bool, int, float)):
            parsed_report[key] = float(value)
        # 其他类型（如字符串、列表）原样保留
        else:
            parsed_report[key] = value
    # 返回归一化后的报告
    return parsed_report


# MLE-Bench 任务类：封装"执行解法代码 -> 生成提交文件 -> 官方评分"的完整流程
class MLEBenchTask(Task):
    """
    Represents an MLE Bench task.

    This task executes a provided solution to train a model and expects the solution
    to produce a submission file (submission.csv). It then runs an evaluation script to
    compute a fitness score.

    Expected configuration keys (accessible via self.cfg):
      - eval_script: The evaluation script code.
      - task_description: The task description.
    """

    # 解法代码在执行环境中保存的文件名
    _solution_script = "solution.py"
    # 提交文件（submission.csv）的本地路径，prepare 阶段才赋值
    _submission_file_path = None

    def __init__(self, cfg: MLEBenchTaskConfig) -> None:
        """
        Initialize the MLEBenchTask.

        Args:
            **cfg: Configuration containing keys like 'eval_script' and 'task_description'.
        """
        # 调用父类构造函数，完成通用配置初始化
        super().__init__(cfg)

        # Get instructions
        # 定位本任务源码所在目录（instructions.txt 与其同级）
        self.task_src_path = Path(__file__).resolve().parent
        # 拼出通用指令文件 instructions.txt 的路径
        self.instructions_path = self.task_src_path / "instructions.txt"
        # 读取通用指令文本（告诉智能体如何训练模型并产出提交文件）
        self.instructions = self.instructions_path.read_text()
        # 展开指令中的环境变量（如 $DATA_DIR），使指令与实际运行环境一致
        self.instructions = os.path.expandvars(self.instructions)

        # Read task description.
        # 定位该比赛的公开描述文件 description.md（公开目录下）
        task_description_path = Path(self.cfg.public_dir).resolve() / "description.md"
        # 通用指令与比赛描述拼接为最终任务描述
        self.task_description = self.instructions + "\n" + task_description_path.read_text()

        # Load competition data
        # 让注册表指向本任务缓存的数据目录，得到新的注册表实例
        new_registry = registry.set_data_dir(Path(self.cfg.cache_dir))
        # 按比赛名称取出比赛对象（含训练/测试数据与评分所需元信息）
        self.competition = new_registry.get_competition(self.cfg.name)

        # Resolve paths
        # 解析公开目录的绝对路径（存放 description.md 等公开材料）
        self.public_dir = Path(self.cfg.public_dir).resolve()
        # 解析私有目录的绝对路径（存放评分用脚本等私有材料）
        self.private_dir = Path(self.cfg.private_dir).resolve()

    def prepare(self, **task_args):
        # 以关键字参数作为初始任务状态
        state = task_args
        # 初始观测置为空字典，后续由 step 填充
        state["init_obs"] = {}

        # 依据求解器解释器的工作目录，确定提交文件的完整路径
        self._submission_file_path = Path(task_args["solver_interpreter"].working_dir) / self.cfg.submission_fname

        # Prepare the task info object
        # 查询该比赛的指标方向（如 RMSE 越小越好），对应适应度的优化方向
        lower_is_better = evaluate.is_lower_better(self.competition)
        # 组装任务信息：任务描述 + 指标优化方向
        task_info = {
            TASK_DESCRIPTION: self.task_description,
            "lower_is_better": lower_is_better,
        }

        # 返回初始化后的状态与任务信息
        return state, task_info

    def step_task(self, state: Dict[str, Any], action: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Execute a single step of the task.

        In this task, a step corresponds to evaluating the provided solution code.

        Args:
            state (Dict[str, Any]): The current state of the task.
            action (Any): The solution code (as a string) to evaluate.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing the updated state and the outcome.
        """
        # 尝试从动作文本中提取代码块；格式非法则直接返回失败结果
        try:
            # 从模型输出中解析出可执行的 Python 代码
            solution = extract_code(action)
        except Exception as e:
            # 记录解析失败的具体原因
            self.logger.error(f"The solution does not follow the required format: {e}")
            # 构造一个空的执行结果对象
            exec_output = ExecutionResult.get_empty()
            # 把错误信息写进终端输出，便于智能体看到失败原因
            exec_output.term_out[0] = f"Invalid solution: {e}"
            # 提前返回：无验证适应度，并标记解法无效
            return state, {EXECUTION_OUTPUT: exec_output, VALIDATION_FITNESS: None, VALID_SOLUTION: False}

        # 从状态中取出求解器解释器（在隔离环境中执行智能体代码）
        interpreter = state["solver_interpreter"]
        # 将解法写入 solution.py 并执行，得到执行结果（动作执行环节）
        exec_output: ExecutionResult = interpreter.run(solution, file_name=self._solution_script)
        # 以执行输出初始化评估结果字典
        eval_result = {EXECUTION_OUTPUT: exec_output}

        # Check if the execution was not successful
        # 执行退出码非 0 或超时都视为执行失败
        if (not exec_output.exit_code == 0) or exec_output.timed_out:
            self.logger.error(
                f"Execution failed - exit code: {exec_output.exit_code} - timed out: {exec_output.timed_out} - execution time: {exec_output.exec_time}"
            )
            # If the execution was not succesful, for sanity reasons, we want to remove the submission file if it exists
            # this ensures that the agent does not have access to a submission file that was not generated by a successful execution
            # 执行失败时删除残留的提交文件，防止智能体拿到非成功执行产出的提交
            if self._submission_file_path.exists():
                # 删除本地提交文件（文件不存在也不报错）
                self._submission_file_path.unlink(missing_ok=True)
        else:
            # 执行成功则记录日志，准备取回提交文件用于评估
            self.logger.info(f"Execution successful - fetching submission file for evaluation.")
            # 把提交文件从执行环境拷贝到宿主机（任务侧）以便评分
            interpreter.fetch_file(self._submission_file_path)
            self.logger.info(f"Submission file fetched: {self._submission_file_path}")

        # 判断本地是否存在提交文件
        has_csv_submission = self._submission_file_path.exists()
        # 先默认标记本次解法无效
        eval_result[VALID_SOLUTION] = False
        if has_csv_submission:
            # 校验提交文件是否符合比赛规定的格式（列名、行数、ID 对齐等）
            is_valid_submission, message = validate_submission(self._submission_file_path, self.competition)
            # 把校验结论写入评估结果
            eval_result[VALID_SOLUTION] = is_valid_submission
            # 同时记录校验返回的说明信息，作为给智能体的反馈
            eval_result[VALID_SOLUTION_FEEDBACK] = message
            self.logger.info(
                f"Submission file found: {self._submission_file_path} || Submission valid: {is_valid_submission}"
            )

            if is_valid_submission:
                self.logger.info(f"Evaluating submission: {self._submission_file_path}")
                # 调用官方评估脚本对提交打分，得到测试集适应度与详细报告
                test_fitness, report = evaluate.evaluate_submission(
                    submission_path=self._submission_file_path,
                    data_dir=Path(self.cfg.cache_dir),
                    competition_id=self.cfg.name,
                    results_output_dir=Path(self.cfg.results_output_dir),
                )
                # 记录测试适应度（论文中用于衡量测试性能的 fitness）
                eval_result[TEST_FITNESS] = test_fitness
                # 记录辅助评估信息（归一化后的报告指标）
                eval_result[AUX_EVAL_INFO] = parse_report(report)
                self.logger.info(f"Test fitness: {test_fitness} || AUX eval info: {eval_result[AUX_EVAL_INFO]}")

            # 无论是否有效都删除本地提交文件，避免下一轮残留干扰
            self._submission_file_path.unlink(missing_ok=True)  # remove the submission_file locally
            # 断言本地已不存在提交文件
            assert not self._submission_file_path.exists(), (
                "At this point, the submissions file should not exists locally!"
            )

        # 若解释器由工厂管理（每次执行使用新容器），直接整体关闭
        if interpreter.factory:
            interpreter.close()
        else:
            interpreter.run(
                f"!rm {self.cfg.submission_fname}"
            )  # remove the submission_file from the agent's environment
            # 断言智能体环境中的提交文件已被清理（取回为 None 说明已不存在）
            assert interpreter.fetch_file(self._submission_file_path) is None, (
                "At this point, the submissions file should not exists in the agent's environment!"
            )

        # 返回更新后的状态与评估结果
        return state, eval_result

    def evaluate_fitness(
        self,
        solution: Optional[Any] = None,
        state: Optional[Dict[str, Any]] = None,
        interpreter: Optional[Interpreter] = None,
        aux_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # 提交文件路径必须在 prepare 阶段已设置，否则无法评估
        if self._submission_file_path is None:
            raise Exception("The path to the submission file must be set.")

        # 运行最终解法代码，得到执行输出
        exec_output = interpreter.run(solution, file_name=self._solution_script)
        # 以执行输出初始化评估结果
        eval_result = {EXECUTION_OUTPUT: exec_output}

        # 从执行环境取回提交文件
        interpreter.fetch_file(self._submission_file_path)
        # 检查提交文件是否成功生成
        has_csv_submission = self._submission_file_path.exists()
        # 最终解法必须产出提交文件，否则视为无效
        assert has_csv_submission, "The final solution is not valid!"

        # 调用官方评估流程对最终提交打分
        test_fitness, report = evaluate.evaluate_submission(
            submission_path=self._submission_file_path,
            data_dir=Path(self.cfg.cache_dir),
            competition_id=self.cfg.name,
            results_output_dir=Path(self.cfg.results_output_dir),
        )
        # 记录测试集适应度与辅助评估信息
        eval_result[TEST_FITNESS] = test_fitness
        eval_result[AUX_EVAL_INFO] = parse_report(report)

        # 返回最终评估结果
        return eval_result

    def close(self, state):
        # 遍历求解器与评估器两类解释器，逐一清理
        for interp_key in ["solver_interpreter", "eval_interpreter"]:
            # 状态中不存在该解释器则跳过
            if interp_key not in state:
                continue

            # 取出解释器实例
            interpreter = state[interp_key]
            # 若支持会话清理则先清理会话资源
            if hasattr(interpreter, "cleanup_session"):
                interpreter.cleanup_session()

            # 若支持资源释放则做整体清理
            if hasattr(interpreter, "clean_up"):
                interpreter.clean_up()
