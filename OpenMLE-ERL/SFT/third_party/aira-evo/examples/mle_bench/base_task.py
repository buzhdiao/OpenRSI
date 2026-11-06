# 启用延迟注解求值（PEP 563），注解以字符串形式保存
from __future__ import annotations

# 导入 asyncio，用于在同步上下文中运行异步沙箱请求
import asyncio
# 导入 os，用于读取环境变量
import os
# 导入 random，用于假沙箱模式的确定性随机打分
import random
# 导入 datetime，用于解析沙箱返回的 ISO 时间戳
from datetime import datetime
# 导入 Path，用于路径操作
from pathlib import Path
# 导入 Any 类型注解
from typing import Any

# 导入 httpx 异步 HTTP 客户端，用于与沙箱服务通信
import httpx

# 导入 dojo 的执行结果与解释器基类
from dojo.core.interpreters.base import ExecutionResult, Interpreter
# 导入 dojo 的任务基类 Task
from dojo.core.tasks.base import Task
# 导入 dojo 任务结果字段的常量键名
from dojo.core.tasks.constants import (
    AUX_EVAL_INFO,
    EXECUTION_OUTPUT,
    TASK_DESCRIPTION,
    VALID_SOLUTION,
    VALID_SOLUTION_FEEDBACK,
    VALIDATION_FITNESS,
)
# 导入 tts_search 的奖励函数工具模块
from tts_search import reward_func_utils
# 导入沙箱反馈格式化、日志清洗、结果获取与分数转奖励的工具函数
from tts_search.reward_func_utils import (
    format_sandbox_feedback,
    get_clear_log,
    get_sandbox_result,
    score2reward,
)
# 导入沙箱并发槽位控制（限流）
from tts_search.airaevo_concurrency import acquire_sandbox_slot


# 基于 dojo.Task 的 MLE-Bench 沙箱评测任务
class SandboxMLEBenchTask(Task):
    # 初始化任务配置
    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        time_budget: float | None = None,
    ) -> None:
        # 调用 dojo Task 基类初始化
        super().__init__(cfg)
        # 拷贝一份配置字典，避免外部修改影响内部状态
        self.cfg = dict(cfg)
        # 验证阶段数据目录
        self.validation_data_dir = str(self.cfg["data_dir"])
        # 提交（测试）目录名
        self.submit_dir = str(self.cfg["submit_dir"])
        # 可选的提交数据根目录
        self.submit_data_dir_root = self.cfg.get("submit_data_dir_root")
        # 验证阶段时间预算（秒），超出后请求停止搜索
        self.time_budget = float(time_budget) if time_budget is not None else None
        # 已使用的验证时间累计
        self.validation_time_used = 0.0
        # 是否已请求停止搜索
        self.stop_requested = False

        # 解析验证数据目录路径
        validation_path = Path(self.validation_data_dir)
        # 记录验证目录名（用于后续代码中的路径替换）
        self.validation_dir_name = validation_path.parent.name
        # 提供了独立提交数据根目录时
        if self.submit_data_dir_root:
            # Keep submit evaluation aligned with the ordinary runner semantics.
            # 提交数据目录 = 提交根目录 / 验证目录名
            self.submit_data_dir = str(
                Path(str(self.submit_data_dir_root)) / validation_path.name
            )
        else:
            # 默认规则：验证目录上两级 / submit_dir / 验证目录名
            self.submit_data_dir = str(
                validation_path.parent.parent / self.submit_dir / validation_path.name
            )
        # 公共系统提示词
        self.public_system_prompt = str(self.cfg.get("public_system_prompt") or "")
        # 公共用户提示词
        self.public_user_prompt = str(self.cfg.get("public_user_prompt") or "")
        # 原始任务描述
        self.raw_task_description = str(self.cfg.get("task_description") or "")
        # 数据描述
        self.data_description = str(self.cfg.get("data_description") or "")
        # 拼装最终任务描述
        self.task_description = self._build_task_description()
        # 指标是否越低越好（取反 higher_is_better）
        self.lower_is_better = not bool(self.cfg["higher_is_better"])

    # 拼装任务描述：任务描述 + 数据描述两个小节
    def _build_task_description(self) -> str:
        # 组织（标题, 内容）小节列表
        sections = [
            ("Task Description", self.raw_task_description),
            ("Data Description", self.data_description),
        ]
        # 过滤空小节后用空行拼接
        return "\n\n".join(
            f"{title}:\n{content.strip()}" for title, content in sections if content.strip()
        )

    # 构建提交用代码：默认把验证目录名替换为提交目录名
    def build_submit_code(self, code: str) -> str:
        # 提供独立提交数据根目录时无需替换路径
        if self.submit_data_dir_root:
            return code
        # 把代码中出现的验证目录名替换为提交目录
        return code.replace(self.validation_dir_name, self.submit_dir)

    # 准备任务初始状态与任务信息
    def prepare(self, **task_args: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        # 由外部传入的参数构造状态字典
        state = dict(task_args)
        # 初始观测为空字典
        state["init_obs"] = {}
        # 写入已使用的验证时间
        state["validation_time_used"] = self.validation_time_used
        # 组装任务信息字典（供求解器使用）
        task_info = {
            TASK_DESCRIPTION: self.task_description,
            "lower_is_better": self.lower_is_better,
            "task_description": self.raw_task_description,
            "data_description": self.data_description,
            "public_system_prompt": self.public_system_prompt,
            "public_user_prompt": self.public_user_prompt,
        }
        # 返回初始状态与任务信息
        return state, task_info

    # 异步执行沙箱评测（或假沙箱模拟）
    async def _run_eval_async(self, code: str, *, data_dir: str) -> dict[str, Any]:
        # 假沙箱模式：不真正执行代码，仅做语法检查并返回模拟分数
        if os.getenv("AIRAEVO_FAKE_SANDBOX") == "1":
            # 取代码文本
            code_text = str(code or "")
            # 空代码视为失败
            if not code_text.strip():
                return self._fake_sandbox_failure(
                    reason="empty code",
                    data_dir=data_dir,
                )
            try:
                # 仅做语法编译检查，不执行
                compile(code_text, "<airaevo_fake_sandbox>", "exec")
            except SyntaxError as exc:
                # 语法错误时返回失败结果
                return self._fake_sandbox_failure(
                    reason=f"syntax error: {exc.msg}",
                    data_dir=data_dir,
                )

            # 生成下一个模拟分数
            score = self._next_fake_score()
            # 按配置决定是否把分数转换为奖励
            reward = (
                score2reward(score, self.cfg, mode="power_sigmoid")
                if bool(self.cfg["sandbox"]["use_score2reward"])
                else score
            )
            # 返回模拟的评测结果字典
            return {
                "status_code": 200,
                "status": "completed",
                "score": score,
                "reward": reward,
                "job_id": f"fake-sandbox-{score:.4f}",
                "queue_time": 0.0,
                "run_time": 1.0,
                "raw_run_log": f"FAKE_SANDBOX score={score:.4f}",
                "clear_run_log": f"FAKE_SANDBOX score={score:.4f}",
                "feedback": f"FAKE_SANDBOX completed with score={score:.4f}",
                "data_dir": data_dir,
            }

        # 真实沙箱模式：拷贝沙箱配置
        sandbox_cfg = dict(self.cfg["sandbox"])
        # 读取资源类型（cpu 或 gpu）
        resource = str(sandbox_cfg["resource"])
        # 按资源类型设置沙箱服务的 API key（模块级全局）
        reward_func_utils.API_KEY = os.environ[
            "SANDBOX_CPU_API_KEY" if resource == "cpu" else "SANDBOX_GPU_API_KEY"
        ]

        # 配置 HTTP 超时（连接/读/写/连接池）
        timeout = httpx.Timeout(connect=10, read=100, write=30, pool=60)
        # 配置连接池上限与保活时间
        limits = httpx.Limits(
            max_connections=16,
            max_keepalive_connections=16,
            keepalive_expiry=30.0,
        )
        # 创建异步 HTTP 客户端与沙箱服务通信
        async with httpx.AsyncClient(
            base_url=str(sandbox_cfg["base_url"]),
            limits=limits,
            timeout=timeout,
            trust_env=False,
            verify=False,
        ) as client:
            # 提交代码并轮询直到拿到执行结果
            status_code, payload = await get_sandbox_result(
                client=client,
                code_str=code,
                data_dir=data_dir,
                resource_type=resource,
                priority=1,
                job_timeout=int(sandbox_cfg["job_timeout"]),
                wait_timeout=int(sandbox_cfg["wait_timeout"]),
                poll_interval=int(sandbox_cfg["poll_interval"]),
            )

        # 提取结果负载
        result_payload = payload.get("result") or {}
        # 读取执行状态字符串
        status = str(result_payload.get("result") or "unknown")
        # 读取分数：仅请求成功且有分数时有效
        score_value = result_payload.get("score")
        score = float(score_value) if status_code == 200 and score_value is not None else None

        # 奖励默认 0
        reward = 0.0
        # 分数有效时按配置转换为奖励
        if score is not None:
            reward = (
                score2reward(score, self.cfg, mode="power_sigmoid")
                if bool(sandbox_cfg["use_score2reward"])
                else score
            )

        # 排队时间与运行时间默认未知
        queue_time = None
        run_time = None
        # 请求成功时尝试由时间戳推算
        if status_code == 200:
            # 读取创建/开始/完成时间戳
            created_at = payload.get("created_at")
            started_at = payload.get("started_at")
            completed_at = payload.get("completed_at")
            # 同时有开始与完成时间时计算运行时长
            if started_at and completed_at:
                # 解析开始时间（ISO8601，Z 后缀转 +00:00）
                started_ts = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                # 解析完成时间
                completed_ts = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
                # 运行时间 = 完成 - 开始
                run_time = (completed_ts - started_ts).total_seconds()
                # 有创建时间时再计算排队时长
                if created_at:
                    created_ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    # 排队时间 = 开始 - 创建
                    queue_time = (started_ts - created_ts).total_seconds()

        # 原始运行日志
        raw_run_log = result_payload.get("run_log")
        # 清洗后的运行日志
        clear_run_log = get_clear_log(raw_run_log)
        # 格式化的反馈文本
        feedback = format_sandbox_feedback(status_code, payload)

        # 返回统一的评测结果字典
        return {
            "status_code": status_code,
            "status": status,
            "score": score,
            "reward": reward,
            "job_id": payload.get("job_id"),
            "queue_time": queue_time,
            "run_time": run_time,
            "raw_run_log": raw_run_log,
            "clear_run_log": clear_run_log,
            "feedback": feedback,
            "data_dir": data_dir,
        }

    # 假沙箱模式下生成下一个模拟分数
    def _next_fake_score(self) -> float:
        # 从环境变量读取可选的分数列表（逗号分隔）
        scores = [
            float(item)
            for item in os.getenv("AIRAEVO_FAKE_SANDBOX_SCORES", "").split(",")
            if item.strip()
        ]
        # 读取计数器文件路径（用于多次调用时递增索引）
        counter_path_value = os.getenv("AIRAEVO_FAKE_SANDBOX_COUNTER_PATH")
        # 转成 Path 对象（未配置时为 None）
        counter_path = Path(counter_path_value) if counter_path_value else None
        # 索引默认 0
        index = 0
        # 计数器文件存在时读取当前索引
        if counter_path is not None and counter_path.exists():
            index = int(counter_path.read_text(encoding="utf-8").strip() or "0")
        # 配置了分数列表时按索引循环取值
        if scores:
            score = scores[index % len(scores)]
        else:
            # 否则用 种子+索引 生成确定性随机分数
            rng = random.Random(int(os.getenv("AIRAEVO_FAKE_SANDBOX_SEED", "42")) + index)
            score = rng.random()
        # 计数器文件可用时写入索引+1，供下次调用递增
        if counter_path is not None:
            counter_path.parent.mkdir(parents=True, exist_ok=True)
            counter_path.write_text(str(index + 1), encoding="utf-8")
        # 返回浮点分数
        return float(score)

    # 假沙箱失败结果的统一构造
    def _fake_sandbox_failure(self, *, reason: str, data_dir: str) -> dict[str, Any]:
        # 构造失败反馈文本
        feedback = f"FAKE_SANDBOX failed: {reason}"
        # 返回状态码 400 的失败结果字典
        return {
            "status_code": 400,
            "status": "failed",
            "score": None,
            "reward": 0.0,
            "job_id": "fake-sandbox-failed",
            "queue_time": 0.0,
            "run_time": 0.0,
            "raw_run_log": feedback,
            "clear_run_log": feedback,
            "feedback": feedback,
            "data_dir": data_dir,
        }

    # 评测代码入口：按阶段选择数据目录并累计时间
    def evaluate_code(self, code: str, *, phase: str) -> dict[str, Any]:
        # 默认使用验证数据目录
        data_dir = self.validation_data_dir
        # 默认直接运行原代码
        code_to_run = code
        # 测试（提交）阶段：切换到提交数据目录并替换代码中的路径
        if phase == "test":
            data_dir = self.submit_data_dir
            code_to_run = self.build_submit_code(code)

        # 占用一个沙箱并发槽位后同步运行异步评测
        with acquire_sandbox_slot():
            eval_payload = asyncio.run(self._run_eval_async(code_to_run, data_dir=data_dir))
        # 记录评测阶段
        eval_payload["phase"] = phase
        # 验证阶段成功执行时累计沙箱运行时间
        if (
            phase == "validation"
            and eval_payload["status_code"] == 200
            and eval_payload.get("run_time") is not None
        ):
            # 累加本次运行时间
            self.validation_time_used += float(eval_payload["run_time"])
            # 超出时间预算时请求停止搜索
            if self.time_budget is not None and self.validation_time_used >= self.time_budget:
                self.stop_requested = True
        # 返回评测结果
        return eval_payload

    # 把评测结果转换为 dojo 的 ExecutionResult
    def _build_execution_result(self, eval_payload: dict[str, Any]) -> ExecutionResult:
        # 读取清洗后的日志
        clear_run_log = str(eval_payload.get("clear_run_log") or "")
        # 读取反馈文本
        feedback = str(eval_payload.get("feedback") or "")
        # 终端输出优先用清洗日志，否则用反馈
        terminal_output = clear_run_log if clear_run_log else feedback
        # 读取状态字符串
        status = str(eval_payload.get("status") or "")
        # 构造执行结果对象
        return ExecutionResult(
            term_out=[terminal_output],
            exec_time=float(eval_payload.get("run_time") or 0.0),
            exit_code=0 if eval_payload.get("status_code") == 200 else 1,
            timed_out=bool(
                eval_payload.get("status_code") == 504 or status.lower() == "timeout"
            ),
        )

    # dojo 步进接口：执行一次验证评测并返回结果
    def step_task(
        self,
        state: dict[str, Any],
        action: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # 把动作（代码字符串）作为验证阶段评测
        eval_payload = self.evaluate_code(str(action), phase="validation")
        # 更新状态中的已用验证时间
        state["validation_time_used"] = self.validation_time_used
        # 状态码 200 视为有效解
        valid_solution = eval_payload["status_code"] == 200
        # 组装评测结果字典（使用 dojo 常量键名）
        eval_result = {
            EXECUTION_OUTPUT: self._build_execution_result(eval_payload),
            VALIDATION_FITNESS: eval_payload.get("score") if valid_solution else None,
            AUX_EVAL_INFO: dict(eval_payload),
            VALID_SOLUTION: valid_solution,
            VALID_SOLUTION_FEEDBACK: str(eval_payload.get("feedback") or ""),
        }
        # 返回更新后的状态与结果
        return state, eval_result

    # 计算最终适应度：用测试（提交）阶段评测解
    def evaluate_fitness(
        self,
        solution: Any | None = None,
        state: dict[str, Any] | None = None,
        interpreter: Interpreter | None = None,
        aux_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # 无解时返回空结果
        if solution is None:
            return {}

        # 以测试阶段评测解代码
        eval_payload = self.evaluate_code(str(solution), phase="test")
        # 状态码 200 视为有效解
        valid_solution = eval_payload["status_code"] == 200
        # 返回与 step_task 结构一致的结果
        return {
            EXECUTION_OUTPUT: self._build_execution_result(eval_payload),
            VALIDATION_FITNESS: eval_payload.get("score") if valid_solution else None,
            AUX_EVAL_INFO: dict(eval_payload),
            VALID_SOLUTION: valid_solution,
            VALID_SOLUTION_FEEDBACK: str(eval_payload.get("feedback") or ""),
        }

    # 关闭任务：清理状态中的解释器会话
    def close(self, state: dict[str, Any]) -> None:
        # 遍历求解器/评测两类解释器
        for interp_key in ("solver_interpreter", "eval_interpreter"):
            # 取出解释器实例
            interpreter = state.get(interp_key)
            # 不存在则跳过
            if interpreter is None:
                continue
            # 支持时调用会话清理
            if hasattr(interpreter, "cleanup_session"):
                interpreter.cleanup_session()
            # 支持时调用清理接口
            if hasattr(interpreter, "clean_up"):
                interpreter.clean_up()
