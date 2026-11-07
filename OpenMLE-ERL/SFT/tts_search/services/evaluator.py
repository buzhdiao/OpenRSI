"""
Evaluator Service - Handles sandbox code evaluation.

This service is responsible for evaluating generated code solutions
using the sandbox environment. It operates independently from the
generation service and communicates through async queues.
"""
# 评估服务模块：用沙箱环境评估生成的代码方案，与生成服务解耦、
# 通过异步队列与调度器通信（类似 SGLang prefill/decode 分离的架构思想）。

from __future__ import annotations

# 导入 asyncio：信号量并发控制与批量并发 gather
import asyncio
# 导入 logging：模块日志
import logging
# 导入 dataclass/field：声明请求与结果数据结构
from dataclasses import dataclass, field
# 导入 datetime：解析沙箱返回的时间戳以计算排队/运行时长
from datetime import datetime
# 导入 Path：输出产物目录处理
from pathlib import Path
# 导入类型注解 Any
from typing import Any

# 导入 httpx：异步 HTTP 客户端
import httpx

# 导入 eval_utils：榜单加载与奖牌计算工具
from tts_search import eval_utils
# 导入奖励/日志工具函数
from tts_search.reward_func_utils import (
    format_sandbox_feedback,
    get_clear_log,
    get_sandbox_result,
    normalize_sandbox_resource_type,
    resolve_sandbox_resource_type,
    score2reward,
)

# 模块级 logger
logger = logging.getLogger(__name__)


@dataclass
class EvaluationRequest:
    """Request for code evaluation."""

    # 字段说明：request_id=请求唯一标识；task_id/task_name=任务标识；
    # code=待评估代码；data_dir=任务数据目录；metadata=任务元信息（含指标配置）；
    # step_index=当前搜索步数；output_dir=产物保存目录（可选）；
    # use_score2reward=是否把分数换算为奖励；generation_*_tokens=生成阶段的 token 用量（透传记录）
    request_id: str
    task_id: str
    task_name: str
    code: str
    data_dir: str
    metadata: dict[str, Any]
    step_index: int
    output_dir: Path | None = None
    use_score2reward: bool = True
    generation_prompt_tokens: int | None = None
    generation_completion_tokens: int | None = None
    generation_total_tokens: int | None = None


@dataclass
class EvaluationResult:
    """Result from code evaluation."""

    # 字段说明：request_id/task_id/task_name=请求与任务标识；
    # status_code=沙箱 HTTP 状态码；status=执行结果状态；score=沙箱分数；reward=换算后的奖励；
    # job_id=沙箱任务 ID；queue_time/run_time=排队/运行时长（秒）；raw_run_log/clear_run_log=原始与清洗日志；
    # feedback=格式化后的反馈文本；submit_grade/submit_medal=提交分数档位与奖牌；metadata=任务元信息；
    # step_index/data_dir/code=请求回传字段；output_dir=产物目录；generation_*_tokens=生成阶段 token 用量；
    # error=内部错误信息；timestamp=结果创建时间
    request_id: str
    task_id: str
    task_name: str
    status_code: int
    status: str
    score: float | None
    reward: float
    job_id: str | None
    queue_time: float | None
    run_time: float | None
    raw_run_log: str | None
    clear_run_log: str
    feedback: str
    submit_grade: float | None
    submit_medal: str
    metadata: dict[str, Any]
    step_index: int
    data_dir: str
    code: str
    output_dir: Path | None = None
    generation_prompt_tokens: int | None = None
    generation_completion_tokens: int | None = None
    generation_total_tokens: int | None = None
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def success(self) -> bool:
        # 仅当 HTTP 成功且拿到了分数才视为评估成功
        return self.status_code == 200 and self.score is not None


class EvaluatorService:
    """
    Service for sandbox code evaluation.

    This service handles:
    - Async sandbox evaluation with retry logic
    - Score and reward calculation
    - Test set evaluation (no validation set)
    - Output artifact persistence
    """

    def __init__(
        self,
        base_url: str,
        concurrency: int = 32,
        job_timeout: int = 3600,
        wait_timeout: int = 86400,
        poll_interval: int = 5,
        cpu_base_url: str | None = None,
        resource_type_override: str | None = None,
    ):
        """
        Initialize the evaluator service.

        Args:
            base_url: Sandbox API base URL for GPU resources
            concurrency: Maximum concurrent evaluation requests
            job_timeout: Job execution timeout in seconds
            wait_timeout: Maximum wait time for job completion
            poll_interval: Polling interval for job status
            cpu_base_url: Optional separate base URL for CPU resources (defaults to base_url)
            resource_type_override: Force all sandbox jobs to use this resource type
        """
        # GPU 沙箱服务地址
        self._base_url = base_url
        # CPU 沙箱服务地址（未提供则复用 GPU 地址）
        self._cpu_base_url = cpu_base_url or base_url
        # 最大并发评估数
        self._concurrency = concurrency
        # 用信号量限制并发
        self._semaphore = asyncio.Semaphore(concurrency)
        # 任务执行超时（秒）
        self._job_timeout = job_timeout
        # 等待任务完成的最大时长（秒）
        self._wait_timeout = wait_timeout
        # 状态轮询间隔（秒）
        self._poll_interval = poll_interval
        # 资源类型强制覆盖：配置了则所有任务统一使用该类型
        self._resource_type_override = (
            normalize_sandbox_resource_type(resource_type_override, default="gpu")
            if resource_type_override is not None
            else None
        )

        # HTTP clients will be created in context manager
        # GPU 客户端句柄（进入异步上下文时创建）
        self._gpu_client: httpx.AsyncClient | None = None
        # CPU 客户端句柄（进入异步上下文时创建）
        self._cpu_client: httpx.AsyncClient | None = None

        # Statistics
        # 以下为运行统计：总评估数、成功数、失败数、累计分数、各状态计数
        self._total_evaluations = 0
        self._successful_evaluations = 0
        self._failed_evaluations = 0
        self._total_score = 0.0
        self._status_counts: dict[str, int] = {}

    def _build_internal_error_result(
        self, request: EvaluationRequest, exc: Exception
    ) -> EvaluationResult:
        """Convert unexpected evaluator exceptions into a normal failed result."""
        # 把评估过程中的意外异常封装成常规失败结果，保证批处理不中断
        status = "internal_error"
        # 统计计数更新：总数与失败数
        self._total_evaluations += 1
        self._failed_evaluations += 1
        self._status_counts[status] = self._status_counts.get(status, 0) + 1

        # 错误文本与反馈信息
        error_text = f"{type(exc).__name__}: {exc}"
        feedback = f"Internal evaluation error: {error_text}"
        # 构造 500 状态、无分数、零奖励的失败结果
        return EvaluationResult(
            request_id=request.request_id,
            task_id=request.task_id,
            task_name=request.task_name,
            status_code=500,
            status=status,
            score=None,
            reward=0.0,
            job_id=None,
            queue_time=None,
            run_time=None,
            raw_run_log=None,
            clear_run_log="",
            feedback=feedback,
            submit_grade=None,
            submit_medal="N/A",
            metadata=request.metadata,
            step_index=request.step_index,
            data_dir=request.data_dir,
            code=request.code,
            output_dir=request.output_dir,
            generation_prompt_tokens=request.generation_prompt_tokens,
            generation_completion_tokens=request.generation_completion_tokens,
            generation_total_tokens=request.generation_total_tokens,
            error=error_text,
        )

    async def __aenter__(self) -> "EvaluatorService":
        """Create HTTP clients on context entry."""
        # 配置了资源类型覆盖时打日志提示
        if self._resource_type_override is not None:
            logger.info(
                "Sandbox resource override enabled: forcing all evaluations to %s",
                self._resource_type_override,
            )

        # HTTP 超时配置：连接 10s / 读 100s / 写 30s / 连接池 60s
        timeout = httpx.Timeout(connect=10, read=100, write=30, pool=60)
        # 连接池上限：按并发数的 1.5 倍并夹在 [16,128] 之间
        max_connections = min(max(int(1.5 * self._concurrency), 16), 128)
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
            keepalive_expiry=30.0,
        )

        # Create GPU client
        # 创建 GPU 沙箱客户端
        self._gpu_client = httpx.AsyncClient(
            base_url=self._base_url, limits=limits, timeout=timeout
        )

        # Create CPU client (reuse GPU client if URLs are the same)
        # 创建 CPU 客户端：地址相同则直接复用 GPU 客户端
        if self._cpu_base_url == self._base_url:
            self._cpu_client = self._gpu_client
        else:
            self._cpu_client = httpx.AsyncClient(
                base_url=self._cpu_base_url, limits=limits, timeout=timeout
            )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close HTTP clients on context exit."""
        # Close GPU client
        # 关闭 GPU 客户端并置空
        if self._gpu_client:
            await self._gpu_client.aclose()
            self._gpu_client = None

        # Close CPU client only if it's different from GPU client
        # CPU 客户端与 GPU 客户端不同实例时才需要单独关闭
        if self._cpu_client and self._cpu_client != self._gpu_client:
            await self._cpu_client.aclose()
        self._cpu_client = None

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """
        Evaluate code in sandbox.

        Args:
            request: Evaluation request with code and task details

        Returns:
            EvaluationResult with score and execution details
        """
        # 必须在异步上下文管理器内使用（客户端已创建）
        if self._gpu_client is None or self._cpu_client is None:
            raise RuntimeError("EvaluatorService must be used as async context manager")

        # Determine the actual data directory based on eval mode
        # 取请求数据目录与代码
        data_dir = request.data_dir
        code = request.code
        print(f"data_dir: {data_dir}")

        # Resolve effective resource_type from config override or task metadata.
        # 解析生效的资源类型：配置覆盖 > 任务 metadata > 默认 gpu
        resource_type = resolve_sandbox_resource_type(
            request.metadata,
            override=self._resource_type_override,
            default="gpu",
        )

        # Select client based on resource_type
        # 按资源类型选择对应的沙箱客户端
        client = (
            self._cpu_client if resource_type.lower() == "cpu" else self._gpu_client
        )

        # 用信号量限流后提交沙箱任务并等待完成
        async with self._semaphore:
            status_code, payload = await get_sandbox_result(
                client=client,
                code_str=code,
                data_dir=data_dir,
                job_timeout=self._job_timeout,
                wait_timeout=self._wait_timeout,
                resource_type=resource_type,
                priority=2,
                poll_interval=self._poll_interval,
            )

        # 提取沙箱任务 ID 与结果负载
        sandbox_job_id = payload.get("job_id")
        result_payload = payload.get("result") or {}
        status = result_payload.get("result") or "unknown"
        status = str(status)

        # 提取分数：HTTP 非 200 或无分数时置 None
        score_value = result_payload.get("score")
        if status_code != 200 or score_value is None:
            score = None
        else:
            score = float(score_value)

        # 计算奖励：开启 use_score2reward 时做分数→奖励映射（power_sigmoid 塑形），否则直接用分数
        reward = 0.0
        if score is not None:
            reward = (
                score2reward(score, request.metadata, mode="power_sigmoid")
                if request.use_score2reward
                else score
            )

        # 初始化提交档位与奖牌
        submit_grade: float | None = None
        submit_medal = "N/A"
        if score is not None:
            try:
                # 加载任务榜单，把分数映射为档位与金银铜奖牌
                leaderboard = eval_utils.load_leaderboard(request.metadata)
                if leaderboard is not None:
                    submit_grade, submit_medal = (
                        eval_utils.build_submit_grade_and_medal(score, leaderboard)
                    )
            except Exception as exc:
                # 奖牌计算失败仅告警，不影响评估主流程
                logger.warning(
                    "Failed to compute submit medal for task=%s score=%s: %s",
                    request.task_name,
                    score,
                    exc,
                )

        # 初始化排队/运行时长
        queue_time = None
        run_time = None
        if status_code == 200:
            # 从沙箱时间戳计算运行与排队时长
            created_at = payload.get("created_at")
            started_at = payload.get("started_at")
            completed_at = payload.get("completed_at")
            # 有开始与完成时间才能算运行时长
            if started_at and completed_at:
                # 解析开始时间（ISO8601，Z 后缀转 +00:00）
                started_ts = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                # 解析完成时间
                completed_ts = datetime.fromisoformat(
                    completed_at.replace("Z", "+00:00")
                )
                # 运行时长 = 完成 - 开始
                run_time = (completed_ts - started_ts).total_seconds()
                if created_at:
                    # 解析创建时间
                    created_ts = datetime.fromisoformat(
                        created_at.replace("Z", "+00:00")
                    )
                    # 排队时长 = 开始 - 创建
                    queue_time = (started_ts - created_ts).total_seconds()

        # 提取原始日志、清洗日志与反馈文本
        raw_run_log = result_payload.get("run_log")
        clear_run_log = get_clear_log(raw_run_log)
        feedback = format_sandbox_feedback(status_code, payload)

        # Save artifacts if output directory is specified
        # 指定了输出目录时保存产物
        if request.output_dir:
            print(f"request.output_dir: {request.output_dir}")
            # 按步数建立子目录
            step_dir = request.output_dir / f"step_{request.step_index}"
            step_dir.mkdir(parents=True, exist_ok=True)
            # 保存原始运行日志
            (step_dir / "raw_run_log.txt").write_text(
                str(raw_run_log or ""), encoding="utf-8"
            )
            # 保存清洗后的运行日志
            (step_dir / "clear_run_log.txt").write_text(
                str(clear_run_log or ""), encoding="utf-8"
            )
            # 保存反馈文本
            (step_dir / "feedback.txt").write_text(
                str(feedback or ""), encoding="utf-8"
            )

        # Update statistics
        # 更新统计：总数与各状态计数
        self._total_evaluations += 1
        self._status_counts[status] = self._status_counts.get(status, 0) + 1
        if score is not None:
            # 有分数计入成功并累计分数
            self._successful_evaluations += 1
            self._total_score += score
        else:
            # 无分数计入失败
            self._failed_evaluations += 1

        # 组装并返回评估结果
        return EvaluationResult(
            request_id=request.request_id,
            task_id=request.task_id,
            task_name=request.task_name,
            status_code=status_code,
            status=status,
            score=score,
            reward=reward,
            job_id=sandbox_job_id,
            queue_time=queue_time,
            run_time=run_time,
            raw_run_log=raw_run_log,
            clear_run_log=clear_run_log,
            feedback=feedback,
            submit_grade=submit_grade,
            submit_medal=submit_medal,
            metadata=request.metadata,
            step_index=request.step_index,
            data_dir=request.data_dir,
            code=request.code,
            output_dir=request.output_dir,
            generation_prompt_tokens=request.generation_prompt_tokens,
            generation_completion_tokens=request.generation_completion_tokens,
            generation_total_tokens=request.generation_total_tokens,
        )

    async def evaluate_batch(
        self, requests: list[EvaluationRequest]
    ) -> list[EvaluationResult]:
        """
        Evaluate multiple code samples concurrently.

        Args:
            requests: List of evaluation requests

        Returns:
            List of evaluation results
        """
        # 为每个请求创建协程任务
        tasks = [self.evaluate(req) for req in requests]
        # 并发执行；异常不立即抛出，收集后逐个处理
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[EvaluationResult] = []
        # 与请求一一配对检查结果
        for request, result in zip(requests, raw_results, strict=True):
            if isinstance(result, Exception):
                # 协程崩溃：记录错误日志并封装为内部错误结果
                logger.error(
                    "Evaluation crashed for request_id=%s task_id=%s step_index=%s",
                    request.request_id,
                    request.task_id,
                    request.step_index,
                    exc_info=(type(result), result, result.__traceback__),
                )
                results.append(self._build_internal_error_result(request, result))
            else:
                # 正常结果直接收集
                results.append(result)
        return results

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        # 返回评估统计：总数/成功/失败/总分/平均分/状态分布
        return {
            "total_evaluations": self._total_evaluations,
            "successful_evaluations": self._successful_evaluations,
            "failed_evaluations": self._failed_evaluations,
            "total_score": self._total_score,
            "average_score": (
                self._total_score / self._successful_evaluations
                if self._successful_evaluations > 0
                else 0.0
            ),
            "status_counts": dict(self._status_counts),
        }
