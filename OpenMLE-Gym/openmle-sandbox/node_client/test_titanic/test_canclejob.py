# 中文说明:沙箱"取消作业"链路的手工联调脚本(node_client/test_titanic 下)。
# 用途:向 node_controller 提交一个"占住 GPU 显存"的长作业,等它进入 running 后
# 发起 DELETE 取消,并继续轮询观察作业是否真的终止(验证取消语义与状态机)。
# 用法:既可只取消已有作业(--job-id),也可走完整"提交 -> 运行 -> 取消 -> 观察"流程;
# 默认地址/密钥都是占位值,实际使用时用命令行参数覆盖。
# 关键设计:作业代码是动态生成的 torch 显存占位脚本(分配后按秒打印持有量),
# 便于肉眼确认"取消前后"显存的占用与释放。

# argparse:命令行参数解析
import argparse
# asyncio:异步主流程与轮询睡眠
import asyncio
# textwrap:生成作业代码的去缩进模板
import textwrap

# httpx:异步 HTTP 客户端(提交/查询/取消)
import httpx


BASE_URL = "http://203.0.113.10:6580"  # 占位地址：替换为你的控制器地址
API_KEY = "mlsandbox-oss-3fad186210e530e6b4cc53576cd2723e"
# 测试任务的数据目录(沙箱内可见的 titanic 数据)
DEFAULT_DATA_DIR = "/mnt/pubdatasets2/MLTasks/Selected_Dojo/titanic"
# 终态状态集合(cancelled 两种拼写都兼容)
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "canceled"}


def build_gpu_hold_code(vram_gb: float, hold_seconds: int) -> str:
    """生成"显存占位"作业代码:分配约 vram_gb GiB 并持有 hold_seconds 秒后退出。"""
    # 目标字节数(GB -> 字节,至少 1)
    target_bytes = max(1, int(vram_gb * 1024 * 1024 * 1024))
    # 去缩进返回整段可执行代码(双花括号是字面量转义)
    return textwrap.dedent(
        f"""
        from __future__ import annotations

        import sys
        import time
        from pathlib import Path

        import pandas as pd
        import torch

        TARGET_BYTES = {target_bytes}
        HOLD_SECONDS = {hold_seconds}


        def main() -> int:
            if not torch.cuda.is_available():
                print("[job] CUDA is not available", file=sys.stderr, flush=True)
                return 1

            device = torch.device("cuda:0")
            print(
                f"[job] allocating about {{TARGET_BYTES / 1024**3:.2f}} GiB on {{device}}",
                flush=True,
            )

            tensor = torch.empty(int(TARGET_BYTES), dtype=torch.uint8, device=device)
            print(
                f"[job] allocation done, requested={{TARGET_BYTES / 1024**3:.2f}} GiB, "
                f"allocated={{torch.cuda.memory_allocated(device) / 1024**3:.2f}} GiB",
                flush=True,
            )

            for sec in range(HOLD_SECONDS):
                if sec % 5 == 0 or sec == HOLD_SECONDS - 1:
                    allocated = torch.cuda.memory_allocated(device) / 1024**3
                    print(
                        f"[job] holding gpu memory: t={{sec + 1}}/{{HOLD_SECONDS}}s, "
                        f"allocated={{allocated:.2f}} GiB",
                        flush=True,
                    )
                time.sleep(1)

            del tensor
            torch.cuda.empty_cache()

            submission = pd.DataFrame({{"PassengerId": [1], "Survived": [0]}})
            submission_path = Path(__file__).resolve().parent / "submission.csv"
            submission.to_csv(submission_path, index=False)
            print(f"[job] wrote submission to {{submission_path}}", flush=True)
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
        """
    ).strip()


def safe_json(response: httpx.Response) -> dict:
    """安全解析响应 JSON:非 dict 或解析失败都返回空字典。"""
    try:
        data = response.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def build_headers(api_key: str) -> dict:
    """构造带 API 密钥的请求头。"""
    return {"X-API-Key": api_key}


async def get_job_status(
    client: httpx.AsyncClient,
    headers: dict,
    job_id: str,
) -> tuple[int, dict]:
    """查询作业状态,返回 (HTTP 状态码, 数据字典)。"""
    response = await client.get(f"/api/v1/jobs/{job_id}", headers=headers)
    return response.status_code, safe_json(response)


async def submit_long_running_job(
    client: httpx.AsyncClient,
    headers: dict,
    data_dir: str,
    resource_type: str,
    priority: int,
    job_timeout: int,
    code_str: str,
) -> str:
    """提交长作业,返回 job_id(提交失败或缺 job_id 直接抛错)。"""
    # 作业载荷:名称/代码/数据目录/超时/资源/优先级/环境变量
    payload = {
        "name": "cancel-job-gpu-memory-test",
        "code": code_str,
        "data_dir": data_dir,
        "timeout": job_timeout,
        "resource_type": resource_type,
        "priority": priority,
        "environment": {
            "EXECUTION_MODE": "shell",
            "DATA_DIR": data_dir,
            "SANDBOX_DATA_DIR": data_dir,
            "HF_ENDPOINT": "https://hf-mirror.com",
        },
    }
    # 发起提交
    response = await client.post("/api/v1/jobs", json=payload, headers=headers)
    data = safe_json(response)
    if response.status_code >= 400:
        raise RuntimeError(f"submit failed: status={response.status_code}, data={data}")
    # 校验返回的作业 ID
    job_id = data.get("job_id")
    if not job_id:
        raise RuntimeError(f"submit succeeded but job_id missing: data={data}")
    print(f"[submit] job_id={job_id}, data={data}")
    return job_id


async def cancel_job(
    client: httpx.AsyncClient,
    headers: dict,
    job_id: str,
) -> tuple[int, dict]:
    """发起 DELETE 取消请求并打印结果。"""
    response = await client.delete(f"/api/v1/jobs/{job_id}", headers=headers)
    data = safe_json(response)
    print(f"[cancel] status_code={response.status_code}, data={data}")
    return response.status_code, data


async def wait_for_status(
    client: httpx.AsyncClient,
    headers: dict,
    job_id: str,
    *,
    target_statuses: set[str] | None,
    timeout_seconds: int,
    poll_seconds: float,
) -> tuple[int, dict]:
    """轮询作业直到进入目标状态(或任意终态);超时抛 TimeoutError。"""
    # 轮询死线(事件循环时钟)
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    # 上一次状态(只在变化时打印,避免刷屏)
    last_status = None
    while asyncio.get_running_loop().time() < deadline:
        status_code, data = await get_job_status(client, headers, job_id)
        status = str(data.get("status", "")).lower()
        if status != last_status:
            print(f"[poll] job_id={job_id}, http={status_code}, status={status or '<empty>'}")
            last_status = status

        # 命中目标状态:返回
        if target_statuses and status in target_statuses:
            return status_code, data
        # 任意终态:返回(防止空转)
        if status in TERMINAL_STATUSES:
            return status_code, data

        await asyncio.sleep(poll_seconds)

    # 超时仍未等到
    raise TimeoutError(
        f"timed out waiting for job {job_id}, last_status={last_status or '<unknown>'}"
    )


async def run(args: argparse.Namespace) -> int:
    """执行取消流程主逻辑:按需提交 -> 等 running -> 取消 -> 观察终态。返回进程退出码。"""
    # 请求头与客户端超时(总 30s,连接 10s)
    headers = build_headers(args.api_key)
    timeout = httpx.Timeout(30.0, connect=10.0)

    async with httpx.AsyncClient(base_url=args.base_url, timeout=timeout) as client:
        # 指定了 job_id 就只做取消;否则先提交长作业
        job_id = args.job_id

        if not job_id:
            # 生成显存占位代码并提交
            code_str = build_gpu_hold_code(
                vram_gb=args.vram_gb,
                hold_seconds=args.hold_seconds,
            )
            job_id = await submit_long_running_job(
                client=client,
                headers=headers,
                data_dir=args.data_dir,
                resource_type=args.resource_type,
                priority=args.priority,
                job_timeout=args.job_timeout,
                code_str=code_str,
            )

            try:
                # 等作业真正进入 running(占用显存后才取消更有意义)
                await wait_for_status(
                    client,
                    headers,
                    job_id,
                    target_statuses={"running"},
                    timeout_seconds=args.start_timeout,
                    poll_seconds=args.poll_seconds,
                )
                print(f"[flow] job {job_id} entered running, sleep {args.cancel_after}s then cancel")
            except TimeoutError:
                # 没等到 running 也继续尝试取消(排队中同样可取消)
                print(
                    f"[flow] job {job_id} did not enter running within {args.start_timeout}s; "
                    f"will still try to cancel",
                )

            # 取消前的额外等待(默认模拟"运行一段时间再取消")
            await asyncio.sleep(args.cancel_after)

        # 发起取消
        await cancel_job(client, headers, job_id)

        # 取消后持续观察(可选):确认作业确实进入 cancelled
        if args.watch_after_cancel > 0:
            try:
                status_code, data = await wait_for_status(
                    client,
                    headers,
                    job_id,
                    target_statuses={"cancelled", "canceled"},
                    timeout_seconds=args.watch_after_cancel,
                    poll_seconds=args.poll_seconds,
                )
                print(f"[done] job_id={job_id}, http={status_code}, final_status={data.get('status')}")
                # 有结果载荷则一并打印(验证无"取消后仍跑完"的泄漏)
                result = data.get("result")
                if result:
                    print(f"[done] result={result}")
            except TimeoutError as exc:
                # 未在窗口内看到 cancelled:视为取消语义异常,退出码 2
                print(f"[warn] {exc}")
                return 2

        print(f"[done] cancel flow finished for job_id={job_id}")
        return 0


def parse_args() -> argparse.Namespace:
    """解析命令行参数(默认值即脚本顶部的占位常量)。"""
    parser = argparse.ArgumentParser(
        description="Submit or cancel a sandbox job for cancellation testing.",
    )
    parser.add_argument("--base-url", default=BASE_URL, help="Sandbox API base URL")
    parser.add_argument("--api-key", default=API_KEY, help="API key")
    parser.add_argument(
        "--job-id",
        default="",
        help="Existing job_id to cancel. If omitted, the script submits a long-running job first.",
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Dataset path used by the test job")
    parser.add_argument("--resource-type", default="gpu", choices=["cpu", "gpu"], help="Resource type")
    parser.add_argument("--priority", type=int, default=1, choices=[1, 2], help="Job priority")
    parser.add_argument("--job-timeout", type=int, default=300, help="Timeout passed to the job")
    parser.add_argument("--start-timeout", type=int, default=60, help="How long to wait for running status")
    parser.add_argument("--cancel-after", type=int, default=30, help="Seconds to wait before cancelling")
    parser.add_argument(
        "--watch-after-cancel",
        type=int,
        default=60,
        help="Seconds to keep polling after cancellation request",
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0, help="Polling interval in seconds")
    parser.add_argument("--vram-gb", type=float, default=10.0, help="Approximate GPU memory to hold in GiB")
    parser.add_argument("--hold-seconds", type=int, default=120, help="How long the job keeps the GPU memory")
    return parser.parse_args()


def main() -> int:
    """脚本入口:参数解析 -> 异步主流程;Ctrl+C 与异常都映射为退出码。"""
    args = parse_args()
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        # 用户中断:130(与 shell 惯例一致)
        print("[exit] interrupted by user")
        return 130
    except Exception as exc:
        # 其他异常:打印并返回 1
        print(f"[error] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
