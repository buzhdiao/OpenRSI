# 启用延迟注解求值（PEP 563）
from __future__ import annotations

# 导入 asyncio，用于并发调度子任务
import asyncio
# 导入 json，用于读写进度与失败信息文件
import json
# 导入 logging，用于记录运行日志
import logging
# 导入 os，用于复制环境变量
import os
# 导入 sys，用于定位子进程解释器
import sys
# 导入 Path，用于路径操作
from pathlib import Path
# 导入 Any 类型注解
from typing import Any

# 导入 hydra，用于基于配置文件驱动的入口
import hydra
# 导入 yaml，用于读写配置文件
import yaml
# 导入 dotenv，用于从 .env 文件加载环境变量
from dotenv import load_dotenv
# 导入 OmegaConf 配置对象类型
from omegaconf import DictConfig, OmegaConf

# 导入共享并发服务器与每 epoch 任务并发度解析工具
from tts_search.airaevo_concurrency import (
    SharedConcurrencyServer,
    resolve_task_concurrency_per_epoch,
)
# 导入评测结果汇总工具模块
from tts_search import eval_utils
# 导入代理环境变量禁用工具（避免影响子进程网络）
from tts_search.eval_network import disable_proxy_env
# 导入紧凑进度快照追加工具
from tts_search.services.result_persistence import append_compact_progress_snapshot


# 这些终止原因表示任务已彻底完成，重跑时可直接跳过
TERMINAL_SKIP_DONE_REASONS = {
    "accepted_target_reached",
    "step_limit_reached",
    "aira_time_limit_reached",
    "aira_step_limit_reached",
    "aira_num_generations_exhausted",
}


# 安全转 float：转换失败时返回默认值
def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# 安全转 int：转换失败时返回默认值
def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# 读取任务目录下的 config.yaml；缺失时返回空字典
def _load_task_config(task_dir: Path) -> dict[str, Any]:
    # 拼接配置文件路径
    config_path = task_dir / "config.yaml"
    # 文件不存在时返回空配置
    if not config_path.exists():
        return {}
    # 读取并解析 YAML 为字典
    return dict(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})


# 读取任务名：优先取配置中的 task_name，否则用目录名
def _load_task_name(task_dir: Path) -> str:
    # 读取任务配置
    task_cfg = _load_task_config(task_dir)
    # 返回任务名（带兜底）
    return str(task_cfg.get("task_name") or task_dir.name)


# 为旧版（无 done_reason 字段）的任务状态推断终止原因
def _infer_legacy_done_reason(
    *,
    task_state: dict[str, Any],
    output_dir: Path,
    runner_cfg: dict[str, Any],
) -> str | None:
    # 任务未标记完成时直接返回 None
    if not bool(task_state.get("done")):
        return None

    # 已有显式终止原因时原样返回
    done_reason = task_state.get("done_reason")
    if done_reason:
        return str(done_reason)

    # 读取求解器配置
    solver_cfg = dict(runner_cfg.get("solver") or {})
    # 推断接受目标数（多处兜底取值）
    accepted_target = _safe_int(
        task_state.get("accepted_target")
        or task_state.get("target")
        or runner_cfg.get("rejection_target")
        or runner_cfg.get("accepted_target")
        or runner_cfg.get("n_samples_per_task"),
        1,
    )
    # 已接受样本数达到目标：判定为达成接受目标
    if _safe_int(task_state.get("accepted")) >= accepted_target:
        return "accepted_target_reached"

    # 读取请求的步数上限
    requested_step_limit = _safe_int(
        task_state.get("max_target") or solver_cfg.get("step_limit"),
        0,
    )
    # 已完成步数达到上限：判定为步数上限终止
    if requested_step_limit and _safe_int(task_state.get("completed")) >= requested_step_limit:
        return "step_limit_reached"

    # 拼接 AIRA checkpoint 状态文件路径
    checkpoint_path = output_dir / "aira_evo" / "checkpoint" / "state.json"
    # checkpoint 存在时进一步推断
    if checkpoint_path.exists():
        try:
            # 读取 checkpoint 状态 JSON
            checkpoint_state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # 解析失败时按空状态处理
            checkpoint_state = {}

        # 读取时间上限（默认 24 小时）
        time_limit_secs = _safe_float(
            runner_cfg.get("model_plus_sandbox_time_budget")
            or solver_cfg.get("time_limit_secs")
            or 86400.0,
            86400.0,
        )
        # 运行时间达到上限：判定为时间终止
        if _safe_float(checkpoint_state.get("running_time")) >= time_limit_secs:
            return "aira_time_limit_reached"

        # single_task_runner passes requested_step_limit + 1 to AIRA because
        # the AIRA journal contains a root node.
        # AIRA 内部步数上限比请求值大 1（journal 含根节点），推断时同步加 1
        aira_step_limit = _safe_int(solver_cfg.get("step_limit"), requested_step_limit)
        if aira_step_limit:
            aira_step_limit += 1
        # 内部步数达到上限：判定为 AIRA 步数终止
        if aira_step_limit and _safe_int(checkpoint_state.get("current_step")) >= aira_step_limit:
            return "aira_step_limit_reached"

        # 读取代数上限
        num_generations = _safe_int(solver_cfg.get("num_generations"), 0)
        # 内部代数达到上限：判定为代数耗尽
        if (
            num_generations
            and "current_generation" in checkpoint_state
            and _safe_int(checkpoint_state.get("current_generation")) >= num_generations
        ):
            return "aira_num_generations_exhausted"

    # 都无法确定时按“旧版完成但无原因”处理
    return "legacy_done_without_reason"


# 查询某任务在某 epoch 下是否已彻底完成（可跳过）
def _terminal_done_reason_for_task(
    *,
    output_root: Path,
    task_dir: Path,
    epoch_index: int,
    runner_cfg: dict[str, Any],
) -> str | None:
    # 拼接全局进度文件路径
    progress_path = output_root / "progress.json"
    # 进度文件不存在时无法判断
    if not progress_path.exists():
        return None
    try:
        # 读取并解析进度 JSON
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # 解析失败时视为未完成
        return None

    # 读取任务配置
    task_cfg = _load_task_config(task_dir)
    # 任务 id 优先取 uuid，否则用目录名
    task_id = str(task_cfg.get("uuid") or task_dir.name)
    # 取出全部任务状态
    task_states = dict(progress.get("task_states") or {})
    # 查询本任务的状态
    task_state = task_states.get(task_id)
    # 状态缺失或类型不对时视为未完成
    if not isinstance(task_state, dict):
        return None

    # 计算本任务本 epoch 的输出目录
    output_dir = output_root / f"program_ep_{epoch_index}" / task_dir.name
    # 推断（含旧版兜底）终止原因
    done_reason = _infer_legacy_done_reason(
        task_state=task_state,
        output_dir=output_dir,
        runner_cfg=runner_cfg,
    )
    # 只有处于“可跳过”集合中的原因才返回
    if done_reason in TERMINAL_SKIP_DONE_REASONS:
        return done_reason
    # 否则视为需要重跑
    return None


# 任务失败时写出错误信息与占位统计文件
def _write_failed_task_outputs(
    output_dir: Path,
    task_name: str,
    error: str,
    return_code: int | None,
) -> None:
    # 写出错误文本文件
    (output_dir / "runner_error.txt").write_text(error, encoding="utf-8")

    # 统计文件路径
    stat_path = output_dir / "stat.json"
    # 已存在统计文件时不覆盖
    if stat_path.exists():
        return

    # 构造全零/空值的失败统计模板
    failure_stat = {
        "task_name": task_name,
        "steps": [],
        "num_steps": 0,
        "step_limit": 0,
        "num_generations": 0,
        "individuals_per_generation": 0,
        "final_score": None,
        "final_reward": 0.0,
        "final_step": None,
        "submit_score": None,
        "submit_reward": 0.0,
        "submit_grade": None,
        "submit_medal": "N/A",
        "total_cost": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "val_time": 0.0,
        "test_time": 0.0,
        "total_time": 0.0,
        "total_model_time": 0.0,
        "total_sandbox_time": 0.0,
        "total_model_plus_sandbox_time": 0.0,
        "sandbox_12h_score": 0.0,
        "model_plus_sandbox_12h_score": 0.0,
        "status_count": {"unknown": 1},
        "error": error,
        "return_code": return_code,
    }
    # 写出失败统计 JSON
    stat_path.write_text(json.dumps(failure_stat, indent=2), encoding="utf-8")


# 运行单个任务（异步）：拉起子进程执行 single_task_runner.py
async def run_task(
    *,
    task_dir: Path,
    epoch_index: int,
    output_root: Path,
    resolved_runner_config_path: Path,
    runner_cfg: dict[str, Any],
    task_concurrency: asyncio.Semaphore,
    subprocess_env: dict[str, str],
) -> dict[str, Any]:
    # 计算输出目录：output_root / program_ep_<i> / 任务目录名
    output_dir = output_root / f"program_ep_{epoch_index}" / task_dir.name
    # 读取任务名
    task_name = _load_task_name(task_dir)
    # 查询任务是否已彻底完成、可以跳过
    skip_reason = _terminal_done_reason_for_task(
        output_root=output_root,
        task_dir=task_dir,
        epoch_index=epoch_index,
        runner_cfg=runner_cfg,
    )
    # 可跳过时直接返回跳过结果
    if skip_reason is not None:
        logging.info("Skipping completed AIRA Dojo task %s (%s).", task_name, skip_reason)
        return {
            "success": True,
            "skipped": True,
            "skip_reason": skip_reason,
            "task_name": task_name,
            "epoch_index": epoch_index,
            "output_dir": str(output_dir),
            "return_code": 0,
            "error": None,
        }

    # 创建输出目录（含父目录）
    output_dir.mkdir(parents=True, exist_ok=True)

    # 定位单任务运行脚本（与本文件同目录）
    runner_script = Path(__file__).resolve().with_name("single_task_runner.py")
    # 组装子进程命令行
    cmd = [
        sys.executable,
        str(runner_script),
        "--task-dir",
        str(task_dir),
        "--output-dir",
        str(output_dir),
        "--runner-config",
        str(resolved_runner_config_path),
        "--sample-index",
        str(epoch_index),
    ]

    try:
        # 受并发信号量约束地启动子进程并等待退出
        async with task_concurrency:
            process = await asyncio.create_subprocess_exec(*cmd, env=subprocess_env)
            return_code = await process.wait()
    except Exception as exc:
        # 子进程未能正常启动/等待时的异常处理
        error = f"AIRA Dojo task crashed before completion: {task_name}\n{exc}"
        logging.exception(error)
        # 写出失败输出
        _write_failed_task_outputs(
            output_dir=output_dir,
            task_name=task_name,
            error=error,
            return_code=None,
        )
        # 返回失败结果
        return {
            "success": False,
            "task_name": task_name,
            "epoch_index": epoch_index,
            "output_dir": str(output_dir),
            "return_code": None,
            "error": error,
        }

    # 退出码非 0：任务执行失败
    if return_code != 0:
        error = f"AIRA Dojo task failed: {task_name} (return_code={return_code})"
        logging.error(error)
        # 写出失败输出
        _write_failed_task_outputs(
            output_dir=output_dir,
            task_name=task_name,
            error=error,
            return_code=return_code,
        )
        # 返回失败结果
        return {
            "success": False,
            "task_name": task_name,
            "epoch_index": epoch_index,
            "output_dir": str(output_dir),
            "return_code": return_code,
            "error": error,
        }

    # 正常完成：返回成功结果
    return {
        "success": True,
        "task_name": task_name,
        "epoch_index": epoch_index,
        "output_dir": str(output_dir),
        "return_code": return_code,
        "error": None,
    }


# hydra 入口：从当前目录读取 runner_config.yaml
@hydra.main(config_path=".", config_name="runner_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # 配置 INFO 级别日志
    logging.basicConfig(level=logging.INFO)
    # 从 .env 文件加载环境变量
    load_dotenv()
    # 清除代理环境变量，避免影响子进程
    disable_proxy_env()

    # 解析输出根目录并创建
    output_root = Path(cfg.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # 把解析后的完整配置落盘，供子进程读取
    resolved_runner_config_path = output_root / "runner_resolved.yaml"
    resolved_runner_config_path.write_text(
        yaml.safe_dump(
            OmegaConf.to_container(cfg, resolve=True),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    # 再读回为普通字典，便于运行期取值
    runner_cfg = dict(yaml.safe_load(resolved_runner_config_path.read_text(encoding="utf-8")) or {})

    # 解析任务根目录
    task_root = Path(cfg.task_root).resolve()
    # 读取任务列表；未配置时扫描目录下含 config.yaml 的子目录
    task_names = list(cfg.task_list) if cfg.task_list else None
    if task_names is None:
        task_names = sorted(
            [
                path.name
                for path in task_root.iterdir()
                if path.is_dir() and (path / "config.yaml").exists()
            ]
        )

    # 每个任务的采样次数即“epoch 数”
    num_epochs = int(cfg.n_samples_per_task)
    # LLM 全局并发度
    llm_concurrency = int(cfg.llm_concurrency)
    # 沙箱全局并发度
    sandbox_concurrency = int(cfg.sandbox_concurrency)
    # 依据 LLM/沙箱并发度计算每个 epoch 内的任务并发度
    task_concurrency = resolve_task_concurrency_per_epoch(
        configured_task_concurrency=(
            int(cfg.task_concurrency) if cfg.task_concurrency is not None else None
        ),
        llm_concurrency=llm_concurrency,
        sandbox_concurrency=sandbox_concurrency,
        num_epochs=num_epochs,
    )
    # 计算各任务目录的完整路径
    task_dirs = [task_root / name for name in task_names]
    # 为每个 epoch 建立独立的并发信号量
    epoch_semaphores = {
        epoch_index: asyncio.Semaphore(task_concurrency)
        for epoch_index in range(num_epochs)
    }
    # 启动共享并发服务器（LLM/沙箱限流）
    concurrency_server = SharedConcurrencyServer(
        llm_concurrency=llm_concurrency,
        sandbox_concurrency=sandbox_concurrency,
    )
    # 复制当前环境并叠加并发服务器变量，供子进程使用
    subprocess_env = os.environ.copy()
    subprocess_env.update(concurrency_server.env)

    # 定义异步主流程：并发运行所有 (任务, epoch) 组合
    async def run_all() -> list[dict[str, Any]]:
        # 进度历史快照间隔（秒），至少 1 秒
        progress_history_interval = max(
            1.0,
            float(getattr(cfg, "progress_history_interval_seconds", 600.0) or 600.0),
        )
        # 完成事件：用于停止快照循环
        done_event = asyncio.Event()

        # 周期性写入紧凑进度快照的后台循环
        async def progress_history_loop() -> None:
            # 启动时先写一次快照
            append_compact_progress_snapshot(output_root)
            # 未收到完成事件前周期性写快照
            while not done_event.is_set():
                try:
                    # 等待完成事件或超时
                    await asyncio.wait_for(
                        done_event.wait(),
                        timeout=progress_history_interval,
                    )
                except asyncio.TimeoutError:
                    # 超时则继续下一轮
                    pass
                # 未完成时写一次快照
                if not done_event.is_set():
                    append_compact_progress_snapshot(output_root)
            # 结束前补写最终快照
            append_compact_progress_snapshot(output_root)

        # 启动快照后台任务
        progress_task = asyncio.create_task(progress_history_loop())
        # 收集全部 (任务, epoch) 的异步任务
        tasks: list[asyncio.Task[dict[str, Any]]] = []
        try:
            # 双层遍历：任务 × epoch
            for task_dir in task_dirs:
                for epoch_index in range(num_epochs):
                    tasks.append(
                        asyncio.create_task(
                            run_task(
                                task_dir=task_dir,
                                epoch_index=epoch_index,
                                output_root=output_root,
                                resolved_runner_config_path=resolved_runner_config_path,
                                runner_cfg=runner_cfg,
                                task_concurrency=epoch_semaphores[epoch_index],
                                subprocess_env=subprocess_env,
                            )
                        )
                    )
            # 并发等待全部任务完成
            return await asyncio.gather(*tasks)
        finally:
            # 无论成败都停止快照循环并等待其退出
            done_event.set()
            await progress_task

    try:
        # 运行异步主流程
        task_results = asyncio.run(run_all())
    finally:
        # 关闭共享并发服务器
        concurrency_server.shutdown()

    # 收集失败的任务结果
    failures = [result for result in task_results if not result["success"]]
    # 写出失败清单 JSON
    (output_root / "runner_failures.json").write_text(
        json.dumps(failures, indent=2),
        encoding="utf-8",
    )
    # 存在失败时打印警告
    if failures:
        logging.warning("AIRA Dojo runner completed with %d failed tasks.", len(failures))

    # 构建 任务名 -> 任务配置 的映射
    task_metadata_map: dict[str, dict[str, Any]] = {}
    for task_dir in task_dirs:
        # 读取任务配置
        task_cfg = yaml.safe_load((task_dir / "config.yaml").read_text(encoding="utf-8"))
        # 以任务名为键登记
        task_metadata_map[str(task_cfg["task_name"])] = task_cfg

    # 为每个 epoch 汇总单 epoch 统计
    for epoch_index in range(num_epochs):
        eval_utils.write_epoch_stat(output_root / f"program_ep_{epoch_index}")

    # 汇总写出总结 CSV
    eval_utils.write_summary_csv(output_root, task_metadata_map)
    # 汇总写出全局统计
    eval_utils.write_global_stat(output_root)

    # 写出运行清单（实验元信息）
    (output_root / "runner_manifest.json").write_text(
        json.dumps(
            {
                "experiment_name": cfg.experiment_name,
                "task_root": str(task_root),
                "task_names": task_names,
                "n_samples_per_task": num_epochs,
                "llm_concurrency": llm_concurrency,
                "sandbox_concurrency": sandbox_concurrency,
                "task_concurrency_per_epoch": task_concurrency,
                "max_concurrent_tasks": num_epochs * task_concurrency,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # 存在失败时以非零码退出
    if failures:
        raise SystemExit(1)


# 脚本入口
if __name__ == "__main__":
    main()
