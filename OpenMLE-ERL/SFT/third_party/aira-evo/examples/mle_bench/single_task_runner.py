#!/usr/bin/env python3
# 启用延迟注解求值（PEP 563）
from __future__ import annotations

# 导入 argparse，用于解析命令行参数
import argparse
# 导入 json，用于读写状态与统计文件
import json
# 导入 logging，用于记录运行日志
import logging
# 导入 os，用于环境变量设置
import os
# 导入 random，用于设置随机种子
import random
# 导入 sys，用于调整模块搜索路径
import sys
# 导入 time，用于搜索时间预算判断
import time
# 导入 asdict，用于把 dataclass 配置转为字典
from dataclasses import asdict
# 导入 datetime，用于记录时间戳
from datetime import datetime
# 导入 Path，用于路径操作
from pathlib import Path
# 导入 SimpleNamespace，用于轻量命名空间包装
from types import SimpleNamespace
# 导入 Any 类型注解
from typing import Any

# 导入 numpy，用于设置随机种子
import numpy as np
# 导入 yaml，用于读取配置文件
import yaml
# 导入 dotenv，用于从 .env 文件加载环境变量
from dotenv import load_dotenv


# 搜索预算耗尽时抛出的自定义异常，用于中断求解过程
class StopSearch(RuntimeError):
    pass


# 读取 YAML 配置并转为字典
def _load_yaml(path: Path) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


# 去掉配置负载中的 hydra _target_ 键（构造配置类时不需要）
def _strip_target(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "_target_"}


# 合并两份生成参数（generation_kwargs）；extra_body 键做深合并而非覆盖
def _merge_generation_kwargs(
    base_kwargs: dict[str, Any],
    override_kwargs: dict[str, Any],
) -> dict[str, Any]:
    # 以基础参数为底本拷贝
    merged = dict(base_kwargs)
    # 拷贝覆盖参数
    override_payload = dict(override_kwargs)
    # 双方都含 extra_body 时按字典合并，保留两侧字段
    if "extra_body" in base_kwargs and "extra_body" in override_kwargs:
        override_payload["extra_body"] = {
            **dict(base_kwargs["extra_body"] or {}),
            **dict(override_kwargs["extra_body"] or {}),
        }
    # 用覆盖参数更新合并结果
    merged.update(override_payload)
    # 返回合并后的生成参数
    return merged


# 提取消息的文本内容；content 为分块列表时拼接其中的 text 字段
def _prompt_content(message: dict[str, Any]) -> str:
    # 读取 content 字段
    content = message.get("content")
    # 分块列表形式：逐块提取 text 并换行拼接
    if isinstance(content, list):
        return "\n".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
    # 字符串形式：原样返回（None 兜底为空串）
    return str(content or "")


# 从算子指标中提取系统提示词、用户提示词与补全文
def _extract_prompt_messages(
    operator_metric: dict[str, Any] | None,
) -> tuple[str, str, str]:
    # 无指标时返回空三元组
    if not operator_metric:
        return "", "", ""

    # 取出提示词消息列表
    prompt_messages = list(operator_metric.get("prompt_messages") or [])
    # 取出补全文本
    completion_text = str(operator_metric.get("completion_text") or "")
    # 系统提示词缓存
    system_prompt = ""
    # 用户提示词缓存
    user_prompt = ""
    # 遍历消息，分别捕获第一条 system 与最后一条 user
    for message in prompt_messages:
        # 尚未有系统提示词且当前是 system 角色时记录
        if not system_prompt and str(message.get("role")) == "system":
            system_prompt = _prompt_content(message)
        # 当前是 user 角色时记录（不断覆盖，保留最后一条）
        if str(message.get("role")) == "user":
            user_prompt = _prompt_content(message)
    # 返回（系统提示词, 用户提示词, 补全文）
    return system_prompt, user_prompt, completion_text


# 从算子指标中提取响应文本、思考内容、用量与原始补全文
def _extract_completion_artifacts(
    operator_metric: dict[str, Any] | None,
    parse_thinking_tags_fn: Any,
) -> tuple[str, str, dict[str, Any], str]:
    # 无指标时返回空值组合
    if not operator_metric:
        return "", "", {}, ""

    # 读取原始补全文本
    completion_text = str(operator_metric.get("completion_text") or "")
    # 读取 token 用量字典
    usage = dict(operator_metric.get("usage") or {})
    # 解析 <think> 标签：得到思考内容与去除标签后的正文
    parsed_reasoning, text_without_thinking = parse_thinking_tags_fn(completion_text)
    # 用量中可能直接携带思考内容字段（多个键名兼容）
    field_reasoning = (
        usage.get("reasoning_content")
        or usage.get("reasoning")
        or usage.get("thinking")
    )
    # 思考内容优先取用量字段，其次取标签解析结果
    reasoning_content = str(field_reasoning or parsed_reasoning or "")
    # 解析出思考内容时返回去除标签后的正文，否则返回原文
    response_text = text_without_thinking if parsed_reasoning else completion_text
    # 返回（正文, 思考内容, 用量, 原始补全文）
    return response_text, reasoning_content, usage, completion_text


# 汇总所有算子调用的用量（费用/token 数/延迟）
def _sum_operator_usage(operator_metrics: list[dict[str, Any]]) -> dict[str, float]:
    # 各项用量累计值初始化为 0
    totals = {
        "cost": 0.0,
        "prompt_tokens": 0.0,
        "completion_tokens": 0.0,
        "total_tokens": 0.0,
        "latency": 0.0,
    }
    # 遍历每个算子调用记录
    for operator_metric in operator_metrics:
        # 取出该次调用的用量字典
        usage = dict(operator_metric.get("usage") or {})
        # 累加费用
        totals["cost"] += float(usage.get("cost") or 0.0)
        # 累加提示词 token 数
        totals["prompt_tokens"] += float(usage.get("prompt_tokens") or 0.0)
        # 累加补全 token 数
        totals["completion_tokens"] += float(usage.get("completion_tokens") or 0.0)
        # 累加总 token 数
        totals["total_tokens"] += float(usage.get("total_tokens") or 0.0)
        # 累加延迟
        totals["latency"] += float(usage.get("latency") or 0.0)
    # 返回累计用量
    return totals


# 选取当前最优可用节点：优先 journal 的最佳节点，否则回退最后一个非根节点
def _select_best_available_node(solver: Any) -> Any | None:
    # 尝试获取 journal 中的最佳节点
    best_node = solver.journal.get_best_node()
    # 存在则直接返回
    if best_node is not None:
        return best_node

    # 无最佳节点时从后往前找第一个非根节点
    for node in reversed(solver.journal.nodes):
        # 跳过根节点
        if not solver.journal.is_root_node(node):
            return node
    # 全部为空时返回 None
    return None


# 生成文件系统安全的任务输出目录名（任务名 + 任务 id）
def _safe_task_output_name(task_name: str, task_id: str) -> str:
    # 把路径分隔符替换为下划线
    clean = task_name.replace("/", "_").replace("\\", "_")
    # 有任务 id 时拼接 id，否则只返回任务名
    return f"{clean}_{task_id}" if task_id else clean


# 建立兼容目录：优先软链接，失败时退化为完整拷贝
def _sync_task_link(src: Path, dst: Path) -> None:
    # 确保父目录存在
    dst.parent.mkdir(parents=True, exist_ok=True)
    # 目标已存在（含悬挂软链）时无需处理
    if dst.exists() or dst.is_symlink():
        return
    try:
        # 创建指向源目录的软链接
        dst.symlink_to(src, target_is_directory=True)
    except OSError:
        # 不支持软链（如某些文件系统）时递归拷贝
        import shutil

        shutil.copytree(src, dst, dirs_exist_ok=True)


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


# 单任务主流程
def main() -> None:
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Run a single AIRA Dojo task")
    # 任务目录（必填）
    parser.add_argument("--task-dir", required=True)
    # 输出目录（必填）
    parser.add_argument("--output-dir", required=True)
    # 运行配置文件路径（必填）
    parser.add_argument("--runner-config", required=True)
    # 采样索引（即 epoch 序号，必填）
    parser.add_argument("--sample-index", required=True, type=int)
    # 解析参数
    args = parser.parse_args()

    # 从 .env 文件加载环境变量
    load_dotenv()
    # 配置 INFO 级别日志
    logging.basicConfig(level=logging.INFO)

    # 计算仓库根目录（本文件向上 4 级）
    repo_root = Path(__file__).resolve().parents[4]
    # 本示例目录（与脚本同目录）
    example_dir = Path(__file__).resolve().parent
    # 解析任务目录绝对路径
    task_dir = Path(args.task_dir).resolve()
    # 解析输出目录绝对路径
    output_dir = Path(args.output_dir).resolve()
    # 解析运行配置文件绝对路径
    runner_config_path = Path(args.runner_config).resolve()
    # AIRA 演化工作子目录
    aira_evo_dir = output_dir / "aira_evo"
    # 创建该子目录
    aira_evo_dir.mkdir(parents=True, exist_ok=True)

    # 设置 AIRA 日志目录环境变量（缺省时）
    os.environ.setdefault("LOGGING_DIR", str(aira_evo_dir))

    # dojo 源码目录（向上 2 级的 src）
    dojo_src_dir = Path(__file__).resolve().parents[2] / "src"
    # 把 dojo 源码目录加入 sys.path
    if str(dojo_src_dir) not in sys.path:
        sys.path.insert(0, str(dojo_src_dir))
    # 把示例目录加入 sys.path（便于导入 base_task）
    if str(example_dir) not in sys.path:
        sys.path.insert(0, str(example_dir))

    # 导入本示例的任务实现与 dojo 各配置类、求解器等（依赖 sys.path，故延迟导入）
    from base_task import SandboxMLEBenchTask
    from dojo.config_dataclasses.client.base import ClientConfig
    from dojo.config_dataclasses.interpreter.python import PythonInterpreterConfig
    from dojo.config_dataclasses.llm.generic_llm import GenericLLMConfig
    from dojo.config_dataclasses.llm.jinjaprompt import JinjaPromptConfig
    from dojo.config_dataclasses.logger import LoggerConfig
    from dojo.config_dataclasses.operators.base import OperatorConfig
    from dojo.config_dataclasses.operators.memory import MemoryOpConfig
    from dojo.config_dataclasses.solver.evo import EvolutionarySolverConfig
    from dojo.core.interpreters.python import PythonInterpreter
    from dojo.core.solvers.utils.response import parse_thinking_tags
    from dojo.core.solvers.utils.search_exporter import export_search_results
    from dojo.solvers.evo.evo import Evolutionary
    from dojo.utils.logger import config_logger
    from tts_search import eval_utils
    from tts_search.data_produce.token_filter import (
        count_chat_template_tokens,
        load_tokenizer,
    )
    from tts_search.services.result_persistence import (
        append_jsonl_record,
        update_progress_snapshot,
    )
    from tts_search.services.rejection import build_rejection_policy
    from tts_search.services.tree_search_state import (
        NODE_EVALUATED,
        NODE_GENERATED,
        SearchEvent,
        SearchState,
        append_search_event,
        atomic_write_search_state,
        read_search_events,
        replay_search_events,
    )

    # 加载运行配置
    runner_cfg = _load_yaml(runner_config_path)
    # 加载任务配置
    task_cfg = _load_yaml(task_dir / "config.yaml")
    # 读取任务名（缺省用目录名）
    task_name = str(task_cfg.get("task_name") or task_dir.name)
    # 读取任务 id（缺省用目录名）
    task_id = str(task_cfg.get("uuid") or task_dir.name)
    # 推断运行根目录：program_ep_0 时上两级，否则上一级
    run_dir = output_dir.parent.parent if output_dir.parent.name == "program_ep_0" else output_dir.parent
    # 计算兼容模式下的任务目录名
    compat_task_dir = run_dir / "tasks" / _safe_task_output_name(task_name, task_id)
    # 建立软链接/拷贝，保持与旧版目录结构兼容
    _sync_task_link(output_dir, compat_task_dir)
    # 读取运行配置中的任务总数
    total_runner_tasks = len(runner_cfg.get("task_list") or [])
    # 未配置任务列表时按输出目录的兄弟目录数估算
    if total_runner_tasks <= 0:
        try:
            total_runner_tasks = sum(1 for p in output_dir.parent.iterdir() if p.is_dir())
        except OSError:
            # 目录不可读时按 1 个任务处理
            total_runner_tasks = 1

    # 运行配置指定了排行榜目录时覆盖任务配置
    if runner_cfg.get("leaderboard_dir"):
        task_cfg["leaderboard_dir"] = runner_cfg["leaderboard_dir"]

    # 计算随机种子：基础种子 + 采样索引，保证不同 epoch 可复现且不同
    seed = int(runner_cfg.get("seed", 42)) + int(args.sample_index)
    # 设置 Python 随机种子
    random.seed(seed)
    # 设置 numpy 随机种子
    np.random.seed(seed)

    # 构造 dojo 日志配置（输出目录/控制台/wandb/json 开关）
    logger_cfg = LoggerConfig(
        output_dir=str(aira_evo_dir),
        use_console=bool(runner_cfg.get("logger", {}).get("use_console", True)),
        use_wandb=bool(runner_cfg.get("logger", {}).get("use_wandb", False)),
        use_json=bool(runner_cfg.get("logger", {}).get("use_json", True)),
        print_config=False,
        write_env_vars=False,
    )
    # 初始化 dojo logger
    logger = config_logger(SimpleNamespace(logger=logger_cfg))

    # 实例化 MLE-Bench 沙箱任务（携带时间预算）
    task = SandboxMLEBenchTask(
        task_cfg,
        time_budget=runner_cfg.get("time_budget"),
    )

    # 各步统计记录列表
    step_stats: list[dict[str, Any]] = []
    # 已写出统计的步号集合
    written_steps: set[int] = set()
    # 搜索事件日志路径
    search_event_log_path = output_dir / "search_events.jsonl"
    # 搜索状态快照路径
    search_state_path = output_dir / "search_state.json"
    # 重放历史搜索事件，恢复累计计数
    replay_summary = replay_search_events(read_search_events(search_event_log_path))
    # 已接受样本数
    accepted_count = int(replay_summary.accepted_count)
    # 已生成样本数
    generated_count = int(replay_summary.generated_count)
    # 已完成（写出）样本数
    completed_count = int(replay_summary.completed_count)
    # 读取接受目标数（多个配置键兜底）
    accepted_target = int(
        runner_cfg.get("rejection_target")
        or runner_cfg.get("accepted_target")
        or runner_cfg.get("n_samples_per_task", 1)
    )
    # 读取拒绝策略名（默认 accept_scored）
    rejection_policy_name = str(runner_cfg.get("rejection_policy") or "accept_scored")
    # 按配置构建拒绝（样本筛选）策略
    rejection_policy = build_rejection_policy(
        name=rejection_policy_name,
        score_threshold=runner_cfg.get("rejection_score_threshold"),
        reward_threshold=runner_cfg.get("rejection_reward_threshold"),
        reference_scores_path=runner_cfg.get("rejection_reference_scores_path"),
        accepted_medals=runner_cfg.get("rejection_accepted_medals") or [],
        apply_baseline_filters=bool(
            runner_cfg.get("rejection_apply_baseline_filters", True)
        ),
        baseline_token_limit=int(runner_cfg.get("rejection_baseline_token_limit", 32768)),
        baseline_relative_gap_limit=float(
            runner_cfg.get("rejection_baseline_relative_gap_limit", 0.12)
        ),
        mixed_leaderboard_target=int(
            runner_cfg.get("rejection_mixed_leaderboard_target", 2)
        ),
        mixed_no_leaderboard_target=int(
            runner_cfg.get("rejection_mixed_no_leaderboard_target", 4)
        ),
    )
    # 基线 tokenizer 缓存（惰性加载）
    baseline_tokenizer: Any | None = None

    # 构建 SFT 样本的 assistant 回复内容（思考块 + 代码块/正文）
    def build_sft_assistant_content(
        *,
        reasoning_content: str,
        code: str,
        response_text: str,
    ) -> str:
        # 取思考内容文本
        reasoning = str(reasoning_content or "")
        # 有思考内容时
        if reasoning.strip():
            # 同时有代码：思考块 + python 代码块
            if code.strip():
                return (
                    f"<think>\n{reasoning.strip()}\n</think>\n\n"
                    f"```python\n{code.rstrip()}\n```"
                )
            # 无代码：思考块 + 正文
            return f"<think>\n{reasoning.strip()}\n</think>\n\n{response_text.strip()}"
        # 无思考内容但有代码：仅 python 代码块
        if code.strip():
            return f"```python\n{code.rstrip()}\n```"
        # 都没有：直接返回正文
        return response_text.strip()

    # 按 slime 的对话模板统计消息 token 数
    def count_slime_message_tokens(
        *,
        system_prompt: str,
        user_prompt: str,
        response_text: str,
        reasoning_content: str,
        code: str,
    ) -> tuple[int | None, str | None]:
        # 引用外层 tokenizer 缓存
        nonlocal baseline_tokenizer
        # 未配置基线 tokenizer 模型时返回空
        tokenizer_model = runner_cfg.get("rejection_baseline_tokenizer_model")
        if not tokenizer_model:
            return None, None
        # 首次使用时惰性加载 tokenizer（仅本地文件）
        if baseline_tokenizer is None:
            baseline_tokenizer = load_tokenizer(
                str(tokenizer_model),
                local_files_only=True,
            )
        # 组装 system/user/assistant 三条消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {
                "role": "assistant",
                "content": build_sft_assistant_content(
                    reasoning_content=reasoning_content,
                    code=code,
                    response_text=response_text,
                ),
            },
        ]
        # 用聊天模板统计 token 数
        token_count = count_chat_template_tokens(messages, baseline_tokenizer)
        # 返回（token 数, tokenizer 模型名）
        return int(token_count), str(tokenizer_model)
    # 已有任务统计文件路径
    existing_task_stat_path = output_dir / "stat.json"
    # 存在历史统计（断点续跑）时恢复
    if existing_task_stat_path.exists():
        # 读取历史统计 JSON
        existing_task_stat = json.loads(
            existing_task_stat_path.read_text(encoding="utf-8")
        )
        # 恢复各步统计
        step_stats = list(existing_task_stat.get("steps") or [])
        # 恢复已写出的步号集合
        written_steps = {int(step["step"]) for step in step_stats}
        # 恢复已用验证时间
        task.validation_time_used = float(existing_task_stat.get("val_time") or 0.0)

    # 构造 Python 解释器配置（工作目录/超时/软链开关）
    interpreter_cfg = PythonInterpreterConfig(
        working_dir=str(aira_evo_dir / "workspace"),
        timeout=int(runner_cfg["solver"]["execution_timeout"]),
        use_symlinks=bool(runner_cfg.get("interpreter", {}).get("use_symlinks", True)),
    )
    # 创建求解器侧解释器（绑定任务数据目录）
    solver_interpreter = PythonInterpreter(
        interpreter_cfg,
        data_dir=Path(task_cfg["data_dir"]),
    )

    # 构造 LLM 客户端配置（api/模型/base_url/azure/provider）
    client_cfg = ClientConfig(
        api=str(runner_cfg["llm"].get("api", "litellm")),
        model_id=str(runner_cfg["llm"]["model_id"]),
        base_url=str(runner_cfg["llm"]["base_url"]),
        use_azure_client=bool(runner_cfg["llm"].get("use_azure_client", False)),
        provider=str(runner_cfg["llm"].get("provider", "selfhosted")),
    )

    # dojo 配置根目录
    configs_root = dojo_src_dir / "dojo" / "configs"
    # 求解器配置子目录
    solver_root = configs_root / "solver"

    # 由 YAML 负载构造 Jinja 提示词配置
    def build_prompt_config(prompt_payload: dict[str, Any]) -> JinjaPromptConfig:
        return JinjaPromptConfig(**_strip_target(dict(prompt_payload)))

    # 负载为空时返回默认提示词配置
    def maybe_build_prompt_config(prompt_payload: dict[str, Any] | None) -> JinjaPromptConfig:
        # 空负载用默认配置
        if prompt_payload is None:
            return JinjaPromptConfig()
        # 否则正常构造
        return build_prompt_config(prompt_payload)

    # 构造算子配置：合并多层生成参数覆盖
    def build_operator_config(name: str, relative_path: str) -> OperatorConfig:
        # 加载算子 YAML 定义
        operator_yaml = _load_yaml(solver_root / "operators" / relative_path)
        # 取出该算子的配置负载
        operator_payload = dict(operator_yaml[name])
        # 取出算子内嵌的 LLM 配置并去掉 _target_
        llm_payload = _strip_target(dict(operator_payload["llm"]))
        # 以运行配置的全局生成参数为底
        generation_kwargs = dict(runner_cfg["llm"].get("generation_kwargs") or {})
        # 叠加算子 YAML 中的生成参数
        generation_kwargs = _merge_generation_kwargs(
            generation_kwargs,
            dict(llm_payload.get("generation_kwargs") or {}),
        )

        # 读取运行配置中针对该算子的覆盖
        solver_operator_overrides = (
            runner_cfg.get("solver", {}).get("operators", {}).get(name, {})
        )
        # 再叠加求解器默认配置中该算子的生成参数
        generation_kwargs = _merge_generation_kwargs(
            generation_kwargs,
            dict(
                solver_defaults["operators"].get(name, {})
                .get("llm", {})
                .get("generation_kwargs", {})
            ),
        )
        # 最后叠加运行配置中该算子的生成参数（优先级最高）
        generation_kwargs = _merge_generation_kwargs(
            generation_kwargs,
            dict(
                solver_operator_overrides.get("llm", {}).get("generation_kwargs", {})
            ),
        )

        # 构造算子配置对象（LLM + 三类提示词模板）
        return OperatorConfig(
            llm=GenericLLMConfig(
                client=client_cfg,
                generation_kwargs=generation_kwargs,
            ),
            system_message_prompt_template=maybe_build_prompt_config(
                operator_payload.get("system_message_prompt_template")
            ),
            init_user_message_prompt_template=maybe_build_prompt_config(
                operator_payload.get("init_user_message_prompt_template")
            ),
            user_message_prompt_template=maybe_build_prompt_config(
                operator_payload.get("user_message_prompt_template")
            ),
        )

    # 加载 evo 求解器基础默认配置
    base_solver_defaults = _load_yaml(solver_root / "evo.yaml")
    # 加载 mlebench 场景的求解器默认配置
    solver_defaults = _load_yaml(solver_root / "mlebench" / "evo.yaml")
    # 运行配置中针对求解器的覆盖项
    solver_cfg_overrides = dict(runner_cfg.get("solver") or {})

    # 构建五个演化算子的配置（草拟/改进/调试/分析/交叉）
    operators = {
        "draft": build_operator_config("draft", "mlebench/aira_operators/draft.yaml"),
        "improve": build_operator_config("improve", "mlebench/aira_operators/improve.yaml"),
        "debug": build_operator_config("debug", "mlebench/aira_operators/debug.yaml"),
        "analyze": build_operator_config("analyze", "mlebench/aide_operators/analyze.yaml"),
        "crossover": build_operator_config(
            "crossover",
            "mlebench/aira_operators/crossover.yaml",
        ),
    }

    # 构造记忆操作配置（普通记忆）
    memory_cfg = MemoryOpConfig(
        **_strip_target(_load_yaml(solver_root / "memory" / "simple_memory.yaml"))
    )
    # 构造调试记忆配置
    debug_memory_cfg = MemoryOpConfig(
        **_strip_target(_load_yaml(solver_root / "memory" / "debug_memory.yaml"))
    )

    # 计算时间上限（模型+沙箱预算优先，其次求解器覆盖/基础默认）
    time_limit_secs = int(
        runner_cfg.get("model_plus_sandbox_time_budget")
        or solver_cfg_overrides.get("time_limit_secs")
        or base_solver_defaults.get("time_limit_secs")
    )
    # 读取请求的步数上限
    requested_step_limit = int(
        solver_cfg_overrides.get("step_limit", solver_defaults["step_limit"])
    )
    # 构造演化求解器配置（下面一批字段均为 覆盖值优先、默认值兜底）
    solver_cfg = EvolutionarySolverConfig(
        # AIRA 内部步数上限比请求值大 1（journal 含根节点）
        step_limit=requested_step_limit + 1,
        # 可用软件包列表
        available_packages=list(
            solver_cfg_overrides.get(
                "available_packages",
                base_solver_defaults["available_packages"],
            )
        ),
        # 算子配置集合
        operators=operators,
        # 普通记忆配置
        memory=memory_cfg,
        # 调试记忆配置
        debug_memory=debug_memory_cfg,
        # 实验名（清洗路径分隔符）
        exp_name=str(task_cfg["task_name"]).replace("/", "_").replace("\\", "_"),
        # 代码执行超时
        execution_timeout=int(
            solver_cfg_overrides.get(
                "execution_timeout",
                base_solver_defaults["execution_timeout"],
            )
        ),
        # 搜索时间上限
        time_limit_secs=time_limit_secs,
        # 是否导出搜索结果
        export_search_results=bool(
            solver_cfg_overrides.get(
                "export_search_results",
                base_solver_defaults.get("export_search_results", True),
            )
        ),
        # checkpoint 保存路径
        checkpoint_path=str(aira_evo_dir / "checkpoint"),
        # 是否使用测试分数
        use_test_score=bool(
            solver_cfg_overrides.get(
                "use_test_score",
                solver_defaults.get("use_test_score", False),
            )
        ),
        # 是否考虑解复杂度
        use_complexity=bool(
            solver_cfg_overrides.get(
                "use_complexity",
                solver_defaults.get("use_complexity", False),
            )
        ),
        # LLM 调用最大重试次数
        max_llm_call_retries=int(
            solver_cfg_overrides.get(
                "max_llm_call_retries",
                base_solver_defaults.get("max_llm_call_retries", 3),
            )
        ),
        # 岛屿数量（岛屿模型演化）
        num_islands=int(solver_cfg_overrides.get("num_islands", solver_defaults["num_islands"])),
        # 每岛最大个体数
        max_island_size=int(
            solver_cfg_overrides.get(
                "max_island_size",
                solver_defaults["max_island_size"],
            )
        ),
        # 交叉概率
        crossover_prob=float(
            solver_cfg_overrides.get(
                "crossover_prob",
                solver_defaults["crossover_prob"],
            )
        ),
        # 迁移概率
        migration_prob=float(
            solver_cfg_overrides.get(
                "migration_prob",
                solver_defaults["migration_prob"],
            )
        ),
        # 初始温度
        initial_temp=float(
            solver_cfg_overrides.get(
                "initial_temp",
                solver_defaults["initial_temp"],
            )
        ),
        # 最终温度
        final_temp=float(
            solver_cfg_overrides.get(
                "final_temp",
                solver_defaults["final_temp"],
            )
        ),
        # 距下次迁移的代数
        num_generations_till_migration=int(
            solver_cfg_overrides.get(
                "num_generations_till_migration",
                solver_defaults["num_generations_till_migration"],
            )
        ),
        # 距下次交叉的代数
        num_generations_till_crossover=int(
            solver_cfg_overrides.get(
                "num_generations_till_crossover",
                solver_defaults["num_generations_till_crossover"],
            )
        ),
        # few-shot 示例配置
        few_shot=dict(
            solver_cfg_overrides.get(
                "few_shot",
                base_solver_defaults["few_shot"],
            )
        ),
        # 总代数上限
        num_generations=int(
            solver_cfg_overrides.get(
                "num_generations",
                solver_defaults["num_generations"],
            )
        ),
        # 每代个体数
        individuals_per_generation=int(
            solver_cfg_overrides.get(
                "individuals_per_generation",
                solver_defaults["individuals_per_generation"],
            )
        ),
        # 单次调试最长时间
        max_debug_time=float(
            solver_cfg_overrides.get(
                "max_debug_time",
                solver_defaults["max_debug_time"],
            )
        ),
        # 最大调试深度
        max_debug_depth=int(
            solver_cfg_overrides.get(
                "max_debug_depth",
                solver_defaults["max_debug_depth"],
            )
        ),
        # 是否展示数据预览
        data_preview=bool(
            solver_cfg_overrides.get(
                "data_preview",
                solver_defaults["data_preview"],
            )
        ),
    )

    # 汇总 dojo 配置快照（含种子/任务/各子配置）
    dojo_config_payload = {
        "seed": seed,
        "sample_index": int(args.sample_index),
        "task": task_cfg,
        "interpreter": asdict(interpreter_cfg),
        "logger": asdict(logger_cfg),
        "solver": asdict(solver_cfg),
    }
    # 落盘 dojo 配置 JSON，便于复盘
    (aira_evo_dir / "dojo_config.json").write_text(
        json.dumps(dojo_config_payload, indent=2),
        encoding="utf-8",
    )

    # 准备任务：得到初始状态与任务信息
    state, task_info = task.prepare(
        solver_interpreter=solver_interpreter,
        eval_interpreter=None,
    )
    # 创建演化求解器
    solver = Evolutionary(solver_cfg, task_info=task_info)
    # 加载已有 checkpoint（断点续跑）
    solver.load_checkpoint()
    # 从历史统计中收集已完成步对应的代数 id
    completed_generation_ids = [
        int(step["generation_id"])
        for step in step_stats
        if step.get("generation_id") is not None
    ]
    # 存在历史代数且求解器仍停在 0 代时，恢复到最大代数+1
    if completed_generation_ids and solver.state.current_generation == 0:
        solver.state.current_generation = max(completed_generation_ids) + 1

    # 汇总写出任务级统计文件 stat.json
    def write_task_stat(
        *,
        best_node: Any | None = None,
        submit_result: dict[str, Any] | None = None,
        test_time: float = 0.0,
    ) -> None:
        # 最终分数/奖励/步号初始化为空
        final_score = None
        final_reward = None
        final_step = None
        # 有最佳节点且带指标时提取
        if best_node is not None and getattr(best_node, "metric", None) is not None:
            # 取出指标附加信息
            aux_info = dict(best_node.metric.info or {})
            # 最终分数优先取 aux 中的 score，否则用指标值
            final_score = aux_info.get("score", best_node.metric.value)
            # 最终奖励
            final_reward = aux_info.get("reward")
            # 最终步号（journal 步号从 1 开始，转 0 起）
            final_step = int(best_node.step) - 1 if best_node.step is not None else None

        # 提交分数与奖励
        submit_score = submit_result.get("score") if submit_result else None
        submit_reward = submit_result.get("reward") if submit_result else None

        # 累计费用
        total_cost = sum(float(step["cost"]) for step in step_stats)
        # 累计提示词 token
        total_prompt_tokens = sum(int(step["prompt_tokens"]) for step in step_stats)
        # 累计补全 token
        total_completion_tokens = sum(
            int(step["completion_tokens"]) for step in step_stats
        )
        # 累计总 token
        total_tokens = sum(int(step["total_tokens"]) for step in step_stats)
        # 累计模型耗时
        total_model_time = sum(float(step["model_time_used"]) for step in step_stats)
        # 沙箱耗时 = 验证累计 + 测试耗时
        total_sandbox_time = float(task.validation_time_used) + test_time
        # 模型+沙箱总耗时
        total_model_plus_sandbox_time = total_model_time + total_sandbox_time

        # 状态计数表
        status_count: dict[str, int] = {}
        # 遍历各步统计
        for step in step_stats:
            # 读取状态字符串
            status = str(step["status"])
            # 计数累加
            status_count[status] = status_count.get(status, 0) + 1

        # 提交等级与奖牌默认为空/N/A
        submit_grade = None
        submit_medal = "N/A"
        # 有提交分数时对照排行榜评级
        if submit_score is not None:
            # 加载排行榜
            leaderboard = eval_utils.load_leaderboard(task_cfg)
            # 计算等级与奖牌
            submit_grade, submit_medal = eval_utils.build_submit_grade_and_medal(
                submit_score,
                leaderboard,
            )

        # 组装任务统计负载
        task_stat_payload = {
            "task_name": task_cfg["task_name"],
            "val_data_dir": task.validation_data_dir,
            "submit_data_dir": task.submit_data_dir,
            "steps": step_stats,
            "num_steps": len(step_stats),
            "step_limit": requested_step_limit,
            "num_generations": int(solver_cfg.num_generations),
            "individuals_per_generation": int(solver_cfg.individuals_per_generation),
            "final_score": final_score,
            "final_reward": float(final_reward) if final_reward is not None else None,
            "final_step": final_step,
            "submit_score": submit_score,
            "submit_reward": float(submit_reward) if submit_reward is not None else None,
            "submit_grade": submit_grade,
            "submit_medal": submit_medal,
            "total_cost": total_cost,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "val_time": float(task.validation_time_used),
            "test_time": test_time,
            "total_time": float(task.validation_time_used) + test_time,
            "total_model_time": total_model_time,
            "total_sandbox_time": total_sandbox_time,
            "total_model_plus_sandbox_time": total_model_plus_sandbox_time,
            # 沙箱时间折算的 12 小时得分（按基准秒数缩放）
            "sandbox_12h_score": total_sandbox_time
            / eval_utils.TIME_SCALING_SCORE_SECONDS,
            "model_plus_sandbox_12h_score": total_model_plus_sandbox_time
            / eval_utils.TIME_SCALING_SCORE_SECONDS,
            "status_count": status_count,
        }
        # 落盘 stat.json
        (output_dir / "stat.json").write_text(
            json.dumps(task_stat_payload, indent=2),
            encoding="utf-8",
        )

    # 解析当前任务的终止原因（未终止时返回 None）
    def resolve_done_reason(*, final: bool = False) -> str | None:
        # 已接受样本数达标
        if accepted_count >= accepted_target:
            return "accepted_target_reached"
        # 已写出步数达到请求上限
        if len(written_steps) >= requested_step_limit:
            return "step_limit_reached"
        # 任务侧请求停止（如时间预算耗尽）
        if task.stop_requested:
            return "task_stop_requested"
        # 非最终检查到此为止
        if not final:
            return None
        # 运行时间达到求解器时间上限
        if _safe_float(getattr(solver.state, "running_time", 0.0)) >= _safe_float(
            solver_cfg.time_limit_secs
        ):
            return "aira_time_limit_reached"
        # 内部步数达到求解器步数上限
        if _safe_int(getattr(solver.state, "current_step", 0)) >= _safe_int(
            solver_cfg.step_limit
        ):
            return "aira_step_limit_reached"
        # 内部代数达到求解器代数上限
        if _safe_int(getattr(solver.state, "current_generation", 0)) >= _safe_int(
            solver_cfg.num_generations
        ):
            return "aira_num_generations_exhausted"
        # 其余情况按“搜索正常返回”处理
        return "aira_search_returned"

    # 构建任务进度状态（写入全局 progress.json 的条目）
    def build_task_progress_state(*, final: bool = False) -> dict[str, Any]:
        # 先解析终止原因
        done_reason = resolve_done_reason(final=final)
        # 组装进度状态字典
        return {
            "task_name": task_name,
            # 非缺陷且有分数的步数
            "success": sum(
                1
                for stat in step_stats
                if not bool(stat.get("is_buggy")) and stat.get("score") is not None
            ),
            # 获得金银铜牌的步数
            "medal": sum(
                1
                for stat in step_stats
                if str(stat.get("submit_medal", "")).lower()
                in {"gold", "silver", "bronze"}
            ),
            # 已接受样本数
            "accepted": accepted_count,
            # 已完成样本数
            "completed": len(written_steps),
            # 目标数（两种键名冗余记录）
            "target": accepted_target,
            "accepted_target": accepted_target,
            # 步数上限
            "max_target": requested_step_limit,
            # 已生成样本数
            "generated": generated_count,
            # 是否终止与原因
            "done": done_reason is not None,
            "done_reason": done_reason,
        }

    # 已达成目标（接受数或步数上限）时无需再搜索，直接落盘并退出
    if accepted_count >= accepted_target or len(written_steps) >= requested_step_limit:
        # 更新全局进度快照
        update_progress_snapshot(
            run_dir / "progress.json",
            task_id=task_id,
            task_state=build_task_progress_state(),
            total_tasks=total_runner_tasks,
        )
        # 清理任务资源
        task.close(state)
        # 停止 dojo logger
        logger.stop()
        # 直接结束主流程
        return

    # 把求解器 journal 中最新的未落盘节点镜像到文件与统计
    def mirror_latest_node() -> None:
        # 引用外层计数器
        nonlocal accepted_count, generated_count, completed_count
        # 候选节点与步号
        node = None
        step_index = None
        # 从后往前找第一个未落盘的非根节点
        for candidate in reversed(list(solver.journal.nodes)):
            # 跳过无步号或根节点
            if candidate.step is None or solver.journal.is_root_node(candidate):
                continue
            # 换算为 0 起步的步号
            candidate_step_index = int(candidate.step) - 1
            # 已写出过则跳过
            if candidate_step_index in written_steps:
                continue
            # 记录候选并退出
            node = candidate
            step_index = candidate_step_index
            break
        # 没有可镜像的新节点时返回
        if node is None or step_index is None:
            return

        # 登记该步号已写出
        written_steps.add(step_index)
        # 创建该步的输出目录
        step_dir = output_dir / f"step_{step_index}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # 取生成该节点时的首个算子指标
        creation_metric = node.operators_metrics[0] if node.operators_metrics else None
        # 提取系统/用户提示词
        system_prompt, user_prompt, _ = _extract_prompt_messages(
            creation_metric
        )
        # 提取正文、思考内容、用量与原始补全文
        (
            response_text,
            reasoning_content,
            creation_usage,
            completion_text,
        ) = _extract_completion_artifacts(creation_metric, parse_thinking_tags)
        # 取节点指标附加信息（无指标时空字典）
        aux_info = dict(node.metric.info or {}) if node.metric is not None else {}
        # 汇总该节点全部算子调用的用量
        usage = _sum_operator_usage(node.operators_metrics)
        # 沙箱耗时优先取 aux 记录，其次节点执行时间
        sandbox_time_used = float(aux_info.get("run_time") or node.exec_time or 0.0)
        # 推断生成模式（算子名）：无记录时按是否缺陷给 debug/unknown
        operator = (
            str(node.operators_used[0])
            if node.operators_used
            else ("debug" if node.is_buggy else "unknown")
        )
        # 加载排行榜
        leaderboard = eval_utils.load_leaderboard(task_cfg)
        # 计算该步分数对应的等级与奖牌
        submit_grade, submit_medal = eval_utils.build_submit_grade_and_medal(
            aux_info.get("score"),
            leaderboard,
        )
        # 统计 slime 消息模板 token 数（未配置时为 None）
        slime_message_tokens, slime_message_tokenizer_model = count_slime_message_tokens(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_text=response_text,
            reasoning_content=reasoning_content,
            code=node.code,
        )

        # 写出该步的提示词与响应文件
        (step_dir / "system_prompt.md").write_text(system_prompt, encoding="utf-8")
        (step_dir / "user_prompt.md").write_text(user_prompt, encoding="utf-8")
        (step_dir / "response.md").write_text(response_text, encoding="utf-8")
        # 写出响应 JSON（含原始文本/思考/用量）
        (step_dir / "response.json").write_text(
            json.dumps(
                {
                    "raw_text": response_text,
                    "reasoning_content": reasoning_content,
                    "usage": creation_usage,
                    "completion_text": completion_text,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        (step_dir / "reasoning.md").write_text(
            reasoning_content,
            encoding="utf-8",
        )
        # 写出该步代码
        (step_dir / "valid_code.py").write_text(node.code, encoding="utf-8")
        # 写出原始运行日志
        (step_dir / "raw_run_log.txt").write_text(
            str(aux_info.get("raw_run_log") or ""),
            encoding="utf-8",
        )
        # 写出清洗后的运行日志
        (step_dir / "clear_run_log.txt").write_text(
            str(aux_info.get("clear_run_log") or node.term_out or ""),
            encoding="utf-8",
        )
        # 写出反馈文本
        (step_dir / "feedback.txt").write_text(
            str(aux_info.get("feedback") or node.analysis or node.term_out or ""),
            encoding="utf-8",
        )

        # 组装该步的统计负载
        step_stat_payload = {
            "mode": operator,
            "operator": operator,
            # 当前代数 id
            "generation_id": int(solver.state.current_generation),
            # 当前代对应的温度（线性衰减）
            "temperature": float(solver.linear_decay(int(solver.state.current_generation))),
            "score": aux_info.get("score"),
            "reward": float(aux_info.get("reward") or 0.0),
            "submit_grade": submit_grade,
            "submit_medal": submit_medal,
            # 适应度取节点指标值
            "fitness": node.metric.value if node.metric is not None else None,
            "cost": float(usage["cost"]),
            "prompt_tokens": int(usage["prompt_tokens"]),
            "completion_tokens": int(usage["completion_tokens"]),
            "total_tokens": int(usage["total_tokens"]),
            # 状态优先取 aux 记录，否则按缺陷标记推断
            "status": str(aux_info.get("status") or ("buggy" if node.is_buggy else "unknown")),
            "status_code": aux_info.get("status_code"),
            # 模型耗时取算子延迟累计
            "model_time_used": float(usage["latency"]),
            "sandbox_time_used": sandbox_time_used,
            "model_plus_sandbox_time_used": float(usage["latency"]) + sandbox_time_used,
            "is_buggy": bool(node.is_buggy),
            # 父节点步号列表（转 0 起点）
            "parent_steps": [
                int(parent.step) - 1 for parent in list(node.parents or []) if parent.step
            ],
            # 节点 id
            "node_id": node.id,
        }
        # 落盘该步 stat.json
        (step_dir / "stat.json").write_text(
            json.dumps(step_stat_payload, indent=2),
            encoding="utf-8",
        )
        # 追加到任务级步统计列表
        step_stats.append({"step": step_index, **step_stat_payload})
        # 程序 id（节点 id 兜底为步号）
        program_id = str(node.id or f"step_{step_index}")
        # 收集非根父节点 id 列表
        parent_ids = [
            str(parent.id)
            for parent in list(node.parents or [])
            if parent is not None and not solver.journal.is_root_node(parent)
        ]
        # 搜索元信息（算法/步号/程序树关系等）
        search_metadata = {
            "search_algorithm": "airaevo",
            "search_step": step_index,
            "program_id": program_id,
            "parent_ids": parent_ids,
            "generation_mode": operator,
            "fitness": node.metric.value if node.metric is not None else None,
            "code_path": str(Path(f"step_{step_index}") / "valid_code.py"),
        }
        # 由拒绝策略判定该样本是否被接受
        accepted_decision = rejection_policy.accepts_record(
            {
                "score": aux_info.get("score"),
                "reward": float(aux_info.get("reward") or 0.0),
                "submit_medal": submit_medal,
                "status": str(aux_info.get("status") or ""),
                "success": not bool(node.is_buggy),
                "feedback": str(aux_info.get("feedback") or node.analysis or node.term_out or ""),
                "slime_message_tokens": slime_message_tokens,
                "generation_prompt_tokens": int(usage["prompt_tokens"]),
                "generation_completion_tokens": int(usage["completion_tokens"]),
                "generation_total_tokens": int(usage["total_tokens"]),
                "total_tokens": int(usage["total_tokens"]),
            },
            task_cfg,
        )
        # 生成数与完成数各加一
        generated_count += 1
        completed_count += 1
        # 被策略接受时更新接受计数
        if accepted_decision.accepted:
            accepted_count += 1
        # 请求 id = 任务id_步号
        request_id = f"{task_id}_{step_index}"
        # 追加生成结果记录（SFT 数据源）
        append_jsonl_record(
            run_dir / "gen_results.jsonl",
            {
                "request_id": request_id,
                "task_id": task_id,
                "task_name": task_name,
                "code": node.code,
                "raw_text": response_text,
                "reasoning_content": reasoning_content,
                "prompt_tokens": int(usage["prompt_tokens"]),
                "completion_tokens": int(usage["completion_tokens"]),
                "total_tokens": int(usage["total_tokens"]),
                "cost": float(usage["cost"]),
                # 模型名取 model_id 的最后一段
                "model_name": str(runner_cfg["llm"]["model_id"]).split("/")[-1],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "metadata": task_cfg,
                "step_index": step_index,
                "data_dir": str(task_cfg.get("data_dir") or ""),
                "output_dir": str(compat_task_dir),
                "success": bool(node.code),
                "error": None,
                "timestamp": datetime.now().isoformat(),
                **search_metadata,
            },
        )
        # 追加评测结果记录
        append_jsonl_record(
            run_dir / "eval_results.jsonl",
            {
                "request_id": request_id,
                "task_id": task_id,
                "task_name": task_name,
                "status_code": aux_info.get("status_code"),
                "status": str(aux_info.get("status") or ("buggy" if node.is_buggy else "unknown")),
                "score": aux_info.get("score"),
                "reward": float(aux_info.get("reward") or 0.0),
                "submit_grade": submit_grade,
                "submit_medal": submit_medal,
                "job_id": aux_info.get("job_id"),
                "queue_time": aux_info.get("queue_time"),
                "run_time": sandbox_time_used,
                "step_index": step_index,
                # 成功 = 非缺陷且有分数
                "success": not bool(node.is_buggy) and aux_info.get("score") is not None,
                "generation_prompt_tokens": int(usage["prompt_tokens"]),
                "generation_completion_tokens": int(usage["completion_tokens"]),
                "generation_total_tokens": int(usage["total_tokens"]),
                "slime_message_tokens": slime_message_tokens,
                "slime_message_tokenizer_model": slime_message_tokenizer_model,
                "feedback_path": str(step_dir / "feedback.txt"),
                "error": None,
                "timestamp": datetime.now().isoformat(),
                # 拒绝策略判定结果与原因
                "accepted_by_rejection_policy": accepted_decision.accepted,
                "rejection_reason": accepted_decision.reason,
                **search_metadata,
            },
        )
        # 更新全局进度快照
        update_progress_snapshot(
            run_dir / "progress.json",
            task_id=task_id,
            task_state=build_task_progress_state(),
            total_tasks=total_runner_tasks,
        )
        # 追加“节点已生成”搜索事件
        append_search_event(
            search_event_log_path,
            SearchEvent(
                event_type=NODE_GENERATED,
                task_id=task_id,
                task_name=task_name,
                search_algorithm="airaevo",
                search_step=step_index,
                program_id=program_id,
                parent_ids=parent_ids,
                generation_mode=operator,
                code_path=str(Path(f"step_{step_index}") / "valid_code.py"),
                generated_count=generated_count,
            ),
        )
        # 追加“节点已评测”搜索事件（含分数/接受判定/计数）
        append_search_event(
            search_event_log_path,
            SearchEvent(
                event_type=NODE_EVALUATED,
                task_id=task_id,
                task_name=task_name,
                search_algorithm="airaevo",
                search_step=step_index,
                program_id=program_id,
                parent_ids=parent_ids,
                generation_mode=operator,
                code_path=str(Path(f"step_{step_index}") / "valid_code.py"),
                score=aux_info.get("score"),
                reward=float(aux_info.get("reward") or 0.0),
                fitness=node.metric.value if node.metric is not None else None,
                feedback_path=str(Path(f"step_{step_index}") / "feedback.txt"),
                raw_log_path=str(Path(f"step_{step_index}") / "raw_run_log.txt"),
                accepted_by_rejection_policy=accepted_decision.accepted,
                rejection_reason=accepted_decision.reason,
                generated_count=generated_count,
                completed_count=completed_count,
                accepted_count=accepted_count,
                stop_requested=accepted_count >= accepted_target,
                extra={
                    "generation_prompt_tokens": int(usage["prompt_tokens"]),
                    "generation_completion_tokens": int(usage["completion_tokens"]),
                    "generation_total_tokens": int(usage["total_tokens"]),
                },
            ),
        )
        # 读取解库状态（存在时）
        solution_database_state = (
            solver.solution_database.state_dict()
            if hasattr(solver, "solution_database")
            else {}
        )
        # 重放全部搜索事件，得到程序树汇总
        replay_summary = replay_search_events(read_search_events(search_event_log_path))
        # 原子写出搜索状态快照
        atomic_write_search_state(
            search_state_path,
            SearchState(
                task_id=task_id,
                task_name=task_name,
                search_algorithm="airaevo",
                # 下一步搜索步号
                next_search_step=step_index + 1,
                generated_count=generated_count,
                completed_count=completed_count,
                accepted_count=accepted_count,
                accepted_target=accepted_target,
                max_generated=requested_step_limit,
                stop_requested=accepted_count >= accepted_target,
                # 程序树信息（由事件重放得到）
                program_ids=list(replay_summary.program_ids),
                program_scores=dict(replay_summary.program_scores),
                program_fitness=dict(replay_summary.program_fitness),
                program_code_paths=dict(replay_summary.program_code_paths),
                parent_map=dict(replay_summary.parent_map),
                generation_modes=dict(replay_summary.generation_modes),
                # 岛屿种群状态（来自解库）
                island_populations=solution_database_state.get("island_populations"),
                # 代缓冲状态
                generation_buffer={
                    "current_generation": int(solver.state.current_generation),
                    "next_counter_id": None,
                },
                # journal 文件相对路径
                journal_path="aira_evo/checkpoint/journal.jsonl",
            ),
        )
        # 接受数达标时请求任务停止
        if accepted_count >= accepted_target:
            task.stop_requested = True
        # 用当前最佳节点刷新任务统计
        write_task_stat(best_node=_select_best_available_node(solver))

    # 保存原版 log_journal 方法
    original_log_journal = solver.log_journal

    # 打补丁后的 log_journal：先执行原逻辑，再镜像最新节点
    def patched_log_journal() -> None:
        # 调用原版日志落盘
        original_log_journal()
        # 镜像最新节点到文件与统计
        mirror_latest_node()

    # 替换求解器的 log_journal 实现（monkey patch）
    solver.log_journal = patched_log_journal

    # 判断是否应停止搜索（预算/目标/停止标记）
    def should_stop_search() -> bool:
        # 任务侧已请求停止
        if task.stop_requested:
            return True
        # 运行时间达到时间上限
        if time.monotonic() - solver.start_time >= float(solver_cfg.time_limit_secs):
            return True
        # 已写出步数达到请求上限
        return len(written_steps) >= requested_step_limit

    # 给算子调用加守卫：预算耗尽时抛 StopSearch 中断
    def guard_operator_call(fn: Any) -> Any:
        # 包装函数
        def wrapped(*operator_args: Any, **operator_kwargs: Any) -> Any:
            # 应停止时抛异常中断搜索
            if should_stop_search():
                raise StopSearch("Search budget reached.")
            # 否则正常执行算子
            return fn(*operator_args, **operator_kwargs)

        # 返回包装后的函数
        return wrapped

    # 对四个生成类算子全部加守卫
    solver._draft = guard_operator_call(solver._draft)
    solver._improve = guard_operator_call(solver._improve)
    solver._debug = guard_operator_call(solver._debug)
    solver._crossover = guard_operator_call(solver._crossover)

    # 最佳节点初始化为空
    best_node = None
    # 提交结果初始化为空
    submit_result: dict[str, Any] | None = None
    # 测试耗时初始化为 0
    test_time = 0.0

    try:
        # 运行演化求解主流程（返回值第三项为最佳节点）
        _, _, best_node = solver(task, state)
    except StopSearch:
        # 预算耗尽：保存 checkpoint
        solver.save_checkpoint()
        # 导出搜索结果
        export_search_results(solver.cfg, solver.journal, logger, "EVO")
        # 从 journal 选取当前最佳可用节点
        best_node = _select_best_available_node(solver)
    finally:
        # 无最佳节点时从 journal 兜底选取
        best_node = _select_best_available_node(solver) if best_node is None else best_node
        # 循环镜像，直到没有新的未落盘节点
        while True:
            # 记录镜像前的步数
            before = len(written_steps)
            mirror_latest_node()
            # 步数不再增长说明已全部落盘
            if len(written_steps) == before:
                break

        # 取最佳节点代码（无节点时空串）
        best_code = best_node.code if best_node is not None else ""
        # 写出最终验证代码
        (output_dir / "valid_code_final.py").write_text(best_code, encoding="utf-8")
        # 构建提交用代码（路径替换）
        submit_code = task.build_submit_code(best_code)
        # 写出提交代码
        (output_dir / "submit_code.py").write_text(submit_code, encoding="utf-8")

        # 用测试阶段评测提交代码
        submit_result = task.evaluate_code(best_code, phase="test")
        # 成功执行时记录测试耗时
        if submit_result["status_code"] == 200 and submit_result.get("run_time") is not None:
            test_time = float(submit_result["run_time"])

        # 写出提交运行的原始日志
        (output_dir / "submit_raw_run_log.txt").write_text(
            str(submit_result.get("raw_run_log") or ""),
            encoding="utf-8",
        )
        # 写出提交运行的清洗日志
        (output_dir / "submit_clear_run_log.txt").write_text(
            str(submit_result.get("clear_run_log") or ""),
            encoding="utf-8",
        )
        # 写出提交运行的反馈
        (output_dir / "submit_feedback.txt").write_text(
            str(submit_result.get("feedback") or ""),
            encoding="utf-8",
        )

        # 用提交结果刷新最终任务统计
        write_task_stat(
            best_node=best_node,
            submit_result=submit_result,
            test_time=test_time,
        )
        # 最终更新全局进度快照（final 标记）
        update_progress_snapshot(
            run_dir / "progress.json",
            task_id=task_id,
            task_state=build_task_progress_state(final=True),
            total_tasks=total_runner_tasks,
        )
        # 清理任务资源
        task.close(state)
        # 停止 dojo logger
        logger.stop()


# 脚本入口
if __name__ == "__main__":
    main()
