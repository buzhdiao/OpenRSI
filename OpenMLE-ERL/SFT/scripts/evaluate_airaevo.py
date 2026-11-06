# 启用 PEP 563 延迟注解求值，类型注解在运行时不立即求值，避免循环导入并兼容新语法
from __future__ import annotations

# 标准库：日志、环境变量、子进程调用、sys.path 操作、路径处理、任意类型注解
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# hydra：基于配置文件驱动的实验入口框架
import hydra
# pandas：读取评测集 parquet 文件，从中提取任务列表
import pandas as pd
# yaml：把派生的 prepare/runner 配置落盘为 YAML 供子进程读取
import yaml
# dotenv：从仓库根目录的 .env 加载 API key 等环境变量
from dotenv import load_dotenv
# OmegaConf：hydra 配置对象及其到普通 Python 容器的转换
from omegaconf import DictConfig, OmegaConf

# 仓库根目录：本脚本位于 <root>/SFT/scripts/ 下，parents[1] 即 SFT 根目录
REPO_ROOT = Path(__file__).resolve().parents[1]
# 把 SFT 根目录插入 sys.path 最前，保证后面能直接导入 tts_search 包
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# tts_search 子项目目录（内含 configs 与 eval_network 模块）
TTS_SEARCH_DIR = REPO_ROOT / "tts_search"
# 本模块专属 logger
logger = logging.getLogger(__name__)

# 从 tts_search 导入禁用代理环境变量的工具（评测需直连内网服务）
from tts_search.eval_network import disable_proxy_env


def _to_plain(value: Any) -> Any:
    # 若传入的是 OmegaConf 配置节点，则递归解析为普通 dict/list 并插值 ${...} 引用
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    # 其他类型原样返回
    return value


def _first_model_payload(cfg: DictConfig) -> dict[str, Any]:
    # 把 litellm.model_list 配置解析为普通列表；为空时视为配置错误
    model_list = list(_to_plain(cfg.litellm.model_list) or [])
    if not model_list:
        raise ValueError("cfg.litellm.model_list is empty")
    # 只取第一个模型条目（评测只使用主模型），返回其 dict 副本
    return dict(model_list[0])


def _generation_kwargs_from_litellm_params(
    litellm_params: dict[str, Any],
) -> dict[str, Any]:
    # 先解析为普通 dict（防止传入 OmegaConf 节点），空值兜底为空 dict
    litellm_params = dict(_to_plain(litellm_params) or {})
    # 白名单：允许透传给后端 LLM 的生成参数键集合（温度、top_p、最大 token 等）
    generation_keys = {
        "frequency_penalty",
        "include_usage",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "min_p",
        "n",
        "presence_penalty",
        "response_format",
        "seed",
        "stop",
        "stream",
        "stream_options",
        "temperature",
        "tool_choice",
        "tools",
        "top_k",
        "top_logprobs",
        "top_p",
    }
    # 只保留白名单内且值非 None 的参数，避免把 None 覆盖后端默认值
    generation_kwargs = {
        key: value
        for key, value in litellm_params.items()
        if key in generation_keys and value is not None
    }
    # extra_body 用于携带非标准扩展参数，整体透传（同样先解析为普通 dict）
    if litellm_params.get("extra_body") is not None:
        generation_kwargs["extra_body"] = dict(_to_plain(litellm_params["extra_body"]))
    # 返回整理后的生成参数字典
    return generation_kwargs


def _configure_airaevo_api_key_env(litellm_params: dict[str, Any]) -> None:
    """Bridge Hydra LiteLLM keys to the AIRA Evo backend env convention."""
    # 取出 api_key 并去掉首尾空白
    api_key = str(litellm_params.get("api_key") or "").strip()
    # 空 key 或占位符 "EMPTY"（常见于本地 vLLM 部署）时不设置环境变量
    if not api_key or api_key.upper() == "EMPTY":
        return
    # 以 setdefault 写入 AIRA Evo 后端约定的 PRIMARY_KEY 环境变量，不覆盖已有值
    os.environ.setdefault("PRIMARY_KEY", api_key)


def _sanitize_task_name(task_name: str) -> str:
    # 把任务名中的路径分隔符替换为下划线，避免其被当作目录层级
    return task_name.replace("/", "_").replace("\\", "_")


def _task_list_from_eval_data(cfg: DictConfig) -> list[str]:
    # 优先使用配置里显式给出的 task_list
    configured = list(_to_plain(cfg.search.runner.get("task_list", [])) or [])
    if configured:
        # 配置了就把每项转为字符串返回
        return [str(item) for item in configured]

    # 否则从评测数据（parquet）的 metadata 列读取任务名列表
    records = pd.read_parquet(
        str(cfg.data.eval_data),
        columns=[str(cfg.data.metadata_key)],
    ).to_dict(orient="records")
    # 收集所有任务名（做路径安全化处理）
    task_names: list[str] = []
    for record in records:
        # 每行的 metadata 字段是嵌套 dict，包含 task_name
        metadata = dict(record[str(cfg.data.metadata_key)])
        task_names.append(_sanitize_task_name(str(metadata["task_name"])))
    # 若配置了 max_tasks，则只截取前 N 个任务
    if cfg.max_tasks is not None:
        task_names = task_names[: int(cfg.max_tasks)]
    # 返回最终参与评测的任务列表
    return task_names


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    # 以 UTF-8 写出 YAML；allow_unicode 保留中文等非 ASCII 字符，sort_keys 保持插入顺序
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


@hydra.main(
    config_path=f"{TTS_SEARCH_DIR}/configs",
    config_name="experiment/evolutionary_glm47",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    # 加载仓库根目录 .env，注入各服务的 API key 等凭据
    load_dotenv(REPO_ROOT / ".env")
    # 禁用 HTTP(S)_PROXY 等代理变量，保证与内网沙箱/LLM 服务直连
    disable_proxy_env()
    # 根日志级别设为 INFO
    logging.basicConfig(level=logging.INFO)

    # 解析输出目录为绝对路径
    output_dir = Path(str(cfg.output_dir)).resolve()
    # hydra 子目录用于存放派生出的 prepare/runner 配置
    hydra_dir = output_dir / ".hydra"
    # 不存在则递归创建（已存在也不报错）
    hydra_dir.mkdir(parents=True, exist_ok=True)

    # 取出 search.runner 段配置并转为普通 dict
    search_runner_cfg = dict(_to_plain(cfg.search.runner) or {})
    # 取第一个模型条目（评测使用的推理模型）
    first_model = _first_model_payload(cfg)
    # 取该模型的 litellm_params（含 base_url、api_key、生成参数等）
    litellm_params = dict(first_model.get("litellm_params") or {})
    # 把 api_key 桥接为 AIRA Evo 后端约定的 PRIMARY_KEY 环境变量
    _configure_airaevo_api_key_env(litellm_params)
    # 从 litellm_params 抽取可透传的生成参数
    llm_generation_kwargs = _generation_kwargs_from_litellm_params(litellm_params)
    # 每步候选个体数（进化搜索的种群宽度），默认为 1
    candidates_per_step = int(getattr(cfg, "candidates_per_step", 1) or 1)
    # 从评测数据或配置解析任务列表
    task_list = _task_list_from_eval_data(cfg)
    # 任务根目录：默认指向 third_party/aira-evo 的 mle_bench 示例目录
    default_task_root = str(
        search_runner_cfg.get("task_root", "third_party/aira-evo/examples/mle_bench")
    )
    # 包根目录：默认去掉任务根目录尾部的 /examples/mle_bench 得到
    package_root_value = str(
        search_runner_cfg.get(
            "package_root",
            default_task_root.removesuffix("/examples/mle_bench"),
        )
    )

    # 组装 AIRA Evo prepare 阶段（构建任务）所需的配置载荷
    prepare_payload = {
        "data": {
            # 评测数据文件路径
            "eval_data": str(cfg.data.eval_data),
            # 排行榜目录：未配置时显式置 None
            "leaderboard_dir": (
                str(cfg.data.leaderboard_dir)
                if cfg.data.leaderboard_dir is not None
                else None
            ),
            # 提交目录（runner 内的答案提交位置）
            "submit_dir": str(
                search_runner_cfg.get("submit_dir", "MLTasks/MLE-Bench-Lite-Modified")
            ),
            # 提交数据的根目录前缀（可为空串）
            "submit_data_dir_root": str(
                search_runner_cfg.get(
                    "submit_data_dir_root",
                    "",
                )
            ),
            # 输入列名与元数据列名（数据集 schema 约定）
            "input_key": str(cfg.data.input_key),
            "metadata_key": str(cfg.data.metadata_key),
        },
        "sandbox": {
            # 沙箱资源映射：cpu/gpu 各自的 base_url，runner 配置优先，缺省回落到 cfg.sandbox.base_url
            "base_url_map": [
                {
                    "resource": "cpu",
                    "base_url": str(
                        search_runner_cfg.get("cpu_base_url") or cfg.sandbox.base_url
                    ),
                },
                {
                    "resource": "gpu",
                    "base_url": str(
                        search_runner_cfg.get("gpu_base_url") or cfg.sandbox.base_url
                    ),
                },
            ],
            # 单个沙箱作业超时（秒）
            "job_timeout": int(cfg.sandbox.job_timeout),
            # 等待作业结束的总超时（秒）
            "wait_timeout": int(cfg.sandbox.wait_timeout),
            # 轮询间隔（秒）
            "poll_interval": int(cfg.sandbox.poll_interval),
            # 是否使用 score2 奖励（排名分位转奖励的 execution-grounded 打分）
            "use_score2reward": bool(cfg.sandbox.use_score2reward),
        },
    }

    # 组装 AIRA Evo runner（进化搜索主循环）所需的配置载荷
    runner_payload = {
        # 实验名与输出目录、随机种子
        "experiment_name": str(cfg.experiment_name),
        "output_dir": str(output_dir),
        "seed": int(cfg.seed),
        # 总时间预算（小时），以及“模型+沙箱”联合预算（可选）
        "time_budget": cfg.time_budget,
        "model_plus_sandbox_time_budget": getattr(
            cfg, "model_plus_sandbox_time_budget", None
        ),
        # 每任务采样的轨迹条数
        "n_samples_per_task": int(cfg.n_samples_per_task),
        # 拒绝采样目标条数：未配置时等于 n_samples_per_task（即不额外多采）
        "accepted_target": int(
            getattr(cfg, "rejection_target", None) or cfg.n_samples_per_task
        ),
        # 拒绝采样策略及各类阈值（分数/奖励门槛、参考分数文件、可接受的奖牌集合）
        "rejection_policy": getattr(cfg, "rejection_policy", None),
        "rejection_target": getattr(cfg, "rejection_target", None),
        "rejection_score_threshold": getattr(cfg, "rejection_score_threshold", None),
        "rejection_reward_threshold": getattr(cfg, "rejection_reward_threshold", None),
        "rejection_reference_scores_path": getattr(
            cfg, "rejection_reference_scores_path", None
        ),
        # 拒绝采样可接受的奖牌等级列表（如 gold/silver/bronze）
        "rejection_accepted_medals": _to_plain(
            getattr(cfg, "rejection_accepted_medals", [])
        ),
        # 是否启用基线过滤（对齐基线的 token 预算与相对差距限制）
        "rejection_apply_baseline_filters": bool(
            getattr(cfg, "rejection_apply_baseline_filters", True)
        ),
        # 基线 token 上限与分词器模型名
        "rejection_baseline_token_limit": int(
            getattr(cfg, "rejection_baseline_token_limit", 32768)
        ),
        "rejection_baseline_tokenizer_model": getattr(
            cfg, "rejection_baseline_tokenizer_model", None
        ),
        # 基线相对差距上限（默认 12%）
        "rejection_baseline_relative_gap_limit": float(
            getattr(cfg, "rejection_baseline_relative_gap_limit", 0.12)
        ),
        # 混合榜单/无榜单场景下的拒绝采样目标条数
        "rejection_mixed_leaderboard_target": int(
            getattr(cfg, "rejection_mixed_leaderboard_target", 2)
        ),
        "rejection_mixed_no_leaderboard_target": int(
            getattr(cfg, "rejection_mixed_no_leaderboard_target", 4)
        ),
        # 每步候选个体数
        "candidates_per_step": candidates_per_step,
        # 三级并发：任务级、LLM 请求级、沙箱请求级（未配置任务级并发时置 None 走后端默认）
        "task_concurrency": (
            int(search_runner_cfg["task_concurrency"])
            if search_runner_cfg.get("task_concurrency") is not None
            else None
        ),
        "llm_concurrency": int(cfg.llm_concurrency),
        "sandbox_concurrency": int(cfg.sandbox.concurrency),
        # 断点续跑是否要求严格一致（任务列表等必须完全匹配）
        "strict_resume": bool(search_runner_cfg.get("strict_resume", False)),
        # 进度历史快照间隔（秒），默认 600s
        "progress_history_interval_seconds": float(
            getattr(cfg, "progress_history_interval_seconds", 600.0) or 600.0
        ),
        # 任务根目录与任务列表
        "task_root": default_task_root,
        "task_list": task_list,
        # 榜单目录（可为 None）
        "leaderboard_dir": (
            str(cfg.data.leaderboard_dir)
            if cfg.data.leaderboard_dir is not None
            else None
        ),
        # LLM 接入配置：API 类型、模型名、base_url、生成参数、是否走 Azure 客户端、provider 归属
        "llm": {
            "api": str(search_runner_cfg.get("llm", {}).get("api", "litellm")),
            "model_id": str(first_model.get("model_name")),
            "base_url": str(litellm_params.get("base_url")),
            "generation_kwargs": llm_generation_kwargs,
            "use_azure_client": bool(
                search_runner_cfg.get("llm", {}).get("use_azure_client", False)
            ),
            "provider": str(
                search_runner_cfg.get("llm", {}).get("provider", "selfhosted")
            ),
        },
        # solver（进化搜索策略）/interpreter（代码解释器）/logger 三段原样透传
        "solver": dict(search_runner_cfg.get("solver") or {}),
        "interpreter": dict(search_runner_cfg.get("interpreter") or {}),
        "logger": dict(search_runner_cfg.get("logger") or {}),
        # 传给子进程 runner 的 hydra 行为配置：日志写入同一 .hydra 目录、不切换工作目录
        "hydra": {
            "run": {"dir": f"{output_dir}/.hydra"},
            "output_subdir": ".",
            "job": {"chdir": False},
        },
    }
    # 配置了 max_steps 时，写入 solver 的步数上限
    if cfg.max_steps is not None:
        runner_payload["solver"]["step_limit"] = int(cfg.max_steps)
    # 每步候选数大于 1 时，把它作为进化搜索每代个体数
    if candidates_per_step > 1:
        runner_payload["solver"]["individuals_per_generation"] = candidates_per_step

    # 两份派生配置的落盘路径
    prepare_config_path = hydra_dir / "airaevo_prepare_config.yaml"
    runner_config_path = hydra_dir / "airaevo_runner_config.yaml"
    # 写出 prepare 与 runner 配置，供下游子进程读取
    _write_yaml(prepare_config_path, prepare_payload)
    _write_yaml(runner_config_path, runner_payload)

    # 解析包根目录为绝对路径（相对路径则相对仓库根）
    package_root = Path(package_root_value)
    if not package_root.is_absolute():
        package_root = REPO_ROOT / package_root_value
    # AIRA Evo 的 MLE 桥接目录（内含 build_tasks.py 与 runner.py）
    mle_bench_dir = package_root / "examples" / "mle_bench"
    if not mle_bench_dir.is_dir():
        raise FileNotFoundError(
            f"AIRA Evo MLE bridge directory does not exist: {mle_bench_dir}"
        )

    # 子进程环境：复制当前环境变量
    subprocess_env = os.environ.copy()
    # PYTHONPATH 需要包含 SFT 根目录与包内 src 目录
    python_path_entries = [str(REPO_ROOT), str(package_root / "src")]
    # 已有 PYTHONPATH 则追加在后面，保持原有可导入路径
    existing_python_path = subprocess_env.get("PYTHONPATH")
    if existing_python_path:
        python_path_entries.append(existing_python_path)
    # 用平台路径分隔符拼回 PYTHONPATH
    subprocess_env["PYTHONPATH"] = os.pathsep.join(python_path_entries)

    # 第一步：运行 build_tasks.py，根据 prepare 配置构建评测任务
    build_tasks_cmd = [
        sys.executable,
        str(mle_bench_dir / "build_tasks.py"),
        "--config",
        str(prepare_config_path),
    ]
    # 记录使用的配置文件路径
    logger.info("Building AIRA Evo tasks with %s", prepare_config_path)
    # 同步执行构建任务脚本；check=True 使非零退出码直接抛异常终止
    subprocess.run(
        build_tasks_cmd,
        check=True,
        cwd=REPO_ROOT,
        env=subprocess_env,
    )

    # 第二步：运行 runner.py 执行 AIRA Evo 进化评测（以 hydra 方式传入 runner 配置）
    runner_cmd = [
        sys.executable,
        str(mle_bench_dir / "runner.py"),
        "--config-path",
        str(runner_config_path.parent),
        "--config-name",
        runner_config_path.stem,
    ]
    # 记录 runner 启动信息
    logger.info("Running AIRA Evo evaluation with %s", runner_config_path)
    # 同步执行评测 runner；失败即抛异常
    subprocess.run(
        runner_cmd,
        check=True,
        cwd=REPO_ROOT,
        env=subprocess_env,
    )


# 直接运行本脚本时进入 hydra 主入口
if __name__ == "__main__":
    main()
