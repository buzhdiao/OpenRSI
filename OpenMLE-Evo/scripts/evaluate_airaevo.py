# 中文说明:airaevo(MLE-Bench)评估编排脚本——Hydra 配置驱动的一键评估入口。
# 在系统中的位置:OpenMLE-Evo/scripts 下,串联"评测数据增强 -> 任务构建 -> aira-evo runner":
# 1) 按需给 eval_data(parquet)注入"验证集切分指导"与"任务技能指导"两种 prompt 补丁;
# 2) 把 litellm 模型路由配置翻译成 runner 可用的单模型 LLM 配置;
# 3) 组装 prepare(任务构建)与 runner(搜索评估)两份 YAML,再以子进程依次执行。
# 关键设计:prompt 补丁只做一次并写到 .hydra 目录,保证多 worker 共享同一份提示词;
# 两阶段都 subprocess + check=True,失败即停,便于上游捕获。

# 延迟注解:类型标注运行期不求值
from __future__ import annotations

# json:解析 metadata 字符串
import json
# logging:模块日志
import logging
# subprocess:依次执行 build_tasks 与 runner 两个子进程
import subprocess
# sys:把仓库根目录加入 sys.path
import sys
# Path:路径对象
from pathlib import Path
# 类型注解
from typing import Any

# hydra:配置驱动入口
import hydra
# pandas:读写 parquet 评测数据
import pandas as pd
# yaml:落盘 prepare/runner 配置
import yaml
# load_dotenv:加载 .env
from dotenv import load_dotenv
# OmegaConf:Hydra 配置对象与取值
from omegaconf import DictConfig, OmegaConf

# 仓库根目录(OpenMLE-Evo)
REPO_ROOT = Path(__file__).resolve().parents[1]
# 保证可 import tts_search 包
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# 配置目录(tts_search/configs)
TTS_SEARCH_DIR = REPO_ROOT / "tts_search"
logger = logging.getLogger(__name__)

# 网络工具:清空代理环境变量(沙箱/自托管服务不走代理)
from tts_search.eval_network import disable_proxy_env  # noqa: E402
# 任务技能指导:技能映射加载器与 prompt 注入器
from tts_search.task_skill_guidance import (  # noqa: E402
    TaskSkillGuidanceInjector,
    load_task_skill_map,
)
# 验证集切分指导:批量修补 eval_data 的工具
from tts_search.validation_split_guidance import (  # noqa: E402
    patch_eval_data_with_guidance,
)


def _to_plain(value: Any) -> Any:
    """把 OmegaConf 节点递归转为原生容器;非配置原样返回。"""
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _first_model_payload(cfg: DictConfig) -> dict[str, Any]:
    """取 litellm 路由里的第一个模型配置(评估只用单模型)。"""
    model_list = list(_to_plain(cfg.litellm.model_list) or [])
    if not model_list:
        raise ValueError("cfg.litellm.model_list is empty")
    return dict(model_list[0])


def _generation_kwargs_from_litellm_params(
    litellm_params: dict[str, Any],
) -> dict[str, Any]:
    """从 litellm 路由参数中筛出"生成参数"子集(runner 只认识这些键)。"""
    litellm_params = dict(_to_plain(litellm_params) or {})
    # 允许透传的生成参数白名单
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
    # 只保留白名单内且非 None 的键
    generation_kwargs = {
        key: value
        for key, value in litellm_params.items()
        if key in generation_keys and value is not None
    }
    # extra_body 单独透传(转为原生 dict)
    if litellm_params.get("extra_body") is not None:
        generation_kwargs["extra_body"] = dict(_to_plain(litellm_params["extra_body"]))
    return generation_kwargs


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """写 YAML(保留键序,允许非 ASCII)。"""
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _resolve_repo_path(path_value: str | None) -> Path | None:
    """把相对路径锚定到仓库根;空值返回 None。"""
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _maybe_patch_eval_data_with_validation_guidance(
    cfg: DictConfig,
    *,
    hydra_dir: Path,
) -> Path:
    """按需给 eval_data 注入"验证集切分指导";未启用则原样返回路径。"""
    # 原始评测数据路径
    eval_data_path = Path(str(cfg.data.eval_data)).resolve()
    # 读取指导配置段
    guidance_cfg = OmegaConf.select(cfg, "data.validation_split_guidance") or {}
    if not bool(guidance_cfg.get("enabled", False)):
        return eval_data_path

    # 指导文档路径(默认 docs 下的 22 任务说明)
    instructions_path = _resolve_repo_path(
        str(
            guidance_cfg.get(
                "instructions_path",
                "docs/mlebench_validation_split_instructions_22.md",
            )
        )
    )
    if instructions_path is None:
        raise ValueError("data.validation_split_guidance.instructions_path is empty")

    # 输出文件名(默认原名 + _validation_split_guidance 后缀)
    output_filename = str(
        guidance_cfg.get(
            "output_filename",
            f"{eval_data_path.stem}_validation_split_guidance{eval_data_path.suffix}",
        )
    )
    output_path = hydra_dir / output_filename
    # 执行批量修补
    stats = patch_eval_data_with_guidance(
        eval_data_path=eval_data_path,
        output_path=output_path,
        instructions_path=instructions_path,
        input_key=str(cfg.data.input_key),
        metadata_key=str(cfg.data.metadata_key),
        insert_marker=str(guidance_cfg.get("insert_marker", "**FINAL OUTPUT**")),
        heading=str(guidance_cfg.get("heading", "Validation Split Guidance:")),
        strict=bool(guidance_cfg.get("strict", False)),
    )
    # 记录修补统计
    logger.info(
        "Validation split guidance patched eval_data: %s -> %s "
        "(patched_rows=%s/%s, skipped_existing=%s, missing_tasks=%s)",
        stats.input_path,
        stats.output_path,
        stats.patched_rows,
        stats.total_rows,
        stats.skipped_existing_rows,
        list(stats.missing_tasks),
    )
    return output_path


def _metadata_task_name(metadata: Any) -> str | None:
    """从 metadata(字符串 JSON 或 dict)提取任务名(多个候选键依次尝试)。"""
    metadata = _to_plain(metadata)
    # 字符串形式:先解析为 dict
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return None
    if not isinstance(metadata, dict):
        return None
    # 依次尝试常见任务名键
    for key in ("task_name", "competition_id", "name"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def _inject_task_skill_into_prompt_value(
    prompt_value: Any,
    *,
    task_name: str,
    injector: TaskSkillGuidanceInjector,
) -> tuple[Any, Any]:
    """把任务技能指导注入一条 prompt:兼容纯文本 / numpy 数组 / 消息列表三种形态。"""
    prompt_value = _to_plain(prompt_value)
    messages: list[Any] | None = None

    # numpy/数组形态:tolist 转消息列表
    if hasattr(prompt_value, "tolist"):
        messages = list(prompt_value.tolist() or [])
    elif isinstance(prompt_value, (list, tuple)):
        # 原生列表/元组形态
        messages = list(prompt_value)

    # 纯文本形态:整段注入
    if messages is None:
        patched_prompt, result = injector.inject(str(prompt_value), task_name)
        return patched_prompt, result

    # 消息列表形态:dict 拷贝,其余原样
    patched_messages = [
        dict(message) if isinstance(message, dict) else message for message in messages
    ]
    # 只注入第一条 user 消息
    for index, message in enumerate(patched_messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).strip().lower() != "user":
            continue
        patched_content, result = injector.inject(
            str(message.get("content", "")),
            task_name,
        )
        patched_messages[index] = {
            **message,
            "content": patched_content,
        }
        return patched_messages, result

    # 列表里没有 user 消息:退回整段注入
    patched_prompt, result = injector.inject(str(prompt_value), task_name)
    return patched_prompt, result


def _maybe_patch_eval_data_with_task_skill_guidance(
    cfg: DictConfig,
    *,
    eval_data_path: Path,
    hydra_dir: Path,
) -> Path:
    """按需给每行样本注入"任务专属技能指导",输出补丁后的 parquet;未启用则原样返回。"""
    # 读取技能指导配置段
    guidance_cfg = OmegaConf.select(cfg, "data.task_skill_guidance") or {}
    if not bool(guidance_cfg.get("enabled", False)):
        return eval_data_path

    # 技能文档目录(相对路径锚定仓库根)
    skills_dir = _resolve_repo_path(str(guidance_cfg.get("skills_dir") or "").strip())
    if skills_dir is None:
        raise ValueError("data.task_skill_guidance.skills_dir is empty")

    # 加载任务名 -> 技能文档映射
    skill_map = load_task_skill_map(
        skills_dir,
        recursive=bool(guidance_cfg.get("recursive", True)),
    )
    if not skill_map:
        raise ValueError(f"No task skill Markdown files found in {skills_dir}")

    # 构造注入器(标题/引言/严格模式可配)
    injector = TaskSkillGuidanceInjector(
        skill_map=skill_map,
        heading=str(
            guidance_cfg.get(
                "heading",
                "Task-Specific Skill Reference:",
            )
        ),
        intro=str(guidance_cfg.get("intro", "")),
        strict=bool(guidance_cfg.get("strict", False)),
    )
    # 输入/元数据列名
    input_key = str(cfg.data.input_key)
    metadata_key = str(cfg.data.metadata_key)
    # 输出文件名与路径
    output_filename = str(
        guidance_cfg.get(
            "output_filename",
            f"{eval_data_path.stem}_task_skill_guidance{eval_data_path.suffix}",
        )
    )
    output_path = hydra_dir / output_filename

    # 读入评测数据
    dataframe = pd.read_parquet(eval_data_path)
    changed = 0
    skipped_existing = 0
    missing: set[str] = set()
    patched_prompts: list[Any] = []
    # 逐行注入
    for index, row in dataframe.iterrows():
        # 从元数据提取任务名
        task_name = _metadata_task_name(row.get(metadata_key))
        if not task_name:
            # 无任务名:记为缺失;strict 模式直接报错
            task_name = f"<row:{index}>"
            missing.add(task_name)
            if injector.strict:
                raise ValueError(f"Missing task name in metadata row: {index}")
            patched_prompts.append(row.get(input_key))
            continue

        # 注入技能指导
        patched_prompt, result = _inject_task_skill_into_prompt_value(
            row.get(input_key),
            task_name=task_name,
            injector=injector,
        )
        patched_prompts.append(patched_prompt)
        if result.changed:
            changed += 1
        if result.skipped_existing:
            skipped_existing += 1
        if result.missing:
            missing.add(result.task_name)

    # 写出补丁后的 parquet(只替换输入列)
    patched_dataframe = dataframe.copy()
    patched_dataframe[input_key] = patched_prompts
    patched_dataframe.to_parquet(output_path, index=False)
    logger.info(
        "Task skill guidance patched eval_data: %s -> %s "
        "(patched_rows=%s/%s, skipped_existing=%s, missing_tasks=%s)",
        eval_data_path,
        output_path,
        changed,
        len(dataframe),
        skipped_existing,
        sorted(missing),
    )
    return output_path


@hydra.main(
    config_path=f"{TTS_SEARCH_DIR}/configs",
    config_name="experiment/openmle_evo",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """评估主流程:增强数据 -> 落盘两份配置 -> 依次执行任务构建与 runner。"""
    # 环境准备:加载 .env、关代理、设日志级别
    load_dotenv()
    disable_proxy_env()
    logging.basicConfig(level=logging.INFO)

    # 输出目录与 hydra 子目录
    output_dir = Path(str(cfg.output_dir)).resolve()
    hydra_dir = output_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    # airaevo 任务构建产物根目录
    task_root = hydra_dir / "airaevo_tasks"
    # 第一步:验证集切分指导补丁
    eval_data_path = _maybe_patch_eval_data_with_validation_guidance(
        cfg,
        hydra_dir=hydra_dir,
    )
    # Patch once before task preparation so all workers share one prompt.
    # 第二步:任务技能指导补丁(在任务构建前只做一次,保证所有 worker 共用同一份 prompt)
    eval_data_path = _maybe_patch_eval_data_with_task_skill_guidance(
        cfg,
        eval_data_path=eval_data_path,
        hydra_dir=hydra_dir,
    )

    # 读取 runner 段配置与首个模型
    search_runner_cfg = dict(_to_plain(cfg.search.runner) or {})
    first_model = _first_model_payload(cfg)
    litellm_params = dict(first_model.get("litellm_params") or {})
    # 抽取生成参数
    llm_generation_kwargs = _generation_kwargs_from_litellm_params(litellm_params)
    # 每步候选数(默认 1)
    candidates_per_step = int(getattr(cfg, "candidates_per_step", 1) or 1)
    # 默认任务根(aira-evo 的 mle_bench 示例)
    default_task_root = str(
        search_runner_cfg.get("task_root", "third_party/aira-evo/examples/mle_bench")
    )
    # 包根:由任务根去掉 examples/mle_bench 后缀推断
    package_root_value = str(
        search_runner_cfg.get(
            "package_root",
            default_task_root.removesuffix("/examples/mle_bench"),
        )
    )
    package_root = Path(package_root_value)
    if not package_root.is_absolute():
        package_root = REPO_ROOT / package_root_value

    # 组装"任务构建"配置载荷:数据段 + 沙箱段
    prepare_payload = {
        "data": {
            # 补丁后的评测数据
            "eval_data": str(eval_data_path),
            # 排行榜目录(可为 None)
            "leaderboard_dir": (
                str(cfg.data.leaderboard_dir)
                if cfg.data.leaderboard_dir is not None
                else None
            ),
            # 提交目录
            "submit_dir": str(
                search_runner_cfg.get("submit_dir", "MLTasks/MLE-Bench-Lite-Modified")
            ),
            # 提交数据根(可为 None)
            "submit_data_dir_root": (
                str(search_runner_cfg["submit_data_dir_root"])
                if search_runner_cfg.get("submit_data_dir_root") is not None
                else None
            ),
            # 评估协议
            "evaluation_protocol": str(
                search_runner_cfg.get("evaluation_protocol", "legacy")
            ),
            # 验证/测试切分与选择分来源、提交超时与次数
            "validation_eval_split": search_runner_cfg.get("validation_eval_split"),
            "test_eval_split": search_runner_cfg.get("test_eval_split"),
            "selection_score_source": search_runner_cfg.get("selection_score_source"),
            "submit_job_timeout": search_runner_cfg.get("submit_job_timeout"),
            "submit_wait_timeout": search_runner_cfg.get("submit_wait_timeout"),
            "submit_repeats": int(search_runner_cfg.get("submit_repeats", 1) or 1),
            # 列名约定
            "input_key": str(cfg.data.input_key),
            "metadata_key": str(cfg.data.metadata_key),
        },
        "sandbox": {
            # CPU/GPU 两类资源的沙箱地址映射(runner 覆盖优先)
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
            # 作业/等待超时与轮询间隔
            "job_timeout": int(cfg.sandbox.job_timeout),
            "wait_timeout": int(cfg.sandbox.wait_timeout),
            "poll_interval": int(cfg.sandbox.poll_interval),
            # 分数->奖励整形与"干净日志分"开关
            "use_score2reward": bool(cfg.sandbox.use_score2reward),
            "use_clear_run_log_score": bool(
                getattr(cfg.sandbox, "use_clear_run_log_score", False)
            ),
        },
    }

    # 组装"runner"配置载荷:实验元信息 + 并发 + LLM + 求解器等
    runner_payload = {
        # 实验名
        "experiment_name": str(cfg.experiment_name),
        # 输出目录
        "output_dir": str(output_dir),
        # 随机种子
        "seed": int(cfg.seed),
        # 总时间预算
        "time_budget": cfg.time_budget,
        # 模型+沙箱时间预算(可为 None)
        "model_plus_sandbox_time_budget": getattr(
            cfg, "model_plus_sandbox_time_budget", None
        ),
        # 每任务采样数
        "n_samples_per_task": int(cfg.n_samples_per_task),
        # 每步候选数
        "candidates_per_step": candidates_per_step,
        # 任务并发(runner 段可覆盖)
        "task_concurrency": (
            int(search_runner_cfg["task_concurrency"])
            if search_runner_cfg.get("task_concurrency") is not None
            else None
        ),
        # LLM 并发与沙箱并发
        "llm_concurrency": int(cfg.llm_concurrency),
        "sandbox_concurrency": int(cfg.sandbox.concurrency),
        # 严格续跑
        "strict_resume": bool(search_runner_cfg.get("strict_resume", False)),
        # 任务根与任务清单
        "task_root": str(task_root),
        "task_list": list(search_runner_cfg.get("task_list", [])),
        # 排行榜目录
        "leaderboard_dir": (
            str(cfg.data.leaderboard_dir)
            if cfg.data.leaderboard_dir is not None
            else None
        ),
        # LLM 配置:翻译自 litellm 首个模型(密钥留在本进程,不下发)
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
        # 求解器/解释器/日志三段原样透传
        "solver": dict(search_runner_cfg.get("solver") or {}),
        "interpreter": dict(search_runner_cfg.get("interpreter") or {}),
        "logger": dict(search_runner_cfg.get("logger") or {}),
        # runner 侧 hydra 运行参数(固定输出到同一 .hydra)
        "hydra": {
            "run": {"dir": f"{output_dir}/.hydra"},
            "output_subdir": ".",
            "job": {"chdir": False},
        },
    }
    # 最大步数映射为求解器的 step_limit
    if cfg.max_steps is not None:
        runner_payload["solver"]["step_limit"] = int(cfg.max_steps)
    # 多候选时映射为每代个体数
    if candidates_per_step > 1:
        runner_payload["solver"]["individuals_per_generation"] = candidates_per_step

    # 落盘两份配置(供复现与子进程读取)
    prepare_config_path = hydra_dir / "airaevo_prepare_config.yaml"
    runner_config_path = hydra_dir / "airaevo_runner_config.yaml"
    _write_yaml(prepare_config_path, prepare_payload)
    _write_yaml(runner_config_path, runner_payload)

    # 第二阶段(子进程):构建 airaevo 任务包
    build_tasks_cmd = [
        sys.executable,
        str(package_root / "examples" / "mle_bench" / "build_tasks.py"),
        "--config",
        str(prepare_config_path),
        "--output-root",
        str(task_root),
    ]
    logger.info("Building AIRA Dojo tasks with %s", prepare_config_path)
    subprocess.run(build_tasks_cmd, check=True, cwd=REPO_ROOT)

    # 第三阶段(子进程):执行 aira-evo 搜索评估
    runner_cmd = [
        sys.executable,
        str(package_root / "examples" / "mle_bench" / "runner.py"),
        "--config-path",
        str(runner_config_path.parent),
        "--config-name",
        runner_config_path.stem,
    ]
    logger.info("Running AIRA Dojo evaluation with %s", runner_config_path)
    subprocess.run(runner_cmd, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
