# 中文说明:本脚本是 NatureBench 评估的编排入口:把 tts_search 的 Hydra 实验配置
# "降维"成 aira-evo 框架可消费的两份 YAML(任务准备配置 + 搜索 runner 配置),
# 然后依次子进程调用 build_tasks.py(构建 NatureBench 任务包)与
# mle_bench/runner.py(执行进化搜索评估),全程在本仓库工作目录下运行。
# 位置:OpenMLE-Evo/scripts 下,是 Evo 侧对接 NatureBench 基准的 glue 层。
# 关键设计:配置转换时对敏感字段(api_key)先落到 .hydra 目录再统一脱敏重写,
# 避免密钥落盘;执行超时等字段按 data -> sandbox -> solver 的优先级回退取值。

# 延迟注解:类型标注运行期不求值
from __future__ import annotations

# logging:脚本自身的日志
import logging
# os:复制环境变量、拼接 PYTHONPATH
import os
# subprocess:调用 build_tasks 与 runner 两个子进程
import subprocess
# sys:把仓库根目录加入 import 路径、取当前解释器路径
import sys
# Path:各类配置/输出路径构造
from pathlib import Path
# Any:配置值类型(可能是 OmegaConf 节点或普通值)
from typing import Any

# hydra:声明 CLI 主入口并加载默认配置
import hydra
# yaml:安全序列化 payload 到 YAML 文件
import yaml
# dotenv:从 .env 文件加载环境变量(本地密钥注入)
from dotenv import load_dotenv
# OmegaConf:读写/查询 Hydra 配置
from omegaconf import DictConfig, OmegaConf

# 仓库根目录(OpenMLE-Evo):scripts 的上一级
REPO_ROOT = Path(__file__).resolve().parents[1]
# 保证仓库根在 sys.path 首位,便于直接导入 tts_search 包
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# tts_search 目录路径(configs 默认目录等)
TTS_SEARCH_DIR = REPO_ROOT / "tts_search"
# 本模块 logger
logger = logging.getLogger(__name__)

# YAML 脱敏工具:重写 yaml 文件,抹掉密钥等敏感字段
from tts_search.config_security import redact_sensitive_yaml_file  # noqa: E402
# 代理清理:评估前禁用代理环境变量,避免本地服务被代理劫持
from tts_search.eval_network import disable_proxy_env  # noqa: E402


def _to_plain(value: Any) -> Any:
    """把 OmegaConf 配置节点转成普通 Python 容器(带插值解析);非配置原样返回。"""
    # 判断是否为 OmegaConf 配置对象
    if OmegaConf.is_config(value):
        # to_container 且 resolve=True:展开 ${} 插值后转 list/dict
        return OmegaConf.to_container(value, resolve=True)
    # 普通值直接返回
    return value


def _first_model_payload(cfg: DictConfig) -> dict[str, Any]:
    """取 litellm 模型列表的第一项作为评估用主模型配置。"""
    # 把 model_list 转成普通列表(可能为 None 则兜底空列表)
    model_list = list(_to_plain(cfg.litellm.model_list) or [])
    # 模型列表为空则无法评估,直接报错
    if not model_list:
        raise ValueError("cfg.litellm.model_list is empty")
    # 只取第一项并拷贝为 dict(避免后续修改污染原配置)
    return dict(model_list[0])


def _generation_kwargs_from_litellm_params(
    litellm_params: dict[str, Any],
) -> dict[str, Any]:
    """从 litellm_params 中筛出与"生成过程"相关的参数,作为 runner 的 generation_kwargs。

    路由/鉴权类参数(base_url/api_key 等)被排除,只保留真正影响采样的字段。
    """
    # 拷贝并转普通 dict(None 兜底为空)
    litellm_params = dict(_to_plain(litellm_params) or {})
    # 允许透传的生成参数白名单(温度/截断/采样/工具等)
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
    # 只挑白名单内且值非 None 的参数,组成 generation_kwargs
    generation_kwargs = {
        key: value
        for key, value in litellm_params.items()
        if key in generation_keys and value is not None
    }
    # extra_body 整体透传(服务商私有参数通道),同样转普通 dict
    if litellm_params.get("extra_body") is not None:
        generation_kwargs["extra_body"] = dict(_to_plain(litellm_params["extra_body"]))
    # 返回筛选后的生成参数
    return generation_kwargs


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """把 dict 安全序列化为 YAML 文件(允许非 ASCII、保持键序)。"""
    # safe_dump 只输出纯数据;allow_unicode 保留中文;sort_keys=False 保持书写顺序便于 diff
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _subprocess_env(*extra_paths: Path) -> dict[str, str]:
    """构造子进程环境变量:PYTHONPATH 前置仓库根与给定的额外路径。"""
    # 复制当前环境变量
    env = os.environ.copy()
    # PYTHONPATH 第一项:仓库根(保证子进程能导入 tts_search 等)
    python_paths = [str(REPO_ROOT)]
    # 追加额外路径,但只加实际存在的目录
    python_paths.extend(str(path) for path in extra_paths if path.exists())
    # 原有 PYTHONPATH 追加在最后,保留用户自定义路径
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    # 用平台分隔符(POSIX 冒号/Windows 分号)拼回 PYTHONPATH
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    # 返回修改后的环境
    return env


def _data_value(cfg: DictConfig, key: str, default: Any = None) -> Any:
    """按 data.<key> 取配置值(带默认值),并转为普通 Python 值。"""
    # OmegaConf.select 支持点路径与缺省;_to_plain 归一化
    return _to_plain(OmegaConf.select(cfg, f"data.{key}", default=default))


def _build_payloads(
    cfg: DictConfig,
    *,
    output_dir: Path,
    task_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """把 Hydra 实验配置拆成 (prepare_payload, runner_payload) 两份普通 dict。

    prepare_payload 交给 build_tasks 构建任务;runner_payload 交给搜索 runner。
    """
    # search.runner 段转普通 dict
    search_runner_cfg = dict(_to_plain(cfg.search.runner) or {})
    # 其中 solver 子段(搜索策略配置)
    solver_payload = dict(search_runner_cfg.get("solver") or {})
    # 主模型配置(取 litellm 列表首项)
    first_model = _first_model_payload(cfg)
    # 模型的 litellm_params(base_url/api_key/生成参数所在)
    litellm_params = dict(first_model.get("litellm_params") or {})
    # 筛出纯生成参数
    llm_generation_kwargs = _generation_kwargs_from_litellm_params(litellm_params)
    # 每步候选数,缺省 1(getattr 兼容配置缺字段;or 1 防止 0 值)
    candidates_per_step = int(getattr(cfg, "candidates_per_step", 1) or 1)
    # 执行超时的多级回退:先取 data.execution_timeout
    execution_timeout = _data_value(cfg, "execution_timeout")
    # 再回退 sandbox.job_timeout
    if execution_timeout is None:
        execution_timeout = OmegaConf.select(cfg, "sandbox.job_timeout")
    # 最后回退 solver 段内的 execution_timeout
    if execution_timeout is None:
        execution_timeout = solver_payload.get("execution_timeout")

    # 组装任务准备配置(全部字段显式罗列,便于审查与 diff)
    prepare_payload = {
        "data": {
            # NatureBench 基准数据根目录
            "naturebench_root": str(_data_value(cfg, "naturebench_root")),
            # 任务输出根目录
            "tasks_root": _data_value(cfg, "tasks_root"),
            # 任务名列表(可选,过滤要跑的任务)
            "task_list": list(_data_value(cfg, "task_list", []) or []),
            # 任务集合清单文件路径
            "task_set_path": _data_value(cfg, "task_set_path"),
            # 评估服务地址,默认本地 8321 端口
            "eval_service_url": str(
                _data_value(cfg, "eval_service_url", "http://127.0.0.1:8321")
            ),
            # 批次名,默认取实验名
            "batch_name": str(_data_value(cfg, "batch_name", str(cfg.experiment_name))),
            # 任务执行模式:默认 docker(也可 local/scm)
            "execution_mode": str(_data_value(cfg, "execution_mode", "docker")),
            # 本地/容器工作区根目录
            "workspace_root": _data_value(cfg, "workspace_root"),
            # docker 镜像名
            "docker_image": _data_value(cfg, "docker_image"),
            # GPU 设备指定
            "gpu_devices": _data_value(cfg, "gpu_devices"),
            # 本地 python 解释器路径
            "local_python": _data_value(cfg, "local_python"),
            # 本地 conda 环境名
            "local_conda_env": _data_value(cfg, "local_conda_env"),
            # conda 可执行文件,默认 conda
            "local_conda_executable": _data_value(
                cfg, "local_conda_executable", "conda"
            ),
            # 本地模式终止进程前的宽限秒数
            "local_terminate_grace_seconds": _data_value(
                cfg, "local_terminate_grace_seconds", 5
            ),
            # 候选环境白名单(限制任务可用的环境)
            "candidate_env_allowlist": _data_value(cfg, "candidate_env_allowlist", []),
            # SCM(源码管理)模式:主机名
            "scm_host": _data_value(cfg, "scm_host"),
            # SCM 工作区根目录
            "scm_workspace_root": _data_value(cfg, "scm_workspace_root"),
            # SCM 任务根目录列表
            "scm_task_roots": _data_value(cfg, "scm_task_roots"),
            # SCM 任务集所在主机
            "scm_task_set_host": _data_value(cfg, "scm_task_set_host"),
            # SCM 任务集根目录
            "scm_task_set_root": _data_value(cfg, "scm_task_set_root"),
            # SCM 资源行(资源限额描述)
            "scm_resource_lines": _data_value(cfg, "scm_resource_lines"),
            # SCM 模式的评估服务地址
            "scm_eval_service_url": _data_value(cfg, "scm_eval_service_url"),
            # 容器内可见的评估服务地址
            "scm_container_eval_service_url": _data_value(
                cfg, "scm_container_eval_service_url"
            ),
            # 候选预检开关(提交前先做基本校验)
            "candidate_preflight": _data_value(cfg, "candidate_preflight", True),
            # 预检是否包含 import 试运行
            "candidate_preflight_imports": _data_value(
                cfg, "candidate_preflight_imports", True
            ),
            # 预检超时秒数,默认 45
            "candidate_preflight_timeout": _data_value(
                cfg, "candidate_preflight_timeout", 45
            ),
            # SCM 模式等 GPU 的超时
            "scm_gpu_wait_timeout": _data_value(cfg, "scm_gpu_wait_timeout"),
            # 可见数据分析目录(候选可读的资料区)
            "visible_data_analysis_root": _data_value(
                cfg, "visible_data_analysis_root"
            ),
            # 提交重复次数,未配置时为 1(None 表示未配置)
            "submit_repeats": int(
                1
                if _data_value(cfg, "submit_repeats", None) is None
                else _data_value(cfg, "submit_repeats")
            ),
            # 单次评估超时,默认 3600 秒(0/None 兜底回默认)
            "eval_timeout": int(_data_value(cfg, "eval_timeout", 3600) or 3600),
            # 公共系统提示词
            "public_system_prompt": str(_data_value(cfg, "public_system_prompt", "")),
            # 公共用户提示词
            "public_user_prompt": str(_data_value(cfg, "public_user_prompt", "")),
        }
    }
    # 仅当三级回退都拿到了执行超时才写入(保持配置最小化)
    if execution_timeout is not None:
        prepare_payload["data"]["execution_timeout"] = int(execution_timeout)

    # 组装搜索 runner 配置
    runner_payload = {
        # 实验名(日志/排行榜目录命名用)
        "experiment_name": str(cfg.experiment_name),
        # 输出目录(绝对路径)
        "output_dir": str(output_dir),
        # 随机种子
        "seed": int(cfg.seed),
        # 时间预算(秒)
        "time_budget": cfg.time_budget,
        # 模型+沙箱合计时间预算(可选字段)
        "model_plus_sandbox_time_budget": getattr(
            cfg,
            "model_plus_sandbox_time_budget",
            None,
        ),
        # 每任务采样数
        "n_samples_per_task": int(cfg.n_samples_per_task),
        # 每步候选数
        "candidates_per_step": candidates_per_step,
        # 任务级并发:未配置则显式置 None(runner 自行决定)
        "task_concurrency": (
            int(search_runner_cfg["task_concurrency"])
            if search_runner_cfg.get("task_concurrency") is not None
            else None
        ),
        # LLM 请求并发
        "llm_concurrency": int(cfg.llm_concurrency),
        # 沙箱并发
        "sandbox_concurrency": int(cfg.sandbox.concurrency),
        # 严格续跑开关(默认 False)
        "strict_resume": bool(search_runner_cfg.get("strict_resume", False)),
        # 构建好的任务根目录(指向 .hydra/naturebench_tasks)
        "task_root": str(task_root),
        # runner 级任务列表覆盖
        "task_list": list(search_runner_cfg.get("task_list", [])),
        # 排行榜目录置 None(NatureBench 评估不写排行榜)
        "leaderboard_dir": None,
        # LLM 接入配置
        "llm": {
            # API 适配器类型,默认 litellm
            "api": str(search_runner_cfg.get("llm", {}).get("api", "litellm")),
            # 模型名(取自 litellm 列表首项)
            "model_id": str(first_model.get("model_name")),
            # 推理服务 base_url
            "base_url": str(litellm_params.get("base_url")),
            # 生成参数(已按白名单筛选)
            "generation_kwargs": llm_generation_kwargs,
            # 是否走 Azure 客户端,默认 False
            "use_azure_client": bool(
                search_runner_cfg.get("llm", {}).get("use_azure_client", False)
            ),
            # 供应商标签,默认 selfhosted
            "provider": str(
                search_runner_cfg.get("llm", {}).get("provider", "selfhosted")
            ),
        },
        # 搜索策略配置原样透传
        "solver": solver_payload,
        # 解释器配置透传
        "interpreter": dict(search_runner_cfg.get("interpreter") or {}),
        # 日志器配置透传
        "logger": dict(search_runner_cfg.get("logger") or {}),
        # 给 runner 内部 hydra 初始化用的配置(输出到 .hydra、禁止 chdir)
        "hydra": {
            "run": {"dir": f"{output_dir}/.hydra"},
            "output_subdir": ".",
            "job": {"chdir": False},
        },
    }
    # 配置了 max_steps 时映射为 solver 的 step_limit
    if cfg.max_steps is not None:
        runner_payload["solver"]["step_limit"] = int(cfg.max_steps)
    # 多候选模式下把候选数映射为 solver 的 individuals_per_generation
    if candidates_per_step > 1:
        runner_payload["solver"]["individuals_per_generation"] = candidates_per_step
    # 返回两份 payload
    return prepare_payload, runner_payload


# hydra CLI 主入口:默认加载 tts_search/configs 下的 naturebench_smoke 实验配置
@hydra.main(
    config_path=f"{TTS_SEARCH_DIR}/configs",
    config_name="experiment/naturebench_smoke",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """NatureBench 评估主流程:配置落盘脱敏 -> 构建任务 -> 运行搜索评估。"""
    # 加载 .env 环境变量(本地密钥/地址)
    load_dotenv()
    # 禁用代理,确保本地评估服务直连
    disable_proxy_env()
    # 日志级别 INFO,便于观察子流程进度
    logging.basicConfig(level=logging.INFO)

    # 输出目录取绝对路径
    output_dir = Path(str(cfg.output_dir)).resolve()
    # hydra 输出子目录(配置与日志都在这)
    hydra_dir = output_dir / ".hydra"
    # 确保该目录存在
    hydra_dir.mkdir(parents=True, exist_ok=True)
    # NatureBench 任务包的输出根:置于 .hydra/naturebench_tasks
    task_root = hydra_dir / "naturebench_tasks"

    # 读取 search.runner 段
    search_runner_cfg = dict(_to_plain(cfg.search.runner) or {})
    # aira-evo 包根目录,默认 third_party/aira-evo
    package_root_value = str(
        search_runner_cfg.get("package_root", "third_party/aira-evo")
    )
    # 转成 Path
    package_root = Path(package_root_value)
    # 相对路径则相对仓库根解析
    if not package_root.is_absolute():
        package_root = REPO_ROOT / package_root_value

    # 生成两份 payload(任务准备 + 搜索 runner)
    prepare_payload, runner_payload = _build_payloads(
        cfg,
        output_dir=output_dir,
        task_root=task_root,
    )
    # 再次取主模型配置,用于提取 api_key
    first_model = _first_model_payload(cfg)
    # 从 litellm_params 中取 api_key(可能为 None)
    llm_api_key = dict(first_model.get("litellm_params") or {}).get("api_key")
    # 两份配置的落盘路径
    prepare_config_path = hydra_dir / "naturebench_prepare_config.yaml"
    runner_config_path = hydra_dir / "naturebench_runner_config.yaml"
    # 写出准备配置
    _write_yaml(prepare_config_path, prepare_payload)
    # 写出 runner 配置
    _write_yaml(runner_config_path, runner_payload)
    # 遍历 .hydra 下所有 yaml(排序保证确定性),统一脱敏敏感字段
    for hydra_config_path in sorted(hydra_dir.glob("*.yaml")):
        redact_sensitive_yaml_file(hydra_config_path)

    # 组装 build_tasks 子进程命令:构建 NatureBench 的 AIRA-Evo 任务包
    build_tasks_cmd = [
        sys.executable,
        str(package_root / "examples" / "nature_bench" / "build_tasks.py"),
        "--config",
        str(prepare_config_path),
        "--output-root",
        str(task_root),
    ]
    # 日志提示开始构建
    logger.info("Building NatureBench AIRA-Evo tasks with %s", prepare_config_path)
    # 子进程环境:PYTHONPATH 前置 aira-evo 的 src 目录
    env = _subprocess_env(package_root / "src")
    # 阻塞运行构建命令,失败(check=True)即中断整个评估
    subprocess.run(build_tasks_cmd, check=True, cwd=REPO_ROOT, env=env)
    # 构建成功后把 LLM api_key 注入子进程环境(密钥只经环境变量传递,不落盘)
    if llm_api_key is not None:
        env["OPENMLE_LLM_API_KEY"] = str(llm_api_key)

    # 组装 runner 子进程命令:以 hydra 方式加载 runner 配置执行评估
    runner_cmd = [
        sys.executable,
        str(package_root / "examples" / "mle_bench" / "runner.py"),
        "--config-path",
        str(runner_config_path.parent),
        "--config-name",
        runner_config_path.stem,
    ]
    # 日志提示开始评估
    logger.info("Running NatureBench AIRA-Evo evaluation with %s", runner_config_path)
    # 阻塞运行评估;失败即以非零码退出(便于外层任务调度感知)
    subprocess.run(runner_cmd, check=True, cwd=REPO_ROOT, env=env)


# 直接运行脚本时进入 hydra 主入口
if __name__ == "__main__":
    main()
