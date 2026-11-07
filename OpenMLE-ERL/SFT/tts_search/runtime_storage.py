"""Runtime storage defaults for long-running evaluations and training."""
# 运行时存储默认配置模块：为长时间运行的评估/训练统一指定仓库本地的临时与缓存目录，
# 避免写入 /tmp 等易失位置，且所有路径均可被环境变量覆盖。

from __future__ import annotations

# 导入 os：读取/设置环境变量
import os
# 导入 Path：路径拼接与目录创建
from pathlib import Path


def _clean_run_id(value: str) -> str:
    # 清洗 run_id：去除首尾空白，并把路径分隔符/冒号/空格替换为下划线
    clean = value.strip().replace("/", "_").replace(":", "_").replace(" ", "_")
    # 清洗后为空则兜底为 "rollout"
    return clean or "rollout"


def configure_runtime_storage(default_run_id: str = "rollout") -> dict[str, str]:
    """Configure repository-local temp and cache paths unless overridden."""
    # 配置仓库本地的临时/缓存路径（除非环境变量已显式指定），返回最终生效的路径表

    # 默认存储根目录取本文件上两级目录下的 .runtime
    default_root = Path(__file__).resolve().parents[1] / ".runtime"
    # 允许用 OPENMLE_STORAGE_ROOT 覆盖根目录
    root = Path(os.environ.get("OPENMLE_STORAGE_ROOT", str(default_root)))
    # run_id 优先级：OPENMLE_RUN_ID > JOB_NAME > TASK_NAME > 默认值，并做清洗
    run_id = _clean_run_id(
        os.environ.get("OPENMLE_RUN_ID")
        or os.environ.get("JOB_NAME")
        or os.environ.get("TASK_NAME")
        or default_run_id
    )
    # 把根目录与 run_id 回写环境变量（已设置时不覆盖）
    os.environ.setdefault("OPENMLE_STORAGE_ROOT", str(root))
    os.environ.setdefault("OPENMLE_RUN_ID", run_id)

    # 各类临时/缓存目录的默认路径（按 run_id 隔离临时目录）
    defaults = {
        "TMPDIR": root / "tmp" / run_id,
        "XDG_CACHE_HOME": root / "cache" / "xdg",
        "HF_HOME": root / "cache" / "huggingface",
        "TORCH_HOME": root / "cache" / "torch",
        "PIP_CACHE_DIR": root / "cache" / "pip",
        "WANDB_DIR": root / "wandb" / run_id,
    }
    # HuggingFace hub 缓存放在 HF_HOME/hub 下
    defaults["HUGGINGFACE_HUB_CACHE"] = defaults["HF_HOME"] / "hub"
    # transformers 缓存放在 HF_HOME/transformers 下
    defaults["TRANSFORMERS_CACHE"] = defaults["HF_HOME"] / "transformers"

    # 逐项写入环境变量（仅当未设置时生效，尊重外部覆盖）
    for key, path in defaults.items():
        os.environ.setdefault(key, str(path))

    # 读取当前 Ray 临时目录配置
    ray_tmpdir = os.environ.get("RAY_TMPDIR", "")
    # 未允许临时目录 / 且 Ray 临时目录缺失或落在 /tmp/ 下时，改到持久化目录
    if os.environ.get("ALLOW_EPHEMERAL_TMP") != "1" and (
        not ray_tmpdir or ray_tmpdir.startswith("/tmp/")
    ):
        os.environ["RAY_TMPDIR"] = str(root / "ray" / run_id)

    # 汇总所有最终生效的路径（从环境变量回读，反映真实值）
    paths = {key: os.environ[key] for key in defaults}
    paths["RAY_TMPDIR"] = os.environ["RAY_TMPDIR"]
    # 确保每个路径目录都已创建（parents 递归创建，已存在则跳过）
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)

    # 返回最终路径表供调用方使用
    return paths
