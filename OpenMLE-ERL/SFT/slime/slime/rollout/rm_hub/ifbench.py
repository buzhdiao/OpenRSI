# 使类型标注支持延迟求值（兼容低版本 Python 的新式标注语法）
from __future__ import annotations

# 引入 importlib，用于动态导入 IFBench 的 evaluation_lib
import importlib
# 引入日志模块
import logging
# 引入 os，用于读写 PYTHONPATH 环境变量
import os
# 引入 subprocess，用于执行 git clone 与 pip install
import subprocess
# 引入 sys，用于操作 sys.path 与获取当前解释器路径
import sys
# 引入 Sequence 类型标注
from collections.abc import Sequence
# 引入 Path，用于路径拼接与存在性判断
from pathlib import Path
# 引入 Any 类型标注
from typing import Any

# 获取本模块专属 logger
logger = logging.getLogger(__name__)

# 工作区根目录：本文件向上三级（slime/rollout/rm_hub -> slime -> SFT 根）
_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
# 工作区的上级目录（IFBench 仓库将被克隆到此处）
_WORKSPACE_PARENT = _WORKSPACE_ROOT.parent
# 本地保存的 IFBench 依赖清单路径
_LOCAL_IFBENCH_REQUIREMENTS = _WORKSPACE_ROOT / "examples" / "eval_multi_task" / "requirements_ifbench.txt"


# 确保 IFBench 仓库就位（必要时自动克隆）并加入模块搜索路径
def _ensure_ifbench_repo() -> Path:
    """Clone IFBench repo if needed and ensure it is available on sys.path."""

    # 目标仓库目录：工作区上级目录下的 IFBench
    repo_path = _WORKSPACE_PARENT / "IFBench"

    # 仓库不存在时自动从 GitHub 克隆
    if not repo_path.exists():
        # 组装 git clone 命令
        clone_cmd = ["git", "clone", "https://github.com/allenai/IFBench.git", str(repo_path)]
        # 执行克隆并捕获输出
        try:
            subprocess.run(clone_cmd, check=True, capture_output=True)
        # 克隆失败：抛出 ImportError 提示手动克隆
        except Exception as exc:
            raise ImportError(
                "Unable to automatically clone IFBench. Please clone "
                "https://github.com/allenai/IFBench.git into the repo root."
            ) from exc

    # 仓库路径字符串形式
    repo_str = str(repo_path)
    # 尚不在 sys.path 时插入到最前，保证优先导入
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    # 读取当前 PYTHONPATH 环境变量
    current_pythonpath = os.environ.get("PYTHONPATH")
    # 尚未设置时直接设为仓库路径
    if current_pythonpath is None:
        os.environ["PYTHONPATH"] = repo_str
    # 已设置但不含仓库路径时前置追加（子进程也能找到）
    elif repo_str not in current_pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = os.pathsep.join([repo_str, current_pythonpath])

    # 返回仓库路径
    return repo_path


# 首次使用时安装 IFBench 的 Python 依赖（用哨兵文件保证只装一次）
def _ensure_ifbench_dependencies(repo_path: Path) -> None:
    """Install IFBench requirements the first time the module is imported."""

    # 依赖清单文件路径
    requirements_file = _LOCAL_IFBENCH_REQUIREMENTS

    # 清单不存在：跳过安装并记录调试日志
    if not requirements_file.exists():
        logger.debug("Local IFBench requirements file not found at %s; skipping install.", requirements_file)
        return

    # 哨兵文件：标记依赖已安装过
    sentinel = repo_path / ".deps_installed"
    # 已有哨兵则不再重复安装
    if sentinel.exists():
        return

    # 组装 pip install 命令（使用当前解释器）
    install_cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
    # 执行安装
    try:
        subprocess.run(install_cmd, check=True)
    # 安装失败：仅记录警告（不中断主流程）
    except Exception as exc:
        logger.warning("Failed to install IFBench dependencies automatically: %s", exc)
    # 安装成功：写入哨兵文件
    else:
        sentinel.write_text("installed\n")


# 懒加载并返回 IFBench 的 evaluation_lib 模块
def _load_evaluation_lib():
    # 先确保仓库就位
    repo_path = _ensure_ifbench_repo()
    # 尝试直接导入
    try:
        return importlib.import_module("evaluation_lib")
    # 导入失败通常是依赖缺失：先装依赖再重试
    except ImportError:
        _ensure_ifbench_dependencies(repo_path)
        return importlib.import_module("evaluation_lib")


# 模块导入时即完成 evaluation_lib 的加载
evaluation_lib = _load_evaluation_lib()
# 从库中引入输入样例数据结构
InputExample = evaluation_lib.InputExample


# JSON 字典类型别名
JsonDict = dict[str, Any]
# 指令 kwargs 字典类型别名（值允许标量或 None）
KwargsDict = dict[str, str | int | float | None]


# 规范化指令 ID 列表：只保留非空字符串
def _normalize_instruction_ids(raw_ids: Sequence[Any]) -> list[str]:
    """Ensure instruction identifiers are clean strings."""

    # 规范化结果列表
    normalized: list[str] = []
    # 逐条处理原始 ID（None 视为空列表）
    for entry in raw_ids or []:
        # 跳过 None 项
        if entry is None:
            continue
        # 转字符串并去首尾空白
        text = str(entry).strip()
        # 空字符串跳过
        if not text:
            continue
        # 收集有效 ID
        normalized.append(text)
    # 返回规范化列表
    return normalized


# 把存储形式的 kwargs 转成 IFBench 期望的"每条指令一个字典"的列表结构
def _coerce_kwargs_list(
    raw_kwargs: Any,
    num_instructions: int,
) -> list[KwargsDict]:
    """Convert stored kwargs into the list structure expected by IFBench."""

    # 输入本身是列表的情形：逐项规整
    if isinstance(raw_kwargs, list):
        # 处理结果列表
        processed: list[KwargsDict] = []
        # 逐项检查
        for entry in raw_kwargs:
            # 字典项浅拷贝入列
            if isinstance(entry, dict):
                processed.append(dict(entry))
            # 非字典项用空字典占位
            else:
                processed.append({})
    # 输入是单个字典的情形：复制给每条指令
    elif isinstance(raw_kwargs, dict):
        processed = [dict(raw_kwargs) for _ in range(num_instructions)]
    # 其他类型（含 None）：全部用空字典
    else:
        processed = [{} for _ in range(num_instructions)]

    # 条目不足时用最后一个条目补齐到指令数
    if len(processed) < num_instructions:
        # 取末位条目作为补齐模板（空列表则用空字典）
        tail = processed[-1] if processed else {}
        # 复制模板补足差额
        processed.extend([dict(tail) for _ in range(num_instructions - len(processed))])
    # 条目超出时截断到指令数
    elif len(processed) > num_instructions:
        processed = processed[:num_instructions]

    # Remove explicit None values to match official preprocessing.
    # 最终清洗：剔除值为 None 的键，与官方预处理保持一致
    sanitized: list[KwargsDict] = []
    # 逐条过滤
    for entry in processed:
        # 仅保留值非 None 的键值对
        sanitized.append({k: v for k, v in entry.items() if v is not None})
    # 返回清洗后的 kwargs 列表
    return sanitized


# 从样本元数据构建 IFBench 的 InputExample 对象
def _build_input_example(metadata: JsonDict) -> InputExample | None:
    # 规范化指令 ID 列表（缺失时用空列表）
    instruction_ids = _normalize_instruction_ids(metadata.get("instruction_id_list") or [])
    # 没有任何指令 ID：无法构造评测样例
    if not instruction_ids:
        logger.debug("Missing instruction identifiers in metadata: %s", metadata)
        return None

    # 取原始 prompt 文本
    prompt_text = metadata.get("prompt_text")
    # 缺失时置空串
    if prompt_text is None:
        prompt_text = ""
    # 存在则统一转字符串
    else:
        prompt_text = str(prompt_text)

    # 取原始 kwargs（可能是列表/字典/None 等多种形态）
    raw_kwargs = metadata.get("kwargs")
    # 规整成与指令数等长的 kwargs 列表
    kwargs_list = _coerce_kwargs_list(raw_kwargs, len(instruction_ids))

    # 构造并返回 InputExample（record_id 缺省为 0）
    return InputExample(
        key=int(metadata.get("record_id") or 0),
        instruction_id_list=instruction_ids,
        prompt=prompt_text,
        kwargs=kwargs_list,
    )


# IFBench 奖励计算：按官方严格规则评测指令遵循，全部满足给 1.0
def compute_ifbench_reward(response: str, label: Any, metadata: JsonDict | None = None) -> float:
    """Score a model response using the official IFBench rules."""

    # 未提供元数据：无法评测，判 0 分
    if metadata is None:
        logger.debug("No metadata provided for IFBench scoring.")
        return 0.0

    # 响应为 None：判 0 分
    if response is None:
        return 0.0

    # 由元数据构造评测输入样例
    inp = _build_input_example(metadata)
    # 构造失败（缺指令 ID 等）：判 0 分
    if inp is None:
        return 0.0

    # 组装 prompt 到 response 的映射（官方评测接口要求的输入形态）
    prompt_to_response = {inp.prompt: str(response or "")}
    # 调用官方严格模式评测：逐条校验所有指令约束
    output = evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
    # 全部指令都满足给 1.0，否则 0.0
    return 1.0 if output.follow_all_instructions else 0.0
