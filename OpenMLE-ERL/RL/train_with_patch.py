#!/usr/bin/env python3
# =====================================================================
# 中文注释导读（OpenRSI / OpenMLE-RL 学习文档）：
# 本脚本是 OpenMLE-RL 在线强化学习的"标准训练入口"（train_with_patch）。
# 它做三件事：
#   1) 固定 Python 模块搜索路径，保证加载的是本仓库内 vendored 的 slime
#      （SLIME_ROOT = OpenMLE-ERL/RL/slime），而不是环境里可能存在的其他 slime；
#   2) 在任何 slime 模块被导入之前，先加载 patch_eval_logger 与 patch_rollout
#      两个"猴子补丁"模块——它们会替换 slime 的评估日志与 rollout 日志函数，
#      注入算子级（Draft/Improve/Debug/Crossover）奖励视图与程序库快照；
#   3) 以 exec 方式在当前命名空间执行 slime/train.py，启动真正的 RL 训练。
# 论文术语：execution-grounded RL——奖励来自沙箱真实执行结果（见 reward_func_utils），
# 而非模型自评；本入口本身不定义奖励，只负责让补丁在正确时机生效。
# =====================================================================
"""
Wrapper script for train.py that applies custom eval logger patch.
This ensures the monkey patch is applied before slime modules are loaded.
"""

import importlib.util  # 用于在钉扎路径后探测 slime 包的实际解析位置
import os  # 路径拼装与目录切换
import sys  # 操作 sys.path（模块搜索路径钉扎）

# 三级路径定位：脚本目录 -> slime 根目录（RL/slime）-> 项目根（OpenMLE-ERL 的上一级）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SLIME_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PROJECT_ROOT = os.path.dirname(SLIME_ROOT)


def _realpath(path: str) -> str:
    # 归一化为真实绝对路径（解析符号链接、统一大小写），用于可靠的前缀比较
    return os.path.normcase(os.path.realpath(path))


def _pin_local_repo_paths() -> None:
    # "钉扎"模块搜索路径：把本仓库的 RL/ 与 RL/slime/ 放到最前，
    # 并过滤掉 sys.path 中所有指向外部 slime 安装的条目——
    # 这防止 pip 安装的同名包或站点包目录抢先于 vendored slime 被导入。
    local_prefix = _realpath(PROJECT_ROOT)
    pinned_paths = [_realpath(SCRIPT_DIR), _realpath(SLIME_ROOT)]
    filtered_sys_path = []
    for entry in sys.path:
        candidate = entry or os.getcwd()
        try:
            candidate_real = _realpath(candidate)
        except OSError:
            filtered_sys_path.append(entry)
            continue
        # 剔除路径片段含 "/slime/" 却不属于本仓库的外部条目
        if f"{os.sep}slime{os.sep}" in f"{candidate_real}{os.sep}" and not candidate_real.startswith(local_prefix + os.sep):
            continue
        filtered_sys_path.append(entry)

    # 钉扎路径置前 + 去重，然后切换工作目录到 slime 根（train.py 依赖相对路径）
    sys.path[:] = pinned_paths + [entry for entry in filtered_sys_path if _realpath(entry or os.getcwd()) not in pinned_paths]
    os.chdir(SLIME_ROOT)
    spec = importlib.util.find_spec("slime")
    origin = spec.origin if spec is not None else "missing"
    # 打印解析结果，便于确认加载的是 vendored slime 而非外部安装
    print(f"[TRAIN_WITH_PATCH] Pinned repo root: {PROJECT_ROOT}")
    print(f"[TRAIN_WITH_PATCH] slime import resolution: {origin}")


_pin_local_repo_paths()

# Import and apply the patch BEFORE importing any slime modules
# 关键顺序：补丁必须先于 slime 模块加载，才能成功替换其模块级函数
print("=" * 60)
print("[TRAIN_WITH_PATCH] Applying custom eval logger patch...")
print("=" * 60)

try:
    import patch_eval_logger  # 评估日志补丁：把验证/评估指标接进 slime 日志流

    print("[TRAIN_WITH_PATCH] Patch module loaded successfully")
except Exception as e:
    # 补丁失败只告警不中断：训练仍可进行，但会缺失定制的评估日志
    print(f"[TRAIN_WITH_PATCH] Warning: Failed to load patch: {e}")
    import traceback

    traceback.print_exc()

print("[TRAIN_WITH_PATCH] Applying rollout patch...")
try:
    import patch_rollout  # rollout 补丁：算子级奖励视图、零方差检测、程序库快照

    print("[TRAIN_WITH_PATCH] Rollout patch module loaded successfully")
except Exception as e:
    print(f"[TRAIN_WITH_PATCH] Warning: Failed to load rollout patch: {e}")
    import traceback

    traceback.print_exc()

# Patch 2: SGLang rollout to pass evaluation parameter
# NOTE: This patch is no longer needed as the fix has been applied directly to the source code
# in slime/rollout/sglang_rollout.py to ensure it works in Ray worker processes
# 第三处补丁已直接改入 vendored 源码（sglang_rollout.py），无需再打
print("SGLang rollout patch is now applied directly in source code")

# Now import and run the actual train.py
print("[TRAIN_WITH_PATCH] Starting train.py...")
print("=" * 60)

# 定位 vendored slime 的 train.py（RL 真正的训练主程序）
TRAIN_PY = os.path.join(SLIME_ROOT, "train.py")

if not os.path.exists(TRAIN_PY):
    print(f"[TRAIN_WITH_PATCH] Error: train.py not found at {TRAIN_PY}")
    sys.exit(1)

# Execute train.py in the current namespace
# 编译后以 __name__="__main__" 在当前命名空间执行：
# 等价于"python train.py"，但保留了本进程已应用的全部补丁与钉扎路径
with open(TRAIN_PY, "r") as f:
    code = compile(f.read(), TRAIN_PY, "exec")
    exec(code, {"__name__": "__main__", "__file__": TRAIN_PY})
