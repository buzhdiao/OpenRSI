#!/usr/bin/env python3
# =====================================================================
# 中文注释导读（OpenRSI / OpenMLE-RL 学习文档）：
# 本脚本是 OpenMLE-RL 的"异步训练入口"（train_async_with_patch），对应 slime 的
# train_async.py：训练与 rollout 重叠执行（上一批样本还在训练时，下一批已在沙箱
# 执行并收集奖励），这是 execution-grounded RL 提高吞吐的关键。
# 与 train_with_patch.py 的差异：
#   - 目标脚本换成 train_async.py；
#   - 显式拒绝回退到 /root/slime 等外部安装（防止静默用了非 vendored 代码）；
#   - 同样要求补丁（patch_eval_logger / patch_rollout）先于 slime 模块加载。
# 论文术语：RL 算子学习——用四个原子算子（Draft/Improve/Debug/Crossover）产生的
# 执行轨迹训练模型，使策略本身学会"何时改、怎么改"。
# =====================================================================
"""
Wrapper script for train_async.py that applies custom patches.
This ensures monkey patches are applied before slime modules are loaded.
"""

import importlib.util  # 探测 slime 包解析位置（确认加载的是 vendored 副本）
import os  # 路径拼装与工作目录切换
import sys  # sys.path 钉扎

# 三级路径定位：RL/ -> RL/slime -> 项目根（与 train_with_patch.py 相同）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SLIME_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PROJECT_ROOT = os.path.dirname(SLIME_ROOT)


def _realpath(path: str) -> str:
    # 真实绝对路径归一化（解析符号链接），保证路径比较可靠
    return os.path.normcase(os.path.realpath(path))


def _pin_local_repo_paths() -> None:
    # 钉扎模块搜索路径：本仓库 RL/ 与 RL/slime/ 优先，剔除外部 slime 安装
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
        # 路径含 "/slime/" 但不在本仓库内的条目一律剔除
        if f"{os.sep}slime{os.sep}" in f"{candidate_real}{os.sep}" and not candidate_real.startswith(local_prefix + os.sep):
            continue
        filtered_sys_path.append(entry)

    sys.path[:] = pinned_paths + [entry for entry in filtered_sys_path if _realpath(entry or os.getcwd()) not in pinned_paths]
    os.chdir(SLIME_ROOT)  # train_async.py 依赖相对路径，切到 slime 根
    spec = importlib.util.find_spec("slime")
    origin = spec.origin if spec is not None else "missing"
    print(f"[TRAIN_ASYNC_WITH_PATCH] Pinned repo root: {PROJECT_ROOT}")
    print(f"[TRAIN_ASYNC_WITH_PATCH] slime import resolution: {origin}")


_pin_local_repo_paths()

print("=" * 60)
print("[TRAIN_ASYNC_WITH_PATCH] Applying custom eval logger patch...")
print("=" * 60)

try:
    import patch_eval_logger  # 评估日志补丁（在任何 slime 模块之前加载）

    print("[TRAIN_ASYNC_WITH_PATCH] Eval logger patch module loaded successfully")
except Exception as e:
    print(f"[TRAIN_ASYNC_WITH_PATCH] Warning: Failed to load eval logger patch: {e}")
    import traceback

    traceback.print_exc()

print("[TRAIN_ASYNC_WITH_PATCH] Applying rollout patch...")
try:
    import patch_rollout  # rollout 补丁：算子级奖励视图 + 程序库快照

    print("[TRAIN_ASYNC_WITH_PATCH] Rollout patch module loaded successfully")
except Exception as e:
    print(f"[TRAIN_ASYNC_WITH_PATCH] Warning: Failed to load rollout patch: {e}")
    import traceback

    traceback.print_exc()

print("[TRAIN_ASYNC_WITH_PATCH] Starting train_async.py...")
print("=" * 60)

# 定位 vendored slime 的 train_async.py（异步训练主程序）
TRAIN_ASYNC_PY = os.path.join(SLIME_ROOT, "train_async.py")

if not os.path.exists(TRAIN_ASYNC_PY):
    # 安全检查：即使 /root/slime 存在 train_async.py 也拒绝回退——
    # 静默使用外部代码会让补丁与实验配置失效，复现性无从谈起
    fallback_root = "/root/slime"
    fallback_train_async = os.path.join(fallback_root, "train_async.py")
    if os.path.exists(fallback_train_async):
        print(f"[TRAIN_ASYNC_WITH_PATCH] Error: expected local train_async.py at {TRAIN_ASYNC_PY}")
        print(f"[TRAIN_ASYNC_WITH_PATCH] Refusing to fall back to {fallback_train_async}")
        sys.exit(1)
    else:
        print(f"[TRAIN_ASYNC_WITH_PATCH] Error: train_async.py not found at {TRAIN_ASYNC_PY}")
        sys.exit(1)

print(f"[TRAIN_ASYNC_WITH_PATCH] Using slime root: {SLIME_ROOT}")

# 在当前命名空间执行 train_async.py：保留已加载的补丁与钉扎路径
with open(TRAIN_ASYNC_PY, "r") as f:
    code = compile(f.read(), TRAIN_ASYNC_PY, "exec")
    exec(code, {"__name__": "__main__", "__file__": TRAIN_ASYNC_PY})
