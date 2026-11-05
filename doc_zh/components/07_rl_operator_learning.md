# 07 RL：算子学习

## 组件职责

`OpenMLE-ERL/RL/` 是 Frontis-MA1 后训练第二阶段：**execution-grounded 在线强化学习**，训练与推理共享的四个原子算子（Draft/Improve/Debug/Crossover）。训练框架是 vendored SLIME + Ray + SGLang，奖励来自沙箱真实执行（OpenMLE Sandbox 或本地评估），不是模型自评。

## 模块地图

| 文件 | 角色 |
| --- | --- |
| `train_with_patch.py` | 同步训练入口（`:105` 定位并 exec `slime/train.py`） |
| `train_async_with_patch.py` | 异步训练入口（`:90` 定位并 exec `slime/train_async.py`，训练与 rollout 重叠） |
| `patch_rollout.py` | rollout 日志补丁：算子级奖励视图 + 程序库快照 |
| `patch_eval_logger.py` | 评估日志补丁 |
| `fully_async_rollout.py` | 全异步 rollout worker（常驻线程 + 连续循环） |
| `program_database.py` | 已执行程序库：存储与父代采样（RL 侧的经验先验） |
| `reward_func_utils.py` / `ttt_reward_postprocess.py` | 奖励计算与后处理 |
| `adaptive_reward_advantage_utils.py` | 自适应奖励/优势处理 |
| `airaevo_experience.py` | 与 AIRA-Evo 经验机制对接 |
| `generate_mle.py` | 自定义生成函数（SLIME 按模块路径加载） |

## 入口的补丁机制（为什么需要 wrapper）

SLIME 按模块路径导入自定义函数，且猴子补丁必须在 slime 模块加载前生效。两个入口都执行三步（见 `train_with_patch.py` 的中文注释）：

1. **路径钉扎**（`train_with_patch.py:35` 的 `_pin_local_repo_paths`）：把 vendored `RL/slime` 提到 `sys.path` 最前，剔除外部 slime 安装，防止 pip 包抢占导入；
2. **预加载补丁**：`import patch_eval_logger`、`import patch_rollout`——模块导入时即完成函数替换（`patch_rollout.py:428` 的 `apply_patch`）；
3. **exec 主程序**：`compile + exec` 以 `__main__` 身份运行 train.py，保留已加载的补丁。异步入口额外**拒绝回退** `/root/slime`（`train_async_with_patch.py:90` 后的安全检查），防止静默用错代码。

## 奖励与观测（patch_rollout.py）

- 奖励视图（`patch_rollout.py:30` 的 `REWARD_VIEW_KEYS`）：`metric_static_base_reward`（相对任务静态基线归一，跨任务可比）与 `dynamic_raw_reward`（任务原始指标）；
- 算子分桶（`patch_rollout.py:248` 的 `patched_log_rollout_data`）：每个 rollout batch 按四算子分桶统计 hack 率、空程序率、成功率、得分率与组内质量代理（组内最优奖励、刷新历史最优比例）；Improve 父代奖励为 0 时重归类 Debug；
- GRPO/GSPO 组内归一的独立复算（`patch_rollout.py:182` 的 `_compute_debug_rewards`）让奖励链路可核对；
- 程序库快照（`patch_rollout.py:217` 的 `patched_save_db_snapshot`）：`SAVE_DB_SNAPSHOTS=1` 时每轮 rollout 落盘 `.db` 快照，供离线审计经验库演化。

## 训练语义要点

- rollout 采样由 `generate_mle.py` 驱动：模型在任务上执行四算子之一 → 代码进沙箱 → 执行分数经奖励后处理变成标量奖励 → 组内归一成优势；
- reward hacking 监控是一等公民：`code_category`（hack/hack_verify/no_verify/empty/valid）与 hack 率逐算子可见；
- 异步档（`fully_async_rollout.py`）常驻 worker 连续采样，训练与生成重叠，配合 `generation_abort` 元数据支持中断重置。

## 上手路径

`OpenMLE-ERL/RL/README.md` 的 Quick Start 展示了克隆锁定版 slime（commit 680824d）、把 `RL/` 拷入 `examples/openmle_rl`、预检（`PRECHECK_ONLY=1`）的流程；四个启动 profile（单/多节点 × 同步/异步）见 `configs/` 与 `scripts/`。
