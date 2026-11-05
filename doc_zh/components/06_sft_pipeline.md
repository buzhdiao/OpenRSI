# 06 SFT：执行接地数据管线

## 组件职责

`OpenMLE-ERL/SFT/` 是 Frontis-MA1 后训练第一阶段：**用执行接地的轨迹做监督微调**。发布管线合并两类样本共 26,574 条（见 `OpenMLE-ERL/SFT/README.md`）：并行全量响应 17,344 条 + 演化轨迹步 9,230 条。所有数据在与评测基准去重的前提下产生（论文去重原则）。

## 目录结构

| 路径 | 用途 |
| --- | --- |
| `scripts/` | rollout、评估与数据选择入口（`run_evolutionary_rollout.sh`、`evaluate_airaevo.py`、`runtime_storage_env.sh` 等） |
| `tts_search/` | 并行与演化 rollout 运行时（`services/` 下分 generator/evaluator/scheduler/rejection/result_persistence；`data_produce/` 下分 collect/alignment/filters/selection/pipeline） |
| `slime_scripts/` | 30B/35B 全参训练启动器（`qwen3_6_35b/train.sh`、`common/run_slime_sft.sh`） |
| `slime/` | vendored 的 SLIME 训练框架（`train.py` / `train_async.py`） |
| `third_party/aira-evo/` | 演化搜索运行时与 MLE 桥接 |

## 数据怎么来（rollout 采集）

演化轨迹的采集由 `OpenMLE-ERL/SFT/scripts/run_evolutionary_rollout.sh` 驱动：在任务上跑演化搜索（复用 tts_search 运行时），把 Draft/Improve/Debug/Crossover 每一步的（提示词、模型响应、执行反馈、分数）完整记录为轨迹样本。并行轨迹则对同一任务独立多次采样全量响应。执行反馈来自沙箱真实评估——这是"execution-grounded SFT"的字面含义：模型模仿的不是静态答案，而是"带着执行反馈改代码"的过程。

## 数据怎么筛（selection 与过滤）

`OpenMLE-ERL/SFT/tts_search/data_produce/` 实现了多道关卡（文件名即语义）：

- `collect.py`：汇总 rollout 产物；
- `alignment.py`：对齐消息格式（多轮轨迹 → 训练样本）；
- `baseline_filter.py` / `gap_filter.py`：按"相对基线的改进幅度"筛选——保留真正带来执行分数提升的轨迹步，而非平均水平的噪声样本；
- `token_filter.py`：按长度预算裁剪；
- `selection.py` + `select_parallel.py` / `select_evolutionary.py`：两条数据线的最终配额选择；
- `pipeline.py`：串起全流程；`exclude_reserved_tasks.py`（scripts 侧）负责把保留评测任务从训练数据中剔除（基准去重的落点之一）。

## 训练怎么跑

全参 SFT 用 vendored SLIME：`OpenMLE-ERL/SFT/slime/train.py`（同步）与 `train_async.py`（异步）。启动器 `slime_scripts/common/run_slime_sft.sh` 组合 Megatron/Transformers 后端与 SGLang 推理引擎。与 RL 侧不同，SFT 训练本身不需要执行沙箱——执行信息已经在数据里了。

## 与 RL 的关系

SFT 让模型先"会做"四个算子（行为克隆执行接地的专家轨迹），RL（见 07 篇）再让模型"做得更好"（在线探索 + 奖励优化）。两者共用同一提示词构建器族（`OpenMLE-ERL/SFT/tts_search/prompt_builder.py` 与 `OpenMLE-ERL/RL/prompt_builder.py`），保证训练阶段内部以及训练-推理之间的动作空间一致。

## 上手路径

`OpenMLE-ERL/SFT/docs/usage.md` 给出锁定环境安装（`uv sync --locked --extra rollout`）、rollout 生成、选择规则、30B/35B 训练与续跑的完整命令。
