# OpenRSI 中文学习文档（doc_zh）

本目录是 OpenRSI 仓库的中文学习文档工程，面向想理解 **Frontis-MA1 / OpenMLE** 技术栈实现细节的读者。全部内容基于仓库真实源码撰写，标注了 `file:line` 位置（行号对应当前带中文注释的代码版本）。

> 论文：*Frontis-MA1: Training an AI4AI Model towards Recursive Self-Improvement in Machine Learning Engineering*（arXiv:2607.28568）。仓库原始代码以 CC BY-NC 4.0 发布，本中文文档同样仅用于非商业学习用途。

## 目录

### 组件原理（doc_zh/components/）

| 编号 | 文档 | 主题 |
| --- | --- | --- |
| 01 | [OpenRSI 总体定位：AI4AI 与机制阶梯](components/01_openrsi_overview.md) | AI4AI 三条路线、机制阶梯（Evolution → Self-Evolution → Meta-Evolution → RSI） |
| 02 | [OpenMLE-Gym：可验证任务环境](components/02_openmle_gym.md) | 任务包构建、确定性验证、metric 冒烟、进程隔离 |
| 03 | [四大进化算子与共享动作空间](components/03_four_operators.md) | Draft / Improve / Debug / Crossover，提示词构建与调用点 |
| 04 | [OpenMLE-Evo：长程搜索](components/04_openmle_evo_search.md) | 标准逐代搜索与 Evo-Max 异步稳态搜索、岛屿模型、执行反馈解析 |
| 05 | [OpenMLE Sandbox：分布式执行与自动评估](components/05_openmle_sandbox.md) | Router/Controller/Worker 三层架构、任务执行流、本地进程隔离执行器 |
| 06 | [SFT：执行接地数据管线](components/06_sft_pipeline.md) | 演化 rollout 采集、数据选择与过滤、全参训练 |
| 07 | [RL：算子学习](components/07_rl_operator_learning.md) | execution-grounded RL、奖励视图、补丁机制、同步/异步训练入口 |
| 08 | [NatureBench 迁移评估与结果解读](components/08_naturebench_transfer.md) | held-out 迁移实验、模型-harness 对照、Medal Average 解读边界 |

### 翻译与技法

| 文档 | 内容 |
| --- | --- |
| [translations/README.zh.md](translations/README.zh.md) | 根 README 完整中文翻译（保留徽章、表格、代码块） |
| [techniques-100.md](techniques-100.md) | 「Python 高级技法与重要库 100 讲」，全部结合仓库真实代码 |

## 阅读建议

- 先读 01 建立全局图景，再按 02 → 03 → 04 的顺序理解"环境 → 算子 → 搜索"的主线；
- 05 讲执行后端，06/07 讲训练侧（SFT + RL），08 讲评测与结论边界；
- 每篇文档中的 `file:line` 均可跳转到带逐段中文注释的源码对照阅读；核心代码注释分布在：
  - `OpenMLE-Evo/third_party/aira-evo/src/dojo/solvers/evo/evo.py`（搜索主循环与四算子调用点）
  - `OpenMLE-Evo/tts_search/airaevo_async_resources.py`（Evo-Max 异步资源解析）
  - `OpenMLE-ERL/RL/train_with_patch.py`、`OpenMLE-ERL/RL/train_async_with_patch.py`（RL 训练入口）
  - `OpenMLE-ERL/RL/patch_rollout.py`（算子级奖励视图与程序库快照）
  - `OpenMLE-Gym/openmle_gym/build.py`（任务构建编排）
  - `OpenMLE-Gym/openmle_gym/process_runner.py`（进程/容器隔离执行）
  - `OpenMLE-Gym/openmle_gym/local_evaluator.py`（任务包质量评估核心）

## 术语表（速查）

| 术语 | 含义 |
| --- | --- |
| AI4AI | AI improving AI，用 AI 改进 AI 研发流程本身 |
| 四个原子算子 | Draft（起草）/ Improve（改进）/ Debug（调试）/ Crossover（交叉），推理与训练共享的动作空间 |
| execution-grounded | 执行接地：奖励/反馈来自真实代码执行与确定性评估，而非模型自评 |
| OpenMLE-Gym | 可验证任务环境：任务包构建 + 执行 + 自动评估 |
| OpenMLE-Evo | 长程搜索框架：把四个算子组合成进化搜索 |
| OpenMLE-Evo-Max | Evo 的异步强化档：基准无关经验先验 + 多 worker 异步稳态搜索（`async_steady_state`） |
| OpenMLE Sandbox | 自托管分布式代码执行与自动评估后端 |
| Medal Average | MLE-Bench 指标：按每任务"获奖等级"（金/银/铜/无）归一平均 |
| Match-SOTA | NatureBench 指标：智能体结果是否达到论文发表 SOTA |
