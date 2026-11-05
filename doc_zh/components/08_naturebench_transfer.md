# 08 NatureBench 迁移评估与结果解读（模型-harness 对照实验）

## 为什么要 held-out 迁移评估

MLE-Bench 上跑分高，可能是模型真的更强，也可能是搜索系统过拟合了基准。论文用 NatureBench（`arXiv:2606.24530`）作 held-out 迁移基准来区分两者：NatureBench 检验编码智能体能否复现 Nature 系列论文的发表 SOTA，与训练数据完全无交集。指标是 **Match-SOTA**（智能体结果达到论文发表水平记 1）。

## 两组对照实验

根 README Results 表（`README.md:125-142`）给出四条受控比较，迁移相关的两条：

1. **模型迁移**（framework 固定）：NatureBench adapter 不变，只换模型——Match-SOTA 从 50%（基座模型）升到 70%（Frontis-MA1-35B）。说明后训练把"会做四个算子"的能力迁移到了从未见过的科研任务上；
2. **框架迁移**（model 固定）：基座模型不变，换上 OpenMLE-Evo——Match-SOTA 从 20%（单次生成）升到 50%。说明搜索框架本身带来正迁移。

## 结果表与解读边界（必须分清 model-harness）

README 明确警告：这些都是 **model–harness 结果**，不是独立的单次模型分。尤其 OpenMLE-Evo-Max 的 71.21% Medal Average 同时改变了搜索系统（基准无关经验先验 + 异步搜索），**不能解读为纯模型增益**。受控比较的完整表格、评测协议（12 小时/任务、单卡 RTX 4090 限 12GB 显存）与解读边界见 `docs/results.md`。

| 比较 | 控制 | 结果 |
| --- | --- | --- |
| 后训练增益 | 基座 → Frontis-MA1-35B，Evo 固定 | 39.39% → 60.61% Medal Average |
| 端到端系统 | Frontis-MA1-35B + OpenMLE-Evo-Max | 71.21% Medal Average |
| held-out 模型迁移 | 换模型，adapter 固定 | 50% → 70% Match-SOTA |
| held-out 框架迁移 | 换框架，基座固定 | 20% → 50% Match-SOTA |

## 本地复现路径

仓库提供单任务 smoke 级别的本地复现（根 README "Quick Start"）：

- 运行器：`OpenMLE-Evo/scripts/run_naturebench_local.py`（736 行，含模型端点健康检查、SGLang 端口转发与 Hydra 覆盖参数）；`--smoke` 只生成并评估一个候选，去掉即完整单任务搜索（4 小时预算）；
- 执行环境：`OpenMLE-Evo/environments/naturebench-local.yml`（conda env，锁定 LightGBM/scanpy/sklearn 等科研常用库版本）；
- 教程与轨迹分析：`OpenMLE-Evo/benchmarks/naturebench_local_quick/README.md` 与 `RESULTS.md`（分数归因与优化动力学）；
- 批量评测：`OpenMLE-Evo/scripts/evaluate_naturebench.py`、搜索启动脚本 `scripts/run_naturebench.sh` / `run_standard.sh` / `run_multi_gpu.sh`（Evo-Max 档：`execution=multi_gpu`，经 `evaluate_airaevo.py` 进入异步稳态搜索）。

## 怎么读结果（给复现者的三条提醒）

1. 对照实验的价值在于"单变量"：复现 70% 那行时只换模型，复现 50% 那行时只换框架；
2. Medal Average / Match-SOTA 都是任务级聚合，看分布（每任务奖牌）比只看均值更能定位系统瓶颈；
3. 经验先验与异步搜索属于 harness 工程，报告任何"模型能力"结论时都应注明所用 harness（这也正是论文把 60.61% 与 71.21% 分开陈述的原因）。
