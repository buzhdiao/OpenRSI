# 01 OpenRSI 总体定位：AI4AI 与机制阶梯

## 一句话定位

OpenRSI 是 FrontisAI 的开源计划，目标是把"AI 改进 AI"（AI4AI）变成一个**可执行、可度量、可复现**的工程问题。它不宣称解决了通用递归自我改进（RSI），而是从**有界、可执行领域**（机器学习工程，MLE）开始训练"改进者"本身——即论文所称的 Meta-Evolution（元进化）档位。

## AI4AI 的三条路线

根 README 的 OpenRSI 一节给出三条相互衔接的路线（见 `README.md` 的 "📖 OpenRSI" 部分）：

1. **AI4AI 基础模型**：把可复用的科研过程内化进模型权重——即 post-train 出的 Frontis-MA1（35B/30B）；
2. **世界模型与科研品味（research taste）**：判断"哪里值得再投入算力"；
3. **开放任务、环境与评测**：把真实科研转化为可规模化的训练与搜索管线——即 OpenMLE-Gym / OpenMLE-Tasks。

三条路线构成一个闭环：**搜索产生经验 → 经验进入训练 → 训练后的模型回到搜索与评测**。README 中反复强调，目标不是某次基准刷榜，而是让每一代 AI 研发"更快、更省、更强"变得可验证、可归因——"OpenRSI makes the rate of improvement itself an optimization target"（把改进速度本身变成优化目标）。

## 机制阶梯

README 用 "mechanism ladder" 描述从低到高的机制档位：

- **Evolution（进化）**：系统产出变异并按表现筛选（无自我模型）；
- **Self-Evolution（自进化）**：智能体在自己的轨迹上改进自己；
- **Meta-Evolution（元进化）**：训练"改进者"——OpenRSI 的起点；
- **RSI（递归自我改进）**：改进过程作用于改进过程本身，尚属开放问题。

对应到代码仓库，Meta-Evolution 落在三件具体的事情上：

- **环境**：`OpenMLE-Gym/openmle_gym/build.py:293`（任务构建）与 `local_evaluator.py:122`（质量评估）把真实比赛变成可验证任务包；
- **算子学习**：`OpenMLE-ERL/RL/train_with_patch.py:105`（同步 RL 入口）等把四个原子算子的改进能力训进模型；
- **长程搜索**：`OpenMLE-Evo/third_party/aira-evo/src/dojo/solvers/evo/evo.py:2169`（`search` 主循环）把学到的算子组合成数小时的进化搜索。

## 首个可执行领域：MLE

为什么选 MLE？README 的回答是：MLE 提供了一个具体且可执行的试验场——AI 系统可以对程序 **draft（起草）、improve（改进）、debug（调试）、crossover（重组）**，真实执行它们，观察**可测量的反馈**，并从轨迹中学习。这四个动作就是贯穿整个仓库的**四个原子算子**（详见 03 篇），也是训练与推理共享的动作空间。

## 阅读路径

- 概览：`README.md:33-49`（OpenRSI 定位与图示）；
- 结果边界：`docs/results.md`（所有结论均为 model–harness 结果，见 08 篇）；
- 发布物地图：`docs/release.md`。
