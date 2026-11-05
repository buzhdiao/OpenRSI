# OpenRSI 中文版 README（README.zh.md）

> 本文件是仓库根 [README.md](../../README.md) 的完整中文翻译，保留原徽章链接、表格与代码块。译文与原文如有出入，以原文为准。原项目遵循 CC BY-NC 4.0 许可（非商业使用）。

<div align="center">

<img src="../../assets/logos/openrsi-logo.svg" width="560" alt="OpenRSI by Frontis">

**让"AI 改进 AI"变得可执行、可度量、可复现。**

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](../../LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2607.28568-b31b1b.svg)](https://arxiv.org/abs/2607.28568)
[![Hugging Face Paper](https://img.shields.io/badge/%F0%9F%A4%97%20Paper-2607.28568-yellow)](https://huggingface.co/papers/2607.28568)
[![Project Page](https://img.shields.io/badge/%F0%9F%8C%90%20Project-Page-blue)](https://frontisai.github.io/OpenRSI/)

[![Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Collection-Frontis--MA1-yellow)](https://huggingface.co/collections/FrontisAI/frontis-ma1)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Frontis--MA1--35B-yellow)](https://huggingface.co/FrontisAI/Frontis-MA1-35B)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-OpenMLE--Tasks-yellow)](https://huggingface.co/datasets/FrontisAI/OpenMLE-Tasks)

[📖 OpenRSI](#openrsi) · [🚀 Frontis-MA1](#frontis-ma1) · [🧩 OpenMLE](#openmle) · [📊 结果](#results) · [✨ 快速开始](#getting-started) · [🤝 参与共建](#contribute)

> **首个发布 · 机器学习工程**<br>
> **Frontis-MA1 + OpenMLE**

</div>

## 📰 新闻

- **[2026-08-25]** 🎉 **EEMA（Experience-Evolving Meta Agents，经验演化元智能体）**——**OpenMLE-Evo** 背后的基础技术之一——已被 **EMNLP 2026 主会**接收。恭喜我们的合作者！相关任务技能已更新至 [OpenMLE-Evo/skills](../../OpenMLE-Evo/skills/)。
- **[2026-08-09]** 🧰 我们开源了 [OpenMLE Sandbox](../../OpenMLE-Gym/openmle-sandbox/README.md)：**OpenMLE-Evo** 与 **OpenMLE-RL** 的自托管分布式代码执行与自动评估后端，支持 CPU/GPU 任务调度与可选的多控制器路由。
- **[2026-07-31]** 🎉 OpenRSI 首次发布：**Frontis-MA1**（[35B](https://huggingface.co/FrontisAI/Frontis-MA1-35B) / [30B](https://huggingface.co/FrontisAI/Frontis-MA1-30B)，含 [GGUF](https://huggingface.co/collections/FrontisAI/frontis-ma1) 衍生版本）、**OpenMLE** 全栈（Gym / RL / Evo），以及 [OpenMLE Tasks](https://huggingface.co/datasets/FrontisAI/OpenMLE-Tasks) 与 [OpenMLE SFT Traces](https://huggingface.co/datasets/FrontisAI/OpenMLE-SFT-Traces) 数据集。
- **[2026-06-25]** 📚 我们发布了 RSI 综述：[*Self-Improving Agents in the Era of Experience: A Survey of Self- to Meta-Evolution*](https://openreview.net/pdf?id=IUltZSgLMm)（[论文列表](https://github.com/FrontisAI/Awesome-Self-Improving-Agents)）——OpenRSI 背后"从自进化到元进化"的概念路线图。
- **[2026-06-23]** 📄 我们发布了 **NatureBench**：[*NatureBench: Can Coding Agents Match the Published SOTA of Nature-Family Papers?*](https://arxiv.org/abs/2606.24530)（[代码](https://github.com/FrontisAI/NatureBench)）——检验编码智能体能否达到 Nature 系列论文发表 SOTA 的基准，后来用作 RSI 的 held-out 迁移评测。

<a id="openrsi"></a>

## 📖 OpenRSI

OpenRSI 是 Frontis 的开源计划，要把"AI 改进 AI"变成一个可执行的工程问题。它通过三条相互衔接的路线发展 AI4AI 系统：

- **AI4AI 基础模型**：把可复用的科研过程内化进模型
- **世界模型与科研品味**：识别"额外的算力投在哪里值得"
- **开放任务、环境与评测**：把真实科研转化为可规模化的训练与搜索管线

搜索产生经验，经验进入训练，训练后的模型回到搜索与评测。目标不是一次性的刷榜成绩，而是让每一代 AI 研发在**可验证、可归因**的意义上变得**更快、更高效、更强大**。

> **OpenRSI 把"改进速度本身"变成优化目标。**

机制阶梯从 **Evolution（进化）** 走向 **Self-Evolution（自进化）**、**Meta-Evolution（元进化）**，最终指向 RSI。OpenRSI 从 Meta-Evolution 起步——在有界、可执行的领域中训练"改进者"本身——并不宣称已解决通用递归自我改进。

<p align="center">
  <img src="../../assets/figures/paper-figure-positioning.jpg" width="96%" alt="OpenRSI 定位：AI4AI、OpenMLE 与从进化到 RSI 的机制层级">
</p>

<a id="frontis-ma1"></a>

## 🚀 Frontis-MA1：首个 OpenRSI 发布

我们的第一个可执行领域是**机器学习工程（MLE）**。MLE 提供了一个具体的环境：AI 系统可以对程序进行起草、改进、调试与重组；真实执行它们；观察可测量的反馈；并从产生的轨迹中学习。

<!-- GitHub 仅播放通过网页编辑器上传的视频（user-attachments URL 单独成行）。
     更新方法：把新 mp4 拖进任意 GitHub 评论/编辑框，把生成的 URL 粘贴到这里。 -->

https://github.com/user-attachments/assets/176247a4-c28e-4e30-b65e-45873a7ae946

<p align="center">
  <sub>视频无法播放？<a href="../../docs/assets/videos/frontis-ma1-openmle-release-teaser.mp4">下载完整画质发布视频</a> 或在 <a href="https://frontisai.github.io/OpenRSI/">项目主页</a> 观看。</sub>
</p>

首次发布包含：

- **OpenMLE**：面向可执行 RSI 研究的开放全栈系统。
- **Frontis-MA1**：经过后训练的 AI4AI 模型，扮演 MLE 的元进化智能体。
- **OpenMLE Tasks**：用于训练与评测的可执行任务工件及构建管线。

### 论文

**[Frontis-MA1: Training an AI4AI Model towards Recursive Self-Improvement in Machine Learning Engineering](https://arxiv.org/abs/2607.28568)**

[📄 arXiv](https://arxiv.org/abs/2607.28568) · [🤗 Hugging Face Papers](https://huggingface.co/papers/2607.28568)

<details>
<summary><strong>摘要（译文）</strong></summary>

递归自我改进（RSI）要求 AI 系统能够改进"构建 AI 的过程"（即 AI4AI）；机器学习工程（MLE）为研究这一能力提供了具体、可执行的试验场。

我们提出 OpenMLE——一个面向 MLE 领域 RSI 研究的开放全栈系统，涵盖带执行反馈的可验证任务环境（OpenMLE-Gym）、算子学习（OpenMLE-RL）与长程搜索（OpenMLE-Evo）。在该栈之上，我们后训练出 Frontis-MA1（35B）作为 MLE 的元进化智能体，把后训练与推理对齐到四个原子程序进化算子（Draft、Improve、Debug、Crossover）上：同样的算子先在与所有评测基准去重后的执行接地（execution-grounded）SFT 与 RL 数据上训练，再被组合成长程搜索，把学习与进化耦合进同一个循环。

在 MLE-Bench Lite（每任务 12 小时预算、单张限制 12GB 显存的 RTX 4090）上，Frontis-MA1（35B）搭配 OpenMLE-Evo 把 Medal Average 从基座模型的 39.39% 提升到 60.61%；搭配 OpenMLE-Evo-Max（基准无关经验先验与异步搜索）达到 71.21%，超过 GPT-5.5 + Codex，接近 GPT-5.6 Sol 与 2.8T 的 Kimi K3。

在 held-out 的 NatureBench Lite 上，两个组件都可迁移：固定框架、只换模型，Match-SOTA 从 50% 升至 70%；固定模型、只换 OpenMLE-Evo，Match-SOTA 从 20% 升至 50%。

我们发布模型权重与完整 OpenMLE 栈，以支持面向 RSI 的可执行 AI4AI 复现研究。

</details>

<a id="openmle"></a>

## 🧩 OpenMLE：可执行的技术栈

OpenMLE 把后训练与推理对齐到一个共享动作空间上。同样的程序进化算子既从执行接地数据中学习，也被组合成长程搜索。

| 组件 | 角色 | 入口 |
| --- | --- | --- |
| **OpenMLE-Gym** | 构建、描述、执行并质检可验证的 MLE 任务包 | [`OpenMLE-Gym/`](../../OpenMLE-Gym/) |
| **OpenMLE-RL** | 通过执行接地的 SFT 与在线 RL 学习程序进化算子 | [`OpenMLE-ERL/SFT/`](../../OpenMLE-ERL/SFT/) · [`OpenMLE-ERL/RL/`](../../OpenMLE-ERL/RL/) |
| **Frontis-MA1** | 把学到的算子用作 MLE 元进化智能体 | [35B](https://huggingface.co/FrontisAI/Frontis-MA1-35B) · [30B](https://huggingface.co/FrontisAI/Frontis-MA1-30B) |
| **OpenMLE-Evo** | 把算子组合成标准或异步长程搜索 | [`OpenMLE-Evo/`](../../OpenMLE-Evo/) |

OpenMLE-Evo-Max 是 `OpenMLE-Evo` 内部的异步多 GPU 搜索档位，不是独立的源码树。OpenMLE-RL 以 [`OpenMLE-ERL/`](../../OpenMLE-ERL/) 目录发布，其中打包了它的两个阶段：执行接地 SFT 与在线 RL。

OpenMLE-Evo 与 OpenMLE-ERL 的分布式代码执行与自动评估由 [OpenMLE Sandbox](../../OpenMLE-Gym/openmle-sandbox/README.md) 提供，作为 OpenMLE-Gym 的一部分维护。

### 可训练的原子算子

| 算子 | 功能 |
| --- | --- |
| **Draft** | 从零生成一个程序。 |
| **Improve** | 利用执行反馈改进父代程序。 |
| **Debug** | 修复一个失败的程序。 |
| **Crossover** | 重组两个父代程序。 |

<p align="center">
  <img src="../../assets/figures/paper-figure-openmle-framework.png" width="96%" alt="OpenMLE 框架：可训练原子算子、进化推理、SFT 与在线 RL">
</p>

<a id="results"></a>

## 📊 结果

在 MLE-Bench Lite 上，Frontis-MA1 与 OpenMLE 通过受控比较把"模型进步"与"搜索系统进步"分离开：

| 发现 | 受控比较 | 结果 |
| --- | --- | ---: |
| **后训练增益** | 基座模型 → Frontis-MA1-35B，OpenMLE-Evo 固定 | **39.39% → 60.61%** Medal Average |
| **端到端系统** | Frontis-MA1-35B + OpenMLE-Evo-Max | **71.21%** Medal Average |
| **held-out 模型迁移** | 更换模型，NatureBench adapter 固定 | **50% → 70%** Match-SOTA |
| **held-out 框架迁移** | 更换框架，基座模型固定 | **20% → 50%** Match-SOTA |

这些是**模型-harness 结果**，不是独立的单次模型分数。OpenMLE-Evo-Max 的结果通过基准无关经验先验与异步搜索改变了搜索系统，不应解读为纯模型增益。

<p align="center">
  <img src="../../assets/figures/frontis-ma1-results.jpg" width="98%" alt="Frontis-MA1 与 OpenMLE-Evo 在 MLE-Bench Lite 上跨模型与 harness 的结果">
</p>

完整的受控比较表、评测协议、搜索效率研究、长程案例与解读边界见 [`docs/results.md`](../../docs/results.md)。

## 📦 发布物

| 工件 | 发布 | 内容 |
| --- | --- | --- |
| **Frontis-MA1 Collection** | [Collection](https://huggingface.co/collections/FrontisAI/frontis-ma1) | 所有 Frontis-MA1 模型与 OpenMLE 数据集汇总 |
| **Frontis-MA1-35B** | [模型权重](https://huggingface.co/FrontisAI/Frontis-MA1-35B) · [GGUF](https://huggingface.co/FrontisAI/Frontis-MA1-35B-GGUF) | BF16 权威模型与 GGUF 衍生版 |
| **Frontis-MA1-30B** | [模型权重](https://huggingface.co/FrontisAI/Frontis-MA1-30B) · [GGUF](https://huggingface.co/FrontisAI/Frontis-MA1-30B-GGUF) | BF16 权威模型与 GGUF 衍生版 |
| **OpenMLE SFT Traces** | [数据集](https://huggingface.co/datasets/FrontisAI/OpenMLE-SFT-Traces) | Frontis-MA1 使用的监督轨迹 |
| **OpenMLE Tasks** | [数据集](https://huggingface.co/datasets/FrontisAI/OpenMLE-Tasks) | 审计后的任务清单与可发布任务工件 |
| **OpenMLE-Gym** | [`OpenMLE-Gym/`](../../OpenMLE-Gym/) | 任务构建与评估源码 |
| **OpenMLE-RL** | [`OpenMLE-ERL/`](../../OpenMLE-ERL/) | SFT 与强化学习源码 |
| **OpenMLE-Evo** | [`OpenMLE-Evo/`](../../OpenMLE-Evo/) | 搜索运行时与基准适配器 |

完整的工件地图、源码边界与外部运行时要求见 [`docs/release.md`](../../docs/release.md)。

<a id="getting-started"></a>

## ✨ 快速开始

```bash
git clone https://github.com/FrontisAI/OpenRSI.git
cd OpenRSI
```

OpenRSI 包含若干可独立运行的阶段，而非一个合成的顶层命令。每个组件管理自己的环境与依赖——从与你的目标匹配的那个开始：

| 目标 | 从这里开始 |
| --- | --- |
| 构建或评估可执行任务包 | [`OpenMLE-Gym/README.md`](../../OpenMLE-Gym/README.md) |
| 生成或筛选 SFT 数据并启动监督训练 | [`OpenMLE-ERL/SFT/README.md`](../../OpenMLE-ERL/SFT/README.md) |
| 配置并启动执行接地 RL | [`OpenMLE-ERL/RL/README.md`](../../OpenMLE-ERL/RL/README.md) |
| 运行 OpenMLE-Evo 或基准适配器 | [`OpenMLE-Evo/README.md`](../../OpenMLE-Evo/README.md) |

### 快速上手：本地运行一个 NatureBench 用例

在 `s42256-023-00611-x`（分类反事实结果估计）上运行一次真实的 OpenMLE-Evo 搜索与评估循环。以下 smoke 档位只生成并评估一个候选；任务数据、生成代码与评估都留在本机。

```bash
# 接在上面的快速开始命令之后。
cd ..
git clone https://github.com/FrontisAI/NatureBench.git
cd OpenRSI/OpenMLE-Evo

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
conda env create -f environments/naturebench-local.yml

export PRIMARY_KEY='your-api-key'
.venv/bin/python scripts/run_naturebench_local.py \
  --naturebench-repo ../../NatureBench \
  --conda-env naturebench-local \
  --model-base-url https://model.example/v1 \
  --model-id served-model-name \
  --smoke
```

同一个运行器也支持本地或远程 SGLang 端点。去掉 `--smoke` 即完整单任务搜索档位。SGLang 命令、四小时搜索预算、高级覆盖项与本地执行安全边界见[本地 NatureBench 教程](../../OpenMLE-Evo/benchmarks/naturebench_local_quick/README.md)，随后可阅读[示例轨迹分析](../../OpenMLE-Evo/benchmarks/naturebench_local_quick/RESULTS.md)了解分数归因与优化动力学。

模型权重、任务工件、训练语料、外部基准环境、服务凭据与私有基础设施配置不随本源码仓库分发，另行提供。

## 🗂️ 仓库内容

```text
.
├── assets/                  # 图表与 Frontis 品牌素材
├── docs/                    # 项目主页、结果、训练与发布范围
├── OpenMLE-Gym/             # 可执行任务构建与评估
├── OpenMLE-ERL/
│   ├── SFT/                 # rollout 采集、数据选择与 SFT
│   └── RL/                  # 执行接地强化学习
├── OpenMLE-Evo/             # 长程搜索与基准适配器
├── LICENSE
└── NOTICE
```

## 📚 文档

| 文档 | 内容 |
| --- | --- |
| [交互式项目主页](https://frontisai.github.io/OpenRSI/) | 可视化总览、发布工件、结果与项目视频 |
| [`docs/results.md`](../../docs/results.md) | 结果表、评测协议与比较边界 |
| [`docs/training.md`](../../docs/training.md) | 任务管线、监督语料与强化学习摘要 |
| [`docs/release.md`](../../docs/release.md) | 公开工件地图、依赖与发布边界 |
| [`OpenMLE-Gym/docs/usage.md`](../../OpenMLE-Gym/docs/usage.md) | 任务构建、元数据、评估与 smoke 工作流 |
| [`OpenMLE-ERL/SFT/docs/usage.md`](../../OpenMLE-ERL/SFT/docs/usage.md) | rollout 生成、选择与监督训练 |
| [`OpenMLE-ERL/RL/docs/usage.md`](../../OpenMLE-ERL/RL/docs/usage.md) | RL 配置、启动档位与验证 |
| [`OpenMLE-Evo/docs/usage.md`](../../OpenMLE-Evo/docs/usage.md) | 安装、配置、启动、续跑与输出 |
| [`OpenMLE-Evo/docs/validation.md`](../../OpenMLE-Evo/docs/validation.md) | 源码级与运行时验证边界 |

<a id="contribute"></a>

## 🤝 与我们共建 OpenRSI

OpenRSI 希望在公开环境中构建。我们欢迎让 AI4AI 研究更可执行、更可复现的贡献：

- 真实科研任务、隔离环境与可验证评估器
- 科研轨迹、经验语料与去重方法
- 算子学习、长程搜索与信用分配方法
- AI4AI 世界模型与"科研品味"奖励
- 迁移基准与可执行的工业领域问题

[提一个 issue](https://github.com/FrontisAI/OpenRSI/issues) 提议任务、方法、基准、复现或合作。

## 🎈 引用

**Frontis-MA1: Training an AI4AI Model towards Recursive Self-Improvement in Machine Learning Engineering**

```bibtex
@misc{yang2026frontisma1trainingai4aimodel,
  title={Frontis-MA1: Training an AI4AI Model towards Recursive Self-Improvement in Machine Learning Engineering},
  author={Junlin Yang and Che Jiang and Yu Fu and Tianwei Luo and Can Ren and Weizhi Wang and Kaikai Zhao and Hongyi Liu and Yuxin Zuo and Yuru Wang and Yuchen Fan and Kai Tian and Zhenzhao Yuan and Xiaojian Lin and Li Sheng and Rushi Qiang and Guoli Jia and Xingtai Lv and Ermo Hua and Dianqiao Lei and Youbang Sun and Ning Ding and Bowen Zhou and Kaiyan Zhang},
  year={2026},
  eprint={2607.28568},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2607.28568},
}
```

## ⚖️ 许可

本仓库中 OpenRSI 与 OpenMLE 的原创内容以 [CC BY-NC 4.0](../../LICENSE) 发布：要求署名、仅限非商业使用。本许可不授予商业使用权。

第三方组件与依赖保留其各自条款，见 [NOTICE](../../NOTICE)。

## 🙏 致谢

感谢 Qwen、SLIME、Ray、SGLang、Megatron-LM、Transformers、MLE-Bench、NatureBench 以及更广泛的可执行 AI4AI 生态的社区。

## Star History

<a href="https://www.star-history.com/?repos=FrontisAI%2FOpenRSI&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=FrontisAI/OpenRSI&type=date&theme=dark&legend=top-left&sealed_token=UWbC_3WnIT3WDncJggzeSTAlJGo18V4FuMpMk_g3mE4bJ4Wf-8gJbcqVJjIIYMkrsCgnZOSiAOdeY7zdVmumjZxTDEMDQ54DVIw7xBdh39hvcZTqtHr76w" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=FrontisAI/OpenRSI&type=date&theme=dark&legend=top-left&sealed_token=UWbC_3WnIT3WDncJggzeSTAlJGo18V4FuMpMk_g3mE4bJ4Wf-8gJbcqVJjIIYMkrsCgnZOSiAOdeY7zdVmumjZxTDEMDQ54DVIw7xBdh39hvcZTqtHr76w" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=FrontisAI/OpenRSI&type=date&legend=top-left&sealed_token=UWbC_3WnIT3WDncJggzeSTAlJGo18V4FuMpMk_g3mE4bJ4Wf-8gJbcqVJjIIYMkrsCgnZOSiAOdeY7zdVmumjZxTDEMDQ54DVIw7xBdh39hvcZTqtHr76w" />
 </picture>
</a>
