# 02 OpenMLE-Gym：可验证任务环境

## 组件职责

OpenMLE-Gym 负责"把真实比赛变成可执行、可验证的 MLE 任务包"：构建（build）、描述（overview）、执行（sandbox/进程隔离）与质量检查（evaluate）。它是整个 OpenMLE 栈 execution-grounded 特性的环境侧基础——没有确定性评估器，后面的 SFT/RL 奖励与进化搜索适应度都无从谈起。

## 任务包契约

一个构建完成的任务目录遵循固定契约（见 `OpenMLE-Gym/openmle_gym/local_evaluator.py:122` 的 `analyze_task_structure`）：

- `data/public/`：`train.csv`、`test.csv`（无标签）、`sample_submission.csv`、`description.txt`；
- `data/private/test_answer.csv`：私有标准答案，供 metric 打分；
- `utils/prepare.py`：从原始数据确定性切分出 train/test；
- `utils/metric.py`：评估管线，含 `validate_submission()` 与 `evaluate()`；
- `raw/`：原始数据，供行数对账（row-usage accounting）。

## 构建流水线

构建入口是 `OpenMLE-Gym/openmle_gym/build.py:293` 的 `build_tasks`，要点：

1. **去重白名单**：`build.py:29` 的 `MLEBENCH_SLUGS` 集合收录 MLE-Bench 全部比赛 slug，`skip_mlebench=True` 时直接跳过——落实论文"训练数据对所有评测基准去重"的原则；
2. **Preflight 预检**（`build.py:331-378` 附近）：任务名合法性（`validate_task_name`）、重名检测、三个产出路径的目录穿越检查；
3. **进程隔离构建**（`build.py:215` 的 `_run_all`）：每个任务通过 `run_task_process` 在独立子进程中执行，子进程只有 builder 根目录与 info.csv 的只读权限、输出根目录的写权限；并发由 `asyncio.Semaphore` 限流；
4. **演练模式**：`execute=False` 时走 `build.py:144` 的 `_dry_run`，只报告将构建/跳过哪些任务。

## 进程隔离执行器

`OpenMLE-Gym/openmle_gym/process_runner.py:170` 的 `run_task_process` 是所有任务操作的统一执行底座：

- 通信协议：父进程原子写 `request.json` → 子进程 `python -m openmle_gym.task_worker` → 子进程写 `result.json` 信封 `{ok, result, error}`（`process_runner.py:52` 的 `TaskProcessOutcome`）；
- `isolated` 模式（`process_runner.py:91` 的 `_container_command`）：容器断网、根文件系统只读、CPU/内存/PID 受限、丢弃全部 capabilities、`/tmp` 无执行位；
- 环境变量白名单转发（`process_runner.py:31`），且 `prepare`/`metric` 阶段一律不透传凭据；
- 超时两阶段终止（`process_runner.py:73`）：SIGTERM 进程组，5 秒宽限后 SIGKILL。

## 质量评估：硬门禁 + AI 判官

`local_evaluator.py` 的评估分两层：

1. **确定性验证**（`local_evaluator.py:371` 的 `_deterministic_validation`）：必需文件齐全、CSV 行数/表头/行宽一致、test 与 sample_submission 行数与标识符逐行对齐（首列 SHA256 摘要比对，见 `local_evaluator.py:55` 的 `_inspect_csv`）、目标列不得泄漏进公共测试集、`metric.py` 冒烟执行返回有限分数（`local_evaluator.py:122` 内的 metric_check 步骤）；
2. **AI 语义评估**（`local_evaluator.py:789` 的 `_evaluate_with_ai`）：把结构证据交给 LLM 按 5 个维度打 0-5 分（`local_evaluator.py:695` 的 `QUALITY_DIMENSIONS`），输出必须通过 `_validate_ai_quality`（`local_evaluator.py:704`）的严格 schema 校验。

硬门禁失败即 0 分不推荐（`local_evaluator.py:589` 的 `_hard_gate_quality`），不浪费 LLM 调用。批量入口为 `local_evaluator.py:1059` 的 `evaluate_tasks`。

## 与其他组件的关系

Gym 产出的任务包被三处消费：OpenMLE-Evo 的搜索（执行反馈）、OpenMLE-ERL 的 SFT/RL（奖励来源）、OpenMLE Sandbox（分布式执行后端，见 05 篇）。
