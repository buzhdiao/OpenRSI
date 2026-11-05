# 03 四大进化算子与共享动作空间

## 核心思想

OpenMLE 把**训练**（SFT + RL）与**推理**（长程搜索）对齐在同一个动作空间上：四个原子程序进化算子。论文摘要的表述是——同样的算子用 execution-grounded 的 SFT 与 RL 训练，再组合成长程搜索，把学习与进化耦合进同一个循环。这避免了"训练学一套、推理用另一套"的错位。

## 四个原子算子

| 算子 | 功能 | 输入 | 代码调用点（evo.py 已注释） |
| --- | --- | --- | --- |
| Draft | 从零生成程序 | 任务描述 + 数据预览 | `_draft`（`OpenMLE-Evo/third_party/aira-evo/src/dojo/solvers/evo/evo.py:1174`） |
| Improve | 依据执行反馈改进父代 | 一个可运行父代 + 其执行输出 | `_improve`（`evo.py:1206`） |
| Debug | 修复失败程序 | 一个报错父代 + 错误日志 | `_debug`（`evo.py:1240`） |
| Crossover | 重组两个父代 | 两个父代 + 各自执行反馈 | `_crossover`（`evo.py:1292`） |

`evo.py:937` 的 `setup_operators` 为每个算子建立独立的 LLM 客户端（`GenericLLM`），并用 `functools.partial` 预绑定配置、记忆组件与共享提示词上下文——训练对齐后，这四个端点通常由同一个 Frontis-MA1 服务。另有辅助算子 `analyze`（`evo.py:1274` 附近的 `_analyze`），仅在沙箱无法给出确定性反馈时兜底（见 `parse_eval_result`，`evo.py:1987`）。

## 共享动作空间的提示词侧

提示词构建器（RL 侧与搜索侧几乎同构，保证训练-推理一致）：

- `OpenMLE-ERL/RL/prompt_builder.py:65`（draft）、`:88`（improve）、`:126`（crossover）、`:440`（统一入口 `build_prompt`）；
- `OpenMLE-Evo/tts_search/prompt_builder.py:70`（draft）、`:102`（improve）、`:127`（debug）、`:163`（crossover）、`:196`（`build_prompt`）。

Crossover 的系统提示强调"提出两个原始解都不具备的新改进，简单拼接不足以取胜"；Improve/Debug 的用户提示都显式给出"上一次尝试 + 分数 + 执行输出"——这正是 execution-grounded 的提示词形态：反馈是真实执行产物，不是模型自评。

## 算子调度（何时用哪个算子）

算子选择发生在父代采样阶段 `evo.py:586` 的 `sample_in_context`：

- 种群为空 → 强制 Draft；
- 重复 Debug 失败（签名检测，`evo.py:1474` 附近的 `_repeated_debug_failure_reason`）或以 `fresh_draft_prob` 概率 → 强制 Draft（`request_fresh_draft`）；
- 否则以 `crossover_prob` 概率选 Crossover（需要岛上父代充足，不足则降级 Improve），其余选 Improve；
- Debug 不在采样时选择，而是执行失败后由 `debug_cycle`（`evo.py:1435`）/ `_debug_cycle_async`（`evo.py:1727`）按需触发。

## 算子的执行闭环

每个算子生成的子代都会：执行（`task.step_task`）→ 解析执行反馈（`parse_eval_result`，`evo.py:1987`）→ 失败进 Debug 循环 → 成功则按"不劣于岛均适应度"准入种群（`add_nodes_to_islands`，`evo.py:402`）。RL 侧则用同样的四算子分桶统计奖励（`OpenMLE-ERL/RL/patch_rollout.py:248` 的 `patched_log_rollout_data`，其中 Improve 父代奖励为 0 时重归类为 Debug）。
