# 04 OpenMLE-Evo：长程搜索（标准与异步 Evo-Max）

## 组件职责

OpenMLE-Evo 把四个原子算子组合成**长程进化搜索**：在数小时预算内反复"采样父代 → 生成子代 → 沙箱执行 → 解析反馈 → 准入种群"，最终返回最优节点代码。核心实现集中在 `OpenMLE-Evo/third_party/aira-evo/src/dojo/solvers/evo/evo.py`（AIRA-Evo 衍生，OpenRSI 加入了经验记忆、Debug 签名检测与 Evo-Max 异步路径）。

## 数据结构

- `Node`（journal 树节点）：保存 plan、code、执行输出、is_buggy、metric，以及父代选择审计轨迹；
- `Island`（`evo.py:120`）：岛屿种群，提供 best/average 适应度、淘汰最差、迁移等方法；
- `SolutionsDatabase`（`evo.py:282`）：多岛数据库，负责**算子选择 + 父代加权采样**（`sample_in_context`，`evo.py:586`）与稳态准入（`add_nodes_to_islands`，`evo.py:402`）、岛屿重置迁移（`reset_islands`，`evo.py:466`）；
- `AsyncWorkItem`（`evo.py:103`）：异步搜索中一次"生成-执行"工作单元的不可变快照（attempt 编号、虚拟代数、温度、算子、父代、目标 worker）。

## 标准逐代搜索

入口 `search`（`evo.py:2169`），由 `__call__`（`evo.py:2355`）触发。第 0 代全部 Draft 均匀播种；第 1 代起每代为 `individuals_per_generation` 个个体执行：

1. 温度线性退火（`linear_decay`）下采样算子与父代；
2. 调用算子生成子代并执行；
3. 失败节点进入 `debug_cycle`（`evo.py:1435`），受 `max_debug_depth` 与 `max_debug_time` 约束；
4. 成功解按"不劣于岛均"准入；早期代数禁用 Crossover 与迁移（`num_generations_till_crossover/migration`）；
5. 每代保存 checkpoint（journal.jsonl），可从任意代恢复。

关键工程细节：排队等待 GPU 的时间不计入搜索预算（`_task_budget_exempt_seconds`），让 12 小时预算花在真搜索上。

## Evo-Max 异步稳态搜索

论文中 OpenMLE-Evo-Max 的两个增量是**基准无关经验先验**与**异步搜索**。异步部分对应执行模式 `async_steady_state`：

- 资源解析在 `OpenMLE-Evo/tts_search/airaevo_async_resources.py`：`resolve_async_worker_count`（`:59`，支持 `auto`=GPU 数）、`build_worker_specs`（`:123`，GPU 轮转绑定 + 沙箱轮转分配）、`fetch_router_idle_worker_count`（`:153`，观测路由器空闲度）；
- 主入口 `search_async_steady_state`（`evo.py:1917`）：为每个 worker 启动协程 `_async_worker_loop`（`evo.py:1810`）；
- 并发模型：**采样加锁（sample_lock）、生成与执行并行（每 worker 绑定独立沙箱 URL）、提交加锁（commit_lock）**——生成/执行是长耗时操作，只有状态写入需要串行化；
- 失败重试用指数退避；worker 数量与 attempt 上限把"代"折算成流水（`_next_async_work_item`，`evo.py:1581`）；
- Debug 循环异步化（`_debug_cycle_async`，`evo.py:1727`），反复同类失败会触发下一次强制 Draft。

跨进程的 LLM/沙箱并发上限由 `OpenMLE-Evo/tts_search/airaevo_concurrency.py` 的共享信号量服务管理（基于 `multiprocessing.SyncManager`，同步/异步两套上下文管理器）。

## 执行反馈解析（execution-grounded 落点）

`parse_eval_result`（`evo.py:1987`）决定节点的生死与适应度，优先级为：

1. 沙箱确定性验证适应度 `VALIDATION_FITNESS` —— 直接采纳，跳过 LLM 自评；
2. 沙箱确定性失败信号（status/status_code/exit_code）—— 标记 buggy；
3. 仅当两者皆无，才回退到 LLM `analyze` 算子。

最终 `is_buggy = LLM判bug ∨ 非零退出码 ∨ 无指标 ∨ 解非法`，失败节点套上 `WorstMetricValue` 哨兵保证排序垫底。搜索结束后 `export_search_results` 导出全树，最优代码交给评测脚本计算 Medal Average。
