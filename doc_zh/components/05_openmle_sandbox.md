# 05 OpenMLE Sandbox：分布式执行与自动评估

## 组件职责

OpenMLE Sandbox（`OpenMLE-Gym/openmle-sandbox/README.md`）是自托管的**分布式代码执行与自动评估后端**，同时服务 OpenMLE-Evo（搜索执行反馈）与 OpenMLE-ERL（RL rollout 奖励）。2026-08-09 开源（见根 README News）。客户端提交"ML 代码 + 任务包路径"，沙箱在隔离 worker 容器中执行代码、评估产出的 `submission.csv`，通过 REST API 返回分数与日志。

## 三层架构

来自 `openmle-sandbox/README.md` 的架构图与组件表：

| 组件 | 职责 |
| --- | --- |
| Router（可选，:6591） | 为每个新任务选择 controller，持久化 job→controller 路由，聚合状态；多控制器横向扩展的入口（对应 Evo-Max 的 `async_sandbox_urls` 多端点） |
| Nginx（:6580） | 暴露 controller 端口，把 `/api/v1/*` 反代到 FastAPI |
| FastAPI | 鉴权（X-API-Key）、校验任务、建任务信封、入库、入队 |
| PostgreSQL | 任务全生命周期：时间戳、结果、分数、错误、幂等元数据 |
| Redis | 优先级队列 + CPU/GPU worker 实时清单 |
| Dispatcher | 选空闲 worker → 暂存代码 → 执行 → 评估提交 → 记录结果 → 归还 worker |
| AIO Sandbox worker | 长驻容器，暴露 HTTP shell API，一次执行一个任务 |
| 共享存储 | 数据集/任务包/评估代码在所有节点同路径可见 |
| worker 本地 NVMe scratch | 活跃代码与输出落在本地盘，避免直接在 NFS 上跑负载 |

值得注意的设计：**worker 不自注册**，Dispatcher 启动时从静态清单 `sandbox_config.json` 读取端点——部署拓扑可复现、可审计。

## 单个任务的执行流

README 中的时序图概括为：Client →（Router 可选）→ Nginx+FastAPI → PostgreSQL 落库 → Redis 入队 → Dispatcher 摘取 → 分派到空闲 AIO Sandbox worker 执行 → 评估 submission → 分数/日志经 API 返回。CPU 任务与 GPU 任务分池调度；Evo-Max 启动前还会用 `fetch_router_idle_worker_count`（`OpenMLE-Evo/tts_search/airaevo_async_resources.py:153`）探测各端点空闲度（仅观测日志）。

## 与 Gym 本地执行器的关系

没有集群时，OpenMLE-Gym 内置的单机执行器承担同样语义：

- `OpenMLE-Gym/openmle_gym/process_runner.py:170` `run_task_process`：process 模式（一次性子进程 + request/result JSON 信封）与 isolated 模式（docker/podman：断网、只读根、CPU/内存/PID 上限、cap-drop ALL、`/tmp` 无执行位，见 `:91` 的 `_container_command`）；
- `OpenMLE-Gym/openmle_gym/sandbox_exec.py` 提供 sandbox 执行的轻量入口；
- `OpenMLE-Gym/openmle_gym/metric_validation.py` 的 `evaluate_sample_submission*` 是 metric.py 冒烟测试的两种执行形态。

两者的共性是：**结果必须可复现、失败必须显式**——超时、非零退出、坏 JSON、worker 自报失败都折叠成结构化 Outcome，绝不静默。

## 在搜索中的位置

Evo-Max 把多个 sandbox URL 配置进 `async_sandbox_urls`，worker 按 round-robin 绑定端点（`airaevo_async_resources.py:123` 的 `build_worker_specs`），实现"多 GPU 生成 × 多沙箱执行"的流水线并行——这是论文中异步搜索吞吐优势的物理基础。
