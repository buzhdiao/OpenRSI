# 导入 asyncio，用于在同步钩子中运行异步函数（临时事件循环）
import asyncio
# 导入 dataclasses，用于声明 EngineGroup/RolloutServer 数据类
import dataclasses
# 导入 inspect，判断函数返回值是否为可等待对象
import inspect
# 导入 itertools，用于展平嵌套列表（chain.from_iterable）
import itertools
# 导入 logging，模块级日志
import logging
# 导入 multiprocessing，用于以子进程方式启动路由器
import multiprocessing
# 导入 os，读取环境变量
import os
# 导入 random，随机选择路由器/普罗米修斯端口范围
import random
# 导入 time，计时与休眠
import time
# 导入 Path，保存调试数据时构造路径
from pathlib import Path
# 导入 Any，用于类型标注
from typing import Any

# 导入 numpy，计算指标均值
import numpy as np
# 导入 ray，创建远程 actor
import ray
# 导入 torch，加载/保存调试 rollout 数据与张量计算
import torch
# 导入 wandb，实验跟踪（结束时调用 finish）
import wandb
# 导入放置组调度策略，把引擎 actor 绑到指定 bundle
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
# 导入 sglang 的显存类型常量：CUDA 图 / KV cache / 权重（offload 分阶段使用）
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

# 导入 SGLang 引擎的 Ray actor 封装
from slime.backends.sglang_utils.sglang_engine import SGLangEngine
# 导入 rollout 函数调用包装（统一处理 evaluation 开关等）
from slime.rollout.base_types import call_rollout_fn
# 导入日志工具模块
from slime.utils import logging_utils
# 导入健康监控器：容错模式下监测引擎存活性
from slime.utils.health_monitor import RolloutHealthMonitor
# 导入 HTTP 工具：IPv6 包装、找空闲端口、取主机信息、初始化 HTTP 客户端
from slime.utils.http_utils import _wrap_ipv6, find_available_port, get_host_info, init_http_client
# 导入日志工具：配置日志器、初始化跟踪后端（wandb 等）
from slime.utils.logging_utils import configure_logger, init_tracking
# 导入指标工具：通过率、步数、统计量、字典加前缀
from slime.utils.metric_utils import compute_pass_rate, compute_rollout_step, compute_statistics, dict_add_prefix
# 导入杂项工具：Box 包装、分组、按路径加载函数
from slime.utils.misc import Box, group_by, load_function
# 导入序列长度均衡分区工具（数据并行分桶）
from slime.utils.seqlen_balancing import get_seqlen_balanced_partitions
# 导入 Sample 类型：rollout 样本的标准数据结构
from slime.utils.types import Sample

# 从上级工具导入重复检测函数（计算重复率指标）
from ..utils.metric_utils import has_repetition
# 从本地 utils 导入环境变量常量与分布式锁
from .utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Lock

# 把 httpx 的日志级别压到 WARNING，避免刷屏
logging.getLogger("httpx").setLevel(logging.WARNING)
# 把 httpcore 的日志级别压到 WARNING
logging.getLogger("httpcore").setLevel(logging.WARNING)

# 创建本模块日志器
logger = logging.getLogger(__name__)


# 执行可能是同步或异步的钩子函数：同步直接调用，异步用临时事件循环跑完
def _run_maybe_awaitable(func, *args, **kwargs):
    """Execute sync hooks directly and async hooks via a temporary event loop."""
    # 先按同步方式调用函数
    result = func(*args, **kwargs)
    # 若返回值是协程/可等待对象
    if inspect.isawaitable(result):
        # 用 asyncio.run 在新事件循环中执行至完成
        return asyncio.run(result)
    # 同步结果直接返回
    return result


# 引擎组数据类：一组配置完全相同的 SGLang 引擎
@dataclasses.dataclass
class EngineGroup:
    """A group of homogeneous SGLang engines with the same configuration.

    All engines in a group share the same tp_size / nodes_per_engine / pg.
    A RolloutServer may contain multiple EngineGroups (e.g. prefill vs decode
    in PD disaggregation).
    """

    # 全局配置（Namespace）
    args: Any
    # 放置组三元组：(placement_group, 重排 bundle 索引, 重排 GPU id)
    pg: Any  # (placement_group, reordered_bundle_indices, reordered_gpu_ids)
    # 全部引擎句柄列表（死亡引擎为 None，便于故障恢复占位）
    all_engines: list
    # 每个引擎占用的节点数（跨节点大引擎 > 1）
    nodes_per_engine: int
    # 本次新启动的引擎数量（用于权重更新时识别新引擎）
    num_new_engines: int
    # 组角色："regular"/"prefill"/"decode"（PD 分离时区分）
    role: str = "regular"  # "regular", "prefill", or "decode"
    # 本组第一个引擎的全局 rank 偏移
    rank_offset: int = 0  # global rank of the first engine in this group

    # 属性：仅返回 node 0 的引擎（多节点引擎只需与 0 号节点交互）
    @property
    def engines(self):
        """Node-0 engines only (for multi-node serving)."""
        # 按每引擎节点数做切片抽样
        return self.all_engines[:: self.nodes_per_engine]

    # 启动本组所有缺失的引擎：创建 Ray actor、分配端口、异步发起 init
    def start_engines(self) -> list:
        """Create Ray actors, allocate ports, and fire ``engine.init()`` without waiting.

        Returns a list of Ray ObjectRefs for the init calls.  The caller
        should ``ray.get()`` on them to block until the engines are healthy.
        """
        # 只训练调试模式：不启动任何引擎
        if self.args.debug_train_only:
            # 新引擎数清零
            self.num_new_engines = 0
            # 返回空句柄列表
            return []

        # 每引擎 GPU 数取每引擎需求与单节点 GPU 数的较小值
        num_gpu_per_engine = min(self.args.rollout_num_gpus_per_engine, self.args.num_gpus_per_node)
        # 引擎总数 = rollout GPU 总数 ÷ 每引擎 GPU 数
        total_num_engines = self.args.rollout_num_gpus // num_gpu_per_engine

        # 解包放置组三元组
        pg, reordered_bundle_indices, reordered_gpu_ids = self.pg

        # 把 SGLangEngine 类包装成 Ray 远程类
        RolloutRayActor = ray.remote(SGLangEngine)

        # 本次新建的引擎列表：(全局 rank, actor 句柄)
        rollout_engines = []
        # 遍历全部引擎槽位
        for i in range(len(self.all_engines)):
            # 槽位已有引擎（存活）则跳过，只补建缺失的
            if self.all_engines[i] is not None:
                # 继续下一个槽位
                continue

            # 计算该槽位引擎的全局 rank
            global_rank = self.rank_offset + i
            # 引擎 actor 只占 0.2 GPU（实际 GPU 由 sglang 子进程占用，Ray 侧少占以便共置）
            num_gpus = 0.2
            # CPU 配额与 GPU 配额一致
            num_cpus = num_gpus

            # Get the base GPU ID from placement group
            # （原注释含义：从放置组获取起始 GPU 编号）
            # 用该引擎首 GPU 的重排 GPU id 作为 base_gpu_id
            base_gpu_id = int(reordered_gpu_ids[global_rank * num_gpu_per_engine])

            # 构造放置组调度策略：绑定首 GPU 所在 bundle，并捕获子任务
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=reordered_bundle_indices[global_rank * num_gpu_per_engine],
            )

            # 组装引擎进程的环境变量
            env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
                # 逐项取当前环境变量值（不存在则用默认值）
                key: os.environ.get(key, default_val)
                for key, default_val in {
                    # 关闭 DeepGEMM JIT 预编译（启动提速、行为稳定）
                    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
                    # 关闭 TP 显存不均衡检查（共置场景显存天然不均）
                    "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                    # 同上的新版变量名
                    "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                    # 内存节省器关闭 CUDA 图时使用该开关
                    "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
                    # 启用批量不变算子的 mm 回退变体（数值一致性）
                    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
                    # 健康检查端点不执行真实生成（避免干扰 offload）
                    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
                    # 空闲期不做严格显存检查
                    "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
                }.items()
            }

            # 以配置的选项创建引擎 actor
            rollout_engine = RolloutRayActor.options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                runtime_env={
                    # 注入环境变量
                    "env_vars": env_vars,
                },
            ).remote(self.args, rank=global_rank, worker_type=self.role, base_gpu_id=base_gpu_id)

            # 记录新建引擎
            rollout_engines.append((global_rank, rollout_engine))
            # 回填引擎槽位
            self.all_engines[i] = rollout_engine

        # 记录本次新建引擎数
        self.num_new_engines = len(rollout_engines)

        # 没有新引擎时直接返回
        if self.num_new_engines == 0:
            # 返回空列表
            return []

        # 外部引擎模式：从配置的地址列表分配地址端口
        if self.args.rollout_external:
            # 调用外部模式分配函数
            addr_and_ports = _allocate_rollout_engine_addr_and_ports_external(
                args=self.args, rollout_engines=rollout_engines
            )
        else:
            # 常规模式：自动分配地址端口
            addr_and_ports = _allocate_rollout_engine_addr_and_ports_normal(
                args=self.args,
                num_engines=total_num_engines,
                rollout_engines=rollout_engines,
                worker_type=self.role,
            )

        # 异步发起每个引擎的 init（传入各自的地址端口配置）
        init_handles = [engine.init.remote(**(addr_and_ports[rank])) for rank, engine in rollout_engines]
        # 返回 init 句柄列表，由调用方 ray.get 等待健康
        return init_handles

    # 对全部 node0 引擎发起显存释放（offload），非阻塞
    def offload(self):
        """Fire release_memory_occupation on all engines (non-blocking).

        Returns a list of Ray ObjectRefs.
        """
        # 对每个存活的 node0 引擎发起 release_memory_occupation
        return [engine.release_memory_occupation.remote() for engine in self.engines if engine is not None]

    # 对全部 node0 引擎发起显存恢复（onload），可按显存类型分阶段
    def onload(self, tags: list[str] | None = None):
        """Fire resume_memory_occupation on all engines (non-blocking).

        Returns a list of Ray ObjectRefs.
        """
        # 对每个存活的 node0 引擎发起 resume_memory_occupation，携带阶段标签
        return [engine.resume_memory_occupation.remote(tags=tags) for engine in self.engines if engine is not None]


# Rollout 服务器数据类：一个共享路由器后挂一个或多个引擎组
@dataclasses.dataclass
class RolloutServer:
    """A model served behind a shared router, with one or more engine groups.

    Corresponds to one entry in the future ``--sglang-config`` YAML::

        rollout_servers:
          - name: policy
            engine_groups:
              - role: prefill
                ...
              - role: decode
                ...

    Currently only a single EngineGroup is used for non-PD mode;
    PD disaggregation creates separate prefill and decode groups.
    """

    # 引擎组列表
    engine_groups: list[EngineGroup]
    # 路由器 IP（可为 None）
    router_ip: str | None = None
    # 路由器端口（可为 None）
    router_port: int | None = None

    # 属性：所有组的 node0 引擎汇总
    @property
    def engines(self):
        """All node-0 engines across all groups."""
        # 双层推导展开
        return [e for g in self.engine_groups for e in g.engines]

    # 属性：所有组的全部引擎（含非 node0）
    @property
    def all_engines(self):
        """All engines (including non-node-0) across all groups."""
        # 双层推导展开
        return [e for g in self.engine_groups for e in g.all_engines]

    # 属性：全部组新引擎数之和
    @property
    def num_new_engines(self):
        # 对各组求和
        return sum(g.num_new_engines for g in self.engine_groups)

    # setter：把新引擎数同步写到所有组
    @num_new_engines.setter
    def num_new_engines(self, value):
        # 遍历各组赋值
        for g in self.engine_groups:
            # 覆盖组内计数
            g.num_new_engines = value

    # 属性：每引擎节点数（要求所有组一致）
    @property
    def nodes_per_engine(self):
        """Nodes per engine. Only valid when all groups share the same value.

        TODO: remove once health_monitor operates per-group.
        """
        # 收集各组的节点数值集合
        values = {g.nodes_per_engine for g in self.engine_groups}
        # 不一致则断言失败
        assert len(values) == 1, f"Heterogeneous nodes_per_engine: {values}"
        # 弹出唯一值返回
        return values.pop()

    # 恢复所有组中死亡的引擎（并发初始化）
    def recover(self):
        """Recover dead engines across all groups, overlapping init."""
        # Record dead indices per group before starting.
        # （原注释含义：启动前记录各组中死亡引擎的槽位）
        # 对每组找出为 None 的槽位下标
        dead_per_group = [[i for i, engine in enumerate(g.all_engines) if engine is None] for g in self.engine_groups]

        # Start all groups concurrently.
        # （原注释含义：所有组并发启动）
        # 汇总所有组的 init 句柄
        all_handles = []
        # 遍历各组
        for g in self.engine_groups:
            # 并入该组新建引擎的 init 句柄
            all_handles.extend(g.start_engines())
        # 有句柄则阻塞等待全部初始化完成
        if all_handles:
            # 等待全部健康
            ray.get(all_handles)

        # Post-recovery: offload then onload weights for newly created engines.
        # （原英文注释：恢复后对新引擎先 offload 再只 onload 权重）
        # 汇总 release 句柄
        release_handles = []
        # 收集全部新引擎
        new_engines_all = []
        # 成对遍历各组与其死亡槽位（strict=True 确保长度一致）
        for g, dead_indices in zip(self.engine_groups, dead_per_group, strict=True):
            # 打印恢复数量与角色
            logger.info(f"Recovered {g.num_new_engines} dead rollout engines (role={g.role})")
            # 校验新建数与死亡数一致
            assert g.num_new_engines == len(dead_indices), "num_new_engines does not match dead_indices length"
            # 开启 rollout offload 且有死亡引擎时：把新引擎对齐到 offload 状态
            if g.args.offload_rollout and dead_indices:
                # 取出该组的新引擎
                new_engines = [g.all_engines[i] for i in dead_indices]
                # 对每个新引擎发起显存释放
                release_handles.extend(engine.release_memory_occupation.remote() for engine in new_engines)
                # 收集新引擎
                new_engines_all.extend(new_engines)

        # 有 release 句柄时：先全部释放，再只恢复权重部分的显存占用
        if release_handles:
            # 等待全部释放完成
            ray.get(release_handles)
            # 只 onload 权重（KV cache 与 CUDA 图等训练开始后再恢复）
            ray.get(
                [engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]) for engine in new_engines_all]
            )

    # 对所有组并发执行显存释放
    def offload(self):
        """Release memory occupation across all groups (concurrent)."""
        # 汇总句柄
        handles = []
        # 遍历各组
        for g in self.engine_groups:
            # 并入该组句柄
            handles.extend(g.offload())
        # 有句柄则等待完成，否则返回空列表
        return ray.get(handles) if handles else []

    # 对所有组并发执行显存恢复
    def onload(self, tags: list[str] | None = None):
        """Resume memory occupation across all groups (concurrent)."""
        # 汇总句柄
        handles = []
        # 遍历各组
        for g in self.engine_groups:
            # 并入该组句柄
            handles.extend(g.onload(tags))
        # 有句柄则等待完成，否则返回空列表
        return ray.get(handles) if handles else []


# RolloutManager：Ray 远程 actor，负责生成 rollout 数据并转换为训练数据
@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""

    # 构造函数：初始化数据源、rollout 函数、rollout 服务器与锁等
    def __init__(self, args, pg):
        # 配置日志器（级别、格式）
        configure_logger()

        # 保存放置组三元组
        self.pg = pg
        # 保存全局配置
        self.args = args

        # 初始化实验跟踪后端（本 actor 是 secondary）
        init_tracking(args, primary=False)

        # 按路径加载数据源类并实例化
        data_source_cls = load_function(self.args.data_source_path)
        # 保存数据源实例
        self.data_source = data_source_cls(args)

        # 加载训练用 rollout 生成函数
        self.generate_rollout = load_function(self.args.rollout_function_path)
        # 加载评估用 rollout 生成函数
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        # 自定义奖励后处理函数初始为 None
        self.custom_reward_post_process_func = None
        # 配置了路径则加载
        if self.args.custom_reward_post_process_path is not None:
            # 加载自定义奖励后处理函数
            self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
        # 自定义样本转训练数据函数初始为 None
        self.custom_convert_samples_to_train_data_func = None
        # 配置了路径则加载
        if self.args.custom_convert_samples_to_train_data_path is not None:
            # 加载自定义转换函数
            self.custom_convert_samples_to_train_data_func = load_function(
                self.args.custom_convert_samples_to_train_data_path
            )
        # 打印训练 rollout 函数来源
        logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
        # 打印评估 rollout 函数来源
        logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        # 只训练调试模式：不启动 rollout 服务器
        if self.args.debug_train_only:
            # 服务器置 None
            self.server = None
        else:
            # 初始化共享 HTTP 客户端（连接池复用）
            init_http_client(args)
            # 启动 rollout 服务器（路由器 + 引擎组）
            self.server = start_rollout_server(args, pg)
        # 创建全局广播锁 actor（权重同步互斥用）
        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()
        # rollout 轮次从 -1 开始（首次 generate 前无轮次）
        self.rollout_id = -1

        # 健康监控器列表
        self._health_monitors = []
        # 非调试且启用容错时：为每个引擎组启动健康监控
        if not self.args.debug_train_only and self.args.use_fault_tolerance:
            # 遍历引擎组
            for group in self.server.engine_groups:
                # 创建该组的健康监控器
                monitor = RolloutHealthMonitor(group, args)
                # 启动监控线程
                monitor.start()
                # 收集监控器
                self._health_monitors.append(monitor)
            # CI 故障注入待执行标志（仅 ci_test 时使用）
            self._ci_fault_injection_pending = self.args.ci_test  # Flag for CI fault injection

    # 尝试注入 CI 故障：在生成过程中（健康监控运行时）模拟引擎崩溃
    def _try_ci_fault_injection(self):
        """Try to inject fault during generate (when health monitor is running)."""
        # 无待注入标志则直接返回
        if not self._ci_fault_injection_pending:
            # 直接返回
            return

        # Only inject fault once
        # （原注释含义：只注入一次故障）
        # 清除待注入标志
        self._ci_fault_injection_pending = False

        # 服务器存在且首组首个引擎存活时才注入
        if self.server and self.server.engine_groups[0].all_engines and self.server.engine_groups[0].all_engines[0]:
            # 打印注入日志
            logger.info("CI Fault Injection: Simulating crash on engine 0 during generate")
            # 包裹异常避免注入失败影响主流程
            try:
                # This will cause the ray actor to exit
                # （原注释含义：这会导致该 ray actor 退出）
                # 远程调用引擎的 simulate_crash
                self.server.engine_groups[0].all_engines[0].simulate_crash.remote()
                # Wait for health monitor to detect the crash and mark engine as None
                # health_check_interval + health_check_timeout + buffer
                # （原英文注释：等健康监控发现崩溃并标记为 None）
                # 等待时间 = 检查间隔 + 检查超时 + 5 秒缓冲
                wait_time = self.args.rollout_health_check_interval + self.args.rollout_health_check_timeout + 5
                # 打印等待日志
                logger.info(f"CI Fault Injection: Waiting {wait_time}s for health monitor to detect crash")
                # 阻塞休眠等待
                time.sleep(wait_time)
            except Exception as e:
                # 注入失败仅告警
                logger.warning(f"CI Fault Injection failed: {e}")

    # 释放资源：停止健康监控并结束 wandb
    def dispose(self):
        # 停止所有健康监控器
        for monitor in self._health_monitors:
            # 逐个停止
            monitor.stop()
        # wandb 运行存在时正常收尾
        if wandb.run is not None:
            # 调用 finish 结束跟踪
            wandb.finish()

    # 属性：node0 引擎列表
    @property
    def rollout_engines(self):
        # 无服务器时返回空
        if self.server is None:
            # 空列表
            return []
        # 返回服务器的 node0 引擎
        return self.server.engines

    # 返回引擎列表与锁以及新引擎数（权重更新方使用）
    def get_rollout_engines_and_lock(self):
        # 无服务器时新引擎数为 0
        num_new_engines = self.server.num_new_engines if self.server else 0
        # 返回三元组
        return self.rollout_engines, self.rollout_engine_lock, num_new_engines

    # 查询每个 epoch 的 rollout 轮数
    def get_num_rollout_per_epoch(self):
        # 断言使用全局数据集
        assert self.args.rollout_global_dataset
        # 每 epoch 轮数 = 数据集样本数 ÷ 每轮 batch 大小
        return len(self.data_source) // self.args.rollout_batch_size

    # 执行一轮训练数据生成：生成 → 调试保存 → 日志 → 转训练数据 → 按 DP 切分
    def generate(self, rollout_id):
        # 记录开始时间
        start_time = time.time()
        # 记录当前轮次
        self.rollout_id = rollout_id
        # 恢复健康监控（生成期间要监控引擎）
        self.health_monitoring_resume()
        # CI 测试且容错开启且轮次达到 2 时注入故障
        if self.args.ci_test and self.args.use_fault_tolerance and rollout_id >= 2:
            # 执行故障注入
            self._try_ci_fault_injection()
        # 获取 rollout 数据与指标
        data, metrics = self._get_rollout_data(rollout_id=rollout_id)
        # 保存调试用 rollout 数据（若配置）
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=False)
        # 记录 rollout 日志与指标
        _log_rollout_data(rollout_id, self.args, data, metrics, time.time() - start_time)
        # 只调试 rollout 模式：不转训练数据直接返回
        if self.args.debug_rollout_only:
            # if debug rollout only, we don't convert samples to train data and directly return
            # （原英文注释：调试模式直接返回）
            # 返回 None
            return
        # 把样本转换为训练数据字典
        data = self._convert_samples_to_train_data(data)
        # 按数据并行度切分并放到对象存储，返回引用列表
        return self._split_train_data_by_dp(data, self.train_parallel_config["dp_size"])

    # 执行评估：生成评估数据并记录指标
    def eval(self, rollout_id):
        # 只训练调试模式：不生成评估数据
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            # （原英文注释：只训练调试时跳过评估）
            # 返回 None
            return
        # 恢复健康监控
        self.health_monitoring_resume()

        # 用评估函数执行 rollout（evaluation=True）
        result = call_rollout_fn(self.eval_generate_rollout, self.args, rollout_id, self.data_source, evaluation=True)
        # 取出样本数据
        data = result.data
        # 保存调试用评估数据（若配置）
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=True)
        # 记录评估日志并返回指标字典
        return _log_eval_rollout_data(rollout_id, self.args, data, result.metrics)

    # 保存数据源状态（如全局数据集游标）
    def save(self, rollout_id):
        # 委托给数据源的 save
        self.data_source.save(rollout_id)

    # 保存程序数据库快照（OpenRSI 的 program database 持久化）
    def save_db_snapshot(self, rollout_id):
        """Save program DB snapshot for the current rollout when enabled.

        This is implemented on the Ray actor class itself so
        `rollout_manager.save_db_snapshot.remote(...)` is always available,
        regardless of external monkey patches.
        """
        # 环境变量开关未打开则跳过
        if os.environ.get("SAVE_DB_SNAPSHOTS", "0") != "1":
            # 直接返回
            return

        # 包裹异常避免快照失败影响训练
        try:
            # 取自定义生成函数路径
            custom_generate_module_path = self.args.custom_generate_function_path
            # 未配置则无需快照
            if not custom_generate_module_path:
                # 直接返回
                return

            # 从函数路径中拆出模块路径
            module_path, _ = custom_generate_module_path.rsplit(".", 1)
            # 加载该模块的 save_db_snapshot 函数
            save_db_snapshot_func = load_function(f"{module_path}.save_db_snapshot")
            # 运行目录取环境变量，缺省为当前目录
            run_dir = os.environ.get("RUN_DIR", ".")
            # 快照文件路径：program_database_iter_{rollout_id}.db
            snapshot_path = os.path.join(run_dir, f"program_database_iter_{rollout_id}.db")
            # 调用快照函数（兼容同步/异步实现）
            _run_maybe_awaitable(save_db_snapshot_func, snapshot_path)
            # 打印保存成功日志
            logger.info("Database snapshot saved to %s", snapshot_path)
        except Exception as e:
            # 失败仅告警
            logger.warning("Failed to save database snapshot: %s", e)

    # 加载数据源状态（恢复训练时使用）
    def load(self, rollout_id=None):
        # 委托给数据源的 load
        self.data_source.load(rollout_id)

    # 卸载 rollout 引擎显存（训练阶段让出 GPU 显存）
    def offload(self):
        # 先暂停健康监控（offload 期间引擎不服务）
        self.health_monitoring_pause()
        # 服务器存在时执行整体 offload
        if self.server:
            # 返回 offload 结果
            return self.server.offload()

    # 恢复 rollout 引擎显存（可按类型分阶段）
    def onload(self, tags: list[str] | None = None):
        # 服务器存在时执行整体 onload
        if self.server:
            # 返回 onload 结果
            return self.server.onload(tags)

    # 只恢复权重部分显存
    def onload_weights(self):
        # 以权重标签 onload
        self.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    # 只恢复 KV cache 与 CUDA 图部分显存
    def onload_kv(self):
        # 以 KV cache + CUDA 图标签 onload
        self.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])

    # 恢复死亡的 rollout 引擎并返回新引擎数（供权重更新识别）
    def recover_rollout_engines(self):
        """Restart any dead rollout engines and update num_new_engines for update_weights detection."""
        # 恢复期间暂停健康监控
        self.health_monitoring_pause()
        # 取服务器引用
        srv = self.server
        # 尚未开始生成时无需恢复
        if self.rollout_id == -1:
            # 直接返回当前引擎与锁
            return self.rollout_engines, self.rollout_engine_lock, srv.num_new_engines

        # 执行服务器级恢复（重建死亡引擎）
        srv.recover()
        # 返回引擎列表、锁与新引擎数
        return self.rollout_engines, self.rollout_engine_lock, srv.num_new_engines

    # 清零新引擎计数（无容错时权重更新后手动清）
    def clear_num_new_engines(self):
        # when fault tolerance is not enabled, we need to manually clear num_new_engines after update_weights
        # （原英文注释：未开启容错时需在权重更新后手动清零）
        # 服务器存在时清零
        if self.server:
            # 设为 0
            self.server.num_new_engines = 0

    # 暂停所有健康监控
    def health_monitoring_pause(self) -> None:
        # 遍历监控器逐个暂停
        for monitor in self._health_monitors:
            # 暂停
            monitor.pause()

    # 恢复所有健康监控
    def health_monitoring_resume(self) -> None:
        # 遍历监控器逐个恢复
        for monitor in self._health_monitors:
            # 恢复
            monitor.resume()

    # 在所有引擎上执行权重校验动作（调试用）
    def check_weights(self, action: str):
        # 对每个 node0 引擎远程调用 check_weights 并等待
        return ray.get([engine.check_weights.remote(action=action) for engine in self.rollout_engines])

    # 获取 rollout 数据：调试加载或调用生成函数，并做裁剪对齐
    def _get_rollout_data(self, rollout_id):
        # 配置了调试数据加载路径时：直接从磁盘读
        if self.args.load_debug_rollout_data:
            # torch.load 反序列化调试数据
            data = torch.load(
                self.args.load_debug_rollout_data.format(rollout_id=rollout_id),
                weights_only=False,
            )["samples"]
            # 把字典还原成 Sample 对象
            data = [Sample.from_dict(sample) for sample in data]
            # 配置了子采样比例时保留头尾各一半（保持分布多样性）
            if (ratio := self.args.load_debug_rollout_data_subsample) is not None:
                # 记录原始行数
                original_num_rows = len(data)
                # 粗略目标行数 = 原行数 × 比例
                rough_subsample_num_rows = int(original_num_rows * ratio)
                # 取头部一半与尾部一半拼接
                data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
                # 打印子采样日志
                logger.info(
                    f"Subsample loaded debug rollout data using {ratio=} and change num rows {original_num_rows} -> {len(data)}"
                )
            # 调试加载无指标
            metrics = None
        else:
            # 正常路径：调用训练 rollout 函数
            data = call_rollout_fn(self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False)
            # 取出指标
            metrics = data.metrics
            # 取出样本列表
            data = data.samples
            # flatten the data if it is a list of lists
            # （原注释含义：若数据是列表的列表则展平）
            # 循环展平直到元素不再是列表
            while isinstance(data[0], list):
                # 用 chain 展平一层
                data = list(itertools.chain.from_iterable(data))

            # 未禁用样本裁剪且非只调试 rollout 时：对齐全局 batch
            if not self.args.disable_rollout_trim_samples and not self.args.debug_rollout_only:
                # 取全局 batch 大小
                global_batch_size = self.args.global_batch_size
                # 启用动态全局 batch 时重新计算
                if self.args.use_dynamic_global_batch_size:
                    # 打印收集样本数日志
                    logger.info(f"Collected {len(data)} samples from rollout to train with dynamic global batch size")
                    # TODO: this is a temporary solution, we should directly save dynamic_global_batch_size to rollout data
                    # （保留原 TODO）
                    # 计算并缓存动态全局 batch
                    self._dynamic_global_batch_size = self._compute_dynamic_global_batch_size(len(data))
                    # 使用动态值
                    global_batch_size = self._dynamic_global_batch_size

                # 样本数不是全局 batch 整数倍时截断多余样本
                if len(data) % global_batch_size != 0:
                    # 截断长度 = 向下取整到整倍数
                    trim_len = (len(data) // global_batch_size) * global_batch_size
                    # 截断后为 0 说明样本不足
                    if trim_len == 0:
                        # 抛出值错误
                        raise ValueError(f"Not enough samples {len(data)} for global_batch_size {global_batch_size}")
                    # 记录原始长度
                    origin_data_length = len(data)
                    # 截断数据
                    data = data[:trim_len]
                    # 打印裁剪日志
                    logger.info(f"trim number of samples from {origin_data_length} to {trim_len}")
                # 打印最终样本数
                logger.info(f"Final collected {len(data)} samples from rollout to train")

        # 返回样本与指标
        return data, metrics

    # 计算动态全局 batch：保证本轮只有一个训练步
    def _compute_dynamic_global_batch_size(self, num_samples: int) -> int:
        """Calculate dynamic global_batch_size to ensure only one training step.

        Strategy: global_batch_size = num_samples rounded down to a multiple of dp_size
        This ensures num_steps_per_rollout = num_samples // global_batch_size = 1
        """
        # 取数据并行度
        dp_size = self.train_parallel_config["dp_size"]
        # 记录原始全局 batch
        original_gbs = self.args.global_batch_size

        # Round down to a multiple of dp_size to ensure only one training step
        # （原英文注释：向下取整到 dp 的倍数保证单步训练）
        # 动态全局 batch = 样本数向下取整到 dp 倍数
        dynamic_gbs = (num_samples // dp_size) * dp_size

        # 样本过少时至少用 dp_size
        if dynamic_gbs == 0:
            # Too few samples, use at least dp_size
            # （原注释含义：样本太少时至少使用 dp_size）
            # 使用 dp_size 作为全局 batch
            dynamic_gbs = dp_size
            # 打印警告
            logger.warning(f"num_samples={num_samples} < dp_size={dp_size}, using dp_size as global_batch_size")

        # Calculate how many samples will be discarded
        # （原注释含义：计算将被丢弃的样本数）
        # 浪费样本数 = 总数 - 动态 batch
        wasted = num_samples - dynamic_gbs

        # 有变化或有浪费时打印信息
        if dynamic_gbs != original_gbs or wasted > 0:
            # 打印动态调整详情
            logger.info(
                f"Dynamic global_batch_size: {original_gbs} -> {dynamic_gbs} (num_samples={num_samples}, dp_size={dp_size}, num_steps=1, wasted={wasted})"
            )

        # 返回动态全局 batch
        return dynamic_gbs

    # 保存调试用 rollout 数据到磁盘（若配置了路径模板）
    def _save_debug_rollout_data(self, data, rollout_id, evaluation: bool):
        # TODO to be refactored (originally Buffer._set_data)
        # （原 TODO 含义：待重构，原实现位于 Buffer._set_data）
        # 未配置保存路径则跳过
        if (path_template := self.args.save_debug_rollout_data) is not None:
            # 模板填入 rollout id（评估时加 eval_ 前缀）
            path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
            # 打印保存路径
            logger.info(f"Save debug rollout data to {path}")
            # 确保父目录存在
            path.parent.mkdir(parents=True, exist_ok=True)

            # TODO may improve the format
            # （原 TODO 含义：导出格式可再改进）
            # 评估数据结构不同（按数据集分组）需展平样本
            if evaluation:
                # 组装评估调试数据
                dump_data = dict(
                    samples=[sample.to_dict() for dataset_name, info in data.items() for sample in info["samples"]]
                )
            else:
                # 训练数据直接逐样本转字典
                dump_data = dict(
                    samples=[sample.to_dict() for sample in data],
                )

            # torch.save 保存（附带 rollout id）
            torch.save(dict(rollout_id=rollout_id, **dump_data), path)

    # 奖励后处理：返回 (原始奖励, 处理后奖励)；GRPO 类做组内去均值/标准化
    def _post_process_rewards(self, samples: list[Sample] | list[list[Sample]]):
        # 配置了自定义后处理函数则直接委托
        if self.custom_reward_post_process_func is not None:
            # 调用自定义函数
            return self.custom_reward_post_process_func(self.args, samples)

        # 逐样本取奖励值
        raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
        # GRPO/GSPO/Reinforce++ 基线且开启奖励归一化时做组内归一化
        if (
            self.args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
            and self.args.rewards_normalization
        ):
            # group norm
            # （原注释含义：组内归一化）
            # 把奖励转为浮点张量
            rewards = torch.tensor(raw_rewards, dtype=torch.float)
            # 样本数与"每提示采样数 × 批大小"一致时按组重排
            if rewards.shape[-1] == self.args.n_samples_per_prompt * self.args.rollout_batch_size:
                # reshape 成 (批大小, 每提示采样数)
                rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
            else:
                # when samples count are not equal in each group
                # （原注释含义：各组样本数不相等时）
                # 按实际样本数均分成组
                rewards = rewards.view(-1, rewards.shape[-1])
            # 计算每组的均值（保留维度便于广播）
            mean = rewards.mean(dim=-1, keepdim=True)
            # 每组减去均值（组内中心化）
            rewards = rewards - mean

            # GRPO/GSPO 且开启 std 归一化时再除以组内标准差
            if self.args.advantage_estimator in ["grpo", "gspo"] and self.args.grpo_std_normalization:
                # 计算组内标准差
                std = rewards.std(dim=-1, keepdim=True)
                # 除以标准差（加小量防零）
                rewards = rewards / (std + 1e-6)

            # 返回原始奖励与展平后的归一化奖励
            return raw_rewards, rewards.flatten().tolist()

        # 不做归一化时：处理后奖励就是原始奖励
        return raw_rewards, raw_rewards

    # 把推理生成的样本转换为训练数据字典
    def _convert_samples_to_train_data(self, samples: list[Sample] | list[list[Sample]]):
        """
        Convert inference generated samples to training data.
        """
        # 配置了自定义转换函数则直接委托
        if self.custom_convert_samples_to_train_data_func is not None:
            # 调用自定义转换
            return self.custom_convert_samples_to_train_data_func(self.args, samples)

        # 先做奖励后处理
        raw_rewards, rewards = self._post_process_rewards(samples)

        # 校验原始奖励数量与样本一致
        assert len(raw_rewards) == len(samples)
        # 校验处理后奖励数量与样本一致
        assert len(rewards) == len(samples)

        # 组装训练数据字典
        train_data = {
            # 每个样本的 token 序列
            "tokens": [sample.tokens for sample in samples],
            # 每个样本的响应长度
            "response_lengths": [sample.response_length for sample in samples],
            # some reward model, e.g. remote rm, may return multiple rewards,
            # we could use key to select the reward.
            # （原英文注释：某些 RM 返回多个奖励，可用 key 选取）
            # 归一化后的奖励
            "rewards": rewards,
            # 原始奖励
            "raw_reward": raw_rewards,
            # 是否被截断（1/0）
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
            # 样本索引（用于去重与追踪）
            "sample_indices": [sample.index for sample in samples],
        }

        # loss mask
        # （原注释：损失掩码）
        # TODO: compress the loss mask
        # （原 TODO 含义：掩码目前未压缩，可优化存储）
        # 初始化掩码列表
        loss_masks = []
        # 遍历样本处理掩码
        for sample in samples:
            # always instantiate loss_mask if not provided
            # （原注释含义：未提供损失掩码时总是构造一个）
            # 缺失时按响应长度填 1
            if sample.loss_mask is None:
                # 全 1 掩码
                sample.loss_mask = [1] * sample.response_length

            # 校验掩码长度与响应长度一致
            assert (
                len(sample.loss_mask) == sample.response_length
            ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
            # 标记移除的样本掩码全 0（不参与损失）
            if sample.remove_sample:
                # 全 0 掩码
                sample.loss_mask = [0] * sample.response_length
            # 收集掩码
            loss_masks.append(sample.loss_mask)
        # 写入训练数据
        train_data["loss_masks"] = loss_masks

        # overwriting the raw reward
        # （原注释含义：覆盖原始奖励）
        # 元数据带 raw_reward 时以元数据为准
        if samples[0].metadata and "raw_reward" in samples[0].metadata:
            # 用元数据中的 raw_reward 覆盖
            train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

        # For rollout buffer
        # （原注释含义：供 rollout buffer 使用——多轮交互的轮次号）
        # 元数据带 round_number 时透传
        if samples[0].metadata and "round_number" in samples[0].metadata:
            # 写入轮次号列表
            train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

        # Add rollout log probabilities for off-policy correction
        # （原英文注释：rollout 时的对数概率，用于 off-policy 修正）
        # 存在 rollout 对数概率时透传
        if samples[0].rollout_log_probs is not None:
            # 写入 rollout_log_probs
            train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

        # MoE 路由重放：rollout 时记录的路由专家
        if samples[0].rollout_routed_experts is not None:
            # 写入 rollout_routed_experts
            train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]

        # 训练侧元数据透传
        if samples[0].train_metadata is not None:
            # 写入 metadata
            train_data["metadata"] = [sample.train_metadata for sample in samples]

        # 任一样本带多模态训练输入时透传
        if any(sample.multimodal_train_inputs is not None for sample in samples):
            # 写入 multimodal_train_inputs
            train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

        # 教师模型对数概率透传（蒸馏用）
        if samples[0].teacher_log_probs is not None:
            # 写入 teacher_log_probs
            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

        # 返回训练数据字典
        return train_data

    # 训练侧并行配置注入（由外部调用）
    def set_train_parallel_config(self, config: dict):
        # 保存配置
        self.train_parallel_config = config

    # 按数据并行度把训练数据切分并放入 Ray 对象存储
    def _split_train_data_by_dp(self, data, dp_size):
        """Split the train data by data parallel size."""
        # 初始化输出字典
        rollout_data = {}

        # prompt 字段是全局共享的，原样保留
        if "prompt" in data:
            # 保留 prompt
            rollout_data["prompt"] = data["prompt"]

        # 计算每条样本的总 token 长度
        total_lengths = [len(t) for t in data["tokens"]]
        # 写入总长度
        data["total_lengths"] = total_lengths

        # 开启均衡时按序列长度均衡分区（各 DP 分片负载相近）
        if self.args.balance_data:
            # 调用均衡分区工具（等分）
            partitions = get_seqlen_balanced_partitions(total_lengths, dp_size, equal_size=True)
        else:
            # 否则按轮转（stride）分区：i, i+dp, i+2dp...
            partitions = [range(i, len(total_lengths), dp_size) for i in range(dp_size)]

        # 初始化输出引用列表
        rollout_data_refs = []

        # 为每个 DP 分片组装数据
        for i in range(dp_size):
            # 每片的输出字典
            rollout_data = {}
            # 取该片的样本下标分区
            partition = partitions[i]
            # 记录分区（训练侧需要知道自己拿哪些样本）
            rollout_data["partition"] = partition
            # 逐字段按分区抽取
            for key in [
                # token 序列
                "tokens",
                # 多模态输入
                "multimodal_train_inputs",
                # 响应长度
                "response_lengths",
                # 奖励
                "rewards",
                # 截断标记
                "truncated",
                # 损失掩码
                "loss_masks",
                # 轮次号
                "round_number",
                # 样本索引
                "sample_indices",
                # rollout 对数概率
                "rollout_log_probs",
                # 路由专家
                "rollout_routed_experts",
                # 提示词
                "prompt",
                # 教师对数概率
                "teacher_log_probs",
            ]:
                # 数据中不存在该字段则跳过
                if key not in data:
                    # 继续
                    continue
                # 按分区下标抽取该字段的值
                val = [data[key][j] for j in partition]
                # 写入分片数据
                rollout_data[key] = val
            # keys that need to be splited at train side
            # （原注释含义：这些字段留到训练侧再切分）
            # 这些字段全量传给每个分片，训练侧再各自切
            for key in [
                # 原始奖励
                "raw_reward",
                # 总长度
                "total_lengths",
            ]:
                # 不存在则跳过
                if key not in data:
                    # 继续
                    continue
                # 全量写入
                rollout_data[key] = data[key]
            # Pass dynamic global_batch_size to training side
            # （原注释含义：把动态全局 batch 大小传给训练侧）
            # 存在动态 batch 时写入分片数据
            if hasattr(self, "_dynamic_global_batch_size"):
                # 写入动态全局 batch
                rollout_data["dynamic_global_batch_size"] = self._dynamic_global_batch_size
            # 把分片数据放入对象存储并用 Box 包装引用
            rollout_data_refs.append(Box(ray.put(rollout_data)))
        # 返回引用列表
        return rollout_data_refs

# 为外部引擎分配地址与端口（直接使用配置好的地址列表）
def _allocate_rollout_engine_addr_and_ports_external(args, rollout_engines):
    # 初始化地址端口列表
    addr_and_ports = []
    # 遍历每个 (rank, 引擎) 对
    for rank, _ in rollout_engines:
        # 从配置中取该 rank 的地址
        addr = args.rollout_external_engine_addrs[rank]
        # 按冒号拆出 host 与 port
        [host, port] = addr.split(":")
        # 组装该引擎的配置字典
        addr_and_ports.append(
            dict(
                # 分布式初始化地址
                dist_init_addr=addr,
                # 外部引擎无需分配 NCCL 端口
                nccl_port=None,
                # 主机名
                host=host,
                # 服务端口（转 int）
                port=int(port),
            )
        )
    # 返回列表
    return addr_and_ports


# 为常规引擎分配地址与端口：服务端口、NCCL 端口、dist init 端口、dp attention 端口
def _allocate_rollout_engine_addr_and_ports_normal(*, args, num_engines, rollout_engines, worker_type="regular"):
    # get ports
    # （原注释含义：分配端口）
    # there are 4 ports we need to allocate
    # 1. server port
    # 2. nccl port
    # 3. dist_init_addr port
    # 4. other ports for dp_attention, which is of size 4 + dp_size
    # （原英文注释：需分配 4 类端口，dp_attention 需要 4+dp_size 个）
    # 每节点引擎数 = min(每节点 GPU, rollout GPU) ÷ 每引擎 GPU（至少 1）
    num_engines_per_node = max(
        1, min(args.num_gpus_per_node, args.rollout_num_gpus) // args.rollout_num_gpus_per_engine
    )
    # 为全部引擎初始化空配置
    addr_and_ports = [{} for _ in range(num_engines)]

    # 已访问节点集合（按节点批量分配端口）
    visited_nodes = set()
    # 遍历新建引擎
    for rank, engine in rollout_engines:
        # 该引擎所在节点已处理过则跳过
        if rank // num_engines_per_node in visited_nodes:
            # 继续
            continue
        # 标记该节点已访问
        visited_nodes.add(rank // num_engines_per_node)
        # TODO: currently when restarting engines, we will set port for all engines on this node starting with this rank.
        # e.g. for 8 gpus, if we are restarting engine on gpu 3, we will set port for engine 3,4,5,6,7 on this node.
        # （保留原 TODO：重启时从该 rank 起给本节点所有引擎重新分配端口）
        # 本节点需要处理的引擎数（从当前 rank 到节点末尾）
        num_engines_on_this_node = num_engines_per_node - (rank % num_engines_per_node)

        # 闭包工厂：为指定引擎生成取地址与取端口的函数
        def get_addr_and_ports(engine):
            # use small ports to prevent ephemeral port between 32768 and 65536.
            # also, ray uses port 10002-19999, thus we avoid near-10002 to avoid racing condition
            # （原英文注释：用小端口避开临时端口段与 Ray 的 10002-19999）
            # 起始端口 15000
            start_port = 15000

            # 内部取端口函数：向引擎所在节点申请空闲端口
            def port(consecutive=1):
                # nonlocal 让闭包记住已分配到的位置
                nonlocal start_port
                # 远程调用引擎的静态方法获取连续空闲端口
                _, port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                # 更新起始端口为本次端口 + 数量，保证下次不重复
                start_port = port + consecutive
                # 返回端口号
                return port

            # 内部取地址函数
            def addr():
                # 远程调用获取节点 IP
                addr, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                # 返回地址
                return addr

            # 返回 (取地址, 取端口) 函数对
            return addr, port

        # 为该节点首个引擎生成取地址/端口闭包
        get_addr, get_port = get_addr_and_ports(engine)

        # 逐个为本节点引擎分配基础端口
        for i in range(num_engines_on_this_node):
            # 当前引擎的全局 rank
            current_rank = rank + i
            # 分配 host
            addr_and_ports[current_rank]["host"] = get_addr()
            # 分配服务端口
            addr_and_ports[current_rank]["port"] = get_port()
            # 分配 NCCL 端口
            addr_and_ports[current_rank]["nccl_port"] = get_port()

            # prefill 引擎还需 PD 分离的引导端口
            if worker_type == "prefill":
                # 分配引导端口
                addr_and_ports[current_rank]["disaggregation_bootstrap_port"] = get_port()

        # 引擎跨节点时（每引擎 GPU 数 > 每节点 GPU 数）需统一 dist_init_addr
        if args.rollout_num_gpus_per_engine > args.num_gpus_per_node:
            # 每引擎占用的节点数
            num_node_per_engine = args.rollout_num_gpus_per_engine // args.num_gpus_per_node
            # 本 rank 是引擎的首节点时分配共享的 dist_init_addr
            if rank % num_node_per_engine == 0:
                # this is the first node in the engine, we need to allocate the dist_init_addr port
                # （原英文注释：引擎首节点分配 dist_init_addr 端口）
                # 地址用本节点 IP；端口申请 30+dp_size 个连续空闲（dp_attention 用）
                dist_init_addr = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"
                # 同一引擎的所有节点共用该地址
                for i in range(num_node_per_engine):
                    # 写入 dist_init_addr
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            # 单节点引擎：每个引擎独立分配 dist_init_addr（同样 30+dp_size 个连续端口）
            for i in range(num_engines_on_this_node):
                # 写入 dist_init_addr
                addr_and_ports[rank + i]["dist_init_addr"] = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"

    # 最终校验并打印每个引擎的端口配置
    for i, _ in rollout_engines:
        # 校验三类必要端口都已分配
        for key in ["port", "nccl_port", "dist_init_addr"]:
            # 缺失则断言失败
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        # 打印该引擎端口信息
        logger.info(f"Ports for engine {i}: {addr_and_ports[i]}")

    # 返回完整配置列表
    return addr_and_ports


# 启动路由器（sgl router 或 slime 自带 router），返回 (IP, 端口)
def _start_router(args) -> tuple[str, int]:
    """Start sgl router or slime router and return (router_ip, router_port).

    If ``args.sglang_router_ip`` is already set (e.g. by the user), skip
    launching and return the existing values.
    """
    # 用户已指定路由器地址时直接复用，不再启动
    if args.sglang_router_ip is not None:
        # 返回既有配置
        return args.sglang_router_ip, args.sglang_router_port

    # 路由器 IP：本机 IP 并做 IPv6 包装
    router_ip = _wrap_ipv6(get_host_info()[1])
    # 先取配置的端口
    router_port = args.sglang_router_port
    # 未指定时在 3000-4000 范围随机找一个空闲端口
    if router_port is None:
        # 随机起找空闲端口
        router_port = find_available_port(random.randint(3000, 4000))

    # 使用 slime 自带 router
    if args.use_slime_router:
        # slime router 不支持 prefill 多服务器配置
        assert args.prefill_num_servers is None, "slime router does not support prefill_num_servers."
        # 延迟导入 slime router 入口
        from slime.router.router import run_router

        # slime router reads ip/port from args at startup
        # （原注释含义：slime router 启动时从 args 中读取 ip/port）
        # 把地址端口写回 args
        args.sglang_router_ip = router_ip
        # 写回端口
        args.sglang_router_port = router_port
        # router 参数直接用全局 args
        router_args = args

    else:
        # 使用 sglang 官方 router
        from sglang_router.launch_router import RouterArgs

        # 从 http_utils 导入官方 router 启动函数
        from slime.utils.http_utils import run_router

        # 从 CLI 参数构造 RouterArgs（启用 router 前缀）
        router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)
        # 设置 host
        router_args.host = router_ip
        # 设置端口
        router_args.port = router_port
        # Prometheus 指标端口在 4000-5000 范围随机找
        router_args.prometheus_port = find_available_port(random.randint(4000, 5000))
        # 日志级别 warn
        router_args.log_level = "warn"
        # 请求超时来自配置（rollout 长请求场景）
        router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

        # 配置了路由策略（如一致性哈希）则覆盖
        if hasattr(args, "sglang_router_policy") and args.sglang_router_policy:
            # 设置策略
            router_args.policy = args.sglang_router_policy

        # 配置 prefill 服务器数量时开启 PD 分离
        if args.prefill_num_servers is not None:
            # 开启 pd_disaggregation
            router_args.pd_disaggregation = True

        # 打印路由器启动参数
        logger.info(f"Launch router with args: {router_args}")

    # 以子进程方式启动路由器
    process = multiprocessing.Process(
        target=run_router,
        args=(router_args,),
    )
    process.daemon = True  # Set the process as a daemon
    # （行尾原注释：把该进程设为守护进程）
    # 启动进程
    process.start()
    # Wait 3 seconds
    # （原注释含义：等待 3 秒）
    # 休眠 3 秒给路由器启动时间
    time.sleep(3)
    # 断言进程还活着（启动失败立即报错）
    assert process.is_alive()
    # 打印路由器地址
    logger.info(f"Router launched at {router_ip}:{router_port}")
    # 返回地址与端口
    return router_ip, router_port


# 启动完整的 rollout 服务器：一个路由器 + 一组引擎组
def start_rollout_server(args, pg) -> RolloutServer:
    """Start a complete rollout server: one router + a set of SGLang engines.

    Combines router startup and engine initialization into a single
    operation. Each RolloutServer represents one model served behind
    a shared router, containing one or more EngineGroups.

    When ``args.prefill_num_servers`` is set, creates separate prefill
    and decode EngineGroups for PD disaggregation.

    Note: init_http_client should be called separately before this,
    as the HTTP client is shared across all servers.
    """
    # 先启动路由器
    router_ip, router_port = _start_router(args)
    # Write back for backward compatibility: downstream code (SGLangEngine,
    # rollout functions, examples) still reads args.sglang_router_ip/port.
    # TODO: remove once all consumers read from RolloutServer directly.
    # （保留原注释与 TODO：写回 args 以兼容旧消费方）
    args.sglang_router_ip = router_ip
    # 写回端口
    args.sglang_router_port = router_port

    # 每引擎 GPU 数（单节点内）
    num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, args.num_gpus_per_node)
    # 引擎总数
    total_num_engines = args.rollout_num_gpus // num_gpu_per_engine
    # 每引擎节点数（跨节点引擎 > 1，至少 1）
    nodes_per_engine = max(1, args.rollout_num_gpus_per_engine // args.num_gpus_per_node)

    # 配置了 prefill 服务器数量时走 PD 分离路径
    if args.prefill_num_servers is not None:
        # prefill 引擎数 = prefill 服务器数 × 每引擎 GPU ÷ 单节点引擎 GPU
        prefill_engine_count = args.prefill_num_servers * args.rollout_num_gpus_per_engine // num_gpu_per_engine
        # decode 引擎数 = 总数 - prefill 数
        decode_engine_count = total_num_engines - prefill_engine_count
        # 断言 decode 引擎数大于 0
        assert decode_engine_count > 0, f"No decode engines: total {total_num_engines}, prefill {prefill_engine_count}"

        # 创建 prefill 引擎组（rank 从 0 开始）
        prefill_group = EngineGroup(
            args=args,
            pg=pg,
            all_engines=[None] * prefill_engine_count,
            nodes_per_engine=nodes_per_engine,
            num_new_engines=0,
            role="prefill",
            rank_offset=0,
        )
        # 创建 decode 引擎组（rank 接在 prefill 之后）
        decode_group = EngineGroup(
            args=args,
            pg=pg,
            all_engines=[None] * decode_engine_count,
            nodes_per_engine=nodes_per_engine,
            num_new_engines=0,
            role="decode",
            rank_offset=prefill_engine_count,
        )

        # Start both groups concurrently — the heavy work (sglang server
        # startup + health-check) runs inside engine.init() on the Ray
        # workers, so overlapping the two groups nearly halves wall time.
        # （原英文注释：两组并发启动，重叠初始化几乎省一半墙钟时间）
        # 合并两组的 init 句柄
        all_handles = prefill_group.start_engines() + decode_group.start_engines()
        # 阻塞等待全部引擎健康
        ray.get(all_handles)

        # 引擎组列表为 [prefill, decode]
        engine_groups = [prefill_group, decode_group]
    else:
        # 非 PD 模式：单一常规引擎组
        group = EngineGroup(
            args=args,
            pg=pg,
            all_engines=[None] * total_num_engines,
            nodes_per_engine=nodes_per_engine,
            num_new_engines=0,
        )
        # 启动并等待全部引擎健康
        ray.get(group.start_engines())
        # 引擎组列表只有一组
        engine_groups = [group]

    # 返回 RolloutServer 实例
    return RolloutServer(
        engine_groups=engine_groups,
        router_ip=router_ip,
        router_port=router_port,
    )


# 记录评估 rollout 的指标日志
def _log_eval_rollout_data(rollout_id, args, data, extra_metrics: dict[str, Any] | None = None):
    # 配置了自定义评估日志函数时先调用，返回真值则跳过默认逻辑
    if args.custom_eval_rollout_log_function_path is not None:
        # 按路径加载自定义函数
        custom_log_func = load_function(args.custom_eval_rollout_log_function_path)
        # 调用并检查是否已处理
        if custom_log_func(rollout_id, args, data, extra_metrics):
            # 已处理直接返回
            return

    # 以额外指标为底，逐步补充
    log_dict = extra_metrics or {}
    # 遍历每个数据集
    for key in data.keys():
        # 取该数据集的奖励列表
        rewards = data[key]["rewards"]
        # 记录平均奖励
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
        # 有样本明细时补充样本级指标
        if (samples := data[key].get("samples")) is not None:
            # 计算样本指标并加 eval/{key}/ 前缀合并
            log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), f"eval/{key}/")
        # 有截断记录时计算截断率
        if "truncated" in data[key]:
            # 取截断标记列表
            truncated = data[key]["truncated"]
            # 计算截断比例
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)
        # 开启通过率日志时计算 pass@k 类指标
        if args.log_passrate:
            # 合并通过率指标（按评估提示的采样数分组）
            log_dict |= dict_add_prefix(
                compute_pass_rate(
                    flat_rewards=rewards,
                    group_size=args.n_samples_per_eval_prompt,
                ),
                f"eval/{key}-",
            )

    # 打印评估摘要
    logger.info(f"eval {rollout_id}: {log_dict}")

    # 计算全局训练步数
    step = compute_rollout_step(args, rollout_id)
    # 写入评估步数
    log_dict["eval/step"] = step
    # 通过统一日志接口上报（step 键为 eval/step）
    logging_utils.log(args, log_dict, step_key="eval/step")

    # 返回日志字典
    return log_dict


# 记录训练 rollout 的指标日志（自定义函数优先，调试加载跳过）
def _log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    # 配置了自定义日志函数时先调用
    if args.custom_rollout_log_function_path is not None:
        # 按路径加载
        custom_log_func = load_function(args.custom_rollout_log_function_path)
        # 返回真值表示已处理
        if custom_log_func(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
            # 直接返回
            return

    # 调试数据加载模式不记录（数据不是真实生成的）
    if args.load_debug_rollout_data:
        # 直接返回
        return

    # 以额外 rollout 指标为底
    log_dict = {**(rollout_extra_metrics or {})}
    # 合并样本级指标（rollout/ 前缀）
    log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), "rollout/")
    # 合并性能指标（perf/ 前缀）
    log_dict |= dict_add_prefix(compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
    # 打印性能摘要
    logger.info(f"perf {rollout_id}: {log_dict}")
    # 计算训练步数
    step = compute_rollout_step(args, rollout_id)
    # 写入 rollout 步数
    log_dict["rollout/step"] = step
    # 上报日志
    logging_utils.log(args, log_dict, step_key="rollout/step")


# 从样本列表计算质量类指标（响应长度/零方差/错误类别/重复率/截断率）
def compute_metrics_from_samples(args, samples):
    # 收集有效响应长度（考虑移除样本等定义）
    response_lengths = [sample.effective_response_length for sample in samples]

    # 初始化日志字典
    log_dict = {}
    # 合并响应长度统计（均值/方差/分位数等）
    log_dict |= dict_add_prefix(compute_statistics(response_lengths), "response_len/")
    # 合并零方差组指标
    log_dict |= _compute_zero_std_metrics(args, samples)
    # 合并奖励类别指标
    log_dict |= _compute_reward_cat_metrics(args, samples)
    # 计算响应出现重复的样本比例
    log_dict["repetition_frac"] = np.mean([int(has_repetition(s.response)) for s in samples]).item()
    # 计算截断样本比例
    log_dict["truncated_ratio"] = np.mean([int(s.status == Sample.Status.TRUNCATED) for s in samples]).item()
    # 返回指标
    return log_dict


# 从样本列表计算性能指标（吞吐、非生成时间等）
def compute_perf_metrics_from_samples(args, samples, rollout_time):
    # 收集每个样本的非生成时间（排队/网络等）
    non_generation_time = [sample.non_generation_time for sample in samples]

    # 初始化日志字典
    log_dict = {}
    # 记录总 rollout 时间
    log_dict["rollout_time"] = rollout_time
    # 存在非零非生成时间时输出其统计
    if max(non_generation_time) > 0:
        # 合并非生成时间统计
        log_dict |= dict_add_prefix(compute_statistics(non_generation_time), "non_generation_time/")

    # 内部函数：按指定响应长度口径计算 token 吞吐
    def token_perf(response_lengths, non_generation_time, key=""):
        # 最长响应的 token 数
        max_response_length = max(response_lengths)
        # 配置了 rollout GPU 数时计算每 GPU 每秒 token 数
        if args.rollout_num_gpus:
            # 总响应 token ÷ 时间 ÷ GPU 数
            log_dict[f"{key}tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
        # 最长样本的每秒 token 数（决定整体时延的"最慢者"）
        log_dict[f"longest_{key}sample_tokens_per_sec"] = max_response_length / rollout_time

        # 全部非生成时间为 0 时无需细分
        if max(non_generation_time) == 0:
            # 提前返回
            return

        # 只保留最长样本对应的非生成时间（长度等于最大值者）
        non_generation_time = [
            t for t, length in zip(non_generation_time, response_lengths, strict=True) if length == max_response_length
        ]
        # 计算其平均非生成时间
        mean_non_generation_time = sum(non_generation_time) / len(non_generation_time)

        # 记录最长样本的平均非生成时间
        log_dict[f"longest_{key}sample_non_generation_time"] = mean_non_generation_time
        # 记录剔除非生成时间后的最长样本吞吐
        log_dict[f"longest_{key}sample_tokens_per_sec_without_non_generation"] = max_response_length / (
            rollout_time - mean_non_generation_time
        )

    # 用名义响应长度计算一次
    token_perf([sample.response_length for sample in samples], non_generation_time, key="")
    # 用有效响应长度再算一次（effective_ 前缀）
    token_perf([sample.effective_response_length for sample in samples], non_generation_time, key="effective_")

    # 返回性能指标
    return log_dict


# 计算零方差组指标：GRPO 类算法中同一提示的奖励全相同的组
def _compute_zero_std_metrics(args, all_samples: list[Sample]):
    # only compute in GRPO-like algorithms where one prompt has multiple responses
    # （原英文注释：仅在 GRPO 类算法中计算）
    # PPO 无组结构，跳过
    if args.advantage_estimator == "ppo":
        # 返回空
        return {}

    # 内部函数：判断一组样本奖励是否零方差
    def _is_zero_std(samples: list[Sample]):
        # 取组内全部奖励值
        rewards = [sample.get_reward_value(args) for sample in samples]
        # 空组或所有奖励与第一个相同视为零方差
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)

    # 按组索引把样本分组
    all_sample_groups = group_by(all_samples, lambda s: s.group_index)
    # 过滤出零方差的组
    interesting_sample_groups = [g for g in all_sample_groups.values() if _is_zero_std(g)]

    # 取每组第一个奖励值（四舍五入到一位小数）作为类别标签
    interesting_rewards = [str(round(g[0].get_reward_value(args), 1)) for g in interesting_sample_groups]

    # 按奖励值统计各组数量，生成 zero_std/count_{reward} 指标
    return {f"zero_std/count_{reward}": len(items) for reward, items in group_by(interesting_rewards).items()}


# 计算投机解码（speculative decoding）相关指标：接受率与平均接受长度
def _compute_spec_metrics(args, all_samples: list[Sample]):
    # 未启用投机解码时跳过
    if args.sglang_speculative_algorithm is None:
        # 返回空
        return {}
    # 样本总数
    num_samples = len(all_samples)
    # 初始化指标字典
    metrics = {}
    # 平均投机接受率
    metrics["spec_accept_rate"] = sum(sample.spec_info.spec_accept_rate for sample in all_samples) / num_samples
    # 平均投机接受长度
    metrics["spec_accept_length"] = sum(sample.spec_info.spec_accept_length for sample in all_samples) / num_samples
    # 返回指标
    return metrics


# 计算前缀缓存（prefix caching）命中率指标
def _compute_prefix_cache_metrics(args, all_samples: list[Sample]):
    # 样本总数
    num_samples = len(all_samples)
    # 初始化指标字典
    metrics = {}
    # 累计命中缓存的 token 数
    total_cached_tokens = sum(sample.prefix_cache_info.cached_tokens for sample in all_samples)
    # 累计提示词 token 总数
    total_prompt_tokens = sum(sample.prefix_cache_info.total_prompt_tokens for sample in all_samples)

    # 命中率 = 命中 token ÷ 总提示 token（分母为 0 时取 0）
    metrics["prefix_cache_hit_rate"] = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    # 每样本平均命中 token 数
    metrics["avg_cached_tokens_per_sample"] = total_cached_tokens / num_samples
    # 返回指标
    return metrics


# 按奖励类别（如错误类型）统计样本占比
def _compute_reward_cat_metrics(args, all_samples: list[Sample]):
    # 取配置的奖励类别字段名
    reward_cat_key = args.log_reward_category
    # 未配置时跳过
    if reward_cat_key is None:
        # 返回空
        return {}

    # 按样本奖励字典中的该字段分组
    samples_of_reward_cat = group_by(all_samples, lambda s: s.reward[reward_cat_key])

    # 每类生成 error_cat/{类别}: 占比 指标
    return {f"error_cat/{reward_cat}": len(s) / len(all_samples) for reward_cat, s in samples_of_reward_cat.items()}
