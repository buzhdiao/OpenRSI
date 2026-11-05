# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are MIT-licensed
# Copyright (c) 2024 Rishi Hazra, Alkis Sygkounas
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/RishiHazra/Revolve/blob/main/LICENSE

# Portions of this file are Apache 2.0 licensed
# Copyright (c) 2023 Google DeepMind
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/google-deepmind/funsearch/blob/main/LICENSE

# =====================================================================
# 中文注释导读（OpenRSI / OpenMLE-Evo 学习文档）：
# 本文件是 OpenMLE-Evo 长程搜索的"搜索主循环"（论文中的 evolutionary search
# 推理引擎），围绕论文定义的四个原子程序进化算子（四个原子算子）组织：
#   Draft（从零起草）/ Improve（依据执行反馈改进）/ Debug（修复失败程序）/
#   Crossover（交叉重组两个父代程序）。
# 论文术语对照：
#   - execution-grounded：执行接地——所有算子的输入输出都锚定在真实代码执行
#     与可验证评估反馈上，而非纯文本自评；
#   - async_steady_state：Evo-Max 异步搜索（OpenMLE-Evo-Max）的稳态执行模式，
#     多个 worker 并发采样-生成-执行-提交，不再按"代(generation)"串行推进；
#   - Medal Average（奖牌平均）：MLE-Bench 的评价指标（每任务按获奖比例计分），
#     本文件产出的 best node 代码将交由评测脚本计算该指标。
# 数据结构总览：
#   Island        —— 岛屿模型中的一个个岛（种群），维护若干解节点；
#   SolutionsDatabase —— 多岛种群数据库，负责按适应度加权采样父代并选择算子；
#   Evolutionary  —— 求解器主体，实现标准逐代搜索与 Evo-Max 异步稳态搜索两条路径；
#   AsyncWorkItem —— 异步搜索中一次"生成-执行"工作单元的描述（算子+父代+沙箱）。
# =====================================================================

from __future__ import annotations  # 允许类型注解使用 list[Node] 等新式写法而不触发运行期求值

import asyncio  # 异步 IO：Evo-Max 异步稳态搜索的并发调度基础
import json  # checkpoint（journal.jsonl）序列化/反序列化
import random  # 岛屿/父代随机采样、迁移与重置的随机性来源
import sys  # 仅用于类型检查分支，保留标准库导入
import threading  # rich memory 摘要缓存的进程内互斥锁
import time  # 墙钟计时：搜索预算（time_limit_secs）统计
from dataclasses import dataclass  # 声明 AsyncWorkItem 不可变工作单元
from functools import partial  # 将 LLM 与配置"预绑定"到各算子函数
from pathlib import Path  # checkpoint 路径处理
from typing import Any, Dict, List, Optional, Tuple, Union  # 类型注解

import numpy  # 岛屿适应度统计（min/max/mean/argmin）与加权采样

# 搜索工具集：normalized() 把分数列表转成带温度的采样概率分布
import dojo.core.solvers.utils.search_utils as utils
from dojo.core.solvers.base import Solver  # 所有求解器的抽象基类
from dojo.core.solvers.operators.analyze import analyze_op  # 分析算子：解读执行输出、判 bug、提指标
from dojo.core.solvers.operators.core import execute_op_plan_code  # 算子执行封装：带重试地产出 (plan, code)
from dojo.core.solvers.operators.crossover import crossover_op  # 交叉算子：Crossover，重组两个父代
from dojo.core.solvers.operators.draft import draft_op  # 起草算子：Draft，从零生成程序
from dojo.core.solvers.operators.improve import improve_op  # 改进算子：Improve，基于父代+执行反馈改写
from dojo.core.solvers.operators.debug import debug_op  # 调试算子：Debug，修复报错程序
from dojo.core.solvers.operators.rich_memory_summary import rich_memory_summary_op  # 富记忆摘要算子（经验卡片）
from dojo.core.solvers.operators.memory import create_memory_op  # 按配置构造记忆组件（few-shot 示例池）
from dojo.core.solvers.utils import data_preview  # 数据预览：给 LLM 提供训练数据的 head/shape 摘要
from dojo.core.solvers.utils.journal import Journal, Node  # 搜索日志树与解节点（每次生成即一个 Node）
from dojo.core.solvers.utils.metric import MetricValue, WorstMetricValue  # 适应度值封装（含最差值哨兵）
from dojo.core.solvers.utils.response import extract_code  # 从模型回复中提取代码块
from dojo.core.solvers.utils.response import prompt_score_sanitization_enabled  # 是否对提示词中的分数脱敏
from dojo.core.solvers.utils.response import sanitize_execution_output_for_prompt  # 执行输出送入提示词前的清洗
from dojo.core.solvers.utils.search_exporter import export_search_results  # 搜索结束后的结果导出（树/指标）
from dojo.solvers.utils import get_complextiy_level  # 估计节点代码复杂度等级，作为提示词上下文
from dojo.solvers.evo.experience import (  # 经验（experience）机制：父代效用与记忆卡片
    collect_operator_memory_nodes,  # 按算子收集"纵向(祖先)/横向(兄弟)"经验节点
    compute_parent_utilities,  # 基于经验卡片的父代效用计算（用于父代加权采样）
    prompt_memory_enabled,  # 经验记忆开关
    prompt_memory_max_related_cards,  # 每算子最多注入的相关经验卡片数
    prompt_memory_operator_k,  # 每算子注入的祖先/_sibling 经验条数配置
    prompt_memory_sibling_rank_weights,  # 兄弟节点按排名衰减的权重
)
from dojo.utils.logger import CollectiveLogger, LogEvent  # 结构化日志器与事件类别
from dojo.utils.code_parsing import parse_json_output  # 从 LLM 回复中鲁棒解析 JSON
from dojo.core.tasks.constants import (  # 任务执行返回字典的键名常量
    EXECUTION_OUTPUT,  # 执行stdout/stderr 文本
    TASK_DESCRIPTION,  # 任务描述键
    VALID_SOLUTION_FEEDBACK,  # 评估器给出的合法性反馈
    VALIDATION_FITNESS,  # 沙箱确定性算出的验证集适应度（execution-grounded 的关键通道）
    AUX_EVAL_INFO,  # 附加评估信息（状态码、反馈等元数据）
    VALID_SOLUTION,  # 解是否合法
)
from dojo.config_dataclasses.solver.evo import EvolutionarySolverConfig  # 进化求解器的全部超参配置
from dojo.utils.state import EvolutionaryState  # 可恢复的搜索状态（当前代数/步数/已用时间）
from dojo.core.solvers.llm_helpers.generic_llm import GenericLLM  # 统一 LLM 客户端（支持多后端）
from tts_search.airaevo_async_resources import (  # Evo-Max 异步搜索的资源解析工具（OpenRSI 自有模块）
    WorkerSpec,  # 单个异步 worker 的规格（id/GPU/沙箱URL）
    build_worker_specs,  # 依据 worker 数与沙箱URL构造 worker 列表
    fetch_router_idle_worker_count,  # 查询沙箱路由器当前空闲 worker 数（观测用）
    resolve_async_sandbox_urls,  # 解析异步沙箱地址列表（多控制器路由）
    resolve_async_worker_count,  # 解析异步 worker 数量（支持 "auto"=GPU 数）
    resolve_execution_mode,  # 解析执行模式：generation（标准逐代）或 async_steady_state（Evo-Max）
)


@dataclass(frozen=True)
class AsyncWorkItem:
    """Evo-Max 异步稳态搜索中的一次"生成-执行"工作单元（不可变快照）。

    采样阶段在 sample_lock 保护下完成，产出该对象后，worker 可以脱离锁、
    并发地执行 LLM 生成与沙箱执行，避免多 worker 之间的状态竞争。
    """

    attempt_id: int  # 全局递增的尝试编号（论文中异步搜索的 attempt 流水号）
    generation_id: int  # 折算出的虚拟代数（attempt_id // 每代个体数），用于温度/迁移调度
    temperature: float  # 本次采样的温度（随代数线性退火）
    island_id: int  # 该工作单元归属的岛屿（新解将注册到该岛）
    operator: str  # 选中的原子算子：draft / improve / crossover（debug 在执行后按需触发）
    parent_nodes: list[Node]  # 父代节点列表（draft 为空；improve 1 个；crossover 2 个）
    parent_selection_trace: dict[str, Any] | None  # 父代选择的可审计轨迹（效用权重、候选等）
    worker: WorkerSpec  # 承担该工作单元的 worker（绑定 GPU 与沙箱 URL）


class Island:
    """A population of solutions (Nodes) in an island."""

    def __init__(
        self,
        island_id: int,
        initial_solution_nodes: List[Node],
        lower_is_better: bool,
        logger: CollectiveLogger,
    ):
        # 岛屿编号：多岛（island model）用于维持种群多样性，避免单一种群早熟收敛
        self.island_id = island_id
        # 优化方向标志：True 表示适应度越小越好（如误差），False 表示越大越好（如准确率）
        self.lower_is_better = lower_is_better
        self.logger = logger
        # 该岛当前持有的解节点（种群个体）
        self.nodes: List[Node] = initial_solution_nodes

    @property
    def size(self) -> int:
        # 种群规模：当前岛内解节点数量
        return len(self.nodes)

    @property
    def fitness_scores(self) -> List[float]:
        """Returns fitness scores, using +/- inf for None metrics."""
        scores = []
        for node in self.nodes:
            fitness = node.metric.value
            if fitness is None:
                # Use +/- inf for comparison purposes
                # 无适应度（如执行失败）的个体用"最差方向的无穷大"占位，保证可比较
                scores.append(float("inf") if self.lower_is_better else float("-inf"))
            else:
                scores.append(fitness)
        return scores

    @property
    def best_fitness_score(self) -> float:
        """Returns the best fitness score in the island."""
        scores = self.fitness_scores
        if not scores:
            # Return worst possible score if island is empty
            # 空岛返回最差分数，确保在岛屿排序中排到最后
            return float("inf") if self.lower_is_better else float("-inf")
        # 依据优化方向选择 min 或 max 作为"最优"定义
        op = numpy.min if self.lower_is_better else numpy.max
        return op(scores)

    @property
    def average_fitness_score(self):
        """Returns the average fitness score."""
        scores = self.fitness_scores
        if not scores:
            # Avoid numpy warning on empty list, return worst score
            return float("inf") if self.lower_is_better else float("-inf")
        # numpy.mean handles inf correctly (inf + finite = inf, inf + inf = inf, inf - inf = nan)
        # However, if all scores are inf/-inf, mean might be inf/-inf.
        # Filter out inf/-inf before calculating mean for a more representative average?
        # Or rely on the fact that inf/-inf indicates poor solutions.
        # 只对有限分数求平均：inf 个体代表失败解，不参与均值，避免均值被失败解淹没
        finite_scores = [s for s in scores if s != float("inf") and s != float("-inf")]
        if not finite_scores:
            # If only inf/-inf scores, return that score
            # 全是失败解时退化为返回其中之一
            return scores[0] if scores else (float("inf") if self.lower_is_better else float("-inf"))
        return numpy.mean(finite_scores)

    @property
    def fittest_individual(self) -> Optional[Node]:  # Renamed to fittest_node
        """Returns the node with the best fitness score."""
        if not self.nodes:
            return None
        scores = self.fitness_scores
        # argmin/argmax 找到最优个体下标（方向由 lower_is_better 决定）
        op = numpy.argmin if self.lower_is_better else numpy.argmax
        fittest_idx = op(scores)
        return self.nodes[fittest_idx]

    @property
    def solution_nodes(self) -> List[Node]:  # Keep this property
        # 对外暴露种群个体的只读视图（兼容旧接口命名）
        return self.nodes

    def register_node_in_island(
        self,
        solution_node: Node,
    ):
        """
        Add a solution Node to the island population.
        """
        # IDs are stored within the node if needed elsewhere (e.g., node.id)
        # 将新解登记进种群；日志记录其适应度便于追踪岛屿演化
        self.logger.info(
            f"Registering Node {solution_node.id} in Island {self.island_id}. Metric: {solution_node.metric.value}",
            LogEvent.SOLVER,
        )
        self.nodes.append(solution_node)

    def remove_lowest(self):
        """
        Removes the node with the worst fitness score in the island.
        """
        if not self.nodes:
            return  # Nothing to remove

        scores = self.fitness_scores
        # 注意方向反转：lower_is_better 时"最差"是最大分数，故用 argmax
        op = numpy.argmax if self.lower_is_better else numpy.argmin
        lowest_score_index = op(scores)
        weakest_node = self.nodes.pop(lowest_score_index)
        self.logger.info(
            f"Removed weakest node {weakest_node.id} (Score: {scores[lowest_score_index]}) from island {self.island_id}",
            LogEvent.SOLVER,
        )

    def remove_node(self, to_remove_node: Node):  # Renamed from remove_individual
        """
        Remove a specific node from the island.
        """
        # 记录删除前规模，用于判断目标节点是否存在
        initial_size = len(self.nodes)
        # 按节点 id 过滤删除（迁移完成后从"捐献岛"移除该节点即调用此处）
        self.nodes = [node for node in self.nodes if node.id != to_remove_node.id]
        if len(self.nodes) < initial_size:
            self.logger.info(f"Removed node {to_remove_node.id} from island {self.island_id}", LogEvent.SOLVER)
        else:
            # 目标节点不存在时告警（幂等删除的防御式设计）
            self.logger.warning(
                f"Attempted to remove node {to_remove_node.id} from island {self.island_id}, but it was not found.",
                LogEvent.SOLVER,
            )

    def only_keep_best(self):
        """
        Remove all nodes except the single best one.
        Handles ties by keeping only one of the best.
        """
        best_node = self.fittest_individual  # Renamed property
        if best_node:
            self.logger.info(
                f"Island {self.island_id}: Keeping only best node {best_node.id} (Score: {best_node.metric.value})",
                LogEvent.SOLVER,
            )
            # 岛屿重置时保留最优个体作为"火种"，其余全部丢弃
            self.nodes = [best_node]
        else:
            self.logger.info(f"Island {self.island_id}: only_keep_best called on empty island.", LogEvent.SOLVER)
            self.nodes = []

    def migrate_node(
        self,
        founder_node: Node,  # Changed type to Node
    ):
        """
        Migrate a node from a founder island to this island.
        """
        # 接收来自"创始岛"的迁移节点，重建被重置的岛屿种群
        self.logger.info(f"Migrating node {founder_node.id} to Island {self.island_id}", LogEvent.SOLVER)
        self.register_node_in_island(founder_node)


class SolutionsDatabase:
    """
    Maintains and updates a Database of all solutions (Nodes).

    Adapted from Fun Search: https://github.com/google-deepmind/funsearch/blob/main
    and from REvolve: https://arxiv.org/pdf/2406.01309
    """

    def __init__(
        self,
        num_islands: int,
        max_size: int,
        lower_is_better: bool,
        logger: CollectiveLogger,
        experience_config: Optional[dict] = None,
    ):
        # 岛屿数量：FunSearch 式多岛并行探索
        self.num_islands = num_islands
        # 每岛最大种群规模：超出即淘汰最差个体（截断选择）
        self.max_size = max_size
        self.lower_is_better = lower_is_better
        self.logger = logger
        # 经验（experience）配置：控制是否启用经验卡片父代选择等增强
        self.experience_config = dict(experience_config or {})
        # 最近一次父代选择的审计轨迹（供下游写入节点、追踪采样依据）
        self.last_parent_selection: Optional[dict[str, Any]] = None
        self._islands: List[Island] = []
        # 全局适应度范围：用于把任意原始分数归一化到 [0,1]（1 恒为最优）
        self.global_min_fitness = float("inf")
        self.global_max_fitness = float("-inf")

        # Initialize empty islands.
        # 初始化 num_islands 个空岛，共享同一个优化方向与日志器
        self._islands = [
            Island(island_id, [], self.lower_is_better, self.logger) for island_id in range(self.num_islands)
        ]

    @property
    def has_nodes(self) -> bool:
        # 是否已有任何解：决定搜索起点走 Draft 还是 Improve/Crossover
        return any(island.size > 0 for island in self._islands)

    def has_island_with_size(self, size: int) -> bool:
        # 是否存在规模足够的岛（例如 Crossover 需要至少 2 个可用父代）
        return any(island.size >= size for island in self._islands)

    def seed_islands_with_nodes(
        self,
        solution_nodes: List[Node],
        island_ids: List[int],
    ):
        """
        Initialize islands with the first generation of Nodes.
        """
        # 第 0 代：把初始 Draft 解按预定 island_id 播种到各岛（均匀初始化）
        for solution_node, island_id in zip(
            solution_nodes,
            island_ids,
        ):
            if 0 <= island_id < len(self._islands):
                self._islands[island_id].register_node_in_island(
                    solution_node,
                )
            else:
                # 非法岛屿编号只记录并跳过，不中断搜索
                self.logger.info(f"Invalid island_id {island_id} during seeding. Max index: {len(self._islands) - 1}")
            # Seed initial global fitness range if nodes have scores
            # 播种同时更新全局分数范围，为后续归一化建立标定
            fitness_score = solution_node.metric.value
            if fitness_score is not None and numpy.isfinite(fitness_score):
                self._update_global_fitness_range(fitness_score)

    def _update_global_fitness_range(self, score: float):
        """Updates the global min and max fitness scores seen so far."""
        # 只统计有限分数，inf 哨兵不参与范围标定
        if numpy.isfinite(score):
            self.global_min_fitness = min(self.global_min_fitness, score)
            self.global_max_fitness = max(self.global_max_fitness, score)
            self.logger.info(
                f"Updated global fitness range: min={self.global_min_fitness}, max={self.global_max_fitness}",
                LogEvent.SOLVER,
            )

    def get_normalized_score(self, score: Optional[float]) -> float:
        """
        Normalizes a raw fitness score to the range [0, 1], where 1.0 is always best.
        Handles None scores by returning 0.0 (worst).
        """
        if score is None or not numpy.isfinite(score):
            # 失败解（None/inf）归一化为 0（最差），确保采样时几乎不被选中
            return 0.0  # Worst normalized score for None or non-finite scores

        # If global range isn't established or is a single point, return neutral 0.5
        # 尚无有效分数范围（或所有解同分）时返回中性 0.5，退化为均匀采样
        if (
            not numpy.isfinite(self.global_min_fitness)
            or not numpy.isfinite(self.global_max_fitness)
            or self.global_min_fitness == self.global_max_fitness
        ):
            return 0.5

        # Normalize to [0, 1]
        # 归一化输出统一约定：值越大越好（后续 normalized() 无需再关心方向）
        if self.lower_is_better:
            # For lower_is_better, higher values are worse.
            # (global_max - score) / (global_max - global_min)
            # If score is min_fitness, result is 1. If score is max_fitness, result is 0.
            # 分数越小越好：做镜像翻转
            normalized = (self.global_max_fitness - score) / (self.global_max_fitness - self.global_min_fitness)
        else:
            # For higher_is_better, higher values are better.
            # (score - global_min) / (global_max - global_min)
            # If score is min_fitness, result is 0. If score is max_fitness, result is 1.
            # 分数越大越好：标准 min-max 归一化
            normalized = (score - self.global_min_fitness) / (self.global_max_fitness - self.global_min_fitness)

        # Clamp to [0, 1] to handle scores outside the current global range robustly
        # 截断到 [0,1]，防御新解超出历史范围导致的越界概率
        return float(numpy.clip(normalized, 0.0, 1.0))

    def add_nodes_to_islands(
        self,
        solution_nodes: List[Node],
        island_ids: List[int],
        migration_prob: float,
    ):
        """
        Add evaluated Nodes to appropriate islands based on fitness improvement.
        Manages island size and triggers migration/reset.
        """
        # 将本轮已评估（execution-grounded 适应度已知）的解登记进各自岛屿
        for solution_node, island_id in zip(
            solution_nodes,
            island_ids,
        ):
            if not (0 <= island_id < len(self._islands)):
                self.logger.error(f"Invalid island_id {island_id} when adding node {solution_node.id}. Skipping.")
                continue

            fitness_score = solution_node.metric.value  # This can be None or a float
            if fitness_score is not None and numpy.isfinite(fitness_score):  # Update global range
                # 每个新分数都可能扩展全局范围，保持归一化标定最新
                self._update_global_fitness_range(fitness_score)

            current_island = self._islands[island_id]
            island_avg_fitness_score = current_island.average_fitness_score

            # Determine if the new node improves the island
            # 准入判据：只有"不劣于岛屿平均适应度"的解才允许入岛（稳态截断选择）
            if current_island.size == 0:
                improvement_condition = True  # Always add to an empty island
            elif fitness_score is None:
                improvement_condition = False  # A solution with None fitness never improves an island
            else:
                # 方向感知比较：小更好则须 <= 平均，大更好则须 >= 平均
                improvement_condition = (self.lower_is_better and fitness_score <= island_avg_fitness_score) or (
                    not self.lower_is_better and fitness_score >= island_avg_fitness_score
                )

            # check if individual is adding any value to the island
            if improvement_condition:
                current_island.register_node_in_island(
                    solution_node,
                )
                self.logger.info(
                    f"Node {solution_node.id} added to island {island_id}. New avg score: {current_island.average_fitness_score:.4f}",
                    LogEvent.SOLVER,
                )

            # if island size exceeds max size, discard individual with the lowest score
            # 超出岛屿容量则淘汰最差个体，保持种群规模恒定
            if current_island.size > self.max_size:
                self.logger.info(
                    f"Exceeded maximum size ({self.max_size}) on island {island_id}, discarding weakest node(s)",
                    LogEvent.SOLVER,
                )
                while current_island.size > self.max_size:
                    current_island.remove_lowest()

        # Migration / Reset Logic
        # 以 migration_prob 概率触发岛屿重置：打散停滞种群、注入多样性
        if len(self._islands) > 1 and random.random() <= migration_prob:
            self.reset_islands()

    def reset_islands(self):
        """
        Resets the weaker half of islands and seeds them
        with nodes migrated from fitter islands.
        """
        if len(self._islands) < 2:  # Cannot reset if less than 2 islands
            self.logger.warning("Reset islands called with less than 2 islands. Skipping.", LogEvent.SOLVER)
            return

        self.logger.info("============ Resetting Islands ============", LogEvent.SOLVER)

        # Get island scores (best score per island)
        # 以每岛"最优分数"代表岛屿整体水平
        island_scores = []
        for island in self._islands:
            best_score = island.best_fitness_score  # Uses +/- inf for empty/None
            island_scores.append(best_score)

        # Add small noise to break ties during sorting
        # 加微小噪声打破同分，避免排序不稳定
        noisy_scores = numpy.array(island_scores) + numpy.random.randn(len(self._islands)) * 1e-9

        # Sort islands by score (ascending for lower_is_better, descending otherwise)
        # 升序排序后：lower_is_better 时前段是优岛；否则需要反转为降序
        indices_sorted_by_score = numpy.argsort(noisy_scores)
        if not self.lower_is_better:
            indices_sorted_by_score = indices_sorted_by_score[::-1]

        num_islands_to_reset = len(self._islands) // 2
        if num_islands_to_reset == 0:  # Ensure at least one island is kept
            self.logger.info("Not enough islands to perform reset.", LogEvent.SOLVER)
            return

        # 排名靠后的一半岛屿被重置，靠前的一半作为"创始岛"捐献个体
        reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]

        self.logger.info(f"Resetting islands: {reset_islands_ids}", LogEvent.SOLVER)
        self.logger.info(f"Keeping islands: {keep_islands_ids}", LogEvent.SOLVER)

        for reset_island_id in reset_islands_ids:
            reset_island = self._islands[reset_island_id]
            # Keep only the single best node in the island being reset
            # 被重置岛仅保留其最优个体（避免完全清空导致的退化）
            reset_island.only_keep_best()

            # Find a suitable founder island (must have > 1 node to donate)
            # 创始岛必须至少有 2 个节点：迁走 1 个后自身仍非空
            possible_founder_ids = [idx for idx in keep_islands_ids if self._islands[idx].size > 1]
            if not possible_founder_ids:
                self.logger.warning(
                    f"No suitable founder island with >1 node found for resetting island {reset_island_id}. Skipping migration.",
                    LogEvent.SOLVER,
                )
                continue  # Skip to next island to reset

            # 随机挑选一个创始岛
            founder_island_id = numpy.random.choice(possible_founder_ids)
            founder_island = self._islands[founder_island_id]

            # Sample a node from the founder island (weighted by fitness, excluding the absolute best)
            candidates = []
            candidate_scores = []
            best_founder_score = founder_island.best_fitness_score
            for node in founder_island.nodes:
                score = node.metric.value
                # Handle None scores - treat as worst possible for sampling
                numeric_score = (
                    score if score is not None else (float("inf") if self.lower_is_better else float("-inf"))
                )
                # Exclude the absolute best node(s) from being migrated
                # 最优个体必须留守创始岛，只迁移次优个体
                if numeric_score != best_founder_score:
                    candidates.append(node)
                    # Use the numeric score for weighting the sampling
                    candidate_scores.append(numeric_score)

            if not candidates:
                self.logger.warning(
                    f"Founder island {founder_island_id} has no non-best nodes to migrate. Skipping migration for island {reset_island_id}.",
                    LogEvent.SOLVER,
                )
                continue  # Skip to next island to reset

            # Perform weighted sampling (using normalized utility)
            # Note: utils.normalized expects scores where higher is better for probability
            # 适应度加权采样迁移个体：好个体有更大概率被选中去重建新岛
            normalized_candidate_scores = [self.get_normalized_score(s) for s in candidate_scores]
            sampling_weights = utils.normalized(
                normalized_candidate_scores, temp=1.0
            )  # Normalized scores are always higher_is_better
            if sum(sampling_weights) == 0:  # Avoid error if all weights are zero
                # Fallback to uniform sampling if weights are zero
                # 权重全零（浮点下溢）时退化为均匀采样
                founder_node_to_migrate = random.choice(candidates)
                self.logger.warning(
                    "Sampling weights were all zero, falling back to uniform selection for migration.",
                    LogEvent.SOLVER,
                )
            else:
                # 按权重抽取一个迁移个体
                founder_node_to_migrate = random.choices(candidates, weights=sampling_weights, k=1)[0]

            # Migrate the selected node
            self.logger.info(
                f"Migrating node {founder_node_to_migrate.id} from Island {founder_island_id} to Island {reset_island_id}",
                LogEvent.SOLVER,
            )
            reset_island.migrate_node(founder_node_to_migrate)

            # Remove the migrated node from the founder island
            # 迁移是"移动"而非"复制"：从创始岛移除，保持总量守恒
            founder_island.remove_node(founder_node_to_migrate)

    def request_fresh_draft(self, reason: str) -> None:
        """Make the next non-initial generation explore a clean candidate."""
        # 记录"强制重新起草"的原因（如反复 Debug 失败）：下一次采样将直接走 Draft，
        # 避免在一个注定失败的父代上继续浪费算子调用
        self._forced_fresh_draft_reason = str(reason)

    def sample_in_context(
        self,
        num_samples: Dict,
        temperature: float,
        crossover_prob: float,
        fresh_draft_prob: float = 0.0,
    ) -> Tuple[List[Node], int, str]:
        """
        Samples nodes for the next generation, selecting islands and then nodes based on fitness.
        Returns sampled nodes, the island they came from, and the selected operator ('improve' or 'crossover').
        """
        # 本方法是"算子选择 + 父代采样"的核心：决定下一个个体由哪个原子算子生成、
        # 以哪些解为上下文（few-shot）。返回 (父代列表, 岛屿id, 算子名)。
        if not any(island.size > 0 for island in self._islands):
            self.logger.warning(
                "Sample in context called, but all islands are empty. Cannot sample. Back to drafting", LogEvent.SOLVER
            )
            self.last_parent_selection = None
            # 空种群时只能从零起草（Draft 是唯一的合法起点）
            return [], 0, "draft"

        forced_reason = getattr(self, "_forced_fresh_draft_reason", None)
        fresh_probability = min(1.0, max(0.0, float(fresh_draft_prob)))
        # 强制 Draft（重复失败后）或以 fresh_draft_prob 概率随机 Draft：
        # 引入"从零重开"的探索通道，防止搜索被早期解锁死
        if forced_reason or (fresh_probability > 0 and random.random() < fresh_probability):
            reason = str(forced_reason or "fresh_draft_probability")
            self._forced_fresh_draft_reason = None  # 用掉一次强制标记即清空
            island_id = random.randrange(len(self._islands))
            self.last_parent_selection = {
                "enabled": False,
                "operator": "draft",
                "island_id": island_id,
                "reason": reason,
                "fresh_draft_prob": fresh_probability,
            }
            self.logger.info(
                f"Starting a fresh draft on island {island_id}: {reason}",
                LogEvent.SOLVER,
            )
            return [], island_id, "draft"

        # Calculate average fitness scores for island sampling
        # Use a default worst score for empty islands to give them zero probability
        # 第一步：给每个岛按"平均适应度"打分，空岛记为最差（天然零概率）
        island_avg_scores = []
        for island in self._islands:
            if island.size > 0:
                island_avg_scores.append(island.average_fitness_score)
            else:
                island_avg_scores.append(float("inf") if self.lower_is_better else float("-inf"))

        # Normalize scores for sampling probabilities (higher score = higher probability)
        # Since get_normalized_score handles lower_is_better and returns higher_is_better output,
        # we pass lower_is_better=False to utils.normalized
        # 岛屿级加权：好岛更可能被选中继续深入搜索（开发），差岛概率低但仍可能被探索
        normalized_island_avg_scores = [self.get_normalized_score(s) for s in island_avg_scores]
        island_sampling_weights = utils.normalized(
            normalized_island_avg_scores,
            temp=temperature,  # Apply temperature to avg scores for island selection
            # Normalized scores are always higher_is_better
        )

        # Ensure weights sum to 1 (handle potential all-zero case)
        # 权重全零时退化为均匀选岛
        if sum(island_sampling_weights) == 0:
            self.logger.warning("Island sampling weights are all zero. Falling back to uniform island selection.")
            island_sampling_weights = [1.0 / len(self._islands)] * len(self._islands)

        self.logger.debug(f"Island sampling weights: {island_sampling_weights}", LogEvent.SOLVER)

        # Determine operator and required number of samples
        # 算子选择：以 crossover_prob 概率走 Crossover（两个父代），否则走 Improve（一个父代）。
        # 这是四个原子算子中"进化类"两个算子的调度点（Draft 见上方分支，Debug 在执行失败后触发）
        operator = "improve" if random.random() >= crossover_prob else "crossover"
        num_in_context_samples = num_samples.get(operator, 1)  # Default to 1 if operator key missing
        operator_fallback_reason = None
        # 只有规模足够（父代数量达标）的岛才有资格被采样
        eligible_island_ids = [
            island_id
            for island_id, island in enumerate(self._islands)
            if island.size >= num_in_context_samples
        ]
        if not eligible_island_ids and operator == "crossover":
            # Crossover 父代不足时降级为 Improve，避免无谓的 Draft
            improve_samples = num_samples.get("improve", 1)
            improve_island_ids = [
                island_id
                for island_id, island in enumerate(self._islands)
                if island.size >= improve_samples
            ]
            if improve_island_ids:
                self.logger.info(
                    "No island has enough parents for crossover; falling back to improve.",
                    LogEvent.SOLVER,
                )
                operator = "improve"
                num_in_context_samples = improve_samples
                eligible_island_ids = improve_island_ids
                operator_fallback_reason = "crossover_parent_shortage"
        if not eligible_island_ids:
            # 所有岛都无法提供父代上下文：回退到全新 Draft
            island_id = random.randrange(len(self._islands))
            self.last_parent_selection = {
                "enabled": False,
                "operator": "draft",
                "island_id": island_id,
                "reason": "no_eligible_parent_context",
            }
            self.logger.info(
                "No island has enough eligible parents; starting a fresh draft.",
                LogEvent.SOLVER,
            )
            return [], island_id, "draft"
        self.last_parent_selection = None

        # Loop until a suitable island and samples are found
        sampled_island_id = -1
        sampled_island = None
        in_context_nodes = []
        attempts = 0
        max_attempts = len(self._islands) * 2  # Heuristic limit to prevent infinite loops

        while attempts < max_attempts:
            attempts += 1
            # STEP 1: Sample an island based on average fitness
            # 在合格岛屿集合内按权重选岛（并发场景下状态可能变化，故带重试）
            eligible_weights = [island_sampling_weights[island_id] for island_id in eligible_island_ids]
            if sum(eligible_weights) == 0:
                eligible_weights = [1.0 / len(eligible_island_ids)] * len(eligible_island_ids)
            sampled_island_id = random.choices(
                eligible_island_ids,
                weights=eligible_weights,
                k=1,
            )[0]
            sampled_island = self._islands[sampled_island_id]

            # Check if the sampled island has enough nodes for the operator
            # 复查岛规模（异步模式下其他 worker 可能刚消费了该岛个体）
            if sampled_island.size < num_in_context_samples:
                self.logger.debug(
                    f"Sampled island {sampled_island_id} size {sampled_island.size} < required {num_in_context_samples}. Resampling island.",
                    LogEvent.SOLVER,
                )
                continue  # Resample island

            # STEP 2: Sample nodes within the island (weighted by individual fitness)
            # 记录父代选择的完整审计轨迹：算子、温度、候选与最终选中者
            parent_selection_trace: dict[str, Any] = {
                "enabled": False,
                "operator": operator,
                "island_id": sampled_island_id,
                "temperature": float(temperature),
                "num_in_context_samples": int(num_in_context_samples),
                "candidates": [],
                "selected_node_ids": [],
            }
            if operator_fallback_reason:
                parent_selection_trace["operator_fallback_reason"] = operator_fallback_reason
            parent_selection_cfg = dict(self.experience_config.get("parent_selection") or {})
            component_normalization_cfg = dict(parent_selection_cfg.get("component_normalization") or {})
            experience_parent_selection_enabled = bool(self.experience_config.get("enabled", False)) and bool(
                parent_selection_cfg.get("enabled", True)
            )
            if experience_parent_selection_enabled:
                # 经验增强的父代选择：除适应度外，还利用经验卡片（方法摘要、失败教训）
                # 计算父代综合效用——这是 OpenRSI 对基础 AIRA-Evo 的扩展点之一
                previous_cards = [
                    card
                    for island in self._islands
                    for node in island.nodes
                    if isinstance((card := getattr(node, "experience_card", None)), dict)
                ]
                utility_items = compute_parent_utilities(
                    sampled_island.nodes,
                    lower_is_better=self.lower_is_better,
                    previous_cards=previous_cards,
                    weights=dict(parent_selection_cfg.get("weights") or {}),
                    component_normalization=component_normalization_cfg,
                    temperature=temperature,
                )
                node_sampling_weights = [item["probability"] for item in utility_items]
                parent_selection_trace.update(
                    {
                        "enabled": True,
                        "weights": dict(parent_selection_cfg.get("weights") or {}),
                        "component_normalization": component_normalization_cfg,
                        "candidates": utility_items,
                    }
                )
                self.logger.debug(
                    f"Experience parent sampling weights on island {sampled_island_id}: {node_sampling_weights}",
                    LogEvent.SOLVER,
                )
            else:
                # 基线父代选择：仅按适应度归一化分数加权（温度控制探索强度）
                island_node_scores = sampled_island.fitness_scores
                normalized_node_scores = [self.get_normalized_score(s) for s in island_node_scores]
                node_sampling_weights = utils.normalized(
                    normalized_node_scores, temperature
                )  # Normalized scores are always higher_is_better

            if sum(node_sampling_weights) == 0:
                self.logger.warning(
                    f"Node sampling weights on island {sampled_island_id} are zero. Falling back to uniform node selection."
                )
                # Use uniform sampling if weights are zero
                # 权重全零时在岛内均匀不放回抽取父代
                indices = numpy.random.choice(range(sampled_island.size), size=num_in_context_samples, replace=False)
            else:
                try:
                    # 按权重不放回抽取 num_in_context_samples 个父代
                    indices = numpy.random.choice(
                        range(sampled_island.size),
                        p=node_sampling_weights,
                        size=num_in_context_samples,
                        replace=False,
                    )
                except ValueError as e:
                    # This might happen if weights don't sum to 1 due to float precision
                    # 浮点误差导致概率和不为一时，退化为均匀采样
                    self.logger.error(
                        f"Error sampling nodes on island {sampled_island_id}: {e}. Weights: {node_sampling_weights}. Falling back to uniform."
                    )
                    indices = numpy.random.choice(
                        range(sampled_island.size), size=num_in_context_samples, replace=False
                    )

            # 产出父代列表并写入审计轨迹，供后续节点记录与复盘
            in_context_nodes = [sampled_island.nodes[i] for i in indices]
            parent_selection_trace["selected_node_ids"] = [node.id for node in in_context_nodes]
            self.last_parent_selection = parent_selection_trace
            self.logger.info(
                f"{operator.capitalize()} | Sampled island: {sampled_island_id}. Nodes: {[n.id for n in in_context_nodes]}",
                LogEvent.SOLVER,
            )
            return in_context_nodes, sampled_island_id, operator

        # This should be unreachable after sampling only eligible islands, but a
        # fresh draft preserves the search if a concurrent mutation changes state.
        # 理论上不可达（只采样合格岛）；但异步并发可能改变状态，保底走 Draft 维持搜索不中断
        island_id = random.randrange(len(self._islands))
        self.last_parent_selection = {
            "enabled": False,
            "operator": "draft",
            "island_id": island_id,
            "reason": "parent_sampling_exhausted",
        }
        self.logger.warning(
            f"Parent sampling exhausted after {max_attempts} attempts; starting a fresh draft.",
            LogEvent.SOLVER,
        )
        return [], island_id, "draft"

    def sample_in_context_with_trace(
        self,
        num_samples: Dict,
        temperature: float,
        crossover_prob: float,
        fresh_draft_prob: float = 0.0,
    ) -> tuple[list[Node], int, str, dict[str, Any] | None]:
        # 带 trace 的包装：异步搜索用它把父代选择轨迹随工作单元一起传递
        nodes, island_id, operator = self.sample_in_context(
            num_samples,
            temperature,
            crossover_prob,
            fresh_draft_prob,
        )
        return nodes, island_id, operator, self.last_parent_selection


class Evolutionary(Solver):
    """进化求解器主体：标准逐代搜索（generation）与 Evo-Max 异步稳态搜索（async_steady_state）。

    对应论文中的长程搜索（long-horizon search）阶段：把四个原子算子组合成
    多代进化，并把 SFT/RL 学到的策略（Frontis-MA1）作为算子的底层模型。
    """

    def __init__(self, cfg: EvolutionarySolverConfig, task_info):
        super().__init__(cfg, task_info=task_info)
        # journal：搜索树日志，保存全部 Node（plan/code/执行反馈/指标），支持断点续跑
        self.journal = Journal()
        self.data_preview: str | None = None

        # 任务描述：Prompt 的核心字段（execution-grounded 提示词的"任务锚"）
        self.task_desc = task_info[TASK_DESCRIPTION]
        # NatureBench 场景可提供更原始的任务描述（含论文级上下文）
        raw_task_description = str(
            task_info.get("naturebench_raw_task_description") or self.task_desc
        )
        # 任务族执行建议（如"该类任务应使用 GPU 训练"）：附加到任务描述尾部
        task_family_guidance = str(task_info.get("task_family_guidance") or "").strip()
        if task_family_guidance:
            raw_task_description = (
                f"{raw_task_description}\n\nTask-Family Execution Guidance:\n"
                f"- {task_family_guidance}"
            )
        # 汇总所有算子共享的提示词上下文（任务/数据/系统提示等）
        self.prompt_context = {
            "task_description": raw_task_description,
            "data_description": str(task_info.get("data_description") or ""),
            "visible_data_analysis": str(
                task_info.get("visible_data_analysis") or ""
            ),
            "public_system_prompt": str(task_info.get("public_system_prompt") or ""),
            "public_user_prompt": str(task_info.get("public_user_prompt") or ""),
        }
        # 优化方向必须由任务显式给出（决定全部比较与归一化逻辑）
        self.lower_is_better = task_info.get("lower_is_better", None)
        assert self.lower_is_better is not None  # Ensure lower_is_better is set
        self.prompt_context["lower_is_better"] = bool(self.lower_is_better)

        # 初始化四个原子算子与辅助算子（各自独立的 LLM 配置）
        self.setup_operators()

        # 可恢复搜索状态：当前代数/步数/累计运行时间
        self.state = EvolutionaryState()
        # 富记忆摘要的逐节点互斥锁表（异步模式下多协程可能同时为一个节点生成摘要）
        self._rich_summary_locks: dict[str, threading.Lock] = {}
        self._rich_summary_locks_guard = threading.Lock()

    def save_checkpoint(self):
        super().save_checkpoint()

        # Write the journal to a jsonl file
        # 将整棵搜索树逐行写入 journal.jsonl（每行一个 Node 的 JSON 快照）
        journal_sd = self.journal.node_list()
        journal_path = Path(self.cfg.checkpoint_path) / "journal.jsonl"
        with open(journal_path, "w") as f:
            for node in journal_sd:
                f.write(json.dumps(node) + "\n")
        self.logger.info(f"Checkpoint saved to {journal_path}")

    def load_checkpoint(self):
        super().load_checkpoint()

        journal_path = Path(self.cfg.checkpoint_path) / "journal.jsonl"
        if not journal_path.exists():
            # 有状态无日志属于不一致：直接断言失败，避免静默丢历史
            assert self.state.current_step == 0, (
                f"No journal found at {journal_path}, but the state was found. This is unexpected."
            )
            return

        self.logger.info(f"Found journal at {journal_path}. Loading...")
        # Load the journal
        # 从 jsonl 恢复全部节点，重建搜索树（恢复后可从任意代继续搜索）
        with open(journal_path, "r") as f:
            journal_export = [json.loads(line) for line in f]
        self.journal = Journal.from_export_data({"nodes": journal_export})

    def setup_operators(self):
        """Setup operator LLMs."""

        # First we set up the LLMs
        # 四个原子算子各配一个独立 LLM 客户端（可分别指向不同模型/端点/采样参数）：
        # 训练对齐（SFT+RL）后，这四个端点通常由同一个 Frontis-MA1 提供服务
        draft_llm = GenericLLM(self.cfg.operators["draft"])
        improve_llm = GenericLLM(self.cfg.operators["improve"])
        debug_llm = GenericLLM(self.cfg.operators["debug"])
        crossover_llm = GenericLLM(self.cfg.operators["crossover"])
        # analyze：辅助算子，解读执行输出、判定 bug、抽取指标（非四个原子算子之一）
        analyze_llm = GenericLLM(self.cfg.operators["analyze"])
        # 富记忆摘要算子：仅在启用经验记忆时创建
        rich_memory_summary_llm = (
            GenericLLM(self.cfg.operators["rich_memory_summary"])
            if prompt_memory_enabled(self.cfg)
            and "rich_memory_summary" in self.cfg.operators
            else None
        )

        # Create the memory for operators
        # 两个记忆组件：常规记忆（Draft/Improve/Crossover 用）与调试记忆（Debug 用）
        self.memory_op = create_memory_op(self.cfg.memory)
        self.debug_memory_op = create_memory_op(self.cfg.debug_memory)

        # Then we create the operators
        # 用 partial 把 LLM、配置、记忆与提示词上下文预绑定进各算子函数，
        # 之后调用 draft_fn(...) 只需再传任务相关参数
        self.draft_fn = partial(
            draft_op, draft_llm, self.cfg, self.memory_op, self.prompt_context
        )
        self.improve_fn = partial(
            improve_op, improve_llm, self.cfg, self.memory_op, self.prompt_context
        )
        # Debug 使用独立的 debug_memory_op：修复程序需要错误日志类记忆
        self.debug_fn = partial(
            debug_op, debug_llm, self.cfg, self.debug_memory_op, self.prompt_context
        )
        self.analyze_fn = partial(analyze_op, analyze_llm, self.cfg)
        self.rich_memory_summary_fn = (
            partial(rich_memory_summary_op, rich_memory_summary_llm, self.cfg)
            if rich_memory_summary_llm is not None
            else None
        )
        self.crossover_fn = partial(
            crossover_op, crossover_llm, self.cfg, self.memory_op, self.prompt_context
        )

    def _rich_memory_cache_dir(self) -> Path:
        # 富记忆摘要缓存目录：位于 checkpoint 目录旁边的 rich_summaries/
        checkpoint_path = Path(str(self.cfg.checkpoint_path))
        return checkpoint_path.parent / "rich_summaries"

    def _rich_memory_cache_path(self, node: Node) -> Path:
        # 每个节点的摘要缓存为 <node_id>.json，重启后无需重新调用 LLM
        return self._rich_memory_cache_dir() / f"{node.id}.json"

    def _rich_summary_lock_for_node(self, node: Node) -> threading.Lock:
        # 懒创建每节点锁：保证同一节点的摘要只被生成一次（线程安全）
        node_id = str(getattr(node, "id", "") or id(node))
        with self._rich_summary_locks_guard:
            lock = self._rich_summary_locks.get(node_id)
            if lock is None:
                lock = threading.Lock()
                self._rich_summary_locks[node_id] = lock
            return lock

    def _load_cached_rich_summary(self, node: Node) -> dict[str, str] | None:
        # 从磁盘读取缓存的富记忆摘要；损坏或不完整的缓存一律视为不存在
        cache_path = self._rich_memory_cache_path(node)
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        summary = payload.get("rich_summary") if isinstance(payload, dict) else None
        # 两个字段齐全才算有效摘要（方法概述 + 与父代的对比经验）
        if isinstance(summary, dict) and summary.get("method_overview") and summary.get("parent_comparison_experience"):
            return {
                "method_overview": str(summary["method_overview"]),
                "parent_comparison_experience": str(summary["parent_comparison_experience"]),
            }
        return None

    def _store_rich_summary(self, node: Node, summary: dict[str, str]) -> None:
        # 把摘要同时写到节点对象、经验卡片和磁盘缓存三处
        node.rich_summary = summary
        card = getattr(node, "experience_card", None)
        if isinstance(card, dict):
            card["rich_summary"] = summary
        cache_path = self._rich_memory_cache_path(node)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "node_id": node.id,
                        "parent_node_ids": [parent.id for parent in list(node.parents or [])],
                        "rich_summary": summary,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            # 缓存写失败不影响搜索主流程，只告警
            self.logger.warning(f"Failed to write rich memory cache for node {node.id}: {exc}")

    def _normalize_rich_summary(self, response: Any) -> dict[str, str] | None:
        # 解析并校验 LLM 返回的摘要 JSON；缺少任一必需字段则视为无效
        parsed = parse_json_output(response)
        if not isinstance(parsed, dict):
            return None
        method_overview = str(parsed.get("method_overview") or "").strip()
        parent_experience = str(parsed.get("parent_comparison_experience") or "").strip()
        if not method_overview or not parent_experience:
            return None
        return {
            "method_overview": method_overview,
            "parent_comparison_experience": parent_experience,
        }

    def _ensure_node_rich_summary(self, node: Node) -> None:
        if node is None:
            return
        # 加锁确保并发下同一节点摘要的幂等生成
        with self._rich_summary_lock_for_node(node):
            self._ensure_node_rich_summary_unlocked(node)

    def _ensure_node_rich_summary_unlocked(self, node: Node) -> None:
        # 摘要获取的三级回退：内存 -> 经验卡片 -> 磁盘缓存 -> 现场生成
        existing = getattr(node, "rich_summary", None)
        if isinstance(existing, dict) and existing.get("method_overview") and existing.get("parent_comparison_experience"):
            card = getattr(node, "experience_card", None)
            if isinstance(card, dict):
                card["rich_summary"] = existing
            return

        card = getattr(node, "experience_card", None)
        if isinstance(card, dict):
            card_summary = card.get("rich_summary")
            if (
                isinstance(card_summary, dict)
                and card_summary.get("method_overview")
                and card_summary.get("parent_comparison_experience")
            ):
                node.rich_summary = card_summary
                return

        cached_summary = self._load_cached_rich_summary(node)
        if cached_summary is not None:
            self._store_rich_summary(node, cached_summary)
            return

        if self.rich_memory_summary_fn is None:
            return

        # 现场调用 LLM 生成摘要：以该节点与其首父代为输入
        parent_node = list(node.parents or [None])[0]
        try:
            result = self.rich_memory_summary_fn(self.task_desc, node, parent_node)
            if isinstance(result, tuple) and len(result) == 2:
                response, metrics = result
            else:
                response, metrics = result, {}
            # 记录"rich_memory_summary"也算一次算子调用（成本审计）
            node.operators_used.append("rich_memory_summary")
            node.operators_metrics.append(metrics)
        except Exception as exc:
            self.logger.warning(f"Rich memory summary failed for node {node.id}: {exc}")
            return

        summary = self._normalize_rich_summary(response)
        if summary is None:
            self.logger.warning(f"Rich memory summary returned invalid JSON for node {node.id}")
            return
        self._store_rich_summary(node, summary)

    def _prepare_operator_rich_memory(
        self,
        operator: str,
        parent_nodes: list[Node],
        *,
        current_node: Node | None = None,
    ) -> None:
        # 在调用 Improve/Debug/Crossover 前准备"富经验记忆"：
        # 收集与该算子相关的祖先（纵向）/兄弟（横向）节点并确保其经验摘要就绪
        if not prompt_memory_enabled(self.cfg):
            return
        sections = collect_operator_memory_nodes(
            operator,
            parent_nodes,
            journal=self.journal,
            lower_is_better=bool(self.lower_is_better),
            current_node=current_node,
            # Improve 是主力算子，默认注入更多经验条目（3 条祖先/3 条兄弟）
            ancestor_k=prompt_memory_operator_k(
                self.cfg,
                operator,
                "ancestor_k",
                3 if str(operator).lower() == "improve" else 2,
            ),
            sibling_k=prompt_memory_operator_k(
                self.cfg,
                operator,
                "sibling_k",
                3 if str(operator).lower() == "improve" else 2,
            ),
            sibling_rank_weights=prompt_memory_sibling_rank_weights(self.cfg),
            max_related_cards=prompt_memory_max_related_cards(self.cfg, operator),
        )
        memory_nodes = []
        # 汇总各记忆分区：主父代、纵向祖先、横向兄弟、双父代各自的纵横分区、调试相关
        for key in [
            "primary",
            "vertical",
            "horizontal",
            "parent_1_vertical",
            "parent_1_horizontal",
            "parent_2_vertical",
            "parent_2_horizontal",
            "debug_related",
        ]:
            memory_nodes.extend(sections.get(key) or [])
        seen_node_ids = set()
        deduped_memory_nodes = []
        # 同一节点可能在多个分区重复出现，去重后统一补齐摘要
        for memory_node in memory_nodes:
            node_id = str(getattr(memory_node, "id", "") or "")
            if not node_id or node_id in seen_node_ids:
                continue
            seen_node_ids.add(node_id)
            deduped_memory_nodes.append(memory_node)
        for memory_node in deduped_memory_nodes:
            self._ensure_node_rich_summary(memory_node)

    def _draft(self) -> Node:
        """
        Generate a new solution from scratch using the draft LLM operator.

        Uses the draft operator to create a new code solution based on the task description.
        The resulting code is packaged into a new Node object with relevant metadata.

        Returns:
            Node: A new node containing the drafted solution
        """
        # 【原子算子 1/4：Draft】从零起草：只依赖任务描述与数据预览，无父代代码。
        # execute_op_plan_code 负责带重试地调用 LLM，产出 (计划文本, 代码, 调用指标)。
        # 剩余时间预算被传入，让模型自行控制方案的规模与训练时长。
        plan, code, metrics = execute_op_plan_code(
            self.draft_fn,
            self.task_desc,
            self.journal,
            self.state.current_step,
            self.cfg.time_limit_secs - self.state.running_time,
            self.data_preview,
            get_complextiy_level(self.root_node) if self.cfg.use_complexity else None,
            self.root_node,
            max_operator_tries=self.cfg.max_llm_call_retries,
        )
        # 包装为节点：父代设为虚拟根节点，算子标记为 "draft"
        node = Node(
            plan=plan, code=code, parents=[self.root_node], operators_used=["draft"], operators_metrics=[metrics]
        )
        self.logger.info(f"Draft Node Created - Metrics: {metrics}")
        self.logger.info(f"Draft Code: {code}")
        return node

    def _improve(self, parent_node: Node) -> Node:
        """
        Improve an existing solution using the improve LLM operator.

        Takes a parent node with a working solution and attempts to enhance it
        using the improve operator.

        Args:
            parent_node: The node containing the solution to improve

        Returns:
            Node: A new node containing the improved solution
        """
        # 【原子算子 2/4：Improve】改进：以一个可运行的父代 + 其执行反馈为基础改写。
        # 先注入富经验记忆（祖先/兄弟节点的方法摘要），再调用算子。
        self._prepare_operator_rich_memory("improve", [parent_node])
        plan, code, metrics = execute_op_plan_code(
            self.improve_fn,
            self.task_desc,
            self.journal,
            parent_node,
            self.state.current_step,
            self.cfg.time_limit_secs - self.state.running_time,
            get_complextiy_level(parent_node) if self.cfg.use_complexity else None,
            self.data_preview,
            max_operator_tries=self.cfg.max_llm_call_retries,
        )
        node = Node(
            plan=plan, code=code, parents=[parent_node], operators_used=["improve"], operators_metrics=[metrics]
        )
        self.logger.info(f"Improve Node Created - Metrics: {metrics}")
        self.logger.info(f"Improve Code: {code}")
        return node

    def _debug(self, parent_node: Node) -> Node:
        """
        Debug a buggy solution using the debug LLM operator.

        Takes a parent node with a buggy solution and attempts to fix it
        using the debug operator, with access to the execution output/error.

        Args:
            parent_node: The node containing the buggy solution to debug

        Returns:
            Node: A new node containing the debugged solution
        """
        # 【原子算子 3/4：Debug】调试：针对执行失败（is_buggy）的节点，
        # 以其报错输出（execution-grounded 错误信号）为依据修复代码。
        self._prepare_operator_rich_memory("debug", [parent_node], current_node=parent_node)
        plan, code, metrics = execute_op_plan_code(
            self.debug_fn,
            self.task_desc,
            self.journal,
            parent_node,
            self.state.current_step,
            self.cfg.time_limit_secs - self.state.running_time,
            self.data_preview,
            max_operator_tries=self.cfg.max_llm_call_retries,
        )
        node = Node(plan=plan, code=code, parents=[parent_node], operators_used=["debug"], operators_metrics=[metrics])
        self.logger.info(f"Debug Node Created - Metrics: {metrics}")
        self.logger.info(f"Debug Code: {code}")
        return node

    def _analyze(self, node: Node) -> Union[str, dict]:
        """
        Analyze a node's execution results using the analyze LLM operator.

        Processes the task description, code, and execution output to determine
        if the solution is buggy and to extract metrics when available.

        Args:
            node: The node to analyze

        Returns:
            Union[str, dict]: Analysis results, either as a string or dictionary
        """
        # 辅助算子（非四原子算子）：让 LLM 阅读执行输出，判断是否 bug 并抽取指标。
        # 仅在沙箱无法给出确定性反馈时作为回退（见 parse_eval_result）。
        analysis, metrics = self.analyze_fn(self.task_desc, node)
        node.operators_used.append("analysis")
        node.operators_metrics.append(metrics)
        self.logger.info(f"Node Analysis Performed - Metrics: {metrics}")
        return analysis

    def _crossover(self, parent_node1: Node, parent_node2: Node) -> Node:
        # 【原子算子 4/4：Crossover】交叉：同时给定两个父代（及其经验记忆），
        # 让模型重组两者的方法片段，产生可能超越任一父代的后代。
        self._prepare_operator_rich_memory("crossover", [parent_node1, parent_node2])
        plan, code, metrics = execute_op_plan_code(
            self.crossover_fn,
            self.task_desc,
            self.journal,
            parent_node1,
            parent_node2,
            self.state.current_step,
            self.cfg.time_limit_secs - self.state.running_time,
            self.data_preview,
            max_operator_tries=self.cfg.max_llm_call_retries,
        )
        node = Node(
            plan=plan,
            code=code,
            parents=[parent_node1, parent_node2],
            operators_used=["crossover"],
            operators_metrics=[metrics],
        )
        self.logger.info(f"Crossover Node Created - Metrics: {metrics}")
        self.logger.info(f"Crossover Code: {code}")
        return node

    def update_data_preview(self, state):
        """
        Generate a data preview to provide context for the LLM operators.

        Creates a small preview of the data (head, shapes, etc.) that can be used
        to help the LLM understand the data structure when generating solutions.

        Args:
            state: The current solver state containing the interpreter
        """
        assert "solver_interpreter" in state, (
            "For generating data previews, the solver needs access to an interpreter."
        )

        # 配置关闭时置空预览，所有算子拿不到数据摘要
        if not self.cfg.data_preview:
            self.data_preview = ""
            return

        self.logger.debug("Generating data preview")
        if state["solver_interpreter"].local:
            # 本地解释器：直接在工作目录上生成 head/shape 等数据预览
            self.data_preview = data_preview.generate(state["solver_interpreter"].working_dir)
        else:
            # 远程沙箱：把预览生成脚本发过去执行，取回 stdout 作为预览
            import inspect

            path = Path(inspect.getsourcefile(data_preview))
            script = path.read_text()
            code = f"{script}\nprint(generate(Path('.').resolve()))"
            exec_result = state["solver_interpreter"].run(code, include_exec_time=False)
            self.data_preview = "\n".join(exec_result.term_out)
        self.logger.debug("Data preview generated")

    def linear_decay(self, iteration: int):
        """defines a temperature schedule for sampling of islands and individuals"""
        # 温度线性退火：早期高温（偏探索、岛屿与父代采样更均匀），后期低温（偏开发、贪心选优）
        initial_temp = self.cfg.initial_temp
        final_temp = self.cfg.final_temp
        num_generations = self.cfg.num_generations
        temperature = (
            initial_temp
            - (initial_temp - final_temp) * max(0, iteration) / num_generations
        )
        # 把结果夹在 [min(initial,final), max(initial,final)] 内防止越界
        return min(
            max(temperature, min(initial_temp, final_temp)),
            max(initial_temp, final_temp),
        )

    def _experience_enabled(self) -> bool:
        # 读取经验机制开关（兼容 dict 与对象两种配置形态）
        experience_cfg = getattr(self.cfg, "experience", {}) or {}
        if hasattr(experience_cfg, "get"):
            return bool(experience_cfg.get("enabled", False))
        return bool(getattr(experience_cfg, "enabled", False))

    def _append_evaluated_node(self, node: Node) -> None:
        """Append an evaluated node and immediately mirror/log it if not already present."""
        # 幂等追加：异步并发下同一节点可能被重复提交，按 id 去重
        node_id = str(getattr(node, "id", "") or "")
        if any(str(getattr(existing, "id", "") or "") == node_id for existing in self.journal.nodes):
            return
        self.journal.append(node)
        self.log_journal()
        # 每个已评估节点计入一步（step_limit 与预算统计的基本单位）
        self.state.current_step += 1

    @staticmethod
    def _task_budget_exempt_seconds(task: Any) -> float:
        # 读取任务报告的"豁免等待秒数"（如排队等 GPU 的时间）：
        # 这段时间不计入搜索时间预算，避免排队惩罚有效搜索时长
        getter = getattr(task, "get_budget_exempt_wait_seconds", None)
        if not callable(getter):
            return 0.0
        try:
            value = float(getter())
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, value)

    @staticmethod
    def _repeated_debug_failure_reason(debug_path: List[Node]) -> str | None:
        """Identify repeated failures where another repair is unlikely to add information."""
        # 检测"重复失败"：连续两次 Debug 以同类错误收场时，再修下去信息增量趋近于零
        if len(debug_path) < 2:
            return None

        def failure_signature(node: Node) -> str | None:
            # 提取失败签名：把失败归入少数可辨识类别（预检/超时/缺依赖/内存不足）
            metric = getattr(node, "metric", None)
            info = dict(getattr(metric, "info", None) or {})
            preflight_category = str(info.get("preflight_category") or "").strip()
            if preflight_category:
                return f"preflight:{preflight_category}"
            status = str(info.get("status") or "").strip().lower()
            status_code = info.get("status_code")
            if status == "timeout" or status_code == 504:
                return "timeout"
            feedback = str(
                info.get("feedback")
                or info.get("validity_feedback")
                or ""
            ).lower()
            if "modulenotfounderror" in feedback or "importerror" in feedback:
                return "import_error"
            if "out of memory" in feedback or "memoryerror" in feedback:
                return "memory_error"
            return None

        previous = failure_signature(debug_path[-2])
        current = failure_signature(debug_path[-1])
        # 两次签名一致 => 判定为重复失败，返回原因供上游触发强制 Draft
        if previous and previous == current:
            return f"repeated_{current}"
        return None

    def debug_cycle(self, state, task, buggy_node: Node):
        # 标准模式下的 Debug 循环：失败节点进入"修复-执行-再判"的小循环，
        # 直到修好、达到最大深度（max_debug_depth）或耗尽调试时间预算。
        if self._experience_enabled():
            # 经验模式下先把失败的原始节点也写入日志（失败经验同样有价值）
            self._append_evaluated_node(buggy_node)
        current_debug_node = buggy_node  # Start with the initial buggy node
        debug_path = [current_debug_node]  # Path includes the initial buggy node
        debug_depth = self.cfg.max_debug_depth
        fixed_metric = None
        # We set the initial debug cycle time to the execution time of the first buggy node
        total_debug_cycle_time = current_debug_node.exec_time if current_debug_node.exec_time is not None else 0

        # We run the debug cycle for a number of times
        # or until time runs out, whichever comes first
        for _ in range(debug_depth):
            # Create the debugged node
            # 调用 Debug 原子算子生成修复候选
            fixed_node_attempt = self._debug(current_debug_node)
            # Evaluate the attempt
            try:
                # 立即在沙箱执行修复候选并解析执行反馈（execution-grounded）
                state, eval_result = task.step_task(state, extract_code(fixed_node_attempt.code))
                self.parse_eval_result(node=fixed_node_attempt, eval_result=eval_result)
                if self._experience_enabled():
                    self._append_evaluated_node(fixed_node_attempt)
                debug_path.append(fixed_node_attempt)
                current_debug_node = fixed_node_attempt  # Update the node for the next iteration
            except Exception as e:
                self.logger.error(f"Error during debug step execution/parsing: {e}", LogEvent.SOLVER)
                break  # Break the debug cycle on execution/parsing error

            # Break if we have a fixed metric - i.e. the solution is no longer buggy
            if not current_debug_node.is_buggy:
                # 修复成功：记录最终指标并退出调试循环
                fixed_metric = current_debug_node.metric.value
                self.logger.info(
                    f"Debug cycle successful for node {buggy_node.id}, final node {current_debug_node.id} has metric: {fixed_metric}",
                    LogEvent.SOLVER,
                )
                break

            # Add exec time if available
            # 累计调试耗时（含每次失败的执行时间）
            if current_debug_node.exec_time is not None:
                total_debug_cycle_time += current_debug_node.exec_time

            # or if the debug time is reached we break
            # 超过调试时间预算则放弃本轮修复
            if total_debug_cycle_time >= self.cfg.max_debug_time:
                self.logger.info(
                    f"Debug cycle time exceeded: {total_debug_cycle_time} seconds for initial node {buggy_node.id}",
                    LogEvent.SOLVER,
                )
                break

        # Return state, the full path, and the metric of the final node (or None if not fixed)
        # 返回完整调试路径：即便未修复，失败链条也会被日志保留（用于重复失败检测）
        return state, debug_path, fixed_metric

    def _new_solution_database(self) -> SolutionsDatabase:
        # 按配置构造新的岛屿数据库
        return SolutionsDatabase(
            num_islands=self.cfg.num_islands,
            max_size=self.cfg.max_island_size,
            lower_is_better=self.lower_is_better,
            logger=self.logger,
            experience_config=dict(getattr(self.cfg, "experience", {}) or {}),
        )

    def _seed_solution_database_from_journal(self, solution_database: SolutionsDatabase) -> None:
        # 从 journal 恢复种群：只挑"非根且非 bug"的可复用节点
        good_nodes = [
            node
            for node in self.journal.nodes
            if not self.journal.is_root_node(node) and not bool(getattr(node, "is_buggy", True))
        ]
        if not good_nodes:
            return
        # 轮转分配到各岛，保证恢复后的种群分布均衡
        island_ids = [
            index % solution_database.num_islands
            for index, _ in enumerate(good_nodes)
        ]
        solution_database.add_nodes_to_islands(good_nodes, island_ids, migration_prob=0.0)

    def _restore_solution_database_from_journal(self) -> SolutionsDatabase:
        """Create the islands database and restore all resumable journal nodes."""
        # 断点续跑入口：新建数据库并把历史好解重新播种进岛屿
        solution_database = self._new_solution_database()
        self.solution_database = solution_database
        self._seed_solution_database_from_journal(solution_database)
        return solution_database

    def _async_worker_count(self) -> int:
        # 解析异步 worker 数：支持整数与 "auto"（=可见 GPU 数）
        return resolve_async_worker_count(getattr(self.cfg, "async_workers", 1))

    def _execution_mode(self) -> str:
        # 解析执行模式：generation（标准逐代）或 async_steady_state（Evo-Max 异步）
        return resolve_execution_mode(getattr(self.cfg, "execution_mode", "generation"))

    def _async_attempt_limit(self) -> int:
        # 异步搜索的尝试上限 = 代数 × 每代个体数（把"代"折算成 attempt 总量）
        return max(
            1,
            int(self.cfg.num_generations)
            * max(1, int(self.cfg.individuals_per_generation)),
        )

    def _async_sandbox_urls(self, task) -> list[str]:
        # 解析异步沙箱地址列表：优先用求解器配置，其次回退到任务侧沙箱 base_url
        sandbox_cfg = dict(getattr(task, "cfg", {}).get("sandbox", {}) or {})
        fallback_url = str(sandbox_cfg.get("base_url") or "")
        return resolve_async_sandbox_urls(
            getattr(self.cfg, "async_sandbox_urls", None),
            fallback_url=fallback_url,
        )

    def _should_stop_async_search(
        self,
        task,
        *,
        include_attempt_limit: bool = True,
    ) -> bool:
        # 异步搜索的统一停止判据：外部请求 / 有效时间预算 / 墙钟上限 / 步数上限 / 尝试数上限
        if getattr(task, "stop_requested", False):
            return True
        wall_elapsed = max(0.0, time.monotonic() - self.start_time)
        exempt_wait = self._task_budget_exempt_seconds(task)
        # 有效耗时 = 墙钟 - 豁免等待（排队不算搜索时间）
        effective_elapsed = max(0.0, wall_elapsed - exempt_wait)
        if effective_elapsed >= float(self.cfg.time_limit_secs):
            return True
        max_wall_time = float(
            getattr(self.cfg, "max_wall_time_secs", 0.0) or 0.0
        )
        if max_wall_time > 0 and wall_elapsed >= max_wall_time:
            return True
        if int(self.state.current_step) >= int(self.cfg.step_limit):
            return True
        return (
            include_attempt_limit
            and int(self.state.current_generation) >= self._async_attempt_limit()
        )

    def _next_async_work_item(
        self,
        *,
        solution_database: SolutionsDatabase,
        worker: WorkerSpec,
    ) -> AsyncWorkItem:
        # 异步搜索的"采样阶段"（须在 sample_lock 内调用）：
        # 分配 attempt 编号、折算虚拟代数、退火温度，并选择算子与父代。
        attempt_id = int(self.state.current_generation)
        self.state.current_generation += 1
        # attempt -> generation 的整数除法：每 individuals_per_generation 个 attempt 记为一代
        generation_id = attempt_id // max(
            1,
            int(self.cfg.individuals_per_generation),
        )
        temperature = self.linear_decay(iteration=generation_id)

        if not solution_database.has_nodes:
            # 种群为空：只能 Draft（搜索的强制起点）
            return AsyncWorkItem(
                attempt_id=attempt_id,
                generation_id=generation_id,
                temperature=temperature,
                island_id=random.choice(range(solution_database.num_islands)),
                operator="draft",
                parent_nodes=[],
                parent_selection_trace=None,
                worker=worker,
            )

        # 前 num_generations_till_crossover 代禁用 Crossover（种群尚小，交叉无意义）
        crossover_prob = 0 if generation_id < self.cfg.num_generations_till_crossover else self.cfg.crossover_prob
        crossover_parent_count = int(dict(self.cfg.few_shot).get("crossover", 2))
        if not solution_database.has_island_with_size(crossover_parent_count):
            # 没有任何岛能提供足够父代时同样禁用 Crossover
            crossover_prob = 0

        try:
            # 复用标准模式的采样逻辑：算子选择 + 父代加权采样（含审计轨迹）
            parent_nodes, island_id, operator, parent_selection_trace = solution_database.sample_in_context_with_trace(
                self.cfg.few_shot,
                temperature,
                crossover_prob,
                float(getattr(self.cfg, "fresh_draft_prob", 0.0) or 0.0),
            )
        except RuntimeError:
            # 采样异常时兜底为 Draft，保证异步流水不断流
            parent_nodes, island_id, operator, parent_selection_trace = (
                [],
                random.choice(range(solution_database.num_islands)),
                "draft",
                None,
            )

        if operator == "draft":
            # Draft 无父代：随机挑一个岛作为新解的归属
            island_id = random.choice(range(solution_database.num_islands))
            parent_nodes = []

        # 打包为不可变工作单元，交给 worker 在锁外并发执行
        return AsyncWorkItem(
            attempt_id=attempt_id,
            generation_id=generation_id,
            temperature=temperature,
            island_id=island_id,
            operator=operator,
            parent_nodes=list(parent_nodes),
            parent_selection_trace=parent_selection_trace,
            worker=worker,
        )

    def _create_node_from_work_item(self, work_item: AsyncWorkItem) -> Node:
        # 异步执行的"生成阶段"：按工作单元分派到对应原子算子（同步阻塞调用，
        # 由外层 asyncio.to_thread 放入线程池，不阻塞事件循环）
        if work_item.operator == "improve":
            node = self._improve(work_item.parent_nodes[0])
        elif work_item.operator == "crossover":
            node = self._crossover(work_item.parent_nodes[0], work_item.parent_nodes[1])
        else:
            node = self._draft()
        # 记录异步元数据：便于事后按 worker/GPU/沙箱归因与复盘
        node.async_work_metadata = {
            "execution_mode": "async_steady_state",
            "attempt_id": work_item.attempt_id,
            "generation_id": work_item.generation_id,
            "temperature": work_item.temperature,
            "worker_id": work_item.worker.worker_id,
            "gpu_index": work_item.worker.gpu_index,
            "sandbox_url": work_item.worker.sandbox_url,
        }
        if work_item.parent_selection_trace is not None:
            node.experience_parent_selection = work_item.parent_selection_trace
        return node

    async def _step_task_async(self, task, state, code: str, sandbox_url: str):
        # 异步执行候选代码：任务支持原生 async 接口则直接 await；
        # 否则把同步 step_task 丢进线程池，避免阻塞事件循环
        if hasattr(task, "step_task_async"):
            return await task.step_task_async(
                dict(state),
                code,
                sandbox_base_url=sandbox_url,
            )
        return await asyncio.to_thread(task.step_task, dict(state), code)

    async def _commit_async_node(
        self,
        *,
        task,
        node: Node,
        solution_database: SolutionsDatabase,
        work_item: AsyncWorkItem,
        commit_lock: asyncio.Lock,
        add_to_database: bool,
        commit_count: list[int],
    ) -> bool:
        # 异步执行的"提交阶段"：串行化（commit_lock）地写入日志树与种群数据库，
        # 保证 journal 与岛屿状态的一致性（生成/执行并行，提交串行）
        async with commit_lock:
            if self._should_stop_async_search(
                task=task,
                include_attempt_limit=False,
            ):
                # 预算已尽：拒绝提交，尽快收尾
                return False
            self._append_evaluated_node(node)
            self.state.running_time = time.monotonic() - self.start_time
            if add_to_database and not bool(node.is_buggy):
                # 早期代数禁用迁移（num_generations_till_migration），让各岛先独立发展
                migration_prob = (
                    0
                    if work_item.generation_id < self.cfg.num_generations_till_migration
                    else self.cfg.migration_prob
                )
                solution_database.add_nodes_to_islands(
                    [node],
                    [work_item.island_id],
                    migration_prob,
                )
            commit_count[0] += 1
            # 每 N 次提交落盘一次 checkpoint（异步模式下的可恢复性保障）
            checkpoint_every = max(1, int(getattr(self.cfg, "async_checkpoint_every_commits", 1)))
            if commit_count[0] % checkpoint_every == 0:
                self.save_checkpoint()
            return True

    async def _debug_cycle_async(
        self,
        *,
        state,
        task,
        buggy_node: Node,
        solution_database: SolutionsDatabase,
        work_item: AsyncWorkItem,
        commit_lock: asyncio.Lock,
        commit_count: list[int],
    ) -> None:
        # 异步版 Debug 循环：与标准 debug_cycle 逻辑一致，但每步都经过
        # commit_lock 提交，且用 asyncio.to_thread 包裹同步 LLM 调用
        await self._commit_async_node(
            task=task,
            node=buggy_node,
            solution_database=solution_database,
            work_item=work_item,
            commit_lock=commit_lock,
            add_to_database=False,  # 失败节点不进种群，只进日志
            commit_count=commit_count,
        )
        current_debug_node = buggy_node
        debug_path = [buggy_node]
        total_debug_cycle_time = current_debug_node.exec_time if current_debug_node.exec_time is not None else 0

        for debug_attempt_index in range(self.cfg.max_debug_depth):
            if self._should_stop_async_search(
                task,
                include_attempt_limit=False,
            ):
                # 预算耗尽立即退出调试循环
                break
            try:
                # 调用 Debug 原子算子（在线程池中执行，避免阻塞事件循环）
                fixed_node_attempt = await asyncio.to_thread(self._debug, current_debug_node)
                fixed_node_attempt.async_work_metadata = {
                    **dict(getattr(buggy_node, "async_work_metadata", {}) or {}),
                    "debug_attempt_index": debug_attempt_index,
                }
                # 在该 worker 绑定的沙箱上执行修复候选（execution-grounded 反馈）
                _, eval_result = await self._step_task_async(
                    task,
                    state,
                    extract_code(fixed_node_attempt.code),
                    work_item.worker.sandbox_url,
                )
                self.parse_eval_result(node=fixed_node_attempt, eval_result=eval_result)
                await self._commit_async_node(
                    task=task,
                    node=fixed_node_attempt,
                    solution_database=solution_database,
                    work_item=work_item,
                    commit_lock=commit_lock,
                    # 修好才允许入种群
                    add_to_database=not bool(fixed_node_attempt.is_buggy),
                    commit_count=commit_count,
                )
                current_debug_node = fixed_node_attempt
                debug_path.append(fixed_node_attempt)
            except Exception as exc:
                if exc.__class__.__name__ == "StopSearch":
                    # StopSearch 是控制流异常：向上抛出以终止整个搜索
                    raise
                self.logger.error(f"Error during async debug step: {exc}", LogEvent.SOLVER)
                break

            if not current_debug_node.is_buggy:
                self.logger.info(
                    f"Async debug cycle fixed node {buggy_node.id} with node {current_debug_node.id}",
                    LogEvent.SOLVER,
                )
                break
            if current_debug_node.exec_time is not None:
                total_debug_cycle_time += current_debug_node.exec_time
            if total_debug_cycle_time >= self.cfg.max_debug_time:
                break
        if current_debug_node.is_buggy:
            # 仍未修好：检测重复失败签名，必要时请求下一次强制 Draft
            repeated_failure_reason = self._repeated_debug_failure_reason(debug_path)
            if repeated_failure_reason:
                solution_database.request_fresh_draft(repeated_failure_reason)

    async def _async_worker_loop(
        self,
        *,
        worker: WorkerSpec,
        state,
        task,
        solution_database: SolutionsDatabase,
        sample_lock: asyncio.Lock,
        commit_lock: asyncio.Lock,
        stop_event: asyncio.Event,
        commit_count: list[int],
    ) -> None:
        # Evo-Max 异步稳态搜索的单 worker 主循环（每个 worker 绑定一个沙箱端点）：
        # 采样(加锁) -> 生成(解锁并发) -> 执行 -> 失败则 Debug -> 提交(加锁)
        while not stop_event.is_set():
            if self._should_stop_async_search(task):
                stop_event.set()
                break

            async with sample_lock:
                # 采样必须串行：岛屿状态与 attempt 计数只能被一个 worker 同时修改
                if self._should_stop_async_search(task):
                    stop_event.set()
                    break
                work_item = self._next_async_work_item(
                    solution_database=solution_database,
                    worker=worker,
                )

            max_retries = max(
                0,
                int(getattr(self.cfg, "async_worker_max_retries", 3) or 0),
            )
            retry_backoff = max(
                0.0,
                float(
                    getattr(
                        self.cfg,
                        "async_worker_retry_backoff_secs",
                        1.0,
                    )
                    or 0.0
                ),
            )
            failure_index = 0
            while not stop_event.is_set():
                try:
                    # 阶段 1：LLM 生成候选程序（线程池执行同步算子）
                    child_node = await asyncio.to_thread(
                        self._create_node_from_work_item,
                        work_item,
                    )
                    # 阶段 2：在该 worker 的沙箱上执行并解析执行反馈
                    _, eval_result = await self._step_task_async(
                        task,
                        state,
                        extract_code(child_node.code),
                        work_item.worker.sandbox_url,
                    )
                    self.parse_eval_result(child_node, eval_result)
                    if child_node.is_buggy:
                        # 阶段 3a：执行失败 -> 进入异步 Debug 循环
                        await self._debug_cycle_async(
                            state=state,
                            task=task,
                            buggy_node=child_node,
                            solution_database=solution_database,
                            work_item=work_item,
                            commit_lock=commit_lock,
                            commit_count=commit_count,
                        )
                    else:
                        # 阶段 3b：执行成功 -> 直接提交入种群
                        await self._commit_async_node(
                            task=task,
                            node=child_node,
                            solution_database=solution_database,
                            work_item=work_item,
                            commit_lock=commit_lock,
                            add_to_database=True,
                            commit_count=commit_count,
                        )
                    break
                except Exception as exc:
                    if exc.__class__.__name__ == "StopSearch":
                        stop_event.set()
                        break
                    if failure_index >= max_retries:
                        # 重试耗尽：放弃该 attempt，worker 继续下一个
                        self.logger.error(
                            f"Async worker {worker.worker_id} exhausted retries "
                            f"for attempt {work_item.attempt_id}: {exc}",
                            LogEvent.SOLVER,
                        )
                        break
                    # 指数退避重试（backoff * 2^failure_index）
                    delay = retry_backoff * (2**failure_index)
                    failure_index += 1
                    self.logger.warning(
                        f"Async worker {worker.worker_id} failed attempt "
                        f"{work_item.attempt_id}; retry {failure_index}/"
                        f"{max_retries} in {delay:.2f}s: {exc}",
                        LogEvent.SOLVER,
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)

    async def search_async_steady_state(self, task, state, *, worker_count: int) -> Optional[str]:
        # 【Evo-Max 异步搜索入口】与标准逐代搜索不同：没有"代"边界，
        # 多个 worker 以流水线方式持续"采样-生成-执行-提交"，最大化 GPU 与沙箱利用率。
        if self.journal.nodes:
            self.root_node = self.journal.nodes[0]
            if self.state.current_step != len(self.journal):
                # 续跑时以 journal 实际长度校准步数
                self.state.current_step = len(self.journal)
        else:
            self.create_root_node()

        # 从 checkpoint 恢复岛屿种群（好解重新播种）
        solution_database = self._restore_solution_database_from_journal()

        sandbox_urls = self._async_sandbox_urls(task)
        worker_specs = build_worker_specs(
            worker_count=worker_count,
            sandbox_urls=sandbox_urls,
            assignment=str(
                getattr(
                    self.cfg,
                    "async_sandbox_assignment",
                    "round_robin",
                )
            ),
        )
        # 探测各沙箱路由器的空闲 worker 数（仅日志观测，不参与调度）
        for url in sandbox_urls:
            idle_count = await fetch_router_idle_worker_count(url)
            if idle_count is not None:
                self.logger.info(f"Sandbox router {url} idle workers: {idle_count}", LogEvent.SOLVER)
        self.logger.info(
            f"Starting async steady-state evolutionary search with {worker_count} workers and sandbox URLs: {sandbox_urls}",
            LogEvent.SOLVER,
        )

        # 两把锁：采样锁（保护岛屿/计数）与提交锁（保护 journal/种群写入）
        sample_lock = asyncio.Lock()
        commit_lock = asyncio.Lock()
        stop_event = asyncio.Event()
        commit_count = [0]
        # 为每个 WorkerSpec 启动一个异步 worker 协程
        workers = [
            asyncio.create_task(
                self._async_worker_loop(
                    worker=worker,
                    state=state,
                    task=task,
                    solution_database=solution_database,
                    sample_lock=sample_lock,
                    commit_lock=commit_lock,
                    stop_event=stop_event,
                    commit_count=commit_count,
                )
            )
            for worker in worker_specs
        ]
        try:
            await asyncio.gather(*workers)
        finally:
            # 无论正常结束还是异常，都要停掉全部 worker 并保存 checkpoint
            stop_event.set()
            for worker_task in workers:
                worker_task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            self.save_checkpoint()

        best_node = self.journal.get_best_node()
        return state, best_node.code if best_node is not None else None

    def parse_eval_result(self, node: Node, eval_result: Dict[str, Any]):
        """
        Parse evaluation results and update the node accordingly.

        Processes the execution output, extracts metrics, determines if the solution
        is buggy, and updates the node with this information. Also applies the analysis
        operator to get additional insights about the solution.

        Args:
            node: The node to update with evaluation results
            eval_result: Dictionary containing evaluation results from task execution
        """
        # 【执行反馈解析核心】把沙箱执行结果变成节点的三项属性：
        # 终端输出（term_out）、是否 bug（is_buggy）、适应度（metric）。
        # 这是 execution-grounded 的关键环节：优先采用沙箱确定性反馈而非 LLM 自评。
        self.logger.debug(f"Parsing execution results for node {node.id}")

        # Safely ensure we have eval_result
        if isinstance(eval_result, dict):
            assert EXECUTION_OUTPUT in eval_result
        else:
            raise ValueError(f"Unexpected eval_result type: {type(eval_result)}")

        # Absorb the execution output into the node
        # 把执行 stdout/stderr 写入节点（后续作为 Improve/Debug 的反馈上下文）
        node.absorb_exec_result(eval_result[EXECUTION_OUTPUT])

        # Extract potential auxliary evaluation information to store
        # in the node. This is more for logging purposes.
        aux_eval_info = eval_result.get(AUX_EVAL_INFO, {}) or {}
        if not isinstance(aux_eval_info, dict):
            aux_eval_info = {}

        def coerce_float(value):
            # 安全转 float：非数值与 inf/nan 一律返回 None
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            if not numpy.isfinite(numeric):
                return None
            return numeric

        def deterministic_failure() -> bool:
            # 依据沙箱元数据判定确定性失败（不依赖 LLM 判断）
            failure_statuses = {
                "buggy",
                "error",
                "failed",
                "failure",
                "invalid",
                "sandbox_error",
                "scoring_failed",
                "submission_missing",
                "timeout",
                "unknown",
            }
            status = str(aux_eval_info.get("status") or "").strip().lower()
            if status in failure_statuses:
                return True
            status_code = aux_eval_info.get("status_code")
            try:
                # HTTP 语义：状态码 >= 400 视为失败
                return status_code is not None and int(status_code) >= 400
            except (TypeError, ValueError):
                return False

        def run_legacy_analyze() -> dict:
            # 传统路径：调用 analyze 算子（LLM）判 bug 并抽取指标
            try:
                analyze_response = self._analyze(node)
            except Exception as e:
                self.logger.error(f"Error during analysis operator: {str(e)}")
                analyze_response = {}

            parsed_response = parse_json_output(analyze_response)
            if not isinstance(parsed_response, dict):
                self.logger.warning(f"Parsed response is not a dictionary: {type(parsed_response)}")
                parsed_response = {}

            if len(parsed_response) == 0:
                # LLM 未给出有效结构 => 保守判定为 bug
                return {"metric": None, "summary": "", "is_bug": True}

            if "metric" not in parsed_response:
                parsed_response["metric"] = None
            if "summary" not in parsed_response:
                parsed_response["summary"] = ""
            if "is_bug" not in parsed_response:
                parsed_response["is_bug"] = parsed_response["metric"] is None
            return parsed_response

        # 沙箱确定性算出的验证集适应度（若可用则完全无需 LLM 自评）
        validation_fitness = coerce_float(eval_result.get(VALIDATION_FITNESS))
        sanitize_prompt_scores = prompt_score_sanitization_enabled(self.cfg)
        experience_enabled = self._experience_enabled()

        if not experience_enabled:
            # Preserve the inference AIRA-Evo path: call analyze for every node,
            # then let deterministic validation fitness override the metric.
            # 基线推理路径：每个节点都跑 analyze，再用确定性适应度覆盖其结果
            response = run_legacy_analyze()
            if validation_fitness is not None:
                response["metric"] = validation_fitness
                response["is_bug"] = False
        elif validation_fitness is not None:
            # 首选路径：沙箱已给出确定性验证适应度，跳过 analyze 省一次 LLM 调用
            response = {
                "metric": validation_fitness,
                "summary": sanitize_execution_output_for_prompt(
                    aux_eval_info.get("feedback") or "Deterministic validation fitness available.",
                    enabled=sanitize_prompt_scores,
                ),
                "is_bug": False,
            }
        elif deterministic_failure() or node.exit_code != 0:
            # 次选路径：沙箱明确报告失败（状态/状态码/退出码），标记 bug 并回传反馈
            response = {
                "metric": None,
                "summary": sanitize_execution_output_for_prompt(
                    aux_eval_info.get("feedback") or aux_eval_info.get("clear_run_log") or "",
                    enabled=sanitize_prompt_scores,
                ),
                "is_bug": True,
            }
        else:
            # Fall back to the legacy analyze operator only when sandbox metadata
            # cannot deterministically provide a metric or failure state.
            # 兜底路径：沙箱元数据既无指标也无失败信号，才退回 LLM analyze
            response = run_legacy_analyze()

        if not isinstance(response["metric"], (float, int)):
            response["metric"] = None

        # 分析摘要存入节点（作为后续算子的文字反馈）
        node.analysis = response["summary"]

        if self.cfg.use_test_score:
            # 显式开启 use_test_score 时直接采用测试分数（谨慎：易造成测试集泄漏式过拟合）
            test_score = aux_eval_info.get("score", None)
            aux_eval_info["validation_score"] = response["metric"]
            response["metric"] = test_score
            self.logger.info(f"Using Test score: {test_score}")

        # Determine if solution is valid
        # If the task does not return this key, we assume the solution is valid
        valid_solution = eval_result.get(VALID_SOLUTION, True)
        validity_feedback = eval_result.get(VALID_SOLUTION_FEEDBACK, None)
        if validity_feedback is not None:
            aux_eval_info["validity_feedback"] = validity_feedback
            validity_feedback = f"\n\n Evaluator Feedback: {validity_feedback}"
            node._term_out.append(validity_feedback)
        else:
            aux_eval_info["validity_feedback"] = "evaluator feedback not available"

        # 最终 is_buggy 判定：LLM 判 bug / 非零退出码 / 无指标 / 解非法 —— 任一成立即失败
        node.is_buggy = (
            response["is_bug"] or (not node.exit_code == 0) or (response["metric"] is None) or (not valid_solution)
        )

        if node.is_buggy:
            # 失败节点打上"最差指标"哨兵，保证其在一切比较中排序垫底
            node.metric = WorstMetricValue(info=aux_eval_info)
            self.logger.debug(f"Node {node.id} marked as buggy")
        else:
            node.metric = MetricValue(response["metric"], maximize=not self.lower_is_better, info=aux_eval_info)
            self.logger.debug(f"Node {node.id} metric: {response['metric']}")

    def create_root_node(self):
        # 创建虚拟根节点：所有 Draft 节点的形式父代，不含真实代码
        self.root_node = Node(
            code="",
            plan="",
            analysis="",
            metric=WorstMetricValue(maximize=not self.lower_is_better),
            is_buggy=True,
        )
        self.root_node.absorb_exec_result(None)
        self.journal.append(self.root_node)
        self.log_journal()
        self.state.current_step += 1

    def search(self, task, state) -> Optional[str]:
        # 【搜索主循环入口】先按执行模式分流：
        # async_steady_state 走 Evo-Max 异步搜索；generation 走下方标准逐代搜索
        execution_mode = self._execution_mode()
        worker_count = self._async_worker_count()
        if execution_mode == "async_steady_state":
            return asyncio.run(
                self.search_async_steady_state(
                    task,
                    state,
                    worker_count=worker_count,
                )
            )
        if worker_count != 1:
            raise ValueError(
                "generation mode requires exactly one worker; "
                "use execution=multi_gpu for async steady-state search"
            )

        if self.journal.nodes:
            self.root_node = self.journal.nodes[0]
            if self.state.current_step != len(self.journal):
                # checkpoint 状态与 journal 不一致时，以 journal 为准（宁可多算不可少算）
                self.logger.warning(
                    f"Checkpoint step {self.state.current_step} does not match journal length {len(self.journal)}. "
                    "Using journal length for resume.",
                    LogEvent.SOLVER,
                )
                self.state.current_step = len(self.journal)
        else:
            self.create_root_node()
        self.logger.info("Starting evolutionary search", LogEvent.SOLVER)

        # Restore resumable solutions into the same islands database used by async mode.
        solution_database = self._restore_solution_database_from_journal()

        # define a schedule for temperature of sampling
        temp_scheduler = self.linear_decay

        # 标准逐代主循环：从续跑代开始直到 num_generations
        for generation_id in range(self.state.current_generation, self.cfg.num_generations):
            # Measure state time of the generation
            start_time = time.monotonic()
            generation_exempt_time_start = self._task_budget_exempt_seconds(task)

            # fix the temperature for sampling
            # 本代采样温度（线性退火）
            temperature = temp_scheduler(iteration=generation_id)
            self.logger.info(
                f"\n========= Generation {generation_id} | temperature: {round(temperature, 2)} ==========",
                LogEvent.SOLVER,
            )

            solution_nodes = []
            island_ids = []
            counter_ids = []
            # 内层个体循环：本代要产出 individuals_per_generation 个个体
            for counter_id in range(self.cfg.individuals_per_generation):
                if generation_id == 0:  # initially, uniformly populate the islands
                    # 第 0 代：全部走 Draft，随机均匀播种到各岛
                    island_id = random.choice(range(solution_database.num_islands))
                    create_node_fn = self._draft
                    in_context_nodes = []
                else:  # gen_id > 0: start the evolutionary process
                    # 第 1 代起：采样决定"算子 + 父代"（进化过程正式开始）
                    in_context_nodes, island_id, operator = solution_database.sample_in_context(
                        self.cfg.few_shot,
                        temperature,
                        (0 if generation_id < self.cfg.num_generations_till_crossover else self.cfg.crossover_prob),
                        self.cfg.fresh_draft_prob,
                    )  # weighted sampling of islands and corresponding individuals
                    # 按采样结果分派到对应原子算子的调用点
                    if operator == "improve":
                        create_node_fn = self._improve
                    elif operator == "draft":
                        island_id = random.choice(range(solution_database.num_islands))
                        create_node_fn = self._draft
                        in_context_nodes = []
                    else:
                        create_node_fn = self._crossover

                island_ids.append(island_id)

                self.logger.info(
                    f"Creating node for individual {counter_id} in generation {generation_id}", LogEvent.SOLVER
                )

                # 调用原子算子生成子代（draft 无参 / improve 一个父代 / crossover 两个父代）
                child_node = create_node_fn(*in_context_nodes)
                if generation_id > 0 and solution_database.last_parent_selection is not None:
                    # 把父代选择审计轨迹挂到子代节点上（可追溯、可复盘）
                    child_node.experience_parent_selection = solution_database.last_parent_selection
                # 执行子代代码并解析执行反馈（execution-grounded 闭环的闭合点）
                state, eval_result = task.step_task(state, extract_code(child_node.code))
                self.parse_eval_result(child_node, eval_result)
                # if the node is buggy, we run a debug cycle
                # and add the fixed node to the generation
                # if the node is not buggy, we add it to the generation
                if not child_node.is_buggy:
                    # 执行成功：写入日志并进入本代的解集合
                    self._append_evaluated_node(child_node)
                    solution_nodes.append(child_node)
                    counter_ids.append(counter_id)
                else:
                    self.logger.info(f"Node {child_node.id} was buggy, entering debug cycle.", LogEvent.SOLVER)
                    # 执行失败：进入 Debug 循环尝试修复
                    state, debug_path, fixed_metric = self.debug_cycle(state, task, child_node)
                    if not self._experience_enabled():
                        for n in debug_path:
                            self._append_evaluated_node(n)
                    if fixed_metric is not None:
                        fixed_node = debug_path[-1]  # Get the last node (the fixed one)
                        self.logger.info(
                            f"Node {child_node.id} was fixed by node {fixed_node.id}, adding fixed node to generation.",
                            LogEvent.SOLVER,
                        )
                        # Add only the final fixed node to the generation
                        # 只有最终修复成功的节点才能代表该个体进入种群
                        solution_nodes.append(fixed_node)
                        counter_ids.append(counter_id)
                    else:
                        self.logger.info(
                            f"Node {child_node.id} could not be fixed by debug cycle, discarding individual {counter_id}.",
                            LogEvent.SOLVER,
                        )
                        # 修复失败：检测重复失败签名，必要时请求下轮强制 Draft
                        repeated_failure_reason = self._repeated_debug_failure_reason(debug_path)
                        if repeated_failure_reason:
                            solution_database.request_fresh_draft(repeated_failure_reason)

            # store individuals solutions only if it improves overall island fitness
            # for initialization, we don't use this step
            if generation_id > 0:
                # 稳态准入：仅"不劣于岛均适应度"的解入岛，并按概率触发岛屿迁移/重置
                solution_database.add_nodes_to_islands(
                    solution_nodes,
                    island_ids,
                    (0 if generation_id < self.cfg.num_generations_till_migration else self.cfg.migration_prob),
                )
            else:  # initialization step (generation = 0)
                # 第 0 代：直接播种（无准入竞争）
                solution_database.seed_islands_with_nodes(
                    solution_nodes,
                    island_ids,
                )

            generation_wall_time = time.monotonic() - start_time
            generation_exempt_time = max(
                0.0,
                self._task_budget_exempt_seconds(task) - generation_exempt_time_start,
            )
            generation_effective_time = max(0.0, generation_wall_time - generation_exempt_time)
            # 只有"有效时间"计入预算（排队等 GPU 的豁免时间被剔除）
            self.state.running_time += generation_effective_time
            self.logger.info(
                f"Step {self.state.current_step} | Generation {self.state.current_generation}: "
                f"effective={generation_effective_time:.3f}s, gpu_wait={generation_exempt_time:.3f}s, "
                f"budget_used={self.state.running_time:.3f}s"
            )

            # Update the state with the current generation
            self.state.current_generation += 1

            self.logger.info(f"Step {self.state.current_step}: Saving checkpoint")
            # 每代结束落盘 checkpoint，保证任意代可恢复
            self.save_checkpoint()

            if self.state.running_time >= self.cfg.time_limit_secs or self.state.current_step >= self.cfg.step_limit:
                self.logger.info("Maximum runtime reached, stopping search")
                break

        best_node = self.journal.get_best_node()
        return state, best_node.code if best_node is not None else None

    def log_journal(self):
        # Get the current best node in the tree.
        # 结构化记录：当前步的节点数据 + 当前最优节点所在步
        best_node = self.journal.get_best_node()
        best_node_step = 0 if best_node is None else best_node.step

        self.logger.log(
            self.journal.get_node_data(self.state.current_step) | {"current_best_node": best_node_step},
            "JOURNAL",
            step=self.state.current_step,
        )

    def __call__(self, task, state):
        # 求解器入口：首次运行先生成数据预览（所有算子共享该上下文）
        # Possibly generate data preview first
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview(state)

        state, solution = self.search(task, state)

        # Export results at the end of the search process
        # 搜索结束导出完整结果（搜索树/节点表/指标曲线）
        export_search_results(self.cfg, self.journal, self.logger, "EVO")

        # Get the best node for returning
        best_node = self.journal.get_best_node()

        # Return the best node
        # 返回最优节点代码（交由评测脚本计算 Medal Average 等最终指标）
        if best_node:
            return state, best_node.code, best_node
        else:
            self.logger.info("No suitable code found after all generations.", LogEvent.SOLVER)
            return state, None, None
