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

# 导入 json：检查点日志(Journal)的序列化/反序列化
import json
# 导入 random：岛屿/个体随机采样与迁移触发
import random
# 导入 sys：系统模块（保留以备扩展）
import sys
# 导入 partial：预绑定 LLM 与配置生成算子函数
from functools import partial
# 导入 Path：路径处理
from pathlib import Path
# 导入类型注解
from typing import Any, Dict, List, Optional, Tuple, Union, cast

# 导入 hydra：注册配置入口（保留以备扩展）
import hydra
# 导入 numpy：适应度统计计算（min/max/mean/argsort/clip 等）
import numpy
# 导入 time：统计每代耗时
import time

# 导入搜索工具模块：如 normalized 归一化采样权重函数
import dojo.core.solvers.utils.search_utils as utils
# 导入求解器基类，Evolutionary 继承自它
from dojo.core.solvers.base import Solver
# 导入分析算子：解读执行结果、判断 bug 并抽取指标
from dojo.core.solvers.operators.analyze import analyze_op
# 导入算子执行封装：统一调用"规划+写码"类算子并解析输出
from dojo.core.solvers.operators.core import execute_op_plan_code
# 导入交叉算子：融合两个父代个体生成新解（遗传算法核心算子）
from dojo.core.solvers.operators.crossover import crossover_op
# 导入起草算子：从零生成新方案
from dojo.core.solvers.operators.draft import draft_op
# 导入改进算子：优化现有方案
from dojo.core.solvers.operators.improve import improve_op
# 导入调试算子：修复有 bug 的代码
from dojo.core.solvers.operators.debug import debug_op
# 导入记忆算子工厂：为算子构建经验记忆上下文
from dojo.core.solvers.operators.memory import create_memory_op
# 导入数据预览工具：生成数据集概要供 LLM 参考
from dojo.core.solvers.utils import data_preview
# 导入搜索日志(Journal)与节点(Node)：以树结构记录解的演化历史
from dojo.core.solvers.utils.journal import Journal, Node
# 导入指标值封装：正常指标与"最差指标"
from dojo.core.solvers.utils.metric import MetricValue, WorstMetricValue
# 导入代码抽取工具：从 LLM 响应中剥离可执行代码
from dojo.core.solvers.utils.response import extract_code
# 导入搜索结果导出工具
from dojo.core.solvers.utils.search_exporter import export_search_results
# 导入复杂度级别判定工具
from dojo.solvers.utils import get_complextiy_level
# 导入集合日志器与日志事件枚举
from dojo.utils.logger import CollectiveLogger, LogEvent
# 导入 JSON 解析工具：稳健解析 LLM 输出
from dojo.utils.code_parsing import parse_json_output
# 导入任务评估结果字典的键名常量
from dojo.core.tasks.constants import (
    EXECUTION_OUTPUT,
    TASK_DESCRIPTION,
    VALID_SOLUTION_FEEDBACK,
    VALIDATION_FITNESS,
    AUX_EVAL_INFO,
    VALID_SOLUTION,
)
# 导入进化求解器配置数据类
from dojo.config_dataclasses.solver.evo import EvolutionarySolverConfig
# 导入进化求解器状态类（当前代数、步数、运行时长等）
from dojo.utils.state import EvolutionaryState
# 导入通用 LLM 封装：按配置实例化不同用途的语言模型
from dojo.core.solvers.llm_helpers.generic_llm import GenericLLM


# 岛屿(Island)：进化算法岛屿模型中的一个子种群，独立维护一组解节点并周期性接受迁移
class Island:
    """A population of solutions (Nodes) in an island."""

    # 构造函数：岛屿编号、优化方向、日志器与初始解集合
    def __init__(
        self,
        island_id: int,
        initial_solution_nodes: List[Node],
        lower_is_better: bool,
        logger: CollectiveLogger,
    ):
        # 岛屿编号
        self.island_id = island_id
        # 优化方向：分数越低越好与否
        self.lower_is_better = lower_is_better
        # 集合日志器
        self.logger = logger
        # 岛屿种群：解节点列表
        self.nodes: List[Node] = initial_solution_nodes

    # 种群规模属性
    @property
    def size(self) -> int:
        # 返回节点数量
        return len(self.nodes)

    # 全体个体的适应度(Fitness)分数列表
    @property
    def fitness_scores(self) -> List[float]:
        """Returns fitness scores, using +/- inf for None metrics."""
        # 初始化得分列表
        scores = []
        # 遍历每个节点
        for node in self.nodes:
            # 取节点指标值
            fitness = node.metric.value
            # 无指标（无效解）时
            if fitness is None:
                # Use +/- inf for comparison purposes
                # 用正/负无穷占位，保证排序比较时垫底
                scores.append(float("inf") if self.lower_is_better else float("-inf"))
            else:
                # 正常节点直接记录得分
                scores.append(fitness)
        # 返回得分列表
        return scores

    # 岛屿最佳适应度分数
    @property
    def best_fitness_score(self) -> float:
        """Returns the best fitness score in the island."""
        # 取全部得分
        scores = self.fitness_scores
        # 空岛屿
        if not scores:
            # Return worst possible score if island is empty
            # 返回最差分数
            return float("inf") if self.lower_is_better else float("-inf")
        # 依优化方向选择 min 或 max 聚合
        op = numpy.min if self.lower_is_better else numpy.max
        # 计算最佳分
        return op(scores)

    # 岛屿平均适应度（用于"新解是否改进岛屿"的判断）
    @property
    def average_fitness_score(self):
        """Returns the average fitness score."""
        # 取全部得分
        scores = self.fitness_scores
        # 空岛屿
        if not scores:
            # Avoid numpy warning on empty list, return worst score
            # 返回最差分数（同时避免 numpy 对空列表告警）
            return float("inf") if self.lower_is_better else float("-inf")
        # numpy.mean handles inf correctly (inf + finite = inf, inf + inf = inf, inf - inf = nan)
        # However, if all scores are inf/-inf, mean might be inf/-inf.
        # Filter out inf/-inf before calculating mean for a more representative average?
        # Or rely on the fact that inf/-inf indicates poor solutions.
        # 剔除正负无穷占位值，只对有限得分求均值
        finite_scores = [s for s in scores if s != float("inf") and s != float("-inf")]
        # 全部为无穷占位时
        if not finite_scores:
            # If only inf/-inf scores, return that score
            # 直接返回该占位得分
            return scores[0] if scores else (float("inf") if self.lower_is_better else float("-inf"))
        # 返回有限得分的算术平均
        return numpy.mean(finite_scores)

    # 最优个体：返回适应度最好的节点
    @property
    def fittest_individual(self) -> Optional[Node]:  # Renamed to fittest_node
        """Returns the node with the best fitness score."""
        # 空岛屿返回 None
        if not self.nodes:
            # 无个体
            return None
        # 取全部得分
        scores = self.fitness_scores
        # 依优化方向选 argmin/argmax
        op = numpy.argmin if self.lower_is_better else numpy.argmax
        # 最优个体的下标
        fittest_idx = op(scores)
        # 返回最优节点
        return self.nodes[fittest_idx]

    # 种群节点列表只读属性
    @property
    def solution_nodes(self) -> List[Node]:  # Keep this property
        # 直接返回内部列表
        return self.nodes

    # 将解节点注册（加入）岛屿种群
    def register_node_in_island(
        self,
        solution_node: Node,
    ):
        """
        Add a solution Node to the island population.
        """
        # IDs are stored within the node if needed elsewhere (e.g., node.id)
        # 日志：登记节点及其指标
        self.logger.info(
            f"Registering Node {solution_node.id} in Island {self.island_id}. Metric: {solution_node.metric.value}",
            LogEvent.SOLVER,
        )
        # 追加到种群列表
        self.nodes.append(solution_node)

    # 淘汰机制：移除岛上适应度最差的个体（用于控制种群规模）
    def remove_lowest(self):
        """
        Removes the node with the worst fitness score in the island.
        """
        # 空岛屿无需淘汰
        if not self.nodes:
            return  # Nothing to remove

        # 取全部得分
        scores = self.fitness_scores
        # 越低越好时最差为最大分，反之最差为最小分
        op = numpy.argmax if self.lower_is_better else numpy.argmin
        # 最差个体的下标
        lowest_score_index = op(scores)
        # 弹出并移除该个体
        weakest_node = self.nodes.pop(lowest_score_index)
        # 日志：输出被淘汰个体及其分数
        self.logger.info(
            f"Removed weakest node {weakest_node.id} (Score: {scores[lowest_score_index]}) from island {self.island_id}",
            LogEvent.SOLVER,
        )

    # 按节点 ID 精确移除（迁移时从源岛移出个体）
    def remove_node(self, to_remove_node: Node):  # Renamed from remove_individual
        """
        Remove a specific node from the island.
        """
        # 记录移除前规模
        initial_size = len(self.nodes)
        # 重建列表：过滤掉目标节点
        self.nodes = [node for node in self.nodes if node.id != to_remove_node.id]
        # 规模变小说明移除成功
        if len(self.nodes) < initial_size:
            # 日志：移除成功
            self.logger.info(f"Removed node {to_remove_node.id} from island {self.island_id}", LogEvent.SOLVER)
        else:
            # 告警：目标节点不在岛上
            self.logger.warning(
                f"Attempted to remove node {to_remove_node.id} from island {self.island_id}, but it was not found.",
                LogEvent.SOLVER,
            )

    # 精英保留：只留最优个体（岛屿重置时使用）
    def only_keep_best(self):
        """
        Remove all nodes except the single best one.
        Handles ties by keeping only one of the best.
        """
        # 取最优个体
        best_node = self.fittest_individual  # Renamed property
        # 岛非空时
        if best_node:
            # 日志：保留最优个体及其分数
            self.logger.info(
                f"Island {self.island_id}: Keeping only best node {best_node.id} (Score: {best_node.metric.value})",
                LogEvent.SOLVER,
            )
            # 种群只保留最优个体
            self.nodes = [best_node]
        else:
            # 日志：空岛屿调用本方法
            self.logger.info(f"Island {self.island_id}: only_keep_best called on empty island.", LogEvent.SOLVER)
            # 置空种群
            self.nodes = []

    # 迁移(Migration)：接收来自源岛屿（founder island）的个体
    def migrate_node(
        self,
        founder_node: Node,  # Changed type to Node
    ):
        """
        Migrate a node from a founder island to this island.
        """
        # 日志：迁移个体到达本岛
        self.logger.info(f"Migrating node {founder_node.id} to Island {self.island_id}", LogEvent.SOLVER)
        # 以注册方式加入本岛种群
        self.register_node_in_island(founder_node)


class SolutionsDatabase:
    """
    Maintains and updates a Database of all solutions (Nodes).

    Adapted from Fun Search: https://github.com/google-deepmind/funsearch/blob/main
    and from REvolve: https://arxiv.org/pdf/2406.01309
    """

    # 构造函数：初始化岛屿群与全局适应度范围
    def __init__(
        self,
        num_islands: int,
        max_size: int,
        lower_is_better: bool,
        logger: CollectiveLogger,
    ):
        # 岛屿数量
        self.num_islands = num_islands
        # 每岛最大种群规模
        self.max_size = max_size
        # 优化方向：分数越低越好与否
        self.lower_is_better = lower_is_better
        # 集合日志器
        self.logger = logger
        # 岛屿列表（稍后统一初始化）
        self._islands: List[Island] = []
        # 全局最小适应度（归一化基准），初始为正无穷
        self.global_min_fitness = float("inf")
        # 全局最大适应度，初始为负无穷
        self.global_max_fitness = float("-inf")

        # Initialize empty islands.
        # 创建 num_islands 个空岛屿
        self._islands = [
            Island(island_id, [], self.lower_is_better, self.logger) for island_id in range(self.num_islands)
        ]

    # 导出数据库状态字典（用于日志与持久化）
    def state_dict(self) -> dict:
        # 包含岛屿数、容量上限、优化方向、全局适应度范围与各岛成员 ID
        return {
            "num_islands": self.num_islands,
            "max_size": self.max_size,
            "lower_is_better": self.lower_is_better,
            "global_min_fitness": self.global_min_fitness,
            "global_max_fitness": self.global_max_fitness,
            "island_populations": [
                [str(node.id) for node in island.nodes]
                for island in self._islands
            ],
        }

    # 播种：把第一代解分配到指定岛屿（初始阶段均匀填充各岛）
    def seed_islands_with_nodes(
        self,
        solution_nodes: List[Node],
        island_ids: List[int],
    ):
        """
        Initialize islands with the first generation of Nodes.
        """
        # 按节点与岛屿 ID 配对逐个放入
        for solution_node, island_id in zip(
            solution_nodes,
            island_ids,
        ):
            # 岛屿 ID 合法性检查
            if 0 <= island_id < len(self._islands):
                # 注册进对应岛屿
                self._islands[island_id].register_node_in_island(
                    solution_node,
                )
            else:
                # 非法 ID：记录日志说明范围
                self.logger.info(f"Invalid island_id {island_id} during seeding. Max index: {len(self._islands) - 1}")
            # Seed initial global fitness range if nodes have scores
            # 取节点得分
            fitness_score = solution_node.metric.value
            # 得分有效（非空且有限）时
            if fitness_score is not None and numpy.isfinite(fitness_score):
                # 顺带初始化全局适应度范围
                self._update_global_fitness_range(fitness_score)

    # 更新全局适应度的最小/最大值范围（作为得分归一化基准）
    def _update_global_fitness_range(self, score: float):
        """Updates the global min and max fitness scores seen so far."""
        # 仅统计有限值
        if numpy.isfinite(score):
            # 更新全局下界
            self.global_min_fitness = min(self.global_min_fitness, score)
            # 更新全局上界
            self.global_max_fitness = max(self.global_max_fitness, score)
            # 日志：输出更新后的范围
            self.logger.info(
                f"Updated global fitness range: min={self.global_min_fitness}, max={self.global_max_fitness}",
                LogEvent.SOLVER,
            )

    def get_normalized_score(self, score: Optional[float]) -> float:
        """
        Normalizes a raw fitness score to the range [0, 1], where 1.0 is always best.
        Handles None scores by returning 0.0 (worst).
        """
        # 空值或非有限值
        if score is None or not numpy.isfinite(score):
            # 视为最差（归一化 0 分）
            return 0.0  # Worst normalized score for None or non-finite scores

        # If global range isn't established or is a single point, return neutral 0.5
        # 全局范围尚未建立或退化为单点时，返回中性 0.5
        if (
            not numpy.isfinite(self.global_min_fitness)
            or not numpy.isfinite(self.global_max_fitness)
            or self.global_min_fitness == self.global_max_fitness
        ):
            # 返回中性值
            return 0.5

        # Normalize to [0, 1]
        # 按优化方向选择归一化公式
        if self.lower_is_better:
            # For lower_is_better, higher values are worse.
            # (global_max - score) / (global_max - global_min)
            # If score is min_fitness, result is 1. If score is max_fitness, result is 0.
            # 反向归一化：分数越接近全局最小值，结果越接近 1
            normalized = (self.global_max_fitness - score) / (self.global_max_fitness - self.global_min_fitness)
        else:
            # For higher_is_better, higher values are better.
            # (score - global_min) / (global_max - global_min)
            # If score is min_fitness, result is 0. If score is max_fitness, result is 1.
            # 正向归一化：分数越接近全局最大值，结果越接近 1
            normalized = (score - self.global_min_fitness) / (self.global_max_fitness - self.global_min_fitness)

        # Clamp to [0, 1] to handle scores outside the current global range robustly
        # 截断到 [0,1]，容忍超出历史范围的分数
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
        # 逐个处理新解及其目标岛屿
        for solution_node, island_id in zip(
            solution_nodes,
            island_ids,
        ):
            # 非法岛屿 ID 检查
            if not (0 <= island_id < len(self._islands)):
                # 记录错误并跳过该解
                self.logger.error(f"Invalid island_id {island_id} when adding node {solution_node.id}. Skipping.")
                # 跳到下一个解
                continue

            # 取新解得分（可能为 None 或浮点数）
            fitness_score = solution_node.metric.value  # This can be None or a float
            # 得分有效时更新全局范围
            if fitness_score is not None and numpy.isfinite(fitness_score):  # Update global range
                # 更新全局适应度范围
                self._update_global_fitness_range(fitness_score)

            # 取目标岛屿
            current_island = self._islands[island_id]
            # 当前岛屿的平均适应度
            island_avg_fitness_score = current_island.average_fitness_score

            # Determine if the new node improves the island
            # 判断新解是否改进岛屿
            if current_island.size == 0:
                # 空岛：无条件接受
                improvement_condition = True  # Always add to an empty island
            elif fitness_score is None:
                # 无得分的新解不视为改进
                improvement_condition = False  # A solution with None fitness never improves an island
            else:
                # 与岛屿平均分比较：不低于（或不高于）平均分即视为改进
                improvement_condition = (self.lower_is_better and fitness_score <= island_avg_fitness_score) or (
                    not self.lower_is_better and fitness_score >= island_avg_fitness_score
                )

            # check if individual is adding any value to the island
            # 通过改进检查才加入岛屿
            if improvement_condition:
                # 注册进岛屿种群
                current_island.register_node_in_island(
                    solution_node,
                )
                # 日志：入岛成功并输出新的平均分
                self.logger.info(
                    f"Node {solution_node.id} added to island {island_id}. New avg score: {current_island.average_fitness_score:.4f}",
                    LogEvent.SOLVER,
                )

            # if island size exceeds max size, discard individual with the lowest score
            # 超出岛屿容量上限时淘汰最差个体
            if current_island.size > self.max_size:
                # 日志：容量超限
                self.logger.info(
                    f"Exceeded maximum size ({self.max_size}) on island {island_id}, discarding weakest node(s)",
                    LogEvent.SOLVER,
                )
                # 循环淘汰直至回到上限
                while current_island.size > self.max_size:
                    # 移除适应度最差者
                    current_island.remove_lowest()

        # Migration / Reset Logic
        # 以迁移概率触发岛屿重置（多岛时才生效）
        if len(self._islands) > 1 and random.random() <= migration_prob:
            # 执行重置：弱岛重建、强岛输出迁移个体
            self.reset_islands()

    def reset_islands(self):
        """
        Resets the weaker half of islands and seeds them
        with nodes migrated from fitter islands.
        """
        # 少于两个岛屿无法重置
        if len(self._islands) < 2:  # Cannot reset if less than 2 islands
            # 告警并跳过
            self.logger.warning("Reset islands called with less than 2 islands. Skipping.", LogEvent.SOLVER)
            # 直接返回
            return

        # 日志：开始重置岛屿
        self.logger.info("============ Resetting Islands ============", LogEvent.SOLVER)

        # Get island scores (best score per island)
        # 收集各岛最佳得分
        island_scores = []
        # 遍历岛屿
        for island in self._islands:
            # 取该岛最佳分
            best_score = island.best_fitness_score  # Uses +/- inf for empty/None
            # 收集
            island_scores.append(best_score)

        # Add small noise to break ties during sorting
        # 加极小噪声打破并列，避免排序不稳定
        noisy_scores = numpy.array(island_scores) + numpy.random.randn(len(self._islands)) * 1e-9

        # Sort islands by score (ascending for lower_is_better, descending otherwise)
        # 按分数对岛屿下标升序排序
        indices_sorted_by_score = numpy.argsort(noisy_scores)
        # 越高越好时需要反转为降序
        if not self.lower_is_better:
            # 翻转排序结果
            indices_sorted_by_score = indices_sorted_by_score[::-1]

        # 重置数量：一半岛屿
        num_islands_to_reset = len(self._islands) // 2
        # 至少保留一个岛
        if num_islands_to_reset == 0:  # Ensure at least one island is kept
            # 日志：岛屿过少
            self.logger.info("Not enough islands to perform reset.", LogEvent.SOLVER)
            # 放弃重置
            return

        # 最弱的若干岛进入重置名单
        reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
        # 其余为保留的强岛
        keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]

        # 日志：输出重置名单
        self.logger.info(f"Resetting islands: {reset_islands_ids}", LogEvent.SOLVER)
        # 日志：输出保留名单
        self.logger.info(f"Keeping islands: {keep_islands_ids}", LogEvent.SOLVER)

        # 逐个重置弱岛
        for reset_island_id in reset_islands_ids:
            # 取被重置岛
            reset_island = self._islands[reset_island_id]
            # Keep only the single best node in the island being reset
            # 精英保留：只留该岛最优个体
            reset_island.only_keep_best()

            # Find a suitable founder island (must have > 1 node to donate)
            # 筛选可捐赠个体的强岛：规模需大于 1
            possible_founder_ids = [idx for idx in keep_islands_ids if self._islands[idx].size > 1]
            # 无可用强岛时
            if not possible_founder_ids:
                # 告警：跳过迁移
                self.logger.warning(
                    f"No suitable founder island with >1 node found for resetting island {reset_island_id}. Skipping migration.",
                    LogEvent.SOLVER,
                )
                # 处理下一个待重置岛
                continue  # Skip to next island to reset

            # 随机选择一个源岛(founder island)
            founder_island_id = numpy.random.choice(possible_founder_ids)
            # 取源岛对象
            founder_island = self._islands[founder_island_id]

            # Sample a node from the founder island (weighted by fitness, excluding the absolute best)
            # 候选迁移个体列表
            candidates = []
            # 对应得分（用于加权采样）
            candidate_scores = []
            # 源岛最佳分（该个体不可迁移）
            best_founder_score = founder_island.best_fitness_score
            # 遍历源岛个体
            for node in founder_island.nodes:
                # 取个体得分
                score = node.metric.value
                # Handle None scores - treat as worst possible for sampling
                # None 得分换算为最差占位分
                numeric_score = (
                    score if score is not None else (float("inf") if self.lower_is_better else float("-inf"))
                )
                # Exclude the absolute best node(s) from being migrated
                # 排除源岛最优个体
                if numeric_score != best_founder_score:
                    # 加入候选列表
                    candidates.append(node)
                    # Use the numeric score for weighting the sampling
                    # 记录其得分供加权用
                    candidate_scores.append(numeric_score)

            # 无候选可迁时
            if not candidates:
                # 告警：跳过该岛迁移
                self.logger.warning(
                    f"Founder island {founder_island_id} has no non-best nodes to migrate. Skipping migration for island {reset_island_id}.",
                    LogEvent.SOLVER,
                )
                # 处理下一个待重置岛
                continue  # Skip to next island to reset

            # Perform weighted sampling (using normalized utility)
            # Note: utils.normalized expects scores where higher is better for probability
            # 候选得分归一化（统一为越大越好语义）
            normalized_candidate_scores = [self.get_normalized_score(s) for s in candidate_scores]
            # 由归一化得分计算 softmax 采样权重（温度 1.0）
            sampling_weights = utils.normalized(
                normalized_candidate_scores, temp=1.0
            )  # Normalized scores are always higher_is_better
            # 权重全零的兜底
            if sum(sampling_weights) == 0:  # Avoid error if all weights are zero
                # Fallback to uniform sampling if weights are zero
                # 退化为均匀随机选择
                founder_node_to_migrate = random.choice(candidates)
                # 告警：权重全零
                self.logger.warning(
                    "Sampling weights were all zero, falling back to uniform selection for migration.",
                    LogEvent.SOLVER,
                )
            else:
                # 按权重加权随机抽一个迁移个体
                founder_node_to_migrate = random.choices(candidates, weights=sampling_weights, k=1)[0]

            # Migrate the selected node
            # 日志：输出迁移方向
            self.logger.info(
                f"Migrating node {founder_node_to_migrate.id} from Island {founder_island_id} to Island {reset_island_id}",
                LogEvent.SOLVER,
            )
            # 迁移个体进入被重置岛
            reset_island.migrate_node(founder_node_to_migrate)

            # Remove the migrated node from the founder island
            # 从源岛移除该个体（移动而非复制，保持总量守恒）
            founder_island.remove_node(founder_node_to_migrate)

    def sample_in_context(
        self, num_samples: Dict, temperature: float, crossover_prob: float
    ) -> Tuple[List[Node], int, str]:
        """
        Samples nodes for the next generation, selecting islands and then nodes based on fitness.
        Returns sampled nodes, the island they came from, and the selected operator ('improve' or 'crossover').
        """
        # 所有岛都为空时无法采样
        if not any(island.size > 0 for island in self._islands):
            # 告警：退回起草模式
            self.logger.warning(
                "Sample in context called, but all islands are empty. Cannot sample. Back to drafting", LogEvent.SOLVER
            )
            # 返回空上下文并标记为起草
            return [], 0, "draft"

        # Calculate average fitness scores for island sampling
        # Use a default worst score for empty islands to give them zero probability
        # 收集各岛平均分（空岛用最差占位分，采样概率为 0）
        island_avg_scores = []
        # 遍历岛屿
        for island in self._islands:
            # 非空岛取平均分
            if island.size > 0:
                # 收集平均分
                island_avg_scores.append(island.average_fitness_score)
            else:
                # 空岛垫底
                island_avg_scores.append(float("inf") if self.lower_is_better else float("-inf"))

        # Normalize scores for sampling probabilities (higher score = higher probability)
        # Since get_normalized_score handles lower_is_better and returns higher_is_better output,
        # we pass lower_is_better=False to utils.normalized
        # 岛屿平均分归一化（输出统一为越大越好）
        normalized_island_avg_scores = [self.get_normalized_score(s) for s in island_avg_scores]
        # 计算岛屿采样权重（带温度的 softmax）
        island_sampling_weights = utils.normalized(
            normalized_island_avg_scores,
            temp=temperature,  # Apply temperature to avg scores for island selection
            # Normalized scores are always higher_is_better
        )

        # Ensure weights sum to 1 (handle potential all-zero case)
        # 权重全零的兜底
        if sum(island_sampling_weights) == 0:
            # 告警
            self.logger.warning("Island sampling weights are all zero. Falling back to uniform island selection.")
            # 退化为均匀分布
            island_sampling_weights = [1.0 / len(self._islands)] * len(self._islands)

        # 调试日志：输出岛屿采样权重
        self.logger.debug(f"Island sampling weights: {island_sampling_weights}", LogEvent.SOLVER)

        # Determine operator and required number of samples
        # 依交叉概率选择算子：改进(Improve)或交叉(Crossover)
        operator = "improve" if random.random() >= crossover_prob else "crossover"
        # 取该算子需要的示例个体数（缺省 1）
        num_in_context_samples = num_samples.get(operator, 1)  # Default to 1 if operator key missing
        # 无任何岛屿有足够个体做交叉时
        if not any(island.size >= num_in_context_samples for island in self._islands) and operator == "crossover":
            # 退回改进算子所需样本数
            fallback_samples = num_samples.get("improve", 1)
            # 若有岛能满足改进采样
            if any(island.size >= fallback_samples for island in self._islands):
                # 告警：交叉降级为改进
                self.logger.warning(
                    "Crossover was selected, but no island has enough nodes. Falling back to improve.",
                    LogEvent.SOLVER,
                )
                # 切换算子为改进
                operator = "improve"
                # 同步更新所需样本数
                num_in_context_samples = fallback_samples

        # Loop until a suitable island and samples are found
        # 已采样岛 ID
        sampled_island_id = -1
        # 已采样岛对象
        sampled_island = None
        # 选中的上下文节点
        in_context_nodes = []
        # 重试计数
        attempts = 0
        # 最大重试次数（启发式上限，防止死循环）
        max_attempts = len(self._islands) * 2  # Heuristic limit to prevent infinite loops

        # 采样重试循环
        while attempts < max_attempts:
            # 计数加一
            attempts += 1
            # STEP 1: Sample an island based on average fitness
            # 第一步：按权重加权抽岛
            sampled_island_id = random.choices(range(len(self._islands)), weights=island_sampling_weights, k=1)[0]
            # 取岛对象
            sampled_island = self._islands[sampled_island_id]

            # Check if the sampled island has enough nodes for the operator
            # 该岛个体数不足所需样本数
            if sampled_island.size < num_in_context_samples:
                # 调试日志：重采样岛
                self.logger.debug(
                    f"Sampled island {sampled_island_id} size {sampled_island.size} < required {num_in_context_samples}. Resampling island.",
                    LogEvent.SOLVER,
                )
                # 重新抽岛
                continue  # Resample island

            # STEP 2: Sample nodes within the island (weighted by individual fitness)
            # 第二步：取岛上全部个体得分
            island_node_scores = sampled_island.fitness_scores
            # 个体得分归一化
            normalized_node_scores = [self.get_normalized_score(s) for s in island_node_scores]
            # 计算个体采样权重
            node_sampling_weights = utils.normalized(
                normalized_node_scores, temperature
            )  # Normalized scores are always higher_is_better

            # 权重全零的兜底
            if sum(node_sampling_weights) == 0:
                # 告警
                self.logger.warning(
                    f"Node sampling weights on island {sampled_island_id} are zero. Falling back to uniform node selection."
                )
                # Use uniform sampling if weights are zero
                # 均匀无放回抽取个体下标
                indices = numpy.random.choice(range(sampled_island.size), size=num_in_context_samples, replace=False)
            else:
                # 按权重无放回抽取个体下标
                try:
                    indices = numpy.random.choice(
                        range(sampled_island.size),
                        p=node_sampling_weights,
                        size=num_in_context_samples,
                        replace=False,
                    )
                except ValueError as e:
                    # This might happen if weights don't sum to 1 due to float precision
                    # 浮点精度导致权重和不为 1 时报错
                    self.logger.error(
                        f"Error sampling nodes on island {sampled_island_id}: {e}. Weights: {node_sampling_weights}. Falling back to uniform."
                    )
                    # 均匀采样兜底
                    indices = numpy.random.choice(
                        range(sampled_island.size), size=num_in_context_samples, replace=False
                    )

            # 取出选中的个体节点
            in_context_nodes = [sampled_island.nodes[i] for i in indices]
            # 日志：输出算子、来源岛与选中节点
            self.logger.info(
                f"{operator.capitalize()} | Sampled island: {sampled_island_id}. Nodes: {[n.id for n in in_context_nodes]}",
                LogEvent.SOLVER,
            )
            # 返回上下文节点、来源岛与算子名
            return in_context_nodes, sampled_island_id, operator

        # If loop finishes without returning, something went wrong
        # 错误：多次尝试均失败
        self.logger.error(f"Failed to sample context after {max_attempts} attempts. All islands might be too small.")
        # 抛出运行时异常，提示检查岛屿规模或采样逻辑
        raise RuntimeError(
            "Failed to sample context nodes after multiple attempts. Check island sizes or sampling logic."
        )


# 进化(Evolutionary)求解器：多岛并行的 LLM 进化搜索（FunSearch/REvolve 风格，含交叉与迁移机制）
class Evolutionary(Solver):
    # 构造函数：初始化日志、任务描述、提示词上下文与各 LLM 算子
    def __init__(self, cfg: EvolutionarySolverConfig, task_info):
        # 调用基类构造
        super().__init__(cfg, task_info=task_info)
        # 创建搜索日志
        self.journal = Journal()
        # 数据预览缓存：初始为空
        self.data_preview: str | None = None

        # 保存任务描述
        self.task_desc = task_info[TASK_DESCRIPTION]
        # 提示词上下文：任务/数据描述与公开系统、用户提示（供各算子渲染 prompt）
        self.prompt_context = {
            "task_description": str(task_info.get("task_description") or self.task_desc),
            "data_description": str(task_info.get("data_description") or ""),
            "public_system_prompt": str(task_info.get("public_system_prompt") or ""),
            "public_user_prompt": str(task_info.get("public_user_prompt") or ""),
        }
        # 读取优化方向：分数越低越好与否
        self.lower_is_better = task_info.get("lower_is_better", None)
        # 优化方向必须显式给出
        assert self.lower_is_better is not None  # Ensure lower_is_better is set

        # 初始化各 LLM 算子
        self.setup_operators()

        # 初始化进化状态（当前代数、步数、运行时长）
        self.state = EvolutionaryState()

    # 保存检查点：在基类保存之外，额外把搜索日志写为 JSONL
    def save_checkpoint(self):
        # 先执行基类的检查点保存
        super().save_checkpoint()

        # Write the journal to a jsonl file
        # 把日志中的全部节点序列化为字典列表
        journal_sd = self.journal.node_list()
        # 拼出 journal.jsonl 的保存路径
        journal_path = Path(self.cfg.checkpoint_path) / "journal.jsonl"
        # 打开文件逐行写出
        with open(journal_path, "w") as f:
            # 遍历所有节点
            for node in journal_sd:
                # JSONL 格式：一行一个节点
                f.write(json.dumps(node) + "\n")
        # 日志：检查点已保存
        self.logger.info(f"Checkpoint saved to {journal_path}")

    # 加载检查点：恢复搜索日志实现断点续跑
    def load_checkpoint(self):
        # 先执行基类的检查点加载
        super().load_checkpoint()

        # 检查点日志路径
        journal_path = Path(self.cfg.checkpoint_path) / "journal.jsonl"
        # 若日志文件不存在
        if not journal_path.exists():
            # 但状态却存在：数据不一致，断言报错
            assert self.state.current_step == 0, (
                f"No journal found at {journal_path}, but the state was found. This is unexpected."
            )
            # 无历史记录，从头开始
            return

        # 日志：发现历史日志，开始加载
        self.logger.info(f"Found journal at {journal_path}. Loading...")
        # Load the journal
        # 打开文件并逐行解析
        with open(journal_path, "r") as f:
            # 每行 JSON 解析为一个节点字典
            journal_export = [json.loads(line) for line in f]
        # 由导出数据重建日志对象
        self.journal = Journal.from_export_data({"nodes": journal_export})

    # 初始化各 LLM 算子（含交叉算子）
    def setup_operators(self):
        """Setup operator LLMs."""

        # First we set up the LLMs
        # 按配置实例化"起草"算子的 LLM
        draft_llm = GenericLLM(self.cfg.operators["draft"])
        # 实例化"改进"算子的 LLM
        improve_llm = GenericLLM(self.cfg.operators["improve"])
        # 实例化"调试"算子的 LLM
        debug_llm = GenericLLM(self.cfg.operators["debug"])
        # 实例化"交叉"算子的 LLM
        crossover_llm = GenericLLM(self.cfg.operators["crossover"])
        # 实例化"分析"算子的 LLM
        analyze_llm = GenericLLM(self.cfg.operators["analyze"])

        # Create the memory for operators
        # 构建经验记忆算子：提供历史成败经验上下文
        self.memory_op = create_memory_op(self.cfg.memory)
        # 构建调试专用记忆算子：提供历史报错与修复经验
        self.debug_memory_op = create_memory_op(self.cfg.debug_memory)

        # Then we create the operators
        # 预绑定起草算子（额外注入提示词上下文）
        self.draft_fn = partial(
            draft_op, draft_llm, self.cfg, self.memory_op, self.prompt_context
        )
        # 预绑定改进算子
        self.improve_fn = partial(
            improve_op, improve_llm, self.cfg, self.memory_op, self.prompt_context
        )
        # 预绑定调试算子
        self.debug_fn = partial(
            debug_op, debug_llm, self.cfg, self.debug_memory_op, self.prompt_context
        )
        # 预绑定分析算子
        self.analyze_fn = partial(analyze_op, analyze_llm, self.cfg)
        # 预绑定交叉(Crossover)算子：融合两个父代个体
        self.crossover_fn = partial(
            crossover_op, crossover_llm, self.cfg, self.memory_op, self.prompt_context
        )

    # 起草(Draft)算子封装：从根节点生成全新方案
    def _draft(self) -> Node:
        """
        Generate a new solution from scratch using the draft LLM operator.

        Uses the draft operator to create a new code solution based on the task description.
        The resulting code is packaged into a new Node object with relevant metadata.

        Returns:
            Node: A new node containing the drafted solution
        """
        # 执行起草算子：传入任务描述、日志、剩余时间、数据预览与复杂度级别，得到计划/代码/指标
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
        # 将计划与代码封装为新节点：父节点为根节点
        node = Node(
            plan=plan, code=code, parents=[self.root_node], operators_used=["draft"], operators_metrics=[metrics]
        )
        # 日志：起草节点创建及其指标
        self.logger.info(f"Draft Node Created - Metrics: {metrics}")
        # 日志：起草得到的代码
        self.logger.info(f"Draft Code: {code}")
        # 返回新节点
        return node

    # 改进(Improve)算子封装：在父节点方案基础上优化并生成新节点
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
        # 执行改进算子：传入父节点、剩余时间、复杂度级别与数据预览等上下文
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
        # 封装为新节点：父节点指向被改进节点
        node = Node(
            plan=plan, code=code, parents=[parent_node], operators_used=["improve"], operators_metrics=[metrics]
        )
        # 日志：改进节点创建及其指标
        self.logger.info(f"Improve Node Created - Metrics: {metrics}")
        # 日志：改进后的代码
        self.logger.info(f"Improve Code: {code}")
        # 返回新节点
        return node

    # 调试(Debug)算子封装：根据父节点的报错信息修复代码
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
        # 执行调试算子：调试 LLM 会看到父节点的执行输出/报错信息
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
        # 封装为新节点：父节点指向被调试节点
        node = Node(plan=plan, code=code, parents=[parent_node], operators_used=["debug"], operators_metrics=[metrics])
        # 日志：调试节点创建及其指标
        self.logger.info(f"Debug Node Created - Metrics: {metrics}")
        # 日志：调试后的代码
        self.logger.info(f"Debug Code: {code}")
        # 返回修复后的新节点
        return node

    # 分析(Analyze)算子封装：解读节点执行结果，判断 bug 并抽取指标
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
        # 调用分析算子：得到分析文本与算子调用指标
        analysis, metrics = self.analyze_fn(self.task_desc, node)
        # 在节点上记录使用过的算子序列
        node.operators_used.append("analysis")
        # 记录分析算子的调用指标
        node.operators_metrics.append(metrics)
        # 日志：分析完成及其指标
        self.logger.info(f"Node Analysis Performed - Metrics: {metrics}")
        # 返回分析结果
        return analysis

    # 交叉(Crossover)算子封装：融合两个父代个体的思路生成新子代（遗传算法核心算子）
    def _crossover(self, parent_node1: Node, parent_node2: Node) -> Node:
        # 执行交叉算子：传入两个父节点及运行上下文
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
        # 新节点记录两个父节点，形成 DAG 演化结构
        node = Node(
            plan=plan,
            code=code,
            parents=[parent_node1, parent_node2],
            operators_used=["crossover"],
            operators_metrics=[metrics],
        )
        # 日志：交叉节点创建及其指标
        self.logger.info(f"Crossover Node Created - Metrics: {metrics}")
        # 日志：交叉得到的代码
        self.logger.info(f"Crossover Code: {code}")
        # 返回子代节点
        return node

    # 生成数据预览（可由配置关闭）
    def update_data_preview(self, state):
        """
        Generate a data preview to provide context for the LLM operators.

        Creates a small preview of the data (head, shapes, etc.) that can be used
        to help the LLM understand the data structure when generating solutions.

        Args:
            state: The current solver state containing the interpreter
        """
        # 生成数据预览必须能访问求解器解释器
        assert "solver_interpreter" in state, (
            "For generating data previews, the solver needs access to an interpreter."
        )

        # 配置关闭数据预览时
        if not self.cfg.data_preview:
            # 预览置空
            self.data_preview = ""
            # 直接返回
            return

        # 调试日志：开始生成预览
        self.logger.debug("Generating data preview")
        # 本地解释器：直接在本地工作目录生成预览
        if state["solver_interpreter"].local:
            # 读取工作目录数据并生成概要文本
            self.data_preview = data_preview.generate(state["solver_interpreter"].working_dir)
        else:
            # 非本地解释器：把预览脚本注入远程解释器执行
            import inspect

            # 定位 data_preview 模块的源码文件
            path = Path(inspect.getsourcefile(data_preview))
            # 读取脚本全文
            script = path.read_text()
            # 拼接脚本与调用语句：在远程工作目录生成预览并打印
            code = f"{script}\nprint(generate(Path('.').resolve()))"
            # 在远程解释器中执行该脚本
            exec_result = state["solver_interpreter"].run(code, include_exec_time=False)
            # 以终端输出作为数据预览内容
            self.data_preview = "\n".join(exec_result.term_out)
        # 调试日志：预览生成完毕
        self.logger.debug("Data preview generated")

    # 温度线性退火：随代数从 initial_temp 线性降到 final_temp（控制采样随机性）
    def linear_decay(self, iteration: int):
        """defines a temperature schedule for sampling of islands and individuals"""
        # 读取初始温度
        initial_temp = self.cfg.initial_temp
        # 读取最终温度
        final_temp = self.cfg.final_temp
        # 读取总代数
        num_generations = self.cfg.num_generations
        # 线性插值得到当前代的温度
        return initial_temp - (initial_temp - final_temp) * iteration / num_generations

    # 调试循环：迭代修复有 bug 节点，直至修复成功、超时或超深度
    def debug_cycle(self, state, task, buggy_node: Node):
        # 以初始 bug 节点为当前处理对象
        current_debug_node = buggy_node  # Start with the initial buggy node
        # 调试路径包含初始节点
        debug_path = [current_debug_node]  # Path includes the initial buggy node
        # 最大调试轮数（取配置上限）
        debug_depth = self.cfg.max_debug_depth
        # 修复后的指标（未修复则保持 None）
        fixed_metric = None
        # We set the initial debug cycle time to the execution time of the first buggy node
        # 调试计时起点（首个节点无耗时则记 0）
        total_debug_cycle_time = current_debug_node.exec_time if current_debug_node.exec_time is not None else 0

        # We run the debug cycle for a number of times
        # or until time runs out, whichever comes first
        # 逐轮调试
        for _ in range(debug_depth):
            # Create the debugged node
            # 生成修复尝试节点
            fixed_node_attempt = self._debug(current_debug_node)
            # Evaluate the attempt
            # 执行并评估修复尝试
            try:
                # 执行修复后的代码并取得评估结果
                state, eval_result = task.step_task(state, extract_code(fixed_node_attempt.code))
                # 解析评估结果回填节点
                self.parse_eval_result(node=fixed_node_attempt, eval_result=eval_result)
                # 加入调试路径
                debug_path.append(fixed_node_attempt)
                # 更新当前节点为最新尝试
                current_debug_node = fixed_node_attempt  # Update the node for the next iteration
            except Exception as e:
                # 执行/解析异常：记录错误
                self.logger.error(f"Error during debug step execution/parsing: {e}", LogEvent.SOLVER)
                # 中断调试循环
                break  # Break the debug cycle on execution/parsing error

            # Break if we have a fixed metric - i.e. the solution is no longer buggy
            # 修复成功（不再有 bug）
            if not current_debug_node.is_buggy:
                # 记录修复后指标
                fixed_metric = current_debug_node.metric.value
                # 日志：调试成功及其最终指标
                self.logger.info(
                    f"Debug cycle successful for node {buggy_node.id}, final node {current_debug_node.id} has metric: {fixed_metric}",
                    LogEvent.SOLVER,
                )
                # 结束调试循环
                break

            # Add exec time if available
            # 有执行耗时则累计
            if current_debug_node.exec_time is not None:
                # 累加到调试总耗时
                total_debug_cycle_time += current_debug_node.exec_time

            # or if the debug time is reached we break
            # 调试时间超限
            if total_debug_cycle_time >= self.cfg.max_debug_time:
                # 日志：调试超时
                self.logger.info(
                    f"Debug cycle time exceeded: {total_debug_cycle_time} seconds for initial node {buggy_node.id}",
                    LogEvent.SOLVER,
                )
                # 结束调试循环
                break

        # Return state, the full path, and the metric of the final node (or None if not fixed)
        # 返回状态、调试路径与修复指标
        return state, debug_path, fixed_metric

    # 解析评估结果：吸收执行输出、运行分析算子、确定指标与 bug 标记并回填节点
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
        # 调试日志：开始解析该节点的执行结果
        self.logger.debug(f"Parsing execution results for node {node.id}")

        # Safely ensure we have eval_result
        # 校验评估结果必须是字典
        if isinstance(eval_result, dict):
            # 且必须包含"执行输出"键
            assert EXECUTION_OUTPUT in eval_result
        else:
            # 其他类型直接报错
            raise ValueError(f"Unexpected eval_result type: {type(eval_result)}")

        # Absorb the execution output into the node
        # 把执行输出（终端输出、退出码、耗时等）吸收进节点
        node.absorb_exec_result(eval_result[EXECUTION_OUTPUT])

        # Safely perform the analyze operation
        # 安全地运行分析算子
        try:
            # 调用分析算子得到响应
            response = self._analyze(node)
        except Exception as e:
            # 分析失败只记录错误，不中断搜索
            self.logger.error(f"Error during analysis operator: {str(e)}")
            # 失败时以空字典兜底
            response = {}

        # Parse response to dictionary
        # If the response is a string, we try to parse it into a dictionary
        # 把分析输出稳健解析为字典（容忍 LLM 输出格式不规范）
        response = parse_json_output(response)

        # Validate it's actually a dictionary
        # 解析结果必须是字典
        if not isinstance(response, dict):
            # 否则告警
            self.logger.warning(f"Parsed response is not a dictionary: {type(response)}")
            # 以空字典兜底
            response = {}

        # If the response is empty, we return a default dictionary
        # 空响应：采用默认值（无指标、判定为 bug）
        if len(response) == 0:
            # 默认字典：无指标、空摘要、有 bug
            response = {"metric": None, "summary": "", "is_bug": True}
        else:
            # 补全缺失的 metric 键
            if "metric" not in response:
                # 缺省置 None
                response["metric"] = None
            # 补全缺失的 summary 键
            if "summary" not in response:
                # 缺省置空串
                response["summary"] = ""
            # 补全缺失的 is_bug 键：按是否有指标推断
            if "is_bug" not in response:
                # 有指标则认为无 bug
                if response["metric"] is not None:
                    # 标记无 bug
                    response["is_bug"] = False
                else:
                    # 无指标则判定有 bug
                    response["is_bug"] = True

        # If the metric isn't a float or int then fill the metric with the worst metric
        # 指标必须是数值，否则视为无效
        if not isinstance(response["metric"], (float, int)):
            # 置空以便按最差指标处理
            response["metric"] = None

        # If a validation fitness value is provided (not test fitness) from the task
        # we replace the validation metric with it.
        # 若任务直接返回了验证集适应度，优先采用官方数值
        if eval_result.get(VALIDATION_FITNESS, None) is not None:
            # 用任务给定的验证指标覆盖分析算子的估计值
            response["metric"] = float(eval_result[VALIDATION_FITNESS])
            # 有验证指标即认为方案有效
            response["is_bug"] = False

        # Store the analysis summary
        # 保存分析摘要到节点（去除首尾空白）
        node.analysis = str(response["summary"]).strip()

        # Extract potential auxliary evaluation information to store
        # in the node. This is more for logging purposes.
        # 取辅助评估信息（主要用于日志记录）
        aux_eval_info = eval_result.get(AUX_EVAL_INFO, {})

        # 若配置允许使用测试集得分（注意：存在信息泄露风险）
        if self.cfg.use_test_score:
            # 取测试集得分
            test_score = aux_eval_info.get("score", None)
            # 把验证得分存入辅助信息备查
            aux_eval_info["validation_score"] = response["metric"]
            # 搜索时改用测试得分作为节点指标
            response["metric"] = test_score
            # 日志：使用测试得分
            self.logger.info(f"Using Test score: {test_score}")

        # Determine if solution is valid
        # If the task does not return this key, we assume the solution is valid
        # 读取任务判定解是否有效（未返回该键时默认视为有效）
        valid_solution = eval_result.get(VALID_SOLUTION, True)
        # 读取评分器(grader)的反馈文本
        validity_feedback = eval_result.get(VALID_SOLUTION_FEEDBACK, None)
        # 若有反馈内容
        if validity_feedback is not None:
            # 反馈写入辅助评估信息
            aux_eval_info["validity_feedback"] = validity_feedback
            # 格式化反馈文本（不改动字符串本身）
            validity_feedback = f"\n\n submission.csv Grader Feedback: {validity_feedback}"
            # 追加到节点终端输出，便于 LLM 后续参考
            node._term_out.append(validity_feedback)
        else:
            # 无反馈时的占位说明
            aux_eval_info["validity_feedback"] = "submission grader feedback not available"

        # 综合判定 bug：分析判定为 bug / 非零退出码 / 无有效指标 / 任务判无效，任一成立即有 bug
        node.is_buggy = (
            response["is_bug"] or (not node.exit_code == 0) or (response["metric"] is None) or (not valid_solution)
        )

        # 按判定结果回填节点指标
        if node.is_buggy:
            # 有 bug：指标记为"最差值"，排序时垫底
            node.metric = WorstMetricValue(info=aux_eval_info)
            # 调试日志：节点已标记为 buggy
            self.logger.debug(f"Node {node.id} marked as buggy")
        else:
            # 无 bug：以真实指标创建 MetricValue（附优化方向）
            node.metric = MetricValue(response["metric"], maximize=not self.lower_is_better, info=aux_eval_info)
            # 调试日志：输出节点指标
            self.logger.debug(f"Node {node.id} metric: {response['metric']}")

    # 创建搜索树的虚拟根节点（不是真实解，仅作为所有起草节点的父节点）
    def create_root_node(self):
        # 根节点：空代码、标记 buggy、指标取"最差值"
        self.root_node = Node(
            code="",
            plan="",
            analysis="",
            metric=WorstMetricValue(maximize=not self.lower_is_better),
            is_buggy=True,
        )
        # 根节点不吸收任何执行结果
        self.root_node.absorb_exec_result(None)
        # 根节点加入搜索日志
        self.journal.append(self.root_node)
        # 记录 JOURNAL 事件
        self.log_journal()
        # 步数计数加一
        self.state.current_step += 1

    # 从检查点日志重建解数据库（断点续跑支持）
    def _build_solution_database_from_journal(self) -> SolutionsDatabase:
        # 按配置创建空数据库（岛数、容量、优化方向、日志器）
        solution_database = SolutionsDatabase(
            num_islands=self.cfg.num_islands,
            max_size=self.cfg.max_island_size,
            lower_is_better=self.lower_is_better,
            logger=self.logger,
        )
        # 筛选可恢复节点：非根、无 bug、且指标有效
        solution_nodes = [
            node
            for node in self.journal.nodes
            if not self.journal.is_root_node(node)
            and not node.is_buggy
            and node.metric is not None
            and node.metric.value is not None
        ]
        # 无可恢复节点时
        if not solution_nodes:
            # 返回空数据库
            return solution_database

        # Historical SFT AIRA Evo runs use a single island. For multi-island
        # checkpoints that predate database persistence, place recovered nodes
        # on island 0 rather than discarding the searchable context.
        # 恢复节点全部放入 0 号岛（兼容旧版单岛检查点）
        solution_database.seed_islands_with_nodes(
            solution_nodes,
            [0 for _ in solution_nodes],
        )
        # 日志：输出恢复的节点数
        self.logger.info(
            f"Recovered {len(solution_nodes)} solution nodes from checkpoint journal",
            LogEvent.SOLVER,
        )
        # 返回重建后的数据库
        return solution_database

    # 进化搜索主循环：逐代生成个体、评估、入岛，并按代数启用交叉与迁移
    def search(self, task, state) -> Optional[str]:
        # 无历史日志时创建根节点
        if not self.journal.nodes:
            # 创建虚拟根节点
            self.create_root_node()
        else:
            # 有历史日志时复用原根节点
            self.root_node = self.journal.nodes[0]
        # 日志：开始进化搜索
        self.logger.info("Starting evolutionary search", LogEvent.SOLVER)

        # database of solutions: load all islands if it > 0, else initialize empty islands
        # 从日志恢复解数据库（无历史则得到空岛群）
        solution_database = self._build_solution_database_from_journal()

        # define a schedule for temperature of sampling
        # 绑定温度调度函数（线性退火）
        temp_scheduler = self.linear_decay

        # 逐代迭代（续跑时从当前代数开始）
        for generation_id in range(self.state.current_generation, self.cfg.num_generations):
            # Measure state time of the generation
            # 记录本代开始时间
            start_time = time.monotonic()

            # fix the temperature for sampling
            # 计算本代采样温度
            temperature = temp_scheduler(iteration=generation_id)
            # 日志：输出代数与温度
            self.logger.info(
                f"\n========= Generation {generation_id} | temperature: {round(temperature, 2)} ==========",
                LogEvent.SOLVER,
            )

            # 本代成功个体集合
            solution_nodes = []
            # 各个体的目标岛屿 ID
            island_ids = []
            # 各个体在本代中的编号
            counter_ids = []
            # 每代生成配置数量的个体
            for counter_id in range(self.cfg.individuals_per_generation):
                # 第 0 代：均匀随机分岛并全部起草
                if generation_id == 0:  # initially, uniformly populate the islands
                    # 均匀随机选岛
                    island_id = random.choice(range(solution_database.num_islands))
                    # 使用起草算子
                    create_node_fn = self._draft
                    # 无上下文示例
                    in_context_nodes = []
                else:  # gen_id > 0: start the evolutionary process
                    # 第 1 代起：按适应度加权采样岛屿与父代个体（温度与交叉概率随代数变化）
                    in_context_nodes, island_id, operator = solution_database.sample_in_context(
                        self.cfg.few_shot,
                        temperature,
                        (0 if generation_id < self.cfg.num_generations_till_crossover else self.cfg.crossover_prob),
                    )  # weighted sampling of islands and corresponding individuals
                    # 改进算子：单父代
                    if operator == "improve":
                        # 绑定改进函数
                        create_node_fn = self._improve
                    # 起草：重新随机选岛
                    elif operator == "draft":
                        # 均匀随机选岛
                        island_id = random.choice(range(solution_database.num_islands))
                        # 绑定起草函数
                        create_node_fn = self._draft
                        # 无上下文示例
                        in_context_nodes = []
                    else:
                        # 交叉算子：两个父代
                        create_node_fn = self._crossover

                # 记录该个体的目标岛
                island_ids.append(island_id)

                # 日志：正在为该个体创建节点
                self.logger.info(
                    f"Creating node for individual {counter_id} in generation {generation_id}", LogEvent.SOLVER
                )

                # 生成子代节点（改进/起草不传父代参数，交叉自动展开两个父代）
                child_node = create_node_fn(*in_context_nodes)
                # 执行子代代码并取得评估结果
                state, eval_result = task.step_task(state, extract_code(child_node.code))
                # 解析评估结果回填节点
                self.parse_eval_result(child_node, eval_result)
                # if the node is buggy, we run a debug cycle
                # and add the fixed node to the generation
                # if the node is not buggy, we add it to the generation
                # 无 bug 的子代直接入日志与本代集合
                if not child_node.is_buggy:
                    # 写入搜索日志
                    self.journal.append(child_node)
                    # 记录 JOURNAL 事件
                    self.log_journal()
                    # 步数计数加一
                    self.state.current_step += 1
                    # 收集进本代个体
                    solution_nodes.append(child_node)
                    # 记录编号
                    counter_ids.append(counter_id)
                else:
                    # 日志：子代有 bug，进入调试循环
                    self.logger.info(f"Node {child_node.id} was buggy, entering debug cycle.", LogEvent.SOLVER)
                    # 尝试通过调试循环修复
                    state, debug_path, fixed_metric = self.debug_cycle(state, task, child_node)
                    # Add the debug path to the journal
                    # 调试路径上所有节点入日志
                    for n in debug_path:
                        # 逐个写入
                        self.journal.append(n)
                        # 记录 JOURNAL 事件
                        self.log_journal()
                        # 步数计数加一
                        self.state.current_step += 1
                    # 修复成功（得到有效指标）
                    if fixed_metric is not None:
                        # 取调试路径最后一个节点（修复版）
                        fixed_node = debug_path[-1]  # Get the last node (the fixed one)
                        # 日志：输出修复结果
                        self.logger.info(
                            f"Node {child_node.id} was fixed by node {fixed_node.id}, adding fixed node to generation.",
                            LogEvent.SOLVER,
                        )
                        # Add only the final fixed node to the generation
                        # 修复后的节点加入本代
                        solution_nodes.append(fixed_node)
                        # 记录编号
                        counter_ids.append(counter_id)
                    else:
                        # 日志：无法修复，丢弃该个体
                        self.logger.info(
                            f"Node {child_node.id} could not be fixed by debug cycle, discarding individual {counter_id}.",
                            LogEvent.SOLVER,
                        )

            # store individuals solutions only if it improves overall island fitness
            # for initialization, we don't use this step
            # 非初始代：按"是否改进岛屿"决定入岛（迁移概率也随代数从 0 起效）
            if generation_id > 0:
                # 新解批量加入岛屿并管理容量/迁移
                solution_database.add_nodes_to_islands(
                    solution_nodes,
                    island_ids,
                    (0 if generation_id < self.cfg.num_generations_till_migration else self.cfg.migration_prob),
                )
            else:  # initialization step (generation = 0)
                # 第 0 代：播种岛屿
                solution_database.seed_islands_with_nodes(
                    solution_nodes,
                    island_ids,
                )

            # 累计本代耗时到总运行时长
            self.state.running_time += time.monotonic() - start_time
            # 日志：输出本代耗时
            self.logger.info(
                f"Step {self.state.current_step} | Generation {self.state.current_generation}: Time taken for generation: {self.state.running_time:.3f} seconds"
            )

            # Update the state with the current generation
            # 代数计数加一
            self.state.current_generation += 1

            # 日志：保存检查点
            self.logger.info(f"Step {self.state.current_step}: Saving checkpoint")
            # 每代落盘一次
            self.save_checkpoint()

            # 达到时间或步数上限则停止
            if self.state.running_time >= self.cfg.time_limit_secs or self.state.current_step >= self.cfg.step_limit:
                # 日志：达到上限，停止搜索
                self.logger.info("Maximum runtime reached, stopping search")
                # 退出代际循环
                break

        # 返回状态与最佳节点的代码
        return state, self.journal.get_best_node().code

    # 记录 JOURNAL 事件（当前步节点数据与最佳节点）
    def log_journal(self):
        # Get the current best node in the tree.
        # 取当前最佳节点
        best_node = self.journal.get_best_node()
        # 最佳节点的步号（无则为 0）
        best_node_step = 0 if best_node is None else best_node.step

        # 输出 JOURNAL 日志事件
        self.logger.log(
            self.journal.get_node_data(self.state.current_step) | {"current_best_node": best_node_step},
            "JOURNAL",
            step=self.state.current_step,
        )

    # 求解器主入口：准备数据预览后执行进化搜索并导出结果
    def __call__(self, task, state):
        # Possibly generate data preview first
        # 首次调用或尚无数据预览时先生成
        if not self.journal.nodes or self.data_preview is None:
            # 生成数据预览
            self.update_data_preview(state)

        # 执行进化搜索主循环
        state, solution = self.search(task, state)

        # Export results at the end of the search process
        # 导出搜索过程与结果到磁盘
        export_search_results(self.cfg, self.journal, self.logger, "EVO")

        # Get the best node for returning
        # 取最佳节点用于返回
        best_node = self.journal.get_best_node()

        # Return the best node
        # 若存在有效最佳节点
        if best_node:
            # 返回更新状态、最佳解代码与最佳节点
            return state, best_node.code, best_node
        else:
            # 日志：所有代结束后仍未找到可用解
            self.logger.info("No suitable code found after all generations.", LogEvent.SOLVER)
            # 返回空解
            return state, None, None
