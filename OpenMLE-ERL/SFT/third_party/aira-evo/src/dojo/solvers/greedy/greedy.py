# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are MIT-licensed
# Copyright (c) 2024 Weco AI Ltd
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/WecoAI/aideml/blob/main/LICENSE

# 导入 os：环境/路径相关操作（保留以备扩展）
import os
# 导入 random：搜索策略中的随机决策（如按概率触发调试）
import random
# 导入 partial：把 LLM 与配置预绑定生成算子函数
from functools import partial
# 导入 Path：检查点与文件路径处理
from pathlib import Path
# 导入类型注解
from typing import Any, Dict, Optional, Union
# 导入 json：检查点日志(Journal)的序列化/反序列化
import json

# 导入 hydra：注册配置入口（保留以备扩展）
import hydra
# 导入求解器基类，Greedy 继承自它
from dojo.core.solvers.base import Solver
# 导入分析算子：解读执行结果、判断是否有 bug 并抽取指标
from dojo.core.solvers.operators.analyze import analyze_op
# 导入算子执行封装：统一调用"规划+写码"类算子并解析其输出
from dojo.core.solvers.operators.core import execute_op_plan_code
# 导入调试算子：根据报错修复有 bug 的代码
from dojo.core.solvers.operators.debug import debug_op
# 导入起草算子：从零生成新方案
from dojo.core.solvers.operators.draft import draft_op
# 导入改进算子：在现有方案基础上优化
from dojo.core.solvers.operators.improve import improve_op
# 导入记忆算子工厂：为算子构建经验记忆(Memory)上下文
from dojo.core.solvers.operators.memory import create_memory_op
# 导入数据预览工具：生成数据集概要供 LLM 参考
from dojo.core.solvers.utils import data_preview
# 导入搜索日志(Journal)与节点(Node)：以树结构记录解的演化历史
from dojo.core.solvers.utils.journal import Journal, Node
# 导入指标值封装：正常指标与"最差指标"（无效解占位用）
from dojo.core.solvers.utils.metric import MetricValue, WorstMetricValue
# 导入代码抽取工具：从 LLM 响应中剥离出可执行代码
from dojo.core.solvers.utils.response import extract_code
# 导入复杂度级别判定工具（控制提示词中方案的目标复杂度）
from dojo.solvers.utils import get_complextiy_level
# 导入 JSON 解析工具：把 LLM 输出稳健地解析为字典
from dojo.utils.code_parsing import parse_json_output
# 导入搜索结果导出工具：把整棵搜索树与结果导出到磁盘
from dojo.core.solvers.utils.search_exporter import (
    export_search_results,
)
# 导入任务评估结果字典的键名常量
from dojo.core.tasks.constants import (
    EXECUTION_OUTPUT,
    TASK_DESCRIPTION,
    VALID_SOLUTION_FEEDBACK,
    VALIDATION_FITNESS,
    AUX_EVAL_INFO,
    VALID_SOLUTION,
)
# 导入 time：统计每步耗时
import time

# 导入通用 LLM 封装：按配置实例化不同用途的语言模型
from dojo.core.solvers.llm_helpers.generic_llm import GenericLLM

# 导入贪心求解器配置数据类
from dojo.config_dataclasses.solver.greedy import GreedySolverConfig
# 导入贪心求解器的状态类（当前步数、累计运行时长等）
from dojo.utils.state import GreedyState


# 贪心(Greedy)求解器：每步在"起草/调试/改进"三种动作间做局部最优选择的 LLM 搜索智能体
class Greedy(Solver):
    """Greedy solver."""

    # 构造函数：初始化搜索日志、任务描述、优化方向与各 LLM 算子
    def __init__(self, cfg: GreedySolverConfig, task_info):
        """
        Initialize the Greedy solver.

        Args:
            task_info: Dictionary containing task information including description and optimization direction
            **cfg: Configuration dictionary with solver parameters
        """
        # 调用基类构造，完成公共初始化
        super().__init__(cfg, task_info=task_info)
        # 创建搜索日志：按树结构记录所有生成的解及其指标
        self.journal = Journal()
        # 数据预览缓存：初始为空，首次迭代时生成
        self.data_preview: str | None = None

        # 保存任务描述（目标、约束、评估方式）
        self.task_desc = task_info[TASK_DESCRIPTION]
        # 读取优化方向：分数越低越好(如误差指标)还是越高越好(如准确率)
        self.lower_is_better = task_info.get("lower_is_better", None)

        # 优化方向必须由任务显式给出
        assert self.lower_is_better is not None

        # 初始化各 LLM 算子
        self.setup_operators()

        # 初始化求解器状态（当前步数、累计运行时长）
        self.state = GreedyState()

    # 保存检查点：在基类保存之外，额外把整棵搜索日志写为 JSONL 文件
    def save_checkpoint(self):
        # 先执行基类的检查点保存（状态与配置等）
        super().save_checkpoint()

        # Write the journal to a jsonl file
        # 把日志中的全部节点序列化为字典列表
        journal_sd = self.journal.node_list()
        # 拼出 journal.jsonl 的保存路径
        journal_path = Path(self.cfg.checkpoint_path) / "journal.jsonl"
        # 打开文件逐行写出每个节点
        with open(journal_path, "w") as f:
            # 遍历所有节点
            for node in journal_sd:
                # JSONL 格式：一行写一个节点的 JSON
                f.write(json.dumps(node) + "\n")
        # 日志：检查点已保存
        self.logger.info(f"Checkpoint saved to {journal_path}")

    # 加载检查点：恢复搜索日志，实现断点续跑
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
        # 由导出数据重建日志对象（恢复树结构与父子关系）
        self.journal = Journal.from_export_data({"nodes": journal_export})

    # 初始化并配置求解器使用的各 LLM 算子
    def setup_operators(self):
        """
        Initialize and configure the LLM operators used in the Greedy solver.

        This method instantiates the draft, improve, debug, and analyze LLMs
        and creates partial functions for each operator with the appropriate parameters.
        """

        # First we set up the LLMs
        # 按配置实例化"起草"算子专用的 LLM
        draft_llm = GenericLLM(self.cfg.operators["draft"])
        # 实例化"改进"算子的 LLM
        improve_llm = GenericLLM(self.cfg.operators["improve"])
        # 实例化"调试"算子的 LLM
        debug_llm = GenericLLM(self.cfg.operators["debug"])
        # 实例化"分析"算子的 LLM
        analyze_llm = GenericLLM(self.cfg.operators["analyze"])

        # Create the memory for operators
        # 构建经验记忆算子：为普通算子提供历史成败经验上下文
        self.memory_op = create_memory_op(self.cfg.memory)
        # 构建调试专用记忆算子：提供历史报错与修复经验
        self.debug_memory_op = create_memory_op(self.cfg.debug_memory)

        # Then we create the operators
        # 预绑定 LLM、配置与记忆，得到可直接调用的起草函数
        self.draft_fn = partial(draft_op, draft_llm, self.cfg, self.memory_op)
        # 预绑定参数得到改进函数
        self.improve_fn = partial(improve_op, improve_llm, self.cfg, self.memory_op)
        # 预绑定参数得到调试函数
        self.debug_fn = partial(debug_op, debug_llm, self.cfg, self.debug_memory_op)
        # 预绑定参数得到分析函数
        self.analyze_fn = partial(analyze_op, analyze_llm, self.cfg)

    # 创建搜索树的虚拟根节点（不是真实解，仅作为所有起草节点的父节点）
    def create_root_node(self):
        # 根节点：空代码、标记为 buggy、指标取"最差值"，保证排名时不会胜出
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
        # 记录 JOURNAL 事件：第 0 步的节点数据与当前最佳节点
        self.logger.log(
            self.journal.get_node_data(self.state.current_step) | {"current_best_node": 0},
            "JOURNAL",
            step=self.state.current_step,
        )
        # 步数计数加一
        self.state.current_step += 1

    # 求解器主入口：迭代执行搜索直至步数或时间上限，返回最佳解
    def __call__(self, task, state):
        """
        Run the Greedy solver for a specified number of iterations.

        Executes the search process for the configured number of steps, tracking the best
        solution found. After all iterations, exports search results and returns the best code.

        Args:
            task: The task object that provides evaluation capabilities
            state: The current solver state

        Returns:
            tuple: Updated state and the best code solution (or None if no valid solution found)
        """
        # 日志：开始贪心搜索
        self.logger.info("Starting Greedy search")

        # Create a blank root node to start.
        # 先创建空白根节点
        self.create_root_node()

        # Run the search
        # 逐步迭代：从当前步数（续跑时可能大于 0）迭代到步数上限
        for _ in range(self.state.current_step, self.cfg.step_limit):
            # 记录本步开始时间（单调时钟，不受系统时间调整影响）
            start_time = time.monotonic()
            # 执行一次完整的搜索迭代
            state, _ = self.step(task, state)
            # 累计总运行时长
            self.state.running_time += time.monotonic() - start_time
            # 日志：输出累计耗时
            self.logger.info(
                f"Step {self.state.current_step}: Time taken for step: {self.state.running_time:.3f} seconds"
            )

            # 步数计数加一
            self.state.current_step += 1

            # 日志：保存检查点
            self.logger.info(f"Step {self.state.current_step}: Saving checkpoint")
            # 每步落盘一次，支持中断续跑
            self.save_checkpoint()

            # 若累计运行时间达到时间上限
            if self.state.running_time >= self.cfg.time_limit_secs:
                # 日志：时间到，停止搜索
                self.logger.info("Maximum runtime reached, stopping search")
                # 提前退出迭代
                break

        # Get the best node
        # 取整棵搜索树中的最佳节点
        best_node = self.journal.get_best_node()

        # Export the search results
        # 导出搜索过程与结果（树、指标等）到磁盘
        try:
            export_search_results(self.cfg, self.journal, self.logger, "Greedy")
        except Exception as e:
            # 导出失败仅记录错误，不影响主流程
            self.logger.error(f"Error exporting search results: {e}")

        # Return the best node
        # 若存在有效最佳节点
        if best_node:
            # 返回更新状态、最佳解代码与最佳节点
            return state, best_node.code, best_node
        else:
            # 日志：迭代结束仍未找到可用代码
            self.logger.info("No suitable code found after all iterations.")
            # 返回空解
            return state, None, None

    # 搜索策略：决定本步"对哪个节点做什么"——探索(Exploration)起草/调试，利用(Exploitation)改进最优解
    def search_policy(self) -> Node | None:
        """
        Determine the next node to work on based on the current state of the journal.

        This node selection policy determines whether to:
        - Draft a new solution (returns None)
        - Debug a buggy node (returns a buggy node)
        - Improve the best node (returns the best performing node)

        The decision is based on:
        1. Number of existing drafts compared to required drafts
        2. Random probability for debugging
        3. Existence of good (non-buggy) nodes to improve

        Returns:
            Node | None: Selected node to work on, or None to indicate a new draft should be created
        """
        # If not enough drafts exist, return None -> draft a new solution.
        # 起草数不足配置值：先补足初始解数量
        if len(self.journal.draft_nodes) < self.cfg.num_drafts:
            # 日志：选择起草新节点（附当前起草数进度）
            self.logger.info(
                f"Search Policy: Drafting a new node (not enough drafts - {len(self.journal.draft_nodes)}/{self.cfg.num_drafts})"
            )
            # 返回 None 表示"起草新方案"
            return None

        # With probability debug_prob, try to debug a buggy node.
        # 以 debug_prob 概率触发调试分支（随机探索有 bug 的节点）
        if random.random() < self.cfg.debug_prob:
            # nodes that are buggy + leaf nodes + debug depth < max debug depth
            # 筛选可调试节点：有 bug、是叶子节点、且调试深度未超上限
            debuggable_nodes = [
                n for n in self.journal.buggy_nodes if (n.is_leaf and n.debug_depth <= self.cfg.max_debug_depth)
            ]
            # 若存在可调试节点
            if debuggable_nodes:
                # 日志：选择调试有 bug 的节点
                self.logger.info("Search Policy: Debugging a buggy node")
                # 随机挑一个待调试节点返回
                return random.choice(debuggable_nodes)
            # 调试日志：随机未命中，跳过调试
            self.logger.debug("Search Policy: Not debugging (by random chance)")

        # If no good nodes exist, return None -> draft a new solution.
        # 取所有无 bug 的有效节点
        good_nodes = self.journal.good_nodes
        # 若一个有效节点都没有
        if not good_nodes:
            # 日志：无有效节点可改进，只能再起草
            self.logger.info("Search Policy: Drafting a new node (no good nodes exist)")
            # 返回 None 表示起草
            return None

        # Otherwise, pick the best node for improvement
        # 日志：选择最佳节点进行改进
        self.logger.info("Search Policy: Selecting best node for improvement")
        # 利用(Exploitation)：返回当前最佳节点
        return self.journal.get_best_node()

    # 起草(Draft)算子封装：调用 draft LLM 从零生成新方案并打包为节点
    def _draft(self) -> Node:
        """
        Generate a new solution from scratch using the draft LLM operator.

        Uses the draft operator to create a new code solution based on the task description.
        The resulting code is packaged into a new Node object with relevant metadata.

        Returns:
            Node: A new node containing the drafted solution
        """
        # 日志：开始起草新方案
        self.logger.info(f"Step {self.state.current_step}: Starting to drafting new solution")
        # 执行起草算子：传入任务描述、日志、剩余时间、数据预览与目标复杂度级别，得到计划/代码/指标
        plan, code, metrics = execute_op_plan_code(
            self.draft_fn,
            self.task_desc,
            self.journal,
            self.state.current_step,
            self.cfg.time_limit_secs - self.state.running_time,
            self.data_preview,
            get_complextiy_level(num=len(self.journal.draft_nodes)) if self.cfg.use_complexity else None,
            self.root_node,
            max_operator_tries=self.cfg.max_llm_call_retries,
        )
        # 将计划与代码封装为新节点：父节点为根节点，并记录所用算子与调用指标
        node = Node(
            plan=plan, code=code, operators_used=["draft"], operators_metrics=[metrics], parents=[self.root_node]
        )
        # 日志：起草节点创建及其指标
        self.logger.info(f"Draft Node Created - Metrics: {metrics}")
        # 日志：起草得到的代码
        self.logger.info(f"Draft Code: {code}")
        # 日志：起草结束
        self.logger.info(f"Step {self.state.current_step}: End of drafting new solution")
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
        # 日志：开始改进现有方案
        self.logger.info(f"Step {self.state.current_step}: Starting to improve existing solution")
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
        # 日志：改进结束
        self.logger.info(f"Step {self.state.current_step}: End of improving existing solution")
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
        # 日志：开始调试有 bug 的方案
        self.logger.info(f"Step {self.state.current_step}: Starting to debug buggy solution")
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
        # 日志：调试结束
        self.logger.info(f"Step {self.state.current_step}: End of debugging buggy solution")
        # 返回修复后的新节点
        return node

    # 分析(Analyze)算子封装：解读节点执行结果，判断是否有 bug 并抽取指标
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
        # 记录分析算子的调用指标（如 token 数、耗时）
        node.operators_metrics.append(metrics)
        # 日志：分析完成及其指标
        self.logger.info(f"Node Analysis Performed - Metrics: {metrics}")
        # 日志：分析结束
        self.logger.info(f"Step {self.state.current_step}: End of analyzing solution")
        # 返回分析结果（后续会解析为 JSON 字典）
        return analysis

    # 生成数据预览：为 LLM 提供数据集概要（表头、形状等）作为上下文
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

        # 调试日志：开始生成预览
        self.logger.debug("Generating data preview")
        # 本地解释器：直接在本地工作目录生成预览
        if state["solver_interpreter"].local:
            # 读取工作目录中的数据文件并生成概要文本
            self.data_preview = data_preview.generate(state["solver_interpreter"].working_dir)
        else:
            # 非本地解释器：把预览生成脚本源码注入远程解释器执行
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

    # 单次迭代核心：选节点 -> 生成/修复代码 -> 执行评估 -> 记入搜索日志
    def step(self, task, state):
        """
        Execute a single iteration of the Greedy solver process.

        This method implements the core Greedy algorithm:
        1. Select a node to work on (draft/debug/improve)
        2. Apply the appropriate operator to generate new code
        3. Evaluate the code using the task
        4. Parse the results and update the journal

        Args:
            task: The task object that provides evaluation capabilities
            state: The current solver state

        Returns:
            tuple: Updated state and evaluation results
        """
        # 日志：迭代开始
        self.logger.info(f"Step {self.state.current_step}: Starting iteration")

        # Possibly generate data preview first
        # 首次迭代（日志为空）或尚无数据预览时先生成
        if not self.journal.nodes or self.data_preview is None:
            # 生成数据预览
            self.update_data_preview(state)

        # Select the parent node
        # 依搜索策略选出本步要处理的父节点
        parent_node = self.search_policy()
        # 调试日志：输出所选节点
        self.logger.debug(f"Step {self.state.current_step}: Selected parent node: {parent_node}")

        # If no parent node is selected, draft a new solution.
        # Otherwise, if the parent node is buggy, debug it.
        # Otherwise, improve the parent node.
        # 无父节点：起草新方案
        if parent_node is None:
            # 调用起草算子
            result_node = self._draft()
        # 父节点有 bug：走调试分支
        elif parent_node.is_buggy:
            # 调用调试算子
            result_node = self._debug(parent_node)
        else:
            # 否则调用改进算子
            result_node = self._improve(parent_node)

        # Evaluate the code
        # 调试日志：开始执行生成的代码
        self.logger.debug(f"Step {self.state.current_step}: Executing generated code")
        # 剥离出纯代码并交给任务执行，得到评估结果
        state, eval_result = task.step_task(state, extract_code(result_node.code))

        # Update running time
        # self.state.running_time += eval_result[EXECUTION_OUTPUT].exec_time

        # Parse the evaluation results
        # 解析评估结果：抽取指标、判定 bug 并回填节点
        self.parse_eval_result(node=result_node, eval_result=eval_result)

        # Store in the journal
        # 新节点写入搜索日志
        self.journal.append(result_node)

        # Log the best node
        # 取当前最佳节点
        best_node = self.journal.get_best_node()
        # 记录最佳节点的步号（无最佳节点则为 0）
        best_node_step = 0 if best_node is None else best_node.step

        # Log the latest node
        # 记录 JOURNAL 事件：本步节点数据与当前最佳节点
        self.logger.log(
            self.journal.get_node_data(self.state.current_step) | {"current_best_node": best_node_step},
            "JOURNAL",
            step=self.state.current_step,
        )

        # Log state
        # 记录 STATE 事件：求解器状态快照
        self.logger.log(
            self.state.state_dict(),
            "STATE",
            step=self.state.current_step,
        )

        # 日志：迭代完成
        self.logger.info(f"Step {self.state.current_step}: Iteration complete")
        # 返回更新后的状态与评估结果
        return state, eval_result

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
        # 保存分析摘要到节点
        node.analysis = response["summary"]

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
            self.logger.info(f"Using test score: {test_score}")

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
            # 格式化反馈文本（不改动字符串本身，仅在其上加注释）
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
