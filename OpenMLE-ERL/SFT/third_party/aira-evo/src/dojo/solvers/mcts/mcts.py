# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 math：UCT 公式中的对数与开方计算
import math
# 导入 partial：预绑定 LLM 与配置生成算子函数
from functools import partial
# 导入 Path：检查点路径处理
from pathlib import Path
# 导入 time：统计每步耗时
import time
# 导入类型注解
from typing import Any, Dict, List, Optional, Tuple, Union
# 导入 json：检查点日志(Journal)的序列化/反序列化
import json

# 导入 hydra：注册配置入口（保留以备扩展）
import hydra
# 导入求解器基类，MCTS 继承自它
from dojo.core.solvers.base import Solver
# 导入分析算子：解读执行结果、判断 bug 并抽取指标
from dojo.core.solvers.operators.analyze import analyze_op
# 导入算子执行封装：统一调用"规划+写码"类算子并解析输出
from dojo.core.solvers.operators.core import execute_op_plan_code
# 导入调试算子：修复有 bug 的代码
from dojo.core.solvers.operators.debug import debug_op
# 导入起草算子：从零生成新方案
from dojo.core.solvers.operators.draft import draft_op
# 导入改进算子：优化现有方案
from dojo.core.solvers.operators.improve import improve_op
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
# 导入复杂度级别判定工具
from dojo.solvers.utils import get_complextiy_level
# 导入通用 LLM 封装：按配置实例化不同用途的语言模型
from dojo.core.solvers.llm_helpers.generic_llm import GenericLLM
# 导入 JSON 解析工具：稳健解析 LLM 输出
from dojo.utils.code_parsing import parse_json_output
# 导入搜索结果导出工具
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
# 导入 MCTS 求解器配置数据类
from dojo.config_dataclasses.solver.mcts import MCTSSolverConfig
# 导入 MCTS 求解器状态类（当前步数、累计运行时长等）
from dojo.utils.state import MCTSState


# 将 Q 值线性归一化到 [0,1] 区间（UCT 中"利用"项的标准化）
def normalise_q_value(q_value: float, global_max_q_val: float, global_min_q_val: float) -> float:
    # Avoid division by zero in case global_max and global_min are equal.
    # 最大最小值相等时无法归一化（避免除零）
    if global_max_q_val == global_min_q_val:
        # 返回 [0,1] 内的中性值 0.5
        return 0.5  # Neutral value in the [0,1] range.
    # 线性归一化：(q-min)/(max-min)
    return (q_value - global_min_q_val) / (global_max_q_val - global_min_q_val)


# UCT 公式：归一化 Q 值(利用项) + 探索常数 × 探索项；UCB1 应用于树搜索即 UCT（论文中的经典节点选择策略）
def uct_value(
    q_value: float,
    explore_count: int,
    parent_explore_count: int,
    uct_c: float,
    global_max_q_val: float,
    global_min_q_val: float,
):
    # 从未被访问过的节点没有统计信息
    if explore_count == 0:
        # 返回极低哨兵值，使其不会被选中
        return -1e8

    # 计算归一化的利用项
    norm_q = normalise_q_value(q_value, global_max_q_val, global_min_q_val)

    # 探索项：sqrt(ln(父访问次数)/本节点访问次数)，访问越少奖励越大
    exploration = math.sqrt(math.log(parent_explore_count) / explore_count)
    # 返回 利用项 + 探索常数 × 探索项
    return norm_q + uct_c * exploration


# MCTS 节点：在普通节点上增加访问计数与累计价值（对应 MCTS 的 N/W 统计量）
class MCTSNode(Node):
    # 访问次数 N（该节点被回传更新过的次数）
    explore_count: int = 0
    # 累计价值 W（所有回传值之和）
    node_value: float = 0

    # Q 值：平均价值 W/N，是 UCT"利用"项的原始值
    def q_value(self, lower_is_better: bool = False):
        # 未访问过时没有 Q 值
        if self.explore_count == 0:
            # 返回极低哨兵值
            return -1e8

        # 计算平均价值 W/N
        q_value = self.node_value / self.explore_count

        # 若优化方向是"越低越好"，取负号统一为越大越好
        if lower_is_better:
            # 符号翻转
            q_value = -q_value

        # 返回 Q 值
        return q_value

    # 设置节点的累计价值
    def set_value(self, value: float) -> None:
        """Set the node's value."""
        # 直接赋值
        self.node_value = value

    # 累加节点价值：回传(Backpropagation)阶段调用
    def add_value(self, value: float) -> None:
        """
        Add to the node's cumulative value.

        Args:
            value: Value to add to node's cumulative value
        """
        # 空值不计入统计
        if value is not None:
            # 累加到节点价值
            self.node_value += value

    # 访问计数加一
    def increment_explore_count(self):
        # 自增 N
        self.explore_count += 1

    # 输出 MCTS 特有的日志指标
    def extra_metrics_to_log(self):
        # 未访问时平均值无意义
        if self.explore_count == 0:
            # 置空
            v_div_ec = None
        else:
            # 平均价值 W/N
            v_div_ec = self.node_value / self.explore_count
        # 返回节点价值、访问次数与平均值
        return {"node_value": self.node_value, "explore_count": self.explore_count, "v_div_ec": v_div_ec}


# 蒙特卡洛树搜索(MCTS)求解器：以 UCT 选择节点、LLM 算子扩展子节点、指标回传的树搜索
class MCTS(Solver):
    # 构造函数：初始化日志、任务描述、算子与全局 Q 值范围
    def __init__(self, cfg: MCTSSolverConfig, task_info):
        """
        Initialize the MCTS solver.

        Args:
            task_info: Dictionary containing task information including description and optimization direction
            **cfg: Configuration dictionary with solver parameters
        """
        # 调用基类构造，完成公共初始化
        super().__init__(cfg, task_info=task_info)
        # 创建搜索日志：按树结构记录所有生成的解及其指标
        self.journal = Journal()
        # 数据预览缓存：初始为空
        self.data_preview: str | None = None

        # 保存任务描述（目标、约束、评估方式）
        self.task_desc = task_info[TASK_DESCRIPTION]
        # 读取优化方向：分数越低越好与否
        self.lower_is_better = task_info.get("lower_is_better", None)

        # 优化方向必须由任务显式给出
        assert self.lower_is_better is not None

        # 初始化求解器状态
        self.state = MCTSState()

        # 初始化各 LLM 算子
        self.setup_operators()

        # 全局 Q 值上界（用于归一化），初始为负哨兵值
        self.global_max_q_val = -1e8
        # 全局 Q 值下界，初始为正哨兵值
        self.global_min_q_val = 1e8

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

    # 初始化并配置各 LLM 算子
    def setup_operators(self):
        """
        Initialize and configure the LLM operators used in the MCTS solver.

        This method instantiates the draft, improve, debug, and analyze LLMs
        and creates partial functions for each operator with the appropriate parameters.
        """

        # First we set up the LLMs
        # 按配置实例化"起草"算子的 LLM
        draft_llm = GenericLLM(self.cfg.operators["draft"])
        # 实例化"改进"算子的 LLM
        improve_llm = GenericLLM(self.cfg.operators["improve"])
        # 实例化"调试"算子的 LLM
        debug_llm = GenericLLM(self.cfg.operators["debug"])
        # 实例化"分析"算子的 LLM
        analyze_llm = GenericLLM(self.cfg.operators["analyze"])

        # Create the memory for operators
        # 构建经验记忆算子：提供历史成败经验上下文
        self.memory_op = create_memory_op(self.cfg.memory)
        # 构建调试专用记忆算子：提供历史报错与修复经验
        self.debug_memory_op = create_memory_op(self.cfg.debug_memory)

        # Then we create the operators
        # 预绑定 LLM、配置与记忆得到起草函数
        self.draft_fn = partial(draft_op, draft_llm, self.cfg, self.memory_op)
        # 预绑定参数得到改进函数
        self.improve_fn = partial(improve_op, improve_llm, self.cfg, self.memory_op)
        # 预绑定参数得到调试函数
        self.debug_fn = partial(debug_op, debug_llm, self.cfg, self.debug_memory_op)
        # 预绑定参数得到分析函数
        self.analyze_fn = partial(analyze_op, analyze_llm, self.cfg)

    # 创建搜索树的虚拟根节点（不是真实解，仅作为树的起点）
    def create_root_node(self):
        # 根节点：空代码、标记 buggy、指标取"最差值"
        self.root_node = MCTSNode(
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

    # 剩余步数属性
    @property
    def remaining_steps(self):
        # 步数上限减去当前步数
        return self.cfg.step_limit - self.state.current_step

    # 求解器主入口：循环执行 MCTS 步骤直至步数或时间上限
    def __call__(self, task, state):
        """
        Run the MCTS solver for a specified number of iterations.

        Executes the search process for the configured number of steps, tracking the best
        solution found. After all iterations, exports search results and returns the best code.

        Args:
            task: The task object that provides evaluation capabilities
            state: The current solver state

        Returns:
            tuple: Updated state and the best code solution (or None if no valid solution found)
        """
        # 日志：开始 MCTS 搜索
        self.logger.info("Starting MCTS search")

        # Create a blank root node to start.
        # 先创建空白根节点
        self.create_root_node()

        # Run the search
        # 主循环：未达步数上限就持续搜索
        while self.state.current_step <= self.cfg.step_limit:
            # 记录本步开始时间
            start_time = time.monotonic()
            # 执行一次 MCTS 迭代（选择 -> 扩展 -> 评估 -> 回传）
            state = self.step(task, state)
            # 累计总运行时长
            self.state.running_time += time.monotonic() - start_time
            # 日志：输出累计耗时
            self.logger.info(
                f"Step {self.state.current_step}: Time taken for step: {self.state.running_time:.3f} seconds"
            )

            # 日志：保存检查点
            self.logger.info(f"Step {self.state.current_step}: Saving checkpoint")
            # 每步落盘一次
            self.save_checkpoint()

            # 若累计运行时间达到时间上限
            if self.state.running_time >= self.cfg.time_limit_secs:
                # 日志：时间到，停止搜索
                self.logger.info("Maximum runtime reached, stopping search")
                # 退出主循环
                break

        # Get the best node
        # 取整棵搜索树中的最佳节点
        best_node = self.journal.get_best_node()

        # Export the search results
        # 导出搜索过程与结果到磁盘
        try:
            export_search_results(self.cfg, self.journal, self.logger, "MCTS")
        except Exception as e:
            # 导出失败仅记录错误
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

    # 选择(Selection)阶段：自根向下按 UCT 最大值走至叶子节点
    def search_policy(self, root_node: MCTSNode) -> List[MCTSNode]:
        """
        Traverse the tree from root to leaf using UCT selection.

        Args:
            root_node: Starting node for path selection

        Returns:
            List of nodes representing path from root to selected leaf
        """
        # 记录从根到叶的路径
        path: List[MCTSNode] = []
        # 从根节点出发
        current_node = root_node

        # 逐层下行直至叶子
        while True:
            # 当前节点加入路径
            path.append(current_node)
            # 无子节点说明到达叶子
            if not current_node.children:
                # It's a leaf
                # 返回完整路径
                return path

            # 选 UCT 值最大的子节点继续下行（利用+探索的折中）
            current_node = max(
                current_node.children,
                key=lambda c: uct_value(
                    q_value=c.q_value(self.lower_is_better),
                    explore_count=c.explore_count,
                    parent_explore_count=current_node.explore_count,
                    uct_c=self.cfg.uct_c,
                    global_max_q_val=self.global_max_q_val,
                    global_min_q_val=self.global_min_q_val,
                ),
            )

    # 起草(Draft)算子封装：从父节点（通常为根）生成全新方案
    def _draft(self, parent: Optional[MCTSNode] = None) -> MCTSNode:
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
            get_complextiy_level(parent) if self.cfg.use_complexity else None,
            self.root_node,
            max_operator_tries=self.cfg.max_llm_call_retries,
        )
        # 将计划与代码封装为 MCTS 节点，父节点指向传入节点
        node = MCTSNode(plan=plan, code=code, parents=[parent], operators_used=["draft"], operators_metrics=[metrics])
        # 日志：起草节点创建及其指标
        self.logger.info(f"Draft Node Created - Metrics: {metrics}")
        # 日志：起草得到的代码
        self.logger.info(f"Draft Code: {code}")
        # 返回新节点
        return node

    # 改进(Improve)算子封装：在父节点方案基础上优化并生成新节点
    def _improve(self, parent_node: MCTSNode) -> MCTSNode:
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
        # 封装为新 MCTS 节点，父节点指向被改进节点
        node = MCTSNode(
            plan=plan, code=code, parents=[parent_node], operators_used=["improve"], operators_metrics=[metrics]
        )
        # 日志：改进节点创建及其指标
        self.logger.info(f"Improve Node Created - Metrics: {metrics}")
        # 日志：改进后的代码
        self.logger.info(f"Improve Code: {code}")
        # 返回新节点
        return node

    # 调试(Debug)算子封装：根据父节点的报错信息修复代码
    def _debug(self, parent_node: MCTSNode) -> MCTSNode:
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
        # 封装为新 MCTS 节点，父节点指向被调试节点
        node = MCTSNode(
            plan=plan, code=code, parents=[parent_node], operators_used=["debug"], operators_metrics=[metrics]
        )
        # 日志：调试节点创建及其指标
        self.logger.info(f"Debug Node Created - Metrics: {metrics}")
        # 日志：调试后的代码
        self.logger.info(f"Debug Code: {code}")
        # 返回修复后的新节点
        return node

    # 分析(Analyze)算子封装：解读节点执行结果，判断 bug 并抽取指标
    def _analyze(self, node: MCTSNode) -> Union[str, dict]:
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

    # 生成数据预览：为 LLM 提供数据集概要作为上下文
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

    # 单次 MCTS 迭代：UCT 选择路径，扩展叶子并回传价值
    def step(self, task, state):
        """
        Execute a single iteration of the MCTS solver process.

        This method implements the core MCTS algorithm:
        1. Select a node to work on using UCT.
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
        # 首次迭代或尚无数据预览时先生成
        if not self.journal.nodes or self.data_preview is None:
            # 生成数据预览
            self.update_data_preview(state)

        # 1) Selection: Find a path from root → leaf using UCT
        # 选择阶段：用 UCT 从根走到叶
        path = self.search_policy(self.root_node)

        # 2) Expansion and Backprop
        # 扩展+回传阶段：生成子节点、评估并回传
        state = self._expand_leaf_and_backprop(path, state, task)

        # 返回更新后的状态
        return state

    # 回传(Backpropagation)一步：沿路径累加价值并增加访问计数
    def _backprop_step(self, path: List[MCTSNode], value_estimate: float):
        """
        Backpropagates the reward/score up the path (increasing explore_count
        and adjusting the metrics).
        """
        # 遍历路径上的每个节点
        for node in path:
            # 访问计数 N 加一
            node.increment_explore_count()
            # 累计价值 W 加上本次估计值
            node.add_value(value_estimate)

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

    # 维护全局 Q 值范围，供 UCT 归一化使用
    def set_global_q_values(self, new_metric_value):
        # 空值不更新范围
        if new_metric_value is not None:
            # 更新全局上界
            self.global_max_q_val = max(self.global_max_q_val, new_metric_value)
            # 更新全局下界
            self.global_min_q_val = min(self.global_min_q_val, new_metric_value)

    # 扩展(Expansion)与回传：为路径末端叶子生成若干子节点，评估并回传价值
    def _expand_leaf_and_backprop(self, path: List[MCTSNode], state: Any, task: Any) -> Tuple[Any, int]:
        """
        Expand a leaf node and backpropagate results.

        Args:
            path: List of nodes from root to leaf
            state: Current state object
            task: Task object for evaluation

        Returns:
            Tuple of (updated state, number of trials performed)
        """
        # 取路径末端的叶子节点
        leaf_node = path[-1]
        # 本轮要生成的子节点数：受配置与剩余步数双重限制
        num_children_to_create = min(self.cfg.num_children, self.remaining_steps)
        # 逐个生成并评估子节点
        for _ in range(num_children_to_create):
            # If root node, we draft otherwise we improve
            # 叶子为根节点（无父节点）：起草全新方案
            if not leaf_node.parents:
                # 调用起草算子
                child_node = self._draft(leaf_node)
            else:
                # 否则在叶子上做改进
                child_node = self._improve(leaf_node)

            # Evaluate the code
            # 调试日志：开始执行生成的代码
            self.logger.debug(f"Step {self.state.current_step}: Executing generated code")
            # 剥离纯代码并交给任务执行，得到评估结果
            state, eval_result = task.step_task(state, extract_code(child_node.code))
            # 解析评估结果回填子节点
            self.parse_eval_result(node=child_node, eval_result=eval_result)

            # Add the child to the journal
            # 子节点写入搜索日志
            self.journal.append(child_node)
            # 记录 JOURNAL 事件
            self.log_journal()
            # 步数计数加一
            self.state.current_step += 1

            # If the child node is not buggy, we backpropagate
            # 子节点无 bug：直接回传其指标
            if not child_node.is_buggy:
                # 沿"路径+子节点"回传价值
                self._backprop_step(path=path + [child_node], value_estimate=child_node.metric.value)
                # 更新全局 Q 值范围
                self.set_global_q_values(child_node.metric.value)
            else:
                # Execute debug cycle
                # 有 bug：进入调试循环尝试修复
                state, debug_path, fixed_metric = self.debug_cycle(state, task, child_node)
                # We now exclude the child node from the path and backprop the fixed metric up the rest of the tree
                # 若修复得到了有效指标
                if fixed_metric is not None:
                    # 沿"路径+调试路径"回传修复后指标
                    self._backprop_step(path=path + debug_path, value_estimate=fixed_metric)
                    # 更新全局 Q 值范围
                    self.set_global_q_values(fixed_metric)

            # If we have used up all the steps we break
            # 步数用尽则停止本轮扩展
            if self.state.current_step > self.cfg.step_limit:
                # 日志：达到步数上限
                self.logger.info(f"Step limit reached: {self.state.current_step} steps")
                # 跳出循环
                break

        # 返回更新后的状态
        return state

    # 调试循环：迭代调用调试算子修复有 bug 节点，直至修复成功或超限
    def debug_cycle(self, state, task, buggy_node: MCTSNode):
        # 调试路径以初始 bug 节点为起点
        debug_path = [buggy_node]
        # 本次最多调试轮数：受剩余步数与配置上限双重限制
        debug_depth = min(self.remaining_steps, self.cfg.max_debug_depth)
        # 修复后的指标（未修复则保持 None）
        fixed_metric = None
        # We set the initial debug cycle time to the execution time of the first buggy node
        # 调试计时起点：计入首个 bug 节点的执行耗时
        total_debug_cycle_time = buggy_node.exec_time
        # We run the debug cycle for a number of times
        # or until time runs out, whichever comes first
        # 逐轮调试
        for _ in range(debug_depth):
            # 调用调试算子生成修复尝试
            buggy_node = self._debug(buggy_node)
            # 执行修复后的代码并取得评估结果
            state, eval_result = task.step_task(state, extract_code(buggy_node.code))
            # 解析评估结果回填节点
            self.parse_eval_result(node=buggy_node, eval_result=eval_result)
            # 节点写入搜索日志
            self.journal.append(buggy_node)
            # 记录 JOURNAL 事件
            self.log_journal()
            # 步数计数加一
            self.state.current_step += 1
            # 加入调试路径
            debug_path.append(buggy_node)
            # Break if we have a fixed metric - i.e. the solution is no longer buggy
            # 指标有效说明修复成功
            if buggy_node.metric.value is not None:
                # 记录修复后指标
                fixed_metric = buggy_node.metric.value
                # 结束调试循环
                break
            # 步数用尽则结束
            elif self.state.current_step > self.cfg.step_limit:
                # 日志：达到步数上限
                self.logger.info(f"Step limit reached: {self.state.current_step} steps")
                # 结束调试
                break
            # or if the debug depth is reached we break
            # 调试深度达上限
            if buggy_node.debug_depth >= self.cfg.max_debug_depth:
                # 结束调试
                break
            # or if the debug time is reached we break
            # 累计调试耗时
            total_debug_cycle_time += buggy_node.exec_time
            # 调试时间超限
            if total_debug_cycle_time >= self.cfg.max_debug_time:
                # 日志：调试超时
                self.logger.info(f"Debug cycle time exceeded: {total_debug_cycle_time} seconds")
                # 结束调试
                break

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
