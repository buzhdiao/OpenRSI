# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are MIT-licensed
# Copyright (c) 2024 Weco AI Ltd
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/WecoAI/aideml/blob/main/LICENSE

"""
The journal is the core datastructure in Greedy that contains:
- the generated code samples
- information how code samples relate to each other (the tree structure)
- code execution results
- evaluation information such as metrics
...
"""

# 导入 time 模块，用于记录节点创建时间
import time
# 导入 uuid 模块，用于生成节点唯一 ID
import uuid
# 导入 dataclass/field，用于声明数据类字段
from dataclasses import dataclass, field
# 导入 total_ordering，只需实现 __eq__/__gt__ 即可自动补全全部比较运算
from functools import total_ordering
# 导入类型注解工具
from typing import Any, Dict, List, Literal, Optional

# 导入 DataClassJsonMixin，让数据类支持 JSON 序列化/反序列化
from dataclasses_json import DataClassJsonMixin
# 导入 OmegaConf（保留以维持接口一致性）
from omegaconf import OmegaConf

# 导入执行结果数据类（终端输出、耗时、退出码）
from dojo.core.interpreters.base import ExecutionResult
# 导入指标值类型：普通指标值与"最差指标"占位值
from dojo.core.solvers.utils.metric import MetricValue, WorstMetricValue
# 导入长字符串截断工具
from dojo.core.solvers.utils.response import trim_long_string

# 导入 logging 模块
import logging

# 获取本模块专属日志器
log = logging.getLogger(__name__)


@total_ordering
@dataclass
class Node(DataClassJsonMixin):
    """A single node in the solution tree. Contains code, execution results, and evaluation information."""

    # ---- code & plan ----
    # 节点保存的代码；compare=False 表示不参与节点间的相等/排序比较
    code: str = field(compare=False)
    # 节点对应的方案说明文本
    plan: str = field(default=None, kw_only=True, compare=False)  # type: ignore

    # ---- general attrs ----
    # 节点在搜索过程中的步数序号
    step: int = field(default=None, kw_only=True, compare=False)  # type: ignore
    # 节点唯一 ID：默认用 uuid4 的十六进制串
    id: str = field(default_factory=lambda: uuid.uuid4().hex, kw_only=True, compare=False)
    # 节点创建时间戳
    ctime: float = field(default_factory=lambda: time.time(), kw_only=True, compare=False)
    # 父节点列表（多亲支持）
    parents: List["Node"] = field(default_factory=list, kw_only=True, compare=False)
    # 子节点集合（用 set 便于去重与 O(1) 增删）
    children: set["Node"] = field(default_factory=set, kw_only=True, compare=False)

    # ---- operator info ----
    # 生成该节点所用的算子名称列表
    operators_used: List[str] = field(default_factory=list, kw_only=True, compare=False)
    # 各算子对应的指标记录
    operators_metrics: List[Dict[str, Any]] = field(default_factory=list, kw_only=True, compare=False)

    # ---- execution info ----
    # 代码执行的原始终端输出（列表形式）
    _term_out: list[str] = field(default=None, kw_only=True, compare=False)  # type: ignore
    # 代码执行耗时（秒）
    exec_time: float = field(default=None, kw_only=True, compare=False)  # type: ignore
    # 代码执行退出码（None 表示未执行）
    exit_code: int | None = field(default=None, kw_only=True, compare=False)

    # ---- evaluation ----
    # post-execution result analysis (findings/feedback)
    # 执行结果的分析文本（LLM 分析算子的产出）
    analysis: str = field(default=None, kw_only=True, compare=False)  # type: ignore
    # 节点的评估指标值（参与比较，故不加 compare=False）
    metric: MetricValue = field(default=None, kw_only=True)  # type: ignore
    # whether the agent decided that the code is buggy
    # -> always True if exc_type is not None or no valid metric
    # 智能体判定该节点代码是否有 bug
    is_buggy: bool = field(default=None, kw_only=True, compare=False)  # type: ignore

    def __post_init__(self) -> None:
        # 数据类初始化后的校验逻辑
        if self.parents is None:
            # 无父节点（根节点）时无需校验
            return

        # 逐个检查父节点引用
        for parent in self.parents:
            if parent is None:
                # 父节点为 None 属于非法状态，尽早报错
                raise ValueError("Parent node is None")

    def attach_to_parents(self) -> None:
        # 把本节点登记到各父节点的 children 集合中，建立树结构链接
        if not self.parents:
            # 无父节点（根节点）则无需挂接
            return

        # 遍历所有父节点
        for parent in self.parents:
            # 把自己加入父节点的子节点集合（set 自动去重）
            parent.children.add(self)

    def remove_child(self, child: "Node"):
        # 从本节点的 children 集合中移除指定子节点
        if child and self.children:
            # discard 与 remove 不同：元素不存在时不抛异常
            self.children.discard(child)

    @property
    def stage_name(self) -> Literal["draft", "debug", "improve"]:
        """
        Return the stage of the node:
        - "stage" if the node is an initial solution draft
        - "debug" if the node is the result of a debugging step
        - "improve" if the node is the result of an improvement step
        """
        if self.parents is None or len(self.parents) == 0:
            # 没有父节点说明是初始起草节点（draft）
            return "draft"
        # 有父节点时：父节点有 bug 则本节点是调试产物，否则是改进产物
        return "debug" if list(self.parents)[0].is_buggy else "improve"

    def absorb_exec_result(self, exec_result: ExecutionResult):
        """Absorb the result of executing the code from this node."""
        if exec_result is None:
            # 执行结果缺失时用空结果对象兜底
            exec_result = ExecutionResult.get_empty()
        # 保存终端输出
        self._term_out = exec_result.term_out
        # 保存执行耗时
        self.exec_time = exec_result.exec_time
        # 保存退出码
        self.exit_code = exec_result.exit_code

    @property
    def term_out(self) -> str:
        """Get the terminal output of the code execution (after truncating it)."""
        if self._term_out:
            # 把输出片段拼接并截断过长文本后返回
            return trim_long_string("".join(self._term_out))
        # 无输出时返回空字符串
        return ""

    @property
    def is_leaf(self) -> bool:
        """Check if the node is a leaf node in the solution tree."""
        # 没有任何子节点即为叶子节点
        return not self.children

    def __eq__(self, other):
        # 节点相等定义为：同为 Node 且唯一 ID 相同
        return isinstance(other, Node) and self.id == other.id

    def __gt__(self, other: "Node") -> bool:
        # 大小比较委托给指标值的比较（用于挑选最优节点）
        return self.metric > other.metric  # type: ignore

    def __hash__(self):
        # 哈希基于唯一 ID，与 __eq__ 保持一致，可安全放入 set/dict
        return hash(self.id)

    @property
    def debug_depth(self) -> int:
        """
        Length of the current debug path
        - 0 if the node is not a debug node (parent is not buggy)
        - 1 if the parent is buggy but the skip parent isn't
        - n if there were n consecutive debugging steps
        """
        if self.stage_name != "debug":
            # 不是调试节点则调试深度为 0
            return 0

        if not self.parents or len(self.parents) > 1:
            # 调试节点必须恰好有一个父节点，否则状态非法
            raise ValueError("Debug node must have exactly one parent.")
        # 解包出唯一父节点
        (parent,) = self.parents
        # 调试深度 = 父节点调试深度 + 1（递归累计连续调试步数）
        return parent.debug_depth + 1  # type: ignore

    def extra_metrics_to_log(self):
        # 供子类扩展：返回需要额外记录的指标（默认无）
        return {}


@dataclass
class InteractiveSession(DataClassJsonMixin):
    """A collection of nodes for an interactive search session."""

    # 会话包含的节点列表
    nodes: list[Node] = field(default_factory=list)
    # 会话是否已完成
    completed: bool = False

    def append(self, node: Node) -> None:
        # 向会话追加节点
        node.step = len(self.nodes)
        self.nodes.append(node)
        # Link the node only after it has a stable step inside the session.
        # step 稳定后再挂接父子关系，保证树结构一致
        node.attach_to_parents()

@dataclass
class Journal(DataClassJsonMixin):
    """A collection of nodes representing the solution tree."""

    # 解方案树中的全部节点（按 step 顺序）
    nodes: list[Node] = field(default_factory=list)

    def __getitem__(self, idx: int) -> Node:
        # 支持按索引访问节点
        return self.nodes[idx]

    def __len__(self) -> int:
        """Return the number of nodes in the journal."""
        # 返回节点总数
        return len(self.nodes)

    def append(self, node: Node) -> None:
        """Append a new node to the journal."""
        # 给新节点分配步数序号
        node.step = len(self.nodes)
        self.nodes.append(node)
        # Link the node only after it has a stable step inside the journal.
        # step 稳定后再挂接父子关系
        node.attach_to_parents()

    def is_root_node(self, node: Node) -> bool:
        """Check if the node is a root node (no parents)."""
        # 检查一：无父节点（parents 为 None 或空列表）
        first_check = node.parents is None or len(node.parents) == 0
        # 检查二：代码、方案、分析均为空字符串（占位根）
        second_check = node.code == "" and node.plan == "" and node.analysis == ""
        # 检查三：指标为"最差值"占位类型
        third_check = isinstance(node.metric, WorstMetricValue)
        # 检查四：步数序号为 0
        fourth_check = node.step == 0
        # 四个条件同时满足才判定为根节点
        return first_check and second_check and third_check and fourth_check

    @property
    def draft_nodes(self) -> list[Node]:
        """Return a list of nodes representing intial coding drafts"""
        # 收集"初始起草"节点：无父节点，或父节点是占位根节点
        return [
            n
            for n in self.nodes
            if ((n.parents is None) or (len(n.parents) == 0) or self.is_root_node(n.parents[0]))
            and not self.is_root_node(n)
        ]

    @property
    def buggy_nodes(self) -> list[Node]:
        """Return a list of nodes that are considered buggy by the agent."""
        # 收集被判为有 bug 的节点（排除占位根）
        return [n for n in self.nodes if n.is_buggy and not self.is_root_node(n)]

    @property
    def good_nodes(self) -> list[Node]:
        """Return a list of nodes that are not considered buggy by the agent."""
        # 收集无 bug 的节点
        return [n for n in self.nodes if not n.is_buggy]

    def get_metric_history(self) -> list[MetricValue]:
        """Return a list of all metric values in the journal."""
        # 按节点顺序收集全部指标值（指标历史轨迹）
        return [n.metric for n in self.nodes]

    def get_best_node(self, only_good: bool = True) -> Optional[Node]:
        """Return the node with the best validation metric, or ``None`` if none exist.

        If *only_good* is True, search only in ``self.good_nodes``; otherwise search
        the full ``self.nodes`` list.
        """
        # 按开关决定只在优质节点中找最优，还是在全部节点中找
        pool = self.good_nodes if only_good else self.nodes
        if not pool:
            # 候选池为空时返回 None
            return None

        # NB: keep metrics whose value is None – they are still orderable.
        # 在拥有 MetricValue 类型指标的节点中取指标最大者（MetricValue 支持排序）
        return max(
            [n for n in pool if isinstance(n.metric, MetricValue)],
            key=lambda n: n.metric,
            default=None,  # empty after filtering
        )

    def generate_summary(
        self, include_code: bool = False, include_buggy_nodes: bool = False, only_plans: bool = False
    ) -> str:
        """Generate a summary of the journal for the agent."""
        # 该方法已废弃，提示改用记忆算子
        log.warning("generate_summary is deprecated. Use memory operator instead.")

        # 摘要片段列表
        summary = []
        # 按开关决定遍历全部节点还是仅优质节点
        nodes = self.nodes if include_buggy_nodes else self.good_nodes
        # 逐节点生成摘要片段
        for n in nodes:
            # 先写入方案说明
            summary_part = f"Design: {n.plan}\n"
            if not only_plans:
                if include_code:
                    # 需要时附上代码
                    summary_part += f"Code: {n.code}\n"
                if n.is_buggy:
                    # 有 bug 的节点附上终端输出以便定位问题
                    summary_part += f"Node was Buggy: {n.term_out}\n"
                else:
                    # 正常节点附上分析结果与验证指标值
                    summary_part += f"Results: {n.analysis}\n"
                    summary_part += f"Validation Metric: {n.metric.value}\n"

            # 收集该节点摘要
            summary.append(summary_part)
        # 用分隔线拼接所有片段
        return "\n-------------------------------\n".join(summary)

    def get_node_data(self, idx: int):
        # 按索引取出单个节点的完整数据字典
        node = self.nodes[idx]
        # If node.metric is a MetricValue, use node.metric.value, else None
        # 提取指标数值：普通 MetricValue 取其 value
        if isinstance(node.metric, MetricValue):
            metric_value = node.metric.value
        elif isinstance(node.metric, WorstMetricValue):
            metric_value = None  # or some sentinel for "worst"
        else:
            # 其他类型一律视为 None
            metric_value = None

        # 收集子节点的 step 序号列表（跳过 None）
        children = [c.step for c in node.children if c is not None and c.step is not None] if node.children else []
        # 收集父节点的 step 序号列表（跳过 None）
        parents = [p.step for p in node.parents if p is not None and p.step is not None] if node.parents else []
        try:
            # 新版字段结构：包含算子相关字段
            node_data = {
                "step": node.step,
                "id": node.id,
                "plan": node.plan,
                "code": node.code,
                "metric": metric_value,
                "metric_info": node.metric.info if node.metric else None,
                "metric_maximize": node.metric.maximize if node.metric else None,
                "is_buggy": node.is_buggy,
                "analysis": node.analysis,
                "operators_metrics": node.operators_metrics,
                "children": children,
                "parents": parents,
                "creation_time": node.ctime,
                "term_out": node.term_out,
                "operators_used": node.operators_used,
                "exec_time": node.exec_time,
                "exit_code": node.exit_code,
                "_term_out": node._term_out,
            }
        except:
            # Legacy data support
            # 旧版数据没有算子相关字段，退化为不包含它们的结构
            node_data = {
                "step": node.step,
                "id": node.id,
                "plan": node.plan,
                "code": node.code,
                "metric": metric_value,
                "metric_info": node.metric.info if node.metric else None,
                "metric_maximize": node.metric.maximize if node.metric else None,
                "is_buggy": node.is_buggy,
                "analysis": node.analysis,
                "children": children,
                "parents": parents,
                "creation_time": node.ctime,
                "term_out": node.term_out,
                "exec_time": node.exec_time,
                "exit_code": node.exit_code,
                "_term_out": node._term_out,
            }

        # 返回节点数据字典
        return node_data

    def export_data(self) -> dict:
        """Gather data into a dictionary structure."""

        # 找出当前最优节点
        best_node = self.get_best_node()
        if best_node:
            # 取最优节点的代码作为最终方案
            best_solution_code = best_node.code
        else:
            # 无可用节点时方案记为空
            best_solution_code = ""

        # 逐节点导出数据
        node_list = []
        for i in range(len(self.nodes)):
            node_data = self.get_node_data(i)
            node_list.append(node_data)

        # 组装搜索数据：全部节点 + 最优方案
        search_data = {
            "nodes": node_list,
            "solution": best_solution_code,
        }

        # 返回导出字典
        return search_data

    @classmethod
    def from_export_data(cls, export_data: dict) -> "Journal":
        """
        Reconstruct a Journal object from exported search data.

        Args:
            export_data: Dictionary containing the exported search data

        Returns:
            Journal: Reconstructed journal with all nodes and their relationships
        """
        # 创建空的日志对象
        journal = cls()

        # First create all nodes without setting relationships
        # 第一遍：先创建全部节点（不建关系），建立 step -> Node 的映射
        step_to_node: Dict[int, Node] = {}

        # 遍历导出数据中的每个节点记录
        for node_data in export_data["nodes"]:
            # Create metric value if it exists
            # 指标对象默认为 None
            metric = None
            if node_data.get("metric", None) is not None:
                # find keys that start with "metric_info/*"
                # 从记录中收集所有以 metric_info/ 开头的键，还原指标附属信息
                metric_info = {
                    k[len("metric_info/") :]: v for k, v in node_data.items() if k.startswith("metric_info/")
                }
                # 用数值与附加信息构造 MetricValue（maximize 缺省为 True，即越大越好）
                metric = MetricValue(
                    value=node_data["metric"],
                    info=metric_info,
                    maximize=node_data.get("metric_maximize", True),
                )
            else:
                # 无有效指标时用"最差值"占位
                metric = WorstMetricValue()

            # Create the node
            try:
                # 新版记录包含算子相关字段
                node = Node(
                    code=node_data["code"],
                    plan=node_data["plan"],
                    step=node_data["step"],
                    id=node_data["id"],
                    ctime=node_data["creation_time"],
                    operators_metrics=node_data["operators_metrics"],
                    operators_used=node_data["operators_used"],
                    _term_out=node_data["_term_out"],
                    exec_time=node_data.get("exec_time"),
                    exit_code=node_data.get("exit_code"),
                    analysis=node_data["analysis"],
                    metric=metric,
                    is_buggy=node_data["is_buggy"],
                    # Initialize relationships as empty
                    # 关系字段先置空，第二遍再补
                    parents=[],
                    children=set(),
                )
            except:
                # Deprecated legacy data support.
                # 旧版记录缺少算子相关字段，退化构造
                node = Node(
                    code=node_data["code"],
                    plan=node_data["plan"],
                    step=node_data["step"],
                    id=node_data["id"],
                    ctime=node_data["creation_time"],
                    _term_out=node_data["_term_out"],
                    exec_time=node_data.get("exec_time"),
                    exit_code=node_data.get("exit_code"),
                    analysis=node_data["analysis"],
                    metric=metric,
                    is_buggy=node_data["is_buggy"],
                    # Initialize relationships as empty
                    # 关系字段先置空
                    parents=[],
                    children=set(),
                )

            # 登记该 step 对应的节点
            step_to_node[node.step] = node

        # Second pass to set up parent-child relationships
        # 第二遍：依据 step 引用还原父子关系
        for node_data in export_data["nodes"]:
            # 按 step 找到对应节点对象
            node = step_to_node[node_data["step"]]

            # Set parents if they exist
            if node_data["parents"]:
                # 把父节点 step 列表解析为实际 Node 列表
                node.parents = [step_to_node[p_step] for p_step in node_data["parents"]]

            # Set children if they exist
            if node_data["children"] and not any([child is None for child in node_data["children"]]):
                # 子节点 step 集合全部有效时才还原为 Node 集合
                node.children = {step_to_node[c_step] for c_step in node_data["children"]}

            # 把节点（复制其字段）追加进日志，完成挂接
            journal.append(Node(**node.__dict__))

        # 返回重建完成的日志
        return journal

    def node_list(self) -> dict:
        """Gather data into a dictionary structure."""

        # 节点数据列表
        node_list = []
        for i in range(len(self.nodes)):
            # 逐节点导出数据
            node_data = self.get_node_data(i)
            node_list.append(node_data)

        # 返回节点列表
        return node_list
