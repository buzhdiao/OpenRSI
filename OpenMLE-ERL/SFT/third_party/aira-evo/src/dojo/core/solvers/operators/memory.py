# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 os 模块（保留以维持接口一致性）
import os
# 导入 time 模块（保留以维持接口一致性）
import time
# 导入类型注解工具
from typing import Any, Callable, Optional
# 导入 xml 的 include（保留原有导入不动）
from xml.etree.ElementInclude import include

# 导入 humanize（保留以维持接口一致性）
import humanize
# 导入 hydra（保留以维持接口一致性）
import hydra
# 导入 OmegaConf 的配置类型
from omegaconf import DictConfig, OmegaConf

# 导入通用 LLM 封装层
from dojo.core.solvers.llm_helpers.generic_llm import GenericLLM
# 导入进化日志（Journal）与节点（Node）类型
from dojo.core.solvers.utils.journal import Journal, Node
# 导入 wrap_code 工具：把文本包进代码围栏
from dojo.core.solvers.utils.response import wrap_code
# 导入记忆算子的配置数据类
from dojo.config_dataclasses.operators.memory import MemoryOpConfig

# 导入 logging 模块
import logging

# 获取本模块专属日志器
log = logging.getLogger(__name__)


# Memory Op Factory
############################################
def create_memory_op(
    cfg: Optional[MemoryOpConfig],
) -> Callable[[Journal, Optional[Node]], str]:
    # 记忆算子工厂：根据配置生成一个"从日志/节点提取记忆文本"的闭包函数
    if cfg is None:
        # 未配置记忆时返回 None，表示不使用记忆
        return None

    # 按配置名从注册表中取出具体的记忆处理器函数
    memory_processor = MEM_OPS[cfg.memory_processor]
    # 取出传给记忆处理器的额外关键字参数
    memory_op_kwargs = cfg.memory_op_kwargs

    def memory_op(journal: Journal, node: Optional[Node] = None) -> str:
        # 闭包：调用所选记忆处理器，从日志与（可选）节点生成记忆文本
        memory = memory_processor(
            journal=journal,
            node=node,
            **memory_op_kwargs,
        )

        if not memory:
            # 记忆为空时返回占位提示，保证提示词中有明确说明
            memory = "(No memory available.)"

        # 返回最终记忆文本
        return memory

    # 返回构造好的记忆算子闭包
    return memory_op


# Memory Processors
############################################


def truncate_string(s: str, max_prefix: int = 100, max_suffix: int = 100) -> str:
    """
    Truncate a string to 200 characters and add ellipsis.
    """
    # 长度不超过 200 的字符串无需截断
    if len(s) <= 200:
        return s
    # 超长时保留前缀与后缀各 max_prefix/max_suffix 个字符，中间以省略标记替代
    return s[:max_prefix] + "...(truncated)..." + s[-max_suffix:]


def get_node_summary(
    node: Node,
    include_code: bool = False,
    only_plans: bool = False,
) -> str:
    """
    Generate a summary of the node for the agent.
    """
    # 摘要文本初始化为空
    summary = ""

    # If the node has no plan and no code, return an empty summary
    # 节点既无方案也无代码时返回空摘要（无记忆价值）
    if not (node.plan and node.code):
        return ""

    if "debug" in node.operators_used:
        # 该节点由调试算子生成时，标注为"调试方案"
        summary += f"Debug plan: {node.plan}\n"
    else:
        # 否则标注为常规"设计"
        summary += f"Design: {node.plan}\n"

    if only_plans:
        # 只要方案时到此为止，直接返回
        return summary

    if include_code:
        # 需要包含代码时，把节点代码包进代码围栏追加
        summary += f"Code: ```{node.code}```\n"

    if node.is_buggy:
        # 有 bug 的节点没有有效指标，跳过
        pass
    else:
        # 正常节点追加其验证指标（Validation Metric，即验证集上的评价指标）
        summary += f"Validation Metric: {node.metric.value}\n"

    # 取出节点的分析文本并去除首尾空白
    analysis = str(node.analysis or "").strip()
    if analysis:
        # 有分析内容时追加；有 bug 的用"失败分析"标签，否则用"分析"
        label = "Failure Analysis" if node.is_buggy else "Analysis"
        summary += f"{label}: {analysis}\n"

    # 返回完整节点摘要
    return summary


def generate_journal_summary(
    journal: Journal,
    include_code: bool = False,
    include_buggy_nodes: bool = False,
    only_plans: bool = False,
) -> str:
    """
    Generate a summary of the journal for the agent.
    """
    # 节点摘要之间的分隔线
    separator = "\n-------------------------------\n"
    # 按开关决定包含全部节点还是仅无 bug 的优质节点
    nodes = journal.nodes if include_buggy_nodes else journal.good_nodes
    # 过滤掉没有方案或代码的节点
    nodes = [node for node in nodes if node.plan and node.code]
    # 对每个节点生成摘要
    summary = [
        get_node_summary(
            node,
            include_code=include_code,
            only_plans=only_plans,
        )
        for node in nodes
    ]
    # 用分隔线把所有摘要连接成一段文本
    summary = separator.join(summary)
    if not summary:
        # 没有任何有效摘要时给占位提示
        summary = "(No memory available.)"
    # 返回日志级记忆摘要
    return summary


def generate_ancestral_summary(
    node: Optional[Node],
    include_code: bool = False,
    include_buggy_nodes: bool = True,
    only_plans: bool = False,
    until_successful_parent: bool = True,
) -> str:
    """
    Generate a summary of the ancestral journal for the agent.
    """
    # 节点摘要之间的分隔线
    separator = "\n-------------------------------\n"
    # 以"从当前节点向祖先回溯"的顺序收集摘要
    summaries_in_reverse = []

    if not include_buggy_nodes:
        # 祖先记忆建议包含失败节点（失败教训同样有价值），否则给出警告
        log.warning("It's not recommended to use ancestral memory without buggy nodes.")

    # 沿父指针逐级向上遍历祖先
    while node is not None:
        # now if this was a successful parent node, we can stop
        # 开关打开且当前节点是成功的父节点时停止回溯
        if until_successful_parent and not node.is_buggy:
            break

        if include_buggy_nodes or not node.is_buggy:
            # 按开关过滤后生成该祖先的摘要
            node_summary = get_node_summary(node, include_code=include_code, only_plans=only_plans)
            if node_summary:
                # 有效摘要才收集
                summaries_in_reverse.append(node_summary)

        # 取当前节点的父节点集合
        parents = node.parents
        if not parents:
            # 没有父节点说明已到根，结束遍历
            node = None
            break

        # if len(parents) > 1:
        #     raise ValueError("AncestralMemory supports maximum one parent.")

        # 取第一个父节点继续向上（祖先记忆假定单亲链）
        node = parents[0]  # unpack the first parent

    # 把倒序收集的摘要反转为"根→当前"的时间顺序
    summaries = reversed(summaries_in_reverse)
    # 用分隔线连接并返回祖先记忆摘要
    return separator.join(summaries)


def simple_memory(
    journal: Journal,
    node: Optional[Node] = None,
    max_length: Optional[int] = None,
    **kwargs,
) -> str:
    # 简单记忆：直接取整个日志的摘要作为记忆文本
    summary = generate_journal_summary(journal, **kwargs)

    if max_length and summary:
        # 指定了最大长度且摘要非空时进行截断
        if len(summary) > max_length:
            # 超长时保留末尾 max_length 个字符，并在前面标注截断信息
            summary = f"...(truncated) {summary[-max_length:]}"

    # 返回记忆文本
    return summary


def ancestral_memory(
    journal: Journal,
    node: Optional[Node] = None,
    **kwargs,
) -> str:
    # 祖先记忆：以当前节点为起点，沿父链生成祖先摘要
    return generate_ancestral_summary(node, **kwargs)


def no_memory(
    journal: Journal,
    node: Optional[Node] = None,
    **kwargs,
) -> str:
    # 空记忆：始终返回空字符串（禁用记忆）
    return ""


def get_sibling_summary(
    parent_node: Optional[Node],
    include_code: bool = False,
    include_buggy_nodes: bool = True,
    only_plans: bool = True,
):
    # 兄弟记忆：汇总同一父节点下所有兄弟节点的经验
    # 节点摘要之间的分隔线
    separator = "\n-------------------------------\n"
    # 取出父节点的全部子节点作为兄弟集合
    previous_siblings = [node for node in parent_node.children]
    if not include_buggy_nodes:
        # 兄弟记忆建议包含失败节点，否则给出警告
        log.warning("It's not recommended to use sibling memory without buggy nodes.")

    # 收集各兄弟节点摘要
    summaries = []
    for node in previous_siblings:
        if include_buggy_nodes or not node.is_buggy:
            # 按开关过滤后生成兄弟摘要
            node_summary = get_node_summary(node, include_code=include_code, only_plans=only_plans)
            if node_summary:
                # 有效摘要才收集
                summaries.append(node_summary)

    if len(previous_siblings) == 0 or previous_siblings is None:
        # 没有任何兄弟节点时返回占位提示
        return "(No memory available.)"

    # 用分隔线连接所有兄弟摘要
    summary = separator.join(summaries)
    if not summary:
        # 全部被过滤导致为空时同样返回占位提示
        summary = "(No memory available.)"
    # 返回兄弟记忆摘要
    return summary


def sibling_memory(
    journal: Journal,
    node: Optional[Node] = None,
    **kwargs,
) -> str:
    # 兄弟记忆入口：以给定节点（作为父节点）聚合其子节点的经验
    if node is None:
        # 兄弟记忆必须提供节点，否则报错
        raise ValueError("Sibling memory requires a node to be provided.")

    # 调用摘要生成函数
    return get_sibling_summary(node, **kwargs)


# 记忆处理器注册表：名称 -> 处理函数，供工厂按配置选择
MEM_OPS = {
    "simple_memory": simple_memory,
    "no_memory": no_memory,
    "sibling_memory": sibling_memory,
    "ancestral_memory": ancestral_memory,
}
