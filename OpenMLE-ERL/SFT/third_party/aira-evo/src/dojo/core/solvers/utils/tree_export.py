# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are MIT-licensed
# Copyright (c) 2024 Weco AI Ltd
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/WecoAI/aideml/blob/main/LICENSE

"""Export journal to HTML visualization of tree + code."""

# 导入 json 模块，用于把树结构序列化为 JSON 字符串
import json
# 导入 textwrap（保留以维持接口一致性）
import textwrap
# 导入 pathlib 的 Path，用于模板文件与输出路径处理
from pathlib import Path

# 导入 numpy，用于布局坐标数组运算
import numpy as np
# 导入 igraph 的 Graph，用于计算树形布局
from igraph import Graph

# 导入进化日志（Journal）类型
from dojo.core.solvers.utils.journal import Journal


def get_edges(journal: Journal):
    # 生成器：产出树中所有 (父节点 step, 子节点 step) 边
    # 遍历日志中的每个节点
    for node in journal:
        if node.step is None:
            # 未分配 step 的节点（尚未入树）直接跳过
            continue
        # 遍历该节点的所有子节点
        for c in node.children:
            if c is None or c.step is None:
                # 子节点为空或未入树时跳过
                continue
            # 产出一条父子边
            yield (node.step, c.step)


def generate_layout(n_nodes, edges):
    # 用 Reingold-Tilford 树布局算法（"rt"）计算 n 个节点在边集 edges 下的二维坐标
    layout = Graph(
        n_nodes,
        edges=edges,
        directed=True,
    ).layout("rt")
    # 取所有节点 y 坐标的最大值（igraph 布局 y 轴向下为正）
    y_max = max(layout[k][1] for k in range(n_nodes))
    # 翻转后的坐标列表
    layout_coords = []
    for n in range(n_nodes):
        # 把每个节点的 y 坐标做镜像翻转（2*y_max - y），使根在顶部
        layout_coords.append((layout[n][0], 2 * y_max - layout[n][1]))
    # 返回 numpy 数组形式的坐标
    return np.array(layout_coords)


def normalize_layout(layout: np.ndarray):
    """Normalize layout to [0, 1]"""
    # 把两个坐标轴分别线性缩放到 [0,1]（加 1e-8 防止除零）
    layout = (layout - layout.min(axis=0)) / (layout.max(axis=0) - layout.min(axis=0) + 1e-8)
    # 再次翻转 y 轴，保证树根在上方
    layout[:, 1] = 1 - layout[:, 1]
    # y 轴的 NaN 一律替换为 0
    layout[:, 1] = np.nan_to_num(layout[:, 1], nan=0)
    # x 轴的 NaN 替换为 0.5（居中），避免单节点树除零产生 NaN
    layout[:, 0] = np.nan_to_num(layout[:, 0], nan=0.5)
    # 返回归一化后的布局坐标
    return layout


def get_human_readable_prompt_text(messages):
    # Iterate over the messages and print them with headers and demarcations
    # 把多轮消息拼成带角色标题、分隔线的人类可读文本
    prompt_text = ""
    # 逐条遍历消息
    for message in messages:
        # 取消息角色（system/user/assistant）
        role = message["role"]

        # 构造形如 "R: USER" 的角色标题
        role_str = f"R: {role.upper()}"
        # 生成与标题等长的分隔线
        header_line = "=" * len(role_str)

        # Print header with color
        # 把标题块（上下夹分隔线）追加进文本
        prompt_text += f"\n{header_line}\n{role_str}\n{header_line}\n"

        # Wrap each line individually while preserving newline characters
        # 追加消息正文并空一行分隔
        prompt_text += f"{message['content']}\n\n"

    # 返回拼接后的完整提示词文本
    return prompt_text


def cfg_to_tree_struct(cfg, jou: Journal):
    # 把日志（Journal）转换为树可视化所需的结构化字典
    # 收集树的全部边
    edges = list(get_edges(jou))
    # 计算并归一化树布局坐标
    layout = normalize_layout(generate_layout(len(jou), edges))

    # 指标数组初始化为全 0（占位，与旧版可视化格式兼容）
    metrics = np.array([0 for n in jou])
    # 收集每个节点的指标数值（numpy 安全形式，无值时为 NaN）
    fitnesses = np.array([n.metric.value_npsafe for n in jou]).tolist()
    # 收集每个节点的完整数据字典
    node_data_list = [jou.get_node_data(n.step) for n in jou]
    # 收集每个节点指标的附属信息（JSON 缩进格式化）
    metric_infos = [json.dumps(n.metric.info, indent=2) for n in jou]

    # 收集每个节点的提示词文本（记录生成该节点时发给 LLM 的完整对话）
    prompts = []
    for n in jou:
        # 该节点的提示词文本初始为空
        prompt_text = ""
        if n.operators_metrics and len(n.operators_metrics) > 0:
            # 遍历该节点各算子的指标记录
            for op_metrics in n.operators_metrics:
                if isinstance(op_metrics, dict):
                    # 仅处理字典形式的记录
                    if "prompt_messages" in op_metrics:
                        # 复制一份提示消息列表，避免修改原数据
                        messages = op_metrics["prompt_messages"].copy()
                    if "completion_text" in op_metrics:
                        # 把模型补全文本作为 assistant 消息追加，还原完整对话
                        messages.append({"role": "assistant", "content": op_metrics["completion_text"]})
                    # 把人类可读的对话文本加重分隔线后追加
                    prompt_text += (
                        get_human_readable_prompt_text(messages)
                        + "\n\n\n ================================================================= \n\n\n"
                    )

        # 收集该节点的提示词文本
        prompts.append(prompt_text)

    # 组装可视化所需的全部字段（方案、代码、输出、分析、指标、提示词等）
    return dict(
        edges=edges,
        layout=layout.tolist(),
        plan=[n.plan for n in jou.nodes],
        code=[n.code for n in jou],
        term_out=[n.term_out for n in jou],
        analysis=[n.analysis for n in jou],
        exp_name=getattr(cfg, "id", getattr(cfg, "exp_name", "")),
        metrics=metrics.tolist(),
        fitnesses=fitnesses,
        prompts=prompts,
        node_data_list=node_data_list,
        metric_infos=metric_infos,
    )


def generate_html(tree_graph_str: str):
    # 把树结构 JSON 字符串填进 HTML/JS 模板，生成独立可打开的可视化页面
    # 模板目录与本文件同级
    template_dir = Path(__file__).parent / "viz_templates"

    with open(template_dir / "template.js") as f:
        # 读取 JS 模板，把占位符替换为树结构 JSON 字符串
        js = f.read()
        js = js.replace("null/*<-replacehere*/;", tree_graph_str)

    with open(template_dir / "template.html") as f:
        # 读取 HTML 模板，把占位注释替换为填充后的 JS
        html = f.read()
        html = html.replace("<!-- placeholder -->", js)

        # 返回最终 HTML 文本
        return html


def generate(cfg, jou: Journal, out_path: Path):
    # 对外入口：生成树+代码的可视化 HTML 并写入 out_path；失败时打印原因并返回 False
    try:
        # 把树结构序列化为 JSON 字符串
        tree_graph_str = json.dumps(cfg_to_tree_struct(cfg, jou))
        # 填充模板生成 HTML
        html = generate_html(tree_graph_str)
        with open(out_path, "w") as f:
            # 写出 HTML 文件
            f.write(html)
        # 成功返回 True
        return True
    except Exception as e:
        # 任何异常都只打印提示，不中断搜索主流程
        print(f"Couldn't generate visualisation due to {e}")
        return False
