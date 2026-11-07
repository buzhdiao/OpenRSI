# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 defaultdict：按竞赛 ID 聚合各实验统计量的默认字典
from collections import defaultdict
# 导入 concurrent.futures：进程池并行处理多个实验
import concurrent.futures
# 导入 itertools：chain 用于把嵌套列表拍平
import itertools
# 导入 re：正则，用于把蛇形/驼峰命名转成可读标题
import re
# 导入 Path：面向对象的路径操作
from pathlib import Path
# 导入类型标注工具
from typing import Any, Dict, List, Union

# 导入 matplotlib：绘图后端，保存统计图
import matplotlib.pyplot as plt
# 导入 seaborn：基于 matplotlib 的统计绘图库（箱线图/直方图/条形图）
import seaborn as sns
# 导入 numpy：数值计算与统计
import numpy as np
# 导入 pandas：构造绘图用 DataFrame
import pandas as pd
# 导入 tiktoken：token 计数（本文件中未直接使用该导入）
import tiktoken
# 导入 yaml：本文件中预留的 YAML 读取能力（未直接使用）
import yaml
# 导入运行配置类：读取实验的 dojo_config.json
from dojo.config_dataclasses.run import RunConfig
# 导入环境工具：获取 mlebench 数据根目录
from dojo.utils.environment import get_mlebench_data_dir
# 导入 Journal 类：研究日志（搜索树）对象
from dojo.core.solvers.utils.journal import Journal
# 导入思考标签解析：剥离模型输出的 <think> 片段
from dojo.core.solvers.utils.response import parse_thinking_tags
# 导入日志转 JSON 的函数：JOURNAL.jsonl -> journal.json
from dojo.analysis_utils.journal_to_tree import save_journal_log_as_json


# 创建并返回 GDM 后端的 LLM 客户端（用于根据日志生成技术报告）
def create_client(api: str = "gdm", model_id: str = "gemini-2.0-flash", provider: str = "gdm"):
    """Creates and returns a configured GDM client."""
    # 函数内延迟导入，避免模块加载时的循环依赖/开销
    from omegaconf import OmegaConf
    from dojo.core.solvers.llm_helpers.backends.gdm import GDMClient

    # 用 OmegaConf 构造客户端配置
    client_cfg = OmegaConf.create({"api": api, "model_id": model_id, "provider": provider})
    # 实例化并返回 GDM 客户端
    return GDMClient(client_cfg)


# Global client instance for reuse across functions
# 模块级全局客户端实例，供本文件各函数复用（避免重复创建）
client = create_client()


# 根据研究日志生成一份结构化的技术报告（markdown 文本）
def generate_journal_report(journal: Journal, task_description: str, include_code: bool = False):
    """Generates a structured technical markdown report from a journal."""

    # 把日志对象序列化为摘要文本，作为提示词输入（可选包含代码）
    report_input = journal.generate_summary(include_code=include_code)

    # 构造系统提示词：角色/目标/输入/输出格式（字符串字面量，勿改）
    system_prompt = f"""
    # Role
    You are a research assistant that always uses concise language.
    # Goal
    The goal is to write a technical report summarising the empirical findings and technical decisions.
    # Input
    You are given a raw research journal with list of design attempts and their outcomes, and a task description.
    # Output
    Your output should be a single markdown document.
    Your report should have the following sections: Introduction, Preprocessing, Modellind Methods, Results Discussion, Future Work.
    You can include subsections if needed.
    Here is the research journal of the agent: <journal>{report_input}<\\journal>, 
    and the task description is: <task>{task_description}<\\task>.
    """

    # 组装消息：提示词作为系统消息
    messages = [{"role": "system", "content": system_prompt}]

    # 调用 LLM 生成报告
    out, _ = client.query(messages=messages)

    # 剥离思考标签，只保留最终回答文本
    _, text_without_thinking = parse_thinking_tags(out)

    # 返回报告的 markdown 文本
    return text_without_thinking


# 从 Journal 搜索树中计算一批统计指标（节点数、深度、耗时、分数等）
def calculate_tree_statistics(journal: Journal) -> Dict[str, Union[int, float]]:
    """Calculates statistical metrics from a Journal tree structure."""
    # 取出搜索树的全部节点（Journal 即 agent 搜索树的数据结构）
    nodes = journal.nodes

    # 初始化统计字典（随后整体覆盖赋值）
    stats = {}

    # 第一组统计：节点总数、根节点数、中间节点数、叶子节点数
    stats = {
        # 节点总数
        "num_total_nodes": len(nodes),
        # 根节点：没有任何父节点的节点个数
        "num_root_nodes": sum([1 if len(node.parents) == 0 else 0 for node in nodes]),
        # 中间节点：既有父又有子的节点个数
        "num_intermediate_nodes": sum(
            [1 if (len(node.parents) > 0 and len(node.children) > 0) else 0 for node in nodes]
        ),
        # 叶子节点：没有子节点的节点个数
        "num_leaf_nodes": sum([1 if len(node.children) == 0 else 0 for node in nodes]),
    }

    # 每个节点的子节点数量列表
    num_children = [len(node.children) for node in nodes]
    # 每个节点的执行耗时列表
    execution_times = [node.exec_time for node in nodes]

    # 空树时用 [0] 兜底，避免后续 max 报错
    if len(num_children) == 0:
        num_children = [0]

    # 第二组统计：平均/最大子节点数，以及执行耗时的均值/最值/中位数/总和
    stats.update(
        {
            # 平均子节点数 = 子节点总数 / 节点总数（分支因子）
            "avg_num_children": sum(num_children) / stats["num_total_nodes"],
            # 最大的子节点数
            "max_num_children": max(num_children),
            # 平均执行耗时（无数据时为 None）
            "avg_exec_time": np.mean(execution_times) if execution_times else None,
            # 最短执行耗时
            "min_exec_time": np.min(execution_times) if execution_times else None,
            # 最长执行耗时
            "max_exec_time": np.max(execution_times) if execution_times else None,
            # 执行耗时中位数
            "median_exec_time": np.median(execution_times) if execution_times else None,
            # 执行耗时总和
            "total_exec_time": np.sum(execution_times) if execution_times else None,
        }
    )

    # Depth calculation helper function
    # 递归计算节点深度的内部辅助函数
    def calculate_depth(node_id, node_dict, depth=0):
        # 取出当前节点
        node = node_dict[node_id]
        # 叶子节点直接返回当前深度
        if not node.children:
            return depth
        # 否则在所有子节点中取最大深度并加一
        return max(calculate_depth(child.id, node_dict, depth + 1) for child in node.children)

    # 建立 节点ID -> 节点 的映射，便于按 id 查找
    node_dict = {node.id: node for node in nodes}
    # 找出全部根节点
    root_nodes = [node for node in nodes if len(node.parents) == 0]
    # 分别从每个根节点递归求子树深度
    depths = [calculate_depth(root_node.id, node_dict=node_dict) for root_node in root_nodes]
    # 树的最大深度（空树记 0）
    stats["max_depth"] = max(depths) if depths else 0

    # Nodes with metrics
    # 统计带有效验证指标的节点个数（metric.value 非空）
    nodes_with_val_metrics = [1 if node.metric.value is not None else 0 for node in nodes]
    # 空列表兜底为 [0]，避免后续求和出错
    if len(nodes_with_val_metrics) == 0:
        nodes_with_val_metrics = [0]
    # 写入带验证指标的节点数
    stats["num_nodes_with_val_metrics"] = sum(nodes_with_val_metrics)

    # Average metrics (if metrics are available)
    # 收集所有节点的验证得分（仅非空值）
    metrics = [node.metric.value for node in nodes if node.metric.value is not None]
    # 平均验证得分（无数据为 None）
    stats["average_val_metric"] = sum(metrics) / len(metrics) if metrics else None

    # Number of buggy nodes
    # 统计缺陷（buggy）节点个数：代码有 bug 的尝试
    stats["num_buggy_nodes"] = sum([1 if node.is_buggy else 0 for node in nodes])
    # 正常节点数 = 总数 - 缺陷节点数
    stats["num_good_nodes"] = stats["num_total_nodes"] - stats["num_buggy_nodes"]

    # Buggy nodes rate
    # 缺陷节点占比（节点总数为 0 时记 0）
    stats["buggy_nodes_proportion"] = (
        stats["num_buggy_nodes"] / stats["num_total_nodes"] if stats.get("num_total_nodes") else 0
    )
    # 正常节点占比 = 1 - 缺陷占比
    stats["good_nodes_proportion"] = 1 - stats["buggy_nodes_proportion"]

    # Nodes without analysis
    # 统计缺少分析文本（analysis 为空）的节点数
    stats["nodes_without_analysis"] = sum([1 if (not node.analysis) else 0 for node in nodes])

    # 收集所有节点使用过的算子（设计决策），用于频次统计
    operators_used_all = []
    # 逐节点展开其算子列表
    for node in nodes:
        operators_used_all.extend(node.operators_used)
    # 去重得到算子集合
    operators_set = set(operators_used_all)
    # 对每种算子统计全局频次与占比
    for op in operators_set:
        # 内部计数函数：统计单节点中该算子的使用次数
        op_counter = lambda n, op: sum([1 if op_used == op else 0 for op_used in n.operators_used])
        # 该算子在整棵树中的使用总次数
        stats[f"{op}_frequency"] = sum([op_counter(node, op) for node in nodes])
        # 该算子使用频次相对节点总数的比例
        stats[f"{op}_proportion"] = stats[f"{op}_frequency"] / stats["num_total_nodes"]

    # 收集每个节点的算子级用量指标（tokens/延迟等），并拍平成单一列表
    all_operators_metrics = [node.operators_metrics for node in nodes]
    all_operators_metrics = list(itertools.chain(*all_operators_metrics))

    # 存在用量数据时统计 token 与延迟
    if all_operators_metrics:
        # 全部调用的总 token 数列表
        all_token_counts = [op_metrics["usage"]["total_tokens"] for op_metrics in all_operators_metrics]

        # 全部调用的提示（prompt）token 数列表
        all_prompt_token_counts = [op_metrics["usage"]["prompt_tokens"] for op_metrics in all_operators_metrics]
        # 全部调用的补全（completion）token 数列表
        all_completion_token_counts = [
            op_metrics["usage"]["completion_tokens"] for op_metrics in all_operators_metrics
        ]
        # 全部调用的延迟列表
        all_latency = [op_metrics["usage"]["latency"] for op_metrics in all_operators_metrics]

        # 平均与总 token 数
        stats["avg_num_all_tokens"] = sum(all_token_counts) / len(all_token_counts)
        stats["total_num_all_tokens"] = sum(all_token_counts)

        # 平均与总提示 token 数
        stats["avg_num_prompt_tokens"] = sum(all_prompt_token_counts) / len(all_prompt_token_counts)
        stats["total_num_prompt_tokens"] = sum(all_prompt_token_counts)

        # 平均与总补全 token 数
        stats["avg_num_completion_tokens"] = sum(all_completion_token_counts) / len(all_completion_token_counts)
        stats["total_num_completion_tokens"] = sum(all_completion_token_counts)

        # 平均与总延迟
        stats["avg_latency"] = sum(all_latency) / len(all_latency)
        stats["total_latency"] = sum(all_latency)

    # MLEbench specific metrics
    # 收集 MLEbench 特有的测试集得分（metric.info 中的 score 字段）
    test_scores = [node.metric.info["score"] for node in nodes if node.metric.info and "score" in node.metric.info]
    # 存在测试得分时统计均值与最值
    if test_scores:
        # 平均测试得分
        stats["avg_test_score"] = sum(test_scores) / len(test_scores)
        # 最低测试得分
        stats["min_test_score"] = min(test_scores)
        # 最高测试得分
        stats["max_test_score"] = max(test_scores)

    # 返回完整统计字典
    return stats


# 把蛇形命名/驼峰命名的键名转换成适合人类阅读的标题格式
def prettify_key(key: str) -> str:
    """Converts a snake_case or camelCase key to a human-readable format."""
    # 下划线换空格；再在大写字母前插入空格（正则零宽断言）；最后词首大写
    return re.sub(r"(?<!^)(?=[A-Z])", " ", key.replace("_", " ")).title()


# 把统计字典格式化为 markdown 项目符号列表
def dict_to_markdown(data: dict) -> str:
    """Formats a dictionary into markdown."""
    # 每个键值对渲染为一行 "- **键:** 值"，按行拼接
    return "\n".join(f"- **{prettify_key(k)}:** {v}" for k, v in data.items())


# 为单个实验生成技术报告与统计报告，并返回统计字典
def write_tree_reports(experiment_path: Path):
    """Writes detailed markdown reports and statistics based on an experiment's journal."""
    # 读取实验运行配置
    config = RunConfig.load_from_json(experiment_path / "dojo_config.json")
    # 拆出竞赛 ID 与实验名称
    competition_id, exp_name = config.task.name, config.meta_id

    # 拼出该竞赛的任务描述文件路径（mlebench prepared/public/description.md）
    task_desc_path = Path(get_mlebench_data_dir()) / competition_id / "prepared/public/description.md"
    # 读取任务描述全文
    task_description = task_desc_path.read_text()

    # 找到原始日志 JSONL 路径
    journal_json_path = experiment_path / "json/JOURNAL.jsonl"
    # 转成 journal.json 并加载为 Journal 搜索树对象
    journal = Journal.from_export_data(save_journal_log_as_json(journal_json_path, experiment_path, "journal.json"))

    # 让 LLM 根据日志与任务描述生成技术报告
    report = generate_journal_report(journal, task_description)
    # 把报告写到 <竞赛ID>_report.md
    (experiment_path / f"{competition_id}_report.md").write_text(report)

    # 计算搜索树的全部统计指标
    stats = calculate_tree_statistics(journal)
    # 把统计字典渲染为 markdown
    stats_report = dict_to_markdown(stats)
    # 写统计报告文件，附上实验名
    (experiment_path / f"{competition_id}_stats_report.md").write_text(stats_report + f"\n\nExperiment: {exp_name}")

    # 统计字典再补充竞赛 ID 与实验名，便于上层聚合
    stats["competition_id"] = competition_id
    stats["exp_name"] = exp_name

    # 返回带标识的统计字典
    return stats


# 把多次运行（多个实验）的统计量汇总绘图：箱线图/直方图/条形图，并写出分位数文本
def plot_aggregate_stats(stats_list: List[Dict[str, Union[int, float]]], comp_name: str, meta_exp_dir: Path):
    """
    Plots various distributions (boxplot, histogram, and bar chart) of all numerical
    metrics across multiple runs using seaborn. Also calculates and writes raw data statistics
    (mean, median, Q1, Q3) for each metric to a text file.

    Args:
        stats_list (List[Dict[str, Union[int, float]]]): A list of dictionaries where keys are metric names
                                           and values are numerical values.
        comp_name (str): Base name for the output files (PNG and TXT).
    """

    # 输出目录：<元实验目录>/tree_stats
    plot_loc = meta_exp_dir / "tree_stats"
    # 递归创建输出目录
    plot_loc.mkdir(parents=True, exist_ok=True)

    # Identify all metric names from stats_list
    # 收集所有统计字典中出现过的指标名（去重）
    metric_names = set()
    for stats in stats_list:
        metric_names.update(stats.keys())

    # Collect numeric values for each metric
    # 为每个指标建立取值列表，逐实验收集数值型数据
    metric_data = {metric: [] for metric in metric_names}
    for stats in stats_list:
        for metric, value in stats.items():
            # 仅收集 int/float 型值，跳过字符串/None 等
            if isinstance(value, (int, float)):
                metric_data[metric].append(value)

    # Remove metrics with no numeric data
    # 剔除没有任何数值数据的指标
    metric_data = {k: v for k, v in metric_data.items() if v}
    # 全部为空时提示并直接返回
    if not metric_data:
        print("No numeric metrics found to plot.")
        return

    # Write summary statistics to a text file
    # 把每个指标的原始统计量（最值/均值/分位数）写入文本文件
    with open(plot_loc / f"{comp_name}_stats.txt", "w") as f:
        for metric, values in metric_data.items():
            # 转为 numpy 数组便于统计计算
            values_array = np.array(values)
            # 最小值
            min_val = np.min(values_array)
            # 均值
            mean_val = np.mean(values_array)
            # 中位数
            median_val = np.median(values_array)
            # 下四分位数 Q1（论文中常用的分位数统计量）
            q1 = np.percentile(values_array, 25)
            # 上四分位数 Q3
            q3 = np.percentile(values_array, 75)
            # 最大值
            max_val = np.max(values_array)
            # 逐行写出该指标的统计结果（保留 3 位小数）
            f.write(f"Metric: {metric}\n")
            f.write(f"  Min: {min_val:.3f}\n")
            f.write(f"  Q1 (25th percentile): {q1:.3f}\n")
            f.write(f"  Mean: {mean_val:.3f}\n")
            f.write(f"  Median: {median_val:.3f}\n")
            f.write(f"  Q3 (75th percentile): {q3:.3f}\n")
            f.write(f"  Max: {max_val:.3f}\n")
            f.write("-" * 40 + "\n")

    # Set up seaborn style
    # 设置 seaborn 白底网格样式
    sns.set(style="whitegrid")

    # Create subplots: one row per metric, three columns for the different plots
    # 指标个数决定子图行数；每行三列：箱线图/直方图/条形图
    num_metrics = len(metric_data)
    fig, axes = plt.subplots(num_metrics, 3, figsize=(18, 4 * num_metrics))

    # If only one metric, wrap axes into a 2D array for consistency
    # 单指标时把 axes 包装成二维数组，统一后续索引方式
    if num_metrics == 1:
        axes = np.array([axes])

    # Loop through each metric and create the plots
    # 逐指标绘制三联图
    for ax_triplet, (metric, values) in zip(axes, metric_data.items()):
        # Prepare a DataFrame for seaborn
        # 构造 seaborn 需要的长表 DataFrame
        df = pd.DataFrame({metric: values})
        df["Run"] = df.index  # Run index for the bar plot

        # Boxplot
        # 第一列：箱线图，展示分布与离群点
        sns.boxplot(data=df, x=metric, ax=ax_triplet[0])
        ax_triplet[0].set_title(f"Boxplot of {metric}")

        # Histogram with KDE for a smoother view
        # 第二列：直方图叠加核密度估计（KDE）曲线
        sns.histplot(data=df, x=metric, bins=20, kde=True, ax=ax_triplet[1])
        ax_triplet[1].set_title(f"Histogram of {metric}")
        ax_triplet[1].set_xlabel(metric)
        ax_triplet[1].set_ylabel("Frequency")

        # Bar chart showing each run's value
        # 第三列：条形图，逐次运行（Run）展示该指标的取值
        sns.barplot(data=df, x="Run", y=metric, ax=ax_triplet[2], errorbar=None)
        ax_triplet[2].set_title(f"Bar Chart of {metric}")
        ax_triplet[2].set_xlabel("Run Index")
        ax_triplet[2].set_ylabel(metric)

    # 自动收紧子图布局避免重叠
    plt.tight_layout()
    # 保存整张图为 PNG
    plt.savefig(plot_loc / f"{comp_name}.png")
    # 关闭图对象释放内存
    plt.close(fig)


# 并行处理元实验目录下的全部实验：生成报告、聚合统计、按竞赛出汇总图
def generate_tree_reports_and_stats(meta_experiment_path: Path):
    """Concurrently processes multiple experiments and aggregates statistics."""
    # 统一转为 Path 对象
    meta_experiment_path = Path(meta_experiment_path)

    # 目录不存在时提示并返回
    if not meta_experiment_path.exists():
        print(f"Experiments path does not exist: {meta_experiment_path}")
        return

    # 按竞赛 ID 聚合各实验统计量的容器
    all_experiment_stats = defaultdict(list)
    # 列出全部实验子目录
    experiment_folders = [folder for folder in meta_experiment_path.iterdir() if folder.is_dir()]

    # 用进程池并行生成各实验的报告与统计
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 为每个实验目录提交 write_tree_reports 任务
        future_to_experiment = {
            executor.submit(write_tree_reports, experiment_folder): experiment_folder
            for experiment_folder in experiment_folders
        }
        # 按完成顺序收集结果
        for future in concurrent.futures.as_completed(future_to_experiment):
            # 由 future 反查对应实验目录
            experiment_folder = future_to_experiment[future]
            try:
                # 拿到该实验的统计字典
                stats = future.result()
                # 读取竞赛 ID 作为聚合键
                comp_id = stats["competition_id"]
                # 按竞赛 ID 归组保存
                all_experiment_stats[comp_id].append(stats)
                # 打印处理成功的实验名
                print(f"Processed experiment: {experiment_folder.name}")
            except Exception as exc:
                # 单个实验失败不影响其他实验
                print(f"Experiment {experiment_folder.name} generated an exception: {exc}")

    # After processing all experiments, plot aggregate stats per competition id
    # 全部实验处理完后，按竞赛 ID 分别绘制聚合统计图
    for comp, stats in all_experiment_stats.items():
        plot_aggregate_stats(stats, comp, meta_experiment_path)


# 命令行入口：传入元实验目录即可运行全流程
if __name__ == "__main__":
    import sys

    # 缺少参数时打印用法并退出
    if len(sys.argv) < 2:
        print("Usage: python meta_tree_analysis.py <meta_experiment_directory>")
        sys.exit(1)

    # 对第一个参数指定的目录执行报告生成与统计绘图
    generate_tree_reports_and_stats(Path(sys.argv[1]))
