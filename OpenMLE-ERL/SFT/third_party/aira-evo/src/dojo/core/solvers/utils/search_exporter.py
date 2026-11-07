# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 datetime，用于给导出文件名打时间戳
from datetime import datetime
# 导入 json 模块，用于写出/读回搜索数据
import json
# 导入 logging 模块
import logging
# 导入 tempfile，用于测试时创建临时目录
import tempfile
# 导入 pathlib 的 Path，用于路径处理
from pathlib import Path

# 导入 asdict，把 OmegaConf 配置转成普通字典
from dataclasses import asdict

# 导入 OmegaConf 的配置类型
from omegaconf import DictConfig, OmegaConf

# 导入 wandb，用于把树可视化以 HTML 形式记录到实验平台
import wandb
# 导入树可视化导出模块
from dojo.core.solvers.utils import tree_export
# 导入日志（Journal）与节点（Node）类型
from dojo.core.solvers.utils.journal import Journal, Node
# 导入指标值类型
from dojo.core.solvers.utils.metric import MetricValue
# 导入集合日志器与日志事件枚举
from dojo.utils.logger import CollectiveLogger, LogEvent

# 获取本模块专属日志器
logger = logging.getLogger(__name__)


class SearchExporter:
    # 搜索结果导出器：把进化日志（Journal）汇总为结构化数据并可选导出 JSON 与树可视化

    def __init__(self, journal: Journal, cfg):
        # 保存进化日志引用
        self.journal = journal
        # 保存运行配置
        self.cfg = cfg

    def gather_and_export_search_results(
        self, tree_path: str | Path | None = None, output_file: str | Path | None = None
    ) -> dict:
        """
        Gather search data into a dictionary structure and optionally
        export it to a JSON file. Allows for a dynamically specified `tree_path`
        for the generated HTML visualization.

        Args:
            tree_path (str | Path | None, optional):
                Where to generate the tree visualization HTML. If None,
                the tree export is skipped, and the returned data has `tree_path=None`.
            output_file (str | Path | None, optional):
                If provided, the search data is written out to this file in JSON format.

        Returns:
            dict: A dictionary containing:
                - "nodes": a list of node dictionaries
                - "config": the config used by Greedy
                - "tree_path": the HTML path for tree visualization (if any)
                - "solution": the best solution code (if any)
        """

        # 把日志导出为基础数据字典（节点列表 + 最优方案）
        journal_data = self.journal.export_data()

        # 树可视化路径字符串初始化为 None（未生成时保持 None）
        tree_path_str = None
        if tree_path is not None:
            # 指定了树路径时转为 Path 对象
            tree_path = Path(tree_path)
            if tree_export.generate(self.cfg, self.journal, tree_path):
                # 生成成功则记录其绝对路径字符串
                tree_path_str = str(tree_path.absolute())

        # 组装完整搜索数据：日志数据 + 配置 + 树可视化路径
        search_data = {
            **journal_data,
            "config": asdict(self.cfg),
            "tree_path": tree_path_str,
        }

        if output_file:
            # 指定了输出文件时把搜索数据写成 JSON
            output_file = Path(output_file)
            with output_file.open("w", encoding="utf-8") as f:
                # 以缩进格式写出 JSON
                json.dump(search_data, f, indent=2)
            # 记录导出成功的日志
            logger.info(f"Exported search results to {output_file.resolve()}")

        # 返回搜索数据字典
        return search_data


def export_search_results(cfg: DictConfig, journal: Journal, logger: CollectiveLogger, search_method: str):
    # 对外入口：按配置把搜索结果（数据 JSON + 树可视化 HTML）导出并登记到日志系统
    if cfg.export_search_results:
        # 配置开启导出时才执行
        # 取实验名称
        exp_name = cfg.exp_name
        # 生成微秒级时间戳字符串，保证文件名唯一
        time = datetime.now().strftime("%Y%m%d%H%M%S%f")
        # 拼出默认的树可视化 HTML 输出路径
        default_tree_path = f"{logger.cfg.logger.output_dir}/{exp_name}_{time}_{search_method}_tree.html"
        # 拼出默认的搜索数据 JSON 输出路径
        default_data_path = f"{logger.cfg.logger.output_dir}/{exp_name}_{time}_{search_method}_search_data.json"
        # 创建搜索导出器
        search_exporter = SearchExporter(journal, cfg)
        # 执行导出，返回搜索数据
        search_data = search_exporter.gather_and_export_search_results(default_tree_path, default_data_path)
        # 记录导出完成日志
        logger.info(f"Visualisation exported to {default_data_path} and {default_tree_path}.", LogEvent.SOLVER)
        # 取数据文件的绝对路径
        data_path = Path(default_data_path).absolute()
        # 把数据文件登记到日志系统（便于追溯）
        logger.log_file(data_path)

        # 读取搜索数据中的树可视化路径
        tree_path = search_data.get("tree_path")
        if tree_path:
            # 生成了树可视化时转为绝对路径
            tree_path = Path(tree_path).absolute()
            # 登记树文件
            logger.log_file(tree_path)
            with tree_path.open(encoding="utf-8") as tree_file:
                # 以 wandb.Html 形式把树可视化记录进实验日志
                logger.log({"tree_vis": wandb.Html(tree_file)}, LogEvent.SOLVER)
        else:
            # 未生成树可视化时仅记录说明日志
            logger.info("Tree visualisation was not generated; exported search data only.", LogEvent.SOLVER)


def test_export_and_reconstruct_search_data():
    # Create a simple test journal with multiple nodes and branches
    # 端到端自测：构建多分支日志 -> 导出 -> 重建 -> 逐项校验一致性
    # 创建空日志
    journal = Journal()

    # Create root node
    # 创建根节点（初始起草方案）
    root = Node(
        code="def hello(): print('hello')",
        plan="Initial implementation",
        analysis="Works fine",
        metric=MetricValue(value=0.8, info={"accuracy": 0.8}, maximize=True),
        is_buggy=False,
        messages=[{"role": "user", "content": "Write a hello function"}],
        operator_used="initial_solution",
    )
    # 根节点入日志
    journal.append(root)

    # Create first improvement child
    # 创建第一个改进子节点
    child1 = Node(
        code="def hello(): print('hello world')",
        plan="Improved version",
        analysis="Better output",
        metric=MetricValue(value=0.9, info={"accuracy": 0.9}, maximize=True),
        is_buggy=False,
        messages=[{"role": "user", "content": "Improve the function"}],
        operator_used="improve",
        parents=[root],
    )
    # 子节点入日志并挂接到父节点
    journal.append(child1)

    # Create a buggy child
    # 创建一个有 bug 的子节点（代码缺冒号）
    buggy_child = Node(
        code="def hello() print('hello world')",  # Missing colon
        plan="Attempt to simplify",
        analysis="Syntax error",
        metric=MetricValue(value=0.0, info={"error": "syntax"}, maximize=True),
        is_buggy=True,
        messages=[{"role": "user", "content": "Simplify the function"}],
        operator_used="improve",
        parents=[child1],
    )
    # 有 bug 的子节点入日志
    journal.append(buggy_child)

    # Create debug fix for buggy child
    # 创建针对有 bug 子节点的调试修复节点
    debug_fix = Node(
        code="def hello(): print('hello world')",
        plan="Fix syntax error",
        analysis="Fixed missing colon",
        metric=MetricValue(value=0.9, info={"accuracy": 0.9}, maximize=True),
        is_buggy=False,
        messages=[{"role": "user", "content": "Fix the syntax error"}],
        operator_used="debug",
        parents=[buggy_child],
    )
    # 调试修复节点入日志
    journal.append(debug_fix)

    # Create alternative improvement branch
    # 创建另一条改进分支（参数化版本）
    alt_child = Node(
        code="def hello(name='world'): print(f'hello {name}')",
        plan="Add parameter",
        analysis="Added name parameter",
        metric=MetricValue(value=0.95, info={"accuracy": 0.95}, maximize=True),
        is_buggy=False,
        messages=[{"role": "user", "content": "Add parameter support"}],
        operator_used="improve",
        parents=[child1],
    )
    # 备选分支节点入日志
    journal.append(alt_child)

    # Create final improvement
    # 创建最终改进节点（加类型注解，指标最高）
    final_node = Node(
        code="def hello(name: str = 'world') -> None: print(f'hello {name}')",
        plan="Add type hints",
        analysis="Added type annotations",
        metric=MetricValue(value=1.0, info={"accuracy": 1.0}, maximize=True),
        is_buggy=False,
        messages=[{"role": "user", "content": "Add type hints"}],
        operator_used="improve",
        parents=[alt_child],
    )
    # 最终节点入日志
    journal.append(final_node)

    # Create test config with required fields
    # 创建包含必需字段的测试配置
    cfg = OmegaConf.create({"test_key": "test_value", "exp_name": "test_experiment", "export_search_results": True})

    # Export the search data using temporary directory without context manager
    # 用临时目录承载导出文件（手动在 finally 中清理）
    temp_dir = Path(tempfile.mkdtemp())
    try:
        # 拼出搜索数据输出文件路径
        output_file = temp_dir / "search_data.json"
        # 拼出树可视化输出文件路径
        tree_path = temp_dir / "tree.html"

        # Export original journal
        # 导出原始日志
        exporter = SearchExporter(journal, cfg)
        search_data = exporter.gather_and_export_search_results(tree_path=tree_path, output_file=output_file)

        # Verify the file was created
        # 断言导出文件确实存在
        assert output_file.exists()

        # Load the data back
        # 把写出的 JSON 读回内存
        with output_file.open() as f:
            loaded_data = json.load(f)

        # Reconstruct journal from the loaded data
        # 由读回的数据重建日志对象
        reconstructed_journal = Journal.from_export_data(loaded_data)

        # Generate visualizations for both journals
        # 为原始与重建日志分别生成树可视化
        orig_vis_path = temp_dir / "original_tree.html"
        recon_vis_path = temp_dir / "reconstructed_tree.html"

        # 生成原始日志的树可视化
        tree_export.generate(cfg, journal, str(orig_vis_path))
        # 生成重建日志的树可视化
        tree_export.generate(cfg, reconstructed_journal, str(recon_vis_path))

        # Compare the visualization files if they exist
        if orig_vis_path.exists() and recon_vis_path.exists():
            # 两个可视化文件都存在时逐字符比较
            with orig_vis_path.open("r") as f1, recon_vis_path.open("r") as f2:
                # 读取原始可视化内容
                orig_vis = f1.read()
                # 读取重建可视化内容
                recon_vis = f2.read()
                # 断言两者完全一致（证明重建无损）
                assert orig_vis == recon_vis, "Visualizations differ between original and reconstructed journals"

        # Verify the reconstruction
        # 断言重建后节点数量与原始一致（6 个）
        assert len(reconstructed_journal.nodes) == len(journal.nodes) == 6, "Journal should have exactly 6 nodes"

        # Check node properties are preserved
        # 逐节点校验基础属性在导出-重建往返后保持不变
        for orig_node, recon_node in zip(journal.nodes, reconstructed_journal.nodes):
            # 校验代码一致
            assert orig_node.code == recon_node.code, f"Code mismatch for node {orig_node.id}"
            # 校验方案一致
            assert orig_node.plan == recon_node.plan, f"Plan mismatch for node {orig_node.id}"
            # 校验步数一致
            assert orig_node.step == recon_node.step, f"Step mismatch for node {orig_node.id}"
            # 校验唯一 ID 一致
            assert orig_node.id == recon_node.id, f"ID mismatch for node {orig_node.id}"
            # 校验指标数值一致
            assert orig_node.metric.value == recon_node.metric.value, f"Metric value mismatch for node {orig_node.id}"
            # 校验指标附属信息一致
            assert orig_node.metric.info == recon_node.metric.info, f"Metric info mismatch for node {orig_node.id}"
            # 校验 bug 标记一致
            assert orig_node.is_buggy == recon_node.is_buggy, f"Buggy status mismatch for node {orig_node.id}"
            # 校验消息历史一致
            assert orig_node.messages == recon_node.messages, f"Messages mismatch for node {orig_node.id}"
            # 校验所用算子一致
            assert orig_node.operator_used == recon_node.operator_used, f"Operator mismatch for node {orig_node.id}"

        # Get reconstructed nodes for relationship testing
        # 按顺序解包重建后的 6 个节点，便于校验树关系
        [
            root_recon,
            child1_recon,
            buggy_recon,
            debug_recon,
            alt_child_recon,
            final_recon,
        ] = reconstructed_journal.nodes

        # Check relationships with detailed messages
        # 校验父子关系结构是否符合原树
        # 根节点应无父节点
        assert root_recon.parents is None, "Root node should have no parents"
        # 根节点应恰有一个孩子
        assert len(root_recon.children) == 1, "Root node should have exactly one child"
        # 根的孩子应是 child1
        assert child1_recon in root_recon.children, "Root's child should be child1"

        # child1 应恰有一个父节点
        assert len(child1_recon.parents) == 1, "child1 should have exactly one parent"
        # child1 的父节点应是根
        assert root_recon in child1_recon.parents, "child1's parent should be root"
        # child1 应有两个孩子
        assert len(child1_recon.children) == 2, "child1 should have exactly two children"
        # child1 的孩子应包含 buggy_child
        assert buggy_recon in child1_recon.children, "child1's children should include buggy_child"
        # child1 的孩子应包含 alt_child
        assert alt_child_recon in child1_recon.children, "child1's children should include alt_child"

        # buggy_child 应恰有一个孩子
        assert len(buggy_recon.children) == 1, "buggy_child should have exactly one child"
        # 其孩子应是 debug_fix
        assert debug_recon in buggy_recon.children, "buggy_child's child should be debug_fix"
        # buggy_child 应恰有一个父节点
        assert len(buggy_recon.parents) == 1, "buggy_child should have exactly one parent"
        # 其父节点应是 child1
        assert child1_recon in buggy_recon.parents, "buggy_child's parent should be child1"

        # debug_fix 应是叶子
        assert len(debug_recon.children) == 0, "debug_fix should have no children"
        # 其父节点应是 buggy_child
        assert debug_recon.parents == [buggy_recon], "debug_fix's parent should be buggy_child"

        # alt_child 应恰有一个孩子
        assert len(alt_child_recon.children) == 1, "alt_child should have exactly one child"
        # 其孩子应是 final_node
        assert final_recon in alt_child_recon.children, "alt_child's child should be final_node"
        # alt_child 应恰有一个父节点
        assert len(alt_child_recon.parents) == 1, "alt_child should have exactly one parent"
        # 其父节点应是 child1
        assert child1_recon in alt_child_recon.parents, "alt_child's parent should be child1"

        # final_node 应是叶子
        assert len(final_recon.children) == 0, "final_node should have no children"
        # 其父节点应是 alt_child
        assert final_recon.parents == [alt_child_recon], "final_node's parent should be alt_child"

        # Check metric values and ordering
        # 校验各节点指标数值逐一对得上
        assert root_recon.metric.value == 0.8, "Root node should have metric value 0.8"
        assert child1_recon.metric.value == 0.9, "child1 should have metric value 0.9"
        assert buggy_recon.metric.value == 0.0, "buggy_child should have metric value 0.0"
        assert debug_recon.metric.value == 0.9, "debug_fix should have metric value 0.9"
        assert alt_child_recon.metric.value == 0.95, "alt_child should have metric value 0.95"
        assert final_recon.metric.value == 1.0, "final_node should have metric value 1.0"

        # Check stage names with detailed messages
        # 校验各节点的阶段判定（draft/debug/improve）
        assert root_recon.stage_name == "draft", "Root node should be in 'draft' stage"
        assert child1_recon.stage_name == "improve", "child1 should be in 'improve' stage"
        assert buggy_recon.stage_name == "improve", "buggy_child should be in 'improve' stage"
        assert debug_recon.stage_name == "debug", "debug_fix should be in 'debug' stage"
        assert alt_child_recon.stage_name == "improve", "alt_child should be in 'improve' stage"
        assert final_recon.stage_name == "improve", "final_node should be in 'improve' stage"

        # Check debug depths with detailed messages
        # 校验各节点的连续调试深度
        assert root_recon.debug_depth == 0, "Root node should have debug_depth 0"
        assert child1_recon.debug_depth == 0, "child1 should have debug_depth 0"
        assert buggy_recon.debug_depth == 0, "buggy_child should have debug_depth 0"
        assert debug_recon.debug_depth == 1, "debug_fix should have debug_depth 1"
        assert alt_child_recon.debug_depth == 0, "alt_child should have debug_depth 0"
        assert final_recon.debug_depth == 0, "final_node should have debug_depth 0"

        # Check leaf status with detailed messages
        # 校验各节点的叶子状态
        assert not root_recon.is_leaf, "Root node should not be a leaf"
        assert not child1_recon.is_leaf, "child1 should not be a leaf"
        assert not buggy_recon.is_leaf, "buggy_child should not be a leaf"
        assert debug_recon.is_leaf, "debug_fix should be a leaf"
        assert not alt_child_recon.is_leaf, "alt_child should not be a leaf"
        assert final_recon.is_leaf, "final_node should be a leaf"

        # Check best node identification
        # 校验最优节点的选取结果
        # 取原始日志的最优节点
        best_node = journal.get_best_node()
        # 打印最优节点与最终节点的 ID/代码/指标以便人工核对
        print(f"Original best node ID: {best_node.id}")
        print(f"Final node ID: {final_node.id}")
        print(f"Original best node code: {best_node.code}")
        print(f"Final node code: {final_node.code}")
        print(f"Original best node metric: {best_node.metric.value}")
        print(f"Final node metric: {final_node.metric.value}")

        # 最优节点的代码应等于 final_node
        assert best_node.code == final_node.code, "Best node code doesn't match final node"
        # 最优指标应为 1.0 且与 final_node 一致
        assert best_node.metric.value == final_node.metric.value == 1.0, "Best node metric doesn't match final node"
        # 最优节点的 ID 应等于 final_node
        assert best_node.id == final_node.id, "Best node ID doesn't match final node"

        # Check reconstructed journal's best node
        # 取重建日志的最优节点并打印信息
        recon_best = reconstructed_journal.get_best_node()
        print(f"Reconstructed best node ID: {recon_best.id}")
        print(f"Reconstructed best node code: {recon_best.code}")
        print(f"Reconstructed best node metric: {recon_best.metric.value}")

        # Additional verification tests:

        # Test metric info preservation
        # 校验指标附属信息的保真性
        assert root_recon.metric.info == {"accuracy": 0.8}, "Metric info not preserved for root node"
        assert buggy_recon.metric.info == {"error": "syntax"}, "Metric info not preserved for buggy node"

        # Test creation time preservation
        # 校验创建时间戳的保真性
        assert root_recon.ctime == root.ctime, "Creation time not preserved for root node"
        assert final_recon.ctime == final_node.ctime, "Creation time not preserved for final node"

        # Test execution information preservation
        # 校验执行信息的保真性
        # 终端输出一致
        assert root_recon._term_out == root._term_out, "Terminal output not preserved"
        # 执行耗时一致
        assert root_recon.exec_time == root.exec_time, "Execution time not preserved"
        # 退出码一致
        assert root_recon.exit_code == root.exit_code, "Exception code not preserved"

        # Test message history preservation
        # 逐节点校验消息历史的保真性
        for orig_node, recon_node in zip(journal.nodes, reconstructed_journal.nodes):
            assert recon_node.messages == orig_node.messages, f"Messages not preserved for node {orig_node.id}"

        # Test operator preservation
        # 校验算子信息的保真性
        assert root_recon.operator_used == "initial_solution", "Operator not preserved for root node"
        assert debug_recon.operator_used == "debug", "Operator not preserved for debug node"
        assert final_recon.operator_used == "improve", "Operator not preserved for final node"

        # Test step numbering preservation and continuity
        # 校验 step 序号连续且保持原序
        steps = [node.step for node in reconstructed_journal.nodes]
        assert steps == list(range(len(steps))), "Step numbering not continuous or preserved"

        # Test UUID preservation
        # 逐节点校验唯一 ID 的保真性
        for orig_node, recon_node in zip(journal.nodes, reconstructed_journal.nodes):
            assert orig_node.id == recon_node.id, f"UUID not preserved for node {orig_node.id}"

        # Test plan preservation
        # 校验方案文本的保真性
        assert root_recon.plan == "Initial implementation", "Plan not preserved for root node"
        assert final_recon.plan == "Add type hints", "Plan not preserved for final node"

        # Test analysis preservation
        # 校验分析文本的保真性
        assert root_recon.analysis == "Works fine", "Analysis not preserved for root node"
        assert buggy_recon.analysis == "Syntax error", "Analysis not preserved for buggy node"

        # Test that the reconstructed journal maintains the same order
        # 校验重建后节点顺序与原始完全一致
        for i, (orig_node, recon_node) in enumerate(zip(journal.nodes, reconstructed_journal.nodes)):
            assert orig_node.step == recon_node.step == i, f"Node order not preserved at position {i}"

        # Test that the search data contains all required fields
        # 校验搜索数据包含全部必需字段
        required_fields = {"nodes", "config", "tree_path", "solution"}
        assert all(field in search_data for field in required_fields), "Missing required fields in search data"

        # Test that the config is properly preserved
        # 校验配置在导出数据中的保真性
        assert search_data["config"]["test_key"] == "test_value", "Config values not preserved"
        assert search_data["config"]["exp_name"] == "test_experiment", "Config values not preserved"

        # Test that the best solution is properly preserved
        # 校验最优方案代码的保真性
        assert search_data["solution"] == final_node.code, "Best solution code not preserved"

        # 全部校验通过后打印成功提示
        print("All reconstruction tests passed successfully!")

    finally:
        # Clean up temporary directory
        # 无论成败都清理临时目录
        import shutil

        # 递归删除临时目录及其全部内容
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # 直接运行本文件时执行导出-重建自测
    test_export_and_reconstruct_search_data()
