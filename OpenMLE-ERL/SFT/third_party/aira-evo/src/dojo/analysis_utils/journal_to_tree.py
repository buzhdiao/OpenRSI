# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 concurrent.futures：提供进程池，用于并行处理多个实验的树可视化
import concurrent.futures
# 导入 datetime：解析 JSONL 日志中的 ISO 时间戳
from datetime import datetime
# 导入 json：读写 JSON 文件
import json
# 导入 curses：终端交互式菜单（TUI）库
import curses
# 导入 Path：面向对象的文件路径操作
from pathlib import Path
# 导入类型标注工具：Any/Dict/List/Optional/Union
from typing import Any, Dict, List, Optional, Union

# 导入 concurrent 模块本体（as_completed 等以模块属性方式使用）
import concurrent
# 导入 jsonlines：逐行读写 JSONL 文件的库
import jsonlines

# 导入树导出工具：把 Journal 对象渲染成 HTML 可视化树
from dojo.core.solvers.utils import tree_export
# 导入 Journal 类：智能体研究日志（journal）的数据结构与加载器
from dojo.core.solvers.utils.journal import Journal
# 导入运行配置类：从 dojo_config.json 恢复实验配置
from dojo.config_dataclasses.run import RunConfig


# 把 JSONL 格式的研究日志解析成节点字典列表；seconds_cutoff 用于按时间窗口截断
def journal_log_into_json(
    file_path: Union[str, Path],
    seconds_cutoff: Optional[float | int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse a JSONL journal and return all nodes whose timestamp is within
    *seconds_cutoff* seconds of the first entry.

    Parameters
    ----------
    file_path
        Path (or str) to a *.jsonl* file whose lines look like::
            {"timestamp": "2025-05-06T10:32:19.614691", "data": {...}}

    seconds_cutoff
        ``None`` → keep the whole file (default).
        ``int`` / ``float`` → keep entries where
        ``(entry_ts - first_ts).total_seconds() <= seconds_cutoff``.
    """
    # 统一将输入路径转为 Path 对象
    file_path = Path(file_path)
    # 若给定了时间窗口则必须非负，否则参数不合法
    if seconds_cutoff is not None and seconds_cutoff < 0:
        raise ValueError("seconds_cutoff must be non-negative or None")

    # 存放解析出的节点列表
    nodes: List[Dict[str, Any]] = []
    # 记录首条日志的时间戳，作为时间窗口的基准点
    first_ts: Optional[datetime] = None

    # 以 jsonlines 逐行打开 JSONL 日志文件
    with jsonlines.open(file_path) as reader:
        # 迭代每一条日志记录
        for entry in reader:
            # 取出该条目的时间戳字符串
            ts_str = entry.get("timestamp")
            # 取出该条目的数据体（节点内容）
            data = entry.get("data")

            # Skip malformed lines
            # 缺时间戳或缺数据体的行视为坏行，直接跳过
            if ts_str is None or data is None:
                continue

            # 把 ISO 格式时间字符串解析为 datetime 对象（"YYYY‑MM‑DDTHH:MM:SS.ssssss"）
            ts = datetime.fromisoformat(ts_str)  # "YYYY‑MM‑DDTHH:MM:SS.ssssss"

            # 第一条有效记录的时间作为窗口起点
            if first_ts is None:
                first_ts = ts

            # Stop if we've exceeded the requested window
            # 当前记录距首条记录已超出时间窗口时，提前结束（日志按时间有序）
            if seconds_cutoff is not None and (ts - first_ts).total_seconds() > seconds_cutoff:
                break

            # 浅拷贝数据体，避免改动调用方字典（node 与 data 共享嵌套引用）
            node = dict(data)  # shallow copy – don’t mutate caller’s dict
            # 保留原始格式的时间戳字符串写入节点
            node["timestamp"] = ts_str  # keep the original format
            # 收集该节点
            nodes.append(node)

    # 以 {"nodes": [...]} 的结构返回全部节点
    return {"nodes": nodes}


# 读取 JSONL 日志并解析为结构化树后另存为 JSON 文件（默认存在实验目录下）
def save_journal_log_as_json(
    file_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    output_filename: str = "journal.json",
    seconds_cutoff: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Reads a journal JSONL file, parses it into a structured tree, and saves it as a JSON file.
    The output file is saved in the experiment folder where the JSONL file is located.

    Args:
        file_path (Union[str, Path]): Path to the input journal JSONL file.
        output_dir (Optional[Union[str, Path]]): Directory where the output JSON file should be saved.
            Defaults to the experiment directory where the JSONL file is located.
        output_filename (str): Name of the output JSON file. Defaults to "journal.json".
        seconds_cutoff (Optional[float]): Time window in seconds to limit the entries.
            If None, all entries are included. Defaults to None.

    Returns:
        Dict[str, Any]: The parsed tree structure saved to the file.
    """
    # 统一转为 Path 对象
    file_path = Path(file_path)
    # 日志位于 <实验目录>/json/JOURNAL.jsonl，向上两级即实验目录
    experiment_dir = file_path.parent.parent  # Navigate up to the experiment folder
    # 未显式指定输出目录时，默认输出到实验目录
    output_dir = Path(output_dir) if output_dir else experiment_dir
    # 递归创建输出目录（已存在则忽略）
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    # 拼出输出文件完整路径
    output_file = output_dir / output_filename

    # 调用上面的解析函数得到结构化节点树（可按时间窗口截断）
    parsed_json = journal_log_into_json(file_path, seconds_cutoff=seconds_cutoff)

    # 以 UTF-8 写出格式化（缩进 2）的 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parsed_json, f, indent=2)

    # 打印保存成功信息
    print(f"Journal successfully saved to {output_file}")

    # 返回解析后的树结构（字典）
    return parsed_json


# 由内存中的 Journal 对象生成并导出树可视化（HTML 页面）
def journal_into_tree(
    journal: Journal, exp_folder: Union[str, Path], tree_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Generates and exports a visualization of the journal tree from a journal instance.

    Args:
        journal (Journal): The journal instance to visualize.
        tree_path (Optional[Union[str, Path]]): Path where the visualization will be saved.
            Defaults to the experiment directory where the JSONL file is located.
        experiment_name (str): Name of the experiment for labeling purposes. Defaults to "tree".
    """
    # 未指定输出路径时，默认写到当前目录的 tree.html
    if tree_path is None:
        tree_path = Path("./tree.html")

    # 统一转为 Path 对象
    exp_folder = Path(exp_folder)
    # Load Experiment config
    # 实验配置文件路径：<实验目录>/dojo_config.json
    config_path = exp_folder / "dojo_config.json"

    # 配置文件缺失则无法还原实验上下文，直接报错
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # 从 JSON 恢复运行配置
    cfg = RunConfig.load_from_json(config_path)

    # 用树导出工具按配置与日志对象生成可视化文件
    tree_export.generate(cfg, journal, str(tree_path))


# 由已结构化的 JSON 数据（journal.json）生成树可视化
def json_into_tree(
    json_data: Dict[str, Any], exp_folder: Union[str, Path], tree_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Generates and exports a visualization of the journal tree from JSON data.

    Args:
        json_data (dict): The json data representing the journal instance to visualize.
        tree_path (Optional[Union[str, Path]]): Path where the visualization will be saved.
            Defaults to the experiment directory where the JSONL file is located.
        experiment_name (str): Name of the experiment for labeling purposes. Defaults to "tree".
    """
    # Load journal from structured JSON and generate tree visualization
    # 从结构化 JSON 数据重建 Journal 对象
    journal = Journal.from_export_data(json_data)

    # 复用 journal_into_tree 完成可视化导出
    journal_into_tree(journal, exp_folder, tree_path)


# 单个实验目录的端到端处理：JSONL -> journal.json -> tree.html
def log_to_tree(exp_folder: Path):
    # 实验日志 JSONL 的固定位置：<实验目录>/json/JOURNAL.jsonl
    log_dir = exp_folder / "json" / "JOURNAL.jsonl"
    # 解析并保存为 journal.json，同时拿到结构化数据
    json_data = save_journal_log_as_json(log_dir)
    # 树可视化的输出文件路径
    output_file = exp_folder / "tree.html"
    # 由 JSON 数据生成 HTML 树
    json_into_tree(json_data, exp_folder, output_file)


# 并行处理元实验目录下的所有实验，为每个实验生成树可视化
def visualise_all_trees(meta_experiment_path: Path):
    """Concurrently processes multiple experiments and visualises trees."""
    # 统一转为 Path 对象
    meta_experiment_path = Path(meta_experiment_path)

    # 元实验目录不存在则直接提示并返回
    if not meta_experiment_path.exists():
        print(f"Experiments path does not exist: {meta_experiment_path}")
        return

    # 列出元实验目录下所有子目录（每个子目录对应一个实验）
    exp_fold = [folder for folder in meta_experiment_path.iterdir() if folder.is_dir()]

    # 记录处理过程中是否发生过异常
    errors_occurred = False

    # 用进程池并行执行各实验的 log_to_tree
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 为每个实验目录提交一个任务，记录 future 到目录的映射
        future_to_experiment = {executor.submit(log_to_tree, log_file): log_file for log_file in exp_fold}
        # 按完成顺序收集结果
        for future in concurrent.futures.as_completed(future_to_experiment):
            # 由 future 反查对应的实验目录
            log_file = future_to_experiment[future]
            try:
                # 等待该任务完成（异常会在此抛出）
                future.result()
                # 打印处理成功的实验
                print(f"Processed experiment: {log_file}")
            except Exception as exc:
                # 标记发生过错误并打印异常
                errors_occurred = True
                print(f"Experiment {log_file} generated an exception: {exc}")

    # 汇总提示：有错误或全部成功
    if errors_occurred:
        print("\nProcessing completed with errors.")
    else:
        print("\nProcessing completed successfully.")


# curses 交互主菜单：上下键选择、回车确认
def main_menu(stdscr):
    # 隐藏终端光标
    curses.curs_set(0)
    # 当前高亮选中的菜单行索引
    current_row = 0
    # 菜单项文本
    menu = ["Visualize Single Experiment", "Visualize All Experiments", "Exit"]

    # 主循环：绘制菜单并响应按键
    while True:
        # 每轮清屏重绘
        stdscr.clear()
        # 获取终端的行数与列数
        h, w = stdscr.getmaxyx()
        # 居中绘制每个菜单项
        for idx, row in enumerate(menu):
            # 水平方向居中的起始列
            x = w // 2 - len(row) // 2
            # 垂直方向整体居中后按序号偏移的行
            y = h // 2 - len(menu) // 2 + idx
            # 当前选中项使用反色高亮显示
            if idx == current_row:
                # 开启颜色对 1 的属性
                stdscr.attron(curses.color_pair(1))
                # 绘制菜单项文本
                stdscr.addstr(y, x, row)
                # 关闭高亮属性
                stdscr.attroff(curses.color_pair(1))
            else:
                # 未选中项按普通样式绘制
                stdscr.addstr(y, x, row)

        # 刷新屏幕使绘制生效
        stdscr.refresh()

        # 阻塞等待用户按键
        key = stdscr.getch()

        # 上方向键：选中行上移（不越界）
        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        # 下方向键：选中行下移（不越界）
        elif key == curses.KEY_DOWN and current_row < len(menu) - 1:
            current_row += 1
        # 回车键：执行当前选中项
        elif key in [curses.KEY_ENTER, ord("\n")]:
            # 第一项：可视化单个实验
            if current_row == 0:
                # 暂时退出 curses 界面以便接收键盘输入
                curses.endwin()
                # 提示输入单个日志所在目录
                log_dir = input("Enter path to single journal log: ")
                # 对该目录执行树生成
                log_to_tree(Path(log_dir))
                # 处理完成后暂停等待回车
                input("\nPress Enter to continue...")
            # 第二项：并行可视化全部实验
            elif current_row == 1:
                # 退出 curses 界面
                curses.endwin()
                # 提示输入元实验目录
                meta_path = input("Enter path to meta-experiment directory: ")
                # 并行处理该目录下的所有实验
                visualise_all_trees(Path(meta_path))
                # 完成后暂停
                input("\nPress Enter to continue...")
            # 第三项：退出菜单循环
            elif current_row == 2:
                break


# curses 初始化：注册颜色对后进入主菜单
def setup_curses(stdscr):
    # 定义颜色对 1：黑字白底，用于菜单高亮
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
    # 进入交互主菜单
    main_menu(stdscr)


# 命令行入口：无参数时进入交互菜单，带参数则直接处理
def main():
    # argparse
    # 延迟导入 argparse（仅命令行运行时需要）
    import argparse

    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Visualize journal logs.")
    # 位置参数：日志目录（可选）
    parser.add_argument("log_dir", type=str, help="Path to the journal log directory", nargs="?")
    # 开关：处理目录下所有实验
    parser.add_argument("--all", action="store_true", help="Visualize all experiments")
    # 解析命令行参数
    args = parser.parse_args()

    # 未提供日志目录时，启动 curses 交互菜单
    if not args.log_dir:
        curses.wrapper(lambda stdscr: setup_curses(stdscr))
        return

    # 把命令行目录转为 Path
    log_dir = Path(args.log_dir)
    # 指定 --all 时并行处理全部实验
    if args.all:
        visualise_all_trees(log_dir)
    else:
        # 否则只处理单个实验目录
        log_to_tree(log_dir)


# 作为脚本直接运行时执行主入口
if __name__ == "__main__":
    main()
