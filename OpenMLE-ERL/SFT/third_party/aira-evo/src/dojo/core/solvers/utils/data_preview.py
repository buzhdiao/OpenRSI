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
Contains functions to manually generate a textual preview of some common file types (.csv, .json,..) for the agent.
"""

# 导入 json 模块，用于解析 JSON/JSONL 文件
import json
# 导入 pathlib 的 Path，用于文件路径操作
from pathlib import Path

# 导入 humanize，用于把字节数转成人类可读的大小描述
import humanize
# 导入 pandas，用于读取并预览 CSV 数据
import pandas as pd
# 导入 genson 的 SchemaBuilder，用于从 JSON 数据自动推断 JSON Schema
from genson import SchemaBuilder
# 导入 pandas 的数值类型判断函数
from pandas.api.types import is_numeric_dtype

# these files are treated as code (e.g. markdown wrapped)
# 视为"代码类"的文件扩展名集合（预览时用代码围栏包裹）
code_files = {".py", ".sh", ".yaml", ".yml", ".md", ".html", ".xml", ".log", ".rst"}
# we treat these files as text (rather than binary) files
# 视为纯文本（而非二进制）的文件扩展名集合（在代码类基础上再并入常见数据文本格式）
plaintext_files = {".txt", ".csv", ".json", ".tsv"} | code_files


def get_file_len_size(f: Path) -> tuple[int, str]:
    """
    Calculate the size of a file (#lines for plaintext files, otherwise #bytes)
    Also returns a human-readable string representation of the size.
    """
    # 文件不存在时返回 (0, "missing")
    if not f.exists():
        return 0, "missing"
    if f.suffix in plaintext_files:
        # 纯文本文件按"行数"计量
        try:
            # 逐行计数统计总行数
            num_lines = sum(1 for _ in open(f))
        except FileNotFoundError:
            # 读取期间文件被删除则返回缺失
            return 0, "missing"
        # 返回行数与可读描述
        return num_lines, f"{num_lines} lines"
    else:
        # 二进制等其他文件按"字节数"计量
        try:
            # 读取文件大小（字节）
            s = f.stat().st_size
        except FileNotFoundError:
            # 读取期间文件被删除则返回缺失
            return 0, "missing"
        # 返回字节数与人类可读的大小（如 1.2 kB）
        return s, humanize.naturalsize(s)


def file_tree(path: Path, depth=0, skip_hidden=True) -> str:
    """Generate a tree structure of files in a directory"""
    # 递归生成目录的树状文本预览
    result = []
    if not Path(path).exists():
        # 目录不存在时输出缩进的缺失标记
        return f"{' ' * depth * 4}(missing)"
    if skip_hidden:
        # 跳过隐藏项：先收集当前层的非隐藏文件
        files = [p for p in Path(path).iterdir() if not p.is_dir() and not p.name.startswith(".")]
        # 再收集非隐藏子目录
        dirs = [p for p in Path(path).iterdir() if p.is_dir() and not p.name.startswith(".")]
    else:
        # 不跳过隐藏项：收集全部文件
        files = [p for p in Path(path).iterdir() if not p.is_dir()]
        # 收集全部子目录
        dirs = [p for p in Path(path).iterdir() if p.is_dir()]
    # 文件很多时少展示一些（最多 4 个），否则最多展示 8 个
    max_n = 4 if len(files) > 30 else 8
    # 按名称排序后逐个列出文件及大小描述
    for p in sorted(files)[:max_n]:
        result.append(f"{' ' * depth * 4}{p.name} ({get_file_len_size(p)[1]})")
    if len(files) > max_n:
        # 超出展示上限时追加"还有 N 个文件"的提示
        result.append(f"{' ' * depth * 4}... and {len(files) - max_n} other files")

    # 递归展开各子目录
    for p in sorted(dirs):
        # 先列出目录名（带斜杠标识）
        result.append(f"{' ' * depth * 4}{p.name}/")
        # 再递归输出该目录的树
        result.append(file_tree(p, depth + 1))

    # 用换行拼接成完整树状文本
    return "\n".join(result)


def _walk(path: Path, skip_hidden=True):
    """Recursively walk a directory (analogous to os.walk but for pathlib.Path)"""
    # 递归遍历目录并逐个产出文件路径（pathlib 版 os.walk）
    if not Path(path).exists():
        # 目录不存在时直接返回（不产出任何内容）
        return
    # 按名称排序遍历当前层条目
    for p in sorted(Path(path).iterdir()):
        # Skip hidden files and directories
        # 跳过以点开头的隐藏文件/目录
        if p.name.startswith("."):
            continue

        # 跳过已不存在的条目（竞态保护）
        if not p.exists():
            continue

        if p.is_dir():
            # 子目录则递归深入
            yield from _walk(p)
            continue

        # 文件则产出该路径
        yield p


def preview_csv(p: Path, file_name: str, simple=True) -> str:
    """Generate a textual preview of a csv file

    Args:
        p (Path): the path to the csv file
        file_name (str): the file name to use in the preview
        simple (bool, optional): whether to use a simplified version of the preview. Defaults to True.

    Returns:
        str: the textual preview
    """
    # 用 pandas 读入 CSV 为 DataFrame
    df = pd.read_csv(p)

    # 预览文本行列表
    out = []

    # 第一行：报告行数与列数
    out.append(f"-> {file_name} has {df.shape[0]} rows and {df.shape[1]} columns.")

    if simple:
        # 简化模式：只列出列名
        cols = df.columns.tolist()
        # 最多展示 15 列
        sel_cols = 15
        # 拼接前 15 列的列名
        cols_str = ", ".join(cols[:sel_cols])
        # 组装列名说明文本
        res = f"The columns are: {cols_str}"
        if len(cols) > sel_cols:
            # 超出 15 列时追加剩余列数提示
            res += f"... and {len(cols) - sel_cols} more columns"
        # 加入输出列表
        out.append(res)
    else:
        # 详细模式：逐列给出统计信息
        out.append("Here is some information about the columns:")
        # 按列名排序后逐列处理
        for col in sorted(df.columns):
            # 取该列的数据类型
            dtype = df[col].dtype
            # 组装"列名 (类型)"标签
            name = f"{col} ({dtype})"

            # 统计该列缺失值（NaN）数量
            nan_count = df[col].isnull().sum()

            if dtype == "bool":
                # 布尔列：报告 True/False 占比
                v = df[col][df[col].notnull()].mean()
                out.append(f"{name} is {v * 100:.2f}% True, {100 - v * 100:.2f}% False")
            elif df[col].nunique() < 10:
                # 唯一值少于 10 的列：直接列出全部取值
                out.append(f"{name} has {df[col].nunique()} unique values: {df[col].unique().tolist()}")
            elif is_numeric_dtype(df[col]):
                # 数值列：报告最小-最大范围与缺失值个数
                out.append(f"{name} has range: {df[col].min():.2f} - {df[col].max():.2f}, {nan_count} nan values")
            elif dtype == "object":
                # 对象（字符串）列：报告唯一值个数与出现频次最高的 4 个示例值
                out.append(
                    f"{name} has {df[col].nunique()} unique values. Some example values: {df[col].value_counts().head(4).index.tolist()}"
                )

    # 用换行拼接成完整预览文本
    return "\n".join(out)


def preview_json(p: Path, file_name: str):
    """Generate a textual preview of a json file using a generated json schema"""
    # 创建 JSON Schema 推断器
    builder = SchemaBuilder()
    with open(p) as f:
        # 读取第一行并去掉首尾空白
        first_line = f.readline().strip()

        try:
            # 尝试把第一行解析为 JSON 对象
            first_object = json.loads(first_line)

            if not isinstance(first_object, dict):
                # 首行是数组等非对象类型时，按"首行不是 JSON 对象"处理
                raise json.JSONDecodeError("The first line isn't JSON", first_line, 0)

            # if the the next line exists and is not empty, then it is a JSONL file
            # 读第二行：非空则说明是 JSONL（每行一个 JSON）文件
            second_line = f.readline().strip()
            if second_line:
                f.seek(0)  # so reset and read line by line
                # 回到文件头逐行读取并喂给 SchemaBuilder
                for line in f:
                    builder.add_object(json.loads(line.strip()))
            # if it is empty, then it's a single JSON object file
            else:
                # 第二行为空说明是单对象 JSON 文件，只喂首行对象
                builder.add_object(first_object)

        except json.JSONDecodeError:
            # if first line isn't JSON, then it's prettified and we can read whole file
            # 首行不是完整 JSON：说明是格式化（多行缩进）的 JSON，回到文件头整体加载
            f.seek(0)
            builder.add_object(json.load(f))

    # 返回"自动生成的 JSON Schema"预览文本
    return f"-> {file_name} has auto-generated json schema:\n" + builder.to_json(indent=2)


def generate(base_path, include_file_details=True, simple=False):
    """
    Generate a textual preview of a directory, including an overview of the directory
    structure and previews of individual files
    """
    # 生成目录树并用代码围栏包裹
    tree = f"```\n{file_tree(base_path)}```"
    # 预览片段列表以目录树开头
    out = [tree]

    if include_file_details:
        # 需要文件详情时逐文件生成预览
        for fn in _walk(base_path):
            # 计算相对路径作为展示文件名
            file_name = str(fn.relative_to(base_path))

            if fn.suffix == ".csv":
                # CSV 文件用 pandas 预览
                out.append(preview_csv(fn, file_name, simple=simple))
            elif fn.suffix == ".json":
                # JSON 文件用 Schema 预览
                out.append(preview_json(fn, file_name))
            elif fn.suffix in plaintext_files:
                # 其他纯文本文件：行数少于 30 时直接展示全文
                if get_file_len_size(fn)[0] < 30:
                    with open(fn) as f:
                        # 读取整个文件内容
                        content = f.read()
                        if fn.suffix in code_files:
                            # 代码类文件用代码围栏包裹
                            content = f"```\n{content}\n```"
                        # 追加"文件内容"预览
                        out.append(f"-> {file_name} has content:\n\n{content}")

    # 用空行拼接全部预览片段
    result = "\n\n".join(out)

    # if the result is very long we generate a simpler version
    # 结果超过 6000 字符且还是详细模式时，改用简化模式重新生成
    if len(result) > 6_000 and not simple:
        return generate(base_path, include_file_details=include_file_details, simple=True)
    # if still too long, we truncate
    # 简化模式仍超过 6000 字符时直接截断
    if len(result) > 6_000 and simple:
        return result[:6_000] + "\n... (truncated)"

    # 返回最终预览文本
    return result
