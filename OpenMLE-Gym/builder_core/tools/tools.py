# 中文说明:builder_core 感知阶段(Perceive)的 LLM 工具集。
# 在系统中的位置:builder_core/tools 下,是 NodeExecutor.Perceive 工具循环的"手":
# list_directory_contents(列目录,超 50 项自动按扩展名聚合)、get_csv_summary
# (CSV 摘要,限前 32 列+分块统计)、read_txt_md(文本预览,6000 字符截断)、
# save(唯一合法出口:整体覆写 fileinfo.txt)。
# 关键设计:进程级路径绑定(configure_task_paths),读取被锁在任务 raw 目录内
# (防穿越),写入只允许 fileinfo.txt;工具 docstring 是给 LLM 看的功能说明,
# 保持原文不动(它们属于运行期消费的字符串字面量)。

# pandas:CSV 摘要分析
import pandas as pd
# os:列目录/取扩展名
import os
# defaultdict:按扩展名聚合文件
from collections import defaultdict
# Path:路径对象
from pathlib import Path
# tool:langchain 工具装饰器
from langchain_core.tools import tool
# atomic_write_text:fileinfo.txt 的原子覆写
from openmle_gym.common import atomic_write_text

# 进程级任务路径绑定:允许读取的根目录 / 唯一允许写入的目标文件
_READ_ROOT: Path | None = None
_SAVE_PATH: Path | None = None


def configure_task_paths(read_root: str | Path, save_path: str | Path) -> None:
    """把工具的读写范围绑定到一个任务(每个任务跑在独立进程里,互不干扰)。"""
    global _READ_ROOT, _SAVE_PATH
    # 记录读取根目录与保存目标(均取真实路径)
    _READ_ROOT = Path(read_root).resolve()
    _SAVE_PATH = Path(save_path).resolve()


def _resolve_read_path(value: str) -> Path:
    """解析读取路径并强制限定在任务 raw 目录内(防路径穿越)。"""
    # 未配置读取根:直接报错
    if _READ_ROOT is None:
        raise RuntimeError("Task tool read root is not configured")
    candidate = Path(value)
    if not candidate.is_absolute():
        # 相对路径:若首段与根目录同名则按兄弟目录处理,否则挂在根下
        if candidate.parts and candidate.parts[0] == _READ_ROOT.name:
            candidate = _READ_ROOT.parent / candidate
        else:
            candidate = _READ_ROOT / candidate
    # 取真实路径(展开符号链接)
    resolved = candidate.resolve()
    try:
        # 必须仍位于读取根之内
        resolved.relative_to(_READ_ROOT)
    except ValueError as exc:
        # 越界读取:拒绝
        raise PermissionError(f"Path is outside the current task raw directory: {value}") from exc
    return resolved


def _resolve_save_path(value: str) -> Path:
    """解析保存路径:只允许写入当前任务的 fileinfo.txt。"""
    # 未配置保存目标:直接报错
    if _SAVE_PATH is None:
        raise RuntimeError("Task tool save path is not configured")
    resolved = Path(value).resolve()
    if resolved != _SAVE_PATH:
        # 写入其他路径:拒绝
        raise PermissionError(f"save may only write the current task fileinfo.txt: {value}")
    return resolved


@tool
def list_directory_contents(path: str) -> str:
    """Lists the contents of the specified path. 
Automatically aggregates files by extension if the item count is too high. 
    Directories always appear first."""
    # (中文)列目录工具:目录优先;超过 50 项时按扩展名聚合成报表。
    try:
        # 解析并校验读取路径(必须在任务 raw 目录内)
        resolved_path = _resolve_read_path(path)
        all_items = os.listdir(resolved_path)
        if not all_items:
            # 空目录:直接提示
            return f"The directory '{path}' is empty"

        # 目录清单与按扩展名分组的文件表
        directories = []
        files_by_ext = defaultdict(list)

        # 逐项归类
        for item in all_items:
            full_path = resolved_path / item
            if os.path.isdir(full_path):
                directories.append(item)
            else:
                # 无扩展名文件单独标记
                _, ext = os.path.splitext(item)
                ext = ext.lower() if ext else "[no extension]"
                files_by_ext[ext].append(item)
                
        # 目录排序;扩展名排序
        directories.sort()
        sorted_exts = sorted(files_by_ext.keys())

        # 条目总数
        total_count = len(all_items)
        
        if total_count <= 50:
            # 少量条目:完整逐行列出(目录在前)
            output = [f"Listing {path} ({total_count} items):"]
            for d in directories:
                output.append(f"  [Dir]  {d}")
            for ext in sorted_exts:
                for f in sorted(files_by_ext[ext]):
                    output.append(f"  [File] {f}")
            return "\n".join(output)

        else:
            # 大量条目:聚合成报表(目录取样,文件按扩展名计数)
            report = [f"Directory Report: {path} ({total_count} items in total)"]
            report.append("-" * 3)
            
            if directories:
                # 子目录:总数 + 前 5 个样本
                dir_sample = directories[:5]
                report.append(f"Subdirectories: {len(directories)} in total")
                report.append(f"   Samples: {', '.join(dir_sample)}" + ("..." if len(directories) > 5 else ""))
            
            # 文件:按数量降序列出每类扩展名(含 2 个样本)
            report.append(f"Categorized File Report:")
            for ext in sorted(files_by_ext, key=lambda k: len(files_by_ext[k]), reverse=True):
                count = len(files_by_ext[ext])
                samples = ", ".join(files_by_ext[ext][:2])
                report.append(f"   - {ext:12} : {count:3} in total (Samples: {samples}...)")
            
            report.append("-" * 3)
            # 提示已聚合,想看细节请细化路径
            report.append("Too many items; results have been automatically aggregated. To view specific files, please refine the search path")
            
            return "\n".join(report)

    except Exception as e:
        # 任何失败都以错误文本返回(LLM 可读),不抛出
        return f"Error: Unable to open '{path}': {str(e)}"

@tool
def get_csv_summary(csv_path: str) -> dict:
    """Analyzes a CSV file and returns a summary. Limits analysis to the first `max_cols` columns to ensure readability and performance."""
    # (中文)CSV 摘要工具:只分析前 32 列,分块统计行数/空值,附 3 行样例。
    try:
        # 校验读取路径
        resolved_csv = _resolve_read_path(csv_path)
        # 列数上限
        max_cols = 32
        
        # 1. Read header only to determine columns
        # 先只读表头拿全部列名
        full_df_columns = pd.read_csv(resolved_csv, nrows=0).columns.tolist()
        
        # Apply the 32-column limit
        # 应用 32 列上限(超出部分标记截断)
        target_columns = full_df_columns[:max_cols]
        is_truncated = len(full_df_columns) > max_cols

        # 2. Get sample data for the limited columns
        # 取目标列的前 3 行样例
        df_sample = pd.read_csv(resolved_csv, nrows=3, usecols=target_columns)
        
        # Initialize counters
        # 行数与各列空值计数器
        row_count = 0
        null_counts = {col: 0 for col in target_columns}

        # 3. Process file in chunks (only reading target columns for speed)
        # 分块遍历(每块 10 万行,只读目标列提速)
        with pd.read_csv(resolved_csv, usecols=target_columns, chunksize=100000) as reader:
            for chunk in reader:
                row_count += len(chunk)
                # Count nulls for the limited set of columns
                # 累加每块各列空值数
                chunk_nulls = chunk.isnull().sum().to_dict()
                for col, count in chunk_nulls.items():
                    null_counts[col] += count

        # 4. Get data types (from a small sample)
        # 用前 100 行推断各列数据类型
        df_types = pd.read_csv(resolved_csv, nrows=100, usecols=target_columns)
        dtypes_dict = df_types.dtypes.apply(lambda x: str(x)).to_dict()

        # 5. Construct column-specific details
        # 组装每列的类型与空值信息
        column_info = {}
        for col in target_columns:
            column_info[col] = {
                "data_type": dtypes_dict.get(col, "unknown"),
                "null_count": int(null_counts[col])
            }

        # 6. Build final summary
        # 汇总最终摘要(文件信息/规模/列明细/样例)
        summary = {
            "file_info": {
                "filename": os.path.basename(csv_path),
                "columns_analyzed": len(target_columns),
                "is_column_limit_reached": is_truncated
            },
            "shape": {
                "total_rows": row_count,
                "total_columns": len(full_df_columns)
            },
            "column_details": column_info,
            "sample_data": df_sample.to_dict(orient='records')
        }
        
        if is_truncated:
            # 截断说明:完整列描述请看数据集说明
            summary["note"] = f"Only the first {max_cols} columns were analyzed due to the limit. For full description of columns, refer to Dataset Description"
            
        return summary

    except Exception as e:
        # 失败返回错误字典并建议跳过该文件
        return {"error": f"Failed to analyze CSV at {csv_path}: {str(e)}. You may consider skipping this file"}

@tool
def read_txt_md(file_path: str) -> str:
    """Display a bounded preview of a text file. One preview is sufficient for inventory."""
    # (中文)文本预览工具:最多读 6000 字符,截断时明确提示"一次预览即足够"。
    try:
        # 校验读取路径
        resolved_file = _resolve_read_path(file_path)
        # Check if the file exists and is a file before opening
        # 字符数上限
        char_limit = 6000
        with open(resolved_file, "r", encoding="utf-8") as f:
            # Read only up to the char_limit
            # 只读上限内的内容
            content = f.read(char_limit)
            # 再读 1 个字符判断是否被截断
            is_truncated = len(f.read(1)) > 0
            if is_truncated:
                # 截断版本:带一次性读取提示(防止 LLM 反复读同一文件)
                return (
                    f"File {file_path} content (preview truncated to "
                    f"{char_limit} chars; this single preview is sufficient, "
                    "do not read the same path again):\n"
                    f"{content}\n... [End of preview]"
                )
            # 未截断:完整返回
            return f"File {file_path} content:\n{content}"
    except UnicodeDecodeError:
        # 非 UTF-8/二进制:提示跳过
        return f"Error: File {file_path} contains non-UTF-8 characters or binary data. Skip this file"
    except Exception as e:
        # 其他读取错误:提示可跳过
        return f"File {file_path} reading error: {e}. You may consider skipping this file"

@tool
def save(info_path: str, content: str):
    """Fully overwrite the content in fileinfo.txt. The only EXIT of tool use. Call this to save, and terminate the tool use."""
    # (中文)保存工具:工具循环的唯一合法出口——整体覆写 fileinfo.txt 后即结束清点。
    # 校验写入目标(只允许 fileinfo.txt)
    path = _resolve_save_path(info_path)
    # 确保父目录存在
    path.parent.mkdir(parents=True, exist_ok=True)
    # 原子写入:内容 + 分隔线(供增量追加阅读)
    atomic_write_text(path, content + "\n" + "-"*20 + "\n")
    return f"Successfully saved to fileinfo.txt"
