# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 GDM 后端的 LLM 客户端：用于调用大模型分析崩溃日志
from dojo.core.solvers.llm_helpers.backends.gdm import GDMClient
# 导入思考标签解析工具：剥离 <think> 类标签，取最终回答文本
from dojo.core.solvers.utils.response import parse_thinking_tags
# 导入 JSON 输出解析工具：从 LLM 回复中抽取 JSON 对象
from dojo.utils.code_parsing import parse_json_output
# 导入实验目录判断工具：识别某目录是否为一次有效实验
from dojo.utils.experiment_logs import is_experiment
# 导入运行配置类：读取各实验的 dojo_config.json
from dojo.config_dataclasses.run import RunConfig
# 导入 os：提供 SEEK_END 等文件定位常量
import os
# 导入 glob：按模式匹配 slurm 日志文件（本文件中未直接使用该导入）
import glob
# 导入 json：序列化汇总结果
import json
# 导入 re：正则表达式，压缩日志文本
import re
# 导入 Path：面向对象的路径操作
from pathlib import Path
# 导入 OmegaConf：以字典形式快速构造客户端配置
from omegaconf import OmegaConf
# 导入 defaultdict：本文件中预留的带默认值字典（未直接使用）
from collections import defaultdict
# 导入 tqdm：进度条显示
from tqdm import tqdm


# 创建并返回一个 GDM 后端的 LLM 客户端（用于崩溃日志的智能分析）
def create_client():
    # 用 OmegaConf 构造客户端配置：gdm 接口 + gemini-2.0-flash 模型
    client_cfg = OmegaConf.create({"api": "gdm", "model_id": "gemini-2.0-flash", "provider": "gdm"})

    # 按配置实例化 GDM 客户端
    client = GDMClient(client_cfg)

    # 返回客户端实例
    return client


# 从文件尾部近似读取指定数量的 token（按平均字符数估算，并压缩空白）
def get_last_approx_tokens(filepath, approx_token_count=70000, avg_chars_per_token=4):
    """
    Returns approximately the last `approx_token_count` tokens from a file.
    Approximation is based on average characters per token.
    Extra whitespace and newlines are compressed to avoid wasting token space.
    """
    # 把 token 数换算成近似字节数（token 数 * 每 token 平均字符数）
    approx_byte_count = approx_token_count * avg_chars_per_token

    # 以二进制方式打开文件，便于从末尾定位读取
    with open(filepath, "rb") as f:
        # 定位到文件末尾以获取文件大小
        f.seek(0, os.SEEK_END)
        # 读取当前指针位置即文件总字节数
        file_size = f.tell()

        # Calculate position from where to start reading
        # 计算起始读取位置：文件大小减去近似字节数，最小为 0
        seek_pos = max(file_size - approx_byte_count, 0)
        # 移动到该起始位置
        f.seek(seek_pos)

        # Read and decode the data
        # 从起始位置读到文件尾并解码（非法字节用替换符，避免报错）
        data = f.read().decode("utf-8", errors="replace")

        # To avoid partial tokens/words at the beginning, skip to the first newline
        # 起始位置可能切断一行，跳到第一个换行符之后，丢弃残缺首行
        first_newline = data.find("\n")
        if first_newline != -1:
            data = data[first_newline + 1 :]

        # Compress multiple whitespace characters (including newlines) into a single space
        # 将连续空白（含换行）压成单个空格，节省 token 开销
        data = " ".join(data.split())
        # 再把零散换行压缩为单个换行（上面 join 已基本消除，此处兜底）
        data = re.sub(r"\n+", "\n", data)

        # 返回压缩后的日志尾部文本
        return data


# 建立 slurm 任务 ID 到实验目录的映射（读取每个实验的 dojo_config.json）
def slurm_id_to_log(logs_folder):
    """
    Extracts 'Output dir: *' from *.out files, cleaning ANSI escape characters.
    Returns a dictionary mapping job names to cleaned output directories.
    """
    # 字典推导：对日志目录下每个有效实验子目录，取其配置里的 slurm_id 作为键
    job_id2experiment_id = {
        RunConfig.load_from_json(subdir / "dojo_config.json").metadata.slurm_id: subdir
        for subdir in logs_folder.iterdir()
        if is_experiment(subdir)
    }
    # 返回 slurm 任务 ID -> 实验目录 的映射
    return job_id2experiment_id


# 简单启发式：在日志末尾若干行中搜索崩溃关键词，猜测任务是否崩溃
def likely_crashed(log_text):
    """
    Simple heuristic to guess if a job crashed
    by scanning the last 50 lines for certain keywords.
    Feel free to make this more/less robust.
    """
    # 崩溃指标关键词列表（大小写不敏感匹配）
    crash_indicators = [
        "error",
        "exception",
        "traceback",
        "killed",
        "fatal",
        "submitit error",
        "exited with exit code 1",
        "Terminating",
        "non-zero exit",
    ]
    # 任一关键词出现在日志文本中即判定为疑似崩溃
    return any(word.lower() in log_text.lower() for word in crash_indicators)


# 汇总 submitit/slurm 日志数据：统计运行数、识别疑似崩溃任务并保存其日志尾部
def gather_submitit_data(logs_folder, job_ids):
    """
    - Walks the logs_folder.
    - Finds subfolders containing *.err files.
    - Reads and stores last tokens.
    - Guesses crash status.
    - Returns a JSON-like list of crashed jobs + statistics.
    """
    # 存放疑似崩溃任务信息的结果列表
    results = []
    # 实际扫描到的运行计数
    num_runs = 0
    # slurm 日志目录位于元实验目录上两级（<...>/slurm_logs）
    slurm_log_path = Path(logs_folder).parent.parent / "slurm_logs"

    # 逐个任务 ID 查找对应的 .err 文件
    for job_id in job_ids:
        # 递归匹配路径中含该任务 ID 的 .err 文件
        err_files = list(slurm_log_path.glob(f"**/*{job_id}*.err"))

        # 仅当恰好找到一个 .err 文件时才处理（0 个或多个都视为不明确）
        if len(err_files) == 1:
            # 取出该唯一的错误日志文件
            err_file = err_files[0]
            # 有效运行数加一
            num_runs += 1

            # 近似读取错误日志尾部约 1 万 token 的文本
            last_tokens = get_last_approx_tokens(err_file, 10_000)
            # 用启发式判断是否疑似崩溃
            crashed = likely_crashed(last_tokens)

            # 疑似崩溃的任务收集其日志尾部信息
            if crashed:
                results.append(
                    {
                        "job_name": job_id,
                        "err_file_path": err_file,
                        "status": "likely_crashed",
                        "last_err_tokens": last_tokens,
                    }
                )
    # Append meta-info as a separate dictionary
    # 在列表末尾追加统计信息字典：总运行数与疑似崩溃数
    results.append({"total_num_runs": num_runs, "num_crashed": len(results)})
    # 返回结果列表（含尾部元信息）
    return results


# 单个任务崩溃摘要的 JSON Schema：约束 LLM 结构化输出的字段与类型
job_summary_schema = """{
    "type": "object",
    "properties": {
        "job_name": {
            "type": "string",
            "description": "The name or identifier of the job."
        },
        "experiment_name": {
            "type": "string",
            "description": "The name of the experiment."
        },
        "details": {
            "type": "string",
            "description": "Brief details or log excerpts explaining the error or crash status."
        },
        "error_category": {
            "type": "string",
            "description": "Short label categorizing the type of error, e.g., 'CUDA OOM', 'AssertionError', 'Python Exception', or 'None' if not truly crashed."
        },
        "is_crashed": {
            "type": "boolean",
            "description": "True if the job actually crashed the outer loop, False if it handled internal exceptions without crashing."
        },
        "error_text": {
            "type": "string",
            "description": "The text responsible for indicating an error occurred."
        }
    },
    "required": ["job_name", "experiment_name", "details", "error_category", "is_crashed", "error_text"]
}"""


# 对单个疑似崩溃任务调用 LLM，判断其是外层循环真崩溃还是内部可容忍异常
def summarize_single_crash(job, job_to_name):
    # If you have a raw text or JSON to embed:
    # 把该任务的错误日志尾部文本拼接为提示词中的日志片段
    last_err_tokens = "\n".join(job.get("last_err_tokens", []))

    # 构造系统提示词：角色、目标、任务详情与输出格式要求（字符串字面量，勿改）
    system_prompt = f"""
        # Role
        You are a technical assistant who provides concise analyses.

        # Goal
        You are given the last tokens of a job's .err logs. 
        The job was heuristically flagged as 'likely_crashed', 
        but this crash might be internal (allowed within the code) 
        rather than an actual crash of the outer loop. Often if the crash information
        is preceded by a Term Out: it most likely is a crash in the inner loop. Make note 
        if it is most likely an inner loop crash such that we can attempt to distinguish 
        it later.

        # Input
        job_name = {job["job_name"]}
        experiment_name = {job_to_name[job["job_name"]]}

        Here are the last tokens of the err log:
        <log_snippet>
        {last_err_tokens}
        </log_snippet>

        # Task
        1. Determine whether this run TRULY crashed the outer loop or if the crash was internal/allowed.
        2. If truly crashed, assign a short error category (e.g., 'CUDA OOM', 'AssertionError', 'Python Exception', etc.).
        3. Provide a brief detail or excerpt from the logs explaining why you concluded that.
        4. Tip: If the last line of the logs show a message with a timestamp, it most likely is still running and has not crashed.

        # Output
        Return a JSON object ONLY with fields:
        - job_name
        - experiment_name
        - is_crashed (true/false)
        - error_category (string, or 'none' if not truly crashed)
        - details (short text)
        - error text (text indicating the error)
            """.strip()

    # 组装消息列表：整段提示作为一条用户消息
    messages = [{"role": "user", "content": system_prompt}]
    # 创建 LLM 客户端
    client = create_client()
    # 按预定义 JSON Schema 调用模型，要求返回结构化的崩溃摘要
    response_text, _ = client.query(
        messages=messages,
        json_schema=job_summary_schema,
        function_name="jobCrashSummary",
        function_description="Return a JSON object summarizing the job's crash status, including the job name, a short set of details from the error log, a categorized error type, and a boolean indicating if the outer loop truly crashed.",
    )

    # 从模型回复文本中解析 JSON 对象
    summary_dict = parse_json_output(response_text)

    # 解析失败（空结果）时用兜底字典替代，标记为未知类别的崩溃
    if len(summary_dict) == 0:
        summary_dict = {
            "job_name": job["job_name"],
            "is_crashed": True,
            "experiment_name": job_to_name[job["job_name"]],
            "error_category": "unknown",
            "details": "LLM returned invalid JSON. Unable to parse.",
            "error_text": "Could not find",
        }

    # 断言关键字段存在，保证下游代码可用
    assert "job_name" in summary_dict
    assert "is_crashed" in summary_dict
    assert "error_category" in summary_dict
    assert "details" in summary_dict

    # 返回该任务的崩溃摘要字典
    return summary_dict


# 把逐任务的崩溃摘要列表再次交给 LLM，生成合并的 markdown 总报告
def final_report_from_summaries(summaries):
    """
    Takes a list of per-job summaries (in JSON form) and does
    one final LLM call to produce a consolidated markdown report.
    """

    # Convert entire summary list to JSON
    # 将全部摘要序列化为带缩进的 JSON 文本，嵌入提示词
    summaries_json = json.dumps(summaries, indent=2)

    # 构造最终汇总的系统提示词（字符串字面量，勿改）
    system_prompt = f"""
        # Role
        You are a technical assistant who provides concise analyses.

        # Goal
        You are given a JSON array, where each element is a summary of a job's crash status:
        (job_name, experiment_name, is_crashed, error_category, details).

        # Input
        JSON Summaries:
        <summaries>
        {summaries_json}
        </summaries>

        # Task
        1. Provide a final markdown report that groups jobs by is_crashed vs not_crashed.
        2. For crashed jobs, group them by error_category.
        3. Provide a short discussion of each category. 
        4. Give overall stats that give insights to help identify areas to fix: total jobs analyzed, total truly crashed, etc.
        5. Give the experiment name alongside the job name.
        
        # Output
        Return a single markdown document with your consolidated analysis. Do not surround it with a code block.
    """.strip()

    # 组装消息：提示词作为系统消息
    messages = [{"role": "system", "content": system_prompt}]
    # 创建 LLM 客户端
    client = create_client()
    # 调用模型生成合并报告文本
    out, _ = client.query(messages=messages)
    # 剥离思考标签，仅保留最终 markdown 内容
    _, final_markdown = parse_thinking_tags(out)
    # 返回最终 markdown 报告
    return final_markdown


# 端到端流程：收集日志数据 -> 逐任务 LLM 分析 -> 汇总生成最终报告并落盘
def generate_error_reports(log_dir, save_dir):
    """
    1. Gather data from logs.
    2. For each 'likely_crashed' job, do an LLM call to refine the analysis.
    3. Accumulate all job-level summaries + meta-stats.
    4. Do a final LLM call on the entire set of summaries.
    5. Write the final output to 'logs_report.md'.
    """
    # 建立 slurm 任务 ID 到实验目录的映射
    job_to_name = slurm_id_to_log(log_dir)

    # 扫描日志，得到疑似崩溃任务列表（末位元素为统计信息）
    data = gather_submitit_data(log_dir, list(job_to_name.keys()))

    # The last dict in 'data' has meta info about total_num_runs and num_crashed
    # 取出列表末尾的元信息字典（若存在）
    meta_info = data[-1] if isinstance(data[-1], dict) and "total_num_runs" in data[-1] else {}

    # Filter out the meta_info from the actual job logs
    # 把元信息从逐任务数据中剥离，只留下任务记录
    job_candidates = data[:-1] if meta_info else data

    # Summarize each 'likely_crashed' job
    # (We only run LLM calls on these, as you requested)
    # 逐个对疑似崩溃任务调用 LLM 精化分析，带进度条
    crash_summaries = []
    for job in tqdm(job_candidates, desc="Summarising Likely Crashed Jobs"):
        # 生成单个任务的崩溃摘要
        summary = summarize_single_crash(job, job_to_name)
        # 收集到汇总列表
        crash_summaries.append(summary)

    # Optionally, you might embed the meta_info right into our final summary:
    # E.g. as a dictionary so the final LLM knows total runs
    # 把统计信息整理为字典，供最终报告的 LLM 参考
    meta_info_dict = {
        "total_num_runs": meta_info.get("total_num_runs", 0),
        "num_likely_crashed": meta_info.get("num_crashed", 0),
    }
    # Combine them in a single list or dict:
    # 合并元信息与逐任务摘要为一个整体结构
    combined_summaries = {"meta_info": meta_info_dict, "per_job_summaries": crash_summaries}

    # 将逐任务原始摘要写入 md 文件留存
    with open(save_dir / "error_analysis_per_job_jsons.md", "w") as f:
        f.write(str(combined_summaries))

    # Now do one final LLM call to create a single markdown report
    # 调用 LLM 生成最终的合并 markdown 报告
    final_md = final_report_from_summaries(combined_summaries)

    # 将最终报告写入文件
    with open(save_dir / "error_analysis_report.md", "w") as f:
        f.write(final_md)

    return final_md  # or just return it if you prefer


# 命令行入口：第一个参数为日志目录，可选第二个参数指定保存目录
if __name__ == "__main__":
    import sys

    # 参数不足时打印用法并退出
    if len(sys.argv) < 2:
        print("Usage: python meta_error_summary.py <meta_exp_directory> <optional_save_dir>")
        sys.exit(1)

    # 第一个参数：元实验日志目录
    log_dir = Path(sys.argv[1])

    # 第二个参数可选：报告保存目录，默认与日志目录相同
    if len(sys.argv) > 2:
        save_dir = Path(sys.argv[2])
    else:
        save_dir = log_dir

    # 执行端到端的错误报告生成
    generate_error_reports(log_dir, save_dir)
