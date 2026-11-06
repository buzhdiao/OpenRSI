"""
Logging utilities for tracking training data and samples.
"""

import csv
# 标准库导入:CSV 读写、JSON 序列化(组内列表字段写盘前转 JSON)、
# 环境/路径操作、时间、默认字典(组缓冲)、线程锁与类型注解
import json
import os
import time
from collections import defaultdict
from threading import Lock
from typing import Any, Dict, List, Optional

# 评估阶段的默认组大小:评估样本缺少 group_index 约束时按此阈值凑组写盘
EVAL_GROUP_SIZE = 8


class TrainingLogger:
    """Manages logging of training data to CSV files."""
    # 训练数据记录器:按组缓冲样本,凑满一组后写一行 CSV(供训练侧审计四算子行为与奖励)

    def __init__(self, log_file_name: str = None, n_samples_per_prompt: int = 8):
        """
        Initialize training logger.

        Args:
            log_file_name: Name of the log file (default: train_log.csv)
            n_samples_per_prompt: Number of samples per prompt group
        """
        # 组大小:凑满多少个样本算完整组
        self.n_samples_per_prompt = n_samples_per_prompt

        # Setup log directory - use RUN_DIR from environment if available
        # 日志目录优先取环境变量 RUN_DIR
        run_dir_from_env = os.environ.get("RUN_DIR")
        self.run_dir = run_dir_from_env
        # 确保目录存在(已存在则忽略)
        os.makedirs(self.run_dir, exist_ok=True)

        if log_file_name is None:
            # 未指定文件名时取环境变量,缺省 train_log.csv
            log_file_name = os.environ.get("LOG_FILE_NAME", "train_log.csv")

        # CSV 完整路径与初始化标志(表头只写一次)
        self.csv_log_file = os.path.join(self.run_dir, log_file_name)
        self.csv_initialized = False

        # Group logging buffer
        # 组缓冲:group_key -> 样本数据列表;配线程锁保证并发安全
        self.group_logging_buffer = defaultdict(list)
        self.group_logging_lock = Lock()

    def init_csv(self):
        """Initialize CSV log file with headers."""
        if not self.csv_initialized:
            # 仅初始化一次:以写模式打开文件并写入表头(覆盖旧文件)
            with open(self.csv_log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # 表头与每组记录一一对应(分数、奖励、静态/动态口径、父代信息、作弊标记等)
                headers = [
                    "group_id",
                    "task_id",
                    "task_name",
                    "higher_is_better",
                    "theoretical_max",
                    "theoretical_min",
                    "leaderboard_max",
                    "leaderboard_min",
                    "prompt_type",
                    # "prompts",
                    "scores",
                    "rewards",
                    "static_base_rewards",
                    "base_rewards",
                    "exploit_coefficients",
                    "explore_coefficients",
                    "cooling_coefficients",
                    "results",
                    "codes",
                    "raw_texts",
                    "parent_ids",
                    "parent_codes",
                    "parent_scores",
                    "parent_rewards",
                    "modes",
                    "hacks",
                    "code_categories",
                    "static_raw_rewards",
                    "static_mercy_raw_rewards",
                    "dynamic_raw_rewards",
                ]
                writer.writerow(headers)
            # 标记已初始化,避免重复写表头
            self.csv_initialized = True
            # 打印初始化信息与运行目录
            print(
                f"[LOG] Initialized CSV log file: {self.csv_log_file} with {self.n_samples_per_prompt} samples per prompt"
            )
            print(f"[LOG] Run directory: {self.run_dir}")

    def get_run_dir(self) -> str:
        """Get the run directory path."""
        # 返回本次运行的日志目录
        return self.run_dir

    def log_group_data(self, group_key: str, group_data: List[Dict], group_id: int, group_size: int):
        """Write a complete group's data to CSV."""
        # 将完整一组的样本数据写为一行 CSV(组内各字段聚合为 JSON 数组)
        print("[LOG_GROUP_DATA]")

        if len(group_data) != group_size:
            # 样本数与组大小不符:跳过并告警(防止写入残缺组)
            print(f"[LOG WARNING] Group {group_key} has {len(group_data)} samples, expected {group_size}. Skipping.")
            return

        # Extract common metadata from first sample
        # 组内样本共用任务级元数据,从首样本提取
        first_sample = group_data[0]
        metadata = first_sample["metadata"]

        # 行首固定列:组ID 与任务级指标上下界等元信息(四算子生成的评分依据)
        row = [
            group_id,  # Use sequential group ID as step
            metadata.get("uuid", "N/A"),  # task_id (uuid in metadata)
            metadata.get("task_name", "N/A"),
            metadata.get("higher_is_better", "N/A"),
            metadata.get("theoretical_max", "N/A"),
            metadata.get("theoretical_min", "N/A"),
            metadata.get("leaderboard_max", "N/A"),
            metadata.get("leaderboard_min", "N/A"),
            metadata.get("prompt_type", "N/A"),
        ]

        # 初始化各组内列表:每列收集一个字段(全部样本聚合成数组后写 JSON)
        (
            # prompts,
            scores,
            rewards,
            static_base_rewards,
            base_rewards,
            exploit_coefficients,
            explore_coefficients,
            cooling_coefficients,
            results,
            codes,
            raw_texts,
            parent_ids,
            parent_codes,
            parent_scores,
            parent_rewards,
            modes,
            hacks,
            code_categories,
            static_raw_rewards,
            static_mercy_raw_rewards,
            dynamic_raw_rewards,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for sample_data in group_data:
            # 逐样本收集各字段到对应列表
            # Collect prompt (serialize if it's a list/dict)
            # prompt = sample_data.get("prompt", "")
            # if isinstance(prompt, (dict, list)):
            #     prompt_serialized = json.dumps(prompt, ensure_ascii=False)
            # else:
            #     prompt_serialized = str(prompt) if prompt else ""
            # prompt_serialized = prompt_serialized.replace("\r", "").replace("\n", "\\n")
            # prompts.append(prompt_serialized)

            # 沙箱执行分数
            scores.append(sample_data["score"])
            # 最终训练奖励
            rewards.append(sample_data["reward"])
            # 指标静态基础奖励
            static_base_rewards.append(sample_data.get("static_base_reward"))
            # 基础奖励(缺省回退最终奖励)
            base_rewards.append(sample_data.get("base_reward", sample_data["reward"]))
            # 利用/探索/冷却系数(airaevo 策略信息)
            exploit_coefficients.append(sample_data.get("exploit_coefficient"))
            explore_coefficients.append(sample_data.get("explore_coefficient"))
            cooling_coefficients.append(sample_data.get("cooling_coefficient"))

            result_info = sample_data.get("result", "")
            if isinstance(result_info, (dict, list)):
                # 结构化结果原样保留(写盘时由 json 序列化)
                result_serialized = result_info
            else:
                # 字符串结果压平换行,避免破坏 CSV 行结构
                result_serialized = "" if result_info is None else str(result_info)
                result_serialized = result_serialized.replace("\r", "").replace("\n", "\\n")
            results.append(result_serialized)

            # 代码文本压平换行
            code = sample_data["code"]
            code = code.replace("\r", "").replace("\n", "\\n")
            codes.append(code)

            # 模型原始回复压平换行
            raw_text = sample_data.get("raw_text", "")
            raw_text = raw_text.replace("\r", "").replace("\n", "\\n")
            raw_texts.append(raw_text)

            # 父代程序 ID(improve/debug 模式的参照)
            parent_ids.append(sample_data.get("parent_id"))

            # 父代代码压平换行
            parent_code = sample_data.get("parent_code", "")
            parent_code = parent_code.replace("\r", "").replace("\n", "\\n")
            parent_codes.append(parent_code)

            # 父代分数与奖励
            parent_scores.append(sample_data.get("parent_score"))
            parent_rewards.append(sample_data.get("parent_reward"))

            # 算子模式(缺省 draft)
            modes.append(sample_data.get("mode", "draft"))
            # 作弊标记(缺省 0 表示合法)
            hacks.append(sample_data.get("hack", 0))  # Default to 0 (valid) if not present
            # 代码类别(缺省 valid)
            code_categories.append(sample_data.get("code_category", "valid"))
            # 静态/仁慈/动态原始奖励
            static_raw_rewards.append(sample_data.get("static_raw_reward"))
            static_mercy_raw_rewards.append(sample_data.get("static_mercy_raw_reward"))
            dynamic_raw_rewards.append(sample_data.get("dynamic_raw_reward"))

        # row.append(json.dumps(prompts, ensure_ascii=False))
        # 把各列表序列化为 JSON 追加到行尾(顺序与表头一致)
        row.append(json.dumps(scores, ensure_ascii=False))
        row.append(json.dumps(rewards, ensure_ascii=False))
        row.append(json.dumps(static_base_rewards, ensure_ascii=False))
        row.append(json.dumps(base_rewards, ensure_ascii=False))
        row.append(json.dumps(exploit_coefficients, ensure_ascii=False))
        row.append(json.dumps(explore_coefficients, ensure_ascii=False))
        row.append(json.dumps(cooling_coefficients, ensure_ascii=False))
        row.append(json.dumps(results, ensure_ascii=False))
        row.append(json.dumps(codes, ensure_ascii=False))
        row.append(json.dumps(raw_texts, ensure_ascii=False))
        row.append(json.dumps(parent_ids, ensure_ascii=False))
        row.append(json.dumps(parent_codes, ensure_ascii=False))
        row.append(json.dumps(parent_scores, ensure_ascii=False))
        row.append(json.dumps(parent_rewards, ensure_ascii=False))
        row.append(json.dumps(modes, ensure_ascii=False))
        row.append(json.dumps(hacks, ensure_ascii=False))
        row.append(json.dumps(code_categories, ensure_ascii=False))
        row.append(json.dumps(static_raw_rewards, ensure_ascii=False))
        row.append(json.dumps(static_mercy_raw_rewards, ensure_ascii=False))
        row.append(json.dumps(dynamic_raw_rewards, ensure_ascii=False))

        # Write to CSV
        # 追加模式写一行(不覆盖已有记录)
        with open(self.csv_log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # 打印写盘完成日志
        print(f"[LOG] Wrote group data for group_id={group_id}, task={metadata.get('task_name', 'N/A')}")

    def log_sample(
        self,
        sample,
        score: Optional[float],
        reward: float,
        static_base_reward: Optional[float],
        base_reward: float,
        result_info: Any,
        code: str,
        raw_text: str = "",
        parent_id: Optional[int] = None,
        parent_code: str = "",
        parent_score: Optional[float] = None,
        parent_reward: Optional[float] = None,
        mode: str = "draft",
        hack: int = 0,
        code_category: str = "valid",
        static_raw_reward: Optional[float] = None,
        static_mercy_raw_reward: Optional[float] = None,
        dynamic_raw_reward: Optional[float] = None,
        exploit_coefficient: Optional[float] = None,
        explore_coefficient: Optional[float] = None,
        cooling_coefficient: Optional[float] = None,
    ):
        """
        Collect sample information and write to CSV when a group is complete.

        Uses sample.group_index to uniquely identify each group.
        This ensures that even when the same prompt appears multiple times in a batch,
        each group is tracked separately.

        Args:
            sample: The sample object
            score: The score from sandbox execution
            reward: The final reward used for training (may be diff for improve mode)
            base_reward: The base reward before considering parent
            result_info: Execution result information
            code: The extracted code
            raw_text: Raw response text from model
            parent_id: Parent program ID (for improve mode)
            parent_code: Parent program code (for improve mode)
            parent_score: Parent program score (for improve mode)
            parent_reward: Parent program reward (for improve mode)
            mode: Generation mode ('draft' or 'improve')
            hack: Hack check result (0 = not hacking, valid ML code, 1 = cheating/guessing)
            code_category: Code category ('valid', 'hack_non_empty', 'empty', 'no_valid_code')
        """
        print("[LOG_SAMPLE_TO_GROUP]")

        # Use group_index as the unique identifier for this group
        # group_index is set by the data source and is unique for each group
        # 以 group_index 作为组唯一标识;缺失时用 -1 走回退
        group_index = sample.group_index if hasattr(sample, "group_index") and sample.group_index is not None else -1

        if group_index < 0:
            print(f"[LOG WARNING] Sample has no valid group_index, using fallback")
            # Fallback to prompt hash if group_index is not available
            # 无有效 group_index 时用 uuid 构造回退组键
            group_key = f"fallback_{sample.metadata.get('uuid', 'N/A')}"
            print(f"[LOG WARNING] Group key: {group_key}")
        else:
            # 常规组键
            group_key = f"group_{group_index}"

        # Extract prompt from sample
        # prompt = sample.prompt if hasattr(sample, "prompt") else ""

        # Prepare sample data
        # 打包本样本待记录的全部字段(凑满组后由 log_group_data 写盘)
        sample_data = {
            "metadata": sample.metadata if hasattr(sample, "metadata") else {},
            # "prompt": prompt,
            "score": score,
            "reward": reward,
            "static_base_reward": static_base_reward,
            "base_reward": base_reward,
            "result": result_info,
            "code": code,
            "raw_text": raw_text,
            "sample_index": sample.index if hasattr(sample, "index") else None,
            "parent_id": parent_id,
            "parent_code": parent_code,
            "parent_score": parent_score,
            "parent_reward": parent_reward,
            "mode": mode,
            "hack": hack,
            "code_category": code_category,
            "static_raw_reward": static_raw_reward,
            "static_mercy_raw_reward": static_mercy_raw_reward,
            "dynamic_raw_reward": dynamic_raw_reward,
            "exploit_coefficient": exploit_coefficient,
            "explore_coefficient": explore_coefficient,
            "cooling_coefficient": cooling_coefficient,
        }

        # Thread-safe buffer update
        # 加锁更新组缓冲(多协程并发记录)
        with self.group_logging_lock:
            # 样本数据加入对应组缓冲
            self.group_logging_buffer[group_key].append(sample_data)
            # 组大小:训练用配置值,评估回退用 EVAL_GROUP_SIZE
            group_size = self.n_samples_per_prompt if group_index >= 0 else EVAL_GROUP_SIZE
            print(
                f"[LOG WARNING] Group: {len(self.group_logging_buffer[group_key])}/{group_size} for group_key: {group_key}"
            )
            # When we have n_samples_per_prompt samples (a complete group), write to CSV
            if len(self.group_logging_buffer[group_key]) == group_size:
                # 凑满一组:取出整组数据
                group_data = self.group_logging_buffer[group_key]

                # Sort samples by their index to ensure consistent order
                # 按样本序号排序,保证 CSV 内顺序稳定
                group_data.sort(key=lambda x: x.get("sample_index", 0))

                # Use the group_index directly as the step number (it's already unique and sequential)
                # 写入完整组(组号即步号)
                self.log_group_data(group_key, group_data, group_index, group_size)
                # Clear this group from buffer
                # 写完清理该组缓冲
                del self.group_logging_buffer[group_key]
            elif len(self.group_logging_buffer[group_key]) > group_size:
                # This shouldn't happen, but log a warning if it does
                # 超出组大小属异常,告警提示
                print(
                    f"[LOG WARNING] Group {group_key} has more than {group_size} samples: {len(self.group_logging_buffer[group_key])}"
                )


# Global logger instance (can be initialized by generate_mle.py)
# 全局单例日志器(由 generate_mle.py 初始化)
_global_logger = None


def get_logger() -> TrainingLogger:
    """Get or create the global logger instance."""
    # 获取全局日志器;未初始化时用默认参数创建
    global _global_logger
    if _global_logger is None:
        _global_logger = TrainingLogger()
        _global_logger.init_csv()
    return _global_logger


def init_logger(log_file_name: str = None, n_samples_per_prompt: int = 8):
    """Initialize the global logger."""
    # 显式初始化全局日志器(可指定文件名与组大小)
    global _global_logger
    _global_logger = TrainingLogger(log_file_name, n_samples_per_prompt)
    _global_logger.init_csv()
    return _global_logger
