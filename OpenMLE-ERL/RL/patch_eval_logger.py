"""
Monkey patch for custom eval logger.
This file should be imported BEFORE slime modules are loaded.

Usage: Add this at the very beginning of train.py or main entry point:
    import sys
    sys.path.insert(0, '/path/to/examples/mle-agent-rl')
    import patch_eval_logger
"""

# 导入 sys 模块：用于操作 Python 模块搜索路径 sys.path，保证能从脚本所在目录导入模块
import sys
# 导入 os 模块：用于读取环境变量、拼接与检测排行榜文件路径
import os
# 导入 pandas：用于加载并清洗公开排行榜（leaderboard）CSV 数据
import pandas as pd
# 导入 numpy：用于计算均值以及判断数值是否有限（过滤 NaN/inf）
import numpy as np
# 导入 slime 框架的日志工具模块，最终用它把评估指标字典写入训练日志系统
from slime.utils import logging_utils
# 导入 slime 的 compute_rollout_step：把 rollout_id 换算成全局评估步数 step
from slime.utils.metric_utils import compute_rollout_step

# Make sure we can import from current directory
# 计算本脚本文件所在的绝对路径目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 若该目录尚未加入模块搜索路径，则插入到最前面，确保可从当前目录导入其他模块
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# 默认排行榜根目录元组（留空），真实根目录由环境变量 LEADERBOARD_ROOT 提供
DEFAULT_LEADERBOARD_ROOTS: tuple[str, ...] = ()
# 读取环境变量 LEADERBOARD_ROOT 作为排行榜根目录，并去掉首尾空白
LEADERBOARD_PTH = os.getenv("LEADERBOARD_ROOT", "").strip()
# 定义需要统计均值的两种奖励视角：静态基准奖励（固定指标口径、execution-grounded 的分数）与动态原始奖励
REWARD_VIEW_KEYS = (
    "metric_static_base_reward",
    "dynamic_raw_reward",
)


def _to_finite_float(value):
    # 工具函数：把任意输入安全转换为有限浮点数；无法转换或非有限（NaN/inf）时返回 None
    if value is None:
        # 输入本身就是 None，直接返回 None 表示"无该指标"
        return None
    try:
        # 尝试把输入转换成 float 类型
        value = float(value)
    except (TypeError, ValueError):
        # 类型不支持或格式非法，转换失败则返回 None
        return None
    # NaN 或正负无穷不属于有效指标，返回 None
    if not np.isfinite(value):
        return None
    # 转换成功且为有限值，返回该浮点数
    return value


def _build_mode_bucket():
    # 构造按"四算子"模式（debug/draft/improve/crossover）聚合统计用的空统计桶
    return {
        "reward_views": {key: [] for key in REWARD_VIEW_KEYS},
        "scores": [],
        "scores_every_tasks": {},
        "response_lengths": [],
        "status_codes": [],
        "hack_count": 0,
        "empty_count": 0,
        "success_count": 0,
        "total_count": 0,
        "samples": [],
    }


def _has_static_bounds(metadata):
    # 判断样本元数据中是否存在静态基准奖励的上下界；无界则该奖励不可比，统计时需跳过
    flag = metadata.get("metric_static_base_reward_has_bounds")
    if flag is not None:
        try:
            # 若显式给出了 has_bounds 标志，直接转成布尔值返回
            return bool(int(flag))
        except (TypeError, ValueError):
            # 标志无法解析为整数时，继续走后备推断逻辑
            pass
    # 读取"分数越高越好"标志（决定最优/最差方向的口径）
    higher_is_better = metadata.get("higher_is_better")
    # 读取理论最小值并转为有限浮点数
    theoretical_min = _to_finite_float(metadata.get("theoretical_min"))
    # 读取理论最大值并转为有限浮点数
    theoretical_max = _to_finite_float(metadata.get("theoretical_max"))
    # 读取排行榜（榜单上实际队伍）的最小分数
    leaderboard_min = _to_finite_float(metadata.get("leaderboard_min"))
    # 读取排行榜（榜单上实际队伍）的最大分数
    leaderboard_max = _to_finite_float(metadata.get("leaderboard_max"))
    if higher_is_better is False:
        # 分数越低越好：最优值取榜单最小（缺失时退回理论最小），最差值取榜单最大（缺失时退回理论最大）
        best = leaderboard_min if leaderboard_min is not None else theoretical_min
        worst = leaderboard_max if leaderboard_max is not None else theoretical_max
    else:
        # 分数越高越好（或方向未知）：最优值取榜单最大（缺失时退回理论最大），最差值取榜单最小（缺失时退回理论最小）
        best = leaderboard_max if leaderboard_max is not None else theoretical_max
        worst = leaderboard_min if leaderboard_min is not None else theoretical_min
    # 只有最优界与最差界都能拿到时，才认为该奖励存在可用边界
    return best is not None and worst is not None


def _log_reward_view_means(log_dict, prefix, reward_views):
    # 把各奖励视角的均值与样本数写入日志字典；静态基准奖励会额外以通用名 "reward" 输出
    for reward_key, values in reward_views.items():
        # 该视角收集到数值才输出统计
        if values:
            # 静态基准奖励在输出时改名为 static_base_reward，其余视角沿用原名
            output_key = "static_base_reward" if reward_key == "metric_static_base_reward" else reward_key
            # 计算该视角的均值
            mean_value = float(np.mean(values))
            # 记录均值
            log_dict[f"{prefix}/{output_key}"] = mean_value
            # 记录参与均值的样本数，便于诊断样本覆盖情况
            log_dict[f"{prefix}/{output_key}_count"] = len(values)
            if reward_key == "metric_static_base_reward":
                # 静态基准奖励额外以通用键 "reward" 再记一份，兼容下游看板
                log_dict[f"{prefix}/reward"] = mean_value
                log_dict[f"{prefix}/reward_count"] = len(values)


def _log_quality_proxy_metrics(log_dict, prefix, samples):
    # 计算质量代理指标：GRPO 组内最优静态基准奖励、组内"任务最优改进"次数、历史最优奖励
    if not samples:
        # 没有样本则无需统计，直接返回
        return

    # 按组索引（group_index，对应 GRPO 的组采样概念）把样本分桶
    group_buckets = {}
    # 遍历样本并带上其下标，便于组标识缺失时兜底
    for sample_idx, sample in enumerate(samples):
        # 取样本元数据；若不存在或不是字典则退化为空字典
        metadata = sample.metadata if hasattr(sample, "metadata") and isinstance(sample.metadata, dict) else {}
        # 优先从元数据中取组索引
        group_key = metadata.get("group_index")
        if group_key is None and hasattr(sample, "group_index"):
            # 元数据里没有时，尝试从 sample 对象的同名属性上取
            group_key = sample.group_index
        if group_key is None:
            # 仍取不到则用样本下标构造伪组键，保证该样本也能参与统计
            group_key = f"sample_{sample_idx}"
        # 为该组惰性初始化两个指标桶：静态基准奖励、是否任务最优改进
        bucket = group_buckets.setdefault(
            group_key,
            {
                "metric_static_base_reward": [],
                "metric_static_base_reward_is_task_best_improvement": [],
            },
        )
        # 遍历桶里的键，逐个收集数值
        for key in bucket:
            # 静态基准奖励要求存在可用上下界，否则该值不可比，跳过
            if key == "metric_static_base_reward" and not _has_static_bounds(metadata):
                continue
            # 把元数据中的指标值安全转换为有限浮点数
            value = _to_finite_float(metadata.get(key))
            if value is not None:
                # 只有有效值才入桶
                bucket[key].append(value)

    if not group_buckets:
        # 没有任何有效分组则直接返回
        return

    # 目标奖励键：静态基准奖励
    reward_key = "metric_static_base_reward"
    # 对每个组取组内最优（max）静态奖励，对应 GRPO 中"组内最优"的信号
    group_bests = [max(bucket[reward_key]) for bucket in group_buckets.values() if bucket[reward_key]]
    if group_bests:
        # 记录"组内最优"的均值：衡量每组最好采样能达到的水平
        log_dict[f"{prefix}/group_best_static_base_reward"] = float(np.mean(group_bests))
        # 记录有组内最优值的组数
        log_dict[f"{prefix}/group_best_static_base_reward_count"] = len(group_bests)
        # 记录组内最优达到 0.95 以上的比例（接近满分的组占比）
        log_dict[f"{prefix}/group_best_static_base_reward_ge_0p95_rate"] = float(
            np.mean([best >= 0.95 for best in group_bests])
        )

    # 对每个组统计"任务最优改进"标志之和（该组内刷新任务最优的次数）
    group_improvement_counts = [
        float(sum(bucket["metric_static_base_reward_is_task_best_improvement"]))
        for bucket in group_buckets.values()
        if bucket["metric_static_base_reward"]
    ]
    if group_improvement_counts:
        # 记录平均每组的任务最优改进次数：反映组内采样带来的探索增益
        log_dict[f"{prefix}/static_base_reward_is_task_best_improvement"] = float(np.mean(group_improvement_counts))

    # 收集每个样本元数据中的"历史最优静态奖励"（跨 rollout 的累计最优）
    history_values = [
        _to_finite_float(
            (sample.metadata if hasattr(sample, "metadata") and isinstance(sample.metadata, dict) else {}).get(
                "history_best_static_base_reward"
            )
        )
        for sample in samples
    ]
    # 过滤掉无法解析的 None 值
    history_values = [value for value in history_values if value is not None]
    if history_values:
        # 记录历史最优奖励的均值：衡量跨 rollout 的累计进展
        log_dict[f"{prefix}/history_best_static_base_reward"] = float(np.mean(history_values))


def is_lower_better(leaderboard: pd.DataFrame) -> bool:
    """
    Determine from the leaderboard whether lower scores are better.
    Returns True if lower scores are better, False otherwise.
    """
    # 取排行榜的 score 列
    scores = leaderboard["score"]
    # 第一名的分数（假设排行榜已按名次排序）
    top_score = scores.iloc[0]
    # 最后一名的分数
    bottom_score = scores.iloc[-1]
    # 第一名分数比最后一名低，说明该任务分数越低越好
    return bool(top_score < bottom_score)


def get_medal_for_score(score: float | None, leaderboard: pd.DataFrame) -> str:
    """
    Determine the medal level for a score using the leaderboard.

    Args:
        score: Score to evaluate.
        leaderboard: DataFrame containing a score column.

    Returns:
        str: Medal level ("Gold", "Silver", "Bronze", or "None").
    """
    # 分数为 None 或 NaN 时无奖牌可判
    if score is None or np.isnan(score):
        return "None"

    # 判断该排行榜是"越低越好"还是"越高越好"
    lower_better = is_lower_better(leaderboard)
    # 排行榜上的队伍总数
    num_teams = len(leaderboard)
    # 排行榜的 score 列
    scores = leaderboard["score"]

    def get_score_at_position(position: int) -> float:
        # 取排行榜上第 position 名（1 基）的分数
        if position - 1 >= len(scores) or position < 1:
            # 名次越界（超出队伍数或小于 1）时报错
            raise IndexError("Position out of bounds in the leaderboard.")
        # 名次转 0 基下标后取值
        return scores.iloc[position - 1]

    def get_thresholds(num_teams: int) -> tuple:
        """Return the gold, silver, and bronze score thresholds."""
        if 1 <= num_teams < 100:
            # 小榜单（<100 队）：金/银/铜线分别取前 10%/20%/40% 名次的分数
            gold_threshold = get_score_at_position(max(1, int(num_teams * 0.1)))
            silver_threshold = get_score_at_position(max(1, int(num_teams * 0.2)))
            bronze_threshold = get_score_at_position(max(1, int(num_teams * 0.4)))
        elif 100 <= num_teams < 250:
            # 中等榜单（100~249 队）：金牌固定取第 10 名，银/铜按 20%/40% 名次
            gold_threshold = get_score_at_position(10)
            silver_threshold = get_score_at_position(max(1, int(num_teams * 0.2)))
            bronze_threshold = get_score_at_position(max(1, int(num_teams * 0.4)))
        elif 250 <= num_teams < 1000:
            # 大榜单（250~999 队）：金牌取第 10+0.2% 名，银/铜固定取第 50/100 名
            gold_threshold = get_score_at_position(10 + int(num_teams * 0.002))
            silver_threshold = get_score_at_position(50)
            bronze_threshold = get_score_at_position(100)
        elif num_teams >= 1000:
            # 超大榜单（>=1000 队）：金牌取第 10+0.2% 名，银/铜按 5%/10% 名次
            gold_threshold = get_score_at_position(10 + int(num_teams * 0.002))
            silver_threshold = get_score_at_position(max(1, int(num_teams * 0.05)))
            bronze_threshold = get_score_at_position(max(1, int(num_teams * 0.1)))
        else:
            # 队伍数为 0 时无法定义奖牌阈值
            raise ValueError("Number of teams in leaderboard must be greater than 0.")

        # 返回金/银/铜三个阈值分数
        return float(gold_threshold), float(silver_threshold), float(bronze_threshold)

    # 计算当前排行榜的金/银/铜阈值
    gold_threshold, silver_threshold, bronze_threshold = get_thresholds(num_teams)

    # Determine the medal level.
    # 依据分数方向与阈值判定奖牌等级
    if lower_better:
        # 分数越低越好：达到或优于金牌线即金牌，依次类推
        if score <= gold_threshold:
            return "Gold"
        elif score <= silver_threshold:
            return "Silver"
        elif score <= bronze_threshold:
            return "Bronze"
    else:
        # 分数越高越好：达到或超过金牌线即金牌，依次类推
        if score >= gold_threshold:
            return "Gold"
        elif score >= silver_threshold:
            return "Silver"
        elif score >= bronze_threshold:
            return "Bronze"

    # 未达到任何奖牌线，返回无奖牌
    return "None"


def load_public_leaderboard(task_name: str) -> pd.DataFrame | None:
    # 根据任务名在多个候选根目录下查找并加载公开排行榜 CSV；找不到返回 None
    if os.getenv("LEADERBOARD_ROOTS"):
        # 优先使用多根目录环境变量 LEADERBOARD_ROOTS（按路径分隔符切分，过滤空项）
        roots = [item for item in os.getenv("LEADERBOARD_ROOTS", "").split(os.pathsep) if item]
    else:
        # 否则退回单根目录 LEADERBOARD_PTH（可能为空）
        roots = [LEADERBOARD_PTH] if LEADERBOARD_PTH else []

    # 记录已处理过的根目录，避免重复扫描
    seen = set()
    # 逐个根目录尝试查找
    for root in roots:
        if root in seen:
            # 重复出现的根目录直接跳过
            continue
        # 标记该根目录已处理
        seen.add(root)
        # 该任务下两个候选排行榜路径（info 子目录优先）
        candidates = [
            os.path.join(root, task_name, "info", "public_leaderboard.csv"),
            os.path.join(root, task_name, "leaderboard.csv"),
        ]
        # 依次尝试每个候选路径
        for path in candidates:
            if not os.path.exists(path):
                # 路径不存在则尝试下一个候选
                continue
            # 读取 CSV 为 DataFrame
            leaderboard = pd.read_csv(path)
            if "score" in leaderboard.columns:
                # 必须包含 score 列才可用
                leaderboard = leaderboard.copy()
                # 把 score 列强转数值，无法解析的值变为 NaN
                leaderboard["score"] = pd.to_numeric(leaderboard["score"], errors="coerce")
                # 丢弃 score 无效（NaN）的行并重置索引
                leaderboard = leaderboard[leaderboard["score"].notna()].reset_index(drop=True)
                if "rank" in leaderboard.columns:
                    # 若有 rank 列，按名次稳定排序，保证名次顺序正确
                    leaderboard = leaderboard.sort_values("rank", kind="stable").reset_index(drop=True)
                # 返回清洗后的排行榜
                return leaderboard
    # 所有候选都不可用时返回 None
    return None


def custom_log_eval_rollout_data(rollout_id, args, data, extra_metrics=None):
    """Extended eval logger with custom metrics for MLE-Agent."""
    # 延迟导入 slime 的通过率与奖励分类指标工具，避免模块加载顺序问题
    from slime.utils.metric_utils import compute_pass_rate, dict_add_prefix
    from slime.ray.rollout import _compute_reward_cat_metrics

    # Keep compatibility with upstream signature:
    # _log_eval_rollout_data(rollout_id, args, data, extra_metrics)
    # and preserve any framework-provided eval metrics.
    # 以框架已提供的 extra_metrics 为基础初始化日志字典，保持与上游签名兼容
    log_dict = {**(extra_metrics or {})}

    # 遍历数据中的每个数据源键（key 通常是评测集名称）
    for key in data.keys():
        # 取该数据源的奖励列表
        rewards = data[key]["rewards"]
        # log_dict[f"eval-avg-rewards/{key}"] = sum(rewards) / len(rewards)

        # Custom metrics from samples
        # 只有存在样本列表时才做自定义统计
        if (samples := data[key].get("samples")) is not None:
            # Original reward category metrics
            # 先记录框架自带的奖励分类指标，并加前缀 eval/{key}-
            log_dict |= dict_add_prefix(_compute_reward_cat_metrics(args, samples), f"eval/{key}-")

            # 🆕 Separate debug/draft/improve/crossover mode statistics by prompt_type
            # 按 prompt_type（对应"四算子"模式）分别建立统计桶
            debug_data = _build_mode_bucket()
            draft_data = _build_mode_bucket()
            improve_data = _build_mode_bucket()
            crossover_data = _build_mode_bucket()

            # 🆕 Custom metrics - Status code distribution (split by prompt_type)
            # 全体样本（不分模式）的 HTTP 状态码收集列表
            status_codes = []
            # 全体样本的分数列表
            scores = []
            # 全体样本按任务名归档的分数，供总体奖牌统计使用
            scores_every_tasks = {}
            # 全体样本的奖励视角收集器（与模式桶同构）
            overall_reward_views = {key: [] for key in REWARD_VIEW_KEYS}
            # 逐样本遍历，按 prompt_type 分桶收集各类统计量
            for sample in samples:
                if hasattr(sample, "metadata") and isinstance(sample.metadata, dict):
                    # 读取该样本的算子模式类型，缺省按 draft 处理
                    prompt_type = sample.metadata.get("prompt_type", "draft")

                    # Select target data dict based on prompt_type
                    # 依据 prompt_type 选择目标统计桶
                    if prompt_type == "debug":
                        target_data = debug_data
                    elif prompt_type == "improve":
                        target_data = improve_data
                    elif prompt_type == "crossover":
                        target_data = crossover_data
                    else:  # draft or default
                        # draft 或未知类型统一归入 draft 桶
                        target_data = draft_data

                    # Collect hack statistics
                    # 累计该模式的样本总数
                    target_data["total_count"] += 1
                    # 保留原始样本，供质量代理指标计算
                    target_data["samples"].append(sample)
                    # 读取作弊标志（hack=1 表示 execution-grounded 校验发现投机取巧行为）
                    hack_value = sample.metadata.get("hack", 0)
                    # 读取代码类别（valid/empty/hack/hack_verify/no_verify）
                    code_category = sample.metadata.get("code_category", "valid")
                    if hack_value == 1:
                        # 作弊样本计数加一
                        target_data["hack_count"] += 1
                    if code_category == "empty":
                        # 空输出样本计数加一
                        target_data["empty_count"] += 1

                    # 收集两种奖励视角的数值
                    for reward_key in REWARD_VIEW_KEYS:
                        # 静态基准奖励需存在可用上下界才可比，否则跳过
                        if reward_key == "metric_static_base_reward" and not _has_static_bounds(sample.metadata):
                            continue
                        # 安全转换为有限浮点数
                        reward_value = _to_finite_float(sample.metadata.get(reward_key))
                        if reward_value is not None:
                            # 数值有效时同时记入该模式桶与总体收集器
                            target_data["reward_views"][reward_key].append(reward_value)
                            overall_reward_views[reward_key].append(reward_value)

                    # Collect status codes
                    # 收集样本元数据里的 HTTP 状态码序列
                    if "status_codes" in sample.metadata:
                        # 同时追加到该模式桶与总体列表
                        target_data["status_codes"].extend(sample.metadata["status_codes"])
                        status_codes.extend(sample.metadata["status_codes"])
                    # 取该样本的状态码列表（缺省为空列表）
                    sample_status_codes = sample.metadata.get("status_codes", [])
                    # 取最后一个状态码作为本次运行的最终请求结果；非列表或为空则记 None
                    latest_status_code = (
                        sample_status_codes[-1]
                        if isinstance(sample_status_codes, list) and sample_status_codes
                        else None
                    )
                    if latest_status_code == 200 or hack_value == 1:
                        # 最终请求成功（200）或为作弊样本时计为"成功"
                        target_data["success_count"] += 1

                    # Collect scores
                    # 收集样本分数
                    if "score" in sample.metadata:
                        score_val = sample.metadata["score"]
                        if score_val is not None:
                            # 分数有效时同时记入该模式桶与总体列表
                            target_data["scores"].append(score_val)
                            scores.append(score_val)

                    # Collect scores per task
                    # 按任务名归档分数（模式内与总体各一份），供奖牌统计使用
                    if "task_name" in sample.metadata and "score" in sample.metadata:
                        task_name = sample.metadata["task_name"]
                        score_val = sample.metadata["score"]
                        if task_name not in target_data["scores_every_tasks"]:
                            # 该模式内首次出现该任务时初始化列表
                            target_data["scores_every_tasks"][task_name] = []
                        # 把分数追加到该模式桶对应任务的列表
                        target_data["scores_every_tasks"][task_name].append(score_val)
                        if task_name not in scores_every_tasks:
                            # 总体首次出现该任务时初始化列表
                            scores_every_tasks[task_name] = []
                        # 把分数追加到总体对应任务的列表
                        scores_every_tasks[task_name].append(score_val)

                    # Collect response lengths (using framework's method, same as training)
                    # 计算响应长度：与训练同口径，优先按 loss_mask 统计有效长度
                    response_length = sum(sample.loss_mask) if sample.loss_mask is not None else sample.response_length
                    # 记入该模式的响应长度列表
                    target_data["response_lengths"].append(response_length)

            # 打印按任务归档的分数，便于调试核对
            print(f"scores_every_tasks: {scores_every_tasks}")
            # Compute overall statistics (all modes combined)
            # 汇总四种模式的响应长度（全体样本）
            all_response_lengths = (
                debug_data["response_lengths"]
                + draft_data["response_lengths"]
                + improve_data["response_lengths"]
                + crossover_data["response_lengths"]
            )
            # 汇总四种模式的作弊样本数
            total_hack_count = (
                debug_data["hack_count"]
                + draft_data["hack_count"]
                + improve_data["hack_count"]
                + crossover_data["hack_count"]
            )
            # 汇总四种模式的空输出样本数
            total_empty_count = (
                debug_data["empty_count"]
                + draft_data["empty_count"]
                + improve_data["empty_count"]
                + crossover_data["empty_count"]
            )
            # 汇总四种模式的成功样本数
            total_success_count = (
                debug_data["success_count"]
                + draft_data["success_count"]
                + improve_data["success_count"]
                + crossover_data["success_count"]
            )
            # 汇总四种模式的样本总数
            total_sample_count = (
                debug_data["total_count"]
                + draft_data["total_count"]
                + improve_data["total_count"]
                + crossover_data["total_count"]
            )

            # Code category statistics
            # 全体样本数
            total_samples = len(samples)
            # 初始化五种代码类别的计数（execution-grounded 校验结果分类）
            code_category_counts = {
                "hack": 0,
                "hack_verify": 0,
                "no_verify": 0,
                "empty": 0,
                "valid": 0,
            }

            # 遍历样本统计代码类别分布
            for sample in samples:
                if hasattr(sample, "metadata") and isinstance(sample.metadata, dict):
                    # 读取样本的代码类别，缺省按 valid 处理
                    code_category = sample.metadata.get("code_category", "valid")
                    if code_category in code_category_counts:
                        # 已知类别则对应计数加一
                        code_category_counts[code_category] += 1

            if total_samples > 0:
                # 有样本时输出每个类别的数量与占比
                for code_category, count in code_category_counts.items():
                    log_dict[f"eval/{key}-code_category/{code_category}/count"] = count
                    log_dict[f"eval/{key}-code_category/{code_category}/ratio"] = count / total_samples

            # 🆕 Log statistics for each prompt_type (debug/draft/improve/crossover)
            # 对每种算子模式分别输出统计
            for mode_name, mode_data in [
                ("debug", debug_data),
                ("draft", draft_data),
                ("improve", improve_data),
                ("crossover", crossover_data),
            ]:
                # 输出该模式两种奖励视角的均值
                _log_reward_view_means(log_dict, f"eval/{key}-{mode_name}", mode_data["reward_views"])
                # 输出该模式的质量代理指标（GRPO 组内最优等）
                _log_quality_proxy_metrics(log_dict, f"eval/{key}-{mode_name}/quality", mode_data["samples"])

                if mode_data["status_codes"]:
                    # 该模式收集到状态码时统计 HTTP 结果分布
                    total_status = len(mode_data["status_codes"])
                    # 200（成功）占比
                    log_dict[f"eval/{key}-{mode_name}/http_200_rate"] = (
                        mode_data["status_codes"].count(200) / total_status
                    )
                    # 503（连接错误）占比
                    log_dict[f"eval/{key}-{mode_name}/connection_error_rate"] = (
                        mode_data["status_codes"].count(503) / total_status
                    )
                    # 其他错误（非 200/503）占比
                    log_dict[f"eval/{key}-{mode_name}/other_error_rate"] = (
                        sum(1 for c in mode_data["status_codes"] if c not in [200, 503]) / total_status
                    )  # A 504 response indicates a timeout.

                if mode_data["response_lengths"]:
                    # 有响应长度数据时输出平均与最大响应长度
                    log_dict[f"eval/{key}-{mode_name}/avg_response_length"] = sum(mode_data["response_lengths"]) / len(
                        mode_data["response_lengths"]
                    )
                    log_dict[f"eval/{key}-{mode_name}/max_response_length"] = max(mode_data["response_lengths"])

                # Log hack statistics per mode
                # 输出该模式的作弊/空输出/成功率统计
                if mode_data["total_count"] > 0:
                    # 该模式作弊率
                    log_dict[f"eval/{key}-{mode_name}/hack_rate"] = mode_data["hack_count"] / mode_data["total_count"]
                    # 该模式作弊样本数
                    log_dict[f"eval/{key}-{mode_name}/hack_count"] = mode_data["hack_count"]
                    # 该模式空输出率
                    log_dict[f"eval/{key}-{mode_name}/empty_rate"] = mode_data["empty_count"] / mode_data["total_count"]
                    # 该模式空输出样本数
                    log_dict[f"eval/{key}-{mode_name}/empty_count"] = mode_data["empty_count"]
                    # 该模式成功率
                    log_dict[f"eval/{key}-{mode_name}/success_rate"] = (
                        mode_data["success_count"] / mode_data["total_count"]
                    )
                    # 该模式成功样本数
                    log_dict[f"eval/{key}-{mode_name}/success_count"] = mode_data["success_count"]
                    # 该模式样本总数
                    log_dict[f"eval/{key}-{mode_name}/total_count"] = mode_data["total_count"]

                # Medal statistics per mode - count medals for EACH sample, not just best score
                # Distinguish scored samples without medals from samples without scores.
                # 该模式各奖牌等级计数（对每个样本分别判奖牌，而非只看最优分）
                mode_medal_cnt = {"Gold": 0, "Silver": 0, "Bronze": 0, "None": 0}
                mode_scored_count = 0  # Total number of samples with valid scores.
                # 各任务获得奖牌的样本数
                mode_task_medal_counts = {}

                # 遍历该模式下每个任务的分数列表，逐一判奖牌
                for task_name, task_scores in mode_data["scores_every_tasks"].items():
                    try:
                        # 加载该任务的公开排行榜
                        leaderboard = load_public_leaderboard(task_name)
                        if task_scores and leaderboard is not None and "score" in leaderboard.columns:
                            # Count medal for EACH score (each sample), not just the best
                            # 对每个分数（每个样本）逐一判奖牌
                            for score in task_scores:
                                if score is not None:
                                    # 分数有效时按排行榜判奖牌等级
                                    medal_per_score = get_medal_for_score(score, leaderboard)
                                    # 累加该等级的计数
                                    mode_medal_cnt[medal_per_score] += 1
                                    # 有分数样本计数加一
                                    mode_scored_count += 1
                                    if medal_per_score != "None":
                                        # 拿到奖牌时给对应任务的奖牌数加一
                                        mode_task_medal_counts[task_name] = (
                                            mode_task_medal_counts.get(task_name, 0) + 1
                                        )
                                    # 打印每个分数的判牌结果，便于调试
                                    print(f"MEDAL: {medal_per_score} for score: {score} in task: {task_name}")
                        else:
                            # 排行榜缺失或格式不符时的诊断与兜底
                            if task_scores is None:
                                print(f"Warning: Task scores are None for task '{task_name}'")
                            if leaderboard is None:
                                print(f"Warning: Leaderboard is None for task '{task_name}'")
                            elif "score" not in leaderboard.columns:
                                print(f"Warning: Score column not found in leaderboard for task '{task_name}'")
                            # 统计有效分数个数，全部按"无奖牌"处理
                            valid_score_count = len([s for s in (task_scores or []) if s is not None])
                            # 计入 medal_none_count
                            mode_medal_cnt["None"] += valid_score_count
                            # 计入有分数样本数
                            mode_scored_count += valid_score_count
                            print(f"MEDAL: Cannot get medal for task '{task_name}'")
                    except Exception as e:
                        # If leaderboard not found, count all scores as None
                        # 判奖牌过程异常时，所有有效分数按"无奖牌"处理
                        valid_score_count = len([s for s in (task_scores or []) if s is not None])
                        mode_medal_cnt["None"] += valid_score_count
                        mode_scored_count += valid_score_count
                        print(f"Warning: Exception when getting medal '{task_name}': {e}")

                # Calculate metrics
                # 无分数样本数 = 该模式总数 - 有分数样本数
                mode_no_score_count = mode_data["total_count"] - mode_scored_count  # Samples without scores.

                # Always record scored and unscored counts, regardless of medal availability.
                # 无论排行榜是否可用，都记录有分数样本数
                log_dict[f"eval/{key}-{mode_name}/scored_count"] = mode_scored_count  # Number of scored samples.

                # 输出该模式的金牌计数
                log_dict[f"eval/{key}-{mode_name}/medal_gold_count"] = mode_medal_cnt["Gold"]
                # 输出该模式的银牌计数
                log_dict[f"eval/{key}-{mode_name}/medal_silver_count"] = mode_medal_cnt["Silver"]
                # 输出该模式的铜牌计数
                log_dict[f"eval/{key}-{mode_name}/medal_bronze_count"] = mode_medal_cnt["Bronze"]
                # 输出该模式奖牌总数（金+银+铜）
                log_dict[f"eval/{key}-{mode_name}/medal_count"] = (
                    mode_medal_cnt["Gold"] + mode_medal_cnt["Silver"] + mode_medal_cnt["Bronze"]
                )  # Total awarded medals.
                # 输出有分数但未获奖牌的样本数
                log_dict[f"eval/{key}-{mode_name}/medal_none_count"] = mode_medal_cnt["None"]  # Scored but no medal.
                # Medal rate across all samples, including samples without scores.
                # 分母含无分数样本的奖牌率与有分数率
                if mode_data["total_count"] > 0:
                    # 该模式奖牌率（分母为全部样本）
                    log_dict[f"eval/{key}-{mode_name}/medal_rate_total"] = (
                        mode_medal_cnt["Gold"] + mode_medal_cnt["Silver"] + mode_medal_cnt["Bronze"]
                    ) / mode_data["total_count"]
                    # 该模式有分数样本占比
                    log_dict[f"eval/{key}-{mode_name}/scored_rate_total"] = (
                        mode_scored_count / mode_data["total_count"]
                    )

                # Log per-task medal counts for this mode
                # 输出该模式下每个任务的奖牌数
                for task_name, task_medal_count in mode_task_medal_counts.items():
                    log_dict[f"eval/{key}-{mode_name}-{task_name}/task_medal_count"] = task_medal_count

            # Overall medal statistics - count medals for EACH sample, not just best score
            # Distinguish scored samples without medals from samples without scores.
            # 总体（不分模式）各奖牌等级计数
            medal_cnt = {"Gold": 0, "Silver": 0, "Bronze": 0, "None": 0}
            overall_scored_count = 0  # Total number of samples with valid scores.
            # 总体各任务获得奖牌的样本数
            overall_task_medal_counts = {}

            # 遍历全体样本按任务归档的分数，逐一判奖牌
            for task_name, task_scores in scores_every_tasks.items():
                try:
                    # 加载该任务的公开排行榜
                    leaderboard = load_public_leaderboard(task_name)
                    if task_scores and leaderboard is not None and "score" in leaderboard.columns:
                        # Count medal for EACH score (each sample), not just the best
                        # 对每个分数（每个样本）逐一判奖牌
                        for score in task_scores:
                            if score is not None:
                                # 分数有效时按排行榜判奖牌等级
                                medal_per_score = get_medal_for_score(score, leaderboard)
                                # 累加该等级的计数
                                medal_cnt[medal_per_score] += 1
                                # 有分数样本计数加一
                                overall_scored_count += 1
                                if medal_per_score != "None":
                                    # 拿到奖牌时给对应任务的奖牌数加一
                                    overall_task_medal_counts[task_name] = (
                                        overall_task_medal_counts.get(task_name, 0) + 1
                                    )
                    else:
                        # 排行榜缺失或格式不符时的诊断与兜底
                        if task_scores is None:
                            print(f"  Warning: Task scores are None for task '{task_name}'")
                        if leaderboard is None:
                            print(f"  Warning: Leaderboard is None for task '{task_name}'")
                        elif "score" not in leaderboard.columns:
                            print(f"  Warning: Score column not found in leaderboard for task '{task_name}'")
                        # 统计有效分数个数，全部按"无奖牌"处理
                        valid_score_count = len([s for s in (task_scores or []) if s is not None])
                        # 计入 medal_none_count
                        medal_cnt["None"] += valid_score_count
                        # 计入总体有分数样本数
                        overall_scored_count += valid_score_count
                        print(f"  Warning: Cannot load leaderboard for task '{task_name}'")
                except Exception as e:
                    # If leaderboard not found, count all scores as None
                    # 异常时所有有效分数按"无奖牌"处理
                    valid_score_count = len([s for s in (task_scores or []) if s is not None])
                    medal_cnt["None"] += valid_score_count
                    overall_scored_count += valid_score_count
                    print(f"  Warning: Cannot load leaderboard for task '{task_name}': {e}")

            # Always record the base statistics.
            # 记录总体有分数样本数
            log_dict[f"eval/{key}-scored_count"] = overall_scored_count  # Number of scored samples.
            # Record medal statistics.
            # 输出总体金牌计数
            log_dict[f"eval/{key}-medal_gold_count"] = medal_cnt["Gold"]
            # 输出总体银牌计数
            log_dict[f"eval/{key}-medal_silver_count"] = medal_cnt["Silver"]
            # 输出总体铜牌计数
            log_dict[f"eval/{key}-medal_bronze_count"] = medal_cnt["Bronze"]
            # 输出总体奖牌总数（金+银+铜）
            log_dict[f"eval/{key}-medal_count"] = (
                medal_cnt["Gold"] + medal_cnt["Silver"] + medal_cnt["Bronze"]
            )  # Total awarded medals.
            # 输出有分数但未获奖牌的样本数
            log_dict[f"eval/{key}-medal_none_count"] = medal_cnt["None"]  # Scored but no medal.

            # Medal rate across all samples, including samples without scores.
            # 分母含无分数样本的总体奖牌率
            if total_sample_count > 0:
                medal_rate_total = (
                    medal_cnt["Gold"] + medal_cnt["Silver"] + medal_cnt["Bronze"]
                ) / total_sample_count
                # 记录总体奖牌率（带数据源前缀）
                log_dict[f"eval/{key}-medal_rate_total"] = medal_rate_total
                # 以通用键再记一份，兼容下游看板
                log_dict[f"eval/{key}/medal_rate"] = medal_rate_total
                # 记录总体有分数样本占比
                log_dict[f"eval/{key}-scored_rate_total"] = overall_scored_count / total_sample_count

            # Log per-task total medal counts (all modes combined)
            # 输出各任务的总体奖牌数
            for task_name, task_medal_count in overall_task_medal_counts.items():
                log_dict[f"eval/{key}-{task_name}/task_medal_count"] = task_medal_count

            # 输出总体两种奖励视角的均值
            _log_reward_view_means(log_dict, f"eval/{key}", overall_reward_views)
            # 输出总体质量代理指标（GRPO 组内最优等）
            _log_quality_proxy_metrics(log_dict, f"eval/{key}/quality", samples)

            # Log status code metrics
            # 输出总体 HTTP 状态码分布
            if status_codes:
                # 状态码总数
                total_status = len(status_codes)
                # 200（成功）占比
                log_dict[f"eval/{key}-http_200_rate"] = status_codes.count(200) / total_status
                # 503（连接错误）占比
                log_dict[f"eval/{key}-connection_error_rate"] = status_codes.count(503) / total_status
                # 其他错误（如 504 超时）占比
                log_dict[f"eval/{key}-other_error_rate"] = (
                    sum(1 for c in status_codes if c not in [200, 503]) / total_status
                )

            # Log score statistics
            # if scores:
            # log_dict[f"eval/{key}-avg_score"] = sum(scores) / len(scores)
            # log_dict[f"eval/{key}-max_score"] = max(scores)
            # log_dict[f"eval/{key}-min_score"] = min(scores)
            # log_dict[f"eval/{key}-score_std"] = (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)) ** 0.5

            # Log overall response length statistics
            # 输出全体样本的平均与最大响应长度
            if all_response_lengths:
                log_dict[f"eval/{key}-avg_response_length"] = sum(all_response_lengths) / len(all_response_lengths)
                log_dict[f"eval/{key}-max_response_length"] = max(all_response_lengths)

            # Log overall hack statistics
            # 输出总体的作弊/空输出/成功率统计
            if total_sample_count > 0:
                # 总体作弊率
                log_dict[f"eval/{key}-hack_rate"] = total_hack_count / total_sample_count
                # 总体作弊样本数
                log_dict[f"eval/{key}-hack_count"] = total_hack_count
                # 总体空输出率
                log_dict[f"eval/{key}-empty_rate"] = total_empty_count / total_sample_count
                # 总体空输出样本数
                log_dict[f"eval/{key}-empty_count"] = total_empty_count
                # 总体成功率
                log_dict[f"eval/{key}-success_rate"] = total_success_count / total_sample_count
                # 总体成功样本数
                log_dict[f"eval/{key}-success_count"] = total_success_count
                # 总体样本总数
                log_dict[f"eval/{key}-total_sample_count"] = total_sample_count

        # Original truncated ratio
        # 记录截断比例（响应被截断的样本占比）
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)

        # Original pass rate metrics
        # 若启用 log_passrate，则额外输出通过率（pass rate）指标
        if args.log_passrate:
            log_dict |= dict_add_prefix(
                compute_pass_rate(
                    flat_rewards=rewards,
                    group_size=args.n_samples_per_eval_prompt,
                ),
                f"eval/{key}-",
            )

    # 打印完整日志字典，便于终端直接查看
    print(f"[CUSTOM EVAL LOGGER] eval {rollout_id}: {log_dict}")

    # 把 rollout_id 换算为全局评估步数
    step = compute_rollout_step(args, rollout_id)
    # 在日志字典中记录评估步数
    log_dict["eval/step"] = step
    # 通过 slime 日志工具上报（以 eval/step 作为 step 键）
    logging_utils.log(args, log_dict, step_key="eval/step")

    # Return True so --custom-eval-rollout-log-function-path skips stock logging.
    # 返回 True 告知框架已完成自定义日志，跳过内置日志逻辑
    return True


# Apply the monkey patch
def apply_patch():
    """Apply monkey patch to _log_eval_rollout_data"""
    try:
        # 导入 slime 的 rollout 模块，准备替换其中的评估日志函数
        import slime.ray.rollout

        # 用本模块的 custom_log_eval_rollout_data 替换 slime 内置的 _log_eval_rollout_data（猴子补丁核心）
        slime.ray.rollout._log_eval_rollout_data = custom_log_eval_rollout_data
        # Stock RolloutManager.eval already forwards metrics to _log_eval_rollout_data;
        # keep a thin wrapper only for compatibility with older call sites.
        # 保存原版 RolloutManager.eval，便于判断是否重复打补丁
        original_eval = slime.ray.rollout.RolloutManager.eval

        def eval_with_log_dict(self, rollout_id):
            # 包装版 eval：先处理调试模式与健康检查，再执行评估并调用（已被替换的）日志函数
            if self.args.debug_train_only:
                # 仅调试训练模式时跳过评估
                return None
            # 恢复健康监控状态
            self.health_monitoring_resume()

            # 调用框架的评估 rollout 生成流程
            result = slime.ray.rollout.call_rollout_fn(
                self.eval_generate_rollout,
                self.args,
                rollout_id,
                self.data_source,
                evaluation=True,
            )
            # 取出生成的评估数据
            data = result.data
            # 保存调试用评估数据
            self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=True)
            # 调用（已被补丁替换的）评估日志函数并返回其结果
            return slime.ray.rollout._log_eval_rollout_data(rollout_id, self.args, data, result.metrics)

        if getattr(original_eval, "__name__", "") != "eval_with_log_dict":
            # 避免重复打补丁：只有原 eval 不是我们的包装版时才替换
            slime.ray.rollout.RolloutManager.eval = eval_with_log_dict
        # 打印补丁成功信息
        print("✅ [PATCH_EVAL_LOGGER] Successfully patched _log_eval_rollout_data")
        # 返回补丁成功标志
        return True
    except Exception as e:
        # 补丁失败时打印异常信息
        print(f"⚠️ [PATCH_EVAL_LOGGER] Failed to patch: {e}")
        import traceback

        # 打印完整异常堆栈，便于定位失败原因
        traceback.print_exc()
        # 返回补丁失败标志
        return False


def test_medal_statistics_integration():
    """Integration test for medal statistics with simulated samples"""
    # 导入临时目录工具，用于构造测试用排行榜
    import tempfile
    import shutil
    # 导入命名元组，用于构造模拟样本
    from collections import namedtuple

    # 打印测试标题分隔线
    print("\n" + "=" * 80)
    print("TESTING MEDAL STATISTICS INTEGRATION")
    print("=" * 80)

    # Create a temporary leaderboard for testing
    # 创建临时目录存放测试排行榜
    temp_dir = tempfile.mkdtemp()
    # 测试用任务名
    test_task_name = "tabular-playground-series-dec-2021"
    # 该任务在临时目录下的路径
    test_leaderboard_dir = os.path.join(temp_dir, test_task_name)
    # 创建任务目录
    os.makedirs(test_leaderboard_dir, exist_ok=True)

    # Create mock samples with different prompt_types and scores
    # 定义仅含 metadata 字段的模拟样本类型
    MockSample = namedtuple("MockSample", ["metadata"])

    # 构造覆盖 debug/draft/improve 三种模式、含缺失分数的模拟样本集合（crossover 在此无样本）
    test_samples = [
        # Debug samples (3 with scores + 1 without score)
        MockSample(metadata={"prompt_type": "debug", "task_name": test_task_name, "score": 0.99}),
        MockSample(metadata={"prompt_type": "debug", "task_name": test_task_name, "score": 0.85}),
        MockSample(metadata={"prompt_type": "debug", "task_name": test_task_name, "score": 0.70}),
        MockSample(metadata={"prompt_type": "debug", "task_name": test_task_name}),  # No score
        # Draft samples (4 with scores + 1 without score)
        MockSample(metadata={"prompt_type": "draft", "task_name": test_task_name, "score": 0.955}),
        MockSample(metadata={"prompt_type": "draft", "task_name": test_task_name, "score": 0.82}),
        MockSample(metadata={"prompt_type": "draft", "task_name": test_task_name, "score": 0.65}),
        MockSample(metadata={"prompt_type": "draft", "task_name": test_task_name, "score": 0.50}),
        MockSample(metadata={"prompt_type": "draft", "task_name": test_task_name}),  # No score
        # Improve samples (5 with scores + 1 without score)
        MockSample(metadata={"prompt_type": "improve", "task_name": test_task_name, "score": 0.93}),
        MockSample(metadata={"prompt_type": "improve", "task_name": test_task_name, "score": 0.88}),
        MockSample(metadata={"prompt_type": "improve", "task_name": test_task_name, "score": 0.75}),
        MockSample(metadata={"prompt_type": "improve", "task_name": test_task_name, "score": 0.62}),
        MockSample(metadata={"prompt_type": "improve", "task_name": test_task_name, "score": 0.45}),
        MockSample(metadata={"prompt_type": "improve", "task_name": test_task_name}),  # No score
    ]

    # Process samples and collect statistics (simulating the logic in custom_log_eval_rollout_data)
    # 模拟 custom_log_eval_rollout_data 的分桶逻辑，为每种模式建立统计结构
    debug_data = {"scores_every_tasks": {}, "total_count": 0}
    draft_data = {"scores_every_tasks": {}, "total_count": 0}
    improve_data = {"scores_every_tasks": {}, "total_count": 0}
    crossover_data = {"scores_every_tasks": {}, "total_count": 0}
    # 总体按任务名归档的分数
    scores_every_tasks = {}

    # 遍历模拟样本完成分桶
    for sample in test_samples:
        # 读取算子模式类型，缺省按 draft 处理
        prompt_type = sample.metadata.get("prompt_type", "draft")
        # 读取任务名
        task_name = sample.metadata["task_name"]
        score = sample.metadata.get("score")  # Use .get() to handle missing scores

        # 依据模式选择目标统计桶
        if prompt_type == "debug":
            target_data = debug_data
        elif prompt_type == "improve":
            target_data = improve_data
        elif prompt_type == "crossover":
            target_data = crossover_data
        else:
            # draft 或未知模式归入 draft 桶
            target_data = draft_data

        # Count all samples
        # 累计该模式样本总数
        target_data["total_count"] += 1

        # Only add scores if they exist and are valid
        # 只归档存在且非 NaN 的有效分数
        if score is not None and not (isinstance(score, float) and np.isnan(score)):
            if task_name not in target_data["scores_every_tasks"]:
                # 该模式内首次出现该任务时初始化列表
                target_data["scores_every_tasks"][task_name] = []
            # 把分数追加到该模式对应任务的列表
            target_data["scores_every_tasks"][task_name].append(score)

            if task_name not in scores_every_tasks:
                # 总体首次出现该任务时初始化列表
                scores_every_tasks[task_name] = []
            # 把分数追加到总体对应任务的列表
            scores_every_tasks[task_name].append(score)

    # Calculate medals for each mode
    # 通过标志（当前逻辑恒为 True，保留原有结构）
    all_passed = True

    # 对每种算子模式分别计算奖牌统计
    for mode_name, mode_data in [
        ("debug", debug_data),
        ("draft", draft_data),
        ("improve", improve_data),
        ("crossover", crossover_data),
    ]:
        # 该模式各奖牌等级计数
        mode_medal_cnt = {"Gold": 0, "Silver": 0, "Bronze": 0, "None": 0}
        # 该模式有分数样本数
        mode_scored_count = 0

        # 遍历该模式各任务的分数，逐一判奖牌
        for task_name, task_scores in mode_data["scores_every_tasks"].items():
            try:
                # 加载该任务的公开排行榜
                leaderboard = load_public_leaderboard(task_name)
                if task_scores and leaderboard is not None and "score" in leaderboard.columns:
                    # 对每个分数逐一判奖牌
                    for score in task_scores:
                        # 按排行榜判奖牌等级
                        medal_per_score = get_medal_for_score(score, leaderboard)
                        # 累加该等级的计数
                        mode_medal_cnt[medal_per_score] += 1
                        # 有分数样本计数加一
                        mode_scored_count += 1
            except Exception as e:
                # 异常时所有分数按"无奖牌"处理
                mode_medal_cnt["None"] += len(task_scores)
                mode_scored_count += len(task_scores)

        # 无分数样本数 = 该模式总数 - 有分数样本数
        mode_no_score_count = mode_data["total_count"] - mode_scored_count
        # 有分数样本总数（各奖牌等级计数之和）
        total = sum(mode_medal_cnt.values())
        # 有分数样本中的奖牌率
        medal_rate = (
            (mode_medal_cnt["Gold"] + mode_medal_cnt["Silver"] + mode_medal_cnt["Bronze"]) / total
            if total > 0
            else 0.0
        )
        # 全部样本（含无分数）中的奖牌率
        medal_rate_total = (
            (mode_medal_cnt["Gold"] + mode_medal_cnt["Silver"] + mode_medal_cnt["Bronze"]) / mode_data["total_count"]
            if mode_data["total_count"] > 0
            else 0.0
        )

        # 以下打印该模式统计结果：金/银/铜计数、有分数但无奖牌数、无分数样本数、有分数样本数、总数及两种口径的奖牌率
        print(f"\n{mode_name.upper()} Results:")
        print(f"  Gold: {mode_medal_cnt['Gold']} ")
        print(f"  Silver: {mode_medal_cnt['Silver']} ")
        print(f"  Bronze: {mode_medal_cnt['Bronze']} ")
        print(f"  None (scored, no medal): {mode_medal_cnt['None']} ")
        print(f"  No Score: {mode_no_score_count} ")
        print(f"  Scored Count: {mode_scored_count} ")
        print(f"  Total Count: {mode_data['total_count']} ")
        print(f"  Medal Rate (scored samples): {medal_rate:.4f} ")
        print(f"  Medal Rate Total (all samples): {medal_rate_total:.4f} ")

    # Calculate overall medals
    # 总体各奖牌等级计数
    medal_cnt = {"Gold": 0, "Silver": 0, "Bronze": 0, "None": 0}
    # 总体有分数样本数
    overall_scored_count = 0

    # 遍历总体按任务归档的分数，逐一判奖牌
    for task_name, task_scores in scores_every_tasks.items():
        try:
            # 加载该任务的公开排行榜
            leaderboard = load_public_leaderboard(task_name)
            if task_scores and leaderboard is not None and "score" in leaderboard.columns:
                # 对每个分数逐一判奖牌
                for score in task_scores:
                    if score is not None:
                        # 分数有效时按排行榜判奖牌等级
                        medal_per_score = get_medal_for_score(score, leaderboard)
                        # 累加该等级的计数
                        medal_cnt[medal_per_score] += 1
                        # 有分数样本计数加一
                        overall_scored_count += 1
        except Exception as e:
            # 异常时所有有效分数按"无奖牌"处理
            medal_cnt["None"] += len([s for s in task_scores if s is not None])
            overall_scored_count += len([s for s in task_scores if s is not None])

    # 总样本数 = 四种模式样本总数之和
    total_sample_count = (
        debug_data["total_count"] + draft_data["total_count"] + improve_data["total_count"] + crossover_data["total_count"]
    )
    # 总体无分数样本数
    overall_no_score_count = total_sample_count - overall_scored_count
    # 有分数样本总数
    total = sum(medal_cnt.values())
    # 有分数样本中的总体奖牌率
    overall_medal_rate = (medal_cnt["Gold"] + medal_cnt["Silver"] + medal_cnt["Bronze"]) / total if total > 0 else 0.0
    # 全部样本（含无分数）中的总体奖牌率
    overall_medal_rate_total = (
        (medal_cnt["Gold"] + medal_cnt["Silver"] + medal_cnt["Bronze"]) / total_sample_count
        if total_sample_count > 0
        else 0.0
    )

    # 以下打印总体统计结果：金/银/铜计数、有分数但无奖牌数、无分数样本数、有分数样本数、总数及两种口径的奖牌率
    print(f"\nOVERALL Results:")
    print(f"  Gold: {medal_cnt['Gold']}")
    print(f"  Silver: {medal_cnt['Silver']} ")
    print(f"  Bronze: {medal_cnt['Bronze']} ")
    print(f"  None (scored, no medal): {medal_cnt['None']} ")
    print(f"  No Score: {overall_no_score_count} ")
    print(f"  Scored Count: {overall_scored_count} ")
    print(f"  Total Count: {total_sample_count} ")
    print(f"  Medal Rate (scored samples): {overall_medal_rate:.4f} ")
    print(f"  Medal Rate Total (all samples): {overall_medal_rate_total:.4f} ")

    # Clean up
    # 删除临时目录，清理测试产物
    shutil.rmtree(temp_dir)

    # 打印测试结果分隔线
    print("\n" + "=" * 80)
    if all_passed:
        # 通过标志为真时打印通过
        print("✅ INTEGRATION TEST PASSED")
    else:
        # 否则打印失败
        print("❌ INTEGRATION TEST FAILED")
    print("=" * 80 + "\n")

    # 返回测试是否全部通过
    return all_passed


# Auto-apply patch when this module is imported
# 模块被导入（而非直接运行）时自动应用补丁
if __name__ != "__main__":
    apply_patch()
else:
    # Run tests when executed directly
    # 直接运行本文件时执行奖牌统计集成测试
    print("Running medal calculation tests...")
    test2_passed = test_medal_statistics_integration()

    # 打印最终测试总结
    print("\n" + "=" * 80)
    print("FINAL TEST SUMMARY")
    print("=" * 80)
    # 打印集成统计测试结果
    print(f"Integration statistics test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print("=" * 80)
