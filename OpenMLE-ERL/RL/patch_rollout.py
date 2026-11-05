"""
Monkey patch for custom rollout behaviors.
This file should be imported BEFORE slime modules are loaded.
"""

# =====================================================================
# 中文注释导读（OpenRSI / OpenMLE-RL 学习文档）：
# 本模块用猴子补丁替换 slime 的 rollout 日志函数（_log_rollout_data）与
# 程序库快照方法（RolloutManager.save_db_snapshot），为 execution-grounded RL
# 增加"算子级"训练观测：把每个 rollout batch 按 Draft/Improve/Debug/Crossover
# 四个原子算子分桶，分别统计奖励视图（奖励已对任务静态基线归一）、hack 率、
# 空程序率、成功率与得分率。训练者据此判断"哪个算子在变强、哪个算子在摆烂"。
# 必须在 slime 模块加载之前 import（train_with_patch.py 已保证该顺序）。
# =====================================================================

import asyncio  # patch 中可能遇到协程返回值（数据库快照为 awaitable 时）
import inspect  # inspect.isawaitable 判断快照结果是否需要 await
import os  # 读取 SAVE_DB_SNAPSHOTS / RUN_DIR 环境变量

import numpy as np  # 均值与奖励张量运算（GRPO 类组内归一）
import torch  # 奖励张量（组内去均值/除标准差）
from slime.utils import logging_utils  # slime 统一日志出口（wandb/tensorboard）
from slime.utils.metric_utils import compute_rollout_step  # 由 rollout_id 计算全局 step


# 两种"奖励视图"：
#   metric_static_base_reward —— 相对任务静态基线（如 0 分/全错基线）归一后的奖励，
#                                跨任务可比，是 execution-grounded 奖励的主视图
#   dynamic_raw_reward        —— 任务原始动态指标，保留任务内绝对量纲
REWARD_VIEW_KEYS = (
    "metric_static_base_reward",
    "dynamic_raw_reward",
)

# 质量代理指标：不直接进奖励，只用于观测策略是否真的在"改进任务"
QUALITY_PROXY_KEYS = (
    "metric_static_base_reward",
    "metric_static_base_reward_is_task_best_improvement",  # 本次是否刷新了组内历史最优
    "history_best_static_base_reward",  # 组内历史最优奖励
)


def _to_finite_float(value):
    # 安全转有限浮点：None/不可转/inf/nan 一律返回 None（统计时跳过）
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _mean_or_none(values):
    # 空列表返回 None（日志里省略该键），否则求均值
    return float(np.mean(values)) if values else None


def _has_static_bounds(metadata):
    # 判断任务是否存在"静态基线边界"（理论或榜单的最好/最差分数）：
    # 只有存在边界，metric_static_base_reward 的归一化才有意义
    flag = metadata.get("metric_static_base_reward_has_bounds")
    if flag is not None:
        try:
            # 任务侧已显式标记
            return bool(int(flag))
        except (TypeError, ValueError):
            pass
    higher_is_better = metadata.get("higher_is_better")
    theoretical_min = _to_finite_float(metadata.get("theoretical_min"))
    theoretical_max = _to_finite_float(metadata.get("theoretical_max"))
    leaderboard_min = _to_finite_float(metadata.get("leaderboard_min"))
    leaderboard_max = _to_finite_float(metadata.get("leaderboard_max"))
    if higher_is_better is False:
        # 分数越小越好：best 是最小值方向
        best = leaderboard_min if leaderboard_min is not None else theoretical_min
        worst = leaderboard_max if leaderboard_max is not None else theoretical_max
    else:
        # 分数越大越好：best 是最大值方向
        best = leaderboard_max if leaderboard_max is not None else theoretical_max
        worst = leaderboard_min if leaderboard_min is not None else theoretical_min
    return best is not None and worst is not None


def _build_mode_bucket():
    # 构造单个算子分桶的空统计容器（draft/debug/improve/crossover 各一个）
    return {
        "samples": [],  # 该桶内样本引用（供质量代理指标复用）
        "reward_views": {key: [] for key in REWARD_VIEW_KEYS},  # 两类奖励视图的取值列表
        "hack_count": 0,  # 被 hack 检测判为作弊的样本数
        "empty_count": 0,  # 产出空程序（无有效代码）的样本数
        "success_count": 0,  # 执行成功（HTTP 200 或被标记 hack）的样本数
        "network_success_count": 0,  # 网络层成功（200 或显式网络错误码族）的样本数
        "scored_count": 0,  # 拿到分数的样本数
        "total_count": 0,  # 桶内样本总数
    }


def _log_reward_view_means(log_dict, prefix, reward_views):
    # 把一组奖励视图的均值写入日志字典（前缀区分算子或全局）
    for reward_key, values in reward_views.items():
        mean_value = _mean_or_none(values)
        if mean_value is not None:
            # 主视图重命名为 static_base_reward（更简洁的日志键）
            output_key = "static_base_reward" if reward_key == "metric_static_base_reward" else reward_key
            log_dict[f"{prefix}/{output_key}"] = mean_value
            log_dict[f"{prefix}/{output_key}_count"] = len(values)
            if reward_key == "metric_static_base_reward":
                # 主视图同时别名为 reward / reward_count，便于在面板中直接观察
                log_dict[f"{prefix}/reward"] = mean_value
                log_dict[f"{prefix}/reward_count"] = len(values)


def _log_quality_proxy_metrics(log_dict, prefix, samples):
    # 质量代理指标：按 rollout 组（同一任务的一组采样）聚合，观测"真实改进率"
    if not samples:
        return

    group_buckets = {}
    for sample_idx, sample in enumerate(samples):
        metadata = sample.metadata if hasattr(sample, "metadata") and isinstance(sample.metadata, dict) else {}
        # 组键优先级：metadata.group_index > sample.group_index > 样本序号兜底
        group_key = metadata.get("group_index")
        if group_key is None and hasattr(sample, "group_index"):
            group_key = sample.group_index
        if group_key is None:
            group_key = f"sample_{sample_idx}"
        bucket = group_buckets.setdefault(
            group_key,
            {
                "metric_static_base_reward": [],
                "metric_static_base_reward_is_task_best_improvement": [],
            },
        )
        for key in bucket:
            if key == "metric_static_base_reward" and not _has_static_bounds(metadata):
                # 任务无静态边界时该视图无意义，跳过
                continue
            value = _to_finite_float(metadata.get(key))
            if value is not None:
                bucket[key].append(value)

    if not group_buckets:
        return

    reward_key = "metric_static_base_reward"
    group_bests = [max(bucket[reward_key]) for bucket in group_buckets.values() if bucket[reward_key]]
    if group_bests:
        # 组内最优奖励的均值：衡量"每组最好能改到什么程度"
        log_dict[f"{prefix}/group_best_static_base_reward"] = float(np.mean(group_bests))
        log_dict[f"{prefix}/group_best_static_base_reward_count"] = len(group_bests)
        # 组内最优 >= 0.95 的比例：近似"接近满分"的任务占比
        log_dict[f"{prefix}/group_best_static_base_reward_ge_0p95_rate"] = float(
            np.mean([best >= 0.95 for best in group_bests])
        )

    group_improvement_counts = [
        float(sum(bucket["metric_static_base_reward_is_task_best_improvement"]))
        for bucket in group_buckets.values()
        if bucket["metric_static_base_reward"]
    ]
    if group_improvement_counts:
        # 刷新历史最优的比例：策略是否还在产生"新知识"
        log_dict[f"{prefix}/static_base_reward_is_task_best_improvement"] = float(np.mean(group_improvement_counts))

    history_values = [
        _to_finite_float(
            (sample.metadata if hasattr(sample, "metadata") and isinstance(sample.metadata, dict) else {}).get(
                "history_best_static_base_reward"
            )
        )
        for sample in samples
    ]
    history_values = [value for value in history_values if value is not None]
    if history_values:
        # 历史最优奖励均值：长期进步水位线
        log_dict[f"{prefix}/history_best_static_base_reward"] = float(np.mean(history_values))


def _compute_debug_rewards(args, samples):
    # 调试用：复算一遍原始奖励与"处理后的优势输入"，让日志里的 reward 可被独立核对
    import slime.ray.rollout as rollout_mod

    if getattr(args, "custom_reward_post_process_path", None):
        # 配置了自定义奖励后处理：直接调用（OpenMLE-RL 用它接入沙箱执行奖励）
        custom_post_process = rollout_mod.load_function(args.custom_reward_post_process_path)
        raw_rewards, processed_rewards = custom_post_process(args, samples)
        return list(raw_rewards), list(processed_rewards)

    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    if (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        # GRPO/GSPO/REINFORCE++(基线) 的组内归一：reshape 成 (组数, 每组样本数) 后去均值
        rewards = torch.tensor(raw_rewards, dtype=torch.float)
        if rewards.shape[-1] == args.n_samples_per_prompt * args.rollout_batch_size:
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)
        else:
            rewards = rewards.view(-1, rewards.shape[-1])
        mean = rewards.mean(dim=-1, keepdim=True)
        rewards = rewards - mean

        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
            # GRPO/GSPO 可选的组内标准差归一（+1e-6 防除零）
            std = rewards.std(dim=-1, keepdim=True)
            rewards = rewards / (std + 1e-6)

        return raw_rewards, rewards.flatten().tolist()

    # 无归一配置：处理后奖励等于原始奖励
    return raw_rewards, raw_rewards


def patched_save_db_snapshot(self, rollout_id):
    """Save a snapshot of the program database if enabled."""
    # Check if DB snapshot saving is enabled
    # 由环境变量 SAVE_DB_SNAPSHOTS=1 控制的程序库快照：
    # 每轮 rollout 落盘一份 .db 快照，训练后可离线审计经验库的演化
    if os.environ.get("SAVE_DB_SNAPSHOTS", "0") != "1":
        return

    try:
        import slime.ray.rollout as rollout_mod

        custom_generate_module_path = self.args.custom_generate_function_path
        if custom_generate_module_path:
            # 快照函数与自定义生成函数同模块：按模块路径动态加载
            module_path, _ = custom_generate_module_path.rsplit(".", 1)
            save_db_snapshot_func = rollout_mod.load_function(f"{module_path}.save_db_snapshot")

            run_dir = os.environ.get("RUN_DIR", ".")
            # 快照文件按 rollout 迭代号命名，保留时间序列
            snapshot_path = os.path.join(run_dir, f"program_database_iter_{rollout_id}.db")

            result = save_db_snapshot_func(snapshot_path)
            if inspect.isawaitable(result):
                # 快照实现可能是协程：在无事件循环的上下文里同步驱动完成
                asyncio.run(result)
            print(f"[RolloutManager] Database snapshot saved to {snapshot_path}")
    except Exception as e:
        # 快照失败只告警：不应让一次落盘问题中断训练
        print(f"[RolloutManager] Warning: Failed to save database snapshot: {e}")


def patched_log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    # 【补丁核心】替换 slime 的 rollout 日志函数：在原有指标之上叠加
    # 四算子分桶统计、奖励视图、质量代理与代码类别分布。
    if args.load_debug_rollout_data:
        # 回放调试数据模式：不记录
        return
    log_dict = {**(rollout_extra_metrics or {})}
    # 有效响应长度：有 loss_mask 用掩码和（只计参与损失的 token）
    response_lengths = [
        sum(sample.loss_mask) if sample.loss_mask is not None else sample.response_length for sample in samples
    ]
    log_dict["perf/rollout_time"] = rollout_time
    if args.rollout_num_gpus:
        # 吞吐：每 GPU 每秒生成的 token 数
        log_dict["perf/tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
    log_dict["perf/longest_sample_tokens_per_sec"] = max(response_lengths) / rollout_time

    import slime.ray.rollout as rollout_mod

    # 保留 slime 原生指标：响应长度统计、零方差检测、规格指标、奖励分类
    log_dict |= rollout_mod.dict_add_prefix(rollout_mod.compute_statistics(response_lengths), "rollout/response_len/")
    log_dict |= rollout_mod._compute_zero_std_metrics(args, samples)
    log_dict |= rollout_mod._compute_spec_metrics(args, samples)
    log_dict |= rollout_mod.dict_add_prefix(rollout_mod._compute_reward_cat_metrics(args, samples), "rollout/")

    # ---- 算子分桶：四个原子算子各一个桶 ----
    mode_buckets = {
        "draft": _build_mode_bucket(),
        "debug": _build_mode_bucket(),
        "improve": _build_mode_bucket(),
        "crossover": _build_mode_bucket(),
    }
    overall_reward_views = {key: [] for key in REWARD_VIEW_KEYS}
    for sample in samples:
        if hasattr(sample, "metadata") and isinstance(sample.metadata, dict):
            # 生成模式由 rollout 侧写入 metadata（数据管线中标记）
            generation_mode = sample.metadata.get("generation_mode", "draft")
            if generation_mode == "improve":
                if sample.metadata.get("parent_reward", None) == 0.0:
                    # Improve 的父代奖励为 0 => 实际语义是"修复零分程序"，归入 Debug 桶
                    generation_mode = "debug"
            if generation_mode not in mode_buckets:
                generation_mode = "draft"
            mode_bucket = mode_buckets[generation_mode]
            mode_bucket["samples"].append(sample)
            mode_bucket["total_count"] += 1
            hack_value = sample.metadata.get("hack", 0)
            code_category = sample.metadata.get("code_category", "valid")
            score = sample.metadata.get("score")
            status_codes = sample.metadata.get("status_codes", [])
            latest_status_code = status_codes[-1] if isinstance(status_codes, list) and status_codes else None
            success_flag = latest_status_code == 200 or hack_value == 1
            network_success_flag = latest_status_code == 200 or latest_status_code in (-1, -2, -3, -4)

            if hack_value == 1:
                # reward hacking 检测命中：策略在"钻空子"而非真改进
                mode_bucket["hack_count"] += 1
            if code_category == "empty":
                # 空程序：模型输出没有可执行代码
                mode_bucket["empty_count"] += 1
            if success_flag:
                mode_bucket["success_count"] += 1
            if network_success_flag:
                mode_bucket["network_success_count"] += 1
            if score is not None:
                mode_bucket["scored_count"] += 1

            for reward_key in REWARD_VIEW_KEYS:
                if reward_key == "metric_static_base_reward" and not _has_static_bounds(sample.metadata):
                    # 无静态边界的任务不产生该视图
                    continue
                reward_value = _to_finite_float(sample.metadata.get(reward_key))
                if reward_value is not None:
                    mode_bucket["reward_views"][reward_key].append(reward_value)
                    overall_reward_views[reward_key].append(reward_value)

    # 逐算子输出统计：hack/empty/success/scored 的比率与计数
    for mode_name, mode_bucket in mode_buckets.items():
        _log_reward_view_means(log_dict, f"rollout/{mode_name}", mode_bucket["reward_views"])
        _log_quality_proxy_metrics(log_dict, f"rollout/{mode_name}/quality", mode_bucket["samples"])
        total_count = mode_bucket["total_count"]
        if total_count > 0:
            log_dict[f"rollout/{mode_name}/hack_rate"] = mode_bucket["hack_count"] / total_count
            log_dict[f"rollout/{mode_name}/hack_count"] = mode_bucket["hack_count"]
            log_dict[f"rollout/{mode_name}/empty_rate"] = mode_bucket["empty_count"] / total_count
            log_dict[f"rollout/{mode_name}/empty_count"] = mode_bucket["empty_count"]
            log_dict[f"rollout/{mode_name}/success_rate"] = mode_bucket["success_count"] / total_count
            log_dict[f"rollout/{mode_name}/success_count"] = mode_bucket["success_count"]
            log_dict[f"rollout/{mode_name}/network_success_rate"] = mode_bucket["network_success_count"] / total_count
            log_dict[f"rollout/{mode_name}/network_success_count"] = mode_bucket["network_success_count"]
            log_dict[f"rollout/{mode_name}/scored_rate"] = mode_bucket["scored_count"] / total_count
            log_dict[f"rollout/{mode_name}/scored_count"] = mode_bucket["scored_count"]
            log_dict[f"rollout/{mode_name}/total_count"] = total_count

    # ---- 代码类别分布（数据管线产出的质量标签）----
    total_samples = len(samples)
    code_category_counts = {
        "hack": 0,  # 已证实作弊
        "hack_verify": 0,  # 疑似作弊待复核
        "no_verify": 0,  # 未能验证
        "empty": 0,  # 空程序
        "valid": 0,  # 正常有效
    }
    for sample in samples:
        if hasattr(sample, "metadata") and isinstance(sample.metadata, dict):
            code_category = sample.metadata.get("code_category", "valid")
            if code_category in code_category_counts:
                code_category_counts[code_category] += 1
    if total_samples > 0:
        for code_category, count in code_category_counts.items():
            log_dict[f"rollout/code_category/{code_category}/count"] = count
            log_dict[f"rollout/code_category/{code_category}/ratio"] = count / total_samples

    # ---- 算子使用比例：策略在四个原子算子上的行为分布 ----
    total_draft_count = mode_buckets["draft"]["total_count"]
    total_debug_count = mode_buckets["debug"]["total_count"]
    total_improve_count = mode_buckets["improve"]["total_count"]
    total_crossover_count = mode_buckets["crossover"]["total_count"]
    total_samples_with_mode = total_draft_count + total_debug_count + total_improve_count + total_crossover_count
    if total_samples_with_mode > 0:
        log_dict["rollout/draft/ratio"] = total_draft_count / total_samples_with_mode
        log_dict["rollout/debug/ratio"] = total_debug_count / total_samples_with_mode
        log_dict["rollout/improve/ratio"] = total_improve_count / total_samples_with_mode
        log_dict["rollout/crossover/ratio"] = total_crossover_count / total_samples_with_mode
        # 汇总（跨算子）的总体质量指标
        total_hack_count = sum(bucket["hack_count"] for bucket in mode_buckets.values())
        total_empty_count = sum(bucket["empty_count"] for bucket in mode_buckets.values())
        total_success_count = sum(bucket["success_count"] for bucket in mode_buckets.values())
        total_network_success_count = (
            sum(bucket["network_success_count"] for bucket in mode_buckets.values())
        )
        total_scored_count = sum(bucket["scored_count"] for bucket in mode_buckets.values())
        log_dict["rollout/hack_rate"] = total_hack_count / total_samples_with_mode
        log_dict["rollout/hack_count"] = total_hack_count
        log_dict["rollout/empty_rate"] = total_empty_count / total_samples_with_mode
        log_dict["rollout/empty_count"] = total_empty_count
        log_dict["rollout/success_rate"] = total_success_count / total_samples_with_mode
        log_dict["rollout/success_count"] = total_success_count
        log_dict["rollout/network_success_rate"] = total_network_success_count / total_samples_with_mode
        log_dict["rollout/network_success_count"] = total_network_success_count
        log_dict["rollout/scored_rate"] = total_scored_count / total_samples_with_mode
        log_dict["rollout/scored_count"] = total_scored_count

    # 全局奖励视图与质量代理
    _log_reward_view_means(log_dict, "rollout", overall_reward_views)
    _log_quality_proxy_metrics(log_dict, "rollout/quality", samples)

    # 独立复算奖励链路（调试用）：原始奖励 + 组内归一后的奖励
    raw_rewards, processed_rewards = _compute_debug_rewards(args, samples)
    raw_reward_mean = _mean_or_none([value for value in raw_rewards if _to_finite_float(value) is not None])
    processed_reward_mean = _mean_or_none([value for value in processed_rewards if _to_finite_float(value) is not None])
    if raw_reward_mean is not None:
        log_dict["rollout/raw_reward_debug"] = raw_reward_mean
    if processed_reward_mean is not None:
        log_dict["rollout/processed_reward"] = processed_reward_mean
    # 原始奖励列表只进 stdout（wandb/tb 不收列表型指标）
    printable_log_dict = {
        **log_dict,
        "rollout/raw_rewards": raw_rewards,
        "rollout/processed_rewards": processed_rewards,
    }
    print(f"perf {rollout_id}: {printable_log_dict}")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    # Keep list metrics out of wandb/tensorboard (can be huge); still printed above.
    logging_utils.log(args, log_dict, step_key="rollout/step")

    # Persist program DB snapshot once per rollout when enabled.
    # (Previously RolloutManager.save_db_snapshot was patched but never called.)
    # 旧补丁挂在 RolloutManager 方法上但从未被调用；这里在日志函数内主动触发
    try:
        patched_save_db_snapshot(type("_RolloutArgsProxy", (), {"args": args})(), rollout_id)
    except Exception as e:
        print(f"[PATCH_ROLLOUT] Warning: Failed to save database snapshot: {e}")

    # Return True so --custom-rollout-log-function-path skips stock logging.
    # 返回 True 表示定制日志已完整接管，slime 跳过自带日志
    return True


def apply_patch():
    """Apply monkey patch to rollout internals."""
    # 把两个补丁函数替换进 slime.ray.rollout 模块（import 本模块时自动执行）
    try:
        import slime.ray.rollout as rollout_mod

        rollout_mod.RolloutManager.save_db_snapshot = patched_save_db_snapshot
        rollout_mod._log_rollout_data = patched_log_rollout_data
        print("✅ [PATCH_ROLLOUT] Successfully patched RolloutManager.save_db_snapshot and _log_rollout_data")
        return True
    except Exception as e:
        print(f"⚠️ [PATCH_ROLLOUT] Failed to patch: {e}")
        import traceback

        traceback.print_exc()
        return False


# 非 __main__ 导入即生效：保证 train_with_patch.py 一 import 就完成替换
if __name__ != "__main__":
    apply_patch()
