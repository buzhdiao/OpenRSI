# 导入 Namespace：仅为 args 参数的类型注解
from argparse import Namespace
# 导入 deepcopy：快照计时字典，避免后续 reset 影响已取数据
from copy import deepcopy
# 导入 Callable：FLOPs 计算回调的类型注解
from typing import Callable

# 导入 psutil：读取 CPU/系统内存统计
import psutil
# 导入 torch：查询 CUDA 可用性与 GPU 显存
import torch

# 导入 wandb 工具：上报指标到 Weights & Biases
from slime.utils import wandb_utils
# 导入全局计时器单例：读取各阶段累计耗时
from slime.utils.timer import Timer


def get_memory_stats():
    """Get current GPU and CPU memory usage statistics."""
    # 汇总的内存指标字典
    memory_stats = {}

    # GPU memory stats
    # GPU 显存统计（CUDA 可用时才采集）
    if torch.cuda.is_available():
        # 取当前使用的 GPU 设备
        device = torch.cuda.current_device()
        # 查询驱动层面的空闲与总显存
        free, total = torch.cuda.mem_get_info(device)
        # 各项显存指标统一换算为 GB（保留两位小数）
        memory_stats.update(
            {
                # PyTorch 张量实际占用的显存
                "gpu_allocated_gb": round(torch.cuda.memory_allocated(device) / (1024**3), 2),
                # 缓存分配器向驱动申请的显存
                "gpu_reserved_gb": round(torch.cuda.memory_reserved(device) / (1024**3), 2),
                # 驱动视角空闲显存
                "gpu_free_gb": round(free / (1024**3), 2),
                # 总显存
                "gpu_total_gb": round(total / (1024**3), 2),
                # 已用显存 = 总量 - 空闲
                "gpu_used_gb": round((total - free) / (1024**3), 2),
                # 显存使用率（百分比）
                "gpu_utilization": round((total - free) / total * 100, 2),
            }
        )

    # CPU memory stats
    # CPU 内存统计（psutil 读取虚拟内存）
    mem = psutil.virtual_memory()
    memory_stats.update(
        {
            # 已用内存
            "cpu_used_gb": round(mem.used / (1024**3), 2),
            # 可用内存
            "cpu_available_gb": round(mem.available / (1024**3), 2),
            # 总内存
            "cpu_total_gb": round(mem.total / (1024**3), 2),
            # 内存使用率（百分比）
            "cpu_percent": round(mem.percent, 2),
        }
    )

    # 返回汇总的内存指标
    return memory_stats


def log_perf_data_raw(
    rollout_id: int, args: Namespace, is_primary_rank: bool, compute_total_fwd_flops: Callable
) -> None:
    # 获取全局计时器单例
    timer_instance = Timer()
    # 深拷贝当前所有累计耗时（作为本步的快照）
    log_dict_raw = deepcopy(timer_instance.log_dict())
    # 清空计时器，开始累计下一步的数据
    timer_instance.reset()

    # 非主 rank 不做指标上报（避免重复）
    if not is_primary_rank:
        return

    # 把各阶段耗时统一加 perf/ 前缀并重命名为 *_time 指标
    log_dict = {f"perf/{key}_time": val for key, val in log_dict_raw.items()}

    # Add memory statistics if enabled
    # 附加内存统计指标
    memory_stats = get_memory_stats()
    for key, val in memory_stats.items():
        # 统一加 memory/ 前缀
        log_dict[f"memory/{key}"] = val

    # 存在训练耗时快照且提供了 FLOPs 计算回调时，推导吞吐指标
    if ("perf/actor_train_time" in log_dict) and (compute_total_fwd_flops is not None):
        # 计算本轮训练的总前向 FLOPs
        total_fwd_flops = compute_total_fwd_flops(seq_lens=timer_instance.seq_lens)

        if "perf/log_probs_time" in log_dict:
            # log probs 重算阶段的等效 TFLOPs = 总 FLOPs / 耗时
            log_dict["perf/log_probs_tflops"] = total_fwd_flops / log_dict["perf/log_probs_time"]

        if "perf/ref_log_probs_time" in log_dict:
            # 参考模型 log probs 阶段的等效 TFLOPs
            log_dict["perf/ref_log_probs_tflops"] = total_fwd_flops / log_dict["perf/ref_log_probs_time"]

        if log_dict["perf/actor_train_time"] > 0:
            # 训练含前向+反向+优化器，约等于 3 倍前向 FLOPs，得到训练 TFLOPs
            log_dict["perf/actor_train_tflops"] = 3 * total_fwd_flops / log_dict["perf/actor_train_time"]
            # 训练吞吐 = 本轮总 token 数 / 训练耗时
            log_dict["perf/actor_train_tok_per_s"] = sum(timer_instance.seq_lens) / log_dict["perf/actor_train_time"]

    # 同时存在训练等待时间与训练时间时，推导整步耗时与等待占比
    if "perf/train_wait_time" in log_dict and "perf/train_time" in log_dict:
        # 整步时间 = 等待（rollout 数据生成）+ 训练
        total_time = log_dict["perf/train_wait_time"] + log_dict["perf/train_time"]
        if total_time > 0:
            # 整步耗时指标
            log_dict["perf/step_time"] = total_time
            # 等待时间占比（衡量训练/生成流水线的失衡程度）
            log_dict["perf/wait_time_ratio"] = log_dict["perf/train_wait_time"] / total_time

    # 直接打印本步性能数据（便于日志检索）
    print(f"perf {rollout_id}: {log_dict}")

    # 计算 wandb/TensorBoard 使用的步数：可选折算为等效训练步
    step = (
        rollout_id
        if not args.wandb_always_use_train_step
        else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    )
    if args.use_wandb:
        # 附带步数指标后上报 wandb
        log_dict["rollout/step"] = step
        wandb_utils.log_metrics(log_dict)

    if args.use_tensorboard:
        # 延迟导入 TensorBoard 适配器
        from slime.utils.tensorboard_utils import _TensorboardAdapter

        # 取单例适配器并按步数写入全部指标
        tb = _TensorboardAdapter(args)
        tb.log(data=log_dict, step=step)
