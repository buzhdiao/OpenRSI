# 启用延迟类型注解求值，兼容较旧 Python 版本中的新式注解写法
from __future__ import annotations

# 导入 json 用于读写 checkpoint 元数据文件
import json
# 导入 logging 用于输出日志
import logging
# 导入 time 用于给 checkpoint 打时间戳
import time
# 导入 Path 提供面向对象的文件路径操作
from pathlib import Path
# 导入 Any 用于类型注解
from typing import Any

# 导入 torch 主库
import torch
# 导入 torch 分布式模块（barrier、rank 查询等）
import torch.distributed as dist
# 导入 PyTorch 分布式检查点（Distributed Checkpoint，DCP）模块，支持分片保存/加载
import torch.distributed.checkpoint as dcp
# 导入 DCP 的 state_dict 工具：获取/恢复模型与优化器（含 FSDP 分片映射）的状态字典
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
# 导入 Stateful 接口：DCP 通过它调用自定义对象的 state_dict/load_state_dict
from torch.distributed.checkpoint.stateful import Stateful

# 获取本模块专属的 logger
logger = logging.getLogger(__name__)


# 模型状态的 Stateful 封装：只包模型权重，便于与优化器状态分开保存/加载
class ModelState(Stateful):
    """Wrapper for model state only."""

    # 构造时持有模型引用
    def __init__(self, model):
        self.model = model

    # 导出模型状态字典：get_state_dict 会处理 FSDP 分片/汇聚（DTensor 转换）逻辑
    def state_dict(self):
        model_state_dict, _ = get_state_dict(self.model, optimizers=[])
        # 以 "model" 为键返回，供 DCP 保存
        return {"model": model_state_dict}

    # 加载模型状态：set_state_dict 按相同映射规则写回模型参数
    def load_state_dict(self, state_dict):
        set_state_dict(self.model, optimizers=[], model_state_dict=state_dict["model"], optim_state_dict=None)


# 优化器状态的 Stateful 封装：只包优化器（动量等）状态
class OptimizerState(Stateful):
    """Wrapper for optimizer state only."""

    # 构造时持有模型与优化器引用（get_state_dict 需要两者配对）
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    # 导出优化器状态字典（get_state_dict 第二个返回值为优化器状态）
    def state_dict(self):
        _, optimizer_state_dict = get_state_dict(self.model, optimizers=self.optimizer)
        # 以 "optim" 为键返回
        return {"optim": optimizer_state_dict}

    # 加载优化器状态：model_state_dict 传 None 表示不动模型参数
    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model, optimizers=self.optimizer, model_state_dict=None, optim_state_dict=state_dict["optim"]
        )


# 学习率调度器状态的 Stateful 封装
class LRSchedulerState(Stateful):
    """Wrapper for LR scheduler state only."""

    # 构造时持有调度器引用
    def __init__(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    # 直接复用 PyTorch 调度器自带的 state_dict
    def state_dict(self):
        return {"lr_scheduler": self.lr_scheduler.state_dict()}

    # 同样复用调度器自带的 load_state_dict 完成恢复
    def load_state_dict(self, state_dict):
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])


# 读取 checkpoint 元数据（meta.json）；文件不存在或解析失败时返回空字典
def _read_checkpoint_metadata(path: Path) -> dict[str, Any]:
    # 路径不存在则直接返回空字典
    if not path.exists():
        return {}
    # 尝试把 JSON 文本解析为字典
    try:
        return json.loads(path.read_text())
    # JSON 损坏时记录警告并返回空字典，避免加载流程中断
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse checkpoint metadata at {path}")
        return {}


# 原子化写入 checkpoint 元数据：先写临时文件再替换，防止写一半被读到坏数据
def _write_checkpoint_metadata(path: Path, metadata: dict[str, Any]) -> None:
    # 计算临时文件路径（在原后缀后追加 .tmp）
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    # 写入格式化并按键排序的 JSON 文本
    tmp_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    # 原子替换为正式文件（同文件系统上 replace 是原子操作）
    tmp_path.replace(path)


# 从磁盘加载 checkpoint；返回包含 rng/元数据/迭代号的字典，或 None 表示未加载
def load(actor: Any) -> dict[str, Any] | None:
    """Load checkpoint from disk.

    Loads model weights and optionally optimizer state from separate directories.
    This allows loading weights without optimizer or deleting optimizer before loading.
    """
    # 从训练参数中取加载根目录；未配置则返回 None
    load_root = getattr(actor.args, "load", None)
    if load_root is None:
        return None

    # 展开用户目录符号（~）并构造 Path
    root_path = Path(load_root).expanduser()
    # 目录不存在时记录日志并跳过加载
    if not root_path.exists():
        logger.info(f"[FSDP] Checkpoint directory {root_path} not found; skipping load.")
        return None

    # 读取目标恢复步数；未显式指定时从 tracker 文件中读取
    target_step = getattr(actor.args, "ckpt_step", None)
    if target_step is None:
        # tracker 文件记录最近一次保存的步数
        tracker_file = root_path / "latest_checkpointed_iteration.txt"
        if not tracker_file.exists():
            logger.info(f"[FSDP] No tracker file at {tracker_file}; skipping load.")
            return None
        # 读取文本并转成整数作为目标步数
        tracker_text = tracker_file.read_text().strip()
        target_step = int(tracker_text)

    # 按 iter_XXXXXXXX 命名规则定位具体 checkpoint 目录及其三个子目录
    checkpoint_dir = root_path / f"iter_{target_step:07d}"
    model_dir = checkpoint_dir / "model"
    optimizer_dir = checkpoint_dir / "optimizer"
    lr_scheduler_dir = checkpoint_dir / "lr_scheduler"

    # 模型权重目录缺失时无法继续，跳过加载
    if not model_dir.exists():
        logger.info(f"[FSDP] Model checkpoint {model_dir} not found; skipping load.")
        return None

    # Load model weights (always)
    # 加载模型权重（必做）：构造 ModelState 封装并交给 DCP
    model_state = ModelState(actor.model)
    state_dict = {"model_state": model_state}

    # 调用分布式检查点加载模型权重
    try:
        dcp.load(state_dict=state_dict, checkpoint_id=str(model_dir))
        logger.info(f"[FSDP] Loaded model from {model_dir}")
    # 加载失败时记录错误并整体放弃（不半途带着部分旧权重继续训练）
    except Exception as e:
        logger.error(f"[FSDP] Failed to load model from {model_dir}: {e}")
        return None

    # Load optimizer state (optional)
    # 优化器状态为可选加载：受 --no-load-optim 控制，且要求 actor 已创建优化器
    load_optimizer = not getattr(actor.args, "no_load_optim", False) and hasattr(actor, "optimizer")
    # 条件满足且对应目录存在时才加载
    if load_optimizer and optimizer_dir.exists():
        optimizer_state = OptimizerState(actor.model, actor.optimizer)
        optim_state_dict = {"optim_state": optimizer_state}
        # DCP 加载优化器状态
        try:
            dcp.load(state_dict=optim_state_dict, checkpoint_id=str(optimizer_dir))
            logger.info(f"[FSDP] Loaded optimizer from {optimizer_dir}")
        # 优化器加载失败只告警不中断（模型权重已就绪，可继续训练）
        except Exception as e:
            logger.warning(f"[FSDP] Failed to load optimizer from {optimizer_dir}: {e}")
    # 想加载但目录不存在：记录提示
    elif load_optimizer:
        logger.info(f"[FSDP] Optimizer checkpoint not found at {optimizer_dir}, skipping optimizer load.")

    # Load LR scheduler state (optional)
    # 调度器状态为可选加载：要求 actor 已有调度器且目录存在
    load_lr_scheduler = hasattr(actor, "lr_scheduler") and lr_scheduler_dir.exists()
    if load_lr_scheduler:
        lr_scheduler_state = LRSchedulerState(actor.lr_scheduler)
        lr_scheduler_state_dict = {"lr_scheduler_state": lr_scheduler_state}
        # DCP 加载调度器状态
        try:
            dcp.load(state_dict=lr_scheduler_state_dict, checkpoint_id=str(lr_scheduler_dir))
            logger.info(f"[FSDP] Loaded LR scheduler from {lr_scheduler_dir}")
        # 失败仅告警，训练可用默认调度状态继续
        except Exception as e:
            logger.warning(f"[FSDP] Failed to load LR scheduler from {lr_scheduler_dir}: {e}")
    elif hasattr(actor, "lr_scheduler"):
        logger.info(f"[FSDP] LR scheduler checkpoint not found at {lr_scheduler_dir}, skipping LR scheduler load.")

    # 初始化 RNG 状态为 None
    rng_state = None
    # 检查 RNG 状态文件是否存在
    rng_path = checkpoint_dir / "rng.pt"
    if rng_path.exists():
        # 从 CPU 上加载保存的随机数状态（用于可复现训练）
        rng_state = torch.load(rng_path, map_location="cpu")

    # 读取该 checkpoint 的元数据
    metadata = _read_checkpoint_metadata(checkpoint_dir / "meta.json")

    # 返回加载结果：RNG 状态、元数据与恢复的迭代号
    return {
        "rng": rng_state,
        "metadata": metadata,
        "iteration": target_step,
    }


# 加载后的收尾：恢复 RNG、回填全局步数/rollout 起始 ID，并做分布式同步
def finalize_load(actor: Any, checkpoint_payload: dict[str, Any] | None) -> None:
    # 无 checkpoint 可恢复时只做 barrier 同步保持所有 rank 一致
    if checkpoint_payload is None:
        dist.barrier()
        return

    # 若保存了 RNG 状态且未指定 --no-load-rng，则恢复随机数状态
    if checkpoint_payload.get("rng") is not None and not getattr(actor.args, "no_load_rng", False):
        # 取出 RNG 状态字典
        rng_state = checkpoint_payload["rng"]
        # 恢复 CPU 侧 torch 随机数状态
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        # CUDA 可用且保存了 cuda 状态时，恢复所有 GPU 设备的随机数状态
        if torch.cuda.is_available() and "cuda" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["cuda"])

    # 取出元数据与迭代号（可能为空）
    metadata = checkpoint_payload.get("metadata") or {}
    iteration = checkpoint_payload.get("iteration")
    # 有元数据时优先用其中的精确进度信息
    if metadata:
        # 恢复全局训练步数（不存在则保留当前值）
        actor.global_step = int(metadata.get("global_step", actor.global_step))
        # 恢复微批次步数
        actor.micro_step = int(metadata.get("micro_step", actor.micro_step))
        # 读取下一次 rollout 的起始 ID
        next_rollout = metadata.get("next_rollout_id")
        if next_rollout is not None:
            # 写回训练参数，让 RL 循环从正确的 rollout 继续
            actor.args.start_rollout_id = next_rollout
    # 无元数据（旧格式 checkpoint）时退化为用迭代号作为 rollout 起点
    elif iteration is not None:
        if getattr(actor.args, "start_rollout_id", None) is None:
            actor.args.start_rollout_id = iteration

    # CUDA 同步确保恢复操作全部落地
    torch.cuda.synchronize()
    # barrier 保证所有进程在恢复完成后再继续
    dist.barrier()


# 保存 checkpoint：模型、优化器、调度器分目录保存，并写 RNG/元数据/tracker
def save(actor: Any, iteration: int) -> None:
    """Save checkpoint to disk.

    Saves model weights and optimizer state to separate directories.
    This allows loading weights without optimizer or deleting optimizer before loading.
    """
    # 先 CUDA 同步，确保所有未完成的计算/拷贝都完成再落盘
    torch.cuda.synchronize()

    # 展开并构造保存根目录
    base_dir = Path(actor.args.save).expanduser()
    # 步数编号采用 iteration+1（与 Megatron 的 1-based 习惯对齐）
    step_id = iteration + 1
    # 构造本次 checkpoint 及三个子目录的路径
    checkpoint_dir = base_dir / f"iter_{step_id:07d}"
    model_dir = checkpoint_dir / "model"
    optimizer_dir = checkpoint_dir / "optimizer"
    lr_scheduler_dir = checkpoint_dir / "lr_scheduler"

    # 仅 rank 0 创建目录，避免多进程竞争写同一目录
    if dist.get_rank() == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        optimizer_dir.mkdir(parents=True, exist_ok=True)
        lr_scheduler_dir.mkdir(parents=True, exist_ok=True)
    # barrier 等待目录创建完成，其他 rank 再继续写入
    dist.barrier()

    # Save model weights
    # 保存模型权重：封装 ModelState 后交给 DCP 分片保存
    model_state = ModelState(actor.model)
    state_dict = {"model_state": model_state}
    dcp.save(state_dict, checkpoint_id=str(model_dir))

    # Save optimizer state (skip if --no-save-optim is set)
    # 决定是否保存优化器状态（--no-save-optim 可跳过以省磁盘/时间）
    save_optimizer_state = not getattr(actor.args, "no_save_optim", False)
    # 满足条件且优化器存在时执行保存
    if save_optimizer_state and hasattr(actor, "optimizer") and actor.optimizer is not None:
        optimizer_state = OptimizerState(actor.model, actor.optimizer)
        optim_state_dict = {"optim_state": optimizer_state}
        dcp.save(optim_state_dict, checkpoint_id=str(optimizer_dir))

    # Save LR scheduler state (skip if --no-save-optim is set)
    # 调度器状态跟随优化器开关一起保存（二者都属于可再生的训练进度状态）
    if save_optimizer_state and hasattr(actor, "lr_scheduler") and actor.lr_scheduler is not None:
        lr_scheduler_state = LRSchedulerState(actor.lr_scheduler)
        lr_scheduler_state_dict = {"lr_scheduler_state": lr_scheduler_state}
        dcp.save(lr_scheduler_state_dict, checkpoint_id=str(lr_scheduler_dir))

    # 以下写单文件的动作仅由 rank 0 执行
    if dist.get_rank() == 0:
        # 收集 CPU 与所有 CUDA 设备的随机数状态并保存
        rng_state = {"torch": torch.get_rng_state()}
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
        torch.save(rng_state, checkpoint_dir / "rng.pt")

        # 组装元数据：进度、并行规模与时间戳
        metadata = {
            "iteration": step_id,
            "rollout_id": iteration,
            "next_rollout_id": iteration + 1,
            "global_step": actor.global_step,
            "micro_step": actor.micro_step,
            "world_size": dist.get_world_size(),
            "timestamp": time.time(),
        }
        # 原子化写入 meta.json
        _write_checkpoint_metadata(checkpoint_dir / "meta.json", metadata)

        # 更新 tracker 文件指向最新步数，供下次默认加载使用
        tracker_file = base_dir / "latest_checkpointed_iteration.txt"
        tracker_file.write_text(str(step_id))
        logger.info(f"[FSDP] Saved checkpoint to {checkpoint_dir}")

    # 最终 barrier：确保所有 rank 的保存动作都完成后再继续训练
    dist.barrier()
