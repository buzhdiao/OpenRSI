import logging
from argparse import Namespace
from collections.abc import Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams

from slime.utils import train_metric_utils
from slime.utils.data import get_minimum_num_micro_batch_size
from slime.utils.flops_utils import calculate_fwd_flops
from slime.utils.metric_utils import compute_pass_rate, compute_rollout_step
from slime.utils.seqlen_balancing import get_seqlen_balanced_partitions
from slime.utils.types import RolloutBatch

from ...utils import logging_utils
from .cp_utils import get_sum_of_sample_mean, slice_with_cp

logger = logging.getLogger(__name__)


def get_batch(
    data_iterator: "DataIterator",
    keys: Sequence[str],
    pad_multiplier: int = 128,
    qkv_format: str = "thd",
    allgather_cp: bool = False,
) -> dict[str, torch.Tensor | PackedSeqParams | list[torch.Tensor] | None]:
    """
    Generate a CP-ready micro-batch with packed sequence parameters.

    Steps:
    - Fetch raw fields via iterator.
    - Save original token tensors under "unconcat_tokens".
    - Slice tokens into two chunks for Context Parallelism (CP), concatenate, and pad to a configurable multiple.
    - Build cu_seqlens and `PackedSeqParams` with T-H-D layout (T: sequence length, H: attention heads, D: head dimension).

    Args:
        data_iterator: Iterator providing micro-batch data.
        keys: List of keys to fetch from the iterator.
        pad_multiplier: Multiplier for padding size calculation (default: 128).

    Returns a dict including:
    - "tokens": torch.LongTensor of shape [1, T_padded] on the current CUDA device
    - "unconcat_tokens": list[torch.LongTensor] for the micro-batch before CP slicing/concat
    - "packed_seq_params": PackedSeqParams with T-H-D settings (cu_seqlens on CUDA, dtype=int)
    Plus any other requested keys forwarded from the iterator.
    """

    # tokens 是必备字段
    assert "tokens" in keys
    # 从迭代器取出当前 micro-batch 的原始数据
    batch = data_iterator.get_next(keys)

    # 透传动态全局 batch size（若存在）
    if "dynamic_global_batch_size" in data_iterator.rollout_data:
        batch["dynamic_global_batch_size"] = data_iterator.rollout_data["dynamic_global_batch_size"]

    # 取出 token 序列列表（每条样本一个 1D 张量）
    tokens = batch["tokens"]
    # use 0 as the pad token id should be fine?
    # 填充用的 pad token id 固定取 0
    pad_token_id = 0
    # pad 的粒度：TP 世界大小乘以倍数，保证各 TP rank 序列长度对齐
    pad_size = mpu.get_tensor_model_parallel_world_size() * pad_multiplier

    # for cp, we need all tokens to calculate logprob
    # 备份 CP 切分前的原始 token 序列，logprob 计算需要完整序列
    batch["unconcat_tokens"] = tokens

    # CP（上下文并行）组大小与当前 rank 编号
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    if qkv_format == "bshd":
        # bshd 格式：按 batch 维堆叠，先取本批最大序列长度
        max_seqlen = batch["max_seq_lens"][0]
        # 断言所有样本都不超过 max_seqlen
        assert max([t.size(0) for t in tokens]) <= max_seqlen
        # 每条样本独立做 CP zigzag 切分（补齐到 max_seqlen）
        tokens = [slice_with_cp(t, pad_token_id, qkv_format, max_seqlen) for t in tokens]
        # 沿 batch 维堆叠成 [B, T]
        tokens = torch.stack(tokens)

    elif qkv_format == "thd":
        if allgather_cp:
            # DSA mode: concatenate all sequences first, then slice once with CP.
            # We also pad the *global* concatenated stream to make per-rank chunks equal.
            # DSA 模式：先把整个 batch 的序列拼接成一条长流，再做一次 CP 切分。
            # cu_seqlens 记录每条序列在拼接流中的起止位置（前缀和，首元素为 0）
            cu_seqlens_list: list[int] = [0]
            for t in tokens:
                cu_seqlens_list.append(cu_seqlens_list[-1] + t.size(0))

            # 拼接所有序列为一维长流
            tokens = torch.cat(tokens, dim=0)

            # Pad global stream so (1) divisible by cp_size (equal chunks),
            # (2) divisible by pad_size (reduce fragmentation).
            # 全局填充量取 cp_size 与 pad_size 的乘积：
            # (1) 保证能均分成 cp_size 份；(2) 减少显存碎片
            global_pad_size = cp_size * pad_size
            pad = (global_pad_size - tokens.size(0) % global_pad_size) % global_pad_size
            if pad != 0:
                # 尾部补 pad token，并把填充区域记为一条新的"序列"加入 cu_seqlens
                tokens = F.pad(tokens, (0, pad), value=pad_token_id)
                cu_seqlens_list.append(cu_seqlens_list[-1] + pad)

            # cu_seqlens 搬到当前 CUDA 设备
            cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int, device=torch.cuda.current_device())
            # 取本 rank 对应的 1/cp_size 分片
            tokens = tokens.chunk(cp_size, dim=0)[cp_rank]
        else:
            # 常规 thd 模式：逐条样本做 CP zigzag 切分
            tokens = [slice_with_cp(t, pad_token_id, qkv_format) for t in tokens]

            # 重新按切分后的序列计算 cu_seqlens 前缀和
            cu_seqlens = [0]
            for t in tokens:
                cu_seqlens.append(cu_seqlens[-1] + t.size(0))

            # 拼接本 rank 的所有序列片段
            tokens = torch.cat(tokens)

            # Always pad to reduce memory fragmentation and maybe make the computation faster
            # 尾部补齐到 pad_size 的整倍数，减少显存碎片并可能加速计算
            pad = (pad_size - tokens.size(0) % pad_size) % pad_size
            if pad != 0:
                tokens = F.pad(tokens, (0, pad), value=pad_token_id)
                cu_seqlens.append(cu_seqlens[-1] + pad)

            # thd requires the cu_seqlens to be of the origin length
            # thd 要求 cu_seqlens 用原始（未切分）长度：乘以 cp_size 把本 rank 的局部坐标还原为全局坐标
            cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int).cuda() * cp_size

        # 最大序列长度 = cu_seqlens 相邻差分的最大值
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        # 构造 packed sequence 参数（varlen attention 所需的 THD 布局元信息）
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format="thd",
        )

        # 增加一个 batch 维，变成 [1, T] 的 THD 布局
        tokens = tokens.unsqueeze(0)
    else:
        # 不支持的 qkv 格式直接报错
        raise ValueError(f"Unsupported qkv_format: {qkv_format}")

    # 把处理后的 tokens 与 packed 参数写回 batch
    batch["tokens"] = tokens
    batch["packed_seq_params"] = packed_seq_params

    # loss masks
    # 处理 loss mask，使其与 CP 切分后的 token 流对齐
    loss_masks = []
    for loss_mask, total_length, response_length in zip(
        batch["loss_masks"],
        batch["total_lengths"],
        batch["response_lengths"],
        strict=True,
    ):
        # prompt 长度
        prompt_length = total_length - response_length
        # Align mask to token stream positions (prompt_length-1 left pad, 1 right pad)
        # 把 mask 对齐到 token 流位置：左边补 prompt_length-1 个 0（prompt 不算 loss），
        # 右边补 1 个 0（最后一个 token 没有"下一个 token"可供预测）
        loss_mask = F.pad(loss_mask, (prompt_length - 1, 1), value=0)
        if allgather_cp:
            # DSA 模式下保持整段 mask，稍后统一拼接切分
            loss_masks.append(loss_mask)
            continue
        # 常规模式：对 mask 同样做 CP zigzag 切分
        loss_mask = slice_with_cp(loss_mask, 0, qkv_format, max_seqlen)
        loss_masks.append(loss_mask)

    # 按 qkv 格式把各样本的 mask 合并成与 tokens 相同形状的张量
    if qkv_format == "bshd":
        # bshd：沿 batch 维堆叠
        loss_masks = torch.stack(loss_masks)
    elif qkv_format == "thd" and allgather_cp:
        # DSA: concatenate first (same as tokens), pad globally (same pad as above), then slice once.
        # DSA：与 tokens 同样先拼接、再按相同的全局 pad 补齐、最后切出本 rank 分片
        loss_masks = torch.cat(loss_masks, dim=0)
        if pad != 0:
            loss_masks = F.pad(loss_masks, (0, pad), value=0)
        loss_masks = loss_masks.chunk(cp_size, dim=0)[cp_rank].unsqueeze(0)
    elif qkv_format == "thd":
        # 常规 thd：拼接后按 pad 补齐并加 batch 维
        loss_masks = torch.cat(loss_masks)
        loss_masks = F.pad(loss_masks, (0, pad), value=0).unsqueeze(0)

    # 断言 mask 与 tokens 形状完全一致
    assert loss_masks.shape == tokens.shape, f"loss_masks.shape: {loss_masks.shape}, tokens.shape: {tokens.shape}"
    # 保存 CP 切分后的完整 loss mask
    batch["full_loss_masks"] = loss_masks

    # Process multimodal training tensors if present
    # 多模态训练张量：把每个样本各自的张量字典按 key 拼接合并
    multimodal_train_inputs = batch.get("multimodal_train_inputs", None)
    if multimodal_train_inputs is not None:
        multimodal_data = {}  # key -> concatenated tensor
        multimodal_num_items = {}  # key -> list of item counts per sequence
        # 遍历每个样本的多模态输入字典
        for mm_input_dict in multimodal_train_inputs:
            if mm_input_dict is not None:
                for key, mm_tensor in mm_input_dict.items():
                    if key not in multimodal_data:
                        # 首次出现该 key：直接记录张量与该样本的条目数
                        multimodal_data[key] = mm_tensor
                        multimodal_num_items[key] = [mm_tensor.size(0)]
                    else:
                        # 后续样本：沿第 0 维拼接张量，并累计条目数
                        multimodal_data[key] = torch.cat([multimodal_data[key], mm_tensor], dim=0)
                        multimodal_num_items[key].append(mm_tensor.size(0))
        # 拼接后的张量与每样本条目数写回 batch
        batch["multimodal_train_inputs"] = multimodal_data
        batch["multimodal_num_items"] = multimodal_num_items

    return batch


def gather_log_data(
    metric_name: str,
    args: Namespace,
    rollout_id: int,
    log_dict: dict[str, float],
) -> dict[str, float] | None:
    """
    Gather per-rank metrics, reduce by mean on the DP source rank, and log.

    Expects `log_dict` to contain plain scalars. The DP source rank prints and
    optionally logs to WandB/TensorBoard with a step derived from `rollout_id` and
    batch sizes. Returns the reduced dict on the DP source rank; returns None on others.
    """

    # 只让 DP（含 CP 维度）源 rank 负责收集与打印
    if mpu.get_data_parallel_rank(with_context_parallel=True) == 0:
        # DP 世界大小（含 CP）
        dp_size = mpu.get_data_parallel_world_size(with_context_parallel=True)

        # 预分配收集容器
        gathered_log_dict = [None] * dp_size
        # Not sure if this will be a performance bottleneck.
        # 用 gloo 组做对象 gather（CPU 序列化收集各 rank 的标量指标）
        dist.gather_object(
            log_dict,
            gathered_log_dict,
            dst=mpu.get_data_parallel_src_rank(with_context_parallel=True),
            group=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
        )

        # 对每个指标在 DP 各 rank 间取平均
        reduced_log_dict = {
            f"{metric_name}/{key}": sum([d[key] for d in gathered_log_dict]) / dp_size for key in log_dict
        }
        # 打印归约后的指标
        logger.info(f"{metric_name} {rollout_id}: {reduced_log_dict}")

        # Calculate step once to avoid duplication
        # 计算并附加全局训练步数，作为 wandb/tensorboard 的 step
        step = compute_rollout_step(args, rollout_id)
        reduced_log_dict["rollout/step"] = step
        # 写入日志后端
        logging_utils.log(args, reduced_log_dict, step_key="rollout/step")

        # 源 rank 返回归约后的指标
        return reduced_log_dict
    else:
        # 非源 rank 只发送自己的 log_dict，返回 None
        dist.gather_object(
            log_dict,
            None,
            dst=mpu.get_data_parallel_src_rank(with_context_parallel=True),
            group=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
        )
        return None


class DataIterator:
    """Micro-batch iterator over rollout dicts.

    Supports either fixed contiguous micro-batches or an explicit per-step
    index schedule (for dynamic batch sizing / sequence-length balancing).
    """

    def __init__(
        self,
        rollout_data: RolloutBatch,
        micro_batch_size: int | None = None,
        micro_batch_indices: list[list[int]] | None = None,
    ) -> None:
        """Initialize an iterator over `rollout_data`.

        Args:
            rollout_data: Dict of per-sample fields for the local step.
            micro_batch_size: Fixed contiguous slice size when not using dynamic scheduling.
            micro_batch_indices: Explicit indices per micro-batch when using dynamic balancing.
                Must be mutually exclusive with `micro_batch_size`.
        """
        # 保存 rollout 数据与两种调度方式之一
        self.rollout_data = rollout_data
        self.micro_batch_size = micro_batch_size
        self.micro_batch_indices = micro_batch_indices
        # 两种调度方式互斥
        assert micro_batch_size is None or micro_batch_indices is None
        # 当前读取偏移
        self.offset = 0

    def get_next(self, keys: Sequence[str]) -> dict[str, list[object] | None]:
        """Return the next micro-batch for the requested keys.

        - If `micro_batch_indices` is provided, selects rows according to the current
          index list for each requested key.
        - Otherwise, slices a contiguous window of size `micro_batch_size` starting
          at the current offset.

        Returns a dict mapping each key to a list subset (or None if absent).
        """
        batch = {}
        # 逐 key 取出当前 micro-batch 的数据
        for key in keys:
            vals = self.rollout_data.get(key, None)
            if vals is None:
                # 该 key 不存在时填 None
                batch[key] = None
            else:
                if self.micro_batch_indices is not None:
                    # 动态调度：按当前步的索引列表选样本（用于序列长度均衡）
                    indices = self.micro_batch_indices[self.offset]
                    batch[key] = [vals[i] for i in indices]
                else:
                    # 固定大小：取 [offset, offset+micro_batch_size) 的连续切片，越界则断言
                    assert self.offset + self.micro_batch_size <= len(
                        vals
                    ), f"offset: {self.offset}, micro_batch_size: {self.micro_batch_size}, len(vals): {len(vals)}"
                    batch[key] = vals[self.offset : self.offset + self.micro_batch_size]

        # 推进偏移：动态调度按步数 +1，固定大小按批大小推进
        if self.micro_batch_indices is not None:
            self.offset += 1
        else:
            self.offset += self.micro_batch_size
        return batch

    def reset(self) -> "DataIterator":
        """Reset internal offset to the start and return self."""
        # 偏移归零并返回自身，便于链式调用
        self.offset = 0
        return self


def get_data_iterator(
    args: Namespace,
    model: torch.nn.Module | Sequence[torch.nn.Module],
    rollout_data: RolloutBatch,
) -> tuple[list[DataIterator], list[int]]:
    """
    Create iterators and a micro-batch schedule for a rollout step.

    - If `use_dynamic_batch_size` is False, splits into fixed-size contiguous
      micro-batches of `micro_batch_size`.
    - If True, computes the number of micro-batches per local step based on
      `max_tokens_per_gpu` and per-sample lengths, all-reduces to a DP-wide
      maximum, optionally enforces divisibility for Virtual Pipeline Parallelism (VPP), and builds a balanced
      index schedule to equalize token counts across micro-batches.

    Returns `(data_iterators, num_microbatches)` where:
    - `data_iterators`: list of `DataIterator`, one per VPP stage (size 1 if VPP disabled)
    - `num_microbatches`: list[int], one per local step in the rollout (length = steps)
    """
    # DP 世界大小（不含 CP）与通信组
    dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
    dp_group = mpu.get_data_parallel_group()
    # 虚拟流水线并行（VPP）大小；未开启时为 None，按 1 处理
    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
    if vpp_size is None:
        vpp_size = 1
    if vpp_size > 1:
        # VPP 开启时需要读取模型配置里的 microbatch 分组大小
        from megatron.core.utils import get_model_config

        config = get_model_config(model[0])
        microbatch_group_size_per_vp_stage = config.microbatch_group_size_per_vp_stage
    # CP 世界大小
    cp_size = mpu.get_context_parallel_world_size()

    # 本 rank 的样本数
    num_local_samples = len(rollout_data["total_lengths"])
    # 全局 batch size：可能是动态计算的（LLM 推理产出），否则用配置值
    global_batch_size = rollout_data.get("dynamic_global_batch_size", args.global_batch_size)
    # 每个 DP rank 分到的全局 batch 大小
    num_local_gbs = global_batch_size // dp_size
    # 每次 rollout 对应的训练步数
    num_steps_per_rollout = num_local_samples // num_local_gbs

    # 动态 batch size 与配置值不同时打印日志
    if global_batch_size != args.global_batch_size:
        logger.info(
            f"Using dynamic global_batch_size={global_batch_size} (original={args.global_batch_size}), "
            f"num_local_samples={num_local_samples}, num_steps_per_rollout={num_steps_per_rollout}"
        )

    # 为每个 VPP 阶段创建一个数据迭代器（VPP 各阶段共享同一份数据调度）
    def _generate_data_iterator(rollout_data, micro_batch_size, micro_batch_indices=None):
        data_iterator = []
        for _ in range(vpp_size):
            data_iterator.append(DataIterator(rollout_data, micro_batch_size, micro_batch_indices))
        return data_iterator

    if not args.use_dynamic_batch_size:
        # 固定 micro-batch 大小：每步 micro-batch 数 = 本地 batch / micro_batch_size
        num_microbatches = [num_local_gbs // args.micro_batch_size for _ in range(num_steps_per_rollout)]
        data_iterator = _generate_data_iterator(rollout_data, args.micro_batch_size)
    else:
        # 动态 batch 模式：按 GPU 最大 token 数决定 micro-batch 数
        assert args.max_tokens_per_gpu is not None
        # calculate the number of mirobatches for each step
        # 每步的样本总长度列表
        samples = rollout_data["total_lengths"]
        assert len(samples) == num_local_samples
        num_microbatches = []
        # 逐步计算满足 max_tokens_per_gpu 约束的最少 micro-batch 数（乘 cp_size 换算全局 token 预算）
        for i in range(num_steps_per_rollout):
            start, end = i * num_local_gbs, (i + 1) * num_local_gbs
            num_microbatches.append(
                get_minimum_num_micro_batch_size(samples[start:end], args.max_tokens_per_gpu * cp_size)
            )

        # DP 各 rank 的 micro-batch 数必须一致：all_reduce 取最大值
        num_microbatches = torch.tensor(num_microbatches, dtype=torch.int, device=torch.cuda.current_device())
        dist.all_reduce(num_microbatches, op=dist.ReduceOp.MAX, group=dp_group)

        if vpp_size > 1:
            # vpp requies the number of microbatches to be divisible by vpp_size
            # VPP 要求 micro-batch 数能被每个 vp 阶段的分组大小整除：向下取整并至少保留 1
            num_microbatches = torch.clamp(
                num_microbatches // microbatch_group_size_per_vp_stage * microbatch_group_size_per_vp_stage,
                min=1,
            )

        num_microbatches = num_microbatches.tolist()

        # balance the each micro batch
        # 对每个 micro-batch 做序列长度均衡（longest-shortest 装箱），减少 padding 浪费
        samples = rollout_data["total_lengths"]
        # balance the number of mirobatches across steps
        micro_batch_indices = []
        for i, num_mbs in enumerate(num_microbatches):
            start, end = i * num_local_gbs, (i + 1) * num_local_gbs
            samples = rollout_data["total_lengths"][start:end]
            # 把本步样本划分成 num_mbs 个 token 数尽量均衡的分区（不要求等大小）
            partitions = get_seqlen_balanced_partitions(samples, num_mbs, equal_size=False)
            # 把局部索引加上本步起始偏移，转换成全局索引
            for j in range(num_mbs):
                for k in range(len(partitions[j])):
                    partitions[j][k] += start
            micro_batch_indices.extend(partitions)

        # 断言所有 micro-batch 的索引合起来恰好覆盖全部本地样本（无重复、无遗漏）
        assert len(set(sum(micro_batch_indices, []))) == num_local_samples

        # 用索引调度方式生成迭代器
        data_iterator = _generate_data_iterator(rollout_data, None, micro_batch_indices)

    return (
        data_iterator,
        num_microbatches,
    )


def log_rollout_data(
    rollout_id: int,
    args: Namespace,
    rollout_data: RolloutBatch,
) -> None:
    """
    Summarize rollout fields and log reduced metrics on PP last stage, TP rank 0.

    - Tensor-valued lists are concatenated and averaged. For token-level metrics
      like log-probs/returns/advantages/values, computes a CP-correct sample mean
      using `loss_masks` and total/response lengths.
    - Non-tensor lists are averaged elementwise.
    - Scalars are converted to Python numbers.
    """
    # 只在 PP 最后阶段、TP rank 0 上统计与上报
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        cp_size = mpu.get_context_parallel_world_size()
        log_dict = {}
        # 取出样本长度与 loss mask 等元信息
        response_lengths = rollout_data["response_lengths"]
        loss_masks = rollout_data["loss_masks"]
        total_lengths = rollout_data["total_lengths"]
        max_seq_lens = rollout_data.get("max_seq_lens", None)

        # 遍历 rollout 数据的每个字段，计算指标
        for key, val in rollout_data.items():
            if key in [
                "tokens",
                "multimodal_train_inputs",
                "loss_masks",
                "sample_indices",
                "rollout_routed_experts",
                "max_seq_lens",
                "dynamic_global_batch_size",
            ]:
                # 这些字段不参与指标统计
                continue
            # Upload per sample mean for each rollout value
            # There are the following assumptions:
            # - Each dp rank has the same number of samples
            if isinstance(val, (list, tuple)):
                if isinstance(val[0], torch.Tensor):
                    # NOTE: Here we have to do the clone().detach(), otherwise the tensor will be
                    # modified in place and will cause problem for the next rollout.
                    # token 级指标（log_prob/回报/优势等）需按样本均值统计
                    if key in [
                        "log_probs",
                        "ref_log_probs",
                        "rollout_log_probs",
                        "returns",
                        "advantages",
                        "values",
                        "teacher_log_probs",
                        "opd_reverse_kl",
                    ]:
                        # 拼接所有样本的张量（clone+detach 防止原地修改影响下一轮 rollout）
                        val = torch.cat(val).clone().detach()
                        # 构造 CP 感知的按样本均值聚合器
                        sum_of_sample_mean = get_sum_of_sample_mean(
                            total_lengths,
                            response_lengths,
                            loss_masks,
                            qkv_format=args.qkv_format,
                            max_seq_lens=max_seq_lens,
                        )
                        # 求所有样本均值之和：乘 cp_size 补偿 CP 分片，再除样本数得到平均
                        val = cp_size * sum_of_sample_mean(val) / len(loss_masks)
                    else:
                        # 普通张量列表：拼接后取全元素均值，乘 cp_size 补偿 CP 分片
                        val = torch.cat(val).clone().detach()
                        val = val.mean() * cp_size
                else:
                    # 标量列表：取算术平均
                    val = sum(val) / len(val)
            elif isinstance(val, torch.Tensor):
                # 单张量：转 float 后取均值
                val = val.float().mean()
            else:
                # 其他类型不支持
                raise ValueError(f"Unsupported type: {type(val)} for key: {key}")
            # 张量转 Python 标量
            log_dict[key] = val.item() if isinstance(val, torch.Tensor) else val

        # 跨 DP gather 并平均后写日志
        reduced_log_dict = gather_log_data("rollout", args, rollout_id, log_dict)
        # CI 测试模式下对关键指标做合理性断言
        if args.ci_test and reduced_log_dict is not None:
            if (
                rollout_id == 0
                and "rollout/log_probs" in reduced_log_dict
                and "rollout/ref_log_probs" in reduced_log_dict
            ):
                # 第 0 步时 actor 与 ref 相同（未更新），log_prob 应相等
                assert reduced_log_dict["rollout/log_probs"] == reduced_log_dict["rollout/ref_log_probs"]
            if "rollout/log_probs" in reduced_log_dict:
                # 平均 log_prob 应为一个较小的负数
                assert -0.5 < reduced_log_dict["rollout/log_probs"] < 0
            if "rollout/entropy" in reduced_log_dict:
                # 熵应在 (0, 0.5) 的合理范围
                assert 0 < reduced_log_dict["rollout/entropy"] < 0.5

    if args.log_multi_turn:
        log_multi_turn_data(rollout_id, args, rollout_data)
    if args.log_passrate:
        log_passrate(rollout_id, args, rollout_data)

    # 统计答对样本（reward==1）的附加指标
    if args.log_correct_samples:
        if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
            cp_size = mpu.get_context_parallel_world_size()
            log_dict = {}
            response_lengths = rollout_data["response_lengths"]
            loss_masks = rollout_data["loss_masks"]
            total_lengths = rollout_data["total_lengths"]

            # 把数据按分位点切成 n 个桶并统计各桶占比（经验分布）
            def quantile(total_value, n_quantiles, data) -> dict:
                import math

                # 至少 2 个分位桶
                assert n_quantiles > 1, f"n_quantiles({n_quantiles}) must be greater than 1."

                # 均匀分位点：1/n, 2/n, ..., 1
                quantiles = [((i + 1) / n_quantiles) for i in range(n_quantiles)]
                # 每个分位桶的上界（按总长度等比划分）
                cut_points = [total_value * q for q in quantiles]
                cut_points[-1] = total_value

                # 统计落入每个桶的样本数（首个满足 d<=上界的桶）
                count = [0] * n_quantiles
                for d in data:
                    for i, point in enumerate(cut_points):
                        if d <= point:
                            count[i] += 1
                            break

                # 归一化为占比（加小量防除零）
                total = sum(count) + 1e-9
                percentile = [c / total for c in count]

                # 以 p25/p50/... 形式输出占比
                percentile = {f"p{min(math.ceil(q*100),100)}": p for q, p in zip(quantiles, percentile, strict=True)}
                return percentile

            # 原始奖励列表
            raw_rewards = rollout_data["raw_reward"]
            # Additional metrics for correct cases are calculated separately below.
            # 收集所有答对样本的长度、mask 与熵
            correct_response_lengths = []
            correct_total_lengths = []
            correct_loss_masks = []
            correct_entropy = []
            for i, raw_reward in enumerate(raw_rewards):
                if raw_reward == 1:
                    # reward 为 1 视为答对，收集其元数据
                    correct_response_lengths.append(response_lengths[i])
                    correct_total_lengths.append(total_lengths[i])
                    correct_loss_masks.append(loss_masks[i])
                    # 答对样本的熵近似用 -log_prob（该位置的对数概率的负值）表示
                    correct_entropy.append(-rollout_data["log_probs"][i])
            # 答对样本数
            num_correct_responses = len(correct_total_lengths)
            rollout_data["correct_response_lengths"] = correct_response_lengths
            # 对答对样本的回答长度做 4 分位统计（上限取配置的最大回答长度）
            correct_response_length_percentile = quantile(
                args.rollout_max_response_len, 4, rollout_data["correct_response_lengths"]
            )
            # 把各分位占比写回 rollout_data，供后续汇总记录
            for p, val in correct_response_length_percentile.items():
                rollout_data[f"correct_length/{p}"] = [val] * num_correct_responses
            if len(correct_entropy) > 0:
                # 有答对样本时：用 CP 感知的样本均值聚合器计算平均"熵"
                sum_of_sample_mean = get_sum_of_sample_mean(
                    correct_total_lengths, correct_response_lengths, correct_loss_masks
                )
                correct_entropy = sum_of_sample_mean(torch.cat(correct_entropy, dim=0))
                rollout_data["correct_entropy"] = [correct_entropy.item()] * num_correct_responses
            else:
                # 无答对样本时填 0
                rollout_data["correct_entropy"] = [0] * num_correct_responses


def log_multi_turn_data(rollout_id: int, args: Namespace, rollout_data: RolloutBatch) -> None:
    """
    Log multi-turn auxiliary metrics such as raw/observed response lengths and rounds.

    Operates only on PP last stage and TP rank 0. Uses GPU tensors when available
    to compute statistics without host transfers.
    """
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        log_dict = {}
        for key, val in rollout_data.items():
            # 统计 loss_masks：原始回答长度（含 observation）与去掉 observation 后的长度
            if key == "loss_masks":
                if val:  # Check if val is not empty
                    device = val[0].device  # Get device from first tensor

                    # Vectorized length calculation using torch
                    # 每个样本 mask 的形状长度即原始回答长度
                    raw_response_lengths = torch.tensor([v.shape[0] for v in val], dtype=torch.float32, device=device)
                    # 均值/最大/最小回答长度
                    log_dict["raw_response_length/response_length_mean"] = raw_response_lengths.mean().item()
                    log_dict["raw_response_length/response_length_max"] = raw_response_lengths.max().item()
                    log_dict["raw_response_length/response_length_min"] = raw_response_lengths.min().item()
                    # 截断率：达到最大回答长度上限的样本占比
                    log_dict["raw_response_length/response_length_clip_ratio"] = (
                        (raw_response_lengths >= args.rollout_max_response_len).float().mean().item()
                    )

                    # Vectorized sum calculation using torch - stay on GPU
                    # mask 求和即有效 token 数：不含 observation 的回答长度
                    wo_obs_response_lengths = torch.tensor(
                        [v.sum().item() for v in val], dtype=torch.float32, device=device
                    )
                    log_dict["wo_obs_response_length/response_length_mean"] = wo_obs_response_lengths.mean().item()
                    log_dict["wo_obs_response_length/response_length_max"] = wo_obs_response_lengths.max().item()
                    log_dict["wo_obs_response_length/response_length_min"] = wo_obs_response_lengths.min().item()
            if key == "round_number":
                # Use numpy for vectorized round number statistics
                # 多轮对话轮数的均值/最大/最小统计
                round_number_array = np.array(val)
                log_dict["multi_turn_metric/round_number_mean"] = np.mean(round_number_array)
                log_dict["multi_turn_metric/round_number_max"] = np.max(round_number_array)
                log_dict["multi_turn_metric/round_number_min"] = np.min(round_number_array)
        # 跨 DP 归约并记录
        gather_log_data("multi_turn", args, rollout_id, log_dict)


def log_passrate(rollout_id: int, args: Namespace, rollout_data: RolloutBatch) -> None:
    """
    Compute pass@k metrics from `raw_reward` groups and log the results.

    `raw_reward` is reshaped to `[group_number, group_size]`, then pass@k is
    estimated per problem and averaged.
    """
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        log_dict = {}
        for key, val in rollout_data.items():
            # 只处理原始奖励字段
            if key != "raw_reward":
                continue

            # 按 prompt 分组计算 pass@k（group_size 为每个 prompt 的采样数）
            log_dict |= compute_pass_rate(
                flat_rewards=val,
                group_size=args.n_samples_per_prompt,
                num_groups=args.rollout_batch_size,
            )

        # 跨 DP 归约并记录
        gather_log_data("passrate", args, rollout_id, log_dict)


def log_perf_data(rollout_id: int, args: Namespace) -> None:
    # 性能统计：委托给通用工具，并指定主 rank 判定与 FLOPs 计算方式
    # 主 rank = TP0 且 PP 最后阶段且 DP0
    # 前向 FLOPs 按全局 world size 均摊并换算成 TFLOPs（1e12）
    train_metric_utils.log_perf_data_raw(
        rollout_id=rollout_id,
        args=args,
        is_primary_rank=(
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.is_pipeline_last_stage()
            and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
        ),
        compute_total_fwd_flops=lambda seq_lens: calculate_fwd_flops(seqlens=seq_lens, args=args)
        / dist.get_world_size()
        / 1e12,
    )


def sync_actor_critic_data(
    args: Namespace,
    rollout_data: RolloutBatch | None = None,
    group: dist.ProcessGroup | None = None,
) -> None:
    """
    Broadcast `values` (from critic) and optionally `log_probs`/`ref_log_probs`
    (from actor) across PP ranks to align data dependencies.

    - Values are broadcast from src=1.
    - Log-probs and ref-log-probs are broadcast from src=0 when KL is used.
    Updates `rollout_data` in place with the synchronized tensors.
    """
    # 根据 rollout logprobs 开关决定使用哪个 log_prob 键
    log_probs_key = "log_probs" if not args.use_rollout_logprobs else "rollout_log_probs"
    values, log_probs, ref_log_probs = map(rollout_data.get, ("values", log_probs_key, "ref_log_probs"))

    # return when not the pp last stage
    # 两者都为空说明当前 PP 阶段不持有这些数据，直接返回
    if not values and not log_probs:
        return

    # 收集异步 broadcast 的句柄
    handles = []

    if not values:
        # 本阶段没有 values（critic 输出）时，按 log_prob 形状预分配接收缓冲
        values = [torch.empty_like(log_prob) for log_prob in log_probs]
    for value in values:
        # 从 src=1（critic 所在的 PP 阶段）广播 values
        handles.append(dist.broadcast(value, src=1, group=group, async_op=True))

    # 需要 KL（KL 散度惩罚/约束）时才同步 log_probs 与 ref_log_probs
    if args.kl_coef != 0 or args.use_kl_loss:
        if not log_probs:
            # 预分配 log_prob 接收缓冲
            log_probs = [torch.empty_like(value) for value in values]
        if not ref_log_probs:
            # 预分配 ref log_prob 接收缓冲
            ref_log_probs = [torch.empty_like(value) for value in values]
        for ref_log_prob, log_prob in zip(ref_log_probs, log_probs, strict=False):
            # 从 src=0（actor 所在的 PP 阶段）广播两份 log_prob
            handles.append(dist.broadcast(log_prob, src=0, group=group, async_op=True))
            handles.append(dist.broadcast(ref_log_prob, src=0, group=group, async_op=True))

    # 等待所有异步广播完成
    for handle in handles:
        handle.wait()

    # 把同步好的数据原地写回 rollout_data（跳过 None）
    rollout_data.update(
        {
            k: v
            for k, v in {
                "values": values,
                log_probs_key: log_probs,
                "ref_log_probs": ref_log_probs,
            }.items()
            if v is not None
        }
    )
