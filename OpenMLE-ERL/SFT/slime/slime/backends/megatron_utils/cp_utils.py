from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu


def get_logits_and_tokens_offset_with_cp(
    total_length: int,
    response_length: int,
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
):
    """
    All offsets start from the begining of the prompt.
    """
    # 当前 rank 在上下文并行（Context Parallel, CP）组中的编号与组大小
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    # 该函数只在 CP>1 时有意义
    assert cp_size > 1

    # prompt（输入部分）长度 = 总长 - 回答（response）长度
    prompt_length = total_length - response_length
    if qkv_format == "thd":
        # thd 格式按整条序列切：每 rank 持有两个 chunk（zigzag 切法），故除以 2*cp_size
        chunk_size = (total_length + 2 * cp_size - 1) // (2 * cp_size)
    else:
        # bshd 格式按 batch 内最大序列长度切，需要显式传入 max_seq_len
        assert max_seq_len is not None, "max_seq_len must be provided for qkv_format=bshd"
        chunk_size = (max_seq_len + 2 * cp_size - 1) // (2 * cp_size)

    # the offset of 2 chunks
    # 本 rank 负责的两个 chunk：一个在序列头部（第 cp_rank 块），一个在尾部（对称的第 cp_rank 块）
    # 这种"头尾配对"的 zigzag 切分能让每个 rank 承载均衡的 attention 负载
    chunk_0 = (cp_rank * chunk_size, (cp_rank + 1) * chunk_size)
    chunk_1 = ((2 * cp_size - cp_rank - 1) * chunk_size, (2 * cp_size - cp_rank) * chunk_size)

    # the offset of 2 logits, note that the logits need a "-1".
    # 计算 logits 的有效区间：只有回答部分（预测下一个 token）需要 log_prob，
    # 预测位置最多到 total_length-1（预测最后一个 token），故上限为 total_length-1
    logits_0 = (max(chunk_0[0], prompt_length - 1), min(chunk_0[1], total_length - 1))
    logits_1 = (max(chunk_1[0], prompt_length - 1), min(chunk_1[1], total_length - 1))

    # when the sequence is empty, make an empty slice to continue the gradient flow.
    # chunk 与 logits 区间有交集时，对应 token 区间为 logits 区间右移一位（logits[i] 预测 token[i+1]）
    if logits_0[0] < logits_0[1]:
        token_0 = (logits_0[0] + 1, logits_0[1] + 1)
    else:
        # 空区间时置为 (0,0)：保留空切片以维持梯度回传链路
        logits_0 = (0, 0)
        token_0 = (0, 0)

    # 同上处理第二个 chunk
    if logits_1[0] < logits_1[1]:
        token_1 = (logits_1[0] + 1, logits_1[1] + 1)
    else:
        logits_1 = (0, 0)
        token_1 = (0, 0)

    # 返回 chunk 大小、两个 chunk 区间、两个 logits 区间、两个 token 区间
    return chunk_size, (chunk_0, chunk_1), (logits_0, logits_1), (token_0, token_1)


def get_sum_of_sample_mean(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    calculate_per_token_loss: bool = False,
    qkv_format: str = "thd",
    max_seq_lens: list[int] | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Calculate correct sample mean for CP
    """
    # 获取上下文并行（CP）组大小
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size == 1:

        # CP=1 时，per-sample 均值：每个样本的 loss 按自身有效 token 数归一（样本均值，而非 token 均值）
        def sum_of_sample_mean(x: torch.Tensor) -> torch.Tensor:
            return sum(
                [
                    # 对每个样本：先乘 loss_mask 掩掉无效位置，再除以该样本有效 token 数（clamp 防除零）
                    (x_i * loss_mask_i).sum() / torch.clamp_min(loss_mask_i.sum(), 1)
                    for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks, strict=False)
                ]
            )

        # per-token 模式：直接对掩码后的 loss 求和（分母在别处统一处理）
        def sum_of_token(x: torch.Tensor) -> torch.Tensor:
            return sum(
                [
                    (x_i * loss_mask_i).sum()
                    for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks, strict=False)
                ]
            )

    else:
        # CP>1 时：序列被切到多个 rank，每个 rank 只持有一段 loss_mask，
        # 但归一化分母必须用整条序列的有效 token 数，因此预计算每个 rank 的分段 mask
        cp_chunk_lengths: list[int] = []
        chunked_loss_masks: list[torch.Tensor] = []

        # 逐样本计算其在本 rank 上的两段 loss mask
        for i, (total_length, response_length, loss_mask) in enumerate(
            zip(total_lengths, response_lengths, loss_masks, strict=False)
        ):
            # bshd 格式需要对应的 max_seq_len
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
            # prompt 长度，用于把全局 token 坐标折算到响应部分内的偏移
            prompt_length = total_length - response_length
            # 拿到该样本在 CP 切分下的两段 token 区间
            _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )
            # 按区间切出两段 loss mask（区间是全局坐标，减 prompt_length 转为响应内坐标）
            loss_mask_0 = loss_mask[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
            loss_mask_1 = loss_mask[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
            # 拼接两段作为本 rank 的分段 mask，并记录其长度
            chunked_loss_masks.append(torch.cat([loss_mask_0, loss_mask_1], dim=0))
            cp_chunk_lengths.append(chunked_loss_masks[i].size(0))

        # CP 版 per-sample 均值：分子用本 rank 的分段 mask，
        # 分母用整条序列的完整 mask 求和，保证各 CP rank 的归一化一致
        def sum_of_sample_mean(x: torch.Tensor) -> torch.Tensor:
            return sum(
                [
                    (x_i * chunked_loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)
                    for x_i, chunked_loss_mask, loss_mask in zip(
                        x.split(cp_chunk_lengths, dim=0), chunked_loss_masks, loss_masks, strict=False
                    )
                ]
            )

        # CP 版 per-token 求和
        def sum_of_token(x: torch.Tensor) -> torch.Tensor:
            return sum(
                [
                    (x_i * chunked_loss_mask).sum()
                    for x_i, chunked_loss_mask in zip(
                        x.split(cp_chunk_lengths, dim=0), chunked_loss_masks, strict=False
                    )
                ]
            )

    # 按 per-token 开关返回对应的聚合函数
    return sum_of_sample_mean if not calculate_per_token_loss else sum_of_token


def all_gather_with_cp(tensor: torch.Tensor, total_length: int, response_length: int) -> torch.Tensor:
    """
    Gather tensors across all ranks in the context parallel group.
    The first dimension of the output tensor will be the `response_length`.
    """
    cp_group = mpu.get_context_parallel_group()
    cp_size = mpu.get_context_parallel_world_size()

    # CP=1 时无需 gather，直接返回
    if cp_size == 1:
        return tensor

    # 获取本 rank 两段 logits 的全局区间
    _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(total_length, response_length)

    prompt_length = total_length - response_length

    # 本 rank 的 log_prob 被切成两段：第一段长为 logits_0 区间长度，剩余为第二段
    chunk_0 = tensor[: logits_offset[0][1] - logits_offset[0][0]]
    chunk_1 = tensor[logits_offset[0][1] - logits_offset[0][0] :]
    # 断言第二段长度与 logits_1 区间长度一致
    assert chunk_1.shape[0] == logits_offset[1][1] - logits_offset[1][0]

    # 构造指定长度的全 0 张量（requires_grad 以便梯度可回传）
    def zero(len: int) -> torch.Tensor:
        return torch.zeros(
            [len] + list(tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
            requires_grad=True,
        )

    # logprob should be within the range of [prompt_length - 1, total_length - 1]
    # 按本 rank 持有的区间情况，把两段数据放回整条 response 对应位置，其余用 0 填充
    if chunk_0.shape[0] == 0 and chunk_1.shape[0] == 0:
        # all empty
        # 本 rank 不持有任何 logits，全部填 0
        full_tensor = zero(response_length)
    elif chunk_0.shape[0] != 0 and chunk_1.shape[0] == 0:
        # only first chunk
        # 只持有第一段：左填 logits_0 起点之前的 0，右填到序列末尾的 0
        left = zero(logits_offset[0][0] - (prompt_length - 1))
        right = zero(total_length - 1 - logits_offset[0][1])
        full_tensor = torch.cat([left, chunk_0, right], dim=0)
    elif chunk_0.shape[0] == 0 and chunk_1.shape[0] != 0:
        # only second chunk
        # 只持有第二段：同理按 logits_1 区间补零
        left = zero(logits_offset[1][0] - (prompt_length - 1))
        right = zero(total_length - 1 - logits_offset[1][1])
        full_tensor = torch.cat([left, chunk_1, right], dim=0)
    else:
        # 同时持有两段：在两段之间再补中间空洞的 0
        left = zero(logits_offset[0][0] - (prompt_length - 1))
        mid = zero(logits_offset[1][0] - logits_offset[0][1])
        right = zero(total_length - 1 - logits_offset[1][1])
        full_tensor = torch.cat([left, chunk_0, mid, chunk_1, right], dim=0)

    # 断言拼接后的长度恰好等于 response 长度
    assert full_tensor.shape[0] == response_length, f"Expected {response_length}, got {full_tensor.shape}"
    # 各 rank 的张量在各自持有的位置有值、其余为 0，all_reduce 求和即得到完整的 response 级张量
    full_tensor = dist.nn.all_reduce(full_tensor, group=cp_group)
    return full_tensor


def slice_with_cp(
    tokens: torch.Tensor,
    pad_value: tuple[int, float, Callable],
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
) -> torch.Tensor:
    # 当前 rank 的 CP 编号与 CP 组大小
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()

    # bshd 格式切分必须知道 max_seq_len
    if qkv_format == "bshd":
        assert max_seq_len is not None

    # 按给定 pad 长度补齐 tokens；pad_value 可以是常量或自定义填充函数
    def pad_tokens(tokens, pad):
        if isinstance(pad_value, Callable):
            # 自定义填充函数：传入 tokens 与需要补的长度
            pad_func = pad_value
            tokens = pad_func(tokens, pad)
        else:
            # pad on the first dimension
            # 用 F.pad 只在第一维（序列维）尾部补 pad 个值
            pad_tuple = (0, 0) * (tokens.dim() - 1) + (0, pad)
            tokens = F.pad(tokens, pad_tuple, value=pad_value)
        return tokens

    # CP=1：不需要切分；bshd 时把序列补齐到 max_seq_len
    if cp_size == 1:
        if qkv_format == "bshd":
            pad = max_seq_len - tokens.size(0)
            tokens = pad_tokens(tokens, pad)
        return tokens

    # 计算每 rank 每 chunk 的大小：thd 按实际 token 数，bshd 按 max_seq_len
    token_len = len(tokens)
    if qkv_format == "thd":
        chunk_size = (token_len + 2 * cp_size - 1) // (2 * cp_size)
    else:
        chunk_size = (max_seq_len + 2 * cp_size - 1) // (2 * cp_size)

    # pad
    # 先把序列补齐到 2*cp_size 个 chunk 的整倍数，保证所有 rank 切分长度一致
    pad = 2 * cp_size * chunk_size - token_len
    tokens = pad_tokens(tokens, pad)

    # get 2 chunk for thd cp
    # zigzag 切分：本 rank 取头部的第 cp_rank 块和尾部的对称块（保证 attention 负载均衡）
    start_1, end_1 = chunk_size * cp_rank, chunk_size * (cp_rank + 1)
    start_2, end_2 = chunk_size * (2 * cp_size - cp_rank - 1), chunk_size * (2 * cp_size - cp_rank)
    # 拼接两个 chunk 作为本 rank 持有的序列片段
    return torch.cat([tokens[start_1:end_1], tokens[start_2:end_2]])


def slice_log_prob_with_cp(
    log_prob: list[float] | torch.Tensor,
    total_length: int,
    response_length: int,
    qkv_format: str = "thd",
    max_token_len: int | None = None,
) -> list[float] | torch.Tensor:
    # 传入的 log_prob 长度必须等于 response 长度
    assert len(log_prob) == response_length

    cp_size = mpu.get_context_parallel_world_size()

    # CP=1：无需切分
    if cp_size == 1:
        return log_prob

    prompt_length = total_length - response_length
    # 拿到本 rank 两段 logits 的全局区间
    _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(
        total_length, response_length, qkv_format, max_token_len
    )

    # 把全局区间折算到 response 内部坐标（减去 prompt_length-1），切出两段 log_prob
    chunk_1 = log_prob[logits_offset[0][0] - (prompt_length - 1) : logits_offset[0][1] - (prompt_length - 1)]
    chunk_2 = log_prob[logits_offset[1][0] - (prompt_length - 1) : logits_offset[1][1] - (prompt_length - 1)]

    # list 用列表拼接，张量用 torch.cat 沿第 0 维拼接
    if isinstance(log_prob, list):
        return chunk_1 + chunk_2
    else:
        return torch.cat([chunk_1, chunk_2], dim=0)
