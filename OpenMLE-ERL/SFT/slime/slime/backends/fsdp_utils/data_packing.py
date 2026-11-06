"""Data packing utilities for FSDP backend to reduce padding overhead."""

# 导入 math 用于向上取整计算打包数
import math

# 导入 torch 用于把打包结果转成张量
import torch

# 导入按序列长度均衡分组（负载均衡装箱）的工具函数
from slime.utils.seqlen_balancing import get_seqlen_balanced_partitions


# 将多条序列打包（sequence packing / Token 打包）成稠密批次，减少 padding 浪费
def pack_sequences(
    tokens: list[list[int]],
    loss_masks: list[list[int]],
    rewards: list[float],
    raw_rewards: list,
    response_lengths: list[int],
    advantages: list[float],
    returns: list[float],
    rollout_log_probs: list[list[float]] | None = None,
    multimodal_train_inputs: list[dict] | None = None,
    max_tokens_per_gpu: int | None = None,
    num_packs: int | None = None,
) -> list[dict]:
    """
    Pack sequences into dense batches with cumulative sequence lengths.

    Args:
        tokens: List of token sequences
        loss_masks: List of loss masks
        rewards: List of rewards per sequence
        raw_rewards: List of raw rewards per sequence
        response_lengths: List of response lengths per sequence
        advantages: List of advantages per sequence
        returns: List of returns per sequence
        rollout_log_probs: List of rollout log probabilities per sequence
        multimodal_train_inputs: List of dict of multimodal tensors for training per sequence
        max_tokens_per_gpu: Maximum tokens per GPU pack
        num_packs: Explicit number of packs to create

    Returns:
        List of packed batches with tokens, masks, cu_seqlens, rewards, raw_rewards, response_lengths, advantages, returns
    """
    # 输入为空时直接返回空列表
    if not tokens:
        return []

    # 计算每条序列的 token 长度，供装箱算法使用
    seq_lengths = [len(t) for t in tokens]

    # Determine number of packs and use balanced partitioning
    # 确定打包份数 k：
    if num_packs:
        # 显式指定了打包数则直接采用
        k_partitions = num_packs
    elif max_tokens_per_gpu:
        # 指定了每包最大 token 数时，用总 token 数除以该上限并向上取整
        total_tokens = sum(seq_lengths)
        k_partitions = max(1, math.ceil(total_tokens / max_tokens_per_gpu))
    else:
        # 两者都未指定时只打成一包
        k_partitions = 1

    # Use balanced partitioning for optimal load distribution
    # 用长度均衡划分（类似 LPT 装箱）把序列分配到 k 个包，尽量让各 GPU 负载均匀；不等长允许更优均衡
    partitions = get_seqlen_balanced_partitions(
        seq_lengths, k_partitions=k_partitions, equal_size=False  # Allow variable sizes for better balance
    )

    # Pack each partition
    # 逐个包构建稠密批次
    result = []
    for indices in partitions:
        # Build cumulative sequence lengths
        # cu_seqlens 为累积序列长度数组（varlen attention 的标准输入），首元素为 0
        cu_seqlens = [0]
        # 各摊平后的一维缓冲区：token、mask、位置 id、优势、回报、rollout 对数概率
        flat_tokens = []
        flat_masks = []
        flat_positionids = []
        flat_advantages = []
        flat_returns = []
        flat_rollout_log_probs = []

        # 遍历本包内的每条序列索引，摊平拼接
        for i in indices:
            # 取该序列的 token 与 loss mask
            seq_tokens = tokens[i]
            seq_mask = loss_masks[i]
            # 位置 id 为 0..len-1 的自然序列
            seq_positionids = list(range(len(seq_tokens)))

            # 把该序列的各字段追加到对应的一维缓冲区
            flat_tokens.extend(seq_tokens)
            flat_positionids.extend(seq_positionids)
            flat_masks.extend(seq_mask)
            flat_advantages.extend(advantages[i])
            flat_returns.extend(returns[i])
            # rollout 对数概率按需拼接
            if rollout_log_probs:
                flat_rollout_log_probs.extend(rollout_log_probs[i])
            # 累积长度追加当前序列结束位置（前值 + 本序列长度）
            cu_seqlens.append(cu_seqlens[-1] + len(seq_tokens))

        # 组装打包批次字典：把摊平结果转成张量，按序列的字段按包内顺序切片
        packed_batch = {
            "tokens": torch.tensor(flat_tokens, dtype=torch.long),
            "loss_masks": torch.tensor(flat_masks, dtype=torch.int),
            "position_ids": torch.tensor(flat_positionids, dtype=torch.int),
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
            "rewards": torch.tensor([rewards[i] for i in indices], dtype=torch.float32),
            "raw_reward": [raw_rewards[i] for i in indices],
            "response_lengths": [response_lengths[i] for i in indices],
            "advantages": torch.tensor(flat_advantages, dtype=torch.float32),
            "returns": torch.tensor(flat_returns, dtype=torch.float32),
            "rollout_log_probs": torch.tensor(
                flat_rollout_log_probs, dtype=torch.float32, device=torch.cuda.current_device()
            ),
        }

        # Collect and add multimodal training tensors for this partition
        # 收集本包的多模态训练张量（如图像嵌入），并记录每条序列的条目数用于反解
        if multimodal_train_inputs:
            multimodal_data = {}  # key -> concatenated tensor
            multimodal_num_items = {}  # key -> list of item counts per sequence
            # 遍历包内每条序列的多模态输入
            for i in indices:
                # 该序列无多模态数据则跳过
                if multimodal_train_inputs[i] is None:
                    continue
                # 按键（如 pixel_values）聚合张量
                for key, mm_tensor in multimodal_train_inputs[i].items():
                    if key not in multimodal_data:
                        # 首次遇到该键：直接存入并记录条目数
                        multimodal_data[key] = mm_tensor
                        multimodal_num_items[key] = [mm_tensor.size(0)]
                    else:
                        # 已存在：沿第 0 维拼接张量，并追加本序列的条目数
                        multimodal_data[key] = torch.cat([multimodal_data[key], mm_tensor], dim=0)
                        multimodal_num_items[key].append(mm_tensor.size(0))
            # 把多模态聚合结果与条目数记录写入打包批次
            packed_batch["multimodal_train_inputs"] = multimodal_data
            packed_batch["multimodal_num_items"] = multimodal_num_items

        # 收集本包结果
        result.append(packed_batch)

    # 返回所有打包批次
    return result


# 把打包批次反解回每条序列独立的字典列表（packing 的逆操作）
def unpack_sequences(packed_batch: dict) -> list[dict]:
    """
    Unpack sequences from a packed batch.

    Args:
        packed_batch: Packed batch

    Returns:
        List of unpacked batches
    """

    # 取累积序列长度数组
    cu_seqlens = packed_batch["cu_seqlens"]
    # 序列条数 = 累积长度数 - 1
    num_sequences = len(cu_seqlens) - 1
    # 取每条序列的响应长度（用于切 response 相关字段）
    response_lengths = packed_batch["response_lengths"]
    # 取多模态条目数记录（可能不存在）
    multimodal_num_items = packed_batch.get("multimodal_num_items", {})

    # 输出实例列表
    instances = []

    # Calculate pad_length by counting trailing zeros
    # 通过统计尾部 0 计算 padding 长度：找到最后一个非零 token 的位置
    tokens = packed_batch["tokens"]
    nonzero_indices = (tokens != 0).nonzero(as_tuple=True)[0]
    if len(nonzero_indices) > 0:
        # Last non-zero index, pad_length is everything after it
        # padding 长度 = 总长度 - 最后一个非零索引 - 1
        pad_length = len(tokens) - nonzero_indices[-1].item() - 1
    else:
        pad_length = 0  # No padding if no non-zero tokens (or all zeros)
    # 逐条序列反解
    for i in range(num_sequences):
        # 该序列在摊平数组中的起止偏移（cu_seqlens 相邻元素之差）
        start_idx = cu_seqlens[i].item()
        end_idx = cu_seqlens[i + 1].item()
        # 本序列的实例字典
        instance = {}

        # Copy any additional attributes that might exist in the packed batch
        # 复制打包批次中的所有附加属性，并按其语义做相应切片
        for key, value in packed_batch.items():
            if key not in instance:
                # Skip multimodal_num_items - it's metadata
                # 条目数记录属于元数据，不进入实例
                if key == "multimodal_num_items":
                    continue
                # Handle multimodal_train_inputs dict: split each tensor using multimodal_num_items
                # 多模态张量：按每条序列记录的条目数从拼接结果中切出本序列的部分
                elif key == "multimodal_train_inputs" and isinstance(value, dict):
                    instance[key] = {}
                    for mm_key, mm_tensor in value.items():
                        if mm_key in multimodal_num_items:
                            # 取该键的条目数列表
                            num_items_list = multimodal_num_items[mm_key]
                            # 计算本序列在拼接张量中的起止索引（前面序列条目数之和）
                            start_mm_idx = sum(num_items_list[:i])
                            end_mm_idx = start_mm_idx + num_items_list[i]
                            # 条目数大于 0 时才切片保存
                            if num_items_list[i] > 0:
                                instance[key][mm_key] = mm_tensor[start_mm_idx:end_mm_idx]
                # For tensor attributes, we need to slice them appropriately
                # 张量属性：按不同字段的打包方式选择切片策略
                elif isinstance(value, torch.Tensor):
                    if key in ["log_probs", "ref_log_probs", "cur_log_probs", "entropy"]:
                        # These are computed from logits[:-1] so they have length seq_len-1
                        # 这些量由 logits[:-1] 计算，长度为 seq_len-1，需按响应段并扣除 padding 定位
                        instance[key] = value[
                            end_idx - 1 - response_lengths[i] - pad_length : end_idx - 1 - pad_length
                        ]
                    elif key == "rollout_log_probs":
                        # rollout_log_probs is packed based on response_lengths, so slice differently
                        # 该张量按响应长度独立摊平打包，用响应长度前缀和定位本序列区段
                        instance[key] = value[sum(response_lengths[:i]) : sum(response_lengths[: i + 1])]
                    elif key in ["tokens", "position_ids"]:
                        # For other tensor attributes, try to slice them
                        # 全序列级摊平张量：直接按 cu_seqlens 区段切片
                        if len(value) > start_idx:
                            instance[key] = value[start_idx:end_idx]
                        else:
                            raise ValueError(f"Attribute {key} is not found in the packed batch")
                    elif key in ["loss_masks", "advantages", "returns"]:
                        # 这些字段只针对响应部分打包，按响应长度前缀和切片
                        instance[key] = value[sum(response_lengths[:i]) : sum(response_lengths[: i + 1])]
                elif isinstance(value, list):
                    # 列表属性为按序列存储，直接取第 i 个元素
                    instance[key] = value[i]
                else:
                    # 未知类型的字段无法反解，报错提示
                    raise ValueError(f"Attribute {key} is not found in the packed batch")

        # 收集本序列实例
        instances.append(instance)

    # 返回全部反解结果
    return instances
