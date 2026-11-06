# Adapt from https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/models/utils.py
# and https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/trainer/ppo_utils/experience_maker.py
# 本文件改编自 OpenRLHF：集中实现 PPO/GRPO/REINFORCE++ 等强化学习算法的核心数值计算

# 导入 Namespace：仅为类型注解使用（args 对象）
from argparse import Namespace

# 导入 torch：所有张量运算基础
import torch
# 导入 torch.distributed：跨 TP/CP 进程组的 all_reduce 通信
import torch.distributed as dist
# 导入 torch.nn.functional：张量 padding 等工具
import torch.nn.functional as F


@torch.compile(dynamic=True)
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_loss_type: str,
    importance_ratio: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        kl_loss_type: Type of KL estimator (k1, k2, k3, low_var_kl).
        importance_ratio: Optional IS ratio (π_θ/π_old) for unbiased KL estimation.
    """
    # 逐 token 的对数概率比 log(π_new/π_old)，先转 float32 保证数值精度
    log_ratio = log_probs.float() - log_probs_base.float()

    if kl_loss_type == "k1":
        # k1 估计量：直接用 log ratio（可为负，是最朴素的无偏估计）
        kl = log_ratio
    elif kl_loss_type == "k2":
        # k2 估计量：log ratio 平方的一半（恒非负但有偏）
        kl = log_ratio**2 / 2.0
    elif kl_loss_type in ["k3", "low_var_kl"]:
        # The non negative kl approximation in
        # http://joschu.net/blog/kl-approx.html
        # Besides non negative, it is also unbiased and have lower variance.
        # k3 估计量：e^(-r) - 1 + r，非负、无偏且低方差
        log_ratio = -log_ratio
        # 代入公式 exp(x) - 1 - x（x = -log ratio）
        kl = log_ratio.exp() - 1 - log_ratio
    else:
        # 未知的 KL 估计器类型直接报错
        raise ValueError(f"Unknown kl_loss_type: {kl_loss_type}")

    # Apply IS ratio for unbiased KL estimation (DeepSeek-V3.2)
    # 提供重要性采样比率时加权，修正策略更新导致的 KL 估计偏差（IS 校正）
    if importance_ratio is not None:
        kl = importance_ratio * kl

    # Clamp only for low_var_kl for numerical stability
    # 仅 low_var_kl 需要截断到 [-10, 10]，防止极端 log ratio 造成数值爆炸
    if kl_loss_type == "low_var_kl":
        kl = torch.clamp(kl, min=-10, max=10)

    # 返回逐 token 的 KL 张量
    return kl


def compute_opsm_mask(
    args: Namespace,
    full_log_probs: list[torch.Tensor],
    full_old_log_probs: list[torch.Tensor],
    advantages: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Off-Policy Sequence Masking (OPSM) mask.

    Args:
        args: Configuration containing `opsm_delta` threshold.
        full_log_probs: Current policy log-probs per sample.
        full_old_log_probs: Old policy log-probs per sample.
        advantages: Advantage values per sample.
        loss_masks: Loss masks per sample.

    Returns:
        Tuple of `(opsm_mask, opsm_clipfrac)` where `opsm_mask` is a
        concatenated tensor of per-token masks and
        `opsm_clipfrac` is the count of masked sequences.
    """
    # 收集每条序列的 OPSM mask
    opsm_mask_list = []
    # 取第一个 advantage 张量的设备，用于放置统计量
    device = advantages[0].device
    # 初始化被剪枝序列的占比统计（标量张量）
    opsm_clipfrac = torch.tensor(0.0, device=device)

    # 逐条序列配对处理（当前/旧 log prob、优势、loss mask）
    for full_log_prob, full_old_log_prob, advantage, loss_mask in zip(
        full_log_probs, full_old_log_probs, advantages, loss_masks, strict=False
    ):
        # Calculate sequence-level KL
        # 序列级 KL：逐 token KL 在有效位置上的平均（分母下限 1 防除零）
        seq_kl = ((full_old_log_prob - full_log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)

        # Create mask: 0 if (advantage < 0 and seq_kl > delta), else 1
        # 负优势且序列级 KL 超过阈值 delta 的序列：判定为"离策略"需要剪枝，置 1
        mask = ((advantage < 0) & (seq_kl > args.opsm_delta)).float()
        # 累计被剪枝的 token 占比，作为剪枝率统计
        opsm_clipfrac += mask.sum() / torch.clamp_min(loss_mask.sum(), 1)

        # 保留 mask：1 表示参与训练、0 表示整条序列被剪掉
        opsm_mask_list.append(1 - mask)

    # 拼接所有序列的 mask 成一个整体张量
    opsm_mask = torch.cat(opsm_mask_list, dim=0)
    # 返回（OPSM mask, 剪枝率）
    return opsm_mask, opsm_clipfrac


def compute_gspo_kl(
    full_log_probs: list[torch.Tensor],
    full_old_log_probs: list[torch.Tensor],
    local_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> torch.Tensor:
    """Compute GSPO-style per-sequence KL divergence.

    Args:
        full_log_probs: Current policy log-probs per sample (full or CP-local).
        full_old_log_probs: Old policy log-probs per sample (full or CP-local).
        local_log_probs: Local (CP-local) log-probs for expansion shape reference.
        loss_masks: Loss masks per sample.

    Returns:
        Concatenated tensor of per-token KL values where each token in a
        sequence has the same KL value (the sequence-level KL).
    """
    # Compute sequence-level KL and expand to per-token
    # 先逐序列计算标量 KL：逐 token KL 在有效位置上的平均
    ppo_kl = [
        ((old_logprob - log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)
        for log_prob, old_logprob, loss_mask in zip(full_log_probs, full_old_log_probs, loss_masks, strict=False)
    ]
    # 把每条序列的标量 KL 广播展开到与该序列 local log prob 相同的形状（GSPO：序列内所有 token 共享同一 KL）
    ppo_kl = [kl.expand_as(log_prob) for kl, log_prob in zip(ppo_kl, local_log_probs, strict=False)]
    # 拼接所有序列成整体张量
    ppo_kl = torch.cat(ppo_kl, dim=0)

    # 返回逐 token 的序列级 KL 张量
    return ppo_kl


@torch.compile(dynamic=True)
def compute_policy_loss(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None = None,
):
    # 由 KL 反推重要性采样比率 ratio = exp(-KL) = π_new/π_old
    ratio = (-ppo_kl).exp()
    # 标准 PPO 的未截断策略梯度损失（取负号把目标最大化转为损失最小化）
    pg_losses1 = -ratio * advantages
    # 截断后的损失：ratio 限制在 [1-eps_low, 1+eps_high] 区间（支持非对称截断）
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    # PPO 的悲观取法：两个损失取最大值（保守更新）
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    # 统计被截断（clip 生效）的 token 占比，用于监控策略更新幅度
    clipfrac = torch.gt(pg_losses2, pg_losses1).float()

    if eps_clip_c is not None:
        # 双重截断（dual-clip PPO）开启：下界 c 必须大于 1
        assert (
            eps_clip_c > 1.0
        ), f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {eps_clip_c}."
        # 第三条损失：ratio 压到常数 c，限制负优势时的过度更新
        pg_losses3 = -eps_clip_c * advantages
        # 三个损失取最小值链：先与截断损失取 min
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        # 负优势样本用双重截断分支，正优势样本用普通截断分支
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    else:
        # 未启用双重截断：直接用普通 PPO 截断损失
        pg_losses = clip_pg_losses1

    # 返回（逐 token 策略损失, 截断率）
    return pg_losses, clipfrac


def compute_log_probs(logits: torch.Tensor, tokens: torch.Tensor, process_group: dist.ProcessGroup | None):
    # TODO: when megatron is not installed, fall back to naive implementation
    # 使用 Megatron 的融合词表并行交叉熵：TP 分片下高效且省显存
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

    # convert to [seq_len, batch_size, vocab_size] as expected by fused_vocab_parallel_cross_entropy
    # 融合算子要求 [seq, batch, vocab] 形状：在 batch 维插入一个大小为 1 的维度
    logits = logits.unsqueeze(1)
    # 目标 token 同样升维成 [seq, batch]
    tokens = tokens.unsqueeze(1)
    # 返回负对数似然（取负得 log prob）
    return -fused_vocab_parallel_cross_entropy(logits, tokens, process_group)


# from https://github.com/volcengine/verl/blob/0bdf7f469854815177e73dcfe9e420836c952e6e/verl/utils/megatron/tensor_parallel.py#L99
# 词表并行（TP 分片）下的熵计算自定义 autograd 函数，改编自 verl
class _VocabParallelEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor, process_group: dist.ProcessGroup) -> torch.Tensor:

        # 内联小工具：逐最后一维做乘加归约，计算 E[logit]
        @torch.compile(dynamic=True)
        def mul_reduce(a, b):
            return (a * b).sum(dim=-1, keepdim=True)

        # 先在本地分片内求逐位置的最大 logit
        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        # 跨 TP 组 all_reduce 取全局最大值（数值稳定的 softmax 基准）
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group)
        # 减去全局最大值，平移 logit 防止 exp 溢出
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        # 原地求指数得到未归一化的 softmax 分子
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        # 本地求和得到分母的本分片部分
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        # 跨 TP 组 all_reduce 求和，得到全局分母
        dist.all_reduce(normalized_sum_exp_logits, group=process_group)
        # 原地除法得到完整 softmax 概率
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        # 计算 sum(p * logit)，即 E_{π}[logit]
        sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
        # 跨 TP 组 all_reduce 得到全局期望
        dist.all_reduce(sum_softmax_times_logits, group=process_group)
        # 熵公式：H = log Z - E[logit]，log Z = logit_max + log(sum exp 归一化项)
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        # 保存反向传播所需的中间量
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        # 去掉最后一维的 keepdim，返回逐 token 熵
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # 取出 forward 保存的张量
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        # reuse softmax_logits as grad
        # 反向梯度推导：dH/dlogit = p * (E[logit] - logit)，用原地运算省显存
        # 先把 logits 原地减去期望，得到 (logit - E[logit])
        vocab_parallel_logits.sub_(sum_softmax_times_logits)
        # softmax_logits 原地乘 logits 差值，得到 p*(logit - E[logit])
        softmax_logits.mul_(vocab_parallel_logits)
        # 再乘以上游梯度（升维对齐 keepdim）
        softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
        # recover vocab_parallel_logits
        # 还原 logits 原值，避免破坏外部持有的张量
        vocab_parallel_logits.add_(sum_softmax_times_logits)
        # 取负号得到最终梯度 dH/dlogit = -p*(logit - E[logit]) 的正确符号
        softmax_logits.mul_(-1)
        # 返回对 logits 的梯度；process_group 无需梯度
        return softmax_logits, None


def compute_entropy_from_logits(logits: torch.Tensor, process_group) -> torch.Tensor:
    # 调用上面的自定义 autograd 函数计算词表并行的逐 token 熵
    return _VocabParallelEntropy.apply(logits, process_group)


def get_grpo_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
):
    # GRPO：标量奖励直接广播到每个 token，返回值就是逐 token 的常数回报
    returns = []
    for i in range(len(rewards)):
        # 用与 KL 张量同形状的全 1 乘上标量奖励实现广播
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    # 返回每条序列的逐 token 回报列表
    return returns


def get_reinforce_plus_plus_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    response_lengths: list[int],
    total_lengths: list[int],
    kl_coef: float,
    gamma: float,
) -> list[torch.Tensor]:
    """
    Calculates discounted returns for REINFORCE++ (https://arxiv.org/pdf/2501.03262)

    Args:
        rewards (Tensor): A tensor of scalar rewards for each sequence.
        kl (List[Tensor]): List of per-token KL divergence tensors for sequence chunks.
        loss_masks (List[Tensor]): List of response-only loss masks for each full sequence.
        response_lengths (List[int]): The full length of each response sequence.
        total_lengths (List[int]): The full length of each sequence (prompt + response).
        kl_coef (float): Coefficient for the KL penalty.
        gamma (float): The discount factor.

    Returns:
        List[torch.Tensor]: A list of return (G_t) tensors for the
                            local sequence chunks owned by the current GPU rank.
    """
    # 导入 Megatron 并行工具，获取 CP（上下文并行）规模
    from megatron.core import mpu

    # 当前 CP 组的大小（>1 表示序列被切分到多个 rank）
    cp_size = mpu.get_context_parallel_world_size()

    # 收集本 rank 拥有的各序列 chunk 的回报
    final_returns_chunks = []
    for i in range(len(rewards)):
        # 本 rank 持有的 KL chunk
        local_kl_chunk = kl[i]
        # 取该序列的总长度与响应长度
        total_len, response_len = total_lengths[i], response_lengths[i]

        if cp_size > 1:
            # Step 1,2:Gather all chunks and token_offsets from all ranks and reconstruct the full response tensor by splitting and placing each part
            # CP 分片时：all-gather 各 rank 的 chunk 并按 token 偏移重组成完整响应张量
            from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

            full_kl_response = all_gather_with_cp(local_kl_chunk, total_len, response_len)
        else:
            # 单卡：KL chunk 本身就是完整响应
            full_kl_response = local_kl_chunk

        # Step 3: Compute returns on full response kl tensor.
        # 取该序列的响应部分 loss mask
        full_mask = loss_masks[i]
        # 整条序列被 mask 光则无法放置奖励，直接报错
        assert full_mask.sum().item() > 0, f"Sequence at index {i} is fully masked."
        # 只在有效位置保留 KL 惩罚
        masked_kl = full_kl_response * full_mask
        # 逐 token 奖励 = -kl_coef × KL（KL 越大惩罚越强，负号转为惩罚项）
        token_level_rewards = -kl_coef * masked_kl
        # 找到最后一个有效 token 的下标
        last_idx = full_mask.nonzero(as_tuple=True)[0][-1]
        # 把整条序列的标量奖励加到最后一个 token 上（回合式奖励的常规做法）
        token_level_rewards[last_idx] += rewards[i]

        # 初始化该序列的逐 token 回报
        returns_for_seq = torch.zeros_like(token_level_rewards)
        # 反向累积的运行回报（从后往前折现）
        running_return = 0.0
        for t in reversed(range(token_level_rewards.size(0))):
            # 标准折扣回报递推：当前奖励加折扣后的后续累计回报
            # G_t = r_t + gamma * G_{t+1}
            running_return = token_level_rewards[t] + gamma * running_return
            # 记录位置 t 的回报
            returns_for_seq[t] = running_return

        # Step 4: Pick up the results corresponding to our local chunk's parts.
        if cp_size > 1:
            # CP 分片时：从完整回报中切回本 rank 负责的区段
            from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

            local_returns_chunk = slice_log_prob_with_cp(returns_for_seq, total_len, response_len)
        else:
            # 单卡：完整回报即本地 chunk
            local_returns_chunk = returns_for_seq

        # 收集该序列的本地回报 chunk
        final_returns_chunks.append(local_returns_chunk)

    # 返回所有序列的本地回报列表
    return final_returns_chunks


def get_reinforce_plus_plus_baseline_advantages(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    kl_coef: float,
) -> list[torch.Tensor]:
    """
    Calculates the unwhitened advantages for the REINFORCE++-baseline algorithm.
    Broadcasting the scalar (reward - group_baseline) to each token.

    Args:
        rewards (Tensor): A tensor of scalar rewards, where the group-wise
                                baseline has already been subtracted.
        kl (list[Tensor]): A list of per-token KL divergence tensors. Used to
                                 get the shape for broadcasting.
        loss_masks (list[Tensor]): A list of per-token loss masks.
        kl_coef (float): Coefficient for the KL penalty.

    Returns:
        list[Tensor]: A list of tensors containing the unwhitened advantages.
    """
    # Broadcast to get unwhitened advantages
    # 组基线已在 rewards 中扣除：把标量 (reward - baseline) 广播到每个 token，并减去 KL 惩罚项
    unwhitened_advantages = [
        torch.ones_like(kl_tensor) * reward_val - kl_coef * kl_tensor
        for kl_tensor, reward_val in zip(kl, rewards, strict=False)
    ]

    # 返回未做白化的逐 token 优势列表
    return unwhitened_advantages


def get_advantages_and_returns(
    total_len: int,
    response_len: int,
    values: torch.Tensor,
    rewards: torch.Tensor,
    gamma: float,
    lambd: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function that computes advantages and returns from rewards and values.
    Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
    Note that rewards may include a KL divergence loss term.

    Advantages looks like this:
    Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
            - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

    Returns looks like this:
    Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

    Input:
    - values: Tensor of shape (response_size,)
    - rewards: Tensor of shape (response_size,)

    Output:
    - advantages: Tensor of shape (response_size,)
    - returns: Tensor of shape (response_size,)
    """
    # 导入 Megatron 并行工具获取 CP 规模
    from megatron.core import mpu

    # 当前 CP 组大小
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size > 1:
        # CP 分片：先把本 rank 的奖励与价值 chunk 汇聚成完整序列
        from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

        full_rewards = all_gather_with_cp(rewards, total_len, response_len)
        full_values = all_gather_with_cp(values, total_len, response_len)
    else:
        # 单卡：直接使用本地张量
        full_rewards = rewards
        full_values = values

    # GAE 递推的中间变量（最后一步的累积优势）
    lastgaelam = 0
    # 逆序保存每步的优势值
    advantages_reversed = []

    # 从最后一个时间步向前递推 GAE（Generalized Advantage Estimation，广义优势估计）
    for t in reversed(range(response_len)):
        # 下一步价值：t 是最后一步时约定为 0（episode 结束）
        nextvalues = full_values[t + 1] if t < response_len - 1 else 0.0
        # TD 残差 δ_t = r_t + γ·V_{t+1} - V_t
        delta = full_rewards[t] + gamma * nextvalues - full_values[t]
        # GAE 递推：A_t = δ_t + γλ·A_{t+1}
        lastgaelam = delta + gamma * lambd * lastgaelam
        # 逆序暂存
        advantages_reversed.append(lastgaelam)
    # 反转回时间正序并构造张量（形状与 dtype/device 与 values 对齐）
    full_advantages = torch.tensor(advantages_reversed[::-1], dtype=full_values.dtype, device=full_values.device)
    # 回报 = 优势 + 价值基线
    full_returns = full_advantages + full_values

    if cp_size > 1:
        # CP 分片：把完整优势/回报切回本 rank 的区段
        from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

        advantages = slice_log_prob_with_cp(full_advantages, total_len, response_len)
        returns = slice_log_prob_with_cp(full_returns, total_len, response_len)
    else:
        # 单卡：直接使用完整结果
        advantages = full_advantages
        returns = full_returns

    # 优势 detach 切断梯度（作为 PG 的权重），回报保留原样
    return advantages.detach(), returns


def get_advantages_and_returns_batch(
    total_lengths,
    response_lengths,
    values_list,
    rewards_list,
    gamma,
    lambd,
    chunked: bool = True,
):
    """
    Batched GAE with CP support.
    Input:
        total_lengths:     list[int], each sample's total_len
        response_lengths:  list[int], each sample's response_len
        values_list:       list[Tensor], each shape = [resp_len_i]
        rewards_list:      list[Tensor], same shape
    Output:
        advantages_list:   list[Tensor], each shape = [resp_len_i]
        returns_list:      list[Tensor], same shape
    """

    # 导入 Megatron 并行工具
    from megatron.core import mpu

    # 整个 GAE 计算不需要梯度（优势是权重而非优化目标）
    with torch.no_grad():
        # 批大小 = 样本条数
        B = len(response_lengths)
        # 校验价值列表长度一致
        assert B == len(values_list)
        # 校验奖励列表长度一致
        assert B == len(rewards_list)

        # CP 组大小
        cp_size = mpu.get_context_parallel_world_size()
        # 统一设备与 dtype（以第一条样本为准）
        device = values_list[0].device
        dtype = values_list[0].dtype

        if cp_size > 1:
            # CP 分片：先逐样本 all-gather 出完整的价值/奖励序列
            from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

            full_values_list = []
            full_rewards_list = []

            for total_len, resp_len, v, r in zip(
                total_lengths, response_lengths, values_list, rewards_list, strict=False
            ):
                # 汇聚完整价值序列
                full_v = all_gather_with_cp(v, total_len, resp_len)
                # 汇聚完整奖励序列
                full_r = all_gather_with_cp(r, total_len, resp_len)
                full_values_list.append(full_v)
                full_rewards_list.append(full_r)

            # full_values_list[i].shape = [total_len_i]
        else:
            # 单卡：直接用输入列表
            full_values_list = values_list
            full_rewards_list = rewards_list

        # pad to max_len for batched GAE
        # 取响应长度最大值作为批量 pad 长度
        max_len = max(response_lengths)

        # 初始化批量 pad 后的价值/奖励矩阵（0 填充）
        full_values = torch.zeros(B, max_len, device=device, dtype=dtype)
        full_rewards = torch.zeros(B, max_len, device=device, dtype=dtype)

        # 把每条样本的有效部分拷入 pad 矩阵
        for i in range(B):
            # 该样本的实际响应长度
            L = response_lengths[i]
            # 拷贝价值序列有效段
            full_values[i, :L] = full_values_list[i][:L]
            # 拷贝奖励序列有效段
            full_rewards[i, :L] = full_rewards_list[i][:L]

        if not chunked:
            # 非分块模式：普通批量 GAE（逐时间步递推）
            full_advantages, full_returns = vanilla_gae(
                rewards=full_rewards,
                values=full_values,
                gamma=gamma,
                lambd=lambd,
            )
        else:
            # 分块模式：分块并行扫描 GAE（长序列更快）
            full_advantages, full_returns = chunked_gae(
                rewards=full_rewards,
                values=full_values,
                gamma=gamma,
                lambd=lambd,
            )

        # 收集每条样本的优势/回报
        advantages_list = []
        returns_list = []

        if cp_size > 1:
            # CP 分片：逐样本切回本 rank 区段
            from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

            for total_len, resp_len, adv_row, ret_row in zip(
                total_lengths,
                response_lengths,
                full_advantages,
                full_returns,
                strict=False,
            ):
                adv_full = adv_row  # shape = [resp_len_i padded to max_len]
                ret_full = ret_row

                # 先截掉 pad，再按 CP 切片取本 rank 区段
                adv_sliced = slice_log_prob_with_cp(adv_full[:resp_len], total_len, resp_len)
                ret_sliced = slice_log_prob_with_cp(ret_full[:resp_len], total_len, resp_len)

                advantages_list.append(adv_sliced)
                returns_list.append(ret_sliced)

        else:
            # 单卡：逐样本截掉 pad 取有效段
            for i in range(B):
                L = response_lengths[i]
                advantages_list.append(full_advantages[i, :L])
                returns_list.append(full_returns[i, :L])

    # 返回（优势列表, 回报列表）
    return advantages_list, returns_list


def vanilla_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambd: float,
):
    # 批量形状：B 条样本、最长 T 步
    B, T = rewards.shape
    # 设备与 dtype 保持一致
    device = rewards.device
    dtype = rewards.dtype

    # 每条样本独立的 GAE 累积变量
    lastgaelam = torch.zeros(B, device=device, dtype=dtype)
    # 逆序收集各时间步的优势
    adv_rev = []

    # 从后向前按时间步递推（batch 维并行）
    for t in reversed(range(T)):
        # 最后一步的"下一步价值"约定为 0
        next_value = values[:, t + 1] if t < T - 1 else 0.0
        # TD 残差 δ_t = r_t + γ·V_{t+1} - V_t（batch 向量化）
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        # GAE 递推：A_t = δ_t + γλ·A_{t+1}
        lastgaelam = delta + gamma * lambd * lastgaelam
        # 暂存该步
        adv_rev.append(lastgaelam)

    # 反转并堆叠成 [B, max_len] 的优势矩阵
    full_advantages = torch.stack(adv_rev[::-1], dim=1)  # [B, max_len]
    # 回报 = 优势 + 价值
    full_returns = full_advantages + values  # [B, max_len]
    # 返回（优势, 回报）
    return full_advantages, full_returns


def chunked_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambd: float,
    chunk_size: int = 128,
):
    """
    Compute Generalized Advantage Estimation (GAE) using a FlashLinearAttention-
    inspired algorithm: parallel prefix scan within chunks and recurrent state
    propagation across chunks.

    This reduces the sequential dependency length from O(T) to O(T / chunk_size),
    while keeping chunk computations fully parallelizable (O(C^2) per chunk).

    Args:
        rewards (Tensor): [B, T] reward sequence.
        values (Tensor):  [B, T] value predictions. The next-value of the final
                          step is assumed to be zero (standard PPO convention).
        gamma (float): discount factor.
        lam (float): GAE lambda.
        chunk_size (int): sequence chunk length for parallel scan.

    Returns:
        advantages (Tensor): [B, T] computed advantages.
        returns (Tensor):    [B, T] advantages + values.
    """

    # -------------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------------
    # 校验输入都是二维 [B, T]
    assert rewards.ndim == 2 and values.ndim == 2
    # 取批量与时间长度
    B, T = rewards.shape
    # 校验 rewards 与 values 形状一致
    assert values.shape == (B, T)

    # 设备与 dtype 保持一致
    device = rewards.device
    dtype = rewards.dtype

    # -------------------------------------------------------------------------
    # Build δ_t = r_t + γ * V_{t+1} - V_t   with V_{T} = 0
    # -------------------------------------------------------------------------
    # 构造"下一步价值"：整体左移一位，末位补 0（episode 结束约定）
    next_values = torch.cat(
        [values[:, 1:], torch.zeros(B, 1, device=device, dtype=dtype)],
        dim=1,
    )
    # 批量化计算 TD 残差 δ
    deltas = rewards + gamma * next_values - values

    # Reformulate backward GAE as a forward scan on the reversed sequence:
    #   S[i] = Δ[i] + w * S[i - 1],   w = γλ
    # 把反向 GAE 改写成反转序列上的前向线性扫描，w = γλ 为衰减系数
    w = gamma * lambd
    # 沿时间维反转 δ
    deltas_rev = torch.flip(deltas, dims=[1])  # [B, T]

    # -------------------------------------------------------------------------
    # Pad to a multiple of chunk_size
    # -------------------------------------------------------------------------
    # 时间长度不是 chunk 大小整数倍时补 0 对齐
    if T % chunk_size != 0:
        # 需要补的长度
        pad = chunk_size - (T % chunk_size)
        # 右侧补 0
        deltas_rev = F.pad(deltas_rev, (0, pad))
    else:
        # 整除则无需补
        pad = 0

    # 更新 padded 形状
    B, T_pad = deltas_rev.shape
    # chunk 数量
    n_chunks = T_pad // chunk_size

    # 重排成 [B, chunk 数, chunk 大小]
    deltas_chunks = deltas_rev.view(B, n_chunks, chunk_size)

    # -------------------------------------------------------------------------
    # Construct the intra-chunk parallel scan kernel M
    #
    # For a chunk Δ[0..C-1], we want:
    #   S_local[t] = sum_{k=0..t} w^(t-k) * Δ[k]
    #
    # This is implemented as:
    #   S_local = Δ @ M
    #
    # where:
    #   M[i, j] = w^(j - i)    if j >= i
    #             0            otherwise
    # -------------------------------------------------------------------------
    # 构造 0..C-1 的下标向量
    idx = torch.arange(chunk_size, device=device)
    # 行下标矩阵 [C, 1]
    row = idx[:, None]
    # 列下标矩阵 [1, C]
    col = idx[None, :]
    # 位置差 j - i
    diff = col - row

    # 初始化扫描核矩阵 M
    M = torch.zeros(chunk_size, chunk_size, device=device, dtype=dtype)
    # 上三角掩码：j >= i 的位置需要衰减权重
    mask = diff >= 0

    if w == 0.0:
        # w=0 退化为无衰减：只有对角线（j==i）权重为 1
        M[mask & (diff == 0)] = 1.0
    else:
        # 上三角填 w^(j-i)，0 的幂自动为 1
        M[mask] = w ** diff[mask].to(dtype)

    # pow_vec[t] = w^(t+1), used to inject the recurrent state s_prev
    # 预计算 w 的幂向量，用于把前一个 chunk 的累积状态注入当前 chunk
    if w == 0.0:
        # w=0 时历史状态不传播
        pow_vec = torch.zeros(chunk_size, device=device, dtype=dtype)
    else:
        # w^1, w^2, ..., w^C
        pow_vec = w ** torch.arange(1, chunk_size + 1, device=device, dtype=dtype)

    # -------------------------------------------------------------------------
    # Parallel compute local chunk results (assuming initial state = 0)
    # -------------------------------------------------------------------------
    # 展平成 [B*chunk 数, chunk 大小]，一次矩阵乘并行算所有 chunk
    deltas_flat = deltas_chunks.reshape(B * n_chunks, chunk_size)
    # 假设初始状态为 0 的 chunk 内局部扫描结果
    S_local_flat = deltas_flat @ M
    # 还原成 [B, chunk 数, chunk 大小]
    S_local_chunks = S_local_flat.view(B, n_chunks, chunk_size)

    # Effective length of each chunk (the last chunk may be padded)
    # 记录每个 chunk 的有效长度（最后一个可能被 pad 过）
    lengths = [chunk_size] * n_chunks
    if pad > 0:
        # 末尾 chunk 扣除 pad 长度
        lengths[-1] = chunk_size - pad

    # -------------------------------------------------------------------------
    # Recurrent propagation between chunks
    #
    # Each chunk contributes:
    #   S_global[t] = S_local[t] + w^(t+1) * s_prev
    #
    # And updates:
    #   s_prev = S_global[last_t]
    # -------------------------------------------------------------------------
    # 初始化全局扫描结果容器
    S_rev = deltas_rev.new_zeros(B, T_pad)
    # 跨 chunk 递推的状态（每条样本一个标量）
    s_prev = torch.zeros(B, device=device, dtype=dtype)

    # 逐 chunk 串行传播（chunk 之间仍有依赖，但长度已降为 T/chunk_size）
    for c in range(n_chunks):
        # 当前 chunk 的有效长度
        Lc = lengths[c]
        # 当前 chunk 在反转序列中的起始位置
        start = c * chunk_size
        # 结束位置（不含）
        end = start + Lc

        # 取当前 chunk 的局部扫描结果（截到有效长度）
        S_local = S_local_chunks[:, c, :Lc]
        # 加上前序状态的衰减贡献：w^(t+1)·s_prev
        S_global = S_local + s_prev.unsqueeze(1) * pow_vec[:Lc]

        # 写入全局结果
        S_rev[:, start:end] = S_global
        # 更新状态为当前 chunk 最后位置的全局值，供下一 chunk 使用
        s_prev = S_global[:, -1]  # state for next chunk

    # Remove padding and flip back to original time order
    # 有 pad 时先截掉
    if pad > 0:
        S_rev = S_rev[:, :T]

    # 翻转回原始时间顺序，得到逐 token 优势
    advantages = torch.flip(S_rev, dims=[1])
    # 回报 = 优势 + 价值
    returns = advantages + values

    # 返回（优势, 回报）
    return advantages, returns


def calculate_log_probs_and_entropy(logits, tokens, tp_group, with_entropy: bool = False, chunk_size: int = -1):
    # 保证 logits 内存连续，便于后续分块与 TP 计算
    logits = logits.contiguous()
    # TODO: not sure why we need to clone the logits here.
    # Without the clone, the backward will trigger inplace edit error.
    # It seems that the function with tp will modify the logits inplace.
    # 熵初始为 None（不计算时保持未定义）
    entropy = None
    if logits.size(0) != 0:
        if chunk_size > 0:
            # 开启分块：按 chunk 大小切分的块数（向上取整）
            num_chunks = (logits.size(0) - 1) // chunk_size + 1
            # token 序列分块
            tokens_chunks = tokens.chunk(num_chunks, dim=0)
            # logits 分块
            logits_chunks = logits.chunk(num_chunks, dim=0)
            # 收集各块的 log prob
            log_probs = []
            for tokens_chunk, logits_chunk in zip(tokens_chunks, logits_chunks, strict=True):
                # 逐块计算 log prob（clone 防止 TP 内核原地修改引发反向报错）
                log_prob = compute_log_probs(logits_chunk.clone(), tokens_chunk, tp_group)
                log_probs.append(log_prob)
            # 拼接所有块的 log prob
            log_prob = torch.cat(log_probs, dim=0)
            if with_entropy:
                # 需要熵时逐块再算一遍（熵与 log prob 的融合实现不通用）
                entropys = []
                for _, logits_chunk in zip(tokens_chunks, logits_chunks, strict=True):
                    # 逐块计算词表并行的熵
                    entropy = compute_entropy_from_logits(logits_chunk.clone(), tp_group)
                    entropys.append(entropy)
                # 拼接所有块的熵
                entropy = torch.cat(entropys, dim=0)
        else:
            # 不分块：一次性计算全部 log prob
            log_prob = compute_log_probs(logits.clone(), tokens, tp_group)
            if with_entropy:
                # 同时计算逐 token 熵
                entropy = compute_entropy_from_logits(logits.clone(), tp_group)
    else:
        # 空批（0 token）：返回空张量占位
        log_prob = logits.new_zeros((0,))
        if with_entropy:
            # 空批的熵占位
            entropy = logits.new_zeros((0,))

    # 返回（log prob, 熵）
    return log_prob, entropy
