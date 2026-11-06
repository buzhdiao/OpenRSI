# 启用延迟类型注解求值，兼容旧版 Python 的新式注解写法
from __future__ import annotations

# 导入 Any 用于类型注解
from typing import Any

# 导入 torch（仅在 host 侧调用内核时使用）
import torch
# 导入 Triton JIT 编译装饰器
import triton
# 导入 Triton 语言模块（tl.load/tl.dot 等设备端原语）
import triton.language as tl


# 反向内核一：计算 dL/dInput（对输入的梯度）
# 前向语义：output = input @ weight.T（可选乘 topk_weights）；反向：grad_input = grad_output @ weight
@triton.jit
def fused_moe_backward_input_kernel(
    # Pointers to matrices
    # 各矩阵的指针参数：上游梯度、专家权重、输入梯度、路由权重梯度（占位）、路由权重
    grad_output_ptr,
    weight_ptr,
    grad_input_ptr,
    grad_topk_weights_ptr,
    topk_weights_ptr,
    # 按专家排序后的 token id、每块的专家 id、padding 后 token 总数
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    # 矩阵维度：N（输出维）、K（输入维）、EM（padding 后的 M）、有效 token 数
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides
    # 各张量沿各维的步长（用于指针偏移计算）
    stride_gom,
    stride_gon,
    stride_we,
    stride_wn,
    stride_wk,
    stride_gim,
    stride_gik,
    # Meta-parameters
    # 编译期元参数：分块大小、M 维分组、是否乘路由权重、top-k、写出精度
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """
    Backward kernel for computing grad_input.

    Forward: output = input @ weight.T (optionally multiplied by topk_weights)
    Backward: grad_input = grad_output @ weight (optionally multiplied by topk_weights)

    This kernel computes: grad_input[token] = sum_over_N(grad_output[token, n] * weight[expert, n, :])
    If MUL_ROUTED_WEIGHT: grad_input[token] *= topk_weights[token]

    Parallelization: Similar to forward, parallel over M and N dimensions, loop over K.
    """
    # Map program ids to blocks (parallel over M and N, similar to forward)
    # 把一维 program id 映射到 (M 块, N 块) 二维网格，并做 M 维分组以改善 L2 缓存局部性
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Check bounds
    # 读出 padding 后的真实 token 总数
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    # Only process if this block is valid
    # 仅当 M 块起始位置落在有效范围内才处理
    if pid_m * BLOCK_SIZE_M < num_tokens_post_padded:
        # Load token information
        # 取本 M 块内各槽位的排序 token id 及有效掩码（padding 槽位 id 超界）
        offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
        offs_token = offs_token.to(tl.int64)
        token_mask = offs_token < num_valid_tokens

        # Get expert ID for this block
        # 取本 M 块对应的专家 id
        off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

        # Only process if expert is valid
        # 专家 id 为 -1 表示空块，跳过
        if off_experts != -1:
            # Initialize offsets for N dimension (current block)
            # 初始化本块的 N 维偏移与 K 维基础偏移
            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
            offs_k = tl.arange(0, BLOCK_SIZE_K)

            # Load grad_output block: shape (BLOCK_SIZE_M, BLOCK_SIZE_N)
            # 加载上游梯度块：按排序 token id 寻址（grad_output 以展开的 token*topk 维组织）
            grad_output_ptrs = grad_output_ptr + (offs_token[:, None] * stride_gom + offs_n[None, :] * stride_gon)
            grad_out = tl.load(
                grad_output_ptrs,
                mask=token_mask[:, None] & (offs_n[None, :] < N),
                other=0.0,
            )

            # Apply topk_weights to grad_output if needed
            if MUL_ROUTED_WEIGHT:
                # 需要时把路由权重乘进上游梯度（对应前向乘权的链式法则）
                moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
                grad_out = grad_out * moe_weight[:, None]

            # Iterate over K dimension
            # 沿 K 维循环累加（K 超过 block 时分块）
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                # Current K offsets
                # 当前 K 分块的偏移
                curr_offs_k = k * BLOCK_SIZE_K + offs_k

                # Load weight block: shape (BLOCK_SIZE_N, BLOCK_SIZE_K)
                # weight: shape (E, N, K)
                # 加载该专家的权重块 [N, K]
                weight_ptrs = (
                    weight_ptr
                    + off_experts * stride_we
                    + offs_n[:, None] * stride_wn
                    + curr_offs_k[None, :] * stride_wk
                )
                w = tl.load(
                    weight_ptrs,
                    mask=(offs_n[:, None] < N) & (curr_offs_k[None, :] < K),
                    other=0.0,
                )

                # Compute contribution: grad_out @ weight
                # grad_out: (BLOCK_SIZE_M, BLOCK_SIZE_N)
                # w: (BLOCK_SIZE_N, BLOCK_SIZE_K)
                # result: (BLOCK_SIZE_M, BLOCK_SIZE_K)
                # 张量核乘：grad_out(N 维) x weight(N->K)，得到对输入 K 维的本块贡献
                contribution = tl.dot(grad_out, w)

                # Atomic add to grad_input because different N blocks contribute to same K
                # 不同的 N 块会对同一 K 位置贡献梯度，故需原子加；token 索引除以 top_k 折回原始 token 维
                grad_input_ptrs = grad_input_ptr + (
                    (offs_token[:, None] // top_k) * stride_gim + curr_offs_k[None, :] * stride_gik
                )
                grad_input_mask = token_mask[:, None] & (curr_offs_k[None, :] < K)
                tl.atomic_add(grad_input_ptrs, contribution.to(compute_type), mask=grad_input_mask)


# 反向内核二：计算 dL/dWeight（对专家权重的梯度）
# 前向语义：output = input @ weight.T；反向：grad_weight = input.T @ grad_output
@triton.jit
def fused_moe_backward_weight_kernel(
    # Pointers to matrices
    # 各矩阵指针：上游梯度、输入、权重梯度缓冲、路由权重
    grad_output_ptr,
    input_ptr,
    grad_weight_ptr,
    topk_weights_ptr,
    # 排序 token id、每块专家 id、padding 后 token 数
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    # 维度：N（输出维）、K（输入维）、EM、有效 token 数
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides
    # 各张量步长
    stride_gom,
    stride_gon,
    stride_im,
    stride_ik,
    stride_gwe,
    stride_gwn,
    stride_gwk,
    # Meta-parameters
    # 编译期元参数
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """
    Backward kernel for computing grad_weight.

    Forward: output = input @ weight.T (optionally multiplied by topk_weights)
    Backward: grad_weight = input.T @ grad_output (optionally multiplied by topk_weights)

    This kernel computes: grad_weight[expert, n, k] = sum_over_tokens(input[token, k] * grad_output[token, n])
    If MUL_ROUTED_WEIGHT: the accumulation is weighted by topk_weights[token]

    Parallelization: Parallel over M and N dimensions with grouping, loop over K.
    """
    # Map program ids to blocks (parallel over M and N with grouping, similar to forward and backward_input)
    # 同样把一维 program id 映射到 (M 块, N 块) 并做 M 维分组
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Check bounds
    # 读出 padding 后 token 总数
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    # Only process if this block is valid
    # 无效 M 块直接返回
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # Get expert ID for this M block
    # 取本 M 块的专家 id
    expert_id = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    # Only process if expert is valid
    # 空专家块直接返回
    if expert_id == -1:
        return

    # Load token information for this M block
    # 取本 M 块的排序 token id 与掩码；越界槽位回填为 num_valid_tokens 便于统一掩码
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_token_id = pid_m * BLOCK_SIZE_M + offs_m.to(tl.int64)
    offs_token = tl.load(
        sorted_token_ids_ptr + offs_token_id, mask=offs_token_id < num_tokens_post_padded, other=num_valid_tokens
    )
    offs_token = offs_token.to(tl.int64)
    token_mask = (offs_token_id < num_tokens_post_padded) & (offs_token < num_valid_tokens)

    # Clamp offs_token to valid range
    # 把越界 token id 截到 0，保证指针运算安全（结果被掩码屏蔽）
    offs_token_clamped = tl.where(token_mask, offs_token, 0)

    # Determine input token indices based on MUL_ROUTED_WEIGHT
    # 根据 MUL_ROUTED_WEIGHT 决定 input 的寻址方式：input 形状可能是 [num_tokens*topk, K] 或 [num_tokens, K]
    if MUL_ROUTED_WEIGHT:
        # 乘路由权重的场景（down 阶段，top_k=1）：input 与 grad_output 一一对应
        input_token_idx = offs_token_clamped
        input_mask = token_mask
    else:
        # 不乘路由权重的场景（gate/up 阶段）：input 按原始 token 索引，需除以 top_k 折回
        input_token_idx = offs_token_clamped // top_k
        num_input_tokens = num_valid_tokens // top_k
        input_mask = token_mask & (input_token_idx < num_input_tokens)

    # Load topk_weights if needed
    if MUL_ROUTED_WEIGHT:
        # 需要时加载各 token 的路由权重
        moe_weight = tl.load(topk_weights_ptr + offs_token_clamped, mask=token_mask, other=0.0)

    # Current N offset for this program
    # 本 program 负责的 N 维偏移
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)

    # Load grad_output for this N block: shape (M, BLOCK_SIZE_N)
    # grad_output is always indexed by sorted_token_ids (offs_token_clamped)
    # because it has shape (num_tokens * topk, N)
    # 加载本 N 块的上游梯度（grad_output 总是按排序 token id 寻址，因为形状是展开的 [tokens*topk, N]）
    grad_output_ptrs = grad_output_ptr + (offs_token_clamped[:, None] * stride_gom + offs_n[None, :] * stride_gon)
    grad_out = tl.load(
        grad_output_ptrs,
        mask=token_mask[:, None] & (offs_n[None, :] < N),
        other=0.0,
    )

    # Apply topk_weights if needed
    if MUL_ROUTED_WEIGHT:
        # 把路由权重乘进上游梯度
        grad_out = grad_out * moe_weight[:, None]

    # Zero out padding tokens
    # padding 槽位的梯度清零（掩码外的 other=0.0 已处理，这里再显式乘掩码兜底）
    token_mask_col = token_mask[:, None]
    grad_out = grad_out * token_mask_col

    # Iterate over K blocks and accumulate
    # 沿 K 维分块循环，把对权重的梯度逐块原子累加
    for k_block in range(tl.cdiv(K, BLOCK_SIZE_K)):
        offs_k = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

        # Load input for this K block
        # 加载本 K 块的输入
        input_ptrs = input_ptr + (input_token_idx[:, None] * stride_im + offs_k[None, :] * stride_ik)
        inp = tl.load(
            input_ptrs,
            mask=input_mask[:, None] & (offs_k[None, :] < K),
            other=0.0,
        )

        # Zero out padding tokens - use input_mask for input, token_mask for grad_output
        # padding 槽位的输入清零
        input_mask_col = input_mask[:, None]
        inp = inp * input_mask_col

        # Compute grad_weight contribution: grad_out.T @ inp
        # 权重梯度贡献 = grad_out^T(M,N) @ inp(M,K) -> (N,K)
        grad_w_contribution = tl.dot(grad_out.T, inp)

        # Write back using atomic add
        # 原子加写回：不同 M 块会累加到同一专家的同一 (n,k) 位置
        grad_weight_ptrs = (
            grad_weight_ptr + expert_id * stride_gwe + offs_n[:, None] * stride_gwn + offs_k[None, :] * stride_gwk
        )
        grad_weight_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        tl.atomic_add(grad_weight_ptrs, grad_w_contribution.to(compute_type), mask=grad_weight_mask)


# 反向内核三：计算 dL/dTopkWeights（对路由权重的梯度）
# 前向语义：output = topk_weights * (input @ weight.T)；反向：grad_topk_weights = sum(grad_output * 前向未乘权输出)
@triton.jit
def fused_moe_backward_topk_weights_kernel(
    # Pointers to matrices
    # 各矩阵指针：上游梯度、输入、专家权重、路由权重梯度输出
    grad_output_ptr,
    input_ptr,
    weight_ptr,
    grad_topk_weights_ptr,
    # 排序 token id、每块专家 id、padding 后 token 数
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    # 维度：N、K、EM、有效 token 数
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides
    # 各张量步长
    stride_gom,
    stride_gon,
    stride_im,
    stride_ik,
    stride_we,
    stride_wn,
    stride_wk,
    # Meta-parameters
    # 编译期元参数（此内核无需 GROUP_SIZE_M）
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """
    Backward kernel for computing grad_topk_weights.

    Forward: output = topk_weights * (input @ weight.T)
    Backward: grad_topk_weights = sum(grad_output * (input @ weight.T))

    This kernel computes the gradient of topk_weights by computing the dot product
    of grad_output with the forward output before weight multiplication.
    """
    # Map program id to token block
    # 一维 program id 直接映射到 token 块（只并行 M 维）
    pid = tl.program_id(axis=0)

    # Check bounds
    # 读出 padding 后 token 总数
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    # Only process if this block is valid
    # 无效块跳过
    if pid * BLOCK_SIZE_M < num_tokens_post_padded:
        # Load token information
        # 取本块排序 token id 与掩码
        offs_token_id = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_token = tl.load(
            sorted_token_ids_ptr + offs_token_id, mask=offs_token_id < num_tokens_post_padded, other=num_valid_tokens
        )
        offs_token = offs_token.to(tl.int64)
        token_mask = (offs_token_id < num_tokens_post_padded) & (offs_token < num_valid_tokens)

        # Clamp offs_token to valid range for safe pointer arithmetic
        # 越界 token id 截到 0 保证指针安全
        offs_token_clamped = tl.where(token_mask, offs_token, 0)

        # Get expert ID for this block
        # 取本块专家 id
        off_experts = tl.load(expert_ids_ptr + pid).to(tl.int64)

        # Only process if expert is valid
        # 空专家块跳过
        if off_experts != -1:
            # Initialize offsets
            # 初始化 N/K 维基础偏移
            offs_n = tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)

            # Accumulator for grad_topk_weights
            # 每行 token 的路由权重梯度累加器（fp32 累加保证精度）
            accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

            # Iterate over N and K dimensions to compute forward output and gradient
            # 外层遍历 N 块：先重算前向未乘权输出，再与上游梯度做内积
            for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
                # Current N offset
                # 当前 N 块偏移
                curr_offs_n = n * BLOCK_SIZE_N + offs_n

                # Load grad_output block: (M, N)
                # 加载本 N 块的上游梯度
                grad_output_ptrs = grad_output_ptr + (
                    offs_token_clamped[:, None] * stride_gom + curr_offs_n[None, :] * stride_gon
                )
                grad_out = tl.load(
                    grad_output_ptrs,
                    mask=token_mask[:, None] & (curr_offs_n[None, :] < N),
                    other=0.0,
                )

                # Compute forward output for this N block: input @ weight[:, n, :].T
                # 为本 N 块重算前向输出（未乘路由权重），fp32 累加
                forward_output_n = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

                # 内层遍历 K 块做矩阵乘累加
                for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                    # Current K offset
                    # 当前 K 块偏移
                    curr_offs_k = k * BLOCK_SIZE_K + offs_k

                    # Load input block: (M, K)
                    # 加载输入块（按原始 token 索引，除以 top_k 折回）
                    input_ptrs = input_ptr + (
                        (offs_token_clamped[:, None] // top_k) * stride_im + curr_offs_k[None, :] * stride_ik
                    )
                    inp = tl.load(
                        input_ptrs,
                        mask=token_mask[:, None] & (curr_offs_k[None, :] < K),
                        other=0.0,
                    )

                    # Load weight block: (N, K)
                    # 加载该专家的本 (N,K) 权重块
                    weight_ptrs = (
                        weight_ptr
                        + off_experts * stride_we
                        + curr_offs_n[:, None] * stride_wn
                        + curr_offs_k[None, :] * stride_wk
                    )
                    w = tl.load(
                        weight_ptrs,
                        mask=(curr_offs_n[:, None] < N) & (curr_offs_k[None, :] < K),
                        other=0.0,
                    )

                    # Accumulate forward output: input @ weight.T
                    # inp: (M, K), w.T: (K, N) -> (M, N)
                    # 累加前向输出块
                    forward_output_n += tl.dot(inp, w.T)

                # Compute contribution to grad_topk_weights: sum(grad_out * forward_output)
                # Sum over N dimension
                # 对 N 维求和：grad_out 与重算的前向输出逐元素积再行内求和，得到本块的权重梯度贡献
                accumulator += tl.sum(grad_out * forward_output_n, axis=1)

            # Write back grad_topk_weights using atomic add with clamped token indices
            # 原子加写回各 token 的路由权重梯度
            tl.atomic_add(grad_topk_weights_ptr + offs_token_clamped, accumulator.to(compute_type), mask=token_mask)


# 对外封装：按需依次调用上述三个反向内核，完成一次 MoE 反向传播
def invoke_fused_moe_backward_kernel(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    grad_input: torch.Tensor,
    grad_weight: torch.Tensor,
    grad_topk_weights: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
) -> None:
    """
    Invoke the fused MOE backward kernels to compute gradients.

    Args:
        grad_output: Gradient of output, shape (num_tokens * topk, N) or (num_tokens, topk, N)
        input: Input tensor, shape (num_tokens, K)
        weight: Weight tensor, shape (E, N, K)
        grad_input: Output gradient for input, shape (num_tokens, K)
        grad_weight: Output gradient for weight, shape (E, N, K)
        grad_topk_weights: Output gradient for topk_weights, shape (num_tokens, topk) or None
        topk_weights: Top-K routing weights, shape (num_tokens, topk)
        topk_ids: Top-K expert IDs, shape (num_tokens, topk)
        sorted_token_ids: Sorted token IDs
        expert_ids: Expert IDs for each block
        num_tokens_post_padded: Number of tokens after padding
        mul_routed_weight: Whether to multiply by routing weights
        top_k: Number of experts per token
        config: Kernel configuration
        compute_type: Computation data type
    """
    # 内核要求 topk_weights 与 sorted_token_ids 的最后一维连续
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    # Flatten grad_output if needed
    # Before: (num_tokens, topk, hidden_size)
    # After: (num_tokens * topk, hidden_size)
    # 三维的 grad_output 摊平为二维，便于统一按展开 token 寻址
    if grad_output.ndim == 3:
        grad_output = grad_output.reshape(-1, grad_output.shape[-1])

    # 从权重形状解出专家数、输出维 N、输入维 K
    E, N, K = weight.shape

    # ===================== Compute grad_input =====================
    # 网格：M 块数 x N 块数（一维展开）
    def grid_input(META):
        return (triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    # 启动 grad_input 反向内核
    fused_moe_backward_input_kernel[grid_input](
        grad_output,
        weight,
        grad_input,
        grad_topk_weights if grad_topk_weights is not None else grad_input,  # dummy pointer
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        sorted_token_ids.shape[0],
        grad_output.shape[0],
        grad_output.stride(0),
        grad_output.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        grad_input.stride(0),
        grad_input.stride(1),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        **config,
    )

    # ===================== Compute grad_weight =====================
    # Initialize grad_weight to zero
    # 权重梯度缓冲先清零（内核内是原子累加）
    grad_weight.zero_()

    # Use same grid configuration as forward kernel: encode both M and N dimensions
    # 与前向内核一致的网格配置：M 块数 x N 块数
    def grid_weight(META):
        return (triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    # 启动 grad_weight 反向内核
    fused_moe_backward_weight_kernel[grid_weight](
        grad_output,
        input,
        grad_weight,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        sorted_token_ids.shape[0],
        grad_output.shape[0],
        grad_output.stride(0),
        grad_output.stride(1),
        input.stride(0),
        input.stride(1),
        grad_weight.stride(0),
        grad_weight.stride(1),
        grad_weight.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        **config,
    )

    # ===================== Compute grad_topk_weights (if needed) =====================
    # 仅当前向乘了路由权重且调用方需要该梯度时才计算
    if mul_routed_weight and grad_topk_weights is not None:

        # 该内核只并行 M 维，网格为 M 块数
        def grid_topk(META):
            return (triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"]),)

        # 启动 grad_topk_weights 反向内核
        fused_moe_backward_topk_weights_kernel[grid_topk](
            grad_output,
            input,
            weight,
            grad_topk_weights.view(-1),
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            N,
            K,
            sorted_token_ids.shape[0],
            grad_output.shape[0],
            grad_output.stride(0),
            grad_output.stride(1),
            input.stride(0),
            input.stride(1),
            weight.stride(0),
            weight.stride(1),
            weight.stride(2),
            top_k=top_k,
            compute_type=compute_type,
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        )
