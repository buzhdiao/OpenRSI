# 启用延迟类型注解求值，兼容旧版 Python 的新式注解写法
from __future__ import annotations

# 导入 torch 主库与自动求导机制
import torch
# 导入 Triton 语言模块（仅用于指定计算精度类型 tl.bfloat16）
import triton.language as tl
# 复用 sglang 提供的融合 MoE Triton 前向内核与辅助函数（分块对齐、求和归约、SiluAndMul）
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    invoke_fused_moe_kernel,
    moe_align_block_size,
    moe_sum_reduce,
    silu_and_mul,
)

# 导入配套的自定义反向传播内核
from .fused_moe_triton_backward_kernels import invoke_fused_moe_backward_kernel


# Gate/Up 融合投影算子（autograd.Function）：MoE 第一层 gate_proj+up_proj 的可反传融合实现
class GateUpProjFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        # token 数与隐藏维度
        num_tokens, _ = hidden_states.shape
        # w1 形状 [E, N, D_in]：专家数、融合输出维（2*中间维）、输入维
        E, N, _ = w1.shape
        # We execute the fused_moe kernel in chunks to circumvent this issue:
        # https://github.com/vllm-project/vllm/issues/5938
        # 分块执行融合内核以规避上游已知问题
        CHUNK_SIZE = 64 * 1024

        # default deterministic config
        # 默认的确定性（deterministic）Triton 分块配置：固定 block 大小保证结果可复现
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }

        # 每个 token 激活的专家数 top-k
        topk = topk_ids.shape[1]

        # 预分配中间缓存 1：形状 [num_tokens*topk, N]（每 token 每专家一行）
        intermediate_cache1 = torch.empty(
            (num_tokens * topk, N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        # 按 CHUNK_SIZE 分块遍历 token
        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx, end_chunk_idx = (
                chunk * CHUNK_SIZE,
                min((chunk + 1) * CHUNK_SIZE, num_tokens),
            )
            # 取当前块的输入与输出切片
            curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
            cur_intermediate_cache1 = intermediate_cache1[begin_chunk_idx * topk : end_chunk_idx * topk]

            # 取当前块的 top-k id 与权重
            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

            # 把 token 按专家排序并对齐到 BLOCK_SIZE_M（分块矩阵乘的预整理）
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], E
            )

            # 调用 sglang 融合 MoE 前向内核完成 gate/up 投影（无路由权重乘法，置 False）
            invoke_fused_moe_kernel(
                curr_hidden_states,
                w1,
                None,
                cur_intermediate_cache1,
                None,
                None,
                None,
                curr_topk_weights,
                curr_topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                False,
                topk_ids.shape[1],
                config,
                compute_type=tl.bfloat16,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=None,
                c_sorted=False,
                filter_expert=True,
            )

        # 保存反向传播所需的张量引用
        ctx.save_for_backward(hidden_states, w1, topk_weights, topk_ids)
        # 记录配置与维度信息到 ctx 供 backward 使用
        ctx.config = config
        ctx.num_tokens = num_tokens
        ctx.topk = topk

        # 返回中间缓存 1
        return intermediate_cache1

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for GateUpProjFunction using Triton kernels.

        Args:
            grad_output: shape (num_tokens * topk, N)

        Returns:
            (grad_hidden_states, grad_w1, grad_topk_weights, None)
        """

        # 取出前向保存的张量与元信息
        hidden_states, w1, topk_weights, topk_ids = ctx.saved_tensors
        config = ctx.config
        num_tokens = ctx.num_tokens
        topk = ctx.topk

        # 从 w1 形状解出专家数、输出维、输入维
        E, N, D_in = w1.shape
        CHUNK_SIZE = 64 * 1024

        # Initialize gradient tensors
        # 初始化各梯度缓冲区
        grad_hidden_states = torch.zeros_like(hidden_states)
        grad_w1 = torch.zeros_like(w1)
        # GateUpProj stage doesn't need topk_weights gradient
        # 该阶段不使用路由权重，故 topk_weights 的梯度恒为零
        grad_topk_weights = torch.zeros_like(topk_weights)

        # Process in chunks to match forward pass
        # 与前向一致地分块处理
        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx, end_chunk_idx = (
                chunk * CHUNK_SIZE,
                min((chunk + 1) * CHUNK_SIZE, num_tokens),
            )

            # 当前块 token 数
            curr_num_tokens = end_chunk_idx - begin_chunk_idx
            if curr_num_tokens == 0:
                # 空块直接跳过
                continue

            # 切出当前块的各输入与上游梯度
            curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
            curr_grad_output = grad_output[begin_chunk_idx * topk : end_chunk_idx * topk]

            # Get aligned metadata
            # 重新计算对齐元数据（与该块前向时一致）
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], E
            )

            # Prepare gradient buffer for this chunk
            # 为当前块准备梯度缓冲区
            curr_grad_hidden_states = torch.zeros_like(curr_hidden_states)
            curr_grad_w1 = torch.zeros_like(w1)

            # Call Triton backward kernel with MUL_ROUTED_WEIGHT=False
            # Use chunk of hidden_states to match sorted_token_ids indices
            # 调用自定义 Triton 反向内核：mul_routed_weight=False 对应前向未乘路由权重
            invoke_fused_moe_backward_kernel(
                grad_output=curr_grad_output,
                input=curr_hidden_states,  # Use chunk of hidden_states to match sorted_token_ids
                weight=w1,
                grad_input=curr_grad_hidden_states,
                grad_weight=curr_grad_w1,
                grad_topk_weights=None,  # Not needed for GateUpProj
                topk_weights=curr_topk_weights,
                topk_ids=curr_topk_ids,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=False,
                top_k=topk,
                config=config,
                compute_type=tl.bfloat16,
            )

            # Accumulate gradients
            # 累加当前块的梯度到全局缓冲区
            grad_hidden_states[begin_chunk_idx:end_chunk_idx] += curr_grad_hidden_states
            grad_w1 += curr_grad_w1

        # 返回四路梯度：hidden_states、w1、topk_weights、topk_ids(None)
        return grad_hidden_states, grad_w1, grad_topk_weights, None


# SiluAndMul 融合激活算子：SwiGLU 风格 silu(x1)*x2 的可反传实现
class SiluAndMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, intermediate_cache1: torch.Tensor):
        # 输入形状 [num_tokens, N]（N 为 2 倍中间维）
        num_tokens, N = intermediate_cache1.shape
        # 预分配输出：每行取前半 silu 后乘以后半，维度减半
        intermediate_cache2 = torch.empty(
            (num_tokens, N // 2),
            device=intermediate_cache1.device,
            dtype=intermediate_cache1.dtype,
        )
        # 调用 sglang 融合激活内核
        silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)

        # 保存原输入供反向使用
        ctx.save_for_backward(intermediate_cache1)
        # 返回激活输出
        return intermediate_cache2

    @staticmethod
    def backward(ctx, grad_output):
        # 取回保存的输入
        (intermediate_cache1,) = ctx.saved_tensors
        # 输入最后一维大小
        N = intermediate_cache1.shape[-1]
        # 沿最后一维切成两半：x1 走 silu，x2 为乘法门
        x1, x2 = intermediate_cache1.view(-1, N).chunk(2, dim=-1)
        # 预计算 silu(x1) 供两路梯度使用
        silu_x1 = torch.nn.functional.silu(x1)

        # 推导 d silu/dx = sig + x*sig*(1-sig)（sigmoid 的导数链式展开）
        sig = torch.sigmoid(x1)
        dsilu_dx1 = sig + x1 * sig * (1 - sig)
        # 对 x1 的梯度：上游梯度 * x2 * silu 导数
        grad_x1 = grad_output * x2 * dsilu_dx1
        # 对 x2 的梯度：上游梯度 * silu(x1)
        grad_x2 = grad_output * silu_x1
        # 拼回两路梯度
        grad_input = torch.cat([grad_x1, grad_x2], dim=-1)

        # 恢复成与输入相同的形状返回
        return grad_input.view_as(intermediate_cache1)


# Down 投影算子（autograd.Function）：MoE 第二层 down_proj 的可反传融合实现
class DownProjFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        intermediate_cache2: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        # 输入为 [num_tokens*topk, I]（激活后中间态）
        num_tokens, _ = intermediate_cache2.shape
        # top-k 数
        topk = topk_ids.shape[1]
        # 还原真实 token 数
        num_tokens //= topk
        # 专家数
        E, _, _ = w2.shape
        # We execute the fused_moe kernel in chunks to circumvent this issue:
        # https://github.com/vllm-project/vllm/issues/5938
        # 与前一层一致：分块执行规避上游问题
        CHUNK_SIZE = 64 * 1024

        # default deterministic config
        # 默认确定性分块配置
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }

        # 预分配中间缓存 3：每 token 每专家一行输出
        intermediate_cache3 = torch.empty(
            (num_tokens, topk, w2.shape[1]),
            device=intermediate_cache2.device,
            dtype=intermediate_cache2.dtype,
        )

        # 分块遍历 token
        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx, end_chunk_idx = (
                chunk * CHUNK_SIZE,
                min((chunk + 1) * CHUNK_SIZE, num_tokens),
            )
            # 切出当前块的输入（含 topk 展开维）与输出
            cur_intermediate_cache2 = intermediate_cache2[begin_chunk_idx * topk : end_chunk_idx * topk]
            cur_intermediate_cache3 = intermediate_cache3[begin_chunk_idx:end_chunk_idx]

            # 切出当前块的路由信息
            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

            # 按专家对齐 token（分块矩阵乘预整理）
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], E
            )
            # 调用融合 MoE 前向内核：第二参数 True 表示乘路由权重（down 阶段应用 routing weight）
            invoke_fused_moe_kernel(
                cur_intermediate_cache2,
                w2,
                None,
                cur_intermediate_cache3,
                None,
                None,
                None,
                curr_topk_weights,
                curr_topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                True,
                1,
                config,
                compute_type=tl.bfloat16,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=None,
                a_use_tma=False,
                b_use_tma=False,
            )

        # 保存反向所需张量与元信息
        ctx.save_for_backward(intermediate_cache2, w2, topk_weights, topk_ids)
        ctx.config = config
        ctx.num_tokens = num_tokens
        ctx.topk = topk

        # 返回中间缓存 3
        return intermediate_cache3

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for DownProjFunction using Triton kernels.

        Args:
            grad_output: shape (num_tokens, topk, hidden_size)

        Returns:
            (grad_intermediate_cache2, grad_w2, grad_topk_weights, None)
        """
        # 取出前向保存的张量与元信息
        intermediate_cache2, w2, topk_weights, topk_ids = ctx.saved_tensors
        config = ctx.config
        num_tokens = ctx.num_tokens
        topk = ctx.topk

        # 从 w2 形状解出专家数、隐藏维、中间维
        E, hidden_size, intermediate_size = w2.shape
        CHUNK_SIZE = 64 * 1024

        # Initialize gradient tensors
        # 初始化各梯度缓冲区（此阶段需要 topk_weights 的梯度）
        grad_intermediate_cache2 = torch.zeros_like(intermediate_cache2)
        grad_w2 = torch.zeros_like(w2)
        grad_topk_weights = torch.zeros_like(topk_weights)

        # Process in chunks to match forward pass
        # 与前向一致地分块处理
        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx, end_chunk_idx = (
                chunk * CHUNK_SIZE,
                min((chunk + 1) * CHUNK_SIZE, num_tokens),
            )

            # 当前块 token 数，空块跳过
            curr_num_tokens = end_chunk_idx - begin_chunk_idx
            if curr_num_tokens == 0:
                continue

            # 切出当前块各输入与上游梯度
            curr_intermediate_cache2 = intermediate_cache2[begin_chunk_idx * topk : end_chunk_idx * topk]
            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
            curr_grad_output = grad_output[begin_chunk_idx:end_chunk_idx]

            # Get aligned metadata
            # 重新计算对齐元数据
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], E
            )

            # Prepare gradient buffers for this chunk
            # 为当前块准备梯度缓冲区
            curr_grad_intermediate_cache2 = torch.zeros_like(curr_intermediate_cache2)
            curr_grad_w2 = torch.zeros_like(w2)
            curr_grad_topk_weights = torch.zeros_like(curr_topk_weights)

            # Call Triton backward kernel with MUL_ROUTED_WEIGHT=True
            # Note: Use top_k=1 to match forward pass indexing
            # 调用反向内核：mul_routed_weight=True 对应前向乘了路由权重；top_k=1 与前向索引方式匹配
            invoke_fused_moe_backward_kernel(
                grad_output=curr_grad_output,
                input=curr_intermediate_cache2,
                weight=w2,
                grad_input=curr_grad_intermediate_cache2,
                grad_weight=curr_grad_w2,
                grad_topk_weights=curr_grad_topk_weights,
                topk_weights=curr_topk_weights,
                topk_ids=curr_topk_ids,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=True,
                top_k=1,
                config=config,
                compute_type=tl.bfloat16,
            )

            # Accumulate gradients
            # 累加/写回当前块梯度
            grad_intermediate_cache2[begin_chunk_idx * topk : end_chunk_idx * topk] = curr_grad_intermediate_cache2
            grad_w2 += curr_grad_w2
            grad_topk_weights[begin_chunk_idx:end_chunk_idx] = curr_grad_topk_weights

        # 返回四路梯度
        return grad_intermediate_cache2, grad_w2, grad_topk_weights, None


# MoE 求和归约算子：把 [tokens, topk, H] 的多专家输出沿 topk 维求和为 [tokens, H]
class MoeSumReduceFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        intermediate_cache3: torch.Tensor,
        hidden_states_shape,
    ):
        # 按原隐藏态形状预分配输出
        out_hidden_states = torch.empty(
            hidden_states_shape, device=intermediate_cache3.device, dtype=intermediate_cache3.dtype
        )
        # 调用 sglang 的求和归约内核（缩放系数 1.0）
        moe_sum_reduce(
            intermediate_cache3,
            out_hidden_states,
            1.0,
        )
        # 保存输入供反向使用
        ctx.save_for_backward(intermediate_cache3)
        # 返回归约结果
        return out_hidden_states

    @staticmethod
    def backward(ctx, grad_output):
        # 取回保存的输入
        (intermediate_cache3,) = ctx.saved_tensors
        # 求和算子的反向即广播：把 [tokens, H] 梯度沿 topk 维复制展开
        return grad_output.unsqueeze(1).expand_as(intermediate_cache3), None
