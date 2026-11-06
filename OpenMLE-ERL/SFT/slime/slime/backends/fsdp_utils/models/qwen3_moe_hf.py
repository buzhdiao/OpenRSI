# 导入 torch 主库
import torch
# 导入函数式接口 F（softmax 等算子）
import torch.nn.functional as F


# 给 HF transformers 的 Qwen3-MoE 稀疏 MoE 模块打补丁，替换其 forward 实现
def apply_fsdp_moe_patch():

    # 延迟导入，避免模块加载时强依赖 transformers
    from transformers.models.qwen3_moe import modeling_qwen3_moe

    # 替换后的 MoE 前向：hidden_states 形状为 [batch, seq, hidden]
    def _forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 解包批次、序列长度与隐藏维度
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # 摊平为 [batch*seq, hidden] 的二维张量便于逐 token 计算
        hidden_states = hidden_states.view(-1, hidden_dim)
        # 路由门控（gate）输出每个专家的 logits
        router_logits = self.gate(hidden_states)

        # 对路由 logits 做 softmax 得到路由权重（以 fp32 计算保证数值稳定）
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        # 取权重最大的 top_k 个专家（稀疏激活，MoE 的 Top-K 路由）
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            # 归一化 top-k 权重使其和为 1（Qwen3-MoE 可选）
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # 权重转回输入精度类型
        routing_weights = routing_weights.to(hidden_states.dtype)

        # 初始化最终输出：逐 token 聚合各专家的加权输出
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # 构造 one-hot 掩码 [num_experts, top_k, tokens]，标记每个专家接收哪些 token
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all experts
        # 逐专家循环处理（朴素实现：非分组 GEMM，便于 FSDP 下兼容）
        for expert_idx in range(self.num_experts):
            # 取该专家的 MLP 层
            expert_layer = self.experts[expert_idx]
            # 找到分派给该专家的 token：idx 是 top-k 槽位，top_x 是 token 索引
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.numel() > 0:
                # 收集该专家负责的 token 输入
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                # 专家前向并乘上对应路由权重，得到加权输出
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
                # 用 index_add_ 把输出按 token 索引累加回结果（一个 token 可能命中多个专家）
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            else:
                # force experts to participate in computation graph
                # 该专家本轮没有 token 也要参与计算图：用零权重的前向保住专家参数的梯度（否则 DDP/FSDP 会因未用参数报错）
                dummy_output = expert_layer(hidden_states[:1]) * 0.0
                final_hidden_states[:1] = final_hidden_states[:1] + dummy_output

        # 恢复 [batch, seq, hidden] 形状
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        # 返回聚合输出与路由 logits（aux loss 等可能需要）
        return final_hidden_states, router_logits

    # 用补丁函数替换 HF Qwen3MoeSparseMoeBlock 的 forward（monkey-patch）
    modeling_qwen3_moe.Qwen3MoeSparseMoeBlock.forward = _forward
