import re

import torch


# 将 DeepSeek-V3（含 MLA、MoE、MTP、索引器 indexer）的 Megatron 参数转换为 HuggingFace 格式
def convert_deepseekv3_to_hf(args, name, param):
    # 词嵌入层：Megatron -> HF 的 embed_tokens
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    # 输出层（LM head）：直接映射到 lm_head
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    # decoder 最后一层 RMSNorm -> HF 的 model.norm
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    try:
        # 计算每个注意力头的维度：优先用 kv_channels，否则用 hidden_size / 注意力头数
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        # args 缺少 kv_channels 属性时的兜底计算
        head_dim = args.hidden_size // args.num_attention_heads
    # 每个 KV 组包含的查询头数量（GQA 分组比，术语：Grouped-Query Attention）
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    # 匹配 decoder 层参数名，提取层号 layer_idx 与层内参数名 rest
    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        # experts
        # 匹配路由专家（routed expert）参数：形如 mlp.experts.linear_fc1.weightN
        expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
        match = re.match(expert_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                # Megatron 把 gate/up 融合在第一个全连接层，按第 0 维切成两半
                gate_weight, up_weight = param.chunk(2, dim=0)
                outputs = [
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight", gate_weight),
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight", up_weight),
                ]
                return outputs
            elif rest == "linear_fc2":
                # 专家的第二个全连接层即 down_proj
                outputs = [
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight", param),
                ]
                return outputs
            else:
                raise ValueError(f"Unknown expert parameter name: {name}")

        # shared expert
        # 匹配共享专家（shared expert，MoE 中所有 token 共享的 FFN）参数
        shared_expert_pattern = r"mlp.shared_experts\.(.+)"
        match = re.match(shared_expert_pattern, rest)
        if match:
            rest = match.groups()[0]
            if rest == "linear_fc1.weight":
                # 同样把融合的 gate/up 拆开
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight", gate_weight),
                    (f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2.weight":
                # 共享专家的下投影
                return [(f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight", param)]
            else:
                raise ValueError(f"Unknown shared expert parameter name: {name}")

        # 注意力输出投影 o_proj
        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        # 普通（非 MLA）Q 投影
        elif rest == "self_attention.linear_q_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_proj.weight", param)]
        # MLA 低秩下投影：q_a_proj（论文术语：Multi-head Latent Attention，Q 的低秩压缩）
        elif rest == "self_attention.linear_q_down_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_a_proj.weight", param)]
        # Q 低秩路径上的 RMSNorm
        elif rest == "self_attention.linear_q_up_proj.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_a_layernorm.weight", param)]

        # indexer
        # 以下为 DeepSeek-V3.2 的稀疏注意力索引器（indexer）参数
        elif rest == "self_attention.wq_b.weight":
            wq_b = param
            # 重排维度：按 (头, 128, d) 视图后交换前 64 与后 64 通道（硬编码 128 为索引器头维）
            wq_b = wq_b.view(-1, 128, wq_b.shape[-1])  # hard code 128
            wq_b = torch.cat([wq_b[:, 64:], wq_b[:, :64]], dim=1).view(-1, wq_b.shape[-1])
            return [(f"model.layers.{layer_idx}.self_attn.indexer.wq_b.weight", wq_b)]
        elif rest == "self_attention.wk.weight":
            wk = param
            # 键投影同样需要交换前 64 与后 64 行
            wk = torch.cat([wk[64:], wk[:64]], dim=0).view(-1, wk.shape[-1])
            return [(f"model.layers.{layer_idx}.self_attn.indexer.wk.weight", wk)]
        elif rest == "self_attention.weights_proj.weight":
            # 索引器打分投影，无需重排直接映射
            return [(f"model.layers.{layer_idx}.self_attn.indexer.weights_proj.weight", param)]
        elif rest == "self_attention.k_norm.weight":
            knorm_weight = param
            # 键归一化权重同样交换前 64 与后 64 行
            knorm_weight = torch.cat([knorm_weight[64:], knorm_weight[:64]], dim=0)
            return [(f"model.layers.{layer_idx}.self_attn.indexer.k_norm.weight", knorm_weight)]
        elif rest == "self_attention.k_norm.bias":
            knorm_bias = param
            # 偏置做同样的通道交换
            knorm_bias = torch.cat([knorm_bias[64:], knorm_bias[:64]], dim=0)
            return [(f"model.layers.{layer_idx}.self_attn.indexer.k_norm.bias", knorm_bias)]

        # MLA 的 Q 上投影 q_b_proj
        elif rest == "self_attention.linear_q_up_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_b_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.bias":
            # 融合的 QKV 偏置：先按 KV 组reshape再切分 Q/K/V 三段
            param = param.view(args.num_query_groups, -1)
            q_bias, k_bias, v_bias = torch.split(
                param,
                split_size_or_sections=[value_num_per_group * head_dim, head_dim, head_dim],
                dim=1,
            )
            # 三段分别展平并保证内存连续，对应 HF 的独立偏置
            q_bias = q_bias.contiguous().flatten()
            k_bias = k_bias.contiguous().flatten()
            v_bias = v_bias.contiguous().flatten()
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.bias", q_bias),
                (f"model.layers.{layer_idx}.self_attn.k_proj.bias", k_bias),
                (f"model.layers.{layer_idx}.self_attn.v_proj.bias", v_bias),
            ]
        # 稠密 MLP 的融合 gate/up 权重拆分
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        # 稠密 MLP 的下投影
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]
        # 注意力前的输入层 RMSNorm（两种 Megatron 命名均归一到 input_layernorm）
        elif rest == "self_attention.linear_qkv.layer_norm_weight" or rest == "input_layernorm.weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        # MLP 前的 RMSNorm
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        # MLA 的 KV 低秩下投影（KV 共享压缩，与 q_a 对应）
        elif rest == "self_attention.linear_kv_down_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight", param)]
        # KV 低秩路径上的 RMSNorm
        elif rest == "self_attention.linear_kv_up_proj.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight", param)]
        # MLA 的 KV 上投影
        elif rest == "self_attention.linear_kv_up_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            # Megatron 的 pre_mlp_layernorm 对应 HF 的 post_attention_layernorm
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        # MoE 路由器（gate）：计算每个专家的选择权重
        elif rest == "mlp.router.weight":
            return [(f"model.layers.{layer_idx}.mlp.gate.weight", param)]
        # 专家偏置（用于负载均衡的修正项，术语：auxiliary-loss-free load balancing bias）
        elif rest == "mlp.router.expert_bias":
            return [(f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", param)]

    # 匹配 MTP（Multi-Token Prediction，多 token 预测）层的参数
    mtp_layer_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
    match = re.match(mtp_layer_pattern, name)
    if match:
        layer_idx, rest = match.groups()
        # MTP 层在 HF 中编号接在主模型层之后
        layer_idx = int(layer_idx) + args.num_layers
        if rest == "eh_proj.weight":
            # 拼接 embedding 与隐状态的投影
            return [(f"model.layers.{layer_idx}.eh_proj.weight", param)]
        elif rest == "enorm.weight":
            # 对 embedding 的 RMSNorm
            return [(f"model.layers.{layer_idx}.enorm.weight", param)]
        elif rest == "hnorm.weight":
            # 对隐状态的 RMSNorm
            return [(f"model.layers.{layer_idx}.hnorm.weight", param)]
        elif rest == "final_layernorm.weight":
            # MTP 共享头的归一化层
            return [(f"model.layers.{layer_idx}.shared_head.norm.weight", param)]
        else:
            # 其余参数与普通 decoder 层结构一致：改写名称后递归复用本函数
            name = f"module.module.decoder.layers.{layer_idx}.{rest}"
            name = name.replace("transformer_layer.", "")
            return convert_deepseekv3_to_hf(args, name, param)

    raise ValueError(f"Unknown parameter name: {name}")
