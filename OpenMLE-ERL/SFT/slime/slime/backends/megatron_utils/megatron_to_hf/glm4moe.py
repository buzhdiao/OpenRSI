import re

import torch


# 将 GLM-4.5 / GLM-4 MoE（含专家、MTP）的 Megatron 参数转换为 HuggingFace 格式
def convert_glm4moe_to_hf(args, name, param):
    # 词嵌入层映射
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    # 输出层（LM head）映射
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    # 最终 RMSNorm 映射
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    try:
        # 每个注意力头的维度：优先 kv_channels，否则 hidden_size 除以头数
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        # args 无 kv_channels 属性时的兜底
        head_dim = args.hidden_size // args.num_attention_heads
    # 每个 KV 组包含的查询头数量（GQA 分组比）
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    # 匹配 decoder 层参数名，提取层号与层内参数名
    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        # experts
        # 匹配路由专家（routed expert）参数
        expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
        match = re.match(expert_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                # 拆开融合的 gate/up 权重
                gate_weight, up_weight = param.chunk(2, dim=0)
                outputs = [
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight", gate_weight),
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight", up_weight),
                ]
                return outputs
            elif rest == "linear_fc2":
                # 专家下投影
                outputs = [
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight", param),
                ]
                return outputs
            else:
                raise ValueError(f"Unknown expert parameter name: {name}")

        # shared expert
        # 匹配共享专家（shared expert）参数
        shared_expert_pattern = r"mlp.shared_experts\.(.+)"
        match = re.match(shared_expert_pattern, rest)
        if match:
            rest = match.groups()[0]
            if rest == "linear_fc1.weight":
                # 拆开共享专家融合的 gate/up 权重
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight", gate_weight),
                    (f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2.weight":
                # 共享专家下投影
                return [
                    (f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight", param),
                ]
            else:
                raise ValueError(f"Unknown shared expert parameter name: {name}")

        # 注意力输出投影
        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":

            # 融合的 QKV 权重：按 KV 组重排后拆分 Q/K/V 三段
            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1)
            # 三段分别展平为二维权重矩阵
            q_param = q_param.reshape(-1, args.hidden_size)
            k_param = k_param.reshape(-1, args.hidden_size)
            v_param = v_param.reshape(-1, args.hidden_size)
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                (f"model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                (f"model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
            ]
        elif rest == "self_attention.linear_qkv.bias":
            # 融合的 QKV 偏置切分
            param = param.view(args.num_query_groups, -1)
            q_bias, k_bias, v_bias = torch.split(
                param,
                split_size_or_sections=[value_num_per_group * head_dim, head_dim, head_dim],
                dim=1,
            )
            # 各段展平并保证内存连续
            q_bias = q_bias.contiguous().flatten()
            k_bias = k_bias.contiguous().flatten()
            v_bias = v_bias.contiguous().flatten()
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.bias", q_bias),
                (f"model.layers.{layer_idx}.self_attn.k_proj.bias", k_bias),
                (f"model.layers.{layer_idx}.self_attn.v_proj.bias", v_bias),
            ]
        elif rest == "mlp.linear_fc1.weight":
            # 稠密 MLP（第一层为 MoE 的层使用稠密 MLP）融合 gate/up 拆分
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        # 稠密 MLP 下投影
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]
        # 注意力前输入 RMSNorm
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        # MLP 前 RMSNorm
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        # 三明治归一化：注意力后的额外 RMSNorm
        elif rest == "post_self_attn_layernorm.weight":
            return [(f"model.layers.{layer_idx}.post_self_attn_layernorm.weight", param)]
        # MLP 后的额外 RMSNorm
        elif rest == "post_mlp_layernorm.weight":
            return [(f"model.layers.{layer_idx}.post_mlp_layernorm.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            # Megatron 的 pre_mlp_layernorm 对应 HF 的 post_attention_layernorm
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        # MoE 路由器（gate）
        elif rest == "mlp.router.weight":
            return [(f"model.layers.{layer_idx}.mlp.gate.weight", param)]
        # 专家偏置（auxiliary-loss-free 负载均衡修正项）
        elif rest == "mlp.router.expert_bias":
            return [(f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", param)]

        # qk norm
        # Q 头 RMSNorm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        # K 头 RMSNorm
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.k_norm.weight", param)]

    # 匹配 MTP（Multi-Token Prediction，多 token 预测）层
    mtp_layer_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
    match = re.match(mtp_layer_pattern, name)
    if match:
        layer_idx, rest = match.groups()
        # MTP 层在 HF 中编号接在主模型层之后
        layer_idx = int(layer_idx) + args.num_layers
        if rest == "eh_proj.weight":
            # embedding 与隐状态拼接后的投影
            return [(f"model.layers.{layer_idx}.eh_proj.weight", param)]
        elif rest == "enorm.weight":
            # embedding 的 RMSNorm
            return [(f"model.layers.{layer_idx}.enorm.weight", param)]
        elif rest == "hnorm.weight":
            # 隐状态的 RMSNorm
            return [(f"model.layers.{layer_idx}.hnorm.weight", param)]
        elif rest == "final_layernorm.weight":
            # MTP 共享头归一化
            return [(f"model.layers.{layer_idx}.shared_head.norm.weight", param)]
        else:
            # 其余参数复用普通 decoder 层的转换逻辑：改写名称后递归
            name = f"module.module.decoder.layers.{layer_idx}.{rest}"
            name = name.replace("transformer_layer.", "")
            return convert_glm4moe_to_hf(args, name, param)

    raise ValueError(f"Unknown parameter name: {name}")
