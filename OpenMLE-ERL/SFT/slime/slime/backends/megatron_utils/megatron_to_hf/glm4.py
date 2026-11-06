import re
import torch


# 将 GLM-4（稠密模型，含 QK-Norm 与三明治归一化）的 Megatron 参数转换为 HuggingFace 格式
def convert_glm4_to_hf(args, name, param):
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
        # 注意力输出投影
        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":

            # 融合的 QKV 权重：按 KV 组重排后把 Q/K/V 三段拆开
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
            # 融合的 QKV 偏置：按 KV 组 reshape 后按 Q/K/V 的大小切分
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
            # GLM-4 的 HF 格式保留融合的 gate_up_proj，无需拆分
            return [
                (f"model.layers.{layer_idx}.mlp.gate_up_proj.weight", param),
            ]
        # MLP 下投影
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]
        # 注意力前输入 RMSNorm
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        # MLP 前 RMSNorm
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]

        # qk norm
        # Q 头上的 RMSNorm（QK-Norm，稳定注意力 logits）
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        # K 头上的 RMSNorm
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.k_norm.weight", param)]

        # sandwitch norm
        # 三明治归一化：注意力之后的额外 RMSNorm
        elif rest == "post_self_attn_layernorm.weight":
            return [(f"model.layers.{layer_idx}.post_self_attn_layernorm.weight", param)]
        # MLP 之后的额外 RMSNorm
        elif rest == "post_mlp_layernorm.weight":
            return [(f"model.layers.{layer_idx}.post_mlp_layernorm.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")
