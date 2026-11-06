import re
import torch


# 将 Llama（稠密模型）的 Megatron 参数转换为 HuggingFace 格式
def convert_llama_to_hf(args, name, param):
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
            # Split QKV weight for Llama
            # 融合 QKV 权重：按 KV 组重排为四维视图
            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            # 在第 1 维上切出 Q（value_num_per_group 个头）、K、V（各 1 个头）
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
        elif rest == "mlp.linear_fc1.weight":
            # Split gate and up projections for SwiGLU
            # SwiGLU 的融合 gate/up 权重按第 0 维切成两半
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
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
        elif rest == "pre_mlp_layernorm.weight":
            # Megatron 的 pre_mlp_layernorm 对应 HF 的 post_attention_layernorm
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")
