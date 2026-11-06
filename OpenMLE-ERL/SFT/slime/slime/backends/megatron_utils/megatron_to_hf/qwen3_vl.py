import re
import torch


# 将 Qwen3-VL（视觉-语言多模态模型）的 Megatron 参数转换为 HuggingFace 格式
def convert_qwen3vl_to_hf(args, name, param):
    # 去掉 Megatron 包装出的 language_model 前缀，使文本侧参数与普通模型命名一致
    if name.startswith("module.module.language_model."):
        name = "module.module." + name[len("module.module.language_model.") :]

    # (Optional safety) if you ever see extra "module." prefixes
    # 防御性处理：剥离重复堆叠的 module. 前缀
    while name.startswith("module.module.module."):
        name = name.replace("module.module.module.", "module.module.", 1)

    if name.startswith("module.module.vision_model."):
        # 视觉塔参数映射到 HF 的 model.visual 命名空间
        hf_name = "model.visual." + name[len("module.module.vision_model.") :]
        return [(hf_name, param)]

    # 词嵌入层映射
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.language_model.embed_tokens.weight", param)]

    if name == "module.module.output_layer.weight":
        # Your key list has lm_head.weight at top-level
        # 输出层在 HF 中位于顶层
        return [("lm_head.weight", param)]

    # 最终 RMSNorm 映射
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.language_model.norm.weight", param)]

    try:
        # 每个注意力头的维度：优先 kv_channels，否则 hidden_size 除以头数
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        # args 无 kv_channels 属性时的兜底
        head_dim = args.hidden_size // args.num_attention_heads
    # 每个 KV 组包含的查询头数量（GQA 分组比）
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    # 匹配 decoder 层参数名
    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()
        # Everything goes under model.language_model.layers.{i}.*
        # 文本层全部映射到 model.language_model.layers 命名空间
        base = f"model.language_model.layers.{layer_idx}"

        # 注意力输出投影
        if rest == "self_attention.linear_proj.weight":
            return [(f"{base}.self_attn.o_proj.weight", param)]

        elif rest == "self_attention.linear_qkv.weight":
            # Keep your original split logic -> q_proj/k_proj/v_proj
            # 融合 QKV 权重拆分
            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1)
            # 三段分别展平为二维权重矩阵
            q_param = q_param.reshape(-1, args.hidden_size)
            k_param = k_param.reshape(-1, args.hidden_size)
            v_param = v_param.reshape(-1, args.hidden_size)
            return [
                (f"{base}.self_attn.q_proj.weight", q_param),
                (f"{base}.self_attn.k_proj.weight", k_param),
                (f"{base}.self_attn.v_proj.weight", v_param),
            ]

        elif rest == "self_attention.linear_qkv.bias":
            # Keep your original split logic -> q_proj/k_proj/v_proj bias
            # 融合 QKV 偏置切分
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
                (f"{base}.self_attn.q_proj.bias", q_bias),
                (f"{base}.self_attn.k_proj.bias", k_bias),
                (f"{base}.self_attn.v_proj.bias", v_bias),
            ]

        elif rest == "mlp.linear_fc1.weight":
            # Keep your original split logic -> gate_proj/up_proj
            # 融合 gate/up 权重拆分
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"{base}.mlp.gate_proj.weight", gate_weight),
                (f"{base}.mlp.up_proj.weight", up_weight),
            ]

        # MLP 下投影
        elif rest == "mlp.linear_fc2.weight":
            return [(f"{base}.mlp.down_proj.weight", param)]

        # 注意力前输入 RMSNorm
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"{base}.input_layernorm.weight", param)]

        # MLP 前 RMSNorm
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"{base}.post_attention_layernorm.weight", param)]

        # qk norm
        # Q 头 RMSNorm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"{base}.self_attn.q_norm.weight", param)]
        # K 头 RMSNorm
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"{base}.self_attn.k_norm.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")
