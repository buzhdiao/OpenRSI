import re

import torch


def _convert_mtp_layer(args, name, param, layer_idx):
    """Convert MTP layer parameters from Megatron to HuggingFace format.

    Handles both wrapper layers (enorm, hnorm, final_layernorm, eh_proj) and
    inner transformer layers for any number of MTP layers.
    """
    # MTP wrapper layers (layer index independent in HF format)
    # MTP 包装层（HF 格式下与层号无关，挂在顶层 mtp 命名空间）
    if "enorm.weight" in name:
        # embedding 侧的预连接归一化
        return [("mtp.pre_fc_norm_embedding.weight", param)]
    if "hnorm.weight" in name:
        # 隐状态侧的预连接归一化
        return [("mtp.pre_fc_norm_hidden.weight", param)]
    if "final_layernorm.weight" in name:
        # MTP 块自身的最终归一化
        return [("mtp.norm.weight", param)]
    if "eh_proj.weight" in name:
        # 拼接投影要求为二维权重
        if param.dim() < 2:
            raise ValueError(f"eh_proj weight expects 2D tensor, got {param.shape}")
        # Megatron 与 HF 的拼接顺序相反：交换两半
        first_half, second_half = param.chunk(2, dim=1)
        new_param = torch.cat([second_half, first_half], dim=1)
        return [("mtp.fc.weight", new_param)]

    # MTP inner transformer layers (keep layer index)
    # MTP 内嵌的 transformer 层：复用主体层的转换逻辑，保留层号
    if "transformer_layer" in name:
        # 伪造一个 decoder 层参数名以便复用主转换函数
        proxy_name = name.replace(f"mtp.layers.{layer_idx}.transformer_layer", f"decoder.layers.{layer_idx}")
        mapped_params = convert_qwen3_next_to_hf(args, proxy_name, param)

        final_params = []
        for hf_name, tensor in mapped_params:
            # 把结果中的层名从 model.layers.N 改为 mtp.layers.N
            target_prefix = f"mtp.layers.{layer_idx}"
            if f"model.layers.{layer_idx}" in hf_name:
                new_hf_name = hf_name.replace(f"model.layers.{layer_idx}", target_prefix)
                final_params.append((new_hf_name, tensor))
            else:
                # 未包含层前缀的参数原样保留
                final_params.append((hf_name, tensor))
        return final_params

    return None


def convert_qwen3_next_to_hf(args, name, param):
    """Convert Qwen3 Next model parameters from Megatron to HuggingFace format."""
    # Handle MTP layers
    # MTP 层参数优先走专门的转换逻辑
    if "mtp.layers" in name:
        # 从参数名中解析层号
        parts = name.split(".")
        try:
            layer_idx_loc = parts.index("layers") + 1
            layer_idx = parts[layer_idx_loc]
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid MTP layer name format: {name}") from e

        result = _convert_mtp_layer(args, name, param, layer_idx)
        if result is not None:
            return result

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

    # 匹配 decoder 层参数名
    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        # experts
        # 匹配路由专家参数
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
        # 匹配共享专家参数
        shared_expert_pattern = r"mlp.shared_experts\.(.+)"
        match = re.match(shared_expert_pattern, rest)
        if match:
            rest = match.groups()[0]
            if rest == "linear_fc1.weight":
                # 拆开共享专家融合的 gate/up 权重
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight", gate_weight),
                    (f"model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2.weight":
                # 共享专家下投影
                return [(f"model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight", param)]
            elif rest == "gate_weight":
                # 共享专家门控投影
                return [(f"model.layers.{layer_idx}.mlp.shared_expert_gate.weight", param)]
            else:
                raise ValueError(f"Unknown shared expert parameter name: {name}")

        # 注意力输出投影
        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":
            # 门控注意力：Q 段为普通头的 2 倍，切分 Q/K/V
            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(
                param, split_size_or_sections=[2 * value_num_per_group, 1, 1], dim=1
            )
            # 门控 Q 需交换"门控两半"与"头"两个维度后再展平
            q_param = (
                q_param.reshape(args.num_query_groups, 2, value_num_per_group, head_dim, args.hidden_size)
                .transpose(1, 2)
                .reshape(-1, args.hidden_size)
            )
            # K/V 直接展平
            k_param = k_param.reshape(-1, args.hidden_size)
            v_param = v_param.reshape(-1, args.hidden_size)
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                (f"model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                (f"model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
            ]
        elif rest == "self_attention.linear_qkv.bias":
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
                (f"model.layers.{layer_idx}.self_attn.q_proj.bias", q_bias),
                (f"model.layers.{layer_idx}.self_attn.k_proj.bias", k_bias),
                (f"model.layers.{layer_idx}.self_attn.v_proj.bias", v_bias),
            ]
        elif rest == "mlp.linear_fc1.weight":
            # 稠密 MLP 融合 gate/up 拆分
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
        # MoE 路由器（gate）
        elif rest == "mlp.router.weight":
            return [(f"model.layers.{layer_idx}.mlp.gate.weight", param)]
        # 专家偏置（负载均衡修正项）
        elif rest == "mlp.router.expert_bias":
            return [(f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", param)]

        # qk norm
        # Q 头 RMSNorm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        # K 头 RMSNorm
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.k_norm.weight", param)]
        elif rest.startswith("self_attention.") and rest[len("self_attention.") :] in [
            # Qwen3 Next 是混合注意力架构：以下为可直接换名的参数清单
            "input_layernorm.weight",
            # linear attn
            # 线性注意力（gated delta net 类）参数
            "linear_attn.A_log",
            "linear_attn.conv1d.weight",
            "linear_attn.dt_bias",
            "linear_attn.in_proj_ba.weight",
            "linear_attn.in_proj_qkvz.weight",
            "linear_attn.norm.weight",
            "linear_attn.out_proj.weight",
            # gated attn
            # 门控全注意力参数
            "self_attn.k_norm.weight",
            "self_attn.k_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.q_proj.weight",
            "self_attn.v_proj.weight",
        ]:
            # 去掉 Megatron 的 self_attention. 前缀后直接沿用组件名
            rest = rest[len("self_attention.") :]
            return [(f"model.layers.{layer_idx}.{rest}", param)]

    raise ValueError(f"Unknown parameter name: {name}")
