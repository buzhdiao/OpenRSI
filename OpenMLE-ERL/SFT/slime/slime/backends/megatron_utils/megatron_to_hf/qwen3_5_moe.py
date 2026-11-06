"""Map Qwen3.5 MoE parameters from Megatron torch_dist to Hugging Face.

This module handles the Qwen3.5-35B-A3B text backbone. The model combines
linear and full attention. Hugging Face stores MoE expert weights as grouped
tensors whose first dimension indexes experts, so the older Qwen2/Qwen3
mapping that emits one key per expert cannot be reused.
"""

from __future__ import annotations

import re

import torch


def _convert_full_attention_qkv(args, layer_idx: str, param: torch.Tensor):
    """Split fused QKV weights using the gated Qwen3.5 Q-projection layout."""

    # 每个注意力头的维度：优先 kv_channels，否则 hidden_size 除以头数
    head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    # 每个 KV 组包含的查询头数量（GQA 分组比）
    value_num_per_group = args.num_attention_heads // args.num_query_groups
    # 按 KV 组重排为四维视图
    param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
    # Qwen3.5 是门控注意力（gated attention）：Q 段是每组普通头的 2 倍
    q_param, k_param, v_param = torch.split(
        param,
        split_size_or_sections=[2 * value_num_per_group, 1, 1],
        dim=1,
    )
    # 门控 Q 的存储顺序与 HF 不同：reshape 后交换"组内门控两半"与"头"两个维度
    q_param = (
        q_param.reshape(args.num_query_groups, 2, value_num_per_group, head_dim, args.hidden_size)
        .transpose(1, 2)
        .reshape(-1, args.hidden_size)
    )
    # HF 侧 Qwen3.5 使用 model.language_model 前缀
    prefix = f"model.language_model.layers.{layer_idx}.self_attn"
    return [
        (f"{prefix}.q_proj.weight", q_param),
        (f"{prefix}.k_proj.weight", k_param.reshape(-1, args.hidden_size)),
        (f"{prefix}.v_proj.weight", v_param.reshape(-1, args.hidden_size)),
    ]


def convert_qwen3_5_moe_to_hf(args, name: str, param: torch.Tensor):
    """Map one Megatron parameter to one or more Qwen3.5 HF parameters."""

    # 词嵌入层映射（注意 Qwen3.5 的 HF 前缀为 model.language_model）
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.language_model.embed_tokens.weight", param)]
    # 输出层（LM head）映射
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    # 最终 RMSNorm 映射
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.language_model.norm.weight", param)]

    # 匹配 decoder 层参数名，提取层号与层内参数名
    match = re.match(r"module\.module\.decoder\.layers\.(\d+)\.(.+)", name)
    if not match:
        raise ValueError(f"Unknown Qwen3.5 parameter name: {name}")

    layer_idx, rest = match.groups()
    # 构造 HF 侧该层的名称前缀
    layer_prefix = f"model.language_model.layers.{layer_idx}"

    # Linear-attention weight names match HF; only the layer prefix changes.
    # 线性注意力（linear attention）权重名与 HF 一致，仅需替换层前缀
    linear_attention_prefix = "self_attention.linear_attn."
    if rest.startswith(linear_attention_prefix):
        # 取出线性注意力组件的后缀名
        suffix = rest.removeprefix(linear_attention_prefix)
        return [(f"{layer_prefix}.linear_attn.{suffix}", param)]

    # Megatron fuses full-attention QKV weights, including the gated Q part.
    # 全注意力的融合 QKV 权重（含门控 Q）走专门的拆分函数
    if rest == "self_attention.linear_qkv.weight":
        return _convert_full_attention_qkv(args, layer_idx, param)

    # 可直接按名称映射的参数表（注意力输出投影、QK-Norm、归一化层、路由器、专家组等）
    direct_names = {
        "self_attention.linear_proj.weight": f"{layer_prefix}.self_attn.o_proj.weight",
        "self_attention.q_layernorm.weight": f"{layer_prefix}.self_attn.q_norm.weight",
        "self_attention.k_layernorm.weight": f"{layer_prefix}.self_attn.k_norm.weight",
        "self_attention.linear_qkv.layer_norm_weight": f"{layer_prefix}.input_layernorm.weight",
        "self_attention.input_layernorm.weight": f"{layer_prefix}.input_layernorm.weight",
        "pre_mlp_layernorm.weight": f"{layer_prefix}.post_attention_layernorm.weight",
        "mlp.router.weight": f"{layer_prefix}.mlp.gate.weight",
        # MoE 专家组：HF 以首维为专家索引的分组张量存储，故保留融合形状不拆
        "mlp.experts.experts.linear_fc1.weight": f"{layer_prefix}.mlp.experts.gate_up_proj",
        "mlp.experts.experts.linear_fc2.weight": f"{layer_prefix}.mlp.experts.down_proj",
        # 共享专家下投影
        "mlp.shared_experts.linear_fc2.weight": f"{layer_prefix}.mlp.shared_expert.down_proj.weight",
        # 共享专家门控（决定共享专家输出参与程度的小型投影）
        "mlp.shared_experts.gate_weight": f"{layer_prefix}.mlp.shared_expert_gate.weight",
    }
    if rest in direct_names:
        # 命中映射表则直接返回
        return [(direct_names[rest], param)]

    # HF stores the shared expert's fused gate/up layer as two parameters.
    # 共享专家融合的 gate/up 权重需拆成两个 HF 参数
    if rest == "mlp.shared_experts.linear_fc1.weight":
        gate_weight, up_weight = param.chunk(2, dim=0)
        return [
            (f"{layer_prefix}.mlp.shared_expert.gate_proj.weight", gate_weight),
            (f"{layer_prefix}.mlp.shared_expert.up_proj.weight", up_weight),
        ]

    raise ValueError(f"Unknown Qwen3.5 parameter name: {name}")
