import re
import torch
from .qwen2 import convert_qwen2_to_hf


def convert_mimo_to_hf(args, name, param):
    """
    Convert MiMo model parameters from Megatron to HuggingFace format.

    MiMo extends Qwen2 with MTP (Multi-Token Prediction) layers.
    """

    # MTP 层参数走专门的 MTP 转换逻辑
    if "mtp" in name:
        return convert_mimo_mtp_param(args, name, param)

    # 其余主体结构与 Qwen2 相同，直接复用 Qwen2 的转换函数
    return convert_qwen2_to_hf(args, name, param)


def convert_mimo_mtp_param(args, name, param):
    """
    Convert MTP layer parameters from Megatron to HuggingFace format.

    MTP layers in MiMo contain:
    - LayerNorms (token_layernorm, hidden_layernorm, final_layernorm)
    - Input projection (input_proj)
    - Self attention (reuses Qwen2 attention structure)
    - MLP (reuses Qwen2 MLP structure)

    Based on MimoBridge._convert_mtp_param logic (reverse mapping)
    """
    # 匹配 MTP 层参数名，提取层号与组件名
    mtp_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
    match = re.match(mtp_pattern, name)

    if not match:
        # 无法匹配说明参数名不符合 MTP 命名规范
        raise ValueError(f"Invalid MTP parameter name: {name}")

    layer_idx, component = match.groups()

    # Direct mappings for MTP-specific components (Megatron -> HF)
    # Based on MimoBridge direct_name_mapping (reversed)
    # MTP 专属组件的直接名称映射表
    direct_mappings = {
        "enorm.weight": f"model.mtp_layers.{layer_idx}.token_layernorm.weight",
        "hnorm.weight": f"model.mtp_layers.{layer_idx}.hidden_layernorm.weight",
        "eh_proj.weight": f"model.mtp_layers.{layer_idx}.input_proj.weight",
        "final_layernorm.weight": f"model.mtp_layers.{layer_idx}.final_layernorm.weight",
    }
    if component == "eh_proj.weight":
        # Megatron 的拼接顺序与 HF 相反：交换两半，使其与 HF 的 input_proj 一致
        first_half, second_half = param.chunk(2, dim=1)
        param = torch.cat([second_half, first_half], dim=1)

    # Check direct mappings first
    # 命中直接映射表则直接返回
    if component in direct_mappings:
        return [(direct_mappings[component], param)]

    # Handle transformer_layer components
    # 处理 MTP 内嵌的 transformer_layer 组件（注意力 / MLP 等）
    if component.startswith("transformer_layer."):
        # Remove "transformer_layer." prefix
        # 去掉 transformer_layer. 前缀
        transformer_component = component[len("transformer_layer.") :]

        # Create proxy name for reusing existing Qwen2 conversion functions
        # 伪造一个普通 decoder 层的参数名，以便复用 Qwen2 转换函数
        proxy_name = f"module.module.decoder.layers.{layer_idx}.{transformer_component}"

        # Use existing convert_qwen2_to_hf function for transformer components
        results = convert_qwen2_to_hf(args, proxy_name, param)

        # Replace model.layers with mtp_layers in results
        # 把转换结果中的层名改回 MTP 层名
        converted_results = []
        for hf_name, hf_param in results:
            # Replace model.layers.{idx} with mtp_layers.{idx}
            # model.layers.N -> model.mtp_layers.N
            hf_name = hf_name.replace(f"model.layers.{layer_idx}", f"model.mtp_layers.{layer_idx}")
            converted_results.append((hf_name, hf_param))

        return converted_results

    raise ValueError(f"Unknown MTP component: {component} in {name}")
