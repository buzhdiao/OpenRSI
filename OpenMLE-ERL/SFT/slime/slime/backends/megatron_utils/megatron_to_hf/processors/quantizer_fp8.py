import re

import torch

from slime.backends.megatron_utils.kernels.fp8_kernel import blockwise_cast_to_fp8_triton

from ...sglang import quant_weight_ue8m0, should_deepgemm_weight_requant_ue8m0, transform_scale_ue8m0


# 将转换后的 HF 参数按 fp8（e4m3、动态激活）方案量化
def quantize_params_fp8(args, megatron_name, converted_named_params, quantization_config):
    # 校验量化方法为 fp8
    assert quantization_config["quant_method"] == "fp8"
    # 仅支持 e4m3 编码格式
    assert quantization_config["fmt"] == "e4m3"
    # 仅支持动态激活量化方案
    assert quantization_config["activation_scheme"] == "dynamic"
    # 读取权重量化的 block size（如 [128,128] 的块级缩放），可能为 None
    weight_block_size = quantization_config.get("weight_block_size", None)

    # 匹配 decoder 层参数名
    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, megatron_name)

    if not match:
        # check mtp layers
        # 非 decoder 层则尝试匹配 MTP（Multi-Token Prediction）层
        mtp_layer_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
        match = re.match(mtp_layer_pattern, megatron_name)
        if not match:
            # 两者都不匹配的参数（embedding 等）不做量化
            return converted_named_params
        layer_idx, rest = match.groups()
        # MTP 内嵌的 transformer 层组件去掉前缀以便统一判断
        rest = rest.replace("transformer_layer.", "")
    else:
        layer_idx, rest = match.groups()

    # experts
    # 匹配路由专家参数
    expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
    match = re.match(expert_pattern, rest)
    if match:
        rest, expert_idx = match.groups()
        if rest in [
            "linear_fc1",
            "linear_fc2",
        ]:
            quantize_named_params = []
            for converted_name, param in converted_named_params:
                # skip bf16 weight_scale and input_scale
                # TODO: find a clearer way.
                # 跳过转换过程附带产生的 bf16 scale（如 MLA 配对时带出的），只量化真正的权重
                if converted_name.endswith("_scale"):
                    continue
                quantize_named_params.extend(_quantize_param(converted_name, param, weight_block_size))

            return quantize_named_params

    # shared expert
    # 匹配共享专家参数
    shared_expert_pattern = r"mlp.shared_experts\.(.+)"
    match = re.match(shared_expert_pattern, rest)
    if match:
        rest = match.groups()[0]
        if rest in [
            "linear_fc1.weight",
            "linear_fc2.weight",
        ]:
            quantize_named_params = []
            for converted_name, param in converted_named_params:
                # 对共享专家的每个权重执行量化
                quantize_named_params.extend(_quantize_param(converted_name, param, weight_block_size))

            return quantize_named_params

    # 需要量化的普通（非专家）权重列表：注意力投影、MLP、MLA 低秩投影、索引器等
    if rest in [
        "self_attention.linear_proj.weight",
        "self_attention.linear_qkv.weight",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc2.weight",
        # mla
        # MLA（Multi-head Latent Attention）的低秩投影
        "self_attention.linear_q_proj.weight",
        "self_attention.linear_q_down_proj.weight",
        "self_attention.linear_q_up_proj.weight",
        "self_attention.linear_kv_down_proj.weight",
        "self_attention.linear_kv_up_proj.weight",
        # indexer
        # DeepSeek-V3.2 稀疏注意力索引器权重
        "self_attention.wq_b.weight",
        "self_attention.wk.weight",
    ]:
        quantize_named_params = []
        for converted_name, param in converted_named_params:
            # 逐个量化命中的权重
            quantize_named_params.extend(_quantize_param(converted_name, param, weight_block_size))

        return quantize_named_params

    # for other parameters, we just return the original converted_named_params
    # 其余参数（LayerNorm、bias 等）保持原样不量化
    return converted_named_params


def _quantize_param(name, weight, weight_block_size):
    # 仅处理权重参数
    assert name.endswith(".weight"), f"Expected weight parameter, got {name}"
    # fp8 e4m3 的取值范围上下限
    FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
    if weight_block_size is not None:
        # 块级量化（block-wise quantization）：按 weight_block_size 分块求 scale
        if should_deepgemm_weight_requant_ue8m0 and should_deepgemm_weight_requant_ue8m0(
            weight_block_size=weight_block_size
        ):
            # DeepGEMM 路径：scale 需要 UE8M0（2 的幂次）编码，先量化再变换 scale
            qweight, scale = quant_weight_ue8m0(weight, weight_block_size=weight_block_size)
            scale = transform_scale_ue8m0(scale, mn=qweight.shape[-2])
        else:
            # 通用路径：Triton 核做块级 fp8 转换
            qweight, scale = blockwise_cast_to_fp8_triton(weight, weight_block_size)
        # 块级量化的 scale 命名后缀为 weight_scale_inv
        scale_name = name.replace(".weight", ".weight_scale_inv")
    else:
        # per tensor quant
        # 整张量量化：scale = 最大绝对值 / fp8 上限（下限钳制避免除零）
        scale = weight.abs().max().clamp(min=1e-12).to(torch.float32) / FP8_MAX
        # 权重除以 scale 后截断到 fp8 范围并转换类型
        qweight = (weight / scale).clamp(min=FP8_MIN, max=FP8_MAX).to(torch.float8_e4m3fn)
        # 标量 scale 整理为形状 (1,)
        scale = scale.view(1)
        # per-tensor 的 scale 命名后缀为 weight_scale
        scale_name = name.replace(".weight", ".weight_scale")
    # 返回量化权重与对应 scale 的命名张量对
    return [(name, qweight), (scale_name, scale)]
