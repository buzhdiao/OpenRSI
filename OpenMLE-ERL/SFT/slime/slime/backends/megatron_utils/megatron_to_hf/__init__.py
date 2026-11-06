# 从各模型专属的转换模块中导入"Megatron 参数名 -> HuggingFace 参数名"的转换函数
from .deepseekv3 import convert_deepseekv3_to_hf
from .glm4 import convert_glm4_to_hf
from .glm4moe import convert_glm4moe_to_hf
from .llama import convert_llama_to_hf
from .mimo import convert_mimo_to_hf
# 导入通用后处理器：去 padding（词表对齐）与量化参数处理
from .processors import quantize_params, remove_padding
from .qwen2 import convert_qwen2_to_hf
from .qwen3_5_moe import convert_qwen3_5_moe_to_hf
from .qwen3_next import convert_qwen3_next_to_hf
from .qwen3_vl import convert_qwen3vl_to_hf
from .qwen3moe import convert_qwen3moe_to_hf


# TODO unify w/ `convert_to_hf`
# 对单个 HF 参数做后处理：目前仅做去 padding（后续计划支持量化）
def postprocess_hf_param(args, megatron_param_name, hf_param_name, param):
    # 按目标词表大小裁掉 padding 部分，保证与 HF 词表尺寸一致
    param = remove_padding(megatron_param_name, param, args.vocab_size)
    # TODO support quant
    return param


# TODO optimize code details
# 把 Megatron 侧的单个参数转换为 HF 侧的（可能是一个或多个）参数，并做去 padding 与量化
def convert_to_hf(args, model_name, name, param, quantization_config=None):
    # 先按词表大小去除 embedding / output_layer 上的 padding 行
    param = remove_padding(name, param, args.vocab_size)

    # 调用核心转换逻辑，得到转换后的命名张量列表
    converted_named_tensors = _convert_to_hf_core(args, model_name, name, param)

    # 若配置了量化方案，则对转换结果做量化处理后再返回
    return quantize_params(args, name, converted_named_tensors, quantization_config)


# TODO optimize
# 缓存待配对的转换结果张量（用于 MLA 投影的顺序配对，见下方 _convert_to_hf_core）
_cached_tensors = {}


# TODO optimize code details
# 核心转换逻辑：根据模型名称分发到对应模型的转换函数
def _convert_to_hf_core(args, model_name, name, param):
    # 按模型名关键字匹配对应的转换函数（注意顺序：glm4moe 需在 glm4 之前判断）
    if "glm4moelite" in model_name or "deepseekv3" in model_name:
        converted_named_tensors = convert_deepseekv3_to_hf(args, name, param)
    elif "glm4moe" in model_name:
        converted_named_tensors = convert_glm4moe_to_hf(args, name, param)
    elif "glm4" in model_name:
        converted_named_tensors = convert_glm4_to_hf(args, name, param)
    elif "qwen3moe" in model_name:
        converted_named_tensors = convert_qwen3moe_to_hf(args, name, param)
    elif "qwen3_5" in model_name:
        converted_named_tensors = convert_qwen3_5_moe_to_hf(args, name, param)
    elif "qwen3next" in model_name:
        converted_named_tensors = convert_qwen3_next_to_hf(args, name, param)
    elif "qwen3vl" in model_name:
        converted_named_tensors = convert_qwen3vl_to_hf(args, name, param)
    elif "qwen2" in model_name or "qwen3" in model_name:
        converted_named_tensors = convert_qwen2_to_hf(args, name, param)
    elif "llama" in model_name:
        converted_named_tensors = convert_llama_to_hf(args, name, param)
    elif "mimo" in model_name:
        converted_named_tensors = convert_mimo_to_hf(args, name, param)
    else:
        # 未支持的模型架构直接抛错
        raise ValueError(f"Unsupported model: {model_name}")

    # to compatible with sglang implementation
    # 以下处理与 sglang 权重加载实现对齐：MLA 结构中 q_a_proj 与 kv_a_proj_with_mqa
    # 这两个投影必须成对出现（sglang 按名称配对加载），但转换顺序可能一先一后，
    # 因此用缓存把先到的一方暂存，等另一方到达后再合并输出。
    if args.q_lora_rank is not None:
        # 仅 MLA（q_lora_rank 非空）架构需要这种配对处理
        old_converted_named_tensors = converted_named_tensors
        converted_named_tensors = []
        for converted_name, converted_param in old_converted_named_tensors:
            if "q_a_proj" in converted_name:
                # 计算其配对投影的名字
                pair_name = converted_name.replace("q_a_proj", "kv_a_proj_with_mqa")
                if pair_name in _cached_tensors:
                    # 配对投影已先到：二者一起按顺序输出
                    converted_named_tensors += [
                        (converted_name, converted_param),
                        (pair_name, _cached_tensors[pair_name]),
                    ]
                    # 从缓存中移除已配对的张量
                    del _cached_tensors[pair_name]
                else:
                    # 配对投影尚未到达：先缓存当前张量，等待配对
                    _cached_tensors[converted_name] = converted_param
            elif "kv_a_proj_with_mqa" in converted_name:
                # 对称地处理 kv_a_proj_with_mqa 一侧的配对
                pair_name = converted_name.replace("kv_a_proj_with_mqa", "q_a_proj")
                if pair_name in _cached_tensors:
                    # 配对投影已先到：合并输出
                    converted_named_tensors += [
                        (converted_name, converted_param),
                        (pair_name, _cached_tensors[pair_name]),
                    ]
                    # 清除缓存
                    del _cached_tensors[pair_name]
                else:
                    # 先缓存等待配对
                    _cached_tensors[converted_name] = converted_param
            else:
                # 其余参数无需配对，直接保留
                converted_named_tensors.append((converted_name, converted_param))
    return converted_named_tensors
