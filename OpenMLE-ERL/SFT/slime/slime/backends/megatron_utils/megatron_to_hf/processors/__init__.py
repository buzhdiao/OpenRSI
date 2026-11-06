# 导入去 padding 处理器与两种量化处理器（compressed-tensors / fp8）
from .padding_remover import remove_padding
from .quantizer_compressed_tensors import quantize_params_compressed_tensors
from .quantizer_fp8 import quantize_params_fp8

__all__ = ["remove_padding", "quantize_param", "quantize_params_fp8", "quantize_params_compressed_tensors"]


# 量化参数的统一入口：按量化配置分发到对应实现
def quantize_params(args, megatron_name, converted_named_params, quantization_config):
    # 未配置量化则原样返回
    if quantization_config is None:
        return converted_named_params
    elif quantization_config["quant_method"] == "fp8":
        # fp8 量化走 fp8 处理器
        return quantize_params_fp8(args, megatron_name, converted_named_params, quantization_config)
    elif quantization_config["quant_method"] == "compressed-tensors":
        # only int4 at the moment.
        # compressed-tensors 方案（目前仅支持 int4）
        return quantize_params_compressed_tensors(converted_named_params, quantization_config)
