# the file to manage all sglang deps in the megatron actor
# 本文件集中管理 Megatron actor 内对 sglang 的全部依赖（便于隔离版本差异）
try:
    # fp8 权重 UE8M0 量化相关工具（DeepGEMM 路径需要）
    from sglang.srt.layers.quantization.fp8_utils import quant_weight_ue8m0, transform_scale_ue8m0
    from sglang.srt.model_loader.utils import should_deepgemm_weight_requant_ue8m0
except ImportError:
    # sglang 缺失或版本不符时置为 None，调用方需自行判空
    quant_weight_ue8m0 = None
    transform_scale_ue8m0 = None
    should_deepgemm_weight_requant_ue8m0 = None

try:
    # 新版 sglang 中位于 utils.patch_torch 的 torch reductions 补丁
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
except ImportError:
    # 旧版 sglang 的兼容导入路径
    from sglang.srt.patch_torch import monkey_patch_torch_reductions


# 多进程序列化工具：用于把张量在进程间（如 Megatron 与推理引擎间）传输
from sglang.srt.utils import MultiprocessingSerializer


try:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket  # type: ignore[import]
except ImportError:
    # 旧版 sglang 从 model_runner 中导入扁平化张量桶（权重同步分桶用）
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket  # type: ignore[import]

__all__ = [
    "quant_weight_ue8m0",
    "transform_scale_ue8m0",
    "should_deepgemm_weight_requant_ue8m0",
    "monkey_patch_torch_reductions",
    "MultiprocessingSerializer",
    "FlattenedTensorBucket",
]
