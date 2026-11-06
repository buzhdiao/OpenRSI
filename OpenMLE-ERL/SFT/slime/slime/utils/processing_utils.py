# 导入 base64：把图像字节编码为 base64 字符串（经 JSON 传给推理引擎）
import base64
# 导入 io：用内存字节流暂存编码后的图像数据
import io
# 导入 logging：模块级日志
import logging

# 导入 transformers 的处理器/分词器类：加载多模态 processor 与 tokenizer
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin

# 创建本模块专属 logger
logger = logging.getLogger(__name__)

# Default image patch size for vision-language models
# Note: Qwen3-VL uses 16, Qwen2.5-VL uses 14
# Reference: https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/README.md
# 视觉语言模型默认的图像 patch 大小（Qwen3-VL 为 16，Qwen2.5-VL 为 14）
DEFAULT_PATCH_SIZE = 14


def load_tokenizer(name_or_path: str, **kwargs):
    # 从模型名或本地路径加载 HuggingFace 分词器，参数透传
    return AutoTokenizer.from_pretrained(name_or_path, **kwargs)


def build_processor_kwargs(multimodal_inputs: dict | None = None) -> dict:

    forced = {
        # force return_tensors to None for input_ids
        # 强制主输出（input_ids 等）返回列表而非张量，由 RL 框架自行组 batch
        "return_tensors": None,
    }
    # 模态专属输出（图像/音频/视频张量）统一要求返回 PyTorch 张量
    modality_forced = {"return_tensors": "pt"}

    # 复制一份多模态输入字典，避免修改调用方的数据
    result = dict(multimodal_inputs) if multimodal_inputs else {}

    # 覆盖强制项：主输出不转张量
    result.update(forced)

    # set return_tensors="pt" for modality-specific outputs
    # 对三类模态专属参数注入 return_tensors="pt"
    for key in ("audio_kwargs", "images_kwargs", "videos_kwargs"):
        if key in result:
            # 已有该模态参数：在原字典基础上合并强制项
            result[key] = {**result[key], **modality_forced}
        else:
            # 没有也注入一份，保证 processor 行为一致
            result[key] = modality_forced.copy()

    # 返回组装好的 processor 调用参数
    return result


def load_processor(name_or_path: str, **kwargs):
    try:
        # 尝试从模型名/路径加载 AutoProcessor（纯文本模型没有 processor，会失败）
        proc = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except (OSError, ValueError) as e:
        # 加载失败说明是纯 LLM，降级为 None 并记录告警
        logger.warning(f"Failed to load processor from {name_or_path}: {e}")
        proc = None

    # If HF returned a tokenizer, discard it.
    # HF 有时会把 tokenizer 当 processor 返回：纯文本场景丢弃之，只保留真正的多模态 Processor
    if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):
        proc = None

    # 返回 processor 或 None（None 表示无多模态处理需求）
    return proc


def process_vision_info(prompt, processor):
    # TODO: temporary solution, will write image utils for slime later
    # 暂时复用 qwen-vl-utils 的官方实现提取多模态信息
    from qwen_vl_utils import process_vision_info as qwen_process_vision_info

    if hasattr(processor.image_processor, "patch_size"):
        # 优先用 processor 自带的 patch 大小（与模型视觉编码器一致）
        image_patch_size = processor.image_processor.patch_size
    else:
        # 拿不到时回退到默认 patch 大小并记录日志
        logger.info(f"Using default patch size: {DEFAULT_PATCH_SIZE}")
        image_patch_size = DEFAULT_PATCH_SIZE
    # 调用 qwen 工具从 prompt 中解析出图像与视频输入
    images, videos = qwen_process_vision_info(prompt, image_patch_size=image_patch_size)
    # 打包成统一的多模态输入字典
    multimodal_inputs = {"images": images, "videos": videos}
    # 返回给调用方喂给 processor
    return multimodal_inputs


def encode_image_for_rollout_engine(image) -> str:
    """Load an image from path, ensure RGB, encode as PNG base64 string."""
    # 创建内存字节流缓冲区
    buffer = io.BytesIO()
    if image.mode != "RGB":
        # 非 RGB 模式（如 RGBA/P）先转换，避免 PNG 编码含 alpha 通道
        image = image.convert("RGB")
    # 把图像以 PNG 格式写入字节流
    image.save(buffer, format="PNG")
    # 字节流转 base64 字符串（UTF-8 解码成纯文本）
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    # 返回 data URL 形式，sglang 等 rollout 引擎可直接解析
    return f"data:image/png;base64,{image_base64}"
