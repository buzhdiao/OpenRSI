import torch

from slime.backends.megatron_utils.misc_utils import strip_param_name_prefix


def remove_padding(name: str, param: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Remove vocab padding: param[:vocab_size] for embedding/output layers, else unchanged.
    """
    # 仅词嵌入与输出层存在词表 padding：训练时为对齐并行度会补足到更大词表
    if strip_param_name_prefix(name) in {"embedding.word_embeddings.weight", "output_layer.weight"}:
        # 截取到真实词表大小，去掉 padding 行
        return param[:vocab_size]
    # 其余参数不做处理
    return param
