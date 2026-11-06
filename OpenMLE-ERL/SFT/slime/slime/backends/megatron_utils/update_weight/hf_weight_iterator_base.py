from abc import ABC, abstractmethod


# HF 权重迭代器的抽象基类：把 Megatron 本地权重转为 HF 格式并分块产出
class HfWeightIteratorBase(ABC):
    @staticmethod
    def create(args, model, **kwargs):
        # 按配置中的转换模式选择具体实现
        from .hf_weight_iterator_bridge import HfWeightIteratorBridge
        from .hf_weight_iterator_direct import HfWeightIteratorDirect

        c = {
            # raw: slime 自带的直接名称映射实现
            "raw": HfWeightIteratorDirect,
            # bridge: 基于 megatron-bridge 的实现
            "bridge": HfWeightIteratorBridge,
        }[args.megatron_to_hf_mode]

        return c(args, model, **kwargs)

    def __init__(self, args, model, model_name, quantization_config):
        # 保存运行参数、模型引用、模型名与量化配置
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config

    @abstractmethod
    def get_hf_weight_chunks(self, megatron_local_weights):
        """
        Mental model of the API:
        megatron_model.to_hf_magically().named_parameters()
        """
        raise NotImplementedError
