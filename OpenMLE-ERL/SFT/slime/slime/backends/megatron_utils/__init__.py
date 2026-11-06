# 导入 logging 模块，用于调整日志级别与输出警告
import logging

# 导入 torch，用于在 monkey-patch 中调用 CUDA 同步
import torch

# 尝试导入 deep_ep 与 torch_memory_saver；两者均为 MoE 训练/显存管理相关可选依赖
try:
    import deep_ep
    from torch_memory_saver import torch_memory_saver

    # 保存 deep_ep.Buffer 原始的 __init__ 实现，便于在补丁中回调
    old_init = deep_ep.Buffer.__init__

    # 定义替换后的 Buffer.__init__：在初始化前后切换 torch_memory_saver 的"关注区域"
    def new_init(self, *args, **kwargs):
        # 若显存保存器已启用，先关闭 interesting region 标记，让 Buffer 初始化期间的分配可被丢弃/换出
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(False)
        # 调用原始的 Buffer.__init__ 完成真正的初始化
        old_init(self, *args, **kwargs)
        # CUDA 同步，确保初始化中的异步显存操作全部完成后再做区域状态切换
        torch.cuda.synchronize()
        # 重新开启 interesting region 标记，恢复对后续分配的保护
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(True)

    # 用补丁函数替换 deep_ep.Buffer 的构造函数（monkey-patch，无需修改 deep_ep 源码）
    deep_ep.Buffer.__init__ = new_init
# 若 deep_ep 未安装则跳过补丁，仅记录警告提示部分功能受限
except ImportError:
    logging.warning("deep_ep is not installed, some functionalities may be limited.")

# 尝试导入 Megatron-Bridge 中 Qwen3-VL 文本模型的旋转位置编码类（RoPE）
try:
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import (
        Qwen3VLMoETextRotaryEmbedding,
        Qwen3VLTextRotaryEmbedding,
    )

    # 定义给 RoPE 类打补丁的函数：去掉 packed_seq_params 相关的变长/打包序列处理逻辑
    def patch_rotary_embedding(cls):
        # 保存该类原始的 forward 方法
        _original_forward = cls.forward

        # 定义替换后的 forward：丢弃 packed_seq_params 参数，直接按原始方式调用
        def _patched_forward(self, *args, packed_seq_params=None, **kwargs):
            return _original_forward(self, *args, **kwargs)

        # 将补丁后的 forward 挂回类上，完成对该类所有实例的行为替换
        cls.forward = _patched_forward

    # 对普通 VL 文本 RoPE 类与 MoE 版本 RoPE 类分别应用补丁
    patch_rotary_embedding(Qwen3VLTextRotaryEmbedding)
    patch_rotary_embedding(Qwen3VLMoETextRotaryEmbedding)
# 若对应 Megatron 模块不可导入（未安装或版本不符），静默跳过补丁
except ImportError:
    pass

# 将 megatron 相关 logger 级别设为 WARNING，屏蔽训练过程中 Megatron 的大量 INFO 日志
logging.getLogger("megatron").setLevel(logging.WARNING)
