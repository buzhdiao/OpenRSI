# 从 abc 导入抽象基类与抽象方法装饰器，用于定义注意力层的抽象接口
from abc import ABC, abstractmethod

# 导入 PyTorch 主库
import torch
# 导入 torch 分布式模块，用于上下文并行（CP）时的集合通信
import torch.distributed as dist
# 导入 Megatron-Core 的模型并行工具（mpu）与张量并行工具
from megatron.core import mpu, tensor_parallel
# 导入 Megatron 推理上下文基类，用于推理时 KV 缓存等状态管理
from megatron.core.inference.contexts import BaseInferenceContext
# 导入 Megatron 的打包序列参数类型（变长序列打包所需）
from megatron.core.packed_seq_params import PackedSeqParams
# 导入 MegatronModule 基类，Megatron 模型层的公共父类
from megatron.core.transformer.module import MegatronModule
# 导入 HuggingFace 的 AutoConfig，用于从 HF checkpoint 加载配置
from transformers import AutoConfig


# HF 风格注意力层抽象类：直接复用 HuggingFace 权重实现前向计算
class HuggingfaceAttention(MegatronModule, ABC):
    """Attention layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.
    """

    def __init__(
        self,
        args,
        config,
        layer_number: int,
        cp_comm_type: str = "p2p",
        pg_collection=None,
    ):
        # 调用 MegatronModule 的初始化，绑定模型配置
        super().__init__(config=config)
        # 保存训练参数引用
        self.args = args
        # 保存 Megatron 模型配置
        self.config = config
        # Note that megatron layer_number starts at 1
        # Megatron 的层号从 1 开始计数
        self.layer_number = layer_number
        # 换算成 HF 侧的层索引（从 0 开始），用于权重对齐
        self.hf_layer_idx = layer_number - 1
        # 从 HF checkpoint 加载模型配置（允许执行仓库自带的自定义代码）
        self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        # hardcode to fa2 at the moment.
        # 当前硬编码使用 FlashAttention-2 作为注意力实现
        self.hf_config._attn_implementation = "flash_attention_2"

    # 前向计算：把 Megatron 布局的 hidden_states 送入 HF 注意力实现
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        rotary_pos_emb: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: int | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 打包序列参数必须提供（变长训练依赖其中的 cu_seqlens）
        assert packed_seq_params is not None
        # 取出查询侧 cu_seqlens：每条序列在展平 batch 中的累计长度
        cu_seqlens = packed_seq_params.cu_seqlens_q

        # 开启序列并行（SP）时，先把各 TP rank 的分片 hidden_states 聚合成完整张量
        if self.args.sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, group=mpu.get_tensor_model_parallel_group()
            )

        # 上下文并行（CP）规模大于 1 时，需要把 CP 各分片重组成完整序列
        if mpu.get_context_parallel_world_size() > 1:
            # 取得 CP 组大小
            cp_size = mpu.get_context_parallel_world_size()
            # 在 CP 组内 all-gather 所有 rank 持有的 hidden_states 分片
            hidden_states_list = dist.nn.all_gather(
                hidden_states,
                group=mpu.get_context_parallel_group(),
            )

            # TODO: preprocess this for each batch to prevent tolist in the training step
            # 收集重组后各序列片段的列表
            whole_hidden_states_list = []

            # 把 cu_seqlens 缩放到单个 CP 分片的局部坐标
            local_cu_seqlens = cu_seqlens // cp_size
            # 逐条序列重组：CP 把每条序列切成前后两半，各 rank 各持有一段
            for i in range(len(cu_seqlens) - 1):
                # 计算第 i 条序列的总长度
                seqlen = cu_seqlens[i + 1] - cu_seqlens[i]
                # 每个 CP rank 持有的半段长度（半段再按 cp_size 均分）
                chunk_size = seqlen // 2 // cp_size
                # 先按 rank 顺序拼接“前半”各分片，再按逆序拼接“后半”各分片，还原完整序列
                whole_hidden_states_list.extend(
                    [
                        hidden_states_list[cp_rank][local_cu_seqlens[i] : local_cu_seqlens[i] + chunk_size]
                        for cp_rank in range(cp_size)
                    ]
                    + [
                        hidden_states_list[cp_rank][local_cu_seqlens[i] + chunk_size : local_cu_seqlens[i + 1]]
                        for cp_rank in range(cp_size)
                    ][::-1],
                )
            # 沿序列维拼接成完整的 hidden_states
            hidden_states = torch.cat(whole_hidden_states_list, dim=0)

        # Megatron 用 [seq, batch, hidden] 布局，HF 需要 [batch, seq, hidden]，做维度转换
        hidden_states = hidden_states.permute(1, 0, 2)  # [bsz, seq_len, hidden_dim]

        # 调用子类实现的 HF 前向（真正的注意力计算）
        output = self.hf_forward(hidden_states, packed_seq_params)
        # Megatron 接口要求返回 bias；HF 实现不含 bias，置 None
        bias = None

        # 恢复为 Megatron 的 [seq, batch, hidden] 布局
        output = output.permute(1, 0, 2)  # [seq_len, bsz, hidden_dim]

        # CP 规模大于 1 时，把完整输出重新切分，只保留本 rank 应持的片段
        if mpu.get_context_parallel_world_size() > 1:
            # 取得当前 CP rank 编号
            cp_rank = mpu.get_context_parallel_rank()
            # 收集本 rank 对每条序列应保留的片段
            output_list = []
            # 逐条序列处理
            for i in range(len(cu_seqlens) - 1):
                # 第 i 条序列的总长度
                seqlen = cu_seqlens[i + 1] - cu_seqlens[i]
                # 半段在单个 CP rank 上的长度
                chunk_size = seqlen // 2 // cp_size
                # 截取本条序列对应的完整输出段
                seq = output[cu_seqlens[i] : cu_seqlens[i + 1]]
                # 把该段切成 2*cp_size 个片段（前后两半各 cp_size 份）
                chunks = torch.chunk(seq, 2 * cp_size, dim=0)
                # 保留本 rank 对应的前半片段
                output_list.append(chunks[cp_rank])
                # 保留与本 rank 镜像对应的后半片段
                output_list.append(chunks[2 * cp_size - 1 - cp_rank])
            # 拼接成本 rank 持有的输出分片
            output = torch.cat(output_list, dim=0)

        # 开启序列并行时，把完整输出重新切分回各 TP rank 的分片
        if self.args.sequence_parallel:
            output = tensor_parallel.scatter_to_sequence_parallel_region(
                output, group=mpu.get_tensor_model_parallel_group()
            )

        # 返回输出与 bias（None）
        return output, bias

    # 抽象方法：由具体模型子类实现 HF 风格的前向
    @abstractmethod
    def hf_forward(self, hidden_states, packed_seq_params):
        """Huggingface forward function"""
