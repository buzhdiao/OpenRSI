# 导入 copy，用于深拷贝层规格
import copy
# 导入 json，用于在 transformers 不认识模型类型时直接读取 config.json
import json
# 导入 os，用于拼接配置文件路径
import os

# 导入 PyTorch 主库
import torch
# 导入 nn 模块，用于构建网络层
import torch.nn as nn
# 导入函数式接口 F，用于 softplus 等计算
import torch.nn.functional as F
# 导入 Megatron GPT 解码器层规格构造函数
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
# 导入 ModuleSpec，用于把自定义模块注入层规格
from megatron.core.transformer.spec_utils import ModuleSpec
# 导入计算当前流水线阶段需构建层数的工具
from megatron.core.transformer.transformer_block import get_num_layers_to_build
# 导入计算当前流水线阶段层偏移的工具
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
# 导入 HF 激活函数映射表（名称 -> 激活函数）
from transformers.activations import ACT2FN


# 加载 HF 配置；transformers 不认识该模型类型时做兜底处理
def _load_hf_config(checkpoint_path):
    """Load HF config, handling cases where transformers doesn't know the model type."""
    try:
        # 优先用 transformers 的 AutoConfig 自动识别并加载
        from transformers import AutoConfig

        # trust_remote_code=True 允许执行模型仓库自带的自定义代码
        return AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
    except (ValueError, KeyError):
        # Fallback: load config.json directly as a SimpleNamespace
        # 回退方案：直接读取并解析 config.json
        config_path = os.path.join(checkpoint_path, "config.json")
        # 打开并解析 JSON 配置
        with open(config_path) as f:
            config_dict = json.load(f)
        # If there's a text_config, also make it a namespace
        # 用动态类型把配置字典包装成属性访问式的命名空间对象
        ns = type("HFConfig", (), config_dict)()
        # 若是多模态配置，把 text_config 子字典也包装成命名空间
        if "text_config" in config_dict:
            ns.text_config = type("TextConfig", (), config_dict["text_config"])()
        # 返回兜底构造的配置对象
        return ns


# 尝试导入 fla（flash-linear-attention）库中线性注意力所需组件
try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except ImportError:
    # 缺少 fla 依赖时静默跳过（仅线性注意力模型需要）
    pass

# 导入 HF 注意力层抽象基类
from .hf_attention import HuggingfaceAttention


# 从（可能的）多模态配置中提取文本侧配置
def _get_text_config(hf_config):
    """Extract text config from a VLM config if needed."""
    # 配置含 text_config 子配置（VLM 结构）时返回文本配置
    if hasattr(hf_config, "text_config"):
        return hf_config.text_config
    # 否则原样返回
    return hf_config


# Adapted from Qwen3NextGatedDeltaNet but with separate in_proj_qkv and in_proj_z
# Qwen3.5 的门控 DeltaNet 线性注意力模块（支持变长序列）
class Qwen3_5GatedDeltaNet(nn.Module):
    """
    Qwen3.5 GatedDeltaNet with varlen support.
    Unlike Qwen3Next which uses a combined in_proj_qkvz, Qwen3.5 uses
    separate in_proj_qkv (for Q,K,V) and in_proj_z (for Z).
    """

    def __init__(self, config, layer_idx: int):
        # 调用 nn.Module 基类初始化
        super().__init__()
        # 记录隐藏层维度
        self.hidden_size = config.hidden_size
        # 值（value）头数
        self.num_v_heads = config.linear_num_value_heads
        # 键（key）头数
        self.num_k_heads = config.linear_num_key_heads
        # 每个 key 头的维度
        self.head_k_dim = config.linear_key_head_dim
        # 每个 value 头的维度
        self.head_v_dim = config.linear_value_head_dim
        # key 侧总维度 = 头维度 * key 头数
        self.key_dim = self.head_k_dim * self.num_k_heads
        # value 侧总维度 = 头维度 * value 头数
        self.value_dim = self.head_v_dim * self.num_v_heads

        # 短卷积核大小
        self.conv_kernel_size = config.linear_conv_kernel_dim
        # 当前层索引（用于参数对齐与调试）
        self.layer_idx = layer_idx
        # 隐藏层激活函数名称
        self.activation = config.hidden_act
        # 激活函数名称映射到可调用对象
        self.act = ACT2FN[config.hidden_act]
        # RMSNorm 的 epsilon
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        # 短卷积作用维度 = key*2 + value（Q、K、V 拼接后的总宽度）
        self.conv_dim = self.key_dim * 2 + self.value_dim
        # 构建因果短卷积层（对拼接后的 QKV 做局部卷积）
        self.conv1d = ShortConvolution(
            hidden_size=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
        )

        # Separate projections for QKV and Z (unlike Qwen3Next which combines QKVZ)
        # QKV 联合投影的输出维度
        projection_size_qkv = self.key_dim * 2 + self.value_dim
        # 门控 Z 投影的输出维度（与 value 维度一致）
        projection_size_z = self.value_dim
        # QKV 联合投影层（无偏置）
        self.in_proj_qkv = nn.Linear(self.hidden_size, projection_size_qkv, bias=False)
        # Z 门控投影层（无偏置）
        self.in_proj_z = nn.Linear(self.hidden_size, projection_size_z, bias=False)
        # beta 系数投影（每个 value 头一个标量）
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        # 衰减系数 a 投影（每个 value 头一个标量）
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        # time step projection
        # 时间步偏置参数（每个 value 头一个），控制衰减节奏
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        # 初始化衰减基值 A（每头一个，均匀分布于 0~16）
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        # 存 log(A) 作为可学习参数，保证 A 恒正且数值稳定
        self.A_log = nn.Parameter(torch.log(A))

        # 门控 RMSNorm：融合激活函数与门控 z 的归一化层
        self.norm = FusedRMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            activation=self.activation,
            device=torch.cuda.current_device(),
            dtype=config.dtype if config.dtype is not None else torch.get_current_dtype(),
        )

        # 输出投影层：把 value 维度映射回隐藏层维度
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    # 前向计算；cu_seqlens 用于变长打包序列（varlen）
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
    ):
        # 读取 batch 大小与序列长度
        batch_size, seq_len, _ = hidden_states.shape

        # Projections (flat layout: [Q_all, K_all, V_all])
        # QKV 联合投影
        mixed_qkv = self.in_proj_qkv(hidden_states)
        # Z 门控投影
        z = self.in_proj_z(hidden_states)
        # 把 z 重排成 [batch, seq, 头数, 头维度]
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        # beta 系数投影
        b = self.in_proj_b(hidden_states)
        # 衰减系数投影
        a = self.in_proj_a(hidden_states)

        # Convolution on the flat QKV
        # 对拼接后的 QKV 做因果短卷积（支持变长 cu_seqlens）
        mixed_qkv, _ = self.conv1d(
            x=mixed_qkv,
            cu_seqlens=cu_seqlens,
        )

        # Split into Q, K, V (flat split, matching HF layout)
        # 按最后一维把 QKV 切开（与 HF 权重排布一致）
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        # 重排 Q 成 [batch, seq, 头数, 头维度]
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        # 重排 K 成 [batch, seq, 头数, 头维度]
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        # 重排 V 成 [batch, seq, 头数, 头维度]
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        # beta = sigmoid(b)：控制新信息写入强度的门控（0~1）
        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        # 计算门控衰减 g = -exp(A_log) * softplus(a + dt_bias)；先转 fp32 防止 fp16 下 A 变 -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        # value 头数多于 key 头数（GQA 结构）时，沿头维复制补齐 Q/K
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        # 调用 fla 的分块门控 Delta Rule 线性注意力核（支持 varlen）
        core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )

        # 记录 z 的原始形状，供归一化后还原
        z_shape_og = z.shape
        # reshape input data into 2D tensor
        # 把注意力输出展平成二维 [token, dim] 以便逐元素门控归一化
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        # 同样把门控 z 展平成二维
        z = z.reshape(-1, z.shape[-1])
        # 用门控 RMSNorm 融合激活与 z 门控
        core_attn_out = self.norm(core_attn_out, z)
        # 还原成 z 的原始形状
        core_attn_out = core_attn_out.reshape(z_shape_og)
        # 再合并头维成 [batch, seq, value_dim]
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        # 输出投影回隐藏层维度
        output = self.out_proj(core_attn_out)
        # 返回模块输出
        return output


# Qwen3.5 的注意力层：继承 HF 注意力基类，内部融合线性注意力模块
class Attention(HuggingfaceAttention):
    def __init__(
        self,
        args,
        config,
        layer_number: int,
        cp_comm_type: str = "p2p",
        pg_collection=None,
    ):
        # 调用基类初始化（加载 HF 配置、记录层号等）
        super().__init__(
            args,
            config,
            layer_number,
            cp_comm_type,
            pg_collection,
        )
        # Qwen3.5 is a VLM model with nested text_config
        # Qwen3.5 是 VLM 模型，配置含嵌套的 text_config，改取文本侧配置
        self.hf_config = _get_text_config(self.hf_config)
        # 强制注意力实现为 FlashAttention-2
        self.hf_config._attn_implementation = "flash_attention_2"

        # 构建本层的线性注意力（GatedDeltaNet）模块
        self.linear_attn = Qwen3_5GatedDeltaNet(self.hf_config, self.hf_layer_idx)

        # Use a simple RMSNorm
        # 构建（简单的）RMSNorm 输入归一化层
        try:
            # 优先使用 transformers 中 Qwen3Next 的 RMSNorm 实现
            from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm

            self.input_layernorm = Qwen3NextRMSNorm(self.hf_config.hidden_size, eps=self.hf_config.rms_norm_eps)
        except ImportError:
            # 回退使用 PyTorch 内置 RMSNorm
            from torch.nn import RMSNorm

            self.input_layernorm = RMSNorm(self.hf_config.hidden_size, eps=self.hf_config.rms_norm_eps)

    # HF 风格前向：先归一化再走线性注意力
    def hf_forward(self, hidden_states, packed_seq_params):
        # 输入 RMSNorm 归一化
        hidden_states = self.input_layernorm(hidden_states)
        # 线性注意力前向（cu_seqlens 来自打包序列参数）
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cu_seqlens=packed_seq_params.cu_seqlens_q,
        )
        # 返回注意力输出
        return hidden_states


# 构造 Qwen3.5 的 Megatron 层规格（spec）
def get_qwen3_5_spec(args, config, vp_stage):
    # always use the moe path for MoE models
    # 非 MoE 模型也统一走 MoE 路径：把每层 MoE 频率置 0
    if not args.num_experts:
        config.moe_layer_freq = [0] * config.num_layers

    # Define the decoder block spec
    # 构造解码器块规格的关键字参数
    kwargs = {
        "use_transformer_engine": True,
    }
    # 存在虚拟流水线阶段时一并传入
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    # 获取 GPT 解码器整体块规格
    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)

    # 当前不支持自定义流水线层排布
    assert config.pipeline_model_parallel_layout is None, "not support this at the moment"

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    # 计算本流水线阶段需要构建的层数
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    # 计算本阶段首层的全局层偏移
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)

    # 加载 HF 配置（带兜底逻辑）
    hf_config = _load_hf_config(args.hf_checkpoint)
    # 提取文本侧配置
    text_config = _get_text_config(hf_config)

    # 遍历本阶段的每一层
    for layer_id in range(num_layers_to_build):
        # 该层在 HF 配置中标记为线性注意力层时
        if text_config.layer_types[layer_id + offset] == "linear_attention":
            # 深拷贝该层规格以便替换子模块
            layer_specs = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
            # 把自注意力子模块替换为自定义 Attention（HF 线性注意力）规格
            layer_specs.submodules.self_attention = ModuleSpec(
                module=Attention,
                params={"args": args},
            )
            # 写回层规格列表
            transformer_layer_spec.layer_specs[layer_id] = layer_specs
    # 返回修改后的整体层规格
    return transformer_layer_spec
