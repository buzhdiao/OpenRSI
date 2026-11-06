# Adapt from https://github.com/NVIDIA/Megatron-LM/blob/b1efb3c7126ef7615e8c333432d76e08038e17ff/pretrain_gpt.py
import argparse
import inspect
import re
from contextlib import nullcontext
from typing import Literal

import torch
from megatron.core import tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import core_transformer_config_from_args

from slime.utils.misc import load_function


# Adapt from https://github.com/volcengine/verl/blob/c3b20575d2bc815fcccd84bddb4c0401fc4b632b/verl/models/llama/megatron/layers/parallel_linear.py#L82
# 专为 RLHF critic（价值头）设计的最后一层线性层
class LinearForLastLayer(torch.nn.Linear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        bias: bool = True,
    ) -> None:
        # 调用标准 Linear 初始化（输出维度为 1，即标量价值）
        super().__init__(in_features=input_size, out_features=output_size, bias=bias)
        # 记录是否启用序列并行（sequence parallelism，序列维切分）
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel:
            # 权重参与序列并行的梯度 all-reduce
            self.weight.sequence_parallel = True
            if bias:
                self.bias.sequence_parallel = True

        # 用 GPT 风格的初始化：权重 N(0, 0.02)，偏置清零
        self.weight.data.normal_(mean=0.0, std=0.02)
        if bias:
            self.bias.data.zero_()

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        # 普通线性前向得到 logits（此处为每个 token 的价值/奖励）
        logits = super().forward(input_)
        # 转为 float32 以保证损失计算精度
        logits = logits.float()
        if self.sequence_parallel:
            # 序列并行时把分散在各序列分片上的结果 gather 起来
            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        return logits, None


# 返回"模型构建器"的工厂函数：根据角色（actor/critic）与配置决定如何创建 GPTModel
def get_model_provider_func(
    args: argparse.Namespace,
    role: Literal["actor", "critic"] = "actor",
):
    # Support custom model provider path (similar to --custom-rm-path for reward models)
    # 支持用户自定义模型构建函数路径（类似奖励模型的 --custom-rm-path）
    if getattr(args, "custom_model_provider_path", None):

        def wrapped_model_provider(
            pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None
        ) -> GPTModel:
            # 动态加载用户自定义的 provider 函数
            custom_model_provider = load_function(args.custom_model_provider_path)
            # Check if the custom provider supports vp_stage parameter
            # 检查自定义 provider 是否支持 vp_stage（虚拟流水线阶段）参数
            has_vp_stage = "vp_stage" in inspect.signature(custom_model_provider).parameters
            if has_vp_stage:
                model = custom_model_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
            else:
                model = custom_model_provider(pre_process=pre_process, post_process=post_process)
            # Apply critic output layer if needed
            # critic 角色且处于流水线最后一级时替换输出层为价值头
            if post_process and role == "critic":
                model.output_layer = LinearForLastLayer(
                    input_size=model.config.hidden_size, output_size=1, config=model.config
                )
            return model

        return wrapped_model_provider

    # bridge 模式：通过 megatron.bridge 从 HF checkpoint 自动推导 Megatron 配置
    if args.megatron_to_hf_mode == "bridge":
        from megatron.bridge import AutoBridge

        # 从 HF 权重目录创建桥接器并生成 Megatron provider（不加载权重）
        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
        provider = bridge.to_megatron_provider(load_weights=False)
        # TODO: we should not manually set this...
        # 手动用命令行参数覆盖并行度设置
        provider.tensor_model_parallel_size = args.tensor_model_parallel_size
        provider.pipeline_model_parallel_size = args.pipeline_model_parallel_size
        provider.expert_model_parallel_size = args.expert_model_parallel_size
        provider.expert_tensor_parallel_size = args.expert_tensor_parallel_size
        provider.sequence_parallel = args.sequence_parallel
        if getattr(args, "decoder_first_pipeline_num_layers", None) is not None:
            # 指定流水线第一阶段的层数（非均匀切分）
            provider.num_layers_in_first_pipeline_stage = args.decoder_first_pipeline_num_layers
        if getattr(args, "decoder_last_pipeline_num_layers", None) is not None:
            # 指定流水线最后阶段的层数
            provider.num_layers_in_last_pipeline_stage = args.decoder_last_pipeline_num_layers
        # finalize 后 provider 才能真正构建模型
        provider.finalize()
        return provider.provide

    def model_provider(pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None) -> GPTModel:
        """Builds the model.

        If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

        Args:
            pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
            post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


        Returns:
            Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
        """
        # 是否使用 TransformerEngine 实现（fp8 训练等）
        use_te = args.transformer_impl == "transformer_engine"

        # Experimental loading arguments from yaml
        # 由命令行参数推导 Megatron 的 TransformerConfig
        config: TransformerConfig = core_transformer_config_from_args(args)

        if args.spec is not None:
            # 用户通过 --spec 指定了自定义层结构模块
            transformer_layer_spec = import_module(args.spec)
            # Allow the spec to be a function so that user can use customized Megatron easier.
            # 允许 spec 是函数，方便用户深度定制 Megatron 层结构
            if callable(transformer_layer_spec):
                transformer_layer_spec = transformer_layer_spec(args, config, vp_stage)
        else:
            if args.num_experts:
                # Define the decoder block spec
                # MoE 模型使用整个 decoder block 的 spec
                kwargs = {
                    "use_transformer_engine": use_te,
                }
                if vp_stage is not None:
                    kwargs["vp_stage"] = vp_stage
                transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)
            else:
                # Define the decoder layer spec
                # 稠密模型按是否用 TE 选择对应层 spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        num_experts=args.num_experts,
                        moe_grouped_gemm=args.moe_grouped_gemm,
                        qk_layernorm=args.qk_layernorm,
                        multi_latent_attention=args.multi_latent_attention,
                        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
                    )
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        num_experts=args.num_experts,
                        moe_grouped_gemm=args.moe_grouped_gemm,
                        qk_layernorm=args.qk_layernorm,
                        multi_latent_attention=args.multi_latent_attention,
                        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
                    )

        # 模型构建的上下文（默认空操作；fp8 参数收集时用 fp8_model_init 包裹）
        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                # 引入 TE 的 fp8 初始化上下文，使参数以 fp8 精度保存/收集
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                # Check if fp8_model_init supports preserve_high_precision_init_val
                # 若支持保留高精度初始值则开启（保证初始化质量）
                if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except Exception as e:
                # 缺少 TE 时无法满足 fp8 参数收集需求，直接报错
                raise RuntimeError(
                    "--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found."
                ) from e

        # 组装 GPTModel 的构造参数
        kwargs = {
            "config": config,
            "transformer_layer_spec": transformer_layer_spec,
            # 注意使用 padding 后的词表大小（与并行度对齐）
            "vocab_size": args.padded_vocab_size,
            "max_sequence_length": args.max_position_embeddings,
            "pre_process": pre_process,
            "post_process": post_process,
            "fp16_lm_cross_entropy": args.fp16_lm_cross_entropy,
            # 输出保持按 TP 分片（后续手动 gather）
            "parallel_output": True,
            # 输入输出词嵌入是否共享
            "share_embeddings_and_output_weights": not args.untie_embeddings_and_output_weights,
            "position_embedding_type": args.position_embedding_type,
            "rotary_percent": args.rotary_percent,
            "rotary_base": args.rotary_base,
            "rope_scaling": args.use_rope_scaling,
        }

        if vp_stage is not None:
            # 虚拟流水线阶段编号（VPP，Virtual Pipeline Parallelism）
            kwargs["vp_stage"] = vp_stage

        if args.mtp_num_layers:
            # 配置了 MTP（Multi-Token Prediction）层时构造对应 block spec
            from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

            mtp_kwargs = {
                "use_transformer_engine": use_te,
            }
            if vp_stage is not None:
                mtp_kwargs["vp_stage"] = vp_stage

            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, **mtp_kwargs)
            kwargs["mtp_block_spec"] = mtp_block_spec

        with build_model_context(**build_model_context_args):
            # 在构建上下文中实例化 GPT 模型
            model = GPTModel(**kwargs)

        if post_process and role == "critic":
            # critic 角色在最后一级流水线替换输出层为标量价值头
            model.output_layer = LinearForLastLayer(input_size=config.hidden_size, output_size=1, config=config)

        return model

    return model_provider


# 把原始 provider 包装一层：构建模型后按参数冻结规则处理
def wrap_model_provider_with_freeze(original_provider, args):
    def wrapped_provider(pre_process=True, post_process=True, vp_stage=None):
        # 兼容不支持 vp_stage 的旧 provider
        sig = inspect.signature(original_provider)
        if "vp_stage" in sig.parameters:
            model = original_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        else:
            model = original_provider(pre_process=pre_process, post_process=post_process)

        # 构建完成后应用参数冻结策略
        freeze_model_params(model, args)

        return model

    return wrapped_provider


# 按命令行配置冻结/解冻模型参数（用于部分参数微调，如仅训练 MoE 路由）
def freeze_model_params(model: GPTModel, args: argparse.Namespace):
    if args.only_train_params_name_list:
        # 白名单模式：先冻结所有参数，再解冻匹配的
        for name, param in model.named_parameters():
            param.requires_grad = False
            for pattern in args.only_train_params_name_list:
                if re.search(pattern, name):
                    # 命中白名单模式则恢复梯度
                    param.requires_grad = True
                    break

    if args.freeze_params_name_list:
        # 黑名单模式：冻结匹配到的参数
        for name, param in model.named_parameters():
            for pattern in args.freeze_params_name_list:
                if re.search(pattern, name):
                    param.requires_grad = False
                    break
