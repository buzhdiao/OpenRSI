import logging

from megatron.training.arguments import parse_args as _megatron_parse_args
from megatron.training.arguments import validate_args as _megatron_validate_args
from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
from transformers import AutoConfig

__all__ = ["validate_args", "megatron_parse_args", "set_default_megatron_args"]

logger = logging.getLogger(__name__)


def validate_args(args):
    """Run megatron's own validate_args plus slime-specific megatron validations."""
    # 先执行 Megatron 自带的参数校验逻辑
    _megatron_validate_args(args)

    # always use varlen
    # 强制使用变长序列（variable sequence length），slime 的训练数据通常长度不齐
    args.variable_seq_lengths = True
    # MoE 的 allgather 分发器不支持变长序列，需要自动切换到 alltoall
    if getattr(args, "moe_token_dispatcher_type", None) == "allgather":
        logger.info(
            "--moe-token-dispatcher-type allgather does not support variable sequence length, "
            "please use alltoall dispatcher instead."
        )
        # 将 token 分发器类型改为 alltoall，以兼容变长序列
        args.moe_token_dispatcher_type = "alltoall"

    # 流水线并行大小为 1 时，不允许设置首/尾流水线阶段层数参数
    if args.pipeline_model_parallel_size == 1:
        assert args.decoder_first_pipeline_num_layers is None and args.decoder_last_pipeline_num_layers is None, (
            "decoder_first_pipeline_num_layers and decoder_last_pipeline_num_layers should be None when "
            "pipeline_model_parallel_size is 1."
        )


def _hf_validate_args(args, hf_config):
    # 内部辅助函数：判断两个值是否相等
    def equal(x, y):
        return x == y

    # 收集所有不一致的错误信息
    errors = []

    # multimodal models have different config structure
    # 多模态模型（如 VLM）的配置结构不同，真实文本模型配置嵌套在 text_config 里
    if hasattr(hf_config, "text_config"):
        hf_config = hf_config.text_config

    # 逐项对比 HF 配置与 Megatron 配置中的关键超参数是否一致
    # （其中 HF 的 tie_word_embeddings 与 Megatron 的 untie_embeddings_and_output_weights 语义相反）
    for hf_config_name, megatron_config_name, compare_fn in [
        ("hidden_size", "hidden_size", equal),
        ("num_attention_heads", "num_attention_heads", equal),
        ("num_hidden_layers", "num_layers", equal),
        ("intermediate_size", "ffn_hidden_size", equal),
        ("tie_word_embeddings", "untie_embeddings_and_output_weights", lambda x, y: not x == y),
        ("rms_norm_eps", "norm_epsilon", equal),
        ("rope_theta", "rotary_base", equal),
    ]:
        # 只有 HF 配置中存在该字段时才做对比
        if hasattr(hf_config, hf_config_name):
            # 用对应的比较函数检查两侧取值是否匹配，不匹配则记录错误
            if not compare_fn(getattr(hf_config, hf_config_name), getattr(args, megatron_config_name)):
                errors.append(
                    f"{hf_config_name} in hf config {getattr(hf_config, hf_config_name)} is not equal to "
                    f"{megatron_config_name} {getattr(args, megatron_config_name)}, please check the config."
                )

    # 存在任何不一致时直接断言失败，避免用错误配置训练
    if len(errors) > 0:
        raise AssertionError("hf_validate_args failed: " + "; ".join(errors))


def _set_default_megatron_args(args):
    # always use zero optimizer
    # 强制使用 ZeRO-1 风格的分布式优化器（optimizer state 分片），节省显存
    args.use_distributed_optimizer = True
    # TODO: maybe change this after megatron has good fp8 support
    # 默认使用 bf16 精度；仅在显式指定 fp16 时才用 fp16
    args.bf16 = not args.fp16
    # placeholders
    # 序列长度未指定时给一个占位默认值 4096（实际值由数据侧决定）
    if args.seq_length is None:
        args.seq_length = 4096
    # 最大位置编码长度与序列长度保持一致
    args.max_position_embeddings = args.seq_length
    # TODO: revisit this when megatron(dev) have solved the optimizer-cpu-offload ckpt saving bug
    # 使用 mcore 0.14 之前的旧式分布式 checkpoint 保存格式，规避优化器 CPU offload 的保存 bug
    args.dist_ckpt_save_pre_mcore_014 = True
    # compatible for megatron
    # RoPE 类型未指定时按注意力机制给默认值：MLA（DeepSeek 式多隐注意力）用 yarn，否则用普通 rope
    if hasattr(args, "rope_type") and args.rope_type is None:
        args.rope_type = "yarn" if args.multi_latent_attention else "rope"

    # 给定了词表大小但未给填充后词表大小时，按并行度对齐计算填充大小
    if args.vocab_size and not args.padded_vocab_size:
        args.padded_vocab_size = _vocab_size_with_padding(args.vocab_size, args)

    # 未指定 tokenizer 模型与类型时，直接复用 HF checkpoint 作为 tokenizer
    if not args.tokenizer_model and not args.tokenizer_type:
        logger.info("--tokenizer-model not set, use --hf-checkpoint as tokenizer model.")
        args.tokenizer_model = args.hf_checkpoint
        args.tokenizer_type = "HuggingFaceTokenizer"
    return args


# Public alias for external tools (e.g. convert_hf_to_torch_dist.py)
set_default_megatron_args = _set_default_megatron_args


def megatron_parse_args(extra_args_provider, skip_hf_validate=False):
    """Parse megatron args, validate HF config, and set defaults."""
    # 调用 Megatron 的参数解析，忽略未知参数（slime 自身会传入额外参数）
    args = _megatron_parse_args(extra_args_provider=extra_args_provider, ignore_unknown_args=True)

    # 指定了 HF checkpoint 且未跳过校验时，加载 HF 配置并做一致性检查
    if args.hf_checkpoint and not skip_hf_validate:
        hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        _hf_validate_args(args, hf_config)

    # rank/world_size 仅为占位：Megatron 需要这两个字段存在，真正的并行初始化由 slime 控制
    args.rank = 0
    # 用 actor 的节点数乘以每节点 GPU 数作为 world_size（与 actor 训练规模一致）
    args.world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
    # 填充上述默认值后返回
    args = _set_default_megatron_args(args)
    return args
