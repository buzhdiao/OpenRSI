# FLOPs 估算工具:按 Transformer 各子层(投影、注意力、MLP、LM head)分别累乘,
# 估算一次前向的计算量;矩阵乘法按 2*M*N*K 计(乘加各算一次浮点运算)。
def calculate_embedding_flops(seqlen, hidden_size):
    # 词嵌入查表本身无乘法,这里按 2*seqlen*hidden_size 计入一次线性变换的开销
    return 2 * seqlen * hidden_size


def calculate_lm_head_flops(seqlen, hidden_size, vocab_size):
    # 输出投影到词表:对每个 token 做 hidden_size -> vocab_size 的矩阵乘,即 2*s*h*v
    return 2 * seqlen * hidden_size * vocab_size


def calculate_qkv_projection_flops(args, seqlen, hidden_size, num_attention_heads, num_query_groups):
    if args.q_lora_rank is None:
        # 标准 MHA/GQA:Q 投影为 hidden_size -> num_heads*kv_channels 的矩阵乘
        q_flops = 2 * seqlen * hidden_size * num_attention_heads * args.kv_channels
    else:
        # MLA(DeepSeek 式低秩注意力):Q 先降维到 q_lora_rank,再升维到 qk 头维(含 RoPE 位置维度)
        q_flops = (
            2
            * seqlen
            * args.q_lora_rank
            * (args.hidden_size + args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim))
        )
    if args.kv_lora_rank is None:
        # 标准 MHA/GQA:K、V 两次投影,各为 hidden_size -> num_query_groups*kv_channels(系数中的 2*2)
        kv_flops = 2 * 2 * seqlen * hidden_size * num_query_groups * args.kv_channels
    else:
        # MLA:K/V 共享低秩压缩,再联合升维并加上解耦 RoPE 的位置编码投影
        kv_flops = (
            2
            * seqlen
            * (
                args.kv_lora_rank
                * (args.hidden_size + args.num_attention_heads * (args.qk_head_dim + args.v_head_dim))
                + args.hidden_size * args.qk_pos_emb_head_dim
            )
        )

    # Q 投影与 KV 投影的 FLOPs 之和
    return q_flops + kv_flops


def calculate_attention_flops(args, seqlen, num_attention_heads):
    # QK^T with causal
    # 注意力得分 QK^T:每头做 s*s*(头维) 的矩阵乘;因果掩码只算一半(/2)
    if args.qk_pos_emb_head_dim:
        # MLA:Q 侧头维还需加上解耦 RoPE 的位置维度
        flops = 2 * num_attention_heads * seqlen * seqlen * (args.qk_head_dim + args.qk_pos_emb_head_dim) / 2
    else:
        # 标准注意力:直接使用 kv_channels 作为 QK 头维
        flops = 2 * num_attention_heads * seqlen * seqlen * args.kv_channels / 2
    # A*V
    # 注意力权重与 V 的加权和:每头做 s*s*(V 头维) 的矩阵乘(因果掩码按一半计,系数 2/2=1)
    if args.v_head_dim:
        # MLA:V 头维由 v_head_dim 单独指定
        flops += num_attention_heads * seqlen * seqlen * args.v_head_dim
    else:
        # 标准注意力:V 头维即 kv_channels
        flops += num_attention_heads * seqlen * seqlen * args.kv_channels
    # 返回注意力核心计算的 FLOPs
    return flops


def calculate_output_flops(seqlen, hidden_size):
    # 注意力输出投影:hidden_size -> hidden_size 的矩阵乘
    return 2 * seqlen * hidden_size * hidden_size


def calculate_mlp_flops(seqlen, hidden_size, ffn_hidden_size):
    # MLP:三个线性变换(hidden->ffn、ffn->hidden 及门控),故系数为 3
    return 2 * seqlen * hidden_size * ffn_hidden_size * 3


def calculate_layer_flops(args, seqlen, hidden_size, num_attention_heads, num_query_groups, ffn_hidden_size):
    # 单个 Transformer 层 = QKV 投影 + 注意力核心 + 输出投影 + MLP
    return (
        calculate_qkv_projection_flops(args, seqlen, hidden_size, num_attention_heads, num_query_groups)
        + calculate_attention_flops(args, seqlen, num_attention_heads)
        + calculate_output_flops(seqlen, hidden_size)
        + calculate_mlp_flops(seqlen, hidden_size, ffn_hidden_size)
    )


def calculate_fwd_flops(
    seqlens,
    args,
):
    # 取出模型结构相关的超参数
    hidden_size = args.hidden_size
    num_attention_heads = args.num_attention_heads
    num_query_groups = args.num_query_groups
    vocab_size = args.vocab_size

    # 累计的总 FLOPs 计数器
    total_flops = 0

    # 稠密层的 FFN 隐层宽度
    dense_ffn = args.ffn_hidden_size
    if args.num_experts is None:
        # 纯稠密模型:所有层都是稠密层,没有 MoE 层
        num_dense_layers = args.num_layers
        num_moe_layers = 0
    else:
        # MoE 模型:读取共享专家的 FFN 宽度(未配置则为 0)
        shared_expert_ffn = getattr(args, "moe_shared_expert_intermediate_size", None)
        if shared_expert_ffn is None:
            shared_expert_ffn = 0

        # MoE 层的有效 FFN 宽度 = topk 个专家宽度之和 + 共享专家宽度
        moe_ffn = args.moe_ffn_hidden_size * args.moe_router_topk + shared_expert_ffn
        if hasattr(args, "moe_layer_freq"):
            if isinstance(args.moe_layer_freq, list):
                # 列表形式:逐层标记,统计其中稠密层(0)与 MoE 层(>0)的数量
                num_dense_layers = sum(1 for freq in args.moe_layer_freq if freq == 0)
                num_moe_layers = sum(1 for freq in args.moe_layer_freq if freq > 0)
            else:
                # 整数形式:每 moe_layer_freq 层一个 MoE 层(以第 0 层为 MoE),其余为稠密层
                num_dense_layers = sum(1 for i in range(args.num_layers) if i % args.moe_layer_freq != 0)
                num_moe_layers = sum(1 for i in range(args.num_layers) if i % args.moe_layer_freq == 0)
        else:
            # 未配置 moe_layer_freq 时视为全部层都是 MoE 层
            num_dense_layers = 0
            num_moe_layers = args.num_layers

    for seqlen in seqlens:
        # 对每个序列长度(每个样本)分别累加
        if num_dense_layers > 0:
            # 稠密层的总 FLOPs = 单层 FLOPs × 稠密层层数
            total_flops += (
                calculate_layer_flops(
                    args,
                    seqlen,
                    hidden_size,
                    num_attention_heads,
                    num_query_groups,
                    dense_ffn,
                )
                * num_dense_layers
            )

        if num_moe_layers > 0:
            # MoE 层的总 FLOPs = 单层 FLOPs(按激活的专家宽度折算)× MoE 层层数
            total_flops += (
                calculate_layer_flops(
                    args,
                    seqlen,
                    hidden_size,
                    num_attention_heads,
                    num_query_groups,
                    moe_ffn,
                )
                * num_moe_layers
            )

        # 每个序列还要加上 LM head(输出词表投影)的 FLOPs
        total_flops += calculate_lm_head_flops(seqlen, hidden_size, vocab_size)

    # 返回本次前向的总 FLOPs 估算值
    return total_flops
