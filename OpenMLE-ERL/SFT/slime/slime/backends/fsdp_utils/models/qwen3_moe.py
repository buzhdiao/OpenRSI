# 导入 torch 主库
import torch
# 导入 nn 模块用于定义网络层
import torch.nn as nn
# 导入函数式接口 F（softmax 等算子）
import torch.nn.functional as F
# 导入 HF 实现的 Qwen3-MoE 专家 MLP 作为专家层
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP

# 导入自定义的融合算子：Gate/Up 投影、SiluAndMul 激活、Down 投影与 MoE 求和归约
from slime.backends.fsdp_utils.kernels.fused_experts import (
    DownProjFunction,
    GateUpProjFunction,
    MoeSumReduceFunction,
    SiluAndMulFunction,
)


# 融合专家计算实现：对全部专家用融合 Triton 算子按 token 分组执行 MoE 前向
def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
):
    # 校验隐藏维度与 w1（gate/up 融合权重）的输入维度一致
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    # 校验 top-k 权重与专家 id 形状一致
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    # 融合算子要求输入张量内存连续
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    # 当前融合内核仅支持 bfloat16 精度
    assert hidden_states.dtype in [torch.bfloat16]

    # 第一步：按 top-k 分派做 gate/up 融合投影（输出中间缓存 1）
    intermediate_cache1 = GateUpProjFunction.apply(
        hidden_states,
        w1,
        topk_weights,
        topk_ids,
    )
    # 第二步：SiLU 并与后半相乘的融合激活
    intermediate_cache2 = SiluAndMulFunction.apply(intermediate_cache1)
    # 第三步：按 top-k 做 down 投影（输出中间缓存 3）
    intermediate_cache3 = DownProjFunction.apply(
        intermediate_cache2,
        w2,
        topk_weights,
        topk_ids,
    )
    # 第四步：把同一 token 命中的多个专家输出求和归约回原形状
    output_hidden_states = MoeSumReduceFunction.apply(
        intermediate_cache3,
        hidden_states.shape,
    )
    # 返回最终聚合的隐藏状态
    return output_hidden_states


# 标准分发器：不做专家并行（EP）切分时原样透传专家 id
class StandardDispatcher:
    def __init__(self, num_experts: int, num_local_experts: int):
        # 单卡场景：专家并行（Expert Parallelism）规模为 1
        self.moe_ep_size = 1
        # 保存专家总数与本地专家数
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        # 本 rank 的 EP 编号为 0
        self.moe_ep_rank = 0
        # 全局专家 id -> 本地专家 id 的映射表，默认未启用
        self.local_expert_mapping = None

        if self.moe_ep_size > 1:
            # EP>1 时构建映射表：全局 id 映射到本地 id，非本地专家记为 -1
            self.local_expert_mapping = torch.full((self.num_experts,), -1, dtype=torch.int32, device="cuda")
            self.local_expert_mapping[
                self.moe_ep_rank * self.num_local_experts : (self.moe_ep_rank + 1) * self.num_local_experts
            ] = torch.arange(0, self.num_local_experts, dtype=torch.int32, device="cuda")

    # 分发 top-k 专家 id：EP 启用时把全局 id 翻译成本地 id
    def dispatch(self, topk_ids) -> torch.Tensor:
        if self.local_expert_mapping is not None:
            return self.local_expert_mapping[topk_ids]
        return topk_ids


# Qwen3-MoE 的稀疏 MoE 模块（自研实现，用融合算子替换 HF 的逐专家循环）
class Qwen3MoeSparseMoeBlock(nn.Module):
    # 类级共享的分发器与运行器，多个 MoE 层实例复用
    dispatcher = None
    runner = None

    def __init__(self, config):
        super().__init__()
        # 专家总数
        self.num_experts = config.num_experts
        # 每个 token 激活的专家数（top-k 路由）
        self.top_k = config.num_experts_per_tok
        # 是否对 top-k 路由权重做归一化
        self.norm_topk_prob = config.norm_topk_prob

        # gating
        # 路由门控：线性层把隐藏态映射到每个专家的 logits（无偏置）
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        # 构建专家列表：每个专家是标准 Qwen3MoeMLP（gate/up/down 三投影）
        self.experts = nn.ModuleList(
            [Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

        if Qwen3MoeSparseMoeBlock.dispatcher is None:
            # 类级分发器只创建一次：此处无 EP 切分，本地专家数等于总数
            Qwen3MoeSparseMoeBlock.dispatcher = StandardDispatcher(
                num_experts=config.num_experts, num_local_experts=config.num_experts
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 解包批次、序列长度与隐藏维度
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # 摊平为 [tokens, hidden] 的二维形式
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        # 门控计算每个 token 对各专家的 logits
        router_logits = self.gate(hidden_states)

        # softmax 得到路由权重（fp32 计算保证数值稳定）
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        # 取 top-k 专家及其权重
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            # 与 Mixtral 稀疏 MoE 的区别之一：可选的 top-k 权重归一化
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        # 权重转回输入精度类型
        routing_weights = routing_weights.to(hidden_states.dtype)

        # 经分发器翻译专家 id（当前为透传）
        selected_experts = Qwen3MoeSparseMoeBlock.dispatcher.dispatch(selected_experts)

        # 把所有专家的 gate/up 权重沿输出维拼接并堆叠为 [E, 2*I, H] 的融合权重张量
        w13_weight = torch.stack(
            [torch.cat([layer.gate_proj.weight, layer.up_proj.weight], dim=0) for layer in self.experts]
        )
        # 同样堆叠所有专家的 down 权重为 [E, H, I]
        w2_weight = torch.stack([layer.down_proj.weight for layer in self.experts], dim=0)

        # 调用融合专家实现完成 MoE 前向（输入转 bf16 以匹配内核要求）
        final_hidden_states = fused_experts_impl(
            hidden_states.to(torch.bfloat16),
            w13_weight,
            w2_weight,
            routing_weights,
            selected_experts,
        )

        # 返回聚合输出与路由 logits
        return final_hidden_states, router_logits


# 对 HF 的 Qwen3-MoE 应用"真 on-policy"补丁：把 HF 的稀疏 MoE 块整体替换为本模块的实现
def apply_true_on_policy_patch_for_qwen3_moe():
    from transformers.models.qwen3_moe import modeling_qwen3_moe

    # 用带融合算子的实现覆盖 HF 模型类，保证训练与推理的 MoE 数值路径一致（true on-policy）
    modeling_qwen3_moe.Qwen3MoeSparseMoeBlock = Qwen3MoeSparseMoeBlock
