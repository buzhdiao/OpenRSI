# 引入 aiohttp 异步 HTTP 客户端，用于请求教师模型（RM 服务）
import aiohttp
# 引入 torch，用于把教师模型 log-probs 组织成张量
import torch

# 引入样本数据结构
from slime.utils.types import Sample


# 奖励函数：调用教师模型（RM 服务）获取 token 级 log-probs（on-policy distillation 的教师信号来源）
async def reward_func(args, sample, **kwargs):
    # 构造发往教师模型的 JSON 请求体（含整条序列的 token id、占位采样参数与 logprob 返回选项）
    payload = {
        # "text": sample.prompt + sample.response,
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    # HTTP 会话附加参数（暂为空）
    session_kwargs = {}
    # 创建异步 HTTP 会话
    async with aiohttp.ClientSession(**session_kwargs) as session:
        # 向教师服务地址 POST JSON 载荷
        async with session.post(args.rm_url, json=payload) as resp:
            # 非 2xx 状态码时抛出异常
            resp.raise_for_status()
            # 解析并返回 JSON 响应（含 token 级 logprobs）
            return await resp.json()


# 后处理教师模型奖励：提取教师 log-probs 并折算成标量奖励
def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the reward response (which contains sglang's logprob output)
    2. Trims them to match the response length
    3. Stores them in sample.teacher_log_probs for OPD KL penalty computation
    4. Returns scalar rewards (0.0 for pure distillation) compatible with GRPO/PPO

    Note: The reward_func calls the teacher server which returns token-level log-probs.
    For pure on-policy distillation without task rewards, we return 0.0 for each sample.
    The actual learning signal comes from the OPD KL penalty applied in compute_advantages_and_returns.
    """
    # 取出每个样本从 RM 响应解析出的原始奖励（此处为完整的 logprob 响应）
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    # 取出每个样本的响应长度（用于截断教师 log-probs）
    response_lengths = [sample.response_length for sample in samples]

    # 从 sglang 响应中提取教师模型 log-probs
    # Extract teacher log-probs from the sglang response
    teacher_log_probs = [
        torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
        for reward in raw_rewards
    ]
    # 右对齐截取最后 response_length 个 log-prob，只保留 response 段（去掉 prompt 前缀部分）
    teacher_log_probs = [
        t_log_prob[-response_length:]
        for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
    ]

    # 把教师 log-probs 写回对应样本，供 OPD（on-policy distillation）KL 惩罚计算使用
    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    # Return scalar rewards for GRPO/PPO advantage estimator
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    # 纯蒸馏场景下任务奖励全为 0，学习信号完全来自 KL 惩罚项
    scalar_rewards = [0.0] * len(samples)

    # 返回两份相同的标量奖励（分别用于优势估计与日志记录）
    return scalar_rewards, scalar_rewards
