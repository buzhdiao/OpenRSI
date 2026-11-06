# 引入 asyncio，用于并发任务调度与信号量限流
import asyncio
# 引入 copy，用于深拷贝评测样本
import copy
# 引入 inspect，用于检查自定义生成函数签名
import inspect
# 引入日志模块
import logging
# 引入 uuid，为样本生成会话 ID
import uuid
# 引入 Namespace 类型标注（命令行参数对象）
from argparse import Namespace
# 引入 Callable 类型标注（数据源取数函数）
from collections.abc import Callable
# 引入 contextmanager，构造 DP rank 上下文管理器
from contextlib import contextmanager
# 引入 Any 类型标注
from typing import Any

# 引入 numpy，用于解码专家路由数据
import numpy as np
# 引入 pybase64，用于高速 base64 解码
import pybase64
# 引入 sglang_router（仅用于版本号判断）
import sglang_router
# 引入版本解析工具，比较 router 版本
from packaging.version import parse
# 引入 tqdm 进度条
from tqdm import tqdm

# 引入 rollout 输出类型（训练/评测输出包装）
from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
# 引入动态过滤器调用工具与指标收集器
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
# 引入同步阻塞运行异步协程的工具
from slime.utils.async_utils import run
# 引入数据集封装（评测 prompt 数据集）
from slime.utils.data import Dataset
# 引入评测数据集配置类型
from slime.utils.eval_config import EvalDatasetConfig
# 引入 HTTP GET/POST 工具
from slime.utils.http_utils import get, post
# 引入单例元类与动态函数加载工具
from slime.utils.misc import SingletonMeta, load_function
# 引入多模态处理器相关工具（kwargs 构建、图像编码、加载 processor/tokenizer）
from slime.utils.processing_utils import (
    build_processor_kwargs,
    encode_image_for_rollout_engine,
    load_processor,
    load_tokenizer,
)
# 引入样本数据结构
from slime.utils.types import Sample

# 引入奖励模型接口：单样本与批量异步 RM
from .rm_hub import async_rm, batched_async_rm

# 声明本模块对外暴露的接口
__all__ = ["generate_rollout"]

# 获取本模块专属 logger
logger = logging.getLogger(__name__)


# 生成过程的全局状态（单例）：持有 tokenizer/信号量/采样参数等持久配置
class GenerateState(metaclass=SingletonMeta):
    """
    The global state for the generation process.
    """

    # 初始化：加载模型组件并配置采样参数
    def __init__(self, args: Namespace) -> None:
        # persistent state for the generation process
        # 保存全局参数
        self.args = args
        # 加载 tokenizer（信任远程代码）
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        # 加载 processor（多模态用，可为 None）
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

        # 创建并发信号量：总并发 = 单引擎并发数 × 引擎数（GPU 总数 / 每引擎 GPU 数）
        self.semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        # 构建全局采样参数字典（训练 rollout 的默认解码配置）：
        # 含采样温度、nucleus 采样 top_p、top_k、最大新生成 token 数、停止字符串与停止 token id、
        # 特殊 token 文本跳过开关、停止后不截断（no_stop_trim）以及特殊 token 间不插空格
        self.sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )

        # 启用确定性推理时：为组内每个样本分配固定采样种子
        if getattr(args, "sglang_enable_deterministic_inference", False):
            # 以 rollout 种子为基准
            sampling_seed_base = args.rollout_seed
            # 组内第 i 个样本的种子 = 基准 + i（保证组内可复现且互不相同）
            self.group_sampling_seeds = [sampling_seed_base + i for i in range(args.n_samples_per_prompt)]

        # dp rank balancing
        # 各数据并行 rank 的在途负载计数（用于负载均衡）
        self.dp_counts = [0] * (args.sglang_dp_size or 1)
        # 当前选中的 DP rank
        self.dp_rank = 0

        # 初始化滚动状态（剩余批次/待完成任务/中止标志）
        self.reset()

    # 上下文管理器：为一次生成任务挑选负载最小的 DP rank，退出时归还计数
    @contextmanager
    def dp_rank_context(self):
        # 找出当前负载最小的所有 DP rank 候选
        candidates = [i for i, count in enumerate(self.dp_counts) if count == min(self.dp_counts)]
        # 在候选中随机挑一个（避免都挤到同一个最小值）
        dp_rank = int(np.random.choice(candidates))
        # 该 rank 在途计数加一
        self.dp_counts[dp_rank] += 1
        # 记录当前 rank
        self.dp_rank = dp_rank
        # 进入上下文：把 rank 暴露给调用方
        try:
            yield dp_rank
        finally:
            # 退出时在途计数减一
            self.dp_counts[dp_rank] -= 1
            # 校验计数不会减成负数
            assert self.dp_counts[dp_rank] >= 0

    # 重置滚动状态：清零剩余批次、待完成任务与中止标志
    def reset(self) -> None:
        # 本轮仍需要的样本组数
        self.remaining_batch_size = 0
        # 在途的生成任务集合
        self.pendings = set()
        # 中止标志复位
        self.aborted = False

    # 把一批样本组提交为异步生成任务
    def submit_generate_tasks(self, samples: list[list[Sample]]) -> None:
        # 逐组创建任务
        for group in samples:
            # 把任务加入在途集合
            self.pendings.add(
                asyncio.create_task(
                    # submit a group of samples as a single task.
                    # 每个样本组作为一个整体任务（组内共享采样参数副本）
                    generate_and_rm_group(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )
        # 累计剩余需要完成的组数
        self.remaining_batch_size += len(samples)


# 单样本生成：经 sglang router 以 token 级 HTTP 工作流完成一次推理
async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """Generate using traditional SGLang router with token-based workflow"""
    # CI 测试模式下断言 prompt 必须是字符串
    if args.ci_test:
        assert isinstance(sample.prompt, str)

    # 获取生成全局状态（单例）
    state = GenerateState(args)
    # 拼 sglang router 的 /generate 接口地址
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # 校验样本状态必须为待生成或已中止（可续写）
    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    # 多模态路径：用 processor 编码文本与图像
    if state.processor:
        # 根据多模态输入构建 processor 参数
        processor_kwargs = build_processor_kwargs(sample.multimodal_inputs)
        # 调用 processor 得到分词结果与多模态特征
        processor_output = state.processor(text=sample.prompt, **processor_kwargs)
        # 取输入 token id 序列
        prompt_ids = processor_output["input_ids"][0]
        # 除 input_ids/attention_mask 外的字段作为多模态训练输入保存（无则置 None）
        sample.multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    # 纯文本路径：直接用 tokenizer 编码
    else:
        prompt_ids = state.tokenizer.encode(sample.prompt, add_special_tokens=False)

    # 已有部分响应（partial rollout 续写）时：按已有 token 数扣减本次最大生成量
    if len(sample.response) > 0:
        sampling_params["max_new_tokens"] -= len(sample.tokens) - len(prompt_ids)

    # 校验剩余可生成 token 数不能为负
    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    # 已无生成空间：标记为截断并直接返回
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    # Prepare payload for sglang server
    # 组装发往 sglang 服务的请求体：采样参数 + 返回 logprob
    payload = {
        "sampling_params": sampling_params,
        "return_logprob": True,
    }

    # 启用 rollout 路由回放时：请求返回被路由到的专家
    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True

    # 多模态样本携带图像时：把图像编码后放进请求体
    if sample.multimodal_inputs and sample.multimodal_inputs["images"]:
        # 取图像列表
        image_data = sample.multimodal_inputs["images"]
        # 逐张编码（base64 等引擎友好格式）后写入载荷
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    # Use existing tokens for multi-turn or tokenize the new prompt
    # 已有响应的多轮场景：直接复用完整 token 序列
    if len(sample.response) > 0:
        payload["input_ids"] = sample.tokens
    # 首轮生成：使用刚编码的 prompt ids
    else:
        payload["input_ids"] = prompt_ids
        if not sample.tokens:  # Initialize sample.tokens for the first turn
            # 首轮时初始化样本的 token 序列
            sample.tokens = prompt_ids

    # Use session_id for consistent hashing routing if router uses consistent_hashing policy
    # 请求头默认为空
    headers = None
    # 路由策略为一致性哈希且样本有会话 ID 时：用会话 ID 作路由键（保证同会话落到同一引擎，提高前缀缓存命中）
    if args.sglang_router_policy == "consistent_hashing" and sample.session_id:
        headers = {"X-SMG-Routing-Key": sample.session_id}

    # 异步 POST 到 router 完成生成
    output = await post(url, payload, headers=headers)

    # 使用 slime 自带 radix tree 中间件时：走其后处理路径（可拿到前缀缓存等增强信息）
    if args.use_slime_router and "RadixTreeMiddleware" in args.slime_router_middleware_paths:
        # 延迟导入 radix tree 后处理函数
        from slime.router.middleware_hub.radix_tree_middleware import postprocess_sample_with_radix_tree

        # 用中间件结果回填样本
        sample = await postprocess_sample_with_radix_tree(args, sample, output)
    # 常规路径：直接解析 sglang 输出
    else:
        # 输出含 token 级 logprob 时逐 token 解析
        if "output_token_logprobs" in output["meta_info"]:
            # 取每个 (logprob, token_id) 对中的 token id
            new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            # 取对应的 logprob 值
            new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        # 无 logprob 输出则置空
        else:
            new_response_tokens, new_response_log_probs = [], []

        # Update sample with tokens directly - avoiding re-tokenization
        # 直接在原 token 序列上追加新 token（避免重新分词）
        sample.tokens = sample.tokens + new_response_tokens
        # 累计响应长度
        sample.response_length += len(new_response_tokens)
        # 追加响应文本
        sample.response += output["text"]

        # When partial rollout and masking off policy is enabled, update the loss mask
        # 已有 loss mask 的 partial rollout 样本：新续写部分标记为监督位 1
        if sample.loss_mask is not None:
            # 校验必须在 partial rollout + offpolicy mask 开启时才会走到这里
            assert args.partial_rollout and args.mask_offpolicy_in_partial_rollout
            # 追加与新 token 等长的监督标记
            sample.loss_mask += [1] * len(new_response_tokens)

        # 初始化 rollout logprob 列表（首次）
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        # 追加本次生成的 logprob
        sample.rollout_log_probs += new_response_log_probs

    # 服务返回了被路由的专家信息时：解码并重塑为 (序列长-1, 层数, topk) 张量存入样本
    if "routed_experts" in output["meta_info"]:
        sample.rollout_routed_experts = np.frombuffer(
            pybase64.b64decode(output["meta_info"]["routed_experts"].encode("ascii")),
            dtype=np.int32,
        ).reshape(
            len(sample.tokens) - 1,
            args.num_layers,
            args.moe_router_topk,
        )

    # 按服务元信息回填样本的其他字段（如 finish reason 等）
    sample.update_from_meta_info(args, output["meta_info"])

    # 返回更新后的样本
    return sample


# 单样本"生成+奖励"流水线：负责部分回滚掩码、状态检查、限流与奖励计算
async def generate_and_rm(
    args: Namespace,
    sample: Sample | list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    # mask previous off-policy generation for partial rollout
    # partial rollout 且开启 off-policy 掩码时：把已有响应段的 loss mask 全部置 0（旧生成视为 off-policy）
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length

    # For samples with existing response, check if they're complete
    # 样本已完成或已截断：无需再生成，直接返回
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        # 校验已有响应文本
        assert sample.response is not None
        # 非 group RM 模式下必须已有奖励
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    # 获取生成全局状态
    state = GenerateState(args)

    # generate
    # 信号量限流：同一时刻在途请求数不超过全局并发上限
    async with state.semaphore:
        # 已被中止：标记状态后直接返回
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        # 选定 DP rank（负载均衡）后执行生成
        with state.dp_rank_context() as _:
            # Check sample.generate_function_path for per-sample custom_generate_function_path (e.g., from eval dataset config)
            # 优先样本级自定义生成函数路径，其次全局配置
            custom_func_path = getattr(sample, "generate_function_path", None) or args.custom_generate_function_path

            # 配置了自定义生成函数：动态加载并调用
            if custom_func_path is not None:
                custom_generate_func = load_function(custom_func_path)
                # if signature has evaluation, pass evaluation
                # 函数签名带 evaluation 参数时把评测标志传入
                if "evaluation" in inspect.signature(custom_generate_func).parameters:
                    sample = await custom_generate_func(args, sample, sampling_params, evaluation=evaluation)
                # 否则只传基础三参
                else:
                    sample = await custom_generate_func(args, sample, sampling_params)
            # 默认走内置 sglang 生成
            else:
                sample = await generate(args, sample, sampling_params)

    # for the rm that need the whole group, we will not do the rm here
    # group RM 模式：奖励统一延后到组级计算
    if args.group_rm:
        return sample

    # multi samples
    # 输入是样本列表（多智能体等场景）：批量处理奖励
    if isinstance(sample, list):
        samples = sample
        # 组内任一样本被中止：直接返回不再算奖励
        if any([sample.status == Sample.Status.ABORTED for sample in samples]):
            return samples

        # for multi agent system, the reward of some sample is calculated during generation.
        # 只对尚未有奖励的样本批量调用 RM
        samples_need_reward = [sample for sample in samples if sample.reward is None]
        rewards = await batched_async_rm(args, samples_need_reward)
        # 把奖励写回各样本
        for sample, reward in zip(samples_need_reward, rewards, strict=False):
            sample.reward = reward
        return samples
    # 单样本情形
    else:
        # 已中止的样本直接返回
        if sample.status == Sample.Status.ABORTED:
            return sample
        # for multi-turn environment, a reward could be assigned to the agent.
        # 尚无奖励时调用单样本 RM（多轮环境的奖励可能已在生成期间赋值）
        if sample.reward is None:
            sample.reward = await async_rm(args, sample)

    # 返回带奖励的样本
    return sample


# 组级"生成+奖励"：为组内样本分配会话/奖励组标识，并发生成后可选做组级 RM
async def generate_and_rm_group(
    args: Namespace, group: list[Sample], sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Sample]:
    # 获取生成全局状态
    state = GenerateState(args)

    # 已整体中止：原样返回该组
    if state.aborted:
        return group

    # Generate a unique session_id for each sample in the group
    # 为组内每个缺会话 ID 的样本生成唯一 session_id
    for sample in group:
        if sample.session_id is None:
            sample.session_id = str(uuid.uuid4())

    # The reward/statistics barrier needs a stable identity for the current
    # group. Assign the group ID, member count, and member index before
    # asynchronous generation so every sandbox result returns to the same
    # barrier and the expected member count can be reached.
    # 先尝试沿用元数据中已有的奖励组 ID（保持跨轮一致）
    existing_reward_group_id = None
    for sample in group:
        metadata = getattr(sample, "metadata", None)
        # 任一成员已带奖励组 ID 即复用
        if isinstance(metadata, dict) and metadata.get("reward_group_id"):
            existing_reward_group_id = str(metadata["reward_group_id"])
            break

    # 没有既有 ID 时用首样本的组号/样本号加随机数构造稳定 ID
    if existing_reward_group_id is None:
        first_sample = group[0] if group else None
        first_group_index = getattr(first_sample, "group_index", "none") if first_sample is not None else "none"
        first_sample_index = getattr(first_sample, "index", "none") if first_sample is not None else "none"
        existing_reward_group_id = f"reward-group:{first_group_index}:{first_sample_index}:{uuid.uuid4()}"

    # 期望成员数即组大小
    reward_group_expected_count = len(group)
    # 给每个成员写入奖励组三元信息（ID/成员序号/期望人数），供异步奖励屏障汇聚
    for member_index, sample in enumerate(group):
        if not isinstance(getattr(sample, "metadata", None), dict):
            sample.metadata = {}
        sample.metadata["reward_group_id"] = existing_reward_group_id
        sample.metadata["reward_group_member_index"] = member_index
        sample.metadata["reward_group_expected_count"] = reward_group_expected_count

    # 为组内每个样本创建生成任务
    tasks = []
    for idx, sample in enumerate(group):
        # 拷贝采样参数避免组间串扰
        current_sampling_params = sampling_params.copy()
        # 确定性推理时：按成员序号取固定种子写入采样参数
        if getattr(args, "sglang_enable_deterministic_inference", False):
            seed = state.group_sampling_seeds[idx]
            current_sampling_params["sampling_seed"] = seed
        tasks.append(
            asyncio.create_task(generate_and_rm(args, sample, current_sampling_params, evaluation=evaluation))
        )

    # 并发执行组内全部任务并收集结果
    group = await asyncio.gather(*tasks)

    # for the rm that need the whole group, we will do the rm here
    # 未中止且为 group RM 模式：在此对整组批量算奖励
    if not state.aborted and args.group_rm:
        rewards = await batched_async_rm(args, group)
        # 奖励写回各样本
        for sample, reward in zip(group, rewards, strict=False):
            sample.reward = reward

    # 返回完整样本组
    return group


# 中止当前 rollout：通知所有 worker 取消请求，并回收 partial rollout 的部分样本
async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    # 收集中止样本组的列表
    aborted_samples = []

    # 获取生成全局状态
    state = GenerateState(args)
    # 校验当前未处于中止状态
    assert not state.aborted
    # 置中止标志（新任务会被拒）
    state.aborted = True

    # 旧版 router 或使用 slime 自带 router：用 /list_workers 接口
    if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_slime_router:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        urls = response["urls"]
    # 新版 router：用 /workers 接口并解析字段
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        urls = [worker["url"] for worker in response["workers"]]

    # 记录即将被中止的 worker 列表
    logger.info(f"Abort request for {urls}")
    # 并发向所有 worker 发送 abort_all 请求
    await asyncio.gather(*[post(f"{url}/abort_request", {"abort_all": True}) for url in urls])

    # make sure all the pending tasks are finished
    # 计数器：统计回收的样本数
    count = 0
    # 等待所有在途任务结束（逐批回收已完成的）
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)

        # 非 partial rollout：只需等任务结束，无需回收
        if not args.partial_rollout:
            continue

        # for partial rollout, collect the partial samples into the data buffer
        # 逐个处理已完成任务
        for task in done:
            group = task.result()
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)
            count += len(group)

    # partial rollout 模式：记录回收的部分样本数
    if args.partial_rollout:
        logger.info(f"Collected {count} partial samples into the data buffer")

    # 返回回收的部分样本组
    return aborted_samples


# 训练 rollout 的异步主循环：动态采样直至凑满有效样本数
async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch

    Returns:
        tuple[RolloutFnTrainOutput, list[list[Sample]]]:
            - data: a list of groups of samples generated by the rollout, length equals `rollout_batch_size`
            - aborted_samples: any partial groups collected during abort when partial_rollout is enabled
    """
    # 训练必须启用全局共享数据集
    assert args.rollout_global_dataset

    # 获取生成全局状态
    state = GenerateState(args)

    # instantiate data filters
    # 加载动态采样过滤器（未配置则为 None，即不过滤）
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )

    # 创建过滤器丢弃原因的指标收集器
    metric_gatherer = MetricGatherer()

    # target_data_size is the total number of valid samples to get
    # 目标有效样本组数 = 配置的 rollout batch 大小
    target_data_size = args.rollout_batch_size

    # 通过过滤的有效样本组
    data = []
    # 所有生成过的样本组（含被过滤的）
    all_data = []
    # 是否打印首个样本日志
    do_print = True
    # 初始化进度条（按目标样本数 × 每组样本数）
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    # 主循环：直到凑满目标数量的有效样本组
    while len(data) < target_data_size:
        # 在途任务不足以覆盖目标时：继续从数据源取超额批次并提交
        while state.remaining_batch_size < target_data_size:
            # get samples from the buffer and submit the generation requests.
            # 从数据源取超额批次（over-sampling 为动态过滤留余量）
            samples = data_source(args.over_sampling_batch_size)
            # 提交生成任务
            state.submit_generate_tasks(samples)

        # wait for the generation to finish
        # 等待任一任务完成（FIRST_COMPLETED），更新在途集合
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        # 处理本轮完成的任务
        for task in done:
            group: list[Sample] = task.result()

            # 首个完成的组打印一次示例日志
            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
                )
                do_print = False

            # 校验返回的组大小与配置一致
            assert len(group) == args.n_samples_per_prompt
            # 无论是否通过过滤都记录到全量数据
            all_data.append(group)
            # 应用动态采样过滤器（如 zero-std 过滤）
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            # 未通过过滤：记录丢弃原因并扣减剩余计数，继续等下一组
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                state.remaining_batch_size -= 1
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            # 未凑满目标时收下该组并推进进度条
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    # 关闭进度条
    pbar.close()
    # 取最后一组的首样本用于日志展示
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        f"Finish rollout: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
    )

    # there are still some unfinished requests, abort them
    # 中止并回收尚未完成的请求（partial rollout 时返回部分样本）
    aborted_samples = await abort(args, rollout_id)

    # 校验最终凑齐了目标数量的样本组
    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    # 按样本编号排序有效数据（恢复与数据集一致的顺序）
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)
    # 全量数据同样排序
    all_samples = sorted(
        all_data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index
    )

    # reset the global state to prevent effects on the next rollout or eval.
    # 重置全局生成状态，避免影响下一轮
    state.reset()
    # 配置了样本过滤函数时：对有效数据应用过滤
    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    # There can be circumstances where users want to process all samples including filtered ones.
    # 配置了全样本处理函数时：对包含被过滤样本的全量数据调用
    if args.rollout_all_samples_process_path is not None:
        process_func = load_function(args.rollout_all_samples_process_path)
        process_func(args, all_samples, data_source)

    # 返回训练输出（样本 + 过滤指标）与中止回收的部分样本
    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples


# 评测 prompt 数据集缓存：键为数据集配置 + 模型 + 模板设置
EVAL_PROMPT_DATASET = {}


# 评测 rollout 入口：并发评测所有配置的数据集
async def eval_rollout(args: Namespace, rollout_id: int) -> tuple[dict[str, dict[str, list[Any]]], list[list[Sample]]]:
    # 评测不支持 group RM
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    # 各数据集的评测协程
    coros = []
    # 遍历配置的评测数据集
    for dataset_cfg in getattr(args, "eval_datasets", []) or []:
        coros.append(eval_rollout_single_dataset(args, rollout_id, dataset_cfg))
    # 并发执行全部数据集评测
    results_list = await asyncio.gather(*coros)
    # 汇总各数据集结果
    results = {}
    for r in results_list:
        results.update(r)
    # 返回评测输出（无中止样本）
    return RolloutFnEvalOutput(data=results), []


# 单个评测数据集的 rollout：构建数据集、批量生成并计算奖励
async def eval_rollout_single_dataset(
    args: Namespace, rollout_id: int, dataset_cfg: EvalDatasetConfig
) -> dict[str, dict[str, list[Any]]]:
    """An example to implement the eval_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        dataset_cfg: configuration of the dataset
    """
    # 评测不支持 group RM
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    # 声明使用模块级数据集缓存
    global EVAL_PROMPT_DATASET

    # 缓存键 = 数据集配置键 + 模型 checkpoint + 是否套用聊天模板
    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template)
    # 缓存未命中时构建数据集
    if cache_key not in EVAL_PROMPT_DATASET:
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[cache_key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        )
    # 取出（可能缓存的）数据集
    dataset = EVAL_PROMPT_DATASET[cache_key]

    # 构建本数据集的基础采样参数（以数据集配置为准）
    base_sampling_params = dict(
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    # 评测任务列表
    tasks = []
    # do multiple samples for eval prompts
    # 全局样本编号计数器
    sample_index = 0
    # 遍历数据集中每个 prompt 样本
    for _i, prompt_sample in enumerate(dataset.samples):
        # 每个 prompt 重复生成 n_samples_per_eval_prompt 次
        for j in range(dataset_cfg.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            # 深拷贝 prompt 模板保证各样本独立
            sample = copy.deepcopy(prompt_sample)
            # 分配全局唯一编号
            sample.index = sample_index
            sample_index += 1
            # 注入数据集配置的元数据（如评测指令约束）
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            # 透传数据集级自定义生成函数路径
            sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            # 默认共用基础采样参数
            sampling_params = base_sampling_params
            # 确定性推理时：每个重复样本用不同固定种子
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
            tasks.append(
                asyncio.create_task(
                    generate_and_rm(
                        args,
                        sample,
                        sampling_params=sampling_params,
                        evaluation=True,
                    )
                )
            )

    # 收集结果
    data = []
    # 是否打印首个样本日志
    do_print = True
    # 初始化进度条
    pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}", disable=not do_print)
    # 按完成顺序消费任务
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        # 首个样本打印一次示例日志
        if do_print:
            logger.info(
                "eval_rollout_single_dataset example data: "
                f"{[str(sample.prompt) + sample.response]} "
                f"reward={sample.reward}"
            )
            do_print = False
        # 自定义生成函数可能返回样本列表：展开收集
        if isinstance(sample, list):
            data.extend(sample)
        # 常规单样本直接追加
        else:
            data.append(sample)
        pbar.update(1)
    # 关闭进度条
    pbar.close()

    # 按样本编号排序，保证输出顺序稳定
    data.sort(key=lambda sample: sample.index)

    # 评测奖励键：数据集专属键优先，其次全局奖励键
    reward_key = args.eval_reward_key or args.reward_key
    # 组装该数据集的评测结果：奖励列表、截断标记与样本本体
    return {
        dataset_cfg.name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }
    }


# 同步入口：根据是否评测分派到 eval_rollout 或训练 rollout
def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to get and store samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        RolloutFnTrainOutput | RolloutFnEvalOutput: the output of the rollout
    """
    # 训练/评测都必须启用全局共享数据集
    assert args.rollout_global_dataset
    # 评测模式：运行评测并直接返回输出
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    # 训练模式：运行异步 rollout 主循环
    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
    # 把中止回收的部分样本回填到数据源缓冲区供复用
    data_source.add_samples(aborted_samples)
    # 返回训练输出
    return output
