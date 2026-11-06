# 引入 asyncio，用于异步并发调用奖励函数
import asyncio
# 引入日志模块
import logging
# 引入 random，用于指数退避抖动与 random 奖励类型
import random

# 引入 aiohttp，用于异步 HTTP 调用远程奖励模型服务
import aiohttp

# 获取本模块专属 logger
logger = logging.getLogger(__name__)

# 引入按点号路径动态加载函数的工具（用于加载自定义奖励函数）
from slime.utils.misc import load_function
# 引入样本数据结构
from slime.utils.types import Sample

# 引入各规则奖励实现：deepscaler 规则奖励
from .deepscaler import get_deepscaler_rule_based_reward
# 引入 F1 分数计算
from .f1 import f1_score
# 引入 GPQA 奖励计算
from .gpqa import compute_gpqa_reward
# 引入 DAPO 数学评分（以 compute_score 别名引入）
from .math_dapo_utils import compute_score as compute_score_dapo
# 引入 \\boxed{} 答案抽取函数
from .math_utils import extract_answer as extract_boxed_answer
# 引入 verl 风格的数学答案判分函数
from .math_utils import grade_answer_verl

# 模块级共享的 HTTP 会话（避免每次请求都重建连接池）
_shared_session: aiohttp.ClientSession | None = None


# 获取（懒创建的）共享 aiohttp 会话
def _get_shared_session() -> aiohttp.ClientSession:
    # 声明使用模块级全局变量
    global _shared_session
    # 会话不存在或已关闭时重建
    if _shared_session is None or _shared_session.closed:
        # 创建 TCP 连接器：最大并发 64 并启用优雅关闭清理
        connector = aiohttp.TCPConnector(
            limit=64,
            enable_cleanup_closed=True,
        )
        # 设置整请求 120 秒总超时
        timeout = aiohttp.ClientTimeout(total=120)
        # 用上述连接器与超时创建新会话
        _shared_session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    # 返回共享会话
    return _shared_session


# 远程奖励模型调用：带指数退避重试地 POST 到 args.rm_url
async def remote_rm(args, sample: Sample, max_retries: int = 10):
    # 组装请求体：prompt、response 与标签
    payload = {
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
    }
    # 获取共享 HTTP 会话
    session = _get_shared_session()
    # 最多重试 max_retries 次
    for attempt in range(max_retries):
        # 尝试发起 POST 请求
        try:
            # 异步 POST JSON 载荷到远程 RM 地址
            async with session.post(args.rm_url, json=payload) as resp:
                # 非 2xx 状态码时抛出异常
                resp.raise_for_status()
                # 成功则解析 JSON 并返回
                return await resp.json()
        # 请求异常（网络/超时/状态码）时进入重试逻辑
        except Exception as e:
            # 已达最大重试次数：记录警告后向上抛出
            if attempt + 1 >= max_retries:
                logger.warning(f"remote_rm failed after {attempt + 1} attempts: {e}")
                raise
            # 计算退避时长：2^attempt 秒封顶 30 秒，再加随机抖动避免惊群
            backoff = min(2 ** attempt, 30) + random.random()
            # 记录重试日志（异常类型、退避时长、当前进度）
            logger.info(f"remote_rm: {type(e).__name__}, retrying in {backoff:.1f}s ({attempt + 1}/{max_retries})")
            # 异步休眠指定退避时长
            await asyncio.sleep(backoff)


# 通用异步奖励入口：支持自定义 RM 或按 rm_type 分发到各规则奖励实现
async def async_rm(args, sample: Sample, **kwargs):
    # 配置了自定义奖励函数路径时优先使用
    if args.custom_rm_path is not None:
        # 动态加载用户自定义奖励函数
        rm_function = load_function(args.custom_rm_path)
        # 调用并返回其结果
        return await rm_function(args, sample, **kwargs)

    # 取样本元数据（非 dict 时退化为空字典）
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    # 确定 RM 类型：优先样本元数据中的 rm_type，其次全局配置 args.rm_type
    rm_type = (metadata.get("rm_type") or args.rm_type or "").strip()
    # 待评判的模型响应文本
    response = sample.response
    # 标准答案标签
    label = sample.label
    # rm_type 以 boxed_ 前缀开头时先抽取 \\boxed{} 内的答案再判分
    if rm_type.startswith("boxed_"):
        # 从响应中抽取 \\boxed{} 答案（失败则为空串）
        response = extract_boxed_answer(response) or ""
        # 去掉 boxed_ 前缀得到真正的 RM 类型
        rm_type = rm_type[len("boxed_") :]

    # This function is intended for remote or time-consuming reward model evaluation.
    # Implement the actual logic as needed.
    # 远程 RM：调用远程奖励服务
    if rm_type == "remote_rm":
        return await remote_rm(args, sample)
    # deepscaler 规则奖励
    elif rm_type == "deepscaler":
        return get_deepscaler_rule_based_reward(response, label)
    # DAPO 数学规则评分
    elif rm_type == "dapo":
        return compute_score_dapo(response, label)
    # math：verl 判分正确为 1 否则 0
    elif rm_type == "math":
        return 1 if grade_answer_verl(response, label) else 0
    # f1：返回词级 F1 分数
    elif rm_type == "f1":
        return f1_score(response, label)[0]
    # gpqa：多选题规则奖励
    elif rm_type == "gpqa":
        return compute_gpqa_reward(response, label, metadata=metadata)
    # ifbench：指令遵循基准奖励（延迟导入以减少无关开销）
    elif rm_type == "ifbench":
        # 在分支内导入 IFBench 奖励实现
        from .ifbench import compute_ifbench_reward

        # 计算并返回 IFBench 奖励
        return compute_ifbench_reward(response, label, metadata=metadata)
    # random：返回 0/1 随机奖励（调试用）
    elif rm_type == "random":
        return random.randint(0, 1)
    # 指定了未知 rm_type：抛出未实现异常
    elif rm_type:
        raise NotImplementedError(f"Rule-based RM for {rm_type} is not implemented.")
    # 未指定 rm_type：抛出异常
    else:
        raise NotImplementedError("Rule-based RM type is not specified.")


# 批量异步奖励：对一组样本并发计算奖励
async def batched_async_rm(
    args,
    samples: list[Sample],
    **kwargs,
) -> list[int | float]:
    # 配置了自定义奖励函数时优先使用批量模式实现
    if args.custom_rm_path is not None:
        # Ensure the custom reward function is implemented in batch mode
        # 动态加载批量版自定义奖励函数
        rm_function = load_function(args.custom_rm_path)
        # 一次性调用并返回所有样本的奖励
        return await rm_function(args, samples, **kwargs)
    # 为每个样本创建异步奖励任务
    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    # 并发执行全部任务并按顺序收集结果
    rewards = await asyncio.gather(*tasks)
    # 返回奖励列表
    return rewards
