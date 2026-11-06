# 引入日志模块
import logging
# 引入 random，用于随机采样输出长度
import random
# 引入 Namespace 类型标注（命令行参数对象）
from argparse import Namespace
# 引入 deepcopy，用于复制采样参数避免污染原配置
from copy import deepcopy
# 引入 Any 类型标注
from typing import Any

# 引入基础的 sglang 生成函数（本模块在其上做包装）
from slime.rollout.sglang_rollout import generate as _generate_base
# 引入样本数据结构
from slime.utils.types import Sample

# 获取本模块专属 logger
logger = logging.getLogger(__name__)


# 基准测试生成器：以随机输出长度（OSL）强制生成长回答，用于压测/评测长序列生成能力
async def generate_with_random_osl(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    # TODO: make it configurable after we have an enhanced arg parser
    # 随机输出长度（OSL）下界：32K token
    min_osl = 32 * 1024
    # 随机输出长度（OSL）上界：64K token
    max_osl = 64 * 1024

    # 深拷贝采样参数，避免修改调用方的原始配置
    modified_sampling_params = deepcopy(sampling_params)
    # 忽略 EOS：强制模型跳过结束符持续生成到目标长度
    modified_sampling_params["ignore_eos"] = True
    # 在 [min_osl, max_osl) 区间内随机指定本次生成的最大 token 数
    modified_sampling_params["max_new_tokens"] = random.randrange(min_osl, max_osl)

    # 调用基础生成函数执行实际推理
    ans = await _generate_base(args, sample, modified_sampling_params)

    # 记录本次生成的响应长度
    logger.info(f"generate_with_random_osl {ans.response_length=}")
    # 返回携带生成结果的样本
    return ans
