# 导入 dataclass 工具：声明数据类与字段默认值工厂
from dataclasses import dataclass, field
# 导入 Enum：定义样本状态的枚举类型
from enum import Enum
# 导入 Any：多模态字段等任意类型注解
from typing import Any

# 导入 torch：张量类型注解（ParamInfo.dtype、RolloutBatch 元素类型）
import torch


@dataclass
class Sample:
    """The sample generated"""
    # 单条 RLHF 样本：贯穿 rollout（生成）与训练两条路径的核心数据结构

    # 样本所属组的下标（同一 prompt 的一组采样共享 group_index，用于 GRPO 组内基线等）
    group_index: int | None = None
    # 组内的样本下标
    index: int | None = None
    # prompt
    # 提示词：纯文本或对话消息列表（多轮 chat 格式）
    prompt: str | list[dict[str, str]] = ""
    # 模型生成的完整 token 序列（prompt + response）
    tokens: list[int] = field(default_factory=list)
    # 原始多模态数据（图像、视频等）
    multimodal_inputs: dict[str, Any] | None = None  # raw multimodal data, e.g. images, videos, etc.
    # 处理后的多模态数据（pixel_values 等张量）
    multimodal_train_inputs: dict[str, Any] | None = None  # processed multimodal data, e.g. pixel_values, etc.
    # response
    # 模型生成的响应文本
    response: str = ""
    # 响应的 token 长度
    response_length: int = 0
    # 标准答案标签（用于奖励计算或评估）
    label: str | None = None
    # 奖励值：标量或按维度细分的字典
    reward: float | dict[str, Any] | None = None
    # loss mask：逐 token 是否参与损失计算（1 参与 / 0 忽略）
    loss_mask: list[int] | None = None
    # 生成时经历过的权重版本列表（部分回滚/过期样本检测用）
    weight_versions: list[str] = field(default_factory=list)
    # rollout 引擎返回的逐 token log prob（重要性采样比率计算用）
    rollout_log_probs: list[float] | None = None  # Log probabilities from rollout engine
    # rollout 引擎记录的 MoE 路由专家（路由重放用）
    rollout_routed_experts: list[list[int]] | None = None  # Routed experts from rollout engine
    # 标记该样本是否应从训练集中剔除
    remove_sample: bool = False
    # 教师模型的 log prob，用于 OPD（on-policy distillation，在线策略蒸馏）
    teacher_log_probs: list[float] | None = None  # Log probabilities from teacher model for OPD

    class Status(Enum):
        # 样本生命周期状态枚举
        # 等待生成
        PENDING = "pending"
        # 正常完成
        COMPLETED = "completed"
        # 达到长度上限被截断
        TRUNCATED = "truncated"
        # 生成被中止
        ABORTED = "aborted"
        # Indicates a recoverable or non-critical failure during generation (e.g., tool call failure,
        # external API error, parsing error). Unlike ABORTED, FAILED samples may still contain partial
        # valid output and can be retried or handled gracefully.
        # 可恢复的非致命失败（工具调用失败、外部 API 错误、解析错误等）；与 ABORTED 不同，可能仍有部分有效输出
        FAILED = "failed"

    # 样本当前状态，默认等待中
    status: Status = Status.PENDING

    # 通用元数据字典
    metadata: dict = field(default_factory=dict)
    # 生成该样本所用的自定义生成函数路径（可复现生成逻辑）
    generate_function_path: str | None = None
    # metadata used during training, e.g., what loss to use for this sample.
    # 训练阶段使用的元数据（例如该样本用哪种损失）
    train_metadata: dict | None = None

    # Session ID for consistent hashing routing (used when router policy is consistent_hashing)
    # 会话 ID：路由策略为一致性哈希时保证同会话请求落到同一 server
    session_id: str | None = None

    # 非生成步骤（如工具调用、预处理）花费的时间
    non_generation_time: float = 0.0  # time spent in non-generation steps

    @dataclass
    class SpecInfo:
        # 投机解码（speculative decoding）统计信息
        # 被接受的草稿 token 数
        spec_accept_token_num: int = 0
        # 草稿 token 总数
        spec_draft_token_num: int = 0
        # 验证（verify）轮数
        spec_verify_ct: int = 0
        # 最终完成的 token 总数
        completion_token_num: int = 0

        @property
        def spec_accept_rate(self) -> float:
            # 投机解码接受率 = 接受 token / 草稿 token（无草稿时为 0）
            return self.spec_accept_token_num / self.spec_draft_token_num if self.spec_draft_token_num > 0 else 0.0

        @property
        def spec_accept_length(self) -> float:
            # 平均每次验证产出的 token 数（衡量投机解码加速比）
            return self.completion_token_num / self.spec_verify_ct if self.spec_verify_ct > 0 else 0.0

        def add(self, meta_info: dict):
            # 从引擎返回的元信息累加各项统计（部分回滚场景需要跨次累加）
            self.spec_accept_token_num += meta_info.get("spec_accept_token_num", 0)
            self.spec_draft_token_num += meta_info.get("spec_draft_token_num", 0)
            self.spec_verify_ct += meta_info.get("spec_verify_ct", 0)
            self.completion_token_num += meta_info.get("completion_tokens", 0)

        def to_dict(self):
            # 序列化为普通字典
            return {
                "spec_accept_token_num": self.spec_accept_token_num,
                "spec_draft_token_num": self.spec_draft_token_num,
                "spec_verify_ct": self.spec_verify_ct,
                "completion_token_num": self.completion_token_num,
            }

        @staticmethod
        def from_dict(data: dict):
            # 从字典反序列化（缺省字段补 0）
            info = Sample.SpecInfo()
            info.spec_accept_token_num = data.get("spec_accept_token_num", 0)
            info.spec_draft_token_num = data.get("spec_draft_token_num", 0)
            info.spec_verify_ct = data.get("spec_verify_ct", 0)
            info.completion_token_num = data.get("completion_token_num", 0)
            return info

    # 投机解码统计，默认空实例
    spec_info: SpecInfo = field(default_factory=SpecInfo)

    @dataclass
    class PrefixCacheInfo:
        # 前缀缓存（prefix caching，KV cache 复用）统计
        # 命中缓存的 token 数
        cached_tokens: int = 0
        # prompt 的 token 总数
        total_prompt_tokens: int = 0

        @property
        def prefix_cache_hit_rate(self) -> float:
            # 前缀缓存命中率 = 命中 token / 总 prompt token（无 token 时为 0）
            return self.cached_tokens / self.total_prompt_tokens if self.total_prompt_tokens > 0 else 0.0

        def add(self, meta_info: dict):
            # 累加缓存命中 token 数
            self.cached_tokens += meta_info.get("cached_tokens", 0)
            # new_tokens = input_tokens - cached_tokens
            # 累加 prompt 总 token 数
            self.total_prompt_tokens += meta_info.get("prompt_tokens", 0)

        def to_dict(self):
            # 序列化为普通字典
            return {
                "cached_tokens": self.cached_tokens,
                "total_prompt_tokens": self.total_prompt_tokens,
            }

        @staticmethod
        def from_dict(data: dict):
            # 从字典反序列化（缺省字段补 0）
            info = Sample.PrefixCacheInfo()
            info.cached_tokens = data.get("cached_tokens", 0)
            info.total_prompt_tokens = data.get("total_prompt_tokens", 0)
            return info

    # 前缀缓存统计，默认空实例
    prefix_cache_info: PrefixCacheInfo = field(default_factory=PrefixCacheInfo)

    def to_dict(self):
        # 拷贝实例属性字典
        value = self.__dict__.copy()
        # 枚举转为字符串值
        value["status"] = self.status.value
        # 嵌套数据类转为字典
        value["spec_info"] = self.spec_info.to_dict()
        value["prefix_cache_info"] = self.prefix_cache_info.to_dict()
        # 返回可序列化字典
        return value

    @staticmethod
    def from_dict(data: dict):
        # 拷贝输入避免污染原字典
        data = dict(data)
        # 字符串还原为状态枚举
        data["status"] = Sample.Status(data["status"])
        # 还原投机解码统计
        data["spec_info"] = Sample.SpecInfo.from_dict(data.get("spec_info", {}))
        # 还原前缀缓存统计
        data["prefix_cache_info"] = Sample.PrefixCacheInfo.from_dict(data.get("prefix_cache_info", {}))

        # 收集 Sample 数据类声明的全部字段名
        field_names = set(Sample.__dataclass_fields__.keys())
        # 只取已知字段用于构造函数（避免非法参数报错）
        init_data = {k: v for k, v in data.items() if k in field_names}
        # 构造样本实例
        sample = Sample(**init_data)

        for key, value in data.items():
            if key not in field_names:
                # 未知字段作为动态属性挂回实例（前向兼容新版本数据）
                setattr(sample, key, value)

        # 返回还原的样本
        return sample

    def get_reward_value(self, args) -> float:
        # 未配置 reward_key 时直接返回标量奖励；配置了则从奖励字典按键取值（多维奖励转标量）
        return self.reward if not args.reward_key else self.reward[args.reward_key]

    @property
    def effective_response_length(self):
        # 有效响应长度：有 loss mask 时为 mask 中 1 的个数，否则退化为整个响应长度
        return sum(self.loss_mask) if self.loss_mask is not None else self.response_length

    def update_from_meta_info(self, args, meta_info: dict):
        """
        Update the sample with new information from meta_info returned by the rollout engine.
        And extract
        """
        if args.sglang_speculative_algorithm:
            # cannot directly use spec info from sglang because of partial rollout.
            # 启用投机解码时累加投机统计（部分回滚导致不能直接用引擎的汇总值）
            self.spec_info.add(meta_info=meta_info)

        # Collect prefix cache statistics
        # 累加前缀缓存统计
        self.prefix_cache_info.add(meta_info=meta_info)

        if "weight_version" in meta_info:
            # 记录生成本段所用的权重版本（检测样本是否由过期权重生成）
            self.weight_versions.append(meta_info["weight_version"])

        # 按引擎给出的结束原因归类样本状态
        match meta_info["finish_reason"]["type"]:
            case "length":
                # 达到长度上限：截断
                self.status = Sample.Status.TRUNCATED
            case "abort":
                # 被主动中止
                self.status = Sample.Status.ABORTED
            case "stop":
                # 正常停止（EOS）
                self.status = Sample.Status.COMPLETED


@dataclass(frozen=True)
class ParamInfo:
    # 参数元信息：权重分发/保存时描述单个参数（frozen 保证不可变，可安全传递）
    # 参数名
    name: str
    # 数据类型
    dtype: torch.dtype
    # 张量形状
    shape: torch.Size
    # 额外属性（如量化参数等）
    attrs: dict
    # 元素总数
    size: int
    # 参数所在（或持有完整副本）的源 rank
    src_rank: int


# A dict-based batch produced along the rollout -> training path
# In Megatron backend, several fields are converted to torch.Tensor lists on GPU
# before being consumed by data iterators (see megatron_utils.actor._get_rollout_data).
# rollout -> training 路径上的字典式批量数据；Megatron 后端中部分字段在使用前会转成 GPU 张量列表
RolloutBatch = dict[str, list[torch.Tensor] | list[int] | list[float] | list[str]]


@dataclass
class MultimodalType:
    # 多模态类型描述
    # 类型标识（用于消息内容中，如 "image"）
    name: str  # Type identifier used in message content (e.g., "image")
    # 对话消息中的占位符（如 "<image>"）
    placeholder: str  # Placeholder token in conversation messages (e.g., "<image>")


class MultimodalTypes:
    # 内置多模态类型常量集
    # 图像类型：占位符 <image>
    IMAGE = MultimodalType(name="image", placeholder="<image>")
    # 视频类型：占位符 <video>
    VIDEO = MultimodalType(name="video", placeholder="<video>")
    # 音频类型：占位符 <audio>
    AUDIO = MultimodalType(name="audio", placeholder="<audio>")

    @classmethod
    def all(cls) -> list[MultimodalType]:
        # 返回全部内置多模态类型
        return [cls.IMAGE, cls.VIDEO, cls.AUDIO]

    @classmethod
    def get(cls, name: str) -> MultimodalType | None:
        # 按名称查找多模态类型，找不到返回 None
        return next((m for m in cls.all() if m.name == name), None)
