# 引入 abc 抽象基类模块，用于定义数据源的统一接口
import abc
# 引入 copy 模块，用于深拷贝 prompt 样本以生成多条训练样本
import copy
# 引入 logging 模块记录日志
import logging
# 引入 os 模块，用于路径拼接与创建目录
import os
# 引入 Path，便于以路径对象方式保存 tokenizer/processor
from pathlib import Path

# 引入 torch，用于以 torch.save/torch.load 持久化数据源状态
import torch

# 引入 slime 的 Dataset 封装，负责读取与预处理 prompt 数据集
from slime.utils.data import Dataset
# 引入按点号路径动态加载函数的工具（用于加载自定义 buffer 过滤器）
from slime.utils.misc import load_function
# 引入加载 tokenizer 与 processor 的工具函数
from slime.utils.processing_utils import load_processor, load_tokenizer
# 引入 rollout 样本数据结构 Sample
from slime.utils.types import Sample

# 获取本模块专属的 logger 实例
logger = logging.getLogger(__name__)


# 数据源抽象基类：约定"取样本、回收样本、保存/加载状态、长度"的统一接口
class DataSource(abc.ABC):
    # 声明 get_samples 为抽象方法，子类必须实现
    @abc.abstractmethod
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

    # 声明 add_samples 为抽象方法，子类必须实现
    @abc.abstractmethod
    def add_samples(self, samples: list[list[Sample]]):
        """
        Add samples to the data source
        """

    # 声明 save 为抽象方法：保存数据源当前状态
    @abc.abstractmethod
    def save(self, rollout_id):
        """
        Save the state of the data source
        """

    # 声明 load 为抽象方法：恢复数据源状态
    @abc.abstractmethod
    def load(self, rollout_id=None):
        """
        Load the state of the data source
        """

    # 声明 __len__ 为抽象方法：返回数据源长度（随样本增删可能变化）
    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Length of the data source. May change when samples are added/fetched.
        """


# TODO may further refactor data-loading part later
# 滚动式数据源：从静态数据集中按顺序（可跨 epoch 循环）取出样本
class RolloutDataSource(DataSource):
    # 初始化：保存全局参数并初始化进度游标
    def __init__(self, args):
        # 保存全局配置参数
        self.args = args

        # 当前训练 epoch 编号（从 0 开始）
        self.epoch_id = 0
        # 样本组编号：同一 prompt 派生的多个样本构成一组（组内对比，论文中 GRPO 的 group 概念）
        self.sample_group_index = 0
        # 全局样本编号（跨组累加，唯一标识每条样本）
        self.sample_index = 0
        # 当前 epoch 数据集中已消费到的偏移量
        self.sample_offset = 0
        # TODO remove this
        # 元数据字典（计划废弃的遗留字段）
        self.metadata = {}

        # 若配置为全局共享数据集（所有 rank 共用同一份数据游标）
        if args.rollout_global_dataset:
            # 加载 tokenizer（trust_remote_code 以支持自定义模型代码）
            tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
            # 加载 processor（多模态模型需要；纯文本模型可能为 None）
            processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

            # TODO move (during the refactor)
            # 若配置了 dump_details 调试目录，则把 tokenizer/processor 落盘便于排查
            if (d := args.dump_details) is not None:
                # 将 tokenizer 保存到调试目录
                tokenizer.save_pretrained(Path(d) / "tokenizer")
                # 仅当 processor 存在时才保存
                if processor:
                    # 将 processor 保存到调试目录
                    processor.save_pretrained(Path(d) / "processor")

            # 构造数据集对象：依次绑定 prompt 数据文件路径、tokenizer、processor（多模态）、
            # prompt 最大长度截断、输入/多模态/标签/元数据/工具各字段映射、聊天模板开关及附加参数、打乱随机种子
            self.dataset = Dataset(
                args.prompt_data,
                tokenizer=tokenizer,
                processor=processor,
                max_length=args.rollout_max_prompt_len,
                prompt_key=args.input_key,
                multimodal_keys=args.multimodal_keys,
                label_key=args.label_key,
                metadata_key=args.metadata_key,
                tool_key=args.tool_key,
                apply_chat_template=args.apply_chat_template,
                apply_chat_template_kwargs=args.apply_chat_template_kwargs,
                seed=args.rollout_seed,
            )
            # 若开启 rollout 阶段打乱数据
            if self.args.rollout_shuffle:
                # 以 epoch 编号为序打乱数据集（保证可复现）
                self.dataset.shuffle(self.epoch_id)
        else:
            # 未启用全局数据集时置空（后续按需生成空样本）
            self.dataset = None

    # 取样本：按序消费数据集，跨 epoch 时自动循环补齐
    def get_samples(self, num_samples):
        # TODO further improve code
        # 数据集存在时走真实数据路径
        if self.dataset is not None:
            # 剩余数据足够本批消费：直接切片返回
            if self.sample_offset + num_samples <= len(self.dataset):
                # 从当前偏移处切出 num_samples 条 prompt 样本
                prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
                # 前移消费偏移量
                self.sample_offset += num_samples
            else:
                # 剩余不足：先取完本 epoch 尾部数据
                prompt_samples = self.dataset.samples[self.sample_offset :]
                # 计算还差多少条需要从新 epoch 头部补齐
                num_samples -= len(prompt_samples)
                # 进入下一个 epoch
                self.epoch_id += 1
                # 若开启打乱，则以新 epoch 编号重新打乱数据
                if self.args.rollout_shuffle:
                    self.dataset.shuffle(self.epoch_id)
                # 从新 epoch 头部补足缺口
                prompt_samples += self.dataset.samples[:num_samples]
                # 更新偏移量为新 epoch 已消费条数
                self.sample_offset = num_samples
        else:
            # 无数据集时生成占位的空样本（如纯生成/自评估场景）
            prompt_samples = [Sample() for _ in range(num_samples)]

        # 最终输出的样本组列表
        samples = []
        # 遍历每条 prompt 样本，展开为一组训练样本
        for prompt_sample in prompt_samples:
            # 一个 prompt 对应的样本组
            group = []
            # 每个 prompt 复制生成 n_samples_per_prompt 条样本（GRPO 类算法组内相对比较的基础）
            for _ in range(self.args.n_samples_per_prompt):
                # 深拷贝 prompt 模板，保证各样本互不影响
                sample = copy.deepcopy(prompt_sample)
                # 标记该样本所属组编号
                sample.group_index = self.sample_group_index
                # 分配全局唯一样本编号
                sample.index = self.sample_index
                # 全局样本编号自增
                self.sample_index += 1
                # 加入当前组
                group.append(sample)
            # 一组完成后组编号自增
            self.sample_group_index += 1
            # 收集该样本组
            samples.append(group)
        # 返回样本组列表
        return samples

    # 只读数据源不支持回填样本
    def add_samples(self, samples: list[list[Sample]]):
        # 抛出运行时错误，提示这是只读数据源
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    # 保存数据源游标状态（断点续训用）
    def save(self, rollout_id):
        # 未启用全局数据集则无需保存
        if not self.args.rollout_global_dataset:
            return

        # 汇总所有进度游标为状态字典（数据消费偏移 / epoch 编号 / 样本组编号 / 全局样本编号 / 元数据）
        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
        }
        # 拼出按 rollout_id 区分的状态文件路径
        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        # 确保父目录存在（已存在则跳过）
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 序列化保存状态字典
        torch.save(state_dict, path)

    # 从检查点恢复数据源游标状态
    def load(self, rollout_id=None):
        # 未启用全局数据集则无需加载
        if not self.args.rollout_global_dataset:
            return

        # 未指定加载目录则直接返回
        if self.args.load is None:
            return

        # 拼出状态文件路径
        path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        # 状态文件不存在时仅记录日志后返回（从头训练）
        if not os.path.exists(path):
            logger.info(f"Checkpoint {path} does not exist.")
            return

        # 记录加载来源路径
        logger.info(f"load metadata from {path}")
        # 记录加载前内存中的 metadata
        logger.info(f"load metadata: {self.metadata}")
        # 反序列化读取状态字典
        state_dict = torch.load(path)
        # 恢复数据消费偏移（缺省 0）
        self.sample_offset = state_dict.get("sample_offset", 0)
        # 恢复 epoch 编号
        self.epoch_id = state_dict.get("epoch_id", 0)
        # 恢复样本组编号
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        # 恢复全局样本编号
        self.sample_index = state_dict.get("sample_index", 0)
        # 恢复元数据
        self.metadata = state_dict.get("metadata", {})

        # 恢复后按当前 epoch 重新打乱，保证与保存时数据顺序一致
        if self.args.rollout_global_dataset and self.args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)

    # 返回数据集总长度
    def __len__(self) -> int:
        # 数据集长度即样本总数
        return len(self.dataset)


# 带缓冲区的数据源：支持把样本回收进 buffer，取数时优先复用（动态采样用）
class RolloutDataSourceWithBuffer(RolloutDataSource):
    # 初始化：在基类之上增加 buffer 与过滤器
    def __init__(self, args):
        # 调用基类初始化
        super().__init__(args)
        # 样本缓冲区：存放回收待复用的样本组
        self.buffer = []
        # 未配置过滤器路径时使用默认的 pop_first（按序弹出）
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            # 按配置的 dotted path 动态加载自定义过滤器函数
            self.buffer_filter = load_function(self.args.buffer_filter_path)

    # 取样本：优先从 buffer 取，不足部分再回落到数据集
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

        # 先从缓冲区取样本
        samples = self._get_samples_from_buffer(num_samples)
        # 扣减已满足的数量，得到剩余缺口
        num_samples -= len(samples)

        # buffer 已满足需求则直接返回
        if num_samples == 0:
            return samples

        # 缺口部分从基类（真实数据集）继续取
        samples += super().get_samples(num_samples=num_samples)
        # 返回最终样本
        return samples

    # 从 buffer 中按过滤器策略取出 num_samples 条样本
    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        # buffer 为空或需求为 0 时直接返回空列表
        if len(self.buffer) == 0 or num_samples == 0:
            return []

        # 调用过滤器从 buffer 就地取出样本
        samples = self.buffer_filter(self.args, None, self.buffer, num_samples)
        # 返回取出的样本
        return samples

    # 把样本组回收进 buffer，供动态采样复用
    def add_samples(self, samples: list[list[Sample]]):
        """
        Add a sample group to buffer.
        """
        # 空输入直接返回
        if not samples:
            return
        # 校验外层必须是列表
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        # 校验内层元素也必须是列表（即样本组）
        assert isinstance(samples[0], list), f"the elements of samples must be list, got {type(samples[0])}"
        # 逐组检查并入库
        for i in range(0, len(samples)):
            # 每组长度必须等于 n_samples_per_prompt
            assert (
                len(samples[i]) == self.args.n_samples_per_prompt
            ), f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(samples[i])} != {self.args.n_samples_per_prompt}"
            # 取出第 i 个样本组
            group = samples[i]  # type: ignore
            # 追加到 buffer 末尾
            self.buffer.append(group)

    # TODO remove
    # 更新元数据字典（遗留接口）
    def update_metadata(self, metadata: dict):
        # 合并传入的键值对
        self.metadata.update(metadata)

    # TODO remove
    # 读取元数据字典（遗留接口）
    def get_metadata(self):
        # 返回元数据
        return self.metadata

    # 返回 buffer 中待复用的样本组数量
    def get_buffer_length(self):
        # buffer 长度
        return len(self.buffer)


# 默认 buffer 过滤器：从 buffer 头部按序弹出 num_samples 条样本组
def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    # 实际弹出数量取"需求量"与"库存量"的较小者
    num_to_pop = min(len(buffer), num_samples)
    # 取出队首若干样本组
    samples = buffer[:num_to_pop]
    # 就地从 buffer 中删除已取出的部分
    del buffer[:num_to_pop]
    # 返回弹出的样本
    return samples
