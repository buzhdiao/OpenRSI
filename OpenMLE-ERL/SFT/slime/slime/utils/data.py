# 数据工具模块:负责读取提示数据集(jsonl/parquet)、构建消息与多模态内容、
# 提供 Dataset 类、以及 rollout 数据在 DP 组间的对齐处理等。
import itertools
import json
import logging
import os
import random
import re

import numpy as np
import ray

# pyarrow 为可选依赖:缺失时置 None,仅在读取 parquet 时报错
try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from slime.utils.types import MultimodalTypes, Sample

from .timer import Timer

# 声明本模块对外暴露的公共接口
__all__ = ["Dataset"]

# 模块级 logger
logger = logging.getLogger(__name__)


def read_file(path):
    # 先解析"路径@切片"语法,得到真实路径与可选的行切片
    path, row_slice = _parse_generalized_path(path)
    reader = None

    if not os.path.exists(path):
        # 数据文件不存在时直接报错,尽早失败
        raise FileNotFoundError(f"Prompt dataset path '{path}' does not exist.")

    if path.endswith(".jsonl"):

        def jsonl_reader(p):
            # 逐行打开 jsonl 文件(UTF-8 编码)
            with open(p, encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    # 去掉行首尾空白(含换行符)
                    line = line.strip()
                    if not line:
                        # 跳过空行
                        continue
                    try:
                        # 把每行 JSON 反序列化为 dict 并产出
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        # 单行解析失败仅打印告警并继续,不让整个数据集读取中断
                        print(f"JSON decode error at line {line_num}: {e}")
                        continue

        # 以生成器形式懒加载 jsonl 数据
        reader = jsonl_reader(path)

    elif path.endswith(".parquet"):
        if pq is None:
            # 读取 parquet 需要 pyarrow,缺失时给出明确报错
            raise ImportError("pyarrow is required for parquet support")

        def parquet_reader(p):
            # 打开 parquet 文件句柄
            pf = pq.ParquetFile(p)

            # 按批迭代读取,逐行产出为 Python dict 列表
            for batch in pf.iter_batches():
                yield from batch.to_pylist()

        # 以生成器形式懒加载 parquet 数据
        reader = parquet_reader(path)

    else:
        # 其他格式暂不支持
        raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl and .parquet.")

    if row_slice is not None:

        # 记录实际生效的切片范围
        logger.info("read_file path=%s applying slice row_slice=%s", path, row_slice)
        # 用 islice 对数据流做惰性切片,只保留指定行区间(支持 data 并行分片读取)
        reader = itertools.islice(reader, row_slice.start, row_slice.stop, row_slice.step)

    # 把(可能切片后的)生成器内容透传给调用方
    yield from reader


def _parse_generalized_path(s: str):
    # 用正则匹配 "真实路径@[start:end]" 形式的广义路径
    if (m := re.match(r"^(?P<real_path>.*)@\[(?P<start>-?\d*):(?P<end>-?\d*)\]$", s)) is not None:
        # 提取 @ 之前的真实文件路径
        path = m.group("real_path")
        # 解析切片起点;留空表示 None(默认从头)
        start = int(x) if (x := m.group("start")) != "" else None
        # 解析切片终点;留空表示 None(默认到尾)
        end = int(x) if (x := m.group("end")) != "" else None
        # 返回路径与 slice 对象
        return path, slice(start, end)

    # 不带切片语法时原样返回路径、无切片
    return s, None


def filter_long_prompt(origin_samples: list[Sample], tokenizer, processor, max_length: int | None) -> list[Sample]:
    # 未设置最大长度限制时直接原样返回
    if max_length is None:
        return origin_samples

    if not isinstance(origin_samples[0].prompt, str):
        # prompt 是消息列表(已应用对话模板)时无法直接 tokenize 估算长度,跳过过滤
        logger.warning(
            "Skipping max_length check for list prompt. Set apply_chat_template=True to enable length filtering."
        )
        return origin_samples

    if processor:
        # 多模态处理器模式:需要借助 processor 才能准确得到含视觉 token 的输入长度
        filtered_samples = []
        for sample in origin_samples:
            # 延迟导入以避免循环依赖
            from slime.utils.processing_utils import process_vision_info

            # 从 prompt 文本中抽取视觉信息(图片/视频等)
            multimodal_inputs = process_vision_info(sample.prompt, processor)
            # 用 processor 完整编码文本与多模态输入
            processor_output = processor(text=sample.prompt, **multimodal_inputs)
            # 取出编码后的 token id 序列
            input_ids = processor_output["input_ids"][0]
            if len(input_ids) <= max_length:
                # 长度达标则保留该样本
                filtered_samples.append(sample)
    else:
        # 纯文本模式:先收集所有 prompt 文本
        prompts = [sample.prompt for sample in origin_samples]
        # 批量 tokenize(不额外加特殊 token)得到各样本的 token 序列
        input_ids_list = tokenizer(prompts, add_special_tokens=False)["input_ids"]
        # 保留 token 数不超过上限的样本(strict=True 保证样本与序列一一对应)
        filtered_samples = [
            sample
            for sample, input_ids in zip(origin_samples, input_ids_list, strict=True)
            if len(input_ids) <= max_length
        ]

    # 打印被过滤掉的样本数量
    logger.info(f"Filtered {len(origin_samples) - len(filtered_samples)} samples longer than max_length={max_length}.")

    # 返回过滤后的样本列表
    return filtered_samples


def _build_messages(data: dict, prompt_key: str, as_conversation: bool, multimodal_keys: dict = None):
    # 从原始数据中按 key 取出 prompt 字段
    prompt = data.get(prompt_key)

    if isinstance(prompt, str):
        # If prompt is a string and we don't apply chat template, return the prompt as is.
        # 字符串 prompt 且不需要对话格式时,原样返回
        if not as_conversation:
            return prompt
        else:
            # 否则包装成 OpenAI 风格的单轮 user 消息列表
            prompt = [{"role": "user", "content": prompt}]

    if multimodal_keys:
        # Build mapping: placeholder -> (MultimodalType, content_list)
        # 构建占位符 -> (多模态类型, 数据内容列表) 的映射
        multimodals = {}
        for type_name, data_key in multimodal_keys.items():
            # 按类型名(如 "image")查找预注册的多模态类型定义
            mt = MultimodalTypes.get(type_name)
            if mt:
                # 从数据中取出该模态对应的内容列表
                multimodal_data = data.get(data_key)
                if multimodal_data is not None:
                    # 登记该占位符及其可用内容(转成 list 便于逐个弹出)
                    multimodals[mt.placeholder] = (mt, list(multimodal_data))

        # 构造用于切分文本的正则:把所有占位符按字面值拼接为捕获组
        pattern = "(" + "|".join(re.escape(p) for p in multimodals.keys()) + ")"

        for message in prompt:
            if isinstance(message["content"], str):
                # 内容为字符串时,按占位符切分文本并重建多模态消息
                content_list = []
                for segment in re.split(pattern, message["content"]):
                    if not segment:
                        # 跳过切分产生的空片段
                        continue
                    if segment in multimodals:
                        # 命中占位符:取出对应的模态类型与内容
                        mt, content = multimodals[segment]
                        assert len(content) > 0, (
                            f"Not enough {mt.name} data: more '{mt.placeholder}' placeholders in prompt "
                            f"than {mt.name}s provided in data"
                        )
                        # 弹出一条该模态数据,组装成带 type 的内容片段
                        content_list.append({"type": mt.name, mt.name: content.pop(0)})
                    else:
                        # 普通文本片段,标记为 text 类型
                        content_list.append({"type": "text", "text": segment})
                # 用重建后的多模态内容列表替换原字符串
                message["content"] = content_list

            elif isinstance(message["content"], list):
                # TODO: handle more general cases. where message['content'] is a dict and contains multiple types of content.
                # e.g.
                #  "content": [
                #     {
                #         "type": "image",
                #         "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                #     },
                #     {"type": "text", "text": "Describe this image."},
                # ],
                # 内容已是结构化列表时暂不处理,仅告警
                logger.warning("message['content'] is a list of dicts, no processing will be done.")
                continue
            else:
                # 其余类型不支持,直接报错
                raise ValueError(
                    f"Unsupported content type: {type(message['content'])}, expected str or list of dicts"
                )

        # 校验每个模态提供的数据条数与占位符出现次数一致(多余的数据说明数量不匹配)
        for placeholder, (mt, remaining) in multimodals.items():
            assert len(remaining) == 0, (
                f"Multimodal data count mismatch: {len(remaining)} more {mt.name}(s)"
                f"than '{placeholder}' placeholders in prompt"
            )

    # 返回(可能已转为对话格式/多模态格式)的 prompt
    return prompt


class Dataset:
    def __init__(
        self,
        path,
        tokenizer,
        processor,
        max_length,
        *,
        prompt_key="text",
        multimodal_keys=None,
        label_key=None,
        tool_key=None,
        metadata_key="metadata",
        seed=42,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
    ):
        # 逐条读取原始数据并构建样本
        origin_samples = []
        for data in read_file(path):
            # Both chat templates and multimodal inputs require conversation format (list of message dicts)
            # 对话模板与多模态输入都要求对话格式,据此决定是否把字符串 prompt 包装成消息列表
            as_conversation = apply_chat_template or (multimodal_keys is not None)
            # 构建消息(处理对话格式与多模态占位符)
            prompt = _build_messages(data, prompt_key, as_conversation, multimodal_keys)

            # 取出元数据,缺失时用空 dict
            metadata = data.get(metadata_key) or {}
            tools = None
            if tool_key is not None and tool_key in data:
                # 取出工具(tools)定义数据
                tools = data[tool_key]
                if isinstance(tools, str):
                    # 字符串形式时反序列化为 JSON
                    tools = json.loads(tools)
                elif isinstance(tools, np.ndarray):
                    # numpy 数组时转成 Python 列表
                    tools = tools.tolist()
                assert isinstance(tools, list), f"tools must be a list, got {type(tools)} instead"
                # 把工具定义写进元数据,供 apply_chat_template 使用
                metadata["tools"] = tools

            if apply_chat_template:
                # 应用 tokenizer 的对话模板,并追加生成提示(add_generation_prompt)
                output_prompt = tokenizer.apply_chat_template(
                    prompt,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                    **(apply_chat_template_kwargs or {}),
                )
            else:
                # 不套模板时直接使用原 prompt
                output_prompt = prompt

            if processor:
                # 延迟导入以避免循环依赖
                from slime.utils.processing_utils import process_vision_info

                assert isinstance(
                    prompt, list
                ), f"prompt must be a list when processor is not None, got {type(prompt)} instead"
                # 使用多模态处理器抽取视觉输入,供后续 rollout 使用
                multimodal_inputs = process_vision_info(prompt, processor)
            else:
                # 纯文本场景没有多模态输入
                multimodal_inputs = None

            # 组装成一个 Sample 并加入样本列表
            origin_samples.append(
                Sample(
                    prompt=output_prompt,
                    label=data[label_key] if label_key is not None else None,
                    metadata=metadata,
                    multimodal_inputs=multimodal_inputs,
                )
            )

        if max_length is not None:
            # 设置了最大长度时过滤超长 prompt 样本
            self.origin_samples = filter_long_prompt(origin_samples, tokenizer, processor, max_length)
        else:
            # 否则保留全部原始样本
            self.origin_samples = origin_samples

        # 初始 epoch 设为 -1,表示尚未洗牌过
        self.epoch_id = -1
        # 保存随机种子,洗牌时按 epoch 派生
        self.seed = seed
        # 当前可访问样本初始与原始样本一致
        self.samples = self.origin_samples

    def shuffle(self, new_epoch_id):
        # 同一 epoch 不重复洗牌
        if self.epoch_id == new_epoch_id:
            return

        # 用 seed+epoch_id 派生确定性随机源,保证各进程洗牌结果一致
        random.seed(self.seed + new_epoch_id)
        # 生成索引序列
        permutation = list(range(len(self.samples)))
        # 就地打乱索引顺序
        random.shuffle(permutation)
        # 按打乱后的索引重排样本列表(原始样本保持不变,便于复现)
        self.samples = [self.origin_samples[i] for i in permutation]
        # 记录已洗牌到的 epoch
        self.epoch_id = new_epoch_id

    def __getitem__(self, idx):
        # 按索引访问当前(可能已洗牌的)样本
        return self.samples[idx]

    def __len__(self):
        # 返回当前样本总数
        return len(self.samples)


def get_minimum_num_micro_batch_size(total_lengths, max_tokens_per_gpu):
    # use first fit to get the number of micro batches
    # 首次适应(first-fit)装箱:估算这些序列至少需要多少个 micro batch
    batches = []
    for length in total_lengths:
        for i in range(len(batches)):
            if batches[i] + length <= max_tokens_per_gpu:
                # 能塞进已有箱子则累加其 token 数
                batches[i] += length
                break
        else:
            # 所有箱子都装不下时新开一个箱子
            batches.append(length)

    # 箱子数即为所需的最少 micro batch 数
    return len(batches)


def process_rollout_data(args, rollout_data_ref, dp_rank, dp_size):
    # 校验 rollout 数据引用数量与 DP(world)大小一致
    assert len(rollout_data_ref) == dp_size
    # 取出本 DP rank 对应的 rollout 数据(Ray ObjectRef 解引用)
    rollout_data = ray.get(rollout_data_ref[dp_rank].inner)

    # 弹出分区信息(该 rank 应取哪些样本的下标)
    partition = rollout_data.pop("partition")
    total_lengths = rollout_data["total_lengths"]

    # save the seqlen of the whole rollout batch
    # 把整个 rollout 批次的序列长度记录进计时器,供性能统计使用
    Timer().seq_lens = total_lengths
    # 按 partition 重排序列长度,使其与本 rank 实际取得的样本对齐
    rollout_data["total_lengths"] = [total_lengths[i] for i in partition]

    # 返回对齐后的 rollout 数据
    return rollout_data
