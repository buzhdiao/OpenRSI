# 导入 transformers 的自动分词器（仅用于类型注解标注 tokenizer 类型）
from transformers import AutoTokenizer


def get_response_lengths(loss_masks: list[list[int]]) -> list[int]:
    # return the lengths starting from the first occurrence of 1 to the end of each loss mask
    # 对每条 loss mask：从第一个 1（响应开始位置）截到末尾并计长度；没有 1 则记 0
    return [len(mask[mask.index(1) :]) if 1 in mask else 0 for mask in loss_masks]


class MultiTurnLossMaskGenerator:
    def __init__(self, tokenizer: AutoTokenizer, tokenizer_type: str = "qwen"):
        # 保存分词器实例，供后续 apply_chat_template / 编码解码使用
        self.tokenizer = tokenizer
        # 预计算两个模板常量：系统消息区段长度、生成提示符（generation prompt）长度
        self.system_message_length, self.gen_token_length = self.get_system_message_length()
        # 保存分词器类型，决定用哪个模板规则生成 loss mask
        self.tokenizer_type = tokenizer_type

    def get_response_lengths(self, loss_masks: list[list[int]]) -> list[int]:
        # 直接复用模块级同名函数：统计每条 mask 中响应部分的长度
        return get_response_lengths(loss_masks)

    def find_all_sublist_indices(self, main_list, sublist):
        # 记录子串（token 子序列）长度
        sublist_len = len(sublist)
        # 收集所有匹配到的起始下标
        indices = []
        # 在主序列上逐个位置滑动窗口
        for i in range(len(main_list) - sublist_len + 1):
            # 窗口内容与目标子序列完全一致则记录该位置
            if main_list[i : i + sublist_len] == sublist:
                indices.append(i)
        # 返回全部匹配位置
        return indices

    def get_system_message_length(self) -> tuple[int, int]:
        # 用一个探测字符串测出 chat 模板中固定区段的 token 长度
        test_string = "FOR TESTING ONLY"
        # 构造两条相同内容的 user 消息，使模板中该内容恰好出现两次
        test_messages = [
            {"role": "user", "content": test_string},
            {"role": "user", "content": test_string},
        ]
        # 单独编码探测字符串（不加特殊 token），得到它的原始 token 序列
        raw_token_ids = self.tokenizer(test_string, add_special_tokens=False)["input_ids"]
        # 把两条消息套上 chat 模板渲染成完整文本（暂不转 token）
        chat_template_token = self.tokenizer.apply_chat_template(
            test_messages, add_special_tokens=False, tokenize=False
        )
        # 将渲染文本再编码成 token 序列
        chat_template_token_ids = self.tokenizer(chat_template_token, add_special_tokens=False)["input_ids"]
        # 在模板序列中定位两次探测字符串的出现位置（idx_1 为第一条消息起点，idx_2 为第二条）
        idx_1, idx_2 = self.find_all_sublist_indices(chat_template_token_ids, raw_token_ids)
        # 第一条消息的结尾区段长度 = 总长 - 最后一次内容 - 其起点
        end_interval = len(chat_template_token_ids) - len(raw_token_ids) - idx_2
        # 追加生成提示符（add_generation_prompt=True）后序列多出的部分即 gen prompt 长度
        gen_token_length = len(
            self.tokenizer.apply_chat_template(
                test_messages, add_special_tokens=False, tokenize=True, add_generation_prompt=True
            )
        ) - len(chat_template_token_ids)

        # 反推系统消息区段长度：idx_1 之前的部分扣除消息自身模板开销
        system_message_length = idx_1 - ((idx_2 - idx_1) - end_interval - len(raw_token_ids))
        # 返回（系统消息长度, 生成提示符长度）二元组
        return system_message_length, gen_token_length

    def gen_multi_turn_loss_mask_qwen(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        # 汇总所有轮次的 loss mask（1 表示参与训练，0 表示忽略）
        all_loss_masks = []
        # 汇总所有轮次拼接后的 token 序列
        all_token_ids = []

        # 逐条消息渲染并生成对应 mask
        for i, message in enumerate(messages):
            if i == 0:
                # 第一条消息：带上 tools 定义一起套模板（工具说明只渲染一次）
                message_ids = self.tokenizer.apply_chat_template([message], tokenize=True, tools=tools)
            else:
                # 后续消息单独套模板渲染为 token 序列
                message_ids = self.tokenizer.apply_chat_template([message], tokenize=True)

            # 非系统消息且不是第一条：去掉模板中系统消息区段的固定长度，避免重复
            if message["role"] != "system" and i > 0:
                message_ids = message_ids[self.system_message_length :]

            if message["role"] == "assistant":
                # assistant 回复：跳过开头的生成提示符（不训练），正文部分置 1 参与 loss
                loss_mask = [0] * self.gen_token_length + [1] * (len(message_ids) - self.gen_token_length)
            else:
                # 非 assistant 消息（system/user/tool）全部置 0，不参与 loss
                loss_mask = [0] * len(message_ids)

            # 数据自带 step_loss_mask 开关时，可整条屏蔽该消息（如拒绝样本/负例）
            if message.get("step_loss_mask", 1) != 1:
                loss_mask = [0] * len(message_ids)

            # 累加到全局 mask
            all_loss_masks.extend(loss_mask)
            # 累加到全局 token 序列
            all_token_ids.extend(message_ids)

        # 返回（拼接 token 序列, 对应 loss mask）
        return all_token_ids, all_loss_masks

    def gen_multi_turn_loss_mask_qwen3(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        # 汇总所有轮次的 loss mask
        all_loss_masks = []
        # 汇总所有轮次拼接后的 token 序列
        all_token_ids = []

        # qwen3 模板会随上下文变化，需要一个"哨兵"消息辅助切分
        prefix_message = {"role": "user", "content": "FOR CALCULATING LOSS MASK ONLY"}
        # 预先渲染哨兵消息单独成模板的 token 序列，作为切分基准
        prefix_token_ids = self.tokenizer.apply_chat_template([prefix_message], tokenize=True)

        # 逐条消息处理
        for i, message in enumerate(messages):
            if i == 0:
                # 第一条：把真实消息与哨兵一起渲染（保留 tools 上下文）
                tailed_message_ids = self.tokenizer.apply_chat_template(
                    [message, prefix_message], tokenize=True, tools=tools
                )
                # 去掉末尾哨兵消息部分，得到该消息的真实 token 序列
                message_ids = tailed_message_ids[: -len(prefix_token_ids)]
            else:
                # 后续：先渲染哨兵再渲染真实消息
                prefixed_message_ids = self.tokenizer.apply_chat_template([prefix_message, message], tokenize=True)
                # 去掉开头哨兵消息部分，得到真实 token 序列
                message_ids = prefixed_message_ids[len(prefix_token_ids) :]

            # 非系统消息且非首条：去掉系统消息区段长度，避免重复计入
            if message["role"] != "system" and i > 0:
                message_ids = message_ids[self.system_message_length :]

            if message["role"] == "assistant":
                # assistant：开头生成提示符置 0，正文置 1 参与 loss
                loss_mask = [0] * self.gen_token_length + [1] * (len(message_ids) - self.gen_token_length)
            else:
                # 其他角色全部置 0
                loss_mask = [0] * len(message_ids)

            # 支持 step_loss_mask 开关整条屏蔽
            if message.get("step_loss_mask", 1) != 1:
                loss_mask = [0] * len(message_ids)

            # 累加 mask
            all_loss_masks.extend(loss_mask)
            # 累加 token
            all_token_ids.extend(message_ids)

        # 返回（拼接 token 序列, 对应 loss mask）
        return all_token_ids, all_loss_masks

    def gen_multi_turn_loss_mask_qwen3_5(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        # qwen3.5：先把全部消息整体渲染为文本（便于做字符级定位）
        rendered_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, tools=tools, return_dict=False
        )
        # 对渲染文本编码，并要求返回每个 token 在原文中的字符区间（offset mapping）
        tokenized = self.tokenizer(
            rendered_text, add_special_tokens=False, return_offsets_mapping=True
        )
        # 取出 token id 序列
        token_ids = tokenized["input_ids"]
        # 取出字符偏移映射（快分词器才支持）
        offset_mapping = tokenized.get("offset_mapping")

        if offset_mapping is None:
            # 没有 offset mapping 就无法建立字符级 mask，直接报错
            raise ValueError(
                "Qwen3.5 loss mask generation requires a fast tokenizer "
                "with `return_offsets_mapping` support."
            )

        # 作为一致性校验：用模板自带的 tokenize=True 路径再编码一次
        expected_token_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, tools=tools, return_dict=False
        )
        if token_ids != expected_token_ids:
            # 两条路径结果不一致说明分词行为有偏差，宁可报错也不产生错误 mask
            raise ValueError(
                "Qwen3.5 rendered text tokenization does not match "
                "`apply_chat_template(..., tokenize=True)` output."
            )

        # assistant 消息在模板文本中的固定起始标记
        assistant_header = "<|im_start|>assistant\n"
        # 思考（thinking）区段的固定前缀
        think_prefix = "<think>\n"
        # 消息结束标记
        end_marker = "<|im_end|>"
        # 初始化字符级 mask：全 0 表示默认不训练
        char_mask = [0] * len(rendered_text)
        # 游标：记录当前已扫描到的文本位置，逐条 assistant 消息向后推进
        cursor = 0

        # 遍历所有消息，只处理 assistant 角色
        for message in messages:
            if message["role"] != "assistant":
                # 非 assistant 消息不参与训练，跳过
                continue

            # 从游标处向后定位该条 assistant 消息的起始标记
            header_pos = rendered_text.find(assistant_header, cursor)
            if header_pos < 0:
                # 找不到说明渲染结果与消息列表不对应，报错
                raise ValueError(
                    "Failed to locate assistant message in rendered Qwen3.5 chat template output."
                )

            # 正文从起始标记之后开始
            content_start = header_pos + len(assistant_header)
            # 定位本条消息的结束标记
            end_pos = rendered_text.find(end_marker, content_start)
            if end_pos < 0:
                # 找不到结束标记说明模板输出异常，报错
                raise ValueError(
                    "Failed to locate <|im_end|> for assistant message in rendered text."
                )

            # 训练区间右端：包含结束标记本身
            span_end = end_pos + len(end_marker)
            # 若结束标记后紧跟换行符，则把换行也纳入训练区间（与官方训练口径一致）
            if span_end < len(rendered_text) and rendered_text[span_end] == "\n":
                span_end += 1
            # 推进游标到本条消息末尾，供下一条查找使用
            cursor = span_end

            # step_loss_mask 开关关闭时整条消息不训练
            if message.get("step_loss_mask", 1) != 1:
                continue

            # 若正文以 <think> 开头，则从思考内容开始训练（跳过 <think> 标记本身）
            if rendered_text[content_start : content_start + len(think_prefix)] == think_prefix:
                mask_start = content_start + len(think_prefix)
            else:
                # 没有思考前缀则从正文起点开始训练
                mask_start = content_start

            # 把 [mask_start, span_end) 的字符全部标记为参与训练
            for position in range(mask_start, span_end):
                char_mask[position] = 1

        # 构建字符级前缀和：prefix_sum[i] 表示前 i 个字符中 1 的个数
        char_mask_prefix_sum = [0]
        for value in char_mask:
            char_mask_prefix_sum.append(char_mask_prefix_sum[-1] + value)

        # 由字符级 mask 推导 token 级 loss mask
        loss_mask = []
        for start, end in offset_mapping:
            if end <= start:
                # 空 span（特殊 token 无对应字符）不参与训练
                loss_mask.append(0)
            else:
                # 用前缀和 O(1) 求该 token 覆盖的字符中是否含训练字符
                covered = char_mask_prefix_sum[end] - char_mask_prefix_sum[start]
                # 只要覆盖到任一训练字符，该 token 即参与 loss
                loss_mask.append(1 if covered > 0 else 0)

        # 返回（token 序列, token 级 loss mask）
        return token_ids, loss_mask

    def gen_multi_turn_loss_mask_distill_qwen(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        # 蒸馏场景：只取首条消息（prompt），并追加生成提示符
        prompt = self.tokenizer.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True, tools=tools
        )
        # 取最后一条消息的内容作为模型要学习的响应
        response = messages[-1]["content"]
        # 编码 prompt（不加特殊 token）
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        # 编码 response（不加特殊 token）
        response_tokens = self.tokenizer(response, add_special_tokens=False)["input_ids"]

        # 记录响应 token 数
        response_length = len(response_tokens)
        # 拼接完整序列 = prompt + response
        token_ids = prompt_tokens + response_tokens
        # prompt 部分置 0 不训练，response 部分置 1 参与 loss
        loss_mask = [0] * len(prompt_tokens) + [1] * response_length

        # step_loss_mask 开关关闭时整条样本不训练
        if messages[-1].get("step_loss_mask", 1) != 1:
            loss_mask = [0] * len(token_ids)
        # 返回（token 序列, loss mask）
        return token_ids, loss_mask

    def get_loss_mask(self, messages: list[dict], tools: list[dict] = None) -> tuple[list[int], list[int]]:
        # 根据 tokenizer 类型分发到对应的 mask 生成实现
        if self.tokenizer_type == "qwen":
            # 词表中含 DeepSeek 风格的 Assistant 标记时按蒸馏模型处理
            if "<｜Assistant｜>" in self.tokenizer.get_added_vocab():
                return self.gen_multi_turn_loss_mask_distill_qwen(messages, tools)

            # 普通 qwen 模型走多轮模板方案
            return self.gen_multi_turn_loss_mask_qwen(messages, tools)
        elif self.tokenizer_type == "qwen3":
            # qwen3 使用哨兵消息切分方案
            return self.gen_multi_turn_loss_mask_qwen3(messages, tools)
        elif self.tokenizer_type == "qwen3_5":
            # qwen3.5 使用整体渲染 + 字符偏移映射方案
            return self.gen_multi_turn_loss_mask_qwen3_5(messages, tools)
        elif self.tokenizer_type == "distill_qwen":
            # 显式声明为蒸馏模型时使用 prompt/response 二段式方案
            return self.gen_multi_turn_loss_mask_distill_qwen(messages, tools)
        else:
            # 未知类型直接报错，避免生成错误 mask
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")

    def get_loss_mask_with_multimodal_alignment(
        self, messages: list[dict], input_ids: list[int], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        # 多模态场景：processor 产出的 input_ids 与纯文本分词不一致，需要按长度对齐
        text = []
        # 第一步：把多模态消息扁平化为纯文本消息
        for msg in messages:
            if isinstance(msg.get("content"), list):
                # content 为列表（多模态格式），抽取其中的文本片段
                text_parts = []
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        # 标准多模态字典项：取 text 字段
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        # 直接就是字符串的项：原样保留
                        text_parts.append(item)
                # 用空格拼接文本片段，重建纯文本消息（保留 role）
                text.append({"role": msg["role"], "content": " ".join(text_parts)})
            else:
                # 本来就是纯文本消息，原样保留
                text.append(msg)

        # 第二步：用文本路径生成 loss mask（不使用其 token 序列）
        _, loss_mask_text = self.get_loss_mask(text, tools=tools)

        # 计算 processor 序列比文本序列多出的长度（多模态占位 token 导致）
        diff = len(input_ids) - len(loss_mask_text)
        assert diff >= 0, (
            f"input_ids (length={len(input_ids)}) is shorter than text loss_mask (length={len(loss_mask_text)}) "
            f"Please check if processor and tokenizer tokenization are consistent."
        )
        # 多出的前缀部分（图像占位等）置 0 不训练，mask 尾部对齐
        loss_mask = [0] * diff + loss_mask_text

        # 返回（processor 的 input_ids, 对齐后的 loss mask）
        return input_ids, loss_mask

    def get_text_from_loss_mask(self, token_ids: list[int], loss_masks: list[int]) -> list[str]:
        # 收集解码出的文本片段
        selected_texts = []
        # 缓存当前连续 mask==1 的 token
        current_tokens = []

        # 逐位置扫描 mask，切分出连续的训练区段
        for idx, mask in enumerate(loss_masks):
            if mask == 1:
                # mask 为 1：token 加入当前区段
                current_tokens.append(token_ids[idx])
            elif current_tokens:
                # 遇到 0 且当前区段非空：解码并收尾
                selected_texts.append(self.tokenizer.decode(current_tokens))
                current_tokens = []

        # 序列结尾仍处于训练区段时，解码最后一段
        if current_tokens:
            selected_texts.append(self.tokenizer.decode(current_tokens))

        # 返回所有被训练（mask==1）的文本片段
        return selected_texts
