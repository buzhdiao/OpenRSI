# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 Callable 类型注解，用于标注算子函数参数
from typing import Callable

# 导入响应解析工具：提取代码块、截取代码前的文本、解析思考标签
from dojo.core.solvers.utils.response import extract_code, extract_text_up_to_code, parse_thinking_tags


def execute_op_plan_code(
    operator_fn: Callable, *operator_args, max_operator_tries: int, requires_plan: bool = False
) -> tuple[str, str, str]:
    """Executes an operator function with the given arguments, attempts to extract the generated plan/code from the output
    and retries if the extraction fails."""
    # 补全文本初始化为 None（全部重试失败时也会被最后返回值引用）
    completion_text = None
    # 去掉思考内容后的文本初始化为 None
    text_without_thinking = None

    # 最多重试 max_operator_tries 次，直到成功提取出代码
    for _ in range(max_operator_tries):
        # 执行算子函数（通常是调 LLM 生成方案/代码），拿到补全文本与指标
        completion_text, metrics = operator_fn(*operator_args)
        # 用 parse_thinking_tags 把 <thinking> 标签内容与正文分离（Chain-of-Thought，思维链内容单独剥离）
        thinking_text, text_without_thinking = parse_thinking_tags(completion_text)
        # 从正文中提取代码块（位于 ``` 代码围栏中的部分）
        code = extract_code(text_without_thinking)
        # 提取代码之前的文本作为方案说明（plan，即计划部分）
        plan = extract_text_up_to_code(text_without_thinking)

        if code:
            # 成功提取到代码
            if requires_plan and not plan:
                # 若该算子要求必须附带方案说明但没有提取到 plan，则打印提示并重试
                print("Plan extraction failed, retrying...")
                continue
            # 满足要求时返回（方案, 代码, 指标）三元组
            return plan, code, metrics

        # 未能提取到代码：打印重试提示并进入下一轮
        print("Retrying Extraction...")

    # 所有重试均失败：返回空方案、最后一次的正文文本与指标
    return "", text_without_thinking, metrics
