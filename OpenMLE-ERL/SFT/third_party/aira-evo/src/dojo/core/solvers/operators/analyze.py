# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 OmegaConf 的 DictConfig 类型注解
from omegaconf import DictConfig

# 导入通用 LLM 封装层，用于发起分析请求
from dojo.core.solvers.llm_helpers.generic_llm import GenericLLM
# 导入 wrap_code 工具：把代码/日志包进代码围栏便于 LLM 阅读
from dojo.core.solvers.utils.response import wrap_code
# 导入进化树的节点类型 Node
from dojo.core.solvers.utils.journal import Node

# 无评估指标场景的分析结果 JSON Schema（约束 LLM 输出 is_bug 与 summary 两个字段）
analyze_schema_without_eval = """{
    "type": "object",
    "properties": {
        "is_bug": {
            "type": "boolean",
            "description": "true if the output log shows that the execution failed or has some bug, otherwise false."
        },
        "summary": {
            "type": "string",
            "minLength": 1,
            "description": "Required. If there is a bug, summarize the root cause and propose a fix. Otherwise, write a short summary (2-3 sentences) of the empirical findings."
        }
    },
    "required": ["is_bug", "summary"],
    "additionalProperties": false
}"""

# 带评估指标场景的分析结果 JSON Schema（额外要求返回 metric 数值或 null）
analyze_schema_with_eval = """{
    "type": "object",
    "properties": {
        "is_bug": {
            "type": "boolean",
            "description": "true if the output log shows that the execution failed or has some bug, otherwise false."
        },
        "summary": {
            "type": "string",
            "minLength": 1,
            "description": "Required. If there is a bug, summarize the root cause and propose a fix. Otherwise, write a short summary (2-3 sentences) of the empirical findings. DO NOT suggest fixes or improvements for successful runs."
        },
        "metric": {
                "type": ["number", "null"],
                "description": "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null."
            }
    },
    "required": ["is_bug", "summary", "metric"],
    "additionalProperties": false
}"""


def analyze_op(
    analyze_llm: GenericLLM,
    cfg: DictConfig,
    task_description: str,
    input_node: Node,
    fetch_metric: bool = True,
) -> str:
    # 分析算子（Analyze Operator）：让 LLM 解读代码执行结果，判断是否有 bug 并总结发现
    # 从输入节点取出已执行的代码
    code = input_node.code
    # 从输入节点取出脚本运行的终端输出
    execution_output = input_node.term_out

    if fetch_metric:
        # 需要评估指标的场景：给 LLM 更严格的输出格式说明（含 metric 字段）
        output_format = (
            "Return exactly one valid JSON object and nothing else. "
            "Use exactly these keys: is_bug, summary, metric. "
            "Do not add keys such as bugs, empirical_findings, validation_auc, "
            "analysis, result, or explanation. "
            "The JSON shape must be: "
            "{\"is_bug\": false, \"summary\": \"non-empty summary\", \"metric\": 0.0}. "
            "Set is_bug to true only when execution failed or the output shows a bug. "
            "If execution failed, still return the same JSON shape with "
            "is_bug=true, metric=null, and a summary that explains the error. "
            "Set metric to the numeric validation metric if it is reported; otherwise set metric to null. "
            "The summary field is required and must be non-empty. "
            "Never return an empty object or an empty code block. "
            "Do not return markdown, code fences, headings, or prose outside the JSON object."
        )
    else:
        # 不需要评估指标的场景：输出格式说明只要求 is_bug 与 summary 两个键
        output_format = (
            "Return exactly one valid JSON object and nothing else. "
            "Use exactly these keys: is_bug, summary. "
            "Do not add keys such as bugs, empirical_findings, analysis, result, or explanation. "
            "The JSON shape must be: "
            "{\"is_bug\": false, \"summary\": \"non-empty summary\"}. "
            "If execution failed, still return the same JSON shape with "
            "is_bug=true and a summary that explains the error. "
            "The summary field is required and must be non-empty. "
            "Never return an empty object or an empty code block. "
            "Do not return markdown, code fences, headings, or prose outside the JSON object."
        )

    # 组装渲染提示词所需的数据：任务描述、代码、执行输出、输出格式要求
    analyze_data = {
        "task_desc": task_description,
        "code": wrap_code(code),
        "execution_output": wrap_code(execution_output, lang=""),
        "output_format": output_format,
    }

    # 按是否需要评估指标选择对应的 JSON Schema
    schema = analyze_schema_with_eval if fetch_metric else analyze_schema_without_eval

    # 发起 LLM 分析请求：以 submit_review 函数调用形式强制输出结构化评审结果
    return analyze_llm(
        query_data=analyze_data,
        json_schema=schema,
        function_name="submit_review",
        function_description="Submit a review evaluating the output of the training script.",
        no_user_message=True,
    )
