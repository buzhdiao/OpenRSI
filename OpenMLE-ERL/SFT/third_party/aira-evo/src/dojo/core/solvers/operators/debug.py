# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 os 模块（保留以维持接口一致性）
import os
# 导入 random 模块，用于打乱可用包列表
import random
# 导入 time 模块（保留以维持接口一致性）
import time
# 导入类型注解工具
from typing import Optional, Callable

# 导入 humanize，用于把秒数转成人类可读的时长描述
import humanize
# 导入 OmegaConf 的 DictConfig 类型注解
from omegaconf import DictConfig

# 导入通用 LLM 封装层
from dojo.core.solvers.llm_helpers.generic_llm import GenericLLM
# 导入进化日志（Journal）与节点（Node）类型
from dojo.core.solvers.utils.journal import Journal, Node
# 导入 wrap_code 工具：把文本包进代码围栏
from dojo.core.solvers.utils.response import wrap_code


def debug_op(
    debug_llm: GenericLLM,
    cfg: DictConfig,
    memory_op: Optional[Callable[[Journal, Optional[Node]], str]],
    prompt_context: dict[str, str],
    task_description: str,
    journal: Journal,
    input_node: Node,
    step_count: int,
    remaining_time: int,
    data_preview: Optional[str] = None,
) -> str:
    # 调试算子（Debug Operator）：根据出错代码与执行输出，让 LLM 定位并修复 bug
    # 取出配置中允许使用的第三方包列表
    pkgs = cfg.available_packages
    # 随机打乱包顺序，避免提示词顺序偏置
    random.shuffle(pkgs)
    # 把包名拼接成反引号包裹的字符串
    pkg_str = ", ".join([f"`{p}`" for p in pkgs])

    if memory_op is not None:
        # 配置了记忆算子时，基于日志与当前节点生成记忆文本
        memory = memory_op(journal, input_node)
    else:
        # 未配置则记为 None
        memory = None

    # 取出出错节点的代码
    prev_buggy_code = input_node.code
    # 取出出错节点的终端执行输出
    execution_output = input_node.term_out

    # 执行超时取"配置超时"与"剩余总时间"中较小者
    exec_timeout = int(min(cfg.execution_timeout, remaining_time))
    # 计算剩余的进化步数
    steps_remaining = cfg.step_limit - step_count

    # 组装渲染调试提示词所需的全部数据
    debug_data = {
        "task_desc": task_description,
        "task_description": prompt_context.get("task_description") or task_description,
        "data_description": prompt_context.get("data_description", ""),
        "public_system_prompt": prompt_context.get("public_system_prompt", ""),
        "public_user_prompt": prompt_context.get("public_user_prompt", ""),
        "prev_buggy_code": wrap_code(prev_buggy_code),
        "execution_output": wrap_code(execution_output, lang=""),
        "time_remaining": humanize.naturaldelta(remaining_time),
        "steps_remaining": steps_remaining,
        "execution_timeout": humanize.naturaldelta(exec_timeout),
        "other_remarks": None,
        "packages": pkg_str,
        "memory": None,
        "data_overview": None,
    }

    if memory:
        # 有记忆内容时才写入模板变量
        debug_data["memory"] = memory

    if cfg.data_preview and data_preview is not None:
        # 配置开启且提供了数据预览时写入数据概览
        debug_data["data_overview"] = data_preview
    else:
        # 否则填入占位说明
        debug_data["data_overview"] = "(No data preview available)"

    # 用组装好的数据调用调试 LLM，返回修复后的方案/代码
    return debug_llm(query_data=debug_data)
