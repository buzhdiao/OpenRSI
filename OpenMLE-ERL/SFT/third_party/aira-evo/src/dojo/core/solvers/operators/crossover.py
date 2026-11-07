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
from typing import Callable, Optional

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
# 导入复杂度等级枚举
from dojo.solvers.utils import Complexity


def crossover_op(
    crossover_llm: GenericLLM,
    cfg: DictConfig,
    memory_op: Optional[Callable[[Journal, Optional[Node]], str]],
    prompt_context: dict[str, str],
    task_description: str,
    journal: Journal,
    input_node1: Node,
    input_node2: Node,
    step_count: int,
    remaining_time: int,
    data_preview: Optional[str] = None,
) -> str:
    # 交叉算子（Crossover Operator）：融合两个父节点的方案/代码，生成新的子代方案（对应进化算法中的交叉重组）
    # 取出配置中允许使用的第三方包列表
    pkgs = cfg.available_packages
    # 随机打乱包顺序，避免提示词中包的排列顺序对模型产生固定偏置
    random.shuffle(pkgs)
    # 把包名拼接成反引号包裹的字符串，方便放进提示词
    pkg_str = ", ".join([f"`{p}`" for p in pkgs])

    # 执行超时取"配置超时"与"剩余总时间"中较小者，防止超预算
    exec_timeout = int(min(cfg.execution_timeout, remaining_time))
    # 计算剩余的进化步数
    steps_remaining = cfg.step_limit - step_count
    if memory_op is not None:
        # 配置了记忆算子时，基于日志与第一个父节点生成记忆文本
        memory = memory_op(journal, input_node1)
    else:
        # 未配置记忆算子则记为空
        memory = ""

    # 组装渲染交叉提示词所需的全部数据
    improve_data = {
        "task_desc": task_description,
        "task_description": prompt_context.get("task_description") or task_description,
        "data_description": prompt_context.get("data_description", ""),
        "public_system_prompt": prompt_context.get("public_system_prompt", ""),
        "public_user_prompt": prompt_context.get("public_user_prompt", ""),
        "prev_code1": wrap_code(input_node1.code),
        "prev_terminal_output1": wrap_code(input_node1.term_out, lang=""),
        "prev_code2": wrap_code(input_node2.code),
        "prev_terminal_output2": wrap_code(input_node2.term_out, lang=""),
        "time_remaining": humanize.naturaldelta(remaining_time),
        "steps_remaining": steps_remaining,
        "execution_timeout": humanize.naturaldelta(exec_timeout),
        "other_remarks": None,
        "improve_complexity": None,
        "memory": None,
        "data_overview": None,
        "packages": pkg_str,
    }

    if memory:
        # 有记忆内容时才写入模板变量，避免空值占位
        improve_data["memory"] = memory

    if cfg.data_preview and data_preview is not None:
        # 配置开启且提供了数据预览时，把数据概览写入模板变量
        improve_data["data_overview"] = data_preview
    else:
        # 否则填入占位说明
        improve_data["data_overview"] = "(No data preview available)"

    # 用组装好的数据调用交叉 LLM，返回生成的新方案/代码
    return crossover_llm(query_data=improve_data)
