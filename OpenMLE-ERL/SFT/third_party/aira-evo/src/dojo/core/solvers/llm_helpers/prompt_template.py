# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 jinja2 的 Environment 与 StrictUndefined：前者负责模板渲染，后者让未定义变量直接报错而非留空
from jinja2 import Environment, StrictUndefined
# 导入 hydra，用于把配置对象实例化
import hydra

# 导入 Jinja 提示词模板的配置数据类
from dojo.config_dataclasses.llm.jinjaprompt import JinjaPromptConfig


class JinjaPrompt:
    r"""This class can be used to generate prompts from jinja templates

    :param \**kwargs:
        See below:
    :Keyword Arguments:
        * *input_variables* (``List[str]``) --
            A list of variables that are required to render the template
        * *partial_variables* (``Dict[str, Any]``) --
            A dictionary of variables and their values that are required to render the template (useful when one has some variables before others)
        * *template* (``str``) --
            The jinja template to render
    """

    def __init__(self, cfg: JinjaPromptConfig):
        # This is temporary -- under the desired setup this will be done in main under a single call
        # 用 hydra 把配置对象实例化为真正的 JinjaPromptConfig（临时做法，后续应统一在 main 中完成）
        cfg = hydra.utils.instantiate(cfg, _recursive_=False)
        # 记录模板渲染所必需的输入变量集合
        self.input_variables: set = set(cfg.input_variables)
        # 记录"部分变量"：提前可得的变量及其取值，渲染时与其他变量合并
        self.partial_variables = cfg.partial_variables
        # 保存 jinja 模板原文
        self.template: str = cfg.template
        # 创建 jinja 渲染环境；StrictUndefined 保证缺失变量时抛错而不是静默输出空值
        self.environment = Environment(undefined=StrictUndefined)

    def format(self, **kwargs):
        r"""format the template with the given input variables

        :param \**kwargs: The input variables to render the template (should be a subset of the input variables)
        :return: The rendered template
        :rtype: str
        """
        # 把模板字符串编译为可渲染的模板对象
        template = self.environment.from_string(self.template)
        # 合并"部分变量"与本次传入的变量，后者优先级更高（同名时覆盖前者）
        merged_args = {**self.partial_variables, **kwargs}
        # 用合并后的变量渲染模板并返回最终提示词文本
        return template.render(**merged_args)
