# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

def get_client(client_cfg):
    # 客户端工厂：根据配置中的 api 类型选择并实例化对应的 LLM 后端客户端
    match client_cfg.api:
        case "openai":
            # openai 类型：延迟导入 OpenAI 客户端，避免加载未用到的后端
            from dojo.core.solvers.llm_helpers.backends.open_ai import OpenAIClient

            # 用配置初始化并返回 OpenAI 客户端
            return OpenAIClient(client_cfg)
        case "litellm":
            # litellm 类型：延迟导入 LiteLLM 统一网关客户端
            from dojo.core.solvers.llm_helpers.backends.lite_llm import LiteLLMClient

            # 用配置初始化并返回 LiteLLM 客户端
            return LiteLLMClient(client_cfg)
        case "gdm":
            # gdm 类型：延迟导入 GDM 客户端
            from dojo.core.solvers.llm_helpers.backends.gdm import GDMClient

            # 用配置初始化并返回 GDM 客户端
            return GDMClient(client_cfg)
        case _:
            # 未知 API 类型：抛出异常提示调用方检查配置
            raise Exception(f"Unknown API: {client_cfg['api']}")
