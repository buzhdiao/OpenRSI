# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are MIT-licensed
# Copyright (c) 2024 John Yang, Carlos E. Jimenez,
# Alexander Wettig, Shunyu Yao, Karthik Narasimhan, Ofir Press
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/SWE-agent/SWE-agent/blob/main/LICENSE

# 延迟注解求值
from __future__ import annotations

# dataclass 装饰器与字段枚举工具
from dataclasses import dataclass, fields


# 配置模型及其行为风格的参数集合
@dataclass
class ModelArguments:
    """Arguments configuring the model and it's behavior."""

    # Name of the model to use
    # 使用的模型名称
    model_name: str
    # Cost limit for every task
    # 单个任务的 API 花费上限
    per_instance_cost_limit: float = 0.0
    # Total cost limit
    # 全程总花费上限（0 表示不限制）
    total_cost_limit: float = 0.0
    # Sampling temperature
    # 采样温度：越高越随机（探索），越低越确定（利用）
    temperature: float = 0.8
    # Sampling top_p
    # 核采样（top-p）参数：只在累计概率前 p 的词表中采样
    top_p: float = 1.0
    # Path to replay file when using the replay model
    # 使用"回放模型"时的录制文件路径（重放历史响应，不真正调用 API）
    replay_path: str | None = None
    # Host URL when using Ollama model
    # 使用 Ollama 本地模型时的服务地址
    host_url: str = "localhost:11434"


# API 调用统计：累计花费、token 数与调用次数
@dataclass
class APIStats:
    # 累计总花费
    total_cost: float = 0.0
    # 当前任务累计花费
    task_cost: float = 0.0
    # 累计发送的 token 数
    tokens_sent: int = 0
    # 累计接收的 token 数
    tokens_received: int = 0
    # 累计 API 调用次数
    api_calls: int = 0

    def __add__(self, other: APIStats):
        # 只允许同类型相加
        if not isinstance(other, APIStats):
            msg = "Can only add APIStats with APIStats, got type {type(other)}"
            raise TypeError(msg)

        # 逐字段相加，返回新的统计对象
        return APIStats(
            **{field.name: getattr(self, field.name) + getattr(other, field.name) for field in fields(self)},
        )

    def replace(self, other):
        # 只允许同类型替换
        if not isinstance(other, APIStats):
            msg = "Can only replace APIStats with APIStats"
            raise TypeError(msg)

        # 用 other 的全部字段值构造新对象
        return APIStats(**{field.name: getattr(other, field.name) for field in fields(self)})


# 上下文窗口（context window）超出限制时抛出的异常
class ContextWindowExceededError(Exception):
    pass


# API 花费超过预设上限时抛出的异常
class CostLimitExceededError(Exception):
    pass


# 通用 API 调用错误
class APIError(Exception):
    pass


# 触发 API 速率限制（rate limit）时抛出的异常
class RateLimitExceededError(Exception):
    pass
