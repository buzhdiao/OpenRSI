# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# 任务状态基类：记录通用运行指标（累计运行时长与启动次数），支持字典序列化/恢复
class BaseState:
    # 已累计的运行时间（秒）
    running_time: float = 0.0
    # 已启动（重启）的次数
    num_starts: int = 0

    def __init__(self):
        # 无需额外初始化，属性沿用类级默认值
        pass

    # 把当前状态导出为字典，便于保存与恢复
    def state_dict(self):
        return {
            "running_time": self.running_time,
            "num_starts": self.num_starts,
        }

    # 从字典恢复状态
    def load_state_dict(self, state_dict):
        # 恢复累计运行时间
        self.running_time = state_dict["running_time"]
        # 恢复启动次数（旧记录缺失时按 0 处理）
        self.num_starts = state_dict.get("num_starts", 0)


# 贪心（Greedy）搜索策略的运行状态
class GreedyState:
    # 当前执行到的步数
    current_step: int = 0
    # 已累计的运行时间（秒）
    running_time: float = 0.0
    # 已启动（重启）的次数
    num_starts: int = 0

    def __init__(self):
        # 无需额外初始化
        pass

    # 导出状态为字典（含世代数，供序列化保存）
    def state_dict(self):
        return {
            "current_step": self.current_step,
            "current_generation": self.current_generation,
            "running_time": self.running_time,
            "num_starts": self.num_starts,
        }

    # 从字典恢复各项状态
    def load_state_dict(self, state_dict):
        # 恢复当前步数
        self.current_step = state_dict["current_step"]
        # 恢复世代数（缺失时按 0 处理）
        self.current_generation = state_dict.get("current_generation", 0)
        # 恢复累计运行时间
        self.running_time = state_dict["running_time"]
        # 恢复启动次数（缺失时按 0 处理）
        self.num_starts = state_dict.get("num_starts", 0)


# 蒙特卡洛树搜索（MCTS, Monte Carlo Tree Search）策略的运行状态
class MCTSState:
    # 当前执行到的步数
    current_step: int = 0
    # 已累计的运行时间（秒）
    running_time: float = 0.0
    # 已启动（重启）的次数
    num_starts: int = 0

    def __init__(self):
        # 无需额外初始化
        pass

    # 导出状态为字典
    def state_dict(self):
        return {
            "current_step": self.current_step,
            "running_time": self.running_time,
            "num_starts": self.num_starts,
        }

    # 从字典恢复各项状态
    def load_state_dict(self, state_dict):
        # 恢复当前步数
        self.current_step = state_dict["current_step"]
        # 恢复累计运行时间
        self.running_time = state_dict["running_time"]
        # 恢复启动次数（缺失时按 0 处理）
        self.num_starts = state_dict.get("num_starts", 0)


# 进化（Evolutionary）策略的运行状态：比基类多记录当前代数
class EvolutionaryState:
    # 当前执行到的步数
    current_step: int = 0
    # 已累计的运行时间（秒）
    running_time: float = 0.0
    # 当前进化到的世代编号
    current_generation: int = 0
    # 已启动（重启）的次数
    num_starts: int = 0

    def __init__(self):
        # 无需额外初始化
        pass

    # 导出状态为字典（含世代数）
    def state_dict(self):
        return {
            "current_step": self.current_step,
            "current_generation": self.current_generation,
            "running_time": self.running_time,
            "num_starts": self.num_starts,
        }

    # 从字典恢复各项状态
    def load_state_dict(self, state_dict):
        # 恢复当前步数
        self.current_step = state_dict["current_step"]
        # 恢复世代数（缺失时按 0 处理）
        self.current_generation = state_dict.get("current_generation", 0)
        # 恢复累计运行时间
        self.running_time = state_dict["running_time"]
        # 恢复启动次数（缺失时按 0 处理）
        self.num_starts = state_dict.get("num_starts", 0)
