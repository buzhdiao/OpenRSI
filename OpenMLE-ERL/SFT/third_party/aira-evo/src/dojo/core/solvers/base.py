# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 时间工具：用单调时钟记录求解开始时间
import time
# 抽象基类与抽象方法
from abc import ABC, abstractmethod
# 文件路径对象
from pathlib import Path
# 检查点状态的 JSON 序列化
import json

# OmegaConf 配置库（供子类使用）
from omegaconf import OmegaConf

# 日志工具：全局日志器与事件枚举
from dojo.utils.logger import get_logger, LogEvent

# 求解器配置类
from dojo.config_dataclasses.solver.base import SolverConfig
# 求解器运行状态容器
from dojo.utils.state import BaseState


# 求解器抽象基类：封装公共配置、日志、计时、运行状态与断点续跑能力；
# 具体搜索策略（贪心/进化/MCTS）由子类实现 __call__
class Solver(ABC):
    def __init__(self, cfg: SolverConfig, task_info=None):
        # 保存求解器配置
        self.cfg = cfg
        # 获取全局聚合日志器
        self.logger = get_logger()
        # 保存任务描述信息（可选）
        self.task_info = task_info
        # 用单调时钟记录起始时间（不受系统时间回拨影响）
        self.start_time = time.monotonic()

        # state
        # 初始化基础状态容器（步数、历史等）
        self.state = BaseState()

    # 主搜索循环接口：子类必须实现
    @abstractmethod
    def __call__(self, task, state):
        raise NotImplementedError()

    def save_checkpoint(self):
        # 保存检查点：把当前状态序列化为 JSON 存到 checkpoint_path
        self.logger.info(f"Saving checkpoint to {self.cfg.checkpoint_path}")
        # 确保检查点目录存在
        Path(self.cfg.checkpoint_path).mkdir(parents=True, exist_ok=True)
        # 把状态对象导出为字典
        state_dict = self.state.state_dict()
        # 写入 state.json 留档
        with open(Path(self.cfg.checkpoint_path) / "state.json", "w") as f:
            json.dump(state_dict, f)

    def load_checkpoint(self):
        # 加载检查点：从 checkpoint_path 恢复状态；不存在则告警并从头开始
        state_path = Path(self.cfg.checkpoint_path) / "state.json"
        # 检查点不存在
        if not state_path.exists():
            self.logger.warning(f"No checkpoint found at {state_path}. Proceeding without loading.")
            return
        # 记录开始加载
        self.logger.info(f"Found state checkpoint at {state_path}. Loading...")
        # 读取并反序列化状态字典
        with open(state_path, "r") as f:
            state_dict = json.load(f)

        # 恢复状态对象
        self.state.load_state_dict(state_dict)

        # Increment the number of starts
        # 启动次数自增（记录这是第几次续跑）
        self.state.num_starts += 1
