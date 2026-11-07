# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 os 模块：操作系统相关的路径与环境访问
import os
# 导入 subprocess 模块：在任务中启动子进程执行外部命令
import subprocess
# 导入 ABC 与 abstractmethod：定义抽象基类与抽象方法（任务接口约定）
from abc import ABC, abstractmethod
# 导入常用类型注解：任意类型、字典、可选、元组
from typing import Any, Dict, Optional, Tuple

# 导入 OmegaConf：分层结构化配置库
from omegaconf import OmegaConf

# 导入日志工厂函数：获取统一配置的 logger 实例
from dojo.utils.logger import get_logger

# 导入任务的配置数据类（dataclass），定义任务参数结构
from dojo.config_dataclasses.task.base import TaskConfig


# Task 抽象基类：类似强化学习中的"环境"抽象，规定智能体所处任务必须实现的接口
class Task(ABC):
    """
    Abstract base class representing a Task.

    A task specifies:
      - The execution environment available to the agent (e.g., local resources or a VM with a specific configuration).
      - The interface through which the agent can interact with the environment (e.g., bash shell or Python Interpreter)
        and the observation space.
      - A description of the task (e.g., the goal, the constraints, the reward, a description of the execution environment
        and the interface).
      - The fitness function that evaluates the agent's performance on the task.
    """

    # 构造函数：保存任务配置并初始化日志器
    def __init__(self, cfg: TaskConfig) -> None:
        """
        Initialize a Task instance.

        Args:
            **cfg: Arbitrary keyword arguments representing the configuration parameters for the task.
        """
        # Convert the provided configuration dictionary into an OmegaConf object for structured access.
        # 保存任务配置对象，供后续方法读取任务参数
        self.cfg = cfg
        # 获取全局 logger，供任务执行过程中记录日志
        self.logger = get_logger()  # Store the logger for use in other methods if needed.

    # 抽象方法：准备任务执行所需的一切（如把数据复制到智能体工作区）
    @abstractmethod
    def prepare(self, **task_args: Optional[Dict]) -> Dict:
        """
        Prepares everything for the task execution to start (e.g., copies data to the agent's workspace).

        This method should be overridden to perform any necessary setup before the task execution begins.

        Args:
            task_args: The arguments needed to prepare for the task execution.
        Returns:
            Dict[str, Any]: The initial state of the task (potentially including the initial observation)
            Dict[str, Any]: Any task_info useful when instantiating the solver.
        """
        # 抽象方法无实现体，必须由子类覆盖
        pass

    # 抽象方法：执行任务的单步交互——给定当前状态与动作，返回新状态与执行结果
    @abstractmethod
    def step_task(self, state: Dict, action: Any) -> Tuple[Dict, Dict]:
        """
        Execute a single step of the task using the provided action.

        This method should be overridden to define how a task processes an action and produces an outcome.

        Args: sv
            state (Dict): The current state of the task environment
            action (Any): The action to be performed in the task environment.

        Returns:
            Tuple[Dict, Dict]: A tuple containing the new state of the task environment and the outcome of the action.
        """
        # 由子类实现具体的状态转移逻辑
        pass

    # 抽象方法：适应度函数(Fitness Function)——评估解的质量/正确性，是搜索过程的优化目标
    @abstractmethod
    def evaluate_fitness(
        self,
        solution: Optional[Dict] = None,
        state: Optional[Dict] = None,
        interpreter: Optional[Dict] = None,
        aux_info: Dict[str, Any] = None,
    ) -> Any:
        """
        Evaluates how good (or correct) the provided solution is for this task.

        Args:
            solution (Dict, optional): The solution to be evaluated.
            state (Dict, optional): The current state of the task environment.
            interpreter (Dict, optional): The interpreter instance that provides an execution environment for the evaluation.
            aux_info (Dict): Additional information that may be needed during the evaluation.

        Returns:
            Any: An object representing the result of the fitness evaluation (e.g., a score, success/failure status, etc.).
        """
        # 由子类实现具体的适应度评估逻辑
        pass

    # 抽象方法：清理并释放任务占用的资源（工作区、进程等）
    @abstractmethod
    def close(self, state: Dict) -> None:
        """
        Perform cleanup and close any resources associated with the task.

        This method should be overridden to properly shut down the task and free resources when the task is completed.

        Args:
            state (Dict): The final state of the task.
        """
        # 由子类实现具体的清理逻辑
        pass
