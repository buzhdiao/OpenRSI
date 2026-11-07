# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 os：读写环境变量、获取当前工作目录
import os
# 导入 logging：标准日志模块，用于模块级 logger
import logging
# 导入 Path：面向对象的文件路径操作
from pathlib import Path
# 导入 inspect：运行时内省，用于获取模块源码文件路径
import inspect

# 导入工具函数：把当前环境变量写入 JSON 文件存档
from dojo.utils.helpers import write_env_variables_to_json
# 导入 hydra：基于配置文件的命令行应用框架（@hydra.main 程序入口）
import hydra
# 导入 load_dotenv：从 .env 文件加载环境变量
from dotenv import load_dotenv
# 导入 DictConfig：OmegaConf 的字典式配置对象类型
from omegaconf import DictConfig

# 导入 rich_utils：基于 rich 的美化输出工具（如配置树打印）
from dojo.utils import rich_utils
# 导入日志配置函数与日志事件枚举
from dojo.utils.logger import config_logger, LogEvent
# 导入 OmegaConf：结构化配置库，用于解析/合并/校验配置
from omegaconf import OmegaConf
# 导入自定义 OmegaConf 插值解析器的注册函数
from dojo.config_dataclasses.omegaconf.resolvers import register_new_resolvers

# 导入单次运行(Run)的配置数据类
from dojo.config_dataclasses.run import RunConfig
# 导入任务名到任务类的映射表
from dojo.config_dataclasses.task import TASK_MAP
# 导入求解器名到求解器类的映射表
from dojo.config_dataclasses.solver import SOLVER_MAP
# 导入解释器名到解释器类的映射表
from dojo.config_dataclasses.interpreter import INTERPRETER_MAP
# 导入 build 工厂函数：按配置从映射表实例化对象
from dojo.utils.config import build
# 导入环境检测工具：硬件信息、PyTorch/TensorFlow GPU 可用性检查、时长格式化
from dojo.utils.environment import (
    get_hardware,
    check_pytorch_gpu,
    check_tensorflow_gpu,
    format_time,
)
# 导入获取 SLURM 作业 ID 的工具函数（HPC 集群调度环境）
from dojo.utils.slurm import get_slurm_id

# 加载 .env 文件中的环境变量（如 API 密钥等）
load_dotenv()
# 创建模块级 logger
log = logging.getLogger(__name__)

# 注册自定义 OmegaConf 插值解析器（如从环境变量取值的 resolver）
register_new_resolvers()


# 主运行流程：准备输出目录、设置环境变量、依次实例化任务/解释器/求解器并执行搜索
def _main(cfg: RunConfig):
    # Create the output directory if it doesn't exist
    # 调试日志：输出实验产物保存目录
    log.debug(f"Saving experiment artifacts to: {cfg.logger.output_dir}")
    # 断言：必须指定输出目录路径，否则中止
    assert cfg.logger.output_dir is not None, (
        "Path to the directory in which the launch artefacts will be written must be specified."
    )
    # 调试日志：再次输出输出目录
    log.debug(f"Output dir: {cfg.logger.output_dir}")
    # 将输出目录字符串转为 Path 对象
    output_dir = Path(cfg.logger.output_dir)
    # 递归创建输出目录（已存在则忽略）
    output_dir.mkdir(parents=True, exist_ok=True)
    # 将最终配置持久化保存到输出目录
    cfg.save()

    # Set environment variables
    # 检测当前硬件类型并写入环境变量（供任务/求解器读取）
    os.environ["HARDWARE"] = get_hardware()
    # 检测 PyTorch 能否使用 GPU 并写入环境变量
    os.environ["PYTORCH_GPU"] = check_pytorch_gpu()
    # 检测 TensorFlow 能否使用 GPU 并写入环境变量
    os.environ["TENSORFLOW_GPU"] = check_tensorflow_gpu()
    # 将求解器时间上限（秒）写入环境变量
    os.environ["TIME_LIMIT_SECS"] = str(cfg.solver.time_limit_secs)
    # 将秒数格式化为人类可读时长并写入环境变量
    os.environ["TIME_LIMIT"] = format_time(int(os.environ["TIME_LIMIT_SECS"]))
    # 将求解器步数上限写入环境变量
    os.environ["STEP_LIMIT"] = str(cfg.solver.step_limit)

    # Store the slurm job ID
    # 记录 SLURM 作业 ID 到元数据（非集群环境下可能为空）
    cfg.metadata.slurm_id = get_slurm_id()

    # Sanity checks
    # 日志：记录当前工作目录（应为快照目录或仓库根目录）
    log.info(f"Current working directory: {os.getcwd()}")
    # 延迟导入 dojo 包，用于确认实际加载的代码位置
    import dojo

    # 日志：输出 dojo 包源码路径，验证运行的是预期代码
    log.info(f"`dojo` package source path: {inspect.getsourcefile(dojo)}")
    # 延迟导入 aira_core 包并确认其位置
    import aira_core

    # 日志：输出 aira_core 包源码路径
    log.info(f"`aira_core` package source path: {inspect.getsourcefile(aira_core)}")
    # 延迟导入 mlebench 包并确认其位置
    import mlebench

    # 日志：输出 mlebench 包源码路径
    log.info(f"`mlebench` package source path: {inspect.getsourcefile(mlebench)}")

    # Create the output directory if it doesn't exist
    # 日志：输出实验产物保存目录
    log.info(f"Saving experiment artifacts to: {cfg.logger.output_dir}")
    # 再次断言输出目录已指定（与开头重复，属冗余保险）
    assert cfg.logger.output_dir is not None, (
        "Path to the directory in which the launch artefacts will be written must be specified."
    )
    # 日志：输出目录
    log.info(f"Output dir: {cfg.logger.output_dir}")
    # 再次构造输出目录 Path 对象
    output_dir = Path(cfg.logger.output_dir)
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    # 保存配置到磁盘
    cfg.save()

    # 若配置要求记录环境变量
    if cfg.logger.write_env_vars:
        # 将当前环境变量写入 JSON 文件，便于复现实验环境
        write_env_variables_to_json(cfg.logger.output_dir)

    # 按配置初始化实验日志器（用于记录指标等事件流）
    logger = config_logger(cfg)

    # 日志：开始实例化任务
    log.info("Instantiating the task...")
    # 根据任务配置从映射表构建任务对象
    task = build(cfg.task, TASK_MAP)

    # Allocate resources for the agent's workspace and instantiate an object that lets you reference and use them
    # 为智能体分配资源并构建其工作区解释器（代码执行环境）
    solver_interpreter = build(cfg.interpreter, INTERPRETER_MAP, data_dir=cfg.task.data_dir)

    # 评估用解释器置空（本流程不使用独立的评估解释器）
    eval_interpreter = None

    # 日志：准备各工作区
    log.info("Preparing the workspaces...")
    # 任务准备：复制数据、搭建工作区，返回初始状态与任务信息（含任务描述、优化方向）
    state, task_info = task.prepare(solver_interpreter=solver_interpreter, eval_interpreter=eval_interpreter)

    # 日志：开始实例化求解器
    log.info("Instantiating the solver...")
    # 根据求解器配置与任务信息构建求解器对象
    solver = build(cfg.solver, SOLVER_MAP, task_info=task_info)

    # Load checkpoint state if it exists
    # 日志：加载检查点（若存在）
    log.info("Loading checkpoints, if any...")
    # 从磁盘恢复求解器历史状态（支持断点续跑）
    solver.load_checkpoint()

    # 日志：启动求解器
    log.info("Starting the solver...")
    # 执行搜索主循环：返回更新后的状态、最佳解代码与最佳节点
    state, solution, best_node = solver(task, state)

    # 若搜索未产生任何有效解
    if solution is None:
        # 错误日志：无有效解生成
        log.error("No valid solution was generated")
    else:
        # 日志：评估最终解
        log.info("Evaluating the final solution...")

        # 若最佳节点已带有评估得分信息（metric.info 中含 "score"），直接复用
        if (
            hasattr(best_node, "metric")
            and hasattr(best_node.metric, "info")
            and "score" in best_node.metric.info
            and best_node.metric.info is not None
        ):
            # 日志：评估得分已计算好
            log.info("We have the evaluation score already computed...")
            # 从最佳节点的指标信息中取出得分作为最终适应度(Fitness)
            fitness = best_node.metric.info["score"]
            # 日志：输出最终适应度
            log.info(f"Final fitness: {fitness}")
        else:
            # 正常流程下必然带有得分，走到此分支说明状态异常
            raise ValueError("This should not be reached and happening.")

        # 以 EVAL 事件记录最终适应度值
        logger.log(fitness, LogEvent.EVAL)

    # 日志：开始清理
    log.info("Clean up...")
    # 关闭任务并释放其资源
    task.close(state)
    # 停止实验日志器（完成事件落盘）
    logger.stop()


# Hydra 入口装饰器：默认读取 configs/default_run.yaml 作为配置基线
@hydra.main(version_base="1.3.2", config_path="configs", config_name="default_run")
def main(_cfg: DictConfig):
    ## Validate and setup config
    # 1) Check structure
    # 按数据类类型注解实例化配置（结构校验）
    cfg_instantiated: RunConfig = hydra.utils.instantiate(_cfg)
    # 2) Resolve interpolations (e.g. experiment ID, time, etc.)
    # 转回 OmegaConf 结构以便解析 ${...} 插值引用
    cfg_dict_config = OmegaConf.structured(cfg_instantiated)
    # 解析所有插值引用（如实验 ID、时间戳等）
    OmegaConf.resolve(cfg_dict_config)
    # 3) Convert back to dataclass and validate
    # 转回数据类对象，得到最终可用的强类型配置
    cfg: RunConfig = OmegaConf.to_object(cfg_dict_config)

    # 执行配置校验，非法配置直接报错
    cfg.validate()

    # Pretty print the config
    # 若配置要求打印配置
    if cfg.logger.print_config:
        # 以树状形式美化打印完整配置
        rich_utils.print_config_tree(cfg_dict_config)

    # 进入主运行流程
    _main(cfg)


if __name__ == "__main__":
    # 脚本直接运行时执行主入口
    main()
