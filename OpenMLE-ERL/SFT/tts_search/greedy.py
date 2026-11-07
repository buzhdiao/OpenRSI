# 让类型注解延迟求值，从而支持新版联合类型等注解语法
from __future__ import annotations

# 标准库导入：日志、随机采样、（可变）数据类
import logging
import random
from dataclasses import dataclass

# 程序数据库与搜索算法基类、prompt 构建器（贪心搜索依赖它们完成节点选择与提示生成）
from tts_search.program_database import Program, ProgramDatabase, SearchAlgorithm
from tts_search.prompt_builder import build_prompt

# 模块级 logger
logger = logging.getLogger(__name__)


@dataclass
class AlgoConfig:
    # 每个任务允许的初始草稿（draft）数量上限；None 表示不强制限制
    num_drafts: int | None = 5
    # 选择“调试已有失败程序（debug）”分支的概率，体现探索/利用权衡
    debug_prob: float = 0.5
    # 已有程序后仍继续生成新草稿的概率；None 视为 0（贪心改进优先）
    draft_prob: float | None = 0.0


class GreedySearch(SearchAlgorithm):
    def __init__(self, config: AlgoConfig):
        # 保存算法配置
        self._config = config
        # 归一化 draft 概率：None 一律按 0.0 处理
        self._draft_prob = config.draft_prob or 0.0
        # 断言 draft 概率与 debug 概率之和不超过 1，保证模式采样合法
        assert self._draft_prob + config.debug_prob <= 1.0

    def select(
        self,
        database: ProgramDatabase,
        task_name: str,
        description: str,
        data_dir: str,
        max_steps: int = 1,
    ) -> tuple[str | None, Program | None, str, str | None]:
        # The first solution must always be a draft.
        # 统计该任务已有的 draft 模式程序数量
        draft_count = database.count_by_generation_mode(task_name, "draft")
        # 读取 draft 数量上限；None 按 0 处理
        num_drafts = self._config.num_drafts or 0
        # 数据库为空：该任务还没有任何程序，只能从草稿开始
        if database.is_empty(task_name):
            # 进入 draft 模式且无父程序
            mode = "draft"
            parent_program = None
            # 记录 draft 进度日志
            logger.info(
                "[SEARCH] Using DRAFT mode for task %s (draft_count=%d/%d)",
                task_name,
                draft_count,
                num_drafts,
            )
        else:
            # 默认未选择调试（debug）分支
            debug_selected = False
            if self._draft_prob > 0.0:
                # 采样 [0,1) 均匀随机数，用于模式判定
                sample = random.random()
                if sample < self._draft_prob:
                    # 落在 draft 概率区间：继续生成新草稿
                    mode = "draft"
                    parent_program = None
                else:
                    # 否则进入改进（improve）模式
                    mode = "improve"
                    # 落在随后的 debug 概率区间内则标记为调试分支
                    debug_selected = sample < self._draft_prob + self._config.debug_prob
            elif draft_count < num_drafts:
                # draft 概率为 0 但草稿数未达标：继续补草稿
                mode = "draft"
                parent_program = None
                # 记录 draft 进度日志
                logger.info(
                    "[SEARCH] Using DRAFT mode for task %s (draft_count=%d/%d)",
                    task_name,
                    draft_count,
                    num_drafts,
                )
            else:
                # 草稿数已达标：进入改进模式
                mode = "improve"
                # 以 debug_prob 的概率选择调试失败程序，否则改进当前最优
                debug_selected = random.random() < self._config.debug_prob

            if mode == "draft":
                # 草稿模式不依赖父程序
                parent_program = None
            else:
                parent_program = None
                if debug_selected:
                    # 调试分支：随机挑选一个 fitness 为 0（失败/含 bug）的节点作为父程序
                    parent_program = database.get_random_by_fitness(task_name, 0.0)
                    if parent_program is not None:
                        # 记录调试目标节点
                        logger.info(
                            "[SEARCH] Debugging buggy node for task %s (parent_id=%s)",
                            task_name,
                            parent_program.id,
                        )
                if parent_program is None:
                    # 调试目标缺失时回退：以当前最优程序作为改进起点
                    parent_program = database.get_best(task_name)
                    # 记录改进起点及其 fitness（论文中的搜索树最优节点）
                    logger.info(
                        "[SEARCH] Using IMPROVE mode for task %s (parent_id=%s, parent_fitness=%.4f)",
                        task_name,
                        parent_program.id,
                        float(parent_program.fitness),
                    )

        # 依据模式与父程序构建下一轮代码生成的 prompt（draft/improve 模板不同）
        prompt = build_prompt(
            mode=mode,
            description=description,
            virtual_data_dir=data_dir,
            parent_program=parent_program,
            max_steps=max_steps,
        )
        # 返回 prompt、父程序、所选模式及空的参考程序
        return prompt, parent_program, mode, None

    def select_best(self, database: ProgramDatabase, task_name: str) -> Program:
        # 直接返回该任务在程序数据库中的最优程序
        return database.get_best(task_name)
