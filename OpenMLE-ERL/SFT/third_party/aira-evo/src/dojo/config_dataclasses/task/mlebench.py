# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 dataclass 装饰器与 field 工厂
from dataclasses import dataclass, field

# OmegaConf 配置库：SI 为字符串插值，MISSING 为必填占位符
from omegaconf import SI, MISSING

# 任务配置基类
from dojo.config_dataclasses.task.base import TaskConfig
# 获取 MLE-Bench 数据目录的工具函数
from dojo.utils.environment import get_mlebench_data_dir

# MLE-Bench 任务各算子环节使用的提示词（prompt）文案表
mlebench_operator_prompts: dict[str, str] = {
    # 首次起草方案（draft）阶段引导语：以 Kaggle 大师身份设计并实现一个方案
    "draft_intro": """You are a Kaggle grandmaster attending a competition.
        In order to win this competition, you need to come up with an excellent and creative plan
        for a solution and then implement this solution in Python. We will now provide a description of the task.""",
    # 改进方案（improve）阶段引导语：基于已有方案提出改进计划并实现
    "improve_intro": """You are a Kaggle grandmaster attending a competition. You are provided with a previously developed
        solution below and should improve it in order to further increase the (test time) performance.
        For this you should first outline a brief plan in natural language for how the solution can be improved and
        then implement this improvement in Python based on the provided previous solution.""",
    # 修复缺陷（debug）阶段引导语：根据报错信息修订上一版方案
    "debug_intro": """You are a Kaggle grandmaster attending a competition.
        Your previous solution had a bug, so based on the information below, you should revise it in order to fix this bug.
        Your response should be an implementation outline in natural language, followed by a single markdown code block which implements the bugfix/solution.""",
    # 结果分析（analyze）阶段引导语：评估代码执行输出、判断是否有 bug 并汇报经验发现（要求返回合法 JSON）
    "analyze_intro": """You are a Kaggle grandmaster attending a competition.
        You have written code to solve this task and now need to evaluate the output of the code execution.
        You should determine if there were any bugs as well as report the empirical findings.
        Return exactly one valid JSON object matching the requested schema, with a non-empty summary field.""",
    # 代码实现（implementation）阶段通用约束：实现方案、打印验证集指标、生成 submission.csv 提交文件
    "implementation_guideline": """The code should **implement the proposed solution**, **print the value of the evaluation metric computed on a hold-out validation set**,
        **AND MOST IMPORTANTLY SAVE PREDICTIONS ON THE PROVIDED UNLABELED TEST DATA IN A `submission.csv` FILE IN THE CURRENT DIRECTORY.**
        The code should be a single-file python program that is self-contained and can be executed as-is.
        No parts of the code should be skipped, don't terminate the before finishing the script.
        Your response should only contain a single code block.
        All the provided input data is stored in "./data" directory.
        **If there is test data provided for this task, please save the test predictions in a `submission.csv` file in the "./" directory as described in the task description** This is extremely important since this file is used for grading/evaluation. DO NOT FORGET THE submission.csv file!
        You can also use the current directory to store any temporary files that your code needs to create.
        REMEMBER THE ./submission.csv FILE!!!!! The correct directory is important too.
        The evaluation should be based on 5-fold cross-validation but only if that's an appropriate evaluation for the task at hand.""",
    # 执行环节引导语：提醒 LLM 注意代码运行时间限制（后接具体秒数）
    "execution_intro": """Be aware of the running time of the code, it should complete within""",
    # 回复格式（response_format）约定：先自然语言简述方案，再紧跟一个 markdown 代码块
    "response_format": """Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences),
        followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric.
        There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block.""",
    # 首版方案草图（solution_sketch）要求：保持简单（不做集成/调参）、给出合理评估指标、避免 EDA
    "solution_sketch": """This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.
        Take the Memory section into consideration when proposing the design,
        don't propose the same modelling solution but keep the evaluation the same.
        The solution sketch should be 3-5 sentences.
        Propose an evaluation metric that is reasonable for this task.
        Don't suggest to do EDA.
        The data is already prepared and available in the `./data` directory. There is no need to unzip any files.""",
    # 改进草图（improvement_sketch）要求：只提出一个原子化、可实验验证的改进点
    "improvement_sketch": """The solution sketch should be a brief natural language description of how the previous solution can be improved.
        You should be very specific and should only propose a single actionable improvement.
        This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.
        Take the Memory section into consideration when proposing the improvement.
        The solution sketch should be 3-5 sentences.
        Don't suggest to do EDA.""",
    # 修复草图（bugfix_sketch）要求：用 3-5 句自然语言描述如何修复上一版实现的问题
    "bugfix_sketch": """You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.
        Don't suggest to do EDA.""",
    # 分析环节工具调用的函数名（function calling 中的 name）
    "analyze_func_name": """submit_review""",
    # 分析环节工具调用的函数描述（function calling 中的 description）
    "analyze_func_desc": """Submit a review evaluating the output of the training script.""",
    # 交叉组合（crossover）阶段引导语：把两份优秀方案的优点融合为一份更强的实现
    "crossover_intro": """You are a Kaggle grandmaster attending a competition.
        You have two previously developed solutions that have both shown strong performance.
        Your task is to combine the best aspects of both solutions into a single, improved implementation.
        First, outline a brief plan in natural language for how to merge the strengths of both approaches while maintaining or improving performance.
        Then, implement this combined solution in Python.""",
    # 交叉组合草图（crossover_sketch）要求：合并互补优点、保持评估指标一致、不引入多余复杂度
    "crossover_sketch": """The solution sketch should be a brief natural language description (3-5 sentences) of how the best aspects of two previous solutions can be combined into a single, more effective implementation.
        Focus on merging complementary strengths while maintaining efficiency and performance.
        Ensure that the proposed approach does not introduce unnecessary complexity.
        The evaluation metric should remain consistent with the previous solutions.
        Do not suggest exploratory data analysis (EDA).""",
}


# MLE-Bench 任务配置：MLE-Bench 是把 Kaggle 竞赛改造为智能体评测基准（benchmark）的测试集，
# 智能体需端到端完成建模并产出符合格式的提交文件
@dataclass
class MLEBenchTaskConfig(TaskConfig):
    # 任务所属基准类型，固定为 mlebench
    benchmark: str = field(
        default="mlebench",
        metadata={
            "help": "Type of the task.",
        },
    )

    # 任务数据存放目录（智能体可见的工作数据）：默认插值为公共数据目录
    data_dir: str = field(
        default=SI("${task.public_dir}"),
        metadata={
            "help": "The directory where the data is stored.",
            "exclude_from_hash": True,
        },
    )
    # 任务数据的缓存根目录（存放下载与预处理后的数据），默认从环境探测
    cache_dir: str = field(
        default=get_mlebench_data_dir(),
        metadata={
            "help": "The directory where the task data is cached.",
            "exclude_from_hash": True,
        },
    )
    # 公共数据目录：智能体可见的训练/测试数据
    public_dir: str = field(
        default=SI("${task.cache_dir}/${task.name}/prepared/public"),
        metadata={
            "help": "The directory where the public data is stored.",
            "exclude_from_hash": True,
        },
    )
    # 私有数据目录：仅供评分使用的真实标签，智能体不可见
    private_dir: str = field(
        default=SI("${task.cache_dir}/${task.name}/prepared/private"),
        metadata={
            "help": "The directory where the private data is stored.",
            "exclude_from_hash": True,
        },
    )
    # 提交文件的文件名，评分时以此为准
    submission_fname: str = field(
        default="submission.csv",
        metadata={
            "help": "The name of the submission file.",
            "exclude_from_hash": True,
        },
    )
    # 任务结果（分数、评测输出）的存放目录
    results_output_dir: str = field(
        default=SI("${logger.output_dir}/results"),
        metadata={
            "help": "The directory where the results are stored.",
            "exclude_from_hash": True,
        },
    )

    # 配置校验入口
    def validate(self) -> None:
        # 复用父类校验逻辑，本类无额外约束
        super().validate()
