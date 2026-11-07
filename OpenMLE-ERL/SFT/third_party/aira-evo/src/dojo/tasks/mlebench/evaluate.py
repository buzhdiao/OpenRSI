# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are MIT-licensed
# Copyright (c) 2024 OpenAI
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/openai/mle-bench/blob/main/LICENSE

# 导入 json：把评分报告序列化写出
import json
# 导入 datetime：报告创建时间戳
from datetime import datetime
# 导入 Path：路径类型注解与操作
from pathlib import Path

# 导入 pandas：读取竞赛排行榜 CSV
import pandas as pd
# 导入 mlebench 数据工具：获取排行榜、检查数据集是否已准备
from mlebench.data import get_leaderboard, is_dataset_prepared
# 导入竞赛评分报告数据结构 CompetitionReport
from mlebench.grade_helpers import CompetitionReport
# 导入竞赛注册表：按竞赛 ID 查找竞赛及其数据目录
from mlebench.registry import Competition, Registry, registry
# 导入 mlebench 通用工具：日志、时间戳、答案加载、CSV/JSONL 读取与终端着色
from mlebench.utils import (
    get_logger,
    get_timestamp,
    load_answers,
    purple,
    read_csv,
    read_jsonl,
)

# 导入 logging 标准日志模块（保持与上游 mle-bench 一致的引入位置）
import logging

# 创建模块级 logger
logger = logging.getLogger(__name__)


# 判断该竞赛是否"分数越低越好"（如 RMSE 类误差指标）
def is_lower_better(competition):
    # 读取竞赛排行榜 CSV 为 DataFrame
    leaderboard_df = pd.read_csv(competition.leaderboard)

    # 通过访问 score 列验证 Git LFS 大文件已正确拉取
    try:
        _ = leaderboard_df["score"]
    except:
        # score 列缺失说明 Git LFS 未拉取，提示用户先执行
        raise Exception("You must run GIT LFS for mlebench.")

    # 交给竞赛评分器判定优化方向
    return competition.grader.is_lower_better(leaderboard_df)


# 评估单个提交：计算得分、奖牌阈值与名次，并把评分报告落盘为 JSON
def evaluate_submission(submission_path: Path, data_dir: Path, competition_id: Path, results_output_dir: Path):
    # Load competition data
    # 复制注册表并指定数据目录
    new_registry = registry.set_data_dir(data_dir)
    # 按竞赛 ID 取出对应的竞赛对象
    competition = new_registry.get_competition(competition_id)

    # 检查评分所需数据（答案/排行榜）是否已准备就绪
    if not is_dataset_prepared(competition, grading_only=True):
        # 未准备则报错，提示先执行 mlebench prepare 命令
        raise ValueError(
            f"Dataset for competition `{competition.id}` is not prepared! "
            f"Please run `mlebench prepare -c {competition.id}` to prepare the dataset."
        )

    # 初始化得分为空（提交无效时保持 None）
    score = None
    # 判断提交文件是否存在且扩展名为 .csv（不区分大小写）
    submission_exists = submission_path.is_file() and submission_path.suffix.lower() == ".csv"

    # 只有有效提交才进行评分
    if submission_exists:
        # 读取提交文件内容为 DataFrame
        submission_df = read_csv(submission_path)
        # 加载竞赛金标答案(Ground Truth)
        answers = load_answers(competition.answers)
        # 调用竞赛评分器计算提交得分
        score = competition.grader(submission_df, answers)
    else:
        # 提交缺失或非 CSV：告警并保持得分为 None
        logger.warning(
            f"Invalid submission file: {submission_path}. Please check that the file exists and it is a CSV."
        )

    # 得分非空即视为有效提交
    valid_submission = score is not None
    # 获取竞赛排行榜（用于计算名次与奖牌阈值）
    competition_leaderboard = get_leaderboard(competition)
    # 计算该得分在排行榜中的名次信息（金/银/铜牌阈值、中位数等）
    rank_info = competition.grader.rank_score(score, competition_leaderboard)
    # 判定该竞赛的优化方向（是否分数越低越好）
    is_lower_better = competition.grader.is_lower_better(competition_leaderboard)

    # 组装评分报告：得分、各级奖牌阈值、是否获奖、提交有效性、时间与路径等
    report = CompetitionReport(
        competition_id=competition.id,
        score=score,
        gold_threshold=rank_info["gold_threshold"],
        silver_threshold=rank_info["silver_threshold"],
        bronze_threshold=rank_info["bronze_threshold"],
        median_threshold=rank_info["median_threshold"],
        any_medal=rank_info["gold_medal"] or rank_info["silver_medal"] or rank_info["bronze_medal"],
        gold_medal=rank_info["gold_medal"],
        silver_medal=rank_info["silver_medal"],
        bronze_medal=rank_info["bronze_medal"],
        above_median=rank_info["above_median"],
        submission_exists=submission_exists,
        valid_submission=valid_submission,
        is_lower_better=is_lower_better,
        created_at=datetime.now(),
        submission_path=str(submission_path),
    )

    # Ensure the output directory exists and write the report to disk.
    # 确保结果输出目录存在
    results_output_dir.mkdir(exist_ok=True)
    # 评分报告的落盘路径
    save_path = results_output_dir / f"grading_report.json"
    # 以 JSON（缩进 2 空格）格式写入报告
    with open(save_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    # 从报告取回最终得分
    score = report.score
    # 统一转为浮点数（无效提交保持 None）
    score = float(score) if score is not None else None
    # 返回得分与报告字典
    return score, report.to_dict()
