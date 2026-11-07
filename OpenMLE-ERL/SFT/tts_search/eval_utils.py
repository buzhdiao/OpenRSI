# 让类型注解延迟求值，从而支持新版联合类型等注解语法
from __future__ import annotations

# 标准库导入：JSON 读写、数学函数（开方用于标准差）
import json
import math
from pathlib import Path
from typing import Any

# pandas 用于读取 Kaggle 公开排行榜 CSV
import pandas as pd

# 低难度任务清单（Kaggle 竞赛 slug）：基线模型容易取得奖牌的任务
LOW_TASK_LIST: list[str] = [
    "aerial-cactus-identification",
    "aptos2019-blindness-detection",
    "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "histopathologic-cancer-detection",
    "jigsaw-toxic-comment-classification-challenge",
    "leaf-classification",
    "mlsp-2013-birds",
    "new-york-city-taxi-fare-prediction",
    "nomad2018-predict-transparent-conductors",
    "plant-pathology-2020-fgvc7",
    "random-acts-of-pizza",
    "ranzcr-clip-catheter-line-classification",
    "siim-isic-melanoma-classification",
    "spooky-author-identification",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "the-icml-2013-whale-challenge-right-whale-redux",
]
# 中难度任务清单：需要一定迭代优化才能拿到奖牌
MIDDLE_TASK_LIST: list[str] = [
    "AI4Code",
    "alaska2-image-steganalysis",
    "billion-word-imputation",
    "cassava-leaf-disease-classification",
    "cdiscount-image-classification-challenge",
    "chaii-hindi-and-tamil-question-answering",
    "champs-scalar-coupling",
    "facebook-recruiting-iii-keyword-extraction",
    "freesound-audio-tagging-2019",
    "google-quest-challenge",
    "h-and-m-personalized-fashion-recommendations",
    "herbarium-2020-fgvc7",
    "herbarium-2021-fgvc8",
    "herbarium-2022-fgvc9",
    "hotel-id-2021-fgvc8",
    "hubmap-kidney-segmentation",
    "icecube-neutrinos-in-deep-ice",
    "imet-2020-fgvc7",
    "inaturalist-2019-fgvc6",
    "iwildcam-2020-fgvc7",
    "jigsaw-unintended-bias-in-toxicity-classification",
    "kuzushiji-recognition",
    "learning-agency-lab-automated-essay-scoring-2",
    "lmsys-chatbot-arena",
    "multi-modal-gesture-recognition",
    "osic-pulmonary-fibrosis-progression",
    "petfinder-pawpularity-score",
    "plant-pathology-2021-fgvc8",
    "seti-breakthrough-listen",
    "statoil-iceberg-classifier-challenge",
    "tensorflow-speech-recognition-challenge",
    "tensorflow2-question-answering",
    "tgs-salt-identification-challenge",
    "tweet-sentiment-extraction",
    "us-patent-phrase-to-phrase-matching",
    "uw-madison-gi-tract-image-segmentation",
    "ventilator-pressure-prediction",
    "whale-categorization-playground",
]
# 高难度任务清单：多模态/大规模数据等强基线任务，最难取得奖牌
HIGH_TASK_LIST: list[str] = [
    "3d-object-detection-for-autonomous-vehicles",
    "bms-molecular-translation",
    "google-research-identify-contrails-reduce-global-warming",
    "hms-harmful-brain-activity-classification",
    "iwildcam-2019-fgvc6",
    "nfl-player-contact-detection",
    "predict-volcanic-eruptions-ingv-oe",
    "rsna-2022-cervical-spine-fracture-detection",
    "rsna-breast-cancer-detection",
    "rsna-miccai-brain-tumor-radiogenomic-classification",
    "siim-covid19-detection",
    "smartphone-decimeter-2022",
    "stanford-covid-vaccine",
    "vesuvius-challenge-ink-detection",
    "vinbigdata-chest-xray-abnormalities-detection",
]
# 全量任务清单 = 低/中/高三档难度任务拼接
ALL_TASK_LIST: list[str] = LOW_TASK_LIST + MIDDLE_TASK_LIST + HIGH_TASK_LIST
# 时间缩放（time-scaling）评测的统一计分时限：12 小时（换算成秒）
TIME_SCALING_SCORE_SECONDS = 12 * 60 * 60


def load_leaderboard(metadata: dict[str, Any]) -> pd.DataFrame | None:
    # 读取可选的排行榜目录配置；未配置时回退到任务数据目录下的 info 子目录
    leaderboard_dir = metadata.get("leaderboard_dir")
    # 取任务名，用于拼出候选排行榜文件路径
    task_name = str(metadata["task_name"])
    # 按优先级列出候选排行榜路径：未指定目录时用数据目录默认位置；指定时依次尝试三种命名布局
    leaderboard_paths = (
        [Path(metadata["data_dir"]) / "info" / "public_leaderboard.csv"]
        if leaderboard_dir is None
        else [
            Path(leaderboard_dir) / f"{task_name}.csv",
            Path(leaderboard_dir) / task_name / "info" / "public_leaderboard.csv",
            Path(leaderboard_dir) / task_name / "public_leaderboard.csv",
        ]
    )
    # 逐个候选路径尝试读取
    for leaderboard_path in leaderboard_paths:
        if leaderboard_path.exists():
            # 文件存在则读入 DataFrame
            leaderboard = pd.read_csv(leaderboard_path)
            # 构造列名归一化映射：去空格并转小写，避免大小写差异导致找不到 score 列
            column_map = {
                column: str(column).strip().lower()
                for column in leaderboard.columns
            }
            # 应用列名重命名
            leaderboard = leaderboard.rename(columns=column_map)
            # 没有 score 列说明该排行榜不可用，返回 None
            if "score" not in leaderboard.columns:
                return None
            # 返回可用的排行榜
            return leaderboard
    # 所有候选路径都不存在时返回 None
    return None


def is_lower_better(leaderboard: pd.DataFrame) -> bool:
    # 取全部排行榜分数
    scores = leaderboard["score"].tolist()
    # 排行榜按名次排列：首行分数低于末行说明该指标越小越好（如 RMSE）
    return bool(scores[0] < scores[-1])


def get_grade_for_score(score: float, leaderboard: pd.DataFrame) -> float:
    # 取全部排行榜分数
    scores = leaderboard["score"].tolist()
    # 先判定指标方向
    lower_better = is_lower_better(leaderboard)
    if lower_better:
        # 越小越好：统计比当前分数更优（更小）的队伍数
        better_count = sum(1 for s in scores if s < score)
    else:
        # 越大越好：统计比当前分数更大（更优）的队伍数
        better_count = sum(1 for s in scores if s > score)
    # 排名 = 更优队伍数 + 1，并夹在排行榜总队伍数以内
    rank = min(better_count + 1, len(scores))
    # 返回排名百分位 grade = rank / N（论文中的 leaderboard 百分位指标，越接近 0 越好）
    return rank / len(scores)


def _medal_positions(num_teams: int) -> tuple[int, int, int]:
    # 依据参赛队伍数计算金/银/铜奖牌的名次分界（对齐 Kaggle 奖牌规则）
    if num_teams < 100:
        # 小规模比赛：按比例取前 10%/20%/40%，但至少保留 1 个名额
        gold_pos = max(1, int(num_teams * 0.1))
        silver_pos = max(1, int(num_teams * 0.2))
        bronze_pos = max(1, int(num_teams * 0.4))
    elif num_teams < 250:
        # 中小规模：金牌固定前 10 名，银牌 20%、铜牌 40%
        gold_pos = 10
        silver_pos = max(1, int(num_teams * 0.2))
        bronze_pos = max(1, int(num_teams * 0.4))
    elif num_teams < 1000:
        # 中大规模：金牌 10 + 0.2%，银牌前 50、铜牌前 100
        gold_pos = 10 + int(num_teams * 0.002)
        silver_pos = 50
        bronze_pos = 100
    else:
        # 大规模比赛（Kaggle 规则）：金牌 10 + 0.2%，银牌前 5%，铜牌前 10%
        gold_pos = 10 + int(num_teams * 0.002)
        silver_pos = max(1, int(num_teams * 0.05))
        bronze_pos = max(1, int(num_teams * 0.1))
    # 返回三档奖牌的名次阈值（前多少名）
    return gold_pos, silver_pos, bronze_pos


def get_medal_for_score(score: float, leaderboard: pd.DataFrame) -> str:
    # 取全部排行榜分数
    scores = leaderboard["score"].tolist()
    # 计算三档奖牌对应的名次位置
    gold_pos, silver_pos, bronze_pos = _medal_positions(len(scores))
    # 取各档名次处的分数作为奖牌阈值（排行榜已按名次排序）
    gold_threshold = scores[gold_pos - 1]
    silver_threshold = scores[silver_pos - 1]
    bronze_threshold = scores[bronze_pos - 1]
    # 判定指标方向
    lower_better = is_lower_better(leaderboard)
    if lower_better:
        # 越小越好：分数不劣于（<=）各档阈值即授予对应奖牌
        if score <= gold_threshold:
            return "gold"
        if score <= silver_threshold:
            return "silver"
        if score <= bronze_threshold:
            return "bronze"
    else:
        # 越大越好：分数达到（>=）各档阈值即授予对应奖牌
        if score >= gold_threshold:
            return "gold"
        if score >= silver_threshold:
            return "silver"
        if score >= bronze_threshold:
            return "bronze"
    # 未进入任何奖牌区间
    return "N/A"


def get_medal_for_grade(grade: float, leaderboard: pd.DataFrame) -> str:
    # 参赛队伍数
    num_teams = len(leaderboard["score"].tolist())
    # 计算三档奖牌的名次位置
    gold_pos, silver_pos, bronze_pos = _medal_positions(num_teams)
    # 把名次位置换算成百分位 grade 阈值（名次/总队伍数）
    gold_grade = gold_pos / num_teams
    silver_grade = silver_pos / num_teams
    bronze_grade = bronze_pos / num_teams
    # grade 越小名次越靠前：依次与三档阈值比较给出奖牌
    if grade <= gold_grade:
        return "gold"
    if grade <= silver_grade:
        return "silver"
    if grade <= bronze_grade:
        return "bronze"
    # 未达奖牌线
    return "N/A"


def build_submit_grade_and_medal(
    submit_score: float | None, leaderboard: pd.DataFrame | None
) -> tuple[float, str]:
    # 缺分数或缺排行榜时按最差成绩处理：grade=1.0、无奖牌
    if submit_score is None or leaderboard is None:
        return 1.0, "N/A"
    # 计算提交分数对应的排名百分位
    grade = get_grade_for_score(submit_score, leaderboard)
    # 计算提交分数对应的奖牌档位
    medal = get_medal_for_score(submit_score, leaderboard)
    return grade, medal


def write_epoch_stat(epoch_output_dir: Path) -> dict[str, Any]:
    # 初始化 epoch 级用量汇总容器：总费用、prompt/completion/总 token 数、总步数
    usage = {
        "total_cost": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "total_steps": 0,
    }

    def mean(values: list[float]) -> float:
        # 局部工具函数：算术平均
        return sum(values) / len(values)

    def std(values: list[float]) -> float:
        # 样本不足 2 个时标准差按 0 处理
        if len(values) < 2:
            return 0.0
        # 先算均值
        avg = mean(values)
        # 总体方差：各点偏离均值的平方取平均
        variance = sum((value - avg) ** 2 for value in values) / len(values)
        # 标准差 = 方差开方
        return math.sqrt(variance)

    # 收集本 epoch 各任务名与对应奖牌、grade 结果
    task_names: list[str] = []
    task_medals: dict[str, str] = {}
    task_grades: dict[str, float] = {}

    # 遍历 epoch 输出目录下按名称排序的各任务子目录
    for task_dir in sorted(
        path for path in epoch_output_dir.iterdir() if path.is_dir()
    ):
        # 每个任务目录的统计文件路径
        stat_path = task_dir / "stat.json"
        # 缺统计文件的任务直接跳过
        if not stat_path.exists():
            continue
        # 读取该任务的统计 JSON
        payload = json.loads(stat_path.read_text(encoding="utf-8"))
        # 任务名缺失时用目录名兜底
        task_name = str(payload.get("task_name") or task_dir.name)
        task_names.append(task_name)
        # 累加本任务的 API 费用
        usage["total_cost"] += float(payload.get("total_cost", 0.0))
        # 累加 prompt 侧 token 数
        usage["total_prompt_tokens"] += int(payload.get("total_prompt_tokens", 0))
        # 累加 completion 侧 token 数
        usage["total_completion_tokens"] += int(
            payload.get("total_completion_tokens", 0)
        )
        # 累加总 token 数
        usage["total_tokens"] += int(payload.get("total_tokens", 0))
        # 累加执行步数
        usage["total_steps"] += int(payload.get("num_steps", 0))

        # 读取提交结果的奖牌与 grade
        submit_medal = payload.get("submit_medal")
        # 无奖牌时统一记为 N/A
        task_medals[task_name] = str(submit_medal) if submit_medal else "N/A"
        submit_grade = payload.get("submit_grade")
        # grade 存在才写入（缺排行榜的任务没有 grade）
        if submit_grade is not None:
            task_grades[task_name] = float(submit_grade)

    # 全量任务清单为空时退化为三档直接拼接
    all_task_list = ALL_TASK_LIST or LOW_TASK_LIST + MIDDLE_TASK_LIST + HIGH_TASK_LIST
    # 把已运行任务按难度档位（low/middle/high/all）分组
    level_tasks = {
        "low": [task for task in task_names if task in LOW_TASK_LIST],
        "middle": [task for task in task_names if task in MIDDLE_TASK_LIST],
        "high": [task for task in task_names if task in HIGH_TASK_LIST],
        "all": [task for task in task_names if task in all_task_list]
        if all_task_list
        else list(task_names),
    }
    # 统计每个难度档的任务数
    task_counts = {
        f"{level}_task_count": len(tasks) for level, tasks in level_tasks.items()
    }

    # 按难度档计算 grade 的均值/标准差（@n 表示对该 epoch 的 n 个任务取统计）
    grade_stats: dict[str, float | None] = {}
    for level, tasks in level_tasks.items():
        # 只收集有 grade 的任务
        grades = [task_grades[task] for task in tasks if task in task_grades]
        # 均值为空列表时置 None
        grade_stats[f"{level}_grade_avg@n"] = mean(grades) if grades else None
        # 标准差同理
        grade_stats[f"{level}_grade_std@n"] = std(grades) if grades else None

    # 按难度档统计金/银/铜牌任务数及任一奖牌任务数
    medal_counts: dict[str, int] = {}
    for level, tasks in level_tasks.items():
        # 逐档枚举三种奖牌
        for medal in ("gold", "silver", "bronze"):
            # 统计拿到该奖牌的任务数
            medal_counts[f"{level}_{medal}_count"] = sum(
                1 for task in tasks if task_medals.get(task) == medal
            )
        # 统计至少拿到一枚奖牌的任务数
        medal_counts[f"{level}_any_count"] = sum(
            1 for task in tasks if task_medals.get(task) in {"gold", "silver", "bronze"}
        )

    # 按难度档把奖牌数换算成奖牌率（无任务时置 None 避免除零）
    medal_rates: dict[str, float | None] = {}
    for level, tasks in level_tasks.items():
        # 该档任务总数
        total = len(tasks)
        # 覆盖金/银/铜及任一奖牌四类比率
        for medal in ("gold", "silver", "bronze", "any"):
            medal_rates[f"{level}_{medal}_rate"] = (
                medal_counts[f"{level}_{medal}_count"] / total if total else None
            )

    # 合并用量、任务数、grade 统计与奖牌统计为 epoch 汇总
    epoch_stat = {
        **usage,
        **task_counts,
        **grade_stats,
        **medal_counts,
        **medal_rates,
    }
    # 把 epoch 统计写入该 epoch 目录下的 stat.json
    (epoch_output_dir / "stat.json").write_text(
        json.dumps(epoch_stat, indent=2), encoding="utf-8"
    )
    # 返回统计内容供调用方继续使用
    return epoch_stat


def write_summary_csv(
    output_dir: Path,
    task_metadata_map: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    def epoch_index(path: Path) -> int:
        # 从目录名 program_ep_<N> 中解析出 epoch 序号，作为排序键
        suffix = path.name.split("program_ep_", 1)[-1]
        # 非数字后缀统一排最前（序号 0）
        return int(suffix) if suffix.isdigit() else 0

    def mean(values: list[float]) -> float:
        # 局部工具函数：算术平均
        return sum(values) / len(values)

    def std(values: list[float]) -> float:
        # 样本不足 2 个时标准差按 0 处理
        if len(values) < 2:
            return 0.0
        # 先算均值
        avg = mean(values)
        # 总体方差：偏差平方取平均
        variance = sum((value - avg) ** 2 for value in values) / len(values)
        # 标准差 = 方差开方
        return math.sqrt(variance)

    # 每个任务跨所有 epoch 的统计样本（用于 @k 汇总）
    task_samples: dict[str, list[dict[str, Any]]] = {}
    # 每个任务跨 epoch 累计的执行状态计数
    task_status_counts: dict[str, dict[str, int]] = {}
    # 固定列的状态枚举，保证 CSV 列稳定
    fixed_statuses = [
        "code_execution_error",
        "code_missing",
        "scoring_failed",
        "submission_missing",
        "success",
        "timeout",
        "unknown",
    ]

    # 收集 output_dir 下所有 program_ep_* 目录并按 epoch 序号排序
    epoch_dirs = sorted(
        [
            path
            for path in output_dir.iterdir()
            if path.is_dir() and path.name.startswith("program_ep_")
        ],
        key=epoch_index,
    )
    # 双层遍历：epoch 目录 -> 任务目录
    for epoch_dir in epoch_dirs:
        for task_dir in sorted(path for path in epoch_dir.iterdir() if path.is_dir()):
            # 该任务在本 epoch 的统计文件
            stat_path = task_dir / "stat.json"
            # 缺文件则跳过
            if not stat_path.exists():
                continue
            # 读取统计 JSON
            payload = json.loads(stat_path.read_text(encoding="utf-8"))
            # 任务名缺失时用目录名兜底
            task_name = str(payload.get("task_name") or task_dir.name)
            # 把该样本挂到任务名下，形成跨 epoch 的样本序列
            task_samples.setdefault(task_name, []).append(payload)
            # 累计各状态发生次数
            status_count = payload.get("status_count") or {}
            # 取出（或初始化）该任务的状态计数字典
            task_status = task_status_counts.setdefault(task_name, {})
            # 把本 epoch 的状态计数累加到全局
            for status, count in dict(status_count).items():
                task_status[status] = task_status.get(status, 0) + int(count)

    # 使用固定状态枚举作为输出列
    all_statuses = fixed_statuses
    # 每任务一行的汇总结果
    summary_rows: list[dict[str, Any]] = []
    # 按任务名排序逐任务汇总
    for task_name in sorted(task_samples):
        # 该任务的所有 epoch 样本
        samples = task_samples[task_name]
        # 查任务元数据（含排行榜目录等）
        metadata = task_metadata_map.get(task_name)
        # 有元数据才尝试加载公开排行榜
        leaderboard = load_leaderboard(metadata) if metadata else None
        # 收集所有非空的提交分数（float 化）
        valid_scores = [
            float(score)
            for score in (sample.get("submit_score") for sample in samples)
            if score is not None
        ]
        # 收集每个 epoch 样本的 API 费用
        costs = [float(sample.get("total_cost", 0.0)) for sample in samples]

        if valid_scores:
            # 分数均值
            score_avg = mean(valid_scores)
            # 分数标准差
            score_std = std(valid_scores)
            # 最优分数按指标方向取：越小越好取最小值，否则取最大值
            score_best = (
                min(valid_scores)
                if leaderboard is not None and is_lower_better(leaderboard)
                else max(valid_scores)
            )
        else:
            # 无有效分数时三项均置空
            score_avg = None
            score_std = None
            score_best = None

        if leaderboard is None:
            # 无排行榜：grade/奖牌列固定为 N/A
            score_grade_avg = "N/A"
            score_medal_avg = "N/A"
        elif score_avg is not None:
            # 有均值分数：计算其排名百分位与对应奖牌
            score_grade_avg = get_grade_for_score(score_avg, leaderboard)
            score_medal_avg = get_medal_for_grade(score_grade_avg, leaderboard)
        else:
            # 有排行榜但无分数：grade 置空、奖牌记 N/A
            score_grade_avg = None
            score_medal_avg = "N/A"

        # 收集所有非空的提交 grade（排名百分位）
        submit_grades = [
            float(grade)
            for grade in (sample.get("submit_grade") for sample in samples)
            if grade is not None
        ]
        if submit_grades:
            # grade 的均值/标准差/最优（grade 越小越好，取最小值）
            grade_avg = mean(submit_grades)
            grade_std = std(submit_grades)
            grade_best = min(submit_grades)
        else:
            # 无 grade 时三项均置空
            grade_avg = None
            grade_std = None
            grade_best = None

        if leaderboard is None or grade_avg is None or grade_best is None:
            # 缺排行榜或缺 grade：平均/最优奖牌均记 N/A
            medal_avg = "N/A"
            medal_best = "N/A"
        else:
            # 由 grade 换算平均与最优奖牌
            medal_avg = get_medal_for_grade(grade_avg, leaderboard)
            medal_best = get_medal_for_grade(grade_best, leaderboard)

        # 组装该任务的汇总行（@k 表示跨 k 个 epoch 的统计）
        row = {
            "Task": task_name,
            "score_avg@k": score_avg,
            "score_std@k": score_std,
            "score_best@k": score_best,
            "score_avg@k_grade": score_grade_avg,
            "score_avg@k_medal": score_medal_avg,
            "grade_avg@k": grade_avg,
            "grade_std@k": grade_std,
            "grade_best@k": grade_best,
            "medal_avg@k": medal_avg,
            "medal_best@k": medal_best,
            "cost_avg@k": mean(costs),
            "cost_best@k": min(costs),
            "cost_sum@k": sum(costs),
        }
        # 取该任务累计的状态计数（无记录则为空字典）
        status_count = task_status_counts.get(task_name, {})
        # 以固定状态枚举初始化各状态计数为 0
        status_rollup = {status: 0 for status in all_statuses}
        # 把累计状态归并到固定列中
        for status, count in dict(status_count).items():
            if status in status_rollup:
                # 已知状态直接累加
                status_rollup[status] += int(count)
            else:
                # Fold unexpected statuses into unknown for stable columns.
                # 未知状态折入 unknown 列，保持列结构稳定
                status_rollup["unknown"] += int(count)
        # 该任务的总执行次数（各状态之和）
        total_status = sum(status_rollup.values())
        # 把各状态计数写入行
        for status in all_statuses:
            row[status] = status_rollup[status]
        # 计算成功率：success 占比（总次数为 0 时记 0.0 防止除零）
        row["success_rate"] = (
            status_rollup["success"] / total_status if total_status else 0.0
        )
        # 收入汇总结果
        summary_rows.append(row)

    # 固定输出列顺序，保证 CSV 结构稳定
    summary_columns = [
        "Task",
        "score_avg@k",
        "score_std@k",
        "score_best@k",
        "score_avg@k_grade",
        "score_avg@k_medal",
        "grade_avg@k",
        "grade_std@k",
        "grade_best@k",
        "medal_avg@k",
        "medal_best@k",
        "cost_avg@k",
        "cost_best@k",
        "cost_sum@k",
        *all_statuses,
        "success_rate",
    ]
    # 构造带固定列序的汇总 DataFrame
    summary_df = pd.DataFrame(summary_rows, columns=summary_columns)
    # 写出 summary.csv（不写行索引）
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    # 返回汇总表供调用方继续使用
    return summary_df


def write_global_stat(
    output_dir: Path,
) -> dict[str, Any]:
    def epoch_index(path: Path) -> int:
        # 从目录名 program_ep_<N> 解析 epoch 序号作排序键
        suffix = path.name.split("program_ep_", 1)[-1]
        # 非数字后缀统一排最前（序号 0）
        return int(suffix) if suffix.isdigit() else 0

    def mean(values: list[float]) -> float:
        # 局部工具函数：算术平均
        return sum(values) / len(values)

    def std(values: list[float]) -> float:
        # 样本不足 2 个时标准差按 0 处理
        if len(values) < 2:
            return 0.0
        # 先算均值
        avg = mean(values)
        # 总体方差：偏差平方取平均
        variance = sum((value - avg) ** 2 for value in values) / len(values)
        # 标准差 = 方差开方
        return math.sqrt(variance)

    def aggregate(values: list[float | None]) -> tuple[float | None, float | None]:
        # 过滤掉 None 后转成数值序列
        numeric_values = [float(value) for value in values if value is not None]
        # 全为空时均值/标准差均置 None
        if not numeric_values:
            return None, None
        # 返回该指标跨 epoch 的均值与标准差
        return mean(numeric_values), std(numeric_values)

    # 收集所有 program_ep_* 目录并按 epoch 序号排序
    epoch_dirs = sorted(
        [
            path
            for path in output_dir.iterdir()
            if path.is_dir() and path.name.startswith("program_ep_")
        ],
        key=epoch_index,
    )
    # 属于“用量直接累加”的键集合
    usage_keys = {
        "total_cost",
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_tokens",
        "total_steps",
    }
    # 全局用量累加容器
    usage = {
        "total_cost": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "total_steps": 0,
    }
    # 各指标跨 epoch 的取值序列（用于计算 avg/std）
    metric_values: dict[str, list[float | None]] = {}
    # 各难度档 grade 均值键集合
    grade_avg_keys = {
        "low_grade_avg@n",
        "middle_grade_avg@n",
        "high_grade_avg@n",
        "all_grade_avg@n",
    }
    # 各难度档 grade 标准差键集合（仅用于识别并跳过）
    grade_std_keys = {
        "low_grade_std@n",
        "middle_grade_std@n",
        "high_grade_std@n",
        "all_grade_std@n",
    }
    # 各难度档 grade 均值的跨 epoch 序列
    grade_avg_values: dict[str, list[float | None]] = {
        "low": [],
        "middle": [],
        "high": [],
        "all": [],
    }
    # 逐 epoch 汇总其 stat.json
    for epoch_dir in epoch_dirs:
        # 该 epoch 的统计文件路径
        epoch_stat_path = epoch_dir / "stat.json"
        # 读入 JSON
        epoch_stat = json.loads(epoch_stat_path.read_text(encoding="utf-8"))
        # 累加费用
        usage["total_cost"] += float(epoch_stat["total_cost"])
        # 累加 prompt token
        usage["total_prompt_tokens"] += int(epoch_stat["total_prompt_tokens"])
        # 累加 completion token
        usage["total_completion_tokens"] += int(epoch_stat["total_completion_tokens"])
        # 累加总 token
        usage["total_tokens"] += int(epoch_stat["total_tokens"])
        # 累加步数
        usage["total_steps"] += int(epoch_stat["total_steps"])
        # Aggregate per-epoch task counts and medal metrics for avg/std.
        # 遍历该 epoch 的其余统计键，按类别分派到对应容器
        for key, value in epoch_stat.items():
            # 用量键已在上面累加过，跳过
            if key in usage_keys:
                continue
            if key in grade_avg_keys:
                # grade 均值键：解析出难度档名并记录该 epoch 的取值
                level = key.split("_", 1)[0]
                grade_avg_values[level].append(value)
                continue
            if key in grade_std_keys:
                # grade 标准差不在全局再统计，跳过
                continue
            # 其余指标统一收集，待会统一算 avg/std
            metric_values.setdefault(key, []).append(value)

    # 对每个指标键计算跨 epoch 的均值/标准差
    metric_stats: dict[str, Any] = {}
    for key, values in sorted(metric_values.items()):
        # 聚合出均值与标准差
        avg, std_value = aggregate(values)
        # 以 <key>_avg / <key>_std 命名写入
        metric_stats[f"{key}_avg"] = avg
        metric_stats[f"{key}_std"] = std_value

    # 合并全局统计：用量 + epoch 数 + 各指标 avg/std
    global_stat = {
        **usage,
        "num_epochs": len(epoch_dirs),
        **metric_stats,
    }
    # 对各难度档 grade 序列再聚合一次，得到 @k（跨 epoch）均值与标准差
    for level, values in grade_avg_values.items():
        # 聚合该档 grade
        avg, std_value = aggregate(values)
        # 写入全局统计
        global_stat[f"{level}_grade_avg@k"] = avg
        global_stat[f"{level}_grade_std@k"] = std_value
    # 写出全局 stat.json
    (output_dir / "stat.json").write_text(
        json.dumps(global_stat, indent=2), encoding="utf-8"
    )
    # 返回全局统计
    return global_stat
