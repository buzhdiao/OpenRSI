# 中文说明:本模块是 tts_search 的评估结果汇总工具集:
# 按"任务目录(stat.json)-> epoch 目录 -> 全局输出目录"三级聚合统计,
# 产出 epoch 级 stat.json、跨 epoch 的 summary.csv、全局 stat.json 与
# 时间-分数扩展曲线(time_scaling.csv/png)。
# 评分口径沿用 Kaggle 语义:以 public leaderboard 计算相对名次(grade)与奖牌
# (gold/silver/bronze),任务按难度分为 low/middle/high 三档分别统计;
# NatureBench 基准不读排行榜(显式跳过),其分数体系由任务自身携带。
# 位置:OpenMLE-Evo/tts_search 下,被评估驱动/汇总脚本导入。

# 延迟注解:类型标注运行期不求值
from __future__ import annotations

# json:stat.json 的读写
import json
# logging:模块日志
import logging
# math:sqrt(总体标准差计算)
import math
# Path:输出目录与统计文件路径
from pathlib import Path
# Any 类型注解
from typing import Any

# pandas:排行榜 CSV 读取与汇总表构建/落盘
import pandas as pd

# 模块 logger
logger = logging.getLogger(__name__)

# 低难度任务清单(Kaggle 比赛名):多为经典分类/回归,资源与建模复杂度较低
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
# 中难度任务清单:多模态/时序/大表等中等复杂度任务
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
# 高难度任务清单:检测/分割/长序列/多阶段管线等复杂任务
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
# 全量任务清单:三档合并(便于整体统计)
ALL_TASK_LIST: list[str] = LOW_TASK_LIST + MIDDLE_TASK_LIST + HIGH_TASK_LIST
# 时间-分数扩展的归一化常数:12 小时(计算 sandbox_12h_score 用)
TIME_SCALING_SCORE_SECONDS = 12 * 60 * 60


def _ordered_unique(values: list[str] | tuple[str, ...] | None) -> list[str] | None:
    """去重并保持首次出现顺序;输入 None 原样返回 None(供调用方走回退逻辑)。"""
    # None 原样返回(调用方据此决定回退)
    if values is None:
        return None
    # 已见集合
    seen: set[str] = set()
    # 保序结果列表
    ordered: list[str] = []
    # 逐项去重收集
    for value in values:
        # 统一转字符串
        item = str(value)
        # 重复项跳过
        if item in seen:
            continue
        # 记录首次出现
        seen.add(item)
        ordered.append(item)
    # 返回保序去重列表
    return ordered


def _read_json_file(path: Path) -> dict[str, Any] | None:
    """读取一个 JSON 文件并要求顶层为字典;不可读/格式不对则告警并返回 None。"""
    try:
        # 读入并反序列化
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        # 读取/解析失败:告警后跳过该文件
        logger.warning("Skipping unreadable JSON file %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        # 顶层不是映射:格式不符合约定,跳过
        logger.warning("Skipping non-mapping JSON file %s", path)
        return None
    # 返回解析结果
    return payload


def _level_tasks_for(task_names: list[str]) -> dict[str, list[str]]:
    """把任务名列表按难度分档,返回 {low, middle, high, all} 四组。"""
    return {
        # 落在低难度清单的任务
        "low": [task for task in task_names if task in LOW_TASK_LIST],
        # 落在中难度清单的任务
        "middle": [task for task in task_names if task in MIDDLE_TASK_LIST],
        # 落在高难度清单的任务
        "high": [task for task in task_names if task in HIGH_TASK_LIST],
        # 全部任务
        "all": list(task_names),
    }


def get_total_model_time(payload: dict[str, Any]) -> float:
    """取任务的累计模型(推理)耗时,缺失按 0。"""
    return float(payload.get("total_model_time", 0.0))


def get_total_sandbox_time(payload: dict[str, Any]) -> float:
    """取任务的累计沙箱执行耗时;兼容旧字段 total_time。"""
    # 新字段存在:直接用
    if payload.get("total_sandbox_time") is not None:
        return float(payload["total_sandbox_time"])
    # 旧数据回退到 total_time
    return float(payload.get("total_time", 0.0))


def get_total_model_plus_sandbox_time(payload: dict[str, Any]) -> float:
    """取"模型+沙箱"总耗时;无该字段时按两项相加重建。"""
    # 显式字段优先
    if payload.get("total_model_plus_sandbox_time") is not None:
        return float(payload["total_model_plus_sandbox_time"])
    # 兜底:模型时间 + 沙箱时间
    return get_total_model_time(payload) + get_total_sandbox_time(payload)


def load_leaderboard(metadata: dict[str, Any]) -> pd.DataFrame | None:
    """按任务元数据查找并加载 public leaderboard;NatureBench 基准显式返回 None。

    支持多种目录布局;列名统一转小写;必须含 score 列。
    """
    # NatureBench 不使用 Kaggle 排行榜口径,直接跳过
    if str(metadata.get("benchmark") or "").strip().lower() == "naturebench":
        return None
    # 排行榜目录(未配置则从任务 data_dir 内找)
    leaderboard_dir = metadata.get("leaderboard_dir")
    # 任务名
    task_name = str(metadata["task_name"])
    # 候选路径列表:无专门目录时找任务包内的 info/public_leaderboard.csv
    leaderboard_paths = (
        [Path(metadata["data_dir"]) / "info" / "public_leaderboard.csv"]
        if leaderboard_dir is None
        else [
            # 布局一:<leaderboard_dir>/<task>.csv
            Path(leaderboard_dir) / f"{task_name}.csv",
            # 布局二:<leaderboard_dir>/<task>/info/public_leaderboard.csv
            Path(leaderboard_dir)
            / task_name
            / "info"
            / "public_leaderboard.csv",
            # 布局三:<leaderboard_dir>/<task>/public_leaderboard.csv
            Path(leaderboard_dir) / task_name / "public_leaderboard.csv",
        ]
    )
    # 依次尝试候选路径
    for leaderboard_path in leaderboard_paths:
        # 不存在则试下一个
        if not leaderboard_path.exists():
            continue
        # 读入 CSV
        leaderboard = pd.read_csv(leaderboard_path)
        # 列名统一去空白并转小写(不同来源格式不一)
        leaderboard = leaderboard.rename(
            columns={
                column: str(column).strip().lower()
                for column in leaderboard.columns
            }
        )
        # 必须含 score 列,否则无法计分
        if "score" not in leaderboard.columns:
            logger.warning(
                "Leaderboard %s has no score column",
                leaderboard_path,
            )
            return None
        # 返回加载好的排行榜
        return leaderboard
    # 全部候选都未命中:告警并返回 None
    logger.warning(
        "No leaderboard found for task %s in candidates: %s",
        task_name,
        ", ".join(str(path) for path in leaderboard_paths),
    )
    return None


def is_lower_better(leaderboard: pd.DataFrame) -> bool:
    """根据排行榜首尾分数判断该任务是否"分越低越好"(如 RMSE)。"""
    # 取全部分数
    scores = leaderboard["score"].tolist()
    # 排行榜按名次排序:第一名分数小于最后一名即为 lower_is_better
    return bool(scores[0] < scores[-1])


def get_grade_for_score(score: float, leaderboard: pd.DataFrame) -> float:
    """把绝对分数换算为相对名次 grade = 名次/参赛队数(越小越好)。"""
    # 排行榜分数
    scores = leaderboard["score"].tolist()
    # 方向判断
    lower_better = is_lower_better(leaderboard)
    # 统计"严格好于我"的队伍数
    if lower_better:
        better_count = sum(1 for s in scores if s < score)
    else:
        better_count = sum(1 for s in scores if s > score)
    # 名次 = 更好数 + 1(封顶队伍数,防并列溢出)
    rank = min(better_count + 1, len(scores))
    # 归一化为 grade
    return rank / len(scores)


def _medal_positions(num_teams: int) -> tuple[int, int, int]:
    """按队伍数确定金/银/铜奖牌的名次分界(近似 Kaggle 奖牌规则)。"""
    # 小于 100 队:按比例(10%/20%/40%)取前若干名
    if num_teams < 100:
        gold_pos = max(1, int(num_teams * 0.1))
        silver_pos = max(1, int(num_teams * 0.2))
        bronze_pos = max(1, int(num_teams * 0.4))
    # 100-249 队:金牌固定前 10,银/铜按比例
    elif num_teams < 250:
        gold_pos = 10
        silver_pos = max(1, int(num_teams * 0.2))
        bronze_pos = max(1, int(num_teams * 0.4))
    # 250-999 队:金=10+0.2%,银=50,铜=100
    elif num_teams < 1000:
        gold_pos = 10 + int(num_teams * 0.002)
        silver_pos = 50
        bronze_pos = 100
    # 1000 队以上:金=10+0.2%,银=5%,铜=10%
    else:
        gold_pos = 10 + int(num_teams * 0.002)
        silver_pos = max(1, int(num_teams * 0.05))
        bronze_pos = max(1, int(num_teams * 0.1))
    # 返回三个名次分界
    return gold_pos, silver_pos, bronze_pos


def get_medal_for_score(score: float, leaderboard: pd.DataFrame) -> str:
    """按绝对分数在排行榜上的位置判定奖牌等级。"""
    # 排行榜分数
    scores = leaderboard["score"].tolist()
    # 计算三类奖牌的名次分界
    gold_pos, silver_pos, bronze_pos = _medal_positions(len(scores))
    # 三个分界对应的分数阈值(第 pos 名的分数)
    gold_threshold = scores[gold_pos - 1]
    silver_threshold = scores[silver_pos - 1]
    bronze_threshold = scores[bronze_pos - 1]
    # 方向判断
    lower_better = is_lower_better(leaderboard)
    # 分数越低越好:依次与阈值比较(<=)
    if lower_better:
        if score <= gold_threshold:
            return "gold"
        if score <= silver_threshold:
            return "silver"
        if score <= bronze_threshold:
            return "bronze"
    else:
        # 分数越高越好:依次与阈值比较(>=)
        if score >= gold_threshold:
            return "gold"
        if score >= silver_threshold:
            return "silver"
        if score >= bronze_threshold:
            return "bronze"
    # 三类奖牌之外
    return "N/A"


def get_medal_for_grade(grade: float, leaderboard: pd.DataFrame) -> str:
    """按相对名次 grade 判定奖牌等级(与分数阈值版等价的 grade 口径)。"""
    # 队伍数
    num_teams = len(leaderboard["score"].tolist())
    # 计算名次分界
    gold_pos, silver_pos, bronze_pos = _medal_positions(num_teams)
    # 名次分界换算成 grade 阈值
    gold_grade = gold_pos / num_teams
    silver_grade = silver_pos / num_teams
    bronze_grade = bronze_pos / num_teams
    # grade 越小名次越高;依次判定
    if grade <= gold_grade:
        return "gold"
    if grade <= silver_grade:
        return "silver"
    if grade <= bronze_grade:
        return "bronze"
    # 奖牌之外
    return "N/A"


def build_submit_grade_and_medal(
    submit_score: float | None, leaderboard: pd.DataFrame | None
) -> tuple[float, str]:
    """由提交分数一步算出 (grade, medal);缺分数或缺排行榜时给最差默认 (1.0, N/A)。"""
    # 无分数或无排行榜:无法计分,返回默认
    if submit_score is None or leaderboard is None:
        return 1.0, "N/A"
    # 计算相对名次
    grade = get_grade_for_score(submit_score, leaderboard)
    # 计算奖牌
    medal = get_medal_for_score(submit_score, leaderboard)
    # 返回二元组
    return grade, medal


def write_epoch_stat(
    epoch_output_dir: Path,
    expected_task_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """聚合单个 epoch 目录下所有任务的 stat.json,写出 epoch 级 stat.json。

    统计内容:资源消耗(费用/token/步数/耗时)、分档 grade 均值与方差、
    奖牌计数与率,以及两类 oracle(自我验证最好成绩)口径的可选统计。
    """
    # 资源消耗累计器(逐任务累加)
    usage = {
        "total_cost": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "total_steps": 0,
        "total_model_time": 0.0,
        "total_sandbox_time": 0.0,
        "total_model_plus_sandbox_time": 0.0,
    }

    def mean(values: list[float]) -> float:
        # 嵌套函数:算术平均
        return sum(values) / len(values)

    def std(values: list[float]) -> float:
        # 嵌套函数:总体标准差(除以 n;样本不足 2 时为 0)
        if len(values) < 2:
            return 0.0
        # 先算均值
        avg = mean(values)
        # 总体方差
        variance = sum((value - avg) ** 2 for value in values) / len(values)
        # 开方
        return math.sqrt(variance)

    # 本 epoch 实际跑到的任务名
    task_names: list[str] = []
    # 任务 -> 提交奖牌
    task_medals: dict[str, str] = {}
    # 任务 -> 提交 grade
    task_grades: dict[str, float] = {}
    # 任务 -> oracle(自我验证最优)奖牌
    oracle_task_medals: dict[str, str] = {}
    # 任务 -> oracle grade
    oracle_task_grades: dict[str, float] = {}
    # 本轮是否出现过 oracle 指标
    has_oracle_metrics = False
    # 任务 -> method2 oracle 奖牌(第二种 oracle 口径)
    method2_oracle_task_medals: dict[str, str] = {}
    # 任务 -> method2 oracle grade
    method2_oracle_task_grades: dict[str, float] = {}
    # 是否出现过 method2 oracle 指标
    has_method2_oracle_metrics = False

    # 按目录名排序遍历本 epoch 下所有任务目录(结果顺序稳定)
    for task_dir in sorted(
        path for path in epoch_output_dir.iterdir() if path.is_dir()
    ):
        # 任务级 stat.json 路径
        stat_path = task_dir / "stat.json"
        # 无统计文件则跳过
        if not stat_path.exists():
            continue
        # 读入该任务的统计载荷
        payload = _read_json_file(stat_path)
        # 不可读则跳过
        if payload is None:
            continue
        # 任务名:优先取载荷里的,退回目录名
        task_name = str(payload.get("task_name") or task_dir.name)
        # 记录任务名
        task_names.append(task_name)
        # 累计费用
        usage["total_cost"] += float(payload.get("total_cost", 0.0))
        # 累计 prompt token
        usage["total_prompt_tokens"] += int(payload.get("total_prompt_tokens", 0))
        # 累计 completion token
        usage["total_completion_tokens"] += int(
            payload.get("total_completion_tokens", 0)
        )
        # 累计总 token
        usage["total_tokens"] += int(payload.get("total_tokens", 0))
        # 累计搜索步数(字段名是 num_steps)
        usage["total_steps"] += int(payload.get("num_steps", 0))
        # 累计模型耗时
        usage["total_model_time"] += get_total_model_time(payload)
        # 累计沙箱耗时
        usage["total_sandbox_time"] += get_total_sandbox_time(payload)
        # 累计模型+沙箱耗时
        usage["total_model_plus_sandbox_time"] += (
            get_total_model_plus_sandbox_time(payload)
        )

        # 提交奖牌(缺失记 N/A)
        submit_medal = payload.get("submit_medal")
        task_medals[task_name] = str(submit_medal) if submit_medal else "N/A"
        # 提交 grade(缺失按最差 1.0)
        submit_grade = payload.get("submit_grade")
        task_grades[task_name] = (
            float(submit_grade) if submit_grade is not None else 1.0
        )
        # oracle 口径一:self_valid_oracle(自我验证下的最好成绩)
        oracle_grade = payload.get("self_valid_oracle_best_grade")
        oracle_medal = payload.get("self_valid_oracle_best_medal")
        # 任一字段出现即认定本轮携带 oracle 指标
        if oracle_grade is not None or oracle_medal is not None:
            has_oracle_metrics = True
            # grade 缺失按 1.0 兜底
            oracle_task_grades[task_name] = (
                float(oracle_grade) if oracle_grade is not None else 1.0
            )
            # medal 缺失记 N/A
            oracle_task_medals[task_name] = (
                str(oracle_medal) if oracle_medal else "N/A"
            )
        # oracle 口径二:method2_oracle
        method2_oracle_grade = payload.get("method2_oracle_best_grade")
        method2_oracle_medal = payload.get("method2_oracle_best_medal")
        if method2_oracle_grade is not None or method2_oracle_medal is not None:
            has_method2_oracle_metrics = True
            method2_oracle_task_grades[task_name] = (
                float(method2_oracle_grade)
                if method2_oracle_grade is not None
                else 1.0
            )
            method2_oracle_task_medals[task_name] = (
                str(method2_oracle_medal) if method2_oracle_medal else "N/A"
            )

    # 统计基准任务集:外部给定的期望任务名(保序去重)优先,否则用实际跑到的任务
    stat_task_names = _ordered_unique(expected_task_names) or task_names
    # 为期望集里但未跑到的任务补默认值(保证分母稳定,跨 epoch 可比)
    for task_name in stat_task_names:
        # 提交奖牌默认 N/A
        task_medals.setdefault(task_name, "N/A")
        # 提交 grade 默认最差 1.0
        task_grades.setdefault(task_name, 1.0)
        # oracle 指标存在时同样补默认
        if has_oracle_metrics:
            oracle_task_medals.setdefault(task_name, "N/A")
            oracle_task_grades.setdefault(task_name, 1.0)
        if has_method2_oracle_metrics:
            method2_oracle_task_medals.setdefault(task_name, "N/A")
            method2_oracle_task_grades.setdefault(task_name, 1.0)

    # 按难度分档
    level_tasks = _level_tasks_for(stat_task_names)
    # 各档任务数
    task_counts = {
        f"{level}_task_count": len(tasks) for level, tasks in level_tasks.items()
    }

    # 各档 grade 的均值与标准差(@n: 按 n 个任务计)
    grade_stats: dict[str, float | None] = {}
    for level, tasks in level_tasks.items():
        # 该档各任务的 grade(缺省 1.0)
        grades = [task_grades.get(task, 1.0) for task in tasks]
        # 均值(空档为 None)
        grade_stats[f"{level}_grade_avg@n"] = mean(grades) if grades else None
        # 标准差
        grade_stats[f"{level}_grade_std@n"] = std(grades) if grades else None

    # 各档奖牌计数(金/银/铜 + 任一奖牌)
    medal_counts: dict[str, int] = {}
    for level, tasks in level_tasks.items():
        # 逐类奖牌计数
        for medal in ("gold", "silver", "bronze"):
            medal_counts[f"{level}_{medal}_count"] = sum(
                1 for task in tasks if task_medals.get(task) == medal
            )
        # 拿到任意奖牌的任务数
        medal_counts[f"{level}_any_count"] = sum(
            1 for task in tasks if task_medals.get(task) in {"gold", "silver", "bronze"}
        )

    # 各档奖牌率(计数/任务数;空档为 None)
    medal_rates: dict[str, float | None] = {}
    for level, tasks in level_tasks.items():
        # 该档任务数
        total = len(tasks)
        for medal in ("gold", "silver", "bronze", "any"):
            medal_rates[f"{level}_{medal}_rate"] = (
                medal_counts[f"{level}_{medal}_count"] / total if total else None
            )

    # oracle 口径一的分档统计(仅当本轮携带该指标)
    oracle_stats: dict[str, float | int | None] = {}
    if has_oracle_metrics:
        for level, tasks in level_tasks.items():
            # 该档 oracle grade 列表
            grades = [oracle_task_grades.get(task, 1.0) for task in tasks]
            # 均值与标准差
            oracle_stats[f"self_valid_oracle_{level}_grade_avg@n"] = (
                mean(grades) if grades else None
            )
            oracle_stats[f"self_valid_oracle_{level}_grade_std@n"] = (
                std(grades) if grades else None
            )
            # 该档任务数
            total = len(tasks)
            # 逐类奖牌计数与率
            for medal in ("gold", "silver", "bronze"):
                count = sum(
                    1 for task in tasks if oracle_task_medals.get(task) == medal
                )
                oracle_stats[f"self_valid_oracle_{level}_{medal}_count"] = count
                oracle_stats[f"self_valid_oracle_{level}_{medal}_rate"] = (
                    count / total if total else None
                )
            # 任一奖牌计数与率
            any_count = sum(
                1
                for task in tasks
                if oracle_task_medals.get(task) in {"gold", "silver", "bronze"}
            )
            oracle_stats[f"self_valid_oracle_{level}_any_count"] = any_count
            oracle_stats[f"self_valid_oracle_{level}_any_rate"] = (
                any_count / total if total else None
            )

    # oracle 口径二的分档统计(仅当本轮携带该指标)
    method2_oracle_stats: dict[str, float | int | None] = {}
    if has_method2_oracle_metrics:
        for level, tasks in level_tasks.items():
            # 该档 method2 oracle grade 列表
            grades = [method2_oracle_task_grades.get(task, 1.0) for task in tasks]
            # 均值与标准差
            method2_oracle_stats[f"method2_oracle_{level}_grade_avg@n"] = (
                mean(grades) if grades else None
            )
            method2_oracle_stats[f"method2_oracle_{level}_grade_std@n"] = (
                std(grades) if grades else None
            )
            # 该档任务数
            total = len(tasks)
            # 逐类奖牌计数与率
            for medal in ("gold", "silver", "bronze"):
                count = sum(
                    1
                    for task in tasks
                    if method2_oracle_task_medals.get(task) == medal
                )
                method2_oracle_stats[f"method2_oracle_{level}_{medal}_count"] = count
                method2_oracle_stats[f"method2_oracle_{level}_{medal}_rate"] = (
                    count / total if total else None
                )
            # 任一奖牌计数与率
            any_count = sum(
                1
                for task in tasks
                if method2_oracle_task_medals.get(task)
                in {"gold", "silver", "bronze"}
            )
            method2_oracle_stats[f"method2_oracle_{level}_any_count"] = any_count
            method2_oracle_stats[f"method2_oracle_{level}_any_rate"] = (
                any_count / total if total else None
            )

    # 汇总 epoch 统计:资源消耗 + 12 小时归一化分数 + 各档任务/成绩统计
    epoch_stat = {
        **usage,
        # 沙箱耗时折算的 12 小时分数(耗时/12h,越小越好)
        "sandbox_12h_score": usage["total_sandbox_time"]
        / TIME_SCALING_SCORE_SECONDS,
        # 模型+沙箱耗时折算
        "model_plus_sandbox_12h_score": usage["total_model_plus_sandbox_time"]
        / TIME_SCALING_SCORE_SECONDS,
        **task_counts,
        **grade_stats,
        **medal_counts,
        **medal_rates,
        **oracle_stats,
        **method2_oracle_stats,
    }
    # 写出 epoch 级 stat.json(缩进 2 便于人工检查)
    (epoch_output_dir / "stat.json").write_text(
        json.dumps(epoch_stat, indent=2), encoding="utf-8"
    )
    # 同时返回统计(调用方可直接使用)
    return epoch_stat


def write_summary_csv(
    output_dir: Path,
    task_metadata_map: dict[str, dict[str, Any]],
    expected_task_names: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """跨 epoch 汇总每个任务的提交成绩,写出 summary.csv 并返回 DataFrame。

    对每个任务统计:分数均值/方差/最优、grade 与奖牌(含两类 oracle 口径)、
    费用统计与运行状态分布(@k: 按 k 个 epoch 计)。
    """
    def epoch_index(path: Path) -> int:
        # 嵌套函数:从目录名提取 epoch 序号(program_ep_12 -> 12;无数字按 0)
        suffix = path.name.split("program_ep_", 1)[-1]
        return int(suffix) if suffix.isdigit() else 0

    def mean(values: list[float]) -> float:
        # 嵌套函数:算术平均
        return sum(values) / len(values)

    def std(values: list[float]) -> float:
        # 嵌套函数:总体标准差
        if len(values) < 2:
            return 0.0
        avg = mean(values)
        variance = sum((value - avg) ** 2 for value in values) / len(values)
        return math.sqrt(variance)

    # 任务 -> 该任务在各 epoch 的统计载荷列表
    task_samples: dict[str, list[dict[str, Any]]] = {}
    # 任务 -> 状态累计计数
    task_status_counts: dict[str, dict[str, int]] = {}
    # 固定的运行状态集合(保证 CSV 列稳定)
    fixed_statuses = [
        "code_execution_error",
        "code_missing",
        "scoring_failed",
        "submission_missing",
        "success",
        "timeout",
        "unknown",
    ]

    # 收集全部 epoch 目录并按序号排序
    epoch_dirs = sorted(
        [
            path
            for path in output_dir.iterdir()
            if path.is_dir() and path.name.startswith("program_ep_")
        ],
        key=epoch_index,
    )
    # 遍历每个 epoch 的每个任务,收集样本与状态计数
    for epoch_dir in epoch_dirs:
        for task_dir in sorted(path for path in epoch_dir.iterdir() if path.is_dir()):
            # 任务级 stat.json
            stat_path = task_dir / "stat.json"
            # 缺失跳过
            if not stat_path.exists():
                continue
            # 读入载荷
            payload = _read_json_file(stat_path)
            # 不可读跳过
            if payload is None:
                continue
            # 任务名(优先载荷,退回目录名)
            task_name = str(payload.get("task_name") or task_dir.name)
            # 追加样本
            task_samples.setdefault(task_name, []).append(payload)
            # 累计该样本的状态计数
            status_count = payload.get("status_count") or {}
            task_status = task_status_counts.setdefault(task_name, {})
            for status, count in dict(status_count).items():
                task_status[status] = task_status.get(status, 0) + int(count)

    # 状态全集沿用固定列表
    all_statuses = fixed_statuses
    # 汇总行收集器
    summary_rows: list[dict[str, Any]] = []
    # 汇总任务集:期望任务名优先(保序去重)
    summary_task_names = _ordered_unique(expected_task_names)
    # 未给定则按实际出现任务排序
    if summary_task_names is None:
        summary_task_names = sorted(task_samples)
    # epoch 总数(用于补齐缺失样本)
    num_epochs = len(epoch_dirs)
    # 逐任务统计
    for task_name in summary_task_names:
        # 该任务的样本列表
        samples = task_samples.get(task_name, [])
        # 任务元数据(排行榜定位用)
        metadata = task_metadata_map.get(task_name)
        # 有元数据才尝试加载排行榜
        leaderboard = load_leaderboard(metadata) if metadata else None
        # 有效的提交分数(过滤 None)
        valid_scores = [
            float(score)
            for score in (sample.get("submit_score") for sample in samples)
            if score is not None
        ]
        # 缺失样本数 = epoch 数 - 实际样本数(用于补 0 保持口径一致)
        missing_sample_count = max(0, num_epochs - len(samples))
        # 费用列表(缺失样本补 0)
        costs = [float(sample.get("total_cost", 0.0)) for sample in samples]
        costs.extend([0.0] * missing_sample_count)

        # 有有效分数才计算均值/方差/最优
        if valid_scores:
            # 分数均值
            score_avg = mean(valid_scores)
            # 分数方差
            score_std = std(valid_scores)
            # 最优分数:lower_is_better 取最小,否则取最大
            score_best = (
                min(valid_scores)
                if leaderboard is not None and is_lower_better(leaderboard)
                else max(valid_scores)
            )
        else:
            # 无分数:三项均为 None
            score_avg = None
            score_std = None
            score_best = None

        # 平均分数的 grade/medal 换算(依赖排行榜)
        if leaderboard is None:
            # 无排行榜:记 N/A
            score_grade_avg = "N/A"
            score_medal_avg = "N/A"
        elif score_avg is not None:
            # 由平均分数算 grade,再由 grade 算奖牌
            score_grade_avg = get_grade_for_score(score_avg, leaderboard)
            score_medal_avg = get_medal_for_grade(score_grade_avg, leaderboard)
        else:
            # 有排行榜但无分数:grade 为 None
            score_grade_avg = None
            score_medal_avg = "N/A"

        # 各 epoch 的提交 grade(缺失按最差 1.0)
        submit_grades = [
            float(grade) if grade is not None else 1.0
            for grade in (sample.get("submit_grade") for sample in samples)
        ]
        # 缺失样本按最差 1.0 补齐
        submit_grades.extend([1.0] * missing_sample_count)
        # grade 的均值/方差/最优(最优=最小 grade)
        if submit_grades:
            grade_avg = mean(submit_grades)
            grade_std = std(submit_grades)
            grade_best = min(submit_grades)
        else:
            grade_avg = None
            grade_std = None
            grade_best = None

        # 提交 grade 对应的奖牌(均值与最优)
        if leaderboard is None or grade_avg is None or grade_best is None:
            medal_avg = "N/A"
            medal_best = "N/A"
        else:
            medal_avg = get_medal_for_grade(grade_avg, leaderboard)
            medal_best = get_medal_for_grade(grade_best, leaderboard)

        # oracle 口径一:self_valid_oracle_best_grade 列表(缺失补最差 1.0)
        oracle_grades = [
            float(grade) if grade is not None else 1.0
            for grade in (
                sample.get("self_valid_oracle_best_grade") for sample in samples
            )
        ]
        oracle_grades.extend([1.0] * missing_sample_count)
        # oracle grade 的均值/方差/最优
        if oracle_grades:
            oracle_grade_avg = mean(oracle_grades)
            oracle_grade_std = std(oracle_grades)
            oracle_grade_best = min(oracle_grades)
        else:
            oracle_grade_avg = None
            oracle_grade_std = None
            oracle_grade_best = None

        # oracle grade 对应奖牌
        if leaderboard is None or oracle_grade_avg is None or oracle_grade_best is None:
            oracle_medal_avg = "N/A"
            oracle_medal_best = "N/A"
        else:
            oracle_medal_avg = get_medal_for_grade(oracle_grade_avg, leaderboard)
            oracle_medal_best = get_medal_for_grade(oracle_grade_best, leaderboard)

        # 是否存在 method2 oracle 指标(任一样本携带即算)
        has_method2_oracle = any(
            "method2_oracle_best_grade" in sample
            or "method2_oracle_best_medal" in sample
            for sample in samples
        )
        if has_method2_oracle:
            # method2 oracle grade 列表(缺失补 1.0)
            method2_oracle_grades = [
                float(grade) if grade is not None else 1.0
                for grade in (
                    sample.get("method2_oracle_best_grade") for sample in samples
                )
            ]
            method2_oracle_grades.extend([1.0] * missing_sample_count)
            # 均值/方差/最优
            method2_oracle_grade_avg = mean(method2_oracle_grades)
            method2_oracle_grade_std = std(method2_oracle_grades)
            method2_oracle_grade_best = min(method2_oracle_grades)
        else:
            method2_oracle_grade_avg = None
            method2_oracle_grade_std = None
            method2_oracle_grade_best = None

        # method2 oracle grade 对应奖牌
        if (
            leaderboard is None
            or method2_oracle_grade_avg is None
            or method2_oracle_grade_best is None
        ):
            method2_oracle_medal_avg = "N/A"
            method2_oracle_medal_best = "N/A"
        else:
            method2_oracle_medal_avg = get_medal_for_grade(
                method2_oracle_grade_avg,
                leaderboard,
            )
            method2_oracle_medal_best = get_medal_for_grade(
                method2_oracle_grade_best,
                leaderboard,
            )

        # 组装该任务的汇总行(列名与下面 summary_columns 严格对应)
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
            "self_valid_oracle_grade_avg@k": oracle_grade_avg,
            "self_valid_oracle_grade_std@k": oracle_grade_std,
            "self_valid_oracle_grade_best@k": oracle_grade_best,
            "self_valid_oracle_medal_avg@k": oracle_medal_avg,
            "self_valid_oracle_medal_best@k": oracle_medal_best,
            "method2_oracle_grade_avg@k": method2_oracle_grade_avg,
            "method2_oracle_grade_std@k": method2_oracle_grade_std,
            "method2_oracle_grade_best@k": method2_oracle_grade_best,
            "method2_oracle_medal_avg@k": method2_oracle_medal_avg,
            "method2_oracle_medal_best@k": method2_oracle_medal_best,
            "cost_avg@k": mean(costs) if costs else 0.0,
            "cost_best@k": min(costs) if costs else 0.0,
            "cost_sum@k": sum(costs),
        }
        # 该任务跨 epoch 的状态累计
        status_count = task_status_counts.get(task_name, {})
        # 以固定状态集为骨架的滚动计数
        status_rollup = {status: 0 for status in all_statuses}
        for status, count in dict(status_count).items():
            if status in status_rollup:
                status_rollup[status] += int(count)
            else:
                # 中文说明:未知状态并入 unknown,保证 CSV 列集合稳定不随数据漂移
                status_rollup["unknown"] += int(count)
        # 缺失样本整体并入 unknown(该 epoch 未产生结果)
        status_rollup["unknown"] += missing_sample_count
        # 状态总数
        total_status = sum(status_rollup.values())
        # 各状态计数写入行
        for status in all_statuses:
            row[status] = status_rollup[status]
        # 成功率 = success / 总状态数(无数据为 0)
        row["success_rate"] = (
            status_rollup["success"] / total_status if total_status else 0.0
        )
        # 收集汇总行
        summary_rows.append(row)

    # 汇总表的固定列顺序(状态列展开 + success_rate 收尾)
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
        "self_valid_oracle_grade_avg@k",
        "self_valid_oracle_grade_std@k",
        "self_valid_oracle_grade_best@k",
        "self_valid_oracle_medal_avg@k",
        "self_valid_oracle_medal_best@k",
        "method2_oracle_grade_avg@k",
        "method2_oracle_grade_std@k",
        "method2_oracle_grade_best@k",
        "method2_oracle_medal_avg@k",
        "method2_oracle_medal_best@k",
        "cost_avg@k",
        "cost_best@k",
        "cost_sum@k",
        *all_statuses,
        "success_rate",
    ]
    # 构建汇总 DataFrame(空缺列自动填 NaN)
    summary_df = pd.DataFrame(summary_rows, columns=summary_columns)
    # 写出 summary.csv(不写索引)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    # 返回 DataFrame 供调用方进一步使用
    return summary_df


def write_global_stat(
    output_dir: Path,
) -> dict[str, Any]:
    """聚合所有 epoch 的 stat.json,写出全局 stat.json(总消耗 + 各指标跨 epoch 均值/方差)。"""
    def epoch_index(path: Path) -> int:
        # 嵌套函数:从目录名提取 epoch 序号
        suffix = path.name.split("program_ep_", 1)[-1]
        return int(suffix) if suffix.isdigit() else 0

    def mean(values: list[float]) -> float:
        # 嵌套函数:算术平均
        return sum(values) / len(values)

    def std(values: list[float]) -> float:
        # 嵌套函数:总体标准差
        if len(values) < 2:
            return 0.0
        avg = mean(values)
        variance = sum((value - avg) ** 2 for value in values) / len(values)
        return math.sqrt(variance)

    def aggregate(values: list[float | None]) -> tuple[float | None, float | None]:
        # 嵌套函数:过滤 None 后求 (均值, 标准差);全空返回 (None, None)
        numeric_values = [float(value) for value in values if value is not None]
        if not numeric_values:
            return None, None
        return mean(numeric_values), std(numeric_values)

    # 收集并按序号排序 epoch 目录
    epoch_dirs = sorted(
        [
            path
            for path in output_dir.iterdir()
            if path.is_dir() and path.name.startswith("program_ep_")
        ],
        key=epoch_index,
    )
    # 资源消耗键集合(这些键跨 epoch 直接求和)
    usage_keys = {
        "total_cost",
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_tokens",
        "total_steps",
        "total_model_time",
        "total_sandbox_time",
        "total_model_plus_sandbox_time",
    }
    # 资源消耗累计器
    usage = {
        "total_cost": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "total_steps": 0,
        "total_model_time": 0.0,
        "total_sandbox_time": 0.0,
        "total_model_plus_sandbox_time": 0.0,
    }
    # 其余指标键 -> 各 epoch 取值列表(之后求均值/方差)
    metric_values: dict[str, list[float | None]] = {}
    # 分档 grade 均值键(这些键按档位单独聚合)
    grade_avg_keys = {
        "low_grade_avg@n",
        "middle_grade_avg@n",
        "high_grade_avg@n",
        "all_grade_avg@n",
    }
    # 分档 grade 方差键(全局口径重算,不直接平均)
    grade_std_keys = {
        "low_grade_std@n",
        "middle_grade_std@n",
        "high_grade_std@n",
        "all_grade_std@n",
    }
    # 各档 grade 均值的取值列表
    grade_avg_values: dict[str, list[float | None]] = {
        "low": [],
        "middle": [],
        "high": [],
        "all": [],
    }
    # oracle 口径一各档 grade 均值列表
    oracle_grade_avg_values: dict[str, list[float | None]] = {
        "low": [],
        "middle": [],
        "high": [],
        "all": [],
    }
    # oracle 口径二各档 grade 均值列表
    method2_oracle_grade_avg_values: dict[str, list[float | None]] = {
        "low": [],
        "middle": [],
        "high": [],
        "all": [],
    }
    # 遍历各 epoch 统计文件
    for epoch_dir in epoch_dirs:
        # epoch 级 stat.json 路径
        epoch_stat_path = epoch_dir / "stat.json"
        # 读入
        epoch_stat = _read_json_file(epoch_stat_path)
        # 不可读跳过
        if epoch_stat is None:
            continue
        # 累计资源消耗各项
        usage["total_cost"] += float(epoch_stat["total_cost"])
        usage["total_prompt_tokens"] += int(epoch_stat["total_prompt_tokens"])
        usage["total_completion_tokens"] += int(epoch_stat["total_completion_tokens"])
        usage["total_tokens"] += int(epoch_stat["total_tokens"])
        usage["total_steps"] += int(epoch_stat["total_steps"])
        usage["total_model_time"] += float(epoch_stat.get("total_model_time", 0.0))
        usage["total_sandbox_time"] += float(
            epoch_stat.get("total_sandbox_time", 0.0)
        )
        usage["total_model_plus_sandbox_time"] += float(
            epoch_stat.get("total_model_plus_sandbox_time", 0.0)
        )
        # 中文说明:把 epoch 级的任务数与奖牌类指标按均值/方差口径再聚合一遍。
        for key, value in epoch_stat.items():
            # 资源消耗键:已求和,跳过
            if key in usage_keys:
                continue
            # 分档 grade 均值:按档位归集
            if key in grade_avg_keys:
                # 从键名提取档位(第一个下划线前)
                level = key.split("_", 1)[0]
                grade_avg_values[level].append(value)
                continue
            # oracle 口径一的分档 grade 均值
            if key.startswith("self_valid_oracle_") and key.endswith("_grade_avg@n"):
                # 去前缀后提取档位
                level = key.removeprefix("self_valid_oracle_").split("_", 1)[0]
                oracle_grade_avg_values[level].append(value)
                continue
            # oracle 口径二的分档 grade 均值
            if key.startswith("method2_oracle_") and key.endswith("_grade_avg@n"):
                level = key.removeprefix("method2_oracle_").split("_", 1)[0]
                method2_oracle_grade_avg_values[level].append(value)
                continue
            # 方差键全局口径重算,原值跳过
            if key in grade_std_keys:
                continue
            if key.startswith("self_valid_oracle_") and key.endswith("_grade_std@n"):
                continue
            if key.startswith("method2_oracle_") and key.endswith("_grade_std@n"):
                continue
            # 其余指标:按键归集供均值/方差统计
            metric_values.setdefault(key, []).append(value)

    # 其余指标的均值/方差统计(按键名排序保证输出稳定)
    metric_stats: dict[str, Any] = {}
    for key, values in sorted(metric_values.items()):
        # 求均值与方差
        avg, std_value = aggregate(values)
        # 分别以 _avg/_std 后缀记录
        metric_stats[f"{key}_avg"] = avg
        metric_stats[f"{key}_std"] = std_value

    # 汇总全局统计
    global_stat = {
        **usage,
        # epoch 数量
        "num_epochs": len(epoch_dirs),
        # 12 小时归一化分数(总耗时口径)
        "sandbox_12h_score": usage["total_sandbox_time"]
        / TIME_SCALING_SCORE_SECONDS,
        "model_plus_sandbox_12h_score": usage["total_model_plus_sandbox_time"]
        / TIME_SCALING_SCORE_SECONDS,
        **metric_stats,
    }
    # 写入各档提交 grade 的全局均值/方差(@k 口径:对 epoch 均值再统计)
    for level, values in grade_avg_values.items():
        avg, std_value = aggregate(values)
        global_stat[f"{level}_grade_avg@k"] = avg
        global_stat[f"{level}_grade_std@k"] = std_value
    # oracle 口径一(仅当有数据)
    for level, values in oracle_grade_avg_values.items():
        if not values:
            continue
        avg, std_value = aggregate(values)
        global_stat[f"self_valid_oracle_{level}_grade_avg@k"] = avg
        global_stat[f"self_valid_oracle_{level}_grade_std@k"] = std_value
    # oracle 口径二(仅当有数据)
    for level, values in method2_oracle_grade_avg_values.items():
        if not values:
            continue
        avg, std_value = aggregate(values)
        global_stat[f"method2_oracle_{level}_grade_avg@k"] = avg
        global_stat[f"method2_oracle_{level}_grade_std@k"] = std_value
    # 写出全局 stat.json
    (output_dir / "stat.json").write_text(
        json.dumps(global_stat, indent=2), encoding="utf-8"
    )
    # 返回统计结果
    return global_stat


def write_time_scaling(output_dir: Path) -> pd.DataFrame:
    """计算"时间预算 -> 覆盖率/成绩"的扩展曲线,写出 CSV 与 PNG 图。

    对每条时间轴(模型+沙箱 / 仅沙箱),以观测到的耗时为预算候选,
    统计在预算内完成的样本占比、平均 grade 与奖牌率。
    """
    def epoch_index(path: Path) -> int:
        # 嵌套函数:从目录名提取 epoch 序号
        suffix = path.name.split("program_ep_", 1)[-1]
        return int(suffix) if suffix.isdigit() else 0

    # 收集并排序 epoch 目录
    epoch_dirs = sorted(
        [
            path
            for path in output_dir.iterdir()
            if path.is_dir() and path.name.startswith("program_ep_")
        ],
        key=epoch_index,
    )

    # 样本收集器:(任务名, grade, 奖牌, 两条时间轴耗时)
    samples: list[dict[str, Any]] = []
    # 遍历各 epoch 的各任务
    for epoch_dir in epoch_dirs:
        for task_dir in sorted(path for path in epoch_dir.iterdir() if path.is_dir()):
            # 任务级 stat.json
            stat_path = task_dir / "stat.json"
            # 缺失跳过
            if not stat_path.exists():
                continue
            # 读入载荷
            payload = _read_json_file(stat_path)
            # 不可读跳过
            if payload is None:
                continue
            # 提交 grade
            submit_grade = payload.get("submit_grade")
            # 收集样本(grade 缺失按最差 1.0,奖牌缺失记 N/A)
            samples.append(
                {
                    "task_name": str(payload.get("task_name") or task_dir.name),
                    "submit_grade": (
                        float(submit_grade) if submit_grade is not None else 1.0
                    ),
                    "submit_medal": str(payload.get("submit_medal") or "N/A"),
                    # 仅沙箱耗时
                    "sandbox_time": get_total_sandbox_time(payload),
                    # 模型+沙箱耗时
                    "model_plus_sandbox_time": get_total_model_plus_sandbox_time(
                        payload
                    ),
                }
            )

    # 时间轴规格:(载荷字段, 展示标签)
    axis_specs = [
        ("model_plus_sandbox_time", "Model + Sandbox Time"),
        ("sandbox_time", "Sandbox Time"),
    ]
    # 曲线行收集器
    rows: list[dict[str, Any]] = []

    # 有样本才计算曲线
    if samples:
        # 样本总数
        total = len(samples)
        # 逐时间轴计算
        for axis_key, axis_label in axis_specs:
            # 预算候选:0、12h 基准,加上每个样本实际耗时(排序去重)
            budgets = sorted(
                {
                    0.0,
                    float(TIME_SCALING_SCORE_SECONDS),
                    *[float(sample[axis_key]) for sample in samples],
                }
            )
            # 逐预算计算指标
            for budget_seconds in budgets:
                # 预算内完成的样本(耗时 <= 预算)
                covered_samples = [
                    sample
                    for sample in samples
                    if float(sample[axis_key]) <= budget_seconds
                ]
                # 覆盖样本数
                covered_count = len(covered_samples)
                # grade 列表:覆盖样本用真实 grade,未覆盖按最差 1.0
                grades = [
                    float(sample["submit_grade"])
                    if float(sample[axis_key]) <= budget_seconds
                    else 1.0
                    for sample in samples
                ]
                # 奖牌率 = 覆盖样本中拿到任意奖牌的比例(分母为全部样本)
                medal_rate = (
                    sum(
                        1
                        for sample in covered_samples
                        if sample["submit_medal"] in {"gold", "silver", "bronze"}
                    )
                    / total
                )
                # 记录该预算点
                rows.append(
                    {
                        "time_axis": axis_key,
                        "time_label": axis_label,
                        "budget_seconds": float(budget_seconds),
                        "budget_hours": float(budget_seconds) / 3600.0,
                        "sample_count": total,
                        "covered_sample_count": covered_count,
                        "coverage_rate": covered_count / total,
                        "grade_avg@k": sum(grades) / len(grades),
                        "medal_rate": medal_rate,
                    }
                )

    # 构建 DataFrame(固定列序)
    time_scaling_df = pd.DataFrame(
        rows,
        columns=[
            "time_axis",
            "time_label",
            "budget_seconds",
            "budget_hours",
            "sample_count",
            "covered_sample_count",
            "coverage_rate",
            "grade_avg@k",
            "medal_rate",
        ],
    )
    # 写出 CSV
    time_scaling_df.to_csv(output_dir / "time_scaling.csv", index=False)

    # 无数据(空表)时跳过绘图
    if time_scaling_df.empty:
        return time_scaling_df

    # 延迟导入 matplotlib:仅在确实需要绘图时引入
    import matplotlib.pyplot as plt

    # 2x2 画布:两行(奖牌率/grade) x 两列(两条时间轴)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    # 逐时间轴绘制一列
    for column, (axis_key, axis_label) in enumerate(axis_specs):
        # 该时间轴的曲线数据
        axis_df = time_scaling_df[time_scaling_df["time_axis"] == axis_key]

        # 上排:奖牌率曲线
        medal_ax = axes[0, column]
        medal_ax.plot(
            axis_df["budget_hours"],
            axis_df["medal_rate"],
            color="tab:blue",
            linewidth=2,
        )
        # 12 小时基准虚线
        medal_ax.axvline(12.0, color="gray", linestyle="--", linewidth=1)
        # 标题与轴标签
        medal_ax.set_title(axis_label)
        medal_ax.set_ylabel("Medal Rate")
        medal_ax.set_xlabel("Time Budget (hours)")
        # 奖牌率取值 [0,1]
        medal_ax.set_ylim(0.0, 1.0)
        medal_ax.grid(alpha=0.3)

        # 下排:平均 grade 曲线(越低越好,故 y 轴反向)
        grade_ax = axes[1, column]
        grade_ax.plot(
            axis_df["budget_hours"],
            axis_df["grade_avg@k"],
            color="tab:orange",
            linewidth=2,
        )
        # 12 小时基准虚线
        grade_ax.axvline(12.0, color="gray", linestyle="--", linewidth=1)
        grade_ax.set_ylabel("Grade (lower is better)")
        grade_ax.set_xlabel("Time Budget (hours)")
        # 反向 y 轴:上 1.0 下 0.0,让"更好"朝上
        grade_ax.set_ylim(1.0, 0.0)
        grade_ax.grid(alpha=0.3)

    # 总标题
    fig.suptitle("Time Scaling")
    # 保存 PNG(200 dpi)
    fig.savefig(output_dir / "time_scaling.png", dpi=200)
    # 关闭画布释放内存
    plt.close(fig)
    # 返回曲线 DataFrame
    return time_scaling_df
