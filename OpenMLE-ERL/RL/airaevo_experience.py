"""AIRA-Evo experience utilities adapted for RL ProgramDatabase.

The original AIRA-Evo implementation stores state in Dojo Node/Journal objects.
The RL loop stores evaluated programs in SQLite, so this module keeps the same
selection and prompt-memory signals while operating on Program-like objects.
"""

# 本模块把 AIRA-Evo 的经验机制移植到 RL 训练循环：以携带测试奖励（test-reward）
# 的 Program 对象（存于 SQLite 的 ProgramDatabase）替代原 Dojo Node/Journal，
# 保留同样的选择信号与提示词记忆信号：
# - 经验卡（experience card）：每个程序的结构化经验摘要（适应度、方法族、新颖度、
#   相对父代提升、错误签名等），信号来自真实执行与评测结果（execution-grounded）；
# - 策略板（strategy board）：全局方向分布、当前最优、探索不足方向与重复错误统计；
# - 算子定向记忆（Targeted Memory Context）：为 Draft/Improve/Crossover/Debug
#   四算子分别裁剪相关历史经验，插入提示词引导 Evo-Max 风格的进化搜索。

# 推迟注解求值（PEP 563），允许在注解中使用较新的联合类型语法
from __future__ import annotations

# ast：解析代码提取 import 依赖特征
import ast
# json：解析 LLM 输出文本中的 JSON 对象
import json
# math：开方（新颖度）、isfinite（有效性过滤）等
import math
# re：空白压缩、正则兜底提取 import、提取 JSON 片段
import re
# 类型注解工具
from typing import Any, Iterable

# 判定为失败的状态集合：这些状态的经验节点不能作为成功样本参与选择与记忆
FAILED_STATUSES = {
    "buggy",
    "error",
    "failed",
    "failure",
    "invalid",
    "sandbox_error",
    "scoring_failed",
    "submission_missing",
    "timeout",
    "unknown",
}
# 判定为成功的状态集合
SUCCESS_STATUSES = {"ok", "passed", "success", "successful", "valid"}
# 父代选择效用（selection utility）默认权重：适应度 1.0、相对父代提升 0.4、新颖度 0.25
DEFAULT_PARENT_UTILITY_WEIGHTS = {"score": 1.0, "delta": 0.4, "novelty": 0.25}


# 安全地把任意值转为有限浮点数：转换失败或为 inf/nan 时返回 None
def finite_float(value: Any) -> float | None:
    # 尝试转 float，类型/取值不符时判为无效
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    # 过滤 inf/nan 等非有限值
    if not math.isfinite(numeric):
        return None
    # 返回有效浮点数
    return numeric


# 压缩文本：把连续空白折叠为单空格并截断到 max_chars（超出部分以 ... 结尾），
# 用于控制经验卡字段与提示词记忆的长度
def compact_text(value: Any, max_chars: int = 240) -> str:
    # 空值转空串、折叠空白并去首尾空格
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    # 未超长时直接返回
    if len(text) <= max_chars:
        return text
    # 超长时截断到 max_chars-3 并补省略号
    return text[: max_chars - 3].rstrip() + "..."


# 读取 program 的 metadata 字典：属性缺失或类型不符时返回空 dict
def _metadata(program: Any) -> dict[str, Any]:
    # getattr 兜底读取 metadata 属性
    metadata = getattr(program, "metadata", {}) or {}
    # 仅当为 dict 时复制返回，否则返回空字典
    return dict(metadata) if isinstance(metadata, dict) else {}


# 返回 RL AIRA-Evo 选择所用的测试奖励适应度信号（test-reward fitness）：
# 依次尝试 metadata 中多个奖励字段，最后回退到对象自身的 base_reward/reward 属性
def program_selection_score(program: Any) -> float | None:
    """Return the test-reward fitness signal used by RL AIRA-Evo selection."""
    # 读取元数据
    metadata = _metadata(program)
    # 按优先级遍历各奖励字段名（静态基础奖励优先于动态奖励）
    for key in (
        "metric_static_base_reward",
        "static_base_reward",
        "base_reward",
        "dynamic_base_reward",
        "reward",
    ):
        # 取字段值并要求是有限浮点数
        value = finite_float(metadata.get(key))
        # 命中第一个有效值即返回
        if value is not None:
            return value
    # 元数据缺失时回退到对象属性 base_reward
    value = finite_float(getattr(program, "base_reward", None))
    # 命中则返回
    if value is not None:
        return value
    # 最后回退到对象属性 reward
    return finite_float(getattr(program, "reward", None))


# 提取程序的状态文本：优先取 metadata 的类别/状态字段，其次 payload.status，
# 最后按 HTTP 风格 status_code 判定（200 视为 valid，否则 unknown）
def _status_text(program: Any) -> str:
    # 读取元数据
    metadata = _metadata(program)
    # 依次尝试 code_category、hack_category、status、sandbox_status 字段
    for key in ("code_category", "hack_category", "status", "sandbox_status"):
        # 取字符串字段并去空白
        value = str(metadata.get(key) or "").strip()
        # 第一个非空字段即作为状态
        if value:
            return value
    # 回退到 payload 中的 status 字段
    payload = getattr(program, "payload", {}) or {}
    # 仅当 payload 是 dict 时才读取
    if isinstance(payload, dict):
        value = str(payload.get("status") or "").strip()
        # 非空即作为状态返回
        if value:
            return value
    # 读取 status_code 属性
    status_code = getattr(program, "status_code", None)
    # 200 视为有效，其余归为 unknown
    return "valid" if status_code == 200 else "unknown"


# 判断状态文本是否属于失败：命中标准失败状态集合或额外的失败别名集合
def status_is_failure(status: str) -> bool:
    # 规范化：转字符串、去空白、转小写
    normalized = str(status or "").strip().lower()
    # 命中标准失败状态集合
    if normalized in FAILED_STATUSES:
        return True
    # 命中额外的失败别名（被判定 hack、空输出、未通过校验、生成中断、无代码等）
    if normalized in {"hack", "hack_verify", "empty", "no_verify", "generation_abort", "generation_abort_group", "no_code"}:
        return True
    # 其余状态不算失败
    return False


# 判断程序是否为成功样本：未被判定 hack、状态码 200、状态非失败、未生成中断，
# 且能提取出有效选择分数（execution-grounded：以真实执行/评测结果为准）
def is_success_program(program: Any) -> bool:
    # 读取元数据
    metadata = _metadata(program)
    # hack 标记非 0 表示被判定为投机取巧，直接不算成功
    if int(getattr(program, "hack", 0) or 0) != 0:
        return False
    # 状态码非 200 视为执行失败
    if int(getattr(program, "status_code", 0) or 0) != 200:
        return False
    # 状态文本属于失败类别则不算成功
    if status_is_failure(_status_text(program)):
        return False
    # 元数据标记生成中断的样本不算成功
    if metadata.get("generation_aborted"):
        return False
    # 最终要求存在有效的测试奖励分数
    return program_selection_score(program) is not None


# 判断程序是否应作为 Debug 算子的候选：非成功样本、无分数或分数<=0 都需要修复
def is_debug_candidate(program: Any) -> bool:
    # 取选择分数
    score = program_selection_score(program)
    # 失败样本、无分数、非正分数均视为调试候选
    return (not is_success_program(program)) or score is None or score <= 0.0


# 提取代码的顶层 import 模块名集合：优先用 AST 精确解析，语法错误时用正则兜底，
# 结果供方法族（method family）识别使用
def extract_imports(code: str) -> list[str]:
    # 用集合去重保存模块名
    imports: set[str] = set()
    # 尝试 AST 解析
    try:
        tree = ast.parse(code or "")
        # 遍历语法树所有节点
        for node in ast.walk(tree):
            # import x, y 语句：取每个别名的顶层模块名（去掉子模块部分）
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".", 1)[0])
            # from x import y 语句：取被导入模块的顶层名
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])
    except SyntaxError:
        # 语法无法解析时，用正则按行匹配 import/from 语句提取模块名
        for match in re.finditer(r"^\s*(?:import|from)\s+([A-Za-z_][\w]*)", code or "", flags=re.MULTILINE):
            imports.add(match.group(1))
    # 排序后返回，保证结果稳定
    return sorted(imports)


# 依据 import 与代码文本识别方法族（method family，如 lightgbm/xgboost/pytorch 等），
# 用于统计方向新颖度与策略板上的方向分布；最多取前 3 个族用 + 拼接
def detect_method_family(code: str, imports: Iterable[str] | None = None) -> str:
    # 把 import 模块名集合转小写便于匹配
    imports_set = {item.lower() for item in (imports or [])}
    # 代码全文转小写用于关键词匹配
    text = (code or "").lower()
    # 按检出顺序保存方法族标签
    families: list[str] = []

    # 内部小工具：向 families 追加去重的族标签
    def add(name: str) -> None:
        # 已存在则跳过，保持首次检出顺序
        if name not in families:
            families.append(name)

    # 检测 LightGBM：import 或文本关键词命中
    if {"lightgbm", "lgbm"} & imports_set or "lgbm" in text or "lightgbm" in text:
        add("lightgbm")
    # 检测 XGBoost
    if "xgboost" in imports_set or "xgb" in text or "xgboost" in text:
        add("xgboost")
    # 检测 CatBoost
    if "catboost" in imports_set or "catboost" in text:
        add("catboost")
    # 检测 PyTorch（torch import、pytorch 或 nn.module 关键词）
    if "torch" in imports_set or "pytorch" in text or "nn.module" in text:
        add("pytorch")
    # 检测 TensorFlow/Keras
    if {"tensorflow", "keras"} & imports_set or "tensorflow" in text or "keras" in text:
        add("tensorflow")
    # 检测 scikit-learn
    if "sklearn" in imports_set or "scikit" in text:
        add("sklearn")
    # 检测 NLP Transformer 方向（transformers/bert/tokenizer 关键词）
    if "transformers" in imports_set or "bert" in text or "tokenizer" in text:
        add("nlp_transformer")
    # 检测视觉方向（cv2/PIL/torchvision import 或 image 关键词）
    if {"cv2", "pil", "torchvision"} & imports_set or "image" in text:
        add("vision")
    # 检测集成方法（ensemble/stacking/voting 关键词）
    if "ensemble" in text or "stacking" in text or "voting" in text:
        add("ensemble")
    # 检测交叉验证技巧（cross_val/KFold/StratifiedKFold 关键词）
    if "cross_val" in text or "kfold" in text or "stratifiedkfold" in text:
        add("cv")
    # 一个族都没检出时归为通用机器学习
    if not families:
        return "general_ml"
    # 最多取前 3 个族拼接为方法族标签
    return "+".join(families[:3])


# 收集程序的全部父代 id：包括主 parent_id 与 metadata 中的交叉（Crossover）父代列表，
# 用于构建谱系与计算相对父代的提升量
def parent_ids(program: Any) -> list[int]:
    # 结果 id 列表
    ids: list[int] = []
    # 读取主 parent_id 属性
    parent_id = getattr(program, "parent_id", None)
    # 存在则尝试转 int 加入
    if parent_id is not None:
        try:
            ids.append(int(parent_id))
        except (TypeError, ValueError):
            pass
    # 读取元数据
    metadata = _metadata(program)
    # 遍历交叉（Crossover）父代 id 列表
    for value in metadata.get("crossover_parent_ids") or []:
        try:
            pid = int(value)
        except (TypeError, ValueError):
            continue
        # 去重后追加
        if pid not in ids:
            ids.append(pid)
    # 返回父代 id 列表
    return ids


# 安全读取程序 id：缺失或无法转 int 时返回 None
def _program_id(program: Any) -> int | None:
    # 读取 id 属性
    value = getattr(program, "id", None)
    try:
        # 尝试转 int
        return int(value)
    except (TypeError, ValueError):
        # 转换失败返回 None
        return None


# 以 id 为键建立 program 索引字典，便于按父代 id 快速查表
def _index(programs: Iterable[Any]) -> dict[int, Any]:
    # 结果字典：id -> program
    result: dict[int, Any] = {}
    # 遍历所有程序
    for program in programs:
        # 读取程序 id
        pid = _program_id(program)
        # id 有效时写入索引
        if pid is not None:
            result[pid] = program
    # 返回索引字典
    return result


# 计算子代相对最优父代的正向提升量：lower_is_better 时为 父-子，否则为 子-父；
# 任一分数缺失时返回 None
def _positive_delta(child_score: float | None, parent_score: float | None, lower_is_better: bool) -> float | None:
    # 双方分数都存在才能计算差值
    if child_score is None or parent_score is None:
        return None
    # 按优化方向计算带符号的提升量
    return float(parent_score - child_score if lower_is_better else child_score - parent_score)


# 提取程序的错误签名（error signature）：优先取元数据中的签名/hack 原因/中断原因，
# 失败状态时用 状态:状态码，再回退 payload 里的错误详情；供 Debug 算子匹配同类历史错误
def error_signature(program: Any) -> str | None:
    # 读取元数据
    metadata = _metadata(program)
    # 依次尝试三个错误相关字段
    for key in ("error_signature", "hack_reason", "generation_abort_reason"):
        # 压缩到 160 字符内
        value = compact_text(metadata.get(key), 160)
        # 第一个非空字段即作为签名
        if value:
            return value
    # 取状态文本与状态码
    status = _status_text(program)
    status_code = getattr(program, "status_code", None)
    # 失败状态时以 状态:状态码 形式生成签名
    if status_is_failure(status):
        return f"{status}:{status_code}"
    # 回退到 payload 中的错误详情
    payload = getattr(program, "payload", {}) or {}
    # 仅当 payload 是 dict 时才读取
    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("error")
        # 有详情则压缩后返回
        if detail:
            return compact_text(detail, 160)
    # 无任何错误信息时返回 None
    return None


# 统计在该程序之前同方法族的经验卡数量：可按 node_id 排除自身，
# 结果用于新颖度（novelty）与"是否新方向"的判定
def _family_count_before(method_family: str, previous_cards: list[dict[str, Any]], node_id: str | None = None) -> int:
    # 同族计数器
    count = 0
    # 遍历历史经验卡
    for card in previous_cards:
        # 跳过与当前节点同 id 的卡片（即排除自身）
        if node_id and str(card.get("node_id")) == str(node_id):
            continue
        # 方法族相同则计数加一
        if str(card.get("method_family_auto") or "unknown") == method_family:
            count += 1
    # 返回同族历史数量
    return count


# 为单个程序构建 AIRA-Evo 经验卡（experience card）：汇总适应度、方法族、新颖度、
# 相对父代提升、错误签名等字段，作为后续父代选择与记忆提示词的结构化依据
def build_experience_card(
    program: Any,
    *,
    all_programs: Iterable[Any] | None = None,
    lower_is_better: bool = False,
) -> dict[str, Any]:
    # 全体程序快照列表
    programs = list(all_programs or [])
    # 建立 id -> program 索引便于父代查表
    by_id = _index(programs)
    # 收集其他程序已缓存的历史经验卡
    previous_cards = []
    # 遍历全体程序
    for candidate in programs:
        # 读取其元数据
        metadata = _metadata(candidate)
        # 取缓存的经验卡字段
        card = metadata.get("airaevo_experience_card")
        # 确实是 dict 时才纳入历史卡列表
        if isinstance(card, dict):
            previous_cards.append(card)

    # 取程序源码文本
    code = str(getattr(program, "code", "") or "")
    # 解析 import 得到依赖列表
    imports = extract_imports(code)
    # 识别方法族标签
    method_family = detect_method_family(code, imports)
    # 节点 id（字符串化）
    node_id = str(getattr(program, "id", "") or "")
    # 统计此前同方法族出现次数（排除自身）
    family_count_before = _family_count_before(method_family, previous_cards, node_id=node_id)
    # 取本程序的适应度分数
    score = program_selection_score(program)
    # 收集各父代的适应度分数
    parent_scores = []
    # 遍历父代 id
    for pid in parent_ids(program):
        # 查表得到父代对象
        parent = by_id.get(pid)
        # 父代存在时读取其分数
        if parent is not None:
            parent_score = program_selection_score(parent)
            # 分数有效才纳入
            if parent_score is not None:
                parent_scores.append(parent_score)
    # 最优父代分数初始化为 None
    best_parent_score = None
    # 有父代分数时按优化方向取最优（越小越好取 min，否则取 max）
    if parent_scores:
        best_parent_score = min(parent_scores) if lower_is_better else max(parent_scores)

    # 读取本程序元数据
    metadata = _metadata(program)
    # 取富摘要（LLM 生成的分析方法与父代对比的内容）
    rich_summary = metadata.get("airaevo_rich_summary")
    # 取状态文本
    status = _status_text(program)
    # 组装经验卡字典：含模式版本、谱系、分数、状态、方法族、新颖度与提升量等
    card = {
        "schema_version": 1,
        "node_id": node_id,
        "step_id": int(getattr(program, "id", 0) or 0),
        "operator": str(getattr(program, "generation_mode", "") or metadata.get("generation_mode") or ""),
        "parents": parent_ids(program),
        "parent_node_ids": [str(pid) for pid in parent_ids(program)],
        "generation_id": int(metadata.get("airaevo_generation_id") or 0),
        "score": getattr(program, "score", None),
        "fitness": score,
        "reward": getattr(program, "reward", None),
        "base_reward": getattr(program, "base_reward", None),
        "status": status,
        "status_code": getattr(program, "status_code", None),
        "is_buggy": not is_success_program(program),
        "sandbox_time_used": getattr(program, "running_time", None),
        "imports": imports,
        "method_family_auto": method_family,
        "family_count_before": family_count_before,
        "is_new_direction": family_count_before == 0,
        "novelty_score": 1.0 / math.sqrt(1.0 + family_count_before),
        "delta_vs_parent": _positive_delta(score, best_parent_score, lower_is_better),
        "error_signature": error_signature(program),
        "rank": None,
        "current_best": None,
        "selection_utility": None,
        "plan": compact_text(getattr(program, "raw_text", ""), 500),
        "analysis": compact_text(metadata.get("clear_run_log") or metadata.get("result_info"), 500),
    }
    # 富摘要为 dict 时附加到卡片
    if isinstance(rich_summary, dict):
        card["rich_summary"] = rich_summary
    # 返回完整经验卡
    return card


# 读取程序的经验卡：metadata 中已有缓存卡直接复用，否则现场构建
def _card(program: Any, all_programs: list[Any]) -> dict[str, Any]:
    # 读取元数据
    metadata = _metadata(program)
    # 取缓存的经验卡
    card = metadata.get("airaevo_experience_card")
    # 已有卡直接返回
    if isinstance(card, dict):
        return card
    # 无缓存时现场构建新卡
    return build_experience_card(program, all_programs=all_programs)


# 把分数列表 min-max 归一化到 [0,1]：lower_is_better 时反向（越小越接近 1），
# None/非有限值映射为 0，全部相同时统一映射为 0.5
def _normalize_values(values: list[float | None], lower_is_better: bool) -> list[float]:
    # 过滤出有限有效值
    finite_values = [value for value in values if value is not None and math.isfinite(value)]
    # 无有效值时全部归 0
    if not finite_values:
        return [0.0 for _ in values]
    # 有效值最小值
    min_value = min(finite_values)
    # 有效值最大值
    max_value = max(finite_values)
    # 最大最小相同（无区分度）时：有值的位置给 0.5，无值给 0
    if min_value == max_value:
        return [0.5 if value is not None else 0.0 for value in values]
    # 逐值归一化
    normalized = []
    # 遍历原列表保持位置对应
    for value in values:
        # 无效值归 0
        if value is None or not math.isfinite(value):
            normalized.append(0.0)
        elif lower_is_better:
            # 越小越好：用 (max-x)/跨度，越小归一化后越大
            normalized.append((max_value - value) / (max_value - min_value))
        else:
            # 越大越好：用 (x-min)/跨度
            normalized.append((value - min_value) / (max_value - min_value))
    # 截断到 [0,1] 区间后返回
    return [float(max(0.0, min(1.0, value))) for value in normalized]


# 把非负值列表按最大值归一化到 [0,1]（相对最大者的比例），全 0 时返回全 0
def _normalize_positive(values: list[float | None]) -> list[float]:
    # 负值/None 一律截断为 0
    positives = [max(0.0, value or 0.0) for value in values]
    # 取最大值（空列表时按 0 处理）
    max_value = max(positives or [0.0])
    # 最大值非正时无区分度，全部归 0
    if max_value <= 0.0:
        return [0.0 for _ in positives]
    # 按最大值等比例缩放
    return [float(value / max_value) for value in positives]


# 数值稳定的 softmax：带温度参数，非有限值按 0 处理，总和异常时退化为均匀分布
def softmax(values: list[float], temperature: float = 1.0) -> list[float]:
    # 空列表直接返回空结果
    if not values:
        return []
    # 温度下限保护，避免除零
    temperature = max(float(temperature or 1.0), 1e-8)
    # 非有限值替换为 0
    finite_values = [value if math.isfinite(value) else 0.0 for value in values]
    # 取最大值用于减法防 exp 溢出
    max_value = max(finite_values)
    # 计算各元素的指数项（已做最大值平移）
    exp_values = [math.exp((value - max_value) / temperature) for value in finite_values]
    # 求和作为配分函数
    total = sum(exp_values)
    # 总和异常（<=0 或非有限）时退化为均匀分布
    if total <= 0.0 or not math.isfinite(total):
        return [1.0 / len(values) for _ in values]
    # 归一化得到概率分布
    return [float(value / total) for value in exp_values]


# 计算候选父代的综合选择效用（selection utility）：
# utility = w_score*归一化分数 + w_delta*归一化提升 + w_novelty*新颖度，
# 再对效用做 softmax 得到采样概率，供 AIRA-Evo/Evo-Max 风格的父代选择使用
def compute_parent_utilities(
    candidates: list[Any],
    *,
    all_programs: Iterable[Any],
    lower_is_better: bool = False,
    temperature: float = 1.0,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    # 合并默认权重与调用方覆盖权重
    weights = {**DEFAULT_PARENT_UTILITY_WEIGHTS, **dict(weights or {})}
    # 全体程序快照
    programs = list(all_programs or [])
    # 建立 id -> program 索引
    by_id = _index(programs)
    # 各候选的适应度分数
    scores = [program_selection_score(program) for program in candidates]
    # 分数分量：min-max 归一化到 [0,1]
    score_components = _normalize_values(scores, lower_is_better=lower_is_better)
    # 提升量列表初始化
    deltas: list[float | None] = []
    # 方法族列表初始化
    families: list[str] = []
    # 预取全体程序的经验卡（供同族统计）
    previous_cards = [_card(program, programs) for program in programs]
    # 遍历候选，收集方法族并计算相对最优父代的提升量
    for idx, program in enumerate(candidates):
        # 取该候选的经验卡
        card = _card(program, programs)
        # 记录方法族标签
        families.append(str(card.get("method_family_auto") or "unknown"))
        # 收集父代分数
        parent_scores = []
        # 遍历父代 id
        for pid in parent_ids(program):
            # 查表取父代对象
            parent = by_id.get(pid)
            # 父代存在时读取分数
            if parent is not None:
                parent_score = program_selection_score(parent)
                # 分数有效才纳入
                if parent_score is not None:
                    parent_scores.append(parent_score)
        # 有父代分数时按方向取最优父代并计算正向提升
        if parent_scores:
            best_parent = min(parent_scores) if lower_is_better else max(parent_scores)
            deltas.append(_positive_delta(scores[idx], best_parent, lower_is_better))
        else:
            # 无父代信息（如 Draft 根节点）时提升量按 0 处理
            deltas.append(0.0)
    # 提升量按最大值归一化为 delta 分量
    delta_components = _normalize_positive(deltas)
    # 效用明细列表
    utilities: list[dict[str, Any]] = []
    # 遍历候选计算各自效用
    for idx, program in enumerate(candidates):
        # 程序 id
        pid = _program_id(program)
        # 方法族标签
        family = families[idx]
        # 统计此前同族数量，换算新颖度（走过的次数越少越新颖）
        family_count = _family_count_before(family, previous_cards, node_id=str(pid or ""))
        novelty = 1.0 / math.sqrt(1.0 + family_count)
        # 加权合成效用（多行表达式，三个分量相加）
        utility = (
            float(weights["score"]) * score_components[idx]
            + float(weights["delta"]) * delta_components[idx]
            + float(weights["novelty"]) * novelty
        )
        # 记录该候选的效用明细
        utilities.append(
            {
                "node_id": pid,
                "method_family_auto": family,
                "fitness": scores[idx],
                "score_source": "rl_test_reward",
                "score_component": score_components[idx],
                "delta_component": delta_components[idx],
                "novelty_component": novelty,
                "utility": utility,
                "probability": 0.0,
            }
        )
    # 对全部效用做 softmax 得到采样概率（温度控制探索/利用）
    probabilities = softmax([item["utility"] for item in utilities], temperature=temperature)
    # 把概率写回各候选明细
    for item, probability in zip(utilities, probabilities):
        item["probability"] = probability
    # 返回效用列表
    return utilities


# 按适应度对程序排序（lower_is_better 时升序，否则降序）；
# 无分数的程序排在"最差"一端，并附加 id 作为稳定排序的次键
def _sort_programs(programs: Iterable[Any], lower_is_better: bool = False) -> list[Any]:
    # 定义排序键：(分数, id) 二元组
    def key(program: Any) -> tuple[float, int]:
        # 读取适应度分数
        score = program_selection_score(program)
        # 无分数时用正/负无穷占位排到最差一端
        score_value = score if score is not None else (float("inf") if lower_is_better else float("-inf"))
        # id 作为次键保证排序稳定
        pid = _program_id(program) or 0
        # 返回二元排序键
        return (score_value, pid)

    # 执行排序：lower_is_better 升序，否则降序
    return sorted(programs, key=key, reverse=not lower_is_better)


# 沿主 parent_id 链向上回溯最近 max_items 个祖先（只走第一父代即 Improve 主线），
# 用于 Improve/Crossover 记忆中的"纵向进化路径"；带已访问集合防环
def _recent_ancestors(program: Any, all_programs: list[Any], max_items: int) -> list[Any]:
    # 建立 id -> program 索引
    by_id = _index(all_programs)
    # 祖先结果列表
    result: list[Any] = []
    # 当前游标从 program 自身开始
    current = program
    # 已访问 id 集合（防死循环）
    seen: set[int] = set()
    # 未达上限且游标有效时继续回溯
    while current is not None and len(result) < max_items:
        # 取当前节点的父代 id 列表
        pids = parent_ids(current)
        # 无父代（根节点）时停止
        if not pids:
            break
        # 查表取第一父代（主进化线）
        parent = by_id.get(pids[0])
        # 父代不在索引中时停止
        if parent is None:
            break
        # 读取父代 id
        parent_pid = _program_id(parent)
        # id 缺失或已访问过（成环）时停止
        if parent_pid is None or parent_pid in seen:
            break
        # 记入已访问集合
        seen.add(parent_pid)
        # 加入祖先结果
        result.append(parent)
        # 游标上移到父代
        current = parent
    # 返回祖先链
    return result


# 找出父代为给定程序的所有子代（同级兄弟，排除自身），
# 用于 Improve 记忆中的"横向兄弟尝试"
def _children_of(parent: Any, all_programs: list[Any]) -> list[Any]:
    # 读取父代 id
    pid = _program_id(parent)
    # 无 id 时不存在子代
    if pid is None:
        return []
    # 过滤出以 pid 为父代且不是自身的程序
    return [program for program in all_programs if pid in parent_ids(program) and _program_id(program) != pid]


# 构建策略板（strategy board）摘要：当前最优节点/分数/方向、各方法族计数、
# 探索不足的方向与重复出现的错误签名，供各算子的记忆提示词引用
def build_strategy_board(programs: list[Any], lower_is_better: bool = False) -> dict[str, Any]:
    # 取全部程序的经验卡
    cards = [_card(program, programs) for program in programs]
    # 过滤出有有效适应度且非 bug 的成功卡
    valid_cards = [card for card in cards if finite_float(card.get("fitness")) is not None and not card.get("is_buggy")]
    # 按适应度排序（默认从优到差）
    valid_cards = sorted(valid_cards, key=lambda card: finite_float(card.get("fitness")) or 0.0, reverse=not lower_is_better)
    # 排名第一的卡即当前最优
    best_card = valid_cards[0] if valid_cards else None
    # 方法族计数表
    family_counts: dict[str, int] = {}
    # 错误签名计数表
    repeated_errors: dict[str, int] = {}
    # 遍历全部卡做统计
    for card in cards:
        # 读取方法族标签
        family = str(card.get("method_family_auto") or "unknown")
        # 累计族计数
        family_counts[family] = family_counts.get(family, 0) + 1
        # 读取错误签名
        signature = str(card.get("error_signature") or "")
        # 有签名时累计重复错误计数
        if signature:
            repeated_errors[signature] = repeated_errors.get(signature, 0) + 1
    # 取出现次数最少的 3 个族作为探索不足方向
    underexplored = [family for family, count in sorted(family_counts.items(), key=lambda item: item[1])[:3]]
    # 汇总策略板字典返回
    return {
        "best_node": best_card.get("node_id") if best_card else None,
        "best_score": best_card.get("fitness") if best_card else None,
        "current_best_family": best_card.get("method_family_auto") if best_card else None,
        "family_counts": family_counts,
        "underexplored_families": underexplored,
        "repeated_errors": repeated_errors,
    }


# 把数值格式化为 6 位有效数字的字符串；None 显示 n/a，其余直接转字符串
def _format_value(value: Any) -> str:
    # 尝试按数值处理
    numeric = finite_float(value)
    # 数值有效时按 6 位有效数字输出
    if numeric is not None:
        return f"{numeric:.6g}"
    # None 显示为 n/a
    if value is None:
        return "n/a"
    # 其他类型直接转字符串
    return str(value)


# 从经验卡的富摘要生成记忆行：优先输出方法概述与父代对比经验，
# 二者皆缺时回退到旧版 analysis 字段
def _rich_summary_lines(prefix: str, card: dict[str, Any]) -> list[str]:
    # 读取富摘要字段
    rich_summary = card.get("rich_summary")
    # 确为 dict 时才解析
    if isinstance(rich_summary, dict):
        # 输出行列表
        lines = []
        # 压缩方法概述文本
        method = compact_text(rich_summary.get("method_overview"), 420)
        # 压缩父代对比经验文本
        parent_exp = compact_text(rich_summary.get("parent_comparison_experience"), 420)
        # 有方法概述时追加对应记忆行
        if method:
            lines.append(f"- {prefix}_method_overview: {method}")
        # 有父代对比经验时追加对应记忆行
        if parent_exp:
            lines.append(f"- {prefix}_parent_comparison_experience: {parent_exp}")
        # 已生成富摘要行则直接返回
        if lines:
            return lines
    # 回退：压缩旧版 analysis 字段
    analysis = compact_text(card.get("analysis"), 360)
    # 有内容才生成行，否则返回空列表
    return [f"- {prefix}_legacy_analysis: {analysis}"] if analysis else []


# 把单个经验节点渲染为多行记忆文本：节点 id、方法族、分数、提升、运行时间、
# 是否当前最优、是否新方向，并追加富摘要经验行
def _memory_node_lines(prefix: str, program: Any, all_programs: list[Any]) -> list[str]:
    # 取该程序的经验卡
    card = _card(program, all_programs)
    # 基础记忆行：节点摘要与两个布尔标记
    lines = [
        (
            f"- {prefix}_node_id: {card.get('node_id')} "
            f"(family={card.get('method_family_auto') or 'unknown'}, "
            f"score={_format_value(card.get('fitness'))}, "
            f"delta_vs_parent={_format_value(card.get('delta_vs_parent'))}, "
            f"runtime_seconds={_format_value(card.get('sandbox_time_used'))})"
        ),
        f"- {prefix}_current_best: {bool(card.get('current_best', False))}",
        f"- {prefix}_is_new_direction: {bool(card.get('is_new_direction', False))}",
    ]
    # 追加富摘要经验行
    lines.extend(_rich_summary_lines(prefix, card))
    # 返回该节点的记忆行
    return lines


# 向行列表追加一个带标题的记忆小节：程序列表为空则跳过；
# 多个程序时给每条记忆前缀加序号
def _append_section(lines: list[str], title: str, prefix: str, programs: list[Any], all_programs: list[Any]) -> None:
    # 无程序时不输出该小节
    if not programs:
        return
    # 追加空行与标题行
    lines.extend(["", title + ":"])
    # 逐个渲染程序记忆行（编号从 1 开始）
    for index, program in enumerate(programs, start=1):
        # 单个程序用原前缀，多个程序加序号后缀
        item_prefix = prefix if len(programs) == 1 else f"{prefix}_{index}"
        # 渲染该程序的节点记忆行
        lines.extend(_memory_node_lines(item_prefix, program, all_programs))


# 按算子类型（Draft/Improve/Crossover/Debug 四算子）构建定向经验记忆
# （Targeted Memory Context）：从策略板与谱系中挑选与当前操作相关的历史节点，
# 渲染为可插入提示词的记忆文本，引导 LLM 提出"有意义的差异化方向"而非重复既有解法
def build_operator_experience_memory(
    operator: str,
    parent_programs: list[Any],
    *,
    all_programs: Iterable[Any],
    current_program: Any | None = None,
    lower_is_better: bool = False,
    max_related_cards: int = 3,
    ancestor_k: int | None = None,
    sibling_k: int | None = None,
) -> str:
    # 全体程序快照
    programs = list(all_programs or [])
    # 算子名规范化为小写
    operator = str(operator or "").lower()
    # 构建策略板统计
    board = build_strategy_board(programs, lower_is_better=lower_is_better)
    # 相关卡数量上限（至少为 1）
    cap = max(1, int(max_related_cards or 3))

    # ---- Draft 算子记忆：为"从零起草"提供最优与最近的历史方向，鼓励差异化 ----
    if operator == "draft":
        # 无任何历史时返回空记忆
        if not programs:
            return ""
        # 最近生成的至多 cap 个程序（倒序取，越新越靠前）
        recent = list(reversed(programs))[:cap]
        # 成功程序中按适应度取前 cap 个最优
        top = _sort_programs([program for program in programs if is_success_program(program)], lower_is_better)[:cap]
        # 记忆文本开头：标题、使用说明与策略板统计
        lines = [
            "Targeted Memory Context for DRAFT",
            "",
            "Use this memory as evidence about previously explored ideas. Propose a meaningfully different solution direction.",
            "",
            "Related board stats:",
            f"- current_best_node: {board.get('best_node') or 'n/a'}",
            f"- current_best_score: {_format_value(board.get('best_score'))}",
            f"- current_best_family: {board.get('current_best_family') or 'unknown'}",
            f"- underexplored_families: {', '.join(board.get('underexplored_families') or []) or 'none'}",
        ]
        # 追加"最优历史记忆"小节
        _append_section(lines, "Best previous memories", "best", top, programs)
        # 追加"最近历史记忆"小节
        _append_section(lines, "Recent previous memories", "recent", recent, programs)
        # 拼接为完整记忆文本
        return "\n".join(lines)

    # ---- Improve 算子记忆：围绕选定父代，给纵向祖先链与横向兄弟尝试 ----
    if operator == "improve" and parent_programs:
        # 取第一个父代作为改进对象
        parent = parent_programs[0]
        # 沿主链回溯至多 3（或 ancestor_k）个祖先
        ancestors = _recent_ancestors(parent, programs, 3 if ancestor_k is None else max(0, int(ancestor_k)))
        # 取父代的兄弟（同父其他子代）按适应度取前 3（或 sibling_k）个
        siblings = _sort_programs(_children_of(parent, programs), lower_is_better)[: 3 if sibling_k is None else max(0, int(sibling_k))]
        # 取父代经验卡
        parent_card = _card(parent, programs)
        # 读父代方法族
        parent_family = str(parent_card.get("method_family_auto") or "unknown")
        # 记忆文本开头：标题与使用说明
        lines = [
            "Targeted Memory Context for IMPROVE",
            "",
            "Use this memory as evidence about the selected parent, its recent evolution path, and nearby sibling attempts. Do not treat it as a complete history.",
        ]
        # 追加"选定父代记忆"小节
        _append_section(lines, "Selected parent memory", "parent", [parent], programs)
        # 追加"纵向祖先记忆（近期进化路径）"小节
        _append_section(lines, "Vertical ancestor memory (recent evolution path)", "ancestor", ancestors, programs)
        # 追加"横向兄弟记忆（邻近替代尝试）"小节
        _append_section(lines, "Horizontal sibling memory (nearby alternative attempts)", "sibling", siblings, programs)
        # 追加策略板统计与父代方向信息
        lines.extend([
            "",
            "Related board stats:",
            f"- current_best_node: {board.get('best_node') or 'n/a'}",
            f"- current_best_score: {_format_value(board.get('best_score'))}",
            f"- current_best_family: {board.get('current_best_family') or 'unknown'}",
            f"- parent_family: {parent_family}",
            f"- underexplored_families: {', '.join(board.get('underexplored_families') or []) or 'none'}",
        ])
        # 拼接为完整记忆文本
        return "\n".join(lines)

    # ---- Crossover 算子记忆：对比两个父代分支的强弱与冲突，提示互补融合而非机械拼接 ----
    if operator == "crossover" and len(parent_programs) >= 2:
        # 取前两个父代
        p1, p2 = parent_programs[:2]
        # 父代 1 经验卡
        c1 = _card(p1, programs)
        # 父代 2 经验卡
        c2 = _card(p2, programs)
        # 父代 1 方法族
        family_a = str(c1.get("method_family_auto") or "unknown")
        # 父代 2 方法族
        family_b = str(c2.get("method_family_auto") or "unknown")
        # 记忆文本开头：标题与使用说明
        lines = [
            "Targeted Memory Context for CROSSOVER",
            "",
            "Use this memory to identify compatible strengths and conflicts between the two selected parent branches. Do not mechanically merge all code.",
            "",
            "Selected parent memories:",
        ]
        # 追加父代 1 的节点记忆行
        lines.extend(_memory_node_lines("parent_1", p1, programs))
        # 追加父代 2 的节点记忆行
        lines.extend(_memory_node_lines("parent_2", p2, programs))
        # 追加父代 1 的纵向祖先记忆小节
        _append_section(lines, "Parent 1 vertical ancestor memory", "parent_1_ancestor", _recent_ancestors(p1, programs, 2), programs)
        # 追加父代 2 的纵向祖先记忆小节
        _append_section(lines, "Parent 2 vertical ancestor memory", "parent_2_ancestor", _recent_ancestors(p2, programs, 2), programs)
        # 追加两父代的互补性提示（方向异同与融合建议）
        lines.extend([
            "",
            "Complementarity:",
            f"- family_complementarity: {'different_method_families' if family_a != family_b else 'same_method_family'}",
            f"- crossover_hint: combine {family_a} with {family_b}",
        ])
        # 拼接为完整记忆文本
        return "\n".join(lines)

    # ---- Debug 算子记忆：围绕当前错误签名，检索同类历史错误与最近失败案例 ----
    if operator == "debug":
        # 调试焦点：优先显式传入的当前程序，否则取第一个父代
        focus = current_program or (parent_programs[0] if parent_programs else None)
        # 提取焦点程序的错误签名
        signature = error_signature(focus) if focus is not None else None
        # 同签名相关失败案例列表
        related = []
        # 其他最近失败案例列表
        recent = []
        # 倒序遍历历史程序（越新越先被看到）
        for candidate in reversed(programs):
            # 跳过焦点程序自身
            if focus is not None and _program_id(candidate) == _program_id(focus):
                continue
            # 错误签名与焦点相同的历史案例归入 related
            if signature and error_signature(candidate) == signature:
                related.append(candidate)
            else:
                # 其余程序作为最近案例归入 recent
                recent.append(candidate)
        # 记忆文本开头：标题与当前错误签名
        lines = [
            "Targeted Memory Context for DEBUG",
            "",
            "Current error:",
            f"- current_error_signature: {signature or 'unknown'}",
        ]
        # 有焦点程序时输出"当前 bug 节点记忆"小节
        if focus is not None:
            _append_section(lines, "Current buggy node memory", "current_buggy", [focus], programs)
        # 输出相关错误与最近案例记忆小节（相关优先，共至多 cap 个）
        _append_section(lines, "Related previous debug/error memories", "related_error", (related + recent)[:cap], programs)
        # 拼接为完整记忆文本
        return "\n".join(lines)

    # 未知算子或条件不满足时返回空记忆
    return ""


# 从 LLM 输出文本中解析 JSON 对象：依次尝试原文、```json 代码块、首个花括号片段，
# 全部失败时返回 None
def parse_json_object(text: str) -> dict[str, Any] | None:
    # 规范化输入文本
    text = str(text or "").strip()
    # 空文本直接失败
    if not text:
        return None
    # 候选片段列表，先放原文
    candidates = [text]
    # 尝试提取 ```json ... ``` 代码块中的对象
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    # 命中时把代码块内容插到最优先
    if match:
        candidates.insert(0, match.group(1))
    # 再尝试提取任意 {...} 片段作为兜底候选
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    # 命中时追加到候选末尾
    if match:
        candidates.append(match.group(0))
    # 依次尝试各候选片段
    for candidate in candidates:
        try:
            # 尝试 JSON 解析
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            # 解析失败换下一个候选
            continue
        # 仅接受 dict 类型的解析结果
        if isinstance(parsed, dict):
            return parsed
    # 所有候选都失败时返回 None
    return None
