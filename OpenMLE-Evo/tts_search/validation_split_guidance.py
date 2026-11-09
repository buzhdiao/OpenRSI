# 中文说明:本模块负责"验证集划分指导"(validation split guidance)的注入。
# 在 OpenRSI 自进化系统中,评估数据集往往要求模型自行做验证/测试集划分(防泄漏、防作弊),
# 但很多任务自身并没有明确说明划分规则;本模块从人工整理的 markdown 指导文档中
# 解析出"每个任务的推荐划分提示",再批量注入到评估数据的 prompt 字段中,
# 使 tts_search/评估器在跑任务时能拿到一致的验证集划分指导,从而保证分数可比。
# 位置:OpenMLE-Evo/tts_search 下,属于评估数据预处理工具层(纯文本处理,无网络/训练依赖)。

# 延迟注解:类型标注在运行期不求值,降低导入开销并兼容旧解释器
from __future__ import annotations

# JSON 编解码:读取 .json 评估数据、解析 metadata 字符串
import json
# 正则表达式:解析 markdown 章节标题与"推荐 prompt 指导"代码块
import re
# dataclass 装饰器:声明统计信息结构体,frozen 保证不可变(统计结果不该被误改)
from dataclasses import dataclass
# Path:文件路径与读写
from pathlib import Path
# Any 类型:metadata 可能是 numpy 标量/字符串/字典等多种形态
from typing import Any


# 注入到 prompt 中的标题行,便于事后审计与去重(已注入的 prompt 靠它识别)
DEFAULT_GUIDANCE_HEADING = "Validation Split Guidance:"
# 默认插入位置标记:指导文本会插在该标记(如 "**FINAL OUTPUT**")之前,保证指导先于输出格式说明出现
DEFAULT_INSERT_MARKER = "**FINAL OUTPUT**"


# 冻结 dataclass:记录一次"打补丁"操作的输入输出路径与统计信息,供脚本打印/日志汇总
@dataclass(frozen=True)
class GuidancePatchStats:
    # 输入评估数据文件路径
    input_path: Path
    # 输出(打过补丁的)评估数据文件路径
    output_path: Path
    # 读入的总行数(样本数)
    total_rows: int
    # 实际被修改(prompt 注入了指导)的行数
    patched_rows: int
    # 没有找到指导文本的任务名集合(转成元组便于冻结存储)
    missing_tasks: tuple[str, ...]
    # 已含指导标题而跳过的行数(幂等:重复运行不会二次注入)
    skipped_existing_rows: int


def load_guidance_map(instructions_path: Path) -> dict[str, str]:
    """从 markdown 指导文档中解析"每个任务的验证集划分指导"映射表。

    文档按 `## N. 任务名` 分节,每节内用 "Recommended prompt instruction:" 后的
    围栏代码块给出推荐 prompt;返回 {任务名: 指导文本} 字典。
    """
    # 以 UTF-8 读入整份 markdown 文档
    text = instructions_path.read_text(encoding="utf-8")
    # 找出所有二级标题 `## 数字. 标题`,用 finditer 拿到每节的起止位置
    sections = list(re.finditer(r"^##\s+\d+\.\s+(.+?)\s*$", text, flags=re.M))
    # 结果映射:任务名 -> 指导文本
    guidance: dict[str, str] = {}
    # 逐节遍历;用下一个标题的位置界定当前节的文本范围
    for idx, match in enumerate(sections):
        # 提取任务名(标题文本),去除首尾空白
        task_name = match.group(1).strip()
        # 当前节正文从标题行末尾开始
        start = match.end()
        # 到下一个标题开头为止;最后一节则到文档末尾
        end = sections[idx + 1].start() if idx + 1 < len(sections) else len(text)
        # 截取本节文本
        section = text[start:end]
        # 在节内查找 "Recommended prompt instruction:" 之后的围栏代码块(``` 或 ```text 开头)
        block = re.search(
            r"Recommended prompt instruction:\s*```(?:text)?\s*(.*?)\s*```",
            section,
            flags=re.S,
        )
        # 找到代码块才收录;没有推荐指导的任务不出现在映射表里
        if block:
            # 代码块内容去除首尾空白后作为该任务的指导文本
            guidance[task_name] = block.group(1).strip()
    # 返回完整映射表
    return guidance


def patch_eval_data_with_guidance(
    *,
    eval_data_path: Path,
    output_path: Path,
    instructions_path: Path,
    input_key: str,
    metadata_key: str,
    insert_marker: str = DEFAULT_INSERT_MARKER,
    heading: str = DEFAULT_GUIDANCE_HEADING,
    strict: bool = False,
) -> GuidancePatchStats:
    """把任务级验证划分指导注入评估数据,并写出新的数据文件。

    支持 .parquet 与 .json 两种评估数据格式;返回统计信息 GuidancePatchStats。
    strict=True 时,只要有任务缺失指导就抛错(用于要求全覆盖的严格流程)。
    """
    # 先解析 markdown 指导文档,得到任务->指导映射
    guidance_map = load_guidance_map(instructions_path)
    # 映射为空说明文档格式不对或没有指导内容,直接报错避免静默产出空补丁
    if not guidance_map:
        raise ValueError(f"No validation split guidance found in {instructions_path}")

    # 按文件扩展名(小写)区分数据格式
    suffix = eval_data_path.suffix.lower()
    # parquet 分支:列式存储,逐行转成 dict 记录列表处理
    if suffix == ".parquet":
        # 延迟导入 pandas:只有真正处理 parquet 时才需要该依赖
        import pandas as pd

        # 读 parquet 并把每行转成 {列名: 值} 的记录字典
        rows = pd.read_parquet(eval_data_path).to_dict(orient="records")
        # 调用核心打补丁逻辑,得到(修改行数, 缺失任务集合, 跳过行数)
        patched_rows, missing_tasks, skipped_existing = _patch_rows(
            rows=rows,
            guidance_map=guidance_map,
            input_key=input_key,
            metadata_key=metadata_key,
            insert_marker=insert_marker,
            heading=heading,
            strict=strict,
        )
        # 确保输出目录存在(多级父目录逐个创建,已存在不报错)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # 把修改后的记录写回 parquet(不保留原索引列)
        pd.DataFrame(rows).to_parquet(output_path, index=False)
    # json 分支:整份读入为列表
    elif suffix == ".json":
        # 读取并反序列化 JSON 文档
        rows = json.loads(eval_data_path.read_text(encoding="utf-8"))
        # 评估数据必须是样本列表,否则数据格式约定被破坏,报错终止
        if not isinstance(rows, list):
            raise ValueError(f"Expected JSON eval data to be a list: {eval_data_path}")
        # 同样走核心打补丁逻辑
        patched_rows, missing_tasks, skipped_existing = _patch_rows(
            rows=rows,
            guidance_map=guidance_map,
            input_key=input_key,
            metadata_key=metadata_key,
            insert_marker=insert_marker,
            heading=heading,
            strict=strict,
        )
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # 写回 JSON:保留非 ASCII 字符(中文指导文本可读)、缩进 2 便于 diff 审查
        output_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    # 其他格式不支持,显式报错
    else:
        raise ValueError(f"Unsupported eval_data format: {eval_data_path}")

    # 汇总统计信息返回;缺失任务名排序后冻结为元组,保证输出稳定可复现
    return GuidancePatchStats(
        input_path=eval_data_path,
        output_path=output_path,
        total_rows=len(rows),
        patched_rows=patched_rows,
        missing_tasks=tuple(sorted(missing_tasks)),
        skipped_existing_rows=skipped_existing,
    )


def _patch_rows(
    *,
    rows: list[dict[str, Any]],
    guidance_map: dict[str, str],
    input_key: str,
    metadata_key: str,
    insert_marker: str,
    heading: str,
    strict: bool,
) -> tuple[int, set[str], int]:
    """对行列表逐条注入指导的核心实现。

    返回 (patched_rows, missing_tasks, skipped_existing):
    修改的行数、缺失指导的任务名集合、因已注入而跳过的行数。
    """
    # 已修改行数计数器
    patched_rows = 0
    # 已存在指导而跳过的行数计数器
    skipped_existing = 0
    # 缺失指导的任务名集合(用于 strict 检查与统计)
    missing_tasks: set[str] = set()

    # 逐行处理(原地修改 rows 中的字典)
    for row in rows:
        # 优先从 metadata 字段解析任务名
        task_name = _task_name_from_metadata(row.get(metadata_key))
        # metadata 里解析不到时,退回从行内 task_name/name 列取
        if not task_name:
            # 多级回退:先 task_name 列,再 name 列,均无则空串
            task_name = str(row.get("task_name") or row.get("name") or "").strip()
        # 查该任务的指导文本;没有则记入缺失集合并跳过
        guidance = guidance_map.get(task_name)
        if not guidance:
            # 未知任务统一记作 "<unknown>" 占位,便于排查
            missing_tasks.add(task_name or "<unknown>")
            # 缺指导的行不做任何修改
            continue
        # 对该行的 prompt 字段做注入,返回(新prompt, 是否修改, 是否因已存在跳过)
        patched_prompt, changed, skipped = _inject_guidance(
            row.get(input_key),
            guidance=guidance,
            insert_marker=insert_marker,
            heading=heading,
        )
        # 确实修改了才写回并计数
        if changed:
            # 原地更新该行的 prompt 字段
            row[input_key] = patched_prompt
            # 修改行数 +1
            patched_rows += 1
        # 因已含指导而跳过的行单独计数(区别于"修改成功")
        if skipped:
            skipped_existing += 1

    # strict 模式下存在缺失指导即报错,保证任务全覆盖后才放行
    if strict and missing_tasks:
        # 拼接排序后的缺失任务名,便于一眼看出漏了哪些
        missing = ", ".join(sorted(missing_tasks))
        raise ValueError(f"Missing validation split guidance for tasks: {missing}")
    # 返回三项统计
    return patched_rows, missing_tasks, skipped_existing


def _task_name_from_metadata(metadata: Any) -> str:
    """从 metadata 字段尽力解析出任务名;失败返回空串。

    metadata 可能是 numpy 标量、JSON 字符串或直接就是字典,需逐层归一化。
    """
    # 情况一:带 .item() 的 numpy 标量(如 parquet 里的 np.str_),先转成 Python 对象
    if hasattr(metadata, "item"):
        try:
            # .item() 取出原生 Python 值
            metadata = metadata.item()
        except Exception:
            # 转换失败就保持原样,继续往下尝试
            pass
    # 情况二:字符串形态的 metadata,尝试按 JSON 解析成字典
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            # 不是合法 JSON:无法提取任务名,返回空串让调用方走回退逻辑
            return ""
    # 解析后仍不是字典(数字、列表等):不认识,返回空串
    if not isinstance(metadata, dict):
        return ""
    # 按优先级尝试常见任务名字段:task_name 最规范,competition_* 次之,name 兜底
    for key in ("task_name", "competition_id", "competition", "name"):
        # 取出该字段值
        value = metadata.get(key)
        # 第一个非空值即为任务名(去空白后返回)
        if value:
            return str(value).strip()
    # 所有字段都缺失:返回空串
    return ""


def _inject_guidance(
    prompt: Any,
    *,
    guidance: str,
    insert_marker: str,
    heading: str,
) -> tuple[Any, bool, bool]:
    """把指导文本注入一条样本的 prompt 字段,兼容多种 prompt 形态。

    返回 (新prompt, changed, skipped):prompt 可能是纯文本,
    也可能是 messages 列表(numpy 数组/元组亦可能),只注入最后一条 user 消息。
    """
    # 情况一:prompt 本身就是纯文本字符串,直接做文本注入
    if isinstance(prompt, str):
        return _inject_into_text(prompt, guidance, insert_marker, heading)

    # 非 str:先处理 numpy 数组形态,用 tolist() 转成 Python 列表
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()

    # 元组形态转列表(后续要按下标替换消息,列表才可变)
    if isinstance(prompt, tuple):
        prompt = list(prompt)

    # 情况二:messages 列表形态(OpenAI 风格对话格式)
    if isinstance(prompt, list):
        # 从后往前找第一条 user 消息:系统消息在前、注入目标是离用户最近的那条
        for idx in range(len(prompt) - 1, -1, -1):
            # 取出当前消息
            message = prompt[idx]
            # 只处理 dict 形态且 role 为 user 的消息
            if isinstance(message, dict) and message.get("role") == "user":
                # 对该消息的 content 做文本注入(content 兜底为空串防 None)
                patched, changed, skipped = _inject_into_text(
                    str(message.get("content") or ""),
                    guidance,
                    insert_marker,
                    heading,
                )
                # 注入成功时,用浅拷贝新 dict 替换原消息(不改动列表里共享的旧对象)
                if changed:
                    # 复制原消息字典
                    new_message = dict(message)
                    # 替换 content 为注入后的文本
                    new_message["content"] = patched
                    # 写回列表对应位置
                    prompt[idx] = new_message
                # 找到第一条 user 消息处理后立即返回
                return prompt, changed, skipped
        # 整个列表里没有 user 消息:原样返回且标记未修改
        return prompt, False, False

    # 其他不认识的 prompt 类型:原样返回,不注入也不报错
    return prompt, False, False


def _inject_into_text(
    text: str,
    guidance: str,
    insert_marker: str,
    heading: str,
) -> tuple[str, bool, bool]:
    """把指导块插入一段 prompt 文本。

    幂等设计:检测到标题已存在则跳过(skipped=True)。
    有插入标记时插到标记之前,否则追加到文末。返回 (新文本, changed, skipped)。
    """
    # 已包含同款标题说明此前注入过,跳过以保证幂等(重复运行不叠加)
    if heading in text:
        return text, False, True

    # 构造要插入的指导块:前后各留一个空行,标题在前、指导文本在后
    block = f"\n\n{heading}\n{guidance.strip()}\n\n"
    # 查找插入标记的位置;无标记或标记为空则记 -1
    marker_index = text.find(insert_marker) if insert_marker else -1
    # 找到标记:指导块插到标记之前,且去掉标记前文本的尾部空白保持紧凑
    if marker_index >= 0:
        patched = f"{text[:marker_index].rstrip()}{block}{text[marker_index:]}"
    else:
        # 没有标记:追加到文本末尾(同样先去掉尾部空白)
        patched = f"{text.rstrip()}{block}"
    # 返回注入后的文本,标记 changed=True、skipped=False
    return patched, True, False
