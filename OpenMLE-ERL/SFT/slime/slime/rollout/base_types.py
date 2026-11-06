# 从 dataclasses 导入 dataclass 装饰器，声明输出数据类
from dataclasses import dataclass
# 导入 Any，用于指标字典的值类型标注
from typing import Any

# 导入 Sample 类型：rollout 样本的标准数据结构
from slime.utils.types import Sample


# 训练用 rollout 函数的输出：嵌套的样本列表（外层按数据集/组划分）与可选指标
@dataclass
class RolloutFnTrainOutput:
    # 样本列表的列表（每个内层列表对应一个提示的一组响应）
    samples: list[list[Sample]]
    # 附加指标（可选，默认 None）
    metrics: dict[str, Any] = None


# 评估用 rollout 函数的输出：按数据集名分组的样本与可选指标
@dataclass
class RolloutFnEvalOutput:
    # 数据字典：{数据集名: {samples: [...], ...}}
    data: dict[str, dict[str, Any]]
    # 附加指标（可选，默认 None）
    metrics: dict[str, Any] = None


# 调用 rollout 函数的统一包装：注入 evaluation 开关并兼容旧版返回格式
def call_rollout_fn(fn, *args, evaluation: bool, **kwargs):
    # 调用实际的 rollout 函数（evaluation 作为关键字参数注入）
    output = fn(*args, **kwargs, evaluation=evaluation)

    # compatibility for legacy version
    # （原注释含义：兼容旧版返回格式）
    # 旧版函数直接返回列表/字典而非 Output 对象，这里自动包装
    if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
        # 评估场景包装为 EvalOutput，训练场景包装为 TrainOutput
        output = RolloutFnEvalOutput(data=output) if evaluation else RolloutFnTrainOutput(samples=output)

    # 返回统一格式的输出对象
    return output
