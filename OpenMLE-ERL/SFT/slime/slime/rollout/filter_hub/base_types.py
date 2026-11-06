# 引入 defaultdict，用于按丢弃原因累计计数
from collections import defaultdict
# 引入 dataclass 装饰器，简化只含字段的输出类定义
from dataclasses import dataclass


# 动态过滤器的统一输出数据结构
@dataclass
class DynamicFilterOutput:
    # 是否保留该样本组
    keep: bool
    # 丢弃原因（保留时为 None）
    reason: str | None = None


# 调用动态过滤器的统一入口，兼容新旧两种返回格式
def call_dynamic_filter(fn, *args, **kwargs):
    # 未配置过滤器时视为全部保留
    if fn is None:
        return DynamicFilterOutput(keep=True)

    # 实际调用过滤器函数
    output = fn(*args, **kwargs)

    # compatibility for legacy version
    # 兼容旧版：旧过滤器直接返回 bool，需包装成 DynamicFilterOutput
    if not isinstance(output, DynamicFilterOutput):
        output = DynamicFilterOutput(keep=output)

    # 返回统一格式的过滤结果
    return output


# 指标收集器：统计动态过滤器各丢弃原因的次数分布
class MetricGatherer:
    # 初始化丢弃原因计数器（默认计数为 0）
    def __init__(self):
        self._dynamic_filter_drop_reason_count = defaultdict(lambda: 0)

    # 记录一次丢弃事件
    def on_dynamic_filter_drop(self, reason: str | None):
        # 原因为空则不统计
        if not reason:
            return
        # 对应原因计数加一
        self._dynamic_filter_drop_reason_count[reason] += 1

    # 汇总输出训练指标字典
    def collect(self):
        # 以 drop_{reason} 为键、次数为值生成指标项
        return {
            f"rollout/dynamic_filter/drop_{reason}": count
            for reason, count in self._dynamic_filter_drop_reason_count.items()
        }
