# 评测配置模块:解析 --eval-config(YAML/JSON)或 --eval-prompt-data,把它们
# 规整成 EvalDatasetConfig 列表,并实现 dataset/default/args 三级字段覆盖优先级。
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

# 哨兵对象:用于区分"字段缺失"与"字段值为 None"
_MISSING = object()

# TODO: This is ugly, temporarily leave this. We should unify all the config name for dataset, default, and args. (advice from Tom.)
# 运行时采样参数的名称映射表:定义每个参数在 dataset 键、default 键、args 属性三处的可能命名
DATASET_RUNTIME_SPECS: dict[str, dict[str, tuple[str, ...]]] = {
    # 每个评测 prompt 采样的回复数
    "n_samples_per_eval_prompt": {
        "dataset_keys": ("n_samples_per_eval_prompt",),
        "default_keys": ("n_samples_per_eval_prompt",),
        "arg_attrs": ("n_samples_per_eval_prompt", "n_samples_per_prompt"),
    },
    # 采样温度
    "temperature": {
        "dataset_keys": ("temperature",),
        "default_keys": ("temperature",),
        "arg_attrs": ("eval_temperature", "rollout_temperature"),
    },
    # 核采样概率阈值 top-p
    "top_p": {
        "dataset_keys": ("top_p",),
        "default_keys": ("top_p",),
        "arg_attrs": ("eval_top_p", "rollout_top_p"),
    },
    # top-k 采样
    "top_k": {
        "dataset_keys": ("top_k",),
        "default_keys": ("top_k",),
        "arg_attrs": ("eval_top_k", "rollout_top_k"),
    },
    # 最大回复长度
    "max_response_len": {
        "dataset_keys": ("max_response_len",),
        "default_keys": ("max_response_len",),
        "arg_attrs": ("eval_max_response_len", "rollout_max_response_len"),
    },
}

# 数据集字段(样本级)的名称映射表,结构与上面一致
DATASET_SAMPLE_SPECS: dict[str, dict[str, tuple[str, ...]]] = {
    # 输入(prompt)字段名
    "input_key": {
        "dataset_keys": ("input_key",),
        "default_keys": ("input_key",),
        "arg_attrs": ("eval_input_key", "input_key"),
    },
    # 标签字段名
    "label_key": {
        "dataset_keys": ("label_key",),
        "default_keys": ("label_key",),
        "arg_attrs": ("eval_label_key", "label_key"),
    },
    # 工具字段名
    "tool_key": {
        "dataset_keys": ("tool_key",),
        "default_keys": ("tool_key",),
        "arg_attrs": ("eval_tool_key", "tool_key"),
    },
    # 元数据字段名
    "metadata_key": {
        "dataset_keys": ("metadata_key",),
        "default_keys": ("metadata_key",),
        "arg_attrs": ("metadata_key",),
    },
}


def _first_not_missing(*values: Any) -> Any:
    # 依次检查候选值,返回第一个非缺失(非 _MISSING 哨兵)的值
    for value in values:
        if value is not _MISSING:
            return value
    # 全部缺失时返回哨兵本身
    return _MISSING


def _pick_from_mapping(data: dict[str, Any], key_names: tuple[str, ...] | None) -> Any:
    if key_names is None:
        # 未提供候选键名则视为缺失
        return _MISSING
    for key_name in key_names:
        if key_name in data:
            # 返回映射中第一个存在的键对应的值
            return data[key_name]
    # 所有候选键都不存在时返回哨兵
    return _MISSING


def pick_from_args(args: Any, attrs: tuple[str, ...]) -> Any:
    # 依次尝试 args 上的多个属性名,返回第一个非 None 的值
    for attr in attrs:
        value = getattr(args, attr, None)
        if value is not None:
            return value
    # 全为 None 时返回 None
    return None


def _ensure_metadata_overrides(value: Any) -> dict[str, Any]:
    if value is None:
        # None 视为空覆盖表
        return {}
    if not isinstance(value, dict):
        # 非映射类型的覆盖表属于配置错误
        raise TypeError("metadata_overrides must be a mapping.")
    # 合法映射原样返回
    return value


@dataclass
class EvalDatasetConfig:
    """Configuration for a single evaluation dataset."""

    # 数据集名称(唯一标识)
    name: str
    # 数据文件路径
    path: str
    # 奖励模型类型(None 表示沿用全局设置)
    rm_type: str | None = None

    # Dataset-specific overrides
    # 以下为该数据集专属的字段覆盖
    input_key: str | None = None
    label_key: str | None = None
    tool_key: str | None = None
    metadata_key: str | None = None

    # 每个评测 prompt 的采样数
    n_samples_per_eval_prompt: int | None = None

    # 采样超参:温度 / top-p / top-k / 最大回复长度
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_response_len: int | None = None
    # 停止字符串列表
    stop: list[str] | None = None
    # 停止 token id 列表
    stop_token_ids: list[int] | None = None
    # 最小生成 token 数
    min_new_tokens: int | None = None

    # per-dataset custom generate function (e.g., for tool calling)
    # 每数据集自定义生成函数路径(如工具调用场景)
    custom_generate_function_path: str | None = None

    # 元数据覆盖表(默认空 dict,需用工厂函数避免共享可变默认值)
    metadata_overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 数据类初始化后校验并规整 metadata_overrides
        self.metadata_overrides = _ensure_metadata_overrides(self.metadata_overrides)

    @property
    def cache_key(self) -> tuple[Any, ...]:
        """Return a tuple uniquely identifying dataset config for caching."""
        # 用名称、路径与各字段 key 组成元组,作为缓存该数据集的唯一键
        return (
            self.name,
            self.path,
            self.input_key,
            self.label_key,
            self.tool_key,
            self.metadata_key,
        )

    def inject_metadata(self, sample_metadata: Any) -> dict[str, Any]:
        """Return updated metadata merging overrides."""
        if not isinstance(sample_metadata, dict):
            # 原始元数据不是 dict 时从空表开始
            metadata = {}
        else:
            # 拷贝一份避免污染原数据
            metadata = dict(sample_metadata)

        if self.rm_type is not None:
            # 数据集级 rm_type 覆盖写入元数据
            metadata["rm_type"] = self.rm_type

        for key, value in self.metadata_overrides.items():
            # 逐项应用用户自定义的元数据覆盖
            metadata[key] = value

        # 返回合并后的元数据
        return metadata


def ensure_dataset_list(config: Any) -> list[dict[str, Any]]:
    """
    Normalize OmegaConf containers into a list of dicts.
    Accepts either a list or dictionary keyed by dataset name.
    """
    if config is None:
        # 未配置任何数据集时返回空列表
        return []

    if isinstance(config, dict):
        # 字典形式:键为数据集名,值为配置项
        datasets = []
        for name, cfg in config.items():
            # 拷贝成普通 dict(None 也归一为空表)
            dataset = dict(cfg or {})
            # 用字典键补齐缺失的 name 字段
            dataset.setdefault("name", name)
            datasets.append(dataset)
        return datasets

    if isinstance(config, (list, tuple)):
        # 列表/元组形式:每项自带 name 字段
        datasets = []
        for item in config:
            dataset = dict(item or {})
            if "name" not in dataset:
                # 列表形式必须显式给出 name
                raise ValueError("Each evaluation dataset entry must include a `name` field.")
            datasets.append(dataset)
        return datasets

    # 其余类型不支持
    raise TypeError("eval.datasets must be either a list or a mapping.")


def _apply_dataset_field_overrides(
    args: Any, dataset_cfg: dict[str, Any], defaults: dict[str, Any], spec_names: dict[str, Any]
) -> None:
    # 遍历每种参数的名称映射规格
    for field_name, spec in spec_names.items():
        # 第一优先级:数据集自身配置中的值
        dataset_value = _pick_from_mapping(dataset_cfg, spec["dataset_keys"])
        # 第二优先级:全局 defaults 中的值
        default_value = _pick_from_mapping(defaults, spec["default_keys"])
        # 取两者中第一个非缺失者
        resolved_value = _first_not_missing(dataset_value, default_value)
        if resolved_value is not _MISSING:
            # 命中前两级则写回数据集配置
            dataset_cfg[field_name] = resolved_value
            continue
        # 第三优先级:从命令行参数(args)上按候选属性名取值
        dataset_cfg[field_name] = pick_from_args(args, spec["arg_attrs"])


def build_eval_dataset_configs(
    args: Any,
    raw_config: Iterable[dict[str, Any]],
    defaults: dict[str, Any],
) -> list[EvalDatasetConfig]:
    # defaults 为空时归一为空 dict
    defaults = defaults or {}
    # 结果列表
    datasets: list[EvalDatasetConfig] = []
    for cfg in raw_config:
        # 拷贝每条原始配置为普通 dict
        cfg_dict = dict(cfg or {})
        # 合并运行时参数与样本字段两套规格
        combined_specs = {**DATASET_RUNTIME_SPECS, **DATASET_SAMPLE_SPECS}
        # 按三级优先级把字段解析进配置
        _apply_dataset_field_overrides(args, cfg_dict, defaults, combined_specs)
        # 构造强类型的 EvalDatasetConfig 实例
        dataset = EvalDatasetConfig(**cfg_dict)
        datasets.append(dataset)
    # 返回全部评测数据集配置
    return datasets
