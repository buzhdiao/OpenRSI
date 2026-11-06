import dataclasses

from slime.utils import megatron_bridge_utils
from slime.utils.misc import chunk_named_params_by_size

from ..megatron_to_hf import postprocess_hf_param
from ..megatron_to_hf.processors import quantize_params
from ..misc_utils import strip_param_name_prefix
from .hf_weight_iterator_base import HfWeightIteratorBase


# 基于 megatron-bridge 的 HF 权重迭代器实现
class HfWeightIteratorBridge(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 从 HF checkpoint 建立桥接器（读取模型配置以生成转换任务）
        from megatron.bridge import AutoBridge

        # 导入插件侧注册的 megatron-bridge 扩展（noqa：仅触发注册副作用）
        import slime_plugins.megatron_bridge  # noqa: F401

        self._bridge = AutoBridge.from_hf_pretrained(self.args.hf_checkpoint, trust_remote_code=True)

    def get_hf_weight_chunks(self, megatron_local_weights):
        # TODO support quantization (e.g. modify megatron-bridge to provide megatron param name)
        # 去掉参数名的 module. 包装前缀，供 bridge 使用
        renamed_megatron_local_weights = {strip_param_name_prefix(k): v for k, v in megatron_local_weights.items()}
        with megatron_bridge_utils.patch_megatron_model(self.model):
            # 获取 bridge 的权重转换任务清单
            conversion_tasks = self._bridge.get_conversion_tasks(self.model)
            # 用最新本地权重替换任务中的占位权重
            conversion_tasks = _process_conversion_tasks(conversion_tasks, renamed_megatron_local_weights)

            # 执行导出：得到 (HF 参数名, 权重, Megatron 参数名) 三元组序列
            named_weights = self._bridge.export_hf_weights(self.model, cpu=False, conversion_tasks=conversion_tasks)

            # 逐参数做后处理（去 padding）与量化
            quantized_results = []

            for hf_param_name, weight, megatron_param_name in named_weights:

                # 后处理：当前为词表 padding 裁剪
                processed_weight = postprocess_hf_param(
                    args=self.args,
                    megatron_param_name=megatron_param_name,
                    hf_param_name=hf_param_name,
                    param=weight,
                )

                converted_named_params = [(hf_param_name, processed_weight)]

                # 若配置了量化方案则量化（否则原样返回）
                quantized_batch = quantize_params(
                    args=self.args,
                    megatron_name=megatron_param_name,
                    converted_named_params=converted_named_params,
                    quantization_config=self.quantization_config,
                )

                quantized_results.extend(quantized_batch)

            # 用量化后的结果替换导出结果
            named_weights = quantized_results

            # 按缓冲区大小分块产出，便于流式传输权重
            yield from chunk_named_params_by_size(quantized_results, chunk_size=self.args.update_weight_buffer_size)


def _process_conversion_tasks(vanilla_conversion_tasks, new_weight_dict):
    # 把 bridge 转换任务中预取的旧权重替换为本次同步的最新 Megatron 权重
    def _handle_one(task):
        # 无权重字段的任务原样返回
        if task.param_weight is None:
            return task

        # 按桥接器的命名约定拼出权重字典键
        weight_dict_key = f"vp_stages.{task.vp_stage}.{task.param_name}"
        assert (
            weight_dict_key in new_weight_dict
        ), f"{weight_dict_key=} not in new_weight_dict ({task.vp_stage=}, {task.param_name=}, {list(new_weight_dict)=})"

        # 取出最新权重并搬到 GPU
        new_param_weight = new_weight_dict[weight_dict_key]
        new_param_weight = new_param_weight.cuda()
        # 用 dataclasses.replace 生成带新权重的任务副本
        return dataclasses.replace(task, param_weight=new_param_weight)

    # 返回惰性映射（保持长度信息供 bridge 使用）
    return _MapWithLen(_handle_one, vanilla_conversion_tasks)


# 惰性 map 包装器：保留 len() 语义，迭代时逐元素应用函数
class _MapWithLen:
    def __init__(self, fn, xs):
        self.fn = fn
        self.xs = xs

    def __len__(self):
        # 转发长度查询
        return len(self.xs)

    def __iter__(self):
        # 迭代时惰性应用处理函数
        for x in self.xs:
            yield self.fn(x)
