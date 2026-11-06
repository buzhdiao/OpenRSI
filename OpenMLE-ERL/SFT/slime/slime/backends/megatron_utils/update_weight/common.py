import inspect
import re
from argparse import Namespace
from collections.abc import Iterator, Sequence

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from slime.backends.megatron_utils.misc_utils import strip_param_name_prefix
from slime.utils.types import ParamInfo


def all_gather_param(name: str, param: torch.nn.Parameter) -> torch.Tensor:
    """
    All-gather TP-sharded param to full tensor. expert_bias→param, non-TP/duplicated→param.data.
    Uses expert-TP for ".experts.", else regular-TP. linear_fc1 rechunked (GLU), linear_fc2 dim fix.
    """
    # expert_bias 不是 TP 分片参数，直接返回
    if "expert_bias" in name:
        return param

    # 参数必须带 TP 分片标记属性
    assert hasattr(param, "tensor_model_parallel"), f"{name} does not have tensor_model_parallel attribute"
    if not param.tensor_model_parallel or getattr(param, "parallel_mode", None) == "duplicated":
        # 非 TP 分片或 duplicated 模式的参数无需 gather，直接取本地数据
        return param.data

    if ".experts." in name:
        # MoE 专家参数使用专家 TP 组（EP 感知的张量并行）
        tp_size = mpu.get_expert_tensor_parallel_world_size()
        tp_group = mpu.get_expert_tensor_parallel_group()
    else:
        # 普通参数使用常规 TP 组
        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_group = mpu.get_tensor_model_parallel_group()

    # 为每个 TP 分片分配接收缓冲
    param_partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
    # 在 TP 组上 all-gather 各分片
    dist.all_gather(param_partitions, param.data, group=tp_group)
    # 记录参数被切分的维度
    partition_dim = param.partition_dim
    assert param.partition_stride == 1, "partition_stride != 1 is not supported"
    # TODO: here we did an extra copy during concat, maybe merge this with convert_to_hf is better?
    # TODO: check only GLU is used.
    if "linear_fc1.weight" in name:
        # GLU 融合权重：Megatron 按"每分片内 gate|up 交错"切分，
        # 需要先在每片内拆成两半，再重排为"先所有 gate 半、后所有 up 半"的顺序
        param_partitions = [p.chunk(2, dim=0) for p in param_partitions]
        param_partitions = [p[0] for p in param_partitions] + [p[1] for p in param_partitions]
    # this is bug in megatron's grouped moe.
    # 修正 Megatron 分组 MoE 的切分维度 bug：fc2 实际沿第 1 维切分
    if "linear_fc2.weight" in name:
        if partition_dim == 0:
            partition_dim = 1
    # 按切分维拼接出完整参数
    param = torch.cat(param_partitions, dim=partition_dim)
    return param


def all_gather_params_async(
    param_infos_and_params: list[tuple[ParamInfo, torch.Tensor]],
) -> list[torch.Tensor]:
    """
    Parallel TP all-gather for multiple params. Loop 1: for each TP param, allocate buffers +
    dist.all_gather(async_op=True) on expert-TP/regular-TP group (skip expert_bias/non-TP/duplicated).
    Loop 2: wait all NCCL handles (enables overlap). Loop 3: concat partitions + apply GLU rechunk/MoE dim fix.
    """
    # Phase 1: Start all async all_gather operations
    # 阶段一：把所有参数的 all-gather 请求并行发出
    gather_tasks = []
    handles = []

    for info, param in param_infos_and_params:
        # Prepare async all_gather
        if "expert_bias" in info.name:
            # expert_bias 无需通信，记录占位任务
            gather_tasks.append((info, param, None, None, None))
            handles.append(None)
        elif not param.tensor_model_parallel or getattr(param, "parallel_mode", None) == "duplicated":
            # 非 TP 分片参数直接用本地数据
            gather_tasks.append((info, param.data, None, None, None))
            handles.append(None)
        else:
            # Start async all_gather
            # 按参数类型选择专家 TP 组或常规 TP 组
            if ".experts." in info.name:
                tp_size = mpu.get_expert_tensor_parallel_world_size()
                tp_group = mpu.get_expert_tensor_parallel_group()
            else:
                tp_size = mpu.get_tensor_model_parallel_world_size()
                tp_group = mpu.get_tensor_model_parallel_group()

            # 分配分片接收缓冲并发起异步 all-gather
            param_partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
            handle = dist.all_gather(param_partitions, param.data, group=tp_group, async_op=True)
            gather_tasks.append((info, None, handle, param_partitions, param.partition_dim))
            handles.append(handle)

    # Phase 2: Wait for ALL async operations to complete at once
    # This ensures maximum parallelism by not blocking on individual operations
    # 阶段二：统一等待全部异步通信完成（通信已在链路上重叠执行）
    for handle in handles:
        if handle is not None:
            handle.wait()

    # Phase 3: Process all results after all communications are done
    # 阶段三：通信全部完成后统一处理拼接与重排
    gathered_params = []
    for info, direct_param, handle, param_partitions, partition_dim in gather_tasks:
        if handle is None:
            # No all_gather needed
            # 无需 gather 的参数直接使用
            param = direct_param
        else:
            # Process the gathered partitions (same logic as original all_gather_param)
            # 以下与 all_gather_param 的同步版本逻辑一致
            assert partition_dim is not None, "partition_stride != 1 is not supported"
            # TODO: here we did an extra copy during concat, maybe merge this with convert_to_hf is better?
            # TODO: check only GLU is used.
            if "linear_fc1.weight" in info.name:
                # GLU 融合权重的分片重排（先 gate 半全部、后 up 半全部）
                param_partitions = [p.chunk(2, dim=0) for p in param_partitions]
                param_partitions = [p[0] for p in param_partitions] + [p[1] for p in param_partitions]
            # this is bug in megatron's grouped moe.
            # 修正分组 MoE 的 fc2 切分维度 bug
            if "linear_fc2.weight" in info.name:
                if partition_dim == 0:
                    partition_dim = 1
            # 拼接分片得到完整参数
            param = torch.cat(param_partitions, dim=partition_dim)

        gathered_params.append(param)

    return gathered_params


def named_params_and_buffers(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    convert_to_global_name: bool = True,
    translate_gpu_to_cpu: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    # 按需选择"全局名"或"本地名"两种参数遍历方式
    if convert_to_global_name:
        ans = _named_params_and_buffers_global(args, model)
    else:
        ans = _named_params_and_buffers_vanilla(model)

    if translate_gpu_to_cpu:
        # 开启 GPU->CPU 翻译时，优先取 torch_memory_saver 的 CPU 备份
        ans = ((name, _maybe_get_cpu_backup(tensor)) for name, tensor in ans)

    return ans


def _maybe_get_cpu_backup(x: torch.Tensor):
    # 尝试从 torch_memory_saver 获取该 GPU 张量的 CPU 备份（权重卸载场景）
    from torch_memory_saver import torch_memory_saver

    if (cpu_tensor := torch_memory_saver.get_cpu_backup(x)) is not None:
        return cpu_tensor

    return x


def _named_params_and_buffers_vanilla(model: Sequence[torch.nn.Module]) -> Iterator[tuple[str, torch.Tensor]]:
    # 本地名遍历：直接按 vp 阶段编号生成参数名
    for vp_stage, model_module in enumerate(model):

        def _compute_fqn(name, vp_stage=vp_stage):
            # 生成带 vp 前缀的完整限定名
            return f"vp_stages.{vp_stage}.{strip_param_name_prefix(name)}"

        for name, param in model_module.named_parameters():
            # 产出全部参数
            yield _compute_fqn(name), param

        for name, buffer in model_module.named_buffers():
            # TODO shall we handle (almost) all buffers like Megatron Bridge
            # 目前只关心 expert_bias 缓冲（MoE 路由的动态偏置）
            if "expert_bias" not in name:
                continue
            yield _compute_fqn(name), buffer


def _named_params_and_buffers_global(
    args: Namespace, model: Sequence[torch.nn.Module]
) -> Iterator[tuple[str, torch.Tensor]]:
    """
    Yield (global_name, param/buffer) with consistent names across PP/EP. Adjusts indices for
    virtual PP + EP offsets. Handles decoder.layers, mtp.layers (Multi-Token Prediction), expert_bias.
    """
    # 取专家并行的规模与 rank，用于计算专家索引偏移
    ep_size = mpu.get_expert_model_parallel_world_size()
    ep_rank = mpu.get_expert_model_parallel_rank()
    if args.num_experts:
        # 本 rank 上专家编号的全局偏移
        expert_offset = ep_rank * args.num_experts // ep_size

    # 检查 Megatron 版本的 get_transformer_layer_offset 是否支持 vp_stage 参数
    sig = inspect.signature(get_transformer_layer_offset)
    need_vp_stage = "vp_stage" in sig.parameters

    for vp_stage, model_module in enumerate(model):
        if need_vp_stage:
            # 计算当前 vp 阶段模型分片对应的全局层偏移
            layer_offset = get_transformer_layer_offset(model_module.config, vp_stage)
        else:
            layer_offset = get_transformer_layer_offset(model_module.config)
        for name, param in model_module.named_parameters():
            # for model without ddp wrap
            # 未被 DDP 包裹的模型补上 module. 前缀以统一命名
            if not name.startswith("module.module."):
                name = "module." + name

            # 匹配普通 decoder 层参数
            decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                # MTP (Multi-Token Prediction) layers for speculative decoding
                # 非 decoder 层则尝试匹配 MTP 层（投机解码用）
                mtp_layers_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
                match = re.match(mtp_layers_pattern, name)
                if not match:
                    # 都不匹配的参数原样产出
                    yield name, param
                    continue

                # MTP layer indices start from 0
                # MTP 内嵌层的专家参数需加 EP 偏移
                layer_idx, rest = match.groups()
                expert_pattern = r"transformer_layer.mlp.experts\.(.+)\.weight(\d+)"
                match = re.match(expert_pattern, rest)
                if not match:
                    yield name, param
                    continue

                # 重写专家编号为全局编号后产出
                rest, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield f"module.module.mtp.layers.{layer_idx}.transformer_layer.mlp.experts.{rest}.weight{expert_idx}", param
                continue

            # 普通层：本地层号加上 vp 偏移得到全局层号
            layer_idx, rest = match.groups()
            layer_idx = int(layer_idx) + layer_offset

            # this is hardcoded for te grouped matmul
            # 专家参数：按 TE 分组 matmul 的命名规则重写专家全局编号
            expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
            match = re.match(expert_pattern, rest)
            if match:
                rest, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield f"module.module.decoder.layers.{layer_idx}.mlp.experts.{rest}.weight{expert_idx}", param
            else:
                # 非专家参数仅替换层号为全局编号
                yield f"module.module.decoder.layers.{layer_idx}.{rest}", param

        # treat expert bias as normal parameters
        # 把 expert_bias 缓冲也按参数方式产出（参与权重同步）
        for name, buffer in model_module.named_buffers():
            # TODO shall we handle (almost) all buffers like Megatron Bridge
            if "expert_bias" not in name:
                continue
            # for model without ddp wrap
            if not name.startswith("module.module."):
                name = "module." + name

            decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                yield name, buffer
            else:
                # 层号换算成全局编号后产出
                layer_idx, rest = match.groups()
                layer_idx = int(layer_idx) + layer_offset
                yield f"module.module.decoder.layers.{layer_idx}.{rest}", buffer
