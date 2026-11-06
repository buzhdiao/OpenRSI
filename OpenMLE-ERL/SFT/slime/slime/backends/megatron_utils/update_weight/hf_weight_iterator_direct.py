# 导入 dataclasses，用于在 EP 分支中以 replace 方式修改 ParamInfo 的字段（如 src_rank）
import dataclasses
# 从 argparse 导入 Namespace，用于类型标注（args 配置对象）
from argparse import Namespace
# 导入 Sequence，用于标注序列类型的入参（如模型列表、参数信息列表）
from collections.abc import Sequence

# 导入 PyTorch 主库，用于张量与分布式操作
import torch
# 导入 torch.distributed，提供集合通信原语（broadcast/all_gather 等）
import torch.distributed as dist
# 导入 Megatron 的并行工具 mpu，用于获取 PP/EP/TP 各并行维度的大小与进程组
from megatron.core import mpu
# 导入 tqdm，用于在主 rank 上显示参数同步的进度条
from tqdm import tqdm

# 导入分布式工具：get_gloo_group 返回基于 gloo 后端的 CPU 通信组（用于元信息同步）
from slime.utils.distributed_utils import get_gloo_group
# 导入 ParamInfo 数据类，描述单个参数的元信息（名字/形状/dtype/并行属性/源 rank 等）
from slime.utils.types import ParamInfo

# 导入 Megatron→HF 的权重转换函数，把 Megatron 内部表示的张量转换为 HF 格式的命名张量
from ..megatron_to_hf import convert_to_hf
# 导入 sglang 后端的 monkey_patch_torch_reductions，修补 torch reductions 以支持跨进程序列化广播
from ..sglang import monkey_patch_torch_reductions
# 导入通用工具：异步 all-gather 参数、收集本 rank 的命名参数与 buffer
from .common import all_gather_params_async, named_params_and_buffers
# 导入基类 HfWeightIteratorBase，本类为其"直接转换"实现
from .hf_weight_iterator_base import HfWeightIteratorBase


# 定义"直接式" HF 权重迭代器：不走 buffer 打包，而是逐桶 all-gather 完整参数后即转 HF 并产出
class HfWeightIteratorDirect(HfWeightIteratorBase):
    # 构造函数，接受与基类一致的参数
    def __init__(self, *args, **kwargs):
        # 先调用基类初始化，得到 self.args 与 self.model 等
        super().__init__(*args, **kwargs)
        # 在初始化阶段就把本 rank 的参数元信息按 buffer 大小分桶（分桶只做一次，供反复使用）
        self.megatron_local_param_info_buckets = _get_megatron_local_param_info_buckets(self.args, self.model)

    # 实现 HF 权重的生成器：输入本 rank 的 Megatron 本地权重，逐桶产出 HF 格式的命名张量
    def get_hf_weight_chunks(self, megatron_local_weights):
        # 获取当前全局 rank，用于决定是否显示进度条（仅 rank 0 显示）
        rank = dist.get_rank()

        # 遍历每个参数信息桶（用 tqdm 在 rank 0 上显示"Update weights"进度）
        for megatron_local_param_infos in tqdm(
            self.megatron_local_param_info_buckets, disable=rank != 0, desc="Update weights"
        ):
            # 将该桶的本地分片参数经 PP/EP 广播 + TP all-gather 重构为 Megatron 完整参数
            megatron_full_params = _get_megatron_full_params(megatron_local_param_infos, megatron_local_weights)
            # 把 Megatron 完整参数转换为 HF 命名张量列表
            hf_named_tensors = self._convert_to_hf_named_tensors(megatron_full_params, megatron_local_param_infos)
            # 以生成器方式把当前桶的 HF 权重交给调用方（如写入推理引擎）
            yield hf_named_tensors
            # 释放本桶的完整参数内存，控制峰值显存占用
            del megatron_full_params

    # 把 Megatron 完整参数逐个转换为 HF 格式张量
    def _convert_to_hf_named_tensors(self, megatron_full_params: Sequence[torch.Tensor], param_infos: list[ParamInfo]):
        # 初始化输出列表：存放 (HF 参数名, 张量) 对
        hf_named_tensors = []
        # 逐对遍历参数元信息与其对应的完整参数
        for info, param in zip(param_infos, megatron_full_params, strict=False):
            # 调用 convert_to_hf 将单个 Megatron 参数转换为 0 个或多个 HF 命名张量（如 QKV 需拆分）
            hf_named_tensors.extend(
                convert_to_hf(self.args, self.model_name, info.name, param, self.quantization_config)
            )
        # 返回本桶全部参数对应的 HF 命名张量
        return hf_named_tensors


# 把本 rank 的一组本地参数信息重构为 Megatron 完整参数（跨 PP/EP 广播 + TP all-gather）
def _get_megatron_full_params(
    megatron_local_param_infos: Sequence[ParamInfo],
    megatron_local_weights,
) -> Sequence[torch.Tensor]:
    # 修补 torch 的 reduction 序列化实现，保证跨进程广播对象时兼容（slime 的常用做法）
    monkey_patch_torch_reductions()
    # 获取流水线并行（PP, Pipeline Parallelism）的世界大小
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    # 获取专家并行（EP, Expert Parallelism）的世界大小
    ep_size = mpu.get_expert_model_parallel_world_size()
    # 获取当前全局 rank
    rank = dist.get_rank()
    # init params:
    # （原注释含义：初始化参数列表）
    # 初始化本地参数列表
    params = []
    # 遍历该桶中每个参数的元信息
    for info in megatron_local_param_infos:
        # 如果该参数的源 rank 是当前 rank，则用本地真实权重初始化
        if dist.get_rank() == info.src_rank:
            # 将本地权重搬到当前 CUDA 设备（异步拷贝），包装成不要求梯度的 Parameter 作为广播源
            params.append(
                torch.nn.Parameter(
                    megatron_local_weights[info.name].to(device=torch.cuda.current_device(), non_blocking=True),
                    requires_grad=False,
                )
            )
        else:
            # 否则按元信息声明的形状与 dtype 在当前设备上分配空张量，等待接收广播
            params.append(torch.empty(info.shape, dtype=info.dtype, device=torch.cuda.current_device()))
    # 同步 CUDA 流，确保上面 non_blocking 的 H2D 拷贝完成后才能开始广播
    torch.cuda.synchronize()

    # broadcast params across pp ranks
    # （原注释含义：跨 PP rank 广播参数）
    # 若流水线并行度大于 1，需要在 PP 组内把参数从 src_rank 广播到其他 PP rank
    if pp_size > 1:
        # 收集异步广播句柄，稍后统一等待以重叠通信
        handles = []
        # 成对遍历参数信息与参数张量
        for info, param in zip(megatron_local_param_infos, params, strict=False):
            # 只有当该参数的源 rank 属于本 PP 组时才需要在 PP 维度广播
            if info.src_rank in dist.get_process_group_ranks(mpu.get_pipeline_model_parallel_group()):
                # 发起异步广播：以 info.src_rank 为源、PP 组为通信域
                handles.append(
                    torch.distributed.broadcast(
                        param, src=info.src_rank, group=mpu.get_pipeline_model_parallel_group(), async_op=True
                    )
                )
        # 统一等待所有异步广播完成，确保各 PP rank 上参数一致
        for handle in handles:
            # 等待单个异步广播句柄
            handle.wait()

    # broadcast params across ep ranks
    # （原注释含义：跨 EP rank 广播参数）
    # 若专家并行度大于 1，专家参数还需在 EP 组内广播对齐
    if ep_size > 1:
        # 收集异步广播句柄
        handles = []
        # 成对遍历参数信息与参数张量
        for info, param in zip(megatron_local_param_infos, params, strict=False):
            # 仅处理专家参数（名字中包含 ".experts."）
            if ".experts." in info.name:
                # 确定广播源：若原 src_rank 在本 EP 组内就用它，否则用当前 rank（本组内的权威副本）
                src_rank = (
                    info.src_rank
                    if info.src_rank in dist.get_process_group_ranks(mpu.get_expert_model_parallel_group())
                    else rank
                )
                # 发起 EP 组内的异步广播，保证各 EP rank 上该专家参数一致
                handles.append(
                    torch.distributed.broadcast(
                        param, src=src_rank, group=mpu.get_expert_model_parallel_group(), async_op=True
                    )
                )
        # 统一等待所有 EP 维度的异步广播完成
        for handle in handles:
            # 等待单个异步广播句柄
            handle.wait()

    # Set tp attrs for all params
    # （原注释含义：为所有参数设置 TP 相关属性）
    # 把元信息中记录的 Megatron 并行属性（如 tensor_model_parallel、partition_dim 等）回填到参数对象上
    for info, param in zip(megatron_local_param_infos, params, strict=False):
        # 遍历该参数保存的属性键值对
        for key, value in info.attrs.items():
            # 用 setattr 把属性挂回参数对象，供 all-gather 逻辑识别 TP 分片方式
            setattr(param, key, value)

    # Batch async all_gather for all parameters
    # （原注释含义：批量异步 all-gather 所有参数）
    # 发起批量异步 all-gather，把 TP 分片聚合成完整参数（内部按需分块、重叠通信）
    gathered_params = all_gather_params_async(list(zip(megatron_local_param_infos, params, strict=False)))

    # 返回重构好的 Megatron 完整参数列表
    return gathered_params


# 把本 rank 的参数元信息切分成多个"桶"，每桶重建所需内存不超过 update_weight_buffer_size（考虑 TP 副本）
def _get_megatron_local_param_info_buckets(args: Namespace, model: Sequence[torch.nn.Module]) -> list[list[ParamInfo]]:
    """
    Partition params into buckets ≤ update_weight_buffer_size (with TP replication).
    """
    # 先收集本 rank 全部参数的元信息（已跨 PP/EP 交换并校验一致）
    param_infos = _get_megatron_local_param_infos(args, model)
    # 初始化桶列表：从一个空桶开始
    param_info_buckets = [[]]  # Start with one empty bucket
    # 记录当前桶已累计的字节数
    buffer_size = 0  # Track current bucket size in bytes

    # 逐个参数尝试装入当前桶
    for info in param_infos:
        # Expert params use expert-TP size, others use regular-TP size
        # （原注释含义：专家参数用专家 TP 度）
        # 专家参数使用专家张量并行（Expert-TP）大小，普通参数使用普通 TP 大小
        if ".experts." in info.name:
            # 取专家张量并行的世界大小
            tp_size = mpu.get_expert_tensor_parallel_world_size()
        else:
            # 取普通张量并行的世界大小
            tp_size = mpu.get_tensor_model_parallel_world_size()

        # Full param size = shard size × TP replicas (all-gather will reconstruct full param)
        # 完整参数大小 = 本地分片大小 × TP 数（因为 all-gather 会把分片重构成完整参数）
        param_size = info.size * tp_size

        # If adding this param exceeds limit AND current bucket has params: start new bucket
        # 若加入该参数会超过 buffer 上限且当前桶非空：则新开一个桶（避免空桶死循环）
        if buffer_size + param_size > args.update_weight_buffer_size and len(param_info_buckets[-1]) > 0:
            # 追加新的空桶
            param_info_buckets.append([])
            # 新桶的累计大小清零
            buffer_size = 0

        # Add param to current bucket and update size
        # （原注释含义：加入当前桶并更新大小）
        # 把该参数元信息追加到当前桶
        param_info_buckets[-1].append(info)
        # 累加当前桶的字节数
        buffer_size += param_size

    # 返回分桶结果（列表的列表）
    return param_info_buckets


# 收集本 rank 全部参数的元信息：本地收集 → 跨 PP/EP 交换 → 用最小 src_rank 去重（MTP 虚拟 PP）→ 全局一致性校验
def _get_megatron_local_param_infos(args: Namespace, model: Sequence[torch.nn.Module]) -> list[ParamInfo]:
    """
    Build global param metadata: collect → exchange PP/EP → resolve duplicates (MTP virtual PP)
    by min src_rank → validate. Returns sorted ParamInfo identical across all ranks.
    """
    # 获取流水线并行（PP）世界大小
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    # 获取专家并行（EP）世界大小
    ep_size = mpu.get_expert_model_parallel_world_size()

    # 用字典按参数名保存元信息，便于跨 rank 合并去重
    param_infos = {}
    # 获取当前全局 rank，作为本地参数的 src_rank（权威来源标识）
    rank = dist.get_rank()
    # 遍历本 rank 上的所有命名参数与 buffer（由 common 工具决定哪些纳入同步）
    for name, param in named_params_and_buffers(args, model):
        # 为每个参数构造 ParamInfo 元信息
        param_infos[name] = ParamInfo(
            # 参数名
            name=name,
            # 张量数据类型
            dtype=param.dtype,
            # 本地分片的形状
            shape=param.shape,
            # 记录 Megatron 并行属性，供后续 all-gather 与 HF 转换使用
            attrs={
                # 是否为 TP 并行参数（沿某维度切分）
                "tensor_model_parallel": getattr(param, "tensor_model_parallel", False),
                # TP 切分的维度（-1 表示未切分）
                "partition_dim": getattr(param, "partition_dim", -1),
                # TP 切分的步长（对 ColumnParallelLinear 等有特殊含义）
                "partition_stride": getattr(param, "partition_stride", 1),
                # 参数的并行模式（如 Replica/Column/Row Parallel）
                "parallel_mode": getattr(param, "parallel_mode", None),
            },
            # 本地分片的字节大小（元素数 × 单元素字节数）
            size=param.numel() * param.element_size(),
            # 标记本参数权威副本所在的全局 rank（此处先记为当前 rank）
            src_rank=rank,
        )

    # 若 PP 度大于 1，需要跨流水线阶段交换参数元信息（不同 PP stage 持有不同层）
    if pp_size > 1:
        # 预分配 PP 度大小的对象列表用于 all_gather_object
        param_infos_list = [None] * pp_size
        # 在 PP 组内用 pickle 方式 all-gather 各 rank 的 (rank, 参数元信息字典)
        dist.all_gather_object(
            obj=(rank, param_infos), object_list=param_infos_list, group=mpu.get_pipeline_model_parallel_group()
        )
        # 遍历其他 PP rank 传来的元信息
        for src_rank, infos in param_infos_list:
            # 跳过自己（自己已经在字典里）
            if src_rank == rank:
                # 直接继续处理下一个
                continue
            # 遍历该 PP rank 的每个参数元信息
            for name, info in infos.items():
                # 若参数名冲突（同名参数出现在多个 PP rank），通常由 MTP（Multi-Token Prediction，多词元预测）虚拟 PP 造成
                if name in param_infos:
                    # 断言确实配置了 MTP 层数，说明冲突是 MTP 引起的
                    assert args.mtp_num_layers is not None
                    # 取出已有的旧元信息
                    old_info = param_infos[name]
                    # 保留 src_rank 更小的一方，保证所有 rank 上去重结果一致
                    if old_info.src_rank > src_rank:
                        # 用更小 src_rank 的新信息替换旧信息
                        param_infos[name] = info
                else:
                    # 名字不冲突则直接收录
                    param_infos[name] = info

    # 若 EP 度大于 1，需要跨专家并行组交换元信息（不同 EP rank 持有不同专家）
    if ep_size > 1:
        # 预分配 EP 度大小的对象列表
        param_infos_list = [None] * ep_size
        # 在 EP 组内 all-gather 各 rank 的 (rank, 参数元信息字典)
        dist.all_gather_object(
            obj=(rank, param_infos), object_list=param_infos_list, group=mpu.get_expert_model_parallel_group()
        )
        # 遍历其他 EP rank 传来的元信息
        for src_rank, infos in param_infos_list:
            # 遍历该 EP rank 的每个参数元信息
            for name, info in infos.items():
                # 只收录本地没有的专家参数（本地已有的无需更新）
                if name not in param_infos:
                    # here we need to set the src_rank to the rank within the expert model parallel group
                    # （原英文注释：需把 src_rank 设为专家并行组内的 rank）
                    # 用 dataclasses.replace 生成新对象，把 src_rank 改成提供该参数的 EP rank
                    info = dataclasses.replace(info, src_rank=src_rank)
                    # 收录该专家参数元信息
                    param_infos[name] = info

    # 把字典转成列表
    param_infos = list(param_infos.values())
    # 按参数名排序，确保所有 rank 上的顺序完全一致（后续逐位校验也依赖此顺序）
    param_infos = sorted(param_infos, key=lambda info: info.name)

    # Check all ranks has the same parameter info
    # （原注释含义：校验所有 rank 的参数元信息一致）
    # 预分配全局世界大小的列表，用于收集所有 rank 的元信息
    all_param_info_list = [None] * dist.get_world_size()
    # 在基于 gloo 的 CPU 通信组上 all-gather 所有 rank 的元信息（元信息小，用 CPU 传输即可）
    dist.all_gather_object(
        obj=param_infos,
        object_list=all_param_info_list,
        group=get_gloo_group(),
    )
    # 以本地排序后的列表为基准逐项校验
    for i, param_info in enumerate(param_infos):
        # 遍历每个 rank 传来的元信息列表
        for infos in all_param_info_list:
            # 断言参数名一致，否则抛出带详细信息的错误
            assert infos[i].name == param_info.name, f"Parameter name mismatch: {infos[i].name} != {param_info.name}"
            # 断言参数形状一致（防止各 rank 对并行切分的理解不一致）
            assert (
                infos[i].shape == param_info.shape
            ), f"Parameter shape mismatch: {infos[i].shape} != {param_info.shape}"
            # 断言参数 dtype 一致
            assert (
                infos[i].dtype == param_info.dtype
            ), f"Parameter dtype mismatch: {infos[i].dtype} != {param_info.dtype}"

    # 返回全局一致、已排序的参数元信息列表
    return param_infos
