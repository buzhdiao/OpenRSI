# 导入 Namespace，标注 args 配置对象类型
from argparse import Namespace
# 导入类型标注工具：Callable/Mapping/Sequence
from collections.abc import Callable, Mapping, Sequence
# 导入 Any，标注可变长存活张量等任意类型
from typing import Any

# 导入 ray，用于远程调用 rollout 引擎方法
import ray
# 导入 PyTorch 主库
import torch
# 导入 torch.distributed，用于 gloo gather 与 NCCL 通信
import torch.distributed as dist
# 导入 Megatron 并行工具 mpu，获取各并行维度 rank
from megatron.core import mpu
# 导入 Ray 的 ObjectRef 类型标注
from ray import ObjectRef
# 导入 Ray 的 ActorHandle 类型标注
from ray.actor import ActorHandle

# 导入分布式工具：get_gloo_group 返回 CPU gloo 通信组
from slime.utils.distributed_utils import get_gloo_group

# 导入 sglang 相关：扁平化张量桶与多进程序列化器（用于 IPC 传输）
from ..sglang import FlattenedTensorBucket, MultiprocessingSerializer
# 导入 HF 权重迭代器基类，用工厂方法创建具体迭代器
from .hf_weight_iterator_base import HfWeightIteratorBase
# 复用分布式方案的建组/销毁组/广播/量化后处理函数
from .update_weight_from_distributed import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    post_process_weights,
    update_weights_from_distributed,
)


# 基于张量字典的权重更新器：本地/远端引擎混合场景（colocated 用 IPC，远端用 NCCL）
class UpdateWeightFromTensor:
    """
    Update rollout engines from tensor dict:
    load(dict→GPU) → broadcast PP/EP(GPU NCCL) → gather TP(GPU NCCL) → convert HF(GPU) → send.
    Colocated: GPU→CPU serialize → gather_object(Gloo CPU, collects from rollout_num_gpus_per_engine ranks) → Ray IPC to engine.
    Distributed: GPU NCCL broadcast to remote engines.
    """

    # 构造函数：保存配置与模型，计算参数桶并建立 IPC 用的 gloo 分组
    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        """
        Compute param buckets, create IPC Gloo groups (rollout_num_gpus_per_engine ranks/group).
        """
        # 保存全局配置
        self.args = args
        # 保存 Megatron 模型
        self.model = model
        # 保存权重获取器（返回当前训练权重的字典）
        self.weights_getter = weights_getter
        # 保存 HF 模型名
        self.model_name = model_name
        # 保存量化配置（可为 None）
        self.quantization_config = quantization_config
        # 权重版本号，从 0 开始
        self.weight_version = 0

        # 通过工厂方法创建具体的 HF 权重迭代器（自动选择 bucket/direct 实现）
        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args, model=model, model_name=model_name, quantization_config=quantization_config
        )

        # create the group within megatron.
        # （原注释含义：在 Megatron 内部建组）
        # 按每个引擎占用的 GPU 数把全部训练 rank 切成若干 gloo 组，用于 IPC gather
        for start_rank in range(0, dist.get_world_size(), self.args.rollout_num_gpus_per_engine):
            # 计算该组的结束 rank（不含）
            end_rank = start_rank + self.args.rollout_num_gpus_per_engine
            # 该组包含的连续 rank 列表
            group_ranks = list(range(start_rank, end_rank))
            # 所有 rank 都要调用 new_group 以保证组创建的一致性
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            # 只有本 rank 属于该组时才记录下来（每个 rank 只属于一个组）
            if dist.get_rank() in group_ranks:
                # 保存本 rank 所属的 IPC gather 组
                self._ipc_gather_group = new_group
                # 组内起始 rank 作为 gather 的目标（src）
                self._ipc_gather_src = start_rank

        # 分布式引擎的 NCCL 组，初始为 None，延迟到 connect_rollout_engines 创建
        self._model_update_groups = None

    # 与 rollout 引擎建立连接：区分本地共置（colocated）与远端（distributed）引擎
    def connect_rollout_engines(
        self, rollout_engines: Sequence[ActorHandle], rollout_engine_lock: ActorHandle
    ) -> None:
        """
        Split colocated/distributed engines. Global source rank (DP=TP=PP=0) creates NCCL
        for distributed. Map ranks to colocated IPC engines.
        """
        # 先把全部引擎保存下来
        self.rollout_engines = rollout_engines
        # 计算训练集群能共置的引擎数量：训练 GPU 总数 ÷ 每引擎 GPU 数
        colocate_engine_nums = (
            self.args.actor_num_nodes * self.args.actor_num_gpus_per_node // self.args.rollout_num_gpus_per_engine
        )
        # 引擎总数超过共置容量时，说明存在远端引擎，需要走分布式路径
        self.use_distribute = len(rollout_engines) > colocate_engine_nums

        # 若存在远端引擎，则拆分列表并为远端部分建 NCCL 组
        if self.use_distribute:
            # 前段是共置引擎（与训练 GPU 同节点）
            self.rollout_engines = rollout_engines[:colocate_engine_nums]
            # 后段是远端引擎（需 NCCL 广播）
            self.distributed_rollout_engines = rollout_engines[colocate_engine_nums:]
            # 远端广播源判定：DP（含 CP）、TP、PP rank 全为 0 的全局源
            self._is_distributed_src_rank = (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                and mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == 0
            )
            # 远端通信组名固定为 "slime"
            self._group_name = "slime"
            # 只有全局源 rank 负责建组
            if self._is_distributed_src_rank:
                # 若已有旧组先销毁，避免泄漏
                if self._model_update_groups is not None:
                    # 销毁训练侧与远端引擎侧的旧组
                    disconnect_rollout_engines_from_distributed(
                        self.args, self._group_name, self._model_update_groups, self.distributed_rollout_engines
                    )

                # 为远端引擎创建新的 NCCL 组
                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args, self._group_name, self.distributed_rollout_engines
                )

        # Here we assume the gpu id of rollout engines and train actors are the same.
        # （原英文注释：假设 rollout 引擎与训练 actor 的 GPU 编号一致）
        # 为共置引擎建立 rank → 引擎 的映射：每段连续 rank 对应一个引擎
        for i, engine in enumerate(self.rollout_engines):
            # 该引擎负责的起始训练 rank
            start_rank = i * self.args.rollout_num_gpus_per_engine
            # 该引擎负责的结束训练 rank（不含）
            end_rank = (i + 1) * self.args.rollout_num_gpus_per_engine
            # 该段 rank 列表
            group_ranks = list(range(start_rank, end_rank))
            # 本 rank 落在某引擎的段内，就记住自己的 IPC 目标引擎
            if dist.get_rank() in group_ranks:
                # 保存本 rank 对应的共置引擎
                self._ipc_engine = engine

    # 用 no_grad 装饰：权重同步无需梯度追踪
    @torch.no_grad()
    def update_weights(self) -> None:
        """
        version++, flush caches, process buckets. Progress on rank 0.
        """
        # 权重版本号自增
        self.weight_version += 1

        # 获取当前全局 rank
        rank = dist.get_rank()
        # 控制面操作由 rank 0 统一执行
        if rank == 0:
            # 暂停所有共置引擎的生成
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            # 清空各引擎 KV cache
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            # compressed-tensors 量化：广播前恢复原始权重
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                # 调用后处理：restore=True、不做事后量化
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )
        # gloo 同步，等 rank 0 完成控制面操作
        dist.barrier(group=get_gloo_group())

        # 从训练框架取当前最新的 Megatron 本地权重字典
        megatron_local_weights = self.weights_getter()

        # 逐桶迭代 HF 权重（迭代器内部完成广播/all-gather/转换）
        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            # 把该桶的 HF 张量发给共置/远端引擎，返回引用与需保活的张量
            refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
            # 阻塞等待引擎确认接收
            ray.get(refs)
            # 引擎已收到，释放保活张量
            del long_lived_tensors

        # gloo 同步，等全部桶发送完成
        dist.barrier(group=get_gloo_group())

        # int4/fp4 post_process
        # （原注释含义：int4/fp4 后处理）
        # 收尾控制面仍由 rank 0 执行
        if rank == 0:
            # compressed-tensors 量化：权重加载完成后重新量化
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                # restore=False、post_quant=True：执行量化后处理
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
            # 恢复所有共置引擎的生成服务
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        # 最终同步，保证 rank 0 收尾完成
        dist.barrier(group=get_gloo_group())

    # 把一桶 HF 张量同时发给共置引擎（IPC）与远端引擎（NCCL）
    def _send_hf_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        # 汇总所有引擎返回的 ObjectRef
        all_refs = []

        # 走 IPC 路径发送给共置引擎（内部完成序列化与 gloo gather）
        refs_colocated, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            weight_version=self.weight_version,
        )
        # 把共置路径的引用并入总列表
        all_refs.extend(refs_colocated)

        # 若有远端引擎且本 rank 是分布式源 rank，则走 NCCL 广播路径
        if self.use_distribute and self._is_distributed_src_rank:
            # 复用分布式方案：发元信息 + NCCL 广播张量
            refs_distributed = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                self.weight_version,
                self.distributed_rollout_engines,
                hf_named_tensors,
            )
            # 有远端引用才并入（源 rank 才会返回非空）
            if refs_distributed:
                # 并入远端引用
                all_refs.extend(refs_distributed)

        # 返回全部引用与保活张量（供调用方 ray.get 与 del）
        return all_refs, long_lived_tensors


# 把 HF 张量经共享内存 IPC 发给同节点的共置引擎
def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    weight_version,
) -> tuple[list[ObjectRef], Any]:
    # TODO improve
    # （原 TODO 含义：待改进）
    # 保存需要延长生命周期的张量容器（防止共享内存被提前回收）
    long_live_tensors = []

    # 若新版 FlattenedTensorBucket 支持混合 dtype，可整体一次打包
    if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
        # 用固定键 "dtype" 包住全部张量，表示不按 dtype 分组
        converted_named_tensors_by_dtypes = {"dtype": hf_named_tensors}
    else:
        # 旧版不支持混合 dtype：需按 dtype 分组打包
        converted_named_tensors_by_dtypes = {}
        # 遍历所有 HF 命名张量
        for name, tensor in hf_named_tensors:
            # 取该张量的 dtype
            dtype = tensor.dtype
            # 该 dtype 的分组不存在则先建空列表
            if dtype not in converted_named_tensors_by_dtypes:
                # 初始化该 dtype 的分组
                converted_named_tensors_by_dtypes[dtype] = []
            # 把 (名字, 张量) 追加进对应 dtype 分组
            converted_named_tensors_by_dtypes[dtype].append((name, tensor))

    # 序列化结果列表：每个 dtype 组对应一个序列化桶
    serialized_tensors = []
    # 遍历每个 dtype 分组
    for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
        # 把该组张量打包成扁平化张量桶（连续内存 + 元信息）
        flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        # 获取桶的元数据（各张量的名字/偏移/形状等）
        metadata = flattened_tensor_bucket.get_metadata()
        # 组装为字典：扁平化数据 + 元数据
        flattened_tensor_data = {
            "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
            "metadata": metadata,
        }
        # 该字典需保活直到引擎加载完成（共享内存引用其缓冲）
        long_live_tensors.append(flattened_tensor_data)
        # 用多进程序列化器打包为可通过 IPC 传输的对象（输出字符串形式）
        serialized_tensors.append(MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True))

    # 准备 gather 结果列表：仅组内 src rank 需要分配接收缓冲
    serialized_named_tensors = (
        [None] * dist.get_world_size(ipc_gather_group) if ipc_gather_src == dist.get_rank() else None
    )
    # 在 IPC gloo 组内 gather 各 rank 的序列化桶到 src rank（一个引擎的各 TP 分片汇总）
    dist.gather_object(
        serialized_tensors,
        object_gather_list=serialized_named_tensors,
        dst=ipc_gather_src,
        group=ipc_gather_group,
    )

    # 本 rank 发起的引擎更新引用
    refs = []
    # 只有 src rank 负责把汇总结果发给引擎
    if dist.get_rank() == ipc_gather_src:
        # TODO: here we assume all ranks have the same number of dtypes, not sure if that is correct.
        # （保留原 TODO：假设各 rank dtype 数一致）
        # 以 rank0 的序列化结果确定 dtype 组数
        num_dtypes = len(serialized_named_tensors[0])
        # 逐 dtype 组向引擎发起一次 IPC 加载
        for i in range(num_dtypes):
            # 组装远端调用参数：各 rank 第 i 组的序列化张量 + 加载格式 + 版本号
            kwargs = {
                "serialized_named_tensors": [tensors[i] for tensors in serialized_named_tensors],
                "load_format": "flattened_bucket",
                "weight_version": str(weight_version),
            }
            # 远程调用引擎的 update_weights_from_tensor，记录引用
            refs.append(ipc_engine.update_weights_from_tensor.remote(**kwargs))

    # 返回引用列表与保活张量
    return refs, long_live_tensors
