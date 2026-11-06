# 导入 abc 用于定义抽象基类
import abc
# 导入 logging 用于日志输出
import logging
# 导入 socket 用于绑定临时端口（NCCL 组初始化）
import socket
# 导入 Namespace 类型注解
from argparse import Namespace
# 导入 Sequence 类型注解
from collections.abc import Sequence

# 导入 Ray 分布式框架（Actor 调用）
import ray
# 导入 torch 主库
import torch
# 导入 torch 分布式模块
import torch.distributed as dist
# 导入 Ray Actor 句柄类型
from ray.actor import ActorHandle
# 导入 DTensor（分布式张量）与 Replicate（复制式放置策略）
from torch.distributed.tensor import DTensor, Replicate

# 兼容不同 sglang 版本导入 torch reductions 补丁函数（新版在 utils.patch_torch，旧版在顶层）
try:
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions  # type: ignore[import]
except ImportError:
    from sglang.srt.patch_torch import monkey_patch_torch_reductions  # type: ignore[import]

# 导入 sglang 的多进程序列化工具（把张量序列化为可跨进程传输的对象）
from sglang.srt.utils import MultiprocessingSerializer

# 导入 slime 的分布式工具：初始化进程组
from slime.utils.distributed_utils import init_process_group


# 兼容不同 sglang 版本导入扁平化张量桶（FlattenedTensorBucket）：把同 dtype 张量拍平成一个连续 buffer
try:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket  # type: ignore[import]
except ImportError:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket  # type: ignore[import]


# 获取本模块 logger
logger = logging.getLogger(__name__)


# 权重同步抽象基类：负责把训练侧模型参数推送到 rollout（推理）引擎
class UpdateWeight(abc.ABC):
    def __init__(self, args: Namespace, model: torch.nn.Module) -> None:
        # 保存训练参数与模型引用
        self.args = args
        self.model = model
        # 权重版本号，每次同步自增（供推理引擎区分新旧权重）
        self.weight_version = 0

    # 抽象方法：与 rollout 引擎建立连接（子类实现具体通信方式）
    @abc.abstractmethod
    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
    ) -> None:
        pass

    # 执行一次完整权重同步：按桶（bucket）分批推送全部模型参数
    def update_weights(self) -> None:
        # 权重版本号加一
        self.weight_version += 1
        # 当前桶：累积 (name, param) 直到达到缓冲区上限
        bucket = []
        bucket_size = 0
        # 遍历模型状态字典中的所有参数
        for name, param in self.model.state_dict().items():
            # 计算参数字节数（元素个数 x 单元素字节数）
            param_size = param.numel() * param.element_size()
            # 若放入本参数会超出缓冲区上限，先冲刷当前桶再开新桶
            if bucket and bucket_size + param_size >= self.args.update_weight_buffer_size:
                self.wait_and_update_bucket_weights(bucket)
                del bucket
                bucket = []
                bucket_size = 0

            # 把参数移到 GPU
            param = param.cuda()
            if isinstance(param, DTensor):
                # async version of param.full_tensor
                # DTensor 场景：异步重分布为各卡复制的放置（Replicate），再取本地份，等价于异步 full_tensor
                param = param.redistribute(
                    placements=[Replicate()] * param.device_mesh.ndim,
                    async_op=True,
                ).to_local()
            # 参数放入桶中并累计大小
            bucket.append((name, param))
            bucket_size += param_size

        # 循环结束后冲刷残余的最后一个桶
        if bucket:
            self.wait_and_update_bucket_weights(bucket)
            del bucket
            bucket = []
            bucket_size = 0

    # 等待桶内异步操作完成后再真正发送权重
    def wait_and_update_bucket_weights(self, bucket):
        # 对每个参数：有 wait 方法的（异步 redistribute 结果）先等待完成
        bucket = [(name, param.wait()) if hasattr(param, "wait") else (name, param) for name, param in bucket]
        # 调用子类实现发送桶内权重，并携带当前版本号
        self.update_bucket_weights(bucket, weight_version=self.weight_version)

    # 抽象方法：发送一个桶的权重（子类实现）
    @abc.abstractmethod
    def update_bucket_weights(self, named_tensors, weight_version=None) -> None:
        pass


# 基于张量直传的权重同步实现：经序列化对象经 Ray RPC 推给推理引擎
class UpdateWeightFromTensor(UpdateWeight):
    """Push model weights to rollout engines using tensors.

    Streams parameters in size-bounded buckets; optionally groups tensors by dtype
    and flattens per dtype, gathers per-rank blobs to the source, and issues one
    RPC per dtype per bucket (or one per bucket if not flattened).
    """

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
    ) -> None:
        """Attach rollout engines and create per-engine IPC (Gloo) groups.

        Sets the gather source rank, engine handle, and `tp_rank` within the
        engine's local group.
        """
        # 保存 rollout 引擎句柄列表
        self.rollout_engines = rollout_engines

        # Here we assume the gpu id of rollout engines and train actors are the same.
        # 为每个引擎创建一个 Gloo 进程组（IPC 收集组），训练进程与引擎进程共享全局 rank 编号
        for i, engine in enumerate(self.rollout_engines):
            # 该引擎占用的全局 rank 区间 [start_rank, end_rank)
            start_rank = i * self.args.rollout_num_gpus_per_engine
            end_rank = (i + 1) * self.args.rollout_num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(
                ranks=group_ranks,
                backend="gloo",
            )
            # 只有本 rank 属于该引擎组时才记录相关信息
            if dist.get_rank() in group_ranks:
                # 组内收集源 rank（区间起点）
                self._ipc_gather_src = start_rank
                # 本 rank 所属的 IPC 收集组
                self._ipc_gather_group = new_group
                # 本 rank 对应的引擎句柄
                self._ipc_engine = engine
                # Calculate TP rank within this SGLang engine group
                # 计算本 rank 在引擎 TP 组内的序号
                self.tp_rank = dist.get_rank() - start_rank

    # 发送一个桶的权重：按 dtype 分组拍平、序列化、收集到源 rank 后统一 RPC 给引擎
    def update_bucket_weights(self, named_tensors, weight_version=None) -> None:
        # 打 sglang 要求的 torch reductions 补丁（保证跨后端集合通信兼容）
        monkey_patch_torch_reductions()
        # Use flattened bucket approach similar to Megatron
        # 采用与 Megatron 相同的扁平化张量桶方案
        logger.info("Using flattened tensor bucket")
        # Group tensors by dtype (same as Megatron)
        # 按 dtype 对张量分组
        named_tensors_by_dtypes = {}
        for name, tensor in named_tensors:
            # 取该张量的 dtype
            dtype = tensor.dtype
            if dtype not in named_tensors_by_dtypes:
                # 首次遇到该 dtype 则建组
                named_tensors_by_dtypes[dtype] = []
            # 加入对应 dtype 组
            named_tensors_by_dtypes[dtype].append((name, tensor))

        # Create flattened bucket for each dtype group
        # 为每个 dtype 组构造扁平化桶并序列化
        serialized_tensors = []
        for _dtype, named_tensors in named_tensors_by_dtypes.items():
            # 用 FlattenedTensorBucket 把同 dtype 的多个张量拍平成一个连续 buffer
            flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            # 取桶的元数据（名字、形状、偏移等）
            metadata = flattened_tensor_bucket.get_metadata()
            # 组装待传输数据：拍平张量 + 元数据
            flattened_tensor_data = {
                "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
                "metadata": metadata,
            }
            # 用多进程序列化器序列化为字符串
            serialized_tensors.append(MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True))

        if self._ipc_gather_src == dist.get_rank():
            # On rank 0, prepare a list to hold the gathered batches from all ranks.
            # 源 rank 准备接收列表（长度为组内世界大小）
            gathered_serialized_batches = [None for _ in range(dist.get_world_size(self._ipc_gather_group))]
        else:
            # 其他 rank 无需接收
            gathered_serialized_batches = None

        # Gather the serialized batches from all ranks to rank 0.
        # 把各 rank 序列化好的张量桶收集到源 rank
        dist.gather_object(
            obj=serialized_tensors,
            object_gather_list=gathered_serialized_batches,
            dst=self._ipc_gather_src,
            group=self._ipc_gather_group,
        )

        if dist.get_rank() == self._ipc_gather_src:
            # Handle flattened bucket format (same as Megatron approach)
            # 源 rank 负责把收集到的桶逐个 RPC 给推理引擎
            # Each rank may have multiple dtype buckets
            # TODO: here we assume all ranks have the same number of dtypes
            # 取每个 rank 的 dtype 桶数（假设各 rank 一致）
            num_dtypes = len(gathered_serialized_batches[0])
            assert num_dtypes > 0
            # 逐 dtype 桶发送
            for i in range(num_dtypes):
                kwargs = {
                    "serialized_named_tensors": [tensors[i] for tensors in gathered_serialized_batches],
                    "load_format": "flattened_bucket",
                    "flush_cache": False,
                    "weight_version": str(weight_version),
                }
                # 远程调用引擎的 update_weights_from_tensor 更新权重
                ref = self._ipc_engine.update_weights_from_tensor.remote(**kwargs)
                # 阻塞等待完成，保证版本顺序
                ray.get(ref)

        if dist.get_rank() == self._ipc_gather_src:
            # 桶发送完毕后让引擎冲刷 KV cache（权重变了，旧 cache 作废）
            ref = self._ipc_engine.flush_cache.remote()
            ray.get(ref)


# 基于临时 NCCL 广播组的权重同步实现：训练源 rank 直接把权重广播给推理引擎
class UpdateWeightFromDistributed(UpdateWeight):
    """Broadcast weights via a temporary NCCL group to rollout engines."""

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
    ) -> None:
        """On rank 0, initialize a temporary NCCL group for parameter broadcast."""
        # 保存引擎句柄与引擎锁
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock

        # For TP:
        #   1. AllGather parameters to rank 0
        #   2. Broadcast parameters from rank 0 to all sglang engines
        # 判断本 rank 是否为广播源（训练 rank 0）
        self._is_src_rank = dist.get_rank() == 0
        if self._is_src_rank:
            # 命名临时进程组
            self._group_name = "slime"
            # 取本节点的 IP 作为 NCCL master 地址
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            ## TODO: why +1?
            # 世界大小 = 所有引擎 GPU 数 + 1（训练源 rank 自己）
            world_size = self.args.rollout_num_gpus + 1

            # 通知各引擎初始化权重更新组（引擎侧 rank 从 1 开始编号）
            refs = [
                engine.init_weights_update_group.remote(
                    master_address,
                    master_port,
                    i * self.args.rollout_num_gpus_per_engine + 1,
                    world_size,
                    self._group_name,
                    backend="nccl",
                )
                for i, engine in enumerate(self.rollout_engines)
            ]
            # 训练侧创建对应的 NCCL 进程组（rank 0）
            self._model_update_groups = init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name=self._group_name,
            )
            # 等待所有引擎完成组初始化
            ray.get(refs)

    def update_bucket_weights(self, named_tensors, weight_version=None) -> None:
        """Send names/dtypes/shapes metadata to engines, then broadcast tensors.

        Ensures tensors are contiguous; when `world_size == 1`, converts DTensors
        to full tensors prior to `dist.broadcast`.
        """
        # 非源 rank 或空桶直接返回
        if not self._is_src_rank or not named_tensors:
            return

        # 先把本桶的名字/dtype/形状等元数据发给各引擎，让它们准备接收缓冲
        refs = [
            engine.update_weights_from_distributed.remote(
                names=[name for name, _ in named_tensors],
                dtypes=[param.dtype for _, param in named_tensors],
                shapes=[param.shape for _, param in named_tensors],
                group_name=self._group_name,
                weight_version=str(weight_version),
            )
            for engine in self.rollout_engines
        ]

        # 异步广播句柄列表
        handles = []
        # Broadcast parameters one by one with memory management
        # 逐参数广播并注意显存管理
        for _name, param in named_tensors:
            # 每次广播前清一下 CUDA 缓存避免显存累积
            torch.cuda.empty_cache()
            # Ensure tensor is contiguous and on the right device
            # 确保张量连续并位于正确的设备
            param_data = param.data.contiguous()

            # avoid `DTensor._op_dispatcher.dispatch` has `assert compute_mesh is not None` error
            # 世界大小为 1 且是 DTensor 时，转成普通完整张量以绕开 DTensor 分发器的断言错误
            if dist.get_world_size() == 1 and isinstance(param_data, DTensor):
                param_data = param_data.full_tensor()

            # Synchronous broadcast to avoid memory buildup
            # 异步发起广播（排队后统一等待），避免同步等待造成显存堆积
            handles.append(dist.broadcast(param_data, 0, group=self._model_update_groups, async_op=True))

        # 等待所有广播完成
        for handle in handles:
            handle.wait()
        # 等待各引擎侧的接收完成
        ray.get(refs)
