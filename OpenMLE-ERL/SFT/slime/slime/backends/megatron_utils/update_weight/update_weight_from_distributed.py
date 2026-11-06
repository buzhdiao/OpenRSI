# 导入 socket，用于临时绑定端口获取空闲的 master 端口号
import socket
# 导入 time，用于获取锁失败时的轮询休眠
import time
# 导入 Namespace，用于标注 args 配置对象类型
from argparse import Namespace
# 导入类型标注工具：Callable/Mapping/Sequence
from collections.abc import Callable, Mapping, Sequence

# 导入 ray，用于远端调用 rollout 引擎的 actor 方法
import ray
# 导入 PyTorch 主库，用于张量操作
import torch
# 导入 torch.distributed，用于训练侧的集合通信（all-gather/broadcast/barrier）
import torch.distributed as dist
# 导入 Megatron 并行工具 mpu，获取各并行维度的 rank 与组
from megatron.core import mpu
# 导入 Ray 的 ObjectRef 类型，用于标注远端对象引用列表
from ray import ObjectRef
# 导入 Ray 的 ActorHandle 类型，标注 rollout 引擎 actor 句柄
from ray.actor import ActorHandle
# 导入 tqdm，用于在源 rank 上显示权重同步进度
from tqdm import tqdm

# 导入分布式工具：get_gloo_group（CPU 通信组）、init_process_group（初始化自定义进程组）
from slime.utils.distributed_utils import get_gloo_group, init_process_group

# 导入 Megatron→HF 转换函数，用于把 Megatron 参数转成 HF 命名张量
from ..megatron_to_hf import convert_to_hf
# 导入通用工具：单个参数的 all-gather、收集本 rank 命名参数与 buffer
from .common import all_gather_param, named_params_and_buffers


# 基于 NCCL 分布式广播的权重更新器：训练集群把参数直接广播给 rollout 推理引擎
class UpdateWeightFromDistributed:
    """
    Update distributed engines via NCCL. Each PP rank: group "slime-pp_{pp_rank}",
    only DP=TP=0 broadcasts. Non-expert (TP) and expert (EP) params separate.
    """

    # 构造函数：保存配置、模型、权重获取器、模型名与量化配置
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
        Initialize. Groups created in connect_rollout_engines.
        """
        # 保存全局参数配置（含 update_weight_buffer_size 等）
        self.args = args
        # 保存 Megatron 模型（多个 chunk 的序列）
        self.model = model
        # 保存 HF 模型名，转换权重时使用
        self.model_name = model_name
        # 保存量化配置（可能为 None），用于 int4/fp4 的前后处理
        self.quantization_config = quantization_config
        # 权重版本号，每同步一次自增 1，引擎侧用它判断权重是否更新
        self.weight_version = 0
        # 模型更新用的 NCCL 进程组，初始为 None，延迟到 connect_rollout_engines 创建
        self._model_update_groups = None

    # 与 rollout 引擎建立连接：为每个 PP rank 创建 NCCL 通信组
    def connect_rollout_engines(
        self, rollout_engines: Sequence[ActorHandle], rollout_engine_lock: ActorHandle
    ) -> None:
        """
        Create NCCL "slime-pp_{pp_rank}" if PP source (DP=TP=0). Lock prevents concurrent broadcasts.
        """
        # 保存 rollout 引擎 actor 句柄列表
        self.rollout_engines = rollout_engines
        # 保存全局广播锁 actor，用于避免多个训练源同时广播造成 NCCL 死锁
        self.rollout_engine_lock = rollout_engine_lock

        # For TP:
        #   1. AllGather parameters to rank 0
        #   2. Broadcast parameters from rank 0 to all sglang engines
        # （原三行注释含义：对 TP 维度——先把参数 all-gather 到 rank 0，再由 rank 0 广播给所有引擎）
        # 判断当前 rank 是否为 PP 广播源：DP（含上下文并行 CP）rank 为 0 且 TP rank 为 0
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        )
        # 获取当前流水线并行 rank，用于命名通信组
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        # 只有 PP 源 rank 才需要组名
        if self._is_pp_src_rank:
            # 组名按 PP rank 区分，每个流水线阶段一个独立 NCCL 组
            self._group_name = f"slime-pp_{pp_rank}"

        # 仅 PP 源 rank 需要与引擎建组
        if self._is_pp_src_rank:
            # 若此前已建过组（如二次连接），先销毁旧组避免泄漏
            if self._model_update_groups is not None:
                # 调用销毁函数同时清理训练侧与引擎侧的进程组
                disconnect_rollout_engines_from_distributed(
                    self.args, self._group_name, self._model_update_groups, self.rollout_engines
                )
            # 创建新的 NCCL 组：训练侧 rank 0 + 所有引擎 GPU
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.args, self._group_name, rollout_engines
            )

    # 用 torch.no_grad 装饰：权重同步纯属推理侧拷贝，无需记录梯度
    @torch.no_grad()
    def update_weights(self) -> None:
        """
        Pause → flush → non-expert (TP) → expert (EP) → continue. Progress on PP source.
        """
        # 权重版本号自增，本次同步产生新版本
        self.weight_version += 1

        # 仅全局 rank 0 负责"暂停生成/清缓存"等控制面操作（避免重复执行）
        if dist.get_rank() == 0:
            # 通过 Ray 远程暂停所有引擎的生成（保证广播期间无推理请求干扰）
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            # 清空各引擎的 KV cache（旧权重对应的缓存不再有效）
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])

            # int4/fp4 pre_process
            # （原注释含义：int4/fp4 的预处理）
            # 若使用 compressed-tensors 量化方案，广播前需要恢复反量化前的权重
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                # 调用后处理函数：restore=True（恢复原始权重）、不做事后量化
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )
        # 在 gloo CPU 组上同步，确保所有 rank 等待 rank 0 完成控制面操作
        dist.barrier(group=get_gloo_group())

        # 当前桶累计字节数，从 0 开始
        buffer_size = 0
        # 暂存已转成 HF 格式的 (名字, 张量) 对，攒够一桶再广播
        converted_named_tensors = []
        # non expert params
        # （原注释含义：非专家参数——本段只处理非专家参数）
        # 仅 PP 源 rank 创建进度条（total=0，按桶次数更新）
        pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_pp_src_rank else None

        # 遍历本 rank 所有命名参数与 buffer，先同步非专家参数
        for name, param in named_params_and_buffers(self.args, self.model):
            # 专家参数跳过（名字含 ".experts."），放到下一阶段处理
            if ".experts." in name:
                # 继续处理下一个参数
                continue
            # 单参数处理：TP all-gather + HF 转换 + 攒桶；返回更新后的桶大小
            buffer_size = self._update_weight_from_distributed(
                name, param, converted_named_tensors, buffer_size, pbar=pbar
            )

        # 循环结束后若桶中还有剩余参数，统一广播最后一桶
        if converted_named_tensors:
            # 广播最后一个非完整的桶
            self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)

        # gloo 同步：所有 rank 等非专家参数同步完成
        dist.barrier(group=get_gloo_group())

        # 专家参数阶段：桶计数清零
        buffer_size = 0
        # 专家参数暂存列表（先攒原始张量，EP all-gather 后再转 HF）
        named_tensors = []
        # 再次遍历所有命名参数与 buffer，这次只处理专家参数
        for name, param in named_params_and_buffers(self.args, self.model):
            # 非专家参数跳过
            if ".experts." not in name:
                # 继续处理下一个
                continue
            # 单个专家参数处理：TP all-gather + 攒桶（考虑 EP 数量放大）
            buffer_size = self._update_expert_weight_from_distributed(
                name, param, named_tensors, buffer_size, pbar=pbar
            )

        # 处理专家参数的最后一个不完整桶
        if named_tensors:
            # 广播剩余的专家参数桶
            self._update_expert_bucket_weights_from_distributed(named_tensors, pbar=pbar)

        # gloo 同步：所有 rank 等专家参数同步完成
        dist.barrier(group=get_gloo_group())
        # 控制面收尾仍由全局 rank 0 执行
        if dist.get_rank() == 0:
            # int4/fp4 post_process
            # （原注释含义：int4/fp4 的后处理）
            # compressed-tensors 量化：权重广播完成后在引擎侧重新做量化
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                # restore=False、post_quant=True：不恢复原始权重，而是执行量化后处理
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
            # 恢复所有引擎的生成服务
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        # 最终 gloo 同步，保证 rank 0 收尾动作完成后再继续训练
        dist.barrier(group=get_gloo_group())

    # 处理单个非专家参数：TP all-gather → 去填充 → HF 转换 → 攒桶（满则先广播）
    def _update_weight_from_distributed(
        self,
        name: str,
        param: torch.nn.Parameter,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        pbar: tqdm | None = None,
    ) -> int | None:
        """
        Non-expert: gather TP → rm pad → HF → buffer (flush if full). All gather, PP source buffers.
        Returns updated bytes on source, None on non-source.
        """
        # 把 TP 分片的参数 all-gather 成完整参数（非源 rank 此后只用它参与集合通信）
        param = all_gather_param(name, param)
        # 非 PP 源 rank 不负责攒桶/转换，直接返回 None
        if not self._is_pp_src_rank:
            # 返回空值（函数不回传桶大小）
            return

        # 计算该完整参数的字节大小
        param_size = param.numel() * param.element_size()
        # 若加入该参数会超出桶上限，先把当前桶广播出去
        if buffer_size + param_size > self.args.update_weight_buffer_size:
            # 广播当前已攒的桶并清空
            self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)
            # 桶计数清零
            buffer_size = 0
        # 把该参数转成 HF 命名张量（可能拆成多个张量）追加进桶
        converted_named_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
        # 累加桶字节计数
        buffer_size += param_size
        # 返回更新后的桶大小给调用方
        return buffer_size

    # 处理单个专家参数：TP all-gather 后只攒原始张量，HF 转换推迟到 EP all-gather 之后
    def _update_expert_weight_from_distributed(
        self,
        name: str,
        param: torch.nn.Parameter,
        named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        pbar: tqdm | None = None,
    ) -> int:
        """
        Expert: gather TP → rm pad → buffer. EP gather + HF deferred. Threshold × EP size.
        """
        # 把 TP 分片的专家参数 all-gather 成完整参数
        param = all_gather_param(name, param)

        # 计算该参数字节大小
        param_size = param.numel() * param.element_size()
        # 桶上限要乘 EP 世界大小：因为后续 EP all-gather 会把桶放大 EP 倍
        if (
            buffer_size + param_size
        ) * mpu.get_expert_model_parallel_world_size() > self.args.update_weight_buffer_size:
            # 先触发当前专家桶的 EP 聚合与广播
            self._update_expert_bucket_weights_from_distributed(named_tensors, pbar=pbar)
            # 桶计数清零
            buffer_size = 0

        # 把 (参数名, 完整参数) 追加进桶（暂不转 HF）
        named_tensors.append((name, param))
        # 累加桶字节计数
        buffer_size += param_size
        # 返回更新后的桶大小
        return buffer_size

    # 专家参数桶处理：先跨 EP all-gather 各 rank 的专家 → 转 HF → 走统一广播
    def _update_expert_bucket_weights_from_distributed(
        self, named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """
        Gather EP → HF → broadcast. Clears buffer.
        """
        # 提取本 rank 桶中的参数名列表
        names = [name for name, _ in named_tensors]
        # 预分配 EP 世界大小的列表，用于收集各 EP rank 的参数名
        all_names = [None] * mpu.get_expert_model_parallel_world_size()
        # 在 EP 组内 all-gather 对象（参数名列表），用于对齐各 rank 的桶内容
        dist.all_gather_object(all_names, names, group=mpu.get_expert_model_parallel_group())

        # 校验各 EP rank 桶内参数个数一致（顺序按 EP rank 排列）
        for names in all_names:
            # 不一致则报错并给出数量差异
            assert len(named_tensors) == len(names), f"mismatch names length: {len(named_tensors)} != {len(names)}"

        # 按 EP rank 组织的收集结果：all_gathered_params[ep_rank] 是该 rank 各参数的列表
        all_gathered_params = [[] for _ in range(mpu.get_expert_model_parallel_world_size())]
        # 保存异步 all-gather 句柄
        handles = []
        # 遍历本桶每个专家参数
        for i, (_name, param) in enumerate(named_tensors):
            # 为每个 EP rank 分配一块与参数同样形状的接收缓冲
            params = [
                torch.empty_like(param.data, device=torch.cuda.current_device())
                for _ in range(mpu.get_expert_model_parallel_world_size())
            ]
            # 发起 EP 组内的异步 all-gather（本参数从各 EP rank 收集其副本）
            handle = dist.all_gather(params, param.data, group=mpu.get_expert_model_parallel_group(), async_op=True)
            # 记录句柄，稍后统一等待
            handles.append(handle)
            # 先按对齐好的名字把缓冲挂到对应 EP rank 的结果列表（数据稍后由 wait 填充）
            for ep_rank, names in enumerate(all_names):
                # 以 all-gather 后的名字为准，配对 (名字, 参数副本)
                all_gathered_params[ep_rank].append((names[i], params[ep_rank]))
        # 统一等待所有异步 all-gather 完成
        for handle in handles:
            # 等待单个句柄
            handle.wait()

        # 清空本地桶（已复制到 all_gathered_params），控制内存
        named_tensors.clear()
        # 仅 PP 源 rank 继续做 HF 转换与广播
        if not self._is_pp_src_rank:
            # 非源 rank 到此结束
            return

        # 把按 EP rank 分组的列表展平成一个长列表
        all_gathered_params = sum(all_gathered_params, [])
        # HF 转换结果暂存列表
        converted_hf_tensors = []
        # 逐个把专家参数转成 HF 命名张量
        for name, param in all_gathered_params:
            # convert_to_hf 可能返回多个张量（如某些权重需重排）
            converted_hf_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)

        # 复用通用广播逻辑，把转换后的专家权重发给引擎
        self._update_bucket_weights_from_distributed(converted_hf_tensors, pbar)

    # 通用桶广播：加锁 → 广播给引擎 → 清空 → 解锁 → 更新进度条
    def _update_bucket_weights_from_distributed(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """
        Lock → broadcast → clear → unlock → pbar++. Lock prevents NCCL deadlock.
        """
        # lock the rollout engines to prevent dead lock on broadcast.
        # （原注释含义：加锁以防广播死锁）
        # 忙等获取全局广播锁（Ray actor 返回 False 表示未抢到），每 0.1 秒重试
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            # 休眠 0.1 秒后重试，避免空转占满 CPU
            time.sleep(0.1)

        # 发起本桶权重的分布式广播：先发元信息再广播张量，返回 ObjectRef 列表
        refs = update_weights_from_distributed(
            self._group_name,
            self._model_update_groups,
            self.weight_version,
            self.rollout_engines,
            converted_named_tensors,
        )

        # 阻塞等待所有引擎确认接收完成
        ray.get(refs)
        # 清空本桶，为下一桶腾出空间
        converted_named_tensors.clear()
        # 释放全局广播锁，允许其他源 rank 广播
        ray.get(self.rollout_engine_lock.release.remote())
        # 进度条推进一桶
        pbar.update(1)


# 建立分布式权重更新组：训练侧 rank 0 + 所有引擎 GPU 组成 NCCL 组
def connect_rollout_engines_from_distributed(
    args: Namespace, group_name: str, rollout_engines: Sequence[ActorHandle]
) -> dist.ProcessGroup:
    """
    Create NCCL group: training rank 0 + all engine GPUs. Blocks until joined.
    """
    # 通过 Ray 获取当前节点 IP，作为 NCCL 初始化的 master 地址
    master_address = ray._private.services.get_node_ip_address()
    # 创建临时 socket 绑定 0 号端口以探测空闲端口
    with socket.socket() as sock:
        # 绑定到所有网卡的随机可用端口
        sock.bind(("", 0))
        # 取出实际分配到的端口号作为 master 端口
        master_port = sock.getsockname()[1]
    # 世界大小 = 所有引擎 GPU 数 + 1（训练侧 rank 0）
    world_size = len(rollout_engines) * args.rollout_num_gpus_per_engine + 1

    # 让每个引擎在其 GPU 上初始化该 NCCL 组（rank 从 1 开始编号）
    refs = [
        engine.init_weights_update_group.remote(
            master_address,
            master_port,
            i * args.rollout_num_gpus_per_engine + 1,
            world_size,
            group_name,
            backend="nccl",
        )
        for i, engine in enumerate(rollout_engines)
    ]
    # 训练侧以 rank 0 身份通过 TCP 方式加入同一 NCCL 组
    model_update_groups = init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_address}:{master_port}",
        world_size=world_size,
        rank=0,
        group_name=group_name,
    )
    # 等待所有引擎完成组初始化（保证组完全建立后才返回）
    ray.get(refs)
    # 返回训练侧持有的进程组
    return model_update_groups


# 销毁分布式权重更新组：训练侧与引擎两侧都要清理
def disconnect_rollout_engines_from_distributed(args, group_name, model_update_groups, rollout_engines):
    """
    Destroy NCCL on training and engines.
    """
    # 让每个引擎销毁对应名字的进程组
    refs = [engine.destroy_weights_update_group.remote(group_name) for engine in rollout_engines]
    # 训练侧销毁自己的进程组
    dist.destroy_process_group(model_update_groups)
    # 等待所有引擎的销毁动作完成
    ray.get(refs)


# 单桶权重的分布式同步：先经 Ray 发元信息，再用 NCCL 广播张量
def update_weights_from_distributed(
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> list[ObjectRef]:
    """
    Send metadata (Ray), broadcast tensors (NCCL rank 0 → engines).
    """
    # 通知每个引擎即将接收的张量元信息（名字/dtype/形状）与版本号，引擎侧按此预分配缓冲
    refs = [
        engine.update_weights_from_distributed.remote(
            names=[name for name, _ in converted_named_tensors],
            dtypes=[param.dtype for _, param in converted_named_tensors],
            shapes=[param.shape for _, param in converted_named_tensors],
            group_name=group_name,
            weight_version=str(weight_version),
        )
        for engine in rollout_engines
    ]

    # 保存异步广播句柄
    handles = []
    # 逐个张量发起 NCCL 广播（源为训练侧 rank 0）
    for _, param in converted_named_tensors:
        # async_op=True 允许连续发起多个广播后再统一等待，重叠通信开销
        handles.append(dist.broadcast(param.data, 0, group=group, async_op=True))
    # 统一等待全部广播完成，确保张量已送达
    for handle in handles:
        # 等待单个广播句柄
        handle.wait()

    # 返回引擎侧的 ObjectRef 列表，由调用方 ray.get 确认完成
    return refs


# 触发所有引擎上的 int4/fp4 量化前后处理
def post_process_weights(
    restore_weights_before_load: bool,
    post_process_quantization: bool,
    rollout_engines: Sequence[ActorHandle],
):
    """
    Trigger post-process for int4/fp4 quantization on all rollout engines.
    """
    # 在所有引擎上远程调用 post_process_weights，并阻塞等待全部完成
    ray.get(
        [
            engine.post_process_weights.remote(
                restore_weights_before_load=restore_weights_before_load,
                post_process_quantization=post_process_quantization,
            )
            for engine in rollout_engines
        ]
    )
