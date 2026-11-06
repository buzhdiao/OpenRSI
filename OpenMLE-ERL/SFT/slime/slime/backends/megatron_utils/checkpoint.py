import logging
import os
import re
from pathlib import Path

# TODO: may need to copy those 2 functions and do refactoring.
from megatron.training.checkpointing import load_checkpoint as _load_checkpoint_megatron
from megatron.training.checkpointing import save_checkpoint
from megatron.training.global_vars import get_args

from slime.utils import megatron_bridge_utils

try:
    # Here we patch out the `validate_non_overlapping_shards_metadata` in both functions
    # because it is really slow for large models with many shards.
    # TODO: find a less hacky way to do this.
    # 下面这段是对 torch ShardedTensor 校验逻辑的猴子补丁（monkey patch），
    # 因为大模型分片极多时 validate_non_overlapping_shards_metadata 校验非常慢，直接跳过它。
    import torch.distributed as dist
    import torch.distributed._shard.sharding_spec as shard_spec
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata
    from torch.distributed._shard.sharded_tensor.shard import Shard
    from torch.distributed._shard.sharded_tensor.utils import _parse_and_validate_remote_device
    from torch.distributed._shard.sharding_spec.api import EnumerableShardingSpec

    # 把 __post_init__ 替换成空函数，从而省去其中的分片元数据校验开销
    def __post_init__(self):
        pass

    EnumerableShardingSpec.__post_init__ = __post_init__

    # 用精简实现替换 ShardedTensor 的初始化方法，跳过跨 rank 的重校验
    @classmethod
    def _init_from_local_shards_and_global_metadata(  # type: ignore[override]
        cls,
        local_shards: list[Shard],
        sharded_tensor_metadata: ShardedTensorMetadata,
        process_group=None,
        init_rrefs=False,
        sharding_spec=None,
    ) -> ShardedTensor:
        """
        Initialize a ShardedTensor with local shards and a global
        ShardedTensorMetadata built on each rank.

        Warning: This API is experimental and subject to change. It does
                 not do cross rank validations, and fully rely on the user
                 for the correctness of sharded_tensor_metadata on each rank
        """
        # 规范化进程组（process group）
        process_group = cls._normalize_pg(process_group)
        current_rank = dist.get_rank()  # intentional to get global rank

        # 取全局分片元数据
        shards_metadata = sharded_tensor_metadata.shards_metadata

        # 本 rank 拥有的分片元数据列表
        local_shard_metadatas = []

        # collect local shard metadatas from the global sharded_tensor_metadata
        # 遍历全局分片元数据，挑出归属当前 rank 的分片
        for shard_metadata in shards_metadata:  # type: ignore[attr-defined]
            # 解析分片放置信息（placement），得到对应的 rank 与本地设备
            rank, local_device = _parse_and_validate_remote_device(process_group, shard_metadata.placement)

            # 只保留本 rank 的分片元数据
            if current_rank == rank:
                local_shard_metadatas.append(shard_metadata)

        # 再次取出分片元数据与张量属性（dtype/layout 等）
        shards_metadata = sharded_tensor_metadata.shards_metadata
        tensor_properties = sharded_tensor_metadata.tensor_properties

        # 未显式给定 sharding spec 时，从分片元数据反推切分规格
        if sharding_spec is None:
            spec = shard_spec._infer_sharding_spec_from_shards_metadata(shards_metadata)
        else:
            spec = sharding_spec

        # 绕过 __init__，直接按 spec 和全局形状/属性构造 ShardedTensor 对象
        sharded_tensor = ShardedTensor.__new__(
            ShardedTensor,
            spec,
            sharded_tensor_metadata.size,
            dtype=tensor_properties.dtype,
            layout=tensor_properties.layout,
            pin_memory=tensor_properties.pin_memory,
            requires_grad=tensor_properties.requires_grad,
        )

        # done validation, add local_shards
        # 挂上本地分片，并完成进程组/RRef 等准备工作
        sharded_tensor._local_shards = local_shards
        sharded_tensor._prepare_init(process_group=process_group, init_rrefs=init_rrefs)

        # run post initialization, i.e. map registration, rpc initialization
        # 执行后初始化（本地分片注册等）
        sharded_tensor._post_init()
        return sharded_tensor

    # 用补丁版初始化方法覆盖 torch 原实现
    ShardedTensor._init_from_local_shards_and_global_metadata = _init_from_local_shards_and_global_metadata

except ImportError:
    # 旧版 torch 没有这些私有 API 时直接跳过补丁
    pass

logger = logging.getLogger(__name__)

__all__ = ["save_checkpoint"]


def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, checkpointing_context, skip_load_to_model_and_opt):
    # ref: how megatron `load_checkpoint` gets directory
    # 读取全局参数并确定加载目录（与 Megatron load_checkpoint 的取路径方式一致）
    args = get_args()
    load_path = args.load

    # 断言加载目录存在且非空，否则提示可能填错了路径
    assert Path(load_path).exists() and _is_dir_nonempty(
        load_path
    ), f"{args.load=} does not exist or is an empty directory. Did you specify the wrong folder?"

    # 按目录特征判断是 Megatron 原生 ckpt 还是 HF ckpt，分别走不同的加载逻辑
    if _is_megatron_checkpoint(load_path):
        # Megatron 原生 checkpoint：直接调用 Megatron 的 load_checkpoint
        return _load_checkpoint_megatron(
            ddp_model=ddp_model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=skip_load_to_model_and_opt,
        )
    else:
        # 否则按 HuggingFace 权重加载
        return _load_checkpoint_hf(
            ddp_model=ddp_model,
            optimizer=optimizer,
            args=args,
            load_path=load_path,
        )


def _is_megatron_checkpoint(path: str | Path) -> bool:
    # 目录下存在 latest_checkpointed_iteration.txt，或目录名形如 iter_0000123，即认为是 Megatron ckpt
    return (Path(path) / "latest_checkpointed_iteration.txt").is_file() or bool(
        re.fullmatch(r"iter_\d{7}", Path(path).name)
    )


def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
    # HF 权重加载只支持 bridge 模式（megatron.bridge 做权重映射）
    assert args.megatron_to_hf_mode == "bridge", "Only bridge mode is supported for loading HF checkpoint"
    from megatron.bridge import AutoBridge

    # 导入插件以注册自定义模型桥接（可能不存在也会在此触发注册副作用）
    import slime_plugins.megatron_bridge  # noqa: F401

    logger.info(f"Load checkpoint from HuggingFace model into Megatron (path={load_path})")

    # 在模型补丁上下文中，用 AutoBridge 把 HF 权重加载进 Megatron 模型
    with megatron_bridge_utils.patch_megatron_model(ddp_model):
        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
        bridge.load_hf_weights(ddp_model)

    # Copied from Megatron-core :: load_checkpoint (with simplifications)
    # 混合精度训练且优化器存在时，把刚加载的权重同步到优化器保存的主参数（master weights）
    if (args.fp16 or args.bf16) and optimizer is not None:
        assert not args.load_main_params_from_ckpt
        optimizer.reload_model_params()

    # We can see `successfully loaded checkpoint from ... [ t 1/2, p 1/1 ] at iteration 0`
    # when loading Megatron, thus it is 0
    # 从 HF 初始化等价于从第 0 步开始训练，因此迭代数与已完成浮点运算量均为 0
    iteration = 0
    num_floating_point_operations_so_far = 0
    return iteration, num_floating_point_operations_so_far


def _is_dir_nonempty(path):
    # 用 scandir 判断目录是否非空（存在任意条目即为非空）
    with os.scandir(path) as it:
        return any(it)
