# 导入 os，读取 NCCL 等环境变量并在环境变量字典中透传
import os

# 导入 ray，用于创建与管理远程 actor
import ray
# 导入 Ray 的 PlacementGroup 类型，用于标注放置组参数
from ray.util.placement_group import PlacementGroup
# 导入放置组调度策略，把 actor 绑定到指定放置组的指定 bundle
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

# 导入常量列表：这些"不应设置 CUDA_VISIBLE_DEVICES"的环境变量名
from slime.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST


# 一组 Ray 训练 actor 的管理器：负责分配 GPU、创建 actor 并转发训练指令
# （docstring 含义：一组 ray actor；async 开头的函数应返回 ObjectRef 列表）
class RayTrainGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs

    Args:
        args (Namespace): Arguments for the actor group.
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
        resources (Dict[str, float], optional): Custom resources to allocate for each actor.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        num_resources_per_node (int, optional): Number of custom resources to allocate for each node.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
    """

    # 构造函数：保存节点/GPU 配置，并立即为 actor 分配 GPU 资源
    def __init__(
        self,
        args,
        num_nodes,
        num_gpus_per_node,
        pg: tuple[PlacementGroup, list[int], list[int]],
        num_gpus_per_actor: float = 1,
        role: str = "actor",
    ) -> None:
        # 保存全局配置
        self.args = args
        # 保存本组节点数
        self._num_nodes = num_nodes
        # 保存本组每节点 GPU 数
        self._num_gpus_per_node = num_gpus_per_node
        # 保存角色（actor 训练方 / critic 批评家等）
        self.role = role

        # Allocate the GPUs for actors w/o instantiating them
        # （原英文注释：先锁定 GPU 资源但不实例化模型）
        # 调用分配函数创建全部 actor
        self._allocate_gpus_for_actor(pg, num_gpus_per_actor)

    # 在放置组上为每个 rank 创建一个训练 actor（每个占 1 GPU）
    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
        # 世界大小 = 节点数 × 每节点 GPU 数
        world_size = self._num_nodes * self._num_gpus_per_node

        # Use placement group to lock resources for models of same type
        # （原英文注释：用放置组锁定同类型模型的资源）
        # 断言放置组已提供
        assert pg is not None
        # 解包放置组三元组：放置组对象、重排后的 bundle 索引、重排后的 GPU id
        pg, reordered_bundle_indices, _reordered_gpu_ids = pg

        # 组装传给 actor 的环境变量字典
        env_vars = {
            # because sglang will always set NCCL_CUMEM_ENABLE to 0
            # we need also set it to 0 to prevent nccl error.
            # （原英文注释：sglang 固定置 0，训练侧也置 0 以防 NCCL 报错）
            # NCCL 累计内存开关，默认 0 与 sglang 对齐
            "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
            # NVTE FP8 块级缩放的 FP32 scale 开关，默认 1
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": os.environ.get("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1"),
            # 这些环境变量置 1 表示不让 Ray 子进程设置 CUDA_VISIBLE_DEVICES（GPU 排他由 Ray 管理）
            **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
            # 透传用户自定义的训练环境变量
            **self.args.train_env_vars,
        }

        # 训练 offload 且后端为 megatron 时，需预加载显存 saver 的动态库
        if self.args.offload_train and self.args.train_backend == "megatron":
            # 导入 torch_memory_saver（显存保存/恢复库）
            import torch_memory_saver

            # 拼出预加载动态库的绝对路径（位于包目录上一级）
            dynlib_path = os.path.join(
                os.path.dirname(os.path.dirname(torch_memory_saver.__file__)),
                "torch_memory_saver_hook_mode_preload.abi3.so",
            )
            # 断言该 .so 文件存在，否则环境有问题
            assert os.path.exists(dynlib_path), f"LD_PRELOAD so file {dynlib_path} does not exist."

            # 通过 LD_PRELOAD 在进程启动时挂载显存 hook
            env_vars["LD_PRELOAD"] = dynlib_path
            # 打开 TMS 初始化使能
            env_vars["TMS_INIT_ENABLE"] = "1"
            # 打开 TMS 的 CPU 备份使能（offload 时把显存数据备份到 CPU）
            env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"

        # We cannot do routing replay for critic.
        # （原注释含义：critic 不能做路由重放）
        # 仅 actor 角色且开启 routing replay 时设置对应环境变量
        if self.args.use_routing_replay and self.role == "actor":
            # 打开路由重放开关（MoE 路由回放）
            env_vars["ENABLE_ROUTING_REPLAY"] = "1"

        # 读取训练后端名称
        backend = self.args.train_backend
        # megatron 后端：用 Megatron 实现的训练 actor
        if backend == "megatron":
            # 延迟导入 Megatron 训练 actor 类
            from slime.backends.megatron_utils.actor import MegatronTrainRayActor

            # 选定 actor 实现类
            actor_impl = MegatronTrainRayActor

        else:
            # FSDP 后端：延迟导入 FSDP 训练 actor 类
            from slime.backends.fsdp_utils import FSDPTrainRayActor

            # 选定 actor 实现类
            actor_impl = FSDPTrainRayActor

        # 用 ray.remote 包装为远程 actor 类：每 actor 1 GPU 并注入环境变量
        TrainRayActor = ray.remote(num_gpus=1, runtime_env={"env_vars": env_vars})(actor_impl)

        # Create worker actors
        # （原注释含义：创建 worker actor）
        # 初始化 actor 句柄列表
        self._actor_handlers = []
        # master 地址与端口先置空，由 rank 0 创建后回填
        master_addr, master_port = None, None
        # 按 rank 顺序逐个创建 actor
        for rank in range(world_size):
            # 用放置组调度策略把该 actor 绑到重排后的 bundle 上
            actor = TrainRayActor.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=reordered_bundle_indices[rank],
                ),
            ).remote(world_size, rank, master_addr, master_port)
            # rank 0 负责确定 master 地址端口，后续 rank 启动时要用
            if rank == 0:
                # 远程调用获取 master 地址与端口（阻塞等待）
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
            # 把 actor 句柄加入列表
            self._actor_handlers.append(actor)

    # 异步初始化全部 actor：分配资源并初始化模型/优化器/本地 ckpt 等
    def async_init(self, args, role, with_ref=False, with_opd_teacher=False):
        """
        Allocate GPU resourced and initialize model, optimzier, local ckpt, etc.
        """
        # 更新最新配置（可能覆盖构造时的 args）
        self.args = args
        # 对每个 actor 发起远程 init，返回 ObjectRef 列表
        return [
            actor.init.remote(args, role, with_ref=with_ref, with_opd_teacher=with_opd_teacher)
            for actor in self._actor_handlers
        ]

    # 异步执行一轮训练
    def async_train(self, rollout_id, rollout_data_ref):
        """Do one rollout training"""
        # 对每个 actor 发起远程 train，传入 rollout id 与数据引用
        return [actor.train.remote(rollout_id, rollout_data_ref) for actor in self._actor_handlers]

    # 同步保存所有 actor 的模型
    def save_model(self, rollout_id, force_sync=False):
        """Save actor model"""
        # ray.get 阻塞等待所有 actor 完成保存
        return ray.get([actor.save_model.remote(rollout_id, force_sync=force_sync) for actor in self._actor_handlers])

    # 把 rank 0 的权重广播到其他 rank（如加载 ckpt 后对齐）
    def update_weights(self):
        """Broadcast weights from rank 0 to all other ranks."""
        # 阻塞等待所有 actor 完成权重广播
        return ray.get([actor.update_weights.remote() for actor in self._actor_handlers])

    # 唤醒：把模型从 CPU 搬回 GPU（onload）
    def onload(self):
        # 阻塞等待所有 actor 完成唤醒
        return ray.get([actor.wake_up.remote() for actor in self._actor_handlers])

    # 休眠：把模型从 GPU 卸载到 CPU（offload）
    def offload(self):
        # 阻塞等待所有 actor 完成休眠
        return ray.get([actor.sleep.remote() for actor in self._actor_handlers])

    # 清理 actor 的缓存显存
    def clear_memory(self):
        # 阻塞等待所有 actor 完成清理
        return ray.get([actor.clear_memory.remote() for actor in self._actor_handlers])

    # 把本组 actor 与 critic 组的 actor 一一连接（actor-critic 训练）
    def connect(self, critic_group):
        # 阻塞等待所有 actor 完成与对应 critic 的连接
        return ray.get(
            [
                actor.connect_actor_critic.remote(critic)
                for actor, critic in zip(self._actor_handlers, critic_group._actor_handlers, strict=False)
            ]
        )

    # 给所有 actor 设置 rollout 管理器引用
    def set_rollout_manager(self, rollout_manager):
        # 阻塞等待所有 actor 完成设置
        return ray.get([actor.set_rollout_manager.remote(rollout_manager) for actor in self._actor_handlers])
