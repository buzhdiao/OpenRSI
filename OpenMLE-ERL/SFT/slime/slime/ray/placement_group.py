# 导入 logging，用于模块级日志
import logging
# 导入 socket，用于把主机名解析为 IP 地址（排序用）
import socket

# 导入 ray，用于创建放置组与 actor
import ray
# 导入放置组创建函数
from ray.util.placement_group import placement_group
# 导入放置组调度策略，把 actor 绑定到指定 bundle
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

# 导入训练 actor 组管理器
from .actor_group import RayTrainGroup
# 导入 rollout 管理器
from .rollout import RolloutManager

# 创建本模块日志器
logger = logging.getLogger(__name__)


# 信息探测 actor：占用 1 GPU，用于获取该 GPU 所在节点 IP 与物理 GPU 编号
@ray.remote(num_gpus=1)
class InfoActor:
    # 返回本 actor 所在节点的 IP 与分配到的第一个 GPU 编号
    def get_ip_and_gpu_id(self):
        # ray.util.get_node_ip_address 取节点 IP，ray.get_gpu_ids 取 GPU 列表
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


# 排序键函数：按节点 IP 数值与 GPU 编号排序 bundle
def sort_key(x):
    # 解包三元组：bundle 索引、节点标识（IP 或主机名）、GPU 编号
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    # （原注释含义：先按节点 IP 数值、再按 GPU 编号排序）
    # 尝试把节点标识按 IPv4 点分十进制解析
    try:
        # try to parse it as an IP address.
        # （原注释含义：尝试按 IP 地址解析）
        # 先假定标识就是 IP 字符串
        ip_address = node_identifier
        # 把各段转成整数列表（便于逐段数值比较）
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        # （原注释含义：尝试把主机名解析为 IP 地址）
        # 解析失败则走主机名解析路径
        try:
            # DNS 解析主机名
            ip_address = socket.gethostbyname(node_identifier)
            # 同样拆成整数列表
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            # （原英文注释：无法解析时用字符 ASCII 值做稳定排序）
            # 兜底方案：把标识字符串逐字符转 ASCII 值作为排序键
            node_ip_parts = [ord(c) for c in node_identifier]

    # 返回 (节点段列表, GPU 编号) 二元组作为排序键
    return (node_ip_parts, gpu_id)


# 创建占用指定数量 GPU 的放置组，并探测各 bundle 的物理位置排序
def _create_placement_group(num_gpus):
    """Create a placement group with the specified number of GPUs."""
    # 构造 bundle 列表：每个 bundle 1 GPU + 1 CPU
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    # 用 PACK 策略创建放置组（尽量紧凑打包）
    pg = placement_group(bundles, strategy="PACK")
    # 记录 bundle 总数
    num_bundles = len(bundles)

    # 阻塞等待放置组资源全部就绪
    ray.get(pg.ready())
    # use info actor to get the GPU id
    # （原注释含义：用信息 actor 来获取 GPU 编号）
    # 初始化信息 actor 列表
    info_actors = []
    # 为每个 bundle 启动一个信息 actor
    for i in range(num_bundles):
        # 用放置组调度策略把信息 actor 固定到第 i 个 bundle
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                )
            ).remote()
        )
    # 收集全部信息 actor 的 (节点 IP, GPU 编号)
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    # 探测完成后杀掉所有信息 actor，释放 GPU
    for actor in info_actors:
        # 逐个杀死
        ray.kill(actor)

    # 组装 (bundle 索引, 节点 IP, GPU 编号) 三元组列表
    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    # 按节点 IP 数值 + GPU 编号排序，使逻辑 rank 与物理拓扑一致
    sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
    # 重排后的 bundle 索引序列：第 i 个逻辑 rank 对应的物理 bundle
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    # Map from logical index -> physical GPU ID
    # （原注释含义：逻辑序号到物理 GPU 编号的映射）
    # 重排后的 GPU 编号序列
    pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]

    # 打印每个逻辑 rank 的物理位置，便于核对拓扑
    for i in range(num_bundles):
        # 取该逻辑 rank 实际的 bundle 索引
        actual_bundle_index = pg_reordered_bundle_indices[i]
        # 打印逻辑序号、物理 bundle 索引、节点与 GPU 编号
        logger.info(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
        )

    # 返回放置组与两个重排序列
    return pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids


# 创建整套实验所需的放置组并划分 actor/critic/rollout 的区间
def create_placement_groups(args):
    """Create placement groups for actor and rollout engines."""

    # 先计算所需 GPU 总数与各角色偏移
    num_gpus = 0
    # 只训练调试模式：仅按 actor（和 critic）数量申请
    if args.debug_train_only:
        # actor GPU 总数
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        # rollout 偏移为 0（不占用）
        rollout_offset = 0
        # 启用 critic 时追加 critic 的 GPU 数
        if args.use_critic:
            # 累加 critic GPU 数
            num_gpus += args.critic_num_nodes * args.critic_num_gpus_per_node
            # critic 区间紧跟 actor 之后
            critic_offset = args.actor_num_nodes * args.actor_num_gpus_per_node
    # 只 rollout 调试模式：仅按 rollout GPU 数申请
    elif args.debug_rollout_only:
        # rollout GPU 总数
        num_gpus = args.rollout_num_gpus
        # rollout 从 0 开始
        rollout_offset = 0
    # 共置模式：actor 与 rollout 共享同一批 GPU
    elif args.colocate:
        # GPU 总数 = actor GPU 数（rollout 复用）
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        # rollout 偏移为 0（与 actor 重叠）
        rollout_offset = 0
        # 启用 critic 时追加
        if args.use_critic:
            # 累加 critic GPU 数
            num_gpus += args.critic_num_nodes * args.critic_num_gpus_per_node
            # critic 偏移在 actor 之后
            critic_offset = args.actor_num_nodes * args.actor_num_gpus_per_node
    # 分离模式：actor、critic、rollout 各占独立 GPU
    else:
        # 总数 = actor GPU 数 + rollout GPU 数
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node + args.rollout_num_gpus
        # rollout 区间排在 actor 之后
        rollout_offset = args.actor_num_nodes * args.actor_num_gpus_per_node
        # 启用 critic 时继续追加
        if args.use_critic:
            # 累加 critic GPU 数
            num_gpus += args.critic_num_nodes * args.critic_num_gpus_per_node
            # critic 紧跟 actor
            critic_offset = args.actor_num_nodes * args.actor_num_gpus_per_node
            # rollout 再排到 critic 之后
            rollout_offset += args.critic_num_nodes * args.critic_num_gpus_per_node

    # 打印创建日志
    logger.info(f"Creating placement group with {num_gpus} GPUs...")
    # 创建放置组并拿到 actor 区间的重排索引与 GPU 编号
    pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids = _create_placement_group(num_gpus)

    # rollout 区间的重排索引 = actor 索引序列的尾部切片
    rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[rollout_offset:]
    # rollout 区间的 GPU 编号切片
    rollout_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[rollout_offset:]
    # 启用 critic 时切出 critic 区间
    if args.use_critic:
        # critic 重排索引从其偏移开始
        critic_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[critic_offset:]
        # critic GPU 编号切片
        critic_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[critic_offset:]

    # 返回三种角色的 (放置组, 索引, GPU 编号) 三元组字典
    return {
        # actor 用完整区间
        "actor": (pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids),
        # critic 仅启用时提供，否则 None
        "critic": (pg, critic_pg_reordered_bundle_indices, critic_pg_reordered_gpu_ids) if args.use_critic else None,
        # rollout 用尾部区间
        "rollout": (pg, rollout_pg_reordered_bundle_indices, rollout_pg_reordered_gpu_ids),
    }


# 分配一个训练 actor 组（每个 actor 0.4 GPU，可与 rollout 引擎共置）
def allocate_train_group(args, num_nodes, num_gpus_per_node, pg):
    # 创建 RayTrainGroup，num_gpus_per_actor=0.4 允许同 GPU 混部
    return RayTrainGroup(
        args=args,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        pg=pg,
        num_gpus_per_actor=0.4,
    )


# 创建训练模型（actor 与可选 critic），完成初始化与相互连接
def create_training_models(args, pgs, rollout_manager):
    # 先创建 actor 训练组
    actor_model = allocate_train_group(
        args=args,
        # actor 节点数
        num_nodes=args.actor_num_nodes,
        # actor 每节点 GPU 数
        num_gpus_per_node=args.actor_num_gpus_per_node,
        # actor 放置组
        pg=pgs["actor"],
    )
    # 启用 critic 时创建 critic 组并异步初始化
    if args.use_critic:
        # 创建 critic 训练组
        critic_model = allocate_train_group(
            args=args,
            # critic 节点数
            num_nodes=args.critic_num_nodes,
            # critic 每节点 GPU 数
            num_gpus_per_node=args.critic_num_gpus_per_node,
            # critic 放置组
            pg=pgs["critic"],
        )
        # 异步初始化 critic（角色为 critic，不加载参考模型）
        critic_init_handle = critic_model.async_init(args, role="critic", with_ref=False)
    else:
        # 未启用 critic 置为 None
        critic_model = None

    # 异步初始化 actor：是否加载参考模型取决于 KL 系数或 KL 损失；OPD 教师按配置决定
    start_rollout_ids = ray.get(
        actor_model.async_init(
            args,
            role="actor",
            with_ref=args.kl_coef != 0 or args.use_kl_loss,
            with_opd_teacher=args.use_opd and args.opd_type == "megatron",
        )
    )

    # 校验所有 actor 汇报的起始 rollout id 一致
    assert len(set(start_rollout_ids)) == 1
    # 未显式指定起始 id 时采用 actor 从 ckpt 恢复出的值
    if args.start_rollout_id is None:
        # 取第一个 actor 汇报的值
        args.start_rollout_id = start_rollout_ids[0]

    # 启用 critic 时：等 critic 初始化完成并建立 actor↔critic 连接
    if args.use_critic:
        # 阻塞等待 critic 初始化
        ray.get(critic_init_handle)
        # 建立两组 actor 的配对连接
        actor_model.connect(critic_model)

    # 把 rollout 管理器句柄传给所有 actor
    actor_model.set_rollout_manager(rollout_manager)
    # 使用全局数据集时，加载 start_rollout_id 的上一轮状态（含数据位置）
    if args.rollout_global_dataset:
        # 远程调用 load 加载数据集游标
        ray.get(rollout_manager.load.remote(args.start_rollout_id - 1))

    # 返回 actor 与 critic 模型组
    return actor_model, critic_model


# 创建 rollout 管理器 actor 并做初始配置
def create_rollout_manager(args, pg):
    # 启动 RolloutManager：只占 1 CPU、不占 GPU
    rollout_manager = RolloutManager.options(
        num_cpus=1,
        num_gpus=0,
    ).remote(args, pg)

    # calculate num_rollout from num_epoch
    # （原注释含义：由 epoch 数推算总 rollout 轮数）
    # 每 epoch 轮数先置 None
    num_rollout_per_epoch = None
    # 未显式指定总轮数时按 epoch 推算
    if args.num_rollout is None:
        # 远程查询每 epoch 的 rollout 轮数
        num_rollout_per_epoch = ray.get(rollout_manager.get_num_rollout_per_epoch.remote())
        # 总轮数 = 每 epoch 轮数 × epoch 数
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
        # 断言总轮数为正
        assert args.num_rollout > 0

    # 开启权重一致性校验时：先对两侧权重做快照再重置张量，供后续对比
    if args.check_weight_update_equal:
        # 快照当前权重
        ray.get(rollout_manager.check_weights.remote(action="snapshot"))
        # 重置引擎侧张量为随机值，验证更新链路确实生效
        ray.get(rollout_manager.check_weights.remote(action="reset_tensors"))

    # rollout offload 开启时，初始化后立即卸载 rollout 显存
    if args.offload_rollout:
        # 远程调用 offload
        ray.get(rollout_manager.offload.remote())

    # 返回管理器句柄与（可能为 None 的）每 epoch 轮数
    return rollout_manager, num_rollout_per_epoch
