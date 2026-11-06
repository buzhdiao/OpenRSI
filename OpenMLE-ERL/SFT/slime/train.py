# 导入 Ray 分布式计算框架
import ray

# 导入放置组与训练管理相关函数
from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
# 导入参数解析函数
from slime.utils.arguments import parse_args
# 导入日志配置与实验跟踪初始化工具
from slime.utils.logging_utils import configure_logger, init_tracking
# 导入周期性动作判断工具（保存/评估是否在本轮执行）
from slime.utils.misc import should_run_periodic_action


# 同步训练主流程（rollout -> 训练 -> 同步权重，逐轮串行）
def train(args):
    # 配置全局日志格式
    configure_logger()
    # allocate the GPUs
    # 创建 GPU 放置组（划分训练与推理资源）
    pgs = create_placement_groups(args)
    # 初始化实验跟踪（如 W&B）
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    # 先创建 rollout 管理器（内含 sglang 引擎），同时得到每 epoch 的 rollout 数
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    # 创建 actor 与 critic 训练模型
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    # rollout 侧权重被卸载时，先重新加载权重
    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    # always update weight first so that sglang has the loaded weights from training.
    # 训练前先同步一次权重，确保 sglang 引擎载入了训练侧的初始权重
    actor_model.update_weights()

    # 需要校验权重更新一致性时，比较训练侧与推理侧的权重
    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    # rollout 侧 KV cache 被卸载时，重新加载 KV
    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    # 仅评估模式：不进行 rollout 训练，直接执行一次评估
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    # 训练后卸载训练模型显存的辅助函数
    def offload_train():
        # 配置为卸载训练模型时
        if args.offload_train:
            # 使用 critic 时：先卸载 critic
            if args.use_critic:
                critic_model.offload()
                # critic-only 预热步之后才卸载 actor（此前 actor 仍需训练）
                if rollout_id >= args.num_critic_only_steps:
                    actor_model.offload()
            else:
                # 不使用 critic 时只卸载 actor
                actor_model.offload()
        else:
            # 未开启卸载时仅清理缓存显存
            actor_model.clear_memory()

    # 保存 checkpoint 的辅助函数
    def save(rollout_id):
        # 非 critic 模式、或已过 critic-only 预热阶段时保存 actor
        if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        # 使用 critic 时同时保存 critic
        if args.use_critic:
            critic_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        # 开启全局数据集时保存数据集状态
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))
        # Save database snapshot if enabled
        # 保存数据库快照
        ray.get(rollout_manager.save_db_snapshot.remote(rollout_id))
        
    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    # 主训练循环：按 rollout_id 迭代
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # 第 0 轮训练前按需先做一次评估
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        # 生成 rollout 数据（采样）并同步等待结果引用
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        # rollout 结束后按需卸载推理引擎权重，为训练腾出显存
        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        # 使用 critic 时：critic 与 actor 依次训练
        if args.use_critic:
            # 异步启动 critic 训练
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            # 过了 critic-only 预热步后同步训练 actor
            if rollout_id >= args.num_critic_only_steps:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            # 等待 critic 训练完成
            ray.get(critic_train_handle)
        else:
            # 无 critic 时同步训练 actor
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        # 到达保存间隔（或 epoch 末尾）时保存 checkpoint
        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            save(rollout_id)

        # 训练后卸载训练模型显存
        offload_train()
        # 卸载后重新加载推理引擎权重
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        # 把训练后的最新权重同步给推理引擎
        actor_model.update_weights()
        # 权重同步后重新加载 KV cache
        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        # 到达评估间隔时执行评估
        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    # 全部训练完成后销毁 rollout 管理器，释放资源
    ray.get(rollout_manager.dispose.remote())


# 脚本入口：解析参数并启动训练
if __name__ == "__main__":
    args = parse_args()
    train(args)
