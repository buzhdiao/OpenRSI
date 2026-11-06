# 导入 os，用于读取环境变量
import os

# 导入 Ray 分布式计算框架
import ray
# 导入 wandb，用于训练结束时正常收尾 run
import wandb

# 导入放置组与训练管理相关函数
from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
# 导入参数解析函数
from slime.utils.arguments import parse_args
# 导入日志配置与实验跟踪初始化工具
from slime.utils.logging_utils import configure_logger, init_tracking
# 导入周期性动作判断工具
from slime.utils.misc import should_run_periodic_action


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
# 异步训练主流程：数据生成与训练重叠执行（一级异步；完全异步见 examples/full_async）
def train(args):
    # 异步训练不支持训练与推理同卡共置（colocate）
    assert not args.colocate, "Colocation is not supported for async training."
    # 配置全局日志格式
    configure_logger()
    # allocate the GPUs
    # 创建 GPU 放置组
    pgs = create_placement_groups(args)
    # 初始化实验跟踪
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    # 先创建 rollout 管理器（内含 sglang 引擎），并得到每 epoch 的 rollout 数
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    # 创建 actor 与 critic 训练模型
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    # always update weight first so that sglang has the loaded weights from training.
    # 先同步一次权重，确保 sglang 引擎载入训练侧的初始权重
    actor_model.update_weights()

    # 需要时校验权重更新的一致性
    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    # 训练开始前按需先做一次评估
    if args.eval_interval is not None and not args.skip_eval_before_train:
        ray.get(rollout_manager.eval.remote(args.start_rollout_id))

    # async train loop.
    # 预先发起第一轮数据生成（异步 future），实现生成与训练的重叠
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    # 主循环：按 rollout_id 迭代
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        # 等待上一轮预先发起的生成结果，作为本轮训练数据
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = ray.get(rollout_data_next_future)

        # Start the next rollout early.
        # 提前发起下一轮数据生成，与当前训练重叠执行
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        # 使用 critic 时：critic 与 actor 依次训练
        if args.use_critic:
            # 异步启动 critic 训练
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_curr_ref)
            # 过了 critic-only 预热步后同步训练 actor
            if rollout_id >= args.num_critic_only_steps:
                ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
            # 等待 critic 训练完成
            ray.get(critic_train_handle)
        else:
            # 无 critic 时同步训练 actor
            ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))

        # 未通过环境变量禁用保存且到达保存间隔时，保存 checkpoint
        if (
            os.environ.get("SLIME_DISABLE_SAVE", "0") != "1"
            and should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout)
        ):
            # 保存 actor 模型
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


        # 每隔 update_weights_interval 轮同步一次权重
        if (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            # 更新权重前先等待进行中的生成完成，避免在生成中途更换权重
            rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
            # 清空下一轮 future（其结果已被取走）
            rollout_data_next_future = None
            # 把最新权重同步给推理引擎
            actor_model.update_weights()

        # 到达评估间隔时执行评估
        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    # 训练结束，销毁 rollout 管理器，释放资源
    ray.get(rollout_manager.dispose.remote())
    # 启用了 wandb 且存在活跃 run 时正常收尾
    if args.use_wandb and wandb.run is not None:
        wandb.finish()


# 脚本入口：解析参数并启动异步训练
if __name__ == "__main__":
    args = parse_args()
    train(args)
