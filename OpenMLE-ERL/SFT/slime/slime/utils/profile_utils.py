# 导入 logging：模块级日志
import logging
# 导入 time：生成快照文件名中的时间戳
import time
# 导入 traceback：OOM 发生时打印调用栈辅助定位
import traceback
# 导入 Path：构造内存快照的输出路径
from pathlib import Path

# 导入 torch：PyTorch profiler 与 CUDA 内存记录接口
import torch

# 导入显存打印工具：OOM 时输出当前显存状态
from slime.utils.memory_utils import print_memory

# 创建本模块专属 logger
logger = logging.getLogger(__name__)


class TrainProfiler:
    def __init__(self, args):
        # 保存训练配置
        self.args = args
        # 训练整体阶段的 torch profiler（未启用时为 None）
        self._torch_profiler_overall = None
        # 训练整体阶段的内存 profiler（未启用时为 None）
        self._memory_profiler_overall = None

        # 启用 PyTorch profiler 且目标是 train_overall 时创建并持有
        if args.use_pytorch_profiler and ("train_overall" in args.profile_target):
            self._torch_profiler_overall = _create_torch_profiler(args, name="train_overall")

        # 启用内存历史记录且目标是 train_overall 时：创建内存 profiler 并立即开始记录
        if args.record_memory_history and ("train_overall" in args.profile_target):
            self._memory_profiler_overall = _BaseMemoryProfiler.create(args)
            self._memory_profiler_overall.start()

    def on_init_end(self):
        # 训练初始化结束后启动整体 profiler（前面初始化阶段的操作不入谱）
        if self._torch_profiler_overall is not None:
            self._torch_profiler_overall.start()

    def step(self, rollout_id: int):
        # 每步推进整体 profiler 的调度（wait/warmup/active 状态机）
        if self._torch_profiler_overall is not None:
            self._torch_profiler_overall.step()

        # 内存 profiler 配置了步数上限且到达最后一步时停止并落盘快照
        if (
            self._memory_profiler_overall is not None
            and ((s := self.args.memory_snapshot_num_steps) is not None)
            and (rollout_id == s - 1)
        ):
            self._memory_profiler_overall.stop()

    def iterate_train_actor(self, iterator):
        # 把训练 actor 的迭代器包一层 profiler（按步推进）
        return _profile_simple_loop(iterator, self.args, name="train_actor")

    def iterate_train_log_probs(self, iterator):
        # 把重算 log probs 的迭代器包一层 profiler
        return _profile_simple_loop(iterator, self.args, name="train_log_probs")


def _profile_simple_loop(iterator, args, name):
    # 未启用 profiler 或该阶段不在目标列表中：原样透传迭代器，零开销
    if not (args.use_pytorch_profiler and (name in args.profile_target)):
        yield from iterator
        return

    # 为该阶段创建独立的 torch profiler
    torch_profiler = _create_torch_profiler(args, name=name)
    # 启动 profiler 记录
    torch_profiler.start()
    # 逐项透传迭代器，每产出一项推进一次 profiler 调度
    for item in iterator:
        yield item
        torch_profiler.step()


def _create_torch_profiler(args, name):
    # 创建配置好的 PyTorch profiler 实例
    return torch.profiler.profile(
        schedule=torch.profiler.schedule(
            # TODO the train_actor and train_log_probs ones may need to have different args to control step
            # 等待阶段：跳过起始步之前的所有步（不计入分析）
            wait=max(args.profile_step_start - 1, 0),
            # 预热步数：首个分析步前留 1 步预热（非 0 起始时），排除 JIT 编译干扰
            warmup=1 if args.profile_step_start > 0 else 0,
            # 活跃记录步数：从 start 到 end 的区间长度
            active=args.profile_step_end - args.profile_step_start,
            # 只循环一轮（不重复采集）
            repeat=1,
        ),
        # 采集完成后的回调：把 trace 写入 TensorBoard 目录（按阶段名与 rank 命名，gzip 压缩）
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            args.tensorboard_dir,
            worker_name=f"{name}_rank_{torch.distributed.get_rank()}",
            use_gzip=True,
        ),
        # 记录算子输入形状，便于区分同算子不同 shape 的耗时
        record_shapes=True,
        # 记录 Python 调用栈，便于定位到具体代码行
        with_stack=True,
        # 记录显存分配信息
        profile_memory=True,
        # 估算每个算子的 FLOPs
        with_flops=True,
    )


class _BaseMemoryProfiler:
    @staticmethod
    def create(args):
        # 按配置选择具体的内存记录器实现类
        c = {
            "torch": _TorchMemoryProfiler,
            "memray": _MemrayMemoryProfiler,
        }[args.memory_recorder]
        # 实例化并返回
        return c(args)

    def __init__(self, args):
        # 预生成快照输出路径：目录 + 时间戳 + rank + 用户后缀，保证多次快照不互相覆盖
        self._path_dump = (
            Path(args.memory_snapshot_dir)
            / f"memory_snapshot_time{time.time()}_rank{torch.distributed.get_rank()}_{args.memory_snapshot_path}"
        )

    def start(self):
        # 子类必须实现启动逻辑
        raise NotImplementedError

    def stop(self):
        # 子类必须实现停止/落盘逻辑
        raise NotImplementedError


class _TorchMemoryProfiler(_BaseMemoryProfiler):
    def start(self):
        # 日志提示开始记录显存分配历史
        logger.info("Attach OOM dump memory history.")

        torch.cuda.memory._record_memory_history(
            # 最多记录 100 万条分配事件，覆盖长训练过程
            max_entries=1000000,
            # record stack information for the trace events
            # trace_alloc_record_context=True,
            # 记录全部分配调用栈（代价较高但信息最全）
            stacks="all",
        )

        # OOM 观察器：CUDA 发生内存不足时被回调
        def oom_observer(device, alloc, device_alloc, device_free):
            # 记录 OOM 信息与快照输出路径
            logger.info(
                f"Observe OOM, will dump snapshot to {self._path_dump}. ({device=} {alloc=} {device_alloc=} {device_free=}; stacktrace is as follows)"
            )
            # 打印当前调用栈，定位触发 OOM 的代码路径
            traceback.print_stack()
            # 立即把显存分配历史快照转储到文件（torch 可视化格式）
            torch.cuda.memory._dump_snapshot(self._path_dump)
            # 打印 OOM 时刻的显存统计
            print_memory("when oom")

        # 把观察器挂到 PyTorch 底层，OOM 时自动触发
        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    def stop(self):
        # 日志提示快照落盘位置
        logger.info(f"Dump memory snapshot to: {self._path_dump}")
        # 主动转储一次完整快照
        torch.cuda.memory._dump_snapshot(self._path_dump)
        # 关闭内存历史记录（enabled=None 表示停止记录）
        torch.cuda.memory._record_memory_history(enabled=None)


class _MemrayMemoryProfiler(_BaseMemoryProfiler):
    def __init__(self, args):
        # 先完成基类的路径初始化
        super().__init__(args)
        # memray 依赖显式配置快照步数，缺失则报错
        assert args.memory_snapshot_num_steps is not None, "In memray, must provide --memory-snapshot-num-steps"

    def start(self):
        # 日志提示 memray 追踪启动
        logger.info("Memray tracker started.")
        # 延迟导入 memray（可选依赖）
        import memray

        # 创建追踪器：输出到快照路径，开启原生栈追踪（能捕获 C/C++ 分配）
        self._tracker = memray.Tracker(
            file_name=self._path_dump,
            native_traces=True,
        )
        # 用上下文协议手动开启追踪
        self._tracker.__enter__()

    def stop(self):
        # 日志提示追踪结束与落盘位置
        logger.info(f"Memray tracker stopped and dump snapshot to: {self._path_dump}")
        # 手动结束追踪上下文，完成文件写入（异常参数传 None 表示正常退出）
        self._tracker.__exit__(None, None, None)
