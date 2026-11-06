# asyncio：协程与事件循环，驱动常驻的异步 rollout 主循环
import asyncio
# atexit：注册进程退出钩子，确保退出时停止全局 worker
import atexit
# queue：线程安全队列，用于 worker 线程与主线程之间传递完成的样本组
import queue
# threading：后台 worker 线程与互斥锁
import threading
# time：统计 rollout 耗时与"无进展"告警计时
import time
# List：完成组列表的类型注解
from typing import List

# Import core functions from sglang_rollout directly to avoid code duplication
# 直接复用 slime 的 sglang rollout 核心实现：生成状态管理、评测 rollout、组生成+奖励模型
from slime.rollout.sglang_rollout import GenerateState, eval_rollout, generate_and_rm_group
# run：在同步上下文中驱动协程的工具函数
from slime.utils.async_utils import run
# Sample：rollout 样本的数据类型
from slime.utils.types import Sample

# Global worker manager
# 全局唯一的异步 rollout worker 实例（惰性创建，跨 rollout 调用复用）
_global_worker = None
# 保护全局 worker 创建/读取/销毁的线程锁
_worker_lock = threading.Lock()
# 生成中断（generation abort）相关的元数据键：重试前需从样本 metadata 中清除
_GENERATION_ABORT_METADATA_KEYS = (
    "generation_aborted",
    "generation_abort_reason",
    "generation_abort_finish_reason",
    "group_generation_aborted",
)
# 标记"生成中断"类别的 code_category 取值集合
_GENERATION_ABORT_CODE_CATEGORIES = {"generation_abort", "generation_abort_group"}


# 判断样本是否属于"生成中断"：状态为 ABORTED，或 metadata 带有单样本/组级中断标记
def _sample_has_generation_abort(sample: Sample) -> bool:
    # 样本状态为 ABORTED 即视为中断
    if sample.status == Sample.Status.ABORTED:
        return True
    # 读取样本 metadata（可能缺失，兜底为空 dict）
    metadata = getattr(sample, "metadata", None) or {}
    # 命中任一中断标记键即视为中断
    return bool(metadata.get("generation_aborted") or metadata.get("group_generation_aborted"))


# 把被中断的样本重置回待生成状态，并清除中断相关元数据，以便放回数据缓冲区重试
def _reset_sample_for_generation_retry(sample: Sample) -> None:
    # 状态复位为 PENDING
    sample.status = Sample.Status.PENDING
    # 清空奖励
    sample.reward = None
    # 清空响应文本
    sample.response = ""
    # 响应长度归零
    sample.response_length = 0
    # 清空 loss 掩码
    sample.loss_mask = []
    # 清空 rollout 对数概率
    sample.rollout_log_probs = []
    # 清空路由专家记录
    sample.rollout_routed_experts = None
    # 清空权重版本记录
    sample.weight_versions = []
    # 清空 token 序列
    sample.tokens = []

    # 读取 metadata；不是 dict 时无需也无法清理
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        return
    # 逐个移除生成中断相关的元数据键
    for key in _GENERATION_ABORT_METADATA_KEYS:
        metadata.pop(key, None)
    # code_category 属于中断类别时一并移除
    if metadata.get("code_category") in _GENERATION_ABORT_CODE_CATEGORIES:
        metadata.pop("code_category", None)
    # 移除历史状态码记录，避免残留旧信息
    metadata.pop("status_codes", None)


def get_global_worker(args, data_buffer):
    """Get or create global worker"""
    # 声明操作全局变量
    global _global_worker
    # 加锁保证并发下只创建一个 worker
    with _worker_lock:
        # 不存在或 worker 线程已死亡时重建
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            # Keep OpenMLE semantics: ctor concurrency is unused for the in-flight
            # cap; continuous_worker_loop limits by rollout_batch_size (not 512×engines).
            # 保持 OpenMLE 语义：在途组上限由 continuous_worker_loop 按 rollout_batch_size
            # 控制，而不是 构造并发数×引擎数，避免请求洪峰打爆 SGLang。
            print("Creating new global async worker...")
            _global_worker = AsyncRolloutWorker(args, data_buffer, concurrency=args.sglang_server_concurrency)
            # 启动 worker 线程
            _global_worker.start()
        # 返回全局 worker
        return _global_worker


def get_existing_global_worker():
    """Return the existing worker without creating a new one."""
    # 声明操作全局变量
    global _global_worker
    # 加锁读取，避免与创建/销毁竞争
    with _worker_lock:
        # 从未创建过则返回 None
        if _global_worker is None:
            return None
        # 线程不存在或已死亡则视为无效
        if _global_worker.worker_thread is None or not _global_worker.worker_thread.is_alive():
            return None
        # 返回仍然存活的全局 worker
        return _global_worker


def stop_global_worker():
    """Stop global worker"""
    # 声明操作全局变量
    global _global_worker
    # 加锁防止与创建逻辑竞争
    with _worker_lock:
        # 存在 worker 时先停止其线程
        if _global_worker is not None:
            _global_worker.stop()
            # 置空引用完成清理
            _global_worker = None


# 简化版异步 rollout worker：用线程（而非进程）承载常驻事件循环，
# 生命周期独立于单次 rollout 函数调用，实现"全异步"的持续采样
class AsyncRolloutWorker:
    """
    Simplified asynchronous rollout worker, using threads instead of processes
    Supports continuous running, independent of rollout function lifecycle
    """

    def __init__(self, args, data_buffer, concurrency=10):
        # 保存训练参数
        self.args = args
        self.data_buffer = data_buffer  # Directly save data_buffer reference
        # 构造传入的并发数（实际在途上限见 continuous_worker_loop 的说明）
        self.concurrency = concurrency
        # 运行开关：置 False 后 worker 循环退出
        self.running = True
        self.output_queue = queue.Queue(maxsize=1000)  # Continuous output queue
        # worker 线程句柄
        self.worker_thread = None
        # 生成状态（采样参数等）由 slime 的 GenerateState 统一管理
        self.state = GenerateState(args)
        # 暂停事件：评测前请求暂停新的训练数据提交
        self.pause_requested = threading.Event()
        # 空闲事件：暂停后所有在途任务完成时置位
        self.idle = threading.Event()

    async def continuous_worker_loop(self):
        """Continuous work loop - constantly get data from data_buffer and process"""
        print("Continuous async rollout worker started")

        # 当前在途任务集合
        active_tasks = set()
        # OpenMLE long trajectories: cap in-flight groups by rollout_batch_size.
        # Do NOT use sglang_server_concurrency×engines here — that floods SGLang.
        # 在途组上限取 rollout_batch_size（OpenMLE 长轨迹语义）；
        # 不能用 并发数×引擎数，否则会把 SGLang 打爆。
        max_concurrent_tasks = self.args.rollout_batch_size
        # 组 id 计数器，为每个提交的组分配顺序标识
        group_id_counter = 0

        # 主循环：运行期间持续取数据、派发任务、回收结果
        while self.running:
            try:
                # Clean up completed tasks
                # 回收已完成任务：取回结果（异常在此打印）并从集合移除
                if active_tasks:
                    done_tasks = {task for task in active_tasks if task.done()}
                    for task in done_tasks:
                        try:
                            task.result()  # Results are already handled in callbacks
                        except Exception as e:
                            print(f"Task failed with exception: {e}")
                    active_tasks -= done_tasks

                # 收到暂停请求（如评测前）时：在途任务清零后置 idle 事件并等待
                if self.pause_requested.is_set():
                    if not active_tasks:
                        self.idle.set()
                    await asyncio.sleep(0.1)
                    continue

                # 循环恢复运行时清除 idle 标记
                if self.idle.is_set():
                    self.idle.clear()

                # If active task count hasn't reached limit, try to get new data and start tasks
                # 在途任务未达上限时，持续从数据缓冲取组并启动生成任务
                while len(active_tasks) < max_concurrent_tasks and self.running:
                    # 从缓冲区取一个组的数据
                    samples = self.data_buffer.get_samples(1)
                    # 取不到数据时跳出，稍后再试
                    if not samples:
                        break

                    for group in samples:
                        # 取当前组 id 并递增计数器
                        group_id = group_id_counter
                        group_id_counter += 1

                        # Create new async task
                        # 为该组创建异步生成任务（生成 + 奖励模型打分）
                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )

                        # Add completion callback
                        # 构造完成回调工厂：闭包捕获组 id，把 (组id, 结果) 放入输出队列
                        def make_callback(gid):
                            def task_done_callback(task):
                                # 取任务结果
                                try:
                                    result = task.result()
                                except Exception as e:
                                    # 任务抛异常时打印并丢弃该组
                                    print(f"fully-async task raised: {e}", flush=True)
                                    return
                                # 结果类型不是 list[Sample] 时丢弃，避免污染训练数据
                                if not isinstance(result, list):
                                    print(
                                        f"fully-async: expected list[Sample], got {type(result).__name__}; dropping",
                                        flush=True,
                                    )
                                    return
                                # 合法结果连同组 id 放入输出队列
                                self.output_queue.put((gid, result))

                            # 返回该组的回调闭包
                            return task_done_callback

                        # 注册该组的完成回调
                        task.add_done_callback(make_callback(group_id))
                        # 加入在途任务集合
                        active_tasks.add(task)
                        break

                # Brief sleep to avoid busy waiting
                # 短暂休眠避免忙等
                await asyncio.sleep(1)

            except Exception as e:
                # 循环级异常兜底：打印后休眠重试，保证 worker 不因单次错误退出
                print(f"Error in continuous worker loop: {e}")
                await asyncio.sleep(1)

        # 退出前等待所有在途任务完成，避免结果丢失
        if active_tasks:
            print(f"Waiting for {len(active_tasks)} continuous tasks to complete...")
            await asyncio.wait(active_tasks)

        print("Continuous async rollout worker stopped")

    def worker_thread_func(self):
        """Worker function running in independent thread"""
        # 在独立线程中运行事件循环
        asyncio.run(self.continuous_worker_loop())

    def start(self):
        """Start continuous work mode"""
        # 线程不存在或已死亡时启动新的守护线程
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self.worker_thread_func, daemon=True)
            self.worker_thread.start()
            print("Started continuous async worker thread")

    def stop(self):
        """Stop worker thread"""
        # 置运行标志为 False 使主循环退出
        self.running = False
        # 清除暂停请求，让循环能走到退出分支
        self.pause_requested.clear()
        # 线程存活时等待其结束（最多 5 秒）
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        print("Stopped async worker thread")

    def pause_and_wait_idle(self, timeout: float | None = None):
        """Pause new training rollout submissions and wait for in-flight groups."""
        # 先清 idle 标记再请求暂停，保证等待语义正确
        self.idle.clear()
        self.pause_requested.set()
        print("Pausing async rollout worker before eval...", flush=True)
        # 线程存活时等待其进入空闲（在途任务全部完成）
        if self.worker_thread and self.worker_thread.is_alive():
            if not self.idle.wait(timeout=timeout):
                raise TimeoutError("Timed out waiting for async rollout worker to become idle before eval")
        print(f"Async rollout worker paused. Buffered groups: {self.get_queue_size()}", flush=True)

    def resume(self):
        """Resume training rollout submissions after eval."""
        # 清 idle 并解除暂停，恢复训练数据提交
        self.idle.clear()
        self.pause_requested.clear()
        print("Resumed async rollout worker after eval.", flush=True)

    def get_completed_groups(self) -> List[tuple]:
        """Get completed sample groups"""
        # 已完成组列表
        completed = []
        # 非阻塞地取空输出队列
        while True:
            try:
                result = self.output_queue.get_nowait()
                completed.append(result)
            except queue.Empty:
                break
        # 返回全部 (组id, 组样本) 结果
        return completed

    def get_queue_size(self) -> int:
        """Get current output queue size"""
        # 返回输出队列当前长度
        return self.output_queue.qsize()


# 全异步 rollout 生成：借助全局常驻 worker 持续采样，
# 从其输出队列收集 rollout_batch_size 个完成的样本组
async def generate_rollout_async(args, rollout_id: int, data_buffer) -> List[List[Sample]]:
    """
    Simplified asynchronous rollout generation - using global continuous worker
    """
    # 要求使用全局数据集（worker 依赖其缓冲策略）
    assert args.rollout_global_dataset

    # Get global worker, which will run continuously
    # 获取（或创建）持续运行的全局 worker
    worker = get_global_worker(args, data_buffer)

    # Simplified: directly use rollout_batch_size as target
    # 目标组数直接取 rollout_batch_size
    target_data_size = args.rollout_batch_size

    # 收集到的最终数据（按组）
    data = []
    # 已完成但尚未按序处理的组：组id -> 组样本
    completed_groups = {}
    # 是否已打印首个样本（只打印一次）
    do_print = True

    print(f"Starting async rollout generation for {target_data_size} groups")
    print(f"Global worker queue size: {worker.get_queue_size()}")

    # Main loop: collect results from global worker's output queue
    # 记录起始时间与最近进展时间，用于无进展告警
    start_time = time.time()
    last_progress_time = start_time
    no_progress_timeout = 30.0  # Warn if no progress for 30 seconds

    # 主循环：直到收齐目标组数
    while len(data) < target_data_size:
        # Collect completed results
        # 从 worker 输出队列取走全部已完成组
        completed = worker.get_completed_groups()

        # 本轮是否有新完成组
        made_progress = False
        # 把新完成组登记进缓存字典
        for group_id, group in completed:
            completed_groups[group_id] = group
            made_progress = True

        # 有进展则刷新最近进展时间
        if made_progress:
            last_progress_time = time.time()

        # Process completed groups in order (try to maintain order, but not strict requirement)
        # 本轮是否处理过任何组
        processed_any = False

        # Process all available completed groups
        # 遍历当前全部已完成组 id
        available_ids = list(completed_groups.keys())
        for group_id in available_ids:
            # 已收齐目标数量则停止处理
            if len(data) >= target_data_size:
                break

            # 从缓存取出该组
            group = completed_groups.pop(group_id)

            # If any sample in the group was aborted, return the whole group to the data buffer
            # and do not forward it to the training engine.
            # 检查组内是否有样本发生生成中断
            try:
                any_aborted = any(_sample_has_generation_abort(sample) for sample in group)
            except Exception:
                # 检查失败时按"未中断"处理
                any_aborted = False

            if any_aborted:
                try:
                    # 重置组内每个被中断的样本
                    for sample in group:
                        _reset_sample_for_generation_retry(sample)
                    # add back to buffer so it can be retried or handled by buffer policy
                    # 整组放回数据缓冲区，等待重试或按缓冲策略处理
                    data_buffer.add_samples([group])
                    print(f"Returned aborted group {group_id} to data buffer for retry", flush=True)
                except Exception as e:
                    # 放回失败仅打印告警，不中断主流程
                    print(f"Failed to return aborted group {group_id} to buffer: {e}", flush=True)
                # don't count as processed for training
                # 被中断的组不计入训练数据
                continue

            if do_print:
                print(
                    f"First rollout sample: {[str(group[0].prompt) + group[0].response]}, "
                    f"label: {group[0].label}, reward: {group[0].reward}",
                    flush=True,
                )
                do_print = False

            # Simplified: directly add samples, no filters used
            # 直接收录该组，不做额外过滤
            data.append(group)
            processed_any = True

        # Check progress
        # 检查是否长时间无进展
        current_time = time.time()
        if current_time - last_progress_time > no_progress_timeout:
            print(
                f"Warning: No progress for {no_progress_timeout}s. "
                f"Queue size: {worker.get_queue_size()}, "
                f"Collected: {len(data)}/{target_data_size}"
            )
            last_progress_time = current_time

        # If no results were processed, brief sleep to avoid busy waiting
        # 本轮没有处理任何结果时短暂休眠，避免忙等
        if not processed_any:
            await asyncio.sleep(0.01)

    # 统计并打印 rollout 总耗时与队列剩余量
    duration = time.time() - start_time
    print(f"Rollout completed in {duration:.2f}s! Global worker queue size: {worker.get_queue_size()}")

    if data:
        print(
            f"Finish rollout: {[str(data[-1][0].prompt) + data[-1][0].response]}, "
            f"label: {data[-1][0].label}, reward: {data[-1][0].reward}",
            flush=True,
        )

    # 按样本 index 排序，尽量恢复数据的原始顺序
    data = sorted(data, key=lambda group: group[0].index)
    return data


# 对外入口：全异步 rollout。评测时暂停训练 worker 独占执行 eval_rollout，
# 训练时经全局常驻 worker 收集 rollout_batch_size 个组
def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation=False):
    # 评测分支：先暂停训练采样，避免与评测争抢 SGLang 资源
    if evaluation:
        worker = get_existing_global_worker()
        if worker is not None:
            worker.pause_and_wait_idle()
        try:
            # eval_rollout currently returns (RolloutFnEvalOutput, []); call_rollout_fn also
            # accepts a bare RolloutFnEvalOutput / legacy dict.
            # 同步执行评测 rollout
            output = run(eval_rollout(args, rollout_id))
            # 兼容 (output, []) 二元组返回，取第一个元素
            if isinstance(output, tuple):
                output = output[0]
            # 返回评测结果
            return output
        finally:
            # 无论成败都恢复训练采样
            if worker is not None:
                worker.resume()

    # 训练分支：同步等待全异步 rollout 收齐目标组数
    completed_samples = run(generate_rollout_async(args, rollout_id, data_buffer))
    return completed_samples


# 进程退出时自动停止全局 worker，防止后台线程悬挂
atexit.register(stop_global_worker)
