# rollout 引擎健康监控模块:后台线程定期对每个推理引擎做 health_generate 探测,
# 失败则杀掉对应 actor;支持在权重 offload 时暂停检查、onload 时恢复。
import logging
import threading

import ray


# 模块级 logger
logger = logging.getLogger(__name__)


class RolloutHealthMonitor:
    """Health monitor for rollout engines.

    The monitor runs continuously once started, but can be paused/resumed
    based on whether the engines are offloaded (cannot health check when offloaded).

    Lifecycle:
    - start(): Start the monitor thread (called once during initialization)
    - pause(): Pause health checking (called when offloading engines)
    - resume(): Resume health checking (called when onloading engines)
    - stop(): Stop the monitor thread completely (called during dispose)
    """

    def __init__(self, engine_group, args):
        # 保存引擎组引用,用于枚举引擎与执行 kill
        self._engine_group = engine_group

        # 监控线程句柄,初始未创建
        self._thread = None
        # 停止事件:置位后监控循环退出
        self._stop_event = None
        # 暂停事件:置位(set)表示暂停检查,清除(clear)表示恢复检查
        self._pause_event = None  # When set, health checking is paused
        # 健康检查轮询间隔(秒)
        self._check_interval = args.rollout_health_check_interval
        # 单次健康检查的超时时间(秒)
        self._check_timeout = args.rollout_health_check_timeout
        # 每次 resume 后的首个宽限期(秒),给大模型编译/初始化留时间
        self._check_first_wait = args.rollout_health_check_first_wait
        # 标记是否需要在(每次)resume 后执行首轮等待
        self._need_first_wait = True  # Need to wait after each resume
        # 当前健康检查是否处于启用状态(用于外部查询)
        self._is_checking_enabled = False  # Track if health checking should be active

    def start(self) -> bool:
        """Start the health monitor thread. Called once during initialization.

        Returns:
            True if the monitor was started, False if there are no engines to monitor.
        """
        if not self._engine_group.all_engines:
            # 没有引擎可监控时无需启动,返回 False
            return False

        if self._thread is not None:
            # 线程已在运行,避免重复启动
            logger.warning("Health monitor thread is already running.")
            return True

        logger.info("Starting RolloutHealthMonitor...")
        # 初始化停止事件
        self._stop_event = threading.Event()
        # 初始化暂停事件
        self._pause_event = threading.Event()
        self._pause_event.set()  # Start in paused state until resume() is called
        # 创建监控守护线程(daemon=True:主进程退出时自动结束)
        self._thread = threading.Thread(
            target=self._health_monitor_loop,
            name="RolloutHealthMonitor",
            daemon=True,
        )
        # 启动监控线程
        self._thread.start()
        logger.info("RolloutHealthMonitor started (in paused state).")
        return True

    def stop(self) -> None:
        """Stop the health monitor thread completely. Called during dispose."""
        if not self._thread:
            # 线程不存在时无需停止
            return

        logger.info("Stopping RolloutHealthMonitor...")
        assert self._stop_event is not None
        # 置位停止事件,通知循环退出
        self._stop_event.set()
        # Also clear pause to let the thread exit
        # 同时清除暂停状态,确保线程不会被卡在内层暂停等待里
        if self._pause_event:
            self._pause_event.clear()
        # 等待上限 = 单次检查超时 + 检查间隔 + 5 秒缓冲
        timeout = self._check_timeout + self._check_interval + 5
        # 阻塞等待线程退出
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            # 超时后线程仍存活,记录告警
            logging.warning("Rollout health monitor thread did not terminate within %.1fs", timeout)
        else:
            # 正常退出
            logger.info("RolloutHealthMonitor stopped.")

        # 清理线程与事件引用,复位检查启用标记
        self._thread = None
        self._stop_event = None
        self._pause_event = None
        self._is_checking_enabled = False

    def pause(self) -> None:
        """Pause health checking. Called when engines are offloaded."""
        if self._pause_event is None:
            # 尚未 start 时无法暂停
            return
        logger.info("Pausing health monitor...")
        # 置位暂停事件,循环将进入等待
        self._pause_event.set()
        # 同步更新启用状态标记
        self._is_checking_enabled = False

    def resume(self) -> None:
        """Resume health checking. Called when engines are onloaded."""
        if self._pause_event is None:
            # 尚未 start 时无法恢复
            return
        logger.info("Resuming health monitor...")
        self._need_first_wait = True  # Need to wait after each resume
        # 清除暂停事件,恢复检查
        self._pause_event.clear()
        # 同步更新启用状态标记
        self._is_checking_enabled = True

    def is_checking_enabled(self) -> bool:
        """Return whether health checking is currently enabled (not paused)."""
        # 返回当前健康检查是否启用
        return self._is_checking_enabled

    def _health_monitor_loop(self) -> None:
        assert self._stop_event is not None
        assert self._pause_event is not None

        # 主循环:直到停止事件被置位
        while not self._stop_event.is_set():
            # Wait while paused
            # 暂停状态下以 0.5 秒为粒度空转等待(同时响应停止事件)
            while self._pause_event.is_set() and not self._stop_event.is_set():
                self._stop_event.wait(timeout=0.5)

            if self._stop_event.is_set():
                # 等待期间收到停止信号则退出
                break

            # Do first wait after each resume (for large MoE models to be ready)
            # resume 后的首轮宽限等待(给大 MoE 模型留出就绪时间)
            if self._need_first_wait:
                logger.info(f"Health monitor doing first wait after resume: {self._check_first_wait}s")
                if self._stop_event.wait(self._check_first_wait):
                    # 等待期间收到停止信号则退出
                    logger.info("Health monitor stopped during first wait.")
                    break
                if self._pause_event.is_set():
                    # Got paused during first wait, skip this round and wait again next resume
                    # 宽限期内又被暂停,跳过本轮并在下次 resume 时重新等待
                    logger.info("Health monitor paused during first wait, will wait again next resume.")
                    continue
                # 首轮等待完成,清除标记
                self._need_first_wait = False

            # Run health checks
            # 确认仍处于启用且未停止状态时执行一轮检查
            if not self._pause_event.is_set() and not self._stop_event.is_set():
                self._run_health_checks()

            # Wait for next check interval
            # 等待下一个检查周期(wait 返回 True 表示停止事件被置位)
            if self._stop_event.wait(self._check_interval):
                break

    def _run_health_checks(self) -> None:
        # 逐个引擎执行健康检查
        for rollout_engine_id, engine in enumerate(self._engine_group.engines):
            if self._stop_event is not None and self._stop_event.is_set():
                # 检查过程中收到停止信号则提前结束
                break
            if self._pause_event is not None and self._pause_event.is_set():
                # 检查过程中被暂停则提前结束
                break
            # 对单个引擎执行健康检查
            self._check_engine_health(rollout_engine_id, engine)

    def _check_engine_health(self, rollout_engine_id, engine) -> None:
        if engine is None:
            # 已被 kill 的引擎槽位为 None,跳过
            logger.info(f"Skipping health check for engine {rollout_engine_id} (None)")
            return

        try:
            # 通过 Ray 远程调用 health_generate,超时时间受 _check_timeout 约束
            ray.get(engine.health_generate.remote(timeout=self._check_timeout))
        except Exception as e:
            # 超时或其他异常都视为引擎不可用,记录错误并杀掉该引擎 actor
            logger.error(
                f"Health check failed for rollout engine {rollout_engine_id} (ray timeout or error). Killing actor. Exception: {e}"
            )
            self._kill_engine(rollout_engine_id=rollout_engine_id)
        else:
            # 检查通过,仅记录调试日志
            logger.debug(f"Health check passed for rollout engine {rollout_engine_id}")

    def _kill_engine(self, rollout_engine_id: int):
        logger.info(f"Killing engine group {rollout_engine_id}...")
        # 遍历该引擎组覆盖的全部底层引擎(按每引擎节点数展开)
        for i in range(
            rollout_engine_id * self._engine_group.nodes_per_engine,
            (rollout_engine_id + 1) * self._engine_group.nodes_per_engine,
        ):
            # 取出对应下标的底层引擎
            engine = self._engine_group.all_engines[i]
            if engine:
                logger.info(f"Shutting down and killing engine at index {i}")
                try:
                    # 先优雅关闭(释放资源),再强制 kill 该 Ray actor
                    ray.get(engine.shutdown.remote())
                    ray.kill(engine)
                    logger.info(f"Successfully killed engine at index {i}")
                except Exception as e:
                    # 关闭/kill 失败仅告警,不中断整体流程
                    logger.warning(f"Fail to kill engine at index {i} (e: {e})")
            else:
                # 该槽位已经是 None(此前已被 kill)
                logger.info(f"Engine at index {i} is already None")
            # 无论结果如何都把槽位置空,避免后续再用到失效引用
            self._engine_group.all_engines[i] = None
