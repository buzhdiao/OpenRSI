# 本模块提供一个全局后台事件循环线程,用于在同步代码中调度并等待协程完成。
import asyncio
import threading

# 声明本模块对外暴露的公共接口
__all__ = ["get_async_loop", "run"]


# Create a background event loop thread
# 封装一个专用于运行事件循环的后台守护线程
class AsyncLoopThread:
    def __init__(self):
        # 为该线程新建一个独立的事件循环对象
        self.loop = asyncio.new_event_loop()
        # 创建守护线程(daemon=True:主进程退出时线程自动结束,不阻塞退出)
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        # 启动后台线程,事件循环随之开始运转
        self._thread.start()

    def _start_loop(self):
        # 把新建的事件循环绑定到当前子线程
        asyncio.set_event_loop(self.loop)
        # 永久运行事件循环,等待外部提交协程
        self.loop.run_forever()

    def run(self, coro):
        # Schedule a coroutine onto the loop and block until it's done
        # 将协程线程安全地提交到后台事件循环执行,并阻塞当前线程直到其完成、返回结果
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()


# Create one global instance
# 全局单例占位:延迟到首次使用时才创建,避免 import 阶段就启动线程
async_loop = None


def get_async_loop():
    # 声明修改模块级全局变量
    global async_loop
    # 懒加载单例:仅在第一次调用时创建后台事件循环线程
    if async_loop is None:
        async_loop = AsyncLoopThread()
    # 返回全局唯一的 AsyncLoopThread 实例
    return async_loop


def run(coro):
    """Run a coroutine in the background event loop."""
    # 便捷入口:取全局事件循环线程并在其中同步地运行协程
    return get_async_loop().run(coro)
