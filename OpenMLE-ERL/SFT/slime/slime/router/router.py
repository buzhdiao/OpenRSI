# Slime 自研路由器:以 FastAPI 服务的形式在多个 SGLang 推理引擎(worker)之间做
# 最少活跃请求数的负载均衡,并带后台健康检查与故障隔离(隔离 dead worker)。
import argparse
import asyncio
import json
import logging

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from slime.utils.misc import load_function

# 模块级 logger,用于路由器内部日志
logger = logging.getLogger(__name__)


def run_router(args):
    """
    Run the Slime router with the specified configuration.
    """
    # Initialize the router with tokenizer and lazy worker initialization
    # 初始化路由器实例(worker 懒加载,此刻不发起任何连接)
    slime_router = SlimeRouter(args, verbose=False)

    # Start the server
    # 用 uvicorn 启动 FastAPI 应用,监听参数指定的 IP 与端口
    uvicorn.run(slime_router.app, host=args.sglang_router_ip, port=args.sglang_router_port, log_level="info")


class SlimeRouter:
    def __init__(self, args, verbose=False):
        """Initialize the slime-router with SGLang router address"""
        # 保存全局配置对象,供健康检查等逻辑读取参数
        self.args = args
        # 是否打印详细调试信息
        self.verbose = verbose

        # 创建 FastAPI 应用实例
        self.app = FastAPI()
        # 注册应用启动时的生命周期钩子:启动后台健康检查循环
        self.app.add_event_handler("startup", self._start_background_health_check)

        # URL -> Active Request Count (load state)
        # 记录每个 worker 的活跃请求数,同时隐式充当已注册 worker 的路由表
        self.worker_request_counts: dict[str, int] = {}
        # URL -> Consecutive Failures
        # 记录每个 worker 连续健康检查失败的次数(用于判定是否隔离)
        self.worker_failure_counts: dict[str, int] = {}
        # Quarantined workers excluded from routing pool
        # 已被隔离的 worker 集合,不再参与请求分发
        self.dead_workers: set[str] = set()
        # 记录最新的模型权重版本号(当前实现中未使用,保留扩展位)
        self.max_weight_version = None

        # 读取可配置的最大 HTTP 连接数;未配置时按推理并发度估算
        max_connections = getattr(args, "slime_router_max_connections", None)
        if max_connections is None:
            # 默认值 = 单引擎并发度 × GPU 总数 ÷ 每引擎 GPU 数,即全部引擎并发之和
            max_connections = (
                args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
            )

        # 读取 HTTP 请求超时时间(秒);None 表示使用 httpx 默认值
        timeout = getattr(args, "slime_router_timeout", None)

        # 创建异步 HTTP 客户端:限制最大连接数并设置超时,供代理转发与健康检查共用
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_connections),
            timeout=httpx.Timeout(timeout),
        )

        # 注册所有 HTTP 路由
        self._setup_routes()

        # 遍历用户配置的中间件路径列表,逐个动态加载并挂载
        for middleware_path in args.slime_router_middleware_paths or []:
            if self.verbose:
                # 详细模式下打印正在加载的中间件路径
                print(f"[slime-router] Loading middleware from: {middleware_path}")
            # 按形如 "module.func" 的路径动态导入中间件工厂函数
            middleware = load_function(middleware_path)
            # 把中间件加入 FastAPI 应用,并传入 router 自身供其访问路由状态
            self.app.add_middleware(middleware, router=self)

    def _setup_routes(self):
        """Setup all the HTTP routes"""
        # sglang-router api
        # 注册与 sglang-router 兼容的管理接口
        self.app.post("/add_worker")(self.add_worker)
        self.app.get("/list_workers")(self.list_workers)
        self.app.post("/retrieve_from_text")(self.retrieve_from_text)
        # Catch-all route for proxying to SGLang - must be registered LAST
        # 兜底路由:其余所有路径全部代理到某个 SGLang worker;必须最后注册以免遮蔽上面的具体路由
        self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])(self.proxy)

    async def _start_background_health_check(self):
        # 应用启动时创建独立的异步任务来跑健康检查死循环
        asyncio.create_task(self._health_check_loop())

    async def _check_worker_health(self, url):
        """Encapsulated health check logic for better maintainability."""
        try:
            # 向 worker 的 /health 端点发 GET 请求,5 秒超时
            response = await self.client.get(f"{url}/health", timeout=5.0)
            if response.status_code == 200:
                # 状态码 200 视为健康,返回 (url, True)
                return url, True
            # 非 200 视为不健康,记录调试日志
            logger.debug(f"[slime-router] Worker {url} is unhealthy (Status: {response.status_code})")
        except Exception as e:
            # 网络异常/超时等同样视为不健康
            logger.debug(f"[slime-router] Worker {url} health check failed: {e}")
        # 走到这里说明检查失败,返回 (url, False)
        return url, False

    async def _health_check_loop(self):
        """Background loop to monitor worker health and adjust routing pool."""
        # 健康检查的轮询间隔(秒)
        interval = self.args.rollout_health_check_interval
        # 连续失败多少次后把 worker 标记为 dead
        threshold = self.args.slime_router_health_check_failure_threshold

        while True:
            try:
                # 先休眠一个间隔再开始本轮检查
                await asyncio.sleep(interval)

                # 取出当前所有未被隔离的 worker URL 作为本轮检查对象
                urls = [u for u in self.worker_request_counts if u not in self.dead_workers]
                if not urls:
                    # 没有任何存活 worker 时跳过本轮
                    continue

                # 并发地对所有 worker 执行健康检查
                results = await asyncio.gather(*(self._check_worker_health(url) for url in urls))

                # 逐个处理检查结果
                for url, is_healthy in results:
                    if not is_healthy:
                        # 失败则累计连续失败次数
                        failures = self.worker_failure_counts.get(url, 0) + 1
                        self.worker_failure_counts[url] = failures

                        if failures >= threshold:
                            # 连续失败达到阈值,把该 worker 加入隔离名单并告警
                            logger.warning(
                                f"[slime-router] Worker {url} failed {threshold} consecutive health checks. Marking as DEAD."
                            )
                            self.dead_workers.add(url)
                            # TODO (chenyang): Connect back 'dead' workers requires a mechanism to sync
                            # model versions to avoid off-policy issues from stale weights, since these
                            # dead workers' parameters may not be refitted.
                    else:
                        # 检查通过则清零失败计数(连续性计数语义)
                        self.worker_failure_counts[url] = 0

                # 打印当前健康 worker 数量的调试日志
                logger.debug(
                    f"[slime-router] Health check complete. {len(self.worker_request_counts) - len(self.dead_workers)} workers healthy."
                )

            except asyncio.CancelledError:
                # 任务被取消时记录日志并继续抛出,保证正常退出
                logger.warning("[slime-router] Background health check loop is being cancelled.")
                raise
            except Exception as e:
                # 捕获其他未知异常,记录错误并稍作等待后继续循环,避免后台线程整体崩溃
                logger.error(f"[slime-router] Unexpected error in health check loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def proxy(self, request: Request, path: str):
        """Proxy all other requests to the SGLang router"""
        # Forward all other paths to SGLang router
        # 按负载均衡策略挑选一个 worker,并把原路径拼接到其 URL 后
        worker_url = self._use_url()
        url = f"{worker_url}/{path}"

        # Get request body and headers
        # 读取原始请求体,并把请求头转成普通 dict 以便转发
        body = await request.body()
        headers = dict(request.headers)

        try:
            # 用相同的 HTTP 方法和内容向选中的 worker 发起转发请求
            response = await self.client.request(request.method, url, content=body, headers=headers)
            # Eagerly read content so we can return JSON (not streaming)
            # 立即读完响应体,以便构造完整的 JSON/普通响应(而非流式)
            content = await response.aread()
            # 取出响应的 content-type,供非 JSON 回退时使用
            content_type = response.headers.get("content-type", "")
            try:
                # Prefer parsing JSON if possible
                # 优先尝试把响应体解析为 JSON 并以 JSONResponse 返回
                data = json.loads(content)
                return JSONResponse(
                    content=data,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )
            except Exception:
                # Fall back to raw body with original content type
                # 解析失败则按原始字节与原 content-type 原样返回
                return Response(
                    content=content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=content_type or None,
                )
            finally:
                if response is not None:
                    # 无论成功与否都释放底层连接
                    await response.aclose()

        finally:
            # 请求结束(含异常),把该 worker 的活跃请求数减一
            self._finish_url(worker_url)

    async def add_worker(self, request: Request):
        """Add a new worker to the router.
        Supports providing the URL via query string or JSON body.
        Examples:
        - POST /add_worker?url=http://127.0.0.1:10090
        - POST /add_worker  with body {"url": "http://127.0.0.1:10090"}
        """
        # 1) Prefer query param
        # 优先从查询串中取 worker URL(兼容 url / worker_url 两种参数名)
        worker_url = request.query_params.get("url") or request.query_params.get("worker_url")

        # 2) Fallback to JSON body
        # 查询串没有时,回退到读取 JSON 请求体
        if not worker_url:
            body = await request.body()
            payload = json.loads(body) if body else {}
            worker_url = payload.get("url") or payload.get("worker_url")

        if not worker_url:
            # 两种方式都拿不到 URL 时返回 400 错误
            return JSONResponse(
                status_code=400, content={"error": "worker_url is required (use query ?url=... or JSON body)"}
            )

        # Add if new, keep a simple request count per worker
        # 只在首次注册时初始化计数器,重复添加同一 URL 无副作用
        if worker_url not in self.worker_request_counts:
            self.worker_request_counts[worker_url] = 0
            self.worker_failure_counts[worker_url] = 0
            if self.verbose:
                # 详细模式下打印新注册的 worker
                print(f"[slime-router] Added new worker: {worker_url}")

        # 返回成功状态与当前全部 worker 的计数字典
        return {"status": "success", "worker_urls": self.worker_request_counts}

    async def list_workers(self, request: Request):
        """List all registered workers"""
        # 返回当前已注册的全部 worker URL 列表
        return {"urls": list(self.worker_request_counts.keys())}

    async def retrieve_from_text(self, request: Request):
        """Get token information from text input"""
        # 读取并解析 JSON 请求体
        body = await request.body()
        payload = json.loads(body) if body else {}

        # 取出待处理的文本,默认空串
        text = payload.get("text", "")

        # Use radix tree's retrieve_from_text method (no need to fetch weight version here)
        # 调用基数树(radix tree)的检索方法,对文本做分词并附带返回各 token 的对数概率
        token_ids, logp, loss_mask = self.radix_tree.retrieve_from_text(text, return_logprob=True)

        # Handle the result based on whether logp was requested
        # 组装返回结果:token 序列、原始文本、损失掩码及各自的长度信息
        result = {
            "tokens": token_ids,  # token IDs
            "response": text,  # The input text
            "loss_mask": loss_mask,  # Loss mask for the tokens
            "token_length": len(token_ids),
            "loss_mask_length": len(loss_mask),
            "rollout_logp": logp,
        }

        # 返回检索结果
        return result

    def _use_url(self):
        """Select worker URL with minimal active requests."""

        if not self.dead_workers:
            # Healthy path: select from all workers
            # 常规路径:没有被隔离的 worker,直接在全部 worker 中选活跃请求数最小者(最少连接负载均衡)
            url = min(self.worker_request_counts, key=self.worker_request_counts.get)
        else:
            # Degraded path: select from workers not in dead_workers
            # 降级路径:先过滤掉被隔离的 worker,再在剩余集合里选最小者
            valid_workers = (w for w in self.worker_request_counts if w not in self.dead_workers)
            try:
                # 对生成器求 min;若集合为空,min 会抛 ValueError
                url = min(valid_workers, key=self.worker_request_counts.get)
            except ValueError:
                # 没有任何健康 worker 可用,抛出运行时错误
                raise RuntimeError("No healthy workers available in the pool") from None

        # 选中后把该 worker 的活跃请求数加一,占用一个连接槽位
        self.worker_request_counts[url] += 1
        # 返回选中的 worker URL
        return url

    def _finish_url(self, url):
        """Mark the request to the given URL as finished"""
        # 校验该 URL 确实是已注册的 worker
        assert url in self.worker_request_counts, f"URL {url} not recognized"
        # 请求完成,活跃请求数减一
        self.worker_request_counts[url] -= 1
        # 防御性断言:计数不允许出现负数
        assert self.worker_request_counts[url] >= 0, f"URL {url} count went negative"


if __name__ == "__main__":
    # 独立启动路由器时的命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--sglang-host", type=str, required=True)
    parser.add_argument("--sglang-port", type=int, required=True)
    parser.add_argument("--tokenizer-name", type=str, help="Name of the tokenizer to use for tokenization")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # 解析命令行参数
    args = parser.parse_args()

    # Run the router
    # 启动路由器服务
    run_router(args)
