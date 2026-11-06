# 导入 asyncio：提供异步事件循环与协程支持（本文件的 HTTP 轮询/重试都用 await asyncio.sleep 等异步原语）
import asyncio
# 导入 ipaddress：用于解析并判断主机字符串是否为合法的 IPv6 地址
import ipaddress
# 导入 json：将 HTTP 响应体（字节串）反序列化为 Python 对象
import json
# 导入 logging：标准库日志模块，用于输出重试告警等信息
import logging
# 导入 multiprocessing：管理子进程（如 sglang_router 路由进程）的生命周期
import multiprocessing
# 导入 os：读取环境变量（如 SLIME_HOST_IP、SLIME_PREFER_IPV6）
import os
# 导入 random：为端口探测生成随机起始偏移，降低多进程端口冲突概率
import random
# 导入 socket：执行端口绑定探测、主机名/IP 解析
import socket

# 导入 httpx：支持 async 的 HTTP 客户端库，作为全局推理请求客户端
import httpx

# 创建本模块专属的 logger，日志中会带上模块名便于定位来源
logger = logging.getLogger(__name__)

# 环境变量名：允许用户手动覆盖本机对外 IP（集群中网卡复杂时使用）
SLIME_HOST_IP_ENV = "SLIME_HOST_IP"


def find_available_port(base_port: int):
    # 从基准端口加一个 100~1000 的随机偏移开始探测，避免多个角色固定撞同一个端口
    port = base_port + random.randint(100, 1000)
    # 不断尝试，直到找到可用端口为止
    while True:
        # 用套接字试绑定，成功则返回该端口
        if is_port_available(port):
            return port
        # 端口未超上限则向下一段跳跃 42 继续找（步长非 1，减少相邻端口冲突）
        if port < 60000:
            port += 42
        else:
            # 超过 60000 后反向跳 43，回到较低端口区间继续搜索
            port -= 43


def is_port_available(port):
    """Return whether a port is available."""
    # 创建 IPv4 TCP 套接字，with 语句保证用完即关闭
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            # 设置 SO_REUSEADDR，允许复用处于 TIME_WAIT 的地址，探测更宽松
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # 尝试绑定到本机所有网卡的指定端口
            s.bind(("", port))
            # 再尝试进入监听状态，进一步确认端口可用
            s.listen(1)
            # 绑定+监听都成功，说明端口空闲
            return True
        except OSError:
            # 绑定失败（端口被占用等），端口不可用
            return False
        except OverflowError:
            # 端口号超出合法范围（0~65535），视为不可用
            return False


def get_host_info():
    # 获取本机主机名，作为 (hostname, ip) 返回值的第一个元素
    hostname = socket.gethostname()

    # 海象表达式：若设置了 SLIME_HOST_IP 环境变量，则直接用它覆盖自动探测的本机 IP
    if env_overwrite_local_ip := os.getenv(SLIME_HOST_IP_ENV, None):
        # 直接返回主机名与用户指定的 IP
        return hostname, env_overwrite_local_ip

    # 内部辅助函数：判断给定 IP 字符串是否为回环地址（127.x.x.x 或 IPv6 的 ::1）
    def _is_loopback(ip):
        # 回环地址不能作为集群内通信地址，需要过滤掉
        return ip.startswith("127.") or ip == "::1"

    # 内部辅助函数：按指定协议族（IPv4/IPv6）解析本机的局域网 IP
    def _resolve_ip(family, test_target_ip):
        """
        Attempt to get the local LAN IP for the specific family (IPv4/IPv6).
        Strategy: UDP Probe (Preferred) -> Hostname Resolution (Fallback) -> None
        """

        # Strategy 1: UDP Connect Probe (Most accurate, relies on routing table)
        # Useful when the machine has a default gateway or internet access.
        try:
            # 创建对应协议族的 UDP 套接字（UDP connect 不会真正发包，只查路由表）
            with socket.socket(family, socket.SOCK_DGRAM) as s:
                # The IP doesn't need to be reachable, but the routing table must exist.
                # 对测试目标 IP 的 80 端口做 connect，让内核选出默认路由对应的源 IP
                s.connect((test_target_ip, 80))
                # 取套接字本地地址中的 IP 部分（getsockname 返回 (ip, port)）
                ip = s.getsockname()[0]
                # 若得到的不是回环地址，即为可用的局域网 IP
                if not _is_loopback(ip):
                    return ip
        except Exception:
            pass  # Route unreachable or network error, move to next strategy.

        # Strategy 2: Hostname Resolution (Fallback for offline clusters)
        # Useful for offline environments where UDP connect fails but /etc/hosts is configured.
        try:
            # getaddrinfo allows specifying the family (AF_INET or AF_INET6)
            # Result format: [(family, type, proto, canonname, sockaddr), ...]
            # 离线集群回退方案：直接解析主机名（依赖 /etc/hosts 配置），并限定协议族
            infos = socket.getaddrinfo(hostname, None, family=family, type=socket.SOCK_STREAM)

            # 遍历所有解析结果，挑出第一个非回环地址
            for info in infos:
                # 取 sockaddr 五元组中的第一个元素，即 IP 地址
                ip = info[4][0]  # The first element of sockaddr is the IP
                # Must filter out loopback addresses to avoid "127.0.0.1" issues
                # 过滤回环地址，避免拿到 127.0.0.1 导致跨机不可达
                if not _is_loopback(ip):
                    return ip
        except Exception:
            # 解析失败则静默跳过，继续走最终回退
            pass

        # 两种策略都失败时返回 None，由调用方使用回环地址兜底
        return None

    # 读取是否偏好 IPv6 的开关（默认关闭，即优先 IPv4）
    prefer_ipv6 = os.getenv("SLIME_PREFER_IPV6", "0").lower() in ("1", "true", "yes", "on")
    # 初始化本机 IP 为 None，等待探测结果
    local_ip = None
    # 最终兜底地址：探测全部失败时使用的回环地址
    final_fallback = "127.0.0.1"

    if prefer_ipv6:
        # [Strict Mode] IPv6 Only
        # 1. Try UDP V6 Probe
        # 2. Try Hostname Resolution (V6)
        # If failed, fallback to V6 loopback. Never mix with V4.
        # 严格 IPv6 模式：仅用 IPv6 探测（UDP 探测目标 Google 的 IPv6 DNS），失败回退 ::1
        local_ip = _resolve_ip(socket.AF_INET6, "2001:4860:4860::8888")
        # 兜底地址改为 IPv6 回环
        final_fallback = "::1"
    else:
        # [Strict Mode] IPv4 Only (Default)
        # 1. Try UDP V4 Probe
        # 2. Try Hostname Resolution (V4)
        # If failed, fallback to V4 loopback. Never mix with V6.
        # 默认严格 IPv4 模式：用 IPv4 探测（目标 8.8.8.8），失败回退 127.0.0.1
        local_ip = _resolve_ip(socket.AF_INET, "8.8.8.8")
        # 兜底地址为 IPv4 回环
        final_fallback = "127.0.0.1"

    # 返回主机名与探测到的 IP；若探测为 None 则使用回环兜底地址
    return hostname, local_ip or final_fallback


def _wrap_ipv6(host):
    """Wrap IPv6 address in [] if needed."""
    try:
        # 去掉可能已有的方括号后验证是否为合法 IPv6 地址
        ipaddress.IPv6Address(host.strip("[]"))
        # 是 IPv6 则统一加上方括号（URL 中 IPv6 必须带括号）
        return f"[{host.strip('[]')}]"
    except ipaddress.AddressValueError:
        # 不是 IPv6（如普通 IPv4 或主机名），原样返回
        return host


def run_router(args):
    # 子进程入口：启动 sglang router，把退出码返回给父进程
    try:
        # 延迟导入 sglang_router，避免未安装路由组件时影响主流程
        from sglang_router.launch_router import launch_router

        # 用传入参数启动路由器，返回 router 实例（或 None 表示启动失败）
        router = launch_router(args)
        if router is None:
            # 启动失败返回非零退出码
            return 1
        # 启动成功返回 0（router 通常在内部阻塞运行）
        return 0
    except Exception as e:
        # 任何异常都记录日志并返回失败退出码
        logger.info(e)
        return 1


def terminate_process(process: multiprocessing.Process, timeout: float = 1.0) -> None:
    """Terminate a process gracefully, with forced kill as fallback.

    Args:
        process: The process to terminate
        timeout: Seconds to wait for graceful termination before forcing kill
    """
    # 进程已经退出则无需处理
    if not process.is_alive():
        return

    # 先发送 SIGTERM 请求优雅退出
    process.terminate()
    # 等待最多 timeout 秒让进程自行结束
    process.join(timeout=timeout)
    if process.is_alive():
        # 超时仍未退出则强制 SIGKILL 杀掉
        process.kill()
        # 再 join 等待其彻底回收，避免僵尸进程
        process.join()


# 模块级全局：共享的异步 HTTP 客户端（惰性创建，仅初始化一次）
_http_client: httpx.AsyncClient | None = None
# 记录客户端允许的最大并发连接数，用于 Ray 分布式 POST 的并发分摊计算
_client_concurrency: int = 0

# Optional Ray-based distributed POST dispatch
# 是否启用基于 Ray 的分布式 POST 分发开关
_distributed_post_enabled: bool = False
# Ray actor 列表：每个节点上部署的 HTTP POST actor
_post_actors: list[object] = []
# 轮询索引：用于在多个 actor 间做负载均衡
_post_actor_idx: int = 0


def _next_actor():
    # 声明修改模块级轮询索引
    global _post_actor_idx
    # 尚未初始化任何 actor 时返回 None，走本地请求路径
    if not _post_actors:
        return None
    # 按轮询（round-robin）方式取出当前 actor
    actor = _post_actors[_post_actor_idx % len(_post_actors)]
    # 索引前移并对 actor 数量取模，实现循环分发
    _post_actor_idx = (_post_actor_idx + 1) % len(_post_actors)
    # 返回本次应使用的 actor
    return actor


async def _post(client, url, payload, max_retries=60, headers=None):
    # 初始化已重试次数
    retry_count = 0
    # 最多重试 max_retries 次（sglang server 滚动更新期间请求会大量失败，需要长重试）
    while retry_count < max_retries:
        # 每轮循环开始时清空响应对象，便于 finally 中安全关闭
        response = None
        try:
            # 发送 POST 请求；payload 为空时发送空 JSON 对象
            response = await client.post(url, json=payload or {}, headers=headers)
            # 4xx/5xx 状态码抛出 HTTPStatusError，纳入统一重试逻辑
            response.raise_for_status()
            # 异步读取完整响应体字节
            content = await response.aread()
            try:
                # 优先把响应体解析为 JSON
                output = json.loads(content)
            except json.JSONDecodeError:
                # 非 JSON 响应则按文本返回（字节则解码为字符串）
                output = content.decode() if isinstance(content, bytes) else content
        except Exception as e:
            # 请求失败：累计重试次数
            retry_count += 1

            # HTTP 状态错误时额外提取响应体文本，方便排查服务端报错
            if isinstance(e, httpx.HTTPStatusError):
                response_text = e.response.text
            else:
                # 其他异常（连接失败等）没有响应体
                response_text = None

            # 打印错误与重试进度日志
            logger.info(
                f"Error: {e}, retrying... (attempt {retry_count}/{max_retries}, url={url}, response={response_text})"
            )
            if retry_count >= max_retries:
                # 达到最大重试次数，记录日志并抛出最后一次的异常
                logger.info(f"Max retries ({max_retries}) reached, failing... (url={url})")
                raise e
            # 每次重试间隔 1 秒，避免打爆服务
            await asyncio.sleep(1)
            # 继续下一轮重试
            continue
        finally:
            # 无论成功失败都关闭响应，释放连接回连接池
            if response is not None:
                await response.aclose()
        # 请求成功则跳出重试循环
        break

    # 返回解析后的响应内容
    return output


def init_http_client(args):
    """Initialize HTTP client and optionally enable distributed POST via Ray."""
    # 声明修改模块级全局状态
    global _http_client, _client_concurrency, _distributed_post_enabled
    # 未配置 rollout GPU 数时（非 rollout 角色）无需初始化客户端
    if not args.rollout_num_gpus:
        return

    # 计算客户端总并发 = 单 server 并发 × 总 GPU 数 / 每 engine 的 GPU 数
    _client_concurrency = args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
    # 客户端尚未创建时才创建（幂等，支持多次调用）
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            # 限制最大连接数与计算出的并发一致
            limits=httpx.Limits(max_connections=_client_concurrency),
            # 不设超时（生成请求可能非常长，交给重试机制兜底）
            timeout=httpx.Timeout(None),
        )

    # Optionally initialize distributed POST via Ray without changing interfaces
    # 若开启分布式 POST 开关，则初始化 Ray actor 并置位启用标志
    if args.use_distributed_post:
        _init_ray_distributed_post(args)
        _distributed_post_enabled = True


def _init_ray_distributed_post(args):
    """Initialize one or more Ray async actors per node for HTTP POST.

    Uses NodeAffinitySchedulingStrategy to place actors on distinct nodes.
    Controlled by SLIME_HTTP_POST_ACTORS_PER_NODE.
    """
    # 声明修改模块级 actor 列表
    global _post_actors
    # 已经初始化过则直接返回（幂等保护）
    if _post_actors:
        return  # Already initialized

    # 延迟导入 ray，只在启用分布式 POST 时才要求安装
    import ray
    # 导入节点亲和调度策略，确保 actor 固定落在指定节点
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    # Discover alive nodes
    # 过滤出 Ray 集群中所有存活节点
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    if not nodes:
        # 没有可用节点则无法部署 actor，直接报错
        raise RuntimeError("No alive Ray nodes to place HTTP POST actors.")

    # Define the async actor
    # 定义 Ray 异步 actor 类（remote 装饰器使其成为分布式 actor）
    @ray.remote
    class _HttpPosterActor:
        def __init__(self, concurrency: int):
            # Lazy creation to this actor's event loop
            # actor 内部持有独立的 httpx 异步客户端，连接上限按分配的并发设置
            self._client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=max(1, concurrency)),
                timeout=httpx.Timeout(None),
            )

        async def do_post(self, url, payload, max_retries=60, headers=None):
            # 复用模块级 _post 函数完成带重试的 POST
            return await _post(self._client, url, payload, max_retries, headers=headers)

    # Create actors per node
    # 收集创建好的 actor
    created = []
    # Distribute client concurrency across actors (at least 1 per actor)
    # 将客户端总并发均摊到每个节点（向上取整，保证至少为 1）
    per_actor_conc = (_client_concurrency + len(nodes)) // len(nodes)

    # 遍历每个存活节点部署 actor
    for node in nodes:
        # 取出节点 ID 用于亲和性调度
        node_id = node["NodeID"]
        # 构造硬亲和策略（soft=False 表示必须调度到该节点）
        scheduling = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        # 每个节点按 GPU 数量创建多个 actor，进一步分摊请求
        for _ in range(args.num_gpus_per_node):
            actor = _HttpPosterActor.options(
                name=None,
                # detached 生命周期：driver 退出后 actor 仍存活
                lifetime="detached",
                # 绑定到当前节点
                scheduling_strategy=scheduling,
                # actor 的最大并发处理能力
                max_concurrency=per_actor_conc,
                # Use tiny CPU to schedule
                # 只占用极小 CPU 配额，避免挤占训练资源
                num_cpus=0.001,
            ).remote(per_actor_conc)
            # 加入待登记列表
            created.append(actor)

    # 登记到模块级 actor 列表，供 post() 轮询使用
    _post_actors = created


async def post(url, payload, max_retries=60, headers=None):
    # If distributed mode is enabled and actors exist, dispatch via Ray.
    # 分布式 POST 已启用且 actor 就绪时，优先走 Ray 分发
    if _distributed_post_enabled and _post_actors:
        try:
            # 延迟导入 ray
            import ray

            # 轮询选择一个 actor
            actor = _next_actor()
            if actor is not None:
                # Use a thread to avoid blocking the event loop on ray.get
                # 远程调用 actor 的 do_post，返回 obj_ref
                obj_ref = actor.do_post.remote(url, payload, max_retries, headers=headers)
                # ray.get 是阻塞调用，放入线程池执行以免卡住本事件循环
                return await asyncio.to_thread(ray.get, obj_ref)
        except Exception as e:
            # 分布式路径失败时记录日志，回退到本地 HTTP 客户端
            logger.info(f"[http_utils] Distributed POST failed, falling back to local: {e} (url={url})")
            # fall through to local

    # 默认（或回退）路径：用本地共享客户端发送带重试的 POST
    return await _post(_http_client, url, payload, max_retries, headers=headers)


async def get(url):
    # 用共享客户端发送 GET 请求
    response = await _http_client.get(url)
    # 非 2xx 状态码直接抛错（GET 失败不做重试）
    response.raise_for_status()
    # 异步读取响应体字节
    content = await response.aread()
    # 将响应体解析为 JSON 后返回
    output = json.loads(content)
    return output
