# 导入 dataclasses，用于枚举 ServerArgs 的字段做参数合并
import dataclasses
# 导入 ipaddress，用于解析与判断 IPv4/IPv6 地址
import ipaddress
# 导入 logging，用于模块级日志
import logging
# 导入 multiprocessing，用于以子进程方式启动 sglang 服务器
import multiprocessing
# 导入 os，读取 CUDA_VISIBLE_DEVICES 等环境变量
import os
# 导入 time，用于健康检查轮询休眠
import time
# 从 urllib 导入 quote，用于 URL 编码（删除路由 worker 时转义 URL）
from urllib.parse import quote

# 导入 requests，用于与 sglang HTTP 服务通信
import requests
# 导入 sglang_router，读取其版本号以适配不同路由器 API
import sglang_router
# 导入 packaging 的 parse，做语义化版本比较
from packaging.version import parse
# 导入 sglang 的服务器参数类，构造启动参数
from sglang.srt.server_args import ServerArgs
# 导入 sglang 工具函数：杀掉进程树（含子进程）
from sglang.srt.utils import kill_process_tree
# 导入 urllib3 的连接错误类型，用于 flush_cache 的异常细分
from urllib3.exceptions import NewConnectionError

# 导入 RayActor 基类，SGLangEngine 是一个 Ray 远程 actor
from slime.ray.ray_actor import RayActor
# 导入 HTTP 工具：get_host_info 获取本机主机名与 IP
from slime.utils.http_utils import get_host_info

# 创建本模块的日志器
logger = logging.getLogger(__name__)


# 计算某 rollout rank 在本节点上的 GPU 起始编号
def get_base_gpu_id(args, rank):
    # 每引擎实际使用的 GPU 数取"每节点 GPU 数"与"每引擎 GPU 数"的较小值
    num_gpus = min(args.num_gpus_per_node, args.rollout_num_gpus_per_engine)
    # 共置（colocate）模式：训练与 rollout 共享 GPU，按 rank 轮转分配
    if args.colocate:
        # 起始编号 = (rank × 每引擎 GPU 数) 对每节点 GPU 数取模
        start_index = (rank * num_gpus) % args.num_gpus_per_node
    else:
        # 分离模式：rollout GPU 排在所有 actor（训练）GPU 之后
        # 训练侧 GPU 总数；debug_rollout_only 时训练侧为 0
        num_actor_gpus = 0 if args.debug_rollout_only else args.actor_num_gpus_per_node * args.actor_num_nodes
        # 起始编号 = actor GPU 数 + rank 偏移，再对节点 GPU 数取模
        start_index = (num_actor_gpus + rank * num_gpus) % args.num_gpus_per_node
        # 若启用 critic（批评家模型），还需跳过 critic 占用的 GPU
        if args.use_critic:
            # 计算 critic 的 GPU 总数
            num_critic_gpus = args.critic_num_gpus_per_node * args.critic_num_nodes
            # 起始编号再往后偏移 critic 的 GPU 数，并取模
            start_index = (num_actor_gpus + num_critic_gpus + rank * num_gpus) % args.num_gpus_per_node
    # 返回本 rank 的 GPU 起始编号
    return start_index


# 把物理 GPU 编号转换为 CUDA_VISIBLE_DEVICES 重映射后的本地编号
def _to_local_gpu_id(physical_gpu_id: int) -> int:
    # 读取环境变量 CUDA_VISIBLE_DEVICES
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    # 未设置时无需重映射
    if not cvd:
        # 直接返回物理编号
        return physical_gpu_id  # no remapping
    # CUDA_VISIBLE_DEVICES can be like "4,5,6,7"
    # （原注释含义：环境变量形如 "4,5,6,7"）
    # 解析出所有可见的物理 GPU 编号列表
    visible = [int(x) for x in cvd.split(",") if x.strip() != ""]
    # In a remapped process, valid torch device indices are 0..len(visible)-1
    # （原英文注释：重映射进程中合法的 torch 设备编号是 0..len-1）
    # 物理编号在可见列表中：返回它在列表中的下标作为本地编号
    if physical_gpu_id in visible:
        # 例如 visible=[4,5,6,7] 时物理 6 → 本地 2
        return visible.index(physical_gpu_id)
    # If we're already getting local IDs, allow them
    # （原英文注释：如果传入的已经是本地编号，也接受）
    # 传入编号落在本地编号范围内时直接使用
    if 0 <= physical_gpu_id < len(visible):
        # 原样返回本地编号
        return physical_gpu_id
    # 两种解释都不成立则抛出运行时错误，说明非法 GPU 编号
    raise RuntimeError(
        f"GPU id {physical_gpu_id} is not valid under CUDA_VISIBLE_DEVICES={cvd}. "
        f"Expected one of {visible} (physical) or 0..{len(visible)-1} (local)."
    )


# 以独立子进程方式启动 sglang HTTP 服务器并等待其就绪
def launch_server_process(server_args: ServerArgs) -> multiprocessing.Process:
    # 延迟导入 sglang 的服务器入口，避免在父进程引入过重依赖
    from sglang.srt.entrypoints.http_server import launch_server

    # 强制使用 spawn 启动方式，保证 CUDA 子进程安全
    multiprocessing.set_start_method("spawn", force=True)
    # 去除 host 两端的方括号（IPv6 场景），服务器内部自行处理
    server_args.host = server_args.host.strip("[]")
    # 创建子进程，目标函数为 launch_server，参数为服务器配置
    p = multiprocessing.Process(target=launch_server, args=(server_args,))
    # 启动子进程
    p.start()

    # 多节点时只有 node 0 负责等待健康检查，其他节点直接返回
    if server_args.node_rank != 0:
        # 非 0 号节点不等待，直接返回进程句柄
        return

    # 阻塞等待服务器健康（传入存活检查回调）
    _wait_server_healthy(
        base_url=server_args.url(),
        api_key=server_args.api_key,
        is_process_alive=lambda: p.is_alive(),
    )

    # 返回服务器进程句柄，供后续 shutdown 使用
    return p


# 等待 sglang 服务器就绪：先 /health_generate 成功，再 /flush_cache 成功
def _wait_server_healthy(base_url, api_key, is_process_alive):
    # 构造请求头：JSON 类型 + Bearer 鉴权（api_key 可能为 None）
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key}",
    }

    # 使用会话复用 TCP 连接，减少握手开销
    with requests.Session() as session:
        # 无限循环直到健康检查通过
        while True:
            # 尝试请求健康检查接口（实际生成一次 token 验证推理链路）
            try:
                # GET /health_generate
                response = session.get(f"{base_url}/health_generate", headers=headers)
                # 返回 200 表示服务已就绪
                if response.status_code == 200:
                    # 跳出健康检查循环
                    break
            except requests.RequestException:
                # 网络异常（服务尚未启动）视为未就绪，继续轮询
                pass

            # 若服务器进程已死亡则立即报错，避免无限等待
            if not is_process_alive():
                # 抛出异常说明进程意外退出
                raise Exception("Server process terminated unexpectedly.")

            # 休眠 2 秒后重试
            time.sleep(2)

        # use flush_cache to make sure the working queue is empty, so that we can do offload
        # （原英文注释：用 flush_cache 确保请求队列为空以便 offload）
        # 第二阶段循环：等 flush_cache 成功（表明服务完全可用）
        while True:
            # 尝试调用 flush_cache 接口
            try:
                # GET /flush_cache
                response = session.get(f"{base_url}/flush_cache", headers=headers)
                # 200 表示缓存已清空、队列空闲
                if response.status_code == 200:
                    # 跳出循环，健康检查完成
                    break

            except requests.RequestException:
                # 忽略网络异常继续轮询
                pass

            # 进程死亡则报错
            if not is_process_alive():
                # 抛出异常说明进程意外退出
                raise Exception("Server process terminated unexpectedly.")

            # 休眠 2 秒后重试
            time.sleep(2)


# SGLang 引擎的 Ray actor 封装：负责引擎生命周期与全部 HTTP 交互
class SGLangEngine(RayActor):
    # 构造函数：只保存配置与基本信息，真正的启动延迟到 init()
    def __init__(self, args, rank: int, worker_type: str = "regular", base_gpu_id: int | None = None):
        # 保存全局配置
        self.args = args
        # 保存引擎 rank
        self.rank = rank
        # 保存 worker 类型：regular/prefill/decode（PD 分离时区分）
        self.worker_type = worker_type
        # 保存指定的 GPU 起始编号（可为 None，由 rank 自动推导）
        self.base_gpu_id = base_gpu_id

    # 初始化引擎：计算服务器参数并启动/连接服务器
    def init(self, dist_init_addr, port, nccl_port, host=None, disaggregation_bootstrap_port=None):
        # 记录路由器 IP（可能为 None，表示不用路由器）
        self.router_ip = self.args.sglang_router_ip
        # 记录路由器端口
        self.router_port = self.args.sglang_router_port

        # 未显式指定 host 时，自动探测本机 IP
        host = host or get_host_info()[1]

        # 内部函数：为 IPv6 地址补上方括号（URI 中必需）
        def _format_v6_uri(addr):
            # 空地址或已带方括号则原样返回
            if not addr or addr.startswith("["):
                # 原样返回
                return addr
            # 尝试解析地址
            try:
                # 解析成功且是 IPv6 时加方括号
                if ipaddress.ip_address(addr).version == 6:
                    # 返回带方括号的形式
                    return f"[{addr}]"
            except ValueError:
                # 不是合法 IP（如主机名）则原样返回
                pass
            # 默认原样返回
            return addr

        # 对 host 做同样的 IPv6 格式化
        host = _format_v6_uri(host)
        # 从分布式初始化地址中按最后一个冒号拆出 IP 与端口
        ip_part, port_part = dist_init_addr.rsplit(":", 1)
        # IP 部分做 IPv6 格式化后重新拼回完整地址
        dist_init_addr = f"{_format_v6_uri(ip_part)}:{port_part}"

        # 计算完整的 sglang 服务器参数字典与需校验字段列表
        server_args_dict, external_engine_need_check_fields = _compute_server_args(
            self.args,
            self.rank,
            dist_init_addr,
            nccl_port,
            host,
            port,
            self.worker_type,
            disaggregation_bootstrap_port,
            base_gpu_id=self.base_gpu_id,
        )

        # 记录本引擎的节点序号（多节点引擎时大于 0）
        self.node_rank = server_args_dict["node_rank"]
        # 记录服务器 host（IPv6 时带方括号）
        self.server_host = server_args_dict["host"]  # with [] if ipv6
        # （行尾原注释：IPv6 时 host 带方括号）
        # 记录服务器端口
        self.server_port = server_args_dict["port"]

        # 外部引擎模式：连接已存在的 sglang 服务并校验参数
        if self.args.rollout_external:
            # 走外部引擎初始化
            self._init_external(server_args_dict, external_engine_need_check_fields=external_engine_need_check_fields)
        else:
            # 常规模式：由本 actor 拉起服务器子进程
            self._init_normal(server_args_dict)

    # 外部引擎初始化：等待服务健康并校验实际参数与期望一致
    def _init_external(self, expect_server_args, external_engine_need_check_fields):
        # 打印日志说明使用外部引擎及期望参数
        logger.info(f"Use external SGLang engine (rank={self.rank}, expect_server_args={expect_server_args})")

        # 内部函数：从外部服务获取实际运行参数
        def _get_actual_server_args():
            # GET /get_server_info
            response = requests.get(f"http://{self.server_host}:{self.server_port}/get_server_info")
            # 非 2xx 状态码抛异常
            response.raise_for_status()
            # 返回 JSON 形式的实际参数
            return response.json()

        # 内部函数：对关键参数做一致性检查
        def _sanity_check_server_args(actual_server_args, expect_server_args):
            # 遍历需要校验的字段
            for name in external_engine_need_check_fields:
                # 期望值（slime 侧推导）
                expect_value = expect_server_args.get(name)
                # 实际值（外部服务返回）
                actual_value = actual_server_args.get(name)
                # 不一致则断言失败并给出双方完整参数
                assert (
                    actual_value == expect_value
                ), f"{name=} {expect_value=} {actual_value=} {expect_server_args=} {actual_server_args=}"

        # 等待外部服务健康（is_process_alive 恒真：外部进程不受本 actor 管理）
        _wait_server_healthy(
            base_url=f"http://{self.server_host}:{self.server_port}",
            api_key=None,
            is_process_alive=lambda: True,
        )
        # 获取外部服务实际参数
        actual_server_args = _get_actual_server_args()
        # 执行参数一致性校验
        _sanity_check_server_args(actual_server_args, expect_server_args)

    # 常规初始化：拉起服务器子进程，并（可选）注册到路由器
    def _init_normal(self, server_args_dict):
        # 打印启动日志
        logger.info(f"Launch HttpServerEngineAdapter at: {self.server_host}:{self.server_port}")
        # 以子进程方式启动 sglang 服务器并保存进程句柄
        self.process = launch_server_process(ServerArgs(**server_args_dict))

        # node 0 且配置了路由器时，把本引擎注册为路由器 worker
        if self.node_rank == 0 and self.router_ip and self.router_port:
            # 旧版路由器或 slime 自带路由器：只支持常规 worker
            if parse(sglang_router.__version__) <= parse("0.2.1") or self.args.use_slime_router:
                # 断言当前是 regular 类型（PD 分离不被旧路由器支持）
                assert (
                    self.worker_type == "regular"
                ), "pd disaggregation is not supported in old router or slime router."
                # 调用旧版 add_worker 接口注册
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/add_worker?url=http://{self.server_host}:{self.server_port}"
                )
            else:
                # 新版路由器：用 JSON 负载注册，支持 worker 类型区分
                payload = {
                    # worker 的服务地址
                    "url": f"http://{self.server_host}:{self.server_port}",
                    # worker 类型（regular/prefill/decode）
                    "worker_type": self.worker_type,
                }
                # prefill worker 还需上报引导端口（PD 分离传输用）
                if self.worker_type == "prefill":
                    # 填入 bootstrap 端口
                    payload["bootstrap_port"] = server_args_dict["disaggregation_bootstrap_port"]
                # 调用新版 workers 注册接口
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/workers",
                    json=payload,
                )
            # 非 2xx 抛异常（新旧两个分支的响应统一在此校验）
            response.raise_for_status()

    # 通用 POST 请求封装：非 0 号节点直接跳过
    def _make_request(self, endpoint: str, payload: dict | None = None):
        """Make a POST request to the specified endpoint with the given payload.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)

        Returns:
            The JSON response from the server
        """
        # 多节点引擎只有 0 号节点对外发请求（其余节点仅参与内部通信）
        if self.node_rank != 0:
            # 直接返回 None
            return

        # 拼接目标 URL
        url = f"http://{self.server_host}:{self.server_port}/{endpoint}"
        # 发送 POST 请求（payload 为空时用空字典）
        response = requests.post(url, json=payload or {})
        # 检查响应状态码
        try:
            # 非 2xx 抛出 HTTPError
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # 把响应正文附到异常 note 便于排查
            e.add_note(f"{response.text=}")
            # 重新抛出
            raise
        # 返回 JSON 响应
        return response.json()

    # 健康检查：调用底层服务的 /health_generate
    def health_generate(self, timeout: float = 5.0) -> bool:
        """Run /health_generate on the underlying SGLang HTTP server.

        Args:
            timeout: Timeout for the health request in seconds.

        Returns:
            True if the server responds with HTTP 200.

        Raises:
            requests.RequestException: If the request fails for any reason, including timeout.
        """
        # 非 0 号节点视为健康（不做检查）
        if self.node_rank != 0:
            # 直接返回 True
            return True

        # GET /health_generate，带超时限制
        response = requests.get(
            f"http://{self.server_host}:{self.server_port}/health_generate",
            timeout=timeout,
        )
        # 非 2xx 抛异常
        response.raise_for_status()
        # 返回健康
        return True

    # 用张量数据更新模型权重：HTTP 只传元数据，真实权重经 GPU 直接拷贝
    def update_weights_from_tensor(
        self,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        flush_cache: bool = False,
        weight_version: str | None = None,
    ):
        """
        Update model weights from tensor data. The HTTP server will only post meta data, and the real weights will be copied directly from GPUs.

        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """
        # 组装请求负载
        payload = {
            # 序列化的命名张量（IPC 共享内存对象）
            "serialized_named_tensors": serialized_named_tensors,
            # 加载格式（如 flattened_bucket）
            "load_format": load_format,
            # 是否同时清空缓存
            "flush_cache": flush_cache,
        }
        # 版本号存在时才带上（兼容旧版服务）
        if weight_version is not None:
            # 写入权重版本
            payload["weight_version"] = weight_version
        # 发送请求并返回结果
        return self._make_request(
            "update_weights_from_tensor",
            payload,
        )

    # 清空服务器 KV cache（等待至多 60 次，每次 1 秒）
    def flush_cache(self):
        """Flush the cache of the server."""
        # 非 0 号节点跳过
        if self.node_rank != 0:
            # 直接返回
            return
        # flush cache will not return status_code 200 when there are pending requests
        # （原英文注释：有挂起请求时 flush_cache 不会返回 200）
        # 重试至多 60 次直到成功
        for _ in range(60):
            # 尝试请求 flush_cache
            try:
                # GET /flush_cache
                response = requests.get(f"http://{self.server_host}:{self.server_port}/flush_cache")
                # 成功则跳出
                if response.status_code == 200:
                    # 跳出循环
                    break
            except NewConnectionError as e:
                # 连接都建立不了说明服务挂了，直接抛出
                raise e
            except Exception as e:
                # 其他异常（如有挂起请求）记录后重试
                logger.info(f"Error flushing cache: {e}")
                # 休眠 1 秒
                time.sleep(1)
                # 继续下一次重试
                continue
        else:
            # 60 次都没成功则抛出超时错误
            raise TimeoutError("Timeout while flushing cache.")

    # 关闭引擎：先从路由器摘除，再杀掉服务器进程树
    def shutdown(self):
        # 外部引擎不由本 actor 管理，直接返回
        if self.args.rollout_external:
            # 直接返回
            return

        # 打印关闭日志
        logger.info(f"Shutdown engine {self.server_host}:{self.server_port}...")
        # 从路由器摘除 worker 由 node 0 执行
        if self.node_rank == 0:
            # 构造本引擎的 worker URL
            worker_url = f"http://{self.server_host}:{self.server_port}"
            # 响应初始化为 None
            response = None
            # 旧版路由器或 slime 路由器：用 remove_worker 接口
            if parse(sglang_router.__version__) <= parse("0.2.1") or self.args.use_slime_router:
                # POST remove_worker 并带 url 查询参数
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/remove_worker?url=http://{self.server_host}:{self.server_port}"
                )
            elif parse(sglang_router.__version__) < parse("0.3.0"):
                # 0.2.x 之后、0.3.0 之前：DELETE /workers/{url}，URL 需编码
                worker_url = quote(worker_url, safe="")
                # 发送 DELETE 请求
                response = requests.delete(f"http://{self.router_ip}:{self.router_port}/workers/{worker_url}")
            else:
                # 0.3.0+：先查 worker 列表找到对应 id 再删除
                try:
                    # GET 全部 workers 列表
                    all_workers = requests.get(f"http://{self.router_ip}:{self.router_port}/workers").json()["workers"]
                    # 遍历每个 worker
                    for worker in all_workers:
                        # URL 匹配到本引擎
                        if worker["url"] == worker_url:
                            # 取出该 worker 的 id
                            worker_id = worker["id"]
                            # 按 id 发送 DELETE 请求
                            response = requests.delete(
                                f"http://{self.router_ip}:{self.router_port}/workers/{worker_id}"
                            )
                            # 找到即退出循环
                            break
                    else:
                        # 循环未 break 说明没找到对应 worker，仅告警
                        logger.warning(f"Worker {worker_url} not found in router during shutdown.")
                except Exception as e:
                    # 获取列表或删除失败只告警，不阻断关闭流程
                    logger.warning(f"Failed to fetch workers list or remove worker: {e}")

            # 有响应时校验状态码
            if response is not None:
                # 非 2xx 抛异常
                response.raise_for_status()
        # 杀掉服务器进程树（含其所有子进程）
        kill_process_tree(self.process.pid)

    # 查询服务器当前的权重版本号
    def get_weight_version(self):
        # 非 0 号节点跳过
        if self.node_rank != 0:
            # 返回 None
            return
        # 拼接版本查询 URL
        url = f"http://{self.server_host}:{self.server_port}/get_weight_version"
        # 发送 GET 请求
        response = requests.get(url)
        # 校验状态码
        response.raise_for_status()
        # 返回版本号字段
        return response.json()["weight_version"]

    # 释放显存占用（训练阶段让 rollout 引擎腾出显存）
    def release_memory_occupation(self):
        # 先清空 KV cache 再请求释放显存
        self.flush_cache()
        # POST release_memory_occupation
        return self._make_request("release_memory_occupation")

    # 恢复显存占用（rollout 阶段重新申请显存）
    def resume_memory_occupation(self, tags: list[str] = None):
        """
        Available tags for multi-stage resume: weights, kv_cache
        """
        # POST resume_memory_occupation，携带阶段标签
        return self._make_request(
            "resume_memory_occupation",
            {"tags": tags},
        )

    # 权重校验器开关（调试用，验证训练/推理两侧权重一致性）
    def check_weights(self, action: str):
        # POST weights_checker，携带动作（如 start/stop）
        return self._make_request("weights_checker", {"action": action})

    # 初始化权重更新用的 NCCL 通信组（训练侧会连进来）
    def init_weights_update_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        # POST init_weights_update_group，携带建组所需的全部信息
        return self._make_request(
            "init_weights_update_group",
            {
                # master 地址
                "master_address": master_address,
                # master 端口
                "master_port": master_port,
                # rank 偏移（本引擎首个 GPU 在组内的 rank）
                "rank_offset": rank_offset,
                # 组的世界大小
                "world_size": world_size,
                # 组名
                "group_name": group_name,
                # 后端（nccl）
                "backend": backend,
            },
        )

    # 销毁权重更新通信组（引擎刚创建还没有该组时静默忽略）
    def destroy_weights_update_group(self, group_name):
        # 尝试发送销毁请求
        try:
            # POST destroy_weights_update_group
            return self._make_request(
                "destroy_weights_update_group",
                {
                    # 组名
                    "group_name": group_name,
                },
            )
        except requests.exceptions.RequestException:
            # catch the case there the engine is just created and does not have the group.
            # （原英文注释：捕获引擎刚创建、组尚不存在的情形）
            # 静默忽略请求异常
            pass

    # 通知引擎即将通过 NCCL 接收权重（先传元信息，张量随后广播到达）
    def update_weights_from_distributed(
        self, names, dtypes, shapes, group_name, flush_cache=False, weight_version: str | None = None
    ):
        # 组装请求负载
        payload = {
            # 参数名列表
            "names": names,
            # dtype 字符串列表（去掉 "torch." 前缀，服务端易解析）
            "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
            # 形状列表
            "shapes": shapes,
            # 通信组名
            "group_name": group_name,
            # 是否同时清缓存
            "flush_cache": flush_cache,
        }
        # 版本号存在时才携带
        if weight_version is not None:
            # 写入权重版本
            payload["weight_version"] = weight_version
        # 发送请求并返回结果
        return self._make_request(
            "update_weights_from_distributed",
            payload,
        )

    # 暂停生成（权重同步期间不接受新请求）
    def pause_generation(self):
        # POST pause_generation
        response = requests.post(f"http://{self.server_host}:{self.server_port}/pause_generation", json={})
        # 校验状态码
        response.raise_for_status()
        # 返回响应
        return response

    # 恢复生成（权重同步完成后恢复服务）
    def continue_generation(self):
        # POST continue_generation
        response = requests.post(f"http://{self.server_host}:{self.server_port}/continue_generation", json={})
        # 校验状态码
        response.raise_for_status()
        # 返回响应
        return response

    # int4/fp4 量化的前后处理（加载前恢复原权重 / 加载后重新量化）
    def post_process_weights(
        self,
        restore_weights_before_load: bool = False,
        post_process_quantization: bool = False,
    ):
        """
        Update model weights from tensor data. The HTTP server will only post meta data, and the real weights will be copied directly from GPUs.
        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """

        # POST post_process_weights，携带两个布尔开关
        return self._make_request(
            "post_process_weights",
            {
                # 是否在加载前恢复原始权重
                "restore_weights_before_load": restore_weights_before_load,
                # 是否执行量化后处理
                "post_process_quantization": post_process_quantization,
            },
        )

    # 启动性能剖析（profiling），参数直接透传给服务端
    def start_profile(
        self,
        # The output directory
        # （原注释含义：输出目录）
        # 剖析结果输出目录（可选）
        output_dir: str | None = None,
        # If set, it profile as many as this number of steps.
        # If it is set, profiling is automatically stopped after this step, and
        # the caller doesn't need to run stop_profile.
        # （原英文注释：设置后自动在若干步后停止剖析）
        # 起始步编号（可选）
        start_step: int | None = None,
        # 剖析的步数（可选）
        num_steps: int | None = None,
        # 剖析活动类型列表（如 CPU/CUDA）（可选）
        activities: list[str] | None = None,
        # 是否按阶段剖析
        profile_by_stage: bool = False,
        # 是否记录调用栈（可选）
        with_stack: bool | None = None,
        # 是否记录张量形状（可选）
        record_shapes: bool | None = None,
    ):
        # POST start_profile，携带全部剖析参数
        response = requests.post(
            f"http://{self.server_host}:{self.server_port}/start_profile",
            json={
                # 输出目录
                "output_dir": output_dir,
                # 起始步
                "start_step": start_step,
                # 步数
                "num_steps": num_steps,
                # 活动类型
                "activities": activities,
                # 按阶段剖析
                "profile_by_stage": profile_by_stage,
                # 记录调用栈
                "with_stack": with_stack,
                # 记录形状
                "record_shapes": record_shapes,
            },
        )
        # 校验状态码
        response.raise_for_status()
        # 返回响应
        return response

    # 停止性能剖析
    def stop_profile(self):
        # POST stop_profile
        response = requests.post(f"http://{self.server_host}:{self.server_port}/stop_profile", json={})
        # 校验状态码
        response.raise_for_status()
        # 返回响应
        return response

    # 模拟引擎崩溃（用于容错测试）：走正常 shutdown 流程
    def simulate_crash(self):
        # 外部引擎或进程句柄不存在时无法模拟
        if self.args.rollout_external or not getattr(self, "process", None):
            # 打印跳过日志并说明原因
            logger.info(
                "simulate_crash called but no local engine process exists (rollout_external=%s); skip kill",
                self.args.rollout_external,
            )
            # 直接返回
            return

        # 打印模拟崩溃日志
        logger.info(f"Simulating crash on engine {self.server_host}:{self.server_port}...")
        # 调用正常关闭流程模拟崩溃
        self.shutdown()


# 根据 slime 的全局配置推导 sglang 服务器的启动参数
def _compute_server_args(
    args,
    rank,
    dist_init_addr,
    nccl_port,
    host,
    port,
    worker_type: str = "regular",
    disaggregation_bootstrap_port: int | None = None,
    base_gpu_id: int | None = None,
):
    # 计算引擎节点数：每引擎 GPU 数 ÷ 每节点 GPU 数（至少 1）
    nnodes = max(1, args.rollout_num_gpus_per_engine // args.num_gpus_per_node)
    # 本引擎的节点序号：rank 对节点数取模
    node_rank = rank % nnodes
    # GPU 起始编号：优先用显式指定的 base_gpu_id，否则按 rank 推导
    base = base_gpu_id if base_gpu_id is not None else get_base_gpu_id(args, rank)
    # 把物理 GPU 编号转换为 CUDA_VISIBLE_DEVICES 重映射后的本地编号
    base = _to_local_gpu_id(base)
    # 组装基础参数字典
    kwargs = {
        # 模型路径：slime 的 HF checkpoint 目录
        "model_path": args.hf_checkpoint,
        # 信任远程代码（RL 场景常需自定义模型代码）
        "trust_remote_code": True,
        # 随机种子：全局种子 + rank，保证各引擎不同
        "random_seed": args.seed + rank,
        # memory
        # （原注释：内存相关）
        # 是否启用内存节省（与 rollout offload 开关一致）
        "enable_memory_saver": args.offload_rollout,
        # distributed
        # （原注释：分布式相关）
        # 服务 host
        "host": host,
        # 服务端口
        "port": port,
        # NCCL 通信端口
        "nccl_port": nccl_port,
        # 引擎节点数
        "nnodes": nnodes,
        # 本引擎节点序号
        "node_rank": node_rank,
        # 分布式初始化地址
        "dist_init_addr": dist_init_addr,
        # GPU 编号步长固定为 1（连续占用）
        "gpu_id_step": 1,
        # GPU 起始编号（已本地化）
        "base_gpu_id": base,
        # parallel
        # （原注释：并行相关）
        # 张量并行大小 = 每引擎 GPU 数 ÷ PP
        "tp_size": args.rollout_num_gpus_per_engine // args.sglang_pp_size,
        # 数据并行大小
        "dp_size": args.sglang_dp_size,
        # 流水线并行大小
        "pp_size": args.sglang_pp_size,
        # 专家并行大小
        "ep_size": args.sglang_ep_size,
        # always skip warmup to prevent warmup timeout.
        # （原注释含义：总是跳过预热以防止预热超时）
        # 跳过服务预热
        "skip_server_warmup": True,
        # always enable draft weights cpu backup so that we run training without mtp weights.
        # （原英文注释：始终启用 draft 权重 CPU 备份，使训练无 MTP 权重也能跑）
        # 启用 MTP draft 权重的 CPU 备份
        "enable_draft_weights_cpu_backup": True,
    }

    # prefill worker：开启 PD 分离的 prefill 模式
    if worker_type == "prefill":
        # 设置分离模式为 prefill
        kwargs["disaggregation_mode"] = "prefill"
        # 负载均衡用轮询
        kwargs["load_balance_method"] = "round_robin"
        # 断言必须提供 bootstrap 端口
        assert (
            disaggregation_bootstrap_port is not None
        ), "disaggregation_bootstrap_port must be set for prefill worker"
        # 写入 bootstrap 端口
        kwargs["disaggregation_bootstrap_port"] = disaggregation_bootstrap_port
    elif worker_type == "decode":
        # decode worker：设置分离模式为 decode
        kwargs["disaggregation_mode"] = "decode"
        # 启用 prefill 轮询均衡
        kwargs["prefill_round_robin_balance"] = True

    # 启用路由重放（routing replay，MoE 路由回放训练技巧）时打开专家路由返回
    if args.use_rollout_routing_replay:
        # 打开 enable_return_routed_experts
        kwargs["enable_return_routed_experts"] = True
    # 指定 fp16 训练时设置 dtype
    if args.fp16:
        # 使用 float16
        kwargs["dtype"] = "float16"
    # 需要对外部引擎校验的字段 = 全部参数中排除免检清单
    external_engine_need_check_fields = [k for k in kwargs.keys() if k not in _EXTERNAL_ENGINE_SKIP_CHECK_FIELDS]

    # 记录尚未匹配到 ServerArgs 字段的键
    unused_keys = set(kwargs.keys())
    # 遍历 ServerArgs 的所有数据类字段，把用户自定义的 sglang_ 参数合并进来
    for attr in dataclasses.fields(ServerArgs):
        # decode 模式不支持分层缓存，跳过该字段避免冲突
        if worker_type == "decode" and attr.name == "enable_hierarchical_cache":
            # 跳过
            continue
        # 用户传了该字段的 sglang_ 版本且基础字典未覆盖时合并
        if hasattr(args, f"sglang_{attr.name}") and attr.name not in kwargs:
            # 写入用户指定值
            kwargs[attr.name] = getattr(args, f"sglang_{attr.name}")
        # 该键已被 ServerArgs 字段识别，从未使用集合中移除
        unused_keys.discard(attr.name)

    # for compatibility with old args
    # （原注释含义：兼容旧版参数）
    # 若仍有未被 sglang 支持的参数，告警并移除
    if len(unused_keys) > 0:
        # 打印不支持的参数列表
        logger.info(f"Warning: The following arguments is not supported in the current sglang: {unused_keys}.")
        # 逐个移除
        for key in unused_keys:
            # 从 kwargs 弹出
            kwargs.pop(key)

    # 返回最终参数字典与需校验字段列表
    return kwargs, external_engine_need_check_fields


# 外部引擎模式下跳过校验的字段清单（外部服务与 slime 推导值天然不同的项）
_EXTERNAL_ENGINE_SKIP_CHECK_FIELDS = [
    # 模型路径（外部服务可自由指定）
    "model_path",
    # 远程代码信任开关
    "trust_remote_code",
    # 随机种子
    "random_seed",
    # NCCL 端口
    "nccl_port",
    # 分布式初始化地址
    "dist_init_addr",
    # 跳过预热开关
    "skip_server_warmup",
    # draft 权重 CPU 备份开关
    "enable_draft_weights_cpu_backup",
    # 静态显存占比
    "mem_fraction_static",
]
