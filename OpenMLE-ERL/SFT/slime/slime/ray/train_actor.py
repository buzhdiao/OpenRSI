# 导入 abc，声明抽象方法（子类必须实现 sleep/wake_up/train 等）
import abc
# 导入 logging，模块级日志
import logging
# 导入 os，设置分布式相关的环境变量
import os
# 导入 random，随机选取 master 端口搜索起点
import random
# 导入 timedelta，设置进程组初始化超时
from datetime import timedelta

# 导入 ray，用于远程调用 rollout 管理器
import ray
# 导入 torch，CUDA 设备设置与序列化安全配置
import torch
# 导入 torch.distributed，初始化进程组
import torch.distributed as dist

# 导入评估配置模块（用于 torch.load 白名单）
import slime.utils.eval_config
# 导入 Ray actor 基类
from slime.ray.ray_actor import RayActor
# 导入分布式工具：初始化 gloo CPU 通信组
from slime.utils.distributed_utils import init_gloo_group
# 导入日志工具：配置日志器
from slime.utils.logging_utils import configure_logger
# 导入显存工具：清理与打印显存
from slime.utils.memory_utils import clear_memory, print_memory

# 创建本模块日志器
logger = logging.getLogger(__name__)


# 计算本进程在 CUDA_VISIBLE_DEVICES 重映射后的本地 GPU 编号
def get_local_gpu_id():
    # 读取 CUDA_VISIBLE_DEVICES（可能为 None）
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    # 未设置时 Ray 分配的 GPU id 就是本地 id
    if cvd is None:
        # 返回 Ray 分配的第一个 GPU 编号
        return ray.get_gpu_ids()[0]
    else:
        # 已设置时返回 Ray GPU id 在可见列表中的下标
        return cvd.split(",").index(str(ray.get_gpu_ids()[0]))


# 训练 actor 基类：封装分布式初始化与训练生命周期接口
class TrainRayActor(RayActor):
    # 构造函数：配置日志并确定 master 地址端口与分布式环境变量
    def __init__(self, world_size, rank, master_addr, master_port):
        # 配置日志器
        configure_logger()

        # 保存世界大小
        self._world_size = world_size
        # 保存自身 rank
        self._rank = rank
        # 已传入 master 地址（非 rank0 场景）时直接使用
        if master_addr:
            # 使用传入的地址与端口
            self.master_addr, self.master_port = master_addr, master_port
        else:
            # rank0 自行探测本机 IP 并在随机段内找一个空闲端口
            self.master_addr, self.master_port = self._get_current_node_ip_and_free_port(
                start_port=random.randint(20000, 21000)
            )

        # 设置 torchrun 风格的环境变量 MASTER_ADDR
        os.environ["MASTER_ADDR"] = self.master_addr
        # 设置 MASTER_PORT（转字符串）
        os.environ["MASTER_PORT"] = str(self.master_port)
        # 设置 WORLD_SIZE
        os.environ["WORLD_SIZE"] = str(self._world_size)
        # 设置 RANK
        os.environ["RANK"] = str(self._rank)
        # TODO: currently this doesn't work as ray has already set torch.cuda.device_count().
        # os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0])
        # （保留原注释与 TODO：Ray 已提前设置 device_count，直接 pop 无效）
        # 用重映射后的本地 GPU 编号设置 LOCAL_RANK
        os.environ["LOCAL_RANK"] = str(get_local_gpu_id())

    # 初始化：进程组、NUMA 亲和、保存角色配置
    def init(self, args, role, with_ref=False, with_opd_teacher=False):
        # 保存配置
        self.args = args
        # 保存角色（actor/critic）
        self.role = role
        # 是否加载参考模型（KL 约束用）
        self.with_ref = with_ref
        # 是否加载 OPD 教师模型
        self.with_opd_teacher = with_opd_teacher

        # 把 EvalDatasetConfig 加入 torch.load 的安全全局白名单（权重安全加载）
        torch.serialization.add_safe_globals([slime.utils.eval_config.EvalDatasetConfig])

        # 读取本地 rank
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # 把当前 CUDA 设备绑定到对应 GPU
        torch.cuda.set_device(f"cuda:{local_rank}")

        # Use hybrid backend when FSDP CPU offload is enabled with a CPU backend
        # （原英文注释：FSDP CPU offload 时用混合后端）
        # 默认后端取配置值
        backend = args.distributed_backend
        # 开启 CPU offload 且指定了 CPU 后端时组合为混合后端
        if getattr(args, "fsdp_cpu_offload", False) and getattr(args, "fsdp_cpu_backend", None):
            # 取 CPU 后端名（如 gloo）
            cpu_backend = args.fsdp_cpu_backend
            # 组合成 "cpu:xxx,cuda:yyy" 形式
            backend = f"cpu:{cpu_backend},cuda:{args.distributed_backend}"
            # 打印混合后端日志
            logger.info(f"FSDP CPU offload enabled, using hybrid backend: {backend}")

        # 初始化默认进程组（超时按配置分钟数）
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
        )
        # 初始化额外的 gloo CPU 组（元信息同步用）
        init_gloo_group()

        # 把实际 rank 写回配置
        args.rank = dist.get_rank()
        # 把实际世界大小写回配置
        args.world_size = dist.get_world_size()

        # NUMA 亲和设置失败不影响训练，故整体 try 包裹
        try:
            # ROCm/HIP 环境暂不支持 NUMA 亲和
            if torch.version.hip is not None:
                # 打印跳过日志
                logger.info("Detected ROCm/HIP environment, skipping NUMA affinity setup")
                # will find the coresponding API to implement ROCm version as below
                # （原英文注释：待实现 ROCm 版本 API）
            else:
                # 延迟导入 pynvml（NVIDIA 管理库）
                import pynvml

                # 初始化 NVML
                pynvml.nvmlInit()

                # 本地 rank = 全局 rank 对每节点 GPU 数取模
                local_rank = int(os.environ["RANK"]) % args.num_gpus_per_node

                # 按本地 rank 拿到 GPU 的 NVML 句柄
                handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
                # 把该 GPU 绑定到就近 CPU 的 NUMA 节点，降低跨 NUMA 访问延迟
                pynvml.nvmlDeviceSetCpuAffinity(handle)

                # 打印设置成功日志
                logger.info(f"Set NUMA affinity for GPU {local_rank}")
                # 关闭 NVML
                pynvml.nvmlShutdown()

        except ImportError:
            # 无 pynvml 时跳过 NUMA 设置
            logger.info("Warning: pynvml not available, skipping NUMA affinity setup")
        except Exception as e:
            # 其他失败也只告警不中断
            logger.info(f"Warning: Failed to set NUMA affinity: {e}")

    # 清理显存：打印前后显存便于观察效果
    def clear_memory(self):
        # 打印清理前的显存状态
        print_memory("before TrainRayActor.clear_memory")
        # 执行显存清理（释放缓存等）
        clear_memory()
        # 打印清理后的显存状态
        print_memory("after TrainRayActor.clear_memory")

    # 抽象方法：休眠（offload 模型到 CPU），tags 指定卸载哪些部分
    @abc.abstractmethod
    def sleep(self, tags):
        # 子类必须实现
        raise NotImplementedError

    # 抽象方法：唤醒（onload 模型到 GPU）
    @abc.abstractmethod
    def wake_up(self, tags):
        # 子类必须实现
        raise NotImplementedError

    # 抽象方法：执行一轮训练
    @abc.abstractmethod
    def train(self, rollout_id, rollout_data_ref):
        # 子类必须实现
        raise NotImplementedError

    # 抽象方法：保存模型
    @abc.abstractmethod
    def save_model(self, rollout_id, force_sync=False):
        # 子类必须实现
        raise NotImplementedError

    # 抽象方法：把 rank0 权重广播到其他 rank
    @abc.abstractmethod
    def update_weights(self):
        # 子类必须实现
        raise NotImplementedError

    # 抽象方法：与 critic actor 建立连接
    @abc.abstractmethod
    def connect_actor_critic(self, critic_group):
        # 子类必须实现
        raise NotImplementedError

    # 抽象方法：返回训练并行配置（DP/TP/PP 等）
    @abc.abstractmethod
    def _get_parallel_config(self):
        # 子类必须实现
        raise NotImplementedError

    # 保存 rollout 管理器句柄，rank0 回传训练并行配置
    def set_rollout_manager(self, rollout_manager):
        # 保存管理器句柄
        self.rollout_manager = rollout_manager
        # 非 rollout 调试模式且是 rank0 时：把训练并行配置告诉 rollout 管理器
        if not self.args.debug_rollout_only and self.args.rank == 0:
            # 远程设置并行配置（供 DP 切分数据用）
            ray.get(self.rollout_manager.set_train_parallel_config.remote(self.train_parallel_config))
