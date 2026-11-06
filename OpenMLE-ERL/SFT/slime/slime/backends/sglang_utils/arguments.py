# 导入 argparse，用于构建与解析命令行参数
import argparse

# 导入 sglang 的服务器参数类 ServerArgs，其中包含全部 sglang 服务端 CLI 参数定义
from sglang.srt.server_args import ServerArgs
# 导入 slime 的 HTTP 工具：_wrap_ipv6 用于给 IPv6 地址补上方括号
from slime.utils.http_utils import _wrap_ipv6


# TODO: use all sglang router arguments with `--sglang-router` prefix
# （原 TODO 含义：将来应使用带 --sglang-router 前缀的全部路由器参数）
# 为解析器添加 SGLang 路由器（router）相关的命令行参数
def add_sglang_router_arguments(parser):
    """
    Add arguments to the parser for the SGLang router.
    """
    # 添加路由器 IP 参数，默认 None 表示不使用路由器
    parser.add_argument(
        "--sglang-router-ip",
        type=str,
        default=None,
        help="IP address of the SGLang router",
    )
    # 添加路由器端口参数，默认 None
    parser.add_argument(
        "--sglang-router-port",
        type=int,
        default=None,
        help="Port of the SGLang router",
    )
    # 添加路由策略参数，如一致性哈希或轮询
    parser.add_argument(
        "--sglang-router-policy",
        type=str,
        default=None,
        help="Routing policy for the SGLang router (e.g., 'consistent_hashing', 'round_robin')",
    )
    # 添加路由请求超时参数，默认 4 小时（RL rollout 的长请求场景）
    parser.add_argument(
        "--sglang-router-request-timeout-secs",
        type=int,
        default=14400,
        help="Timeout for requests to the SGLang router in seconds",
    )
    # 返回解析器，支持链式调用
    return parser


# 为解析器添加 SGLang 服务器相关的全部命令行参数
def add_sglang_arguments(parser):
    """
    Add arguments to the parser for the SGLang server.
    """
    # 先追加路由器参数
    parser = add_sglang_router_arguments(parser)
    # 为路由负载均衡阈值设置默认值（绝对阈值 10、相对阈值 1.2）
    parser.set_defaults(router_balance_abs_threshold=10, router_balance_rel_threshold=1.2)
    # 添加 sglang 服务端并发上限参数，默认 512
    parser.add_argument("--sglang-server-concurrency", type=int, default=512)

    # 备份原生的 add_argument 方法，稍后用包装版替换再还原
    old_add_argument = parser.add_argument

    # 需要跳过注册的 sglang 服务端参数列表（slime 会自行管理或与训练侧冲突）
    skipped_args = [
        # 模型路径由 slime 的 rollout 配置决定
        "model_path",
        # 信任远程代码由训练侧统一控制
        "trust_remote_code",
        # 随机种子由 slime 统一管理
        "random_seed",
        # memory
        # （原注释：内存相关）
        # 内存节省开关与训练侧显存管理冲突
        "enable_memory_saver",
        # distributed
        # （原注释：分布式相关）
        # 张量并行大小由 rollout_num_gpus_per_engine 推导
        "tp_size",
        # 服务端口由 slime 分配
        "port",
        # 节点数由 slime 的集群配置决定
        "nnodes",
        # 节点序号由 Ray/训练框架决定
        "node_rank",
        # 分布式初始化地址由框架注入
        "dist_init_addr",
        # GPU 编号步长由 slime 决定
        "gpu_id_step",
        # GPU 起始编号由 slime 决定
        "base_gpu_id",
        # NCCL 端口由框架分配
        "nccl_port",
        # 服务预热由 slime 控制流程管理
        "skip_server_warmup",
        # 路由专家返回功能不暴露给用户
        "enable_return_routed_experts",
    ]

    # 定义 add_argument 的包装函数：加前缀、跳过被排除参数
    def new_add_argument_wrapper(*name_or_flags, **kwargs):
        """
        Add arguments to the parser, ensuring that the server arguments are prefixed and skippable.
        """
        # Determine the canonical name for skip check (e.g., "model_path")
        # （原英文注释：确定用于跳过检查的规范名）
        # 初始化规范名为 None
        canonical_name_for_skip_check = None
        # 若显式指定了 dest，则直接用 dest 作为规范名
        if "dest" in kwargs:
            # 取出 dest 值
            canonical_name_for_skip_check = kwargs["dest"]
        else:
            # 否则从第一个长选项推导规范名
            for flag_name_candidate in name_or_flags:
                # 只处理字符串且以 "--" 开头的长选项
                if isinstance(flag_name_candidate, str) and flag_name_candidate.startswith("--"):
                    # Derive from first long flag: --foo-bar -> foo_bar
                    # （原英文注释：从长选项推导，把连字符换下划线）
                    # 去掉前导连字符得到词干
                    stem = flag_name_candidate[2:]
                    # 连字符替换为下划线得到规范名
                    canonical_name_for_skip_check = stem.replace("-", "_")
                    # 找到第一个长选项后即退出
                    break
            # If no long flag and no dest, skip logic might not catch it unless short flags imply a dest.
            # （原英文注释：无长选项且无 dest 时跳过逻辑可能不生效）

        # 若规范名在被跳过列表中，则不注册该参数
        if canonical_name_for_skip_check and canonical_name_for_skip_check in skipped_args:
            # 直接返回，跳过整个参数定义
            return  # Skip this entire argument definition

        # If not skipped, proceed to prefix flags and dest
        # （原英文注释：未跳过则给选项加前缀）
        # 构建新的选项列表
        new_name_or_flags_list = []
        # 遍历原始的选项/位置参数
        for item_flag in name_or_flags:
            # 字符串且以 "-" 开头的是选项，需要加前缀
            if isinstance(item_flag, str) and item_flag.startswith("-"):
                original_flag_stem = item_flag.lstrip("-")  # "foo-bar" from "--foo-bar", or "f" from "-f"
                # 给词干统一加 "--sglang-" 前缀，避免与训练侧参数冲突
                prefixed_item = f"--sglang-{original_flag_stem}"
                # 追加带前缀的新选项
                new_name_or_flags_list.append(prefixed_item)
            else:
                # Positional arguments or non-string items
                # （原英文注释：位置参数或非字符串项原样保留）
                # 非选项项原样追加
                new_name_or_flags_list.append(item_flag)

        # Prepare kwargs for the actual add_argument call.
        # Make a copy to avoid modifying the original kwargs dict.
        # （原英文注释：拷贝 kwargs 避免修改原字典）
        # 浅拷贝关键字参数
        final_kwargs = kwargs.copy()

        # If 'dest' is explicitly provided and is a string, prefix it.
        # This ensures the attribute on the args namespace becomes, e.g., args.sglang_dest_name.
        # （原英文注释：显式 dest 也加前缀，保证属性名统一为 sglang_ 开头）
        # 仅当 dest 是字符串时处理
        if "dest" in final_kwargs and isinstance(final_kwargs["dest"], str):
            # 取出原始 dest
            original_dest = final_kwargs["dest"]
            # Avoid double prefixing if dest somehow already starts with sglang_
            # （原英文注释：避免重复加前缀）
            # 若 dest 尚无 sglang_ 前缀则添加
            if not original_dest.startswith("sglang_"):
                # 给 dest 加上 sglang_ 前缀
                final_kwargs["dest"] = f"sglang_{original_dest}"
        # If 'dest' is not explicitly provided (or is None/not a string),
        # argparse will derive 'dest' from the (now prefixed) flag names.
        # E.g., if the first flag is "--sglang-foo-bar", argparse sets dest to "sglang_foo_bar".
        # （原英文注释：未显式 dest 时 argparse 会从带前缀的选项自动推导）

        # 用原生的 add_argument 注册处理后的参数
        old_add_argument(*new_name_or_flags_list, **final_kwargs)

    # 临时替换解析器的 add_argument 为包装版，拦截 ServerArgs 的所有参数注册
    parser.add_argument = new_add_argument_wrapper
    # 把 sglang ServerArgs 的全部 CLI 参数注册进解析器（经过包装处理）
    ServerArgs.add_cli_args(parser)
    # 注册完成后还原原生 add_argument
    parser.add_argument = old_add_argument

    # 返回配置好的解析器
    return parser


# 校验并推导 sglang 相关参数的一致性（TP/DP/EP/PP 之间的关系）
def validate_args(args):
    # 用简写别名记录 sglang 的数据并行大小
    args.sglang_dp_size = args.sglang_data_parallel_size
    # 记录 sglang 的流水线并行大小
    args.sglang_pp_size = args.sglang_pipeline_parallel_size
    # 记录 sglang 的专家并行大小
    args.sglang_ep_size = args.sglang_expert_parallel_size

    # Compute effective TP size considering PP size
    # （原英文注释：结合 PP 计算有效 TP 大小）
    # 当流水线并行度大于 1 时，TP = 每引擎 GPU 数 ÷ PP
    if args.sglang_pp_size > 1:
        # 断言每引擎 GPU 数能被 PP 整除，否则配置非法
        assert args.rollout_num_gpus_per_engine % args.sglang_pp_size == 0, (
            f"rollout_num_gpus_per_engine ({args.rollout_num_gpus_per_engine}) must be divisible by "
            f"sglang_pipeline_parallel_size ({args.sglang_pp_size})"
        )
        # 推导张量并行大小
        args.sglang_tp_size = args.rollout_num_gpus_per_engine // args.sglang_pp_size
    else:
        # 无 PP 时 TP 就等于每引擎 GPU 数
        args.sglang_tp_size = args.rollout_num_gpus_per_engine

    # 数据并行度大于 1 时必须开启 DP attention（sglang 的限制）
    if args.sglang_dp_size > 1:
        # 断言已启用 dp attention
        assert args.sglang_enable_dp_attention

    # 若配置了路由器 IP 且是 IPv6 地址，补上方括号便于拼接 URL
    if getattr(args, "sglang_router_ip", None):
        # 调用工具函数包装 IPv6 地址
        args.sglang_router_ip = _wrap_ipv6(args.sglang_router_ip)


# 独立解析 sglang 服务端参数：只消费 sglang 相关的 argv，其余留给 megatron/fsdp
def sglang_parse_args():
    """
    Parse sglang server arguments independently using a separate ArgumentParser.
    Uses parse_known_args() to only consume sglang-related arguments from sys.argv,
    allowing the remaining arguments to be parsed by megatron/fsdp separately.

    Returns:
        argparse.Namespace: Parsed sglang arguments (all attributes prefixed with sglang_).
    """
    # 创建独立解析器，禁用 --help 以免与主解析器冲突
    parser = argparse.ArgumentParser(add_help=False)
    # 注册全部 sglang 参数
    add_sglang_arguments(parser)

    # Compute default sglang_tensor_parallel_size from CLI args
    # （原英文注释：从 CLI 参数推导默认 TP 大小）
    # 建一个临时解析器做预解析
    temp_parser = argparse.ArgumentParser(add_help=False)
    # 注册每引擎 GPU 数参数（默认 1）
    temp_parser.add_argument("--rollout-num-gpus-per-engine", type=int, default=1)
    # 注册 sglang PP 简写参数
    temp_parser.add_argument("--sglang-pp-size", type=int, default=1)
    # 注册 sglang PP 全名参数
    temp_parser.add_argument("--sglang-pipeline-parallel-size", type=int, default=1)
    # 只解析已知参数，忽略其余 argv
    temp_args, _ = temp_parser.parse_known_args()
    # 取 PP 大小：简写优先，简写为默认值 1 时退回全名参数
    pp_size = temp_args.sglang_pp_size if temp_args.sglang_pp_size != 1 else temp_args.sglang_pipeline_parallel_size
    # 推导默认 TP 大小 = 每引擎 GPU 数 ÷ PP
    sglang_tp_size = temp_args.rollout_num_gpus_per_engine // pp_size
    # 把推导出的 TP 默认值设置进主解析器
    parser.set_defaults(sglang_tensor_parallel_size=sglang_tp_size)

    # 正式解析：只消费 sglang 相关参数，未识别的参数留在忽略列表中
    args, _ = parser.parse_known_args()
    # 返回解析结果（属性均带 sglang_ 前缀）
    return args
