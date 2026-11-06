# slime 命令行参数模块:在 Megatron/FSDP/SGLang 后端参数之上叠加 slime 自身的集群、
# 训练、rollout、算法、评测、日志等参数,并完成多阶段解析、参数合并与交叉校验。
import argparse
import json
import logging
import os
from typing import Any

import yaml
from sglang_router.launch_router import RouterArgs

from slime.backends.sglang_utils.arguments import sglang_parse_args
from slime.backends.sglang_utils.arguments import validate_args as sglang_validate_args
from slime.utils.eval_config import EvalDatasetConfig, build_eval_dataset_configs, ensure_dataset_list
from slime.utils.logging_utils import configure_logger

# 模块级 logger
logger = logging.getLogger(__name__)


def reset_arg(parser, name, **kwargs):
    """
    Reset the default value of a Megatron argument.
    :param parser: The argument parser.
    :param name: The name of the argument to reset.
    :param default: The new default value.
    """
    # 遍历解析器中已注册的所有参数动作,查找目标选项
    for action in parser._actions:
        if name in action.option_strings:
            if "default" in kwargs:
                # 参数已存在:仅覆盖其默认值(slime 对 Megatron 参数的常用改默认值手段)
                action.default = kwargs["default"]
            break
    else:
        # for 循环未 break 说明参数不存在:按给定 kwargs 新增该参数
        parser.add_argument(name, **kwargs)


def get_slime_extra_args_provider(add_custom_arguments=None):
    # 工厂函数:返回向 parser 注册全部 slime 扩展参数的回调,供各后端的 parse_args 调用
    def add_slime_arguments(parser):
        # Ray
        # ---------- 集群与 Ray 资源相关参数 ----------
        def add_cluster_arguments(parser):
            # 训练 actor 占用的节点数
            parser.add_argument("--actor-num-nodes", type=int, default=1, help="Number of nodes for training actor")
            # 训练 actor 每节点 GPU 数
            parser.add_argument(
                "--actor-num-gpus-per-node", type=int, default=8, help="Number of gpus per node for training actor"
            )
            # critic 占用的节点数(可选,默认取 actor 配置)
            parser.add_argument(
                "--critic-num-nodes", type=int, default=None, help="Number of nodes for training actor"
            )
            # critic 每节点 GPU 数(可选,默认取 actor 配置)
            parser.add_argument(
                "--critic-num-gpus-per-node", type=int, default=None, help="Number of gpus per node for training actor"
            )

            # 推理(rollout)引擎占用的 GPU 总数;colocate 模式下忽略并按 actor GPU 数推定
            parser.add_argument(
                "--rollout-num-gpus",
                type=int,
                default=None,
                help=(
                    "Number of GPUs for inference. Note that when using --colocate, "
                    "i.e. the training and the inference engines are on the same gpus, this param will be ignored and will be set as "
                    "actor_num_gpus_per_node * actor_num_nodes."
                ),
            )
            # 每个推理引擎的 GPU 数,等价于 sglang 的 tp_size(张量并行度)
            parser.add_argument(
                "--rollout-num-gpus-per-engine",
                type=int,
                default=1,
                help="Number of GPUs per inference engine, just like the tp_size in sglang.",
            )
            # rollout 每节点 GPU 数;colocate 下节点 GPU 少于 8 卡时需要显式设置
            parser.add_argument(
                "--num-gpus-per-node",
                type=int,
                default=8,
                help=(
                    "Number of gpus per node for rollout."
                    "Notice: If you are going to use less than 8 gpus per node under colocate mode, you should set this number."
                ),
            )
            # 是否把推理引擎与训练 actor 放置在同一批 GPU 上(混合部署)
            parser.add_argument(
                "--colocate",
                action="store_true",
                default=False,
                help=(
                    "Whether to colocate the inference engines and the actor. "
                    "Turning this on will also set --offload to true."
                ),
            )
            # 一键开关:同时 offload 训练侧与推理侧
            parser.add_argument(
                "--offload",
                action="store_true",
                default=False,
                help=("Equivalent to --offload-train + --offload-rollout. "),
            )
            # 训练阶段是否把训练 actor 卸载(offload)到 CPU 内存以腾出显存
            parser.add_argument(
                "--offload-train",
                action=argparse.BooleanOptionalAction,
                help=(
                    "Whether to offload the training actor to CPU during training. "
                    "This will always be true when --colocate is set."
                ),
            )
            # 训练阶段是否把 rollout 生成器卸载到 CPU 内存
            parser.add_argument(
                "--offload-rollout",
                action=argparse.BooleanOptionalAction,
                help=(
                    "Whether to offload the rollout generator to CPU during training. "
                    "This will always be true when --colocate is set."
                ),
            )

            # 重置 Megatron 参数默认值:分布式通信后端默认用 NCCL
            reset_arg(parser, "--distributed-backend", type=str, default="nccl")
            # 重置分布式超时时间默认值为 10 分钟
            reset_arg(parser, "--distributed-timeout-minutes", type=int, default=10)

            # 返回追加了集群参数的 parser
            return parser

        # ---------- 训练相关参数 ----------
        def add_train_arguments(parser):
            # --train-backend is parsed early in _pre_parse_mode() and merged later.
            # Megatron 后端的 QKV 张量排布格式:thd(打包序列)或 bshd(批优先)
            parser.add_argument(
                "--qkv-format",
                type=str,
                choices=["thd", "bshd"],
                default="thd",
                help="The qkv layout for Megatron backend.",
            )
            # 是否开启"真 on-policy"模式(每个训练步都同步最新权重再采样)
            parser.add_argument(
                "--true-on-policy-mode",
                action="store_true",
                default=False,
                help="Whether to enable true-on-policy mode.",
            )
            # 训练进程的额外环境变量(JSON 字符串),如 PyTorch 显存管理相关
            parser.add_argument(
                "--train-env-vars",
                type=json.loads,
                default="{}",
                help="Extra environment variables for training process, e.g. PyTorch memory management ones.",
            )
            # 训练显存分配预留的安全余量,默认 1GB
            parser.add_argument(
                "--train-memory-margin-bytes",
                type=int,
                default=1024**3,
                help="Add margin for train memory allocation. By default we will reserve 1GB as margin.",
            )
            # 通过 --disable-weights-backuper 把 enable_weights_backuper 置 False,省主机内存
            parser.add_argument(
                "--disable-weights-backuper",
                action="store_false",
                dest="enable_weights_backuper",
                help="Whether to disable weights backuper to save host memory.",
            )
            # Megatron 权重转 HuggingFace 权重的方式:raw 直转或 bridge 桥接
            parser.add_argument(
                "--megatron-to-hf-mode",
                choices=["raw", "bridge"],
                default="raw",
                help="The method to convert megatron weights to hugging face weights for SGLang.",
            )
            # 自定义模型构造函数路径(替代默认的 model provider)
            parser.add_argument(
                "--custom-model-provider-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom model provider function. "
                    "If set, we will use this function instead of the default model provider. "
                    "The function should have the signature "
                    "`def custom_model_provider(pre_process: bool, post_process: bool, vp_stage: int | None = None) -> GPTModel`. "
                    "Example: 'my_module.my_model_provider'."
                ),
            )
            # 是否用重计算(loss function recompute)方式省训练显存
            parser.add_argument(
                "--recompute-loss-function",
                action="store_true",
                help="Whether to disable recompute loss function to save memory during training.",
            )
            # 分块(chunk)计算 log probs 的块大小,用于省显存;-1 表示不分块
            parser.add_argument(
                "--log-probs-chunk-size", type=int, default=-1, help="Chunk size to compute log probs to save memory"
            )
            # 只训练名称匹配这些正则模式的参数,其余参数全部冻结(参数名正则白名单)
            parser.add_argument(
                "--only-train-params-name-list",
                type=str,
                nargs="*",
                default=None,
                help=r"""List of regex patterns of parameter names to TRAIN. All other parameters will be FROZEN. 
                        Supports Python regex syntax (re.search).

                        Examples:
                        1. Train ONLY MoE experts:
                            --only-train-params-name-list experts

                        2. Train ONLY Indexer parameters:
                            --only-train-params-name-list self_attention.wq_b self_attention.wk self_attention.k_norm self_attention.weights_proj

                        3. Train ONLY Layer 20 to 23:
                            --only-train-params-name-list layers\.2[0-3]\.
                        """,
            )

            # 冻结名称匹配这些正则模式的参数,其余参数保持可训练(参数名正则黑名单)
            parser.add_argument(
                "--freeze-params-name-list",
                type=str,
                nargs="*",
                default=None,
                help="""List of regex patterns of parameter names to FREEZE. Other parameters will remain trainable.
                        Supports Python regex syntax (re.search).

                        Examples:
                        1. Freeze Embeddings and Output Layer (common for fine-tuning):
                            --freeze-params-name-list embedding output_layer

                        2. Freeze Indexer parameters:
                            --freeze-params-name-list self_attention.wq_b self_attention.wk self_attention.k_norm self_attention.weights_proj

                        3. Freeze specific projection layers (e.g., all Gate/Up projections):
                            --freeze-params-name-list linear_fc1
                        """,
            )
            # 是否在上下文并行(CP)中使用 allgather 方式收集序列分片
            parser.add_argument(
                "--allgather-cp",
                action="store_true",
                default=False,
            )

            # 返回追加了训练参数的 parser
            return parser

        # rollout
        # ---------- rollout(推理采样)相关参数 ----------
        def add_rollout_arguments(parser):
            # 训练模型的 HuggingFace 权重路径,用于初始化 sglang 引擎并提供 tokenizer
            parser.add_argument(
                "--hf-checkpoint",
                type=str,
                default=None,
                help=(
                    "The huggingface checkpoint of the trained model. "
                    "This is used to initialize sglang and also provide the tokenizer. "
                    "Note that, we will always update the parameters in sglang with that of megatron before training, "
                    "so you only need to provide a huggingface checkpoint that has the same architecture as the model you want to train. "
                    "It doesn't necessary need to contain the most up-to-date parameters."
                ),
            )
            # 模型名称,用于 Megatron->HuggingFace 权重格式转换;不设时从 hf_checkpoint 自动推断
            parser.add_argument(
                "--model-name",
                type=str,
                default=None,
                help=(
                    "The name of the model, this is used to convert the megatron weights into huggingface format. "
                    "If not set, we will use `type(AutoConfig.from_pretrained(args.hf_checkpoint)).__name__.lower()` as model_name. "
                    "Also, sometimes this will help alleviate the bug that transformers cannot find certain model."
                ),
            )
            # rollout 生成函数的模块路径,可替换为自定义采样逻辑
            parser.add_argument(
                "--rollout-function-path",
                type=str,
                default="slime.rollout.sglang_rollout.generate_rollout",
                help=(
                    "Path to the rollout generation function."
                    "You should use this model to create your own custom rollout function, "
                    "and then set this to the path of your custom rollout function. "
                    "The signature of the function should be "
                    "`def generate_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnTrainOutput | RolloutFnEvalOutput`"
                    "and within the output sample, you should at least set `tokens`, `response_length`, `reward` "
                    "and `status`."
                ),
            )
            # rollout 阶段推理引擎的采样温度
            parser.add_argument(
                "--rollout-temperature",
                type=float,
                default=1.0,
                help="the temperature for the inference engine during rollout.",
            )
            # rollout 阶段的核采样(nucleus sampling)概率阈值 top-p
            parser.add_argument(
                "--rollout-top-p", type=float, default=1.0, help="the top-p for the inference engine during rollout."
            )
            # rollout 阶段的 top-k 采样;-1 表示不启用
            parser.add_argument(
                "--rollout-top-k", type=int, default=-1, help="the top-k for the inference engine during rollout."
            )
            # rollout 阶段推理引擎的最大上下文长度(不得超过模型 config 的 max_position_embeddings)
            parser.add_argument(
                "--rollout-max-context-len",
                type=int,
                default=None,
                help=(
                    "The maximum context size for the inference engine during rollout."
                    "It should no exceed the `max_position_embeddinds` in Huggingface model's `config.json`"
                ),
            )
            # rollout 阶段允许的最大 prompt 长度;设置后会在构建全局数据集时过滤超长 prompt
            parser.add_argument(
                "--rollout-max-prompt-len",
                type=int,
                default=None,
                help=(
                    "The maximum length of the prompt for the inference engine during rollout. "
                    "If set, we will filter out the long prompts during initialization of the global dataset. "
                    "This is not recommended if the dataset is large."
                ),
            )
            # rollout 阶段允许的最大生成长度,等价于 sglang 的 max_tokens
            parser.add_argument(
                "--rollout-max-response-len",
                type=int,
                default=None,
                help=(
                    "The maximum length of the response for the inference engine during rollout. "
                    "It is basically `max_tokens` in sglang."
                ),
            )
            # 是否在生成的回复中跳过特殊 token(便于把回复再作为下轮 prompt 使用)
            parser.add_argument(
                "--rollout-skip-special-tokens",
                action="store_true",
                default=False,
                help=(
                    "Whether to skip special tokens in the response during rollout. "
                    "This is useful when you want to use the response as a prompt for the next rollout."
                ),
            )
            # rollout 的停止词列表(命令行难以传特殊 token 时可改用 stop_token_ids)
            parser.add_argument(
                "--rollout-stop",
                type=str,
                nargs="+",
                default=None,
                help=(
                    "The stop words for the inference engine during rollout. "
                    "It can be a list of strings or a single string. "
                    "It may be hard to pass special tokens in command line, in that case rollout_stop_token_ids can be used."
                ),
            )
            # rollout 的停止 token id 列表
            parser.add_argument(
                "--rollout-stop-token-ids",
                type=int,
                nargs="+",
                default=None,
                help=(
                    "The stop token ids for the inference engine during rollout. "
                    "It can be a list of integers or a single integer."
                ),
            )
            # 是否在 rollout 时打乱 prompt 顺序
            parser.add_argument(
                "--rollout-shuffle",
                action="store_true",
                default=False,
                help=("Whether to shuffle the prompts during rollout."),
            )
            # rollout 随机数种子,用于打乱 prompt 及随机抽样
            parser.add_argument(
                "--rollout-seed",
                type=int,
                default=42,
                help=(
                    "The seed for the random number generator during rollout. "
                    "This is used to shuffle the prompts and also for the random sampling of the prompts."
                ),
            )

            # sampling
            # 过采样(over-sampling)批大小:可用样本不足目标数时按该粒度补采;None 时取 rollout_batch_size
            parser.add_argument(
                "--over-sampling-batch-size",
                type=int,
                default=None,
                help=(
                    "This defines the granularity of the sampling batch in the rollout function. "
                    "When the number of available samples falls below the target, a sampling "
                    "operation of size over_sampling_batch_size will be triggered."
                    "Regardless of whether partial rollout is used or filters are applied, "
                    "the sampling granularity is always determined by this value. "
                    "If this value is None, rollout_batch_size will be used as the default over_sampling_batch_size."
                ),
            )
            # 动态采样过滤函数路径:按 DAPO 思路筛掉全对/全错等无梯度信号的 prompt 组
            parser.add_argument(
                "--dynamic-sampling-filter-path",
                type=str,
                default=None,
                help=(
                    "This is the filter function for dynamic sampling. "
                    "It should be able to judge whether the result of a prompt should be selected or not."
                    "We will do dynamic filter for sampling as in DAPO. e.g. not all correct or all wrong samples."
                    "You could use `slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std` as an example."
                ),
            )

            # partial rollout
            # 是否启用部分 rollout:动态采样中未完成的样本回收到数据缓冲区,适合长回复场景
            parser.add_argument(
                "--partial-rollout",
                action="store_true",
                default=False,
                help=(
                    "Whether to use partial rollout. "
                    "If set, the unfinished samples during dynamic sampling will be recycled back to data buffer. "
                    "This is useful for long responses."
                ),
            )
            # 部分 rollout 中是否屏蔽(previous generation)旧生成 token,只训练本轮 on-policy 产生的 token
            parser.add_argument(
                "--mask-offpolicy-in-partial-rollout",
                action="store_true",
                default=False,
                help=(
                    "Whether to mask previous generation in partial rollout. "
                    "If set, only on-policy generated tokens will be used in training"
                ),
            )
            # 仅替换示例 rollout 函数内部的 generate(args, sample, sampling_params),适合多轮/函数调用等特殊逻辑
            parser.add_argument(
                "--custom-generate-function-path",
                type=str,
                default=None,
                help=(
                    "Only substitue the `def generate(args, sample, sampling_params)` function within the example rollout function. "
                    "This should be useful if you need to implement some special rollout logic, e.g. multi-turn, function calling."
                ),
            )
            # 自定义 rollout 数据日志函数路径;返回 True 表示跳过默认日志逻辑
            parser.add_argument(
                "--custom-rollout-log-function-path",
                type=str,
                default=None,
                help=(
                    "The custom function for logging rollout data. The signature of the functions is: "
                    "def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool. "
                    "The return value indicates whether to skip the default logging. "
                ),
            )
            # 自定义评测 rollout 数据日志函数路径;返回 True 表示跳过默认日志逻辑
            parser.add_argument(
                "--custom-eval-rollout-log-function-path",
                type=str,
                default=None,
                help=(
                    "The custom function for logging eval rollout data. "
                    "def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool. "
                    "The return value indicates whether to skip the default logging. "
                ),
            )

            # 缓冲区过滤函数路径:从样本缓冲(list[list[Sample]])中筛选样本
            parser.add_argument(
                "--buffer-filter-path",
                type=str,
                default=None,
                help=(
                    "Path to the buffer filter function. "
                    "It should be able to select the samples in the buffer. "
                    "The function should take list[list[Sample]] and return list[list[Sample]]."
                ),
            )
            # update weight
            # 权重更新使用的分块缓冲区大小(字节),MoE 大模型分块同步权重时使用
            parser.add_argument(
                "--update-weight-buffer-size",
                type=int,
                default=512 * 1024**2,
                help=(
                    "buffer size for update weight, in bytes. "
                    "This is used for updating weights by chunk and should be useful for MoE models."
                ),
            )
            # 权重更新的间隔(每多少 rollout 步同步一次训练权重到推理引擎)
            parser.add_argument(
                "--update-weights-interval",
                type=int,
                default=1,
                help="Interval for updating the weights",
            )
            # 是否在训练进程中保留旧的 rollout 模型
            parser.add_argument(
                "--keep-old-actor",
                action="store_true",
                help="Whether to keep the rollout model on training process",
            )

            # rollout 数据(含 log_probs)收集完成后的后处理函数路径,可用于更新 loss mask
            parser.add_argument(
                "--rollout-data-postprocess-path",
                type=str,
                default=None,
                help=(
                    "The called after we have all the rollout data including log_probs. "
                    "It may be helpful for updating loss mask."
                ),
            )
            # 是否使用外部(自行拉起的)SGLang 实例,而非由框架内部启动
            parser.add_argument(
                "--rollout-external",
                action="store_true",
                default=False,
                help="Use external SGLang instances instead of launching them inside the framework.",
            )
            # 外部推理引擎的地址与端口列表
            parser.add_argument(
                "--rollout-external-engine-addrs",
                type=str,
                default=None,
                nargs="+",
                help="Address and ports of the external engines.",
            )
            # 返回追加了 rollout 参数的 parser
            return parser

        # ---------- 容错相关参数 ----------
        def add_fault_tolerance_arguments(parser):
            # 是否在 rollout 期间启用容错机制
            parser.add_argument(
                "--use-fault-tolerance",
                action="store_true",
                default=False,
                help="Whether to enable the fault tolerance function during rollout.",
            )
            # 两次 /health_generate 健康检查之间的间隔(秒)
            parser.add_argument(
                "--rollout-health-check-interval",
                type=float,
                default=30.0,
                help="Interval in seconds between rollout engine /health_generate checks during generate/eval.",
            )
            # 等待 /health_generate 响应的超时时间(秒),超时即杀掉该引擎
            parser.add_argument(
                "--rollout-health-check-timeout",
                type=float,
                default=30.0,
                help="Timeout in seconds to wait for a rollout engine /health_generate response before killing it.",
            )
            # 开始健康检查前的初始宽限期(秒),给模型编译/初始化留时间(使用 deepgemm 时应加大)
            parser.add_argument(
                "--rollout-health-check-first-wait",
                type=float,
                default=0,
                help="Initial grace period (in seconds) before starting health checks. This allows time for model compilation and initialization. Increase this value significantly when using deepgemm.",
            )
            # 返回追加了容错参数的 parser
            return parser

        # data
        # ---------- 数据相关参数 ----------
        def add_data_arguments(parser):
            # dataset
            # TODO: maybe add an num_epoch and calculate the num_rollout from buffer
            # rollout 总步数;不设置时按数据集规模与 num_epoch 推算
            parser.add_argument(
                "--num-rollout",
                type=int,
                default=None,
                help="Number of rollout steps. If not set, we will calculate the number of rollout steps from the dataset size.",
            )
            # 训练 epoch 数,用于从数据集规模推算 rollout 步数;与 --num-rollout 同设时本项被忽略
            parser.add_argument(
                "--num-epoch",
                type=int,
                default=None,
                help=(
                    "Number of epochs for the training. "
                    "This is used to calculate the number of rollout steps from the dataset size. "
                    "If set, we will calculate the number of rollout steps as `num_rollout = num_epoch * dataset_size // rollout_batch_size`."
                    "If both `--num-epoch` and `--num-rollout` are set, `--num-epoch` will be ignored."
                ),
            )

            # 通过 store_false 把 rollout_global_dataset 置 False:关闭全局数据集,改为用户自行管理数据
            parser.add_argument(
                "--disable-rollout-global-dataset",
                action="store_false",
                dest="rollout_global_dataset",
                help=(
                    "Whether to use a global dataset for rollout. "
                    "If set, the rollout will use the `--prompt-data` as the prompt dataset, "
                    "and the prompts for rollout will be sampled from the dataset. "
                    "If not set, you need to manage the data by your self."
                ),
            )

            # rollout 数据源类路径(含缓冲区管理的数据源实现)
            parser.add_argument(
                "--data-source-path",
                type=str,
                default="slime.rollout.data_source.RolloutDataSourceWithBuffer",
                help="The data source class for rollout data.",
            )
            # 提示数据集路径(支持 jsonl/parquet),每行含 input/label 等 key
            parser.add_argument(
                "--prompt-data",
                type=str,
                default=None,
                help=(
                    "The path to the prompt data. "
                    "Currently we only support jsonl format, and each line should contains --input-key and --label-key, "
                    "which will be used as the prompt and the label respectively. "
                    "If you want to use a custom template, you can set --apply-chat-template to true, in that case, "
                    "the input should be the same structure as an openai message, e.g. [{'role': 'user', 'content': 'blabla'}]. "
                ),
            )
            # 是否对数据集中的 prompt 应用对话模板(chat template)
            parser.add_argument("--apply-chat-template", action="store_true", default=False)
            # Temporarily be JSON-serialized str, will be a real dict after using Omegaconf
            # 应用对话模板时的附加参数(JSON 字符串)
            parser.add_argument("--apply-chat-template-kwargs", type=json.loads, default="{}")
            # 数据集中 prompt 字段的 key 名
            parser.add_argument("--input-key", type=str, default="input", help="JSON dataset key")
            # 数据集中标签(label)字段的 key 名
            parser.add_argument("--label-key", type=str, default=None, help="JSON dataset key")
            # 多模态数据映射(JSON 字符串):媒体类型 -> 数据集字段名
            parser.add_argument(
                "--multimodal-keys",
                type=json.loads,
                default=None,
                help=(
                    'JSON string for multimodal data mapping media types to data keys. Example: \'{"image": "image_file"}\''
                ),
            )
            # 数据集中元数据字段的 key 名
            parser.add_argument("--metadata-key", type=str, default="metadata", help="JSON dataset key")
            # 数据集中工具(tools)定义字段的 key 名,供对话模板注入工具描述
            parser.add_argument(
                "--tool-key",
                type=str,
                default="tools",
                help=(
                    "When need to add tools during apply_chat_template, you should provide the key for the tools in the prompt dataset."
                ),
            )

            # 起始 rollout 步号;续训时从 --load 恢复,否则从 0 开始
            parser.add_argument(
                "--start-rollout-id",
                type=int,
                default=None,
                help=(
                    "The starting rollout step, if not set, will try to load the step from --load when doing continue training, "
                    "otherwise will be set to 0, meaning training from start."
                ),
            )

            # batch sizes
            # 每个 rollout 步采样的 prompt 数(必填);一次返回的总样本数为 prompt 数 × 每条样本采样数
            parser.add_argument(
                "--rollout-batch-size",
                type=int,
                required=True,
                help=(
                    "The number of prompts in each rollout step. "
                    "The total data returned should be rollout_batch_size * n_samples_per_prompt. "
                ),
            )
            # 每个 prompt 生成的回复条数(GRPO 中的组大小 G)
            parser.add_argument(
                "--n-samples-per-prompt", type=int, default=1, help="Number of responses for each prompt in generation"
            )

            # gbs of the training, note that the gbs is of sample, not of prompts,
            # so if you hope to train 1 step for each rollout, the global_bach_size should be set as
            # `rollout_batch_size * n_samples_per_prompt`.
            # 重置训练全局批大小(gbs)默认值;注意 gbs 以样本为单位而非 prompt
            reset_arg(parser, "--global-batch-size", type=int, default=None)
            # 每个 rollout 步对应的训练步数,等价于把 gbs 设为 rollout_batch_size*n_samples_per_prompt//该值
            parser.add_argument(
                "--num-steps-per-rollout",
                type=int,
                default=None,
                help=(
                    "Number of steps per rollout, e.g. It is equivalent to setting gbs as "
                    "`rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout`."
                ),
            )
            # mbs for the training, will be ignored if `use_dynamic_batch_size` is set.
            # 重置训练 micro batch size 默认值;启用动态批大小时本项被忽略
            reset_arg(parser, "--micro-batch-size", type=int, default=1)
            # 是否用 karmarkar_karp 算法在 DP 各 rank 间均衡 token 数
            parser.add_argument(
                "--balance-data",
                action="store_true",
                default=False,
                help=(
                    "Balance the number of tokens between data parallel ranks with `karmarkar_karp` for verl. "
                    "Note that this may allocate the different response of the same prompt into different training steps."
                ),
            )

            # 是否按每卡最大 token 数动态切分 micro batch,以最大化 GPU 利用率
            parser.add_argument(
                "--use-dynamic-batch-size",
                action="store_true",
                default=False,
                help=(
                    "Because the sample length varies, to maximize the GPU utilization, "
                    "we will use the dynamic batch size to adjust the micro batch size according to the maximum number of tokens each gpu can run. "
                    "For example, if we have 3 samples, with the length of 100, 200, and 300, and the max_tokens_per_gpu is 300, when enabling "
                    "dynamic batch size, slime will make 2 micro batches, i.e. [100, 200], [300]."
                ),
            )
            # 动态批大小下每 GPU 的最大 token 数(启用 CP 时约取 max_response_len//cp_size)
            parser.add_argument(
                "--max-tokens-per-gpu",
                type=int,
                default=None,
                help=(
                    "The maximum number of tokens per GPU for dynamic batch size. "
                    "Note that when enabling context parallel (CP), the max tokens per gpu should be around "
                    "`max_response_len // cp_size` instead of `max_response_len`."
                ),
            )
            # 计算 log probs 时每 GPU 的最大 token 数(可比 max_tokens_per_gpu 大以提升性能)
            parser.add_argument(
                "--log-probs-max-tokens-per-gpu",
                type=int,
                default=None,
                help=(
                    "The maximum number of tokens per GPU for calculating log probs. "
                    "This is used to calculate the log probs of the responses during rollout, "
                    "and should be set to a larger value than `max_tokens_per_gpu` if you want better performance. "
                ),
            )
            # 返回追加了数据参数的 parser
            return parser

        # ---------- 评测相关参数 ----------
        def add_eval_arguments(parser):
            # 评测生成函数路径;不设置时复用 rollout_function_path
            parser.add_argument(
                "--eval-function-path",
                type=str,
                default=None,
                help=(
                    "Path to the eval generation function."
                    "If not set, we will use rollout_function_path as the default. "
                ),
            )

            # change the default value of eval_interval from Megatron to None
            # 把 Megatron 的 eval_interval 默认值改为 None(默认不周期评测)
            reset_arg(parser, "--eval-interval", type=int, default=None)

            # 评测提示数据:按 "名称 路径" 成对给出,如 aime /path/to/aime.jsonl
            parser.add_argument(
                "--eval-prompt-data",
                type=str,
                default=None,
                nargs="+",
                help=(
                    "Path to the evaluation prompt data, "
                    "should first input the name of the eval dataset and then the path, e.g. "
                    "aime /path/to/aime.jsonl"
                ),
            )
            # 评测数据集的 OmegaConf YAML/JSON 配置文件路径;设置后覆盖 --eval-prompt-data
            parser.add_argument(
                "--eval-config",
                type=str,
                default=None,
                help=(
                    "Path to an OmegaConf YAML/JSON file describing evaluation datasets. "
                    "When provided, this overrides --eval-prompt-data."
                ),
            )
            # 是否跳过训练开始前的评测
            parser.add_argument(
                "--skip-eval-before-train",
                action="store_true",
                default=False,
                help="Whether to skip evaluation before training.",
            )

            # The following keys are used to override the rollout version during eval.
            # 评测时覆盖输入字段 key 名
            parser.add_argument("--eval-input-key", type=str, default=None, help="JSON dataset key")
            # 评测时覆盖标签字段 key 名
            parser.add_argument("--eval-label-key", type=str, default=None, help="JSON dataset key")
            # 评测时覆盖工具字段 key 名
            parser.add_argument("--eval-tool-key", type=str, default=None, help="JSON dataset key")
            # 评测时每个 prompt 生成的回复数
            parser.add_argument(
                "--n-samples-per-eval-prompt",
                type=int,
                default=1,
                help="number of responses for each prompt in generation",
            )
            # 评测采样温度(不设时回退 rollout/数据集配置)
            parser.add_argument("--eval-temperature", type=float, default=None)
            # 评测 top-p(不设时回退 rollout/数据集配置)
            parser.add_argument("--eval-top-p", type=float, default=None)
            # 评测 top-k(不设时回退 rollout/数据集配置)
            parser.add_argument("--eval-top-k", type=int, default=None)
            # 评测最大生成长度
            parser.add_argument("--eval-max-response-len", type=int, default=None)
            # 评测最大 prompt 长度
            parser.add_argument("--eval-max-prompt-len", type=int, default=None)
            # 评测最小生成 token 数
            parser.add_argument("--eval-min-new-tokens", type=int, default=None)
            # 评测最大上下文长度(不设时取 rollout_max_context_len)
            parser.add_argument("--eval-max-context-len", type=int, default=None)

            # 返回追加了评测参数的 parser
            return parser

        # ---------- 强化学习算法相关参数 ----------
        def add_algo_arguments(parser):
            # 参考模型(ref model)权重路径;未设 --load 时兼作训练初始权重
            parser.add_argument(
                "--ref-load",
                type=str,
                default=None,
                help=(
                    "The checkpoint for reference model. "
                    "When --load is not set, this will be used as the initial checkpoint for training. "
                ),
            )
            # 参考模型权重所在的 checkpoint 步号
            parser.add_argument(
                "--ref-ckpt-step", type=int, default=None, help="The checkpoint step for reference model. "
            )
            # 重置 Megatron 的加载路径默认值为 None
            reset_arg(parser, "--load", type=str, default=None)
            # 重置保存路径默认值为 None
            reset_arg(parser, "--save", type=str, default=None)
            # 重置保存间隔默认值为 None(默认不周期保存)
            reset_arg(parser, "--save-interval", type=int, default=None)
            # 重置异步保存开关
            reset_arg(parser, "--async-save", action="store_true")
            # 重置"不保存优化器状态"开关:省 checkpoint 体积但无法从该 checkpoint 恢复训练
            reset_arg(
                parser,
                "--no-save-optim",
                action="store_true",
                default=False,
                help=(
                    "If set, do not save the optimizer state when saving checkpoints. "
                    "This reduces checkpoint size but disables training resumption from the saved checkpoint."
                ),
            )
            # Megatron 后端以 HuggingFace 格式保存模型的路径(按 rollout_id 格式化)
            parser.add_argument(
                "--save-hf",
                type=str,
                default=None,
                help=(
                    "Path to save the model in HuggingFace format when using Megatron backend. "
                    "The model will be saved to `save_hf.format(rollout_id)`. "
                ),
            )
            # 重置随机种子默认值为 1234
            reset_arg(parser, "--seed", type=int, default=1234)
            # 重置梯度裁剪阈值默认值为 1.0
            reset_arg(parser, "--clip-grad", type=float, default=1.0)
            # 重置按 token 计损失(per-token loss)开关
            reset_arg(parser, "--calculate-per-token-loss", action="store_true")
            # 重置学习率默认值为 1e-6(RL 微调常用小学习率)
            reset_arg(parser, "--lr", type=float, default=1e-6)

            # 每轮只训练 critic(不训练 actor)的步数
            parser.add_argument("--num-critic-only-steps", type=int, default=0, help="Number of critic only steps")
            # critic 模型的加载路径
            parser.add_argument("--critic-load", type=str, default=None, help="The checkpoint for critic model.")
            # critic 模型的保存路径
            parser.add_argument("--critic-save", type=str, default=None, help="The checkpoint for critic model.")
            # critic 的学习率(不设时取 --lr)
            parser.add_argument("--critic-lr", type=float, default=None, help="The lr for critic model")
            # critic 学习率线性预热的迭代步数
            parser.add_argument(
                "--critic-lr-warmup-iters",
                type=int,
                default=0,
                help="number of iterations to linearly warmup for critic model.",
            )

            # PPO 策略梯度裁剪范围 epsilon(论文 PPO 的 clip 参数)
            parser.add_argument("--eps-clip", type=float, default=0.2, help="PPO clip range")
            # PPO 裁剪上界(非对称裁剪,如 DAPO 的 clip-higher);不设时取 eps_clip
            parser.add_argument("--eps-clip-high", type=float, default=None, help="PPO clip upper range")
            # Dual-clip PPO 的价值下界 c(见论文 https://arxiv.org/pdf/1912.09729)
            parser.add_argument(
                "--eps-clip-c",
                type=float,
                default=None,
                help="lower bound of the value for Dual-clip PPO from https://arxiv.org/pdf/1912.09729",
            )
            # 价值损失(value loss)的裁剪范围
            parser.add_argument("--value-clip", type=float, default=0.2, help="the clip for value loss")
            # KL 惩罚系数:作用于奖励塑形(reward shaping),在优势计算前施加
            parser.add_argument(
                "--kl-coef",
                type=float,
                default=0.00,
                help="KL penalty coefficient for reward shaping. This is applied to the reward signal before advantage calculation.",
            )
            # 损失类型:policy_loss(PPO/GRPO 策略损失)、sft_loss 或 custom_loss(自定义函数)
            parser.add_argument(
                "--loss-type",
                type=str,
                choices=["policy_loss", "sft_loss", "custom_loss"],
                default="policy_loss",
                help=(
                    "Choose loss type, currently support ppo policy_loss or sft_loss, "
                    "if custom_loss is set, we will use the function path from `--custom-loss-function-path`."
                ),
            )
            # 自定义损失函数路径(loss_type=custom_loss 时启用)
            parser.add_argument(
                "--custom-loss-function-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom loss function, if the loss_type is `custom_loss`, "
                    "we will use this function to calculate the loss. "
                ),
            )
            # KL 散度损失的形式:k1/k2/k3 或低方差估计 low_var_kl
            parser.add_argument(
                "--kl-loss-type",
                type=str,
                choices=["k1", "k2", "k3", "low_var_kl"],
                default="k1",
                help="Choose KL loss type: kl, k2, k3, low_var_kl",
            )
            # 优势估计器(advantage estimator)类型:GRPO/GSPO/REINFORCE++/PPO 等;OPD 与之正交
            parser.add_argument(
                "--advantage-estimator",
                type=str,
                choices=[
                    "grpo",
                    "gspo",
                    "reinforce_plus_plus",
                    "reinforce_plus_plus_baseline",
                    "ppo",
                ],
                default="grpo",
                help=(
                    "Advantage estimator to use. Note: on-policy distillation (OPD) is now orthogonal "
                    "to the advantage estimator. Use --opd-kl-coef > 0 to enable OPD on top of any estimator."
                ),
            )
            # 通过 store_false 关闭优势与回报的计算(SFT 或自定义损失时使用)
            parser.add_argument(
                "--disable-compute-advantages-and-returns",
                action="store_false",
                dest="compute_advantages_and_returns",
                help=(
                    "Whether to disable computing advantages and returns. "
                    "If set, we will not compute the advantages and returns, "
                    "This is useful for sft or custom loss function."
                ),
            )
            # 是否在损失中加入 GRPO 式的 KL 惩罚项
            parser.add_argument(
                "--use-kl-loss", action="store_true", default=False, help="whether to use KL loss from GRPO"
            )
            # 损失函数中的 KL 惩罚系数,直接加到最终 PPO 损失上
            parser.add_argument(
                "--kl-loss-coef",
                type=float,
                default=0.0,
                help="KL penalty coefficient for the loss function. This is added to the final PPO loss.",
            )
            # 是否启用无偏 KL 估计
            parser.add_argument(
                "--use-unbiased-kl",
                action="store_true",
                default=False,
                help="Whether to enable unbiased KL estimation.",
            )
            # 参考模型从 actor 同步更新的间隔(rollout 步数);None 表示不更新
            parser.add_argument(
                "--ref-update-interval",
                type=int,
                default=None,
                help="Interval (in rollout steps) to update ref model from actor. If None, ref model is not updated.",
            )
            # 熵损失(entropy bonus)系数,鼓励探索
            parser.add_argument("--entropy-coef", type=float, default=0.0, help="Entropy loss coef")
            # PPO GAE 的折扣因子 gamma
            parser.add_argument("--gamma", type=float, default=1.0, help="PPO GAE gamma")
            # PPO GAE 的 GAE 参数 lambda
            parser.add_argument("--lambd", type=float, default=1.0, help="PPO GAE lambd")
            # 是否对优势(advantage)做归一化
            parser.add_argument("--normalize-advantages", action="store_true", default=False)
            # 关闭 GRPO 组内标准差归一化(见 Dr.GRPO 论文 https://arxiv.org/pdf/2503.20783)
            parser.add_argument(
                "--disable-grpo-std-normalization",
                action="store_false",
                dest="grpo_std_normalization",
                help="from Dr.GRPO https://arxiv.org/pdf/2503.20783",
            )
            # 关闭奖励归一化
            parser.add_argument(
                "--disable-rewards-normalization",
                action="store_false",
                dest="rewards_normalization",
                help="Disable rewards normalization",
            )
            # 计算 actor/ref 的 logprobs 时是否同时计算熵(可用于特殊 loss mask)
            parser.add_argument(
                "--use-rollout-entropy",
                action="store_true",
                default=False,
                help=(
                    "Whether to calculate the entropy when calculating the logprobs from actor and reference model. "
                    "This is useful for doing special loss mask."
                ),
            )
            # 是否计算训练/推理引擎 logprobs 的失配(mismatch)诊断指标
            parser.add_argument(
                "--get-mismatch-metrics",
                action="store_true",
                default=False,
                help="Whether to calculate the mismatch metrics.",
            )
            # 是否在每个 rollout 结束后清空优化器状态(有助于训练稳定性或特定实验)
            parser.add_argument(
                "--reset-optimizer-states",
                action="store_true",
                default=False,
                help=(
                    "Whether to reset optimizer states after each rollout. "
                    "If enabled, the optimizer's history will be cleared at the end of each rollout, which can sometimes help with training stability or fulfill specific experiment requirements."
                ),
            )
            # 计算重要性采样比率时是否直接采用 rollout 引擎返回的 logprobs(而非 actor 重算值)
            parser.add_argument(
                "--use-rollout-logprobs",
                action="store_true",
                default=False,
                help=(
                    "Whether to use the rollout logprobs when calculating the importance sampling ratios. "
                    "If not set, we will use the logprobs from the actor model."
                ),
            )
            # Off-Policy Correction using Importance Sampling: https://fengyao.notion.site/off-policy-rl
            # 启用 TIS:对 off-policy 样本做重要性采样校正
            parser.add_argument(
                "--use-tis",
                action="store_true",
                default=False,
                help="Enable TIS from https://fengyao.notion.site/off-policy-rl for off-policy importance sampling.",
            )
            # 重要性采样比率的上界裁剪阈值 C,用于控制方差
            parser.add_argument(
                "--tis-clip",
                type=float,
                default=2.0,
                help="Clipping threshold C for importance sampling ratios to control variance.",
            )
            # 重要性采样比率的下界裁剪阈值 C
            parser.add_argument(
                "--tis-clip-low",
                type=float,
                default=0,
                help="Lower bound clipping threshold C for importance sampling ratios to control variance.",
            )
            # 自定义 TIS/RS(重要性采样/奖励塑形)函数路径
            parser.add_argument(
                "--custom-tis-function-path",
                type=str,
                default=None,
                help="Path to the custom TIS/RS function (e.g., examples/train_infer_mismatch_helper/mis.py:compute_mis_weights_with_cp).",
            )
            # 自定义 pg_loss 归约器(reducer)路径;其余指标仍用默认 sum_of_sample_mean
            parser.add_argument(
                "--custom-pg-loss-reducer-function-path",
                type=str,
                default=None,
                help="Path to a custom reducer function for pg_loss only. When set, pg_loss will use this custom reducer while other metrics (pg_clipfrac, ppo_kl, entropy_loss, etc.) still use the default sum_of_sample_mean. (e.g., examples/Dr.GRPO/custom_reducer.py:get_pg_loss_reducer).",
            )

            # 启用 MoE 路由回放(routing replay)技术(论文 https://arxiv.org/abs/2507.18071)
            parser.add_argument(
                "--use-routing-replay",
                action="store_true",
                default=False,
                help="The routing replay technique from https://arxiv.org/abs/2507.18071",
            )
            # 启用 rollout 路由回放技术(论文 https://arxiv.org/abs/2510.11370)
            parser.add_argument(
                "--use-rollout-routing-replay",
                action="store_true",
                default=False,
                help="The rollout routing replay technique from https://arxiv.org/abs/2510.11370",
            )
            # 是否启用离线策略序列掩码(Off-Policy Sequence Masking, OPSM)
            parser.add_argument(
                "--use-opsm",
                action="store_true",
                default=False,
                help="Whether to enable Off-Policy Sequence Masking (OPSM).",
            )
            # OPSM 判定离线策略序列的阈值 delta
            parser.add_argument(
                "--opsm-delta",
                type=float,
                default=1e-4,
                help="The threshold for Off-Policy Sequence Masking (OPSM).",
            )
            # 返回追加了算法参数的 parser
            return parser

        # ---------- 在线策略蒸馏(OPD)参数 ----------
        def add_on_policy_distillation_arguments(parser):
            """Add on-policy distillation (OPD) related arguments.

            OPD is orthogonal to advantage estimators and can be applied on top of
            any estimator (GRPO, PPO, etc.) by adding a KL penalty to advantages.
            """
            # 是否启用在线策略蒸馏(on-policy distillation)
            parser.add_argument(
                "--use-opd",
                action="store_true",
                default=False,
                help="Enable on-policy distillation (OPD). Must specify --opd-type when enabled.",
            )
            # OPD 实现方式:sglang(rollout 期从外部服务取教师 logprobs)或 megatron(训练期加载教师前向)
            parser.add_argument(
                "--opd-type",
                type=str,
                choices=["sglang", "megatron"],
                default=None,
                help=(
                    "Type of on-policy distillation. "
                    "'sglang': Teacher log-probs are obtained from external SGLang server during rollout. "
                    "'megatron': Teacher model is loaded via --opd-teacher-load and forwarded during training."
                ),
            )
            # OPD 的 KL 惩罚系数,默认 1.0
            parser.add_argument(
                "--opd-kl-coef",
                type=float,
                default=1.0,
                help="On-policy distillation KL penalty coefficient. Default is 1.0.",
            )
            # OPD 教师模型权重路径(opd_type=megatron 时必填,结构与策略模型一致)
            parser.add_argument(
                "--opd-teacher-load",
                type=str,
                default=None,
                help=(
                    "The checkpoint for OPD teacher model. Required when --opd-type=megatron. "
                    "The teacher model should have the same architecture as policy/ref model."
                ),
            )
            # OPD 教师模型的 checkpoint 步号
            parser.add_argument(
                "--opd-teacher-ckpt-step", type=int, default=None, help="The checkpoint step for OPD teacher model."
            )
            # 返回追加了 OPD 参数的 parser
            return parser

        # ---------- Slime 路由器参数 ----------
        def add_router_arguments(parser):
            # 是否用 SlimeRouter 做基于文本的路由(替代 SGLang 的 token 级路由)
            parser.add_argument(
                "--use-slime-router",
                action="store_true",
                default=False,
                help="Whether to use SlimeRouter for text-based routing instead of SGLang token-based routing",
            )
            # SlimeRouter 要加载的中间件模块路径列表
            parser.add_argument(
                "--slime-router-middleware-paths",
                type=str,
                nargs="+",
                default="",
            )
            # SlimeRouter HTTP 请求超时时间(秒)
            parser.add_argument(
                "--slime-router-timeout",
                type=float,
                default=None,
                help="Timeout for SlimeRouter HTTP requests in seconds.",
            )
            # SlimeRouter HTTP 客户端的最大连接数
            parser.add_argument(
                "--slime-router-max-connections",
                type=int,
                default=None,
                help="Max connections for SlimeRouter HTTP client.",
            )
            # worker 连续失败多少次健康检查后才被标记为不健康并隔离
            parser.add_argument(
                "--slime-router-health-check-failure-threshold",
                type=int,
                default=3,
                help="Number of consecutive failures before marking a worker as unhealthy.",
            )
            # 把 sglang-router 的 CLI 参数以 slime_router_ 前缀注册进来(排除 host/port)
            RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)
            # 返回追加了路由器参数的 parser
            return parser

        # wandb
        # ---------- W&B 实验追踪参数 ----------
        def add_wandb_arguments(parser):
            # wandb parameters
            # 是否启用 Weights & Biases 日志
            parser.add_argument("--use-wandb", action="store_true", default=False)
            # W&B 运行模式:online/offline/disabled,覆盖 WANDB_MODE 环境变量
            parser.add_argument(
                "--wandb-mode",
                type=str,
                default=None,
                choices=["online", "offline", "disabled"],
                help="W&B mode: online (default), offline (local only), or disabled. Overrides WANDB_MODE env var.",
            )
            # wandb 日志存放目录,默认当前目录下 ./wandb
            parser.add_argument(
                "--wandb-dir",
                type=str,
                default=None,
                help="Directory to store wandb logs. Default is ./wandb in current directory.",
            )
            # W&B API key
            parser.add_argument("--wandb-key", type=str, default=None)
            # W&B 服务主机地址(自建 W&B 时使用)
            parser.add_argument("--wandb-host", type=str, default=None)
            # W&B 团队名
            parser.add_argument("--wandb-team", type=str, default=None)
            # W&B 分组名
            parser.add_argument("--wandb-group", type=str, default=None)
            # 重置 wandb 项目名默认值为 None
            reset_arg(parser, "--wandb-project", type=str, default=None)
            # 用 store_false 关闭 run 名随机后缀(默认追加 6 位随机字符串)
            parser.add_argument(
                "--disable-wandb-random-suffix",
                action="store_false",
                dest="wandb_random_suffix",
                default=True,
                help=(
                    "Whether to add a random suffix to the wandb run name. "
                    "By default, we will add a random 6 length string with characters to the run name."
                ),
            )
            # 是否始终以训练步号作为 wandb 的 step 指标(否则多数信息用 rollout 步号)
            parser.add_argument(
                "--wandb-always-use-train-step",
                action="store_true",
                default=False,
                help=(
                    "Whether to always use train step as the step metric in wandb. "
                    "If set, we will always use the train steps for wandb logging, "
                    "otherwise, will use rollout step for most info other than train/*. "
                ),
            )
            # 是否记录多轮(multi-turn)rollout 的详细信息
            parser.add_argument(
                "--log-multi-turn",
                action="store_true",
                default=False,
                help="Whether to log information for multi-turn rollout.",
            )
            # 是否开启 pass@n 通过率日志
            parser.add_argument(
                "--log-passrate",
                action="store_true",
                default=False,
                help="Whether to turn on passrate logging, which will log the pass@n of the responses in the rollout.",
            )
            # 记录奖励失败原因类别统计:指定奖励 dict 中取类别的 key
            parser.add_argument(
                "--log-reward-category",
                type=str,
                default=None,
                help=(
                    "Log statistics of the category of reward, such as why the reward function considers it as failed. "
                    "Specify the key in the reward dict using this argument.",
                ),
            )
            # 是否记录正确样本相关统计
            parser.add_argument(
                "--log-correct-samples",
                action="store_true",
                default=False,
                help="Whether to turn on passrate logging, which will log the pass@n of the responses in the rollout.",
            )
            # 指定 wandb run id(用于续接同一个 run)
            parser.add_argument("--wandb-run-id", type=str, default=None)
            # 返回追加了 W&B 参数的 parser
            return parser

        # tensorboard
        # ---------- TensorBoard 日志参数 ----------
        def add_tensorboard_arguments(parser):
            # tb_project_name, tb_experiment_name
            # 是否启用 TensorBoard 日志
            parser.add_argument("--use-tensorboard", action="store_true", default=False)
            # TensorBoard 日志目录(默认取环境变量 TENSORBOARD_DIR)
            parser.add_argument(
                "--tb-project-name",
                type=str,
                default=None,
                help="Directory to store tensorboard logs. Default is  os.environ.get('TENSORBOARD_DIR') directory.",
            )
            # TensorBoard 实验名
            parser.add_argument("--tb-experiment-name", type=str, default=None)

            # 返回追加了 TensorBoard 参数的 parser
            return parser

        # debug
        # ---------- 调试相关参数 ----------
        def add_debug_arguments(parser):
            # 把 rollout 数据保存到指定路径(按 rollout_id 格式化)用于调试
            parser.add_argument(
                "--save-debug-rollout-data",
                type=str,
                default=None,
                help=(
                    "Save the rollout data to this path for debugging. "
                    "The file will be saved to `save_debug_rollout_data.format(rollout_id)`."
                ),
            )
            # --load-debug-rollout-data, --debug-rollout-only, --debug-train-only
            # are parsed early in _pre_parse_mode() and merged later.
            # 对调试 rollout 数据做子采样(比例),加快调试迭代
            parser.add_argument(
                "--load-debug-rollout-data-subsample",
                type=float,
                default=None,
                help="Subsample a portion of the debug rollout data for faster debugging.",
            )
            # 把训练数据保存到指定路径(按 rollout_id 格式化)用于调试
            parser.add_argument(
                "--save-debug-train-data",
                type=str,
                default=None,
                help=(
                    "Save the train data to this path for debugging. "
                    "The file will be saved to `save_debug_train_data.format(rollout_id)`."
                ),
            )
            # 导出训练全过程明细,供事后分析与可视化
            parser.add_argument(
                "--dump-details",
                type=str,
                default=None,
                help=("Dump all details of training for post-hoc analysis and visualization."),
            )
            # use together with --record-memory-history and --memory-snapshot-path (defined in Megatron)
            # 显存快照(snapshot)输出目录;配合 Megatron 的 record-memory-history 等参数使用
            parser.add_argument(
                "--memory-snapshot-dir",
                type=str,
                default=".",
            )
            # 采集显存快照的训练步数
            parser.add_argument(
                "--memory-snapshot-num-steps",
                type=int,
                default=None,
            )
            # 性能剖析目标:整体训练 / actor / log_probs 计算(可多选)
            parser.add_argument(
                "--profile-target",
                type=str,
                choices=["train_overall", "train_actor", "train_log_probs"],
                default=["train_overall"],
                nargs="+",
            )
            # 显存记录器实现:torch 或 memray
            parser.add_argument(
                "--memory-recorder",
                type=str,
                choices=["torch", "memray"],
                default="torch",
            )
            # 是否校验权重更新前后一致(调试权重同步)
            parser.add_argument("--check-weight-update-equal", action="store_true")
            # 返回追加了调试参数的 parser
            return parser

        # ---------- 网络相关参数 ----------
        def add_network_arguments(parser):
            # HTTP 代理地址(供数据下载/外部服务访问使用)
            parser.add_argument("--http-proxy", type=str, default=None)
            # 是否启用分布式 POST(多节点并行向远程服务发请求)
            parser.add_argument("--use-distributed-post", action="store_true", default=False)
            # 返回追加了网络参数的 parser
            return parser

        # ---------- 奖励模型(RM)参数 ----------
        def add_reward_model_arguments(parser):
            # 奖励模型类型(如 remote_rm、自定义函数等)
            parser.add_argument(
                "--rm-type",
                type=str,
                default=None,
                help="Type of the reward model",
            )
            # 奖励函数返回 dict 时从中取奖励值的 key
            parser.add_argument(
                "--reward-key",
                type=str,
                default=None,
                help=(
                    "Some reward model may return a dict instead of a value, "
                    "this is the key to extract the reward value from the dict. "
                ),
            )
            # --reward-key 的评测专用变体
            parser.add_argument(
                "--eval-reward-key",
                type=str,
                default=None,
                help="The eval variant for --reward-key",
            )
            # 是否对整组(group)样本统一打分
            parser.add_argument(
                "--group-rm", action="store_true", default=False, help="Whether to do rm on a whole group."
            )
            # remote_rm 类型奖励模型的服务地址
            parser.add_argument(
                "--rm-url",
                type=str,
                default=None,
                help="URL for the reward model service for --rm-type remote_rm, e.g. http://localhost:8000",
            )
            # 自定义奖励函数路径(签名 def custom_rm(args, sample) -> float)
            parser.add_argument(
                "--custom-rm-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom reward model function. "
                    "If set, we will use this function to calculate the reward instead of the default one. "
                    "The function should have the signature `def custom_rm(args, sample) -> float`."
                ),
            )
            # 奖励后处理函数路径(默认做 GRPO 式归一化)
            parser.add_argument(
                "--custom-reward-post-process-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom function that will post process reward, by default it will be the normalization for grpo. "
                ),
            )
            # 自定义"样本转训练数据"函数路径(替换默认的 _convert_samples_to_train_data)
            parser.add_argument(
                "--custom-convert-samples-to-train-data-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom function that converts samples to training data. "
                    "If set, this function will replace the default _convert_samples_to_train_data. "
                    "The function should have the signature `def convert_samples_to_train_data(args, samples) -> dict`."
                ),
            )
            # 返回追加了奖励模型参数的 parser
            return parser

        # ---------- rollout 缓冲区参数 ----------
        def add_rollout_buffer_arguments(parser):
            # rollout 样本缓冲区(buffer)服务的 URL
            parser.add_argument(
                "--rollout-buffer-url",
                type=str,
                default=None,
                help="URL for the rollout buffer",
            )

            # 拉取轨迹(trajectory)的重试次数;-1 表示无限重试
            parser.add_argument(
                "--fetch-trajectory-retry-times",
                type=int,
                default=-1,
                help="Number of times to retry fetching trajectory, -1 means unlimited retry",
            )
            # 允许的最小批次收集比例(实际收到的样本数/目标数)
            parser.add_argument(
                "--min-batch-collection-ratio",
                type=float,
                default=1,
                help="Minimum batch collection ratio",
            )
            # rollout 任务类型(默认 math)
            parser.add_argument(
                "--rollout-task-type",
                type=str,
                default="math",
            )
            # 损失掩码(loss mask)的生成风格:qwen/qwen3/qwen3_5/distill_qwen
            parser.add_argument(
                "--loss-mask-type",
                type=str,
                default="qwen",
                choices=["qwen", "qwen3", "qwen3_5", "distill_qwen"],
                help="Loss mask type",
            )
            # 数据处理时填充(pad)尺寸的倍数
            parser.add_argument(
                "--data-pad-size-multiplier",
                type=int,
                default=128,
                help="Multiplier for data padding size in data processing.",
            )
            # 样本过滤函数路径:决定样本是否参与损失计算(直接设置 Sample.remove_sample 属性)
            parser.add_argument(
                "--rollout-sample-filter-path",
                type=str,
                default=None,
                help=(
                    "Path to the rollout sample filter function. "
                    "This function determines whether a sample will participate in loss calculation. "
                    "The function should take args and samples (list[Sample]) as input, and return None. "
                    "Please directly modify the remove_sample attribute of Sample. "
                    "Note: This attribute does not determine whether the sample participates in advantage normalization."
                ),
            )
            # 对全部样本(含被过滤者)统一处理的函数路径
            parser.add_argument(
                "--rollout-all-samples-process-path",
                type=str,
                default=None,
                help=(
                    "Path to the rollout all samples process function that "
                    "can process all samples including filtered ones."
                ),
            )
            # 关闭样本转训练数据时的裁剪(trim samples)逻辑
            parser.add_argument(
                "--disable-rollout-trim-samples",
                action="store_true",
                default=False,
                help="disable trim samples in rollout buffer when converting samples to train data",
            )
            # 启用动态全局批大小(同时关闭样本裁剪)
            parser.add_argument(
                "--use-dynamic-global-batch-size",
                action="store_true",
                default=False,
                help="enable dynamic global batch size, disable trim samples in rollout buffer when converting samples to train data",
            )
            return parser

        # ---------- 自定义 Megatron 插件参数 ----------
        def add_custom_megatron_plugins_arguments(parser):
            """
            Add custom Megatron plugins arguments.
            This is a placeholder for any additional arguments that might be needed.
            """
            # Custom arguments can be added here
            # 自定义 Megatron 初始化函数路径
            parser.add_argument(
                "--custom-megatron-init-path",
                type=str,
                default=None,
            )
            # 计算 log probs 之前调用的钩子函数路径
            parser.add_argument(
                "--custom-megatron-before-log-prob-hook-path",
                type=str,
                default=None,
            )
            # 每个训练步之前调用的钩子函数路径
            parser.add_argument(
                "--custom-megatron-before-train-step-hook-path",
                type=str,
                default=None,
            )
            # 返回追加了插件参数的 parser
            return parser

        # ---------- MTP(多 token 预测)训练参数 ----------
        def add_mtp_training_arguments(parser):
            """Add MTP training specific arguments."""
            # 重置 MTP 层数默认值为 None
            reset_arg(parser, "--mtp-num-layers", type=int, default=None)
            # 重置 MTP 损失缩放因子默认值为 0.2
            reset_arg(parser, "--mtp-loss-scaling-factor", type=float, default=0.2)
            # 是否在训练中更新 MTP(Multi-Token Prediction)层参数
            parser.add_argument(
                "--enable-mtp-training",
                action="store_true",
                default=False,
                help="Enable MTP layer parameter updates during training",
            )

            # 返回追加了 MTP 参数的 parser
            return parser

        # ---------- Prefill/Decode 分离部署参数 ----------
        def add_prefill_decode_disaggregation_arguments(parser):
            # 分离架构中 prefill 服务器的数量
            parser.add_argument(
                "--prefill-num-servers",
                type=int,
                default=None,
                help="Number of prefill servers for disaggregation.",
            )
            # 返回追加了 PD 分离参数的 parser
            return parser

        # ---------- CI(持续集成)测试参数 ----------
        def add_ci_arguments(parser):
            # 标记本次运行为 CI 测试
            parser.add_argument(
                "--ci-test",
                action="store_true",
            )
            # CI 下禁用 KL 检查器
            parser.add_argument(
                "--ci-disable-kl-checker",
                action="store_true",
            )
            # 把梯度范数保存到指定路径(CI 对拍用)
            parser.add_argument(
                "--ci-save-grad-norm",
                type=str,
                default=None,
            )
            # 从指定路径加载梯度范数(CI 对拍用)
            parser.add_argument(
                "--ci-load-grad-norm",
                type=str,
                default=None,
            )
            # 返回追加了 CI 参数的 parser
            return parser

        # Add custom arguments in front to prevent overwritten some slime arguments.
        # 先注册用户自定义参数,避免覆盖 slime 同名参数
        if add_custom_arguments is not None:
            parser = add_custom_arguments(parser)

        # 依次注册各分组参数(顺序即帮助信息中的展示顺序)
        parser = add_cluster_arguments(parser)
        parser = add_train_arguments(parser)
        parser = add_rollout_arguments(parser)
        parser = add_fault_tolerance_arguments(parser)
        parser = add_data_arguments(parser)
        parser = add_eval_arguments(parser)
        parser = add_algo_arguments(parser)
        parser = add_on_policy_distillation_arguments(parser)
        parser = add_wandb_arguments(parser)
        parser = add_tensorboard_arguments(parser)
        parser = add_router_arguments(parser)
        parser = add_debug_arguments(parser)
        parser = add_network_arguments(parser)
        parser = add_reward_model_arguments(parser)
        parser = add_rollout_buffer_arguments(parser)
        parser = add_mtp_training_arguments(parser)
        parser = add_prefill_decode_disaggregation_arguments(parser)
        parser = add_ci_arguments(parser)
        parser = add_custom_megatron_plugins_arguments(parser)
        # 注册自定义函数 YAML 配置文件路径参数
        reset_arg(
            parser,
            "--custom-config-path",
            type=str,
            default=None,
            help="Path to the YAML config for custom function arguments.",
        )
        # 注册对齐到硬件效率的填充词表大小参数
        reset_arg(parser, "--padded-vocab-size", type=int, default=None)

        # 返回注册完全部 slime 参数的 parser
        return parser

    # 对外返回内部的 add_slime_arguments 闭包
    return add_slime_arguments


def _pre_parse_mode():
    """Pre-parse CLI to extract arguments that control parsing flow.

    These arguments are removed from add_slime_arguments to avoid
    registering them twice.  The returned namespace is merged into
    the final ``args`` after Phase 2 parsing.
    """
    # 构造一个临时解析器(不带 help、不允许缩写),只探测决定解析流程的少数参数
    temp_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    # 训练后端选择:megatron 或 fsdp
    temp_parser.add_argument("--train-backend", type=str, choices=["megatron", "fsdp"], default="megatron")
    # 仅跑 rollout 调试模式
    temp_parser.add_argument("--debug-rollout-only", action="store_true", default=False)
    # 仅跑训练调试模式
    temp_parser.add_argument("--debug-train-only", action="store_true", default=False)
    # 从文件加载调试 rollout 数据(跳过推理引擎)
    temp_parser.add_argument("--load-debug-rollout-data", type=str, default=None)
    # 用 parse_known_args 忽略其余未知参数,得到预解析结果
    temp_args, _ = temp_parser.parse_known_args()
    # 返回预解析的命名空间
    return temp_args


def parse_args(add_custom_arguments=None):
    # Users may call `parse_args` very early, thus we ensure logger is configured here
    # 用户可能在很早阶段调用本函数,这里先确保 logger 已配置
    configure_logger()

    # 构造 slime 参数注册回调
    add_slime_arguments = get_slime_extra_args_provider(add_custom_arguments)

    # 先做预解析,拿到影响解析流程的参数
    pre = _pre_parse_mode()
    # 纯训练调试或加载调试数据时,无需启动 sglang,跳过其参数解析
    skip_sglang = pre.debug_train_only or pre.load_debug_rollout_data is not None

    # Phase 1: Parse sglang args independently (separate parser, parse_known_args).
    # Skipped when sglang servers are not needed.
    # 阶段一:独立解析 sglang 参数(独立解析器 + parse_known_args);不需要时跳过
    sglang_ns = None
    if not skip_sglang:
        sglang_ns = sglang_parse_args()

    # Phase 2: Parse megatron/fsdp + slime args.
    # Uses ignore_unknown_args=True so that --sglang-* and pre-parsed CLI flags
    # are silently ignored by the megatron/fsdp parser.
    # 阶段二:解析 megatron/fsdp + slime 参数;忽略 --sglang-* 等预解析过的旗标
    if pre.train_backend == "megatron":
        # 按需导入 Megatron 后端的解析与校验函数
        from slime.backends.megatron_utils.arguments import megatron_parse_args
        from slime.backends.megatron_utils.arguments import validate_args as megatron_validate_args

        # 调用 Megatron 解析(内部会调用 extra_args_provider 注册 slime 参数)
        args = megatron_parse_args(
            extra_args_provider=add_slime_arguments,
            skip_hf_validate=pre.debug_rollout_only,
        )
    else:
        # FSDP 后端尚在重写,提示稳定性风险
        logger.warning(
            "🚧 🚧 🚧 FSDP backend is being rewritten, please use Megatron backend for better stability. 🚧 🚧 🚧"
        )

        # 按需导入 FSDP 后端的解析函数
        from slime.backends.fsdp_utils.arguments import fsdp_parse_args

        # 调用 FSDP 解析,忽略未知参数
        args = fsdp_parse_args(extra_args_provider=add_slime_arguments, ignore_unknown_args=True)

    # Merge pre-parsed args into the main namespace
    # 把预解析的控制参数合并进主命名空间
    for key, value in vars(pre).items():
        setattr(args, key, value)

    # Merge sglang args into the main namespace
    # 把 sglang 参数合并进主命名空间
    if sglang_ns is not None:
        for key, value in vars(sglang_ns).items():
            setattr(args, key, value)

    # 执行 slime 侧的交叉校验与参数推导
    slime_validate_args(args)

    if pre.train_backend == "megatron" and not args.debug_rollout_only:
        # Megatron 后端(非纯 rollout 调试)再做一次后端级校验
        megatron_validate_args(args)

    if not args.debug_train_only:
        # 非纯训练调试时校验 sglang 参数
        sglang_validate_args(args)

    # 返回最终参数命名空间
    return args


def _resolve_eval_datasets(args) -> list[EvalDatasetConfig]:
    """
    Build evaluation dataset configurations from either --eval-config or --eval-prompt-data.
    """
    # 原始数据集配置列表
    datasets_config = []
    # 全局默认值(来自 eval.defaults)
    defaults: dict[str, Any] = {}

    if args.eval_config:
        # 按需导入 OmegaConf 加载 YAML/JSON 配置
        from omegaconf import OmegaConf

        # 加载配置文件
        cfg = OmegaConf.load(args.eval_config)
        # 转成普通 dict 并解析变量插值
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg_dict, dict):
            # 配置根节点必须是映射
            raise ValueError("--eval-config must contain a mapping at the root.")

        # 允许整个文件就是映射,或包含 eval 子映射
        eval_cfg = cfg_dict.get("eval", cfg_dict)
        if not isinstance(eval_cfg, dict):
            raise ValueError("--eval-config must define an `eval` mapping or be a mapping itself.")

        # 取出全局默认项
        defaults = dict(eval_cfg.get("defaults") or {})
        # 规整数据集列表(支持列表或以名称为键的字典)
        datasets_config = ensure_dataset_list(eval_cfg.get("datasets"))
        if not datasets_config:
            # 配置文件里没有任何数据集视为错误
            raise ValueError("--eval-config does not define any datasets under `eval.datasets`.")
    elif args.eval_prompt_data:
        # 兼容旧版命令行:按 "名称 路径" 成对解析
        values = list(args.eval_prompt_data)
        if len(values) == 1:
            # 只给一个值时按旧约定视作 aime 数据集
            logger.info("[legacy] only one eval_prompt_data detected, will assume it is data for aime")
            values = ["aime", values[0]]
        if len(values) % 2 != 0:
            # 数量为奇数说明 name/path 没有成对
            raise ValueError("eval prompt data must be provided as name/path pairs.")
        # 两两组合成 {name, path} 配置字典
        datasets_config = [{"name": values[i], "path": values[i + 1]} for i in range(0, len(values), 2)]
    else:
        # 两者都未配置:无评测数据集
        datasets_config = []

    # 按三级优先级(dataset > defaults > args)构建最终配置对象
    eval_datasets = build_eval_dataset_configs(args, datasets_config, defaults)
    if eval_datasets:
        # 回写 --eval-prompt-data 为规整后的 "名称 路径" 对,供下游统一使用
        args.eval_prompt_data = [item for dataset in eval_datasets for item in (dataset.name, dataset.path)]
    else:
        # 没有评测数据集时置 None
        args.eval_prompt_data = None

    # 返回评测数据集配置列表
    return eval_datasets


def slime_validate_args(args):
    # 解析并挂载评测数据集配置
    args.eval_datasets = _resolve_eval_datasets(args)

    if args.kl_coef != 0 or args.use_kl_loss:
        # 使用 KL 惩罚时必须提供参考模型权重
        if not os.path.exists(args.ref_load):
            raise FileNotFoundError(f"ref_load {args.ref_load} does not exist, please check the path.")

        if not os.path.exists(os.path.join(args.ref_load, "latest_checkpointed_iteration.txt")):
            # 缺少迭代号标记文件时提醒:可能不是合法的 Megatron checkpoint 目录
            logger.info(
                f"ref_load {args.ref_load} does not have latest_checkpointed_iteration.txt, "
                "please make sure it is a valid megatron checkpoint directory."
            )

    # Validate on-policy distillation (OPD) arguments
    # 校验在线策略蒸馏(OPD)参数的合法性
    if args.use_opd:
        if args.opd_type is None:
            # 启用 OPD 时必须指定实现方式
            raise ValueError("--opd-type must be specified when --use-opd is enabled. Choose 'sglang' or 'megatron'.")

        if args.opd_type == "megatron":
            if args.opd_teacher_load is None:
                # megatron 模式必须有教师模型权重路径
                raise ValueError(
                    "--opd-teacher-load is required when --opd-type=megatron. "
                    "Please provide the path to the teacher model checkpoint."
                )
            if not os.path.exists(args.opd_teacher_load):
                # 教师权重路径不存在则报错
                raise FileNotFoundError(
                    f"opd_teacher_load {args.opd_teacher_load} does not exist, please check the path."
                )
            if not os.path.exists(os.path.join(args.opd_teacher_load, "latest_checkpointed_iteration.txt")):
                # 提醒教师权重目录可能不合法
                logger.info(
                    f"opd_teacher_load {args.opd_teacher_load} does not have latest_checkpointed_iteration.txt, "
                    "please make sure it is a valid megatron checkpoint directory."
                )

        elif args.opd_type == "sglang":
            if args.opd_teacher_load is not None:
                # sglang 模式下教师 logprobs 来自外部服务,不应再给教师权重
                raise ValueError(
                    "--opd-teacher-load should not be set when --opd-type=sglang. "
                    "In sglang mode, teacher log-probs are obtained from external server during rollout."
                )
    else:
        # If OPD is not enabled, opd_teacher_load should not be set
        # 未启用 OPD 却给了教师权重属于配置冲突
        if args.opd_teacher_load is not None:
            raise ValueError("--opd-teacher-load is set but --use-opd is not enabled. Please add --use-opd flag.")

    if args.megatron_to_hf_mode == "bridge":
        if (
            args.load is not None
            and os.path.exists(args.load)
            and os.path.exists(os.path.join(args.load, "latest_checkpointed_iteration.txt"))
        ):
            # If is a Megatron checkpoint, won't use bridge to load hf weight.
            # --load 已是合法 Megatron checkpoint:直接加载,无需 bridge 转 HF 权重
            pass
        else:
            if args.load is None:
                # 未指定 load 时回退到参考权重或 HF checkpoint
                args.load = args.ref_load or args.hf_checkpoint
            # If is a HF checkpoint, set start_rollout_id to 0 here.
            # HF checkpoint 没有 rollout 步号信息,起始步归零
            args.start_rollout_id = 0
    else:
        if (
            args.load is None
            or not os.path.exists(args.load)
            or not os.path.exists(os.path.join(args.load, "latest_checkpointed_iteration.txt"))
        ):
            # load 不可用时退化为从参考模型权重开始微调
            args.no_load_optim = True
            # 不加载随机数状态
            args.no_load_rng = True
            # 标记为微调模式(不恢复训练状态)
            args.finetune = True
            # 用参考模型权重作为初始权重
            args.load = args.ref_load
            if args.ref_ckpt_step is not None:
                # 沿用参考模型的 checkpoint 步号
                args.ckpt_step = args.ref_ckpt_step
            # 从零开始计 rollout 步
            args.start_rollout_id = 0

    if args.eval_interval is not None:
        # 配置了周期评测就必须有评测数据集
        assert args.eval_datasets, "Evaluation datasets must be configured when eval_interval is set."

    if args.save_interval is not None:
        # 配置了周期保存就必须指定保存路径
        assert args.save is not None, "'--save' is required when save_interval is set."

    # kl_coef(奖励塑形)与 kl_loss_coef(损失项)不允许同时生效
    assert not (args.kl_coef != 0 and args.kl_loss_coef != 0), "Only one of kl_coef and kl_loss_coef can be set"

    if args.advantage_estimator in ["reinforce_plus_plus", "reinforce_plus_plus_baseline"]:
        # REINFORCE++ 系列估计器依赖优势归一化
        assert args.normalize_advantages, (
            "The 'reinforce_plus_plus' and 'reinforce_plus_plus_baseline' advantage estimators "
            "require advantage normalization. Please add `--normalize-advantages` to your command."
        )

    if args.use_rollout_logprobs:
        # 直接用引擎 logprobs 与 TIS 互斥(两者都是对 off-policy 的处理)
        assert not args.use_tis, "use_rollout_logprobs and use_tis cannot be set at the same time."

    if args.get_mismatch_metrics:
        # 计算 mismatch 指标必须提供自定义 TIS 函数
        assert (
            args.custom_tis_function_path is not None
        ), "custom_tis_function_path must be set when get_mismatch_metrics is set"

        if args.use_rollout_logprobs:
            # 提示:指标计算仍需训练引擎重算 logprobs,会多一次前向
            logger.info(
                "get_mismatch_metrics is set; For metrics calculation, the log probs will still be recomputed by training engine. One more forward pass will be applied."
            )

    if args.use_dynamic_batch_size:
        # 动态批大小必须给出每 GPU token 上限
        assert args.max_tokens_per_gpu is not None, "max_tokens_per_gpu must be set when use_dynamic_batch_size is set"
        if args.log_probs_max_tokens_per_gpu is None:
            # 未单独设置时,log probs 的 token 上限沿用 max_tokens_per_gpu
            args.log_probs_max_tokens_per_gpu = args.max_tokens_per_gpu

    if args.eps_clip_high is None:
        # 未设置非对称裁剪上界时与下界一致
        args.eps_clip_high = args.eps_clip

    if args.eval_reward_key is None:
        # 评测奖励 key 缺省取训练奖励 key
        args.eval_reward_key = args.reward_key

    if args.dump_details is not None:
        # 一键导出明细时,自动派生 rollout/训练数据的调试保存路径
        args.save_debug_rollout_data = f"{args.dump_details}/rollout_data/{{rollout_id}}.pt"
        args.save_debug_train_data = f"{args.dump_details}/train_data/{{rollout_id}}_{{rank}}.pt"

    if args.load_debug_rollout_data is not None:
        # 加载调试数据模式:不启动 sglang,仅跑训练
        logger.info(
            f"load_debug_rollout_data {args.load_debug_rollout_data} is set, "
            "will not instantiate sglang servers and will only run the training process."
        )
        args.debug_train_only = True

    # 仅 PPO 估计器需要 critic 模型
    args.use_critic = args.advantage_estimator == "ppo"
    if args.critic_num_gpus_per_node is None:
        # critic 每节点 GPU 数缺省同 actor
        args.critic_num_gpus_per_node = args.actor_num_gpus_per_node
    if args.critic_num_nodes is None:
        # critic 节点数缺省同 actor
        args.critic_num_nodes = args.actor_num_nodes
    if args.critic_load is None:
        # critic 权重缺省取 actor 的加载路径
        args.critic_load = args.load
    if args.critic_lr is None:
        # critic 学习率缺省同 actor
        args.critic_lr = args.lr

    if args.offload:
        # 一键 offload 展开为训练侧与推理侧同时卸载
        args.offload_train = True
        args.offload_rollout = True
    # 展开完成后删除原开关,避免下游误读
    del args.offload

    if args.debug_rollout_only:
        if args.colocate and (not args.rollout_num_gpus):
            # 纯 rollout 调试 + colocate:直接用 actor 的 GPU 总数
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
        else:
            # 由 rollout GPU 数反推 actor 的节点数与每节点卡数(每节点至多 8 卡)
            args.actor_num_gpus_per_node = min(8, args.rollout_num_gpus)
            args.actor_num_nodes = args.rollout_num_gpus // args.actor_num_gpus_per_node
        # 纯 rollout 调试时不 colocate、不卸载
        args.colocate = False
        args.offload_train = args.offload_rollout = False
        if args.train_memory_margin_bytes > 0:
            # 纯 rollout 调试不支持显存余量,强制清零并告警
            logger.warning("Force train_memory_margin_bytes=0 since debug_rollout_only does not support it")
            args.train_memory_margin_bytes = 0

    # 两种调试模式互斥
    assert not (args.debug_rollout_only and args.debug_train_only), (
        "debug_rollout_only and debug_train_only cannot be set at the same time, " "please set only one of them."
    )

    # always true on offload for colocate at the moment.
    # colocate 模式:训练与推理共用 GPU,必须交替卸载
    if args.colocate:
        if args.offload_train is None:
            args.offload_train = True
        if args.offload_rollout is None:
            args.offload_rollout = True
        if args.rollout_num_gpus != args.actor_num_gpus_per_node * args.actor_num_nodes:
            # rollout GPU 数与 actor 不一致时强制对齐并记录日志
            logger.info(
                f"rollout_num_gpus {args.rollout_num_gpus} != actor_num_gpus_per_node {args.actor_num_gpus_per_node} "
                f"* actor_num_nodes {args.actor_num_nodes}, overriding rollout_num_gpus to match actor_num_gpus_per_node * actor_num_nodes."
            )
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
            if args.use_critic:
                # PPO 有 critic 时,推理 GPU 总数还要加上 critic 占用
                args.rollout_num_gpus += args.critic_num_gpus_per_node * args.critic_num_nodes

    if args.offload_train is None:
        # 训练侧卸载缺省关闭
        args.offload_train = False
    if args.offload_rollout is None:
        # 推理侧卸载缺省关闭
        args.offload_rollout = False

    if args.eval_function_path is None:
        # 评测生成函数缺省复用 rollout 函数
        args.eval_function_path = args.rollout_function_path

    if args.num_steps_per_rollout is not None:
        # 由"每 rollout 步训练步数"推导全局批大小 gbs = rollout批 × 组大小 ÷ 步数
        global_batch_size = args.rollout_batch_size * args.n_samples_per_prompt // args.num_steps_per_rollout
        if args.global_batch_size is not None:
            # 显式给出的 gbs 必须与推导值一致
            assert args.global_batch_size == global_batch_size, (
                f"global_batch_size {args.global_batch_size} is not equal to "
                f"rollout_batch_size {args.rollout_batch_size} * n_samples_per_prompt {args.n_samples_per_prompt} "
                f"// num_steps_per_rollout {args.num_steps_per_rollout}"
            )
        # 覆写全局批大小为推导值
        args.global_batch_size = global_batch_size

    if args.n_samples_per_prompt == 1:
        # 组大小为 1 时组内标准差无意义,关闭 GRPO 标准差归一化
        args.grpo_std_normalization = False
        logger.info("n_samples_per_prompt is set to 1, grpo_std_normalization will be set to False.")

    if args.over_sampling_batch_size is None:
        # 过采样批大小缺省取 rollout 批大小(即不过采)
        args.over_sampling_batch_size = args.rollout_batch_size

    # 过采样批大小不得小于 rollout 批大小
    assert args.over_sampling_batch_size >= args.rollout_batch_size, (
        f"over_sampling_batch_size {args.over_sampling_batch_size} should be greater than or equal to "
        f"rollout_batch_size {args.rollout_batch_size}"
    )

    if args.num_epoch is not None:
        if args.num_rollout is not None:
            # 两者同设时以 num_rollout 为准,epoch 被忽略
            logger.info("Both num_epoch and num_rollout are set, num_epoch will be ignored.")
        else:
            # 用 epoch 推步数需要全局数据集支持
            assert args.rollout_global_dataset, (
                "num_epoch is set, but rollout_global_dataset is not set, "
                "please remove --disable-rollout-global-dataset to use num_epoch"
            )
    else:
        # if num_epoch is not set, we should set num_rollout
        # 未给 epoch 时必须显式给出 rollout 步数
        assert args.num_rollout is not None, (
            "num_epoch is not set, but num_rollout is not set, " "please set --num-rollout or --num-epoch"
        )

    if args.enable_mtp_training:
        # 开启 MTP 训练必须指定 MTP 层数
        assert args.mtp_num_layers, "mtp_num_layers must be set when enable_mtp_training is set"

    if args.use_rollout_routing_replay:
        # rollout 路由回放隐含启用基础路由回放
        args.use_routing_replay = True

    if args.custom_config_path:
        # 读取自定义 YAML 配置并把键值覆盖到 args 上
        with open(args.custom_config_path) as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if hasattr(args, k):
                # 已有同名参数时先告警再覆盖
                logger.info(f"Warning: Argument {k} is already set to {getattr(args, k)}, will override with {v}.")
            # 逐键写入 args
            setattr(args, k, v)

    if args.eval_max_context_len is None:
        # 评测最大上下文缺省取 rollout 的设置
        logger.info(
            f"args.eval_max_context_len is not set. Use args.rollout_max_context_len {args.rollout_max_context_len} as default value."
        )
        args.eval_max_context_len = args.rollout_max_context_len

    if args.rollout_max_context_len is not None:
        if args.rollout_max_prompt_len is None:
            # prompt 上限缺省为上下文减一,保证至少留 1 个生成 token 可算损失
            args.rollout_max_prompt_len = args.rollout_max_context_len - 1
            logger.info(
                f"args.rollout_max_prompt_len is not set. Use args.rollout_max_context_len - 1 ({args.rollout_max_context_len} - 1) as default value so that there is at least one generated token to compute loss."
            )
        # prompt 上限必须严格小于上下文长度
        assert (
            args.rollout_max_prompt_len <= args.rollout_max_context_len - 1
        ), f"args.rollout_max_prompt_len ({args.rollout_max_prompt_len}) must be smaller than args.rollout_max_context_len ({args.rollout_max_context_len}) so that there is at least one generated token to compute loss."

    # PD 分离(prefill 服务器)与外部引擎模式互斥
    assert not (
        args.prefill_num_servers is not None and args.rollout_external
    ), "prefill_num_servers cannot be set when rollout_external is set."

    if args.qkv_format == "bshd":
        # bshd 布局仅 Megatron 后端支持
        assert args.train_backend == "megatron", "bshd format is only supported for megatron backend."
        # bshd 布局不支持动态批大小,必须改用固定 micro batch size
        assert (
            args.use_dynamic_batch_size is False
        ), "Dynamic batch size is not supported for bshd format. Please specify --micro-batch-size instead."

    if args.only_train_params_name_list and args.freeze_params_name_list:
        # 参数白名单与黑名单互斥,只能二选一
        raise ValueError("You can only specify ONE of: --only-train-params-name-list, or --freeze-params-name-list.")
