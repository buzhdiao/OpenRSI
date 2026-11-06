# 导入 argparse，用于解析命令行参数
import argparse
# 导入 json，用于写出 safetensors 索引文件
import json
# 导入 os，用于路径与目录操作
import os
# 导入 pickle，用于自定义反序列化包装
import pickle
# 导入 re，用于从参数名中匹配层号/专家号
import re
# 导入 shutil，用于拷贝 tokenizer 等附属文件
import shutil
# 导入 time，用于统计耗时
import time


# 导入 safetensors 的 torch 接口，用于保存权重文件
import safetensors.torch
# 导入 PyTorch 主库
import torch
# 导入 torch 分布式 checkpoint（torch_dist）读写模块
import torch.distributed.checkpoint as dist_cp
# 导入 HF AutoConfig，用于从原始 HF 目录推断模型名
from transformers import AutoConfig
# 导入 override 装饰器，标记对父类方法的重写
from typing_extensions import override

# 导入 slime 的 Megatron->HF 权重名映射与去 padding 工具
from slime.backends.megatron_utils.megatron_to_hf import convert_to_hf, remove_padding


# 自定义反序列化器：把 megatron/glm 模块中的类替换为空壳类
class UnpicklerWrapper(pickle.Unpickler):
    # 重写 find_class，控制反序列化时类的解析方式
    @override
    def find_class(self, mod_name, name):
        # 定义空壳类：接受任意参数但不做任何事
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        # megatron/glm 相关类在当前环境无法导入，用空壳类代替
        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return DummyClass
        # 其余类按 pickle 默认逻辑解析
        return super().find_class(mod_name, name)


# 用自定义 Unpickler 替换全局 pickle.Unpickler，使 torch.load 等也走该逻辑
pickle.Unpickler = UnpicklerWrapper


# 包装的文件系统读取器：用上面的 UnpicklerWrapper 读取元数据
class WrappedStorageReader(dist_cp.FileSystemReader):
    # 重写 read_metadata，绕开对 megatron 类的依赖
    @override
    def read_metadata(self):
        # 拼接 .metadata 元数据文件路径
        path = self.fs.concat_path(self.path, ".metadata")
        # 以二进制流打开并用自定义反序列化器加载元数据
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        # 元数据缺少 storage_meta 时补上默认值
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = dist_cp.StorageMeta()
        # 记录本次加载的 load_id
        metadata.storage_meta.load_id = self.load_id
        # planner_data 为空时初始化为空字典
        if metadata.planner_data is None:
            metadata.planner_data = {}
        # 返回处理后的元数据
        return metadata


# “空状态字典”加载规划器：按元数据主动构建待加载的张量容器
class EmptyStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    # 重写 set_up_planner：从元数据枚举所有条目并预分配张量
    @override
    def set_up_planner(
        self,
        state_dict: dist_cp.metadata.STATE_DICT_TYPE,
        metadata: dist_cp.metadata.Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        # 遍历 checkpoint 元数据中记录的所有状态条目
        for k, v in metadata.state_dict_metadata.items():
            # 跳过优化器状态与内部 _state 条目
            if "optimizer" in k or "_state" in k:
                continue
            # 打印发现的键，便于确认内容
            print(f"find {k} in torch_dist ckpt")
            # 张量类条目：按元数据中的形状与 dtype 预分配空张量作容器
            if isinstance(v, dist_cp.metadata.TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)  # type: ignore[assignment]
            # 放入待加载的 state_dict
            state_dict[k] = v
        # 调用父类完成规划器初始化
        super().set_up_planner(state_dict, metadata, is_coordinator)


# 展开“专家维”参数：把合并存储的专家权重拆成逐专家参数
def get_expert_param(args, name, param):
    # 名称不含专家标记时原样产出
    if ".experts." not in name:
        yield name, param
        return

    # Qwen3.5 HF weights keep all experts in one grouped tensor, so the older
    # rule that emits weight0, weight1, and so on cannot be used.
    # Qwen3.5 的 HF 权重把所有专家放在一个分组张量里，不能按 weight0/weight1 拆分
    if "qwen3_5" in getattr(args, "hf_model_name", ""):
        yield name, param
        return

    # 读取专家数量
    num_experts = args.num_experts
    # 尝试匹配名称中已带专家编号的形式
    match = re.search(r"mlp.experts\.(.+)\.weight(\d+)", name)
    # 未匹配：说明专家维折叠在参数第 0 维，需按专家逐个拆开
    if not match:
        # 校验第 0 维大小等于专家数
        assert param.shape[0] == num_experts
        # 逐专家产出名称与切片
        for expert_id in range(num_experts):
            # 生成形如 .experts.<expert_id> 的专家参数名
            expert_name = name.replace(".experts.experts.", ".experts.") + str(expert_id)
            # 按第 0 维取该专家的参数
            expert_param = param[expert_id]
            # 产出一对（名称, 参数）
            yield expert_name, expert_param
    else:
        # 已带编号的形式原样产出
        yield name, param


# 展开“层维”参数：把合并存储的层参数拆成逐层参数
def get_layer_param(args, name, param):
    # 名称不含层标记时原样产出
    if ".layers." not in name:
        yield name, param
        return

    # 读取总层数
    num_layers = args.num_layers
    # 尝试匹配名称中已带层号的形式
    match = re.search(r"\.layers\.(\d+)\.", name)
    # 未匹配：层维折叠在参数第 0 维，需逐层拆开
    if not match:
        # 校验第 0 维大小等于层数
        assert param.shape[0] == num_layers
        # 逐层产出
        for layer_id in range(num_layers):
            # 生成形如 .layers.<layer_id>. 的层参数名
            layer_name = name.replace(".layers.", f".layers.{layer_id}.")
            # 取该层的参数切片
            layer_param = param[layer_id]
            # 继续交给专家维展开处理
            yield from get_expert_param(args, layer_name, layer_param)
    else:
        # 已带层号：直接走专家维展开
        yield from get_expert_param(args, name, param)


# 给所有参数名补上 Megatron 的 module.module 前缀后逐个展开
def get_named_params(args, state_dict):
    # 遍历 checkpoint 中的全部参数
    for name, param in state_dict.items():
        # 补前缀以匹配 convert_to_hf 的名称约定
        name = f"module.module.{name}"
        # 交给层维展开处理
        yield from get_layer_param(args, name, param)


# 将转换后的权重按分片大小保存为 safetensors 文件
def save_tensors(args, model_name, state_dict, output_dir, chunk_size, vocab_size=None):
    # 打印保存开始信息
    print(f"start saving to {output_dir}")
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    # 2GB
    # 当前分片累计大小（字节）
    current_size = 0
    # 全部张量的总大小（字节）
    total_size = 0
    # 分片列表，首个分片为空字典
    modeltensors = [{}]
    # 遍历展开后的全部参数
    for name, param in get_named_params(args, state_dict):
        # 提供词表大小时先去掉词表 padding
        if vocab_size:
            param = remove_padding(name, param, vocab_size)
        # 把 Megatron 参数名/布局转换为 HF 命名与布局
        converted_named_tensors = convert_to_hf(args, model_name, name, param)
        # 遍历每个转换结果
        for converted_name, converted_param in converted_named_tensors:
            # 计算该张量占用的字节数
            tensor_size = converted_param.numel() * converted_param.element_size()
            # 超过分片上限时开启新分片
            if tensor_size + current_size > chunk_size:
                modeltensors.append({})
                current_size = 0
            # 张量放入当前分片
            modeltensors[-1][converted_name] = converted_param
            # 累加当前分片大小
            current_size += tensor_size
            # 累加总大小
            total_size += tensor_size

    # 组装 safetensors 索引元数据（总大小 + 权重映射表）
    metadata = {"metadata": {"total_size": total_size}, "weight_map": {}}

    # 分片数量
    num_files = len(modeltensors)
    # 先写索引：记录每个权重所在的文件
    for i, tensors in enumerate(modeltensors):
        # 生成分片文件名
        filename = f"model-{i:05d}-of-{num_files:05d}.safetensors"
        # 把该分片内的每个键映射到对应文件名
        for key in tensors.keys():
            metadata["weight_map"][key] = filename
    # 索引文件路径
    index_filepath = os.path.join(output_dir, "model.safetensors.index.json")
    # 写出索引 JSON
    json.dump(metadata, open(index_filepath, "w"), indent=2)
    # 打印索引保存完成信息
    print(f"{index_filepath} saved.")

    # 再逐个分片写出 safetensors 文件
    for i, tensors in enumerate(modeltensors):
        # 生成分片文件名
        filename = f"model-{i:05d}-of-{num_files:05d}.safetensors"
        # 记录开始时间
        t = time.time()
        # 拼接文件完整路径
        filepath = os.path.join(output_dir, filename)
        # 保存该分片的全部张量
        safetensors.torch.save_file(tensors, filepath)
        # 打印该分片保存耗时
        print(f"{filename} saved in {time.time() - t:.2f} sec.")


# 从原始 HF 目录拷贝 tokenizer/config 等附属文件到输出目录
def copy_assets(origin_hf_dir, output_dir):
    # 遍历原 HF 目录中的全部文件
    for filename in os.listdir(origin_hf_dir):
        # 跳过权重索引与权重文件（这些由 save_tensors 重新生成）
        if filename == "model.safetensors.index.json" or filename.endswith(".safetensors"):
            continue
        # 源文件完整路径
        origin_filename = os.path.join(origin_hf_dir, filename)
        # 非普通文件（如子目录）跳过
        if not os.path.isfile(origin_filename):
            print(f"Skip {filename}, not a file.")
            continue
        # 确定源与目标路径
        src, dst = origin_filename, os.path.join(output_dir, filename)
        # 打印拷贝信息
        print(f"copy from {src} to {dst}")
        # 执行拷贝
        shutil.copy(src, dst)


# 脚本入口
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 模型名（可选，用于确定参数映射规则）
    parser.add_argument("--model-name", type=str, default=None)
    # torch 分布式 checkpoint 输入目录（必填）
    parser.add_argument("--input-dir", type=str, required=True)
    # HF 格式输出目录（必填）
    parser.add_argument("--output-dir", type=str, required=True)
    # 原始 HF 目录：用于拷贝 tokenizer/config 等文件
    parser.add_argument(
        "--origin-hf-dir",
        type=str,
        default=None,
        help="use the origin hf dir to copy files like tokenizer, config.json, etc.",
    )
    # 是否强制覆盖已存在的输出目录
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite the output directory if it exists."
    )
    # 分片保存大小（默认 5GB）
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5 * 1024**3,
        help="Chunk size for saving tensors, default is 2GB.",
    )
    # 词表大小：用于去除词表 padding（可选）
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Vocab size for removing padding, if applicable. If not provided, no padding will be removed.",
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 输出目录已存在且未指定 --force 时报错
    if os.path.exists(args.output_dir) and not args.force:
        raise ValueError(f"Output directory {args.output_dir} already exists. Use --force to overwrite it.")

    # 模型名与原始 HF 目录都未提供时报错（无法确定参数名规则）
    if args.model_name is None and args.origin_hf_dir is None:
        raise ValueError(
            "Either --model-name or --origin-hf-dir must be provided, so that we can know the name of the params."
        )

    # 未显式给模型名时，从原始 HF 配置的类名推断
    if args.model_name is None:
        # 加载原始 HF 配置
        hf_config = AutoConfig.from_pretrained(args.origin_hf_dir, trust_remote_code=True)
        # 用配置类名的小写作为模型名
        args.model_name = type(hf_config).__name__.lower()

    # 待加载的权重字典
    state_dict = {}
    # 打印加载开始信息
    print(f"loading model from {args.input_dir}")
    # 记录开始时间
    t = time.time()
    # 从 common.pt 读取保存时的 Megatron 参数（含层数、专家数等）
    megatron_args = torch.load(os.path.join(args.input_dir, "common.pt"), weights_only=False)["args"]
    # Select the expert layout from the HF architecture named on the CLI.
    # 用命令行指定的 HF 架构名覆盖模型名，决定专家权重的展开方式
    megatron_args.hf_model_name = args.model_name
    # 用自定义读取器与规划器加载 torch 分布式 checkpoint（单机模式）
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=WrappedStorageReader(args.input_dir),
        planner=EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    # 打印加载耗时
    print(f"model loaded in {time.time()-t:.2f} sec.")

    # 执行转换并保存
    save_tensors(megatron_args, args.model_name, state_dict, args.output_dir, args.chunk_size, args.vocab_size)

    # 提供原始 HF 目录时拷贝附属文件
    if args.origin_hf_dir:
        copy_assets(args.origin_hf_dir, args.output_dir)
