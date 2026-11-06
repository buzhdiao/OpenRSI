import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Get CUDA arch list
# 探测本机所有 GPU 的计算能力（compute capability），生成编译目标架构列表
arch_list = []
if torch.cuda.is_available():
    # 逐个 GPU 读取其主/次版本号，拼接成 "主.次" 形式的架构字符串
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        arch_list.append(f"{major}.{minor}")
    # 去重并排序
    arch_list = sorted(set(arch_list))

# Fallback to TORCH_CUDA_ARCH_LIST env var or default architectures when GPU is not available
# 编译机上无 GPU 时，回退到环境变量 TORCH_CUDA_ARCH_LIST 或默认架构列表
if not arch_list:
    env_arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    if env_arch:
        # Parse TORCH_CUDA_ARCH_LIST format: "7.0 7.5 8.0 8.6 9.0+PTX"
        # 解析环境变量：按空格/分号切分，去掉 +PTX 后缀（只编译 sm 目标）
        arch_list = [a.strip().replace("+PTX", "") for a in env_arch.replace(";", " ").split() if a.strip()]
    else:
        # Default to common architectures (Volta, Turing, Ampere, Ada, Hopper)
        # 默认覆盖常见数据中心 GPU 架构（Ampere/Ada/Hopper）
        arch_list = ["8.0", "8.6", "8.9", "9.0"]

# 声明 CUDA 扩展构建配置：包名、源文件与编译参数
setup(
    name="fake_int4_quant_cuda",
    ext_modules=[
        CUDAExtension(
            name="fake_int4_quant_cuda",
            # 唯一的 CUDA 源文件（fake INT4 伪量化算子，用于 QAT 训练）
            sources=["fake_int4_quant_cuda.cu"],
            extra_compile_args={
                # C++ 编译选项：O3 优化与 C++17 标准
                "cxx": [
                    "-O3",
                    "-std=c++17",
                ],
                # NVCC 编译选项：O3、C++17、放宽 constexpr 限制、生成位置无关代码；
                # 并为每个探测到的架构生成对应的 gencode 编译目标，
                # 额外编译 sm_90a（Hopper 的架构专属变体，支持特定新指令）
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "-Xcompiler",
                    "-fPIC",
                ]
                + [
                    f'-gencode=arch=compute_{arch.replace(".", "")},code=sm_{arch.replace(".", "")}'
                    for arch in arch_list
                ]
                + ["-gencode=arch=compute_90a,code=sm_90a"],
            },
        )
    ],
    # 使用 torch 的 BuildExtension 处理构建流程
    cmdclass={"build_ext": BuildExtension},
)
