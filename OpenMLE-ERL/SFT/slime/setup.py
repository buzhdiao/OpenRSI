# 导入 sys 用于获取当前 Python 解释器的版本信息
import sys
# 导入 platform 用于识别当前操作系统平台（如 Linux/Windows/Darwin）
import platform

# 从 setuptools 导入 find_packages（自动发现包目录）与 setup（打包入口函数）
from setuptools import find_packages, setup
# 从 wheel 库导入 bdist_wheel 命令基类，用于自定义 wheel 构建行为
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


# 读取依赖清单文件的辅助函数：path 为 requirements 文件路径
def _fetch_requirements(path):
    # 打开文件（with 保证自动关闭），逐行读取
    with open(path) as fd:
        # 返回处理后的依赖列表：去掉首尾空白、跳过空行和以 # 开头的注释行
        return [r.strip() for r in fd.readlines() if r.strip() and not r.startswith("#")]


# Custom wheel class to modify the wheel name
# 自定义 wheel 构建命令类，用于修改 wheel 文件的命名标签（继承原生 bdist_wheel）
class bdist_wheel(_bdist_wheel):
    # finalize_options 在构建选项确定后被调用，用于覆盖默认选项
    def finalize_options(self):
        # 先调用父类的实现，完成默认的选项处理
        _bdist_wheel.finalize_options(self)
        # 强制标记为非纯 Python 包（root_is_pure=False），使 wheel 带平台标签而非 py2.py3 通用标签
        self.root_is_pure = False

    # get_tag 返回 wheel 文件名中的三段标签：(python 标签, abi 标签, 平台标签)
    def get_tag(self):
        # 根据 Python 主/次版本号生成 cp 标签，例如 Python 3.10 -> "cp310"
        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        # ABI 标签直接复用 Python 版本标签（自定义行为，保持文件名格式一致）
        abi_tag = f"{python_version}"

        # 判断当前操作系统是否为 Linux
        if platform.system() == "Linux":
            # Linux 平台使用 manylinux1_x86_64 兼容标签，便于在多数 Linux 发行版上安装
            platform_tag = "manylinux1_x86_64"
        else:
            # 其他平台使用系统名的小写形式作为平台标签
            platform_tag = platform.system().lower()

        # 以元组形式返回三段标签，wheel 构建工具会用它们拼接文件名
        return python_version, abi_tag, platform_tag


# Setup configuration
# 调用 setup() 完成包的打包配置
setup(
    # 作者信息
    author="slime Team",
    # 包名
    name="slime",
    # 当前版本号
    version="0.2.2",
    # 自动发现要打包的子包：包含 slime* 与 slime_plugins* 前缀的所有包
    packages=find_packages(include=["slime*", "slime_plugins*"]),
    # 将包内非 Python 数据文件一并打包
    include_package_data=True,
    # 从 requirements.txt 读取运行时依赖
    install_requires=_fetch_requirements("requirements.txt"),
    # 可选依赖组：安装 [fsdp] extra 时额外安装 torch>=2.0（FSDP 后端所需）
    extras_require={
        "fsdp": [
            "torch>=2.0",
        ]
    },
    # 要求 Python 3.10 及以上版本
    python_requires=">=3.10",
    # PyPI 分类器：声明支持的 Python 版本与应用领域
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
    # 注册自定义的 bdist_wheel 命令类，使打包走上面的自定义命名逻辑
    cmdclass={"bdist_wheel": bdist_wheel},
)
