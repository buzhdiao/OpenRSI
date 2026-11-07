# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 日期时间：获取当前时间
import datetime
# 子进程：执行 git 命令获取 commit id
import subprocess
# 深拷贝：复制配置对象
import copy

# OmegaConf 配置库主模块
import omegaconf
# 字典式配置对象与配置主入口
from omegaconf import DictConfig, OmegaConf
# 读取 PyTorch 版本号
import torch
# 导入 transformers（保证依赖环境加载）
import transformers

# 总配置类，用于确定哈希排除字段集合
from dojo.config_dataclasses.run import RunConfig
# Runner 配置
from dojo.config_dataclasses.runner import RunnerConfig
# 环境相关目录的解析函数
from dojo.utils.environment import get_mlebench_data_dir, get_superimage_dir


def generate_id(*, _parent_: DictConfig) -> str:
    """
    Generate a unique ID for an experiment given an experiment config.
    """
    # Some OmegaConf magic: _parent_ gives access to the parent
    # of the field being interpolated. In our case, the interpolation
    # is in the Config.id field so this gives us the original Config.
    # 深拷贝一份未解析的配置（_parent_ 即被插值字段所在的原始配置对象），避免污染原始配置
    unresolved_config = copy.deepcopy(_parent_)
    # 取出所有"不参与哈希"的字段列表
    fields_to_exclude = RunConfig.fields_to_exclude_from_hash()
    # Accessing fields of unresolved_config will resolve them
    # 访问字段会触发插值解析：取出用户名、issue 编号与随机种子，作为 ID 前缀
    user, issue, seed = (
        unresolved_config.metadata.user,
        unresolved_config.metadata.git_issue_id,
        unresolved_config.metadata.seed,
    )
    # 遍历排除字段，把它们的值重置为类型默认值，保证 ID 只由"真正区分实验"的字段决定
    for field, field_type in fields_to_exclude:
        # 数值类型字段重置为 0
        if field_type in (int, float):
            # 动态赋值（字段路径可能是点分嵌套形式，故用 exec）
            exec(f"unresolved_config.{field} = 0")
        # 布尔字段重置为 False
        elif field_type == bool:
            exec(f"unresolved_config.{field} = False")
        # 其余类型（str 等）一律重置为空字符串
        else:
            exec(f'unresolved_config.{field} = ""')
    # 把重置后的配置固化为 RunConfig 对象
    config: RunConfig = OmegaConf.to_object(unresolved_config)  # type: ignore[assignment]
    # 断言转换结果类型正确
    assert isinstance(config, RunConfig)

    # 拼出唯一 ID：用户_issue_种子_配置哈希
    return f"user_{user}_issue_{issue}_seed_{seed}_id_{config.hash()}"


def get_current_time() -> str:
    # 返回格式化的当前时间字符串
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_torch_version() -> str:
    # 返回当前安装的 PyTorch 版本号
    return str(torch.__version__)


def get_git_commit_id() -> str:
    # git 命令：获取当前 HEAD 指向的 commit 哈希
    command = "git rev-parse HEAD"
    # 执行命令并解码输出（strip 去掉尾部换行）
    commit_id = subprocess.check_output(command.split()).strip().decode("utf-8")
    # 返回 commit 哈希字符串
    return commit_id


def register_new_resolvers() -> None:
    # 注册表：解析器名 -> 实现函数（generate_id 生成实验唯一 ID，其余获取时间/版本/目录等）
    for resolver, function in {
        "generate_id": generate_id,
        "get_current_time": get_current_time,
        "get_torch_version": get_torch_version,
        "get_git_commit_id": get_git_commit_id,
        "get_superimage_dir": get_superimage_dir,
        "get_mlebench_data_dir": get_mlebench_data_dir,
    }.items():
        # 防重复注册：只有尚未注册的解析器才注册
        if not omegaconf.OmegaConf._get_resolver(resolver):
            # 向 OmegaConf 注册自定义解析器
            omegaconf.OmegaConf.register_new_resolver(resolver, function)
