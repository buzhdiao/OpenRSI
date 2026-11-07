# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 os，用于读取进程环境变量
import os
# 导入 json，用于把环境变量序列化为 JSON 文件
import json
# 导入 Path，用于拼接输出路径
from pathlib import Path


# 把当前进程的环境变量完整导出为 JSON 文件，便于复现实验运行环境
def write_env_variables_to_json(output_dir):
    """
    Write the environment variables to a JSON file.
    """
    # Path to file
    # 拼出输出文件路径：<output_dir>/env_variables.json
    output_file = Path(output_dir) / "env_variables.json"

    # Convert the environment variables to a dictionary
    # 把环境变量快照为普通字典
    env_vars = dict(os.environ)

    # Write the dictionary to a JSON file
    # 以 4 空格缩进格式写入 JSON 文件
    with open(output_file, "w") as file:
        json.dump(env_vars, file, indent=4)

    # 打印完成提示
    print("Environment variables written as JSON to env_variables.json")
