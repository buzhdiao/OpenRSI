# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入 Path，用于定位划分文件所在目录
from pathlib import Path


# 为指定数据划分生成手动覆盖（manual_overrides）配置文件，
# 把该划分包含的比赛 ID 写入 YAML，供后续准备数据时限定范围
def generate_manual_overrides(split_name):
    # 定位 mlebench 任务目录（本文件在 utils 下，向上一级）
    file_dir = Path(__file__).resolve().parent.parent
    # 拼出划分文件所在目录 splits
    base_dir = file_dir / "splits"
    # 拼出该划分对应的文本文件路径
    split_file = base_dir / (split_name + ".txt")
    # 读取划分文件全文
    split_comp_ids = split_file.read_text()
    # 去掉首尾空白后按行拆分，得到比赛 ID 列表
    split_comp_ids = split_comp_ids.strip().split("\n")

    # 以写模式打开输出的 YAML 配置文件
    with open(f"mlebench_{split_name}.yaml", "w") as f:
        # 写入 YAML 头部：vars 下的 manual_overrides 列表起始标记
        f.write("vars:\n\tmanual_overrides: [")
        # 逐个把比赛 ID 以 competition_id=<id> 的形式写入列表
        for comp_id in split_comp_ids:
            f.write(f"competition_id={comp_id},")
        # 写入列表结束标记
        f.write("]")


# 脚本直接运行时，为 low_dev 划分生成覆盖配置
if __name__ == "__main__":
    generate_manual_overrides("low_dev")
