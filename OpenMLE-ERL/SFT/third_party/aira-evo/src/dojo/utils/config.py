# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# 通用配置构建工厂：按配置对象的类名在映射表中查找对应实现
def build(cfg, cfg_obj_map, **kwargs):
    # 用类名取出具体实现类，传入配置与额外参数完成实例化
    return cfg_obj_map[cfg.__class__.__name__](cfg, **kwargs)
