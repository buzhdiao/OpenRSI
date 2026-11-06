# Adapted from https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/trainer/ray/utils.py#L1
# （原英文注释：改编自 OpenRLHF 的 ray utils）
import os

import ray
import torch
from slime.ray.ray_actor import RayActor


# Refer to
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L102-L103
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L94-L95
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/hpu.py#L116-L117
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/neuron.py#L108-L109
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/tpu.py#L171-L172
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L97-L98
# （原英文注释：参考 Ray 各加速器实现中"禁止设置可见设备"的环境变量名）
# 各硬件平台对应的"不让 Ray 设置 CUDA/ROCm 等可见设备变量"的实验性开关名列表
NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
    # NVIDIA CUDA 平台开关
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    # AMD ROCm 平台开关
    "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    # 华为昇腾 NPU 平台开关
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    # Intel HPU（Gaudi）平台开关
    "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
    # AWS Neuron 平台开关
    "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
    # Google TPU 平台开关
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
    # oneAPI（Intel GPU）平台开关
    "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
]


# 判断当前环境是否启用了任一"禁止设置可见设备"的开关
def ray_noset_visible_devices(env_vars=os.environ):
    # 任一开关被设为真值即返回 True
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)


# 获取当前进程所用 GPU 的物理 UUID
def get_physical_gpu_id():
    # 取当前 CUDA 设备
    device = torch.cuda.current_device()
    # 查询该设备的属性
    props = torch.cuda.get_device_properties(device)
    # 返回设备的 UUID 字符串（跨进程唯一标识物理 GPU）
    return str(props.uuid)


# 基于 Ray actor 实现的分布式锁：训练侧广播权重时的互斥原语
@ray.remote
class Lock(RayActor):
    # 构造函数：初始为未锁定
    def __init__(self):
        # False 表示未锁，True 表示已锁（原注释说明状态含义）
        self._locked = False  # False: unlocked, True: locked

    # 尝试获取锁：成功返回 True，失败返回 False（调用方需重试）
    def acquire(self):
        """
        Try to acquire the lock. Returns True if acquired, False otherwise.
        Caller should retry until it returns True.
        """
        # 未被占用时抢锁成功
        if not self._locked:
            # 置为已锁
            self._locked = True
            # 返回成功
            return True
        # 已被占用则返回 False
        return False

    # 释放锁：允许其他调用方获取
    def release(self):
        """Release the lock, allowing others to acquire."""
        # 断言当前确实持有锁，防止误释放
        assert self._locked, "Lock is not acquired, cannot release."
        # 置回未锁状态
        self._locked = False
