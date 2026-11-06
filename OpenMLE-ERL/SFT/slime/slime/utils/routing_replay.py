# 导入 os：读取路由重放（routing replay）相关的环境变量开关
import os
# 导入 torch：张量拷贝、gather 与 CUDA 设备管理
import torch


# 模块级全局：当前生效的路由重放记录器（由 forward pre-hook 设置）
ROUTING_REPLAY = None


def set_routing_replay(replay):
    # 声明修改模块级全局记录器
    global ROUTING_REPLAY
    # 设置当前的记录器实例
    ROUTING_REPLAY = replay


class RoutingReplay:
    # MoE 路由重放记录器：保存 forward 时的专家选择，backward/重放时复用，省去重复 topk 计算
    # 类级列表：登记所有实例，便于统一清理
    all_routing_replays = []

    def __init__(self):
        # forward 阶段的读取游标（逐层递增）
        self.forward_index = 0
        # backward 阶段的读取游标（逐层递增）
        self.backward_index = 0
        # 每层 MoE 的 top-k 专家下标缓存（CPU 锁页内存）
        self.top_indices_list = []
        # 把新实例登记到类级列表
        RoutingReplay.all_routing_replays.append(self)

    def record(self, top_indices):
        # offload top_indices to CPU pinned memory
        # 在 CPU 上创建同形状的锁页内存缓冲（可异步 H2D 拷贝且规避反传阶段 GPU 数据被覆盖）
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        # 把 GPU 上的专家下标拷贝到 CPU 缓冲
        buf.copy_(top_indices)
        # 追加到缓存列表
        self.top_indices_list.append(buf)

    def pop_forward(self):
        # 按游标取出 forward 阶段的第 index 层缓存
        top_indices = self.top_indices_list[self.forward_index]
        # 游标前移
        self.forward_index += 1
        # 拷回当前 GPU 设备并返回
        return top_indices.to(torch.cuda.current_device())

    def pop_backward(self):
        # 按游标取出 backward 阶段的第 index 层缓存（与 forward 同序）
        top_indices = self.top_indices_list[self.backward_index]
        # 游标前移
        self.backward_index += 1
        # 拷回当前 GPU 设备并返回
        return top_indices.to(torch.cuda.current_device())

    def clear(self):
        # 完全清空：游标与缓存全部重置（新一批数据开始时）
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []

    def clear_forward(self):
        # 仅重置 forward 游标（缓存保留给 backward 复用）
        self.forward_index = 0

    @staticmethod
    def clear_all():
        # 清空所有层的记录器（完整重置）
        for replay in RoutingReplay.all_routing_replays:
            replay.clear()

    @staticmethod
    def clear_all_forward():
        # 重置所有记录器的 forward 游标
        for replay in RoutingReplay.all_routing_replays:
            replay.clear_forward()


def get_routing_replay_compute_topk(old_compute_topk):
    # 包装 MoE 的 top-k 路由函数：按环境变量指定的阶段决定记录还是重放
    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        # 未开启路由重放开关时走原始逻辑
        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            # 读取当前阶段标识
            routing_replay_stage = os.environ["ROUTING_REPLAY_STAGE"]
            if routing_replay_stage == "fallthrough":
                # 直通阶段：正常计算（不记录不重放）
                return old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
            if routing_replay_stage == "record":
                # 记录阶段：正常计算 top-k，并把专家下标存入缓存
                probs, top_indices = old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
                ROUTING_REPLAY.record(top_indices)
            elif routing_replay_stage == "replay_forward":
                # forward 重放阶段：直接取缓存，跳过 topk 计算
                top_indices = ROUTING_REPLAY.pop_forward()
                assert (
                    top_indices.shape[0] == scores.shape[0] and top_indices.shape[1] == topk
                ), f"[{torch.distributed.get_rank()}] top_indices shape {top_indices.shape} does not match scores shape {scores.shape} and topk {topk}"
                # 用 gather 按缓存下标取回对应专家得分作为概率
                probs = scores.gather(1, top_indices)
            elif routing_replay_stage == "replay_backward":
                # backward 重放阶段：取缓存以保证梯度路径与 forward 一致
                top_indices = ROUTING_REPLAY.pop_backward()
                assert (
                    top_indices.shape[0] == scores.shape[0] and top_indices.shape[1] == topk
                ), f"top_indices shape {top_indices.shape} does not match scores shape {scores.shape} and topk {topk}"
                # 同样用 gather 还原概率
                probs = scores.gather(1, top_indices)
            # 返回（概率, 专家下标）
            return probs, top_indices
        else:
            # 开关关闭：完全走原始 top-k 逻辑
            return old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)

    # 返回包装后的路由函数
    return compute_topk


def register_routing_replay(module):
    # 仅在开关开启时为模块挂载重放记录器
    if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
        # 给模块挂一个专属记录器
        module.routing_replay = RoutingReplay()

        def pre_forward_hook(*args, **kwargs):
            # forward 前把本模块的记录器设为全局当前记录器
            set_routing_replay(module.routing_replay)

        # 注册 forward pre-hook：保证每层 forward 前全局记录器指向正确的层
        module.register_forward_pre_hook(pre_forward_hook)
