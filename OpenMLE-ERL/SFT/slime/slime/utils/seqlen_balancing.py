# Copied from https://github.com/volcengine/verl/blob/468adf22c43b744348051fccd7a5d830c6c3c36a/verl/utils/seqlen_balancing.py
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 本文件复制自 verl：序列长度均衡划分算法，用于把样本均衡切分到各 DP rank / microbatch，减小流水线空泡

# 导入 copy：深拷贝映射表（构造逆映射时避免修改原数据）
import copy
# 导入 heapq：Karmarkar-Karp 算法用最小堆维护状态优先队列
import heapq


def karmarkar_karp(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    # see: https://en.wikipedia.org/wiki/Largest_differencing_method
    # Karmarkar-Karp 最大差值法（Largest Differencing Method）：把序列长度集合划分成 k 份，最小化各份总和的最大差

    # 内部类：一个分区的抽象（保存成员及其长度和）
    class Set:

        def __init__(self) -> None:
            # 分区内元素长度之和
            self.sum = 0
            # 分区内元素列表：(原始下标, 长度)
            self.items = []

        def add(self, idx: int, val: int):
            # 加入一个元素并累计总和
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other):
            # 合并另一个分区的全部元素
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other):
            # 比较规则：先比总和，再比元素数，最后逐元素比较（保证排序确定性）
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    # 内部类：一个"划分状态"，即 k 个 Set 的组合
    class State:

        def __init__(self, items: list[tuple[int, int]], k: int) -> None:
            # 目标分区数
            self.k = k
            # sets should always be decreasing order
            # 初始化 k 个空分区
            self.sets = [Set() for _ in range(k)]
            # 元素数要么是 1（单元素状态）要么恰为 k（等分初始状态）
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            # 把初始元素放入前几个分区
            for i, (idx, seqlen) in enumerate(items):
                self.sets[i].add(idx=idx, val=seqlen)
            # 按总和从大到小排序，保持约定
            self.sets = sorted(self.sets, reverse=True)

        def get_partitions(self):
            # 从最终状态导出划分结果：每个分区取其元素下标列表
            partitions = []
            for i in range(len(self.sets)):
                # 收集当前分区的全部下标
                cur_partition = []
                for idx, _ in self.sets[i].items:
                    cur_partition.append(idx)
                partitions.append(cur_partition)
            # 返回 k 个下标列表
            return partitions

        def merge(self, other):
            # 合并两个状态：本状态第 i 大的分区吸收对方第 i 小的分区（KK 算法核心，抵消大数差）
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            # 重新按总和降序排列
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> int:
            # 极差：最大分区和减最小分区和（越小说明划分越均衡）
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other):
            # least heap, let the state with largest spread to be popped first,
            # if the spread is the same, let the state who has the largest set
            # to be popped first.
            # 最小堆反序定义：极差大的状态先出堆（优先合并最不均衡的状态）
            if self.spread != other.spread:
                return self.spread > other.spread
            # 极差相同时首分区大者先出堆，保证确定性
            return self.sets[0] > other.sets[0]

        def __repr__(self) -> str:
            # 调试用：按分区打印各元素长度
            repr_str = "["
            for i in range(self.k):
                if i > 0:
                    repr_str += ","
                repr_str += "{"
                for j, (_, seqlen) in enumerate(self.sets[i].items):
                    if j > 0:
                        repr_str += ","
                    repr_str += str(seqlen)
                repr_str += "}"
            repr_str += "]"
            return repr_str

    # 按 (长度, 下标) 排序，保证后续处理顺序确定
    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    # 状态优先队列（最小堆）
    states_pq = []
    if equal_size:
        # 等分模式：元素数必须能被 k 整除
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"
        # 每次取连续 k 个元素构造一个初始状态（长度相近的元素分到不同分区）
        for offset in range(0, len(sorted_seqlen_list), k_partitions):
            items = []
            for i in range(k_partitions):
                # 取出该组的长度与原始下标
                seqlen, idx = sorted_seqlen_list[offset + i]
                items.append((idx, seqlen))
            # 入堆
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        # 非等分模式：每个元素单独作为一个初始状态
        for seqlen, idx in sorted_seqlen_list:
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    # 反复取出两个状态合并，直到只剩一个
    while len(states_pq) > 1:
        # 取出当前最不均衡的状态
        state0 = heapq.heappop(states_pq)
        # 取出次不均衡的状态
        state1 = heapq.heappop(states_pq)
        # merge states
        # 合并两者
        state0.merge(state1)
        # 合并结果回堆
        heapq.heappush(states_pq, state0)

    # 最终唯一的状态即为结果
    final_state = states_pq[0]
    # 导出 k 个分区的下标列表
    partitions = final_state.get_partitions()
    if equal_size:
        # 等分模式校验：每个分区元素数应相等
        for _i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(
                seqlen_list
            ), f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
    # 返回划分结果
    return partitions


def greedy_partition(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    # 贪心划分：把元素从大到小依次放入当前总和最小的分区（近似均衡，速度快）
    # 等分模式下给长度加同一个大偏置，使"元素数均衡"优先于"长度均衡"
    bias = sum(seqlen_list) + 1 if equal_size else 0
    # 构造带偏置的 (长度, 下标) 列表并降序处理
    sorted_seqlen = [(seqlen + bias, i) for i, seqlen in enumerate(seqlen_list)]
    # 初始化 k 个空分区
    partitions = [[] for _ in range(k_partitions)]
    # 各分区的当前总和
    partition_sums = [0 for _ in range(k_partitions)]
    for seqlen, i in sorted_seqlen:
        # 找当前总和最小的分区
        min_idx = None
        for j in range(k_partitions):
            if min_idx is None or partition_sums[j] < partition_sums[min_idx]:
                min_idx = j
        # 元素放入该分区
        partitions[min_idx].append(i)
        # 更新该分区总和
        partition_sums[min_idx] += seqlen
    if equal_size:
        # 等分模式校验：各分区元素数相等
        for _i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(
                seqlen_list
            ), f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
    # 返回划分结果
    return partitions


def get_seqlen_balanced_partitions(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    """get order of seq lengths to make partitions balanced, this is
        used in balacing sum of seqlength across dp ranks and microbatches
    Parameters:
        seqlen_list (List[int]):
            seq lengths of each items
        k_partitions (int):
            resulting number of partitions
        equal_size (bool):
            if True, number of items in each partitions must be equal.
            if False, only consider balancing the sum, each partition can have
            variable number of items
    Returns:
        partitions (List[List[int]]):
            return k_partitions list containing the index of items.
    """
    # 元素数必须不少于分区数
    assert len(seqlen_list) >= k_partitions, f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]"

    def _check_and_sort_partitions(partitions):
        # 校验分区数量正确
        assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"
        # 记录出现过的下标，用于完整性校验
        seen_idx = set()
        # 预排序结果容器
        sorted_partitions = [None] * k_partitions
        for i, partition in enumerate(partitions):
            # 不允许空分区
            assert len(partition) > 0, f"the {i}-th partition is empty"
            # 收集该分区下标
            for idx in partition:
                seen_idx.add(idx)
            # 分区内下标排序，保证确定性
            sorted_partitions[i] = sorted(partition)
        # 校验所有下标恰好出现一次（无遗漏无重复）
        assert seen_idx == set(range(len(seqlen_list)))
        # 返回排序后的分区
        return sorted_partitions

    # 使用 Karmarkar-Karp 算法计算均衡划分
    partitions = karmarkar_karp(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    # 校验并排序后返回
    return _check_and_sort_partitions(partitions)


def get_reverse_idx(idx_map):
    # 深拷贝原映射作为逆映射的底板
    reverse_idx_map = copy.deepcopy(idx_map)

    # 遍历正向映射，把"位置 -> 值"翻转为"值 -> 位置"
    for i, idx in enumerate(idx_map):
        reverse_idx_map[idx] = i

    # 返回逆映射
    return reverse_idx_map
