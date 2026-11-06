# 导入抽象基类工具：定义 TensorBackuper 的抽象接口
from abc import ABC, abstractmethod
# 导入 defaultdict：带默认值的嵌套备份字典
from collections import defaultdict
# 导入类型注解：可调用对象与可迭代对象
from collections.abc import Callable, Iterable

# 导入 torch：张量备份/恢复的拷贝操作
import torch

# 类型别名：来源获取器——返回 (名字, 张量) 可迭代序列的可调用对象
_SourceGetter = Callable[[], Iterable[tuple[str, torch.Tensor]]]


class TensorBackuper(ABC):
    # 张量备份器抽象基类：把一组参数快照到 CPU 锁页内存，随后可恢复（用于训练/推理权重切换）
    @staticmethod
    def create(source_getter, single_tag):
        # 未指定单标签：创建真正执行备份的正常实现
        if single_tag is None:
            return _TensorBackuperNormal(source_getter=source_getter)
        else:
            # 指定单标签：创建"零拷贝"空实现（权重从不被修改时无需真正备份）
            return _TensorBackuperNoop(source_getter=source_getter, single_tag=single_tag)

    def __init__(self, source_getter: _SourceGetter):
        # 保存来源获取器（每次调用返回当前待备份的参数集合）
        self._source_getter = source_getter

    @property
    @abstractmethod
    def backup_tags(self):
        # 子类必须实现：列出当前所有备份标签
        raise NotImplementedError

    @abstractmethod
    def get(self, tag: str):
        # 子类必须实现：按标签取出备份内容
        raise NotImplementedError

    @abstractmethod
    def backup(self, tag: str):
        # 子类必须实现：按标签执行备份
        raise NotImplementedError

    def copy(self, *, src_tag: str, dst_tag: str):
        # 从一个标签复制到另一个标签（子类可覆盖）
        raise NotImplementedError

    @abstractmethod
    def restore(self, tag: str):
        # 子类必须实现：按标签恢复参数
        raise NotImplementedError


class _TensorBackuperNormal(TensorBackuper):
    def __init__(self, source_getter):
        # 委托基类保存来源获取器
        super().__init__(source_getter=source_getter)
        # 备份存储：标签 -> (参数名 -> CPU 张量副本)
        self._backups: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)

    @property
    def backup_tags(self):
        # 返回已有的全部备份标签
        return list(self._backups)

    def get(self, tag: str):
        # 返回指定标签的备份字典（参数名到 CPU 张量的映射）
        return self._backups[tag]

    @torch.no_grad()
    def backup(self, tag: str) -> None:
        # 取出该标签对应的备份字典
        backup_dict = self._backups[tag]
        # 遍历当前所有需要备份的参数
        for name, param in self._source_getter():
            if name not in backup_dict:
                # 首次备份：在 CPU 锁页内存上预分配同形状缓冲（锁页可异步传输）
                backup_dict[name] = torch.empty_like(param, device=torch.device("cpu"), pin_memory=True)
            # 非阻塞地把 GPU 参数拷贝到 CPU 缓冲
            backup_dict[name].copy_(param.detach(), non_blocking=True)
        # 同步 CUDA 流，确保异步拷贝全部完成后再返回
        torch.cuda.synchronize()

    @torch.no_grad()
    def copy(self, *, src_tag: str, dst_tag: str):
        # 把源标签的全部备份张量拷贝到目标标签（覆盖同名参数）
        for name in self._backups[dst_tag]:
            self._backups[dst_tag][name].copy_(self._backups[src_tag][name])

    @torch.no_grad()
    def restore(self, tag: str) -> None:
        # 取出该标签的备份字典
        backup_dict = self._backups[tag]
        # 遍历当前参数集合，逐个恢复
        for name, param in self._source_getter():
            # 恢复前校验该参数确实被备份过
            assert name in backup_dict
            # 非阻塞地把 CPU 备份拷回 GPU 参数
            param.copy_(backup_dict[name], non_blocking=True)
        # 同步确保恢复完成
        torch.cuda.synchronize()


class _TensorBackuperNoop(TensorBackuper):
    def __init__(self, source_getter, single_tag):
        # 委托基类保存来源获取器
        super().__init__(source_getter=source_getter)
        # 唯一允许的标签
        self._single_tag = single_tag
        # Sanity check for safety
        # 备份时的哈希指纹（用于恢复时校验权重从未被改动）
        self._backup_hash_dict = None

    @property
    def backup_tags(self):
        # 永远只有一个"虚拟"备份标签
        return [self._single_tag]

    def get(self, tag: str):
        # 直接从来源取当前参数（无真实备份）
        ans = dict(self._source_getter())
        # 与来源张量断开计算图
        ans = {k: v.detach() for k, v in ans.items()}
        # 校验当前权重哈希与备份时一致，确保"无需备份"的假设成立
        assert _compute_hash_dict(ans) == self._backup_hash_dict
        # 返回参数快照
        return ans

    def backup(self, tag: str) -> None:
        # 只允许使用预设的单标签
        assert tag == self._single_tag
        # 不拷贝数据，仅记录当前权重的哈希指纹
        self._backup_hash_dict = _compute_hash_dict(dict(self._source_getter()))
        # 同步 CUDA，保证计算哈希时数据可见
        torch.cuda.synchronize()

    def restore(self, tag: str) -> None:
        # 只允许使用预设的单标签
        assert tag == self._single_tag
        # 恢复时校验权重哈希未变（数据从未被改，无需真正恢复）
        assert _compute_hash_dict(dict(self._source_getter())) == self._backup_hash_dict
        # 同步 CUDA
        torch.cuda.synchronize()


def _compute_hash_dict(tensors: dict[str, torch.Tensor]):
    # 对张量字典逐项计算哈希指纹
    return {k: _compute_hash_tensor(v) for k, v in tensors.items()}


def _compute_hash_tensor(x: torch.Tensor):
    # Not a real/good hash, but pretty fast
    # 伪哈希：把张量摊平成 uint32 后求和（快但非严格哈希，仅作一致性校验）
    # 保证内存连续以便视图重解释
    x = x.contiguous()
    # 摊平为一维
    x = x.view(-1)
    # 按字节重解释为 uint32（int32/float32 均为 4 字节）
    x = x.view(torch.uint32)
    # 求和得到指纹
    x = x.sum()
    # 转回 Python 标量返回
    return x.item()
