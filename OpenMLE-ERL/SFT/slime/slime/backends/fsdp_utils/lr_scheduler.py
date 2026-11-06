# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Learning rate scheduler for FSDP training."""

# 导入 logging 用于记录日志
import logging
# 导入 math 用于余弦衰减与开方运算
import math

# 导入 torch 主库（优化器类型注解）
import torch
# 导入 PyTorch 学习率调度器基类 LRScheduler
from torch.optim.lr_scheduler import LRScheduler
# 导入 override 装饰器用于静态检查方法重写签名
from typing_extensions import override

# 获取本模块 logger
logger = logging.getLogger(__name__)


# FSDP 训练的学习率调度器：支持线性预热 + constant/linear/cosine/inv-sqrt/WSD 等衰减方式
class FSDPLRScheduler(LRScheduler):
    """Learning rate scheduler for FSDP training.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to be used.
        init_lr (float): Initial learning rate.
        max_lr (float): Maximum learning rate.
        min_lr (float): Minimum learning rate.
        lr_warmup_steps (int): Number of warmup steps.
        lr_decay_steps (int): Number of decay steps.
        lr_decay_style (str): Decay style for learning rate.
        use_checkpoint_lr_scheduler (bool, optional): Whether to use the checkpoint values
            for the lr scheduler.
        override_lr_scheduler (bool, optional): Whether to override the lr scheduler values
            with the class values.
        wsd_decay_steps (int, optional): Number of weight decay decay steps.
        lr_wsd_decay_style (str, optional): Decay style for learning rate during weight decay decay
            steps.
        last_epoch (int, optional): The index of last epoch. Default: -1.
    """

    # 构造函数：保存调度相关的自定义参数并做合法性校验
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        init_lr: float,
        max_lr: float,
        min_lr: float,
        lr_warmup_steps: int,
        lr_decay_steps: int,
        lr_decay_style: str,
        use_checkpoint_lr_scheduler: bool | None = True,
        override_lr_scheduler: bool | None = False,
        wsd_decay_steps: int | None = None,
        lr_wsd_decay_style: str | None = None,
        last_epoch: int = -1,
    ) -> None:
        # Store our custom parameters
        # 保存预热起始学习率
        self.init_lr = init_lr
        # 保存学习率上界（转 float 防止传入 int）
        self.max_lr = float(max_lr)
        # 保存学习率下界
        self.min_lr = min_lr
        # 校验最小学习率非负
        assert self.min_lr >= 0.0
        # 校验 max >= min
        assert self.max_lr >= self.min_lr
        # 校验初始 lr 不超过上界
        assert self.init_lr <= self.max_lr

        # 保存预热步数
        self.lr_warmup_steps = lr_warmup_steps
        # 保存衰减总步数
        self.lr_decay_steps = lr_decay_steps
        # 保存 WSD（Warmup-Stable-Decay）衰减段步数
        self.wsd_decay_steps = wsd_decay_steps
        # 保存 WSD 衰减方式
        self.lr_wsd_decay_style = lr_wsd_decay_style

        # 校验衰减步数为正
        assert self.lr_decay_steps > 0
        # 校验预热步数小于衰减步数（否则曲线无意义）
        assert self.lr_warmup_steps < self.lr_decay_steps

        # 保存衰减风格字符串
        self.lr_decay_style = lr_decay_style
        if self.lr_decay_style == "WSD":
            # WSD 风格必须给出衰减段步数
            assert self.wsd_decay_steps is not None

        # 保存是否用类参数覆盖 checkpoint 中的调度器配置
        self.override_lr_scheduler = override_lr_scheduler
        # 保存是否使用 checkpoint 恢复的调度器配置
        self.use_checkpoint_lr_scheduler = use_checkpoint_lr_scheduler

        if self.override_lr_scheduler:
            # 覆盖与使用 checkpoint 两种语义互斥，同时设置视为配置错误
            assert not self.use_checkpoint_lr_scheduler, "both override and use-checkpoint are set."

        # Initialize parent class
        # 初始化父类（会触发基类的调度注册逻辑，包括按 param_group 写入初始 lr）
        super().__init__(optimizer, last_epoch)

        # 记录本次使用的衰减风格
        logger.info(f"> learning rate decay style: {self.lr_decay_style}")

    # 为指定参数组计算当前步的学习率
    def _get_lr_for_group(self, param_group: dict) -> float:
        """Compute learning rate for a specific parameter group.

        Args:
            param_group (dict): parameter group from the optimizer.

        Returns:
            float: learning rate for this parameter group.
        """
        # 优先使用参数组内自定义的上/下界，缺省回退到调度器全局值
        max_lr = param_group.get("max_lr", self.max_lr)
        min_lr = param_group.get("min_lr", self.min_lr)

        # Use linear warmup for the initial part.
        # 预热阶段：从 init_lr 线性升到 max_lr（线性预热，warmup）
        if self.lr_warmup_steps > 0 and self.last_epoch <= self.lr_warmup_steps:
            return self.init_lr + ((max_lr - self.init_lr) * float(self.last_epoch) / float(self.lr_warmup_steps))

        # If the learning rate is constant, just return the initial value.
        # 恒定学习率风格：预热结束后保持 max_lr 不变
        if self.lr_decay_style == "constant":
            return max_lr

        # For any steps larger than `self.lr_decay_steps`, use `min_lr`.
        # 超过衰减总步数后固定在最小学习率
        if self.last_epoch > self.lr_decay_steps:
            return min_lr

        # If we are done with the warmup period, use the decay style.
        # 逆平方根衰减：lr = max_lr * sqrt(warmup_steps / step)，实现 1/sqrt(t) 的衰减形状
        if self.lr_decay_style == "inverse-square-root":
            warmup_steps = max(self.lr_warmup_steps, 1)
            num_steps = max(self.last_epoch, 1)
            lr = max_lr * warmup_steps**0.5 / (num_steps**0.5)
            # 不低于 min_lr
            return max(min_lr, lr)

        # 计算预热结束后的相对进度：已走步数与衰减段步数之比
        num_steps_ = self.last_epoch - self.lr_warmup_steps
        decay_steps_ = self.lr_decay_steps - self.lr_warmup_steps
        decay_ratio = float(num_steps_) / float(decay_steps_)
        # 校验进度比例在 [0, 1] 区间内
        assert decay_ratio >= 0.0
        assert decay_ratio <= 1.0

        # 学习率的总下降幅度
        delta_lr = max_lr - min_lr
        # 衰减系数 coeff：1 表示保持 max_lr，0 表示降到 min_lr
        coeff = None

        # 按衰减风格计算系数
        if self.lr_decay_style == "linear":
            # 线性衰减：随进度从 1 降到 0
            coeff = 1.0 - decay_ratio
        elif self.lr_decay_style == "cosine":
            # 余弦衰减：半周期余弦从 1 平滑降到 0
            coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        elif self.lr_decay_style == "WSD":
            # WSD 风格：先保持 max_lr（稳定段），最后 wsd_decay_steps 内退火到 min_lr
            wsd_anneal_start_ = self.lr_decay_steps - self.wsd_decay_steps
            if self.last_epoch <= wsd_anneal_start_:
                # 稳定段：系数保持 1
                coeff = 1.0
            else:
                # 退火段：计算退火相对进度
                wsd_steps = self.last_epoch - wsd_anneal_start_
                wsd_decay_ratio = float(wsd_steps) / float(self.wsd_decay_steps)
                # 按退火方式计算系数
                if self.lr_wsd_decay_style == "linear":
                    coeff = 1.0 - wsd_decay_ratio
                elif self.lr_wsd_decay_style == "cosine":
                    coeff = 0.5 * (math.cos(math.pi * wsd_decay_ratio) + 1.0)
                elif self.lr_wsd_decay_style == "exponential":
                    coeff = (2.0 * math.pow(0.5, wsd_decay_ratio)) - 1.0
                elif self.lr_wsd_decay_style == "minus_sqrt":
                    coeff = 1.0 - math.sqrt(wsd_decay_ratio)
        else:
            # 未知的衰减风格直接抛错
            raise Exception(f"{self.lr_decay_style} decay style is not supported.")

        # 确保系数已计算
        assert coeff is not None
        # 最终 lr = min_lr + coeff * 幅度
        return min_lr + coeff * delta_lr

    # 重写 get_lr：为每个参数组分别计算学习率
    @override
    def get_lr(self) -> list[float]:
        """Compute the learning rates for each parameter group.

        Returns:
            list[float]: A list of learning rates, one for each parameter group.
        """
        # 逐参数组调用内部计算函数
        return [self._get_lr_for_group(group) for group in self.optimizer.param_groups]


# 工厂函数：根据训练参数与优化器创建配置好的学习率调度器
def get_lr_scheduler(args, optimizer: torch.optim.Optimizer) -> FSDPLRScheduler:
    """Create and configure the learning-rate scheduler.

    This configures iteration-based schedules derived from the global batch size
    and run-time arguments.

    Args:
        args: Training/runtime arguments (namespace).
        optimizer (torch.optim.Optimizer): Optimizer bound to the model.

    Returns:
        FSDPLRScheduler: Initialized scheduler bound to ``optimizer``.
    """
    # 由 RL 招样规模推算总训练迭代数：num_rollout * rollout_batch_size * n_samples_per_prompt / global_batch_size
    args.train_iters = args.num_rollout * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    # 未显式指定衰减步数时默认等于总迭代数
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    # 衰减步数取自参数
    lr_decay_steps = args.lr_decay_iters
    # WSD 衰减步数默认为 None
    wsd_decay_steps = None
    if args.lr_wsd_decay_iters is not None:
        # 指定了 WSD 衰减迭代数时使用
        wsd_decay_steps = args.lr_wsd_decay_iters
    if args.lr_warmup_fraction is not None:
        # 以比例方式给定预热时，换算为步数
        lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
    else:
        # 否则直接使用预热迭代数
        lr_warmup_steps = args.lr_warmup_iters
    # 用换算后的参数构造 FSDP 调度器
    lr_scheduler = FSDPLRScheduler(
        optimizer,
        init_lr=args.lr_warmup_init,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        use_checkpoint_lr_scheduler=args.use_checkpoint_lr_scheduler,
        override_lr_scheduler=args.override_lr_scheduler,
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=args.lr_wsd_decay_style,
    )

    # 返回调度器实例
    return lr_scheduler
