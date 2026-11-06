"""CI utilities for Megatron backend testing."""

import logging
from collections.abc import Sequence

from megatron.core.distributed import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


def check_mtp_only_grad(model: Sequence[DDP], step_id: int) -> None:
    """Check that only MTP parameters have non-zero gradients.

    This is used for CI testing to verify that when all outputs are truncated,
    only the MTP layers receive gradients (since only mtp_loss contributes).

    Args:
        model: Sequence of DDP-wrapped model chunks.
        step_id: Current step index for logging.

    Raises:
        AssertionError: If any non-MTP parameter has a non-zero gradient.
    """
    # 分别收集非零梯度的非 MTP 参数与 MTP 参数（参数名, 梯度绝对值最大值）
    non_mtp_nonzero_grads = []
    mtp_nonzero_grads = []

    # 遍历每个模型分片（model chunk）的全部参数
    for model_chunk in model:
        for name, param in model_chunk.named_parameters():
            # Get the main_grad from the distributed optimizer if available
            # 优先取分布式优化器写入的 main_grad（Megatron 的梯度累积缓冲）
            grad = getattr(param, "main_grad", None)
            # 回退到普通 grad
            if grad is None:
                grad = param.grad
            # 两者都没有说明该参数本步未参与计算，跳过
            if grad is None:
                continue

            # 取梯度绝对值的最大值作为该参数梯度是否为零的判据
            grad_norm = grad.abs().max().item()
            # 通过参数名中是否包含 ".mtp." 判断是否为 MTP（Multi-Token Prediction）参数
            is_mtp = ".mtp." in name

            # 按 MTP 与否分别记录非零梯度
            if is_mtp:
                if grad_norm > 0:
                    mtp_nonzero_grads.append((name, grad_norm))
            else:
                if grad_norm > 0:
                    non_mtp_nonzero_grads.append((name, grad_norm))

    # Log the results
    # 打印本轮检查的统计结果
    logger.info(
        f"[CI MTP Grad Check] Step {step_id}: "
        f"MTP params with non-zero grad: {len(mtp_nonzero_grads)}, "
        f"non-MTP params with non-zero grad: {len(non_mtp_nonzero_grads)}"
    )

    if non_mtp_nonzero_grads:
        # Log the first few non-MTP params with non-zero gradients for debugging
        # 打印前几个出现非零梯度的非 MTP 参数，便于定位泄漏来源
        for name, grad_norm in non_mtp_nonzero_grads[:5]:
            logger.error(f"[CI MTP Grad Check] Non-MTP param with non-zero grad: {name}, max_grad={grad_norm}")

    # 断言所有非 MTP 参数梯度为零（输出全被截断时只有 mtp_loss 有贡献）
    assert len(non_mtp_nonzero_grads) == 0, (
        f"Expected all non-MTP parameters to have zero gradients, "
        f"but found {len(non_mtp_nonzero_grads)} with non-zero gradients. "
        f"First few: {non_mtp_nonzero_grads[:5]}"
    )

    # Also verify that MTP params do have gradients (otherwise the test is not valid)
    # 同时确认 MTP 参数确实收到了梯度，否则说明 MTP loss 没有被计算，测试本身无效
    assert len(mtp_nonzero_grads) > 0, (
        "Expected MTP parameters to have non-zero gradients, but all were zero. "
        "This may indicate the MTP loss is not being computed."
    )


def check_mtp_loss(mtp_loss: float, max_mtp_loss: float = 1.0) -> None:
    """Check that MTP loss is within expected bounds.

    Args:
        mtp_loss: The computed MTP loss value.
        max_mtp_loss: Maximum allowed MTP loss (default: 1.0).

    Raises:
        AssertionError: If MTP loss exceeds the maximum allowed value.
    """
    # 断言 MTP loss 小于上限，超过说明 MTP（Multi-Token Prediction）训练可能异常
    assert mtp_loss < max_mtp_loss, (
        f"MTP loss {mtp_loss} exceeds maximum allowed value {max_mtp_loss}. "
        "This may indicate an issue with MTP training."
    )
