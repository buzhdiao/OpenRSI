import torch
import triton
import triton.language as tl

# FP8 量化目标格式：e4m3（4 位指数 + 3 位尾数），fn 表示有限数（无 inf/nan 编码）
fp8_dtype = torch.float8_e4m3fn
# 该 FP8 格式能表示的最大值，用作 per-block 缩放的基准
fp8_max = torch.finfo(fp8_dtype).max
# 最小可表示值为 -max（对称格式）
fp8_min = -fp8_max


def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
            x: the dividend.
            y: the divisor.

    Returns:
            The result of the ceiling division.
    """
    # 向上取整除法：常用于计算 block 数量
    return (x + y - 1) // y


@triton.jit
def _blockwise_cast_to_fp8_triton(
    X,
    Y,
    S,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_sm,
    stride_sn,
    M,
    N,
    eps,
    fp8_min,
    fp8_max,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 128,
):
    # 二维 grid：axis 0 对应 M 方向块号，axis 1 对应 N 方向块号（转 int64 防大矩阵溢出）
    pid_m = tl.cast(tl.program_id(axis=0), tl.int64)
    pid_n = tl.cast(tl.program_id(axis=1), tl.int64)
    # 计算本 block 覆盖的行/列全局索引
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # 行/列越界掩码，用于处理不能整除的边界 block
    mask_m = off_m < M
    mask_n = off_n < N
    # 组合成二维掩码
    mask = mask_m[:, None] & mask_n[None, :]

    # 按掩码加载本 block 的数据并转成 fp32 计算，越界处填 0
    x = tl.load(X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn, mask=mask, other=0.0).to(tl.float32)
    # 求 block 内绝对值最大值；与 eps 取 max 防止除零（全 0 block 的情况）
    _absmax = tl.maximum(tl.max(tl.abs(x)), eps)
    # 计算 per-block 缩放因子 scale = absmax / fp8_max
    x_s = _absmax / fp8_max
    # 取其倒数，避免逐元素除法
    s_inv = 1.0 / x_s
    # 反缩放后截断到 FP8 表示范围，再转成 FP8 类型（per-block 量化）
    y_q = tl.clamp(x * s_inv, fp8_min, fp8_max).to(Y.dtype.element_ty)

    # 写回量化后的 FP8 数据（掩码防止越界写入）
    tl.store(Y + off_m[:, None] * stride_ym + off_n[None, :] * stride_yn, y_q, mask=mask)
    # 把该 block 的 scale 写到 scales 张量对应位置
    tl.store(S + pid_m * stride_sm + pid_n * stride_sn, x_s)


def blockwise_cast_to_fp8_triton(x: torch.Tensor, block_size=None) -> tuple[torch.Tensor, torch.Tensor]:
    # 默认 block 形状为 128x128
    BLOCK_M, BLOCK_N = 128, 128
    # 允许调用方自定义 block 大小
    if block_size:
        BLOCK_M, BLOCK_N = block_size[0], block_size[1]
    # 输入矩阵形状
    M, N = x.shape
    # 预分配量化输出的 FP8 张量
    y = torch.empty(M, N, device=x.device, dtype=torch.float8_e4m3fn)
    # 预分配 per-block scale 张量（fp32），形状为 block 网格大小
    s = torch.empty(ceil_div(M, BLOCK_M), ceil_div(N, BLOCK_N), dtype=torch.float32, device=x.device)

    # Triton grid 回调：按实际 block 大小计算网格维度
    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    # 连续内存布局可用更多 warp/更少 stage 提升吞吐；非连续时保守配置
    if x.is_contiguous():
        kwargs = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "num_warps": 8, "num_stages": 2}
    else:
        kwargs = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "num_warps": 1, "num_stages": 4}
    # 启动 Triton kernel：传入两个矩阵及各自 stride、形状、eps 和 FP8 范围
    _blockwise_cast_to_fp8_triton[grid](
        x, y, s, *x.stride(), *y.stride(), *s.stride(), M, N, 1e-10, fp8_min, fp8_max, **kwargs
    )
    # 返回量化数据与对应的 per-block scale
    return y, s
