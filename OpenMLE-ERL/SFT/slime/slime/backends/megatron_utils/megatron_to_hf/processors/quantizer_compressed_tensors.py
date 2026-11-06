import logging
import math
import re

import torch
import torch.nn as nn

try:
    # 尝试导入 int4 伪量化（fake quant）CUDA 扩展，用于 compressed-tensors int4 量化
    import fake_int4_quant_cuda
except ImportError:
    # 缺少扩展时置为 None，调用处会依赖它（int4 量化不可用）
    fake_int4_quant_cuda = None

logger = logging.getLogger(__name__)


__all__ = ["quantize_params_compressed_tensors"]


# 模拟 AWQ GEMM 风格的 4bit 线性层：把 FP16 权重打包成 int32 的压缩存储格式
class WQLinear_GEMM(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev, training=False):
        super().__init__()

        # 目前仅支持 4bit 权重量化
        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        # group_size 为 -1 表示整行作为一组（per-channel）
        self.group_size = group_size if group_size != -1 else in_features
        self.training = training

        # quick sanity check (make sure alignment)
        # 对齐检查：输入维度必须能被分组大小整除
        assert self.in_features % self.group_size == 0
        # 输出维度必须能被“每个 int32 打包的权重数”整除
        assert out_features % (32 // self.w_bit) == 0

        # 打包后的量化权重：每个 int32 装 32/w_bit 个 4bit 权重
        self.register_buffer(
            "qweight",
            torch.zeros(
                (in_features, out_features // (32 // self.w_bit)),
                dtype=torch.int32,
                device=dev,
            ),
        )
        # 打包后的零点：每行每输出列组一个 int32
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (in_features // self.group_size, out_features // (32 // self.w_bit)),
                dtype=torch.int32,
                device=dev,
            ),
        )
        # 反量化缩放因子：每个分组一个 FP16 scale
        self.register_buffer(
            "scales",
            torch.zeros(
                (in_features // self.group_size, out_features),
                dtype=torch.float16,
                device=dev,
            ),
        )
        if bias:
            # 需要偏置时申请 FP16 的 bias 缓冲
            self.register_buffer(
                "bias",
                torch.zeros(
                    (out_features),
                    dtype=torch.float16,
                    device=dev,
                ),
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None):
        # 由普通 nn.Linear 构造本量化层（按 AWQ 打包格式填充 qweight/qzeros）
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )
        if init_only:  # just prepare for loading sd
            # 仅创建空壳用于加载 state_dict，不做真正量化
            return awq_linear

        # need scales and zeros info for real quantization
        # 真正量化必须提供 scale 与 zero-point
        assert scales is not None and zeros is not None

        awq_linear.scales = scales.clone().half()
        if linear.bias is not None:
            # 偏置转 FP16 保存
            awq_linear.bias = linear.bias.clone().half()

        # 每个 int32 可打包的 4bit 数量
        pack_num = 32 // awq_linear.w_bit
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        # 把 scale 沿分组维 repeat 展开到与权重同形状，便于逐元素除法
        repeat_scales = scales.to(device).t().repeat_interleave(group_size, 1)
        if isinstance(zeros, torch.Tensor):
            # 零点为张量时同样展开
            repeat_zeros = zeros.to(device).t().repeat_interleave(group_size, 1)
        else:
            # 零点为标量时直接使用
            repeat_zeros = zeros
        # 量化：round(weight / scale + zero)，转置为 (out, in) 并保证连续
        intweight = torch.round(linear.weight.to(device) / repeat_scales + repeat_zeros).to(torch.int).t().contiguous()
        intweight = intweight.to(dtype=torch.int32)
        # 释放临时展开的 scale
        del repeat_scales

        # 重塑为 (…, 组内每 pack_num 个元素) 的形状，准备按位打包
        intweight = intweight.reshape(-1, intweight.shape[1] // pack_num, pack_num)

        # AWQ 打包的通道交错顺序：低半字节/高半字节交错排列
        new_order_map = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=device) * awq_linear.w_bit
        intweight = intweight << new_order_map
        # 按位或求和压入同一个 int32
        intweight = torch.sum(intweight, dim=-1).to(torch.int32)
        awq_linear.qweight = intweight

        if isinstance(zeros, torch.Tensor):
            # 零点张量：同样的打包方式压成 qzeros
            zeros = zeros.to(dtype=torch.int32, device=device)
            zeros = zeros.reshape(-1, zeros.shape[1] // pack_num, pack_num)
            zeros = zeros << new_order_map
            qzeros = torch.sum(zeros, dim=-1).to(torch.int32)

        else:
            # 标量零点：构造全为该打包值的 qzeros 矩阵
            value = 0
            for i in range(pack_num):
                # 把同一个零点按各字节位置拼进一个 int32
                value |= zeros << (i * awq_linear.w_bit)
            qzeros = (
                torch.ones(
                    (scales.shape[0], scales.shape[1] // pack_num),
                    dtype=torch.int32,
                    device=device,
                )
                * value
            )

        awq_linear.qzeros = qzeros

        return awq_linear


def pack_to_int32(
    value,
    num_bits,
    packed_dim=1,
    sym=False,
):
    # if value.dtype is not torch.int8:
    #     raise ValueError("Tensor must be quantized to torch.int8 before packing")

    # 打包要求位宽小于 8（否则无法压进 int32 的多子字节布局）
    if num_bits > 8:
        raise ValueError("Packing is only supported for less than 8 bits")

    if num_bits < 1:
        raise ValueError(f"num_bits must be at least 1, got {num_bits}")

    # Convert to unsigned range for packing, matching quantization offset
    # 对称量化时把带符号值平移到无符号区间，便于直接打包
    if sym:
        offset = 1 << (num_bits - 1)
        value = (value + offset).to(torch.uint8)
    device = value.device

    # 每个 int32 能装下的数值个数
    pack_factor = 32 // num_bits

    if packed_dim == 0:
        # 沿第 0 维打包时先转置成沿列打包
        value = value.transpose(0, 1)

    rows, cols = value.shape
    # 把列数补齐到 pack_factor 的整数倍
    padded_cols = math.ceil(cols / pack_factor) * pack_factor
    pad_len = padded_cols - cols

    if pad_len > 0:
        # 尾部补零
        value = torch.nn.functional.pad(value, (0, pad_len))

    # 打包后的组数
    num_groups = padded_cols // pack_factor

    # Use int32 here
    # 重塑为 (rows, num_groups, pack_factor) 并转 int32 以便位移
    reshaped = value.view(rows, num_groups, pack_factor).to(torch.int32)
    # 每个元素对应的移位量：第 i 个元素左移 i*num_bits 位
    bit_shifts = torch.arange(pack_factor, device=device, dtype=torch.int32) * num_bits
    # 位移后按组求和即完成打包
    packed = (reshaped << bit_shifts).sum(dim=2, dtype=torch.int32)

    if packed_dim == 0:
        # 转置回来恢复原始形状布局
        packed = packed.transpose(0, 1)

    return packed


def round_to_quantized_type_dtype(
    tensor,
    dtype,
    cast_to_original_dtype=False,
):
    # 把浮点张量按目标整数量化类型的范围做 clamp + round
    original_dtype = tensor.dtype
    # 取该整型类型的取值范围
    iinfo = torch.iinfo(dtype)
    # 先截断到 [min, max] 再四舍五入并转换类型
    rounded = torch.round(torch.clamp(tensor, iinfo.min, iinfo.max)).to(dtype)
    if cast_to_original_dtype:
        # 可选：再转回原浮点类型（模拟量化值）
        return rounded.to(original_dtype)
    return rounded


@torch.no_grad()
def quantize(
    x,
    scale,
    zero_point,
    dtype=torch.int8,
):
    # 由 scale 的宽度反推分组大小
    group_size = x.shape[-1] // scale.shape[-1]
    output_dtype = dtype
    output = torch.zeros_like(x).to(output_dtype)

    # 把最后一维拆成 (组数, 组内大小) 的二维视图
    reshaped_dims = (
        math.ceil(x.shape[-1] / group_size),
        group_size,
    )
    x = x.unflatten(-1, reshaped_dims)

    # 除以 scale 完成缩放
    scaled = x / scale.unsqueeze(-1)

    if zero_point is not None:
        # 有零点时加上零点偏移（非对称量化）
        zero_point = zero_point.unsqueeze(-1)
        scaled += zero_point.to(x.dtype)

    # clamp and round
    # 截断并四舍五入到目标量化类型
    output = round_to_quantized_type_dtype(tensor=scaled, dtype=dtype)

    # 把分组视图重新展平回原形状
    output = output.flatten(start_dim=-2)
    output = output.to(output_dtype)

    return output


def if_quant(name, patterns):
    # 判断参数名是否命中任一模式（决定是否量化）
    for pattern in patterns:
        if re.search(pattern, name):
            return True
    return False


def pack_layer(weight, group_size, sym=True):
    # 对单个权重层做 int4 伪量化并打包为 compressed-tensors 的存储格式
    # 调用 CUDA 伪量化核：返回量化索引、scale、zero-point
    w, scale, zp = fake_int4_quant_cuda.fake_int4_quant_cuda(weight, (1, group_size), sym)
    # 整理形状为 (out, 组数?, 组数, 组内大小) 的分组视图
    w = w.view(weight.shape[0], 1, weight.shape[1] // group_size, group_size)
    scale = scale.view(weight.shape[0], 1, weight.shape[1] // group_size, 1)
    zp = zp.view(weight.shape[0], 1, weight.shape[1] // group_size, 1)
    if sym:
        # 对称量化：用 scale 把量化值还原为近似浮点
        w = w * scale
    else:
        # 非对称量化：先减零点再乘 scale
        w = (w - zp) * scale
    # 恢复原始形状
    w = w.view(weight.shape)
    # scale 整理为 (out, 组数) 的二维张量
    scale = scale.view(weight.shape[0], -1).contiguous()
    if not sym:
        # 非对称：打包零点
        zp = zp.view(weight.shape[0], -1)
        # 转置成 (组数, out) 并转 int32 准备打包
        zeros = zp.t().contiguous().to(torch.float32)
        zeros = zeros.to(dtype=torch.int32, device=w.device)
        # 按 8 列一组reshape，准备把 8 个 4bit 压进一个 int32
        zeros = zeros.reshape(-1, zeros.shape[1] // 8, 8)
        # AWQ 风格的字节交错顺序
        new_order_map = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=zeros.device) * 4
        zeros = zeros << new_order_map
        # 位移后求和完成打包
        packed_zp = torch.sum(zeros, dim=-1).to(torch.int32)
    else:
        # 对称量化没有零点
        zp = None
        packed_zp = None

    # 用还原后的近似浮点权重重新量化为 int8/uint8（保证与 packed 权重一致）
    quantized_weight = quantize(
        x=w,
        scale=scale,
        zero_point=zp,
        dtype=torch.int8 if sym else torch.uint8,
    )
    # 把量化值按 4bit 打包进 int32
    packed_weight = pack_to_int32(quantized_weight, 4, sym=sym)
    return packed_weight, scale, packed_zp


def quantize_params_compressed_tensors(converted_named_params, quantization_config):
    # 读取量化配置：权重分组配置、分组大小、是否对称、忽略列表
    w_cfg = quantization_config["config_groups"]["group_0"]["weights"]
    group_size = w_cfg["group_size"]
    is_symmetric = w_cfg["symmetric"]
    ignore_rules = quantization_config.get("ignore", [])

    results = []

    for name, param in converted_named_params:
        # 判断该参数是否在忽略列表中：支持 "re:" 前缀的正则、全名匹配、前缀匹配
        is_ignored = any(
            (r.startswith("re:") and re.match(r[3:], name)) or r == name or name.startswith(r) for r in ignore_rules
        )

        # 被忽略、非权重、或维度不足二维的参数跳过量化
        if is_ignored or not name.endswith(".weight") or param.dim() < 2:
            results.append((name, param))
            continue

        # 执行 int4 量化并打包
        qw, s, zp = pack_layer(param, group_size, is_symmetric)
        # compressed-tensors 的命名约定：weight -> weight_packed / weight_scale / weight_shape
        qweight_name = name.replace(".weight", ".weight_packed")
        scale_name = name.replace(".weight", ".weight_scale")
        # 记录原始权重形状（反量化时需要还原形状）
        weight_shape = torch.tensor(param.shape, dtype=torch.int32, device="cuda")
        weight_shape_name = name.replace(".weight", ".weight_shape")
        if zp is not None:
            # 非对称量化额外输出零点
            zp_name = name.replace(".weight", ".weight_zero_point")
            results.append((zp_name, zp))
        # 依次加入打包权重、scale 与形状
        results.append((qweight_name, qw))
        results.append((scale_name, s))
        results.append((weight_shape_name, weight_shape))

    return results
