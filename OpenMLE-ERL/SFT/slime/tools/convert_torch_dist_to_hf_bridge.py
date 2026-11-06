# 导入 argparse，用于解析命令行参数
import argparse
# 导入 os，用于目录存在性检查
import os

# 导入 Megatron Bridge 的模型加载/保存模块（后续对其打补丁）
import megatron.bridge.training.model_load_save as _model_load_save_module
# 导入 Megatron Bridge 的 AutoBridge，用于 HF<->Megatron 权重桥接
from megatron.bridge import AutoBridge


# Here we need to patch Megatron Bridge's `load_model_config`, since the checkpoint is saved
# by Megatron and lack of provider information.
# 全局 provider 覆盖表：供打补丁后的 load_model_config 查询
_provider_override = {}
# 保存原始的 load_model_config 实现，供补丁函数内部调用
_original_load_model_config = _model_load_save_module.load_model_config


# 补丁版 load_model_config：用 Bridge provider 替换信息缺失的模型配置
def _patched_load_model_config(checkpoint_path):
    # 先调用原始实现加载模型配置与参数
    model_cfg, mlm_args = _original_load_model_config(checkpoint_path)
    # 查询是否设置了 provider 覆盖
    provider = _provider_override.get("provider")
    # 存在覆盖时才考虑替换
    if provider is not None:
        # 导入 provider 混入类用于类型判断
        from megatron.bridge.models.model_provider import ModelProviderMixin

        # 仅当原始配置不是 provider 类型时替换（打印替换的类名）
        if not isinstance(model_cfg, ModelProviderMixin):
            print(f"[convert] Overriding MLM TransformerConfig with Bridge provider: " f"{type(provider).__name__}")
            # 返回 provider 作为模型配置
            return provider, mlm_args
    # 无覆盖或已是 provider 类型时返回原始配置
    return model_cfg, mlm_args


# 用补丁版本替换模块级 load_model_config（monkey patch）
_model_load_save_module.load_model_config = _patched_load_model_config


# 脚本入口
if __name__ == "__main__":
    # 创建带描述的参数解析器
    parser = argparse.ArgumentParser(
        description="Convert torch distributed checkpoint to HuggingFace format using Megatron Bridge"
    )
    # torch 分布式 checkpoint 输入目录（必填）
    parser.add_argument(
        "--input-dir", type=str, required=True, help="Path to the torch distributed checkpoint directory"
    )
    # HF 输出目录（必填）
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save the HuggingFace checkpoint")
    # 原始 HF 模型目录：提供配置信息（必填）
    parser.add_argument(
        "--origin-hf-dir",
        type=str,
        required=True,
        help="Path to the original HuggingFace model directory (for config)",
    )
    # 是否强制覆盖已存在的输出目录
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite the output directory if it exists."
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 输出目录已存在且未指定 --force 时报错
    if os.path.exists(args.output_dir) and not args.force:
        raise ValueError(f"Output directory {args.output_dir} already exists. Use --force to overwrite it.")

    # 打印配置加载来源
    print(f"Loading config from {args.origin_hf_dir}")
    # 从原始 HF 目录创建 bridge（读取模型配置）
    bridge = AutoBridge.from_hf_pretrained(args.origin_hf_dir, trust_remote_code=True)

    # Use Bridge's provider so the correct model class is created (e.g., Qwen3VLModel
    # instead of GPTModel). This is needed because MLM checkpoints lack run_config.yaml.
    # 使用 Bridge 的 provider 构建正确的模型类（如 Qwen3VLModel 而非 GPTModel）
    provider = bridge.to_megatron_provider(load_weights=False)
    # 登记到全局覆盖表，供补丁版 load_model_config 使用
    _provider_override["provider"] = provider
    # 打印使用的 provider 类型
    print(f"[convert] Using Bridge provider: {type(provider).__name__}")

    # 打印导出开始信息
    print(f"Exporting checkpoint from {args.input_dir} to {args.output_dir}")
    # 由 bridge 执行 checkpoint 导出（torch dist -> HF）
    bridge.export_ckpt(args.input_dir, args.output_dir)

    # 完成提示
    print("Done!")
