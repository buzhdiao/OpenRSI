# 导入上下文管理器装饰器：用 with 语句实现"临时修改-恢复"语义
from contextlib import contextmanager

# 尝试导入 Megatron-Core 的模型解包工具（去掉 DDP/ wrappers 拿到内层模型）
try:
    from megatron.core.utils import unwrap_model
except ImportError:
    # 未安装 Megatron 时置为 None，调用方自行处理
    unwrap_model = None


@contextmanager
def patch_megatron_model(model):
    # 解开 Megatron 的外层封装（DDP/Float16Module 等），拿到真实的内层模型
    unwrapped_model = unwrap_model(model)[0]
    # 取内层模型的 TransformerConfig 配置对象
    model_config = unwrapped_model.config
    # 标记属性是否由本函数临时添加（用于退出时恢复现场）
    attribute_was_added = False
    # 旧版本 Megatron 的 config 上可能没有该属性，但权重建模需要它
    if not hasattr(model_config, "share_embeddings_and_output_weights"):
        # 从模型上把"输入输出词嵌入共享"标志补写到 config 上
        model_config.share_embeddings_and_output_weights = unwrapped_model.share_embeddings_and_output_weights
        # 记录：该属性是本函数临时加上的
        attribute_was_added = True

    try:
        # yield 前临时补丁已生效，把控制权交给 with 语句体
        yield
    finally:
        # 无论 with 体是否抛异常，都要恢复原状
        if attribute_was_added:
            # 只有是本函数添加的属性才删除，避免误删用户配置
            delattr(model_config, "share_embeddings_and_output_weights")
