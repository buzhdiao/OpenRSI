def strip_param_name_prefix(name: str):
    # 去掉 Megatron 模型参数名上包装用的 "module." 前缀（可能有多层嵌套）
    prefix = "module."
    while name.startswith(prefix):
        name = name.removeprefix(prefix)
    return name
