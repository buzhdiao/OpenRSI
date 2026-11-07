# 让类型注解延迟求值，从而支持新版联合类型等注解语法
from __future__ import annotations

# os 模块用于读写进程环境变量
import os

# 需要清理的代理相关环境变量名（大小写两种形式都覆盖，防止代理劫持沙箱/LLM 请求）
PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def disable_proxy_env() -> None:
    # 逐个删除代理环境变量；第二个参数 None 表示变量不存在时不抛 KeyError
    for name in PROXY_ENV_VARS:
        os.environ.pop(name, None)
