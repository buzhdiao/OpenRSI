# 导入工具函数：获取当前节点 IP 与连续空闲端口
from slime.utils.misc import get_current_node_ip, get_free_port


# Ray actor 基类：提供节点 IP/端口探测与 master 地址查询等通用能力
class RayActor:
    # 静态方法：获取本节点 IP 以及从 start_port 起连续 consecutive 个空闲端口
    @staticmethod
    def _get_current_node_ip_and_free_port(start_port=10000, consecutive=1):
        # 返回 (当前节点 IP, 空闲端口)；consecutive>1 时供多端口需求（如 NCCL 多组）
        return get_current_node_ip(), get_free_port(start_port=start_port, consecutive=consecutive)

    # 返回本 actor 作为 master 的地址与端口（rank 0 初始化后持有）
    def get_master_addr_and_port(self):
        # 直接返回成员变量 master_addr 与 master_port
        return self.master_addr, self.master_port
