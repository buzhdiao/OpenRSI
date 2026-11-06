# 使类型标注支持延迟求值（兼容低版本 Python 的新式标注语法）
from __future__ import annotations

"""
String-based Radix Trie for efficient prefix matching and token caching.
Optimized for string prefixes with corresponding token IDs.
"""

# 引入 threading，提供可重入锁保证树操作的线程安全
import threading
# 引入 time，用于 LRU 访问时间戳（monotonic 时钟不受系统调时影响）
import time
# 引入 dataclass，用于定义匹配结果数据结构
from dataclasses import dataclass
# 引入 Any 类型标注
from typing import Any


# 前缀匹配结果的数据结构
@dataclass
class MatchResult:
    """Result of prefix matching operation."""

    # 匹配到的前缀字符串
    matched_prefix: str
    # 该前缀对应的 token id 序列
    token_ids: list[int]
    # 各 token 的对数概率
    logp: list[float]
    loss_mask: list[int]  # Added loss mask for model generation parts
    # 未匹配到的剩余字符串
    remaining_string: str
    # 匹配终止所在的树节点（后续插入的挂载点）
    last_node: StringTreeNode


# 字符串基数树（Radix Trie）节点：每个节点存储一段字符串片段及其 token 信息
class StringTreeNode:
    """Tree node for string-based radix trie."""

    # 类级节点 ID 计数器（自增分配）
    counter = 0

    # 节点初始化：建好结构、数据与统计字段
    def __init__(self, node_id: int | None = None):
        # Core tree structure
        # 用列表存储全部子节点（按插入序，不做排序）
        self.children: list[StringTreeNode] = []  # Use list to store children
        # 父节点引用（根节点为 None）
        self.parent: StringTreeNode | None = None

        # Node data
        # 本节点代表的字符串片段（非累计，仅为该段）
        self.string_key: str = ""  # The string fragment this node represents
        # 本节点（不含子节点）的 token id 序列
        self.token_ids: list[int] | None = None  # Token IDs for this node only (not cumulative)
        # 本节点 token 的对数概率序列
        self.logp: list[float] | None = None  # Log probabilities for this node's tokens
        # 本节点 token 的损失掩码（标记哪些 token 属于模型生成部分）
        self.loss_mask: list[int] | None = None  # Loss mask for model generation parts

        # Access tracking
        # 最近一次访问时间（LRU 依据）
        self.last_access_time = time.monotonic()
        # 累计访问次数
        self.access_count = 0

        # Reference counting for protection from eviction
        # 引用计数：大于 0 的节点不会被逐出（正在使用的保护机制）
        self.ref_count = 0

        # Weight version tracking
        # 记录该节点所属的权重版本（用于按版本做缓存失效 GC）
        self.weight_version: int | None = None  # Weight version for this node

        # Node identification
        # 分配节点 ID（允许外部指定，否则用类计数器自增）
        self.id = StringTreeNode.counter if node_id is None else node_id
        StringTreeNode.counter += 1

    # 是否为叶子节点（没有子节点）
    @property
    def is_leaf(self) -> bool:
        """Check if this node is a leaf node."""
        return len(self.children) == 0

    # 是否已存储 token 数据
    @property
    def has_value(self) -> bool:
        """Check if this node has token IDs stored."""
        return self.token_ids is not None

    # 校验 token_ids/logp/loss_mask 三者长度一致性
    def validate_token_logp_consistency(self) -> bool:
        """Validate that token_ids, logp, and loss_mask have consistent lengths."""
        # 三者都为空视为一致
        if self.token_ids is None and self.logp is None and self.loss_mask is None:
            return True

        # Check if at least one is not None
        # 有 token 数据时以 token 长度为基准
        if self.token_ids is not None and len(self.token_ids) > 0:
            token_len = len(self.token_ids)
            # logp 存在但长度不符则不一致
            if self.logp is not None and len(self.logp) != token_len:
                return False
            # loss_mask 存在但长度不符则不一致
            if self.loss_mask is not None and len(self.loss_mask) != token_len:
                return False

        # 通过校验
        return True

    # 是否可被逐出：无引用且已持有 token 数据
    @property
    def is_evictable(self) -> bool:
        """Check if this node can be evicted."""
        return self.ref_count == 0 and self.token_ids is not None

    # 更新访问时间与访问计数（LRU 触碰）
    def touch(self):
        """Update access time and count."""
        self.last_access_time = time.monotonic()
        self.access_count += 1

    # 小于比较：按最近访问时间排序（越久未用越靠前），供堆做 LRU
    def __lt__(self, other: StringTreeNode) -> bool:
        """For heap operations - least recently used first."""
        return self.last_access_time < other.last_access_time


# 字符串基数树主类：前缀匹配 + token 缓存 + 按权重版本的垃圾回收
class StringRadixTrie:
    """
    String-based Radix Trie for efficient prefix matching and token caching.
    Features:
    - Efficient string prefix matching
    - Token ID caching for matched prefixes
    - Thread-safe operations
    - Weight version tracking
    - Automatic garbage collection based on weight version thresholds
    """

    # 初始化：设定缓存上限、GC 阈值与可选 tokenizer
    def __init__(self, max_cache_size: int = 10000, gc_threshold_k: int = 5, tokenizer=None, verbose: bool = False):
        """
        Initialize the String Radix Trie.
        Args:
            max_cache_size: Maximum number of cached token IDs (triggers GC when exceeded)
            gc_threshold_k: GC threshold - nodes with weight_version < (current_version - k) will be removed
            tokenizer: Optional tokenizer for converting text to tokens when not found in cache
            verbose: Whether to print debug information and tree structure
        """
        # 缓存 token 总量上限（超出触发 GC）
        self.max_cache_size = max_cache_size
        # GC 阈值 k：权重版本落后当前版本超过 k 的节点将被清除
        self.gc_threshold_k = gc_threshold_k
        # 可选 tokenizer（缓存未命中时用它给文本分词）
        self.tokenizer = tokenizer
        # 调试开关
        self.verbose = verbose

        # Tree structure
        # 创建根节点（空串），始终受保护
        self.root = StringTreeNode()
        # 根节点的 key 为空串
        self.root.string_key = ""
        self.root.ref_count = 1  # Root is always protected

        # Cache statistics
        # 累计插入的节点条目数
        self.total_entries = 0
        # 缓存命中次数
        self.cache_hits = 0
        # 缓存未命中次数
        self.cache_misses = 0
        self.cur_cache_size = 0  # Total number of token IDs across all nodes

        # Thread safety
        # 可重入锁保护所有树操作
        self._lock = threading.RLock()

    # 在树中查找给定文本的最长已缓存前缀（论文中 Radix Cache / 前缀缓存的核心查找）
    def find_longest_prefix(self, text: str) -> MatchResult:
        """
        Find the longest cached prefix for the given text.
        Args:
            text: Input string to find prefix for
        Returns:
            MatchResult containing matched prefix, token IDs, logp, and remaining string
        """
        # 全程持锁，保证线程安全
        with self._lock:
            # 空文本：返回空匹配（终止于根节点）
            if not text:
                return MatchResult("", [], [], [], text, self.root)

            # 累计匹配到的 token id
            matched_tokens = []
            # 累计匹配到的 logp
            matched_logp = []
            # 累计匹配到的 loss mask
            matched_loss_mask = []
            # 累计匹配到的前缀文本
            matched_prefix = ""
            # 当前所在节点（从根开始）
            current_node = self.root
            # 尚未匹配的剩余文本
            remaining_text = text

            # 逐层向下贪心匹配，直到无法继续
            while remaining_text:
                # Find the best matching child that completely matches from start
                # 本轮最佳子节点
                best_child = None
                # 本轮最佳匹配长度
                best_key_len = 0

                # 遍历当前节点的所有子节点
                for child_node in current_node.children:
                    # Only consider complete startswith matches using node's string_key
                    # 只接受"剩余文本以子节点 key 完整开头"的匹配（基数树要求整段匹配）
                    if remaining_text.startswith(child_node.string_key):
                        if len(child_node.string_key) > best_key_len:
                            best_child = child_node
                            best_key_len = len(child_node.string_key)

                # 找不到任何可整段匹配的子节点：匹配结束
                if best_child is None:
                    # No complete startswith match found
                    break

                # Move to the best matching child
                # 触碰该节点（更新 LRU 信息）
                best_child.touch()
                # 下移到最佳子节点
                current_node = best_child
                # 把该段文本并入已匹配前缀
                matched_prefix += best_child.string_key
                # 从剩余文本中裁掉该段
                remaining_text = remaining_text[best_key_len:]

                # Accumulate tokens, logp, and loss_mask from this node
                # 该节点持有 token 数据时才累计
                if best_child.has_value:
                    matched_tokens.extend(best_child.token_ids)
                    matched_logp.extend(best_child.logp)
                    if best_child.loss_mask is not None:
                        matched_loss_mask.extend(best_child.loss_mask)
                    else:
                        # If no loss_mask is stored, create default mask same as logp
                        # 未存 loss mask 时按全 1 兜底（等长于 token 数）
                        matched_loss_mask.extend([1] * len(best_child.token_ids))
                    self.cache_hits += 1

            # 一个 token 都没匹配到则记一次未命中
            if not matched_tokens:
                self.cache_misses += 1

            # 组装匹配结果
            result = MatchResult(
                matched_prefix, matched_tokens, matched_logp, matched_loss_mask, remaining_text, current_node
            )

            # Print tree structure if verbose is enabled
            # 调试模式下打印匹配后的树结构
            if self.verbose:
                print("Tree structure after find_longest_prefix:")
                self.pretty_print()

            # 返回匹配结果
            return result

    # 插入接口：校验参数后把文本/token/logp/loss_mask 写入树，并按需触发 GC
    def insert(
        self,
        text: str,
        token_ids: list[int],
        logp: list[float] | None = None,
        loss_mask: list[int] | None = None,
        weight_version: int | None = None,
    ) -> bool:
        """
        Insert a string and its corresponding token IDs, log probabilities, and loss mask into the trie.
        Args:
            text: String to insert
            token_ids: Corresponding token IDs
            logp: Corresponding log probabilities (must match token_ids length)
            loss_mask: Corresponding loss mask for model generation parts (must match token_ids length)
            weight_version: Optional weight version for this insertion
        Returns:
            True if insertion was successful
        """
        # 全程持锁
        with self._lock:
            # 空文本或空 token 序列：拒绝插入
            if not text or not token_ids:
                if self.verbose:
                    print("[RadixTree] Insertion failed: text or token_ids is empty")
                return False

            # Use provided weight version
            # 沿用调用方给的权重版本
            current_weight_version = weight_version

            # Validate logp consistency
            # 校验 logp 与 token 等长
            if logp is not None and len(logp) != len(token_ids):
                if self.verbose:
                    print(
                        f"[WARNING] Logp length {len(logp)} does not match token length {len(token_ids)} for text: {text}"
                    )
                    print(f"[WARNING] Logp: {logp}")
                    print(f"[WARNING] Token IDs: {token_ids}")
                return False

            # Validate loss_mask consistency
            # 校验 loss_mask 与 token 等长
            if loss_mask is not None and len(loss_mask) != len(token_ids):
                if self.verbose:
                    print(
                        f"[WARNING] Loss mask length {len(loss_mask)} does not match token length {len(token_ids)} for text: {text}"
                    )
                    print(f"[WARNING] Loss mask: {loss_mask}")
                    print(f"[WARNING] Token IDs: {token_ids}")
                return False

            # If logp is not provided, create default values (0.0)
            # 未提供 logp 时用全 0 兜底
            if logp is None:
                logp = [0.0] * len(token_ids)

            # If loss_mask is not provided, create default values (1 for model generation parts)
            # 未提供 loss_mask 时默认全 0（非模型生成部分）
            if loss_mask is None:
                loss_mask = [0] * len(token_ids)

            # 调用内部插入实现
            result = self._insert(text, token_ids, logp, loss_mask, current_weight_version)

            # Check if GC should be triggered after insert
            # 缓存超限且带了权重版本时触发按版本的垃圾回收
            if self.cur_cache_size > self.max_cache_size and weight_version is not None:
                if self.verbose:
                    print(
                        f"[RadixTree] Cache size {self.cur_cache_size} exceeds limit {self.max_cache_size}, triggering GC"
                    )
                gc_removed = self.gc_by_weight_version(weight_version)
                if self.verbose:
                    print(f"[RadixTree] GC removed {gc_removed} nodes, new cache size: {self.cur_cache_size}")

            # Print tree structure if verbose is enabled
            # 调试模式下打印插入后的树结构
            if self.verbose:
                print("Tree structure after insert:")
                self.pretty_print()

            # 返回插入是否成功
            return result

    # 内部插入：沿树下沉跳过已存在的文本段（token 同步跳过），为剩余文本建新节点
    def _insert(
        self,
        text: str,
        token_ids: list[int],
        logp: list[float],
        loss_mask: list[int],
        weight_version: int | None = None,
    ) -> bool:
        """Insert tokens - skip tokens for existing nodes just like we skip text."""

        # 从根节点开始
        current_node = self.root
        # 剩余待插入文本
        remaining_text = text
        remaining_tokens = token_ids[:]  # Copy the tokens list
        remaining_logp = logp[:]  # Copy the logp list
        remaining_loss_mask = loss_mask[:]  # Copy the loss_mask list

        # Track all nodes traversed during insert for weight version update
        # 记录下沉路径上的所有节点
        traversed_nodes = [current_node]
        # 本次新建的节点
        new_node = None

        # 循环消费剩余文本
        while remaining_text:
            # Find best startswith match
            # 最佳子节点
            best_child = None
            # 最佳匹配长度
            best_key_len = 0

            # 找出能整段匹配且最长的子节点
            for child_node in current_node.children:
                if remaining_text.startswith(child_node.string_key) and len(child_node.string_key) > best_key_len:
                    best_child = child_node
                    best_key_len = len(child_node.string_key)

            # 命中已有节点：跳过其文本与对应 token
            if best_child is not None:
                # Found existing node - skip its text and tokens
                # 下移
                current_node = best_child
                traversed_nodes.append(current_node)
                # 裁掉已覆盖文本
                remaining_text = remaining_text[best_key_len:]

                # Skip the tokens that this existing node covers
                # 已有节点持有 token 时按其 token 数同步裁剪剩余序列
                if best_child.has_value:
                    tokens_to_skip = len(best_child.token_ids)
                    remaining_tokens = remaining_tokens[tokens_to_skip:]
                    remaining_logp = remaining_logp[tokens_to_skip:]
                    remaining_loss_mask = remaining_loss_mask[tokens_to_skip:]
            # 无匹配：为剩余文本创建新节点
            else:
                # Create new node for remaining text with remaining tokens
                new_node = StringTreeNode()
                # 挂到当前节点下
                new_node.parent = current_node
                # 剩余文本整段作为新节点的 key
                new_node.string_key = remaining_text

                if remaining_tokens:  # Only assign if there are tokens left
                    # 还有 token 剩余才写入数据
                    new_node.token_ids = remaining_tokens
                    new_node.logp = remaining_logp
                    new_node.loss_mask = remaining_loss_mask
                    new_node.touch()
                    # Increment cache size by number of tokens added
                    # 缓存总量增加相应 token 数
                    self.cur_cache_size += len(remaining_tokens)

                # 把新节点加入子节点列表
                current_node.children.append(new_node)
                traversed_nodes.append(new_node)
                # 条目计数加一
                self.total_entries += 1
                # 插入完成
                break

        # If we've traversed the entire text and the last node doesn't have tokens,
        # assign remaining tokens to it
        # 文本全部走完但终止节点还没有 token 数据时：把剩余 token 写到该节点
        if remaining_text == "" and not current_node.has_value:
            if remaining_tokens:  # Only assign if there are tokens left
                current_node.token_ids = remaining_tokens
                current_node.logp = remaining_logp
                current_node.loss_mask = remaining_loss_mask
                current_node.touch()
                self.cur_cache_size += len(remaining_tokens)

        # Update weight version for all traversed nodes
        # 新建节点记录权重版本
        if weight_version is not None and new_node:
            new_node.weight_version = weight_version

        # 插入总是成功（入参已校验）
        return True

    # 删除指定文本对应的节点及其整个子树
    def remove(self, text: str) -> bool:
        """
        Remove a string and all nodes with this text as prefix from the trie.
        Args:
            text: String to remove (will also remove all strings starting with this text)
        Returns:
            True if any removal was performed
        """
        # 全程持锁
        with self._lock:
            # 按精确文本定位节点
            node = self._find_node_by_text(text)
            # 找到节点则清理其子树
            if node:
                removed_count = self._clean_node_subtree(node)

                # Print tree structure if verbose is enabled
                # 调试模式下打印删除后的树结构
                if self.verbose:
                    print("Tree structure after remove:")
                    self.pretty_print()

                # 至少删除了一个节点才算成功
                return removed_count > 0
            # 未找到节点
            return False

    # 按精确文本查找节点（必须整串匹配，而非仅前缀）
    def _find_node_by_text(self, text: str) -> StringTreeNode | None:
        """
        Find node by exact text match.
        Args:
            text: Text to find
        Returns:
            Node if found, None otherwise
        """
        # 复用最长前缀查找
        result = self.find_longest_prefix(text)
        # 只有当匹配前缀与目标文本完全相等时才算找到
        if result.matched_prefix == text:
            return result.last_node
        # 否则返回 None
        return None

    # 清理节点及其全部后代（根节点除外）
    def _clean_node_subtree(self, node: StringTreeNode) -> int:
        """
        Clean a node and all its descendants.
        This is the core cleanup function.
        Args:
            node: Node to clean (including all descendants)
        Returns:
            Number of nodes removed
        """
        # 不允许删除根节点
        if node == self.root:
            return 0
        return self._remove_node_and_descendants(node)

    # 递归删除节点及其所有后代，返回删除的数据节点数
    def _remove_node_and_descendants(self, node: StringTreeNode) -> int:
        """
        Remove a node and all its descendants from the trie.
        Args:
            node: The node to remove along with all its descendants
        Returns:
            Number of nodes removed
        """
        if node == self.root:
            # Never remove root node
            # 根节点永远不删
            return 0

        # 已删除节点计数
        removed_count = 0

        # First, recursively remove all descendants
        # 先递归删除所有子节点（拷贝列表避免迭代中修改）
        for child in list(node.children):  # Create a copy to avoid modification during iteration
            removed_count += self._remove_node_and_descendants(child)

        # Count this node if it has data and decrement cache size
        # 本节点持有数据时计数加一并扣减缓存总量
        if node.has_value:
            removed_count += 1
            # Decrement cache size by number of tokens removed
            self.cur_cache_size -= len(node.token_ids)

        # Remove this node from its parent
        # 把本节点从父节点摘除
        if self._remove_node_from_parent(node):
            # Update count for the node structure itself
            pass  # _remove_node_from_parent already decrements total_entries

        # 返回删除的数据节点数
        return removed_count

    # 把节点从其父节点的 children 列表中移除并维护条目计数
    def _remove_node_from_parent(self, node: StringTreeNode) -> bool:
        """Remove a node from its parent's children list."""
        # 有父节点且确实在 children 中才删除
        if node.parent and node in node.parent.children:
            node.parent.children.remove(node)
            self.total_entries -= 1
            return True
        # 摘除失败
        return False

    # 按权重版本执行垃圾回收：清掉版本落后超过 k 的整棵子树
    def gc_by_weight_version(self, current_weight_version: int | None = None) -> int:
        """
        Perform garbage collection based on weight version.
        Remove nodes with weight_version < (current_weight_version - gc_threshold_k).
        Args:
            current_weight_version: Current weight version to use for GC threshold
        Returns:
            Number of nodes removed
        """
        # 全程持锁
        with self._lock:
            # 未提供版本号则跳过 GC
            if current_weight_version is None:
                if self.verbose:
                    print("[RadixTree GC] No weight version provided, skipping GC")
                return 0

            # 计算 GC 阈值：当前版本 - k
            gc_threshold = current_weight_version - self.gc_threshold_k
            if self.verbose:
                print(
                    f"[RadixTree GC] Starting GC with threshold: {gc_threshold} (current_version: {current_weight_version}, k: {self.gc_threshold_k})"
                )

            # 找出所有过期待删节点
            nodes_to_remove = self._find_outdated_nodes(gc_threshold)
            # 已删除节点数
            removed_count = 0

            # 逐个清理过期的子树
            for node in nodes_to_remove:
                # Validate that subtree weight versions are <= parent weight version
                # 先校验子树版本号关系的一致性
                self._validate_subtree_weight_versions(node)
                removed_count += self._clean_node_subtree(node)

            if self.verbose:
                print(f"[RadixTree GC] Completed GC, removed {removed_count} nodes")

            # 返回删除节点数
            return removed_count

    # 找出权重版本低于阈值的数据节点（父节点已过期则整棵子树不重复检查）
    def _find_outdated_nodes(self, gc_threshold: int) -> list[StringTreeNode]:
        """
        Find nodes that should be removed based on weight version threshold.
        Uses layer-by-layer traversal - if parent is outdated, children are not checked.
        Args:
            gc_threshold: Weight version threshold (nodes < this value will be removed)
        Returns:
            List of nodes to remove
        """
        # 待删除节点列表
        outdated_nodes = []

        # 深度优先检查的内部递归函数
        def check_node(node):
            # 根节点永远不删，只检查其子节点
            if node == self.root:
                # Root is never removed, check its children
                for child in node.children:
                    check_node(child)
                return

            # Check if this node should be removed
            # 版本落后于阈值且持有数据：记为待删
            if node.weight_version is not None and node.weight_version <= gc_threshold and node.has_value:
                outdated_nodes.append(node)
                return  # Don't check children since entire subtree will be removed

            # Node is not outdated, check its children
            # 未过期则继续检查子节点
            for child in node.children:
                check_node(child)

        # 从根开始全树检查
        check_node(self.root)
        # 返回待删节点列表
        return outdated_nodes

    # 校验子树内所有节点的权重版本不超过父节点版本（数据不变式检查）
    def _validate_subtree_weight_versions(self, node: StringTreeNode):
        """
        Validate that all nodes in subtree have weight_version <= parent weight_version.
        Args:
            node: Root node of subtree to validate
        """

        # 递归校验内部函数
        def validate_recursive(current_node, parent_weight_version):
            # 双方版本号都存在时才比较
            if current_node.weight_version is not None and parent_weight_version is not None:
                assert current_node.weight_version <= parent_weight_version, (
                    f"Child node weight_version {current_node.weight_version} > "
                    f"parent weight_version {parent_weight_version}"
                )

            # Recursively validate children
            # 继续校验子节点（以当前节点版本为父版本）
            for child in current_node.children:
                validate_recursive(child, current_node.weight_version)

        # Start validation from the node itself
        # 从指定节点开始校验
        validate_recursive(node, node.weight_version)

    # 返回缓存统计信息（条目数、命中率等）
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        # 持锁读取统计
        with self._lock:
            # 总请求数 = 命中 + 未命中
            total_requests = self.cache_hits + self.cache_misses
            # 命中率（无请求时为 0）
            hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

            # 汇总统计字典
            return {
                "total_entries": self.total_entries,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate": hit_rate,
                "max_cache_size": self.max_cache_size,
                "cur_cache_size": self.cur_cache_size,
                "gc_threshold_k": self.gc_threshold_k,
            }

    # 清空整棵树并复位统计
    def clear(self):
        """Clear all entries from the trie."""
        # 持锁执行
        with self._lock:
            # 用全新的根节点替换整棵树
            self.root = StringTreeNode()
            self.root.string_key = ""
            self.root.ref_count = 1
            # 各项统计归零
            self.total_entries = 0
            self.cache_hits = 0
            self.cache_misses = 0
            self.cur_cache_size = 0

    # 以可读格式打印整棵树及统计
    def pretty_print(self):
        """Print the trie structure in a readable format."""
        print("String Radix Trie Structure:")
        print("=" * 50)
        self._print_node(self.root, 0)
        print("=" * 50)
        stats = self.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")

    # 递归打印单个节点及其子树
    def _print_node(self, node: StringTreeNode, depth: int):
        """Recursively print node structure."""
        # 按深度生成缩进
        indent = "  " * depth
        # 节点 key 的展示形式（根节点显示 <root>）
        key_repr = repr(node.string_key) if node.string_key else "<root>"
        # token 信息展示串
        token_info = ""
        if node.has_value:
            token_info = f" -> tokens: {node.token_ids}"
            if node.logp:
                token_info += f", logp: {[round(p, 3) for p in node.logp]}"
            if node.loss_mask:
                token_info += f", loss_mask: {node.loss_mask}"
        access_info = f" (accessed: {node.access_count}, ref: {node.ref_count})"

        print(f"{indent}{key_repr}{token_info}{access_info}")

        for child in node.children:
            self._print_node(child, depth + 1)

    # 从文本取 token：优先查 radix tree 缓存，未命中部分用 tokenizer 兜底
    def retrieve_from_text(self, text: str, return_logprob: bool = True):
        """
        Get tokens from text by looking up in radix tree or using tokenizer.
        Also fetches weight version from worker during this operation.
        Args:
            text: Input text to get tokens for
            return_logprob: If True, also return log probabilities
        Returns:
            List of token IDs corresponding to the input text if return_logprob is False.
            Tuple of (token_ids, logp) if return_logprob is True.
        """
        # Call find_longest_prefix to get the match result
        # 先做最长前缀匹配
        result = self.find_longest_prefix(text)

        # If we have a match and it covers the entire text, return the tokens
        # 有匹配且缓存了 token 时：剩余文本用 tokenizer 分词补齐
        if result.matched_prefix and result.token_ids:
            additional_tokens = self.tokenizer(result.remaining_string, add_special_tokens=False)["input_ids"]
            return (
                result.token_ids + additional_tokens,
                (
                    result.logp + len(additional_tokens) * [0.0]
                    if return_logprob
                    else [0] * len(result.token_ids + additional_tokens)
                ),
                result.loss_mask + len(additional_tokens) * [0],
            )
        # If result is empty and input text is not empty, tokenize with tokenizer
        # This is needed because we cannot get the prompt token id from engine response
        # We have to manually insert the text and token into the tree
        # 完全未命中且 tokenizer 可用时：整体分词并手动插入树中建立缓存
        if self.tokenizer and text:
            # Tokenize the text using the provided tokenizer
            tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            # Insert the text and tokens into the tree
            self.insert(text, tokens)
            # Return the tokens
            return (tokens, [0.0] * len(tokens), [0] * len(tokens))
        else:
            raise ValueError("Tokenizer or input text can't be empty")


# Example usage and testing
# 独立运行本模块时的自测试代码
if __name__ == "__main__":
    # Create trie instance for testing
    # 创建测试用 trie 实例（小缓存上限 + 调试输出）
    trie = StringRadixTrie(max_cache_size=100, verbose=True)

    # Example usage with simplified insert
    # 三组测试数据：(文本, token 序列, logp 序列)
    test_cases = [
        ("Hello world", [1, 2, 3], [-0.1, -0.2, -0.3]),
        ("Hello", [1, 2], [-0.1, -0.2]),
        ("Hi there", [4, 5, 6], [-0.4, -0.5, -0.6]),
    ]

    # Insert test data with weight version and loss masks
    print("Inserting test data...")
    for text, tokens, logp in test_cases:
        # Create loss_mask to match tokens length, 1 for model generation parts
        # 构造与 token 等长的全 1 loss mask
        loss_mask = [1] * len(tokens)
        success = trie.insert(text, tokens, logp, loss_mask, weight_version=1)
        print(f"Inserted '{text}' -> {tokens}: {success}")

    print("\nTrie structure:")
    trie.pretty_print()

    # Test prefix matching
    print("\nTesting prefix matching:")
    # 各类边界查询用例
    test_queries = [
        "Hello world!",  # Should match "Hello world" completely
        "Hello everyone",  # Should match "Hello" only
        "Hi there",  # Should match "Hi" only
        "How are you doing?",  # Should match "How are you" completely
        "Goodbye",  # Should not match anything
        "Hell",  # Should not match anything (not complete startswith)
    ]

    # 逐个查询并打印匹配细节
    for query in test_queries:
        result = trie.find_longest_prefix(query)
        print(f"Query: '{query}'")
        print(
            f"  Matched: '{result.matched_prefix}' -> tokens: {result.token_ids}, logp: {result.logp}, loss_mask: {result.loss_mask}"
        )
        print(f"  Remaining: '{result.remaining_string}'")
        print()

    # Test removal
    print("Testing removal:")
    removed = trie.remove("Hello")
    print(f"Removed 'Hello': {removed}")

    result = trie.find_longest_prefix("Hello world")
    print(
        f"After removal - 'Hello world' -> matched: '{result.matched_prefix}', tokens: {result.token_ids}, logp: {result.logp}, loss_mask: {result.loss_mask}"
    )

    # Show final stats
    print("\nFinal statistics:")
    stats = trie.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Test GC with weight version
    print("\nTesting GC with weight version 5:")
    gc_removed = trie.gc_by_weight_version(5)
    print(f"GC removed {gc_removed} nodes")
