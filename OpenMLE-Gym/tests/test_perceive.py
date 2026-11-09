# 中文说明:builder_core 感知节点(Perceive)及相关防护的回归测试:
# 1) 只有 save 工具的成功回执才算"文件清点完成",其他工具消息里出现 saved 字样必须降级;
# 2) 各类异常(LLM 无工具调用/LLM 抛异常/工具调用全被白名单过滤/超过调用上限)
#    都走"fail-open"退出:记错误并置 perceive=True 降级到 Describe,不卡死流程;
# 3) read_txt_md 截断预览契约(6000 字符上限、提示一次性读取);
# 4) fileinfo 为空时即使请求删 raw 也保守保留;
# 5) 解压防路径穿越("../outside.txt" 必须被拒)。
# 位置:OpenMLE-Gym/tests 下,依赖 builder 全套依赖,缺失则整类跳过。

# 延迟注解:类型标注运行期不求值
from __future__ import annotations

# importlib.util:探测 builder 依赖是否齐全
import importlib.util
# tempfile:用例隔离临时目录
import tempfile
# unittest:测试组织与条件跳过
import unittest
# zipfile:构造带路径穿越的恶意压缩包
import zipfile
# Path:路径构造
from pathlib import Path
# SimpleNamespace:替身对象(llm_provider_tools/structure)
from types import SimpleNamespace


# builder 感知链路必需的模块清单
REQUIRED_MODULES = (
    "langchain_core",
    "langgraph",
    "py7zr",
    "requests",
    "tqdm",
)
# 依赖齐全才运行本测试类
HAS_BUILDER_DEPS = all(importlib.util.find_spec(name) is not None for name in REQUIRED_MODULES)


@unittest.skipUnless(HAS_BUILDER_DEPS, "builder dependencies are required")
class PerceiveTests(unittest.TestCase):
    """Perceive 节点 fail-open 语义与解压/删除防护的测试集。"""

    def _node(self, root: Path):
        """绕过 __init__ 造一个最小 NodeExecutor(forge 目录 + 基础 todo 状态)。"""
        from builder_core.utils.nodes import NodeExecutor

        # 伪造 forge 目录与两个关键文件
        forge = root / "forge" / "task"
        forge.mkdir(parents=True)
        (forge / "rawtree.txt").write_text("raw/\n", encoding="utf-8")
        (forge / "fileinfo.txt").write_text("", encoding="utf-8")
        # 跳过 __init__ 的部分构造(只填本测试需要的字段)
        node = NodeExecutor.__new__(NodeExecutor)
        node.todo = {"name": "task", "perceive": False, "errors": []}
        node.max_tool_call = 40
        node.tools = ["list_directory_contents", "get_csv_summary", "read_txt_md", "save"]
        node.structure = SimpleNamespace(comp_id_dir=lambda _: forge)
        return node

    def test_only_successful_save_tool_message_completes_normally(self) -> None:
        """save 工具的成功回执应直接结束感知阶段且不产生错误。"""
        from langchain_core.messages import ToolMessage

        with tempfile.TemporaryDirectory() as directory:
            node = self._node(Path(directory))
            # 消息流末尾是 save 工具的成功 ToolMessage
            state = {
                "messages": [
                    ToolMessage(
                        content="Successfully saved to fileinfo.txt",
                        tool_call_id="save-1",
                        name="save",
                    )
                ]
            }
            node.Perceive(state)
            # 感知完成、无错误
            self.assertTrue(node.todo["perceive"])
            self.assertEqual(node.todo["errors"], [])

    def test_saved_substring_from_other_tool_degrades(self) -> None:
        """其他工具回执里出现 "saved" 字样不算成功:走 LLM 轮并降级(有错误记录)。"""
        from langchain_core.messages import AIMessage, ToolMessage

        with tempfile.TemporaryDirectory() as directory:
            node = self._node(Path(directory))
            # LLM 只回文本(无工具调用) -> fail-open 降级
            node.llm_provider_tools = SimpleNamespace(
                query=lambda _: AIMessage(content="done")
            )
            state = {
                "messages": [
                    ToolMessage(
                        content="The word saved appeared in a text file",
                        tool_call_id="read-1",
                        name="read_txt_md",
                    )
                ]
            }
            node.Perceive(state)
            # 降级退出:perceive=True 且 errors 非空
            self.assertTrue(node.todo["perceive"])
            self.assertTrue(node.todo["errors"])

    def test_no_tool_call_uses_fail_open_exit(self) -> None:
        """LLM 不带工具调用:fai-open 退出,错误信息含 "no tool calls"。"""
        from langchain_core.messages import AIMessage, HumanMessage

        with tempfile.TemporaryDirectory() as directory:
            node = self._node(Path(directory))
            node.llm_provider_tools = SimpleNamespace(
                query=lambda _: AIMessage(content="no tool")
            )
            node.Perceive({"messages": [HumanMessage(content="start")]})
            self.assertTrue(node.todo["perceive"])
            self.assertIn("no tool calls", node.todo["errors"][0])

    def test_llm_exception_uses_fail_open_exit(self) -> None:
        """LLM 查询抛异常:fail-open 退出,异常信息被记录。"""
        from langchain_core.messages import HumanMessage

        with tempfile.TemporaryDirectory() as directory:
            node = self._node(Path(directory))

            # 替身:查询即抛错
            def fail(_):
                raise RuntimeError("simulated LLM failure")

            node.llm_provider_tools = SimpleNamespace(query=fail)
            node.Perceive({"messages": [HumanMessage(content="start")]})
            self.assertTrue(node.todo["perceive"])
            self.assertIn("simulated LLM failure", node.todo["errors"][0])

    def test_filtered_tool_calls_use_fail_open_exit(self) -> None:
        """LLM 返回白名单外工具调用:被过滤后为空,fail-open 退出。"""
        from langchain_core.messages import AIMessage, HumanMessage

        with tempfile.TemporaryDirectory() as directory:
            node = self._node(Path(directory))
            # 构造一个越权工具调用
            response = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "outside_tool",
                        "args": {},
                        "id": "outside-1",
                        "type": "tool_call",
                    }
                ],
            )
            node.llm_provider_tools = SimpleNamespace(query=lambda _: response)
            node.Perceive({"messages": [HumanMessage(content="start")]})
            self.assertTrue(node.todo["perceive"])
            self.assertIn("no allowed tool calls", node.todo["errors"][0])

    def test_tool_call_limit_uses_fail_open_exit(self) -> None:
        """历史工具回执达到上限:判定死循环风险,fail-open 退出。"""
        from langchain_core.messages import ToolMessage

        with tempfile.TemporaryDirectory() as directory:
            node = self._node(Path(directory))
            # 造 max_tool_call 条历史工具回执
            messages = [
                ToolMessage(
                    content="read",
                    tool_call_id=f"read-{index}",
                    name="read_txt_md",
                )
                for index in range(node.max_tool_call)
            ]
            node.Perceive({"messages": messages})
            self.assertTrue(node.todo["perceive"])
            self.assertIn("exceeded maximum", node.todo["errors"][0])

    def test_truncated_text_preview_is_larger_and_explicitly_one_shot(self) -> None:
        """read_txt_md 契约:超长文本截断到 6000 字符并提示"不要重复读同一文件"。"""
        from builder_core.tools.tools import (
            configure_task_paths,
            read_txt_md,
        )
        from builder_core.utils.prompts import gen_perceiver

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # 写一个 7000 字符的长文本
            raw = root / "raw"
            raw.mkdir()
            text = raw / "long.txt"
            text.write_text("x" * 7000, encoding="utf-8")
            # 绑定任务路径后调用工具
            configure_task_paths(raw, root / "fileinfo.txt")
            result = read_txt_md.invoke({"file_path": "long.txt"})
        # 截断提示与一次性读取提示都在返回里
        self.assertIn("truncated to 6000 chars", result)
        self.assertIn("do not read the same path again", result)
        # 感知 prompt 也明确"截断预览即足够"
        self.assertIn("A truncated preview is sufficient", gen_perceiver)

    def test_empty_fileinfo_preserves_raw_on_delete_request(self) -> None:
        """删除请求下 fileinfo 为空:保守保留 raw 目录并记录错误(不丢原始数据)。"""
        from builder_core.utils.nodes import NodeExecutor
        from builder_core.utils.struct import Structure

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # 真实 Structure 布局 + 一个非空 raw
            structure = Structure(root, "batch")
            raw = structure.raw_dir("task")
            raw.mkdir(parents=True)
            (raw / "data.csv").write_text("x\n1\n", encoding="utf-8")
            # forge 目录与空 fileinfo(未完成清点)
            forge = structure.comp_id_dir("task")
            forge.mkdir(parents=True, exist_ok=True)
            (forge / "fileinfo.txt").write_text("", encoding="utf-8")
            # 手工构造完成态的 todo(delete_raw=True)
            node = NodeExecutor.__new__(NodeExecutor)
            node.structure = structure
            node.delete_raw = True
            node.todo = {
                "name": "task",
                "download": True,
                "copy": True,
                "web_info": True,
                "describe": True,
                "prepare": True,
                "metric": True,
                "success": False,
                "errors": [],
                "prepares": [],
            }
            node.Next({"messages": []})
            # raw 仍在、整体成功、最后一条错误说明保留原因
            self.assertTrue(raw.is_dir())
            self.assertTrue(node.todo["success"])
            self.assertIn("Raw data retained", node.todo["errors"][-1])

    def test_archive_path_traversal_is_rejected(self) -> None:
        """压缩包含 ../ 成员:解压必须报错,且不产生越界文件。"""
        from builder_core.utils.nodes import NodeExecutor

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # 构造带路径穿越成员的恶意 zip
            archive = root / "bad.zip"
            destination = root / "raw"
            destination.mkdir()
            with zipfile.ZipFile(archive, "w") as file:
                file.writestr("../outside.txt", "not allowed")
            node = NodeExecutor.__new__(NodeExecutor)
            # 解压应抛 RuntimeError(解压失败包装)
            with self.assertRaises(RuntimeError):
                node.extract(archive, destination, recursive=False)
            # 目标之外不得出现文件
            self.assertFalse((root / "outside.txt").exists())


if __name__ == "__main__":
    unittest.main()
