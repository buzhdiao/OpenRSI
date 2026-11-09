# 中文说明:NodeExecutor——OpenMLE-Gym 构建器(builder)的"节点执行器",
# 负责把一个 Kaggle 竞赛任务从"下载 -> 解压 -> 情报抓取 -> 文件感知 -> 描述 ->
# 数据准备 -> 指标准备"编排成一串状态机节点(Download/Copy/Scrape/Perceive/
# Describe/Prepare/Metric),各节点返回消息增量,Router* 按 todo 状态决定
# 下一步走"成功/重试/失败"分支。
# 在系统中的位置:builder_core/utils 下,是任务构建管线的核心;产物是结构化的
# public/private 数据目录与 prepare.py、metric.py 等工具脚本。
# 关键设计:
# - 代码执行支持 in-process 与 isolated(子进程隔离)两种模式;
# - 解压带路径逃逸与软链接成员校验(防 zip 炸弹/越权写);
# - Perceive 阶段限制工具调用次数上限(防 LLM 死循环)。

# sys:动态模块注册表(sys.modules)与标准错误输出
import sys
# os:读取 Kaggle 凭据等环境变量
import os
# re:从 LLM 输出提取 markdown 代码围栏
import re
# shutil:目录/文件的复制与删除(copy2/rmtree)
import shutil
# json:读写 webinfo.json、prepares.json
import json
# Path:路径对象化
from pathlib import Path
# 类型注解
from typing import List, Optional, Dict, Any, Set, Tuple

# pandas:读任务信息 CSV、预览提交/答案样例
import pandas as pd
# py7zr:解压 7z 归档
import py7zr
# requests:流式下载 Kaggle 竞赛数据
import requests
# zipfile:解压 zip 归档(含软链接成员校验)
import zipfile
# importlib.util:按文件路径动态加载 prepare.py / metric.py
import importlib.util
# tqdm:进度条展示工具
from tqdm.auto import tqdm
# load_dotenv:加载 .env(Kaggle/LLM 密钥等)
from dotenv import load_dotenv
# HTTPBasicAuth:Kaggle API 的 Basic 认证
from requests.auth import HTTPBasicAuth
# atomic_write_text:临时文件+替换的原子写(避免半写文件)
from openmle_gym.common import atomic_write_text
# run_task_process:isolated 模式下以子进程执行 prepare 脚本
from openmle_gym.process_runner import run_task_process

# langchain 消息类型:状态机中逐节点累积的消息流
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
# Task:任务待办字典类型(todo)
from .task import Task
# AgentState:图状态(消息列表载体)
from .state import AgentState
# Structure:批次/竞赛目录布局的唯一事实来源
from .struct import Structure
# OpenAILLMProvider:OpenAI 兼容 LLM 客户端(可选带工具)
from .chat import OpenAILLMProvider
# 各阶段 prompt 模板:感知/描述/数据信息/准备脚本/指标脚本
from .prompts import (
    gen_perceiver,
    gen_description, 
    description_info, 
    gen_prepare_script, 
    gen_metric_script, 
    metric_info,
)


# ============================================================================
# 主执行器类 NodeExecutor
# ============================================================================

class NodeExecutor:
    """编排 Kaggle 竞赛数据的下载、处理与准备(构建流水线的主执行器)。"""
    
    # 支持的压缩包扩展名
    COMPRESSED_EXTENSIONS = {".zip", ".7z"}
    # Kaggle 官方"下载全部数据"接口模板
    KAGGLE_DATA_URL = "https://www.kaggle.com/api/v1/competitions/data/download-all/{comp_id}"
    
    def __init__(
        self,
        base_dir: str,
        batch_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        todo: Task = None,
        tools: List[Any] = None,
        info_csv: Optional[Path] = None,
        sample_dir: Optional[Path] = None,
        delete_raw: bool = False,
        code_execution_mode: str = "process",
        code_timeout: float = 600,
    ) -> None:
        """初始化 NodeExecutor。
        
        Args:
            base_dir: 数据存储根目录
            batch_name: 批次(处理会话)名称
            web_timeout: Web 操作超时(秒)
            api_key: OpenAI API 密钥
            base_url: API 调用地址
            model: LLM 模型名
            tools: LLM 工具列表
        """
        # 加载 .env 环境变量
        load_dotenv()
        
        # 目录布局助手(批次/竞赛各目录的拼装)
        self.structure = Structure(base_dir, batch_name)
        # 目录树文本缓存(由 _generate_tree 递归填充)
        self.tree = ""
        # 任务信息 CSV(竞赛概览与数据描述来源)
        self.info_csv = Path(info_csv) if info_csv is not None else Path("info.csv")
        # 样例目录(描述/准备/指标脚本的 few-shot 样例)
        self.sample_dir = Path(sample_dir) if sample_dir is not None else Path(base_dir) / "samples"
        # 成功后是否删除原始下载数据(留 fileinfo 说明)
        self.delete_raw = delete_raw
        # 代码执行模式:process(进程内)或 isolated(子进程隔离)
        self.code_execution_mode = code_execution_mode
        # 隔离执行的超时(秒)
        self.code_timeout = code_timeout
        # 预读任务信息 CSV 供 Scrape 阶段查询
        self.csvinfo = pd.read_csv(self.info_csv)
        
        # 无工具时置空;否则把工具对象规范成"名字"列表(白名单)
        if tools is None:
            self.tools = []
        else:
            self.tools = [
                getattr(tool, "name", getattr(tool, "__name__", str(tool)))
                for tool in tools
            ]
        # 感知阶段的工具调用次数上限(防死循环)
        self.max_tool_call = 40

        # 普通对话 LLM 客户端(不带工具)
        self.llm_provider = OpenAILLMProvider(
            api_key=api_key,
            base_url=base_url,
            model=model
        )
            
        # 带工具的 LLM 客户端(Perceive 阶段用)
        self.llm_provider_tools = OpenAILLMProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            tools=tools
        )
        
        # 当前任务的待办状态字典(各阶段完成标记/错误/retry 计数)
        self.todo = todo
            
    # ========================================================================
    # 核心数据处理方法
    # ========================================================================
    
    def _is_compressed(self, file_path: Path) -> bool:
        """按扩展名判断文件是否为压缩包(仅认 .zip/.7z)。"""
        # 后缀(忽略大小写)是否在支持集合里
        return file_path.suffix.lower() in self.COMPRESSED_EXTENSIONS

    def _load_kaggle_auth(self) -> HTTPBasicAuth:
        """加载 Kaggle 认证:优先环境变量,回退 KAGGLE_CONFIG_DIR 或 ~/.kaggle/kaggle.json。"""
        # 先取环境变量中的用户名/密钥
        username = os.environ.get("KAGGLE_USERNAME")
        key = os.environ.get("KAGGLE_KEY")
        # 两者齐全则直接构造 Basic 认证
        if username and key:
            return HTTPBasicAuth(username, key)

        # 收集候选配置文件路径:KAGGLE_CONFIG_DIR 优先,再退到 ~/.kaggle
        config_paths = []
        kaggle_config_dir = os.environ.get("KAGGLE_CONFIG_DIR")
        if kaggle_config_dir:
            config_paths.append(Path(kaggle_config_dir).expanduser() / "kaggle.json")
        config_paths.append(Path("~/.kaggle/kaggle.json").expanduser())

        # 逐个尝试读取凭据文件
        for config_path in config_paths:
            # 不存在则看下一个
            if not config_path.exists():
                continue
            try:
                # 解析 JSON 凭据
                data = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception as exc:
                # 文件损坏/不可读:显式报错(避免静默失败)
                raise RuntimeError(f"Failed to read Kaggle credentials from {config_path}: {exc}") from exc
            # 取出用户名与密钥
            username = data.get("username")
            key = data.get("key")
            # 齐全则返回认证对象
            if username and key:
                return HTTPBasicAuth(username, key)

        # 环境变量与配置文件都没有:明确报错
        raise RuntimeError(
            "Kaggle credentials not found. Set KAGGLE_USERNAME/KAGGLE_KEY "
            "or KAGGLE_CONFIG_DIR containing kaggle.json."
        )
    
    def download(self, comp_id: str, target_dir: Path) -> Dict[str, Any]:
        """通过 Kaggle API 流式下载竞赛数据(失败返回错误信息而不是抛异常)。"""
        try:
            # 建目标目录(允许已存在)
            target_dir.mkdir(parents=True, exist_ok=True)
            # 目标压缩包路径
            target_file = target_dir / f"{comp_id}.zip"
            
            # 打印下载开始
            print(f"Downloading competition: {comp_id}")
            
            # 流式请求"下载全部数据"(连接超时 120 秒)
            response = requests.get(
                self.KAGGLE_DATA_URL.format(comp_id=comp_id),
                auth=self._load_kaggle_auth(),
                stream=True,
                timeout=120,
            )
            # 非 200:截取响应前 1000 字符作为错误上下文
            if response.status_code != 200:
                preview = response.text[:1000].replace("\n", " ")
                return {
                    "success": False,
                    "error": f"Kaggle API HTTP {response.status_code}: {preview}",
                }

            # 按 1MB 分块写盘(避免大文件占满内存)
            with target_file.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    # 跳过空块
                    if chunk:
                        file.write(chunk)

            # 文件缺失或 0 字节:视为下载失败
            if not target_file.exists() or target_file.stat().st_size == 0:
                return {
                    "success": False,
                    "error": f"Downloaded empty file: {target_file}",
                }
            
            # 打印完成并返回成功结果与文件路径
            print(f"Download completed: {comp_id}")
            return {"success": True, "path": target_file}
            
        except Exception as e:
            # 任何异常都收敛为失败结果(单任务下载失败不中断整批)
            return {
                "success": False, 
                "error": f"Unexpected error: {e}"
            }
    
    def extract(
        self, 
        compressed_path: Path, 
        dst: Path, 
        recursive: bool = True, 
        already_extracted: Optional[Set[Path]] = None
    ) -> None:
        """安全解压压缩包到 dst,支持嵌套归档递归解压(带防逃逸/防软链接校验)。
        
        Args:
            compressed_path: 压缩包路径
            dst: 解压目标目录
            recursive: 是否递归解压嵌套归档
            already_extracted: 已处理集合(防循环)
            
        Raises:
            Exception: 解压失败时
        """
        # 首次调用时初始化已处理集合
        if already_extracted is None:
            already_extracted = set()

        # 已处理过或文件不存在:直接返回(防死循环)
        if compressed_path in already_extracted or not compressed_path.exists():
            return

        # 打印解压对象
        print(f"Extracting: {compressed_path.name}")
        
        try:
            # 目标目录取真实路径(供逃逸校验)
            destination = dst.resolve()

            # 成员路径校验:解压后必须仍位于目标目录内(防 ../ 逃逸)
            def validate_member(name: str) -> None:
                member_path = (destination / name).resolve()
                try:
                    member_path.relative_to(destination)
                except ValueError as exc:
                    raise ValueError(f"Archive member escapes destination: {name}") from exc

            # 按文件类型分别解压
            if compressed_path.suffix == ".7z":
                # 7z:先校验全部成员名再整体解压
                with py7zr.SevenZipFile(compressed_path, mode="r") as ref:
                    for name in ref.getnames():
                        validate_member(name)
                    ref.extractall(dst)
            elif compressed_path.suffix == ".zip":
                # zip:逐成员校验路径逃逸 + 拒绝符号链接成员(防指向任意文件)
                with zipfile.ZipFile(compressed_path, "r") as ref:
                    for member in ref.infolist():
                        validate_member(member.filename)
                        # 取外部属性高 16 位中的 Unix 权限位
                        unix_mode = member.external_attr >> 16
                        # 文件类型位为 S_IFLNK(软链接)则拒绝
                        if (unix_mode & 0o170000) == 0o120000:
                            raise ValueError(f"Archive symlink is not allowed: {member.filename}")
                    ref.extractall(dst)
            else:
                # 不认识的格式:提示后返回(不抛错)
                print(f"Unsupported compression format: {compressed_path.suffix}")
                return
            
            # 登记已处理并删除原压缩包(解压即释放空间)
            already_extracted.add(compressed_path)
            compressed_path.unlink()

            # 递归解压嵌套归档
            if recursive:
                # 找出解压产物中的新压缩包
                new_compressed_files = [
                    f for f in dst.iterdir() 
                    if f.is_file() and self._is_compressed(f) and f.exists()
                ]
                
                # 未处理过的继续递归解压
                for file_path in new_compressed_files:
                    if file_path not in already_extracted:
                        self.extract(
                            file_path, 
                            dst, 
                            recursive=True, 
                            already_extracted=already_extracted
                        )

        except Exception as e:
            # 统一包装为 RuntimeError(带上出错的压缩包路径)
            raise RuntimeError(f"Failed to extract {compressed_path}: {e}") from e
    
    def _generate_tree(
            self, 
            start_path: Path, 
            max_depth: int = 6, 
            current_depth: int = 0, 
            prefix: str = ''
        ) -> None:
        """递归生成 ASCII 目录树写入 self.tree(目录优先、名称排序、超 30 项折叠)。"""
        # 超过最大深度:停止下钻
        if current_depth > max_depth:
            return
        
        # 根层:先写根目录行
        if current_depth == 0:
            self.tree = f"{start_path}/\n"
        
        # 非目录(文件):不展开
        if not start_path.is_dir():
            return

        try:
            # 列出并排序:目录在前,名称不区分大小写
            items = sorted(
                list(start_path.iterdir()), 
                key=lambda x: (not x.is_dir(), x.name.lower())
            )
        except PermissionError:
            # 无权限目录:静默跳过
            return

        # 条目过多:折叠成一行摘要(避免树文本爆炸)
        if len(items) > 30:
            ext = items[0].suffix if items[0].suffix else "items"
            self.tree += f"{prefix}`-- *There are {len(items)} {ext}*\n"
            return

        # 逐项绘制树形字符
        for i, item in enumerate(items):
            # 是否本层最后一项(决定连接符)
            is_last = (i == len(items) - 1)
            connector = '`-- ' if is_last else '|-- '
            
            # 目录名带 / 后缀便于区分
            item_type = "/" if item.is_dir() else ""
            self.tree += f"{prefix}{connector}{item.name}{item_type}\n"
            
            # 子目录且未到深度上限:递归展开(子前缀按是否末位缩进)
            if item.is_dir() and current_depth < max_depth:
                new_prefix = prefix + ('    ' if is_last else '|   ')
                self._generate_tree(
                    item, 
                    max_depth, 
                    current_depth + 1, 
                    new_prefix
                )
    
    def get_directory_tree(self, start_path: Path, max_depth: int = 6) -> str:
        """获取目录的树状文本表示。
        
        Args:
            start_path: 起始目录
            max_depth: 最大展示深度
            
        Returns:
            目录树字符串
        """
        # 清空缓存后递归生成
        self.tree = ""
        self._generate_tree(start_path, max_depth)
        return self.tree
    
    # ========================================================================
    # 文件加载与处理方法
    # ========================================================================
    
    def _load_sample_file(self, filename: str) -> str:
        """加载样例文件内容(缺失返回空串,不阻断流程)。
        
        Args:
            filename: 样例文件名
            
        Returns:
            文件内容或空字符串
        """
        # 样例完整路径
        sample_path = self.sample_dir / filename
        # 存在才读
        if sample_path.exists():
            with open(sample_path, 'r', encoding='utf-8') as f:
                return f.read()
        # 缺失返回空串(调用方以"无样例"继续)
        return ""
    
    def _clean_code_output(self, message: str) -> str:
        """解析 LLM 输出:拆出"思考文本"与"代码"(支持 <code> 标签或 ``` 围栏两种格式)。
        
        Args:
            code: LLM 原始输出
            
        Returns:
            (思考文本, 代码) 二元组
            
        Raises:
            Exception: 两种格式都不完整时
        """
        # 去首尾空白
        message = message.strip()
        # 优先按 <code>...</code> 标签拆(取最后一个标签块)
        if "<code>" in message and "</code>" in message:
            parts = message.rsplit("<code>", 1)
            # 标签前是思考文本
            think = parts[0].strip()
            # 标签内是代码
            code = parts[1].split("</code>", 1)[0].strip()
            return think, code

        # 回退:提取 ```python 围栏代码块
        fenced_blocks = re.findall(
            r"```(?:python|py)?\s*\n(.*?)```",
            message,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fenced_blocks:
            # 取最后一个代码块
            code = fenced_blocks[-1].strip()
            # 围栏之前的部分作为思考文本
            think = message.rsplit("```", 2)[0].strip()
            return think, code

        # 两种格式都缺失:抛错让上层要求模型重写
        raise Exception(
            "Parsing failed: The returned text does not contain "
            "complete wrapping tags <code> and </code> or a Python code fence. "
            "Please regenerate the code with strict adherence to the format."
        )

    # ========================================================================
    # 模块加载与准备方法
    # ========================================================================
    
    def _load_module(self, module_name: str, file_path: Path):
        """从文件动态加载 Python 模块(加载失败返回 None 并清理注册表)。
        
        Args:
            module_name: 赋予模块的名字
            file_path: Python 文件路径
            
        Returns:
            加载的模块,失败返回 None
        """
        # 文件不存在:直接失败
        if not file_path.exists():
            return None
        
        try:
            # 由路径构造加载 spec
            spec = importlib.util.spec_from_file_location(
                module_name, 
                str(file_path)
            )
            # spec/loader 缺失则放弃
            if spec is None or spec.loader is None:
                return None
            
            # 创建模块对象并注册到 sys.modules(供模块内 import 自省)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            # 执行模块顶层代码
            spec.loader.exec_module(module)
            return module
            
        except Exception as e:
            # 打印加载错误
            print(f"Error loading module {module_name}: {e}")
            # 失败时从注册表摘除半初始化模块(避免污染后续加载)
            if module_name in sys.modules:
                del sys.modules[module_name]
            return None
    
    def get_prepare_function(self, comp_id: str):
        """从 prepare.py 中取出可调用的 prepare 函数(找不到返回 None)。
        
        Args:
            comp_id: 竞赛 ID
            
        Returns:
            prepare 函数或 None
        """
        # prepare.py 路径
        prepare_path = self.structure.comp_id_dir(comp_id) / "prepare.py"
        # 动态加载
        module = self._load_module(f"prepare_{comp_id}", prepare_path)
        
        # 模块加载成功才继续找函数
        if module:
            prepare_func = getattr(module, 'prepare', None)
            # 必须是可调用对象
            if callable(prepare_func):
                return prepare_func
        
        # 找不到返回 None
        return None
    
    def prepare_validation(self, comp_id: str) -> List[str]:
        """校验准备产物必需文件是否齐全。
        
        Args:
            comp_id: 竞赛 ID
            
        Returns:
            缺失文件名列表
        """
        # 必需文件清单:公开侧 4 个 + 私有答案 1 个
        files = [
            self.structure.public_dir(comp_id) / "description.txt",
            self.structure.public_dir(comp_id) / "train.csv",
            self.structure.public_dir(comp_id) / "test.csv",
            self.structure.public_dir(comp_id) / "sample_submission.csv",
            self.structure.private_dir(comp_id) / "test_answer.csv"
        ]

        # 逐个检查,收集缺失文件名
        missing = []
        for file in files:
            if not file.is_file():
                missing.append(file.name)
        return missing
    
    def prepare_single_competition(self, comp_id: str) -> Tuple[bool, Optional[str]]:
        """为单个竞赛执行数据准备(清空重建目录 -> 跑 prepare -> 校验 -> 归档脚本)。
        
        Args:
            comp_id: 竞赛 ID
            
        Returns:
            (成功标志, 错误信息) 元组
        """
        
        try:
            # 打印开始
            print(f"Preparing competition: {comp_id}")
            
            # 清空并重建 data/utils 目录(保证幂等重跑)
            for directory in [
                self.structure.data_dir(comp_id), 
                self.structure.utils_dir(comp_id)
            ]:
                shutil.rmtree(directory, ignore_errors=True)
            
            # 重建四类目录:数据/公开/私有/工具
            self.structure.data_dir(comp_id).mkdir(parents=True, exist_ok=True)
            self.structure.public_dir(comp_id).mkdir(exist_ok=True)
            self.structure.private_dir(comp_id).mkdir(exist_ok=True)
            self.structure.utils_dir(comp_id).mkdir(exist_ok=True)
            
            # prepare 脚本路径
            prepare_path = self.structure.comp_id_dir(comp_id) / "prepare.py"
            if self.code_execution_mode == "isolated":
                # 隔离模式:在受限子进程里跑(只读源数据,仅可写 public/private)
                outcome = run_task_process(
                    "prepare",
                    {
                        # 脚本路径
                        "prepare_path": str(prepare_path),
                        # 原始数据目录
                        "raw_dir": str(self.structure.raw_dir(comp_id)),
                        # 公开输出目录
                        "public_dir": str(self.structure.public_dir(comp_id)),
                        # 私有输出目录
                        "private_dir": str(self.structure.private_dir(comp_id)),
                    },
                    # 超时
                    timeout=self.code_timeout,
                    # 隔离执行
                    execution_mode="isolated",
                    # 只读路径:脚本与原始数据
                    readonly_paths=(prepare_path, self.structure.raw_dir(comp_id)),
                    # 可写路径:public 与 private
                    writable_paths=(
                        self.structure.public_dir(comp_id),
                        self.structure.private_dir(comp_id),
                    ),
                )
                # 回放子进程输出
                if outcome.stdout:
                    print(outcome.stdout, end="")
                if outcome.stderr:
                    print(outcome.stderr, end="", file=sys.stderr)
                # 失败则抛错
                if not outcome.ok:
                    raise RuntimeError(
                        outcome.error or "Isolated prepare worker returned an invalid result"
                    )
            else:
                # 进程内模式:直接调用 prepare(raw, public, private)
                prepare_func = self.get_prepare_function(comp_id)
                if prepare_func is None:
                    raise ValueError("No prepare function found, code inexecutable")
                prepare_func(
                    self.structure.raw_dir(comp_id),
                    self.structure.public_dir(comp_id),
                    self.structure.private_dir(comp_id)
                )
            
            # 校验准备产物
            missing = self.prepare_validation(comp_id)
            if missing:
                # 有缺失:报错并提示检查命名
                raise ValueError(
                    f"Prepare script executed successfully, but failed to "
                    f"pass the file validation: {missing} MISSING. Check your NAMING."
                )
            
            # 把 prepare.py 归档进 utils 目录(供后续复现)
            prepare_source = self.structure.comp_id_dir(comp_id) / "prepare.py"
            if prepare_source.exists():
                shutil.copy2(
                    prepare_source, 
                    self.structure.utils_dir(comp_id) / "prepare.py"
                )
            
            # 成功返回
            print(f"Successfully prepared {comp_id}")
            return True, None
                    
        except Exception as e:
            # 任何失败都收敛为 (False, 错误信息)
            error_msg = str(e)
            return False, error_msg
    
    def prepare_implement(self, comp_id: str) -> None:
        """prepare_single_competition 的包装:失败时清理目录并抛异常。
        
        Args:
            comp_id: 竞赛 ID
            
        Raises:
            Exception: 准备失败时
        """
        # 执行准备
        success_flag, error_msg = self.prepare_single_competition(comp_id)
        
        # 成功直接返回
        if success_flag:
            return
        else:
            # 失败:清理该竞赛的半成品目录
            self._cleanup_failed_competition(comp_id)

            # 再抛出错误信息
            raise Exception(error_msg)
    
    def _cleanup_failed_competition(self, comp_id: str) -> None:
        """清理失败竞赛的目录并重建空目录(保证批次内其他任务不受残留影响)。
        
        Args:
            comp_id: 竞赛 ID
        """
        # 删除 data/utils 目录
        for directory in [
            self.structure.data_dir(comp_id), 
            self.structure.utils_dir(comp_id)
        ]:
            shutil.rmtree(directory, ignore_errors=True)
        
        # 重建空目录
        self.structure.data_dir(comp_id).mkdir(exist_ok=True)
        self.structure.public_dir(comp_id).mkdir(exist_ok=True)
        self.structure.private_dir(comp_id).mkdir(exist_ok=True)
        self.structure.utils_dir(comp_id).mkdir(exist_ok=True)
            
    # ========================================================================
    # 智能体工作流集成节点
    # ========================================================================
    
    def Download(self, state: AgentState) -> AgentState:
        """下载单个竞赛数据(状态机节点:Phase 1)。
        
        Args:
            state: 当前智能体状态
            
        Returns:
            消息增量字典
        """
        # 当前任务名即竞赛 ID
        comp_id = self.todo["name"]
        # 竞赛目录(提前建好)
        comp_dir = self.structure.comp_id_dir(comp_id)
        comp_dir.mkdir(parents=True, exist_ok=True)
        
        # 打印阶段横幅
        print(f"\n{'='*60}")
        print(f"[PROCESSING COMPETITION: {comp_id}")
        print(f"{'='*60}")
        print("\nPhase 1: Downloading competition...")
        # 消息增量
        update = []
        
        # 下载目标目录
        target_dir = self.structure.DOWNLOAD_DIR / comp_id
        # 执行下载
        result = self.download(comp_id, target_dir)
        
        if result["success"]:
            # 成功:登记下载完成标记
            self.todo["download"] = True
            update += [HumanMessage(content="[OK] Download completed")]
            print(f"[OK] Download completed for {comp_id}")
        else:
            # 失败:记录错误并写入消息流
            error_msg = result.get('error', 'Unknown error')
            self.todo["errors"].append(
                f"Download dataset failed: {error_msg}"
            )
            update += [HumanMessage(content=f"[ERROR] Download failed: {error_msg}")]
            print(f"[ERROR] Download failed for {comp_id}: {error_msg}")
        
        # 返回消息增量
        return {"messages": update}
        # return {"messages": []}
    
    def Copy(self, state: AgentState) -> AgentState:
        """复制并解压数据到原始目录(状态机节点:Phase 2)。
        
        Args:
            state: 当前智能体状态
            
        Returns:
            消息增量字典
        """
        # 阶段横幅
        print("\nPhase 2: Copying and extracting dataset...")
        # 任务名
        comp_id = self.todo["name"]
        # 消息增量
        update = []
        try:
            # 打印处理对象
            print(f"Processing competition: {comp_id}")
            
            # 源 zip(下载阶段产物)
            source_zip = self.structure.DOWNLOAD_DIR / comp_id / f"{comp_id}.zip"
            if not source_zip.exists():
                raise Exception(f"Source zip not found: {source_zip}")
            
            # 创建全部必需目录
            directories = [
                self.structure.comp_id_dir(comp_id),
                self.structure.data_comp_id_dir(comp_id),
                self.structure.raw_dir(comp_id),
                self.structure.data_dir(comp_id),
                self.structure.utils_dir(comp_id),
                self.structure.public_dir(comp_id),
                self.structure.private_dir(comp_id),
                # self.structure.evo_dir(comp_id)
            ]
            
            # 逐个创建(parents 允许多级)
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                
            # 复制 zip 到 raw 目录并在原地解压
            target_zip = self.structure.raw_dir(comp_id) / f"{comp_id}.zip"
            shutil.copy2(source_zip, target_zip)
            self.extract(target_zip, self.structure.raw_dir(comp_id))

            # 解压完成后删除下载暂存目录
            shutil.rmtree(source_zip.parent, ignore_errors=True)
            
            # 生成原始目录树并写 rawtree.txt(供感知/描述阶段使用)
            tree_info = self.get_directory_tree(
                self.structure.raw_dir(comp_id)
            )
            tree_file = self.structure.comp_id_dir(comp_id) / "rawtree.txt"
            with open(tree_file, 'w', encoding='utf-8') as file:
                file.write(tree_info)
                
            # 建空的文件信息文件(Perceive 阶段的写入目标)
            file_file = self.structure.comp_id_dir(comp_id) / "fileinfo.txt"
            with open(file_file, 'w', encoding='utf-8') as file:
                file.write("")
            
            # 登记完成标记与消息
            self.todo["copy"] = True
            update += [HumanMessage(content="[OK] Dataset copied and extracted")]
            print(f"[OK] Dataset copied and extracted for {comp_id}")
            
        except Exception as e:
            # 失败也要留 rawtree.txt(标记为错误文本,保证下游文件存在)
            tree_file = self.structure.comp_id_dir(comp_id) / "rawtree.txt"
            with open(tree_file, 'w', encoding='utf-8') as file:
                file.write("Error copying files.")
            
            # 记录错误
            self.todo["errors"].append(f"Copy dataset failed: {e}")
            update += [HumanMessage(content=f"[ERROR] Copy dataset failed: {e}")]
            print(f"[ERROR] Copy dataset failed for {comp_id}: {e}")
        
        # 返回消息增量
        return {"messages": update}

    def Scrape(self, state: AgentState) -> AgentState:
        """从任务信息 CSV 抓取竞赛概览与数据描述,落盘 webinfo.json(状态机节点:Phase 3)。"""
        # 阶段横幅
        print("\nPhase 3: Fetching web information...")
        # 任务名
        comp_id = self.todo["name"]
        # 消息增量
        update = []
        # 抓取到的信息
        info = {}

        # 打印抓取对象
        print(f"Fetching information for: {comp_id}")

        try:
            # 按竞赛 Slug 过滤信息表
            match = self.csvinfo[self.csvinfo["Slug"] == comp_id]

            # 命中则取概览与数据描述(缺失/NaN 不写入)
            if not match.empty:
                overview = match["Overview"].iloc[0]
                dataset = match["DatasetDescription"].iloc[0]
                if not pd.isna(overview):
                    info["overview"] = str(overview)
                if not pd.isna(dataset):
                    info["dataset"] = str(dataset)

            # 两项都缺失:视为失败(下游描述阶段没有素材)
            if info.get("overview") is None and info.get("dataset") is None:
                print(
                    f"[ERROR] CSV information fetching failed for {comp_id}: "
                    "overview and dataset-information Info missing"
                )
                update += [HumanMessage(content=
                                "[ERROR] CSV information fetching failed: "
                                "overview and dataset-information Info missing")]
                # 直接返回失败消息
                return {"messages": update}

            # 写 webinfo.json(缩进 4,保留非 ASCII)
            info_file = self.structure.comp_id_dir(comp_id) / "webinfo.json"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=4, ensure_ascii=False)

            # 登记完成标记与成功消息
            self.todo["web_info"] = True
            update += [HumanMessage(content="[OK] CSV information fetched")]
            print(f"[OK] CSV information fetched for {comp_id}")

        except Exception as e:
            # 异常记入消息流(不设完成标记)
            update += [HumanMessage(content=f"[ERROR] CSV information fetching failed: {e}")]
            print(f"[ERROR] CSV information fetching failed for {comp_id}: {e}")

        # 返回消息增量
        return {"messages": update}
    
    
    def Perceive(self, state: AgentState) -> AgentState:
        """执行可选的文件清点(工具循环);todo["perceive"]=True 表示循环已终止。

        清点失败只记录错误并主动降级到 Describe 阶段(不阻断构建)。
        """
        # 阶段横幅
        print("\nPhase 4: Perceiving dataset information...")
        # 任务名
        comp_id = self.todo["name"]
        # 消息增量
        update = []
        
        try:
            # 终止判定:上一条消息是 save 工具的成功回执
            if state.get("messages") and isinstance(state["messages"][-1], ToolMessage):
                last_tool = state["messages"][-1]
                if (
                    getattr(last_tool, "name", None) == "save"
                    and last_tool.content.strip() == "Successfully saved to fileinfo.txt"
                ):
                    # 清点完成:登记标记并退出
                    self.todo["perceive"] = True
                    update += [HumanMessage(content="[OK] File info digested")]
                    print(f"[OK] File info digested for {comp_id}")
                    return {"messages": update}
                
            # 收集历史工具回执内容,检查是否超过调用上限
            tool_messages = [tool_info.content for tool_info in state["messages"] if isinstance(tool_info, ToolMessage)]
            if len(tool_messages) >= self.max_tool_call:
                # 超限视为不安全死循环,主动终止
                raise Exception(f"Tool call times exceeded maximum({self.max_tool_call}), which indicates unsafe deadloop")
            
            # 读取原始目录树
            tree_file = self.structure.comp_id_dir(comp_id) / "rawtree.txt"
            with open(tree_file, 'r', encoding='utf-8') as file:
                tree_info = file.read()
            
            # 给出 fileinfo.txt 的绝对路径(LLM 的 save 工具写入目标)
            info_path = self.structure.comp_id_dir(comp_id) / "fileinfo.txt"
            # 构造感知轮次的系统/用户消息
            query = [
                SystemMessage(content=gen_perceiver),
                HumanMessage(content=f"Now perform inventory of this directory. FILE TREE:\n{tree_info}\n'fileinfo.txt' PATH:'{info_path}'"),
            ]
            # 系统与用户消息入增量
            update += query
            
            # 带工具上下文请求 LLM(历史工具回执夹在中间)
            response = self.llm_provider_tools.query(
                [query[0]] +
                tool_messages + 
                [query[1]]
            )
            
            # 回复入增量
            update += [response]
            
            if hasattr(response, "tool_calls") and response.tool_calls:
                # 只保留白名单内的工具调用(丢弃越权调用)
                response.tool_calls = [tc for tc in response.tool_calls if tc["name"] in self.tools]
                tool_use = [tc["name"] for tc in response.tool_calls]
                print(f"USING TOOLS: {tool_use}")
                if not response.tool_calls:
                    # 全部被过滤:报错终止
                    raise RuntimeError("LLM returned no allowed tool calls")
            else:
                # 没有任何工具调用:感知轮无法继续,报错降级
                raise RuntimeError("LLM returned no tool calls")
            
        except Exception as e:
            # 失败:记录错误,并强制 perceive=True 降级到 Describe 阶段
            update += [HumanMessage(content=f"[ERROR] File info digestion failed halfway: {e}")]
            self.todo["errors"].append(
                f"File info digestion failed: {e}"
            )
            self.todo["perceive"] = True
            print(f"[ERROR] File info digestion for {comp_id} failed halfway: {e}")
            
        # 返回消息增量
        return {"messages": update}
        
        
    def Describe(self, state: AgentState) -> AgentState:
        """用 LLM 汇总生成任务描述 description.txt(状态机节点:Phase 5)。
        
        Args:
            state: 当前智能体状态
            
        Returns:
            消息增量字典
        """
        # 阶段横幅
        print("\nPhase 5: Generating description...")

        # 任务名
        comp_id = self.todo["name"]
        # 消息增量
        update = []
        
        try:
            # 读取 webinfo 与目录树
            web_file = self.structure.comp_id_dir(comp_id) / "webinfo.json"
            with open(web_file, 'r', encoding='utf-8') as file:
                web_info = json.load(file)
            tree_file = self.structure.comp_id_dir(comp_id) / "rawtree.txt"
            with open(tree_file, 'r', encoding='utf-8') as file:
                tree_info = file.read()
            
            # 读取感知阶段的文件内容清点
            file_file = self.structure.comp_id_dir(comp_id) / "fileinfo.txt"
            with open(file_file, 'r', encoding='utf-8') as file:
                file_info = file.read()
            
            # 加载描述样例(few-shot)
            sample_description = self._load_sample_file("sample_description.txt")
            
            # 构造 prompt
            # infos = description_info.format(
            #     description=web_info.get('description', ''),
            #     evaluation=web_info.get('evaluation', ''),
            #     submission=web_info.get('submission-instructions', ''),
            #     dataset_description=web_info.get('dataset-description', ''),
            #     tree_info=tree_info,
            #     file_info=file_info
            # )
            # 当前版本只用 overview/dataset 两个字段
            infos = description_info.format(
                description=web_info.get('overview', ''),
                dataset_description=web_info.get('dataset', ''),
                tree_info=tree_info,
                file_info=file_info
            )
            
            # 构造系统+用户消息
            query = [
                SystemMessage(content=gen_description.format(sample_description=sample_description)), 
                HumanMessage(content=f"Now perform detailed summarization on these TEXTUAL MATERIALS:\n{infos}")
            ]
            # 入增量
            update += query
            
            # 无工具 LLM 生成描述
            response = self.llm_provider.query(query)
            
            # 写 description.txt
            desc_file = self.structure.comp_id_dir(comp_id) / "description.txt"
            with open(desc_file, 'w', encoding='utf-8') as file:
                file.write(response.content)
            
            # 登记完成标记与消息
            self.todo["describe"] = True
            update += [response, HumanMessage(content="[OK] Description generated")]
            print(f"[OK] Description generated for {comp_id}")
                
        except Exception as e:
            # 失败记录错误(不设完成标记,Router 会走失败分支)
            update += [HumanMessage(content=f"[ERROR] Description generation failed: {e}")]
            self.todo["errors"].append(
                f"Description generation failed: {e}"
            )
            print(f"[ERROR] Description generation failed for {comp_id}: {e}")

        # 返回消息增量
        return {"messages": update}
    
    def Prepare(self, state: AgentState) -> AgentState:
        """生成并执行数据准备脚本 prepare.py(状态机节点:Phase 6,含重试机制)。
        
        Args:
            state: 当前智能体状态
            
        Returns:
            消息增量字典
        """
        # 阶段横幅
        print("\nPhase 6: Generating prepare script and Preparing...")
        # 任务名
        comp_id = self.todo["name"]
        # 消息增量
        update = []
        
        try:
            # 读取描述文件
            desc_file = self.structure.comp_id_dir(comp_id) / "description.txt"
            with open(desc_file, 'r', encoding='utf-8') as f:
                description = f.read()
            
            # 加载准备脚本样例(few-shot)
            prepare_sample = self._load_sample_file("sample_prepare.py")
            
            # 构造 prompt:系统样例 + 用户(描述+格式要求)
            query = [
                SystemMessage(content=gen_prepare_script.format(prepare_sample=prepare_sample)), 
                HumanMessage(content=
                    f"Now THINK and WRITE the CODE based on the DESCRIPTION:\n{description}\nDO NOT FORGET to wrap the CODE part in <code> and </code>!"
                )
            ]
            # 入增量
            update += query
            
            # 请求 LLM;附最近两轮 prepares 上下文(供自我修正)
            response = self.llm_provider.query(
                [query[0]] + self.todo["prepares"][-2:] + [query[1]]
            )
            
            # 登记生成完成
            update += [response, HumanMessage(content="[OK] Prepare script generated")]
            self.todo["prepares"] += [query[1]]
            self.todo["gen_prep"] = True
            print(f"[OK] Prepare script generated for {comp_id}")
            
        except Exception as e:
            # 生成失败:retry 置 -1(不再重试),记录错误并返回
            update += [HumanMessage(content=f"[ERROR] Prepare script generation failed: {e}")]
            self.todo["retry"] = -1
            self.todo["errors"].append(
                f"Prepare script generation failed: {e}"
            )
            print(f"[ERROR] Prepare script generation failed for {comp_id}: {e}")
            return {"messages": update}
        
        # 思考/代码占位(失败时用于回填原始输出)
        think = "VOID"
        prepare_code = "VOID"
        
        try:
            # 解析输出并取出代码
            think, prepare_code = self._clean_code_output(response.content)
            # prepare.py 路径
            prepare_file = self.structure.comp_id_dir(comp_id) / "prepare.py"
            # 先编译检查语法(不合法直接失败,不落盘)
            compile(prepare_code, str(prepare_file), "exec")
            # 原子写盘
            atomic_write_text(prepare_file, prepare_code)
            
            # 实际执行准备(隔离或进程内)
            self.prepare_implement(comp_id)
            
            # 登记准备完成
            self.todo["prepare"] = True
            update += [HumanMessage(content="[OK] Data preparation completed")]
            print(f"[OK] Data preparation completed for {comp_id}")
            
            # 回答与反馈:记录思考/代码/成功反馈(供下轮上下文)
            self.todo["prepares"] += [
                AIMessage(content=think),
                AIMessage(content=prepare_code)                          
            ]
            self.todo["prepares"] += [
                HumanMessage(
                    content="Data preparation successful using the latest code script"
                )
            ]
            
        except Exception as e:
            # 失败:重试计数减一
            self.todo["retry"] -= 1
            self.todo["errors"].append(f"[ERROR] Data preparation failed: {e}")
            
            # 回答与反馈:有解析结果用解析值,否则回填原始输出/错误
            self.todo["prepares"] += [
                AIMessage(content=think if think != "VOID" else response.content),
                AIMessage(content=prepare_code if prepare_code != "VOID" else str(e))                          
            ]
            self.todo["prepares"] += [
                HumanMessage(content=f"Data preparation failed using the latest code script: {e}")
            ]
            
            if self.todo["retry"] > -1:
                # 还有重试机会:写入重试提示
                retry_msg = (
                    f"[ERROR] Data preparation failed: {e}\n"
                    f"Retrying (Remaining attempts: {self.todo['retry']})..."
                )
                update += [HumanMessage(content=retry_msg)]
                print(
                    f"[ERROR] Data preparation failed for {comp_id}: {e}\n"
                    f"Retrying (Remaining attempts: {self.todo['retry']})..."
                )
            else:
                # 重试耗尽:标记最终失败
                update += [HumanMessage(content=f"[ERROR] Data preparation FAILED: {e}")]
                print(f"[ERROR] Data preparation FAILED for {comp_id}: {e}")
        
        # 返回消息增量
        return {"messages": update}
    
    def Metric(self, state: AgentState) -> AgentState:
        """生成评估指标脚本 metric.py(状态机节点:Phase 7)。
        
        Args:
            state: 当前智能体状态
            
        Returns:
            消息增量字典
        """
        # 阶段横幅
        print("\nPhase 7: Generating metric...")
        # 任务名
        comp_id = self.todo["name"]
        # 消息增量
        update = []
        
        try: 
            # 读取公开侧描述文件
            desc_file = self.structure.public_dir(comp_id) / "description.txt"
            with open(desc_file, 'r', encoding='utf-8') as f:
                description = f.read()
            
            # 采样提交文件前 2 行作为格式样例
            submission_sample = ""
            submission_path = self.structure.public_dir(comp_id) / "sample_submission.csv"
            if submission_path.exists():
                df = pd.read_csv(submission_path, nrows=2)
                submission_sample = str(df)
                    
            # 采样私有答案前 2 行作为格式样例
            answer_sample = ""
            answer_path = self.structure.private_dir(comp_id) / "test_answer.csv"
            if answer_path.exists():
                df = pd.read_csv(answer_path, nrows=2)
                answer_sample = str(df)
                    
            # 加载指标脚本样例(few-shot)
            metric_sample = self._load_sample_file("sample_metric.py")
            
            # 构造信息块:描述+两个样例
            infos = metric_info.format(
                description=description,
                submission_sample=submission_sample,
                answer_sample=answer_sample,
            )
            
            # 构造系统+用户消息
            query = [
                SystemMessage(content=gen_metric_script.format(metric_sample=metric_sample)), 
                HumanMessage(content=
                    f"Now THINK and WRITE the CODE based on the INFORMATION:\n{infos}\nDO NOT FORGET to wrap the CODE part in <code> and </code>!"
                )
            ]
            # 入增量
            update += query
            
            # 请求 LLM 生成指标代码
            response = self.llm_provider.query(query)
            # metric.py 落盘路径
            metric_file = self.structure.utils_dir(comp_id) / "metric.py"
            # 解析代码并编译校验
            _, metric_code = self._clean_code_output(response.content)
            compile(metric_code, str(metric_file), "exec")
            # 原子写盘
            atomic_write_text(metric_file, metric_code)
                
            # 登记完成标记与消息
            self.todo["metric"] = True
            update += [response, HumanMessage(content="[OK] Metric script generated")]
            print(f"[OK] Metric script generated for {comp_id}")

        except Exception as e:
            # 失败记录错误(不设完成标记)
            update += [HumanMessage(content=f"[ERROR] Metric script generation failed: {e}")]
            self.todo["errors"].append(
                f"Metric script generation failed: {e}"
            )
            print(f"[ERROR] Metric script generation failed for {comp_id}: {e}")

        # 返回消息增量
        return {"messages": update}
    
    
    # ========================================================================
    # 工作流管理方法
    # ========================================================================
    
    def Next(self, state: AgentState):
        """收尾节点:汇总成功标记、可选清理原始数据、把 prepares 历史落盘 prepares.json。"""
        # 当前任务名
        comp_id = self.todo["name"]

        # 成功 = 六个阶段标记全部为真
        self.todo["success"] = all([
            self.todo["download"],
            self.todo["copy"],
            self.todo["web_info"],
            self.todo["describe"],
            self.todo["prepare"],
            self.todo["metric"]
        ])

        # 需要删原始数据且整体成功:把 fileinfo 转存为 raw.txt 占位后删 raw
        if self.delete_raw and self.todo["success"]:
            # 占位文件路径
            raw_placeholder = self.structure.data_comp_id_dir(comp_id) / "raw.txt"
            # 确保父目录存在
            raw_placeholder.parent.mkdir(parents=True, exist_ok=True)
            # 文件清点说明
            fileinfo = self.structure.comp_id_dir(comp_id) / "fileinfo.txt"
            if fileinfo.is_file() and fileinfo.stat().st_size > 0:
                # 原子写入占位说明
                atomic_write_text(
                    raw_placeholder,
                    fileinfo.read_text(encoding="utf-8"),
                )
                # 删除原始数据目录
                shutil.rmtree(self.structure.raw_dir(comp_id), ignore_errors=True)
            else:
                # 无清点说明:保守保留原始数据并记录
                self.todo["errors"].append(
                    "Raw data retained: fileinfo.txt is missing or empty"
                )

        # 打印该任务成败横幅
        print(
            f"\n[OK] Successful for {comp_id}"
            if self.todo["success"]
            else f"\n[ERROR] Failed for {comp_id}"
        )

        # prepares 历史(每 4 条一组:请求/思考/代码/反馈)落盘
        preps_file = self.structure.comp_id_dir(comp_id) / "prepares.json"
        preps = []

        # 按 4 条一组切分(不完整的尾部丢弃)
        for idx in range(0, len(self.todo["prepares"]), 4):
            if idx + 3 < len(self.todo["prepares"]):
                prep_entry = {
                    "description": self.todo["prepares"][idx].content,
                    "think": self.todo["prepares"][idx + 1].content,
                    "code": self.todo["prepares"][idx + 2].content,
                    "feedback": self.todo["prepares"][idx + 3].content
                }
                preps.append(prep_entry)

        # 写 prepares.json
        with open(preps_file, "w", encoding="utf-8") as f:
            json.dump(preps, f, indent=4, ensure_ascii=False)

        # 返回空消息增量 + 任务结果(供上层聚合)
        return {
            "messages": [],
            "task_result": self.todo
        }
    
    # ========================================================================
    # 路由方法
    # ========================================================================
    
    def Router1(self, state: AgentState) -> str:
        """按下载状态路由(successful/failed)。"""
        # 下载完成走成功边
        return "successful" if self.todo["download"] else "failed"
    
    def Router2(self, state: AgentState) -> str:
        """按复制解压状态路由(successful/failed)。"""
        # 复制完成走成功边
        return "successful" if self.todo["copy"] else "failed"
    
    def Router3(self, state: AgentState) -> str:
        """按网络信息状态路由(successful/failed)。"""
        # 信息抓取完成走成功边
        return "successful" if self.todo["web_info"] else "failed"
    
    def Router4(self, state: AgentState) -> str: # added
        """按文件清点状态路由(successful/continue)。"""
        # 清点未完成时继续留在感知节点(工具循环)
        return "successful" if self.todo["perceive"] else "continue"
    
    def Router5(self, state: AgentState) -> str:
        """按描述状态路由(successful/failed)。"""
        # 描述完成走成功边
        return "successful" if self.todo["describe"] else "failed"
    
    def Router6(self, state: AgentState) -> str:
        """按准备状态路由(successful/retry/failed):成功/仍有重试额度/彻底失败。"""
        # 准备成功
        if self.todo["prepare"]:
            return "successful"
        # 还有重试额度
        elif self.todo["retry"] > -1:
            return "retry"
        else:
            # 重试耗尽:失败
            return "failed"
