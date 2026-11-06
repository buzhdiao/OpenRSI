"""
Program Database for storing and managing generated programs during rollout.
Uses SQLite for persistent storage with thread-safety.
"""

# 从 dataclasses 导入 dataclass 装饰器与 field，用于定义 Program 数据类
from dataclasses import dataclass, field
# 导入类型注解工具
from typing import List, Optional, Dict, Any
# 导入 Enum，用于定义数据库生命周期枚举
from enum import Enum
# 导入 sqlite3：持久化存储生成程序的数据库后端
import sqlite3
# 导入 json：序列化 payload/metadata 等字典字段
import json
# 导入 threading：提供线程锁与线程本地存储，保证多线程安全
import threading
# 导入 os：目录创建等文件系统操作
import os
# 导入 Path：路径处理
from pathlib import Path
# 导入 random：父代选择的随机采样
import random
# 导入 logging：模块级日志
import logging
# 导入 math：有限性判断（isfinite）与数值钳制
import math
# 导入 sys：读取 float 最大值（SQLite REAL 钳制用）
import sys
# 导入 statistics：统计工具（方差等）
import statistics
# 重复导入 statistics（保留原样，无实际影响）
import statistics
# 导入 AIRA-Evo 经验系统的工具：按算子构建经验记忆、计算父代效用、调试候选与成功程序判定、程序选择得分
from airaevo_experience import (
    build_operator_experience_memory,
    compute_parent_utilities,
    is_debug_candidate,
    is_success_program,
    program_selection_score,
)

# 获取名为 mle_agent 的模块级 logger
logger = logging.getLogger("mle_agent")


class DatabaseLifecycle(Enum):
    """Lifecycle modes for Program Database."""

    # 生命周期模式：每个 rollout 一个数据库
    ROLLOUT = "rollout"  # One database per rollout
    # 生命周期模式：每个训练步一个数据库
    STEP = "step"  # One database per step
    # 生命周期模式：每个任务一个数据库
    TASK = "task"  # One database per task


@dataclass
class Program:
    """Represents a program in the database."""

    # 数据库自增主键 ID（插入前可为 None）
    id: Optional[int] = None  # Database ID
    # 任务标识（uuid）
    task_id: str = ""  # Task identifier (uuid)
    # 任务名称
    task_name: str = ""  # Task name
    # 沙箱执行返回的 HTTP 状态码
    status_code: int = 0  # Status code from sandbox execution
    # 沙箱执行返回的结果 payload（字典）
    payload: dict = field(default_factory=dict)  # Payload from sandbox execution
    # 生成的代码文本
    code: str = ""  # The code text
    # 沙箱执行得分（None 表示无分数，区别于 0.0 分）
    score: Optional[float] = None  # Score from sandbox execution (None = no score, different from 0.0)
    # 观测到的沙箱总运行时间（秒）
    running_time: float = 0.0  # Total observed sandbox running time in seconds
    # 由分数计算的奖励（base_reward）
    reward: float = 0.0  # Reward calculated from score (base_reward)
    # 搜索选择使用的适应度（fitness）
    fitness: float | None = None  # Fitness used by search selection
    # 未考虑父代信息的基础奖励
    base_reward: float = 0.0  # Base reward before considering parent
    # 由 base_reward 归一化得到的利用（exploit）系数，对应 FunSearch 式程序库的利用项
    exploit_coefficient: float = 0.5  # Normalized coefficient from base_reward
    # 由子代 base_reward 方差归一化得到的探索（explore）系数
    explore_coefficient: float = 0.5  # Normalized coefficient from child_base_reward_variance
    # 由访问次数推导的冷却（cooling）系数（类似 UCT 的未访问偏好）
    cooling_coefficient: float = 1.0  # Cooling value derived from visit_count
    # 该节点被选为父代的次数
    visit_count: int = 0  # Number of times this node is selected as parent
    # 直接子代 base_reward 的方差（None 表示无子代）
    child_base_reward_variance: float | None = None  # Variance of direct child base_reward values (None=no child)
    # 父代程序 ID（improve/debug 等模式使用）
    parent_id: Optional[int] = None  # Parent program ID (for improve mode)
    # 父代程序代码（仅用于日志）
    parent_code: str = ""  # Parent program code (for logging)
    # 生成模式：四算子之一 'draft' | 'improve' | 'debug' | 'crossover'
    generation_mode: str = "draft"  # Generation mode: 'draft' | 'improve' | 'debug' | 'crossover'
    # 模型原始响应文本
    raw_text: str = ""  # Raw response text from model
    # 作弊检查结果：0 表示合法 ML 代码，1 表示投机取巧/瞎猜（execution-grounded 校验）
    hack: int = 0  # Hack check result: 0 = valid ML code, 1 = cheating/guessing
    # 附加元数据字典
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata

    def __post_init__(self) -> None:
        # 数据类初始化后的钩子：处理 fitness 缺省与运行日志提取
        if self.fitness is None:
            # Fitness is recomputed at task level from multiple components after insertion.
            # fitness 为 None 时先置 0.0，插入后会在任务级由多个分量重新计算
            self.fitness = 0.0
        # 从元数据中读取清洗后的运行日志
        clear_run_log = self.metadata.get("clear_run_log")
        if clear_run_log is not None:
            # 存在时挂到 run_log 属性上（非 dataclass 字段）
            self.run_log = str(clear_run_log)

    def to_dict(self) -> Dict[str, Any]:
        """Convert program to dictionary for storage."""
        # 把程序对象转换为可写入 SQLite 的字典（metadata 序列化为 JSON 字符串）
        return {
            "id": self.id,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "status_code": self.status_code,
            "payload": self.payload,
            "code": self.code,
            "score": self.score,
            "running_time": self.running_time,
            "reward": self.reward,
            "fitness": self.fitness,
            "base_reward": self.base_reward,
            "exploit_coefficient": self.exploit_coefficient,
            "explore_coefficient": self.explore_coefficient,
            "cooling_coefficient": self.cooling_coefficient,
            "visit_count": self.visit_count,
            "child_base_reward_variance": self.child_base_reward_variance,
            "parent_id": self.parent_id,
            "parent_code": self.parent_code,
            "generation_mode": self.generation_mode,
            "raw_text": self.raw_text,
            "hack": self.hack,
            "metadata": json.dumps(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Program":
        """Create program from dictionary."""
        # 从字典构造程序对象；metadata 若是 JSON 字符串则反序列化
        metadata = (
            json.loads(data.get("metadata", "{}"))
            if isinstance(data.get("metadata"), str)
            else data.get("metadata", {})
        )
        # payload 若是 JSON 字符串则反序列化，否则直接使用
        payload = (
            json.loads(data.get("payload", "{}")) if isinstance(data.get("payload"), str) else data.get("payload", {})
        )
        # 逐字段构造 Program（score 允许 None，与 0 分区分）
        return cls(
            id=data.get("id"),
            task_id=data.get("task_id", ""),
            task_name=data.get("task_name", ""),
            status_code=data.get("status_code", 0),
            payload=payload,
            code=data.get("code", ""),
            score=data.get("score"),  # Allow None to distinguish from 0
            running_time=data.get("running_time", 0.0),
            reward=data.get("reward", 0.0),
            fitness=data.get("fitness", 0.0),
            base_reward=data.get("base_reward", 0.0),
            exploit_coefficient=data.get("exploit_coefficient", 0.5),
            explore_coefficient=data.get("explore_coefficient", 0.5),
            visit_count=data.get("visit_count", 0),
            child_base_reward_variance=data.get("child_base_reward_variance"),
            cooling_coefficient=data.get("cooling_coefficient", 1.0),
            parent_id=data.get("parent_id"),
            parent_code=data.get("parent_code", ""),
            generation_mode=data.get("generation_mode", "draft"),
            raw_text=data.get("raw_text", ""),
            hack=data.get("hack", 0),  # Default to 0 (not hacking, valid) for backward compatibility
            metadata=metadata,
        )


class ProgramDatabase:
    """Database for storing and managing programs during rollout.

    Uses SQLite for persistent storage with connection pooling and thread-safety.
    Each task maintains its top-k programs by reward.
    """

    # 线程本地存储：每个线程持有独立的 SQLite 连接
    _local = threading.local()
    # 全局写锁：串行化所有写操作，保证线程安全
    _lock = threading.Lock()
    # SQLite 整数下界（-2^63）
    _SQLITE_INT_MIN = -(2**63)
    # SQLite 整数上界（2^63-1）
    _SQLITE_INT_MAX = 2**63 - 1
    # Python float 最大值（SQLite REAL 钳制用）
    _SQLITE_FLOAT_MAX = sys.float_info.max

    def __init__(self, db_path: str = "program_database.db", max_per_task: int = 10):
        """
        Initialize Program Database.

        Args:
            db_path: Path to SQLite database file
            max_per_task: Maximum number of programs to keep per task (top k by reward)
        """
        # 记录数据库文件路径
        self.db_path = db_path
        # 记录每个任务保留的最大程序数（按 fitness 取 top-k）
        self.max_per_task = max_per_task
        # 标记 child_base_reward_variance 列是否允许 NULL（老库兼容）
        self._child_variance_nullable = True

        # Ensure directory exists
        # 确保数据库文件所在目录存在（无目录部分时用当前目录）
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) if os.path.dirname(db_path) else ".", exist_ok=True)

        # Initialize database schema
        # 初始化数据库表结构与索引
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            # 当前线程尚无连接时新建（check_same_thread=False 允许跨线程创建，30 秒锁等待）
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
            # 使用 Row 工厂，使查询结果可按列名访问
            self._local.conn.row_factory = sqlite3.Row
        # 返回当前线程的连接
        return self._local.conn

    def _init_db(self):
        """Initialize database schema."""
        # 获取线程本地连接与游标
        conn = self._get_connection()
        cursor = conn.cursor()

        # 创建 programs 表（若不存在），包含程序全部字段与父代外键
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS programs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                task_name TEXT NOT NULL,
                status_code INTEGER NOT NULL,
                payload TEXT NOT NULL,
                code TEXT NOT NULL,
                score REAL,
                running_time REAL NOT NULL DEFAULT 0.0,
                reward REAL NOT NULL,
                fitness REAL NOT NULL,
                base_reward REAL NOT NULL,
                exploit_coefficient REAL NOT NULL DEFAULT 0.5,
                explore_coefficient REAL NOT NULL DEFAULT 0.5,
                cooling_coefficient REAL NOT NULL DEFAULT 1.0,
                visit_count INTEGER NOT NULL DEFAULT 0,
                child_base_reward_variance REAL,
                parent_id INTEGER,
                parent_code TEXT,
                generation_mode TEXT DEFAULT 'draft',
                raw_text TEXT,
                hack INTEGER DEFAULT 0,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_id) REFERENCES programs(id)
            )
        """
        )

        cursor.execute("PRAGMA table_info(programs)")  # Read table schema metadata.
        # 读取现有表的列信息，用于老库迁移
        table_info = cursor.fetchall()
        # 收集现有列名集合
        columns = {row["name"] for row in table_info}  # Collect column names.

        if "fitness" not in columns:  # Add fitness column when missing.
            # 老库缺 fitness 列：补列（默认 0.0，非空）
            cursor.execute("ALTER TABLE programs ADD COLUMN fitness REAL NOT NULL DEFAULT 0.0")  # Add fitness column.
            # 用 reward 值回填 fitness，保证历史数据可用
            cursor.execute("UPDATE programs SET fitness = reward")  # Backfill fitness from reward.

        if "hack" not in columns:  # Add hack column when missing.
            # 老库缺 hack 列：补列（默认 0）
            cursor.execute("ALTER TABLE programs ADD COLUMN hack INTEGER DEFAULT 0")  # Add hack column.
            # 历史程序统一回填为 0（合法，非作弊）
            cursor.execute("UPDATE programs SET hack = 0")  # Backfill hack as 0 (valid) for existing programs.
        if "running_time" not in columns:
            # 老库缺 running_time 列：补列并回填 0.0
            cursor.execute("ALTER TABLE programs ADD COLUMN running_time REAL NOT NULL DEFAULT 0.0")
            cursor.execute("UPDATE programs SET running_time = 0.0")
        if "exploit_coefficient" not in columns:
            # 老库缺 exploit_coefficient 列：补列并回填中性值 0.5
            cursor.execute("ALTER TABLE programs ADD COLUMN exploit_coefficient REAL NOT NULL DEFAULT 0.5")
            cursor.execute("UPDATE programs SET exploit_coefficient = 0.5")
        if "explore_coefficient" not in columns:
            # 老库缺 explore_coefficient 列：补列并回填中性值 0.5
            cursor.execute("ALTER TABLE programs ADD COLUMN explore_coefficient REAL NOT NULL DEFAULT 0.5")
            cursor.execute("UPDATE programs SET explore_coefficient = 0.5")
        if "visit_count" not in columns:
            # 老库缺 visit_count 列：补列并回填 0
            cursor.execute("ALTER TABLE programs ADD COLUMN visit_count INTEGER NOT NULL DEFAULT 0")
            cursor.execute("UPDATE programs SET visit_count = 0")
        if "child_base_reward_variance" not in columns:
            # 老库缺 child_base_reward_variance 列：补列（可空）
            cursor.execute("ALTER TABLE programs ADD COLUMN child_base_reward_variance REAL")
        if "cooling_coefficient" not in columns:
            # 老库缺 cooling_coefficient 列：补列并回填 1.0
            cursor.execute("ALTER TABLE programs ADD COLUMN cooling_coefficient REAL NOT NULL DEFAULT 1.0")
            cursor.execute("UPDATE programs SET cooling_coefficient = 1.0")

        # Backward compatibility: old DB may enforce NOT NULL on child_base_reward_variance.
        # 再次读取表结构，检测 child_base_reward_variance 是否允许 NULL
        cursor.execute("PRAGMA table_info(programs)")
        table_info = cursor.fetchall()
        # 按列名建立信息索引
        info_by_name = {row["name"]: row for row in table_info}
        # 取 child_base_reward_variance 列的 schema 信息
        cbrv = info_by_name.get("child_base_reward_variance")
        # notnull=0 表示可空；记录下来供插入时决定是否需要占位值
        self._child_variance_nullable = bool(cbrv is None or int(cbrv["notnull"]) == 0)

        # Create index for faster task-based queries
        # 为 task_name 建索引，加速按任务查询
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_task_name ON programs(task_name)
        """
        )
        # 为 task_id 建索引，加速按任务 ID 查询
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_task_id ON programs(task_id)
        """
        )
        # 为 (task_name, reward DESC) 建复合索引，加速按任务取最高奖励
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_task_reward ON programs(task_name, reward DESC)
        """
        )
        # 为 (task_id, reward DESC) 建复合索引
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_task_id_reward ON programs(task_id, reward DESC)
        """
        )
        # 为 (task_name, fitness DESC) 建复合索引，加速按适应度选父代
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_task_fitness ON programs(task_name, fitness DESC)
        """
        )

        # 提交建表与索引变更
        conn.commit()

    def get_task_scores(self, *, task_id: str = "", task_name: str = "") -> list[float]:
        """Return all finite stored scores for a task."""
        # 未提供任何任务标识时返回空列表
        if not task_id and not task_name:
            return []
        with self._lock:
            # 加锁后获取连接与游标
            conn = self._get_connection()
            cursor = conn.cursor()
            if task_id:
                # 优先按 task_id 查询该任务的所有非空 score，按插入顺序返回
                cursor.execute(
                    """
                    SELECT score
                    FROM programs
                    WHERE task_id = ? AND score IS NOT NULL
                    ORDER BY id ASC
                """,
                    (task_id,),
                )
            else:
                # 否则按 task_name 查询
                cursor.execute(
                    """
                    SELECT score
                    FROM programs
                    WHERE task_name = ? AND score IS NOT NULL
                    ORDER BY id ASC
                """,
                    (task_name,),
                )
            # 收集有效分数的列表
            scores: list[float] = []
            # 逐行解析分数
            for row in cursor.fetchall():
                try:
                    # 转为浮点数
                    score = float(row["score"])
                except (TypeError, ValueError):
                    # 无法解析的行直接跳过
                    continue
                if math.isfinite(score):
                    # 只保留有限值（过滤 NaN/inf）
                    scores.append(score)
            # 返回全部有效分数
            return scores

    def merge_program_metadata(self, program_id: int, metadata_updates: Dict[str, Any]) -> bool:
        """Merge metadata fields into an existing program row."""
        # 无更新内容时直接返回 False
        if not metadata_updates:
            return False
        with self._lock:
            # 加锁后获取连接与游标
            conn = self._get_connection()
            cursor = conn.cursor()
            # 查询目标程序的现有 metadata
            cursor.execute(
                """
                SELECT metadata
                FROM programs
                WHERE id = ?
            """,
                (int(program_id),),
            )
            # 取一行结果
            row = cursor.fetchone()
            if row is None:
                # 程序不存在时返回 False
                return False
            try:
                # 反序列化现有 metadata（空值时用空字典）
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            except Exception:
                # 反序列化失败时退化为空字典
                metadata = {}
            if not isinstance(metadata, dict):
                # 非字典类型时退化为空字典
                metadata = {}
            # 合并更新字段（新值覆盖旧值）
            metadata.update(metadata_updates)
            # 把合并后的 metadata 写回数据库
            cursor.execute(
                """
                UPDATE programs
                SET metadata = ?
                WHERE id = ?
            """,
                (json.dumps(metadata), int(program_id)),
            )
            # 提交事务
            conn.commit()
            # 返回合并成功
            return True

    def add(self, program: Program) -> int:
        """
        Add a program to the database, maintaining top k per task by reward.

        Returns:
            The database ID of the inserted program
        """

        def _to_sqlite_int(value: Any, *, default: int | None = None) -> int | None:
            # 嵌套工具：把值安全转为 SQLite 可接受的整数，越界时钳制
            if value is None:
                # 空值返回默认值
                return default
            try:
                # 尝试转为整数
                iv = int(value)
            except (TypeError, ValueError):
                # 转换失败返回默认值
                return default
            if iv > self._SQLITE_INT_MAX:
                # 超过 SQLite 整数上界时钳制并告警
                logger.warning("Clamping int value %s to SQLite max", iv)
                return self._SQLITE_INT_MAX
            if iv < self._SQLITE_INT_MIN:
                # 低于 SQLite 整数下界时钳制并告警
                logger.warning("Clamping int value %s to SQLite min", iv)
                return self._SQLITE_INT_MIN
            # 正常范围直接返回
            return iv

        def _to_sqlite_real(value: Any, *, default: float | None = None) -> float | None:
            # 嵌套工具：把值安全转为 SQLite 可接受的浮点数，非有限值时钳制
            if value is None:
                # 空值返回默认值
                return default
            try:
                # 尝试转为浮点数
                fv = float(value)
            except (TypeError, ValueError):
                # 转换失败返回默认值
                return default
            if not math.isfinite(fv):
                if fv == 0.0:
                    # float('nan')==0 为 False，此处捕获的是 -0.0/0.0 边界情形，直接返回 0.0
                    return 0.0
                # 非有限值（NaN/inf）按符号钳制到 float 极值并告警
                logger.warning("Clamping non-finite float value %s to max", fv)
                return math.copysign(self._SQLITE_FLOAT_MAX, fv)
            # 正常值直接返回
            return fv

        with self._lock:
            # 加锁后获取连接与游标，保证插入原子性
            conn = self._get_connection()
            cursor = conn.cursor()

            # 状态码钳制为合法整数（缺省 0）
            status_code = _to_sqlite_int(program.status_code, default=0)
            # 父代 ID 钳制为合法整数（可为 None）
            parent_id = _to_sqlite_int(program.parent_id, default=None)
            # 作弊标志钳制为合法整数（缺省 0）
            hack_value = _to_sqlite_int(program.hack, default=0)
            # 分数钳制为合法浮点数（可为 None 表示无分数）
            score_value = _to_sqlite_real(program.score, default=None)
            # 运行时间钳制为合法浮点数，None/0 归一为 0.0
            running_time_value = _to_sqlite_real(program.running_time, default=0.0) or 0.0
            # 奖励钳制为合法浮点数（缺省 0.0）
            reward_value = _to_sqlite_real(program.reward, default=0.0)
            # Persist a placeholder value; task-level recomputation immediately updates fitness.
            # 先持久化占位 fitness，插入后立即在任务级重算更新
            fitness_value = _to_sqlite_real(program.fitness, default=0.0)
            # 基础奖励钳制为合法浮点数
            base_reward_value = _to_sqlite_real(program.base_reward, default=0.0)
            # 利用系数钳制（缺省 0.5）
            exploit_coef_value = _to_sqlite_real(program.exploit_coefficient, default=0.5)
            # 探索系数钳制（缺省 0.5）
            explore_coef_value = _to_sqlite_real(program.explore_coefficient, default=0.5)
            # 访问次数钳制为合法整数，None/0 归一为 0
            visit_count_value = _to_sqlite_int(program.visit_count, default=0) or 0
            # 子代奖励方差钳制；列不可空且值为 None 时用中性占位 0.5
            child_variance_value = _to_sqlite_real(
                program.child_base_reward_variance,
                default=None if self._child_variance_nullable else 0.5,
            )
            # 冷却系数钳制（缺省 1.0）
            cooling_value = _to_sqlite_real(program.cooling_coefficient, default=1.0)

            # Insert the program
            # 把程序插入 programs 表（payload/metadata 序列化为 JSON 字符串）
            cursor.execute(
                """
                INSERT INTO programs (task_id, task_name, status_code, payload, code, score, running_time, reward, fitness, base_reward,
                                     exploit_coefficient, explore_coefficient, cooling_coefficient,
                                     visit_count, child_base_reward_variance,
                                     parent_id, parent_code, generation_mode, raw_text, hack, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    program.task_id,
                    program.task_name,
                    status_code,
                    json.dumps(program.payload),  # Convert dict to JSON string
                    program.code,
                    score_value,
                    running_time_value,
                    reward_value,
                    fitness_value,
                    base_reward_value,
                    exploit_coef_value,
                    explore_coef_value,
                    cooling_value,
                    visit_count_value,
                    child_variance_value,
                    parent_id,
                    program.parent_code,
                    program.generation_mode,
                    program.raw_text,
                    hack_value,
                    json.dumps(program.metadata),
                ),
            )
            # 取自增主键作为程序 ID
            program_id = cursor.lastrowid

            # Recompute task fitness after each insertion.
            # 每次插入后重算该任务下所有程序的 fitness 与系数
            self._recompute_task_fitness(program.task_name, conn, cursor)

            # Maintain top k programs per task when max_per_task > 0.
            # 统计该任务下的程序总数，用于维护 top-k 容量
            cursor.execute(
                """
                SELECT COUNT(*) as count FROM programs WHERE task_name = ?
            """,
                (program.task_name,),
            )
            # 取出计数值
            count = cursor.fetchone()["count"]

            if self.max_per_task > 0 and count > self.max_per_task:
                # Delete programs beyond top k
                # 超出容量时删除 fitness 最低的多余程序，仅保留 top-k
                cursor.execute(
                    """
                    DELETE FROM programs
                    WHERE task_name = ? AND id NOT IN (
                        SELECT id FROM programs
                        WHERE task_name = ?
                        ORDER BY fitness DESC
                        LIMIT ?
                    )
                """,
                    (program.task_name, program.task_name, self.max_per_task),
                )

            # 提交事务
            conn.commit()
            # 返回新插入程序的 ID
            return program_id

    @staticmethod
    def _normalize_minmax(values: list[float], neutral: float = 0.5) -> list[float]:
        # 静态工具：min-max 归一化；全部相同或空列表时返回中性值
        if not values:
            # 空列表返回空
            return []
        # 最小值
        lo = min(values)
        # 最大值
        hi = max(values)
        if abs(hi - lo) < 1e-12:
            # 最大最小几乎相等（无区分度）时全部返回中性值
            return [neutral for _ in values]
        # 归一化分母（极差）
        denom = hi - lo
        # 逐值做 (v-lo)/(hi-lo) 线性归一化
        return [(v - lo) / denom for v in values]

    @staticmethod
    def _normalize_minmax_optional(values: list[float | None], neutral: float = 0.5) -> list[float]:
        # 静态工具：带 None 的 min-max 归一化，仅对有限值计算极差，None 位置保持中性值
        if not values:
            # 空列表返回空
            return []
        # 收集非 None 的 (下标, 值) 对
        finite_idx_vals = [(i, v) for i, v in enumerate(values) if v is not None]
        if not finite_idx_vals:
            # 全部为 None 时整体返回中性值
            return [neutral for _ in values]
        # 提取有限值列表用于计算极差
        finite_vals = [float(v) for _, v in finite_idx_vals]
        # 最小值
        lo = min(finite_vals)
        # 最大值
        hi = max(finite_vals)
        # 初始化输出：全部为中性值
        out = [neutral for _ in values]
        if abs(hi - lo) < 1e-12:
            # 无区分度时非 None 位置也置为中性值后返回
            for i, _ in finite_idx_vals:
                out[i] = neutral
            return out
        # 归一化分母（极差）
        denom = hi - lo
        # 仅对非 None 位置做线性归一化
        for i, v in finite_idx_vals:
            out[i] = (float(v) - lo) / denom
        # 返回归一化结果
        return out

    @staticmethod
    def _safe_float(v: Any, default: float = 0.0) -> float:
        # 静态工具：安全转换为有限浮点数，失败或非有限时返回默认值
        try:
            # 尝试转换
            fv = float(v)
            if math.isfinite(fv):
                # 有限值直接返回
                return fv
        except Exception:
            # 转换异常走默认值
            pass
        # 返回默认值
        return default

    def _extract_parent_ids_from_metadata(self, metadata: dict[str, Any]) -> list[int]:
        # 从元数据的 crossover_parent_ids 字段提取交叉算子的全部父代 ID
        parent_ids: list[int] = []
        # 读取候选父代 ID 列表
        maybe_ids = metadata.get("crossover_parent_ids")
        if isinstance(maybe_ids, list):
            # 仅当字段为列表时逐个解析
            for pid in maybe_ids:
                try:
                    # 尝试转为整数并加入结果
                    pid_i = int(pid)
                    parent_ids.append(pid_i)
                except Exception:
                    # 非法条目直接跳过
                    continue
        # 返回提取到的父代 ID 列表
        return parent_ids

    def _recompute_task_fitness(self, task_name: str, conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
        # 任务级重算 fitness：利用项 + 探索项 + 冷却项（FunSearch/UCT 风格的进化程序库核心）
        cursor.execute(
            """
            SELECT * FROM programs
            WHERE task_name = ?
            """,
            (task_name,),
        )
        # 取出该任务下所有程序行
        rows = cursor.fetchall()
        if not rows:
            # 任务下无程序时无需重算
            return

        # 把数据库行反序列化为 Program 对象列表
        programs = [Program.from_dict(dict(r)) for r in rows]
        # 建立 id 到 Program 的映射（跳过无 id 的对象）
        by_id = {p.id: p for p in programs if p.id is not None}
        if not by_id:
            # 无有效 id 映射时直接返回
            return

        # 1) Utilization value: node base_reward (higher is better).
        # 1) 利用项：节点自身 base_reward 越高越值得被再利用
        utilization_raw = {pid: self._safe_float(p.base_reward, default=0.0) for pid, p in by_id.items()}

        # 2) Learning value: variance of child base_reward (higher is better), default 0.5 when no children.
        # 2) 学习项：子代 base_reward 的方差越大（后代分歧度高）越有探索价值
        child_rewards: dict[int, list[float]] = {pid: [] for pid in by_id}
        # 遍历每个程序，把它登记为其父代的子代
        for child in programs:
            # 收集该子代关联的父代 id 集合（parent_id 与交叉父代并集）
            linked_parents: set[int] = set()
            if child.parent_id is not None and child.parent_id in by_id:
                # 显式 parent_id 存在且在库内时加入
                linked_parents.add(int(child.parent_id))
            if isinstance(child.metadata, dict):
                # 元数据为字典时再提取交叉算子的额外父代
                for pid in self._extract_parent_ids_from_metadata(child.metadata):
                    if pid in by_id:
                        # 仅登记库内存在的父代
                        linked_parents.add(pid)
            for pid in linked_parents:
                # 把该子代的 base_reward 记到每个关联父代名下
                child_rewards[pid].append(self._safe_float(child.base_reward, default=0.0))

        # 计算每个节点的学习项原始值（子代 base_reward 总体方差）
        learning_raw: dict[int, float | None] = {}
        for pid, rewards in child_rewards.items():
            if len(rewards) == 0:
                # 无子代时记 None（归一化时取中性值）
                learning_raw[pid] = None
            elif len(rewards) == 1:
                # 单个子代方差定义为 0
                learning_raw[pid] = 0.0
            else:
                # 子代奖励均值
                mean_reward = sum(rewards) / len(rewards)
                # 总体方差（除以 n），衡量后代质量分歧度
                learning_raw[pid] = float(sum((x - mean_reward) ** 2 for x in rewards) / len(rewards))

        # 3) Cooling coefficient: fewer visits is better.
        # 3) 冷却项：被选中次数越少的节点越"新鲜"，优先探索
        # Visit count is parent selection count from child links.
        # 访问次数由子代链接（父代选择记录）统计而来
        visit_counts: dict[int, int] = {pid: 0 for pid in by_id}
        # 再次遍历所有程序统计每个父代被选中的次数
        for child in programs:
            # 收集该子代关联的父代 id 集合
            linked_parents: set[int] = set()
            if child.parent_id is not None and child.parent_id in by_id:
                # 显式 parent_id 在库内时加入
                linked_parents.add(int(child.parent_id))
            if isinstance(child.metadata, dict):
                # 元数据中提取交叉算子的额外父代
                for pid in self._extract_parent_ids_from_metadata(child.metadata):
                    if pid in by_id:
                        # 仅统计库内存在的父代
                        linked_parents.add(pid)
            for pid in linked_parents:
                # 每条子代链接为对应父代的访问次数加一
                visit_counts[pid] += 1

        # 全库最大访问次数（用于归一化冷却值）
        max_visit = max(visit_counts.values()) if visit_counts else 0
        # 每个节点的冷却原始值
        cooling_raw: dict[int, float] = {}
        for pid, count in visit_counts.items():
            if max_visit <= 0:
                # 无人被访问过时全部节点冷却值取 1.0（最大化探索）
                cooling_raw[pid] = 1.0
            else:
                # 冷却值 = 1 - 访问占比，且下界钳为 0（访问越多越"冷却"）
                cooling_raw[pid] = max(0.0, 1.0 - (float(count) / float(max_visit)))

        # 按库内 id 顺序建立稳定排列
        ordered_ids = list(by_id.keys())
        # 利用项 min-max 归一化
        exploit_coef = self._normalize_minmax([utilization_raw[pid] for pid in ordered_ids], neutral=0.5)
        # 学习项（带 None）min-max 归一化
        explore_coef = self._normalize_minmax_optional([learning_raw[pid] for pid in ordered_ids], neutral=0.5)
        # 冷却项 min-max 归一化
        cool_norm = self._normalize_minmax([cooling_raw[pid] for pid in ordered_ids], neutral=0.5)

        # 逐节点把 fitness 与各系数写回数据库
        for idx, pid in enumerate(ordered_ids):
            # fitness = 利用 + 探索 + 冷却三项之和
            fitness = float(exploit_coef[idx] + explore_coef[idx] + cool_norm[idx])
            # 取该节点的子代方差原始值（可能为 None）
            variance_value = learning_raw.get(pid)
            if variance_value is None and not self._child_variance_nullable:
                # 列不可空且无子代时用中性占位 0.5
                variance_value = 0.5
            # 更新该节点的 fitness、各系数、访问次数与子代方差
            cursor.execute(
                """
                UPDATE programs
                SET fitness = ?,
                    exploit_coefficient = ?,
                    explore_coefficient = ?,
                    visit_count = ?,
                    child_base_reward_variance = ?,
                    cooling_coefficient = ?
                WHERE id = ?
                """,
                (
                    fitness,
                    float(exploit_coef[idx]),
                    float(explore_coef[idx]),
                    int(visit_counts.get(pid, 0)),
                    None if variance_value is None else float(variance_value),
                    float(cooling_raw.get(pid, 1.0)),
                    pid,
                ),
            )

    def get_by_id(self, program_id: int) -> Optional[Program]:
        """Get a program by its ID."""
        # 获取线程本地连接与游标
        conn = self._get_connection()
        cursor = conn.cursor()

        # 按主键查询程序行
        cursor.execute("SELECT * FROM programs WHERE id = ?", (program_id,))
        # 取一行结果
        row = cursor.fetchone()

        if row:
            # 命中时反序列化为 Program 返回
            return Program.from_dict(dict(row))
        # 未命中返回 None
        return None

    def is_empty(self, task_name: str = None, task_id: str = None) -> bool:
        """Check if database is empty for a task.

        Args:
            task_name: Task name to check (deprecated, use task_id instead)
            task_id: Task ID to check
        """
        # 获取连接与游标
        conn = self._get_connection()
        cursor = conn.cursor()

        if task_id is not None:
            # 优先按 task_id 统计该任务程序数
            cursor.execute("SELECT COUNT(*) as count FROM programs WHERE task_id = ?", (task_id,))
        elif task_name is not None:
            # 兼容旧接口按 task_name 统计
            cursor.execute("SELECT COUNT(*) as count FROM programs WHERE task_name = ?", (task_name,))
        else:
            # 两者都未提供时报错
            raise ValueError("Either task_id or task_name must be provided")

        # 取出计数值
        count = cursor.fetchone()["count"]
        # 计数为 0 即任务为空
        return count == 0

    def size(self, task_name: str) -> int:
        """Get the number of programs for a task."""
        # 获取连接与游标
        conn = self._get_connection()
        cursor = conn.cursor()

        # 统计指定任务的程序数
        cursor.execute("SELECT COUNT(*) as count FROM programs WHERE task_name = ?", (task_name,))
        # 返回计数值
        return cursor.fetchone()["count"]

    def clear(self, task_name: Optional[str] = None):
        """Clear programs from the database.

        Args:
            task_name: If provided, only clear programs for this task.
                      If None, clear all programs.
        """
        with self._lock:
            # 加锁后获取连接与游标，保证删除原子性
            conn = self._get_connection()
            cursor = conn.cursor()

            if task_name:
                # 指定任务名时仅删除该任务的程序
                cursor.execute("DELETE FROM programs WHERE task_name = ?", (task_name,))
            else:
                # 未指定时清空全表
                cursor.execute("DELETE FROM programs")

            # 提交事务
            conn.commit()

    def close(self):
        """Close database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            # 关闭当前线程的连接
            self._local.conn.close()
            # 置空引用，下次访问时重新建立
            self._local.conn = None

    def get_all(self, task_name: str) -> list[Program]:
        # 按任务名取全部程序（按插入顺序），反序列化为 Program 列表
        conn = self._get_connection()
        cursor = conn.cursor()
        # 查询该任务下全部程序行
        cursor.execute(
            """
            SELECT * FROM programs
            WHERE task_name = ?
            ORDER BY id ASC
            """,
            (task_name,),
        )
        # 取出所有行
        rows = cursor.fetchall()
        # 逐行反序列化为 Program 对象
        return [Program.from_dict(dict(row)) for row in rows]

    def sample_by_fitness(
        self, task_name: str, sample_size: int = 1, exclude_ids: Optional[set[int]] = None
    ) -> list[Program]:
        # 按任务名取全部候选程序
        candidates = self.get_all(task_name)
        if exclude_ids:
            # 排除指定的程序 id
            candidates = [p for p in candidates if p.id is not None and p.id not in exclude_ids]
        if not candidates or sample_size <= 0:
            # 无候选或采样数为非正时返回空
            return []

        # 已选中的程序列表（无放回）
        selected: list[Program] = []
        # 复制候选池，逐轮移除已选
        pool = list(candidates)
        # 实际采样数不超过池大小
        k = min(sample_size, len(pool))
        # 逐轮 fitness 加权无放回采样
        for _ in range(k):
            # 以 fitness（下界钳 0）作为权重
            weights = [max(self._safe_float(p.fitness, default=0.0), 0.0) for p in pool]
            if sum(weights) <= 0.0:
                # 全零权重（无有效 fitness）时退化为均匀随机
                idx = random.randrange(len(pool))
            else:
                # 按权重轮盘赌抽取一个下标
                idx = random.choices(range(len(pool)), weights=weights, k=1)[0]
            # 移出池并记录选中者
            selected.append(pool.pop(idx))
        # 返回采样结果
        return selected

    def get_all_base_rewards(self, *, include_hack: bool = False) -> list[float]:
        """
        Fetch base_reward for all programs in the DB.

        NOTE: This returns the stored `base_reward` field (i.e., the score->reward mapped value
        before any "improve" shaping). It is useful for computing global baselines / percentiles.

        Args:
            include_hack: If False, exclude programs marked as hacking (hack=1).
        """
        with self._lock:
            # 加锁后获取连接与游标
            conn = self._get_connection()
            cursor = conn.cursor()
            if include_hack:
                # 包含作弊程序时查全部 base_reward
                cursor.execute("SELECT base_reward FROM programs")
            else:
                # 默认排除作弊（hack=1）程序的 base_reward
                cursor.execute("SELECT base_reward FROM programs WHERE hack = 0")
            # 取出所有行
            rows = cursor.fetchall()

        # base_reward column is NOT NULL by schema, but be defensive.
        # 逐行防御性解析 base_reward
        vals: list[float] = []
        for r in rows:
            try:
                # 兼容 sqlite3.Row 与普通元组两种取值方式
                v = r["base_reward"] if isinstance(r, sqlite3.Row) else r[0]
            except Exception:
                # 取值异常按 None 处理
                v = None
            if v is None:
                # 空值跳过
                continue
            try:
                # 转为浮点数并收集
                vals.append(float(v))
            except Exception:
                # 转换失败跳过
                continue
        # 返回全部 base_reward
        return vals

    def save_snapshot(self, snapshot_path: str):
        """
        Create a snapshot copy of the database.

        Args:
            snapshot_path: Path where the snapshot should be saved
        """
        # 延迟导入 shutil：文件复制用
        import shutil

        with self._lock:
            # Ensure all pending transactions are committed
            # 加锁并提交所有挂起事务，保证快照一致性
            conn = self._get_connection()
            conn.commit()

            # Create directory if needed
            # 确保快照目标目录存在
            os.makedirs(
                os.path.dirname(os.path.abspath(snapshot_path)) if os.path.dirname(snapshot_path) else ".",
                exist_ok=True,
            )

            # Copy the database file
            # 复制数据库文件（保留元数据）作为快照
            shutil.copy2(self.db_path, snapshot_path)
            # 打印快照保存信息
            print(f"[ProgramDatabase] Saved snapshot to {snapshot_path}")


class SearchAlgorithm:
    """Base class for search algorithms to select programs from database."""

    def __init__(self, draft_probability: float = 0.5):
        """
        Initialize search algorithm.

        Args:
            draft_probability: Probability of using draft mode instead of improve mode
        """
        # 记录 draft（起草）模式的触发概率
        self.draft_probability = draft_probability

    def select(
        self,
        database: ProgramDatabase,
        task_id: str,
        task_description: str,
        data_description: str,
        public_system_prompt: str,
        public_user_prompt: str,
        data_dir: str,
        max_steps: int = 1,
        task_name: str = None,
        evaluation: bool = False,
    ) -> tuple[tuple[str, str] | None, Optional[Program], str, Optional[Program]]:
        """
        Select a program and build prompt.

        This method decides whether to use draft or improve mode, selects a parent program
        if needed, and builds the appropriate prompt.

        Args:
            database: The ProgramDatabase to select from
            task_id: The task ID to select programs for
            task_description: Task description for prompt building
            data_description: Data description for prompt building
            public_system_prompt: Existing system prompt
            public_user_prompt: Existing user prompt
            data_dir: Virtual data directory path
            max_steps: Maximum number of steps
            task_name: Task name (optional, for logging purposes)
            evaluation: If True, force draft mode (draft_probability=1.0)

        Returns:
            Tuple of (prompts, parent_program, mode, secondary_parent_program)
            - prompts: Tuple of (system_prompt, user_prompt)
            - parent_program: Selected parent Program or None if draft mode
            - mode: One of 'draft' | 'improve' | 'debug' | 'crossover'
            - secondary_parent_program: Used only in crossover mode
        """
        # 基类接口：由子类实现具体的算子选择与提示构建逻辑
        raise NotImplementedError


class GreedySearch(SearchAlgorithm):
    """Greedy search: always select the program with highest reward."""

    def _select_parent(self, database: ProgramDatabase, task_id: str) -> Optional[Program]:
        """Select the program with the highest reward for a given task_id."""
        # 直接访问数据库连接，按 reward 降序取第一名
        conn = database._get_connection()
        cursor = conn.cursor()

        # 查询该任务下 reward 最高的程序
        cursor.execute(
            """
            SELECT * FROM programs
            WHERE task_id = ?
            ORDER BY reward DESC
            LIMIT 1
        """,
            (task_id,),
        )

        # 取一行结果
        row = cursor.fetchone()
        if row:
            # 命中时反序列化为 Program 返回
            return Program.from_dict(dict(row))
        # 未命中返回 None
        return None

    def select(
        self,
        database: ProgramDatabase,
        task_id: str,
        task_description: str,
        data_description: str,
        public_system_prompt: str,
        public_user_prompt: str,
        data_dir: str,
        max_steps: int = 1,
        task_name: str = None,
        evaluation: bool = False,
    ) -> tuple[tuple[str, str] | None, Optional[Program], str, Optional[Program]]:
        """
        Select best program and decide draft vs improve mode.

        Args:
            database: The ProgramDatabase to select from
            task_id: The task ID to select programs for
            task_description: Task description for prompt building
            data_description: Data description for prompt building
            public_system_prompt: Existing system prompt
            public_user_prompt: Existing user prompt
            data_dir: Virtual data directory path
            max_steps: Maximum number of steps
            task_name: Task name (optional, for logging purposes)
            evaluation: If True, force draft mode (draft_probability=1.0)

        Returns:
            Tuple of (prompt, parent_program, mode)
        """

        # Decide whether to use draft or improve mode
        # 初始化：默认走 draft（起草）模式
        use_improve = False
        parent_program = None

        # Force draft mode during evaluation
        # 评估阶段强制使用原始 draft 提示
        if evaluation:
            # 日志用任务名（缺省用 task_id）
            log_task = task_name if task_name else task_id
            # 模式固定为 draft
            mode = "draft"
            # 打印评估模式提示
            print(f"[SEARCH] Using RAW EVAL prompt for task {log_task} (task_id={task_id}) [EVALUATION MODE]")
            # 直接返回原始公共提示，不选父代
            return (public_system_prompt, public_user_prompt), None, mode, None
        # Only choose between draft and improve if task_id exists in database
        elif not database.is_empty(task_id=task_id):
            # Database has programs for this task_id, decide whether to improve or draft
            # 库内有该任务的程序时按概率决定是否走 improve（改进）模式
            if random.random() > self.draft_probability:  # (1 - draft_prob) chance to use improve
                # 贪心选择 reward 最高的程序作为父代
                parent_program = self._select_parent(database, task_id)
                if parent_program is not None:
                    # 成功选到父代则切换为 improve 模式
                    use_improve = True
                    # 日志用任务名（缺省用 task_id）
                    log_task = task_name if task_name else task_id
                    print(
                        f"[SEARCH] Using IMPROVE mode for task {log_task} (task_id={task_id}, parent_id={parent_program.id}, parent_reward={parent_program.reward:.4f})"
                    )

        if not use_improve and not evaluation:
            # 未走 improve 且非评估时，打印 draft 模式日志
            log_task = task_name if task_name else task_id
            print(f"[SEARCH] Using DRAFT mode for task {log_task} (task_id={task_id})")

        # Build prompt based on mode
        # 依据是否 improve 决定最终模式字符串
        mode = "improve" if use_improve else "draft"
        # 延迟导入提示构建器，避免循环依赖
        from prompt_builder import build_prompt

        # 依据模式构建（系统提示, 用户提示）
        prompt = build_prompt(
            mode=mode,
            task_description=task_description,
            data_description=data_description,
            public_system_prompt=public_system_prompt,
            public_user_prompt=public_user_prompt,
            virtual_data_dir=data_dir,
            parent_program=parent_program,
            max_steps=max_steps,
        )

        # 返回提示、父代、模式与次级父代（贪心搜索无交叉，恒为 None）
        return prompt, parent_program, mode, None


class AIRAGreedySearch(SearchAlgorithm):
    def __init__(self, draft_probability: float = 0.5, debug_probability: float = 0.2):
        # 记录 draft 模式概率（None/0 归一为 0.0）
        self.draft_prob = draft_probability or 0.0
        # 记录 debug（调试）模式概率（None/0 归一为 0.0）
        self.debug_prob = debug_probability or 0.0
        # 校验：draft 与 debug 概率之和不得超过 1（其余留给 improve）
        assert self.draft_prob + self.debug_prob <= 1.0

    @staticmethod
    def _best_scored_parent(database: ProgramDatabase, task_name: str) -> Program | None:
        # 静态工具：取任务下 reward>0（有分）的候选中 fitness 最高者
        cands = [p for p in database.get_all(task_name) if float(p.reward or 0.0) > 0.0]
        if not cands:
            # 无有分候选时返回 None
            return None
        # 按 fitness 取最优
        return max(cands, key=lambda p: float(p.fitness or 0.0))

    @staticmethod
    def _random_debug_parent(database: ProgramDatabase, task_name: str) -> Program | None:
        # 静态工具：从任务下 reward<=0（失败/零分）候选中随机取一个作为调试对象
        cands = [p for p in database.get_all(task_name) if float(p.reward or 0.0) <= 0.0]
        if not cands:
            # 无失败候选时返回 None
            return None
        # 均匀随机选一个
        return random.choice(cands)

    def select(
        self,
        database: ProgramDatabase,
        task_id: str,
        task_description: str,
        data_description: str,
        public_system_prompt: str,
        public_user_prompt: str,
        data_dir: str,
        max_steps: int = 1,
        task_name: str = None,
        evaluation: bool = False,
    ) -> tuple[tuple[str, str] | None, Program | None, str, Program | None]:
        # The first solution must always be a draft.
        # Also force draft mode during evaluation
        # 评估阶段强制 draft 原始提示
        if evaluation:
            # 模式固定 draft，无父代
            mode = "draft"
            parent_program = None
            # 打印评估模式日志
            logger.info("[SEARCH] Using RAW EVAL prompt for task %s [EVALUATION MODE]", task_name)
            # 直接返回原始公共提示
            return (public_system_prompt, public_user_prompt), parent_program, mode, None

        if database.is_empty(task_id=task_id):
            # 任务为空（首个解）时必须走 draft
            mode = "draft"
            parent_program = None
            logger.info("[SEARCH] Using DRAFT mode for task %s", task_name)
        else:
            # 库内已有程序：先随机采样初始动作，再按兜底链修正
            mode = "draft"
            parent_program = None

            # Sample initial action first, then apply fallback chain.
            # Probabilities: draft=self.draft_prob, debug=self.debug_prob, improve=remaining.
            # 三算子概率分配：draft / debug / improve（剩余概率）
            sample = random.random()
            if sample < self.draft_prob:
                # 落在 draft 概率区间
                mode = "draft"
            elif sample < self.draft_prob + self.debug_prob:
                # 落在 debug 概率区间
                mode = "debug"
            else:
                # 其余概率走 improve
                mode = "improve"

            if mode == "improve":
                # improve：选 fitness 最高的有分父代
                parent_program = self._best_scored_parent(database, task_name)
                if parent_program is None:
                    # Requested fallback: improve -> debug.
                    # 兜底链：improve 找不到有分父代时退回 debug
                    parent_program = self._random_debug_parent(database, task_name)
                    # debug 候选也缺失时退回 draft
                    mode = "debug" if parent_program is not None else "draft"
            elif mode == "debug":
                # debug：随机选失败候选
                parent_program = self._random_debug_parent(database, task_name)
                if parent_program is None:
                    # Requested fallback: debug -> improve.
                    # 兜底链：debug 找不到失败候选时退回 improve
                    parent_program = self._best_scored_parent(database, task_name)
                    # improve 候选也缺失时退回 draft
                    mode = "improve" if parent_program is not None else "draft"

            if parent_program is not None:
                # 选到父代时打印所选模式与父代信息
                logger.info(
                    "[SEARCH] Using %s mode for task %s (parent_id=%s, parent_fitness=%.4f)",
                    mode.upper(),
                    task_name,
                    parent_program.id,
                    float(parent_program.fitness),
                )
            else:
                # 无父代（draft）时打印日志
                logger.info("[SEARCH] Using DRAFT mode for task %s", task_name)
        # 延迟导入提示构建器
        from prompt_builder import build_prompt

        # 依据最终模式构建提示
        prompt = build_prompt(
            mode=mode,
            task_description=task_description,
            data_description=data_description,
            public_system_prompt=public_system_prompt,
            public_user_prompt=public_user_prompt,
            virtual_data_dir=data_dir,
            parent_program=parent_program,
            max_steps=max_steps,
        )
        # 返回提示、父代、模式与次级父代（无交叉，恒为 None）
        return prompt, parent_program, mode, None


class AIRAEvoSearch(SearchAlgorithm):
    """Evolution-style search with fitness-proportional sampling and optional crossover."""

    def __init__(
        self,
        draft_probability: float = 0.5,
        improve_probability: float = 0.25,
        debug_probability: float = 0.25,
        crossover_probability: float = 0.5,
    ):
        # 四算子各自的触发概率：draft（起草）
        self.draft_prob = max(float(draft_probability or 0.0), 0.0)
        # improve（改进）概率
        self.improve_prob = max(float(improve_probability or 0.0), 0.0)
        # debug（调试）概率
        self.debug_prob = max(float(debug_probability or 0.0), 0.0)
        # crossover（交叉）概率
        self.crossover_prob = max(float(crossover_probability or 0.0), 0.0)
        # 交叉概率大于 0 时启用交叉算子
        self.enable_crossover = self.crossover_prob > 0.0

    @staticmethod
    def _weighted_sample_without_replacement(candidates: list[Program], k: int) -> list[Program]:
        # 静态工具：按 fitness 加权（轮盘赌）无放回抽取 k 个候选
        if k <= 0 or not candidates:
            # 采样数非正或候选为空时返回空
            return []
        # 复制候选池
        pool = list(candidates)
        # 已选中列表
        selected: list[Program] = []
        # 最多抽 min(k, 池大小) 轮
        for _ in range(min(k, len(pool))):
            # 以 fitness（下界钳 0）为权重
            weights = [max(float(p.fitness or 0.0), 0.0) for p in pool]
            # 权重总和
            total_weight = sum(weights)
            if total_weight <= 0.0:
                # 全零权重时退化为均匀随机
                idx = random.randrange(len(pool))
                # 选中概率为均匀分布
                selected_prob = 1.0 / float(len(pool))
                # 各候选概率均等
                probs = [selected_prob for _ in pool]
            else:
                # 按权重轮盘赌抽取
                idx = random.choices(range(len(pool)), weights=weights, k=1)[0]
                # 计算各候选被抽中的概率
                probs = [w / total_weight for w in weights]
                # 记录选中者的概率
                selected_prob = probs[idx]

            # Emit sampling diagnostics for rollout debugging.
            # 输出采样诊断信息，便于调试 rollout
            prob_rows = []
            # 把池中每个程序与其概率配对
            for i, prog in enumerate(pool):
                prob_rows.append((prog, probs[i]))
            # 按概率降序排序
            prob_rows_sorted = sorted(prob_rows, key=lambda x: x[1], reverse=True)
            # 拼出概率前五的候选描述
            top5 = ", ".join(
                f"id={p.id},fitness={float(p.fitness or 0.0):.6f},prob={prob:.6f}" for p, prob in prob_rows_sorted[:5]
            )

            # 取出本轮选中者
            chosen = pool[idx]
            # 记录选中者与 top5 概率分布
            logger.info(
                "[EVO SAMPLE] selected id=%s fitness=%.6f prob=%.6f | top5_probs=[%s]",
                chosen.id,
                float(chosen.fitness or 0.0),
                float(selected_prob),
                top5,
            )
            # 移出池并记录选中者（无放回）
            selected.append(pool.pop(idx))
        # 返回采样结果
        return selected

    def _sample_improve_parent(self, database: ProgramDatabase, task_name: str) -> Program | None:
        # Improve/crossover should sample from "scored" nodes only.
        # improve 算子只从"有分"（reward>0）节点中加权采样
        all_nodes = database.get_all(task_name)
        # 过滤出有分候选
        improve_candidates = [p for p in all_nodes if float(p.reward or 0.0) > 0.0]
        # 加权无放回抽取 1 个
        picked = self._weighted_sample_without_replacement(improve_candidates, k=1)
        # 返回选中的父代（可能为 None）
        return picked[0] if picked else None

    def _sample_crossover_parents(
        self, database: ProgramDatabase, task_name: str
    ) -> tuple[Program | None, Program | None]:
        # crossover 算子：从有分节点中加权无放回抽取两个父代
        all_nodes = database.get_all(task_name)
        # 过滤出有分候选
        improve_candidates = [p for p in all_nodes if float(p.reward or 0.0) > 0.0]
        # 加权无放回抽取 2 个
        picked = self._weighted_sample_without_replacement(improve_candidates, k=2)
        if len(picked) != 2:
            # 候选不足两个时返回空对
            return None, None
        # 返回两个交叉父代
        return picked[0], picked[1]

    def _sample_debug_parent(self, database: ProgramDatabase, task_name: str) -> Program | None:
        # Debug should prefer "no reward" parents.
        # debug 算子只从"零分/失败"（reward<=0）节点中加权采样
        # print(1111)
        all_nodes = database.get_all(task_name)
        # 过滤出失败候选
        debug_candidates = [p for p in all_nodes if float(p.reward or 0.0) <= 0.0]
        if not debug_candidates:
            # 无失败候选时返回 None
            return None
        # 加权无放回抽取 1 个
        picked = self._weighted_sample_without_replacement(debug_candidates, k=1)
        # 返回选中的调试对象（可能为 None）
        return picked[0] if picked else None

    def select(
        self,
        database: ProgramDatabase,
        task_id: str,
        task_description: str,
        data_description: str,
        public_system_prompt: str,
        public_user_prompt: str,
        data_dir: str,
        max_steps: int = 1,
        task_name: str = None,
        evaluation: bool = False,
    ) -> tuple[tuple[str, str] | None, Program | None, str, Program | None]:
        # 评估阶段强制 draft 原始提示
        if evaluation:
            # 模式固定 draft，无父代
            mode = "draft"
            parent_program = None
            secondary_parent = None
            logger.info("[EVO SEARCH] Using RAW EVAL prompt for task %s [EVALUATION MODE]", task_name)
            # 直接返回原始公共提示
            return (public_system_prompt, public_user_prompt), parent_program, mode, secondary_parent

        if database.is_empty(task_id=task_id):
            # 任务为空（首个解）时必须走 draft
            mode = "draft"
            parent_program = None
            secondary_parent = None
            logger.info("[EVO SEARCH] Using DRAFT mode for task %s", task_name)
        else:
            # 库内已有程序：初始化为 draft，再按四算子概率采样
            mode = "draft"
            parent_program = None
            secondary_parent = None

            # 候选动作列表：draft/improve/debug 三算子
            actions = ["draft", "improve", "debug"]
            # 对应权重
            weights = [self.draft_prob, self.improve_prob, self.debug_prob]
            if self.enable_crossover:
                # 启用交叉时追加 crossover 动作与权重
                actions.append("crossover")
                weights.append(self.crossover_prob)
            if sum(weights) <= 0.0:
                # Keep a safe fallback when all configured probabilities are zero.
                # 全部概率为 0 时的安全兜底：均匀分布
                actions = ["draft", "improve", "debug"] + (["crossover"] if self.enable_crossover else [])
                weights = [1.0 for _ in actions]
            # 按权重随机抽取本次算子
            mode = random.choices(actions, weights=weights, k=1)[0]

            if mode == "improve":
                # improve：从有分节点加权采样父代
                parent_program = self._sample_improve_parent(database, task_name)
                if parent_program is None:
                    # Fallback rule: improve -> debug (if possible), else draft.
                    # 兜底链：improve 退回 debug，再退回 draft
                    parent_program = self._sample_debug_parent(database, task_name)
                    mode = "debug" if parent_program is not None else "draft"
            elif mode == "crossover":
                # crossover：抽取两个交叉父代
                parent_program, secondary_parent = self._sample_crossover_parents(database, task_name)
                if parent_program is None or secondary_parent is None:
                    # Fallback rule for crossover: first try improve, then debug, then draft.
                    # 交叉兜底链：先试 improve，再试 debug，最后 draft
                    parent_program = self._sample_improve_parent(database, task_name)
                    # 交叉必然失效，清空次级父代
                    secondary_parent = None
                    if parent_program is not None:
                        # 有分父代可用时降级为 improve
                        mode = "improve"
                    else:
                        # 再试 debug
                        parent_program = self._sample_debug_parent(database, task_name)
                        mode = "debug" if parent_program is not None else "draft"
            elif mode == "debug":
                # debug：从失败节点加权采样
                parent_program = self._sample_debug_parent(database, task_name)
                if parent_program is None:
                    # Fallback rule: debug -> improve (if possible), else draft.
                    # 兜底链：debug 退回 improve，再退回 draft
                    parent_program = self._sample_improve_parent(database, task_name)
                    mode = "improve" if parent_program is not None else "draft"

            if mode == "debug" and parent_program is not None:
                # 打印 debug 模式与父代信息
                logger.info(
                    "[EVO SEARCH] Using DEBUG mode for task %s (parent_id=%s, parent_fitness=%.4f)",
                    task_name,
                    parent_program.id,
                    float(parent_program.fitness),
                )
            elif mode == "improve" and parent_program is not None:
                # 打印 improve 模式与父代信息
                logger.info(
                    "[EVO SEARCH] Using IMPROVE mode for task %s (parent_id=%s, parent_fitness=%.4f)",
                    task_name,
                    parent_program.id,
                    float(parent_program.fitness),
                )
            elif mode == "crossover" and parent_program is not None and secondary_parent is not None:
                # 打印 crossover 模式与双父代信息
                logger.info(
                    "[EVO SEARCH] Using CROSSOVER mode for task %s (parent_ids=%s,%s)",
                    task_name,
                    parent_program.id,
                    secondary_parent.id,
                )
            else:
                # 其余情况按 draft 打印日志
                logger.info("[EVO SEARCH] Using DRAFT mode for task %s", task_name)

        # 延迟导入提示构建器
        from prompt_builder import build_prompt

        # 依据最终模式构建提示
        prompt = build_prompt(
            mode=mode,
            task_description=task_description,
            data_description=data_description,
            public_system_prompt=public_system_prompt,
            public_user_prompt=public_user_prompt,
            virtual_data_dir=data_dir,
            parent_program=parent_program,
            secondary_parent_program=secondary_parent,
            max_steps=max_steps,
        )
        # 返回提示、父代、模式与次级父代
        return prompt, parent_program, mode, secondary_parent


class AIRAInferenceEvoSearch(SearchAlgorithm):
    """AIRA-Evo search adapted to the RL ProgramDatabase interface."""

    def __init__(
        self,
        policy: str = "rl_mixed",
        num_islands: int = 1,
        max_island_size: int = 500,
        crossover_after_generation: int = 2,
        crossover_probability: float = 0.5,
        execution_timeout: int = 7200,
        initial_temp: float = 1.0,
        final_temp: float = 1.0,
    ):
        # 归一化策略名：转小写去空白
        self.policy = str(policy or "rl_mixed").strip().lower()
        if self.policy not in {"rl_mixed", "original"}:
            # 非法策略名回退为 rl_mixed
            self.policy = "rl_mixed"
        # 岛（island）数量，至少为 1
        self.num_islands = max(1, int(num_islands or 1))
        # 单岛/活跃种群容量上限，至少为 1
        self.max_island_size = max(1, int(max_island_size or 500))
        # 允许交叉算子的最小代数，非负
        self.crossover_after_generation = max(0, int(crossover_after_generation or 0))
        # 交叉概率钳制到 [0,1]
        self.crossover_probability = max(0.0, min(1.0, float(crossover_probability or 0.0)))
        # 沙箱执行超时（秒），缺省 7200
        self.execution_timeout = int(execution_timeout or 7200)
        # 采样初始温度
        self.initial_temp = float(initial_temp or 1.0)
        # 采样终止温度（缺省与初始一致）
        self.final_temp = float(final_temp or self.initial_temp)
        # 各任务的代数计数器（draft 也计一代）
        self._task_generation: dict[str, int] = {}
        # 最近一次选择的元数据（供 RL 侧记录轨迹）
        self.last_selection_metadata: dict[str, Any] = {}

    def _all_programs(self, database: ProgramDatabase, task_name: str) -> list[Program]:
        # 取任务下全部程序节点
        return database.get_all(task_name)

    def _active_success_programs(self, programs: list[Program]) -> list[Program]:
        # 筛选"活跃成功"节点：执行成功且有分数的候选种群
        active = [program for program in programs if is_success_program(program)]
        # 按选择得分（次键 id）降序排序，得分高者优先
        active.sort(key=lambda program: (program_selection_score(program) or float("-inf"), int(program.id or 0)), reverse=True)
        if len(active) > self.max_island_size:
            # 超出种群容量时截断（保留得分最高的前 max_island_size 个）
            active = active[: self.max_island_size]
        # 返回活跃种群
        return active

    def _debug_programs(self, programs: list[Program]) -> list[Program]:
        # 筛选 debug 候选（失败/待修复节点），按 id 降序（越新越优先）
        debug = [program for program in programs if is_debug_candidate(program)]
        # 按 id 降序排序
        debug.sort(key=lambda program: int(program.id or 0), reverse=True)
        # 截断到容量上限
        return debug[: self.max_island_size]

    def _temperature(self, generation_id: int) -> float:
        # 采样温度调度：当前实现恒用初始温度（预留代数相关退火接口）
        _ = generation_id
        return self.initial_temp if self.initial_temp == self.final_temp else self.initial_temp

    def _island_id(self, program: Program) -> int:
        # 计算程序所属的岛编号：单岛恒为 0，多岛按 id 取模分配
        if self.num_islands <= 1:
            # 单岛模式
            return 0
        # 按 id 对岛数取模
        return int(program.id or 0) % self.num_islands

    @staticmethod
    def _weighted_pick(candidates: list[Program], utility_items: list[dict[str, Any]], k: int) -> tuple[list[Program], list[dict[str, Any]]]:
        # 静态工具：按效用条目中的 probability 权重无放回抽取 k 个（程序, 效用条目）对
        pool = list(zip(candidates, utility_items))
        # 已选程序列表
        selected: list[Program] = []
        # 已选效用条目列表
        selected_items: list[dict[str, Any]] = []
        # 最多抽 min(k, 池大小) 轮
        for _ in range(min(k, len(pool))):
            # 以条目中的 probability（下界钳 0）为权重
            weights = [max(float(item.get("probability") or 0.0), 0.0) for _, item in pool]
            # 权重总和
            total = sum(weights)
            if total <= 0.0:
                # 全零权重时退化为均匀随机
                idx = random.randrange(len(pool))
            else:
                # 按权重轮盘赌抽取
                idx = random.choices(range(len(pool)), weights=weights, k=1)[0]
            # 弹出选中的（程序, 条目）对
            program, item = pool.pop(idx)
            # 记录程序
            selected.append(program)
            # 记录条目
            selected_items.append(item)
        # 返回选中的程序与对应条目
        return selected, selected_items

    def _sample_by_utility(
        self,
        candidates: list[Program],
        all_programs: list[Program],
        k: int,
        temperature: float,
    ) -> tuple[list[Program], dict[str, Any]]:
        # 按父代效用（AIRA-Evo 的 utility 计算含利用/探索/新鲜度等信号）加权采样
        if not candidates:
            # 无候选时返回空与占位 trace
            return [], {"enabled": True, "candidates": [], "selected_node_ids": []}
        # 调用经验系统计算每个候选的效用与采样概率
        utilities = compute_parent_utilities(
            candidates,
            all_programs=all_programs,
            lower_is_better=False,
            temperature=temperature,
        )
        # 按概率加权抽取 k 个
        selected, selected_items = self._weighted_pick(candidates, utilities, k)
        # 构建选择 trace（记录温度、候选效用、选中节点）
        trace = {
            "enabled": True,
            "temperature": temperature,
            "candidates": utilities[:20],
            "selected_node_ids": [program.id for program in selected],
            "selected_items": selected_items,
        }
        # 返回选中列表与 trace
        return selected, trace

    def _sample_original(
        self,
        active: list[Program],
        all_programs: list[Program],
        mode: str,
        temperature: float,
    ) -> tuple[list[Program], int, dict[str, Any]]:
        # original 策略：先按岛均分选择岛，再在岛内按效用采样
        # 交叉需要 2 个父代，其余算子 1 个
        required = 2 if mode == "crossover" else 1
        if not active or len(active) < required:
            # 活跃种群不足时返回空与原因
            return [], 0, {"enabled": True, "reason": "not_enough_active_nodes"}
        # 按岛编号初始化分岛容器
        islands: dict[int, list[Program]] = {idx: [] for idx in range(self.num_islands)}
        # 把活跃节点分配到各岛
        for program in active:
            islands[self._island_id(program)].append(program)
        # 保留节点数满足需求的岛
        island_ids = [idx for idx, nodes in islands.items() if len(nodes) >= required]
        if not island_ids:
            # 无任何岛有足够节点时返回空与原因
            return [], 0, {"enabled": True, "reason": "no_island_with_enough_nodes"}
        # 计算每个候选岛的均分
        island_scores = []
        for island_id in island_ids:
            # 岛内各节点选择得分的均值
            scores = [program_selection_score(program) or 0.0 for program in islands[island_id]]
            island_scores.append(sum(scores) / float(len(scores)))
        # 岛均分的最小值
        min_score = min(island_scores)
        # 岛均分的最大值
        max_score = max(island_scores)
        if min_score == max_score:
            # 各岛无差异时等权
            weights = [1.0 for _ in island_scores]
        else:
            # 按岛均分线性归一化为权重（好岛更可能被选中）
            weights = [(score - min_score) / (max_score - min_score) for score in island_scores]
        # 按权重随机选岛（权重全零时均匀选）
        island_id = random.choices(island_ids, weights=weights, k=1)[0] if sum(weights) > 0 else random.choice(island_ids)
        # 在选中岛内按效用采样所需数量的父代
        selected, trace = self._sample_by_utility(islands[island_id], all_programs, required, temperature)
        # 在 trace 中补充岛级采样信息
        trace["island_sampling"] = {"candidate_island_ids": island_ids, "weights": weights, "selected_island_id": island_id}
        # 返回选中父代、岛编号与 trace
        return selected, island_id, trace

    def _sample_rl_mixed(self, mode: str, programs: list[Program], active: list[Program], temperature: float) -> tuple[Program | None, Program | None, int, dict[str, Any]]:
        # rl_mixed 策略：按算子类型从不同候选池做效用加权采样
        if mode == "debug":
            # debug：从失败候选中采 1 个（不需要岛信息）
            candidates = self._debug_programs(programs)
            selected, trace = self._sample_by_utility(candidates, programs, 1, temperature)
            # 返回单个调试父代与空岛编号
            return (selected[0] if selected else None), None, 0, trace
        if mode == "crossover":
            # crossover：从活跃种群中采 2 个父代
            selected, trace = self._sample_by_utility(active, programs, 2, temperature)
            # 岛编号取首父代所属岛
            island_id = self._island_id(selected[0]) if selected else 0
            if len(selected) == 2:
                # 采满两个时返回双父代
                return selected[0], selected[1], island_id, trace
            # 不足两个时返回空对
            return None, None, island_id, trace
        if mode == "improve":
            # improve：从活跃种群中采 1 个父代
            selected, trace = self._sample_by_utility(active, programs, 1, temperature)
            # 岛编号取父代所属岛
            island_id = self._island_id(selected[0]) if selected else 0
            # 返回单个父代
            return (selected[0] if selected else None), None, island_id, trace
        # draft 不需要采样
        return None, None, 0, {"enabled": False}

    @staticmethod
    def _choose_rl_mixed_mode() -> str:
        # rl_mixed 的固定算子分布：draft 0.5，improve/debug/crossover 各 1/6
        return random.choices(
            ["draft", "improve", "debug", "crossover"],
            weights=[0.5, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
            k=1,
        )[0]

    def _fallback_mode(
        self,
        requested_mode: str,
        programs: list[Program],
        active: list[Program],
        temperature: float,
        include_requested: bool = True,
    ) -> tuple[str, Program | None, Program | None, int, dict[str, Any]]:
        # 兜底链：请求的算子采样失败时按优先级依次尝试其他算子
        # 各请求算子的尝试顺序表
        search_order = {
            "crossover": ["crossover", "improve", "debug", "draft"],
            "improve": ["improve", "debug", "draft"],
            "debug": ["debug", "improve", "draft"],
            "draft": ["draft"],
        }.get(requested_mode, ["draft"])
        # 不含请求算子时从第二个开始尝试
        fallback_order = search_order if include_requested else search_order[1:] or ["draft"]
        # 依次尝试每个候选算子
        for mode in fallback_order:
            if mode == "draft":
                # draft 无需采样，直接成功（返回"未启用采样"的 trace）
                return "draft", None, None, 0, {"enabled": False, "fallback_from": requested_mode}
            # 尝试按该算子采样父代
            parent, secondary, island_id, trace = self._sample_rl_mixed(mode, programs, active, temperature)
            if mode == "crossover" and parent is not None and secondary is not None:
                # 交叉成功需要双父代
                trace["fallback_from"] = requested_mode
                return mode, parent, secondary, island_id, trace
            if mode in {"improve", "debug"} and parent is not None:
                # improve/debug 成功需要单父代
                trace["fallback_from"] = requested_mode
                return mode, parent, None, island_id, trace
        # 所有兜底都失败时强制走 draft
        return "draft", None, None, 0, {"enabled": False, "fallback_from": requested_mode}

    def select(
        self,
        database: ProgramDatabase,
        task_id: str,
        task_description: str,
        data_description: str,
        public_system_prompt: str,
        public_user_prompt: str,
        data_dir: str,
        max_steps: int = 1,
        task_name: str = None,
        evaluation: bool = False,
    ) -> tuple[tuple[str, str] | None, Program | None, str, Program | None]:
        # 任务主键：优先任务名，其次任务 ID
        task_key = task_name or task_id or "unknown"
        # 读取该任务当前代数（draft 也会推进代数）
        generation_id = int(self._task_generation.get(task_key, 0))
        # 重置最近一次选择元数据
        self.last_selection_metadata = {}
        if evaluation:
            # 评估阶段：记录元数据并强制 draft 原始提示
            self.last_selection_metadata = {"airaevo_policy": self.policy, "airaevo_operator": "draft", "airaevo_generation_id": generation_id}
            return (public_system_prompt, public_user_prompt), None, "draft", None

        # 取任务下全部程序
        programs = self._all_programs(database, task_key)
        # 筛选活跃成功种群
        active = self._active_success_programs(programs)
        # 计算本次采样温度
        temperature = self._temperature(generation_id)
        # 请求的算子（可能与最终落地的算子不同）
        requested_mode = "draft"
        # 主父代
        parent_program: Program | None = None
        # 次级父代（仅交叉使用）
        secondary_parent: Program | None = None
        # 岛编号
        island_id = 0
        # 选择 trace（默认未启用）
        selection_trace: dict[str, Any] = {"enabled": False}

        if not programs:
            # 库中无任何程序（首个解）时强制 draft
            mode = "draft"
        elif self.policy == "original":
            if generation_id == 0 or not active:
                # 第一代或无活跃种群时必须 draft
                mode = "draft"
            else:
                # 判断是否满足交叉条件：达到最低代数且活跃种群至少 2 个
                crossover_ready = generation_id >= self.crossover_after_generation and len(active) >= 2
                # 以 crossover_probability 概率请求交叉，否则请求改进
                requested_mode = "crossover" if crossover_ready and random.random() < self.crossover_probability else "improve"
                # original 策略：先选岛再在岛内效用采样
                selected, island_id, selection_trace = self._sample_original(active, programs, requested_mode, temperature)
                if requested_mode == "crossover" and len(selected) == 2:
                    # 交叉成功：取双父代
                    mode = "crossover"
                    parent_program, secondary_parent = selected[0], selected[1]
                elif requested_mode == "improve" and len(selected) == 1:
                    # 改进成功：取单父代
                    mode = "improve"
                    parent_program = selected[0]
                else:
                    # 采样失败时走兜底链（不再包含请求算子）
                    mode, parent_program, secondary_parent, island_id, selection_trace = self._fallback_mode(requested_mode, programs, active, temperature, include_requested=False)
        else:
            # rl_mixed 策略：按固定分布抽取请求算子
            requested_mode = self._choose_rl_mixed_mode()
            if requested_mode == "draft":
                # 抽中 draft 时无需采样
                mode = "draft"
            else:
                # 其余算子走完整兜底链（含请求算子）
                mode, parent_program, secondary_parent, island_id, selection_trace = self._fallback_mode(requested_mode, programs, active, temperature)

        # 汇总全部非空父代，供经验记忆构建
        parents = [program for program in [parent_program, secondary_parent] if program is not None]
        # 按 AIRA-Evo 算子构建经验记忆（调试模式附带当前程序与更多相关卡片）
        memory = build_operator_experience_memory(
            mode,
            parents,
            all_programs=programs,
            current_program=parent_program if mode == "debug" else None,
            lower_is_better=False,
            max_related_cards=8 if mode == "debug" else 3,
        )
        # 延迟导入 AIRA-Evo 提示构建器
        from prompt_builder import build_airaevo_prompt

        # 构建 AIRA-Evo 对齐的四算子提示（附带记忆、执行预算等）
        prompt = build_airaevo_prompt(
            mode=mode,
            task_description=task_description,
            data_description=data_description,
            public_system_prompt=public_system_prompt,
            public_user_prompt=public_user_prompt,
            virtual_data_dir=data_dir,
            parent_program=parent_program,
            secondary_parent_program=secondary_parent,
            max_steps=max_steps,
            memory=memory,
            data_overview="",
            execution_timeout=self.execution_timeout,
        )
        # 记录本次选择的完整元数据（策略、算子、代数、岛、trace 等），供 RL 轨迹使用
        self.last_selection_metadata = {
            "airaevo_policy": self.policy,
            "airaevo_operator": mode,
            "airaevo_requested_operator": requested_mode,
            "airaevo_generation_id": generation_id,
            "airaevo_island_id": island_id,
            "airaevo_active_population_size": len(active),
            "airaevo_active_population_cap": self.max_island_size,
            "airaevo_num_islands": self.num_islands,
            "airaevo_selection_trace": selection_trace,
            "airaevo_memory_present": bool(memory),
        }
        # 推进该任务的代数计数
        self._task_generation[task_key] = generation_id + 1
        # 打印本次搜索决策摘要
        logger.info(
            "[AIRAEVO SEARCH] task=%s policy=%s mode=%s requested=%s parent=%s secondary=%s active=%s/%s",
            task_key,
            self.policy,
            mode,
            requested_mode,
            getattr(parent_program, "id", None),
            getattr(secondary_parent, "id", None),
            len(active),
            self.max_island_size,
        )
        # 返回提示、父代、模式与次级父代
        return prompt, parent_program, mode, secondary_parent
