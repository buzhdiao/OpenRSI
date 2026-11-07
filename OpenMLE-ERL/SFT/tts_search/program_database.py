"""
Program Database for storing and managing generated programs during rollout.
Uses SQLite for persistent storage with thread-safety.
"""
# 程序数据库模块：在 rollout 期间存储与管理生成的程序（对应论文中的 program archive 思想），
# 使用 SQLite 做持久化存储，并通过线程本地连接 + 写锁保证线程安全。
# 导入抽象基类与 abstractmethod 装饰器，用于定义搜索算法的抽象接口
from abc import ABC, abstractmethod
# 导入 dataclass 装饰器与 field 工厂，用于声明 Program 记录结构
from dataclasses import dataclass, field
# 导入 Enum，用于定义数据库生命周期模式枚举
from enum import Enum
# 导入 json，用于 metadata 字段与 JSON 字符串之间的序列化/反序列化
import json
# 导入 os，用于创建数据库文件所在目录
import os
# 导入 sqlite3，作为程序库的持久化存储引擎
import sqlite3
# 导入 threading，提供线程本地存储（连接隔离）与互斥锁（写入串行化）
import threading
# 导入类型注解 Any
from typing import Any


class DatabaseLifecycle(Enum):
    """Lifecycle modes for Program Database."""
    # 枚举值 rollout：每次 rollout（整轮采样过程）使用一个独立数据库
    ROLLOUT = "rollout"  # One database per rollout
    # 枚举值 step：每个训练 step 使用一个数据库
    STEP = "step"  # One database per step
    # 枚举值 task：每个任务长期共用一个数据库
    TASK = "task"  # One database per task


@dataclass
class Program:
    """Represents a program in the database."""
    # 字段说明（程序库中的一条程序记录）：
    # id=数据库主键；task_id/task_name=任务标识（uuid/名称）；code=生成的代码文本；
    # score=沙箱执行得分；reward=由得分换算的奖励（即 base_reward 语义）；fitness=搜索选样所用的适应度；
    # base_reward=未叠加父程序信息的基础奖励；run_log=沙箱执行日志；feedback=评估器/搜索策略给出的反馈文本；
    # parent_id/parent_code=父程序 ID 与代码（improve 模式用）；generation_mode=生成模式('draft' 从零草稿 / 'improve' 改进)；
    # raw_text=模型原始回复文本；metadata=附加元数据字典
    id: int | None = None  # Database ID
    task_id: str = ""  # Task identifier (uuid)
    task_name: str = ""  # Task name
    code: str = ""  # The code text
    score: float | None = None  # Score from sandbox execution
    reward: float = 0.0  # Reward calculated from score (base_reward)
    fitness: float | None = None  # Fitness used by search selection
    base_reward: float = 0.0  # Base reward before considering parent
    run_log: str = ""  # Execution log
    feedback: str = ""  # Feedback text from evaluator or search policy
    parent_id: int | None = None  # Parent program ID (for improve mode)
    parent_code: str = ""  # Parent program code (for logging)
    generation_mode: str = "draft"  # Generation mode: 'draft' or 'improve'
    raw_text: str = ""  # Raw response text from model
    metadata: dict[str, Any] = field(default_factory=dict)  # Additional metadata

    def __post_init__(self) -> None:
        # 数据类实例化后的钩子：负责补齐默认值与日志覆写逻辑
        # 若调用方未显式给定适应度，则默认与 reward 对齐
        if self.fitness is None:  # Default fitness when not provided.
            self.fitness = self.reward  # Align fitness with reward by default.
        # 从 metadata 中取出 clear_run_log（清洗后的运行日志，若有）
        clear_run_log = self.metadata.get("clear_run_log")
        # 若存在清洗后的日志，则用它覆盖原始 run_log
        if clear_run_log is not None:
            self.run_log = str(clear_run_log)

    def to_dict(self) -> dict[str, Any]:
        """Convert program to dictionary for storage."""
        # 将程序对象展开为与数据库列一一对应的字典（metadata 序列化为 JSON 字符串以便入库）
        return {
            'id': self.id,
            'task_id': self.task_id,
            'task_name': self.task_name,
            'code': self.code,
            'score': self.score,
            'reward': self.reward,
            'fitness': self.fitness,
            'base_reward': self.base_reward,
            'run_log': self.run_log,
            'feedback': self.feedback,
            'parent_id': self.parent_id,
            'parent_code': self.parent_code,
            'generation_mode': self.generation_mode,
            'raw_text': self.raw_text,
            'metadata': json.dumps(self.metadata)
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Program":
        """Create program from dictionary."""
        # 读取 metadata 字段（可能是 JSON 字符串，也可能已是字典）
        metadata_value = data.get("metadata", {})
        # 若为字符串则解析 JSON（空串按空对象 "{}" 处理）
        if isinstance(metadata_value, str):
            metadata = json.loads(metadata_value or "{}")
        else:
            # 已是字典则直接使用，空值兜底为空字典
            metadata = metadata_value or {}
        # 读取原始运行日志（缺省为空串）
        run_log = data.get('run_log', '')
        # 若 metadata 中记录了清洗后的日志，则优先采用它
        clear_run_log = metadata.get("clear_run_log")
        if clear_run_log is not None:
            run_log = str(clear_run_log)
        # 用字典中的各字段重建 Program 对象（缺失字段使用默认值）
        return cls(
            id=data.get('id'),
            task_id=data.get('task_id', ''),
            task_name=data.get('task_name', ''),
            code=data.get('code', ''),
            score=data.get('score'),
            reward=data.get('reward', 0.0),
            fitness=data.get('fitness'),
            base_reward=data.get('base_reward', 0.0),
            run_log=run_log,
            feedback=data.get('feedback') or '',
            parent_id=data.get('parent_id'),
            parent_code=data.get('parent_code', ''),
            generation_mode=data.get('generation_mode', 'draft'),
            raw_text=data.get('raw_text', ''),
            metadata=metadata
        )


class ProgramDatabase:
    """Database for storing and managing programs during rollout.

    Uses SQLite for persistent storage with connection pooling and thread-safety.
    Each task maintains its top-k programs by fitness.
    """

    def __init__(
        self, db_path: str = "program_database.db", max_per_task: int | None = 10
    ):
        """
        Initialize Program Database.

        Args:
            db_path: Path to SQLite database file
            max_per_task: Maximum number of programs to keep per task (top k by fitness).
                          Use None to disable pruning.
        """
        # 保存 SQLite 数据库文件路径
        self.db_path = db_path
        # 保存每个任务保留的最大程序数（按 fitness 取 top-k），None 表示不裁剪
        self.max_per_task = max_per_task

        # 线程本地存储槽：每个线程持有独立连接，避免共享连接带来的相互干扰
        self._local = threading.local()  # Per-instance connections to avoid epoch cross-talk.
        # 写操作互斥锁：将本数据库的写入串行化
        self._lock = threading.Lock()  # Serialize writes per database.

        # Ensure directory exists
        # 确保数据库文件所在目录存在（路径无目录部分时退回当前目录 "."）
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) if os.path.dirname(db_path) else ".", exist_ok=True)

        # Initialize database schema
        # 初始化数据库表结构与索引
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        # 当前线程还没有缓存连接时才新建
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            # 建立连接：check_same_thread=False 允许句柄跨线程传递，timeout=30 秒为写锁忙等待上限
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
            # 行工厂设为 sqlite3.Row，使查询结果可按列名访问
            self._local.conn.row_factory = sqlite3.Row
        # 返回本线程缓存的连接
        return self._local.conn

    def _init_db(self):
        """Initialize database schema."""
        # 获取本线程连接与游标
        conn = self._get_connection()
        cursor = conn.cursor()

        # 若 programs 表不存在则创建（包含程序全部字段、创建时间与指向父程序的外键）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS programs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                task_name TEXT NOT NULL,
                code TEXT NOT NULL,
                score REAL,
                reward REAL NOT NULL,
                fitness REAL NOT NULL,
                base_reward REAL NOT NULL,
                run_log TEXT,
                feedback TEXT NOT NULL DEFAULT '',
                parent_id INTEGER,
                parent_code TEXT,
                generation_mode TEXT DEFAULT 'draft',
                raw_text TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_id) REFERENCES programs(id)
            )
        ''')

        # 读取 programs 表现有列的元信息（用于旧库兼容判断）
        cursor.execute('PRAGMA table_info(programs)')  # Read table schema metadata.
        # 把现有列名汇总成集合，便于后续成员判断
        columns = {row['name'] for row in cursor.fetchall()}  # Collect column names.

        # 旧库迁移：缺少 fitness 列时自动补列
        if 'fitness' not in columns:  # Add fitness column when missing.
            # 新增 fitness 列（NOT NULL，默认 0.0）
            cursor.execute('ALTER TABLE programs ADD COLUMN fitness REAL NOT NULL DEFAULT 0.0')  # Add fitness column.
            # 用已有的 reward 值回填 fitness，保持历史数据语义一致
            cursor.execute('UPDATE programs SET fitness = reward')  # Backfill fitness from reward.
        # 缺少 feedback 列时补列（默认空串）
        if 'feedback' not in columns:
            cursor.execute("ALTER TABLE programs ADD COLUMN feedback TEXT NOT NULL DEFAULT ''")
        # 缺少 created_at 列时补列（默认取当前时间戳）
        if 'created_at' not in columns:
            cursor.execute(
                'ALTER TABLE programs ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
            )
        # if 'score' in columns and columns['score']['notnull']:
        #     cursor.execute('''
        #         CREATE TABLE IF NOT EXISTS programs_migrated (
        #             id INTEGER PRIMARY KEY AUTOINCREMENT,
        #             task_id TEXT,
        #             task_name TEXT NOT NULL,
        #             code TEXT NOT NULL,
        #             score REAL,
        #             reward REAL NOT NULL,
        #             fitness REAL NOT NULL,
        #             base_reward REAL NOT NULL,
        #             run_log TEXT,
        #             parent_id INTEGER,
        #             parent_code TEXT,
        #             generation_mode TEXT DEFAULT 'draft',
        #             raw_text TEXT,
        #             metadata TEXT,
        #             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        #             FOREIGN KEY (parent_id) REFERENCES programs(id)
        #         )
        #     ''')
        #     cursor.execute('''
        #         INSERT INTO programs_migrated (
        #             id, task_id, task_name, code, score, reward, fitness, base_reward,
        #             run_log, parent_id, parent_code, generation_mode, raw_text, metadata, created_at
        #         )
        #         SELECT
        #             id, task_id, task_name, code, score, reward, fitness, base_reward,
        #             run_log, parent_id, parent_code, generation_mode, raw_text, metadata, created_at
        #         FROM programs
        #     ''')
        #     cursor.execute('DROP TABLE programs')
        #     cursor.execute('ALTER TABLE programs_migrated RENAME TO programs')

        # Create index for faster task-based queries
        # 为 task_name 建索引，加速按任务名过滤的查询
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_task_name ON programs(task_name)
        ''')
        # 为 (task_name, fitness 降序) 建联合索引，加速每任务 top-k 选取
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_task_fitness ON programs(task_name, fitness DESC)
        ''')

        # 提交事务，使建表、补列与索引变更落盘
        conn.commit()

    def add(self, program: Program) -> int:
        """
        Add a program to the database, maintaining top k per task by fitness.

        Returns:
            The database ID of the inserted program
        """
        # 加写锁：插入与 top-k 裁剪必须原子完成，避免并发竞争
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Insert the program
            # 将程序记录插入 programs 表（参数元组与 SQL 占位符按列一一对应）
            cursor.execute('''
                INSERT INTO programs (task_id, task_name, code, score, reward, fitness, base_reward,
                                     run_log, feedback, parent_id, parent_code, generation_mode,
                                     raw_text, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                program.task_id,
                program.task_name,
                program.code,
                program.score,
                program.reward,
                program.fitness,
                program.base_reward,
                program.run_log,
                program.feedback,
                program.parent_id,
                program.parent_code,
                program.generation_mode,
                program.raw_text,
                json.dumps(program.metadata)
            ))
            # 取回自增主键，作为该程序在库中的 ID
            program_id = cursor.lastrowid

            # Maintain top k programs per task
            # 统计该任务当前已有的程序条数，用于判断是否超出 top-k 上限
            cursor.execute('''
                SELECT COUNT(*) as count FROM programs WHERE task_name = ?
            ''', (program.task_name,))
            # 取出计数值
            count = cursor.fetchone()['count']

            # 配置了上限且当前条数超额时，执行裁剪
            if self.max_per_task is not None and count > self.max_per_task:
                # Delete programs beyond top k
                # 删除不属于该任务 fitness 前 k 名的多余记录（子查询保留 top-k 的 id）
                cursor.execute('''
                    DELETE FROM programs
                    WHERE task_name = ? AND id NOT IN (
                        SELECT id FROM programs
                        WHERE task_name = ?
                        ORDER BY fitness DESC
                        LIMIT ?
                    )
                ''', (program.task_name, program.task_name, self.max_per_task))

            # 提交事务，使插入与裁剪生效
            conn.commit()
            # 返回新插入程序的数据库 ID
            return program_id

    def get_by_id(self, program_id: int) -> Program | None:
        """Get a program by its ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # 按主键 id 查询单条程序记录
        cursor.execute('SELECT * FROM programs WHERE id = ?', (program_id,))
        row = cursor.fetchone()

        # 查到则转换为 Program 对象，否则返回 None
        if row:
            return Program.from_dict(dict(row))
        return None

    def is_empty(self, task_name: str) -> bool:
        """Check if database is empty for a task."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # 统计该任务名下的程序条数
        cursor.execute('SELECT COUNT(*) as count FROM programs WHERE task_name = ?', (task_name,))
        count = cursor.fetchone()['count']
        # 条数为 0 即视为该任务的库为空
        return count == 0

    def size(self, task_name: str) -> int:
        """Get the number of programs for a task."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # 返回该任务名下的程序条数
        cursor.execute('SELECT COUNT(*) as count FROM programs WHERE task_name = ?', (task_name,))
        return cursor.fetchone()['count']

    def count_by_generation_mode(self, task_name: str, generation_mode: str) -> int:
        """Count programs for a task by generation mode."""
        conn = self._get_connection()
        cursor = conn.cursor()
        # 按任务名 + 生成模式（draft/improve）联合过滤计数
        cursor.execute(
            'SELECT COUNT(*) as count FROM programs WHERE task_name = ? AND generation_mode = ?',
            (task_name, generation_mode),
        )
        return cursor.fetchone()['count']

    def clear(self, task_name: str | None = None):
        """Clear programs from the database.

        Args:
            task_name: If provided, only clear programs for this task.
                      If None, clear all programs.
        """
        # 加写锁：删除操作需与其他写入互斥
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # 指定任务名时只清空该任务的记录
            if task_name:
                cursor.execute('DELETE FROM programs WHERE task_name = ?', (task_name,))
            else:
                # 未指定任务名则清空整张表
                cursor.execute('DELETE FROM programs')

            # 提交删除事务
            conn.commit()

    def close(self):
        """Close database connection."""
        # 仅当本线程存在缓存连接时才关闭并置空引用
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    def get_best(self, task_name: str) -> Program:
        conn = self._get_connection()
        cursor = conn.cursor()
        # 查询该任务 fitness 最高的 1 条记录（top-1）
        cursor.execute(
            '''
            SELECT * FROM programs
            WHERE task_name = ?
            ORDER BY fitness DESC
            LIMIT 1
            ''',
            (task_name,),
        )
        row = cursor.fetchone()
        # 将查询行转换为 Program 对象返回
        return Program.from_dict(dict(row))

    def get_random_by_fitness(self, task_name: str, fitness: float) -> Program | None:
        """Randomly sample a program by fitness."""
        conn = self._get_connection()
        cursor = conn.cursor()
        # 在该任务中按给定 fitness 值随机抽取 1 条（ORDER BY RANDOM() 实现随机采样）
        cursor.execute(
            '''
            SELECT * FROM programs
            WHERE task_name = ? AND fitness = ?
            ORDER BY RANDOM()
            LIMIT 1
            ''',
            (task_name, fitness),
        )
        row = cursor.fetchone()
        # 抽到则转换为 Program，否则返回 None
        if row:
            return Program.from_dict(dict(row))
        return None

    def list_by_task(self, task_name: str) -> list[Program]:
        """List programs for a task in creation order."""
        conn = self._get_connection()
        cursor = conn.cursor()
        # 按创建时间与 id 升序列出该任务全部程序（保证创建顺序稳定）
        cursor.execute(
            '''
            SELECT * FROM programs
            WHERE task_name = ?
            ORDER BY created_at ASC, id ASC
            ''',
            (task_name,),
        )
        rows = cursor.fetchall()
        # 批量把查询行转换为 Program 对象列表
        return [Program.from_dict(dict(row)) for row in rows]


class SearchAlgorithm(ABC):
    """Abstract base class for search algorithms to select programs from database."""

    @abstractmethod
    def select(
        self,
        database: ProgramDatabase,
        task_name: str,
        description: str,
        data_dir: str,
        max_steps: int = 1,
    ) -> tuple[str | None, Program | None, str, str | None]:
        """
        Select a program and build prompt.

        This method decides whether to use draft or improve mode, selects a parent program
        if needed, and builds the appropriate prompt.

        Args:
            database: The ProgramDatabase to select from
            task_name: The task name to select programs for
            description: Task description for prompt building
            data_dir: Virtual data directory path
            max_steps: Maximum number of steps

        Returns:
            Tuple of (prompt, parent_program, mode, model_name)
            - prompt: The built prompt string
            - parent_program: Selected parent Program or None if draft mode
            - mode: Either 'draft' or 'improve'
            - model_name: Selected model name or None
        """
        # 抽象方法：具体搜索算法（如 FunSearch 风格的 island 搜索）需实现选样与提示词构建
        raise NotImplementedError

    @abstractmethod
    def select_best(self, database: ProgramDatabase, task_name: str) -> Program:
        # 抽象方法：返回指定任务在库中的当前最优程序
        raise NotImplementedError
