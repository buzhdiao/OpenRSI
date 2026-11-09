"""
Program Database for storing and managing generated programs during rollout.
Uses SQLite for persistent storage with thread-safety.
"""
# 中文说明:程序数据库模块——在 rollout(整轮采样)期间存储与管理生成的程序
# (对应进化搜索里的 program archive 思想),使用 SQLite 做持久化存储,
# 通过线程本地连接 + 写锁保证线程安全;每个任务按 fitness 维护 top-k 程序,
# 并在此之上抽象出搜索算法接口(draft/improve/debug/crossover 的父程序选择)。
# 位置:OpenMLE-Evo/tts_search 下,是 tts_search 进化搜索的核心数据层。

# 抽象基类与 abstractmethod:定义搜索算法的抽象接口
from abc import ABC, abstractmethod
# dataclass 与 field:声明 Program 记录结构(field 用于 dict 默认工厂)
from dataclasses import dataclass, field
# Enum:定义数据库生命周期模式
from enum import Enum
# json:metadata 字段与 JSON 字符串互转
import json
# os:创建数据库文件所在目录
import os
# sqlite3:程序库的持久化存储引擎
import sqlite3
# threading:线程本地存储(连接隔离)与互斥锁(写串行化)
import threading
# Any 类型注解
from typing import Any


class DatabaseLifecycle(Enum):
    """程序数据库的生命周期模式:决定库的复用粒度。"""
    # 每次 rollout(整轮采样)一个独立数据库
    ROLLOUT = "rollout"  # One database per rollout
    # 每个训练 step 一个数据库
    STEP = "step"  # One database per step
    # 每个任务长期共用一个数据库(经验跨轮积累)
    TASK = "task"  # One database per task


@dataclass
class Program:
    """程序库中的一条程序记录(进化搜索的基本个体单元)。"""
    # 字段说明:id=数据库主键;task_id/task_name=任务标识(uuid/名称);code=生成的代码文本;
    # score=沙箱执行得分;reward=由得分换算的奖励(base_reward 语义);
    # fitness=搜索选样所用的适应度;base_reward=未叠加父程序信息的基础奖励;
    # run_log=沙箱执行日志;feedback=评估器/搜索策略给出的反馈文本;
    # parent_id/parent_code=父程序 ID 与代码(improve 模式用);
    # generation_mode=生成模式('draft'/'improve'/'debug'/'crossover');
    # raw_text=模型原始回复;metadata=附加元数据
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
    generation_mode: str = "draft"  # Generation mode: 'draft' | 'improve' | 'debug' | 'crossover'
    raw_text: str = ""  # Raw response text from model
    metadata: dict[str, Any] = field(default_factory=dict)  # Additional metadata

    def __post_init__(self) -> None:
        # 数据类实例化后的钩子:负责补齐默认值与日志覆写逻辑
        # 未显式给定 fitness 时,默认与 reward 对齐(适应度即奖励的语义)
        if self.fitness is None:  # Default fitness when not provided.
            self.fitness = self.reward  # Align fitness with reward by default.
        # 从 metadata 取出清洗后的运行日志(clear_run_log)
        clear_run_log = self.metadata.get("clear_run_log")
        # 若存在清洗日志,则用它覆盖原始 run_log(评估口径统一)
        if clear_run_log is not None:
            self.run_log = str(clear_run_log)

    def to_dict(self) -> dict[str, Any]:
        """把程序对象展开为与数据库列一一对应的字典(metadata 序列化为 JSON 以便入库)。"""
        # 逐字段映射;metadata 转 JSON 字符串
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
        """从数据库行字典重建 Program 对象(兼容 metadata 为字符串或 dict 两种形态)。"""
        # 先取 metadata 原始值(缺省空 dict)
        metadata_value = data.get("metadata", {})
        # 情况一:metadata 是 JSON 字符串,需要反序列化
        if isinstance(metadata_value, str):
            metadata = json.loads(metadata_value or "{}")
        # 情况二:已是 dict(或 None 兜底)
        else:
            metadata = metadata_value or {}
        # 先按原始 run_log 取值
        run_log = data.get('run_log', '')
        # 若 metadata 里有清洗后的日志,则覆盖(与 __post_init__ 逻辑保持一致)
        clear_run_log = metadata.get("clear_run_log")
        if clear_run_log is not None:
            run_log = str(clear_run_log)
        # 逐字段构造 Program(缺失字段走默认值)
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
    """rollout 期间的程序存储与管理数据库。

    基于 SQLite 持久化,连接按线程隔离(线程本地连接)、写入按库串行(互斥锁);
    每个任务按 fitness 只保留 top-k 条程序(max_per_task 控制剪枝)。
    """

    def __init__(
        self, db_path: str = "program_database.db", max_per_task: int | None = 10
    ):
        """初始化程序数据库。

        Args:
            db_path: SQLite 数据库文件路径
            max_per_task: 每个任务保留的最大程序数(按 fitness 取 top-k);
                          传 None 表示不剪枝。
        """
        # 记录数据库文件路径
        self.db_path = db_path
        # 记录每任务保留上限(top-k 剪枝阈值)
        self.max_per_task = max_per_task

        # 线程本地连接:每个线程一个独立连接,避免不同线程间的 cursor/epoch 串扰
        self._local = threading.local()  # Per-instance connections to avoid epoch cross-talk.
        # 写锁:同一数据库的写入串行化(SQLite 写并发受限)
        self._lock = threading.Lock()  # Serialize writes per database.

        # 确保数据库所在目录存在(相对路径时落到当前目录)
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) if os.path.dirname(db_path) else ".", exist_ok=True)

        # 初始化表结构与索引
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """获取当前线程的本地数据库连接(懒创建)。"""
        # 本线程尚无连接则新建
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            # check_same_thread=False 允许跨线程对象持有,实际由线程本地隔离保证安全;超时 30 秒防锁死
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
            # 行工厂返回 sqlite3.Row,可用列名取值
            self._local.conn.row_factory = sqlite3.Row
        # 返回线程本地连接
        return self._local.conn

    def _init_db(self):
        """初始化数据库 schema:建表、做旧版本字段迁移、建索引。"""
        # 取当前线程连接与游标
        conn = self._get_connection()
        cursor = conn.cursor()

        # 建 programs 主表(若不存在):自增主键、任务/代码/分数/奖励/适应度/日志/父程序等列,
        # parent_id 外键自引用(程序进化谱系),created_at 记录插入时间
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

        # 读取已存在表的列信息(旧库迁移用)
        cursor.execute('PRAGMA table_info(programs)')  # Read table schema metadata.
        # 汇总列名集合
        columns = {row['name'] for row in cursor.fetchall()}  # Collect column names.

        # 旧库缺 fitness 列:补列并用 reward 回填(历史数据适应度=奖励)
        if 'fitness' not in columns:  # Add fitness column when missing.
            cursor.execute('ALTER TABLE programs ADD COLUMN fitness REAL NOT NULL DEFAULT 0.0')  # Add fitness column.
            cursor.execute('UPDATE programs SET fitness = reward')  # Backfill fitness from reward.
        # 旧库缺 feedback 列:补列(默认空串)
        if 'feedback' not in columns:
            cursor.execute("ALTER TABLE programs ADD COLUMN feedback TEXT NOT NULL DEFAULT ''")
        # 旧库缺 created_at 列:补列(默认当前时间)
        if 'created_at' not in columns:
            cursor.execute(
                'ALTER TABLE programs ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
            )
        # 下面注释掉的整段是"重建表"式迁移方案的备份(重建含全部新 schema 的表再改名),
        # 当前用轻量 ALTER TABLE 逐步加列即可,故保留注释以备回溯
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

        # 为按任务名查询建索引,加速任务级检索
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_task_name ON programs(task_name)
        ''')
        # 为"按任务取 top-k 适应度"建联合索引(任务名+适应度倒序),加速排序剪枝
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_task_fitness ON programs(task_name, fitness DESC)
        ''')

        # 提交建表/迁移/建索引事务
        conn.commit()

    def add(self, program: Program) -> int:
        """把一条程序加入数据库,并维护每任务按 fitness 的 top-k。

        Returns:
            新插入程序的数据库 ID
        """
        # 写操作全程持锁,保证多线程写安全
        with self._lock:
            # 取当前线程连接与游标
            conn = self._get_connection()
            cursor = conn.cursor()

            # 插入程序记录(metadata 序列化为 JSON)
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
            # 取新插入行的自增主键
            program_id = cursor.lastrowid

            # 统计该任务当前的程序数量,决定是否需要剪枝
            cursor.execute('''
                SELECT COUNT(*) as count FROM programs WHERE task_name = ?
            ''', (program.task_name,))
            # 取出行数
            count = cursor.fetchone()['count']

            # 配置了上限且超出时,删除 fitness 排名靠后(前 k 名之外)的程序
            if self.max_per_task is not None and count > self.max_per_task:
                # 删除不属于 top-k 的旧程序(按 fitness 降序保留 LIMIT 条)
                cursor.execute('''
                    DELETE FROM programs
                    WHERE task_name = ? AND id NOT IN (
                        SELECT id FROM programs
                        WHERE task_name = ?
                        ORDER BY fitness DESC
                        LIMIT ?
                    )
                ''', (program.task_name, program.task_name, self.max_per_task))

            # 提交插入与剪枝
            conn.commit()
            # 返回新程序 ID
            return program_id

    def get_by_id(self, program_id: int) -> Program | None:
        """按数据库 ID 取一条程序;不存在返回 None。"""
        # 取连接与游标(读操作无需持锁)
        conn = self._get_connection()
        cursor = conn.cursor()

        # 按主键查询
        cursor.execute('SELECT * FROM programs WHERE id = ?', (program_id,))
        # 取一行
        row = cursor.fetchone()

        # 有结果则重建 Program 对象
        if row:
            return Program.from_dict(dict(row))
        # 无结果返回 None
        return None

    def is_empty(self, task_name: str) -> bool:
        """判断某任务在库中是否没有任何程序。"""
        # 取连接与游标
        conn = self._get_connection()
        cursor = conn.cursor()

        # 统计该任务的程序数
        cursor.execute('SELECT COUNT(*) as count FROM programs WHERE task_name = ?', (task_name,))
        # 取计数
        count = cursor.fetchone()['count']
        # 计数为 0 即为空
        return count == 0

    def size(self, task_name: str) -> int:
        """返回某任务当前的程序数量。"""
        # 取连接与游标
        conn = self._get_connection()
        cursor = conn.cursor()

        # 统计该任务的程序数
        cursor.execute('SELECT COUNT(*) as count FROM programs WHERE task_name = ?', (task_name,))
        # 返回计数
        return cursor.fetchone()['count']

    def count_by_generation_mode(self, task_name: str, generation_mode: str) -> int:
        """统计某任务在指定生成模式下的程序数量。"""
        # 取连接与游标
        conn = self._get_connection()
        cursor = conn.cursor()
        # 按任务名+生成模式过滤计数
        cursor.execute(
            'SELECT COUNT(*) as count FROM programs WHERE task_name = ? AND generation_mode = ?',
            (task_name, generation_mode),
        )
        # 返回计数
        return cursor.fetchone()['count']

    def clear(self, task_name: str | None = None):
        """清空程序;指定任务名时只清该任务,否则清空全库。"""
        # 删除属写操作,持锁执行
        with self._lock:
            # 取连接与游标
            conn = self._get_connection()
            cursor = conn.cursor()

            # 指定任务名:按任务名删除
            if task_name:
                cursor.execute('DELETE FROM programs WHERE task_name = ?', (task_name,))
            # 未指定:清空整表
            else:
                cursor.execute('DELETE FROM programs')

            # 提交删除事务
            conn.commit()

    def close(self):
        """关闭当前线程的数据库连接(仅关闭本线程本地连接)。"""
        # 本线程存在连接才关闭
        if hasattr(self._local, 'conn') and self._local.conn:
            # 关闭连接
            self._local.conn.close()
            # 置空引用,下次访问会重新创建
            self._local.conn = None

    def get_best(self, task_name: str) -> Program:
        """返回某任务 fitness 最高的程序(库为空时 SQL 会抛错,由调用方保证非空)。"""
        # 取连接与游标
        conn = self._get_connection()
        cursor = conn.cursor()
        # 按任务过滤并按 fitness 降序取第一条
        cursor.execute(
            '''
            SELECT * FROM programs
            WHERE task_name = ?
            ORDER BY fitness DESC
            LIMIT 1
            ''',
            (task_name,),
        )
        # 取一行
        row = cursor.fetchone()
        # 重建 Program 返回
        return Program.from_dict(dict(row))

    def get_random_by_fitness(self, task_name: str, fitness: float) -> Program | None:
        """在指定 fitness 等级的程序中随机采样一条(保持多样性,避免每次选同一条)。"""
        # 取连接与游标
        conn = self._get_connection()
        cursor = conn.cursor()
        # 按任务+fitness 过滤,用 RANDOM() 随机取一条
        cursor.execute(
            '''
            SELECT * FROM programs
            WHERE task_name = ? AND fitness = ?
            ORDER BY RANDOM()
            LIMIT 1
            ''',
            (task_name, fitness),
        )
        # 取一行
        row = cursor.fetchone()
        # 命中则重建返回
        if row:
            return Program.from_dict(dict(row))
        # 无匹配返回 None
        return None

    def get_top_k(self, task_name: str, k: int) -> list[Program]:
        """返回某任务按 fitness 降序的前 k 条程序。"""
        # 取连接与游标
        conn = self._get_connection()
        cursor = conn.cursor()
        # 按任务过滤,fitness 降序,限制 k 条
        cursor.execute(
            """
            SELECT * FROM programs
            WHERE task_name = ?
            ORDER BY fitness DESC
            LIMIT ?
            """,
            (task_name, int(k)),
        )
        # 取全部行
        rows = cursor.fetchall()
        # 逐行重建 Program 列表
        return [Program.from_dict(dict(row)) for row in rows]

    def list_by_task(self, task_name: str) -> list[Program]:
        """按创建时间先后列出某任务的全部程序(审计/回放用)。"""
        # 取连接与游标
        conn = self._get_connection()
        cursor = conn.cursor()
        # 按任务过滤,先按 created_at 再按 id 升序(同秒插入也能稳定排序)
        cursor.execute(
            '''
            SELECT * FROM programs
            WHERE task_name = ?
            ORDER BY created_at ASC, id ASC
            ''',
            (task_name,),
        )
        # 取全部行
        rows = cursor.fetchall()
        # 逐行重建 Program 列表
        return [Program.from_dict(dict(row)) for row in rows]


class SearchAlgorithm(ABC):
    """搜索算法抽象基类:决定如何从程序库中选父程序并构造下一步的 prompt。"""

    @abstractmethod
    def select(
        self,
        database: ProgramDatabase,
        task_name: str,
        task_description: str,
        data_description: str,
        public_system_prompt: str,
        public_user_prompt: str,
        data_dir: str,
        max_steps: int = 1,
    ) -> tuple[tuple[str, str] | None, Program | None, str, str | None, Program | None]:
        """选择父程序并构造生成 prompt(核心决策点)。

        该方法决定本轮用 draft(从零草稿)还是 improve/improve 类模式,
        需要时从数据库选出父程序,并组装相应的 prompt。

        Args:
            database: 供选择的程序数据库
            task_name: 任务名
            task_description: 任务描述文本(任务特有)
            data_description: 数据描述文本(任务特有)
            public_system_prompt: 数据集提供的系统提示词片段(需拼接)
            public_user_prompt: 数据集提供的用户提示词片段(需拼接)
            data_dir: 虚拟数据目录路径
            max_steps: 最大步数

        Returns:
            元组 (prompts, parent_program, mode, model_name, secondary_parent_program):
            - prompts: (system_prompt, user_prompt)
            - parent_program: 选中的父程序;draft 模式为 None
            - mode: 'draft' | 'improve' | 'debug' | 'crossover' 之一
            - model_name: 选定的模型名,可为 None
            - secondary_parent_program: 仅 crossover(交叉)模式使用的第二父程序
        """
        # 子类必须实现
        raise NotImplementedError

    @abstractmethod
    def select_best(self, database: ProgramDatabase, task_name: str) -> Program:
        """选出某任务的最终最优程序(评估收尾时调用)。"""
        # 子类必须实现
        raise NotImplementedError
