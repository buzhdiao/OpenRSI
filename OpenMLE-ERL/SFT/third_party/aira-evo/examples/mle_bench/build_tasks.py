#!/usr/bin/env python3
# 启用延迟注解求值（PEP 563）
from __future__ import annotations

# 导入 argparse，用于解析命令行参数
import argparse
# 导入 json，用于写出任务元数据 JSON
import json
# 导入 sys，用于调整模块搜索路径
import sys
# 导入 Path，用于路径操作
from pathlib import Path
# 导入 Any 类型注解
from typing import Any

# 导入 pandas，用于读取 parquet 数据
import pandas as pd
# 导入 yaml，用于写出任务配置
import yaml
# 导入 OmegaConf，用于读取构建配置
from omegaconf import OmegaConf

# 计算仓库根目录（本文件向上 4 级）
REPO_ROOT = Path(__file__).resolve().parents[4]
# 根目录不在 sys.path 中则插入，保证能导入 tts_search
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 导入旧版公共提示词拆分工具
from tts_search.prompt_builder import split_legacy_public_prompts


# 清洗任务名：把路径分隔符替换为下划线，避免构造非法目录名
def _sanitize_task_name(task_name: str) -> str:
    return task_name.replace("/", "_").replace("\\", "_")


# 从数据记录中提取提示词相关字段
def _extract_prompt_fields(
    record: dict[str, Any],
    input_key: str,
    metadata: dict[str, Any],
) -> dict[str, str]:
    # 取出输入列的 messages 值
    messages_value = record[input_key]
    # numpy 数组时转成普通列表
    if hasattr(messages_value, "tolist"):
        messages = list(messages_value.tolist() or [])
    else:
        # 普通可迭代对象时直接转列表
        messages = list(messages_value or [])

    # 公共系统提示词缓存
    public_system_prompt = ""
    # 公共用户提示词缓存
    public_user_prompt = ""
    # 遍历消息列表，找第一条 system 与第一条 user 消息
    for message in messages:
        # 读取角色并标准化为小写
        role = str(message.get("role", "")).strip().lower()
        # 读取消息内容
        content = str(message.get("content", ""))
        # 尚未有系统提示词且当前是 system 角色时记录
        if role == "system" and not public_system_prompt:
            public_system_prompt = content
        # 尚未有用户提示词且当前是 user 角色时记录
        elif role == "user" and not public_user_prompt:
            public_user_prompt = content
        # 两者都拿到后提前结束
        if public_system_prompt and public_user_prompt:
            break

    # 从元数据读取任务描述（兼容两种键名）
    task_description = str(metadata.get("task_description") or metadata.get("task_desc") or "")
    # 从元数据读取数据描述（兼容两种键名）
    data_description = str(metadata.get("data_description") or metadata.get("data_desc") or "")

    # 元数据缺失两个描述时，从旧版公共提示词中拆分还原
    if task_description.strip() == "" and data_description.strip() == "":
        (
            task_description,
            data_description,
            public_system_prompt,
            public_user_prompt,
        ) = split_legacy_public_prompts(public_system_prompt, public_user_prompt)

    # 返回四个提示词相关字段
    return {
        "task_description": task_description,
        "data_description": data_description,
        "public_system_prompt": public_system_prompt,
        "public_user_prompt": public_user_prompt,
    }


# 构建任务：把 parquet 数据逐行展开为任务目录
def build_tasks(config_path: Path) -> None:
    # 加载 OmegaConf/YAML 构建配置
    cfg = OmegaConf.load(config_path)
    # 以本文件所在目录作为任务输出根目录
    base_dir = Path(__file__).resolve().parent
    # 读取评测数据并转为字典列表
    records = pd.read_parquet(cfg.data.eval_data).to_dict(orient="records")
    # 构建 资源类型 -> 沙箱地址 的映射
    base_url_map = {item.resource: item.base_url for item in cfg.sandbox.base_url_map}

    # 逐条记录生成任务
    for record in records:
        # 取出该记录的元数据字典
        metadata = dict(record[cfg.data.metadata_key])
        # 读取任务名
        task_name = str(metadata["task_name"])
        # 计算任务输出目录
        task_dir = base_dir / _sanitize_task_name(task_name)
        # 创建任务目录（已存在则复用）
        task_dir.mkdir(parents=True, exist_ok=True)

        # 提取提示词相关字段
        prompt_fields = _extract_prompt_fields(
            record=record,
            input_key=str(cfg.data.input_key),
            metadata=metadata,
        )
        # 读取沙箱资源类型（cpu/gpu）并小写化
        resource = str(metadata["cpu_gpu"]).lower()

        # 组装任务配置负载
        task_payload = {
            "uuid": metadata["uuid"],
            "task_name": task_name,
            "task": metadata["task"],
            "source": str(metadata.get("source") or "MLE-Bench"),
            "modality": metadata.get("modality"),
            "data_dir": str(metadata["data_dir"]),
            "higher_is_better": bool(metadata["higher_is_better"]),
            "theoretical_max": metadata.get("theoretical_max"),
            "theoretical_min": metadata.get("theoretical_min"),
            "leaderboard_max": metadata.get("leaderboard_max"),
            "leaderboard_min": metadata.get("leaderboard_min"),
            "leaderboard_dir": cfg.data.leaderboard_dir,
            "submit_dir": str(cfg.data.submit_dir),
            "submit_data_dir_root": cfg.data.get("submit_data_dir_root"),
            "sandbox": {
                "resource": resource,
                "base_url": str(base_url_map[resource]),
                "job_timeout": int(cfg.sandbox.job_timeout),
                "wait_timeout": int(cfg.sandbox.wait_timeout),
                "poll_interval": int(cfg.sandbox.poll_interval),
                "use_score2reward": bool(cfg.sandbox.use_score2reward),
            },
            **prompt_fields,
        }

        # 写出 YAML 格式的任务配置
        (task_dir / "config.yaml").write_text(
            yaml.safe_dump(task_payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        # 写出 JSON 格式的任务元数据
        (task_dir / "task_metadata.json").write_text(
            json.dumps(task_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# 命令行入口
def main() -> None:
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Build AIRA Dojo MLE-Bench tasks")
    # 构建配置文件路径（必填）
    parser.add_argument("--config", required=True, help="Path to build config yaml")
    # 解析参数
    args = parser.parse_args()
    # 执行任务构建
    build_tasks(Path(args.config))


# 作为脚本运行时执行入口
if __name__ == "__main__":
    main()
