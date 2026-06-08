"""全局工作流状态 - 对应 docs/开发思路框架.md §6.1 ComicState。

LangGraph 的状态字典定义。使用 TypedDict 以兼容 LangGraph 的状态合并语义,
但同时也提供 Pydantic 模型 ComicProject 用于序列化与持久化。
"""
from __future__ import annotations

import re
import time
import uuid
from typing import TypedDict

from pydantic import BaseModel, Field


def _sanitize_project_name(name: str, max_len: int = 20) -> str:
    """将项目名称转换为安全的文件夹名。

    保留中文、英文、数字、下划线、连字符，去除其他特殊字符。
    如果 sanitize 后为空，返回 "project"。
    """
    cleaned = re.sub(r"[^\w\u4e00-\u9fff\-]", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        return "project"
    return cleaned[:max_len]


class ComicState(TypedDict, total=False):
    """LangGraph 工作流状态字典。"""
    # 输入
    user_input: str
    title: str
    creation_mode: str
    style_preset: str
    interaction_mode: str
    target_pages: int

    # v0.4: 故事开发
    developed_story: dict
    narrative_structure: dict

    # 剧本理解
    structured_script: dict
    emotion_curve: list[float]

    # 角色管理
    character_db: dict
    reference_chain: list

    # 分镜
    storyboard_plan: list[dict]
    layout_grids: list

    # 图像
    panel_images: list[dict]
    generation_metadata: list

    # 台词气泡
    bubble_placements: dict

    # 排版
    final_pages: list[dict]
    exports: list[dict]
    export_format: str

    # 交互反馈
    retry_counts: dict
    pending_checkpoint: dict | None
    user_decisions: list
    stream_messages: list[dict]

    # 控制
    current_phase: str
    errors: list
    project_id: str


def make_initial_state(
    user_input: str,
    creation_mode: str = "simple",
    style_preset: str = "manga",
    interaction_mode: str = "semi_auto",
    target_pages: int = 4,
    title: str = "",
) -> ComicState:
    """构造初始状态。"""
    resolved_title = title or user_input[:30]
    safe_name = _sanitize_project_name(resolved_title)
    short_uid = str(uuid.uuid4())[:6]
    project_id = f"{safe_name}_{short_uid}"

    return ComicState(
        user_input=user_input,
        title=resolved_title,
        creation_mode=creation_mode,
        style_preset=style_preset,
        interaction_mode=interaction_mode,
        target_pages=target_pages,
        developed_story={},
        narrative_structure={},
        structured_script={},
        emotion_curve=[],
        character_db={},
        reference_chain=[],
        storyboard_plan=[],
        layout_grids=[],
        panel_images=[],
        generation_metadata=[],
        bubble_placements={},
        final_pages=[],
        exports=[],
        export_format="png",
        retry_counts={},
        pending_checkpoint=None,
        user_decisions=[],
        stream_messages=[],
        current_phase="init",
        errors=[],
        project_id=project_id,
    )


class ComicProject(BaseModel):
    """项目持久化模型 - 用于保存与恢复。"""
    project_id: str
    title: str = "Untitled Project"
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    state: dict = Field(default_factory=dict)
