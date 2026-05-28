"""全局工作流状态 - 对应 docs/开发思路框架.md §6.1 ComicState。

LangGraph 的状态字典定义。使用 TypedDict 以兼容 LangGraph 的状态合并语义,
但同时也提供 Pydantic 模型 ComicProject 用于序列化与持久化。
"""
from __future__ import annotations

import time
import uuid
from typing import TypedDict

from pydantic import BaseModel, Field


class ComicState(TypedDict, total=False):
    """LangGraph 工作流状态字典。"""
    # 输入
    user_input: str
    creation_mode: str
    style_preset: str
    interaction_mode: str
    target_pages: int

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

    # 排版
    final_pages: list[dict]
    exports: list[dict]
    export_format: str

    # 审查与交互
    review_results: list[dict]
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
) -> ComicState:
    """构造初始状态。"""
    return ComicState(
        user_input=user_input,
        creation_mode=creation_mode,
        style_preset=style_preset,
        interaction_mode=interaction_mode,
        target_pages=target_pages,
        structured_script={},
        emotion_curve=[],
        character_db={},
        reference_chain=[],
        storyboard_plan=[],
        layout_grids=[],
        panel_images=[],
        generation_metadata=[],
        final_pages=[],
        exports=[],
        export_format="png",
        review_results=[],
        retry_counts={},
        pending_checkpoint=None,
        user_decisions=[],
        stream_messages=[],
        current_phase="init",
        errors=[],
        project_id=str(uuid.uuid4())[:8],
    )


class ComicProject(BaseModel):
    """项目持久化模型 - 用于保存与恢复。"""
    project_id: str
    title: str = "Untitled Project"
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    state: dict = Field(default_factory=dict)
