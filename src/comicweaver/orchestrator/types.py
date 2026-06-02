"""编排器类型定义 - 工作流事件、确认信号与交互模式判断。

设计目标:
1. 与 LangGraph 工作流引擎配合,提供类型安全的流式事件与确认机制
2. WorkflowEvent / WorkflowMessage / CheckpointSignal 是编排层与 UI 层之间的契约
3. should_pause() 根据交互模式决定是否在关键节点暂停
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from comicweaver.core import (
    ComicState,
    InteractionMode,
)


class WorkflowEvent(str, Enum):
    NODE_START = "node_start"
    NODE_END = "node_end"
    REVIEW_PASS = "review_pass"
    REVIEW_REVISE = "review_revise"
    REVIEW_ESCALATE = "review_escalate"
    CHECKPOINT = "checkpoint"
    WORKFLOW_DONE = "workflow_done"
    ERROR = "error"


@dataclass
class WorkflowMessage:
    """工作流向上层吐出的消息(在StreamEvent之上)。"""
    event: WorkflowEvent
    node: str
    payload: Any = None
    timestamp: float = field(default_factory=time.time)


# 关键确认节点(半自动模式触发)
KEY_CHECKPOINTS = {
    "after_script": "剧本结构确认",
    "after_character": "角色设计确认",
    "after_storyboard": "分镜规划确认",
    "after_layout": "最终漫画确认",
}


@dataclass
class CheckpointSignal:
    """确认点信号 - 工作流暂停,等待用户响应。"""
    checkpoint_id: str
    label: str
    payload: dict
    user_options: list[str] = field(
        default_factory=lambda: ["accept", "regenerate", "skip"]
    )
    timestamp: float = field(default_factory=time.time)


def should_pause(state: ComicState, checkpoint_id: str) -> bool:
    """判断当前模式下是否应在该节点暂停。"""
    mode = state.get("interaction_mode", InteractionMode.SEMI_AUTO.value)
    if mode == InteractionMode.FULL_AUTO.value:
        return False
    if mode == InteractionMode.FULL_INTERACTIVE.value:
        return True
    return checkpoint_id in KEY_CHECKPOINTS
