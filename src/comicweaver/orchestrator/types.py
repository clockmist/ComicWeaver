"""轻量级编排器 - 顺序工作流引擎,带审查节点与确认点。

设计目标:
1. 不强依赖 LangGraph(可在无该依赖时正常工作)
2. 接口仿照 LangGraph 的 add_node/conditional_edges,便于未来替换
3. 支持流式事件透传与暂停/恢复
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from comicweaver.core import (
    AgentContext,
    BaseAgent,
    ComicState,
    InteractionMode,
    ReviewDecision,
    ReviewFeedback,
    ReviewInput,
    StreamEvent,
    StreamEventType,
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
