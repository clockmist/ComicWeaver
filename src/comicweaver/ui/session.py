"""应用会话状态管理（单进程单用户简化方案）。"""
from __future__ import annotations

import asyncio
import time

from comicweaver.core import ComicState
from comicweaver.orchestrator import ComicWorkflow


class Session:
    """单个用户会话的状态容器。"""

    def __init__(self) -> None:
        self.state: ComicState | None = None
        self.workflow: ComicWorkflow | None = None
        self.event_log: list[dict] = []
        self.done_agents: list[str] = []
        self.active_agent: str = ""
        self.review_scores: dict[str, float] = {}
        self.last_checkpoint: dict | None = None
        self.workflow_done: bool = False
        self.error: str | None = None
        self.agent_outputs: dict[str, list[dict]] = {}
        self.dev_log: list[dict] = []
        self.panel_images_preview: list[dict] = []
        self.character_preview: list[dict] = []
        self.agent_start_time: float = 0.0
        self.current_phase: str = "init"
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def reset(self) -> None:
        self.event_log = []
        self.done_agents = []
        self.active_agent = ""
        self.review_scores = {}
        self.last_checkpoint = None
        self.workflow_done = False
        self.error = None
        self.workflow = None
        self.agent_outputs = {}
        self.dev_log = []
        self.panel_images_preview = []
        self.character_preview = []
        self.agent_start_time = 0.0
        self.current_phase = "init"
        self._task = None
        self._loop = None


SESSION = Session()
