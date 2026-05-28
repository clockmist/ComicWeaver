"""Agent 基类 - 对应 docs/agents/00-base.md §2。

所有生产Agent与审查Agent都必须继承 BaseAgent。
"""
from __future__ import annotations

import asyncio
import hashlib
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Generic, TypeVar

from pydantic import BaseModel

from comicweaver.config import AppConfig, load_config

from .schema import (
    AgentContext,
    ReviewFeedback,
    StreamCallback,
    StreamEvent,
    StreamEventType,
)

TIn = TypeVar("TIn", bound=BaseModel)
TOut = TypeVar("TOut", bound=BaseModel)


class BaseAgent(ABC, Generic[TIn, TOut]):
    """所有Agent的基类。

    子类必须设置 name / version / rubric_id 类属性,并实现 run() 与 astream()。
    """
    name: str = "base_agent"
    version: str = "0.0.0"
    rubric_id: str = "rubric_unknown"

    def __init__(
        self,
        stream_callback: StreamCallback | None = None,
        config: AppConfig | None = None,
    ):
        self._stream_callback = stream_callback
        self.config = config or load_config()

    # ---------- 必须实现 ----------

    @abstractmethod
    async def run(self, inputs: TIn, context: AgentContext) -> TOut:
        """同步执行。"""

    async def astream(
        self, inputs: TIn, context: AgentContext
    ) -> AsyncIterator[StreamEvent]:
        """流式执行。默认实现:发送启动LOG → 调用 run() → 发送 DONE。

        子类可以覆盖以提供更细粒度的流式输出。
        """
        yield self._make_event(StreamEventType.LOG, f"{self.name} 启动")
        try:
            output = await self.run(inputs, context)
            yield self._make_event(StreamEventType.DONE, output.model_dump())
        except Exception as exc:  # noqa: BLE001
            yield self._make_event(
                StreamEventType.ERROR,
                {"error": str(exc), "type": exc.__class__.__name__},
            )
            raise

    # ---------- 通用工具 ----------

    async def revise(
        self,
        inputs: TIn,
        previous_output: TOut,
        review_feedback: ReviewFeedback,
        context: AgentContext,
    ) -> TOut:
        """携带审查反馈的返工。默认实现:把feedback注入inputs.feedback并重新run。"""
        revised_inputs = inputs.model_copy(update={"feedback": review_feedback})
        return await self.run(revised_inputs, context)

    def _inputs_hash(self, inputs: TIn) -> str:
        """计算输入哈希,用于缓存。"""
        try:
            payload = inputs.model_dump_json(exclude={"feedback"})
        except Exception:  # noqa: BLE001
            payload = str(inputs)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _make_event(
        self,
        event_type: StreamEventType,
        content: object = None,
        metadata: dict | None = None,
    ) -> StreamEvent:
        return StreamEvent(
            agent=self.name,
            type=event_type,
            content=content,
            metadata=metadata or {},
        )

    async def _emit(
        self,
        event_type: StreamEventType,
        content: object = None,
        metadata: dict | None = None,
    ) -> None:
        """通过回调推送事件。"""
        if self._stream_callback is None:
            return
        evt = self._make_event(event_type, content, metadata)
        await self._stream_callback(evt)

    async def _sleep_for_demo(self, seconds: float) -> None:
        """Small delay used only by the local fallback implementation."""
        await asyncio.sleep(seconds)
