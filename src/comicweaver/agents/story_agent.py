"""Story development agent — v0.4.

Expands raw user input into a complete story outline with theme, structure,
character arcs, and emotional throughline. This is the FIRST step in the
revised pipeline, feeding into ScriptAgent for comic adaptation.
"""
from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator

from comicweaver.api import ApiBackendError, OpenAICompatibleLLMClient
from comicweaver.core import (
    AgentContext,
    AgentOutputMeta,
    BaseAgent,
    StoryInput,
    StoryOutput,
    StreamEvent,
    StreamEventType,
)

_STORY_SYSTEM_PROMPT = """You are a professional story developer for a black-and-white manga comic system. Your job is to take a raw story idea and expand it into a complete, well-structured story outline that will later be adapted into a comic script.

CRITICAL: You are NOT writing the comic script yet. You are developing the STORY that the comic will tell. Think like a novelist or screenwriter, not a comic artist.

Return a single JSON object matching this EXACT structure:
{
  "title": "A compelling, creative title for the story — never generic",
  "premise": "1-2 sentence high-concept summary that captures the hook",
  "theme": "The central theme or message — what is this story really about?",
  "genre": ["action", "drama"],
  "summary": "A 3-5 paragraph complete story summary covering the full narrative arc from beginning to end. Include key plot points, turning points, and the resolution.",
  "core_conflict": "What is the central conflict? Who/what is the protagonist struggling against?",
  "act_structure": "Describe the narrative structure. For a 3-act structure: Setup (establishing the world and conflict), Confrontation (rising stakes and complications), Resolution (climax and denouement). Be specific about what happens in each act.",
  "character_arcs": [
    {
      "char_id": "char_000",
      "name": "Character Name",
      "arc_description": "How this character changes across the story",
      "starting_state": "Who they are at the beginning",
      "ending_state": "Who they become by the end"
    }
  ],
  "emotional_throughline": "Describe the READER'S emotional journey. How should the reader feel as they progress through the story? Where are the peaks and valleys?",
  "target_pages": 4
}

RULES:
1. Title must be creative and specific — NOT "Untitled", NOT generic like "The Story".
2. Genre should be specific: e.g., "cyberpunk noir", "fantasy adventure", "school romance", "psychological thriller".
3. Summary must cover the COMPLETE story with a clear ending. Don't leave it open-ended unless that's the artistic intent.
4. Character arcs should show genuine change/growth. Characters who stay the same are boring.
5. act_structure should be concrete about plot events, not abstract about "the stakes rise".
6. emotional_throughline should describe the reader's emotional experience, not the characters'.
7. Focus on VISUAL storytelling potential — great comics come from stories with strong visual moments.
8. Output ONLY the JSON object, no markdown, no extra text."""


class StoryAgent(BaseAgent[StoryInput, StoryOutput]):
    """Story development agent — expands raw ideas into full story outlines."""

    name = "story_agent"
    version = "0.1.0"
    rubric_id = "rubric_script_v1"  # reviewed alongside script for now

    async def run(self, inputs: StoryInput, context: AgentContext) -> StoryOutput:
        if self.config.llm.is_available:
            try:
                return await self._run_api(inputs, context)
            except ApiBackendError:
                if not self.config.runtime.fallback_to_local:
                    raise
        return await self._run_local(inputs, context)

    async def _run_api(self, inputs: StoryInput, context: AgentContext) -> StoryOutput:
        from pydantic import ValidationError

        client = OpenAICompatibleLLMClient(self.config.llm)

        user_message = (
            f"Develop a complete story outline based on this idea:\n\n"
            f'"{inputs.raw_text}"\n\n'
            f"Parameters:\n"
            f"- Target comic pages: {inputs.target_pages}\n"
            f"- Genre hint: {inputs.genre_hint or 'not specified'}\n"
            f"- Style: {inputs.style_hint or 'manga'}\n"
            f"- Language: {inputs.language}\n\n"
            f"Create 2-3 main characters with clear arcs. "
            f"Design a story with strong visual potential for black-and-white manga. "
            f"Think about moments that would make striking comic panels."
        )

        try:
            data = await asyncio.to_thread(
                client.complete_json,
                _STORY_SYSTEM_PROMPT,
                {"user_message": user_message},
            )
        except ApiBackendError:
            if not self.config.runtime.fallback_to_local:
                raise
            return await self._run_local(inputs, context)

        try:
            output = StoryOutput.model_validate(data)
        except ValidationError:
            # Try unwrapping nested structure
            if len(data) == 1 and isinstance(list(data.values())[0], dict):
                try:
                    output = StoryOutput.model_validate(list(data.values())[0])
                except ValidationError:
                    if self.config.runtime.fallback_to_local:
                        return await self._run_local(inputs, context)
                    raise
            elif self.config.runtime.fallback_to_local:
                return await self._run_local(inputs, context)
            else:
                raise

        # Quality check: ensure title is meaningful
        title = output.title
        if not title or title.strip().lower() in ("untitled", "无题", ""):
            title = _make_story_title(inputs.raw_text)
            output = output.model_copy(update={"title": title})

        return output.model_copy(update={
            "meta": AgentOutputMeta(
                agent=self.name,
                version=self.version,
                inputs_hash=self._inputs_hash(inputs),
                retry_count=context.retry_count,
                self_check_notes=[
                    f"LLM API provider: {self.config.llm.provider}",
                    f"Story developed with {len(output.character_arcs)} character arcs",
                ],
            )
        })

    async def _run_local(self, inputs: StoryInput, context: AgentContext) -> StoryOutput:
        """Local fallback: generate a basic story outline from the raw text."""
        await self._sleep_for_demo(0.2)

        raw = inputs.raw_text.strip()
        title = _make_story_title(raw)

        return StoryOutput(
            title=title,
            premise=f"A story about: {raw[:100]}",
            theme="Self-discovery and growth",
            genre=[inputs.genre_hint or "drama"],
            summary=(
                f"The story follows the events surrounding: {raw}. "
                f"The protagonist faces challenges that test their resolve, "
                f"leading to a moment of truth where everything changes. "
                f"Through perseverance and growth, they find a resolution "
                f"that transforms their understanding of the world."
            ),
            core_conflict="The protagonist must overcome their inner demons while facing external threats.",
            act_structure=(
                "Setup: The protagonist's ordinary world is established, then disrupted by an inciting incident. "
                "Confrontation: Stakes rise as the protagonist faces escalating challenges and setbacks. "
                "Resolution: The climax forces a final confrontation, leading to transformation and resolution."
            ),
            character_arcs=[
                {
                    "char_id": "char_000",
                    "name": "Protagonist",
                    "arc_description": "Journey from uncertainty to purpose",
                    "starting_state": "Uncertain, searching for meaning",
                    "ending_state": "Determined, having found their path",
                },
                {
                    "char_id": "char_001",
                    "name": "Antagonist",
                    "arc_description": "Represents the opposing force or ideal",
                    "starting_state": "Powerful and confident",
                    "ending_state": "Defeated or transformed by the confrontation",
                },
            ],
            emotional_throughline=(
                "The reader begins with curiosity, builds empathy for the protagonist, "
                "feels tension during the confrontation, experiences catharsis at the climax, "
                "and ends with a sense of resolution and reflection."
            ),
            target_pages=inputs.target_pages,
            meta=AgentOutputMeta(
                agent=self.name,
                version=self.version,
                inputs_hash=self._inputs_hash(inputs),
                self_check_notes=[
                    "Local fallback — story generated from rules",
                    "Configure LLM API for richer story development",
                ],
            ),
        )

    async def astream(
        self, inputs: StoryInput, context: AgentContext
    ) -> AsyncIterator[StreamEvent]:
        """Stream events for UI progress display."""
        yield self._make_event(StreamEventType.LOG, "开始分析故事创意...")
        yield self._make_event(StreamEventType.PROGRESS, 0.1)

        await self._sleep_for_demo(0.15)
        yield self._make_event(StreamEventType.THINKING, "构思故事主题与核心冲突...")
        yield self._make_event(StreamEventType.PROGRESS, 0.3)

        await self._sleep_for_demo(0.15)
        yield self._make_event(StreamEventType.THINKING, "设计角色成长弧线...")
        yield self._make_event(StreamEventType.PROGRESS, 0.55)

        await self._sleep_for_demo(0.15)
        yield self._make_event(StreamEventType.THINKING, "构建叙事结构与情感旅程...")
        yield self._make_event(StreamEventType.PROGRESS, 0.8)

        output = await self.run(inputs, context)
        yield self._make_event(
            StreamEventType.PARTIAL_OUTPUT,
            {
                "title": output.title,
                "genre": output.genre,
                "character_count": len(output.character_arcs),
            },
        )
        yield self._make_event(StreamEventType.PROGRESS, 1.0)
        yield self._make_event(StreamEventType.DONE, output.model_dump())


def _make_story_title(raw: str) -> str:
    cleaned = raw.strip().replace("\n", " ")
    if not cleaned:
        return "无题故事"
    # Take first meaningful segment as title base
    title = cleaned[:20]
    if len(cleaned) > 20:
        title = title.rsplit(" ", 1)[0] if " " in title else title
    return f"《{title}...》" if len(cleaned) > 20 else f"《{title}》"
