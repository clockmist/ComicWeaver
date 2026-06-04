"""Story development agent — v0.2.

Expands raw user input into a complete, readable narrative story. The story is
prose-focused (not a structured script), designed to be adapted into a comic
script by the downstream ScriptAgent.

Story length scales with target_pages:
  1-2 pages → ~200 words (micro fiction)
  3-4 pages → ~400-500 words (short scene)
  5-8 pages → ~800-1000 words (short story)
  9-12 pages → ~1200-1500 words (developed short story)
  13-16 pages → ~1800-2200 words (detailed story)
  17-20 pages → ~2500-3000 words (rich narrative)
"""
from __future__ import annotations

import asyncio
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

# ---------------------------------------------------------------------------
# Page-count → word-count mapping
# ---------------------------------------------------------------------------

_PAGE_TO_WORDS: dict[int, tuple[int, int]] = {
    1: (150, 200),
    2: (200, 300),
    3: (400, 500),
    4: (400, 500),
    5: (800, 1000),
    6: (800, 1000),
    7: (800, 1000),
    8: (800, 1000),
    9: (1200, 1500),
    10: (1200, 1500),
    11: (1200, 1500),
    12: (1200, 1500),
    13: (1800, 2200),
    14: (1800, 2200),
    15: (1800, 2200),
    16: (1800, 2200),
    17: (2500, 3000),
    18: (2500, 3000),
    19: (2500, 3000),
    20: (2500, 3000),
}


def _word_count_guidance(target_pages: int) -> str:
    """Return human-readable word count guidance for the given page count."""
    entry = _PAGE_TO_WORDS.get(target_pages, (800, 1000))
    return f"{entry[0]}-{entry[1]} words"


# ---------------------------------------------------------------------------
# LLM System Prompt (v0.2 — prose-focused storytelling)
# ---------------------------------------------------------------------------

_STORY_SYSTEM_PROMPT = """You are a professional story writer. Your job is to take a raw story idea and expand it into a complete, self-contained narrative story that will later be adapted into a black-and-white manga comic.

CRITICAL: You are writing a STORY, not a comic script. Do NOT write panel descriptions, camera directions, shot lists, or visual/cinematography instructions. Just tell the story in narrative prose. The comic adaptation will be handled by a separate process.

Return a single JSON object matching this EXACT structure:
{
  "title": "A compelling, creative title for the story",
  "author_note": "1-2 sentence high-concept summary that captures the essence and hook",
  "tone": "The narrative tone/mood in 2-5 words (e.g., dark noir mystery, whimsical adventure, tense psychological thriller, melancholic romance)",
  "genre": ["cyberpunk noir", "psychological thriller"],
  "story_text": "THE COMPLETE NARRATIVE STORY — this is the main output. Write in flowing prose with clear paragraphs. The story should have a clear beginning (setup), middle (conflict/development), and end (resolution). Include vivid sensory details that paint visual pictures. Develop characters through their actions and dialogue. The story must feel complete and satisfying. Target length will be specified in the user prompt.",
  "characters": [
    {
      "name": "Character Name",
      "role": "protagonist",
      "brief_description": "1-2 sentence character sketch covering personality and key traits"
    }
  ],
  "core_conflict": "What is the central dramatic conflict? Be specific.",
  "setting": "Where and when does this story take place? Describe the world briefly.",
  "target_pages": 4
}

=== RULES ===
1. story_text is the PRIMARY output — pour your creative energy into making it a compelling read.
2. Write in the language specified in the user prompt. For Chinese (zh), write in Chinese. For English, write in English.
3. Genre should be specific and evocative: "cyberpunk noir" not "sci-fi", "school romance with supernatural elements" not "romance".
4. tone should help downstream artists understand the visual mood without being visual instructions.
5. Characters should be introduced naturally within the story. The characters list is a quick reference, not the main characterization.
6. The story must have strong visual potential — moments that would make striking images — but describe them as STORY moments, not camera shots.
7. Output ONLY the JSON object, no markdown, no extra text."""


class StoryAgent(BaseAgent[StoryInput, StoryOutput]):
    """Story development agent — expands raw ideas into narrative stories."""

    name = "story_agent"
    version = "0.2.0"
    rubric_id = "rubric_script_v1"

    async def run(self, inputs: StoryInput, context: AgentContext) -> StoryOutput:
        if not self.config.llm.is_available:
            raise ApiBackendError(
                "LLM API is not configured — StoryAgent requires a working LLM "
                "to generate narrative stories. Please configure llm in configs/comicweaver.yaml."
            )
        return await self._run_api(inputs, context)

    async def _run_api(self, inputs: StoryInput, context: AgentContext) -> StoryOutput:
        from pydantic import ValidationError

        client = OpenAICompatibleLLMClient(self.config.llm)

        word_guidance = _word_count_guidance(inputs.target_pages)
        user_message = (
            f"Write a complete narrative story based on this idea:\n\n"
            f'"{inputs.raw_text}"\n\n'
            f"Parameters:\n"
            f"- Target comic pages: {inputs.target_pages}\n"
            f"- Target story length: approximately {word_guidance}\n"
            f"- Genre hint: {inputs.genre_hint or 'not specified'}\n"
            f"- Style context: {inputs.style_hint or 'manga'}\n"
            f"- Language: {inputs.language}\n\n"
            f"Write in {'Chinese' if inputs.language == 'zh' else inputs.language}. "
            f"Create a story with strong emotional impact and memorable characters. "
            f"Focus on WHAT happens in the story, not HOW to draw it."
        )

        try:
            data = await asyncio.to_thread(
                client.complete_json,
                _STORY_SYSTEM_PROMPT,
                {"user_message": user_message},
            )
        except ApiBackendError:
            raise ApiBackendError(
                f"StoryAgent LLM call failed — cannot generate story. "
                f"Check your LLM configuration (provider: {self.config.llm.provider})."
            ) from None

        try:
            output = StoryOutput.model_validate(data)
        except ValidationError:
            if len(data) == 1 and isinstance(list(data.values())[0], dict):
                try:
                    output = StoryOutput.model_validate(list(data.values())[0])
                except ValidationError:
                    raise ApiBackendError(
                        "StoryAgent received invalid JSON from LLM — "
                        "the model did not return the expected story format."
                    )
            else:
                raise ApiBackendError(
                    "StoryAgent received invalid JSON from LLM — "
                    "the model did not return the expected story format."
                )

        # Quality check: ensure title is meaningful
        title = output.title
        if not title or title.strip().lower() in ("untitled", "无题", ""):
            title = _make_story_title(inputs.raw_text)
            output = output.model_copy(update={"title": title})

        word_count = len(output.story_text) if output.story_text else 0

        return output.model_copy(update={
            "meta": AgentOutputMeta(
                agent=self.name,
                version=self.version,
                inputs_hash=self._inputs_hash(inputs),
                retry_count=context.retry_count,
                self_check_notes=[
                    f"LLM: {self.config.llm.provider}",
                    f"Story: {word_count} words, {len(output.characters)} characters",
                    f"Target: {inputs.target_pages} pages",
                ],
            )
        })

    async def astream(
        self, inputs: StoryInput, context: AgentContext
    ) -> AsyncIterator[StreamEvent]:
        """Stream events for UI progress display."""
        pages = inputs.target_pages
        yield self._make_event(StreamEventType.LOG, "开始分析故事创意...")
        yield self._make_event(StreamEventType.PROGRESS, 0.1)

        await self._sleep_for_demo(0.15)
        yield self._make_event(
            StreamEventType.THINKING,
            f"根据{pages}页篇幅规划故事结构..."
        )
        yield self._make_event(StreamEventType.PROGRESS, 0.3)

        await self._sleep_for_demo(0.15)
        yield self._make_event(StreamEventType.THINKING, "塑造角色并确立故事基调...")
        yield self._make_event(StreamEventType.PROGRESS, 0.55)

        await self._sleep_for_demo(0.15)
        yield self._make_event(StreamEventType.THINKING, "撰写完整叙事故事...")
        yield self._make_event(StreamEventType.PROGRESS, 0.8)

        output = await self.run(inputs, context)
        word_count = len(output.story_text) if output.story_text else 0
        yield self._make_event(
            StreamEventType.PARTIAL_OUTPUT,
            {
                "title": output.title,
                "genre": output.genre,
                "tone": output.tone,
                "word_count": word_count,
                "character_count": len(output.characters),
            },
        )
        yield self._make_event(StreamEventType.PROGRESS, 1.0)
        yield self._make_event(StreamEventType.DONE, output.model_dump())


def _make_story_title(raw: str) -> str:
    cleaned = raw.strip().replace("\n", " ")
    if not cleaned:
        return "无题故事"
    title = cleaned[:20]
    if len(cleaned) > 20:
        title = title.rsplit(" ", 1)[0] if " " in title else title
    return f"《{title}...》" if len(cleaned) > 20 else f"《{title}》"
