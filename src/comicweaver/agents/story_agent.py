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
            if not self.config.runtime.fallback_to_local:
                raise
            return await self._run_local(inputs, context)

        try:
            output = StoryOutput.model_validate(data)
        except ValidationError:
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

    async def _run_local(self, inputs: StoryInput, context: AgentContext) -> StoryOutput:
        """Local fallback: generate a basic story from the raw idea."""
        await self._sleep_for_demo(0.2)

        raw = inputs.raw_text.strip()
        title = _make_story_title(raw)
        pages = inputs.target_pages

        # Scale story detail with page count
        if pages <= 2:
            story_text = (
                f"在一条寂静的街道上，{raw}。\n\n"
                f"这个瞬间虽然短暂，却改变了故事的走向。"
            )
        elif pages <= 4:
            story_text = (
                f"故事从{raw}开始。\n\n"
                f"主角面对着未知的挑战，内心充满矛盾。每一步都考验着意志，"
                f"每一个选择都通向不同的命运。\n\n"
                f"在关键时刻，主角做出了决定性的行动。这个选择带来了改变，"
                f"也揭示了故事真正的主题——关于勇气、成长与自我发现。\n\n"
                f"故事的结尾留下了回味的空间：世界已经不同，而主角也已不再是当初的那个人。"
            )
        elif pages <= 8:
            story_text = (
                f"在这个故事的开端，{raw}。\n\n"
                f"起初看似平静的日常很快被打破，主角被卷入了一场超出预想的冒险。"
                f"随着故事展开，新的角色陆续登场——有盟友，也有对手。"
                f"每个人都带着自己的秘密和目标，让局势变得更加复杂。\n\n"
                f"中段的转折让主角面临真正的考验。曾经相信的一切都被质疑，"
                f"而前方的道路充满危险。但正是在最黑暗的时刻，主角发现了内在的力量。\n\n"
                f"高潮来临，所有的线索汇聚到一起。一场决定性的对抗让故事达到顶峰。"
                f"主角做出了关键的选择——不是为了胜利，而是为了守护重要的东西。\n\n"
                f"故事的结尾既完整又意味深长。世界恢复了平静，但主角知道，"
                f"这段旅程改变了一切。回望来路，每一步都是必要的。"
            )
        else:
            story_text = (
                f"这是一个关于{raw}的故事。\n\n"
                f"开篇，世界被细致地描绘出来——{inputs.style_hint or '漫画'}风格下，"
                f"每一个场景都充满了氛围。主角的日常生活被一个事件打破，"
                f"这个契机开启了长达{pages}页的叙事旅程。\n\n"
                f"第一幕建立了故事的基础：主角是谁，这个世界是什么样的，"
                f"以及即将到来的冲突。配角的登场丰富了故事的层次，"
                f"每个人都有自己想要守护的东西。\n\n"
                f"第二幕将冲突逐步升级。主角遭遇挫折，发现真相并非表面那么简单。"
                f"信任被考验，联盟被打破又重新建立。每一次失败都让主角更接近"
                f"故事的核心——关于人性、关于选择、关于代价。\n\n"
                f"第三幕将所有线索汇聚于高潮。在最终的对抗中，主角不仅要面对"
                f"外在的敌人，更要面对内心最大的恐惧。结局不是简单的胜利或失败，"
                f"而是一种转变——主角已经成长，世界也因此改变。\n\n"
                f"尾声给读者留下思考：故事虽然结束，但它的影响仍在延续。"
                f"那些选择、那些牺牲、那些不期而遇的温暖——共同构成了这个故事的意义。"
            )

        return StoryOutput(
            title=title,
            author_note=f"一个关于{raw[:60]}的故事",
            tone="dramatic",
            genre=[inputs.genre_hint or "drama"],
            story_text=story_text,
            characters=[
                {"name": "主角", "role": "protagonist",
                 "brief_description": raw[:80]},
                {"name": "对手", "role": "antagonist",
                 "brief_description": "代表对立力量的角色"},
            ],
            core_conflict="主角必须在外在挑战与内心挣扎之间找到平衡",
            setting="一个充满可能性的世界",
            target_pages=pages,
            meta=AgentOutputMeta(
                agent=self.name,
                version=self.version,
                inputs_hash=self._inputs_hash(inputs),
                self_check_notes=[
                    f"Local fallback — {len(story_text)} chars",
                    "Configure LLM API for richer story generation",
                ],
            ),
        )

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
