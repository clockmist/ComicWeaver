"""Script agent — v1.0 panel-level narrative tasks.

Transforms a developed story + character designs into a page-by-page,
panel-by-panel script. Each panel gets a clear narrative purpose — WHAT
this panel needs to communicate to the reader.

Does NOT define visual/cinematography details (shot size, camera angle,
pose hints). Those are left for the StoryboardAgent downstream.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from comicweaver.api import ApiBackendError, OpenAICompatibleLLMClient
from comicweaver.core import (
    AgentContext,
    AgentOutputMeta,
    BaseAgent,
    CharacterDB,
    CharacterDraft,
    CreationMode,
    Dialogue,
    NarrativeStructure,
    PanelTask,
    Scene,
    ScriptInput,
    ScriptOutput,
    ScriptPage,
    StreamEvent,
    StreamEventType,
)


def _build_character_context(character_db: CharacterDB | None) -> str:
    """Build a character reference block for the LLM prompt (v1.0)."""
    if not character_db or not character_db.characters:
        return ""

    lines = ["=== CHARACTER DESIGNS (use these in your script) ==="]
    for cid, cp in character_db.characters.items():
        vt = cp.visual_traits
        lines.append(
            f"- {cid}: {cp.name}"
            f" | gender_tag: {cp.gender_tag}"
            f" | hair: {vt.hair}"
            f" | eyes: {vt.eyes}"
            f" | body: {vt.body}"
            f" | clothing: {vt.clothing}"
            f" | tags: {cp.core_tags[:120] if cp.core_tags else 'N/A'}"
        )
    lines.append("")
    return "\n".join(lines)


_PANEL_SCRIPT_SYSTEM_PROMPT = """You are a professional comic script writer. Your job is to take a developed story and character designs, then produce a PANEL-BY-PANEL comic script. Each panel gets a clear NARRATIVE PURPOSE — what this panel needs to communicate to the reader.

CRITICAL: You are writing a PERFORMABLE SCRIPT, not visual directions. Do NOT specify shot sizes, camera angles, poses, expressions, lighting, or weather. Those are cinematography decisions handled by a downstream agent. You define WHAT each panel must accomplish narratively.

Each panel typically focuses on ONE character. Some panels may have NO character (establishing shots, scene transitions, atmosphere panels, narration-only panels).

Return a single JSON object matching this EXACT structure:
{
  "title": "Comic title",
  "summary": "1-2 sentence summary",
  "genre": ["genre1", "genre2"],
  "pages": [
    {
      "page_number": 1,
      "page_note": "Optional page-level narrative note (e.g., 'This page establishes the world and introduces the protagonist')",
      "panels": [
        {
          "panel_id": "page_001_p01",
          "page_number": 1,
          "order_in_page": 1,
          "narrative_purpose": "What this panel must accomplish narratively. Be specific. Examples: 'Establish the rainy city setting and mood of isolation', 'Introduce the protagonist — show his determination despite exhaustion', 'Reaction beat — his hope shatters as he reads the letter', 'Deliver the key revelation dialogue', 'Transition — time passes, storm clears'",
          "character_id": "char_000",
          "character_action": "WHAT the character does in this panel (narrative action, not visual pose). e.g., 'searches frantically through old photographs', 'slumps against the wall in defeat', 'reaches out to grab the falling object'",
          "dialogue_text": "The exact dialogue line for this panel (leave empty if no dialogue)",
          "dialogue_tone": "Emotional tone of the dialogue: angry, sad, joyful, fearful, determined, sarcastic, desperate, calm, nervous, cold, warm, curious, urgent. Leave empty if no dialogue.",
          "is_thought": false,
          "narration": "Narrator text for this panel (leave empty if no narration)",
          "emotion": "Primary emotion of this panel: tension, sorrow, joy, fear, determination, surprise, anger, calm, hope, despair, wonder, dread",
          "emotion_intensity": 0.7,
          "is_key_panel": false,
          "location": "WHERE this panel takes place (narrative location, not background description)",
          "time_of_day": "morning/afternoon/evening/night",
          "atmosphere": "Mood of the environment: tense, peaceful, oppressive, bright, gloomy, eerie, warm, cold"
        }
      ]
    }
  ]
}

=== RULES ===

1. PANEL COUNT: Each page should have 3-6 panels. Vary panel count for rhythm — not every page needs the same number. The total number of panels across all pages should feel appropriate for the story's pacing.

2. NARRATIVE PURPOSE: Every panel MUST have a clear, specific narrative_purpose. This is the MOST IMPORTANT field. Ask yourself: "What does the reader learn or feel from this panel?" The narrative_purpose should describe the narrative function, NOT the visual look. Good: "Reveal the antagonist's true motive through a tense confrontation." Bad: "Close-up shot of the antagonist's face."

3. CHARACTER ACTIONS: Describe WHAT the character does in narrative terms. Good: "confronts the antagonist about the betrayal." Bad: "standing with arms crossed, angry expression." The downstream agent will figure out the visual pose.

4. DIALOGUE: Assign each dialogue line to a specific panel. One panel = one key dialogue beat. If a conversation needs multiple lines, spread them across consecutive panels for natural pacing. dialogue_tone must be specific — never use "neutral" unless truly neutral.

5. EMOTIONAL ARC: emotion_intensity should flow naturally across panels. Build tension gradually, peak at key moments, and allow release. Not every panel needs high intensity — quiet moments create contrast.

6. PAGE RHYTHM: Each page should have one is_key_panel (focal panel). This is the most important narrative beat on the page — often the climax of the page's mini-arc. Vary the position of the key panel across pages.

7. CHARACTER USAGE: Reference characters by their char_id (char_000, char_001, etc.) from the character designs provided. Panels with no character (character_id="") are for establishing shots, transitions, and atmosphere.

8. LOCATION/ATMOSPHERE: Provide narrative context for each panel's setting. These help the downstream agent understand WHERE the story is happening, not HOW to draw it.

9. LANGUAGE: Write in the language specified in the user prompt. For Chinese (zh), write all text fields in Chinese (narrative_purpose, character_action, dialogue_text, narration, location, atmosphere, etc.).

10. Output ONLY the JSON object, no markdown, no extra text."""


class ScriptAgent(BaseAgent[ScriptInput, ScriptOutput]):
    """Script agent — story + character designs → panel-level script."""

    name = "script_agent"
    version = "1.0.0"
    rubric_id = "rubric_script_v1"

    async def run(self, inputs: ScriptInput, context: AgentContext) -> ScriptOutput:
        if not self.config.llm.is_available:
            raise ApiBackendError(
                "LLM API is not configured — ScriptAgent requires a working LLM "
                "to generate panel-level scripts."
            )
        return await self._run_api(inputs, context)

    async def _run_api(self, inputs: ScriptInput, context: AgentContext) -> ScriptOutput:
        from pydantic import ValidationError

        client = OpenAICompatibleLLMClient(self.config.llm)

        # Build story context
        story = inputs.story
        if story:
            story_context = _build_story_context(story)
        else:
            story_context = f'Create a comic script based on this idea:\n\n"{inputs.raw_text}"\n\n'

        # Build character context (v1.0)
        char_context = _build_character_context(inputs.character_db)

        target_panels = inputs.target_pages * inputs.target_panels_per_page

        user_message = (
            f"{story_context}"
            f"{char_context}"
            f"\n=== SCRIPT PARAMETERS ===\n"
            f"- Target pages: {inputs.target_pages}\n"
            f"- Target panels per page: {inputs.target_panels_per_page}\n"
            f"- Total target panels: approximately {target_panels}\n"
            f"- Style: {inputs.style_hint or 'manga'}\n"
            f"- Language: {inputs.language}\n\n"
            f"IMPORTANT:\n"
            f"- Write ALL text fields in {'Chinese' if inputs.language == 'zh' else inputs.language}\n"
            f"- Assign specific char_id values from the character designs above\n"
            f"- Some panels may have no character (character_id='') for establishing/transition panels\n"
            f"- Every panel MUST have a meaningful narrative_purpose\n"
            f"- Vary emotion_intensity across panels for dramatic rhythm\n"
        )

        try:
            data = await asyncio.to_thread(
                client.complete_json,
                _PANEL_SCRIPT_SYSTEM_PROMPT,
                {"user_message": user_message},
            )
        except ApiBackendError:
            raise ApiBackendError(
                "ScriptAgent LLM call failed — cannot generate panel script."
            ) from None

        try:
            output = ScriptOutput.model_validate(data)
        except ValidationError:
            if len(data) == 1 and isinstance(list(data.values())[0], dict):
                try:
                    output = ScriptOutput.model_validate(list(data.values())[0])
                except ValidationError:
                    raise ApiBackendError(
                        "ScriptAgent received invalid JSON from LLM."
                    )
            else:
                raise ApiBackendError(
                    "ScriptAgent received invalid JSON from LLM."
                )

        title = output.title
        if not title or title.strip().lower() in ("untitled", "无题", ""):
            title = story.title if story else "Untitled"
            output = output.model_copy(update={"title": title})

        total_panels = sum(len(p.panels) for p in output.pages)

        return output.model_copy(update={
            "meta": AgentOutputMeta(
                agent=self.name,
                version=self.version,
                inputs_hash=self._inputs_hash(inputs),
                retry_count=context.retry_count,
                self_check_notes=[
                    f"LLM: {self.config.llm.provider}",
                    f"Output: {len(output.pages)} pages, {total_panels} panels",
                    f"Characters in context: {len(inputs.character_db.characters) if inputs.character_db else 0}",
                ],
            )
        })

    async def astream(
        self, inputs: ScriptInput, context: AgentContext
    ) -> AsyncIterator[StreamEvent]:
        yield self._make_event(StreamEventType.LOG, "开始设计面板级剧本...")
        yield self._make_event(StreamEventType.PROGRESS, 0.1)

        await self._sleep_for_demo(0.15)
        char_count = len(inputs.character_db.characters) if inputs.character_db else 0
        yield self._make_event(
            StreamEventType.THINKING,
            f"基于{char_count}个角色设计，规划{inputs.target_pages}页面板叙事..."
        )
        yield self._make_event(StreamEventType.PROGRESS, 0.3)

        await self._sleep_for_demo(0.15)
        yield self._make_event(StreamEventType.THINKING, "为每格分配叙事任务与对白...")
        yield self._make_event(StreamEventType.PROGRESS, 0.6)

        await self._sleep_for_demo(0.15)
        yield self._make_event(StreamEventType.THINKING, "设计情感曲线与页面节奏...")
        yield self._make_event(StreamEventType.PROGRESS, 0.85)

        output = await self.run(inputs, context)
        total_panels = sum(len(p.panels) for p in output.pages)
        yield self._make_event(
            StreamEventType.PARTIAL_OUTPUT,
            {
                "title": output.title,
                "pages": len(output.pages),
                "total_panels": total_panels,
            },
        )
        yield self._make_event(StreamEventType.PROGRESS, 1.0)
        yield self._make_event(StreamEventType.DONE, output.model_dump())


# ---------------------------------------------------------------------------
# Helpers (shared with _build_story_context from v0.2)
# ---------------------------------------------------------------------------

def _build_story_context(story) -> str:
    """Build story context for ScriptAgent LLM prompt from StoryOutput."""
    story_text = getattr(story, "story_text", "") or ""
    author_note = getattr(story, "author_note", "") or ""
    tone = getattr(story, "tone", "") or ""
    setting = getattr(story, "setting", "") or ""
    characters = getattr(story, "characters", []) or []

    if story_text:
        char_lines = "\n".join(
            f"  - {c.get('name', '?')} ({c.get('role', '?')}): {c.get('brief_description', '')}"
            for c in characters
        )
        return (
            f"=== DEVELOPED STORY ===\n"
            f"Title: {story.title}\n"
            f"Author's Note: {author_note}\n"
            f"Tone/Mood: {tone}\n"
            f"Genre: {', '.join(story.genre)}\n"
            f"Setting: {setting}\n"
            f"Core Conflict: {story.core_conflict}\n"
            f"Story Characters (informational — use character designs below for visual details):\n{char_lines}\n"
            f"\n=== FULL STORY TEXT ===\n"
            f"{story_text}\n"
        )

    # Legacy v0.1 fallback
    summary = getattr(story, "summary", "") or ""
    premise = getattr(story, "premise", "") or ""
    if summary or premise:
        return (
            f"=== DEVELOPED STORY (legacy) ===\n"
            f"Title: {story.title}\n"
            f"Premise: {premise}\n"
            f"Summary: {summary}\n"
            f"Genre: {', '.join(story.genre)}\n"
        )

    return ""
