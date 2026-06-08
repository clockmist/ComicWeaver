"""Script agent — v2.1 sliding-window chunking for panelization.

Two-stage architecture:
  Stage 1 (Pagination):  1 LLM call → page-level outlines with narrative arcs
  Stage 2 (Panelization): Chunks of pages processed sequentially; pages within
                          a chunk are handled in ONE LLM call for max coherence.

v1.0: single LLM call for all pages + panels
v2.1: chunked — sequential chunks, single LLM call per chunk (Option A)
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from pydantic import ValidationError

from comicweaver.api import ApiBackendError, OpenAICompatibleLLMClient
from comicweaver.core import (
    AgentContext,
    AgentOutputMeta,
    BaseAgent,
    CharacterDB,
    PanelTask,
    ScriptInput,
    ScriptOutput,
    ScriptPage,
    StreamEvent,
    StreamEventType,
)


# ============================================================================
# CONFIG
# ============================================================================
DEFAULT_CHUNK_SIZE = 4  # 每块生成多少页的面板


# ============================================================================
# STAGE 1: Pagination System Prompt
# ============================================================================

_PAGINATION_SYSTEM_PROMPT = """You are a comic pagination expert. Divide a developed story into PAGE-LEVEL outlines.

CRITICAL: A comic page is a VISUAL experience. Every page outline must provide rich visual context for the panel writer. The "setting" field is NOT a label — it is a vivid visual description of the environment the reader will SEE.

Each page outline must include:
- narrative_arc: what story beats this page covers
- emotional_shift: how emotions change (e.g., "curiosity → tension → shock")
- key_moment: the single most important narrative beat — described as a VISUAL moment
- key_panel_hint: revelation / confrontation / quiet_moment / action_peak / emotional_climax / transition
- involved_characters: list of char_ids
- setting: ★ VISUAL DESCRIPTION of the environment — NOT just "street" or "room". Describe what fills the space: architecture, props, lighting conditions, distinctive visual features. Example: "废弃工厂内部，破碎天窗投下光柱，生锈传送带横贯，翻倒货箱和散落零件" NOT "工厂"
- time_of_day: morning/afternoon/evening/night/dawn/dusk
- atmosphere: tense/peaceful/oppressive/bright/gloomy/eerie/warm/cold/foggy/rainy/smoky/dusty — what the reader FEELS looking at this page
- panel_count_hint: 3-6
- page_note: why this page matters

Return JSON:
{{
  "title": "...",
  "summary": "...",
  "genre": ["..."],
  "pages": [
    {{
      "page_number": 1,
      "narrative_arc": "...",
      "emotional_shift": "...",
      "key_moment": "...",
      "key_panel_hint": "...",
      "involved_characters": ["char_000"],
      "setting": "Rich visual environment description here...",
      "time_of_day": "morning/afternoon/evening/night/dawn/dusk",
      "atmosphere": "...",
      "panel_count_hint": 4,
      "page_note": "..."
    }}
  ]
}}

LANGUAGE: {language}. Output ONLY valid JSON object, no markdown, no extra text.

=== REVISION MODE ===
If the user message contains a "REVISION REQUEST" section, you are in revision mode. In this case:
- You will see both the previous script structure and the user's feedback.
- Revise the page outlines according to the user's specific requests.
- ONLY change what the user asked to change — preserve all other elements from the previous version.
- If the user's feedback is unclear, err on the side of keeping the original."""


# ============================================================================
# STAGE 2: Chunked Panelization System Prompt
# ============================================================================

_CHUNK_PANELIZATION_SYSTEM_PROMPT = """You are a panel-level comic script writer. You receive {chunk_size} consecutive page outlines and expand EACH into individual panels.

=== THE MOST IMPORTANT RULE: SCENE-FIRST WRITING ===
You are writing for a VISUAL medium. Every panel description must answer ONE question:
"What does the reader actually SEE in this frame?"

A comic panel is NOT a character portrait. It is a VIEW into a world where something is happening.
- BAD: "林晓在观察周围环境" (describes a character's mental state — invisible to the reader)
- GOOD: "废弃工厂车间，生锈的传送带横贯画面，破碎的天窗投下条状光线，林晓蹲在翻倒的货箱后，只露出半张脸和一只警惕的眼睛" (describes what the reader SEES)

CRITICAL CONTEXT AWARENESS:
- These pages are CONSECUTIVE and narratively connected.
- The "story_so_far" field tells you what happened before this block — use it to maintain continuity.
- The last page of this block should set up a smooth transition to the next block (if there is one).

For EACH page, return panels with this structure.

=== DIALOGUE & NARRATION RULES (CRITICAL — READ THIS FIRST) ===

COMIC STORYTELLING IS DRIVEN BY WORDS AND IMAGES TOGETHER.
Dialogue and narration are NOT optional decoration — they ARE the story.
A comic with empty bubbles is a broken comic.

DIALOGUE RULES (character panels):
- EVERY character panel where the character is visible and engaged SHOULD have dialogue.
- Dialogue should be NATURAL and SUBSTANTIAL — 1 to 3 full sentences, not one-word grunts.
- Characters should sound like real people talking: incomplete sentences, emotional outbursts, questions, interruptions are all valid.
- Write dialogue that REVEALS CHARACTER — what they say and HOW they say it tells the reader who they are.
- When multiple characters are in a scene, create CONVERSATION — back-and-forth exchanges across consecutive panels.
- ONLY leave dialogue_text empty for: true silent reaction panels (shock without words), pure action panels (character mid-combat/mid-chase), or establishing shots where the character is tiny in the frame.
- "dialogue_text" MUST be the character's exact spoken words — NO "name:" prefix, NO quotation marks around the entire line.

NARRATION RULES (no-character panels):
- EVERY no-character panel (character_id="") MUST have narration filled.
- Narration serves as the "voice-over" of the comic — it provides context, foreshadowing, internal monologue, or atmospheric description.
- Narration should be 1-2 full sentences, written in a literary/narrative style.
- Narration panels are the STORYTELLER'S VOICE — use them to set mood, reveal backstory, or bridge scenes.
- Exception: the very first panel of each page can be a pure establishing shot without narration if the visuals alone set the scene.

DENSITY REQUIREMENT:
- At least 60% of ALL panels across the story MUST have either dialogue_text OR narration filled.
- A page with 4 panels should typically have 3 panels with text (dialogue or narration).
- Consecutive silent panels are ONLY allowed during pure action sequences.

Example — character panel WITH dialogue (note: multi-sentence, natural speech):
{{
  "panel_id": "page_NNN_pNN",
  "page_number": N,
  "order_in_page": 1,
  "narrative_purpose": "废弃工厂车间，生锈传送带横贯画面，林晓从翻倒货箱后站起身，表情凝重地看向远方",
  "character_id": "char_001",
  "character_action": "从货箱后站起身，拍掉膝盖上的灰尘，目光直视前方",
  "dialogue_text": "这个地方……已经没什么可留恋的了。我明天一早就走。",
  "dialogue_tone": "determined",
  "is_thought": false,
  "narration": "",
  "emotion": "determination",
  "emotion_intensity": 0.7,
  "is_key_panel": false,
  "location": "废弃工厂内部，破碎天窗投下光柱，生锈机器和散落零件",
  "time_of_day": "afternoon",
  "atmosphere": "tense"
}}

Example — consecutive character panels showing CONVERSATION (back-and-forth dialogue):
Panel 1:
{{
  "panel_id": "page_NNN_p01",
  "character_id": "char_001",
  "dialogue_text": "你终于来了。我等了你整整三个小时。",
  "dialogue_tone": "cold"
}}
Panel 2:
{{
  "panel_id": "page_NNN_p02",
  "character_id": "char_002",
  "dialogue_text": "抱歉。路上遇到了点麻烦——有人跟踪我，我绕了远路。",
  "dialogue_tone": "nervous"
}}
Panel 3:
{{
  "panel_id": "page_NNN_p03",
  "character_id": "char_001",
  "dialogue_text": "跟踪？你还记得那个人长什么样吗？",
  "dialogue_tone": "urgent"
}}

Example — no-character panel WITH narration (1-2 full sentences):
{{
  "panel_id": "page_NNN_pNN",
  "page_number": N,
  "order_in_page": 2,
  "narrative_purpose": "雨夜的城市街道，霓虹灯倒映在积水里，远处有车灯闪过，空无一人的巷口只有风吹动的塑料袋",
  "character_id": "",
  "character_action": "",
  "dialogue_text": "",
  "dialogue_tone": "calm",
  "is_thought": false,
  "narration": "这座城市的夜晚从不真正安静。每一条暗巷里都藏着不愿被人知晓的秘密，而今晚，又多了一个。",
  "emotion": "dread",
  "emotion_intensity": 0.5,
  "is_key_panel": false,
  "location": "夜晚的城市街道，霓虹灯和积水",
  "time_of_day": "night",
  "atmosphere": "gloomy"
}}

Return a single JSON object with a "pages" key containing the array of expanded pages:
{{
  "pages": [
    {{ "page_number": 1, "panels": [...] }},
    {{ "page_number": 2, "panels": [...] }}
  ]
}}

RULES:
1. Generate EXACTLY the requested number of panels per page (from panel_count_hint).
2. Pages in this block must feel CONNECTED — visual and emotional arcs should flow across page boundaries.
3. The LAST page of the block should end with narrative momentum (hook for next block).
4. EXACTLY ONE panel per page has is_key_panel=true.
5. ★ DIALOGUE DENSITY: At least 60% of all panels MUST have dialogue_text or narration filled. For a 4-panel page, at least 3 panels should contain text. Character panels without dialogue should be rare (<20% of character panels). Create genuine CONVERSATIONS across consecutive panels — characters talking to each other, not isolated one-liners.
6. Use character_id from designs; empty string for no-character panels.
7. panel_id format: page_NNN_pNN (e.g. page_001_p01, page_001_p02).
8. Follow the DIALOGUE & NARRATION RULES above strictly: character present → fill dialogue_text (multi-sentence, natural speech); no character → fill narration (1-2 full sentences). Never fill both in one panel.
9. LANGUAGE: {language}. Output ONLY valid JSON object, no markdown, no extra text.

=== REVISION MODE ===
If the user message contains a "REVISION REQUEST" section, you are in revision mode. In this case:
- You will see both the previous panelization results and the user's feedback.
- Revise the panels according to the user's specific requests.
- ONLY change what the user asked to change — preserve all other panels and their structure from the previous version.
- If the user's feedback is unclear, err on the side of keeping the original.

=== SCENE-FIRST WRITING EXAMPLES ===

BAD narrative_purpose (character-centric, no visual information):
  "林晓决定跟踪那个可疑的人"
  "角色表达愤怒和决心"
  "展示主角的内心挣扎"
These are INVISIBLE to the reader. They describe thoughts, not images.

GOOD narrative_purpose (scene-first, visually concrete):
  "雨夜后巷，积水倒映着霓虹招牌的红色光晕，一个穿风衣的模糊人影在巷口闪过，林晓在二十米外的消防梯阴影下，手按在腰间的刀柄上，呼吸凝成白雾"
  "废弃工厂内部，阳光从破碎的屋顶斜射下来形成光柱，粉尘在光中悬浮，林晓跪在倒塌的机器旁，双手扒开瓦砾，露出下面一只苍白的手"
  "拥挤的夜市街道，摊位灯笼连成暖色光带，人群熙攘，林晓站在鱼摊前，身后的玻璃鱼缸映出一个戴兜帽的人正在接近她"

=== COMIC PANEL VARIETY ===
Comics tell stories through varied visual rhythm, but TEXT carries the narrative forward.
- First panel of each page SHOULD be an establishing/location shot: set character_id="" with narration, or include rich environment description with the character. This orients the reader.
- ~10-20% of panels MAY be pure environment/atmosphere shots (character_id="") — these MUST have narration filled.
- The MAJORITY of panels (~70%) should feature characters WITH dialogue — this is a character-driven comic.
- For character panels: show the character IN CONTEXT, with their environment. Dialogue reveals personality.
- Silent/reaction panels are SPECIAL tools — use sparingly (1-2 per page max), only for moments of genuine emotional impact or fast action.
- Vary panel density: some pages can have 3 panels (slower, more atmospheric), others 5-6 (faster, action-heavy)."""


# ============================================================================
# ScriptAgent v2.1
# ============================================================================

class ScriptAgent(BaseAgent[ScriptInput, ScriptOutput]):
    """Script agent v2.1 — story → pages (Stage 1) → chunked panels (Stage 2)."""

    name = "script_agent"
    version = "2.1.0"

    def __init__(self, *args, chunk_size: int = DEFAULT_CHUNK_SIZE, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_size = chunk_size

    async def run(self, inputs: ScriptInput, context: AgentContext) -> ScriptOutput:
        if not self.config.llm.is_available:
            raise ApiBackendError("LLM API not configured.")

        import traceback as _tb

        client = OpenAICompatibleLLMClient(self.config.llm)

        # ---- Shared contexts ----
        story = inputs.story
        story_context = _build_story_context(story) if story else (
            f'Create comic based on: "{inputs.raw_text}"\n\n'
        )
        char_context = _build_character_context(inputs.character_db)
        lang = "Chinese" if inputs.language == "zh" else inputs.language

        # =====================================================================
        # STAGE 1: Pagination (1 call)
        # =====================================================================
        try:
            page_outlines = await self._run_pagination(
                client, inputs, story_context, char_context, lang
            )
        except Exception as exc:
            raise ApiBackendError(
                f"[Stage 1: Pagination] {type(exc).__name__}: {exc}\n{_tb.format_exc()}"
            ) from exc

        all_pages_data = page_outlines["pages"]

        # Validate each page outline has required keys before Stage 2
        _REQUIRED_OUTLINE_KEYS = (
            "page_number", "narrative_arc", "key_moment", "panel_count_hint",
        )
        for i, outline in enumerate(all_pages_data):
            if not isinstance(outline, dict):
                raise ApiBackendError(
                    f"Stage 1 page outline [{i}] is not a dict: "
                    f"{type(outline).__name__} = {str(outline)[:100]}"
                )
            missing = [k for k in _REQUIRED_OUTLINE_KEYS if k not in outline]
            if missing:
                raise ApiBackendError(
                    f"Stage 1 page outline [{i}] missing keys: {missing}. "
                    f"Got keys: {list(outline.keys())[:10]}"
                )

        # =====================================================================
        # STAGE 2: Chunked Panelization
        # =====================================================================
        try:
            pages = await self._run_chunked_panelization(
                client=client,
                all_pages_data=all_pages_data,
                char_context=char_context,
                language=lang,
                chunk_size=self.chunk_size,
                user_guidance=inputs.user_guidance,
                previous_output=inputs.previous_output,
            )
        except Exception as exc:
            raise ApiBackendError(
                f"[Stage 2: Panelization] {type(exc).__name__}: {exc}\n{_tb.format_exc()}"
            ) from exc

        # ---- Assemble ----
        total_panels = sum(len(p.panels) for p in pages)

        return ScriptOutput(
            title=page_outlines.get("title", story.title if story else "Untitled"),
            summary=page_outlines.get("summary", ""),
            genre=page_outlines.get("genre", []),
            pages=pages,
            meta=AgentOutputMeta(
                agent=self.name,
                version=self.version,
                inputs_hash=self._inputs_hash(inputs),
                retry_count=context.retry_count,
                self_check_notes=[
                    f"LLM: {self.config.llm.provider}",
                    f"Chunk size: {self.chunk_size}",
                    f"Stage 1: 1 call → {len(all_pages_data)} page outlines",
                    f"Stage 2: {(len(all_pages_data) + self.chunk_size - 1) // self.chunk_size} chunks",
                    f"Output: {len(pages)} pages, {total_panels} panels",
                ],
            ),
        )

    # -------------------------------------------------------------------------
    # Stage 1: Pagination
    # -------------------------------------------------------------------------
    async def _run_pagination(
        self, client, inputs, story_context, char_context, lang
    ) -> dict[str, Any]:
        user_msg = (
            f"{story_context}"
            f"{char_context}"
            f"\n=== PAGINATION PARAMETERS ===\n"
            f"- Target pages: {inputs.target_pages}\n"
            f"- Panels per page hint: {inputs.target_panels_per_page}\n"
            f"- Approx total panels: {inputs.target_pages * inputs.target_panels_per_page}\n"
            f"- Style: {inputs.style_hint or 'manga'}\n"
            f"- Language: {inputs.language}\n\n"
            f"Divide into EXACTLY {inputs.target_pages} pages. "
            f"panel_count_hint between 3-6, vary for rhythm."
        )

        # v0.5: 用户反馈驱动修订
        if inputs.user_guidance:
            prev_preview = ""
            if inputs.previous_output:
                prev_preview = json.dumps(inputs.previous_output, ensure_ascii=False, indent=2)[:1200]
            user_msg += (
                f"\n\n=== REVISION REQUEST ===\n"
                f"Your previous script structure was:\n---\n{prev_preview}\n---\n\n"
                f"The user reviewed it and provided this feedback:\n"
                f'"{inputs.user_guidance}"\n\n'
                f"Please revise the page outlines incorporating the user's feedback. "
                f"Only change what the user asked you to change; keep everything else from the previous version."
            )

        system_prompt = _PAGINATION_SYSTEM_PROMPT.format(language=lang)

        try:
            data = await asyncio.to_thread(
                client.complete_json,
                system_prompt,
                {"user_message": user_msg},
            )
        except ApiBackendError:
            raise ApiBackendError("Stage 1 (pagination) failed.") from None

        # Validate pagination output structure
        if not isinstance(data, dict):
            raise ApiBackendError(
                f"Pagination returned unexpected type: {type(data).__name__}"
            )
        if "pages" not in data:
            raise ApiBackendError(
                f"Pagination response missing 'pages' key. "
                f"Got keys: {list(data.keys())[:10]}"
            )
        if len(data["pages"]) != inputs.target_pages:
            raise ApiBackendError(
                f"Pagination returned {len(data['pages'])} pages, "
                f"expected {inputs.target_pages}."
            )
        return data

    # -------------------------------------------------------------------------
    # Stage 2: Chunked Panelization
    # -------------------------------------------------------------------------
    async def _run_chunked_panelization(
        self,
        client: OpenAICompatibleLLMClient,
        all_pages_data: list[dict[str, Any]],
        char_context: str,
        language: str,
        chunk_size: int,
        user_guidance: str = "",
        previous_output: dict | None = None,
    ) -> list[ScriptPage]:
        """Process pages in chunks. Chunks are sequential; pages within a chunk
        are handled in ONE LLM call (Option A: max coherence)."""

        # Split into chunks
        chunks: list[list[dict[str, Any]]] = []
        for i in range(0, len(all_pages_data), chunk_size):
            chunks.append(all_pages_data[i : i + chunk_size])

        all_script_pages: list[ScriptPage] = []
        story_so_far = "Beginning of story."

        for chunk_idx, chunk_pages in enumerate(chunks):
            # Preview of next chunk for continuity hint
            next_block_preview = None
            if chunk_idx + 1 < len(chunks):
                next_first = chunks[chunk_idx + 1][0]
                next_block_preview = (
                    f"Next block starts with Page {next_first['page_number']}: "
                    f"{next_first['narrative_arc']} (Key: {next_first['key_moment']})"
                )

            chunk_pages_result = await self._panelize_chunk_together(
                client=client,
                chunk_pages=chunk_pages,
                story_so_far=story_so_far,
                next_block_preview=next_block_preview,
                char_context=char_context,
                language=language,
                user_guidance=user_guidance,
                previous_output=previous_output,
            )

            all_script_pages.extend(chunk_pages_result)

            # Update story_so_far for next chunk
            last_page = chunk_pages[-1]
            story_so_far = (
                f"Up to Page {last_page['page_number']}: "
                f"{last_page['narrative_arc']} — {last_page['key_moment']}"
            )

        # Sort by page_number
        all_script_pages.sort(key=lambda p: p.page_number)
        return all_script_pages

    # -------------------------------------------------------------------------
    # Option A: One LLM call processes entire chunk — max coherence
    # -------------------------------------------------------------------------
    async def _panelize_chunk_together(
        self,
        client: OpenAICompatibleLLMClient,
        chunk_pages: list[dict[str, Any]],
        story_so_far: str,
        next_block_preview: str | None,
        char_context: str,
        language: str,
        user_guidance: str = "",
        previous_output: dict | None = None,
    ) -> list[ScriptPage]:
        """Single LLM call for all pages in a chunk."""
        page_nums = [p["page_number"] for p in chunk_pages]

        # Build page descriptions
        pages_desc = []
        for p in chunk_pages:
            pages_desc.append(
                f"--- Page {p['page_number']} ---\n"
                f"Narrative Arc: {p['narrative_arc']}\n"
                f"Emotional Shift: {p['emotional_shift']}\n"
                f"Key Moment: {p['key_moment']}\n"
                f"Key Panel Type: {p['key_panel_hint']}\n"
                f"Setting: {p['setting']} | Time: {p['time_of_day']} | Atmosphere: {p['atmosphere']}\n"
                f"Characters: {', '.join(p.get('involved_characters', []))}\n"
                f"Panel Count: {p['panel_count_hint']}\n"
                f"Note: {p.get('page_note', '')}\n"
            )

        next_hint = (
            f"\n=== NEXT BLOCK PREVIEW ===\n{next_block_preview}\n"
            if next_block_preview
            else ""
        )

        nl = '\n'
        user_msg = (
            f"=== STORY CONTEXT ===\n"
            f"Story so far (before this block): {story_so_far}\n"
            f"{next_hint}"
            f"\n=== CHARACTER DESIGNS ===\n{char_context}\n"
            f"\n=== PAGES TO EXPAND ({len(chunk_pages)} consecutive pages) ===\n"
            f"{nl.join(pages_desc)}\n"
            f"\nExpand each page into its specified number of panels. "
            f"Maintain narrative flow ACROSS pages in this block. "
            f"Return a JSON object with a 'pages' key containing the array "
            f"of expanded page objects."
        )

        # v0.5: 用户反馈驱动修订
        if user_guidance:
            prev_preview = ""
            if previous_output:
                prev_pages = previous_output.get("pages", [])
                prev_preview = json.dumps(prev_pages, ensure_ascii=False, indent=2)[:1200] if prev_pages else ""
            user_msg += (
                f"\n\n=== REVISION REQUEST ===\n"
                f"Your previous panelization was:\n---\n{prev_preview}\n---\n\n"
                f"The user reviewed it and provided this feedback:\n"
                f'"{user_guidance}"\n\n'
                f"Please revise the panels incorporating the user's feedback. "
                f"Only change what the user asked you to change; keep everything else from the previous version."
            )

        system_prompt = _CHUNK_PANELIZATION_SYSTEM_PROMPT.format(
            chunk_size=len(chunk_pages),
            language=language,
        )

        # Estimate required tokens: each panel ~350 tokens, plus JSON overhead
        total_panels_in_chunk = sum(p.get("panel_count_hint", 4) for p in chunk_pages)
        est_tokens = max(client.config.max_tokens, total_panels_in_chunk * 350)

        try:
            data = await asyncio.to_thread(
                client.complete_json,
                system_prompt,
                {"user_message": user_msg},
                est_tokens,
            )
        except ApiBackendError as e:
            raise ApiBackendError(
                f"Chunk {page_nums} panelization failed: {e}"
            ) from e

        # Parse response — expect {"pages": [...]} dict
        if not isinstance(data, dict):
            raise ApiBackendError(
                f"Chunk {page_nums}: expected JSON object, got {type(data).__name__}"
            )
        if "pages" not in data:
            raise ApiBackendError(
                f"Chunk {page_nums}: response missing 'pages' key. "
                f"Got keys: {list(data.keys())[:10]}"
            )
        pages_list = data["pages"]
        if not isinstance(pages_list, list):
            raise ApiBackendError(
                f"Chunk {page_nums}: 'pages' must be an array, "
                f"got {type(pages_list).__name__}"
            )

        script_pages: list[ScriptPage] = []
        for page_data in pages_list:
            try:
                page_num = page_data["page_number"]
                panels = []
                for i, p in enumerate(page_data.get("panels", []), 1):
                    if "panel_id" not in p:
                        p["panel_id"] = f"page_{page_num:03d}_p{i:02d}"
                    panels.append(PanelTask(**p))

                # Find matching outline for page_note
                outline = next(
                    (o for o in chunk_pages if o["page_number"] == page_num), {}
                )

                script_pages.append(ScriptPage(
                    page_number=page_num,
                    page_note=outline.get("page_note", ""),
                    panels=panels,
                ))
            except (ValidationError, KeyError, TypeError) as e:
                raise ApiBackendError(
                    f"Invalid panel data in chunk: {e}"
                ) from e

        return script_pages

    # -------------------------------------------------------------------------
    # Streaming
    # -------------------------------------------------------------------------
    async def astream(
        self, inputs: ScriptInput, context: AgentContext
    ) -> AsyncIterator[StreamEvent]:
        yield self._make_event(StreamEventType.LOG, "阶段一：全局故事分页...")
        yield self._make_event(StreamEventType.PROGRESS, 0.05)

        await self._sleep_for_demo(0.1)
        yield self._make_event(
            StreamEventType.THINKING,
            f"分析故事结构，规划{inputs.target_pages}页的叙事弧线..."
        )
        yield self._make_event(StreamEventType.PROGRESS, 0.15)

        output = await self.run(inputs, context)

        yield self._make_event(StreamEventType.PROGRESS, 0.9)
        total_panels = sum(len(p.panels) for p in output.pages)
        n_chunks = (len(output.pages) + self.chunk_size - 1) // self.chunk_size
        yield self._make_event(
            StreamEventType.PARTIAL_OUTPUT,
            {
                "title": output.title,
                "pages": len(output.pages),
                "total_panels": total_panels,
                "chunks": n_chunks,
            },
        )
        yield self._make_event(StreamEventType.PROGRESS, 1.0)
        yield self._make_event(StreamEventType.DONE, output.model_dump())


# ============================================================================
# Helper functions
# ============================================================================

def _build_character_context(character_db: CharacterDB | None) -> str:
    """Build a character reference block for the LLM prompt."""
    if not character_db or not character_db.characters:
        return ""

    lines = ["=== CHARACTER DESIGNS ==="]
    for cid, cp in character_db.characters.items():
        vt = cp.visual_traits
        lines.append(
            f"- {cid}: {cp.name}"
            f" | gender: {cp.gender_tag}"
            f" | hair: {vt.hair}"
            f" | eyes: {vt.eyes}"
            f" | body: {vt.body}"
            f" | clothing: {vt.clothing}"
            f" | tags: {cp.core_tags[:120] if cp.core_tags else 'N/A'}"
        )
    lines.append("")
    return "\n".join(lines)


def _build_story_context(story) -> str:
    """Build story context for ScriptAgent LLM prompt from StoryOutput."""
    story_text = getattr(story, "story_text", "") or ""
    author_note = getattr(story, "author_note", "") or ""
    tone = getattr(story, "tone", "") or ""
    setting = getattr(story, "setting", "") or ""
    characters = getattr(story, "characters", []) or []

    if story_text:
        char_lines = "\n".join(
            f"  - {c.get('name', '?')} ({c.get('role', '?')}): "
            f"{c.get('brief_description', '')}"
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
            f"Story Characters (informational — use character designs below "
            f"for visual details):\n{char_lines}\n"
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
