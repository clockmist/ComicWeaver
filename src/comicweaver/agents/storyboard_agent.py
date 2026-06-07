"""Storyboard planning agent — LLM-driven panel cinematography design.

v0.7 (v5): LLM handles character tag filtering based on shot_size. The LLM receives
the full core_tags from CharacterAgent and decides which tags are visible in the
current shot. PromptPack assembly is pure concatenation — no tag parsing logic.

v0.6: narrative-driven prompt with PanelTask direct consumption, visual_traits
awareness, shot-pose-expression validation, dynamic negative prompts.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from comicweaver.api import ApiBackendError, OpenAICompatibleLLMClient
from comicweaver.core import (
    AgentContext,
    AgentOutputMeta,
    BaseAgent,
    BubbleHint,
    CameraAngle,
    Dialogue,
    NarrativeStructure,
    PageLayout,
    PanelPlan,
    PanelShape,
    PreviousPanelContext,
    PromptPack,
    Scene,
    ShotSize,
    StoryboardInput,
    StoryboardOutput,
    StreamEvent,
    StreamEventType,
)

# ============================================================================
# Layout hints
# ============================================================================


def _select_layout_hint(panel_count: int, climax: bool, action_heavy: bool) -> str:
    if climax:
        return "climax"
    if action_heavy:
        return "action"
    if panel_count >= 5:
        return "dialogue"
    if panel_count <= 1:
        return "climax"
    return "standard"


# ============================================================================
# Hardcoded prompt constants (v0.7 / v5)
# ============================================================================

POSITIVE_PREFIX = (
    "masterpiece, high score, great score, absurdres, safe"
)

POSITIVE_SUFFIX = (
    "monochrome, greyscale, black and white manga style, "
    "screentone, ink drawing, clean lineart, high contrast"
)

NEGATIVE_PROMPT = (
    "lowres, worst quality, bad quality, displeasing, low resolution, "
    "bad anatomy, bad hands, bad fingers, extra digits, fewer digits, "
    "cropped, signature, watermark, username, blurry, text, error, "
    "color, colored, multicolored, gradient, rainbow, "
    "3d, realistic, photo, photograph, photorealistic, "
    "nsfw, explicit"
)

NEGATIVE_BY_SHOT: dict[str, str] = {
    "extreme_close": "bad face, bad eyes, asymmetrical eyes, mutated face, wide shot, full body",
    "close": "bad face, bad eyes, asymmetrical eyes, mutated face, wide shot, full body",
    "extreme_long": "tiny figure, unclear silhouette, cluttered composition, close-up, portrait, face focus, looking at viewer",
    "long": "tiny figure, unclear silhouette, cluttered composition, close-up, portrait, face focus, looking at viewer",
    "full": "close-up, portrait, face focus, looking at viewer, upper body",
    "medium": "close-up, portrait, face focus, wide shot, panoramic",
}


# ============================================================================
# LLM System Prompt (v0.7 / v5: LLM responsible for tag filtering)
# ============================================================================

_STORYBOARD_SYSTEM_PROMPT = """You are a prompt engineer for Animagine-XL-4.0 black-and-white manga.

For each panel, translate narrative description into visual tags. Your job is to describe WHAT FILLS THE FRAME — not just the character, but the entire visual world of this panel.

Output JSON:
{
  "filtered_char_tags": "character tags visible in this shot, or empty string if no character",
  "scene_description": "6-12 visual tags describing THE ENTIRE SCENE. THIS IS THE MOST IMPORTANT FIELD. See rules below.",
  "shot_size": "extreme_long|long|full|medium|close|extreme_close",
  "camera_angle": "eye_level|high_angle|low_angle|dutch_angle"
}

=== SCENE_DESCRIPTION: ENVIRONMENT FIRST (CRITICAL — this makes or breaks the image) ===
scene_description MUST describe the COMPLETE visual scene, not just the character. The character is only ONE element in a larger visual composition.

TAG ORDER (strict — earlier = more weight in image generation):
  1. Environment/setting tags FIRST (architecture, terrain, props, objects, weather, lighting, atmosphere)
  2. Character pose/action tags SECOND
  3. Character expression/face tags LAST

MINIMUM ENVIRONMENT TAGS by shot type:
  - extreme_long / long: at least 5-8 environment tags, 0-2 character tags (character is small in frame)
  - full: at least 3-5 environment tags, 2-3 character tags
  - medium: at least 2-3 environment/background tags, 2-3 character tags
  - close: at least 1-2 environment/background tags, 3-4 character face/expression tags
  - extreme_close: 0-1 environment tags, 4-5 character detail tags

CONCRETE VISUAL TAGS ONLY:
  - YES: "broken windows", "steam pipes", "graffiti wall", "flickering lamp", "puddled floor", "rusty railing"
  - NO: "lonely atmosphere", "tense mood", "sad feeling" (these are abstract, not visual)
  - NO: "morning light", "sunlight", "natural light", "soft light", "harsh light"
  - Instead use: "light shafts", "backlit", "lantern glow", "neon reflection", "candlelit", "shadow patterns"

NO quality/style tags (added automatically): masterpiece, monochrome, screentone, etc.

=== SHOT DISTRIBUTION (CRITICAL — think like a cinematographer) ===
Real comics use VARIED shot sizes. Do NOT default to medium/close for every panel.
- extreme_long / long: ~30-35% of panels — establishing shots, environment, action scenes, character small in context
- full: ~15-20% of panels — full-body character with environment visible
- medium: ~25-30% of panels — upper body with some background
- close / extreme_close: ~10-15% of panels — ONLY for emotional climax, key reaction, or critical detail
- VARY shot sizes across consecutive panels — never use the same shot twice in a row for character panels.
- First panel of each page should usually be an establishing/location shot (extreme_long or long) to orient the reader.
- camera_angle should vary: use high_angle for vulnerability, low_angle for power/threat, dutch_angle for disorientation/unease.

=== CHARACTER DETECTION ===
CRITICAL: Check if the panel has a character.

- If character_id is EMPTY or character_action is EMPTY:
  - This is a BACKGROUND/ESTABLISHING panel with NO character.
  - filtered_char_tags MUST be "" (empty string).
  - scene_description should be 100% environment/atmosphere/architecture tags.
  - shot_size must be extreme_long or long.

- If character_id exists and character_action exists:
  - This panel HAS a character.
  - Filter character tags based on shot_size (see below).
  - Choose shot_size based on narrative purpose (NOT just defaulting to medium):
    * Action/movement in a big space → long or full to show the action in context
    * Dialogue/conversation in a room → medium or full, show the space
    * Emotional reaction in context → medium (show face + surroundings)
    * Pure emotional climax → close or extreme_close (rare!)
    * Character introduction → full (show the whole person in their world)

=== TAG FILTERING (character panels only) ===
You receive a COMPLETE tag list from the character designer. REMOVE only tags not visible in this shot. NEVER add new tags.

- extreme_long: gender + solo + 1 silhouette tag
- long: gender + solo + 2-3 silhouette tags
- full: all tags
- medium: head + upper body clothing
- close: head + eyes + face accessories
- extreme_close: eyes + mouth + eyebrows

NEVER invent tags. Use ONLY provided tags.

=== ACTION TAGS ===
- At most 2 action tags per panel.
- Pick the MOST IMPORTANT pose for this moment.
- BAD: "jumping, aiming, firing" → GOOD: "jumping, aiming pistol"

=== EXAMPLES (study the scene_description carefully — ENVIRONMENT ALWAYS FIRST) ===

Example 1 — NO CHARACTER (establishing shot, 100% environment):
Input: character_id="", character_action="", narrative_purpose="废铁城全景：灰黄烟尘笼罩街道，机械塔楼高耸入云，废弃车辆和锈蚀管道堆积成山，远处工厂烟囱喷着黑烟", emotion="tension", emotion_intensity=0.6, location="废铁城工业区，机械塔楼下", time_of_day="evening"
Output: {"filtered_char_tags":"","scene_description":"industrial ruins,tall mechanical towers,smoke and dust,smashed vehicles,rusty pipes piled high,factory chimneys,dark sky,harsh shadows,dim orange glow","shot_size":"extreme_long","camera_angle":"high_angle"}

Example 2 — CHARACTER LONG SHOT (environment dominates, character is small):
Input: character_id="char_000", character_action="running through debris-filled street, dodging falling pipes", narrative_purpose="雨夜后巷，积水倒映霓虹招牌红光，穿风衣的模糊人影在巷口闪过，林晓在消防梯阴影下手按刀柄", emotion="tension", emotion_intensity=0.7, location="雨夜后巷，霓虹招牌，消防梯", time_of_day="night", character_tags="1boy,solo,young_adult,short hair,spiky hair,black hair,sharp eyes,grey eyes,jacket,torn clothing,athletic"
Output: {"filtered_char_tags":"1boy,solo,short hair,spiky hair,black hair","scene_description":"rainy alley,puddle reflections,neon sign glow,fire escape shadows,steam from manhole,brick walls,running figure in distance,wet pavement","shot_size":"long","camera_angle":"eye_level"}

Example 3 — CHARACTER FULL SHOT (balanced environment + full body):
Input: character_id="char_000", character_action="kneeling beside collapsed machine, digging through rubble with bare hands", narrative_purpose="废弃工厂内部，阳光从破碎屋顶斜射形成光柱，粉尘悬浮，林晓跪在倒塌机器旁双手扒开瓦砾，露出苍白的手", emotion="fear", emotion_intensity=0.6, location="废弃机械工厂内部", time_of_day="afternoon", character_tags="1boy,solo,young_adult,short hair,spiky hair,black hair,sharp eyes,grey eyes,jacket,torn clothing,athletic"
Output: {"filtered_char_tags":"1boy,solo,young_adult,short hair,spiky hair,black hair,sharp eyes,grey eyes,jacket,torn clothing,athletic","scene_description":"abandoned factory interior,shattered roof,light shafts through ceiling,dust motes floating,collapsed machinery,rubble and debris,kneeling,digging through rubble,pale hand visible in debris","shot_size":"full","camera_angle":"high_angle"}

Example 4 — CHARACTER MEDIUM SHOT (visible background):
Input: character_id="char_000", character_action="leaning out from behind wrecked car, observing", narrative_purpose="废车残骸后，林晓探出半身，一只手紧握钢管，远处黑影在烟尘中逼近", emotion="tension", emotion_intensity=0.6, location="废铁城街道，废车残骸", time_of_day="evening", character_tags="1boy,solo,young_adult,short hair,spiky hair,black hair,sharp eyes,grey eyes,jacket,torn clothing,athletic"
Output: {"filtered_char_tags":"1boy,solo,young_adult,short hair,spiky hair,black hair,sharp eyes,grey eyes,jacket,torn clothing,athletic","scene_description":"wrecked car in foreground,smoke and dust,approaching shadowy figures,distant industrial structures,dim streetlight,leaning forward,gripping steel pipe,tense expression","shot_size":"medium","camera_angle":"eye_level"}

Example 5 — CHARACTER CLOSE-UP (emotional climax, rare):
Input: character_id="char_001", character_action="peering through rifle scope, finger on trigger, sweat dripping", narrative_purpose="极近距离：瞄准镜后的眼睛，汗水沿眉骨滑落，准星中对准远处目标", emotion="tension", emotion_intensity=0.9, location="废墟楼顶边缘", time_of_day="evening", character_tags="1girl,solo,young_adult,long hair,straight hair,black hair,large eyes,green eyes,tactical vest,fitted shirt,cargo pants,slim"
Output: {"filtered_char_tags":"1girl,solo,long hair,straight hair,black hair,large eyes,green eyes","scene_description":"scope lens reflection,crosshair overlay,rooftop edge visible,sweat on brow,focused eyes,tense finger","shot_size":"extreme_close","camera_angle":"eye_level"}

Output ONLY valid JSON. No markdown, no extra text."""


# ============================================================================
# Tag helpers (minimal — no semantic parsing)
# ============================================================================

def _deduplicate_tags(tags_str: str) -> str:
    """Remove duplicate tags while preserving order."""
    if not tags_str:
        return ""
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags_str.split(","):
        t = tag.strip()
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return ", ".join(result)


# ============================================================================
# PromptPack builder (v0.7 / v5: pure concatenation, no logic)
# ============================================================================

def _build_prompt_from_llm(llm_output: dict, char_gender_tag: str) -> PromptPack:
    """v2.0: Shot-aware prompt assembly — all decisions made by the LLM.

    Tag order varies by shot_size to weight the image model correctly:
    - extreme_long/long/full/medium:  environment >> character (scene_description first)
    - close/extreme_close:            character > environment (character tags first)

    v1.0 fix: shot_size and camera_angle composition tags injected into prompt.
    """
    filtered_char_tags = (llm_output.get("filtered_char_tags") or "").strip()
    scene_description = (llm_output.get("scene_description") or "").strip()
    shot = str(llm_output.get("shot_size", "medium"))
    angle = str(llm_output.get("camera_angle", "eye_level"))

    # Resolve shot/angle to Animagine-compatible composition tags
    shot_tag = _SHOT_SIZE_TAGS.get(shot, "")
    angle_tag = _CAMERA_ANGLE_TAGS.get(angle, "")

    parts = [POSITIVE_PREFIX]

    # v2.0: Shot-aware tag ordering — environment-first for non-close-up shots.
    # In Animagine XL, earlier tags have more weight, so this directly controls
    # whether the model prioritizes the environment or the character.
    if shot in ("extreme_long", "long", "full", "medium"):
        # Environment-first: scene_description (environment tags) before character
        if scene_description:
            parts.append(scene_description)
        if filtered_char_tags:
            parts.append(filtered_char_tags)
    else:
        # Character-first: for close/extreme_close, character IS the focus
        if filtered_char_tags:
            parts.append(filtered_char_tags)
        if scene_description:
            parts.append(scene_description)

    # Inject composition tags BEFORE the style suffix
    if shot_tag:
        parts.append(shot_tag)
    if angle_tag:
        parts.append(angle_tag)

    parts.append(POSITIVE_SUFFIX)

    positive = ", ".join(p for p in parts if p)
    positive = _deduplicate_tags(positive)

    # Negative prompt — dynamic per shot
    # v1.0: extended NEGATIVE_BY_SHOT blocks wrong framing for each shot type
    neg = NEGATIVE_PROMPT
    extra_neg = NEGATIVE_BY_SHOT.get(shot, "")
    if extra_neg:
        neg = f"{neg}, {extra_neg}"

    return PromptPack(
        positive_prompt=positive,
        negative_prompt=neg,
        style_tags=["monochrome", "greyscale", "black and white manga", "screentone"],
        composition_tags=[shot, angle],
        quality_tags=["masterpiece", "high score", "great score", "absurdres", "clean lineart"],
    )


# ============================================================================
# Animagine XL 4.0 shot-size / camera-angle mapping
# ============================================================================

_SHOT_SIZE_TAGS: dict[str, str] = {
    "extreme_long": "panoramic view",
    "long": "long shot",
    "full": "full shot",
    "medium": "medium shot",
    "close": "close-up",
    "extreme_close": "extreme close-up",
}

_CAMERA_ANGLE_TAGS: dict[str, str] = {
    "eye_level": "",
    "high_angle": "from above",
    "low_angle": "from below",
    "dutch_angle": "dutch angle",
}


def _map_shot_size(shot_value: str) -> str:
    return _SHOT_SIZE_TAGS.get(shot_value, shot_value.replace("_", " "))


# ============================================================================
# StoryboardAgent
# ============================================================================


class StoryboardAgent(BaseAgent[StoryboardInput, StoryboardOutput]):
    """Storyboard planning agent — LLM-driven cinematography design."""

    name = "storyboard_agent"
    version = "0.7.0-llm"
    rubric_id = "rubric_storyboard_v1"

    async def run(
        self, inputs: StoryboardInput, context: AgentContext
    ) -> StoryboardOutput:
        if self.config.llm.is_available:
            try:
                return await self._run_with_llm(inputs, context)
            except ApiBackendError:
                if not self.config.runtime.fallback_to_local:
                    raise
        return await self._run_local(inputs, context)

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------

    async def _run_with_llm(
        self, inputs: StoryboardInput, context: AgentContext
    ) -> StoryboardOutput:
        """v0.7: LLM-driven panel design — LLM handles tag filtering + scene description.

        When inputs.pages is available, directly consumes PanelTask.
        Falls back to inputs.scenes for legacy compatibility.
        """
        await self._sleep_for_demo(0.3)

        # Build character reference info — only core_tags and gender_tag needed
        char_refs: dict[str, dict] = {}
        if inputs.character_db:
            for cid, cp in inputs.character_db.characters.items():
                char_refs[cid] = {
                    "char_id": cid,
                    "name": cp.name,
                    "core_tags": cp.core_tags or cp.appearance_prompt or "",
                    "gender_tag": cp.gender_tag or "1girl",
                }

        # --- Choose data source: pages (PanelTask) or scenes (legacy) ---
        if inputs.pages:
            all_panels: list[tuple[dict, int]] = []
            for page in inputs.pages:
                for panel_task in page.panels:
                    all_panels.append((panel_task.model_dump(), len(all_panels)))

            total = len(all_panels)
            prev_narrative = ""
            panel_designs: list[dict] = []

            if not self.config.llm.is_available:
                raise ApiBackendError("LLM is not available — cannot design storyboard panels")

            for idx, (pt, global_idx) in enumerate(all_panels):
                char_for_panel = pt.get("character_id", "")

                design = await self._design_single_panel_via_llm(
                    panel_task=pt,
                    char_name=char_for_panel,
                    char_refs=char_refs,
                    prev_narrative=prev_narrative,
                    global_idx=global_idx,
                    total_panels=total,
                )
                panel_designs.append(design)
                prev_narrative = pt.get("narrative_purpose", "")

        else:
            # Legacy Scene bridge
            panel_specs: list[tuple[Scene, int, str]] = []
            for scene in inputs.scenes:
                chars = scene.characters_present
                for i in range(max(1, scene.panel_hint)):
                    char_for_panel = chars[i % len(chars)] if chars else ""
                    panel_specs.append((scene, i, char_for_panel))

            total = len(panel_specs)
            prev_narrative = ""
            panel_designs = []

            if not self.config.llm.is_available:
                raise ApiBackendError("LLM is not available — cannot design storyboard panels")

            for idx, (scene, panel_idx, char_for_panel) in enumerate(panel_specs):
                legacy_pt = {
                    "narrative_purpose": getattr(scene, "visual_hook", "") or scene.atmosphere,
                    "emotion": scene.atmosphere,
                    "emotion_intensity": scene.emotion_intensity,
                    "character_action": scene.actions[0].description if scene.actions else "",
                    "character_id": char_for_panel,
                    "location": scene.location,
                    "time_of_day": scene.time_of_day,
                    "dialogue_text": "",
                    "dialogue_tone": "",
                    "is_thought": False,
                    "panel_id": scene.scene_id,
                }

                design = await self._design_single_panel_via_llm(
                    panel_task=legacy_pt,
                    char_name=char_for_panel,
                    char_refs=char_refs,
                    prev_narrative=prev_narrative,
                    global_idx=idx,
                    total_panels=total,
                )
                panel_designs.append(design)
                prev_narrative = legacy_pt.get("narrative_purpose", "")

        # --- Group into pages (common to both paths) ---
        per_page = max(1, inputs.target_panels_per_page)
        page_list: list[PageLayout] = []
        page_idx = 0

        if inputs.pages:
            source_items = all_panels

            def _pt_val(item, key, default=""):
                return item[0].get(key, default)
            def _pt_emotion(item):
                return float(item[0].get("emotion_intensity", 0.5))
            def _pt_char(item):
                return item[0].get("character_id", "")
            def _pt_dialogues(item):
                pt = item[0]
                text = pt.get("dialogue_text", "")
                if text:
                    return [Dialogue(
                        speaker=pt.get("character_id", ""),
                        text=text,
                        tone=pt.get("dialogue_tone", "neutral"),
                        is_thought=pt.get("is_thought", False),
                    )]
                return []
            def _pt_action_heavy(_chunk_designs):
                return any(len(item.get("character_action", "")) > 30 for item, _ in chunk_source_items)
        else:
            source_items = panel_specs

            def _pt_val(item, key, default=""):
                s = item[0]
                return getattr(s, key, default)
            def _pt_emotion(item):
                return item[0].emotion_intensity
            def _pt_char(item):
                return item[2]
            def _pt_dialogues(item):
                return item[0].dialogues
            def _pt_action_heavy(_chunk_designs):
                s_list = [s for s, _, _ in chunk_source_items]
                return any(s.actions and len(s.actions) >= 2 for s in s_list)

        for start in range(0, total, per_page):
            chunk_source_items = source_items[start:start + per_page]
            chunk_designs = panel_designs[start:start + per_page]
            climax = any(_pt_emotion(si) >= 0.85 for si in chunk_source_items)
            action_heavy = _pt_action_heavy(chunk_designs)

            layout_hint = _select_layout_hint(len(chunk_source_items), climax, action_heavy)
            page_id = f"page_{page_idx + 1:03d}"
            panels: list[PanelPlan] = []

            for i, si in enumerate(chunk_source_items):
                design = chunk_designs[i]
                char_for_panel = _pt_char(si)
                location = _pt_val(si, "location", "")
                atmosphere = _pt_val(si, "atmosphere", "")
                time_of_day = _pt_val(si, "time_of_day", "")
                scene_id = _pt_val(si, "panel_id", _pt_val(si, "scene_id", ""))

                # Parse LLM output fields
                try:
                    shot = ShotSize(design.get("shot_size", "medium"))
                except ValueError:
                    shot = ShotSize.MEDIUM
                try:
                    angle = CameraAngle(design.get("camera_angle", "eye_level"))
                except ValueError:
                    angle = CameraAngle.EYE_LEVEL

                scene_description = design.get("scene_description", "")
                emotion = float(design.get("emotion_intensity", _pt_emotion(si)))
                mood_val = design.get("mood", atmosphere)

                # Character info
                char_core_tags = ""
                char_gender_tag = "1girl"
                if char_for_panel and char_for_panel in char_refs:
                    ref = char_refs[char_for_panel]
                    char_core_tags = ref.get("core_tags", "")
                    char_gender_tag = ref.get("gender_tag", "1girl")

                single_char_list = [char_for_panel] if char_for_panel else []

                # v0.7: PromptPack built from LLM output (PREFIX + filtered + scene + SUFFIX)
                prompt_pack = _build_prompt_from_llm(design, char_gender_tag)

                # Dialogues
                dialogues_for_panel = _pt_dialogues(si)
                if inputs.pages:
                    panel_dialogues = list(dialogues_for_panel)
                else:
                    panel_hint = getattr(si[0], "panel_hint", 1)
                    panel_dialogues = _distribute_dialogues(dialogues_for_panel, i, panel_hint)

                bubble_hints = [
                    BubbleHint(
                        dialogue_index=di,
                        suggested_position="top_right" if di % 2 == 0 else "bottom_left",
                        bubble_type="speech" if not d.is_thought else "thought",
                    )
                    for di, d in enumerate(panel_dialogues)
                ]

                pctx = _build_panel_context(
                    idx=i, chunk_idx=start + i, panel_designs=panel_designs, start=start
                )

                panels.append(
                    PanelPlan(
                        panel_id=design.get("panel_id", f"{page_id}_p{i + 1:02d}"),
                        page_id=page_id,
                        order_in_page=i + 1,
                        source_scene_id=scene_id,
                        source_dialogue_indices=list(range(len(panel_dialogues))),
                        shape=PanelShape.RECTANGLE,
                        shot_size=shot,
                        camera_angle=angle,
                        characters_in_panel=single_char_list,
                        primary_action=scene_description,
                        setting=location,
                        mood=mood_val,
                        emotion_intensity=emotion,
                        prompt_pack=prompt_pack,
                        dialogues_in_panel=panel_dialogues,
                        speech_bubble_hints=bubble_hints,
                        pose_hint=scene_description,
                        expression="",
                        scene_lighting="",
                        weather="",
                        time_of_day=time_of_day,
                        previous_panel_context=pctx,
                    )
                )

            avg_emotion = sum(_pt_emotion(si) for si in chunk_source_items) / len(chunk_source_items)
            page_list.append(
                PageLayout(
                    page_id=page_id,
                    page_number=page_idx + 1,
                    layout_template=layout_hint,
                    layout_hint=layout_hint,
                    panels=panels,
                    page_emotion_avg=avg_emotion,
                    is_climax_page=climax,
                )
            )
            page_idx += 1

        source_label = "pages(v1.0)" if inputs.pages else "scenes(legacy)"
        return StoryboardOutput(
            pages=page_list,
            total_panels=sum(len(p.panels) for p in page_list),
            meta=AgentOutputMeta(
                agent=self.name,
                version=self.version,
                inputs_hash=self._inputs_hash(inputs),
                self_check_notes=[
                    f"LLM panel design [{source_label}]: {len(page_list)} pages, {sum(len(p.panels) for p in page_list)} panels",
                ],
            ),
        )

    # ------------------------------------------------------------------
    # Local fallback path
    # ------------------------------------------------------------------

    async def _run_local(
        self, inputs: StoryboardInput, context: AgentContext
    ) -> StoryboardOutput:
        """v0.7: Local fallback — deterministic panel design without LLM."""
        await self._sleep_for_demo(0.2)

        char_refs: dict[str, dict] = {}
        if inputs.character_db:
            for cid, cp in inputs.character_db.characters.items():
                char_refs[cid] = {
                    "char_id": cid,
                    "name": cp.name,
                    "core_tags": cp.core_tags or cp.appearance_prompt or "",
                    "gender_tag": cp.gender_tag or "1girl",
                }

        if inputs.pages:
            panel_specs: list[tuple[dict, int]] = []
            for page in inputs.pages:
                for pt in page.panels:
                    panel_specs.append((pt.model_dump(), len(panel_specs)))
            total = len(panel_specs)
        elif inputs.scenes:
            old_specs: list[tuple[Scene, int, str]] = []
            for scene in inputs.scenes:
                chars = scene.characters_present
                for i in range(max(1, scene.panel_hint)):
                    char_for_panel = chars[i % len(chars)] if chars else ""
                    old_specs.append((scene, i, char_for_panel))
            total = len(old_specs)
        else:
            total = 0

        if total == 0:
            return StoryboardOutput(
                pages=[], total_panels=0,
                meta=AgentOutputMeta(agent=self.name, version=self.version,
                    self_check_notes=["Local fallback — no panels to design"]),
            )

        per_page = max(1, inputs.target_panels_per_page)
        page_list: list[PageLayout] = []
        page_idx = 0
        panel_counter = 0

        for start in range(0, total, per_page):
            if inputs.pages:
                chunk_specs = panel_specs[start:start + per_page]
            else:
                chunk_specs = old_specs[start:start + per_page]
            chunk_len = len(chunk_specs)

            page_id = f"page_{page_idx + 1:03d}"
            panels: list[PanelPlan] = []

            for i in range(chunk_len):
                panel_id = f"{page_id}_p{i + 1:02d}"

                # v1.0: Context-aware shot selection for local fallback
                if inputs.pages:
                    pt = chunk_specs[i][0]
                    char_for_panel = pt.get("character_id", "")
                    emotion = float(pt.get("emotion_intensity", 0.5))
                else:
                    scene, panel_idx, char_for_panel = chunk_specs[i]
                    emotion = scene.emotion_intensity

                # No character → establishing shot (extreme_long or long)
                if not char_for_panel:
                    shot = ShotSize.EXTREME_LONG if panel_counter % 3 == 0 else ShotSize.LONG
                # High emotion → close-up
                elif emotion >= 0.85:
                    shot = ShotSize.EXTREME_CLOSE if emotion >= 0.95 else ShotSize.CLOSE
                # Normal character panel → varied distribution
                else:
                    shots = [ShotSize.LONG, ShotSize.FULL, ShotSize.MEDIUM, ShotSize.FULL, ShotSize.LONG, ShotSize.MEDIUM]
                    shot = shots[panel_counter % len(shots)]
                angle = CameraAngle.EYE_LEVEL

                if inputs.pages:
                    pt = chunk_specs[i][0]
                    location = pt.get("location", "")
                    atmosphere = pt.get("atmosphere", "")
                    time_of_day = pt.get("time_of_day", "")
                    scene_description = pt.get("character_action", "") or pt.get("narrative_purpose", "")
                    scene_id = pt.get("panel_id", "")
                    dialogues_for_panel = []
                    if pt.get("dialogue_text"):
                        dialogues_for_panel = [Dialogue(
                            speaker=char_for_panel,
                            text=pt.get("dialogue_text", ""),
                            tone=pt.get("dialogue_tone", "neutral"),
                            is_thought=pt.get("is_thought", False),
                        )]
                else:
                    scene, panel_idx, char_for_panel_legacy = chunk_specs[i]
                    location = scene.location
                    atmosphere = scene.atmosphere
                    time_of_day = scene.time_of_day
                    scene_description = scene.actions[0].description if scene.actions else scene.atmosphere
                    scene_id = scene.scene_id
                    dialogues_for_panel = _distribute_dialogues(
                        scene.dialogues, panel_idx, scene.panel_hint
                    )

                char_gender_tag = "1girl"
                if char_for_panel and char_for_panel in char_refs:
                    char_gender_tag = char_refs[char_for_panel].get("gender_tag", "1girl")

                # Build a fake LLM output so _build_prompt_from_llm works.
                # v2.0: environment-first — shot_tag is injected separately by
                # _build_prompt_from_llm, so scene_description here focuses on
                # location + atmosphere + character action.
                fake_llm = {
                    "filtered_char_tags": char_refs.get(char_for_panel, {}).get("core_tags", ""),
                    "scene_description": f"{location}, {atmosphere}, {time_of_day}, {scene_description}",
                    "shot_size": shot.value,
                    "camera_angle": angle.value,
                }
                prompt_pack = _build_prompt_from_llm(fake_llm, char_gender_tag)

                prev_ctx = _build_panel_context_local(i, panel_counter, panels)

                panels.append(
                    PanelPlan(
                        panel_id=panel_id,
                        page_id=page_id,
                        order_in_page=i + 1,
                        source_scene_id=scene_id,
                        shape=PanelShape.RECTANGLE,
                        shot_size=shot,
                        camera_angle=angle,
                        characters_in_panel=[char_for_panel] if char_for_panel else [],
                        primary_action=scene_description,
                        setting=location,
                        mood=atmosphere,
                        emotion_intensity=emotion,
                        prompt_pack=prompt_pack,
                        dialogues_in_panel=list(dialogues_for_panel),
                        pose_hint=scene_description,
                        expression="",
                        scene_lighting="",
                        weather="",
                        time_of_day=time_of_day,
                        previous_panel_context=prev_ctx,
                    )
                )
                panel_counter += 1

            if inputs.pages:
                avg_emotion = sum(float(si[0].get("emotion_intensity", 0.5)) for si in chunk_specs) / chunk_len
                climax = any(float(si[0].get("emotion_intensity", 0.5)) >= 0.85 for si in chunk_specs)
            else:
                avg_emotion = sum(s.emotion_intensity for s, _, _ in chunk_specs) / chunk_len
                climax = any(s.emotion_intensity >= 0.85 for s, _, _ in chunk_specs)

            page_list.append(
                PageLayout(
                    page_id=page_id,
                    page_number=page_idx + 1,
                    layout_template="grid_2x2",
                    layout_hint=_select_layout_hint(chunk_len, climax, False),
                    panels=panels,
                    page_emotion_avg=avg_emotion,
                    is_climax_page=climax,
                )
            )
            page_idx += 1

        source_label = "pages(v1.0)" if inputs.pages else "scenes(legacy)"
        return StoryboardOutput(
            pages=page_list,
            total_panels=sum(len(p.panels) for p in page_list),
            meta=AgentOutputMeta(
                agent=self.name,
                version=self.version,
                self_check_notes=[f"Local fallback [{source_label}] — deterministic panel design"],
            ),
        )

    # ------------------------------------------------------------------
    # Single-panel LLM design (v0.7 / v5)
    # ------------------------------------------------------------------

    async def _design_single_panel_via_llm(
        self,
        panel_task: dict,
        char_name: str = "",
        char_refs: dict | None = None,
        prev_narrative: str = "",
        global_idx: int = 0,
        total_panels: int = 1,
    ) -> dict:
        """v0.7: LLM receives full char_tags + narrative description, returns
        filtered_char_tags + scene_description + shot_size + camera_angle.
        """
        client = OpenAICompatibleLLMClient(self.config.llm)
        char_refs = char_refs or {}

        char_info = char_refs.get(char_name, {})
        full_char_tags = char_info.get("core_tags", "")
        char_gender = char_info.get("gender_tag", "1girl")


        user_payload = {
    "character_id": panel_task.get('character_id', ''),  # 空字符串表示无角色
    "character_action": panel_task.get('character_action', ''),
    "narrative_purpose": panel_task.get('narrative_purpose', ''),
    "emotion": panel_task.get('emotion', ''),
    "emotion_intensity": panel_task.get('emotion_intensity', 0.5),
    "location": panel_task.get('location', ''),
    "time_of_day": panel_task.get('time_of_day', ''),
    "character_tags": full_char_tags if char_name else "",  # 无角色时传空
    "panel_index": f"{global_idx + 1}/{total_panels}",
}

        data = await asyncio.to_thread(
            client.complete_json,
            _STORYBOARD_SYSTEM_PROMPT,
            user_payload,
        )
        if not isinstance(data, dict):
            raise ApiBackendError(
                f"LLM panel design returned unexpected format: {type(data).__name__}"
            )

        # Enrich with panel_task metadata for downstream grouping
        data["panel_id"] = panel_task.get("panel_id", f"panel_{global_idx:03d}")
        data["setting"] = panel_task.get("location", "")
        data["mood"] = panel_task.get("emotion", "")
        data["emotion_intensity"] = panel_task.get("emotion_intensity", 0.5)
        return data

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def astream(
        self, inputs: StoryboardInput, context: AgentContext
    ) -> AsyncIterator[StreamEvent]:
        yield self._make_event(StreamEventType.LOG, "Starting storyboard planning...")
        yield self._make_event(StreamEventType.PROGRESS, 0.1)

        await self._sleep_for_demo(0.15)
        yield self._make_event(StreamEventType.THINKING, "Assigning panels per scene...")
        yield self._make_event(StreamEventType.PROGRESS, 0.3)

        await self._sleep_for_demo(0.15)
        yield self._make_event(StreamEventType.THINKING, "Selecting layout templates and shot sizes...")
        yield self._make_event(StreamEventType.PROGRESS, 0.6)

        await self._sleep_for_demo(0.15)
        yield self._make_event(StreamEventType.THINKING, "Assembling prompts with mood and weather...")

        output = await self.run(inputs, context)
        yield self._make_event(
            StreamEventType.PARTIAL_OUTPUT,
            {"pages": len(output.pages), "panels": output.total_panels},
        )
        yield self._make_event(StreamEventType.PROGRESS, 1.0)
        yield self._make_event(StreamEventType.DONE, output.model_dump())


# ============================================================================
# Helper functions (shared by LLM path and local fallback)
# ============================================================================

def _distribute_dialogues(
    dialogues: list[Dialogue], panel_idx: int, total_panels: int
) -> list[Dialogue]:
    """Distribute dialogues across panels within a scene."""
    if total_panels <= 1:
        return list(dialogues)
    if not dialogues:
        return []

    per_panel = max(1, len(dialogues) // total_panels)
    start = panel_idx * per_panel
    end = start + per_panel if panel_idx < total_panels - 1 else len(dialogues)
    return list(dialogues[start:end])


def _build_panel_context_local(
    idx: int, global_idx: int, panels: list[PanelPlan]
) -> PreviousPanelContext | None:
    """Build PreviousPanelContext from within-page panels (local fallback)."""
    if idx == 0:
        return None
    prev = panels[idx - 1] if idx > 0 else None
    if not prev:
        return None
    return PreviousPanelContext(
        panel_id=prev.panel_id,
        shot_size=prev.shot_size.value,
        camera_angle=prev.camera_angle.value,
        pose_hint=prev.pose_hint,
        expression=prev.expression,
        setting_summary=prev.setting,
        emotion=prev.mood,
        emotion_intensity=prev.emotion_intensity,
    )


def _build_panel_context(
    idx: int, chunk_idx: int, panel_designs: list[dict], start: int
) -> PreviousPanelContext | None:
    """Build PreviousPanelContext from the preceding panel's design data."""
    if idx == 0:
        if chunk_idx > 0 and chunk_idx - 1 < len(panel_designs):
            prev = panel_designs[chunk_idx - 1]
            return PreviousPanelContext(
                panel_id=prev.get("panel_id", ""),
                shot_size=str(prev.get("shot_size", "medium")),
                camera_angle=str(prev.get("camera_angle", "eye_level")),
                pose_hint=str(prev.get("pose_hint", "")),
                expression=str(prev.get("expression", "")),
                setting_summary=str(prev.get("setting", "")),
                emotion=str(prev.get("mood", "")),
                emotion_intensity=float(prev.get("emotion_intensity", 0.5)),
            )
        return None

    prev = panel_designs[idx - 1] if idx > 0 else {}
    if not prev:
        return None
    return PreviousPanelContext(
        panel_id=prev.get("panel_id", ""),
        shot_size=str(prev.get("shot_size", "medium")),
        camera_angle=str(prev.get("camera_angle", "eye_level")),
        pose_hint=str(prev.get("pose_hint", "")),
        expression=str(prev.get("expression", "")),
        setting_summary=str(prev.get("setting", "")),
        emotion=str(prev.get("mood", "")),
        emotion_intensity=float(prev.get("emotion_intensity", 0.5)),
    )
