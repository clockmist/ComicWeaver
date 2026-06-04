"""Storyboard planning agent — LLM-driven panel cinematography design.

v0.5: mood/time_of_day/weather enter prompt, previous-panel context, all actions visible,
dialogues distributed across panels.
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
# Mood → visual tags mapping (v0.5: mood now enters the prompt)
# ============================================================================

_MOOD_VISUAL_TAGS: dict[str, str] = {
    "tense": "tense atmosphere, dramatic shadows",
    "dramatic": "dramatic lighting, intense contrast",
    "mysterious": "mysterious atmosphere, dim lighting",
    "joyful": "bright mood, soft lighting",
    "sorrowful": "somber atmosphere, soft shadows",
    "peaceful": "peaceful atmosphere, gentle lighting",
}


# ============================================================================
# LLM System Prompt
# ============================================================================

_STORYBOARD_SYSTEM_PROMPT = """You are a professional manga storyboard director for Animagine-XL-4.0, producing BLACK AND WHITE MANGA.

Design cinematography for ONE panel. Output ONLY valid JSON — no markdown, no extra text.

=== FIELDS (flat JSON, not wrapped) ===
- order_in_page: integer (1-based)
- shot_size: "extreme_long" | "long" | "full" | "medium" | "close" | "extreme_close"
- camera_angle: "eye_level" | "high_angle" | "low_angle" | "dutch_angle"
- pose_hint: FULL body pose Danbooru tags describing the EXACT action pose.
  Examples: "running, one leg forward, arms pumping" / "crouching, reaching out, alert" /
  "standing tall, arms crossed, confident stance" / "kicking, one leg raised, arms out for balance"
- expression: facial expression Danbooru tags (e.g., "determined expression, furrowed brow" /
  "surprised, wide eyes, open mouth")
- setting: 1-2 key background tags ONLY. For close-up/extreme_close: use "plain background" or
  empty. For medium: 1 background element. For long/full: 2 elements max.
  Examples: "school hallway, lockers" / "rooftop, night sky" / "plain background"
- scene_lighting: 1-2 words max. Use VISUAL lighting terms (rim lighting, backlight, harsh
  shadows, moonlight, dim). NEVER use "natural light", "natural sunlight".
- mood: "tense" | "dramatic" | "mysterious" | "joyful" | "sorrowful" | "peaceful"
- emotion_intensity: 0.0-1.0
- weather: visual weather tags if scene is outdoors. Use "rain", "snow", "fog", "storm",
  "overcast", "clear sky". Leave empty "" for indoor scenes.

=== SHOT SIZE vs ACTION MATCHING ===
| shot_size      | frame  | allowed actions                          | FORBIDDEN actions        |
|----------------|--------|------------------------------------------|--------------------------|
| extreme_close  | face   | facial expression ONLY, detailed eyes    | walking, running, kicking|
| close          | head   | facial expression, talking, slight turn  | walking, running, jumping|
| medium         | waist  | hand gestures, holding objects, reaching | running, kicking         |
| full           | body   | walking, standing, sitting, crouching    | -                        |
| long           | body+  | running, jumping, dynamic fight poses    | -                        |
| extreme_long   | scene  | small figure in environment              | detailed facial expression|

=== PREVIOUS PANEL CONTEXT (when provided) ===
- Vary shot_size from the previous panel for visual rhythm.
- If previous panel used a specific angle, consider a complementary angle.
- Maintain character facing direction for continuous actions within the same scene.
- If previous panel showed a reaction, this panel can show the subject of that reaction.

=== FORBIDDEN WORDS ===
- NO abstract mood words: "quiet", "soft", "serene", "calm atmosphere"
- NO "natural light", "natural sunlight", "daylight"
- NO vague body descriptors in setting or lighting

=== RULES ===
- Vary shot sizes across adjacent panels for visual rhythm.
- Action panels: dynamic angles (low_angle, dutch_angle).
- Dialogue/expression panels: medium or close shots, eye_level.
- Establishing panels: long or extreme_long shots.
- pose_hint MUST be consistent with shot_size.
- Weather should enhance the scene's mood (rain for sorrow, storm for tension, etc.)."""


# Animagine XL 4.0 standard tag mapping
_SHOT_SIZE_TAGS: dict[str, str] = {
    "extreme_long": "panoramic view",
    "long": "long shot",
    "full": "full shot",
    "medium": "medium shot",
    "close": "close-up",
    "extreme_close": "extreme close-up",
}


def _map_shot_size(shot_value: str) -> str:
    return _SHOT_SIZE_TAGS.get(shot_value, shot_value.replace("_", " "))


def _ensure_str(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value) if value else ""


def _strip_gender_tag_prefix(char_tags: str, gender_tag: str) -> str:
    char_tags = _ensure_str(char_tags)
    if not char_tags or not gender_tag:
        return char_tags
    tags = [t.strip() for t in char_tags.split(",")]
    if tags and tags[0] == gender_tag:
        tags = tags[1:]
    return ", ".join(tags)


def _deduplicate_tags(tags_str: str) -> str:
    tags_str = _ensure_str(tags_str)
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags_str.split(","):
        t = tag.strip()
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return ", ".join(result)


def _build_prompt_from_llm_data(panel_data: dict, char_gender_tag: str) -> PromptPack:
    """Assemble Animagine-XL-4.0 prompt from LLM-designed panel data.

    v0.5: mood, time_of_day, and weather now enter the prompt.
    """
    style_prefix = "masterpiece, high score, great score, absurdres, safe"

    # Character tags
    char_tags = panel_data.get("_char_tags", "")
    char_tags = _strip_gender_tag_prefix(char_tags, char_gender_tag)
    char_tags = char_tags.replace("solo, ", "").replace(", solo", "").replace("solo", "")

    # Composition & camera
    shot = panel_data.get("shot_size", "medium")
    angle = panel_data.get("camera_angle", "eye_level")
    shot_desc = _map_shot_size(str(shot))
    angle_desc = str(angle).replace("_", " ")
    comp_parts = [shot_desc, angle_desc]

    # Action & expression
    for key in ("pose_hint", "expression"):
        v = panel_data.get(key, "")
        if v:
            comp_parts.append(v)

    # Background: setting + lighting + time_of_day + weather + mood visual tags
    scene_parts = []
    setting = _ensure_str(panel_data.get("setting", ""))
    if setting:
        setting_tags = [s.strip() for s in setting.split(",")][:2]
        scene_parts.extend(setting_tags)

    scene_lighting = panel_data.get("scene_lighting", "")
    if scene_lighting:
        scene_parts.append(scene_lighting)

    # v0.5: time_of_day enters prompt
    time_of_day = panel_data.get("time_of_day", "")
    if time_of_day:
        scene_parts.append(time_of_day)

    # v0.5: weather enters prompt
    weather = panel_data.get("weather", "")
    if weather:
        scene_parts.append(weather)

    # v0.5: mood enters prompt as visual tags
    mood_val = panel_data.get("mood", "")
    if mood_val and mood_val in _MOOD_VISUAL_TAGS:
        scene_parts.append(_MOOD_VISUAL_TAGS[mood_val])

    # Build positive prompt
    tag_parts = [
        style_prefix,
        f"{char_gender_tag}, solo, {char_tags}",
        ", ".join(comp_parts),
    ]
    if scene_parts:
        tag_parts.append(", ".join(scene_parts))
    tag_parts.append(
        "monochrome, greyscale, black and white manga style, "
        "screentone, halftone, ink drawing, clean lineart, high contrast"
    )
    pos = ", ".join(p for p in tag_parts if p)
    pos = _deduplicate_tags(pos)

    neg = (
        "lowres, worst quality, bad quality, displeasing, low resolution, "
        "bad anatomy, bad hands, bad fingers, extra digits, fewer digits, "
        "cropped, signature, watermark, username, blurry, text, error, "
        "color, colored, multicolored, gradient, rainbow, "
        "3d, realistic, photo, photograph, photorealistic, "
        "nsfw, explicit"
    )

    return PromptPack(
        positive_prompt=pos,
        negative_prompt=neg,
        style_tags=["monochrome", "greyscale", "black and white manga", "screentone"],
        composition_tags=[shot, angle],
        quality_tags=["masterpiece", "high score", "great score", "absurdres", "clean lineart"],
    )


class StoryboardAgent(BaseAgent[StoryboardInput, StoryboardOutput]):
    """Storyboard planning agent — LLM-driven cinematography design."""

    name = "storyboard_agent"
    version = "0.5.0-llm"
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

    async def _run_with_llm(
        self, inputs: StoryboardInput, context: AgentContext
    ) -> StoryboardOutput:
        """LLM-driven panel design with previous-panel context."""
        await self._sleep_for_demo(0.3)

        # Build character reference info
        char_refs: dict[str, dict] = {}
        if inputs.character_db:
            for cid, cp in inputs.character_db.characters.items():
                char_refs[cid] = {
                    "char_id": cid,
                    "name": cp.name,
                    "core_tags": (cp.core_tags or cp.appearance_prompt)[:300],
                    "gender_tag": cp.gender_tag or "1girl",
                }

        # Expand scenes into panel specs
        panel_specs: list[tuple[Scene, int, str]] = []
        for scene in inputs.scenes:
            chars = scene.characters_present
            for i in range(max(1, scene.panel_hint)):
                char_for_panel = chars[i % len(chars)] if chars else ""
                panel_specs.append((scene, i, char_for_panel))

        if not self.config.llm.is_available:
            raise ApiBackendError("LLM is not available — cannot design storyboard panels")

        # Design each panel with previous-panel context (v0.5)
        panel_designs: list[dict] = []
        total = len(panel_specs)
        prev_context: PreviousPanelContext | None = None

        for idx, (scene, panel_idx, char_for_panel) in enumerate(panel_specs):
            # v0.5: Build all-actions summary for LLM
            all_actions = _format_all_actions(scene)

            design = await self._design_single_panel_via_llm(
                scene, panel_idx, char_for_panel, char_refs,
                idx, total, prev_context, all_actions,
                story_summary=inputs.story_summary,
                narrative_structure=inputs.narrative_structure,
            )
            panel_designs.append(design)

            # v0.5: Build PreviousPanelContext for the NEXT panel
            prev_context = PreviousPanelContext(
                panel_id=design.get("panel_id", f"panel_{idx:03d}"),
                shot_size=str(design.get("shot_size", "medium")),
                camera_angle=str(design.get("camera_angle", "eye_level")),
                characters_in_panel=[char_for_panel] if char_for_panel else [],
                pose_hint=str(design.get("pose_hint", "")),
                expression=str(design.get("expression", "")),
                setting_summary=str(design.get("setting", scene.location)),
                emotion=str(design.get("mood", scene.atmosphere)),
                emotion_intensity=float(design.get("emotion_intensity", scene.emotion_intensity)),
            )

        # Group into pages
        per_page = max(1, inputs.target_panels_per_page)
        pages: list[PageLayout] = []
        page_idx = 0
        for start in range(0, len(panel_specs), per_page):
            chunk = panel_specs[start:start + per_page]
            chunk_designs = panel_designs[start:start + per_page]
            climax = any(s.emotion_intensity >= 0.85 for s, _, _ in chunk)
            action_heavy = any(
                s.actions and len(s.actions) >= 2 for s, _, _ in chunk
            )

            layout_hint = _select_layout_hint(len(chunk), climax, action_heavy)
            page_id = f"page_{page_idx + 1:03d}"
            panels: list[PanelPlan] = []

            for i, (scene, panel_idx, char_for_panel) in enumerate(chunk):
                design = chunk_designs[i]
                panel_id = design.get("panel_id", f"{page_id}_p{i + 1:02d}")
                try:
                    shot = ShotSize(design.get("shot_size", "medium"))
                except ValueError:
                    shot = ShotSize.MEDIUM
                try:
                    angle = CameraAngle(design.get("camera_angle", "eye_level"))
                except ValueError:
                    angle = CameraAngle.EYE_LEVEL

                # v0.5: Use ALL actions, not just actions[0]
                pose_hint = design.get("pose_hint", (
                    scene.actions[0].description if scene.actions else "standing"
                ))
                setting = design.get("setting", scene.location)
                mood_val = design.get("mood", None) or scene.atmosphere
                emotion = float(design.get("emotion_intensity", scene.emotion_intensity))
                expression = design.get("expression", "")
                scene_lighting = design.get("scene_lighting", "")
                weather_val = design.get("weather", "")  # v0.5: weather populated by LLM
                time_of_day = scene.time_of_day

                # Character info
                char_core_tags = ""
                char_gender_tag = "1girl"
                if char_for_panel and char_for_panel in char_refs:
                    ref = char_refs[char_for_panel]
                    char_core_tags = ref.get("core_tags", "")
                    char_gender_tag = ref.get("gender_tag", "1girl")

                single_char_list = [char_for_panel] if char_for_panel else []

                # Build PromptPack with mood/time/weather in prompt (v0.5)
                prompt_data = {
                    "_char_tags": char_core_tags if char_core_tags else "",
                    "shot_size": shot.value,
                    "camera_angle": angle.value,
                    "pose_hint": pose_hint,
                    "expression": expression,
                    "setting": setting,
                    "scene_lighting": scene_lighting,
                    "mood": mood_val,
                    "time_of_day": time_of_day,
                    "weather": weather_val,
                }
                prompt_pack = _build_prompt_from_llm_data(prompt_data, char_gender_tag)

                # v0.5: Distribute dialogues across panels, not just first panel
                panel_dialogues = _distribute_dialogues(scene.dialogues, panel_idx, scene.panel_hint)
                bubble_hints = [
                    BubbleHint(
                        dialogue_index=di,
                        suggested_position="top_right" if di % 2 == 0 else "bottom_left",
                        bubble_type="speech" if not d.is_thought else "thought",
                    )
                    for di, d in enumerate(panel_dialogues)
                ]

                # v0.5: Build previous_panel_context for this panel
                pctx = _build_panel_context(idx=i, chunk_idx=start + i, panel_designs=panel_designs, start=start)

                panels.append(
                    PanelPlan(
                        panel_id=panel_id,
                        page_id=page_id,
                        order_in_page=i + 1,
                        source_scene_id=scene.scene_id,
                        source_dialogue_indices=list(range(len(panel_dialogues))),
                        shape=PanelShape.RECTANGLE,
                        shot_size=shot,
                        camera_angle=angle,
                        characters_in_panel=single_char_list,
                        primary_action=pose_hint,
                        setting=setting,
                        mood=mood_val,
                        emotion_intensity=emotion,
                        prompt_pack=prompt_pack,
                        dialogues_in_panel=panel_dialogues,
                        speech_bubble_hints=bubble_hints,
                        pose_hint=pose_hint,
                        expression=expression,
                        scene_lighting=scene_lighting,
                        weather=weather_val,
                        time_of_day=time_of_day,
                        previous_panel_context=pctx,
                    )
                )

            avg_emotion = sum(s.emotion_intensity for s, _, _ in chunk) / len(chunk)
            pages.append(
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

        return StoryboardOutput(
            pages=pages,
            total_panels=sum(len(p.panels) for p in pages),
            meta=AgentOutputMeta(
                agent=self.name,
                version=self.version,
                inputs_hash=self._inputs_hash(inputs),
                self_check_notes=[
                    f"LLM panel design: {len(pages)} pages, {sum(len(p.panels) for p in pages)} panels",
                ],
            ),
        )

    async def _run_local(
        self, inputs: StoryboardInput, context: AgentContext
    ) -> StoryboardOutput:
        """Local fallback: deterministic panel design without LLM (v0.5)."""
        await self._sleep_for_demo(0.2)

        char_refs: dict[str, dict] = {}
        if inputs.character_db:
            for cid, cp in inputs.character_db.characters.items():
                char_refs[cid] = {
                    "char_id": cid,
                    "name": cp.name,
                    "core_tags": (cp.core_tags or cp.appearance_prompt)[:300],
                    "gender_tag": cp.gender_tag or "1girl",
                }

        panel_specs: list[tuple[Scene, int, str]] = []
        for scene in inputs.scenes:
            chars = scene.characters_present
            for i in range(max(1, scene.panel_hint)):
                char_for_panel = chars[i % len(chars)] if chars else ""
                panel_specs.append((scene, i, char_for_panel))

        per_page = max(1, inputs.target_panels_per_page)
        pages: list[PageLayout] = []
        page_idx = 0
        panel_counter = 0

        for start in range(0, len(panel_specs), per_page):
            chunk = panel_specs[start:start + per_page]
            page_id = f"page_{page_idx + 1:03d}"
            panels: list[PanelPlan] = []

            for i, (scene, panel_idx, char_for_panel) in enumerate(chunk):
                panel_id = f"{page_id}_p{i + 1:02d}"
                # Alternate shots for visual rhythm
                shots = [ShotSize.LONG, ShotSize.MEDIUM, ShotSize.CLOSE, ShotSize.FULL]
                shot = shots[panel_counter % len(shots)]
                angle = CameraAngle.EYE_LEVEL

                char_core_tags = ""
                char_gender_tag = "1girl"
                if char_for_panel and char_for_panel in char_refs:
                    ref = char_refs[char_for_panel]
                    char_core_tags = ref.get("core_tags", "")
                    char_gender_tag = ref.get("gender_tag", "1girl")

                prompt_data = {
                    "_char_tags": char_core_tags,
                    "shot_size": shot.value,
                    "camera_angle": angle.value,
                    "pose_hint": scene.actions[0].description if scene.actions else "standing",
                    "expression": "neutral expression",
                    "setting": scene.location,
                    "scene_lighting": "",
                    "mood": scene.atmosphere,
                    "time_of_day": scene.time_of_day,
                    "weather": "",
                }
                prompt_pack = _build_prompt_from_llm_data(prompt_data, char_gender_tag)

                panel_dialogues = _distribute_dialogues(
                    scene.dialogues, panel_idx, scene.panel_hint
                )

                prev_ctx = _build_panel_context_local(i, panel_counter, panels)

                panels.append(
                    PanelPlan(
                        panel_id=panel_id,
                        page_id=page_id,
                        order_in_page=i + 1,
                        source_scene_id=scene.scene_id,
                        shape=PanelShape.RECTANGLE,
                        shot_size=shot,
                        camera_angle=angle,
                        characters_in_panel=[char_for_panel] if char_for_panel else [],
                        primary_action=scene.actions[0].description if scene.actions else "standing",
                        setting=scene.location,
                        mood=scene.atmosphere,
                        emotion_intensity=scene.emotion_intensity,
                        prompt_pack=prompt_pack,
                        dialogues_in_panel=panel_dialogues,
                        pose_hint=scene.actions[0].description if scene.actions else "standing",
                        expression="neutral expression",
                        scene_lighting="",
                        weather="",
                        time_of_day=scene.time_of_day,
                        previous_panel_context=prev_ctx,
                    )
                )
                panel_counter += 1

            avg_emotion = sum(s.emotion_intensity for s, _, _ in chunk) / len(chunk)
            climax = any(s.emotion_intensity >= 0.85 for s, _, _ in chunk)
            pages.append(
                PageLayout(
                    page_id=page_id,
                    page_number=page_idx + 1,
                    layout_template="grid_2x2",
                    layout_hint=_select_layout_hint(len(chunk), climax, False),
                    panels=panels,
                    page_emotion_avg=avg_emotion,
                    is_climax_page=climax,
                )
            )
            page_idx += 1

        return StoryboardOutput(
            pages=pages,
            total_panels=sum(len(p.panels) for p in pages),
            meta=AgentOutputMeta(
                agent=self.name,
                version=self.version,
                self_check_notes=["Local fallback — deterministic panel design"],
            ),
        )

    async def _design_single_panel_via_llm(
        self,
        scene: Scene,
        panel_idx: int,
        char_name: str,
        char_refs: dict,
        global_idx: int = 0,
        total_panels: int = 1,
        prev_context: PreviousPanelContext | None = None,
        all_actions: str = "",
        story_summary: str = "",
        narrative_structure: NarrativeStructure | None = None,
    ) -> dict:
        """Design a single panel via LLM with full context (v0.5)."""
        client = OpenAICompatibleLLMClient(self.config.llm)

        char_info = char_refs.get(char_name, {})
        char_tags = char_info.get("core_tags", "")[:200]
        char_gender = char_info.get("gender_tag", "1girl")

        # Build scene info with all actions
        scene_info = {
            "scene_id": scene.scene_id,
            "location": scene.location,
            "time_of_day": scene.time_of_day,
            "atmosphere": scene.atmosphere,
            "emotion_intensity": scene.emotion_intensity,
            "all_actions": all_actions,
            "visual_hook": getattr(scene, "visual_hook", ""),
            "shot_sequence_hint": getattr(scene, "shot_sequence_hint", ""),
        }

        user_payload: dict = {
            "panel_index": f"{global_idx + 1}/{total_panels}",
            "panel_idx_in_scene": panel_idx,
            "scene": scene_info,
            "character": {
                "name": char_name,
                "gender_tag": char_gender,
                "tags": char_tags,
            } if char_name else None,
            "style": "black and white manga, monochrome, screentone, ink drawing",
        }

        # v0.5: Add previous panel context
        if prev_context:
            user_payload["previous_panel"] = {
                "shot_size": prev_context.shot_size,
                "camera_angle": prev_context.camera_angle,
                "pose_hint": prev_context.pose_hint,
                "expression": prev_context.expression,
                "emotion": prev_context.emotion,
                "setting": prev_context.setting_summary,
            }

        # v0.5: Add story context
        if story_summary:
            user_payload["story_context"] = story_summary[:300]
        if narrative_structure:
            user_payload["narrative_phase"] = narrative_structure.model_dump()

        data = await asyncio.to_thread(
            client.complete_json,
            _STORYBOARD_SYSTEM_PROMPT,
            user_payload,
        )
        if not isinstance(data, dict):
            raise ApiBackendError(
                f"LLM panel design returned unexpected format: {type(data).__name__}"
            )
        return data

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
# Helper functions (v0.5)
# ============================================================================

def _format_all_actions(scene: Scene) -> str:
    """Format all actions in a scene for the LLM (v0.5: not just actions[0])."""
    if not scene.actions:
        return "standing"
    return "; ".join(
        f"{a.actor}: {a.description}" + (f" (target: {a.target})" if a.target else "")
        for a in scene.actions
    )


def _distribute_dialogues(
    dialogues: list[Dialogue], panel_idx: int, total_panels: int
) -> list[Dialogue]:
    """Distribute dialogues across panels within a scene (v0.5: not just first panel).

    If there are multiple panels in a scene, dialogues are spread proportionally.
    Panel 0 gets the first portion, panel 1 gets the next, etc.
    """
    if total_panels <= 1:
        return list(dialogues)
    if not dialogues:
        return []

    # Split dialogues evenly across panels
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
        # First panel in page — check if there's a previous page's last panel
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

    # Previous panel within same page
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
