"""Storyboard planning agent — LLM-driven panel cinematography design.

参照 comic_agent.py 的 StoryboardAgent：由 LLM 设计每个分镜的
镜头大小、机位角度、角色姿势、表情、场景光照、氛围等细节，
而非使用硬编码的规则选取。
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
    PageLayout,
    PanelPlan,
    PanelShape,
    PromptPack,
    Scene,
    ShotSize,
    StoryboardInput,
    StoryboardOutput,
    StreamEvent,
    StreamEventType,
)

# ============================================================================
# Layout hints — intent passed to LayoutAgent (NO coordinates).
# LayoutAgent uses these to guide its recursive binary-partition solver.
# ============================================================================


def _select_layout_hint(panel_count: int, climax: bool, action_heavy: bool) -> str:
    """Return a layout *hint* for the page (intent, not coordinates).

    The LayoutAgent uses this to bias its split-direction heuristics.
    """
    if climax:
        return "climax"
    if action_heavy:
        return "action"
    if panel_count >= 5:
        return "dialogue"  # many panels → likely dialogue-heavy
    if panel_count <= 1:
        return "climax"     # single panel → splash / establishing
    return "standard"


# ============================================================================
# LLM 分镜设计的 System Prompt（参照 comic_agent.py StoryboardAgent）
# ============================================================================

_STORYBOARD_SYSTEM_PROMPT = """You are a professional manga storyboard director for Animagine-XL-4.0, producing BLACK AND WHITE MANGA.

Design cinematography for ONE panel. Output ONLY valid JSON — no markdown, no extra text.

=== FIELDS (flat JSON, not wrapped) ===
- order_in_page: integer (1-based)
- shot_size: "extreme_long" | "long" | "full" | "medium" | "close" | "extreme_close"
- camera_angle: "eye_level" | "high_angle" | "low_angle" | "dutch_angle"
- pose_hint: FULL body pose Danbooru tags describing the EXACT action pose — this replaces
  "primary_action". Describe the entire body position during the action.
  Examples: "running, one leg forward, arms pumping" / "crouching, reaching out, alert" /
  "standing tall, arms crossed, confident stance" / "kicking, one leg raised, arms out for balance"
- expression: facial expression Danbooru tags (e.g., "determined expression, furrowed brow" /
  "surprised, wide eyes, open mouth")
- setting: 1-2 key background tags ONLY. For close-up/extreme_close: use "plain background" or
  empty. For medium: 1 background element. For long/full: 2 elements max.
  Examples: "school hallway, lockers" / "rooftop, night sky" / "plain background"
- scene_lighting: 1-2 words max. Use VISUAL lighting terms (rim lighting, backlight, harsh
  shadows, moonlight, dim). NEVER use "natural light", "natural sunlight" — these are invisible
  in B&W manga.
- mood: "tense" | "dramatic" | "mysterious" | "joyful" | "sorrowful" | "peaceful"
- emotion_intensity: 0.0 — 1.0

=== CRITICAL: SHOT SIZE vs ACTION MATCHING ===
MISMATCHED shot+action is the #1 cause of broken images. Follow these rules strictly:

| shot_size      | frame  | allowed actions                          | FORBIDDEN actions        |
|----------------|--------|------------------------------------------|--------------------------|
| extreme_close  | face   | facial expression ONLY, detailed eyes    | walking, running, kicking|
| close          | head   | facial expression, talking, slight turn  | walking, running, jumping|
| medium         | waist  | hand gestures, holding objects, reaching | running, kicking         |
| full           | body   | walking, standing, sitting, crouching    | —                        |
| long           | body+  | running, jumping, dynamic fight poses    | —                        |
| extreme_long   | scene  | small figure in environment              | detailed facial expression|

Always think: "Can the viewer SEE this action at this shot distance?" If not, change the shot_size.

=== FORBIDDEN WORDS (will confuse Animagine) ===
- NO abstract mood words: "quiet", "soft", "serene", "calm atmosphere" — use visual equivalents
- NO "natural light", "natural sunlight", "daylight" — use "harsh lighting", "soft shadows" etc.
- NO vague body descriptors in setting or lighting

=== ADDITIONAL RULES ===
- Vary shot sizes across adjacent panels for visual rhythm.
- Action panels → dynamic angles (low_angle, dutch_angle).
- Dialogue/expression panels → medium or close shots, eye_level.
- Establishing panels → long or extreme_long shots.
- pose_hint MUST be consistent with shot_size — close-up must NOT describe leg/foot positions."""


# Animagine XL 4.0 标准标签映射：ShotSize → 官方 Danbooru tag
_SHOT_SIZE_TAGS: dict[str, str] = {
    "extreme_long": "panoramic view",
    "long": "long shot",
    "full": "full shot",
    "medium": "medium shot",
    "close": "close-up",
    "extreme_close": "extreme close-up",
}


def _map_shot_size(shot_value: str) -> str:
    """Convert internal shot_size enum value to Animagine XL 4.0 standard tag."""
    return _SHOT_SIZE_TAGS.get(shot_value, shot_value.replace("_", " "))


def _ensure_str(value) -> str:
    """Normalise a value that may be str or list[str] to a comma-separated string."""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value) if value else ""


def _strip_gender_tag_prefix(char_tags: str, gender_tag: str) -> str:
    """Remove the gender_tag prefix from char_tags to avoid duplication.

    Example: _strip_gender_tag_prefix("1girl, solo, young_adult, ...", "1girl")
            -> "solo, young_adult, ..."
    """
    char_tags = _ensure_str(char_tags)
    if not char_tags or not gender_tag:
        return char_tags
    tags = [t.strip() for t in char_tags.split(",")]
    if tags and tags[0] == gender_tag:
        tags = tags[1:]
    return ", ".join(tags)


def _deduplicate_tags(tags_str: str) -> str:
    """Remove duplicate tags while preserving first-occurrence order."""
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

    Tag weight ordering (highest to lowest):
    1. Quality/style prefix
    2. Character traits (CORE — drives character consistency)
    3. Composition & camera (shot/angle)
    4. Action & expression (character dynamics)
    5. Background (MINIMAL — 1-2 key tags only)
    6. B&W manga style suffix
    """
    # Animagine XL 4.0 质量前缀（官方训练标签）
    style_prefix = "masterpiece, high score, great score, absurdres, safe"

    # Character tags — strip duplicate gender_tag prefix
    char_tags = panel_data.get("_char_tags", "")
    char_tags = _strip_gender_tag_prefix(char_tags, char_gender_tag)
    char_tags = char_tags.replace("solo, ", "").replace(", solo", "").replace("solo", "")

    # Composition & camera（使用 Animagine 标准标签）
    shot = panel_data.get("shot_size", "medium")
    angle = panel_data.get("camera_angle", "eye_level")
    shot_desc = _map_shot_size(str(shot))
    angle_desc = str(angle).replace("_", " ")
    comp_parts = [shot_desc, angle_desc]

    # Action & expression (pose_hint 已包含完整动作姿态)
    for key in ("pose_hint", "expression"):
        v = panel_data.get(key, "")
        if v:
            comp_parts.append(v)

    # Background — minimal, only 1-2 key tags
    scene_parts = []
    setting = _ensure_str(panel_data.get("setting", ""))
    if setting:
        setting_tags = [s.strip() for s in setting.split(",")][:2]
        scene_parts.extend(setting_tags)
    scene_lighting = panel_data.get("scene_lighting", "")
    if scene_lighting:
        scene_parts.append(scene_lighting)

    # Build positive prompt: character weight > composition > background
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

    # Deduplicate
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
    version = "0.4.0-llm"
    rubric_id = "rubric_storyboard_v1"

    async def run(
        self, inputs: StoryboardInput, context: AgentContext
    ) -> StoryboardOutput:
        """LLM 驱动的分镜设计 —— 不再区分 API/Local，统一走 LLM 路径。"""
        return await self._run_with_llm(inputs, context)

    async def _run_with_llm(
        self, inputs: StoryboardInput, context: AgentContext
    ) -> StoryboardOutput:
        """LLM 驱动分镜设计 + 机械布局分配。

        每个 panel 单独调用一次 LLM，减轻单次调用的 token 压力。
        """
        await self._sleep_for_demo(0.3)

        # 构建角色参考信息（含 LLM 生成的 core_tags）
        char_refs: dict[str, dict] = {}
        if inputs.character_db:
            for cid, cp in inputs.character_db.characters.items():
                char_refs[cid] = {
                    "char_id": cid,
                    "name": cp.name,
                    "core_tags": (cp.core_tags or cp.appearance_prompt)[:300],
                    "gender_tag": cp.gender_tag or "1girl",
                }

        # 展开场景为 panel 规格
        panel_specs: list[tuple[Scene, int, str]] = []  # (scene, panel_idx, char_for_panel)
        for scene in inputs.scenes:
            chars = scene.characters_present
            for i in range(max(1, scene.panel_hint)):
                char_for_panel = chars[i % len(chars)] if chars else ""
                panel_specs.append((scene, i, char_for_panel))

        # === 逐 panel 调用 LLM，每次只设计一个分镜 ===
        if not self.config.llm.is_available:
            raise ApiBackendError("LLM is not available — cannot design storyboard panels")
        panel_designs: list[dict] = []
        total = len(panel_specs)
        for idx, (scene, panel_idx, char_for_panel) in enumerate(panel_specs):
            design = await self._design_single_panel_via_llm(
                scene, panel_idx, char_for_panel, char_refs, idx, total,
            )
            panel_designs.append(design)

        # 按目标 panels_per_page 分页
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

            # Layout hint only — LayoutAgent computes actual bboxes
            layout_hint = _select_layout_hint(len(chunk), climax, action_heavy)

            page_id = f"page_{page_idx + 1:03d}"
            panels: list[PanelPlan] = []
            for i, (scene, panel_idx, char_for_panel) in enumerate(chunk):
                # 从 LLM 返回中读取设计数据（不可用默认值 — LLM 失败已 raise）
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
                pose_hint = design.get("pose_hint", scene.actions[0].description if scene.actions else "standing")
                setting = design.get("setting", scene.location)
                mood_val = design.get("mood", None) or scene.atmosphere
                emotion = float(design.get("emotion_intensity", scene.emotion_intensity))
                expression = design.get("expression", "")
                scene_lighting = design.get("scene_lighting", "")

                # 获取角色信息（来自 CharacterDB）
                char_core_tags = ""
                char_gender_tag = "1girl"
                if char_for_panel and char_for_panel in char_refs:
                    ref = char_refs[char_for_panel]
                    char_core_tags = ref.get("core_tags", "")
                    char_gender_tag = ref.get("gender_tag", "1girl")

                single_char_list = [char_for_panel] if char_for_panel else []

                # 构建 PromptPack（使用 LLM 数据）
                prompt_data = {
                    "_char_tags": char_core_tags if char_core_tags else "",
                    "shot_size": shot.value,
                    "camera_angle": angle.value,
                    "pose_hint": pose_hint,
                    "expression": expression,
                    "setting": setting,
                    "scene_lighting": scene_lighting,
                    "mood": mood_val,
                }
                prompt_pack = _build_prompt_from_llm_data(prompt_data, char_gender_tag)
                bubble_hints = [
                    BubbleHint(
                        dialogue_index=di,
                        suggested_position="top_right" if di % 2 == 0 else "bottom_left",
                        bubble_type="speech" if not d.is_thought else "thought",
                    )
                    for di, d in enumerate(scene.dialogues)
                ]

                panels.append(
                    PanelPlan(
                        panel_id=panel_id,
                        page_id=page_id,
                        order_in_page=i + 1,
                        source_scene_id=scene.scene_id,
                        source_dialogue_indices=list(range(len(scene.dialogues))),
                        # bbox uses default (full page) — LayoutAgent overrides
                        shape=PanelShape.RECTANGLE,
                        shot_size=shot,
                        camera_angle=angle,
                        characters_in_panel=single_char_list,
                        primary_action=pose_hint,  # pose_hint 已替代旧的 primary_action
                        setting=setting,
                        mood=mood_val,
                        emotion_intensity=emotion,
                        prompt_pack=prompt_pack,
                        dialogues_in_panel=list(scene.dialogues) if panel_idx == 0 else [],
                        speech_bubble_hints=bubble_hints if panel_idx == 0 else [],
                        pose_hint=pose_hint,
                        expression=expression,
                        scene_lighting=scene_lighting,
                        weather="",
                        time_of_day=scene.time_of_day,
                    )
                )

            avg_emotion = sum(s.emotion_intensity for s, _, _ in chunk) / len(chunk)
            pages.append(
                PageLayout(
                    page_id=page_id,
                    page_number=page_idx + 1,
                    layout_template=layout_hint,  # keep for backward compat; real intent in layout_hint
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
                    f"LLM分镜设计: {len(pages)}页, {sum(len(p.panels) for p in pages)}格",
                ],
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
    ) -> dict:
        """为单个 panel 调用 LLM 设计摄影参数。

        每次只发送一个场景 + 一个角色的信息，token 消耗极低。
        """
        client = OpenAICompatibleLLMClient(self.config.llm)

        # 获取角色信息
        char_info = char_refs.get(char_name, {})
        char_tags = char_info.get("core_tags", "")[:200]
        char_gender = char_info.get("gender_tag", "1girl")

        user_payload = {
            "panel_index": f"{global_idx + 1}/{total_panels}",
            "scene": {
                "scene_id": scene.scene_id,
                "location": scene.location,
                "time_of_day": scene.time_of_day,
                "atmosphere": scene.atmosphere,
                "emotion_intensity": scene.emotion_intensity,
                "key_action": scene.actions[0].description if scene.actions else "",
            },
            "character": {
                "name": char_name,
                "gender_tag": char_gender,
                "tags": char_tags,
            } if char_name else None,
            "style": "black and white manga, monochrome, screentone, ink drawing",
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
        return data

    async def astream(
        self, inputs: StoryboardInput, context: AgentContext
    ) -> AsyncIterator[StreamEvent]:
        yield self._make_event(StreamEventType.LOG, "开始分镜规划...")
        yield self._make_event(StreamEventType.PROGRESS, 0.1)

        await self._sleep_for_demo(0.15)
        yield self._make_event(StreamEventType.THINKING, "为每场景分配分镜数...")
        yield self._make_event(StreamEventType.PROGRESS, 0.3)

        await self._sleep_for_demo(0.15)
        yield self._make_event(StreamEventType.THINKING, "选择布局模板与景别...")
        yield self._make_event(StreamEventType.PROGRESS, 0.6)

        await self._sleep_for_demo(0.15)
        yield self._make_event(StreamEventType.THINKING, "拼装提示词...")

        output = await self.run(inputs, context)
        yield self._make_event(
            StreamEventType.PARTIAL_OUTPUT,
            {"pages": len(output.pages), "panels": output.total_panels},
        )
        yield self._make_event(StreamEventType.PROGRESS, 1.0)
        yield self._make_event(StreamEventType.DONE, output.model_dump())
