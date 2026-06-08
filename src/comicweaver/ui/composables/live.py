"""实时预览面板 — 组合多个渲染器输出为右侧面板。

设计原则：
- 文字面板（render_phase_text_content）：按工作流顺序展示各阶段的详细输出，含内联图片。
- 图像预览已完全并入文字面板（角色设计、图像生成、台词气泡各自展示缩略图），
  不再需要独立的 render_live_panel_preview。
"""
from __future__ import annotations

from ..renderers.bubbles import _render_bubble_summary
from ..renderers.characters import _render_detailed_characters
from ..renderers.images import _render_detailed_images
from ..renderers.layout import _render_detailed_final_pages
from ..renderers.story import _render_detailed_script, _render_story_outline
from ..renderers.storyboard import _render_detailed_storyboard


def render_phase_text_content(state: dict | None, agent_outputs: dict[str, list[dict]]) -> str:
    """按工作流顺序渲染所有已完成阶段的详细文字内容（含内联图片）。

    顺序：story → character → script → storyboard → image → bubble → layout
    """
    if not state:
        return '<div class="cw-phase-text-empty">尚未创建项目</div>'

    parts: list[str] = []

    # 1. 故事阶段
    story = state.get("developed_story", {})
    if story:
        parts.append(_render_story_outline(story))

    # 2. 角色阶段（先于剧本）
    character_db = state.get("character_db", {})
    char_id_to_name: dict[str, str] = {}
    if character_db:
        for cid, cp in character_db.get("characters", {}).items():
            char_id_to_name[cid] = cp.get("name", cid)
        parts.append(_render_detailed_characters(character_db))

    # 3. 剧本阶段
    script = state.get("structured_script", {})
    if script:
        parts.append(_render_detailed_script(script, char_id_to_name, story))

    # 4. 分镜阶段
    storyboard = state.get("storyboard_plan", [])
    if storyboard:
        parts.append(_render_detailed_storyboard(storyboard, char_id_to_name))

    # 5. 图像阶段
    panel_images = state.get("panel_images", [])
    if panel_images:
        parts.append(_render_detailed_images(panel_images))

    # 6. 台词气泡阶段（含面板缩略图）
    bubble_placements = state.get("bubble_placements", {})
    if bubble_placements:
        parts.append(_render_bubble_summary(bubble_placements, panel_images))

    # 7. 排版阶段
    final_pages = state.get("final_pages", [])
    if final_pages:
        parts.append(_render_detailed_final_pages(final_pages, storyboard))

    if not parts:
        return '<div class="cw-phase-text-empty">等待工作流启动...</div>'

    return "".join(parts)


def render_live_panel_preview(
    panel_images: list[dict],
    character_images: list[dict] | None = None,
    bubble_placements: dict | None = None,
) -> str:
    """[已废弃] 图像预览已并入文字面板，此函数保留仅为向后兼容。"""
    return ""


def render_live_content(
    panel_images: list[dict],
    character_images: list[dict],
    text_html: str,
    bubble_placements: dict | None = None,
) -> str:
    """渲染创作Tab右侧面板：文字内容。图像预览已并入文字面板。"""
    return f"""
    <div class="cw-live-content">
        {text_html}
    </div>
    """
