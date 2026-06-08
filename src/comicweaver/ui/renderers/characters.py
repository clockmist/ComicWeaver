"""角色档案渲染：角色卡片、详细特征。"""
from __future__ import annotations

import html as html_mod
import os

from .common import _gradio_img_src


def render_character_profiles(character_db: dict | None,
                               agent_outputs: dict[str, list[dict]] | None = None) -> str:
    """渲染角色档案卡片（含参考图）。"""
    if not character_db:
        return (
            '<div style="color:#64748b;padding:12px;">角色数据尚未生成。<br>'
            '请在「创作流程」Tab 运行工作流后查看。</div>'
        )

    characters = character_db.get("characters", {})
    if not characters:
        return '<div style="color:#64748b;padding:12px;">角色列表为空</div>'

    cards = []
    for char_id, profile in characters.items():
        name = html_mod.escape(profile.get("name", char_id))
        traits = profile.get("visual_traits", {})
        ref = profile.get("base_reference", {})

        trait_items = []
        for tk, tv in traits.items():
            if tv:
                trait_items.append(
                    f'<span class="cw-badge" style="margin:2px;">'
                    f'{html_mod.escape(tk)}: {html_mod.escape(str(tv))}</span>'
                )

        img_path = ref.get("image_path", "")
        img_html = ""
        if img_path and os.path.exists(str(img_path)):
            safe_path = _gradio_img_src(str(img_path))
            img_html = (
                f'<div style="text-align:center;margin:8px 0;">'
                f'<img src="/gradio_api/file={html_mod.escape(safe_path)}" '
                f'style="max-width:150px;max-height:200px;border:1px solid #e2e8f0;'
                f'border-radius:8px;" alt="{name} 参考图"/>'
                f'</div>'
            )

        meta_info = (
            f'<div style="font-size:11px;color:#94a3b8;">'
            f'来源: {ref.get("source", "?")} · '
            f'置信度: {ref.get("confidence", 0):.2f} · '
            f'锁定: {"是" if profile.get("locked") else "否"}'
            f'</div>'
        )

        cards.append(f"""
        <div class="cw-character-card" style="display:inline-block;width:220px;
                    margin:6px;vertical-align:top;padding:12px;">
            <h4 style="margin:0 0 4px 0;color:#1e293b;">{name}</h4>
            <div style="font-size:11px;color:#64748b;margin-bottom:8px;">ID: {html_mod.escape(char_id)}</div>
            {img_html}
            <div style="margin:6px 0;">{''.join(trait_items)}</div>
            {meta_info}
        </div>
        """)

    return f"""
    <div class="cw-card">
        <h3 style="margin:0 0 12px 0;color:#1e293b;">👤 角色档案 ({len(characters)})</h3>
        <div style="display:flex;flex-wrap:wrap;">{''.join(cards)}</div>
    </div>
    """


def _render_detailed_characters(character_db: dict) -> str:
    """渲染完整角色设计：外貌特征、标签、参考图。"""
    characters = character_db.get("characters", {})
    if not characters:
        return ""

    items = ""
    for cid, cp in characters.items():
        name = html_mod.escape(cp.get("name", cid))
        gender = html_mod.escape(cp.get("gender_tag", ""))
        appearance_prompt = html_mod.escape((cp.get("appearance_prompt") or "")[:200])
        core_tags = html_mod.escape((cp.get("core_tags") or "")[:200])
        seed = cp.get("seed", 0)

        traits = cp.get("visual_traits", {})
        hair = html_mod.escape(str(traits.get("hair", ""))[:40]) if traits.get("hair") else ""
        eyes = html_mod.escape(str(traits.get("eyes", ""))[:40]) if traits.get("eyes") else ""
        body = html_mod.escape(str(traits.get("body", ""))[:40]) if traits.get("body") else ""
        clothing = html_mod.escape(str(traits.get("clothing", ""))[:40]) if traits.get("clothing") else ""
        distinctive = traits.get("distinctive", [])
        distinctive_str = ", ".join(html_mod.escape(str(d)[:30]) for d in distinctive) if distinctive else ""

        trait_rows = ""
        if hair:
            trait_rows += f'<div><span class="key">发型:</span> <span class="val">{hair}</span></div>'
        if eyes:
            trait_rows += f'<div><span class="key">眼睛:</span> <span class="val">{eyes}</span></div>'
        if body:
            trait_rows += f'<div><span class="key">身材:</span> <span class="val">{body}</span></div>'
        if clothing:
            trait_rows += f'<div><span class="key">服装:</span> <span class="val">{clothing}</span></div>'
        if distinctive_str:
            trait_rows += f'<div><span class="key">特征:</span> <span class="val">{distinctive_str}</span></div>'

        ref = cp.get("base_reference", {})
        img_path = ref.get("image_path", "")
        img_html = ""
        if img_path and os.path.exists(str(img_path)):
            safe_path = _gradio_img_src(str(img_path))
            img_html = (
                f'<img src="/gradio_api/file={html_mod.escape(safe_path)}" '
                f'style="max-width:120px;max-height:160px;border:1px solid #e2e8f0;'
                f'border-radius:6px;margin:4px 0;" alt="{name} 参考图"/>'
            )

        items += f"""
        <div class="cw-phase-text-scene">
            <b>{name}</b>
            {f' <span class="cw-badge">{gender}</span>' if gender else ''}
            {f' <span class="cw-badge" style="background:#f1f5f9;color:#64748b;">种子: {seed}</span>' if seed else ''}
            {img_html}
            {f'<div style="font-size:11px;color:#475569;margin-top:4px;">{trait_rows}</div>' if trait_rows else ''}
            {f'<div style="font-size:10px;color:#94a3b8;margin-top:2px;">标签: {core_tags}</div>' if core_tags else ''}
            {f'<div style="font-size:10px;color:#94a3b8;">描述: {appearance_prompt}</div>' if appearance_prompt else ''}
        </div>
        """

    return f"""
    <div class="cw-phase-text-panel">
        <h4>👤 角色设计 ({len(characters)}人)</h4>
        <div class="cw-phase-text-scroll">
        {items}
        </div>
    </div>
    """
