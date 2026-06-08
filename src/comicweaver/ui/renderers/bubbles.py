"""台词气泡渲染 — 展示带气泡的最终面板预览图。

v2.1: 优先使用 BubbleAgent 生成的带气泡实际渲染图（bubbled_panel_images），
      仅在渲染图不可用时回退到 CSS 模拟叠加。
"""
from __future__ import annotations

import html as html_mod
import os as _os


def _normalize_bubble(b) -> dict:
    """将 dict 或 BubblePlacement 对象统一转为标准 dict。"""
    if isinstance(b, dict):
        return {
            "panel_id": b.get("panel_id", "?"),
            "bubble_type": b.get("bubble_type", "speech"),
            "speaker": b.get("speaker", ""),
            "text": b.get("text", ""),
            "font_size_pt": b.get("font_size_pt", 14),
            "tail_direction": b.get("tail_direction", "auto"),
            "face_count": b.get("face_count", 0),
            "bbox": b.get("bbox", {}),
            "occlusion_score": b.get("occlusion_score", 0),
        }
    return {
        "panel_id": getattr(b, "panel_id", "?"),
        "bubble_type": getattr(b, "bubble_type", "speech"),
        "speaker": getattr(b, "speaker", ""),
        "text": getattr(b, "text", ""),
        "font_size_pt": getattr(b, "font_size_pt", 14),
        "tail_direction": getattr(b, "tail_direction", "auto"),
        "face_count": getattr(b, "face_count", 0),
        "bbox": getattr(b, "bbox", {}),
        "occlusion_score": getattr(b, "occlusion_score", 0),
    }


def _render_bubble_summary(
    bubble_placements: dict,
    panel_images: list[dict] | None = None,
    bubbled_images: dict[str, str] | None = None,
) -> str:
    """渲染台词气泡放置摘要 — 优先展示实际带气泡的面板预览图。

    如果 BubbleAgent 已生成带气泡的预览图（bubbled_panel_images），
    则直接展示这些渲染图；否则回退到 CSS 叠加模拟。

    与 layout 输出的区别：这里展示的单张面板预览图，layout 将多张
    面板拼合到一页中。
    """
    if panel_images is None:
        panel_images = []

    if bubbled_images is None:
        bubbled_images = {}

    from .common import _gradio_img_src

    # 按 panel_id 分组所有气泡元信息
    panel_bubbles_meta: dict[str, list[dict]] = {}
    for page_id, bubbles in bubble_placements.items():
        for b_raw in bubbles:
            b = _normalize_bubble(b_raw)
            panel_bubbles_meta.setdefault(b["panel_id"], []).append(b)

    total = sum(len(v) for v in panel_bubbles_meta.values())
    if total == 0:
        return """<div class="cw-phase-text-panel">
            <h4>💬 台词气泡放置</h4>
            <div class="cw-phase-text-empty">无对话气泡</div>
        </div>"""

    panels_html: list[str] = []

    # 收集所有 panel_id（从气泡数据 + 面板图像数据）
    all_panel_ids: set[str] = set(panel_bubbles_meta.keys())
    for pi in panel_images:
        pid = pi.get("panel_id", "") if isinstance(pi, dict) else getattr(pi, "panel_id", "")
        if pid:
            all_panel_ids.add(pid)

    for pid in sorted(all_panel_ids):
        bubbles = panel_bubbles_meta.get(pid, [])
        bubbled_path = bubbled_images.get(pid, "")

        has_bubbled = bool(bubbled_path and _os.path.exists(bubbled_path))

        if has_bubbled:
            # ★ 优先：使用实际渲染的带气泡面板图
            safe_path = _gradio_img_src(bubbled_path)
            img_html = (
                f'<img src="/gradio_api/file={html_mod.escape(safe_path)}" '
                f'style="display:block; max-width:320px; max-height:240px; '
                f'width:auto; height:auto; border-radius:4px;" '
                f'alt="{html_mod.escape(pid)}" loading="lazy"/>'
            )
        else:
            # 回退：查找原始面板图像
            img_path = ""
            for pi in panel_images:
                p = pi.get("panel_id", "") if isinstance(pi, dict) else getattr(pi, "panel_id", "")
                if p == pid:
                    img_path = pi.get("image_path", "") if isinstance(pi, dict) else getattr(pi, "image_path", "")
                    break

            if img_path and _os.path.exists(img_path):
                safe_path = _gradio_img_src(img_path)
                img_html = (
                    f'<img src="/gradio_api/file={html_mod.escape(safe_path)}" '
                    f'style="display:block; max-width:320px; max-height:240px; '
                    f'width:auto; height:auto; border-radius:4px; opacity:0.5;" '
                    f'alt="{html_mod.escape(pid)}" loading="lazy"/>'
                )
                # 添加提示
                img_html += (
                    f'<div style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); '
                    f'color:#ef4444; font-size:12px; font-weight:600; background:rgba(255,255,255,0.8); '
                    f'padding:4px 8px; border-radius:4px;">等待气泡渲染...</div>'
                )
            else:
                img_html = ""

        # 气泡信息摘要
        bubble_info_parts = []
        for b in bubbles:
            btype = b["bubble_type"]
            speaker = b["speaker"]
            text = b["text"]
            short = text[:20] + "…" if len(text) > 20 else text
            icon = {"speech": "💬", "thought": "☁️", "shout": "📢", "whisper": "🤫", "narration": "📋"}.get(btype, "💬")
            speaker_label = f"{speaker}:" if speaker else ""
            bubble_info_parts.append(f"{icon} {speaker_label}{short}")

        bubble_list_html = "<br>".join(
            f'<span style="font-size:10px; color:#475569;">{html_mod.escape(info)}</span>'
            for info in bubble_info_parts[:6]
        )
        if len(bubble_info_parts) > 6:
            bubble_list_html += f'<br><span style="font-size:10px; color:#94a3b8;">... +{len(bubble_info_parts) - 6} 条</span>'

        panels_html.append(f"""
        <div style="display: inline-block; position: relative; margin: 6px;
                    border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden;
                    background: #f8fafc; vertical-align: top;">
            {img_html}
            <div style="padding:4px 8px; font-size:10px; color:#64748b;
                        background: rgba(255,255,255,0.9); border-top:1px solid #e2e8f0;
                        text-align:left;">
                <div style="font-weight:600; margin-bottom:2px; color:#1e293b;">
                    {html_mod.escape(pid)} · {len(bubbles)}个气泡
                </div>
                {bubble_list_html}
            </div>
        </div>""")

    return f"""
    <div class="cw-phase-text-panel">
        <h4>💬 台词气泡放置 ({total} 个气泡，{len(panel_bubbles_meta)} 个面板)</h4>
        <div style="color:#64748b; font-size:11px; margin-bottom:4px;">
            气泡已直接绘制在面板图像上 — 与最终排版效果一致（排版阶段会将多张面板拼合为一页）
        </div>
        <div class="cw-phase-text-scroll" style="text-align:center;">
            {"".join(panels_html)}
        </div>
    </div>"""


def render_bubble_preview(
    bubble_placements: dict,
    panel_images: list[dict] | None = None,
    bubbled_images: dict[str, str] | None = None,
) -> str:
    """渲染台词气泡实时预览区 — 优先展示实际带气泡的面板图。"""
    if not bubble_placements:
        return ""

    if panel_images is None:
        panel_images = []
    if bubbled_images is None:
        bubbled_images = {}

    from .common import _gradio_img_src

    panel_bubbles_meta: dict[str, list[dict]] = {}
    for page_id, bubbles in bubble_placements.items():
        for b_raw in bubbles:
            b = _normalize_bubble(b_raw)
            panel_bubbles_meta.setdefault(b["panel_id"], []).append(b)

    total_bubbles = sum(len(v) for v in panel_bubbles_meta.values())

    panels_html: list[str] = []
    for pid in sorted(panel_bubbles_meta.keys()):
        bubbles = panel_bubbles_meta[pid]
        bubbled_path = bubbled_images.get(pid, "")
        has_bubbled = bool(bubbled_path and _os.path.exists(bubbled_path))

        if has_bubbled:
            safe_path = _gradio_img_src(bubbled_path)
            img_html = (
                f'<img src="/gradio_api/file={html_mod.escape(safe_path)}" '
                f'style="display:block; max-width:200px; max-height:150px; '
                f'width:auto; height:auto; border-radius:4px;" '
                f'alt="{html_mod.escape(pid)}" loading="lazy"/>'
            )
        else:
            img_html = (
                f'<div style="width:160px;height:120px;background:#f1f5f9;'
                f'display:flex;align-items:center;justify-content:center;color:#94a3b8;'
                f'font-size:10px;">等待气泡渲染...</div>'
            )

        speaker_info = ", ".join(
            f"{b['speaker']}: {b['text'][:15]}…" if b['speaker'] else b['text'][:20]
            for b in bubbles[:3]
        )

        panels_html.append(f"""
        <div style="display: inline-block; position: relative; margin: 4px;
                    border: 1px solid #e2e8f0; border-radius: 6px; overflow: hidden;
                    background: #f8fafc; vertical-align: top;">
            {img_html}
            <div style="padding:2px 6px; font-size:9px; color:#94a3b8; text-align:left;
                        max-width:200px;">
                <b>{html_mod.escape(pid)}</b> · {len(bubbles)}气泡 ·
                {html_mod.escape(speaker_info[:60])}
            </div>
        </div>""")

    return f"""
    <div style="margin-bottom:8px;padding:8px;background:#fffbeb;border:1px solid #fcd34d;
                border-radius:8px;">
        <div style="font-size:13px;font-weight:700;color:#92400e;margin-bottom:6px;">
            💬 台词气泡放置结果 ({total_bubbles}个)
        </div>
        <div style="max-height:360px;overflow-y:auto;text-align:center;">
            {"".join(panels_html) if panels_html else '<div style="color:#94a3b8;font-size:11px;">等待面板图像...</div>'}
        </div>
    </div>"""
