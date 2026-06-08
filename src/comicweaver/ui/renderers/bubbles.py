"""台词气泡渲染 — bug 集中修复区。

关键改进：_normalize_bubble() 统一将 dict 和 Pydantic 对象转为标准 dict，
后续所有处理只使用 dict，消灭所有 isinstance(b, dict) 判断。
"""
from __future__ import annotations

import html as html_mod


def _normalize_bubble(b) -> dict:
    """将 dict 或 BubblePlacement 对象统一转为标准 dict。

    这是修复气泡数据不一致问题的核心函数。
    所有气泡处理前都必须经过此函数归一化。
    """
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
    # Pydantic / dataclass 对象
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


def _render_bubble_summary(bubble_placements: dict, panel_images: list[dict] | None = None) -> str:
    """渲染台词气泡放置摘要（含对应面板缩略图）。

    参照 _render_detailed_images / _render_detailed_characters 的模式，
    每条气泡条目左侧展示匹配的面板缩略图。
    """
    if panel_images is None:
        panel_images = []

    # 构建 panel_id → image_path 映射
    panel_img_map: dict[str, str] = {}
    for pi in panel_images:
        pid = pi.get("panel_id", "") if isinstance(pi, dict) else getattr(pi, "panel_id", "")
        path = pi.get("image_path", "") if isinstance(pi, dict) else getattr(pi, "image_path", "")
        if pid and path:
            panel_img_map[pid] = str(path)

    total = 0
    rows: list[str] = []
    for page_id, bubbles in bubble_placements.items():
        page_num = page_id.replace("page_", "").lstrip("0") or "?"
        for b_raw in bubbles:
            b = _normalize_bubble(b_raw)
            total += 1
            panel_id = b["panel_id"]
            bubble_type = b["bubble_type"]
            speaker = b["speaker"]
            text = b["text"]
            font_size = b["font_size_pt"]
            tail = b["tail_direction"]
            face_count = b["face_count"]

            # 匹配面板缩略图（参照 _render_detailed_images 的内联图片方式）
            img_html = ""
            if panel_id in panel_img_map:
                import os as _os
                img_path = panel_img_map[panel_id]
                if _os.path.exists(img_path):
                    from .common import _gradio_img_src
                    safe = _gradio_img_src(img_path)
                    img_html = (
                        f'<img src="/gradio_api/file={html_mod.escape(safe)}" '
                        f'style="max-width:120px;max-height:90px;border:1px solid #e2e8f0;'
                        f'border-radius:4px;margin:2px 0;display:block;" '
                        f'alt="{html_mod.escape(panel_id)}" loading="lazy"/>'
                    )

            speaker_label = f"<strong>{html_mod.escape(speaker)}:</strong> " if speaker else ""
            type_color = {
                "speech": "#3b82f6", "thought": "#8b5cf6", "shout": "#ef4444",
                "whisper": "#6b7280", "narration": "#92400e",
            }.get(bubble_type, "#3b82f6")

            rows.append(f"""
            <div class="cw-phase-text-scene" style="display:flex;gap:8px;align-items:flex-start;">
                {f'<div style="flex-shrink:0;">{img_html}</div>' if img_html else ''}
                <div style="flex:1;min-width:0;">
                    <span class="cw-badge" style="border-color:{type_color};color:{type_color};">{html_mod.escape(bubble_type)}</span>
                    {speaker_label}
                    "<span style="color:#334155;">{html_mod.escape(str(text)[:80])}{'…' if len(str(text))>80 else ''}</span>"
                    <div style="font-size:10px;color:#94a3b8;margin-top:2px;">
                        {html_mod.escape(panel_id)} · 字号{font_size}pt · 尾部{html_mod.escape(tail)}
                        {" · 人脸" + str(face_count) if face_count else ""}
                    </div>
                </div>
            </div>
            """)

    return f"""
    <div class="cw-phase-text-panel">
        <h4>💬 台词气泡放置 ({total} 个气泡)</h4>
        <div class="cw-phase-text-scroll">
        {"".join(rows) if rows else '<div class="cw-phase-text-empty">无对话气泡</div>'}
        </div>
    </div>
    """


def render_bubble_preview(bubble_placements: dict, panel_images: list[dict] | None = None) -> str:
    """渲染台词气泡实时预览区。

    当 panel_images 提供时，匹配显示对应面板图像缩略图；
    否则仅展示气泡文本列表。
    返回 HTML 片段，供 composables/live.py 组合使用。
    """
    if not bubble_placements:
        return ""

    if panel_images is None:
        panel_images = []

    # 构建 panel_id → image_path 映射
    panel_img_map: dict[str, str] = {}
    for pi in panel_images:
        pid = pi.get("panel_id", "") if isinstance(pi, dict) else getattr(pi, "panel_id", "")
        path = pi.get("image_path", "") if isinstance(pi, dict) else getattr(pi, "image_path", "")
        if pid and path:
            panel_img_map[pid] = str(path)

    total_bubbles = sum(len(v) for v in bubble_placements.values())
    bubble_items = []
    for page_id, bubbles in bubble_placements.items():
        page_num = page_id.replace("page_", "").lstrip("0") or "?"
        for b_raw in bubbles:
            b = _normalize_bubble(b_raw)
            pid = b["panel_id"]
            btype = b["bubble_type"]
            speaker = b["speaker"]
            text = b["text"]
            pos = b["tail_direction"]
            font_size = b["font_size_pt"]

            type_icon = {
                "speech": "💬", "thought": "☁️", "shout": "📢",
                "whisper": "🤫", "narration": "📋",
            }.get(btype, "💬")
            type_color = {
                "speech": "#3b82f6", "thought": "#8b5cf6", "shout": "#ef4444",
                "whisper": "#6b7280", "narration": "#92400e",
            }.get(btype, "#3b82f6")

            # 匹配面板缩略图
            img_tag = ""
            if pid in panel_img_map:
                import os as _os
                img_path = panel_img_map[pid]
                if _os.path.exists(img_path):
                    from .common import _gradio_img_src
                    safe = _gradio_img_src(img_path)
                    img_tag = (
                        f'<img src="/gradio_api/file={html_mod.escape(safe)}" '
                        f'style="width:80px;height:60px;object-fit:cover;border-radius:4px;'
                        f'margin-right:8px;border:1px solid #e2e8f0;flex-shrink:0;" '
                        f'alt="{html_mod.escape(pid)}" loading="lazy"/>'
                    )

            speaker_label = f"<b>{html_mod.escape(str(speaker))}:</b> " if speaker else ""
            bubble_items.append(f"""
            <div style="display:flex;align-items:flex-start;gap:6px;padding:6px 8px;
                        margin:2px 0;background:#f8fafc;border-left:3px solid {type_color};
                        border-radius:0 4px 4px 0;font-size:12px;">
                {img_tag}
                <div style="flex:1;min-width:0;">
                    <span style="font-size:14px;">{type_icon}</span>
                    <div style="color:#1e293b;">
                        {speaker_label}<span style="color:#334155;">"{html_mod.escape(str(text)[:60])}{'…' if len(str(text))>60 else ''}"</span>
                    </div>
                    <div style="color:#94a3b8;font-size:10px;margin-top:2px;">
                        {html_mod.escape(pid)} · 位置:{html_mod.escape(pos)} · {font_size}pt
                    </div>
                </div>
            </div>
            """)

    return f"""
    <div style="margin-bottom:8px;padding:8px;background:#fffbeb;border:1px solid #fcd34d;
                border-radius:8px;">
        <div style="font-size:13px;font-weight:700;color:#92400e;margin-bottom:6px;">
            💬 台词气泡放置结果 ({total_bubbles}个)
        </div>
        <div style="max-height:360px;overflow-y:auto;">
            {"".join(bubble_items)}
        </div>
    </div>
    """
