"""排版和阅读器渲染：逐页阅读器、分镜对比、排版详情。"""
from __future__ import annotations

import html as html_mod
import os

from .common import _gradio_img_src
from .storyboard import _render_page_svg


def render_page_reader(
    final_pages: list[dict],
    current_idx: int = 0,
    show_bubbles: bool = False,
) -> str:
    """渲染逐页漫画阅读器。"""
    if not final_pages:
        return (
            '<div style="color:#64748b;padding:24px;text-align:center;">暂无漫画页面。<br>'
            '请先在「创作流程」Tab 运行工作流。</div>'
        )

    total = len(final_pages)
    idx = max(0, min(current_idx, total - 1))
    page = final_pages[idx]

    page_id = page.get("page_id", "?")
    page_num = page.get("page_number", idx + 1)
    image_path = page.get("image_path", "")
    width = page.get("width_px", 1240)
    height = page.get("height_px", 1754)
    bubbles = page.get("bubbles", [])

    img_html = ""
    if image_path and os.path.exists(str(image_path)):
        safe_path = _gradio_img_src(str(image_path))
        img_html = (
            f'<a href="/gradio_api/file={html_mod.escape(safe_path)}" target="_blank">'
            f'<img src="/gradio_api/file={html_mod.escape(safe_path)}" '
            f'style="max-width:100%;max-height:70vh;cursor:zoom-in;border-radius:8px;'
            f'box-shadow:0 4px 16px rgba(0,0,0,0.15);" '
            f'alt="第{page_num}页" />'
            f'</a>'
        )
    else:
        img_html = (
            f'<div style="width:{min(width, 600)}px;height:{min(height, 800)}px;'
            f'background:#f1f5f9;border-radius:8px;display:flex;align-items:center;'
            f'justify-content:center;color:#94a3b8;">'
            f'页面图像未生成</div>'
        )

    bubble_overlay = ""
    if show_bubbles and bubbles:
        bubble_divs = []
        for b in bubbles:
            bbox = b.get("bbox", {})
            bx = bbox.get("x", 0) if isinstance(bbox, dict) else getattr(bbox, "x", 0)
            by = bbox.get("y", 0) if isinstance(bbox, dict) else getattr(bbox, "y", 0)
            bw = bbox.get("width", 0) if isinstance(bbox, dict) else getattr(bbox, "width", 0)
            bh = bbox.get("height", 0) if isinstance(bbox, dict) else getattr(bbox, "height", 0)
            text = b.get("text", "")[:30] if isinstance(b, dict) else getattr(b, "text", "")[:30]
            btype = b.get("bubble_type", "speech") if isinstance(b, dict) else getattr(b, "bubble_type", "speech")

            type_colors = {
                "speech": "rgba(59,130,246,0.2)", "thought": "rgba(139,92,246,0.2)",
                "shout": "rgba(239,68,68,0.25)", "whisper": "rgba(168,85,247,0.2)",
                "narration": "rgba(245,158,11,0.2)",
            }
            bg = type_colors.get(btype, "rgba(59,130,246,0.15)")

            bubble_divs.append(
                f'<div class="cw-bubble-overlay" style="left:{bx*100}%;top:{by*100}%;'
                f'width:{bw*100}%;height:{bh*100}%;background:{bg};">'
                f'<span class="cw-bubble-text">{html_mod.escape(text)}</span>'
                f'</div>'
            )
        bubble_overlay = (
            f'<div style="position:absolute;top:0;left:0;width:100%;height:100%;'
            f'pointer-events:none;">{"".join(bubble_divs)}</div>'
        )

    return f"""
    <div class="cw-page-reader" style="text-align:center;">
        <div style="margin-bottom:12px;font-size:14px;color:#64748b;">
            第 <strong style="color:#1e293b;">{page_num}</strong> / {total} 页
            <span style="margin-left:8px;font-size:12px;color:#94a3b8;">({html_mod.escape(str(page_id))})</span>
        </div>
        <div style="position:relative;display:inline-block;max-width:100%;">
            {img_html}
            {bubble_overlay}
        </div>
        <div class="cw-page-nav" style="margin-top:16px;display:flex;justify-content:center;gap:16px;">
            <button class="cw-page-nav-btn" {"disabled" if idx == 0 else ""}
                    style="padding:8px 20px;border:1px solid #d1d5db;border-radius:6px;
                           background:white;cursor:pointer;font-size:14px;color:#374151;"
                    onclick="document.dispatchEvent(new CustomEvent('cw-page-nav', {{detail: 'prev'}}))">
                ◀ 上一页
            </button>
            <span style="padding:8px 12px;font-size:14px;color:#64748b;align-self:center;">
                {idx + 1} / {total}
            </span>
            <button class="cw-page-nav-btn" {"disabled" if idx >= total - 1 else ""}
                    style="padding:8px 20px;border:1px solid #d1d5db;border-radius:6px;
                           background:white;cursor:pointer;font-size:14px;color:#374151;"
                    onclick="document.dispatchEvent(new CustomEvent('cw-page-nav', {{detail: 'next'}}))">
                下一页 ▶
            </button>
        </div>
        <div style="margin-top:8px;font-size:11px;color:#94a3b8;">
            💡 点击图片可在新标签页中查看原图
        </div>
    </div>
    """


def render_comparison_view(plan: list[dict], final_pages: list[dict]) -> str:
    """渲染分镜规划 vs 最终成品的并排对比。"""
    if not plan and not final_pages:
        return (
            '<div style="color:#64748b;padding:12px;">暂无数据。<br>'
            '请先在「创作流程」Tab 运行工作流后查看。</div>'
        )

    final_by_id: dict[str, dict] = {}
    for fp in final_pages:
        pid = fp.get("page_id", "") if isinstance(fp, dict) else getattr(fp, "page_id", "")
        if pid:
            final_by_id[pid] = fp

    rows = []
    for page_data in plan:
        page_id = page_data.get("page_id", "") if isinstance(page_data, dict) else getattr(page_data, "page_id", "")
        page_num = (
            page_data.get("page_number", "?")
            if isinstance(page_data, dict)
            else getattr(page_data, "page_number", "?")
        )

        storyboard_svg = _render_page_svg(page_data)

        final_img = ""
        matched = final_by_id.get(page_id)
        if matched:
            path = matched.get("image_path", "") if isinstance(matched, dict) else getattr(matched, "image_path", "")
            if path and os.path.exists(str(path)):
                safe_path = _gradio_img_src(str(path))
                final_img = (
                    f'<img src="/gradio_api/file={html_mod.escape(safe_path)}" '
                    f'style="max-width:100%;max-height:400px;border-radius:6px;'
                    f'box-shadow:0 2px 8px rgba(0,0,0,0.1);" '
                    f'alt="第{page_num}页成品" />'
                )
            else:
                final_img = '<div style="color:#94a3b8;padding:40px;">尚未渲染</div>'
        else:
            final_img = '<div style="color:#94a3b8;padding:40px;">尚未渲染</div>'

        rows.append(f"""
        <div class="cw-comparison-row" style="display:flex;gap:16px;margin-bottom:16px;
                    background:white;border:1px solid #e2e8f0;border-radius:8px;padding:12px;">
            <div style="flex:1;min-width:0;">
                <div style="font-size:12px;color:#64748b;margin-bottom:6px;">📐 分镜规划 — 第{page_num}页</div>
                {storyboard_svg}
            </div>
            <div style="flex:1;min-width:0;">
                <div style="font-size:12px;color:#64748b;margin-bottom:6px;">🎨 最终成品 — 第{page_num}页</div>
                {final_img}
            </div>
        </div>
        """)

    return f"""
    <div class="cw-comparison-grid">
        <h3 style="margin:0 0 12px 0;color:#1e293b;">🔄 分镜 vs 成品对比</h3>
        {"".join(rows) if rows else '<div style="color:#64748b;padding:12px;">没有可对比的页面。</div>'}
    </div>
    """


def _render_detailed_final_pages(final_pages: list[dict], storyboard: list[dict]) -> str:
    """渲染完整排版结果：布局网格、对话气泡、最终图片。"""
    if not final_pages:
        return ""

    storyboard_by_page: dict[str, dict] = {}
    for p in storyboard:
        pid = p.get("page_id", "")
        if pid:
            storyboard_by_page[pid] = p

    items = ""
    for fp in final_pages:
        page_num = fp.get("page_number", "?")
        page_id = html_mod.escape(fp.get("page_id", "?"))
        w = fp.get("width_px", 0)
        h = fp.get("height_px", 0)

        sb = storyboard_by_page.get(fp.get("page_id", ""), {})
        panels_in_page = sb.get("panels", [])
        panel_count = len(panels_in_page)
        grid_desc = _infer_grid(panel_count)

        warnings = fp.get("layout_warnings", [])
        warn_html = ""
        if warnings:
            warn_items = "".join(
                f'<div style="color:#f59e0b;font-size:10px;">⚠ {html_mod.escape(str(wrn)[:100])}</div>'
                for wrn in warnings[:3]
            )
            warn_html = f'<div style="margin:4px 0;">{warn_items}</div>'

        bubbles = fp.get("bubbles", [])
        bubble_html = ""
        if bubbles:
            bubble_items = ""
            for b in bubbles[:8]:
                btype = html_mod.escape(b.get("bubble_type", "speech"))
                text = html_mod.escape((b.get("text") or "")[:60])
                font_size = b.get("font_size_pt", 12)
                occlusion = b.get("occlusion_score", 0)
                occ_color = "#10b981" if occlusion < 0.3 else ("#f59e0b" if occlusion < 0.7 else "#ef4444")
                bubble_items += (
                    f'<div style="font-size:10px;margin:1px 0;">'
                    f'<span class="cw-badge" style="font-size:9px;">{btype}</span>'
                    f' {text}'
                    f' <span style="color:{occ_color};">(遮挡:{occlusion:.2f})</span>'
                    f' <span style="color:#94a3b8;">字号:{font_size}pt</span>'
                    f'</div>'
                )
            bubble_html = f"""
            <details style="margin-top:4px;font-size:11px;">
                <summary style="cursor:pointer;color:#3b82f6;">💬 对话气泡 ({len(bubbles)}个)</summary>
                {bubble_items}
            </details>
            """

        img_path = fp.get("image_path", "")
        img_html = ""
        if img_path and os.path.exists(str(img_path)):
            safe_path = _gradio_img_src(str(img_path))
            img_html = (
                f'<a href="/gradio_api/file={html_mod.escape(safe_path)}" target="_blank">'
                f'<img src="/gradio_api/file={html_mod.escape(safe_path)}" '
                f'style="max-width:100%;max-height:200px;border:1px solid #e2e8f0;'
                f'border-radius:6px;margin:4px 0;cursor:zoom-in;" '
                f'alt="第{page_num}页" loading="lazy"/>'
                f'</a>'
            )

        items += f"""
        <div class="cw-phase-text-scene">
            <b>第 {page_num} 页</b> · {w}×{h}px · {grid_desc} ({panel_count}格)
            {img_html}
            {warn_html}
            {bubble_html}
        </div>
        """

    return f"""
    <div class="cw-phase-text-panel">
        <h4>📐 排版完成 ({len(final_pages)}页)</h4>
        <div class="cw-phase-text-scroll">
        {items}
        </div>
    </div>
    """


def _infer_grid(panel_count: int) -> str:
    """根据面板数量推断网格布局名称。"""
    mapping = {
        1: "1×1", 2: "1×2 或 2×1", 3: "1×3 或 L形",
        4: "2×2", 5: "2×3(少1)", 6: "2×3", 7: "2×4(少1)", 8: "2×4",
    }
    return mapping.get(panel_count, f"{panel_count}格")
