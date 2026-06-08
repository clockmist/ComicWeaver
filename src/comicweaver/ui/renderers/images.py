"""面板图像渲染：图像画廊、生成参数详情。"""
from __future__ import annotations

import html as html_mod
import os

from .common import _gradio_img_src


def render_panel_images_gallery(panel_images: list[dict]) -> str:
    """渲染单张面板图像画廊（非最终合成页）。"""
    if not panel_images:
        return (
            '<div style="color:#64748b;padding:12px;">面板图像尚未生成。<br>'
            '请在「创作流程」Tab 运行工作流后查看。</div>'
        )

    items = []
    for pi in panel_images:
        pid = pi.get("panel_id", "?")
        path = pi.get("image_path", "")
        backend = pi.get("backend", "?")
        gen_time = pi.get("generation_time_ms", 0)
        prompt = pi.get("prompt_used", "")[:80] or ""

        img_html = ""
        if path and os.path.exists(str(path)):
            safe_path = _gradio_img_src(str(path))
            img_html = (
                f'<img src="/gradio_api/file={html_mod.escape(safe_path)}" '
                f'style="width:100%;height:180px;object-fit:cover;border-radius:6px;" '
                f'alt="{html_mod.escape(pid)}" loading="lazy"/>'
            )
        else:
            img_html = (
                f'<div style="width:100%;height:180px;background:#f1f5f9;border-radius:6px;'
                f'display:flex;align-items:center;justify-content:center;color:#94a3b8;font-size:12px;">'
                f'无图像</div>'
            )

        items.append(f"""
        <div style="display:inline-block;width:200px;margin:6px;vertical-align:top;
                    background:white;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;">
            {img_html}
            <div style="padding:8px;">
                <div style="font-size:12px;font-weight:600;color:#1e293b;"
                     title="{html_mod.escape(pid)}">{html_mod.escape(pid)}</div>
                <div style="font-size:11px;color:#64748b;">后端: {html_mod.escape(backend)} · {gen_time}ms</div>
                <div style="font-size:10px;color:#94a3b8;margin-top:2px;overflow:hidden;
                            text-overflow:ellipsis;white-space:nowrap;"
                     title="{html_mod.escape(prompt)}">{html_mod.escape(prompt)}</div>
            </div>
        </div>
        """)

    return f"""
    <div class="cw-card">
        <h3 style="margin:0 0 12px 0;color:#1e293b;">🎨 面板图像 ({len(panel_images)})</h3>
        <div style="white-space:nowrap;overflow-x:auto;padding:4px;">{''.join(items)}</div>
    </div>
    """


def _render_detailed_images(panel_images: list[dict]) -> str:
    """渲染每张面板图像的生成详情：提示词、种子、参数。"""
    if not panel_images:
        return ""

    backends: dict[str, int] = {}
    total_time = 0
    for pi in panel_images:
        backend = pi.get("backend", "?")
        backends[backend] = backends.get(backend, 0) + 1
        total_time += pi.get("generation_time_ms", 0)

    backend_str = " · ".join(f"{k}: {v}张" for k, v in backends.items())
    avg_time = total_time / len(panel_images) if panel_images else 0
    time_str = f"{avg_time/1000:.1f}s" if avg_time >= 1000 else f"{avg_time:.0f}ms"

    panel_items = ""
    for pi in panel_images:
        pid = html_mod.escape(pi.get("panel_id", "?"))
        backend = html_mod.escape(pi.get("backend", "?"))
        seed = pi.get("seed", 0)
        steps = pi.get("steps", 0)
        cfg = pi.get("cfg_scale", 0)
        gen_time = pi.get("generation_time_ms", 0)
        gen_str = f"{gen_time/1000:.1f}s" if gen_time >= 1000 else f"{gen_time:.0f}ms"
        prompt = html_mod.escape((pi.get("prompt_used") or "")[:300])
        neg_prompt = html_mod.escape((pi.get("negative_prompt_used") or "")[:200])
        chars = pi.get("characters_present", [])
        char_str = ", ".join(html_mod.escape(str(c)) for c in chars) if chars else "无"

        img_path = pi.get("image_path", "")
        img_html = ""
        if img_path and os.path.exists(str(img_path)):
            safe_path = _gradio_img_src(str(img_path))
            img_html = (
                f'<img src="/gradio_api/file={html_mod.escape(safe_path)}" '
                f'style="max-width:100%;max-height:160px;border:1px solid #e2e8f0;'
                f'border-radius:4px;margin:4px 0;" alt="{pid}" loading="lazy"/>'
            )

        panel_items += f"""
        <details style="margin:2px 0;font-size:11px;">
            <summary style="cursor:pointer;color:#3b82f6;font-weight:500;">
                {pid} · {backend} · ⏱{gen_str} · 🌱{seed}
            </summary>
            <div style="padding:4px 8px;background:#f8fafc;border-radius:4px;">
                {img_html}
                <div><span class="key">角色:</span> {char_str}</div>
                <div><span class="key">参数:</span> steps={steps} cfg={cfg}</div>
                <div style="font-size:10px;color:#64748b;max-height:60px;overflow-y:auto;">
                    <span class="key">正向提示词:</span> {prompt}
                </div>
                {f'<div style="font-size:10px;color:#94a3b8;max-height:40px;overflow-y:auto;"><span class="key">负向:</span> {neg_prompt}</div>' if neg_prompt else ''}
            </div>
        </details>
        """

    return f"""
    <div class="cw-phase-text-panel">
        <h4>🎨 图像生成 ({len(panel_images)}张面板)</h4>
        <div class="cw-phase-text-scroll">
        <div class="cw-phase-text-item">
            <span class="key">后端分布:</span> <span class="val">{backend_str}</span>
            &nbsp;&nbsp;<span class="key">平均耗时:</span> <span class="val">{time_str}/张</span>
        </div>
        <div style="margin-top:4px;">
            {panel_items}
        </div>
        </div>
    </div>
    """
