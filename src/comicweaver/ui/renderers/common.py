"""通用渲染工具：路径转换、缩略图、字典预览、角色名解析。"""
from __future__ import annotations

import html as html_mod
import os


def _gradio_img_src(file_path: str) -> str:
    """将绝对/相对文件路径转为 Gradio 可用的 img src URL。

    自动添加文件修改时间作为 cache-busting 参数，
    确保重新生成后的图片不会被浏览器缓存。
    """
    fp = str(file_path)
    try:
        rel = os.path.relpath(fp, os.getcwd())
    except (ValueError, OSError):
        rel = fp

    safe = rel.replace("\\", "/")

    # 添加 cache-busting 参数：文件修改时间（毫秒精度）
    try:
        mtime_ms = int(os.path.getmtime(fp) * 1000)
        safe = f"{safe}?t={mtime_ms}"
    except (OSError, TypeError):
        pass

    return safe


def _preview_thumbnail(path: str, alt: str) -> str:
    """单个预览缩略图（120px 高，cover 裁剪）。"""
    if path and os.path.exists(str(path)):
        safe_path = _gradio_img_src(str(path))
        return (
            f'<img src="/gradio_api/file={html_mod.escape(safe_path)}" '
            f'style="width:100%;height:120px;object-fit:cover;border-radius:4px;" '
            f'alt="{html_mod.escape(alt)}" loading="lazy"/>'
        )
    return (
        f'<div style="width:100%;height:120px;background:#f1f5f9;border-radius:4px;'
        f'display:flex;align-items:center;justify-content:center;color:#94a3b8;font-size:10px;">'
        f'无图像</div>'
    )


def _dict_preview(d: dict, max_depth: int = 2, current_depth: int = 0) -> str:
    """递归渲染字典为缩进 HTML，键带颜色、值截断。"""
    if current_depth >= max_depth or not isinstance(d, dict):
        if isinstance(d, str) and len(d) > 200:
            return html_mod.escape(d[:200] + "...")
        return html_mod.escape(str(d)[:200])

    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(
                f'<div style="margin-left:{current_depth*16}px;">'
                f'<b style="color:#1e40af;">{html_mod.escape(str(k))}:</b> '
                f'<span style="color:#64748b;">{{... {len(v)} keys}}</span>'
                f'{_dict_preview(v, max_depth, current_depth + 1)}'
                f'</div>'
            )
        elif isinstance(v, list):
            if len(v) == 0:
                lines.append(f'<b style="color:#1e40af;">{html_mod.escape(str(k))}:</b> []')
            elif isinstance(v[0], dict):
                items = "".join(
                    f'<details style="margin-left:8px;"><summary style="cursor:pointer;color:#3b82f6;">'
                    f'[{i}]</summary>{_dict_preview(item, max_depth, current_depth + 1)}</details>'
                    for i, item in enumerate(v[:15])
                )
                more = f' ... +{len(v) - 15} more' if len(v) > 15 else ""
                lines.append(
                    f'<div style="margin-left:{current_depth*16}px;">'
                    f'<b style="color:#1e40af;">{html_mod.escape(str(k))}:</b> [{len(v)} items]{more}'
                    f'{items}</div>'
                )
            else:
                preview = ", ".join(html_mod.escape(str(x)) for x in v[:10])
                more = f" ... +{len(v) - 10}" if len(v) > 10 else ""
                lines.append(
                    f'<div style="margin-left:{current_depth*16}px;">'
                    f'<b style="color:#1e40af;">{html_mod.escape(str(k))}:</b> '
                    f'[{preview}{more}]</div>'
                )
        elif isinstance(v, str):
            display = v[:100] + "..." if len(v) > 100 else v
            lines.append(
                f'<div style="margin-left:{current_depth*16}px;">'
                f'<b style="color:#1e40af;">{html_mod.escape(str(k))}:</b> '
                f'<span style="color:#334155;">{html_mod.escape(display)}</span></div>'
            )
        else:
            lines.append(
                f'<div style="margin-left:{current_depth*16}px;">'
                f'<b style="color:#1e40af;">{html_mod.escape(str(k))}:</b> '
                f'<span style="color:#0f172a;">{html_mod.escape(str(v))}</span></div>'
            )
    return "\n".join(lines)


def _resolve_char_name(char_id: str, char_map: dict[str, str]) -> str:
    """将角色ID解析为显示名称。"""
    return char_map.get(char_id, char_id)
