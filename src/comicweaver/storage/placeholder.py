"""占位图生成工具 - 用于本地 fallback 提供可视的占位图。

真实实现替换为 ImageAgent + diffusion 模型生成。
"""
from __future__ import annotations

import hashlib
import os

from PIL import Image, ImageDraw, ImageFont

from .paths import project_dir

_DEFAULT_FONT_SIZE = 28
_PALETTE = [
    (240, 200, 200), (200, 220, 240), (220, 240, 200),
    (240, 220, 200), (220, 200, 240), (200, 240, 220),
]


def _color_for(seed: str) -> tuple[int, int, int]:
    h = int(hashlib.md5(seed.encode("utf-8")).hexdigest(), 16)
    return _PALETTE[h % len(_PALETTE)]


def _safe_font(size: int) -> ImageFont.ImageFont:
    """Best-effort字体加载,失败则用默认。"""
    for path in (
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:  # noqa: BLE001
                continue
    return ImageFont.load_default()


def generate_character_placeholder(
    project_id: str,
    char_id: str,
    name: str,
    appearance: str,
    size: tuple[int, int] = (512, 768),
) -> str:
    """生成角色参考占位图,返回路径。"""
    out_dir = project_dir(project_id) / "characters" / char_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "base.png"

    color = _color_for(char_id)
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)

    # 简单占位人形
    cx, cy = size[0] // 2, size[1] // 2
    head_r = size[0] // 6
    draw.ellipse(
        (cx - head_r, cy - head_r * 3, cx + head_r, cy - head_r),
        fill=(255, 255, 255), outline=(80, 80, 80), width=3,
    )
    draw.rectangle(
        (cx - head_r, cy - head_r, cx + head_r, cy + head_r * 2),
        fill=(255, 255, 255), outline=(80, 80, 80), width=3,
    )

    font = _safe_font(_DEFAULT_FONT_SIZE)
    draw.text((20, 20), f"角色: {name}", fill=(40, 40, 40), font=font)
    small = _safe_font(18)
    _wrap_draw(draw, f"外观: {appearance}", (20, 60), size[0] - 40, small)
    _wrap_draw(draw, f"[Local 占位图 · {char_id}]", (20, size[1] - 40), size[0] - 40, small)

    img.save(path)
    return str(path)


def generate_panel_placeholder(
    project_id: str,
    panel_id: str,
    prompt: str,
    characters: list[str],
    shot_size: str = "medium",
    size: tuple[int, int] = (768, 1024),
) -> str:
    """生成漫画分镜占位图,返回路径。"""
    out_dir = project_dir(project_id) / "panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{panel_id}.png"

    color = _color_for(panel_id)
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)

    # 边框
    draw.rectangle(
        (4, 4, size[0] - 4, size[1] - 4),
        outline=(40, 40, 40), width=4,
    )

    # 角色简笔
    n_chars = max(1, len(characters))
    for i, char in enumerate(characters[:3]):
        cx = size[0] * (i + 1) // (n_chars + 1)
        cy = size[1] // 2
        head_r = size[0] // 10
        char_color = _color_for(char)
        draw.ellipse(
            (cx - head_r, cy - head_r * 2, cx + head_r, cy),
            fill=char_color, outline=(60, 60, 60), width=2,
        )
        f = _safe_font(20)
        draw.text((cx - head_r, cy + 5), char, fill=(40, 40, 40), font=f)

    # 元信息
    title_font = _safe_font(22)
    draw.text((20, 20), f"[{panel_id}]  shot={shot_size}",
              fill=(20, 20, 20), font=title_font)

    small = _safe_font(16)
    _wrap_draw(draw, prompt[:120], (20, 60), size[0] - 40, small)
    draw.text((20, size[1] - 30), "[Local 漫画分镜]",
              fill=(80, 80, 80), font=small)

    img.save(path)
    return str(path)



def _wrap_draw(
    draw: ImageDraw.ImageDraw,
    text: str,
    pos: tuple[int, int],
    max_width: int,
    font: ImageFont.ImageFont,
    max_lines: int = 4,
) -> None:
    """简单文本换行绘制。"""
    if not text:
        return
    x, y = pos
    line = ""
    lines: list[str] = []
    for ch in text:
        candidate = line + ch
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] > max_width:
            lines.append(line)
            line = ch
            if len(lines) >= max_lines:
                line = ""
                break
        else:
            line = candidate
    if line and len(lines) < max_lines:
        lines.append(line)

    line_h = font.size + 4 if hasattr(font, "size") else 18
    for i, ln in enumerate(lines):
        draw.text((x, y + i * line_h), ln, fill=(30, 30, 30), font=font)
