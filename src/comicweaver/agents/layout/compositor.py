"""Page image compositor.

Assembles a final comic page from individual panel images, placing them at
their solved bbox positions and rendering dialogue bubbles on top.

This replaces the old ``storage/placeholder.py::generate_page_placeholder()``
which is being removed from the storage layer.
"""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from comicweaver.core.schema import BoundingBox, PanelImage

    from .bubbles import BubblePlacement

# ---------------------------------------------------------------------------
# Fit strategy
# ---------------------------------------------------------------------------


class FitStrategy(str, Enum):
    """How to fit a panel image into its target bbox."""
    COVER = "cover"          # Scale to cover, center-crop.  No distortion.  **Default.**
    SMART_COVER = "smart_cover"  # Cover with content-aware anchor (faces up, action center).
    CONTAIN = "contain"      # Scale to fit, pad with background.  No crop, no distortion.
    STRETCH = "stretch"      # Direct resize.  May distort.  Only for <5% ratio mismatch.


@dataclass
class PanelSlot:
    """Bundles a panel's identity, bbox, and fit metadata for the compositor."""
    panel_id: str
    bbox: "BoundingBox"
    shot_size: str = "medium"        # ShotSize value → drives fit strategy
    emotion_intensity: float = 0.5   # High emotion → prefer smart_crop


def _choose_strategy(shot_size: str, emotion: float) -> FitStrategy:
    """Select the best fit strategy based on the panel's shot type and emotion.

    Rationale:
    - close / extreme_close → faces are critical; use smart_crop with top anchor
    - extreme_long / long → establishing shots; don't crop, use contain
    - medium / full → default cover is fine
    - high emotion → face matters more; prefer smart_crop
    """
    if shot_size in ("extreme_long", "long"):
        return FitStrategy.CONTAIN
    if shot_size in ("close", "extreme_close") or emotion >= 0.85:
        return FitStrategy.SMART_COVER
    return FitStrategy.COVER


# ---------------------------------------------------------------------------
# Colour utilities
# ---------------------------------------------------------------------------

_PALETTE = [
    (240, 200, 200), (200, 220, 240), (220, 240, 200),
    (240, 220, 200), (220, 200, 240), (200, 240, 220),
]


def _color_for(seed: str) -> tuple[int, int, int]:
    h = int(hashlib.md5(seed.encode("utf-8")).hexdigest(), 16)
    return _PALETTE[h % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Font loading
# ---------------------------------------------------------------------------

_DEFAULT_FONT_SIZE = 28


def _safe_font(size: int) -> ImageFont.ImageFont:
    """Best-effort CJK font loading; falls back to PIL default."""
    for path in (
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    font: ImageFont.ImageFont,
) -> list[str]:
    """Wrap CJK text to fit within *max_width* pixels.

    Characters are broken individually (CJK has no word boundaries).
    """
    if not text:
        return []
    lines: list[str] = []
    line = ""
    for ch in text:
        candidate = line + ch
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] > max_width and line:
            lines.append(line)
            line = ch
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# Image fitting  (the core fix for aspect-ratio distortion)
# ---------------------------------------------------------------------------

def _fit_image(
    img: Image.Image,
    target_w: int,
    target_h: int,
    strategy: FitStrategy,
) -> Image.Image:
    """Fit *img* into *target_w*×*target_h* without distortion.

    - COVER:        scale so the image fully covers the target, then center-crop
    - SMART_COVER:  like cover, but anchors toward the top-center (faces area)
    - CONTAIN:      scale so the entire image fits, pad with transparent margin
    - STRETCH:      direct resize (only for near-identical ratios)
    """
    src_w, src_h = img.size
    if src_w <= 0 or src_h <= 0 or target_w <= 0 or target_h <= 0:
        return img

    src_ratio = src_w / src_h
    tgt_ratio = target_w / target_h

    # If aspect ratios are close, prefer a small resize over cropping content.
    # Precomputed generation sizes are latent-aligned, so a few percent of
    # aspect drift can happen after rounding.
    ratio_diff = abs(src_ratio - tgt_ratio) / max(src_ratio, tgt_ratio)
    if strategy == FitStrategy.STRETCH or ratio_diff < 0.08:
        return img.resize((target_w, target_h), Image.LANCZOS)

    if strategy == FitStrategy.CONTAIN:
        # Scale to fit INSIDE the target, maintaining aspect ratio
        if src_ratio > tgt_ratio:
            # Image is wider than target → fit by width
            new_w = target_w
            new_h = int(target_w / src_ratio)
        else:
            # Image is taller than target → fit by height
            new_h = target_h
            new_w = int(target_h * src_ratio)
        scaled = img.resize((new_w, new_h), Image.LANCZOS)
        # Pad to target size (centered)
        canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
        px = (target_w - new_w) // 2
        py = (target_h - new_h) // 2
        canvas.paste(scaled, (px, py))
        return canvas

    # COVER / SMART_COVER: scale to cover the target, then crop
    if src_ratio > tgt_ratio:
        # Image is wider → scale by height, crop width
        scale = target_h / src_h
    else:
        # Image is taller → scale by width, crop height
        scale = target_w / src_w

    scaled_w = int(src_w * scale)
    scaled_h = int(src_h * scale)
    scaled = img.resize((scaled_w, scaled_h), Image.LANCZOS)

    # Crop to target size
    if strategy == FitStrategy.SMART_COVER:
        # Anchor toward top-center (faces are usually in the upper 40-60%)
        # Crop from the top-third, biased upward
        crop_x = (scaled_w - target_w) // 2   # horizontal: center
        crop_y = int((scaled_h - target_h) * 0.25)  # vertical: 25% from top
    else:
        # Center crop
        crop_x = (scaled_w - target_w) // 2
        crop_y = (scaled_h - target_h) // 2

    # Clamp crop coordinates
    crop_x = max(0, min(crop_x, scaled_w - target_w))
    crop_y = max(0, min(crop_y, scaled_h - target_h))

    return scaled.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))


def _load_panel_image(img_path: str) -> Image.Image | None:
    """Load a panel image from disk.  Returns *None* on failure."""
    if not os.path.exists(img_path):
        return None
    try:
        return Image.open(img_path)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main compositor entry-point
# ---------------------------------------------------------------------------


def compose_page(
    project_id: str,
    page_id: str,
    page_number: int,
    panel_slots: list[PanelSlot],
    panel_images: dict[str, "PanelImage"],
    bubbles: list["BubblePlacement"],
    *,
    output_dir: str | None = None,
    width_px: int = 1240,
    height_px: int = 1754,
    margin_px: int = 40,
    font_size_pt: int = 14,
    border_color: tuple[int, int, int] = (20, 20, 20),
    border_width: int = 3,
    bg_color: tuple[int, int, int] = (250, 250, 250),
) -> str:
    """Render a single comic page and save to disk.

    Parameters
    ----------
    project_id:
        Used to construct the output path.
    page_id:
        Identifier for the page (used in filename).
    page_number:
        Displayed page number.
    panel_slots:
        Ordered list of ``PanelSlot`` objects (id + bbox + fit metadata).
    panel_images:
        Panel images keyed by panel_id (from ImageAgent).
    bubbles:
        Pre-computed bubble placements for this page.
    output_dir:
        Where to write the PNG.  Defaults to ``projects/<id>/pages/``.
    width_px / height_px:
        Canvas dimensions.
    margin_px:
        Page margin in pixels.
    border_color / border_width / bg_color:
        Visual settings.

    Returns
    -------
    str
        Absolute path to the saved PNG file.
    """
    # Output path
    if output_dir is None:
        from comicweaver.storage.paths import project_dir
        out_dir = project_dir(project_id) / "pages"
    else:
        out_dir = os.path.abspath(output_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(str(out_dir), f"{page_id}.png")

    # Canvas
    page = Image.new("RGB", (width_px, height_px), bg_color)
    draw = ImageDraw.Draw(page)

    inner_w = width_px - margin_px * 2
    inner_h = height_px - margin_px * 2

    # --- Paste panel images ---
    for slot in panel_slots:
        pi = panel_images.get(slot.panel_id)
        if pi is None:
            continue

        px = margin_px + int(inner_w * slot.bbox.x)
        py = margin_px + int(inner_h * slot.bbox.y)
        pw = int(inner_w * slot.bbox.width)
        ph = int(inner_h * slot.bbox.height)

        if pw < 10 or ph < 10:
            continue

        src_img = _load_panel_image(pi.image_path)
        if src_img is None:
            # Fallback placeholder
            _draw_placeholder(draw, slot.panel_id, px, py, pw, ph, border_color, border_width)
        else:
            strategy = _choose_strategy(slot.shot_size, slot.emotion_intensity)
            fitted = _fit_image(src_img, pw, ph, strategy)
            # Convert RGBA → RGB for paste (CONTAIN strategy returns RGBA)
            if fitted.mode == "RGBA":
                bg = Image.new("RGB", fitted.size, bg_color)
                bg.paste(fitted, mask=fitted.split()[3])
                fitted = bg
            page.paste(fitted, (px, py))

        # Panel border
        draw.rectangle(
            (px, py, px + pw, py + ph),
            outline=border_color,
            width=border_width,
        )

    # --- Render dialogue bubbles ---
    _render_bubbles(draw, bubbles, margin_px, inner_w, inner_h, bg_color, font_size_pt)

    # --- Page number ---
    num_font = _safe_font(20)
    num_text = f"- {page_number} -"
    num_bbox = draw.textbbox((0, 0), num_text, font=num_font)
    num_w = num_bbox[2] - num_bbox[0]
    draw.text(
        (width_px // 2 - num_w // 2, height_px - 34),
        num_text,
        fill=(100, 100, 100),
        font=num_font,
    )

    page.save(out_path)
    return str(out_path)


def _draw_placeholder(
    draw: ImageDraw.ImageDraw,
    panel_id: str,
    px: int, py: int, pw: int, ph: int,
    border_color: tuple[int, int, int],
    border_width: int,
) -> None:
    """Draw a fallback placeholder when the panel image is missing."""
    draw.rectangle(
        (px, py, px + pw, py + ph),
        fill=_color_for(panel_id),
        outline=border_color,
        width=border_width,
    )
    font = _safe_font(16)
    draw.text((px + 8, py + 8), f"[{panel_id}]", fill=(40, 40, 40), font=font)


# ---------------------------------------------------------------------------
# Bubble rendering
# ---------------------------------------------------------------------------

_TAIL_SIZE = 12    # pixels – tail triangle size
_THOUGHT_DOT_R = 5  # pixels – thought-bubble dot radius


def _render_bubbles(
    draw: ImageDraw.ImageDraw,
    bubbles: list["BubblePlacement"],
    margin_px: int,
    inner_w: int,
    inner_h: int,
    bg_color: tuple[int, int, int],
    font_size_pt: int = 14,
) -> None:
    """Dispatch each bubble to its type-specific renderer.

    每个气泡使用自身的 font_size_pt（由 BubbleAgent 根据面板大小自适应计算），
    渲染函数内部会从 bp.style 读取样式参数（fill/outline/border/radius/tail）。
    """
    for bp in bubbles:
        bx = margin_px + int(inner_w * bp.x)
        by = margin_px + int(inner_h * bp.y)
        bw = int(inner_w * bp.w)
        bh = int(inner_h * bp.h)
        body = (bx, by, bx + bw, by + bh)

        # 使用气泡自身的字体大小（兜底全局参数）
        bp_font = getattr(bp, 'font_size_pt', None) or font_size_pt

        if bp.bubble_type == "thought":
            _draw_thought(draw, body, bp, bg_color, bp_font)
        elif bp.bubble_type == "shout":
            _draw_shout(draw, body, bp, bg_color, bp_font)
        elif bp.bubble_type == "whisper":
            _draw_whisper(draw, body, bp, bg_color, bp_font)
        elif bp.bubble_type == "narration":
            _draw_narration(draw, body, bp, bp_font)
        else:  # "speech" or unknown
            _draw_speech(draw, body, bp, bp_font)


# ---------------------------------------------------------------------------
# Speech bubble — rounded rectangle + triangular tail
# ---------------------------------------------------------------------------

def _tail_polygon(
    body: tuple[int, int, int, int],
    direction: str,
    tail_size: int = _TAIL_SIZE,
) -> list[tuple[int, int]]:
    """Return a 3-point polygon for the tail, given the bubble body rect.

    The tail extends OUTWARD from the bubble edge, pointing toward the
    panel centre (i.e., opposite to *direction* — if direction is "down_right"
    the tail sits at the bottom-right corner of the bubble).
    """
    x1, y1, x2, y2 = body
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    ts = tail_size

    # Map "where should tail point" → "which edge of bubble" + "triangle coords"
    # Tail base sits on the bubble edge, tip extends outward.
    if direction == "down":
        return [(cx - ts, y2), (cx + ts, y2), (cx, y2 + ts)]
    elif direction == "up":
        return [(cx - ts, y1), (cx + ts, y1), (cx, y1 - ts)]
    elif direction == "left":
        return [(x1, cy - ts), (x1, cy + ts), (x1 - ts, cy)]
    elif direction == "right":
        return [(x2, cy - ts), (x2, cy + ts), (x2 + ts, cy)]
    elif direction == "down_right":
        return [(x2 - ts, y2), (x2, y2 - ts), (x2 + ts // 2, y2 + ts // 2)]
    elif direction == "down_left":
        return [(x1 + ts, y2), (x1, y2 - ts), (x1 - ts // 2, y2 + ts // 2)]
    elif direction == "up_right":
        return [(x2 - ts, y1), (x2, y1 + ts), (x2 + ts // 2, y1 - ts // 2)]
    elif direction == "up_left":
        return [(x1 + ts, y1), (x1, y1 + ts), (x1 - ts // 2, y1 - ts // 2)]
    else:
        # "auto" / fallback — tail at bottom edge
        return [(cx - ts, y2), (cx + ts, y2), (cx, y2 + ts)]


def _draw_speech(
    draw: ImageDraw.ImageDraw,
    body: tuple[int, int, int, int],
    bp: "BubblePlacement",
    font_size_pt: int = 14,
) -> None:
    """Rounded rectangle speech bubble with triangular tail."""
    x1, y1, x2, y2 = body

    # 从 bp.style 读取样式参数，兜底默认值
    style: dict = getattr(bp, 'style', None) or {}
    radius = style.get("radius", 12)
    border = style.get("border", 2)
    outline = tuple(style.get("outline", (20, 20, 20)))
    fill = tuple(style.get("fill", (255, 255, 255)))
    tail_size = style.get("tail", _TAIL_SIZE)
    effective_font = getattr(bp, 'font_size_pt', font_size_pt) or font_size_pt

    tail = _tail_polygon(body, bp.tail_direction, tail_size)

    # Draw tail first (filled, no outline)
    draw.polygon(tail, fill=fill)

    # Bubble body
    draw.rounded_rectangle(
        (x1, y1, x2, y2), radius=radius,
        fill=fill, outline=outline, width=border,
    )

    # Tail outline — draw the two edges of the triangle that aren't on the bubble
    _draw_tail_outline(draw, tail, body, outline_color=outline, border_width=border)

    # Text
    _draw_bubble_text(draw, bp, x1, y1, x2, y2, font_size_pt=effective_font)


def _draw_tail_outline(
    draw: ImageDraw.ImageDraw,
    tail: list[tuple[int, int]],
    body: tuple[int, int, int, int],
    outline_color: tuple[int, int, int] = (20, 20, 20),
    border_width: int = 2,
) -> None:
    """Draw thin lines for the two 'free' edges of the tail triangle."""
    x1, y1, x2, y2 = body
    for i in range(3):
        a = tail[i]
        b = tail[(i + 1) % 3]
        # Skip edge if both endpoints lie on the bubble border
        on_border_a = (a[0] == x1 or a[0] == x2 or a[1] == y1 or a[1] == y2)
        on_border_b = (b[0] == x1 or b[0] == x2 or b[1] == y1 or b[1] == y2)
        if not (on_border_a and on_border_b):
            draw.line([a, b], fill=outline_color, width=border_width)


# ---------------------------------------------------------------------------
# Thought bubble — rounded rect + small circles (cloud outline)
# ---------------------------------------------------------------------------

def _draw_thought(
    draw: ImageDraw.ImageDraw,
    body: tuple[int, int, int, int],
    bp: "BubblePlacement",
    bg_color: tuple[int, int, int],
    font_size_pt: int = 14,
) -> None:
    """Cloud-like thought bubble: rounded rect + circles along the bottom."""
    x1, y1, x2, y2 = body

    style: dict = getattr(bp, 'style', None) or {}
    radius = style.get("radius", 16)
    border = style.get("border", 2)
    outline = tuple(style.get("outline", (180, 180, 200)))
    fill = tuple(style.get("fill", (255, 255, 255)))
    effective_font = getattr(bp, 'font_size_pt', font_size_pt) or font_size_pt

    r = _THOUGHT_DOT_R
    # Dot positions: leading from the bubble toward the panel centre
    tail_dir = bp.tail_direction or "down"
    dots = _thought_dot_positions(body, tail_dir)

    # Draw dots
    for (dx, dy) in dots:
        draw.ellipse((dx - r, dy - r, dx + r, dy + r),
                     fill=fill, outline=outline, width=1)

    # Bubble body (draw after dots so it covers overlapping dot edges)
    draw.rounded_rectangle(
        (x1, y1, x2, y2), radius=radius,
        fill=fill, outline=outline, width=border,
    )

    _draw_bubble_text(draw, bp, x1, y1, x2, y2, font_size_pt=effective_font)


def _thought_dot_positions(
    body: tuple[int, int, int, int],
    direction: str,
) -> list[tuple[int, int]]:
    """Compute (x, y) centres for 3 thought-bubble dots."""
    x1, y1, x2, y2 = body
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    gap = _THOUGHT_DOT_R * 3

    if "down" in direction:
        base_y = y2 + gap
        base_x = cx - gap if "left" in direction else (cx + gap if "right" in direction else cx)
        return [(base_x - gap, base_y), (base_x, base_y + gap // 2), (base_x + gap // 2, base_y + gap)]
    elif "up" in direction:
        base_y = y1 - gap
        base_x = cx - gap if "left" in direction else (cx + gap if "right" in direction else cx)
        return [(base_x - gap, base_y), (base_x, base_y - gap // 2), (base_x + gap // 2, base_y - gap)]
    elif "left" in direction:
        base_x = x1 - gap
        return [(base_x, cy - gap), (base_x - gap // 2, cy), (base_x, cy + gap)]
    else:
        base_x = x2 + gap
        return [(base_x, cy - gap), (base_x + gap // 2, cy), (base_x, cy + gap)]


# ---------------------------------------------------------------------------
# Shout bubble — jagged-edge polygon
# ---------------------------------------------------------------------------

def _draw_shout(
    draw: ImageDraw.ImageDraw,
    body: tuple[int, int, int, int],
    bp: "BubblePlacement",
    bg_color: tuple[int, int, int],
    font_size_pt: int = 14,
) -> None:
    """Jagged / spiky shout bubble for intense dialogue."""
    x1, y1, x2, y2 = body

    style: dict = getattr(bp, 'style', None) or {}
    outline = tuple(style.get("outline", (200, 40, 40)))
    fill = tuple(style.get("fill", (255, 255, 240)))
    border = style.get("border", 2)
    tail_size = style.get("tail", _TAIL_SIZE)
    effective_font = getattr(bp, 'font_size_pt', font_size_pt) or font_size_pt

    # Generate jagged outline points
    points = _jagged_outline(x1, y1, x2, y2, amplitude=6, frequency=8)
    tail_pts = _tail_polygon(body, bp.tail_direction, tail_size)

    # Draw filled polygon (body + tail)
    draw.polygon(points + tail_pts, fill=fill, outline=outline, width=border)

    _draw_bubble_text(draw, bp, x1 + 6, y1 + 6, x2 - 6, y2 - 6, font_size_pt=effective_font)


def _jagged_outline(
    x1: int, y1: int, x2: int, y2: int,
    amplitude: int = 6,
    frequency: int = 8,
) -> list[tuple[int, int]]:
    """Generate a jagged polygon approximating a rectangle.

    The edge is modulated by a sine wave, creating the spiky comic-shout look.
    Returns points going clockwise: top → right → bottom → left.
    """
    pts: list[tuple[int, int]] = []
    w, h = x2 - x1, y2 - y1

    # Top edge (left → right)
    for i in range(frequency):
        t = i / frequency
        x = x1 + int(w * t)
        y = y1 + int(amplitude * math.sin(t * math.pi * frequency))
        pts.append((x, y))

    # Right edge (top → bottom)
    for i in range(frequency):
        t = i / frequency
        x = x2 + int(amplitude * math.sin(t * math.pi * frequency + math.pi / 2))
        y = y1 + int(h * t)
        pts.append((x, y))

    # Bottom edge (right → left)
    for i in range(frequency):
        t = i / frequency
        x = x2 - int(w * t)
        y = y2 + int(amplitude * math.sin(t * math.pi * frequency + math.pi))
        pts.append((x, y))

    # Left edge (bottom → top)
    for i in range(frequency):
        t = i / frequency
        x = x1 + int(amplitude * math.sin(t * math.pi * frequency + 3 * math.pi / 2))
        y = y2 - int(h * t)
        pts.append((x, y))

    return pts


# ---------------------------------------------------------------------------
# Whisper bubble — dashed-border rounded rectangle
# ---------------------------------------------------------------------------

def _draw_whisper(
    draw: ImageDraw.ImageDraw,
    body: tuple[int, int, int, int],
    bp: "BubblePlacement",
    bg_color: tuple[int, int, int],
    font_size_pt: int = 14,
) -> None:
    """Dashed-border whisper bubble for quiet / secretive dialogue."""
    x1, y1, x2, y2 = body

    style: dict = getattr(bp, 'style', None) or {}
    radius = style.get("radius", 12)
    border = style.get("border", 1)
    outline = tuple(style.get("outline", (140, 140, 160)))
    fill = tuple(style.get("fill", (250, 250, 255)))
    effective_font = getattr(bp, 'font_size_pt', font_size_pt) or font_size_pt
    # 派生文字颜色：outline 加深一点
    text_color = tuple(max(0, c - 40) for c in outline)

    # Fill
    draw.rounded_rectangle(
        (x1, y1, x2, y2), radius=radius,
        fill=fill,
    )
    # Dashed outline — draw as short line segments
    _draw_dashed_rounded_rect(draw, x1, y1, x2, y2, radius=radius,
                              dash_len=6, gap_len=4,
                              color=outline, width=border)

    _draw_bubble_text(draw, bp, x1, y1, x2, y2, text_color=text_color, font_size_pt=effective_font)


def _draw_dashed_rounded_rect(
    draw: ImageDraw.ImageDraw,
    x1: int, y1: int, x2: int, y2: int,
    radius: int, dash_len: int, gap_len: int,
    color: tuple[int, int, int], width: int,
) -> None:
    """Draw a rounded-rectangle outline with dash pattern.

    Approximates the rounded rect as straight segments (top/right/bottom/left)
    plus four 90° arcs at the corners.  Each is drawn as dashed lines.
    """
    # Straight segments: [ (start_x, start_y), (end_x, end_y) ]
    segments = [
        (x1 + radius, y1, x2 - radius, y1),           # top
        (x2, y1 + radius, x2, y2 - radius),            # right
        (x2 - radius, y2, x1 + radius, y2),            # bottom
        (x1, y2 - radius, x1, y1 + radius),            # left
    ]

    for sx, sy, ex, ey in segments:
        _draw_dashed_line(draw, sx, sy, ex, ey, dash_len, gap_len, color, width)

    # Corner arcs — draw as dashed too
    corners = [
        (x2 - radius, y1 + radius),   # top-right
        (x2 - radius, y2 - radius),   # bottom-right
        (x1 + radius, y2 - radius),   # bottom-left
        (x1 + radius, y1 + radius),   # top-left
    ]
    angles = [270, 0, 90, 180]  # start angles for each corner
    for (cx, cy), start_angle in zip(corners, angles):
        _draw_dashed_arc(draw, cx, cy, radius, start_angle, dash_len, gap_len, color, width)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    x1: int, y1: int, x2: int, y2: int,
    dash_len: int, gap_len: int,
    color: tuple[int, int, int], width: int,
) -> None:
    """Draw a straight dashed line from (x1,y1) to (x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    length = int(math.hypot(dx, dy))
    if length < 2:
        return
    ux, uy = dx / length, dy / length
    pos = 0
    drawing = True
    while pos < length:
        seg_len = dash_len if drawing else gap_len
        seg_len = min(seg_len, length - pos)
        if drawing:
            sx, sy = x1 + ux * pos, y1 + uy * pos
            ex, ey = x1 + ux * (pos + seg_len), y1 + uy * (pos + seg_len)
            draw.line([(sx, sy), (ex, ey)], fill=color, width=width)
        pos += seg_len
        drawing = not drawing


def _draw_dashed_arc(
    draw: ImageDraw.ImageDraw,
    cx: int, cy: int, radius: int, start_angle: int,
    dash_len: int, gap_len: int,
    color: tuple[int, int, int], width: int,
) -> None:
    """Draw a 90° dashed arc centred at (cx, cy)."""
    arc_len = int(math.pi / 2 * radius)  # quarter-circle length
    steps = max(1, arc_len // (dash_len + gap_len))
    for i in range(steps):
        a1 = math.radians(start_angle + i * 90 / steps)
        a2 = math.radians(start_angle + (i + 0.6) * 90 / steps)  # 0.6 for dash, 0.4 for gap
        xa, ya = cx + radius * math.cos(a1), cy - radius * math.sin(a1)
        xb, yb = cx + radius * math.cos(a2), cy - radius * math.sin(a2)
        draw.line([(xa, ya), (xb, yb)], fill=color, width=width)


# ---------------------------------------------------------------------------
# Narration box — simple square (placeholder for future page-level rendering)
# ---------------------------------------------------------------------------

def _draw_narration(
    draw: ImageDraw.ImageDraw,
    body: tuple[int, int, int, int],
    bp: "BubblePlacement",
    font_size_pt: int = 14,
) -> None:
    """Narration box with dynamic style from BubblePlacement."""
    x1, y1, x2, y2 = body

    style: dict = getattr(bp, 'style', None) or {}
    radius = style.get("radius", 4)
    border = style.get("border", 2)
    outline = tuple(style.get("outline", (80, 80, 80)))
    fill = tuple(style.get("fill", (255, 255, 240)))
    effective_font = getattr(bp, 'font_size_pt', font_size_pt) or font_size_pt
    text_color = tuple(max(0, c - 20) for c in outline)

    draw.rounded_rectangle(
        (x1, y1, x2, y2), radius=radius,
        fill=fill, outline=outline, width=border,
    )
    _draw_bubble_text(draw, bp, x1, y1, x2, y2, text_color=text_color, font_size_pt=effective_font)


# ---------------------------------------------------------------------------
# Shared text rendering
# ---------------------------------------------------------------------------

def _draw_bubble_text(
    draw: ImageDraw.ImageDraw,
    bp: "BubblePlacement",
    x1: int, y1: int, x2: int, y2: int,
    text_color: tuple[int, int, int] = (30, 30, 30),
    font_size_pt: int = 14,
) -> None:
    """Render wrapped dialogue text inside the bubble (no speaker label)."""
    text_font = _safe_font(font_size_pt)

    pad = 10
    text_y = y1 + pad

    text_max_w = (x2 - x1) - pad * 2
    lines = _wrap_text(draw, bp.text, text_max_w, text_font)
    line_h = text_font.size + 4 if hasattr(text_font, "size") else 18
    for i, ln in enumerate(lines):
        draw.text(
            (x1 + pad, text_y + i * line_h), ln,
            fill=text_color, font=text_font,
        )
