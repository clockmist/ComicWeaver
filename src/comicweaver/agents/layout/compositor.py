"""Page image compositor.

Assembles a final comic page from individual panel images, placing them at
their solved bbox positions and rendering dialogue bubbles on top.

This replaces the old ``storage/placeholder.py::generate_page_placeholder()``
which is being removed from the storage layer.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from comicweaver.core.schema import BoundingBox, PanelImage, ShotSize

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

    # If aspect ratios are very close (< 3% diff), stretch is acceptable
    ratio_diff = abs(src_ratio - tgt_ratio) / max(src_ratio, tgt_ratio)
    if strategy == FitStrategy.STRETCH or ratio_diff < 0.03:
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
    _render_bubbles(draw, bubbles, margin_px, inner_w, inner_h, bg_color)

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

def _render_bubbles(
    draw: ImageDraw.ImageDraw,
    bubbles: list["BubblePlacement"],
    margin_px: int,
    inner_w: int,
    inner_h: int,
    bg_color: tuple[int, int, int],
) -> None:
    """Draw all dialogue bubbles onto the page."""
    speaker_font = _safe_font(18)
    text_font = _safe_font(15)

    for bp in bubbles:
        bx = margin_px + int(inner_w * bp.x)
        by = margin_px + int(inner_h * bp.y)
        bw = int(inner_w * bp.w)
        bh = int(inner_h * bp.h)

        # Bubble background based on type
        if bp.bubble_type == "narration":
            # Narration box – square, no tail
            draw.rectangle(
                (bx, by, bx + bw, by + bh),
                fill=(255, 255, 240),
                outline=(80, 80, 80),
                width=2,
            )
        elif bp.bubble_type == "thought":
            # Thought bubble – rounded with lighter border
            draw.rounded_rectangle(
                (bx, by, bx + bw, by + bh),
                radius=16,
                fill=(255, 255, 255),
                outline=(180, 180, 200),
                width=2,
            )
        elif bp.bubble_type == "shout":
            # Shout bubble – thicker outline, warm tint
            draw.rounded_rectangle(
                (bx, by, bx + bw, by + bh),
                radius=8,
                fill=(255, 255, 240),
                outline=(200, 40, 40),
                width=3,
            )
        else:  # "speech" or unknown
            draw.rounded_rectangle(
                (bx, by, bx + bw, by + bh),
                radius=12,
                fill=(255, 255, 255),
                outline=(20, 20, 20),
                width=2,
            )

        # Speaker label
        if bp.speaker:
            draw.text(
                (bx + 10, by + 8),
                f"{bp.speaker}:",
                fill=(40, 40, 80),
                font=speaker_font,
            )

        # Dialogue text (wrapped)
        text_offset_y = by + 32 if bp.speaker else by + 10
        text_max_w = bw - 20
        lines = _wrap_text(draw, bp.text, text_max_w, text_font)
        line_h = text_font.size + 4 if hasattr(text_font, "size") else 18
        for i, ln in enumerate(lines):
            draw.text(
                (bx + 10, text_offset_y + i * line_h),
                ln,
                fill=(30, 30, 30),
                font=text_font,
            )
