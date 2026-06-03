"""Dialogue-bubble placement engine.

Given panel bboxes, dialogue content, and bubble hints from the storyboard,
this module computes the final (x, y, w, h) for each speech/thought/narration
bubble in normalised page coordinates.

Design principle:
    BubbleHint from StoryboardAgent is treated as a **suggestion for the
    ImageAgent** (so it knows to leave space when generating the panel image).
    The LayoutAgent uses BubbleHint as a starting candidate but may adjust
    positions based on text length, panel boundaries, and multi-bubble
    stacking.  A future VLM path will additionally analyse the actual image
    to avoid faces and focal regions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from comicweaver.core.schema import BoundingBox, BubbleHint

if TYPE_CHECKING:
    from comicweaver.core.schema import Dialogue, PanelImage, PanelPlan


# ---------------------------------------------------------------------------
# Bubble placement result
# ---------------------------------------------------------------------------

@dataclass
class BubblePlacement:
    """Finalised bubble coordinates (normalised 0-1, relative to full page)."""
    panel_id: str
    dialogue_index: int
    bubble_type: str       # "speech" | "thought" | "shout" | "whisper" | "narration"
    speaker: str
    text: str
    x: float
    y: float
    w: float
    h: float
    occlusion_score: float = 0.0
    tail_direction: str = "auto"  # "up" | "down" | "left" | "right" | "up_left" | ...


def _derive_tail_direction(
    bubble_center_x: float, bubble_center_y: float,
    panel_bbox: BoundingBox,
) -> str:
    """Determine which way the bubble's tail should point.

    The tail points from the bubble's EDGE toward the panel CENTER
    (where the speaker is most likely located).  So a bubble in the
    top-left corner gets a tail pointing "down_right".
    """
    pcx = panel_bbox.x + panel_bbox.width / 2
    pcy = panel_bbox.y + panel_bbox.height / 2

    dx = pcx - bubble_center_x
    dy = pcy - bubble_center_y

    # Determine primary direction
    v = ""
    if dy < -0.02:
        v = "up"
    elif dy > 0.02:
        v = "down"

    h = ""
    if dx < -0.02:
        h = "left"
    elif dx > 0.02:
        h = "right"

    if v and h:
        return f"{v}_{h}"
    if v:
        return v
    if h:
        return h
    return "down"  # fallback: most bubbles get a tail pointing down


# ---------------------------------------------------------------------------
# Text measurement  (CJK-aware)
# ---------------------------------------------------------------------------

def _load_font(size: int) -> "ImageFont.ImageFont":
    """Best-effort CJK font loading; falls back to PIL default."""
    import os as _os
    from PIL import ImageFont
    for path in (
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ):
        if _os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def estimate_text_size(
    text: str,
    font_size_pt: int = 12,
    max_width_px: int = 200,
) -> tuple[int, int]:
    """Measure the actual pixel size needed to render *text* using PIL.

    Loads the real CJK font at *font_size_pt*, measures each character's
    width with ``textbbox()``, and wraps lines at *max_width_px*.

    Returns (width_px, height_px) including padding.
    """
    if not text:
        return (80, 30)

_EMPTY_IMAGE = None


def _get_draw() -> "ImageDraw.ImageDraw":
    """Get a reusable PIL ImageDraw for text measurement (lazy init)."""
    global _EMPTY_IMAGE
    from PIL import Image, ImageDraw
    if _EMPTY_IMAGE is None:
        _EMPTY_IMAGE = Image.new("RGB", (1, 1))
    return ImageDraw.Draw(_EMPTY_IMAGE)


def estimate_text_size(
    text: str,
    font_size_pt: int = 12,
    max_width_px: int = 200,
) -> tuple[int, int]:
    """Measure the actual pixel size needed to render *text* using PIL.

    Loads the real CJK font at *font_size_pt*, measures each character's
    width with ``textbbox()``, and wraps lines at *max_width_px*.

    Returns (width_px, height_px) including padding.
    """
    if not text:
        return (80, 30)

    from PIL import ImageDraw

    font = _load_font(font_size_pt)
    draw = _get_draw()

    # Line breaking: accumulate characters until exceeding max_width_px
    lines: list[str] = []
    current_line = ""
    for ch in text:
        candidate = current_line + ch
        bbox = draw.textbbox((0, 0), candidate, font=font)
        w = bbox[2] - bbox[0]
        if w > max_width_px and current_line:
            lines.append(current_line)
            current_line = ch
        else:
            current_line = candidate
    if current_line:
        lines.append(current_line)

    # Measure width of the longest rendered line
    max_w = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        if w > max_w:
            max_w = w

    # Height: number of lines × line height
    # TrueType font .size returns the em height in pixels
    line_h = font.size + 4 if hasattr(font, "size") else font_size_pt + 4
    text_h = len(lines) * line_h

    # Add padding (matching compositor's _draw_bubble_text)
    pad = 10
    width = int(max_w) + pad * 2 + 4  # +4 for safety margin
    height = text_h + pad * 2 + 28    # +28 for speaker label area

    # Clamp to reasonable limits
    width = max(80, min(width, 500))
    height = max(40, min(height, 350))

    return width, height


# ---------------------------------------------------------------------------
# Position suggestion  (from BubbleHint)
# ---------------------------------------------------------------------------

# Map BubbleHint position → relative anchor within the panel bbox
_POSITION_ANCHORS: dict[str, tuple[float, float]] = {
    "top_left":      (0.02, 0.02),
    "top_right":     (0.55, 0.02),
    "bottom_left":   (0.02, 0.65),
    "bottom_right":  (0.55, 0.65),
    "center":        (0.25, 0.40),
    "auto":          (0.55, 0.02),   # default → top-right
}


def _bubble_anchor(panel_bbox: BoundingBox, hint: BubbleHint | None) -> tuple[float, float]:
    """Return the suggested (x, y) anchor within *panel_bbox* (normalised)."""
    if hint and hint.suggested_position:
        anchor = _POSITION_ANCHORS.get(hint.suggested_position, _POSITION_ANCHORS["auto"])
    else:
        anchor = _POSITION_ANCHORS["auto"]

    return (
        panel_bbox.x + panel_bbox.width * anchor[0],
        panel_bbox.y + panel_bbox.height * anchor[1],
    )


# ---------------------------------------------------------------------------
# Bubble-type resolution
# ---------------------------------------------------------------------------

def _resolve_bubble_type(dialogue: Dialogue, hint: BubbleHint | None) -> str:
    """Merge dialogue tone and BubbleHint to determine the bubble style."""
    # BubbleHint overrides if it specifies something beyond speech/thought
    if hint and hint.bubble_type not in ("speech", "thought"):
        return hint.bubble_type

    # Map Dialogue.tone → bubble type
    tone_map = {
        "angry":   "shout",
        "excited": "shout",
        "sad":     "whisper",
        "shy":     "whisper",
        "neutral": "speech",
    }
    tone = getattr(dialogue, "tone", "neutral") or "neutral"
    if dialogue.is_thought:
        return "thought"
    return tone_map.get(tone, "speech")


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------


def place_bubbles(
    panels: list[PanelPlan],
    panel_bboxes: list[BoundingBox],
    panel_images: dict[str, PanelImage],
    *,
    font_size_pt: int = 12,
    page_width_px: int = 1240,
    page_height_px: int = 1754,
    margin_px: int = 40,
    faces_by_panel: dict[str, list] | None = None,
) -> list[BubblePlacement]:
    """Compute bubble positions for all panels on a page.

    Parameters
    ----------
    panels:
        Panel plans in reading order (from StoryboardAgent).
    panel_bboxes:
        Solved bboxes, parallel to *panels*.
    panel_images:
        Generated panel images keyed by panel_id (for future image analysis).
    font_size_pt:
        Base font size for text-size estimation.
    faces_by_panel:
        Optional dict of panel_id → list[FaceRegion] from VLM detection.
        When provided, bubble positions are optimized to avoid occluding faces.

    Returns
    -------
    list[BubblePlacement]
        All bubbles for the page, with normalised coordinates.
    """
    if faces_by_panel is None:
        faces_by_panel = {}

    all_bubbles: list[BubblePlacement] = []

    for panel, bbox in zip(panels, panel_bboxes, strict=False):
        dialogues = panel.dialogues_in_panel
        if not dialogues:
            continue

        # Build hint lookup keyed by dialogue_index
        hint_map: dict[int, BubbleHint] = {}
        for h in panel.speech_bubble_hints:
            hint_map[h.dialogue_index] = h

        # Place each dialogue
        for di, dialogue in enumerate(dialogues):
            hint = hint_map.get(di)

            # Resolve bubble type
            bubble_type = _resolve_bubble_type(dialogue, hint)

            # Estimate text size → bubble dimensions in normalised space
            text_w_px, text_h_px = estimate_text_size(dialogue.text, font_size_pt)
            # Convert px to normalised using inner dimensions (match compositor)
            inner_w = page_width_px - margin_px * 2
            inner_h = page_height_px - margin_px * 2
            w_norm = text_w_px / max(inner_w, 1)
            h_norm = text_h_px / max(inner_h, 1)

            # Get anchor position from hint
            ax, ay = _bubble_anchor(bbox, hint)

            # Multi-bubble stacking: offset vertically within the panel
            stack_offset = di * 0.07

            bx = ax
            by = ay + stack_offset

            # Boundary clamp – keep bubble inside panel bbox
            bx = max(bbox.x + 0.01, min(bx, bbox.x + bbox.width - w_norm - 0.01))
            by = max(bbox.y + 0.01, min(by, bbox.y + bbox.height - h_norm - 0.01))

            # Derive tail direction
            bubble_cx = bx + w_norm / 2
            bubble_cy = by + h_norm / 2
            tail_dir = _derive_tail_direction(bubble_cx, bubble_cy, bbox)

            all_bubbles.append(
                BubblePlacement(
                    panel_id=panel.panel_id,
                    dialogue_index=di,
                    bubble_type=bubble_type,
                    speaker=dialogue.speaker,
                    text=dialogue.text,
                    x=bx,
                    y=by,
                    w=w_norm,
                    h=h_norm,
                    occlusion_score=0.0,
                    tail_direction=tail_dir,
                )
            )

    # ---- VLM face-aware optimization ----
    for panel, bbox in zip(panels, panel_bboxes, strict=False):
        faces = faces_by_panel.get(panel.panel_id)
        if not faces:
            continue
        # Extract bubbles belonging to this panel
        panel_bubbles = [b for b in all_bubbles if b.panel_id == panel.panel_id]
        if not panel_bubbles:
            continue
        from .face_detector import FaceRegion, optimize_bubble_positions
        face_regions = [
            FaceRegion(x=f.x, y=f.y, w=f.w, h=f.h, char_name=f.char_name)
            if hasattr(f, 'x') else
            FaceRegion(
                x=float(f.get("x", 0)), y=float(f.get("y", 0)),
                w=float(f.get("w", 0)), h=float(f.get("h", 0)),
                char_name=str(f.get("char_name", "")),
            )
            for f in faces
        ]
        optimized = optimize_bubble_positions(panel_bubbles, face_regions, bbox)
        # Replace in-place
        for i, b in enumerate(all_bubbles):
            for ob in optimized:
                if b.panel_id == ob.panel_id and b.dialogue_index == ob.dialogue_index:
                    all_bubbles[i] = ob

    return all_bubbles
