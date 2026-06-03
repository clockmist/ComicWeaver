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


# ---------------------------------------------------------------------------
# Text measurement  (CJK-aware)
# ---------------------------------------------------------------------------

# Approximate character widths as fraction of font height.
# CJK chars are roughly square; Latin chars are ~0.5× the CJK width.
_CJK_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0x3040, 0x309F),   # Hiragana
    (0x30A0, 0x30FF),   # Katakana
    (0xAC00, 0xD7AF),   # Hangul Syllables
]


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def estimate_text_size(
    text: str,
    font_size_pt: int = 12,
    max_width_chars: int = 18,
    line_height_ratio: float = 1.5,
) -> tuple[int, int]:
    """Estimate the pixel size needed to render *text*.

    Uses a character-count heuristic that works well for CJK without PIL.

    Returns (width_px, height_px).
    """
    if not text:
        return (60, 30)  # minimum bubble size

    # Count effective character widths
    char_units = 0.0
    for ch in text:
        char_units += 1.0 if _is_cjk(ch) else 0.55

    # Line breaking
    lines = max(1, int(char_units / max_width_chars) + (1 if char_units % max_width_chars > 0 else 0))
    chars_per_line = min(char_units, max_width_chars)

    # Pixel dimensions
    char_px = font_size_pt * 1.0   # approximate CJK char width in px at given pt
    width = int(chars_per_line * char_px) + 20   # + padding
    height = int(lines * font_size_pt * line_height_ratio) + 16

    # Clamp
    width = max(80, min(width, 450))
    height = max(30, min(height, 300))

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

    Returns
    -------
    list[BubblePlacement]
        All bubbles for the page, with normalised coordinates.
    """
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
            # Convert px to normalised (assuming a reference page width ~1240px)
            w_norm = text_w_px / 1240.0
            h_norm = text_h_px / 1754.0

            # Get anchor position from hint
            ax, ay = _bubble_anchor(bbox, hint)

            # Multi-bubble stacking: offset vertically within the panel
            stack_offset = di * 0.07

            bx = ax
            by = ay + stack_offset

            # Boundary clamp – keep bubble inside panel bbox
            bx = max(bbox.x + 0.01, min(bx, bbox.x + bbox.width - w_norm - 0.01))
            by = max(bbox.y + 0.01, min(by, bbox.y + bbox.height - h_norm - 0.01))

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
                    occlusion_score=0.0,  # will be updated with image analysis
                )
            )

    return all_bubbles
