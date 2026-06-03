"""Panel visual-weight calculation.

Each panel receives a relative weight ∈ (0, 1] based on multiple signals from the
upstream agents (ScriptAgent → StoryboardAgent).  Weights are normalised within a
page so they sum to 1.0 — the fraction of page area the panel deserves.
"""
from __future__ import annotations

from dataclasses import dataclass

from comicweaver.core.schema import PanelPlan, ShotSize

# ---------------------------------------------------------------------------
# Shot-size → ideal-area multiplier
# ---------------------------------------------------------------------------
# Rationale:
#   extreme_long / establishing  → needs width to show the setting
#   extreme_close                → needs area for dramatic impact
#   medium                       → most flexible, can be smaller
SHOT_WEIGHT: dict[str, float] = {
    ShotSize.EXTREME_LONG.value:  1.30,
    ShotSize.LONG.value:          1.20,
    ShotSize.FULL.value:          1.00,
    ShotSize.MEDIUM.value:        0.85,
    ShotSize.CLOSE.value:         1.10,
    ShotSize.EXTREME_CLOSE.value: 1.30,
}

# ---------------------------------------------------------------------------
# Context object (lightweight – no Pydantic overhead needed here)
# ---------------------------------------------------------------------------


@dataclass
class PageContext:
    """Per-page signals that influence every panel's weight."""
    is_climax_page: bool = False
    page_emotion_avg: float = 0.5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate_panel_weight(panel: PanelPlan, page_ctx: PageContext) -> float:
    """Return the raw (un-normalised) visual weight for *panel*.

    The weight is built from these signals, each contributing a multiplicative
    factor ≥ 1.0 so that the baseline (all-default panel) == 1.0:

    1. **emotion_intensity**  — strongest driver;  mapped to [0.3, 1.0]
    2. **shot_size**           — ideal-area lookup table
    3. **dialogue burden**     — longer / more dialogues → need bubble space
    4. **character count**     — mild boost per character
    5. **climax-page bonus**   — ×1.15 when the page contains a climax scene
    """
    # 1. Emotion  (0-1  →  0.3-1.0 base)
    w_emotion = 0.3 + panel.emotion_intensity * 0.7

    # 2. Shot size
    shot_val = getattr(panel.shot_size, "value", str(panel.shot_size))
    w_shot = SHOT_WEIGHT.get(shot_val, 1.0)

    # 3. Dialogue burden
    total_chars = sum(len(d.text) for d in panel.dialogues_in_panel)
    w_dialogue = 1.0 + min(total_chars / 150.0, 0.35)  # capped +35 %

    # 4. Character count
    w_chars = 1.0 + min(len(panel.characters_in_panel), 4) * 0.04

    # 5. Climax page
    w_climax = 1.15 if page_ctx.is_climax_page else 1.0

    return w_emotion * w_shot * w_dialogue * w_chars * w_climax


def normalize_weights(
    panels: list[PanelPlan],
    page_ctx: PageContext,
) -> list[float]:
    """Compute normalised weights so that ``sum(result) == 1.0``.

    Returns a list parallel to *panels*.  If all raw weights are zero
    (shouldn't happen), falls back to equal distribution.
    """
    raw = [calculate_panel_weight(p, page_ctx) for p in panels]
    total = sum(raw)
    if total <= 0:
        # Degenerate fallback – equal split
        n = len(panels)
        return [1.0 / n] * n if n > 0 else []
    return [w / total for w in raw]
