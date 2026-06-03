"""Layout template library and layout-hint definitions.

Templates are expressed as lists of normalised (x, y, w, h) tuples in [0, 1].
They serve two purposes:

1. **Fallback** – when the recursive solver cannot produce a valid layout, a
   template is used instead.
2. **Reference** – the solver may use template geometry as a starting point
   that it then adjusts according to panel weights.

The ``LayoutHint`` type encodes the *intent* that StoryboardAgent passes to
LayoutAgent, replacing the old ``layout_template`` string that carried both
intent AND coordinates.
"""
from __future__ import annotations

from typing import Literal

# ---------------------------------------------------------------------------
# Layout hint – intent, NOT coordinates
# ---------------------------------------------------------------------------

LayoutHint = Literal[
    "standard",       # normal narrative page, alternating split directions
    "climax",         # climax page, prefer horizontal splits → splash panels
    "action",         # action-heavy page, may trigger diagonal splits / bleed
    "dialogue",       # dialogue-heavy page, reserve more area for bubbles
    "establishing",   # opening page, first panel gets extra width
]

# ---------------------------------------------------------------------------
# Template definitions  (normalised bbox tuples: x, y, w, h)
# ---------------------------------------------------------------------------

# Standard 2×2 grid
GRID_2X2: list[tuple[float, float, float, float]] = [
    (0.0, 0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5, 0.5),
    (0.0, 0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5, 0.5),
]

# 3-row × 2-column grid (6 panels)
GRID_3X2: list[tuple[float, float, float, float]] = [
    (0.0, 0.00, 0.5, 0.33),
    (0.5, 0.00, 0.5, 0.33),
    (0.0, 0.33, 0.5, 0.33),
    (0.5, 0.33, 0.5, 0.33),
    (0.0, 0.66, 0.5, 0.34),
    (0.5, 0.66, 0.5, 0.34),
]

# Wide hero panel on top, two columns below
SPLASH_TOP: list[tuple[float, float, float, float]] = [
    (0.0, 0.0, 1.0, 0.5),
    (0.0, 0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5, 0.5),
]

# Staggered / diagonal layout (3 panels)
DIAGONAL: list[tuple[float, float, float, float]] = [
    (0.0, 0.0,  0.6, 0.45),
    (0.4, 0.0,  0.6, 0.45),
    (0.0, 0.55, 1.0, 0.45),
]

# Single full-page panel
FULL_BLEED: list[tuple[float, float, float, float]] = [
    (0.0, 0.0, 1.0, 1.0),
]

# 2-row layout  –  1 wide panel on top, 2 below (same as splash_top)
TWO_ROW_HEAVY_TOP: list[tuple[float, float, float, float]] = [
    (0.0, 0.0, 1.0, 0.55),
    (0.0, 0.55, 0.5, 0.45),
    (0.5, 0.55, 0.5, 0.45),
]

# Stair-step layout (4 panels, manga style)
STAIR_4: list[tuple[float, float, float, float]] = [
    (0.0, 0.0,  0.55, 0.40),
    (0.55, 0.0, 0.45, 0.40),
    (0.0, 0.40, 0.45, 0.60),
    (0.45, 0.40, 0.55, 0.60),
]

# Inset layout  –  1 large panel with 2 small insets
INSET: list[tuple[float, float, float, float]] = [
    (0.0, 0.0, 1.0, 1.0),       # full background
    (0.02, 0.02, 0.30, 0.25),    # top-left inset
    (0.68, 0.73, 0.30, 0.25),    # bottom-right inset
]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LAYOUT_TEMPLATES: dict[str, list[tuple[float, float, float, float]]] = {
    "grid_2x2":          GRID_2X2,
    "grid_3x2":          GRID_3X2,
    "splash_top":        SPLASH_TOP,
    "diagonal":          DIAGONAL,
    "full_bleed":        FULL_BLEED,
    "two_row_heavy_top": TWO_ROW_HEAVY_TOP,
    "stair_4":           STAIR_4,
    "inset":             INSET,
}

# Map hint → preferred templates (first = default)
HINT_TEMPLATE_MAP: dict[LayoutHint, list[str]] = {
    "standard":      ["grid_2x2", "grid_3x2"],
    "climax":        ["splash_top", "full_bleed", "diagonal"],
    "action":        ["diagonal", "stair_4", "splash_top"],
    "dialogue":      ["grid_2x2", "grid_3x2", "two_row_heavy_top"],
    "establishing":  ["two_row_heavy_top", "splash_top", "grid_2x2"],
}


def get_template(name: str) -> list[tuple[float, float, float, float]] | None:
    """Look up a template by name.  Returns *None* if unknown."""
    return LAYOUT_TEMPLATES.get(name)


def template_for_hint(hint: LayoutHint, panel_count: int) -> str:
    """Pick a suitable template name for the given hint and panel count."""
    candidates = HINT_TEMPLATE_MAP.get(hint, ["grid_2x2"])
    for name in candidates:
        tmpl = LAYOUT_TEMPLATES.get(name)
        if tmpl and len(tmpl) >= panel_count:
            return name
    # Fallback: return the first candidate regardless of fit
    return candidates[0]
