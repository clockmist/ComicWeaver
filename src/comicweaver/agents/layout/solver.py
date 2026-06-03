"""Recursive binary-partition layout solver.

Given N panels with normalised weights [w₁ … wₙ] (∑wᵢ = 1), the solver
recursively subdivides the page rectangle so that each panel's area is
proportional to its weight.

This approach naturally produces the "grid-with-variations" look of hand-drawn
manga while remaining deterministic and fast (no LLM needed).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from comicweaver.core.schema import BoundingBox, PanelPlan

from .templates import LayoutHint

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

SplitDir = Literal["horizontal", "vertical"]


@dataclass
class PageGeometry:
    """Canvas definition for a single page."""
    width_px: int = 1240
    height_px: int = 1754
    margin_px: int = 40
    gutter_px: int = 10

    @property
    def inner_bbox(self) -> BoundingBox:
        """Return the page region excluding margins (normalised 0-1)."""
        return BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)


@dataclass
class LayoutRegion:
    """A rectangular region on the normalised page [0, 1]×[0, 1]."""
    x: float
    y: float
    w: float
    h: float
    # Track the split direction that *created* this region  (None = root)
    parent_split: SplitDir | None = None

    @property
    def aspect_ratio(self) -> float:
        return self.w / max(self.h, 1e-6)

    @property
    def area(self) -> float:
        return self.w * self.h

    def to_bbox(self) -> BoundingBox:
        return BoundingBox(x=self.x, y=self.y, width=self.w, height=self.h)

    def sub_region(
        self,
        direction: SplitDir,
        start_ratio: float,  # fraction along the split axis where the *first* region begins
        end_ratio: float,    # fraction along the split axis where the *first* region ends
    ) -> LayoutRegion:
        """Return a sub-region by splitting along *direction*."""
        if direction == "horizontal":
            new_y = self.y + self.h * start_ratio
            new_h = self.h * (end_ratio - start_ratio)
            return LayoutRegion(
                x=self.x, y=new_y, w=self.w, h=new_h, parent_split=direction,
            )
        else:  # vertical
            new_x = self.x + self.w * start_ratio
            new_w = self.w * (end_ratio - start_ratio)
            return LayoutRegion(
                x=new_x, y=self.y, w=new_w, h=self.h, parent_split=direction,
            )


def _make_root_region() -> LayoutRegion:
    return LayoutRegion(x=0.0, y=0.0, w=1.0, h=1.0, parent_split=None)


# ---------------------------------------------------------------------------
# Split-point selection
# ---------------------------------------------------------------------------

def _find_split_point(weights: list[float]) -> int:
    """Choose index *k* such that panels[:k] and panels[k:] have balanced weight.

    Returns k ∈ [1, len(weights)-1].
    Prefers splitting near the middle when weights are uniform.
    """
    n = len(weights)
    if n <= 1:
        return 1
    total = sum(weights)
    half = total / 2.0
    best_k = 1
    best_diff = float("inf")
    acc = 0.0
    for k in range(1, n):
        acc += weights[k - 1]
        diff = abs(acc - half)
        if diff < best_diff:
            best_diff = diff
            best_k = k
    return best_k


# ---------------------------------------------------------------------------
# Split-direction decision
# ---------------------------------------------------------------------------

def _choose_direction(
    panels: list[PanelPlan],
    weights: list[float],
    region: LayoutRegion,
    hint: LayoutHint,
) -> SplitDir:
    """Decide whether to split *region* horizontally or vertically.

    The rules encode manga-layout conventions and are the main source of
    "creative" variation in the solver.
    """
    n = len(panels)

    # Rule 0: single panel – shouldn't be called, but guard
    if n <= 1:
        return "horizontal"

    # Rule 1: climax / establishing hint – prefer horizontal (creates a wide
    #         hero panel that spans the full width)
    if hint in ("climax", "establishing") and region.w >= 0.9:
        return "horizontal"

    # Rule 2: many small panels in a wide region → vertical split
    #         (produces a row of small panels, typical for rapid cuts)
    if n >= 4 and region.aspect_ratio >= 1.3:
        return "vertical"

    # Rule 3: first panel dominates (> 2× the average of the rest)
    #         → horizontal split so the big panel gets a full row
    if n >= 2:
        first_w = weights[0]
        rest_avg = sum(weights[1:]) / (n - 1) if n > 1 else 1.0
        if rest_avg > 0 and first_w / rest_avg > 2.0:
            return "horizontal"

    # Rule 4: the region is very tall and narrow → horizontal split
    if region.aspect_ratio < 0.6:
        return "horizontal"

    # Rule 5: the region is very short and wide → vertical split
    if region.aspect_ratio > 2.0:
        return "vertical"

    # Rule 6: alternate from parent  (the classic "grid" feel)
    if region.parent_split == "vertical":
        return "horizontal"
    if region.parent_split == "horizontal":
        return "vertical"

    # Rule 7: default
    return "horizontal" if region.aspect_ratio <= 1.0 else "vertical"


# ---------------------------------------------------------------------------
# Recursive solver
# ---------------------------------------------------------------------------

def solve_layout(
    panels: list[PanelPlan],
    weights: list[float],
    region: LayoutRegion | None = None,
    hint: LayoutHint = "standard",
) -> list[BoundingBox]:
    """Recursively partition *region* to fit *panels* according to *weights*.

    Parameters
    ----------
    panels:
        The panels to place, in reading order.
    weights:
        Normalised weights (sum == 1.0), parallel to *panels*.
    region:
        The current region to subdivide.  Defaults to the full page.
    hint:
        Layout intent from the storyboard.

    Returns
    -------
    list[BoundingBox]
        One bbox per panel, in the same order.
    """
    if region is None:
        region = _make_root_region()

    n = len(panels)
    if n == 0:
        return []
    if n == 1:
        # Apply style modifiers to the final single-panel region
        bbox = _apply_panel_modifiers(region.to_bbox(), panels[0])
        return [bbox]

    # 1. Find split point
    k = _find_split_point(weights)
    # Clamp k to valid range
    k = max(1, min(k, n - 1))

    # 2. Choose direction
    direction = _choose_direction(panels, weights, region, hint)

    # 3. Calculate split ratio
    left_weight = sum(weights[:k])
    total_weight = sum(weights)
    ratio = left_weight / total_weight if total_weight > 0 else 0.5
    # Clamp to avoid degenerate panels
    ratio = max(0.15, min(0.85, ratio))

    # 4. Create sub-regions
    left_region = region.sub_region(direction, 0.0, ratio)
    right_region = region.sub_region(direction, ratio, 1.0)

    # 5. Recurse
    left_bboxes = solve_layout(panels[:k], weights[:k], left_region, hint)
    right_bboxes = solve_layout(panels[k:], weights[k:], right_region, hint)

    return left_bboxes + right_bboxes


# ---------------------------------------------------------------------------
# Style modifiers  (per-panel post-processing)
# ---------------------------------------------------------------------------

def _apply_panel_modifiers(bbox: BoundingBox, panel: PanelPlan) -> BoundingBox:
    """Apply per-panel style adjustments to a solved bbox.

    Current modifiers:
    - **bleed**: panels with emotion ≥ 0.85 extend to the page edge
    - **aspect-ratio clamp**: prevent panels that are too narrow or too flat
    """
    x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height

    # Bleed: extend to the nearest page edges
    if panel.emotion_intensity >= 0.85:
        if x < 0.05:
            x = 0.0
            w += bbox.x  # reclaim the left margin
        if x + w > 0.95:
            w = 1.0 - x
        if y < 0.05:
            y = 0.0
            h += bbox.y
        if y + h > 0.95:
            h = 1.0 - y

    # Aspect-ratio clamp
    min_dim = 0.10
    w = max(w, min_dim)
    h = max(h, min_dim)

    # Prevent panels from exceeding page bounds
    x = max(0.0, min(x, 1.0 - w))
    y = max(0.0, min(y, 1.0 - h))

    return BoundingBox(x=x, y=y, width=w, height=h)


# ---------------------------------------------------------------------------
# Gutter-aware bbox shrink
# ---------------------------------------------------------------------------

def apply_gutters(
    bboxes: list[BoundingBox],
    geometry: PageGeometry,
) -> list[BoundingBox]:
    """Shrink each bbox inward to create gutter space between panels.

    The gutter is subtracted from panel edges that are NOT at the page boundary
    (i.e., page-edge panels keep their outer margin).
    """
    if not bboxes or geometry.gutter_px <= 0:
        return bboxes

    # Convert gutter from pixels to normalised space
    inner_w = geometry.width_px - 2 * geometry.margin_px
    inner_h = geometry.height_px - 2 * geometry.margin_px
    gutter_norm_x = geometry.gutter_px / inner_w if inner_w > 0 else 0
    gutter_norm_y = geometry.gutter_px / inner_h if inner_h > 0 else 0

    half_gx = gutter_norm_x / 2.0
    half_gy = gutter_norm_y / 2.0

    result: list[BoundingBox] = []
    for bbox in bboxes:
        x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height

        # Shrink from each side unless the panel touches the page edge
        left = x + half_gx if x > 0.01 else x
        top = y + half_gy if y > 0.01 else y
        right = x + w - half_gx if (x + w) < 0.99 else x + w
        bottom = y + h - half_gy if (y + h) < 0.99 else y + h

        new_x = left
        new_y = top
        new_w = max(0.05, right - left)
        new_h = max(0.05, bottom - top)

        result.append(BoundingBox(x=new_x, y=new_y, width=new_w, height=new_h))

    return result
