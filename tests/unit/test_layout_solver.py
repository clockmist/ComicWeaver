"""Unit tests for the recursive binary-partition layout solver."""
from __future__ import annotations

import pytest

from comicweaver.agents.layout.solver import (
    LayoutRegion,
    _choose_direction,
    _find_split_point,
    apply_gutters,
    solve_layout,
    PageGeometry,
)
from comicweaver.core.schema import BoundingBox, PanelPlan, ShotSize


def _make_panel(
    panel_id: str = "p1",
    page_id: str = "page_001",
    order: int = 1,
    emotion: float = 0.5,
    shot: ShotSize = ShotSize.MEDIUM,
) -> PanelPlan:
    return PanelPlan(
        panel_id=panel_id,
        page_id=page_id,
        order_in_page=order,
        emotion_intensity=emotion,
        shot_size=shot,
    )


# ---------------------------------------------------------------------------
# _find_split_point
# ---------------------------------------------------------------------------

class TestFindSplitPoint:
    def test_uniform_weights_split_middle(self):
        # [0.25, 0.25, 0.25, 0.25] → should split at k=2
        k = _find_split_point([0.25, 0.25, 0.25, 0.25])
        assert k == 2

    def test_two_panels_always_split_at_1(self):
        k = _find_split_point([0.7, 0.3])
        assert k == 1

    def test_skewed_weights(self):
        # [0.6, 0.2, 0.2] → half ≈ 0.5, cumulative at k=1 is 0.6 (closest)
        k = _find_split_point([0.6, 0.2, 0.2])
        assert k == 1


# ---------------------------------------------------------------------------
# _choose_direction
# ---------------------------------------------------------------------------

class TestChooseDirection:
    def test_climax_hint_prefers_horizontal_in_wide_region(self):
        region = LayoutRegion(0, 0, 1.0, 1.0)  # full page, aspect ~1.0
        panels = [_make_panel(f"p{i}") for i in range(3)]
        weights = [0.33, 0.33, 0.34]
        d = _choose_direction(panels, weights, region, "climax")
        assert d == "horizontal"

    def test_many_panels_in_wide_region_prefers_vertical(self):
        region = LayoutRegion(0, 0, 1.0, 0.3)  # wide strip, aspect ~3.3
        panels = [_make_panel(f"p{i}") for i in range(5)]
        weights = [0.2] * 5
        d = _choose_direction(panels, weights, region, "standard")
        assert d == "vertical"

    def test_dominating_first_panel_prefers_horizontal(self):
        region = LayoutRegion(0, 0, 1.0, 1.0)
        panels = [_make_panel("big"), _make_panel("small"), _make_panel("small")]
        weights = [0.7, 0.15, 0.15]  # first > 2× avg of rest
        d = _choose_direction(panels, weights, region, "standard")
        assert d == "horizontal"

    def test_alternates_from_parent_split(self):
        region = LayoutRegion(0, 0, 1.0, 1.0, parent_split="vertical")
        panels = [_make_panel(f"p{i}") for i in range(2)]
        d = _choose_direction(panels, [0.5, 0.5], region, "standard")
        assert d == "horizontal"


# ---------------------------------------------------------------------------
# solve_layout
# ---------------------------------------------------------------------------

class TestSolveLayout:
    def test_single_panel_returns_full_page(self):
        panels = [_make_panel("p1")]
        weights = [1.0]
        bboxes = solve_layout(panels, weights)
        assert len(bboxes) == 1
        b = bboxes[0]
        assert 0.0 <= b.x <= 1.0
        assert 0.0 <= b.y <= 1.0
        assert b.width > 0
        assert b.height > 0

    def test_two_panels_cover_full_page(self):
        panels = [_make_panel("p1", order=1), _make_panel("p2", order=2)]
        weights = [0.5, 0.5]
        bboxes = solve_layout(panels, weights)
        assert len(bboxes) == 2
        # Each bbox has positive area
        for b in bboxes:
            assert b.width > 0.05
            assert b.height > 0.05

    def test_four_panels_all_have_positive_area(self):
        panels = [
            _make_panel("p1", order=1, emotion=0.3),
            _make_panel("p2", order=2, emotion=0.5),
            _make_panel("p3", order=3, emotion=0.8),
            _make_panel("p4", order=4, emotion=0.4),
        ]
        ws = [0.17, 0.32, 0.40, 0.11]
        bboxes = solve_layout(panels, ws)
        assert len(bboxes) == 4
        for b in bboxes:
            assert b.width >= 0.10
            assert b.height >= 0.10

    def test_climax_hint_produces_valid_layout(self):
        panels = [_make_panel(f"p{i}", emotion=0.9) for i in range(3)]
        ws = [0.5, 0.3, 0.2]
        bboxes = solve_layout(panels, ws, hint="climax")
        assert len(bboxes) == 3
        # The climax panel (first, largest weight) should be the biggest
        assert bboxes[0].width * bboxes[0].height >= bboxes[1].width * bboxes[1].height

    def test_empty_panels_returns_empty(self):
        bboxes = solve_layout([], [])
        assert bboxes == []


# ---------------------------------------------------------------------------
# apply_gutters
# ---------------------------------------------------------------------------

class TestApplyGutters:
    def test_gutters_reduce_internal_panel_size(self):
        geom = PageGeometry(width_px=1240, height_px=1754, margin_px=40, gutter_px=20)
        bboxes = [
            BoundingBox(x=0.0, y=0.0, width=0.5, height=1.0),
            BoundingBox(x=0.5, y=0.0, width=0.5, height=1.0),
        ]
        result = apply_gutters(bboxes, geom)
        # Right edge of left panel should be < 0.5 (shrunk by half-gutter)
        assert result[0].x + result[0].width < 0.5
        # Left edge of right panel should be > 0.5
        assert result[1].x > 0.5

    def test_zero_gutter_returns_unchanged(self):
        geom = PageGeometry(width_px=1240, height_px=1754, margin_px=40, gutter_px=0)
        bboxes = [BoundingBox(x=0.0, y=0.0, width=0.5, height=0.5)]
        result = apply_gutters(bboxes, geom)
        assert result[0].x == bboxes[0].x
        assert result[0].width == bboxes[0].width

    def test_empty_input_returns_empty(self):
        result = apply_gutters([], PageGeometry())
        assert result == []
