"""Unit tests for layout weight calculation."""
from __future__ import annotations

import pytest

from comicweaver.agents.layout.weights import (
    PageContext,
    SHOT_WEIGHT,
    calculate_panel_weight,
    normalize_weights,
)
from comicweaver.core.schema import (
    Dialogue,
    PanelPlan,
    ShotSize,
)


def _make_panel(
    emotion: float = 0.5,
    shot: ShotSize = ShotSize.MEDIUM,
    dialogues: list[Dialogue] | None = None,
    characters: list[str] | None = None,
) -> PanelPlan:
    return PanelPlan(
        panel_id="p1",
        page_id="page_001",
        order_in_page=1,
        emotion_intensity=emotion,
        shot_size=shot,
        dialogues_in_panel=dialogues or [],
        characters_in_panel=characters or [],
    )


class TestCalculatePanelWeight:
    def test_baseline_weight_is_positive(self):
        w = calculate_panel_weight(_make_panel(), PageContext())
        assert w > 0.0

    def test_high_emotion_increases_weight(self):
        lo = calculate_panel_weight(_make_panel(emotion=0.2), PageContext())
        hi = calculate_panel_weight(_make_panel(emotion=0.9), PageContext())
        assert hi > lo

    def test_climax_page_boosts_all_panels(self):
        ctx_normal = PageContext(is_climax_page=False)
        ctx_climax = PageContext(is_climax_page=True)
        w_normal = calculate_panel_weight(_make_panel(), ctx_normal)
        w_climax = calculate_panel_weight(_make_panel(), ctx_climax)
        assert w_climax > w_normal

    def test_longer_dialogue_increases_weight(self):
        short = calculate_panel_weight(
            _make_panel(dialogues=[Dialogue(speaker="A", text="Hi")]),
            PageContext(),
        )
        long = calculate_panel_weight(
            _make_panel(dialogues=[Dialogue(speaker="A", text="This is a much longer dialogue line with many characters in it")]),
            PageContext(),
        )
        assert long >= short  # dialogue burden is capped, but still monotonic

    def test_shot_size_extreme_long_gets_higher_weight(self):
        w_long = calculate_panel_weight(_make_panel(shot=ShotSize.EXTREME_LONG), PageContext())
        w_medium = calculate_panel_weight(_make_panel(shot=ShotSize.MEDIUM), PageContext())
        assert w_long > w_medium

    def test_shot_size_extreme_close_gets_higher_weight(self):
        w_close = calculate_panel_weight(_make_panel(shot=ShotSize.EXTREME_CLOSE), PageContext())
        w_medium = calculate_panel_weight(_make_panel(shot=ShotSize.MEDIUM), PageContext())
        assert w_close > w_medium


class TestNormalizeWeights:
    def test_sums_to_one(self):
        panels = [
            _make_panel(emotion=0.3),
            _make_panel(emotion=0.5),
            _make_panel(emotion=0.8),
        ]
        ws = normalize_weights(panels, PageContext())
        assert len(ws) == 3
        assert abs(sum(ws) - 1.0) < 0.001
        # The highest-emotion panel should get the largest share
        assert ws[2] > ws[0]

    def test_equal_panels_get_equal_weights(self):
        panels = [_make_panel(emotion=0.5) for _ in range(4)]
        ws = normalize_weights(panels, PageContext())
        assert len(ws) == 4
        for w in ws:
            assert abs(w - 0.25) < 0.01

    def test_single_panel_gets_full_weight(self):
        ws = normalize_weights([_make_panel()], PageContext())
        assert len(ws) == 1
        assert abs(ws[0] - 1.0) < 0.001

    def test_empty_list_returns_empty(self):
        ws = normalize_weights([], PageContext())
        assert ws == []
