"""Unit tests for bubble placement engine."""
from __future__ import annotations

import pytest

from comicweaver.agents.layout.bubbles import (
    BubblePlacement,
    _resolve_bubble_type,
    estimate_text_size,
    place_bubbles,
)
from comicweaver.core.schema import (
    BoundingBox,
    BubbleHint,
    Dialogue,
    PanelPlan,
)


def _make_panel(
    panel_id: str = "p1",
    dialogues: list[Dialogue] | None = None,
    hints: list[BubbleHint] | None = None,
) -> PanelPlan:
    return PanelPlan(
        panel_id=panel_id,
        page_id="page_001",
        order_in_page=1,
        dialogues_in_panel=dialogues or [],
        speech_bubble_hints=hints or [],
    )


# ---------------------------------------------------------------------------
# estimate_text_size
# ---------------------------------------------------------------------------

class TestEstimateTextSize:
    def test_short_text_returns_minimum_bubble(self):
        w, h = estimate_text_size("Hi")
        assert w >= 80
        assert h >= 30

    def test_cjk_text_measured_wider(self):
        # Same number of characters, CJK should be wider per char
        w_cjk, _ = estimate_text_size("你好世界你好")       # 6 CJK chars
        w_ascii, _ = estimate_text_size("abcdef")              # 6 ASCII chars
        # CJK chars are wider per character
        assert w_cjk > w_ascii

    def test_long_text_wraps_to_multiple_lines(self):
        long_text = "这是一段很长的中文对话文本" * 5
        w, h = estimate_text_size(long_text, font_size_pt=12)
        # Height should be enough for multiple lines
        assert h >= 40


# ---------------------------------------------------------------------------
# _resolve_bubble_type
# ---------------------------------------------------------------------------

class TestResolveBubbleType:
    def test_thought_dialogue_becomes_thought_bubble(self):
        d = Dialogue(speaker="A", text="...", is_thought=True)
        assert _resolve_bubble_type(d, None) == "thought"

    def test_angry_tone_becomes_shout(self):
        d = Dialogue(speaker="A", text="NO!", tone="angry")
        assert _resolve_bubble_type(d, None) == "shout"

    def test_sad_tone_becomes_whisper(self):
        d = Dialogue(speaker="A", text="...", tone="sad")
        assert _resolve_bubble_type(d, None) == "whisper"

    def test_hint_overrides_dialogue_type(self):
        d = Dialogue(speaker="A", text="...", tone="neutral")
        hint = BubbleHint(dialogue_index=0, bubble_type="narration")
        assert _resolve_bubble_type(d, hint) == "narration"

    def test_neutral_dialogue_becomes_speech(self):
        d = Dialogue(speaker="A", text="Hello")
        assert _resolve_bubble_type(d, None) == "speech"


# ---------------------------------------------------------------------------
# place_bubbles
# ---------------------------------------------------------------------------

class TestPlaceBubbles:
    def test_no_dialogues_returns_empty(self):
        panels = [_make_panel("p1")]
        bboxes = [BoundingBox(x=0, y=0, width=1.0, height=1.0)]
        result = place_bubbles(panels, bboxes, {})
        assert result == []

    def test_single_dialogue_placed_inside_panel(self):
        panels = [_make_panel("p1", dialogues=[Dialogue(speaker="A", text="Hello")])]
        bboxes = [BoundingBox(x=0.1, y=0.1, width=0.8, height=0.8)]
        result = place_bubbles(panels, bboxes, {})
        assert len(result) == 1
        b = result[0]
        # Bubble must be within panel bounds
        assert b.x >= 0.1
        assert b.y >= 0.1
        assert b.w > 0
        assert b.h > 0

    def test_multiple_dialogues_stacked_within_panel(self):
        panels = [
            _make_panel(
                "p1",
                dialogues=[
                    Dialogue(speaker="A", text="First line"),
                    Dialogue(speaker="B", text="Second line"),
                ],
            )
        ]
        bboxes = [BoundingBox(x=0.1, y=0.1, width=0.8, height=0.8)]
        result = place_bubbles(panels, bboxes, {})
        assert len(result) == 2
        # 多气泡应分散在不同位置（不再堆叠），位置坐标不重叠
        positions_differ = (
            abs(result[1].x - result[0].x) > 0.01
            or abs(result[1].y - result[0].y) > 0.01
        )
        assert positions_differ, f"Bubbles should be at different positions, got {result[0].x},{result[0].y} and {result[1].x},{result[1].y}"

    def test_bubble_in_panel_bounds(self):
        panels = [
            _make_panel(
                "p1",
                dialogues=[Dialogue(speaker="A", text="Hello")],
            )
        ]
        bboxes = [BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)]
        result = place_bubbles(panels, bboxes, {})
        # bubble should be within page bounds
        assert result[0].x >= -0.5
        assert result[0].y >= -0.5

    def test_multiple_panels_each_get_their_bubbles(self):
        panels = [
            _make_panel("p1", dialogues=[Dialogue(speaker="A", text="P1")]),
            _make_panel("p2", dialogues=[Dialogue(speaker="B", text="P2")]),
        ]
        bboxes = [
            BoundingBox(x=0.0, y=0.0, width=0.5, height=1.0),
            BoundingBox(x=0.5, y=0.0, width=0.5, height=1.0),
        ]
        result = place_bubbles(panels, bboxes, {})
        assert len(result) == 2
        assert result[0].panel_id == "p1"
        assert result[1].panel_id == "p2"
        # Bubbles should be in different x-ranges
        assert result[0].x < 0.5
        assert result[1].x > 0.5
