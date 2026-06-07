"""Tests for pre-image layout target sizing."""
from __future__ import annotations

from comicweaver.core import BoundingBox, PageLayout, PanelPlan
from comicweaver.orchestrator.graph import (
    _generation_size_from_bbox,
    _precompute_layout_targets,
)


def test_generation_size_preserves_wide_bbox_orientation():
    width, height = _generation_size_from_bbox(
        BoundingBox(x=0.0, y=0.0, width=1.0, height=0.25)
    )

    assert width > height
    assert width % 64 == 0
    assert height % 64 == 0


def test_generation_size_preserves_tall_bbox_orientation():
    width, height = _generation_size_from_bbox(
        BoundingBox(x=0.0, y=0.0, width=0.25, height=1.0)
    )

    assert height > width
    assert width % 64 == 0
    assert height % 64 == 0


def test_precompute_layout_targets_writes_panel_bboxes_and_sizes():
    panels = [
        PanelPlan(panel_id=f"p{i}", page_id="page_001", order_in_page=i)
        for i in range(1, 4)
    ]
    page = PageLayout(
        page_id="page_001",
        page_number=1,
        panels=panels,
    )

    layout_grids, sizes = _precompute_layout_targets([page])

    assert len(layout_grids) == 1
    assert {entry["panel_id"] for entry in layout_grids[0]["panels"]} == {"p1", "p2", "p3"}
    assert set(sizes) == {"p1", "p2", "p3"}
    for panel in page.panels:
        assert panel.bbox.width > 0
        assert panel.bbox.height > 0
        assert panel.size_ratio == panel.bbox.width * panel.bbox.height
