"""VLM-based face detection and bubble-position optimization.

Uses a vision-capable LLM to identify character face locations in panel images,
then adjusts bubble positions to avoid occluding those regions.

When VLM is unavailable or fails, the caller falls back to heuristic positioning
(the existing behaviour in bubbles.py).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from comicweaver.config import VLMConfig
    from comicweaver.core.schema import BoundingBox
    from .bubbles import BubblePlacement


async def _run_vlm_in_thread(func, *args) -> Any:
    """Run a sync VLM call in a thread pool to avoid blocking the event loop."""
    return await asyncio.to_thread(func, *args)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class FaceRegion:
    """A detected face with normalised bounding box."""
    char_name: str = ""
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)

    @property
    def area(self) -> float:
        return self.w * self.h


# ---------------------------------------------------------------------------
# VLM detection entry-point
# ---------------------------------------------------------------------------


async def detect_faces_vlm(
    image_path: str,
    vlm_config: "VLMConfig",
) -> list[FaceRegion]:
    """Call the VLM to find faces in *image_path*.

    Returns a list of FaceRegion with normalised [0,1] coordinates.
    Returns an empty list if VLM is unavailable, fails, or finds nothing.
    """
    if not vlm_config.is_available:
        return []

    from comicweaver.api import detect_faces_vlm_api

    raw_faces = await _run_vlm_in_thread(detect_faces_vlm_api, image_path, vlm_config)

    result: list[FaceRegion] = []
    for item in raw_faces:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox") or {}
        try:
            result.append(FaceRegion(
                char_name=str(item.get("char_name", "")),
                x=float(bbox.get("x", 0)),
                y=float(bbox.get("y", 0)),
                w=float(bbox.get("w", 0)),
                h=float(bbox.get("h", 0)),
            ))
        except (ValueError, TypeError):
            continue
    return result


# ---------------------------------------------------------------------------
# Position optimization
# ---------------------------------------------------------------------------


def _distance_to_nearest_face(
    bx: float, by: float, bw: float, bh: float,
    faces: list[FaceRegion],
) -> float:
    """Compute the minimum distance from bubble centre to any face centre."""
    if not faces:
        return float("inf")
    bcx = bx + bw / 2
    bcy = by + bh / 2
    best = float("inf")
    for f in faces:
        dx = bcx - f.center[0]
        dy = bcy - f.center[1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < best:
            best = dist
    return best


def optimize_bubble_positions(
    bubbles: list["BubblePlacement"],
    faces: list[FaceRegion],
    panel_bbox: "BoundingBox",
) -> list["BubblePlacement"]:
    """Adjust bubble positions to avoid occluding detected faces.

    Strategy: try several candidate positions around the panel edges,
    pick the one farthest from any detected face that still keeps the
    bubble inside the panel.
    """
    if not faces:
        return bubbles

    import copy

    # Candidate positions as (anchor_x, anchor_y) relative to panel
    # Each placement anchors a corner of the bubble
    candidates = [
        ("top_left",      0.02, 0.02),
        ("top_right",     0.55, 0.02),
        ("bottom_left",   0.02, 0.65),
        ("bottom_right",  0.55, 0.65),
        ("top_center",    0.30, 0.02),
        ("bottom_center", 0.30, 0.65),
    ]

    result: list[BubblePlacement] = []
    for bp in bubbles:
        best_bubble = copy.copy(bp)
        best_dist = _distance_to_nearest_face(bp.x, bp.y, bp.w, bp.h, faces)

        for _label, ax, ay in candidates:
            bx = panel_bbox.x + panel_bbox.width * ax
            by = panel_bbox.y + panel_bbox.height * ay

            # Clamp inside panel
            bx = max(panel_bbox.x + 0.01, min(bx, panel_bbox.x + panel_bbox.width - bp.w - 0.01))
            by = max(panel_bbox.y + 0.01, min(by, panel_bbox.y + panel_bbox.height - bp.h - 0.01))

            dist = _distance_to_nearest_face(bx, by, bp.w, bp.h, faces)
            if dist > best_dist:
                best_dist = dist
                best_bubble.x = bx
                best_bubble.y = by

        result.append(best_bubble)
    return result
