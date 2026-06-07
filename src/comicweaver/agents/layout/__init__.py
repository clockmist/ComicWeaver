"""Layout engine sub-package — weight calculation, recursive partition solver,
page compositing, bubble placement, and face detection.

All modules are independent of LLM calls; the layout engine is purely algorithmic.
Optional LLM enhancement is handled through layout_agent's _run_api path.
"""
from .bubbles import BubblePlacement, place_bubbles
from .compositor import FitStrategy, PanelSlot, compose_page
from .face_detector import FaceRegion, detect_faces_vlm, optimize_bubble_positions
from .face_detector_yolo import detect_faces_yolo
from .solver import LayoutRegion, PageGeometry, solve_layout
from .templates import LAYOUT_TEMPLATES, LayoutHint, get_template
from .weights import PageContext, calculate_panel_weight, normalize_weights

__all__ = [
    "PageContext",
    "calculate_panel_weight",
    "normalize_weights",
    "LAYOUT_TEMPLATES",
    "LayoutHint",
    "get_template",
    "LayoutRegion",
    "PageGeometry",
    "solve_layout",
    "FitStrategy",
    "PanelSlot",
    "compose_page",
    "BubblePlacement",
    "place_bubbles",
    "FaceRegion",
    "detect_faces_vlm",
    "detect_faces_yolo",
    "optimize_bubble_positions",
]
