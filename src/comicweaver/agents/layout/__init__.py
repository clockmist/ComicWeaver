"""Layout engine sub-package — weight calculation, recursive partition solver,
page compositing, and bubble placement.

All modules are independent of LLM calls; the layout engine is purely algorithmic.
Optional LLM enhancement is handled through layout_agent's _run_api path.
"""
from .bubbles import BubblePlacement, place_bubbles
from .compositor import FitStrategy, PanelSlot, compose_page
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
]
