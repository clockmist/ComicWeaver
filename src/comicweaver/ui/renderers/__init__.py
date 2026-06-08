"""渲染引擎 — 按数据类型拆分的 HTML 渲染函数。

每个模块负责一类数据的可视化，所有函数返回 HTML 字符串供 Gradio gr.HTML 组件使用。
"""

from .agents import render_agent_outputs
from .characters import render_character_profiles
from .common import _resolve_char_name, _gradio_img_src, _preview_thumbnail, _dict_preview
from .dev import render_dev_log, render_performance_summary
from .images import render_panel_images_gallery
from .layout import render_page_reader, render_comparison_view
from .project import render_project_cards, render_project_info
from .story import render_emotion_curve, render_script_summary
from .storyboard import render_storyboard_detail, render_storyboard_preview
from .workflow import (
    render_agent_grid,
    render_agent_status_bar,
    render_checkpoint,
    render_inline_checkpoint,
    render_log,
    render_phase_jump_buttons,
    render_stream_log,
    render_workflow_left,
)
