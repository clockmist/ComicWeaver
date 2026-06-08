"""Gradio 前端主入口（重构版）— 纯 UI 骨架，不含业务逻辑。

设计：三层架构
- Layer 1 (app_new.py) : Gradio Blocks 构建 + 事件绑定
- Layer 2 (callbacks/)  : 业务回调函数
- Layer 3 (renderers/)  : HTML 渲染函数
- composables/          : 跨 renderer 组合
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr

from .callbacks.project import (
    create_project,
    list_projects_cards,
    open_project_by_id,
    save_current_project,
)
from .callbacks.results import (
    get_download_files,
    load_all_results,
    load_dev_log_compact,
    navigate_page,
)
from .callbacks.workflow import respond_checkpoint, start_workflow
from .composables.live import render_live_content
from .renderers.layout import render_page_reader
from .renderers.project import render_project_cards, render_project_info
from .renderers.workflow import render_workflow_left
from .session import SESSION
from .styles import CUSTOM_CSS

HEAD_HTML = """<script>
(function() {
    function _findInput(wrapId) {
        var wrap = document.getElementById(wrapId);
        if (!wrap) return null;
        return wrap.querySelector('textarea, input');
    }
    function _setNativeValue(field, value) {
        var proto = field instanceof HTMLTextAreaElement
            ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
        var desc = Object.getOwnPropertyDescriptor(proto, 'value');
        if (desc && desc.set) desc.set.call(field, value);
        else field.value = value;
        field.dispatchEvent(new InputEvent('input', {bubbles: true, inputType: 'insertText', data: value}));
        field.dispatchEvent(new Event('change', {bubbles: true}));
        field.focus();
    }
    document.addEventListener('click', function(e) {
        var card = e.target.closest('.cw-project-card');
        if (!card) return;
        var pid = card.getAttribute('data-project-id');
        if (!pid) return;
        e.preventDefault();
        var field = _findInput('open-project-id-input');
        if (field) {
            _setNativeValue(field, pid);
        }
    });
})();
</script>"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ComicWeaver") as demo:
        # ==== 头部 ====
        gr.HTML("""
        <div class="cw-header">
            <h1>🎨 ComicWeaver</h1>
            <div class="subtitle">
                多Agent协作的自动化漫画创作系统 · v0.4.0
            </div>
        </div>
        """)

        with gr.Tabs() as tabs:
            # ================================================================
            # Tab 1: 项目
            # ================================================================
            with gr.Tab("📋 项目"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### ✨ 创建新项目")
                        user_input = gr.Textbox(
                            label="剧本内容",
                            placeholder="输入一句话主题或完整剧本...",
                            lines=4,
                        )
                        with gr.Row():
                            project_name = gr.Textbox(
                                label="项目名称（可选）",
                                placeholder="留空则自动截取...",
                                scale=2,
                            )
                            target_pages = gr.Slider(
                                label="目标页数", minimum=1, maximum=20,
                                value=4, step=1, scale=1,
                            )
                        with gr.Row():
                            creation_mode = gr.Dropdown(
                                label="创作模式",
                                choices=[("简单", "simple"), ("专业", "detailed"), ("改编", "adaptation")],
                                value="simple", scale=1,
                            )
                            style_preset = gr.Dropdown(
                                label="风格",
                                choices=[("日漫", "manga"), ("美漫", "comic"), ("条漫", "webtoon"), ("写实", "realistic")],
                                value="manga", scale=1,
                            )
                            interaction_mode = gr.Dropdown(
                                label="交互",
                                choices=[("半自动", "semi_auto"), ("全自动", "full_auto"), ("全交互", "full_interactive")],
                                value="semi_auto", scale=1,
                            )
                        create_btn = gr.Button(
                            "🚀 创建项目", variant="primary",
                            elem_classes="cw-btn-primary", size="lg",
                        )
                        create_status = gr.Textbox(label="", interactive=False, elem_id="create-project-status")

                    with gr.Column(scale=3):
                        with gr.Row():
                            gr.Markdown("### 📂 已保存的项目")
                            refresh_projects_btn = gr.Button(
                                "🔄 刷新", scale=0, elem_classes="cw-btn-secondary",
                            )
                        project_cards_html = gr.HTML(render_project_cards([]))
                        with gr.Row():
                            open_project_id = gr.Textbox(
                                label="输入项目ID打开", placeholder="粘贴项目ID...",
                                scale=2, elem_id="open-project-id-input",
                            )
                            open_btn = gr.Button(
                                "📂 打开", scale=0, elem_classes="cw-btn-primary",
                                elem_id="open-project-btn",
                            )
                            save_current_btn = gr.Button(
                                "💾 保存当前", scale=0, elem_classes="cw-btn-secondary",
                            )

                create_btn.click(
                    create_project,
                    inputs=[user_input, project_name, creation_mode, style_preset,
                            interaction_mode, target_pages],
                    outputs=[create_status],
                ).then(
                    list_projects_cards, outputs=[project_cards_html],
                )
                refresh_projects_btn.click(
                    list_projects_cards, outputs=[project_cards_html],
                )
                save_current_btn.click(
                    save_current_project,
                    outputs=[create_status],
                ).then(
                    list_projects_cards, outputs=[project_cards_html],
                )

            # ================================================================
            # Tab 2: 创作
            # ================================================================
            with gr.Tab("⚡ 创作"):
                project_info_html = gr.HTML(render_project_info())

                with gr.Row():
                    start_phase = gr.Dropdown(
                        label="起始阶段",
                        choices=[
                            ("从头开始", "init"),
                            ("故事阶段", "story"),
                            ("角色阶段", "character"),
                            ("剧本阶段", "script"),
                            ("分镜阶段", "storyboard"),
                            ("图像阶段", "image"),
                            ("台词阶段", "bubble"),
                            ("排版阶段", "layout"),
                        ],
                        value="init", scale=1,
                    )
                    start_btn = gr.Button(
                        "▶ 启动工作流", variant="primary",
                        elem_classes="cw-btn-primary", size="lg", scale=2,
                    )
                    workflow_status = gr.Textbox(
                        label="", interactive=False, scale=4,
                        placeholder="就绪 — 请先创建或打开项目",
                    )
                    save_wf_btn = gr.Button(
                        "💾 保存", elem_classes="cw-btn-secondary", scale=0,
                    )

                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        left_panel_html = gr.HTML(
                            render_workflow_left("", [], [], None)
                        )
                        guidance_input = gr.Textbox(
                            label="🎯 用户指导（可选）",
                            placeholder="例如：'让角色看起来更年轻一些'、'给第二页增加一个特写镜头'...",
                            lines=2,
                            interactive=False,
                            elem_classes="cw-guidance-input",
                        )
                        with gr.Row():
                            accept_btn = gr.Button(
                                "✅ 接受并继续", variant="primary",
                                elem_classes="cw-btn-primary", interactive=False,
                            )
                            regen_btn = gr.Button(
                                "↻ 重新生成", elem_classes="cw-btn-secondary",
                                interactive=False,
                            )
                    with gr.Column(scale=1):
                        panel_preview_html = gr.HTML(
                            render_live_content([], [], "")
                        )

                with gr.Accordion("🔧 开发者调试", open=False):
                    with gr.Row():
                        dev_filter_category = gr.Dropdown(
                            label="类别", value="all", scale=1,
                            choices=[("全部","all"),("输入","agent_input"),("输出","agent_output"),
                                     ("状态","state_change"),
                                     ("错误","error"),("性能","performance"),("工作流","workflow")],
                        )
                        dev_filter_level = gr.Dropdown(
                            label="级别", value="all", scale=1,
                            choices=[("全部","all"),("TRACE","trace"),("DEBUG","debug"),
                                     ("INFO","info"),("WARN","warn"),("ERROR","error")],
                        )
                        dev_refresh_btn = gr.Button("🔄 刷新", scale=0)
                    dev_html = gr.HTML()

                wf_outputs = [
                    workflow_status, left_panel_html, panel_preview_html,
                    project_info_html, accept_btn, regen_btn, guidance_input,
                ]

                start_btn.click(
                    start_workflow,
                    inputs=[start_phase],
                    outputs=wf_outputs,
                )
                accept_btn.click(
                    lambda g: (yield from respond_checkpoint("accept", g)),
                    inputs=[guidance_input],
                    outputs=wf_outputs,
                )
                regen_btn.click(
                    lambda g: (yield from respond_checkpoint("regenerate", g)),
                    inputs=[guidance_input],
                    outputs=wf_outputs,
                )
                save_wf_btn.click(
                    save_current_project,
                    outputs=[workflow_status],
                )
                dev_refresh_btn.click(
                    load_dev_log_compact,
                    inputs=[dev_filter_category, dev_filter_level],
                    outputs=[dev_html],
                )

            # ================================================================
            # Tab 3: 预览
            # ================================================================
            with gr.Tab("📖 预览"):
                with gr.Row():
                    load_results_btn = gr.Button("🔄 刷新", variant="primary", scale=0)
                    download_btn = gr.DownloadButton("📥 下载所有页面", variant="secondary", scale=0)

                gr.Markdown("### 🎨 漫画页面")
                page_state = gr.State(0)
                bubble_toggle_state = gr.State(False)
                with gr.Row():
                    prev_btn = gr.Button("◀ 上一页", scale=1)
                    next_btn = gr.Button("下一页 ▶", scale=1)
                    bubble_checkbox = gr.Checkbox(
                        label="显示对话气泡边界", value=False, scale=2,
                    )
                page_reader_html = gr.HTML(render_page_reader([], 0))

                with gr.Accordion("📝 剧本 & 分镜", open=False):
                    with gr.Row():
                        script_html = gr.HTML(scale=1)
                        curve_html = gr.HTML(scale=1)
                    storyboard_html = gr.HTML()
                    comparison_html = gr.HTML()
                with gr.Accordion("👤 角色", open=False):
                    character_profiles_html = gr.HTML()

                load_results_btn.click(
                    load_all_results,
                    inputs=[page_state, bubble_toggle_state],
                    outputs=[script_html, curve_html, storyboard_html,
                              page_reader_html, comparison_html,
                              character_profiles_html],
                )
                prev_btn.click(
                    lambda s, b: navigate_page("prev", s, b),
                    inputs=[page_state, bubble_toggle_state],
                    outputs=[page_reader_html, page_state],
                )
                next_btn.click(
                    lambda s, b: navigate_page("next", s, b),
                    inputs=[page_state, bubble_toggle_state],
                    outputs=[page_reader_html, page_state],
                )
                bubble_checkbox.change(
                    lambda s, b: (render_page_reader(
                        SESSION.state.get("final_pages", []) if SESSION.state else [],
                        s, b), b),
                    inputs=[page_state, bubble_checkbox],
                    outputs=[page_reader_html, bubble_toggle_state],
                )
                download_btn.click(
                    get_download_files,
                    outputs=[download_btn],
                )

        # ==== 跨 Tab 事件注册 ====
        _open_outputs = [create_status, project_cards_html, project_info_html,
                         left_panel_html, panel_preview_html, start_phase, tabs]
        open_btn.click(
            open_project_by_id,
            inputs=[open_project_id],
            outputs=_open_outputs,
        )
        open_project_id.submit(
            open_project_by_id,
            inputs=[open_project_id],
            outputs=_open_outputs,
        )

        gr.HTML("""
        <div style="text-align:center;padding:16px;color:#94a3b8;font-size:12px;">
            ComicWeaver © 2026 · v0.4.0 Claude Code Edition
        </div>
        """)

    return demo


def main() -> None:
    demo = build_ui()
    demo.queue(default_concurrency_limit=4)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=False,
        head=HEAD_HTML,
        css=CUSTOM_CSS,
        allowed_paths=[str(Path.cwd())],
    )


if __name__ == "__main__":
    main()
