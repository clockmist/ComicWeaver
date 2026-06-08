"""Agent 输出面板渲染。"""
from __future__ import annotations

import html as html_mod

from .common import _dict_preview


def render_agent_outputs(agent_outputs: dict[str, dict]) -> str:
    """渲染所有 Agent 的输出面板（可折叠卡片）。"""
    if not agent_outputs:
        return (
            '<div style="color:#64748b;padding:12px;">工作流尚未运行，无 Agent 输出数据。<br>'
            '请在「创作流程」Tab 启动工作流后再查看。</div>'
        )

    agents_order = [
        ("story_agent", "📖 故事 Agent", "故事创作"),
        ("script_agent", "📝 剧本 Agent", "剧本理解"),
        ("character_agent", "👤 角色 Agent", "角色管理"),
        ("storyboard_agent", "🎬 分镜 Agent", "分镜设计"),
        ("image_agent", "🎨 图像 Agent", "面板图像生成"),
        ("layout_agent", "📐 排版 Agent", "页面排版合成"),
        ("reviewer_agent", "✅ 审查 Agent", "质量审查"),
    ]

    cards = []
    for agent_id, label, subtitle in agents_order:
        outputs = agent_outputs.get(agent_id, [])
        card_content = ""
        if not outputs:
            card_content = '<div style="color:#94a3b8;font-size:13px;padding:8px;">暂无输出数据</div>'
        else:
            for i, out in enumerate(outputs):
                out_id = out.get("panel_id") or out.get("title") or f"#{i + 1}"
                output_data = out.get("output", {})
                card_content += (
                    f'<details style="margin-bottom:4px;" open>'
                    f'<summary style="cursor:pointer;font-size:13px;font-weight:600;'
                    f'color:#1e293b;padding:6px 0;">'
                    f'{html_mod.escape(str(out_id))}</summary>'
                    f'<div style="font-size:12px;font-family:monospace;background:#f8fafc;'
                    f'padding:8px 12px;border-radius:6px;max-height:400px;overflow-y:auto;">'
                    f'{_dict_preview(output_data, max_depth=3)}'
                    f'</div></details>'
                )

        cards.append(f"""
        <div class="cw-agent-output-card">
            <div class="cw-agent-output-header">
                <span style="font-size:18px;">{label}</span>
                <span style="font-size:12px;color:#64748b;">{subtitle}</span>
            </div>
            <div style="padding:8px 12px;">{card_content}</div>
        </div>
        """)

    return f'<div style="display:flex;flex-direction:column;gap:12px;">{"".join(cards)}</div>'
