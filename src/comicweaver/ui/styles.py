"""自定义CSS样式 - 让Gradio界面更接近最终产品形态。"""

CUSTOM_CSS = """
/* ===== 全局 ===== */
.gradio-container {
    font-family: 'Inter', 'Segoe UI', 'PingFang SC', sans-serif !important;
    max-width: 1400px !important;
}

/* ===== 头部 ===== */
.cw-header {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 60%, #60a5fa 100%);
    color: white;
    padding: 24px 32px;
    border-radius: 16px;
    margin-bottom: 16px;
    box-shadow: 0 4px 12px rgba(30, 64, 175, 0.2);
}
.cw-header h1 {
    margin: 0;
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.cw-header .subtitle {
    margin-top: 6px;
    font-size: 14px;
    opacity: 0.92;
}

/* ===== 卡片 ===== */
.cw-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

/* ===== Agent状态卡片 ===== */
.cw-agent-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 8px;
    margin: 12px 0;
}
.cw-agent-pill {
    padding: 10px 8px;
    border-radius: 8px;
    text-align: center;
    font-size: 12px;
    font-weight: 500;
    background: #f1f5f9;
    color: #64748b;
    border: 1px solid #e2e8f0;
}
.cw-agent-pill.active {
    background: linear-gradient(135deg, #fbbf24, #f59e0b);
    color: white;
    border-color: #f59e0b;
    animation: cw-pulse 1.5s ease-in-out infinite;
}
.cw-agent-pill.done {
    background: #dcfce7;
    color: #166534;
    border-color: #86efac;
}
.cw-agent-pill.review {
    background: linear-gradient(135deg, #c084fc, #a855f7);
    color: white;
    border-color: #a855f7;
}
@keyframes cw-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

/* ===== 进度日志 ===== */
.cw-log {
    background: #0f172a;
    color: #e2e8f0;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 12px;
    padding: 12px 16px;
    border-radius: 8px;
    max-height: 240px;
    overflow-y: auto;
    line-height: 1.6;
}
.cw-log .log-thinking { color: #c084fc; }
.cw-log .log-progress { color: #60a5fa; }
.cw-log .log-done { color: #86efac; }
.cw-log .log-error { color: #f87171; }
.cw-log .log-checkpoint { color: #fbbf24; font-weight: 600; }

/* ===== 评分条 ===== */
.cw-score-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 0;
}
.cw-score-bar .label { width: 120px; font-size: 13px; color: #475569; }
.cw-score-bar .bar {
    flex: 1;
    height: 8px;
    background: #e2e8f0;
    border-radius: 4px;
    overflow: hidden;
}
.cw-score-bar .fill {
    height: 100%;
    background: linear-gradient(90deg, #10b981, #34d399);
    transition: width 0.3s ease;
}
.cw-score-bar .fill.low { background: linear-gradient(90deg, #f87171, #ef4444); }
.cw-score-bar .fill.mid { background: linear-gradient(90deg, #fbbf24, #f59e0b); }
.cw-score-bar .value { width: 50px; text-align: right; font-weight: 600; color: #1e293b; }

/* ===== 确认面板 ===== */
.cw-checkpoint {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border-left: 4px solid #f59e0b;
    padding: 16px 20px;
    border-radius: 8px;
    margin: 12px 0;
}
.cw-checkpoint h3 {
    margin: 0 0 8px 0;
    color: #92400e;
    font-size: 16px;
}
.cw-checkpoint .preview {
    background: rgba(255, 255, 255, 0.6);
    padding: 8px 12px;
    border-radius: 6px;
    margin-top: 8px;
    font-size: 13px;
    color: #78350f;
}

/* ===== 角色卡片 ===== */
.cw-character-card {
    border: 2px solid #e2e8f0;
    border-radius: 12px;
    padding: 12px;
    background: #ffffff;
    transition: all 0.2s ease;
}
.cw-character-card:hover {
    border-color: #3b82f6;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
}

/* ===== 漫画展示 ===== */
.cw-page-viewer {
    background: #1e293b;
    padding: 24px;
    border-radius: 12px;
    text-align: center;
}

/* ===== 按钮 ===== */
.cw-btn-primary {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
}
.cw-btn-secondary {
    background: #f1f5f9 !important;
    color: #475569 !important;
    border: 1px solid #cbd5e1 !important;
}

/* ===== 标签 ===== */
.cw-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 500;
    background: #dbeafe;
    color: #1e40af;
}
"""
