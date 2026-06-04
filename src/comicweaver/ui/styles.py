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

/* ===== Agent 输出面板 ===== */
.cw-agent-output-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    overflow: hidden;
    transition: box-shadow 0.2s;
}
.cw-agent-output-card:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.cw-agent-output-header {
    padding: 12px 16px;
    background: #f8fafc;
    border-bottom: 1px solid #e2e8f0;
    display: flex;
    align-items: center;
    gap: 12px;
}

/* ===== 开发者日志 ===== */
.cw-dev-log {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 16px;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 12px;
}

/* ===== 筛选按钮组 ===== */
.cw-filter-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 10px;
}
.cw-filter-btn {
    padding: 4px 10px;
    border-radius: 14px;
    border: 1px solid #cbd5e1;
    background: #f8fafc;
    font-size: 11px;
    cursor: pointer;
    color: #475569;
    transition: all 0.15s;
}
.cw-filter-btn:hover {
    background: #e2e8f0;
    border-color: #94a3b8;
}
.cw-filter-btn.active {
    background: #1e40af;
    color: white;
    border-color: #1e40af;
    font-weight: 600;
}

/* ===== 状态统计卡片 ===== */
.cw-stat-row {
    display: flex;
    gap: 16px;
    margin-bottom: 12px;
    flex-wrap: wrap;
}
.cw-stat-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 16px;
    min-width: 100px;
    text-align: center;
}
.cw-stat-card .number {
    font-size: 24px;
    font-weight: 700;
    color: #1e293b;
}
.cw-stat-card .label {
    font-size: 11px;
    color: #64748b;
    margin-top: 2px;
}

/* ===== 页面阅读器 ===== */
.cw-page-reader {
    background: #f8fafc;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}
.cw-page-nav {
    display: flex;
    justify-content: center;
    gap: 16px;
    margin-top: 16px;
}
.cw-page-nav-btn {
    padding: 8px 20px;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    background: white;
    cursor: pointer;
    font-size: 14px;
    color: #374151;
    transition: all 0.2s;
}
.cw-page-nav-btn:hover:not(:disabled) {
    background: #3b82f6;
    color: white;
    border-color: #3b82f6;
}
.cw-page-nav-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
}

/* ===== 分镜对比 ===== */
.cw-comparison-grid {
    padding: 4px;
}
.cw-comparison-row {
    display: flex;
    gap: 16px;
    margin-bottom: 16px;
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px;
}

/* ===== 气泡叠加层 ===== */
.cw-bubble-overlay {
    position: absolute;
    border: 2px dashed #f59e0b;
    border-radius: 4px;
    pointer-events: none;
    min-width: 20px;
    min-height: 20px;
}
.cw-bubble-text {
    position: absolute;
    top: 2px;
    left: 2px;
    font-size: 8px;
    color: #92400e;
    background: rgba(255,255,255,0.85);
    padding: 1px 3px;
    border-radius: 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 90%;
}

/* ===== 实时面板预览 ===== */
.cw-live-preview {
    background: #f8fafc;
    border: 1px dashed #cbd5e1;
    border-radius: 8px;
    padding: 8px;
    min-height: 48px;
}

/* ===== v0.3: Agent 状态条（紧凑水平）===== */
.cw-agent-status {
    display: flex;
    gap: 6px;
    padding: 6px 0;
    flex-wrap: wrap;
}
.cw-as-pill {
    padding: 4px 12px;
    border-radius: 14px;
    font-size: 12px;
    font-weight: 500;
    transition: all 0.3s;
}
.cw-as-pending { background: #f1f5f9; color: #94a3b8; }
.cw-as-active {
    background: linear-gradient(135deg, #fbbf24, #f59e0b);
    color: #1e293b;
    animation: cw-pulse 1.5s ease-in-out infinite;
}
.cw-as-done { background: #dcfce7; color: #166534; }

/* ===== v0.3: 流式日志（Claude Code 暗色终端风格）===== */
.cw-stream-log {
    background: #0d1117;
    color: #c9d1d9;
    font-family: 'JetBrains Mono', 'Consolas', 'Cascadia Code', monospace;
    font-size: 12.5px;
    padding: 12px;
    border-radius: 10px;
    min-height: 280px;
    max-height: 480px;
    overflow-y: auto;
    line-height: 1.7;
    border: 1px solid #30363d;
}
.cw-stream-empty {
    color: #8b949e;
    padding: 20px;
    text-align: center;
    font-style: italic;
}
.cw-stream-entry {
    padding: 2px 0;
    border-bottom: 1px solid #161b22;
}
.cw-stream-time { color: #484f58; font-size: 10px; margin-right: 6px; }
.cw-stream-icon { margin-right: 6px; }
.cw-stream-content { color: #c9d1d9; }
.cw-stream-thinking .cw-stream-content { color: #a371f7; }
.cw-stream-done .cw-stream-content { color: #3fb950; font-weight: 600; }
.cw-stream-error .cw-stream-content { color: #f85149; font-weight: 600; }
.cw-stream-checkpoint {
    background: rgba(210,153,34,0.12);
    border-left: 3px solid #d29922;
    padding-left: 8px;
    margin: 4px 0;
}
.cw-stream-checkpoint .cw-stream-content { color: #e3b341; font-weight: 600; }
.cw-stream-debug .cw-stream-content { color: #8b949e; font-size: 11px; }
.cw-stream-partial .cw-stream-content { color: #79c0ff; }
.cw-stream-info .cw-stream-content { color: #8b949e; }

/* ===== v0.3: 项目信息栏 ===== */
.cw-proj-bar {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 10px 16px;
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border-radius: 10px;
    color: white;
    font-size: 13px;
    margin-bottom: 10px;
}
.cw-proj-bar-title { font-weight: 700; font-size: 15px; }
.cw-proj-bar-id { opacity: 0.65; font-size: 11px; font-family: monospace; }
.cw-proj-bar-phase {
    margin-left: auto;
    background: rgba(255,255,255,0.15);
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 11px;
}

/* ===== v0.3: 项目卡片网格 ===== */
.cw-project-cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 10px;
}
.cw-project-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px;
    cursor: pointer;
    transition: all 0.2s;
}
.cw-project-card:hover {
    border-color: #3b82f6;
    box-shadow: 0 4px 12px rgba(59,130,246,0.12);
    transform: translateY(-1px);
}
.cw-project-card-title {
    font-weight: 600;
    font-size: 14px;
    color: #1e293b;
    margin-bottom: 4px;
}
.cw-project-card-id {
    font-size: 11px;
    color: #94a3b8;
    font-family: monospace;
    margin-bottom: 8px;
}
.cw-project-card-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* ===== v0.3: 内联 Checkpoint 面板 ===== */
.cw-inline-checkpoint {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border-left: 4px solid #f59e0b;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 8px 0;
    animation: cw-fade-in 0.3s ease;
}
.cw-checkpoint-header { font-size: 14px; color: #92400e; margin-bottom: 4px; }
.cw-checkpoint-meta { font-size: 11px; color: #78350f; opacity: 0.8; }
.cw-checkpoint-hint { font-size: 11px; color: #92400e; margin-top: 6px; font-style: italic; }
@keyframes cw-fade-in {
    from { opacity: 0; transform: translateY(-4px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ===== v0.3: 调试面板（折叠）===== */
.cw-debug-toggle {
    padding: 6px 14px;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    background: #f8fafc;
    cursor: pointer;
    font-size: 12px;
    color: #64748b;
    transition: all 0.15s;
}
.cw-debug-toggle:hover { background: #e2e8f0; }
.cw-debug-panel {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px;
    margin-top: 8px;
}

/* ===== v0.3: 工作流左右分栏 ===== */
.cw-workflow-left { min-width: 0; }
.cw-workflow-right { min-width: 0; }
"""
