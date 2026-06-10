"""结构化开发者日志 — 增强中文可读性版本。

通过 LangGraph 的 get_stream_writer() 通道发射 DevLogEntry，
在 UI 的 "开发者日志" Tab 中分类展示。

v2.0: 所有日志消息增加中文标注，提供 write_dev_log_to_file() 持久化。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class DevLogLevel(str, Enum):
    TRACE = "trace"    # 所有细粒度事件
    DEBUG = "debug"    # Agent 输入/输出详情
    INFO = "info"      # 工作流节点转换、阶段开始/结束
    WARN = "warn"      # 重试、fallback、降级
    ERROR = "error"    # 异常、Schema 校验失败
    PERF = "perf"      # 性能计时


class DevLogCategory(str, Enum):
    AGENT_INPUT = "agent_input"
    AGENT_OUTPUT = "agent_output"
    STATE_CHANGE = "state_change"
    REVIEW_DECISION = "review_decision"
    ERROR = "error"
    PERFORMANCE = "performance"
    CHECKPOINT = "checkpoint"
    WORKFLOW = "workflow"


# ── 中英文标签映射 ──────────────────────────────────────────────

_AGENT_CN: dict[str, str] = {
    "story_agent": "故事创作Agent",
    "character_agent": "角色设计Agent",
    "script_agent": "剧本生成Agent",
    "storyboard_agent": "分镜规划Agent",
    "image_agent": "图像生成Agent",
    "bubble_agent": "台词气泡Agent",
    "layout_agent": "排版合成Agent",
    "workflow": "工作流引擎",
}

_CATEGORY_CN: dict[str, str] = {
    "agent_input": "📥 输入参数",
    "agent_output": "📤 输出结果",
    "state_change": "🔄 状态变更",
    "review_decision": "🔍 审查决策",
    "error": "❌ 错误",
    "performance": "⏱ 性能计时",
    "checkpoint": "⏸ 用户确认",
    "workflow": "📋 工作流事件",
}

_LEVEL_CN: dict[str, str] = {
    "trace": "追踪",
    "debug": "调试",
    "info": "信息",
    "warn": "警告",
    "error": "错误",
    "perf": "性能",
}

_KIND_CN: dict[str, str] = {
    "character_reference": "角色参考图",
    "character_expression": "角色表情图",
    "character_pose": "角色姿态图",
    "panel": "面板图",
    "panel_retry": "面板重试",
    "cover": "封面图",
}


def _agent_label(agent: str) -> str:
    """获取 Agent 的中文标签。"""
    return _AGENT_CN.get(agent, agent)


def _category_label(category: str) -> str:
    """获取分类的中文标签。"""
    return _CATEGORY_CN.get(category, category)


def _level_label(level: str) -> str:
    """获取级别的中文标签。"""
    return _LEVEL_CN.get(level, level)


@dataclass
class DevLogEntry:
    """一条结构化开发者日志条目。

    通过 stream writer 发射到 UI，同时累积到 ComicState["dev_log"] 中。
    """
    level: DevLogLevel
    category: DevLogCategory
    agent: str                     # 关联的 agent 名称
    message: str                   # 人类可读的摘要（含中文标注）
    data: dict[str, Any] = field(default_factory=dict)  # 结构化详情
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0       # 仅 PERFORMANCE 类别有意义

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "category": self.category.value,
            "agent": self.agent,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DevLogEntry:
        return cls(
            level=DevLogLevel(d["level"]),
            category=DevLogCategory(d["category"]),
            agent=d["agent"],
            message=d["message"],
            data=d.get("data", {}),
            timestamp=d.get("timestamp", time.time()),
            duration_ms=d.get("duration_ms", 0.0),
        )

    def to_readable_line(self) -> str:
        """将日志条目格式化为一条人类可读的中文文本行。"""
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        cat_label = _category_label(self.category.value)
        agent_label = _agent_label(self.agent)
        level_label = _level_label(self.level.value)
        dur = f" [{self.duration_ms:.0f}ms]" if self.duration_ms > 0 else ""

        line = f"[{ts}] {cat_label} [{agent_label}] [{level_label}] {self.message}{dur}"

        # 附加关键数据摘要
        if self.data:
            key_info = _extract_key_info(self.data)
            if key_info:
                line += f"  |  {key_info}"

        return line


def _extract_key_info(data: dict, max_len: int = 120) -> str:
    """从日志 data 中提取关键信息摘要。"""
    parts = []
    # 优先展示这些字段
    priority_keys = [
        "page_count", "panel_count", "character_count", "total_bubbles",
        "decision", "score", "duration_ms", "retry_count", "status",
        "total_panels", "title", "operation", "backend", "seed",
    ]
    for k in priority_keys:
        if k in data:
            v = data[k]
            if isinstance(v, float):
                parts.append(f"{k}={v:.1f}")
            else:
                parts.append(f"{k}={v}")

    # 如果有 error 相关信息优先展示
    if "error_type" in data:
        parts.append(f"错误类型={data['error_type']}")

    result = ", ".join(parts[:8])
    return result[:max_len]


# ============================================================================
# 工厂函数 —— 便捷创建各类型日志条目
# ============================================================================


def log_agent_input(agent: str, input_data: dict) -> DevLogEntry:
    """记录 Agent 输入参数（含中文标签）。"""
    agent_cn = _agent_label(agent)
    display = {}
    for k, v in input_data.items():
        if isinstance(v, str) and len(v) > 200:
            display[k] = v[:200] + f"... [{len(v)} chars]"
        elif isinstance(v, list):
            display[k] = f"[{len(v)} 项]"
        elif isinstance(v, dict) and len(v) > 10:
            display[k] = f"{{... {len(v)} 个键}}"
        else:
            display[k] = v

    # 构建中文摘要
    info_parts = []
    for k, v in display.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            info_parts.append(f"{k}={v}")
        elif isinstance(v, str) and not v.startswith("["):
            info_parts.append(f"{k}={v[:60]}")

    summary = f"{agent_cn} 接收输入参数"
    if info_parts:
        summary += f" ({', '.join(info_parts[:4])})"

    return DevLogEntry(
        level=DevLogLevel.DEBUG,
        category=DevLogCategory.AGENT_INPUT,
        agent=agent,
        message=summary,
        data=display,
    )


def log_agent_output(agent: str, output_summary: dict) -> DevLogEntry:
    """记录 Agent 输出摘要（含中文标签）。"""
    agent_cn = _agent_label(agent)

    # 从输出摘要构建一句话描述
    desc_parts = []
    if "title" in output_summary:
        desc_parts.append(f"标题「{output_summary['title']}」")
    if "page_count" in output_summary:
        desc_parts.append(f"{output_summary['page_count']}页")
    if "panel_count" in output_summary or "total_panels" in output_summary:
        desc_parts.append(f"{output_summary.get('panel_count', output_summary.get('total_panels', 0))}格")
    if "character_count" in output_summary:
        desc_parts.append(f"{output_summary['character_count']}个角色")
    if "total_bubbles" in output_summary:
        desc_parts.append(f"{output_summary['total_bubbles']}个气泡")

    desc = "，".join(desc_parts) if desc_parts else "输出完成"
    return DevLogEntry(
        level=DevLogLevel.DEBUG,
        category=DevLogCategory.AGENT_OUTPUT,
        agent=agent,
        message=f"{agent_cn} 输出结果: {desc}",
        data=output_summary,
    )


def log_agent_error(agent: str, error: str, context: dict | None = None) -> DevLogEntry:
    """记录 Agent 错误（含中文标签）。"""
    agent_cn = _agent_label(agent)
    return DevLogEntry(
        level=DevLogLevel.ERROR,
        category=DevLogCategory.ERROR,
        agent=agent,
        message=f"{agent_cn} 出错: {error[:120]}",
        data=context or {},
    )


def log_fallback(agent: str, from_backend: str, to_backend: str) -> DevLogEntry:
    """记录后端降级（含中文标签）。"""
    agent_cn = _agent_label(agent)
    return DevLogEntry(
        level=DevLogLevel.WARN,
        category=DevLogCategory.AGENT_OUTPUT,
        agent=agent,
        message=f"{agent_cn} 后端降级: {from_backend} → {to_backend}",
        data={"from": from_backend, "to": to_backend},
    )


def log_state_change(from_phase: str, to_phase: str) -> DevLogEntry:
    """记录工作流阶段转换（含中文标签）。"""
    phase_cn = {
        "init": "初始化", "story": "故事创作", "character": "角色设计",
        "script": "剧本生成", "storyboard": "分镜规划", "image": "图像生成",
        "bubble": "台词气泡", "layout": "排版合成",
    }
    from_label = phase_cn.get(from_phase, from_phase)
    to_label = phase_cn.get(to_phase, to_phase)
    return DevLogEntry(
        level=DevLogLevel.INFO,
        category=DevLogCategory.STATE_CHANGE,
        agent="workflow",
        message=f"工作流阶段切换: {from_label} → {to_label}",
        data={"from": from_phase, "to": to_phase},
    )


def log_review(agent: str, decision: str, score: float,
               dimension_scores: dict | None = None,
               issues: list | None = None) -> DevLogEntry:
    """记录审查决策（含中文标签）。"""
    agent_cn = _agent_label(agent)
    decision_cn = {"pass": "✅ 通过", "revise": "🔄 需修改", "escalate": "⚠️ 升级"}.get(decision, decision)
    return DevLogEntry(
        level=DevLogLevel.INFO,
        category=DevLogCategory.REVIEW_DECISION,
        agent=agent,
        message=f"审查 {agent_cn}: {decision_cn} (评分={score:.1f})",
        data={
            "decision": decision,
            "score": score,
            "dimension_scores": dimension_scores or {},
            "issues": issues or [],
        },
    )


def log_performance(agent: str, duration_ms: float, retry_count: int = 0,
                    status: str = "ok") -> DevLogEntry:
    """记录性能计时（含中文标签）。"""
    agent_cn = _agent_label(agent)
    status_cn = {"ok": "正常", "partial": "部分完成"}.get(status, status)
    dur_str = f"{duration_ms:.0f}ms" if duration_ms < 1000 else f"{duration_ms / 1000:.1f}s"
    return DevLogEntry(
        level=DevLogLevel.PERF,
        category=DevLogCategory.PERFORMANCE,
        agent=agent,
        message=f"{agent_cn}: 耗时 {dur_str} (重试{retry_count}次, {status_cn})",
        duration_ms=duration_ms,
        data={
            "agent": agent,
            "duration_ms": duration_ms,
            "retry_count": retry_count,
            "status": status,
        },
    )


def log_checkpoint(checkpoint_id: str, decision: str) -> DevLogEntry:
    """记录用户确认点交互（含中文标签）。"""
    checkpoints_cn = {
        "after_story": "故事确认",
        "after_character": "角色设计确认",
        "after_script": "剧本确认",
        "after_storyboard": "分镜确认",
        "after_image": "图像确认",
        "after_bubble": "台词气泡确认",
        "after_layout": "排版确认",
    }
    cp_label = checkpoints_cn.get(checkpoint_id, checkpoint_id)
    decision_cn = {"accept": "接受并继续", "regenerate": "重新生成"}.get(decision, decision)
    return DevLogEntry(
        level=DevLogLevel.INFO,
        category=DevLogCategory.CHECKPOINT,
        agent="workflow",
        message=f"确认点「{cp_label}」: 用户选择了「{decision_cn}」",
        data={"checkpoint_id": checkpoint_id, "decision": decision},
    )


def log_comfyui_request(agent: str, kind: str, prompt: str,
                       negative_prompt: str, seed: int, width: int,
                       height: int, workflow_path: str,
                       metadata: dict | None = None) -> DevLogEntry:
    """记录发送给 ComfyUI 的完整生图参数（含中文标签）。"""
    kind_cn = _KIND_CN.get(kind, kind)
    return DevLogEntry(
        level=DevLogLevel.INFO,
        category=DevLogCategory.AGENT_OUTPUT,
        agent=agent,
        message=f"🎨 ComfyUI 生图请求 [{kind_cn}] seed={seed} {width}×{height}",
        data={
            "kind": kind,
            "kind_cn": kind_cn,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "workflow_path": workflow_path,
            "metadata": metadata or {},
        },
    )


def log_comfyui_response(agent: str, kind: str, image_path: str,
                         backend: str, seed: int) -> DevLogEntry:
    """记录 ComfyUI 生图结果（含中文标签）。"""
    kind_cn = _KIND_CN.get(kind, kind)
    return DevLogEntry(
        level=DevLogLevel.INFO,
        category=DevLogCategory.AGENT_OUTPUT,
        agent=agent,
        message=f"✅ ComfyUI 生图完成 [{kind_cn}] backend={backend} seed={seed}",
        data={
            "kind": kind,
            "kind_cn": kind_cn,
            "image_path": image_path,
            "backend": backend,
            "seed": seed,
        },
    )


def log_workflow_event(event: str, detail: str = "") -> DevLogEntry:
    """记录工作流级别事件（含中文标签）。"""
    event_cn = {
        "进入阶段": "进入新阶段",
        "阶段完成": "阶段完成",
        "用户指导": "用户提供指导",
    }.get(event, event)
    return DevLogEntry(
        level=DevLogLevel.INFO,
        category=DevLogCategory.WORKFLOW,
        agent="workflow",
        message=f"工作流事件: {event_cn}" + (f" — {detail}" if detail else ""),
        data={"event": event, "detail": detail},
    )


# ============================================================================
# 输出摘要构建 —— 从 Agent 输出中提取关键信息
# ============================================================================


def summarize_script_output(output: dict) -> dict:
    """从 ScriptOutput 构建可读摘要（v1.0 pages 格式）。"""
    pages = output.get("pages", [])
    all_panels = []
    for page in pages:
        all_panels.extend(page.get("panels", []))

    char_ids_in_panels: set[str] = set()
    for p in all_panels:
        cid = p.get("character_id", "")
        if cid:
            char_ids_in_panels.add(cid)

    return {
        "title": output.get("title", "Untitled"),
        "summary": (output.get("summary", "") or "")[:200],
        "genre": output.get("genre", []),
        "page_count": len(pages),
        "panel_count": len(all_panels),
        "character_count": max(
            len(char_ids_in_panels),
            len(output.get("characters", [])),
        ),
        "panel_locations": list({
            p.get("location", "?")
            for p in all_panels
            if p.get("location")
        })[:10],
        "character_names": sorted(char_ids_in_panels),
    }


def summarize_character_output(output: dict) -> dict:
    """从 CharacterOutput 构建可读摘要。"""
    cdb = output.get("character_db") or {}
    characters = cdb.get("characters", {})
    return {
        "operation": output.get("operation", "?"),
        "character_count": len(characters),
        "characters": {
            cid: {
                "name": cp.get("name", cid),
                "visual_traits": cp.get("visual_traits", {}),
                "has_reference": bool(cp.get("base_reference", {}).get("image_path")),
                "reference_source": cp.get("base_reference", {}).get("source", "?"),
                "reference_confidence": cp.get("base_reference", {}).get("confidence", 0),
                "archive_size": len(cp.get("archive", [])),
                "seed": cp.get("seed", 0),
                "appearance_prompt": cp.get("appearance_prompt", ""),
                "gender_tag": cp.get("gender_tag", "1girl"),
                "core_tags": cp.get("core_tags", ""),
            }
            for cid, cp in characters.items()
        },
    }


def summarize_storyboard_output(output: dict) -> dict:
    """从 StoryboardOutput 构建可读摘要。"""
    pages = output.get("pages", [])
    panels = []
    for p in pages:
        for pn in p.get("panels", []):
            panels.append({
                "panel_id": pn.get("panel_id", "?"),
                "page": p.get("page_number", "?"),
                "order": pn.get("order_in_page", "?"),
                "shot": pn.get("shot_size", "?"),
                "angle": pn.get("camera_angle", "?"),
                "characters": pn.get("characters_in_panel", []),
                "action": (pn.get("primary_action", "") or "")[:80],
                "emotion": pn.get("emotion_intensity", 0),
                "pose_hint": pn.get("pose_hint", ""),
                "expression": pn.get("expression", ""),
                "scene_lighting": pn.get("scene_lighting", ""),
                "weather": pn.get("weather", ""),
                "prompt_preview": (pn.get("prompt_pack", {}).get("positive_prompt", "") or "")[:120],
            })
    return {
        "page_count": len(pages),
        "total_panels": output.get("total_panels", 0),
        "layout_templates": [p.get("layout_template") for p in pages],
        "climax_pages": [p.get("page_number") for p in pages if p.get("is_climax_page")],
        "panels_detail": panels,
    }


def summarize_image_output(output: dict) -> dict:
    """从 ImageOutput 构建可读摘要。"""
    pi = output.get("panel_image") or {}
    sc = output.get("self_check") or {}
    prompt_full = pi.get("prompt_used", "") or ""
    negative_full = pi.get("negative_prompt_used", "") or ""
    return {
        "panel_id": pi.get("panel_id", "?"),
        "image_path": pi.get("image_path", ""),
        "backend": pi.get("backend", output.get("backend_used", "?")),
        "width": pi.get("width", 0),
        "height": pi.get("height", 0),
        "seed": pi.get("seed", 0),
        "steps": pi.get("steps", 0),
        "cfg_scale": pi.get("cfg_scale", 0),
        "generation_time_ms": pi.get("generation_time_ms", 0),
        "prompt_full": prompt_full,
        "prompt_preview": prompt_full[:200],
        "negative_prompt_full": negative_full,
        "negative_prompt_preview": negative_full[:150],
        "characters_present": pi.get("characters_present", []),
        "fallback_chain": output.get("fallback_chain", []),
        "self_check": {
            "matches_prompt": sc.get("matches_prompt", 0),
            "character_consistency": sc.get("character_consistency", 0),
            "artifact_warnings": sc.get("artifact_warnings", []),
            "suggested_retry": sc.get("suggested_retry", False),
        },
    }


def summarize_layout_output(output: dict) -> dict:
    """从 LayoutOutput 构建可读摘要。"""
    return {
        "page_count": len(output.get("final_pages", [])),
        "export_count": len(output.get("exports", [])),
        "overall_metrics": output.get("overall_metrics", {}),
        "pages": [
            {
                "page_id": p.get("page_id", "?"),
                "page_number": p.get("page_number", "?"),
                "image_path": p.get("image_path", ""),
                "bubble_count": len(p.get("bubbles", [])),
                "bubbles_detail": [
                    {
                        "panel_id": b.get("panel_id", ""),
                        "text": (b.get("text", "") or "")[:60],
                        "bubble_type": b.get("bubble_type", "speech"),
                        "occlusion": round(b.get("occlusion_score", 0), 2),
                    }
                    for b in p.get("bubbles", [])
                ],
                "layout_warnings": p.get("layout_warnings", []),
            }
            for p in output.get("final_pages", [])
        ],
    }


def summarize_bubble_output(output: dict) -> dict:
    """从 BubbleOutput 构建可读摘要。"""
    placements = output.get("bubble_placements", {})
    total = output.get("total_bubbles", 0)
    face_stats = output.get("face_detection_stats", {})

    panel_summary = {}
    for page_id, bubbles in placements.items():
        for b in bubbles:
            pid = b.get("panel_id", "?")
            if pid not in panel_summary:
                panel_summary[pid] = {"count": 0, "types": []}
            panel_summary[pid]["count"] += 1
            panel_summary[pid]["types"].append(b.get("bubble_type", "speech"))

    return {
        "total_bubbles": total,
        "pages_processed": len(placements),
        "face_detection_stats": face_stats,
        "panel_summary": {
            pid: {
                "count": info["count"],
                "types": list(set(info["types"])),
            }
            for pid, info in panel_summary.items()
        },
    }


# ============================================================================
# 持久化日志文件 —— 将内存中的 dev_log 写入可读文本文件
# ============================================================================


def write_dev_log_to_file(
    dev_entries: list[dict],
    project_dir: str | Path,
    mode: str = "write",
    agent_name: str = "",
) -> Path | None:
    """将开发者日志条目写入项目目录下的 dev_log.txt 文件。

    每条日志格式化为一行带中文标注的可读文本，按 Agent 分段。
    文件路径: {project_dir}/dev_log.txt

    Args:
        dev_entries: DevLogEntry.to_dict() 的列表
        project_dir: 项目目录路径
        mode: "write"(全量写入,默认) 或 "append_agent"(增量追加单个agent)
        agent_name: mode="append_agent" 时指定 agent 名称

    Returns:
        写入的文件路径，如果列表为空则返回 None
    """
    if not dev_entries:
        return None

    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    log_path = project_dir / "dev_log.txt"

    if mode == "append_agent":
        _append_agent_log_section(log_path, dev_entries, agent_name)
    else:
        _write_full_dev_log(log_path, dev_entries)

    return log_path


def _write_full_dev_log(log_path: Path, dev_entries: list[dict]) -> None:
    """全量写入 dev_log.txt（mode="write"）。"""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  ComicWeaver 开发者日志 — 按 Agent 分段展示")
    lines.append(f"  生成时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    lines.append(f"  总条目数: {len(dev_entries)}")
    lines.append("=" * 80)
    lines.append("")

    last_agent = ""
    for e in dev_entries:
        agent = e.get("agent", "workflow")
        if agent != last_agent:
            agent_cn = _agent_label(agent)
            lines.append("")
            lines.append(f"{'─' * 60}")
            lines.append(f"  【{agent_cn}】({agent}) 的日志")
            lines.append(f"{'─' * 60}")
            last_agent = agent

        lines.append(_format_log_entry_line(e))

    lines.append("")
    lines.append("=" * 80)
    lines.append("  日志结束")
    lines.append("=" * 80)

    log_path.write_text("\n".join(lines), encoding="utf-8")


def _append_agent_log_section(
    log_path: Path, dev_entries: list[dict], agent_name: str
) -> None:
    """增量追加单个 agent 的日志段落到 dev_log.txt。

    如果文件不存在，先写入头部。
    """
    agent_cn = _agent_label(agent_name)

    lines: list[str] = []

    if not log_path.exists():
        lines.append("=" * 80)
        lines.append("  ComicWeaver 开发者日志 — 按 Agent 分段展示")
        lines.append(f"  生成时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        lines.append("=" * 80)
        lines.append("")

    lines.append("")
    lines.append(f"{'─' * 60}")
    lines.append(f"  【{agent_cn}】({agent_name}) 的日志")
    lines.append(f"{'─' * 60}")

    for e in dev_entries:
        lines.append(_format_log_entry_line(e))

    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _format_log_entry_line(e: dict) -> str:
    """将单条日志条目格式化为一行可读文本。"""
    ts = time.strftime("%H:%M:%S", time.localtime(e.get("timestamp", time.time())))
    cat_cn = _category_label(e.get("category", ""))
    lvl_cn = _level_label(e.get("level", "info"))
    msg = e.get("message", "")

    line = f"  [{ts}] [{lvl_cn}] {cat_cn}  {msg}"

    dur = e.get("duration_ms", 0)
    if dur > 0:
        dur_str = f"{dur:.0f}ms" if dur < 1000 else f"{dur / 1000:.1f}s"
        line += f"  (耗时: {dur_str})"

    # 重要数据缩进展示
    data = e.get("data", {})
    if data:
        for k in ("page_count", "panel_count", "character_count",
                  "total_bubbles", "decision", "score", "status"):
            if k in data:
                v = data[k]
                if isinstance(v, float):
                    return line + f"\n        ↳ {k} = {v:.2f}"
                else:
                    return line + f"\n        ↳ {k} = {v}"

    return line
