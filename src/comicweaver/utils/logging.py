"""结构化开发者日志。

通过 LangGraph 的 get_stream_writer() 通道发射 DevLogEntry，
在 UI 的 "开发者日志" Tab 中分类展示。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
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


@dataclass
class DevLogEntry:
    """一条结构化开发者日志条目。

    通过 stream writer 发射到 UI，同时累积到 ComicState["dev_log"] 中。
    """
    level: DevLogLevel
    category: DevLogCategory
    agent: str                     # 关联的 agent 名称（workflow 级别用 "workflow"）
    message: str                   # 人类可读的摘要
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


# ============================================================================
# 工厂函数 —— 便捷创建各类型日志条目
# ============================================================================


def log_agent_input(agent: str, input_data: dict) -> DevLogEntry:
    """Agent 输入日志。"""
    # 截断过长的文本字段以便日志可读
    display = {}
    for k, v in input_data.items():
        if isinstance(v, str) and len(v) > 200:
            display[k] = v[:200] + f"... [{len(v)} chars]"
        elif isinstance(v, list):
            display[k] = f"[{len(v)} items]"
        elif isinstance(v, dict) and len(v) > 10:
            display[k] = f"{{... {len(v)} keys}}"
        else:
            display[k] = v
    return DevLogEntry(
        level=DevLogLevel.DEBUG,
        category=DevLogCategory.AGENT_INPUT,
        agent=agent,
        message=f"{agent} 输入参数",
        data=display,
    )


def log_agent_output(agent: str, output_summary: dict) -> DevLogEntry:
    """Agent 输出摘要日志。"""
    return DevLogEntry(
        level=DevLogLevel.DEBUG,
        category=DevLogCategory.AGENT_OUTPUT,
        agent=agent,
        message=f"{agent} 输出结果",
        data=output_summary,
    )


def log_agent_error(agent: str, error: str, context: dict | None = None) -> DevLogEntry:
    """Agent 错误日志。"""
    return DevLogEntry(
        level=DevLogLevel.ERROR,
        category=DevLogCategory.ERROR,
        agent=agent,
        message=f"{agent} 出错: {error[:120]}",
        data=context or {},
    )


def log_fallback(agent: str, from_backend: str, to_backend: str) -> DevLogEntry:
    """Fallback 降级日志。"""
    return DevLogEntry(
        level=DevLogLevel.WARN,
        category=DevLogCategory.AGENT_OUTPUT,
        agent=agent,
        message=f"{agent} 回退: {from_backend} → {to_backend}",
        data={"from": from_backend, "to": to_backend},
    )


def log_state_change(from_phase: str, to_phase: str) -> DevLogEntry:
    """状态转换日志。"""
    return DevLogEntry(
        level=DevLogLevel.INFO,
        category=DevLogCategory.STATE_CHANGE,
        agent="workflow",
        message=f"阶段转换: {from_phase} → {to_phase}",
        data={"from": from_phase, "to": to_phase},
    )


def log_review(agent: str, decision: str, score: float,
               dimension_scores: dict | None = None,
               issues: list | None = None) -> DevLogEntry:
    """审查决策日志。"""
    return DevLogEntry(
        level=DevLogLevel.INFO,
        category=DevLogCategory.REVIEW_DECISION,
        agent=agent,
        message=f"审查 {agent}: {decision} (score={score:.1f})",
        data={
            "decision": decision,
            "score": score,
            "dimension_scores": dimension_scores or {},
            "issues": issues or [],
        },
    )


def log_performance(agent: str, duration_ms: float, retry_count: int = 0,
                    status: str = "ok") -> DevLogEntry:
    """性能计时日志。"""
    return DevLogEntry(
        level=DevLogLevel.PERF,
        category=DevLogCategory.PERFORMANCE,
        agent=agent,
        message=f"{agent}: {duration_ms:.0f}ms (重试{retry_count}次, {status})",
        duration_ms=duration_ms,
        data={
            "agent": agent,
            "duration_ms": duration_ms,
            "retry_count": retry_count,
            "status": status,
        },
    )


def log_checkpoint(checkpoint_id: str, decision: str) -> DevLogEntry:
    """确认点交互日志。"""
    return DevLogEntry(
        level=DevLogLevel.INFO,
        category=DevLogCategory.CHECKPOINT,
        agent="workflow",
        message=f"确认点 {checkpoint_id}: 用户选择了 {decision}",
        data={"checkpoint_id": checkpoint_id, "decision": decision},
    )


def log_comfyui_request(agent: str, kind: str, prompt: str,
                       negative_prompt: str, seed: int, width: int,
                       height: int, workflow_path: str,
                       metadata: dict | None = None) -> DevLogEntry:
    """记录发送给 ComfyUI 的完整生图参数。"""
    return DevLogEntry(
        level=DevLogLevel.INFO,
        category=DevLogCategory.AGENT_OUTPUT,
        agent=agent,
        message=f"🎨 ComfyUI 生图请求 [{kind}] seed={seed} {width}×{height}",
        data={
            "kind": kind,
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
    """记录 ComfyUI 生图结果。"""
    return DevLogEntry(
        level=DevLogLevel.INFO,
        category=DevLogCategory.AGENT_OUTPUT,
        agent=agent,
        message=f"✅ ComfyUI 生图完成 [{kind}] backend={backend} seed={seed}",
        data={
            "kind": kind,
            "image_path": image_path,
            "backend": backend,
            "seed": seed,
        },
    )


def log_workflow_event(event: str, detail: str = "") -> DevLogEntry:
    """工作流级别事件日志。"""
    return DevLogEntry(
        level=DevLogLevel.INFO,
        category=DevLogCategory.WORKFLOW,
        agent="workflow",
        message=event + (f": {detail}" if detail else ""),
        data={"event": event, "detail": detail},
    )


# ============================================================================
# 输出摘要构建 —— 从 Agent 输出中提取关键信息
# ============================================================================


def summarize_script_output(output: dict) -> dict:
    """从 ScriptOutput 构建可读摘要。"""
    return {
        "title": output.get("title", "Untitled"),
        "summary": (output.get("summary", "") or "")[:200],
        "genre": output.get("genre", []),
        "scene_count": len(output.get("scenes", [])),
        "character_count": len(output.get("characters", [])),
        "emotion_curve_length": len(output.get("emotion_curve", [])),
        "narrative_structure": {
            k: len(v) for k, v in
            (output.get("narrative_structure") or {}).items()
            if isinstance(v, list)
        },
        "scene_locations": list({
            s.get("location", "?")
            for s in output.get("scenes", [])
            if s.get("location")
        }),
        "character_names": [
            c.get("name") for c in output.get("characters", [])
        ],
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


def summarize_review_output(output: dict) -> dict:
    """从 ReviewOutput 构建可读摘要。"""
    fb = output.get("feedback") or {}
    sc = output.get("schema_check") or {}
    qc = output.get("quality_check") or {}
    es = output.get("escalation_summary") or {}
    return {
        "decision": fb.get("decision", "?"),
        "overall_score": fb.get("overall_score", 0),
        "dimension_scores": fb.get("dimension_scores", qc.get("dimension_scores", {})),
        "issues": fb.get("issues", qc.get("issues", [])),
        "suggestions": fb.get("suggestions", qc.get("suggestions", [])),
        "strengths": qc.get("strengths", []),
        "confidence": qc.get("confidence", 0),
        "schema_check": {
            "passed": sc.get("passed", True),
            "missing_fields": sc.get("missing_fields", []),
            "invalid_fields": sc.get("invalid_fields", []),
            "errors": sc.get("errors", []),
        },
        "revise_prompt": fb.get("revise_prompt", ""),
        "escalation": {
            "headline": es.get("headline", ""),
            "suggested_action": es.get("suggested_action", ""),
        } if es else None,
    }
