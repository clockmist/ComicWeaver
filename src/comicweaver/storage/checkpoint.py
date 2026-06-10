"""Agent 级 checkpoint 持久化 — 将每个 agent 的输入/输出/状态快照独立保存。

v3.0: 每个 agent 完成后在 projects/<id>/checkpoints/<NN_agent>/ 下写入:
  - output.json         ← agent 输出（可手动编辑台词/prompt）
  - input.json          ← agent 输入参数（调试/审计）
  - state_snapshot.json ← 完整 ComicState（断点恢复）
  - metadata.json       ← 时间戳、版本、hash（修改检测）
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from comicweaver.core.state import ComicState

# ── Agent → checkpoint 映射表 ──────────────────────────────────────────
# 按流水线执行顺序排列

AGENT_CHECKPOINTS: dict[str, dict[str, str]] = {
    "story": {
        "folder": "01_story",
        "checkpoint_id": "after_story",
        "output_key": "developed_story",
    },
    "character": {
        "folder": "02_character",
        "checkpoint_id": "after_character",
        "output_key": "character_db",
    },
    "script": {
        "folder": "03_script",
        "checkpoint_id": "after_script",
        "output_key": "structured_script",
    },
    "storyboard": {
        "folder": "04_storyboard",
        "checkpoint_id": "after_storyboard",
        "output_key": "storyboard_plan",
    },
    "image": {
        "folder": "05_image",
        "checkpoint_id": "after_image",
        "output_key": "panel_images",
    },
    "bubble": {
        "folder": "06_bubble",
        "checkpoint_id": "after_bubble",
        "output_key": "bubble_placements",
    },
    "layout": {
        "folder": "07_layout",
        "checkpoint_id": "after_layout",
        "output_key": "final_pages",
    },
}

# 执行顺序（用于确定 checkpoint 先后关系）
_EXECUTION_ORDER: list[str] = [
    "story", "character", "script", "storyboard", "image", "bubble", "layout"
]

# agent_name → 下一个 agent（用于 resume）
_NEXT_AGENT: dict[str, str] = {
    "story": "character",
    "character": "script",
    "script": "storyboard",
    "storyboard": "image",
    "image": "bubble",
    "bubble": "layout",
}


def _sha256(data: str) -> str:
    """计算字符串的 SHA-256 哈希（短格式）。"""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def _write_json(path: Path, data: Any) -> None:
    """写入格式化的 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict | None:
    """读取 JSON 文件，不存在则返回 None。"""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ── 路径 ──────────────────────────────────────────────────────────────

def get_checkpoint_dir(project_id: str, agent_name: str) -> Path:
    """获取指定 agent 的 checkpoint 目录路径。"""
    from .paths import project_dir

    folder = AGENT_CHECKPOINTS[agent_name]["folder"]
    return project_dir(project_id) / "checkpoints" / folder


# ── 保存 ──────────────────────────────────────────────────────────────

def save_agent_checkpoint(
    project_id: str,
    agent_name: str,
    state: ComicState,
    duration_ms: float = 0.0,
    retry_count: int = 0,
    agent_version: str = "",
) -> Path:
    """保存单个 agent 的 checkpoint。

    在 agent 产出结果后立即调用（用户确认之前）。

    Args:
        project_id: 项目 ID
        agent_name: agent 名称（"story"/"script"/...）
        state: 当前完整 ComicState
        duration_ms: agent 执行耗时（毫秒）
        retry_count: 重试次数
        agent_version: agent 版本号

    Returns:
        checkpoint 目录路径
    """
    cp_dir = get_checkpoint_dir(project_id, agent_name)
    cp_dir.mkdir(parents=True, exist_ok=True)

    # 1. output.json — agent 输出
    output_key = AGENT_CHECKPOINTS[agent_name]["output_key"]
    output_data = state.get(output_key, {})

    # 对于 layout agent，也保存 exports
    if agent_name == "layout":
        output_data = {
            "final_pages": state.get("final_pages", []),
            "exports": state.get("exports", []),
        }

    output_json = _serialize_for_json(output_data)
    _write_json(cp_dir / "output.json", output_json)
    output_hash = _sha256(json.dumps(output_json, sort_keys=True, default=str))

    # 2. input.json — agent 输入（从 state 提取相关字段）
    input_data = _extract_agent_input(state, agent_name)
    input_json = _serialize_for_json(input_data)
    _write_json(cp_dir / "input.json", input_json)
    input_hash = _sha256(json.dumps(input_json, sort_keys=True, default=str))

    # 3. state_snapshot.json — 完整状态快照
    snapshot = dict(state)
    _write_json(cp_dir / "state_snapshot.json", _serialize_for_json(snapshot))

    # 4. metadata.json
    metadata = {
        "agent": agent_name,
        "agent_version": agent_version,
        "checkpoint_id": AGENT_CHECKPOINTS[agent_name]["checkpoint_id"],
        "generated_at": time.time(),
        "duration_ms": duration_ms,
        "retry_count": retry_count,
        "input_hash": input_hash,
        "output_hash": output_hash,
    }
    _write_json(cp_dir / "metadata.json", metadata)

    return cp_dir


def _serialize_for_json(obj: Any) -> Any:
    """将 ComicState 中的值转换为 JSON 可序列化格式。

    Pydantic 模型的 model_dump() 已在调用前处理，此处处理
    dict/list/基本类型 的兜底序列化。
    """
    if isinstance(obj, dict):
        return {str(k): _serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_for_json(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "value"):  # Enum
        return obj.value
    return obj


def _extract_agent_input(state: ComicState, agent_name: str) -> dict[str, Any]:
    """从 state 中提取该 agent 的输入参数。

    只提取与该 agent 相关的关键字段，避免保存整个 state。
    """
    # 公共字段（所有 agent 都需要的基本信息）
    common = {
        "project_id": state.get("project_id", ""),
        "creation_mode": state.get("creation_mode", "simple"),
        "style_preset": state.get("style_preset", "manga"),
        "target_pages": state.get("target_pages", 4),
        "interaction_mode": state.get("interaction_mode", "semi_auto"),
    }

    # 各 agent 特定输入
    agent_inputs: dict[str, dict[str, Any]] = {
        "story": {
            "user_input": state.get("user_input", ""),
        },
        "character": {
            "developed_story": state.get("developed_story", {}),
        },
        "script": {
            "developed_story": state.get("developed_story", {}),
            "character_db": state.get("character_db", {}),
        },
        "storyboard": {
            "structured_script": state.get("structured_script", {}),
            "character_db": state.get("character_db", {}),
            "emotion_curve": state.get("emotion_curve", []),
        },
        "image": {
            "storyboard_plan": state.get("storyboard_plan", []),
            "character_db": state.get("character_db", {}),
        },
        "bubble": {
            "structured_script": state.get("structured_script", {}),
            "storyboard_plan": state.get("storyboard_plan", []),
            "panel_images": state.get("panel_images", []),
        },
        "layout": {
            "storyboard_plan": state.get("storyboard_plan", []),
            "panel_images": state.get("panel_images", []),
            "bubble_placements": state.get("bubble_placements", {}),
        },
    }

    specific = agent_inputs.get(agent_name, {})
    return {**common, **specific}


# ── 加载 ──────────────────────────────────────────────────────────────

def load_agent_checkpoint(
    project_id: str, agent_name: str
) -> dict[str, Any] | None:
    """加载单个 agent 的 checkpoint。

    Returns:
        {
            "output": dict,        # output.json 内容
            "input": dict,         # input.json 内容
            "state_snapshot": dict,# state_snapshot.json 内容
            "metadata": dict,      # metadata.json 内容
        }
        如果 checkpoint 不存在则返回 None
    """
    cp_dir = get_checkpoint_dir(project_id, agent_name)
    if not cp_dir.exists():
        return None

    output = _read_json(cp_dir / "output.json")
    inp = _read_json(cp_dir / "input.json")
    snapshot = _read_json(cp_dir / "state_snapshot.json")
    metadata = _read_json(cp_dir / "metadata.json")

    if output is None and snapshot is None:
        return None

    return {
        "output": output or {},
        "input": inp or {},
        "state_snapshot": snapshot or {},
        "metadata": metadata or {},
    }


def load_latest_state(project_id: str) -> ComicState | None:
    """从最新的 checkpoint 重建完整 ComicState。

    按执行顺序逆序查找最新的 state_snapshot 作为基底，
    然后遍历所有 checkpoint，将手动修改过的 output.json 应用到 state 中。

    如果没有任何 checkpoint，返回 None。
    """
    # 1. 找到最新的 state_snapshot 作为基底
    base_snapshot: dict | None = None
    latest_agent: str = ""
    for agent_name in reversed(_EXECUTION_ORDER):
        cp = load_agent_checkpoint(project_id, agent_name)
        if cp and cp["state_snapshot"]:
            base_snapshot = cp["state_snapshot"]
            latest_agent = agent_name
            break

    if base_snapshot is None:
        return None

    # 2. 遍历所有 checkpoint，应用手动修改过的 output.json
    for agent_name in _EXECUTION_ORDER:
        cp_dir = get_checkpoint_dir(project_id, agent_name)
        if not cp_dir.exists():
            continue

        output = _read_json(cp_dir / "output.json")
        metadata = _read_json(cp_dir / "metadata.json")
        if not output or not metadata:
            continue

        # 检查是否被手动修改（或始终应用最新 output 覆盖 snapshot 中的旧数据）
        output_key = AGENT_CHECKPOINTS[agent_name]["output_key"]
        if agent_name == "layout":
            base_snapshot["final_pages"] = output.get("final_pages", [])
            base_snapshot["exports"] = output.get("exports", [])
        else:
            base_snapshot[output_key] = output

    return ComicState(**base_snapshot)  # type: ignore[arg-type]


def load_state_for_resume(
    project_id: str, resume_agent: str
) -> ComicState | None:
    """加载适合从指定 agent 续跑的状态。

    策略:
    1. 以 resume_agent 之前一个 checkpoint 的 state_snapshot 为基底
       （如果 resume_agent 是第一个 agent，使用它的 state_snapshot）
    2. 应用 resume_agent 的 output.json（可能已被手动修改）
    3. 清除 resume_agent 及之后所有 agent 的输出字段

    例如 resume_agent="bubble":
      - 基底: 04_storyboard 的 state_snapshot
      - 应用: 04_storyboard 的 output.json（含手动修改的台词）
      - 清除: bubble_placements, final_pages, exports

    Args:
        project_id: 项目 ID
        resume_agent: 要恢复到的 agent 名称（"story"/"character"/.../"layout"）

    Returns:
        可用于续跑的 ComicState，如果 checkpoint 不存在则返回 None
    """
    # 确定基底 checkpoint（resume_agent 的前一个）
    resume_idx = _EXECUTION_ORDER.index(resume_agent) if resume_agent in _EXECUTION_ORDER else 0
    base_agent = _EXECUTION_ORDER[resume_idx - 1] if resume_idx > 0 else resume_agent

    # 加载基底 snapshot
    cp = load_agent_checkpoint(project_id, base_agent)
    if not cp or not cp["state_snapshot"]:
        return None

    snapshot = cp["state_snapshot"]

    # 如果基底不是 resume_agent 本身，则应用基底 agent 的 output.json（含手动修改）
    if base_agent != resume_agent:
        output_key = AGENT_CHECKPOINTS[base_agent]["output_key"]
        if base_agent == "layout":
            snapshot["final_pages"] = cp["output"].get("final_pages", [])
            snapshot["exports"] = cp["output"].get("exports", [])
        else:
            snapshot[output_key] = cp["output"]

    # 如果 resume_agent 与基底不同（即从非第一个 agent 续跑），
    # 也需要加载 resume_agent 之前的其他 checkpoint 的 output.json
    for agent_name in _EXECUTION_ORDER[:resume_idx]:
        if agent_name == base_agent:
            continue  # 已经处理过
        agent_cp = load_agent_checkpoint(project_id, agent_name)
        if agent_cp and agent_cp["output"]:
            output_key = AGENT_CHECKPOINTS[agent_name]["output_key"]
            if agent_name == "layout":
                snapshot["final_pages"] = agent_cp["output"].get("final_pages", [])
                snapshot["exports"] = agent_cp["output"].get("exports", [])
            else:
                snapshot[output_key] = agent_cp["output"]

    # 清除 resume_agent 及之后的输出字段
    _clear_agent_outputs(snapshot, resume_agent)

    snapshot["current_phase"] = base_agent
    return ComicState(**snapshot)  # type: ignore[arg-type]


def _clear_agent_outputs(state: dict, from_agent: str) -> None:
    """清除指定 agent 及之后所有 agent 的输出字段。"""
    clear_keys: dict[str, list[str]] = {
        "story": ["developed_story", "narrative_structure", "structured_script",
                   "emotion_curve", "character_db", "reference_chain",
                   "storyboard_plan", "layout_grids", "panel_images",
                   "generation_metadata", "bubble_placements", "final_pages", "exports"],
        "character": ["character_db", "reference_chain", "structured_script",
                      "emotion_curve", "storyboard_plan", "layout_grids",
                      "panel_images", "generation_metadata",
                      "bubble_placements", "final_pages", "exports"],
        "script": ["structured_script", "emotion_curve", "storyboard_plan",
                    "layout_grids", "panel_images", "generation_metadata",
                    "bubble_placements", "final_pages", "exports"],
        "storyboard": ["storyboard_plan", "layout_grids", "panel_images",
                       "generation_metadata", "bubble_placements",
                       "final_pages", "exports"],
        "image": ["panel_images", "generation_metadata",
                  "bubble_placements", "final_pages", "exports"],
        "bubble": ["bubble_placements", "final_pages", "exports"],
        "layout": ["final_pages", "exports"],
    }

    keys = clear_keys.get(from_agent, [])
    for k in keys:
        if k in state:
            if isinstance(state.get(k), list):
                state[k] = []
            elif isinstance(state.get(k), dict):
                state[k] = {}
            else:
                state[k] = None


def load_all_checkpoints(project_id: str) -> list[dict[str, Any]]:
    """按执行顺序加载所有已存在的 checkpoint 元信息。

    Returns:
        [{"agent": "story", "folder": "01_story", "metadata": {...}}, ...]
    """
    result: list[dict[str, Any]] = []
    for agent_name in _EXECUTION_ORDER:
        cp = load_agent_checkpoint(project_id, agent_name)
        if cp:
            result.append({
                "agent": agent_name,
                "folder": AGENT_CHECKPOINTS[agent_name]["folder"],
                "checkpoint_id": AGENT_CHECKPOINTS[agent_name]["checkpoint_id"],
                "metadata": cp["metadata"],
            })
    return result


# ── 查询 ──────────────────────────────────────────────────────────────

def list_project_checkpoints(project_id: str) -> list[str]:
    """返回已完成的 checkpoint 文件夹名列表（如 ["01_story", "02_character"]）。"""
    completed: list[str] = []
    for agent_name in _EXECUTION_ORDER:
        cp_dir = get_checkpoint_dir(project_id, agent_name)
        if cp_dir.exists() and (cp_dir / "output.json").exists():
            completed.append(AGENT_CHECKPOINTS[agent_name]["folder"])
    return completed


def detect_checkpoint_modification(
    project_id: str, agent_name: str
) -> bool:
    """检测某个 checkpoint 的 output.json 是否被手动修改过。

    对比当前 output.json 的 hash 与 metadata.json 中记录的 output_hash。
    """
    cp_dir = get_checkpoint_dir(project_id, agent_name)
    metadata = _read_json(cp_dir / "metadata.json")
    output = _read_json(cp_dir / "output.json")
    if not metadata or not output:
        return False

    recorded_hash = metadata.get("output_hash", "")
    current_hash = _sha256(json.dumps(output, sort_keys=True, default=str))
    return current_hash != recorded_hash


def get_modified_checkpoints(project_id: str) -> list[str]:
    """返回所有 output.json 被手动修改过的 agent 名称列表。"""
    modified: list[str] = []
    for agent_name in _EXECUTION_ORDER:
        if detect_checkpoint_modification(project_id, agent_name):
            modified.append(agent_name)
    return modified


# ── 迁移 ──────────────────────────────────────────────────────────────

def migrate_legacy_project(project_id: str) -> bool:
    """将 v1 格式（project.json 内嵌完整 state）的项目迁移到 v2 checkpoint 格式。

    从 project.json 读取完整 state，按 current_phase 生成已完成阶段的 checkpoint。
    如果已经存在 checkpoint 目录则跳过。

    Returns:
        True 表示迁移成功或已经迁移，False 表示失败
    """
    from .project import load_project, save_project

    cp_root = get_checkpoint_dir(project_id, "story").parent
    # 已经迁移过（存在 checkpoints 目录且有内容）
    if cp_root.exists() and list(cp_root.iterdir()):
        return True

    project = load_project(project_id)
    if project is None:
        return False

    state_data = project.state
    if not state_data:
        return False

    state = ComicState(**state_data)  # type: ignore[arg-type]
    current_phase = state.get("current_phase", "init")

    # 找到 current_phase 在执行顺序中的位置，迁移之前的所有 agent
    if current_phase == "init":
        # 没有任何 agent 完成，不需要迁移 checkpoint
        pass
    else:
        # 迁移所有已完成的 agent
        for agent_name in _EXECUTION_ORDER:
            agent_folder = AGENT_CHECKPOINTS[agent_name]["folder"]
            # 如果已经到达 current_phase，说明该 agent 刚完成
            # current_phase 标记的是"刚完成的 agent"，所以需要包含它
            save_agent_checkpoint(
                project_id, agent_name, state,
                duration_ms=0.0,
                retry_count=0,
                agent_version="(migrated)",
            )
            if agent_folder == _phase_to_folder(current_phase):
                break

    # 更新 project.json 为新格式
    project.format_version = 2
    project.latest_checkpoint = _phase_to_folder(current_phase)
    project.checkpoints_completed = list_project_checkpoints(project_id)
    project.state_summary = _build_state_summary(state)
    # v2: 保留启动所需的必要字段，完整状态通过 checkpoints 管理
    project.state = {
        "project_id": state.get("project_id", ""),
        "user_input": state.get("user_input", ""),
        "title": state.get("title", ""),
        "creation_mode": state.get("creation_mode", "simple"),
        "style_preset": state.get("style_preset", "manga"),
        "interaction_mode": state.get("interaction_mode", "semi_auto"),
        "target_pages": state.get("target_pages", 4),
        "current_phase": state.get("current_phase", "init"),
    }
    save_project(project)

    return True


def _phase_to_folder(phase: str) -> str:
    """将 current_phase 值映射为 checkpoint 文件夹名。"""
    phase_map: dict[str, str] = {
        "story": "01_story",
        "character": "02_character",
        "script": "03_script",
        "storyboard": "04_storyboard",
        "image": "05_image",
        "bubble": "06_bubble",
        "layout": "07_layout",
    }
    return phase_map.get(phase, "")


def _build_state_summary(state: ComicState) -> dict:
    """从 ComicState 构建轻量摘要（用于 UI 列表展示）。"""
    script = state.get("structured_script", {})
    pages = script.get("pages", []) if isinstance(script, dict) else []
    all_panels = []
    for p in pages:
        all_panels.extend(p.get("panels", []) if isinstance(p, dict) else [])

    cdb = state.get("character_db", {})
    characters = cdb.get("characters", {}) if isinstance(cdb, dict) else {}

    return {
        "title": state.get("title", ""),
        "user_input": state.get("user_input", ""),
        "current_phase": state.get("current_phase", "init"),
        "page_count": len(pages),
        "panel_count": len(all_panels),
        "character_count": len(characters),
        "interaction_mode": state.get("interaction_mode", "semi_auto"),
        "creation_mode": state.get("creation_mode", "simple"),
        "style_preset": state.get("style_preset", "manga"),
        "target_pages": state.get("target_pages", 4),
    }


def get_agent_next_phase(agent_name: str) -> str:
    """返回指定 agent 的下一个 agent 名称。"""
    return _NEXT_AGENT.get(agent_name, agent_name)


def get_resume_agent(phase: str) -> str:
    """根据 current_phase 确定从哪个 agent 开始恢复。

    例如 phase="character" → resume from "script"
    """
    if phase == "init":
        return "story"
    return _NEXT_AGENT.get(phase, "story")
