"""存储层。"""
from .checkpoint import (
    AGENT_CHECKPOINTS,
    detect_checkpoint_modification,
    get_modified_checkpoints,
    load_agent_checkpoint,
    load_all_checkpoints,
    load_latest_state,
    list_project_checkpoints,
    migrate_legacy_project,
    save_agent_checkpoint,
)
from .paths import cache_root, outputs_root, project_dir, projects_root, root_dir
from .placeholder import (
    generate_character_placeholder,
    generate_panel_placeholder,
)
from .project import (
    list_projects,
    load_project,
    load_project_meta,
    project_to_state,
    save_dev_log,
    save_project,
    state_to_project,
)

__all__ = [
    # paths
    "cache_root",
    "outputs_root",
    "project_dir",
    "projects_root",
    "root_dir",
    # placeholder
    "generate_character_placeholder",
    "generate_panel_placeholder",
    # project
    "list_projects",
    "load_project",
    "load_project_meta",
    "project_to_state",
    "save_dev_log",
    "save_project",
    "state_to_project",
    # checkpoint (v3.0)
    "AGENT_CHECKPOINTS",
    "detect_checkpoint_modification",
    "get_modified_checkpoints",
    "load_agent_checkpoint",
    "load_all_checkpoints",
    "load_latest_state",
    "list_project_checkpoints",
    "migrate_legacy_project",
    "save_agent_checkpoint",
]
