"""存储层。"""
from .paths import cache_root, outputs_root, project_dir, projects_root, root_dir
from .placeholder import (
    generate_character_placeholder,
    generate_panel_placeholder,
)
from .project import (
    list_projects,
    load_project,
    project_to_state,
    save_dev_log,
    save_project,
    state_to_project,
)

__all__ = [
    "cache_root",
    "outputs_root",
    "project_dir",
    "projects_root",
    "root_dir",
    "generate_character_placeholder",
    "generate_panel_placeholder",
    "list_projects",
    "load_project",
    "project_to_state",
    "save_dev_log",
    "save_project",
    "state_to_project",
]
