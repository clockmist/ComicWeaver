"""Application configuration loading.

Configuration is intentionally plain: YAML for shared defaults, optional .env for
local secrets, and environment variables for deployment overrides.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Text-only LLM backend (script, storyboard, review, character tags)."""
    provider: str = "local"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    enabled: bool = False
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout_seconds: float = 60.0

    @property
    def is_available(self) -> bool:
        return self.enabled and bool(self.base_url and self.api_key and self.model)


class VLMConfig(BaseModel):
    """Vision-capable LLM backend (face detection, bubble placement, image QA).

    Shares the same API shape as LLMConfig (OpenAI-compatible /chat/completions)
    but may point to a different provider/model optimized for vision tasks.
    When *enabled* is false or fields are empty, callers fall back to heuristics.
    """
    provider: str = "local"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    enabled: bool = False
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout_seconds: float = 30.0

    @property
    def is_available(self) -> bool:
        return self.enabled and bool(self.base_url and self.api_key and self.model)


class ImageConfig(BaseModel):
    provider: str = "placeholder"
    server_url: str = "http://127.0.0.1:8188"
    model: str = "placeholder"
    enabled: bool = False
    timeout_seconds: float = 180.0
    workflow_character_path: str = "generate_character.json"
    workflow_panel_path: str = "generate_picture.json"
    fallback_to_placeholder: bool = True

    @property
    def is_available(self) -> bool:
        return self.enabled and self.provider != "placeholder"


class StorageConfig(BaseModel):
    projects_dir: str = "projects"
    cache_dir: str = "cache"
    outputs_dir: str = "outputs"


class RuntimeConfig(BaseModel):
    fallback_to_local: bool = True
    request_retries: int = 1


class YoloConfig(BaseModel):
    """YOLO face detection model configuration."""
    model_path: str = "yolo/face_yolov8n.pt"
    confidence_threshold: float = 0.3


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    yolo: YoloConfig = Field(default_factory=YoloConfig)


def _coerce_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def _set_nested(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = data
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def _apply_env(data: dict[str, Any]) -> None:
    mapping: dict[str, tuple[str, Any]] = {
        # LLM (text-only)
        "COMICWEAVER_LLM_PROVIDER": ("llm.provider", str),
        "COMICWEAVER_LLM_BASE_URL": ("llm.base_url", str),
        "COMICWEAVER_LLM_API_KEY": ("llm.api_key", str),
        "COMICWEAVER_LLM_MODEL": ("llm.model", str),
        "COMICWEAVER_LLM_ENABLED": ("llm.enabled", _coerce_bool),
        "COMICWEAVER_LLM_TEMPERATURE": ("llm.temperature", float),
        "COMICWEAVER_LLM_MAX_TOKENS": ("llm.max_tokens", int),
        "COMICWEAVER_LLM_TIMEOUT_SECONDS": ("llm.timeout_seconds", float),
        # VLM (vision)
        "COMICWEAVER_VLM_PROVIDER": ("vlm.provider", str),
        "COMICWEAVER_VLM_BASE_URL": ("vlm.base_url", str),
        "COMICWEAVER_VLM_API_KEY": ("vlm.api_key", str),
        "COMICWEAVER_VLM_MODEL": ("vlm.model", str),
        "COMICWEAVER_VLM_ENABLED": ("vlm.enabled", _coerce_bool),
        "COMICWEAVER_VLM_TEMPERATURE": ("vlm.temperature", float),
        "COMICWEAVER_VLM_MAX_TOKENS": ("vlm.max_tokens", int),
        "COMICWEAVER_VLM_TIMEOUT_SECONDS": ("vlm.timeout_seconds", float),
        # Image
        "COMICWEAVER_IMAGE_PROVIDER": ("image.provider", str),
        "COMICWEAVER_IMAGE_SERVER_URL": ("image.server_url", str),
        "COMICWEAVER_IMAGE_MODEL": ("image.model", str),
        "COMICWEAVER_IMAGE_ENABLED": ("image.enabled", _coerce_bool),
        "COMICWEAVER_IMAGE_TIMEOUT_SECONDS": ("image.timeout_seconds", float),
        "COMICWEAVER_IMAGE_CHARACTER_WORKFLOW": ("image.workflow_character_path", str),
        "COMICWEAVER_IMAGE_PANEL_WORKFLOW": ("image.workflow_panel_path", str),
        "COMICWEAVER_IMAGE_FALLBACK": ("image.fallback_to_placeholder", _coerce_bool),
        # Storage
        "COMICWEAVER_PROJECTS_DIR": ("storage.projects_dir", str),
        "COMICWEAVER_CACHE_DIR": ("storage.cache_dir", str),
        "COMICWEAVER_OUTPUTS_DIR": ("storage.outputs_dir", str),
        # Runtime
        "COMICWEAVER_FALLBACK_TO_LOCAL": ("runtime.fallback_to_local", _coerce_bool),
        "COMICWEAVER_REQUEST_RETRIES": ("runtime.request_retries", int),
        # YOLO
        "COMICWEAVER_YOLO_MODEL_PATH": ("yolo.model_path", str),
        "COMICWEAVER_YOLO_CONFIDENCE": ("yolo.confidence_threshold", float),
    }
    for env_key, (config_key, caster) in mapping.items():
        raw = os.getenv(env_key)
        if raw is None:
            continue
        _set_nested(data, config_key, caster(raw))


@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> AppConfig:
    root = Path.cwd()
    _load_dotenv(root / ".env")
    path = Path(config_path or os.getenv("COMICWEAVER_CONFIG", "configs/comicweaver.yaml"))
    if not path.is_absolute():
        path = root / path
    data = _load_yaml(path)
    _apply_env(data)
    return AppConfig.model_validate(data)


def reload_config(config_path: str | None = None) -> AppConfig:
    load_config.cache_clear()
    return load_config(config_path)
