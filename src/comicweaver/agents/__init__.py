"""智能体集合。"""
from .character_agent import CharacterAgent
from .image_agent import ImageAgent
from .layout_agent import LayoutAgent
from .reviewer_agent import ReviewerAgent
from .script_agent import ScriptAgent
from .storyboard_agent import StoryboardAgent

__all__ = [
    "CharacterAgent",
    "ImageAgent",
    "LayoutAgent",
    "ReviewerAgent",
    "ScriptAgent",
    "StoryboardAgent",
]


# Agent 注册表 - 用于编排层动态查找
AGENT_REGISTRY: dict[str, type] = {
    "script_agent": ScriptAgent,
    "character_agent": CharacterAgent,
    "storyboard_agent": StoryboardAgent,
    "image_agent": ImageAgent,
    "layout_agent": LayoutAgent,
    "reviewer_agent": ReviewerAgent,
}
