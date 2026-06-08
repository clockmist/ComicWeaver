"""智能体集合。"""
from .bubble_agent import BubbleAgent
from .character_agent import CharacterAgent
from .image_agent import ImageAgent
from .layout_agent import LayoutAgent
from .script_agent import ScriptAgent
from .story_agent import StoryAgent
from .storyboard_agent import StoryboardAgent

__all__ = [
    "BubbleAgent",
    "CharacterAgent",
    "ImageAgent",
    "LayoutAgent",
    "ScriptAgent",
    "StoryAgent",
    "StoryboardAgent",
]


# Agent 注册表 - 用于编排层动态查找
AGENT_REGISTRY: dict[str, type] = {
    "story_agent": StoryAgent,
    "script_agent": ScriptAgent,
    "character_agent": CharacterAgent,
    "storyboard_agent": StoryboardAgent,
    "image_agent": ImageAgent,
    "bubble_agent": BubbleAgent,
    "layout_agent": LayoutAgent,
}
