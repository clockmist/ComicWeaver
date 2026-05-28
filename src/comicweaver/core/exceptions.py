"""异常分类 - 对应 docs/agents/00-base.md §4。"""
from __future__ import annotations


class AgentError(Exception):
    """所有Agent异常的基类。"""
    recoverable: bool = True


class SchemaValidationError(AgentError):
    """输出未通过Pydantic校验。可被审查Agent返工修复。"""
    recoverable = True


class ResourceError(AgentError):
    """资源不可用(模型/GPU/API)。触发降级。"""
    recoverable = False


class UserAbortError(AgentError):
    """用户主动中止。终止工作流。"""
    recoverable = False


class EscalationRequired(AgentError):  # noqa: N818 - public API name used by docs/tests
    """审查Agent决定升级到用户。"""
    recoverable = False
