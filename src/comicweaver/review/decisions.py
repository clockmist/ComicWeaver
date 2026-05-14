"""决策器 - 把 quality_score 映射到 PASS/REVISE/ESCALATE。"""
from __future__ import annotations

from comicweaver.core import ReviewDecision


def decide(
    overall_score: float,
    retry_count: int,
    max_retries: int,
    pass_threshold: float = 7.0,
    escalate_threshold: float = 4.0,
) -> ReviewDecision:
    if overall_score >= pass_threshold:
        return ReviewDecision.PASS
    if overall_score < escalate_threshold:
        return ReviewDecision.ESCALATE
    if retry_count >= max_retries:
        return ReviewDecision.ESCALATE
    return ReviewDecision.REVISE
