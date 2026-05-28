"""Review agent with schema checks, optional LLM scoring, and local rubric fallback."""
from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator
from dataclasses import asdict

from comicweaver.api import ApiBackendError, OpenAICompatibleLLMClient
from comicweaver.core import (
    AgentContext,
    AgentOutputMeta,
    BaseAgent,
    EscalationSummary,
    QualityCheckResult,
    ReviewDecision,
    ReviewFeedback,
    ReviewInput,
    ReviewOutput,
    SchemaCheckResult,
    StreamEvent,
    StreamEventType,
    UserOption,
)
from comicweaver.review import decide, get_rubric
from comicweaver.review.rubrics_data import Rubric

# ---- 本地打分规则 ----------------------------------------------------------

def _heuristic_script_score(out: dict) -> dict[str, float]:
    n_scenes = len(out.get("scenes", []))
    n_chars = len(out.get("characters", []))
    has_curve = bool(out.get("emotion_curve"))
    base = 7.0
    if n_scenes >= 4 and n_chars >= 2 and has_curve:
        base = 8.0
    return {
        "completeness": base + random.uniform(-0.3, 0.3),
        "coherence": base + random.uniform(-0.3, 0.3),
        "emotion_arc": (8.0 if has_curve else 5.0) + random.uniform(-0.3, 0.3),
        "character_clarity": (7.5 if n_chars >= 2 else 5.5),
        "visualizability": 7.5 + random.uniform(-0.3, 0.3),
    }


def _heuristic_character_score(out: dict) -> dict[str, float]:
    db = out.get("character_db") or {}
    chars = (db.get("characters") if isinstance(db, dict) else {}) or {}
    n = len(chars)
    base = 7.5 if n >= 2 else 6.0
    return {
        "specificity": base + random.uniform(-0.2, 0.4),
        "trait_clarity": base + random.uniform(-0.2, 0.4),
        "style_consistency": 7.5,
        "recognizability": base + random.uniform(-0.5, 0.3),
    }


def _heuristic_storyboard_score(out: dict) -> dict[str, float]:
    pages = out.get("pages", [])
    panels = sum(len(p.get("panels", [])) for p in pages)
    base = 7.5 if panels >= 4 else 6.5
    return {
        "shot_diversity": base + random.uniform(-0.3, 0.3),
        "reading_flow": 7.8,
        "emotion_mapping": base,
        "prompt_quality": base + random.uniform(-0.3, 0.4),
    }


def _heuristic_image_score(out: dict) -> dict[str, float]:
    self_check = out.get("self_check") or {}
    cc = float(self_check.get("character_consistency", 0.7))
    pm = float(self_check.get("matches_prompt", 0.7))
    return {
        "character_consistency": cc * 10,
        "prompt_match": pm * 10,
        "artifact_free": 8.0 if not self_check.get("artifact_warnings") else 5.0,
        "composition": 7.5,
    }


def _heuristic_layout_score(out: dict) -> dict[str, float]:
    metrics = out.get("overall_metrics") or {}
    overflow = int(metrics.get("bubbles_overflow_count", 0))
    base = 7.5 if overflow == 0 else 5.5
    return {
        "bubble_avoidance": base,
        "flow": base + random.uniform(-0.2, 0.4),
        "text_fit": (8.0 if overflow == 0 else 4.5),
        "aesthetics": 7.0,
    }


_SCORERS = {
    "rubric_script_v1": _heuristic_script_score,
    "rubric_character_v1": _heuristic_character_score,
    "rubric_storyboard_v1": _heuristic_storyboard_score,
    "rubric_image_v1": _heuristic_image_score,
    "rubric_layout_v1": _heuristic_layout_score,
}


def _rubric_payload(rubric: Rubric | None) -> dict | None:
    return asdict(rubric) if rubric is not None else None


def _weighted_overall(rubric_id: str, scores: dict[str, float]) -> float:
    rubric = get_rubric(rubric_id)
    if rubric is None:
        return sum(scores.values()) / max(len(scores), 1)
    total = 0.0
    for dim in rubric.dimensions:
        total += scores.get(dim.dim_id, 0.0) * dim.weight
    return total


def _build_revise_prompt(scores: dict[str, float]) -> tuple[list[str], list[str], str]:
    """根据低分维度生成 issues/suggestions/revise_prompt."""
    issues: list[str] = []
    suggestions: list[str] = []
    weak = sorted(scores.items(), key=lambda kv: kv[1])[:2]
    for dim_id, score in weak:
        if score >= 7.0:
            continue
        issues.append(f"维度「{dim_id}」评分较低 ({score:.1f}/10)")
        suggestions.append(_SUGGESTION_BANK.get(dim_id, "请增强该维度的细节与具体性"))

    revise_prompt = ""
    if suggestions:
        revise_prompt = "请重点改进以下方面:\n" + "\n".join(f"- {s}" for s in suggestions)
    return issues, suggestions, revise_prompt


_SUGGESTION_BANK = {
    "completeness": "为每个场景补充缺失的角色动作、地点、对话信息",
    "coherence": "重新检查场景间因果链,确保事件递进合理",
    "emotion_arc": "调整情感曲线,确保有清晰的高潮与低谷",
    "character_clarity": "为每个角色增加显著的视觉识别特征",
    "visualizability": "把抽象情感描述替换为具体的视觉元素",
    "specificity": "扩充角色外观描述至少包含发型/眼睛/服装",
    "trait_clarity": "补充至少1项独特视觉标志(配饰/疤痕/发型)",
    "style_consistency": "对齐当前style_preset的视觉规范",
    "recognizability": "确保角色之间有明显视觉差异",
    "shot_diversity": "至少使用4种不同景别,避免单调",
    "reading_flow": "重新排列panel顺序,确保从左上到右下",
    "emotion_mapping": "高潮场景使用大画幅或splash布局",
    "prompt_quality": "为每个panel提示词加入具体的角色与动作细节",
    "character_consistency": "强化IP-Adapter权重或更换seed",
    "prompt_match": "提高cfg_scale,加重关键词权重",
    "artifact_free": "增加steps,使用更高质量sampler",
    "composition": "重新构图,主体居中或遵循三分法",
    "bubble_avoidance": "重新放置气泡,避开角色脸与关键道具",
    "flow": "重排气泡顺序,跟随阅读方向",
    "text_fit": "拆分长对话或减小字号",
    "aesthetics": "整理边框宽度与gutter间距",
}


# ---- 主 Agent --------------------------------------------------------------

class ReviewerAgent(BaseAgent[ReviewInput, ReviewOutput]):
    """审查 Agent。"""

    name = "reviewer_agent"
    version = "0.2.0-api"
    rubric_id = "rubric_reviewer_meta"

    async def run(self, inputs: ReviewInput, context: AgentContext) -> ReviewOutput:
        await self._sleep_for_demo(0.15)

        # Layer 1: 格式校验(轻量版,只检查必填字段)
        schema_check = self._schema_check(inputs.target_agent, inputs.target_output)

        if not schema_check.passed:
            feedback = ReviewFeedback(
                decision=ReviewDecision.REVISE,
                overall_score=0.0,
                issues=schema_check.errors,
                suggestions=["输出格式校验失败,请按 Schema 重新生成"],
                revise_prompt="\n".join(schema_check.errors),
                rubric_id=inputs.target_rubric_id,
                reviewer_version=self.version,
            )
            return ReviewOutput(
                feedback=feedback,
                schema_check=schema_check,
                meta=AgentOutputMeta(agent=self.name, version=self.version),
            )

        if self.config.llm.is_available:
            try:
                return await self._run_api(inputs, context, schema_check)
            except ApiBackendError:
                if not self.config.runtime.fallback_to_local:
                    raise

        return self._run_local(inputs, schema_check)

    async def _run_api(
        self,
        inputs: ReviewInput,
        context: AgentContext,
        schema_check: SchemaCheckResult,
    ) -> ReviewOutput:
        client = OpenAICompatibleLLMClient(self.config.llm)
        rubric = get_rubric(inputs.target_rubric_id)
        payload = {
            "task": "review_comicweaver_agent_output",
            "input_schema": ReviewInput.model_json_schema(),
            "output_schema": ReviewOutput.model_json_schema(),
            "input": inputs.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
            "rubric": _rubric_payload(rubric),
            "decision_policy": {
                "pass_threshold": inputs.pass_threshold,
                "escalate_threshold": inputs.escalate_threshold,
                "max_retries": inputs.max_retries,
            },
            "requirements": [
                "Return JSON only.",
                "Scores use a 0-10 scale.",
                "feedback.decision must be pass, revise, or escalate.",
                "Include actionable suggestions for weak dimensions.",
            ],
        }
        data = await asyncio.to_thread(
            client.complete_json,
            "You are the ComicWeaver quality reviewer.",
            payload,
        )
        output = ReviewOutput.model_validate(data)
        return output.model_copy(update={
            "schema_check": schema_check,
            "meta": AgentOutputMeta(
                agent=self.name,
                version=self.version,
                inputs_hash=self._inputs_hash(inputs),
                retry_count=context.retry_count,
                self_check_notes=[f"LLM API provider: {self.config.llm.provider}"],
            ),
        })

    def _run_local(
        self,
        inputs: ReviewInput,
        schema_check: SchemaCheckResult,
    ) -> ReviewOutput:
        # Layer 2: 质量评分
        scorer = _SCORERS.get(inputs.target_rubric_id)
        dim_scores = scorer(inputs.target_output) if scorer else {}
        # clamp
        dim_scores = {k: max(0.0, min(10.0, v)) for k, v in dim_scores.items()}
        overall = _weighted_overall(inputs.target_rubric_id, dim_scores)

        decision = decide(
            overall_score=overall,
            retry_count=inputs.retry_count,
            max_retries=inputs.max_retries,
            pass_threshold=inputs.pass_threshold,
            escalate_threshold=inputs.escalate_threshold,
        )

        issues, suggestions, revise_prompt = _build_revise_prompt(dim_scores)

        feedback = ReviewFeedback(
            decision=decision,
            overall_score=round(overall, 2),
            dimension_scores=dim_scores,
            issues=issues,
            suggestions=suggestions,
            revise_prompt=revise_prompt,
            rubric_id=inputs.target_rubric_id,
            reviewer_version=self.version,
        )

        quality = QualityCheckResult(
            overall_score=round(overall, 2),
            dimension_scores=dim_scores,
            rubric_id=inputs.target_rubric_id,
            strengths=[
                f"维度「{k}」评分良好 ({v:.1f})"
                for k, v in dim_scores.items() if v >= 8.0
            ],
            issues=issues,
            suggestions=suggestions,
            confidence=0.7,
        )

        escalation = None
        if decision == ReviewDecision.ESCALATE:
            escalation = EscalationSummary(
                headline=f"{inputs.target_agent} 输出质量不达标,需要您决策",
                user_options=[
                    UserOption(
                        option_id="force_accept",
                        label="接受当前结果",
                        description=f"质量评分 {overall:.1f},确认接受?",
                    ),
                    UserOption(
                        option_id="retry_with_hint",
                        label="补充提示后重试",
                        description="提供额外指引让 Agent 重新尝试",
                        requires_input=True,
                    ),
                    UserOption(
                        option_id="manual_edit",
                        label="手动编辑",
                        description="打开编辑器直接修改",
                        requires_input=True,
                    ),
                ],
                technical_details=str(dim_scores),
                suggested_action="retry_with_hint",
            )

        return ReviewOutput(
            feedback=feedback,
            schema_check=schema_check,
            quality_check=quality,
            escalation_summary=escalation,
            meta=AgentOutputMeta(
                agent=self.name,
                version=self.version,
                inputs_hash=self._inputs_hash(inputs),
            ),
        )

    async def astream(
        self, inputs: ReviewInput, context: AgentContext
    ) -> AsyncIterator[StreamEvent]:
        yield self._make_event(
            StreamEventType.LOG,
            f"审查 {inputs.target_agent} 输出 (Rubric: {inputs.target_rubric_id})...",
        )
        yield self._make_event(StreamEventType.PROGRESS, 0.3)
        await self._sleep_for_demo(0.1)
        yield self._make_event(StreamEventType.THINKING, "格式校验...")
        yield self._make_event(StreamEventType.PROGRESS, 0.6)
        await self._sleep_for_demo(0.1)
        yield self._make_event(StreamEventType.THINKING, "质量评分...")

        output = await self.run(inputs, context)
        yield self._make_event(
            StreamEventType.PARTIAL_OUTPUT,
            {
                "decision": output.feedback.decision.value,
                "score": output.feedback.overall_score,
            },
        )
        yield self._make_event(StreamEventType.PROGRESS, 1.0)
        yield self._make_event(StreamEventType.DONE, output.model_dump())

    # ---- Schema check (轻量版) -------------------------------------------

    _REQUIRED_FIELDS = {
        "script_agent": ["title", "scenes", "characters", "emotion_curve"],
        "character_agent": ["operation"],
        "storyboard_agent": ["pages", "total_panels"],
        "image_agent": ["panel_image"],
        "layout_agent": ["final_pages"],
    }

    def _schema_check(self, target_agent: str, output: dict) -> SchemaCheckResult:
        required = self._REQUIRED_FIELDS.get(target_agent, [])
        missing = [k for k in required if k not in output]
        if missing:
            return SchemaCheckResult(
                passed=False,
                errors=[f"缺少必填字段: {k}" for k in missing],
                missing_fields=missing,
            )
        return SchemaCheckResult(passed=True)
