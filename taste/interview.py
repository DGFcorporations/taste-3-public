"""Deterministic questions for explicitly unresolved task dimensions."""

from __future__ import annotations

import re
from typing import Any

from taste.core import RetrievalResult, StandardsEngine
from taste.models import InterviewQuestion, InterviewRequest, InterviewResponse
from taste.score import tokens

DIMENSIONS = ("goal", "audience", "output_shape", "success_criteria", "constraints")
ALIASES = {
    "goal": ("goal", "intent", "objective"),
    "audience": ("audience", "reader"),
    "output_shape": ("output_shape", "output", "format"),
    "success_criteria": ("success_criteria", "success", "criteria"),
    "constraints": ("constraints", "limits"),
}
_QUESTIONS = {
    "goal": ("What should this task achieve?", ["Explain a result", "Recommend a decision", "Create a draft"]),
    "audience": ("Who will use the result and what do they already know?", ["Operators", "Decision makers", "Technical readers"]),
    "output_shape": ("What output format and length should the result have?", ["Short memo", "Detailed document", "Checklist"]),
    "success_criteria": ("What concrete result would make this work successful?", ["Actionable next step", "Clear explanation", "Evidence-backed decision"]),
    "constraints": ("What limits or requirements must the output respect?", ["Length limit", "Required evidence", "No additional limits"]),
}


def _present(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_present(child) for child in value)
    if isinstance(value, dict):
        return bool(value)
    return isinstance(value, int | float) and not isinstance(value, bool)


def dimension_value(task_ctx: dict[str, Any], dimension: str) -> Any:
    return next((task_ctx[key] for key in ALIASES[dimension]
                 if key in task_ctx and _present(task_ctx[key])), None)


def clarified_context(task_ctx: dict[str, Any], answers: dict[str, str]) -> dict[str, Any]:
    """Interview question IDs fill omissions without replacing explicit intent."""
    clarified = dict(task_ctx)
    for dimension in DIMENSIONS:
        if dimension_value(clarified, dimension) is None:
            answer = dimension_value(answers, dimension)
            if answer is None:
                answer = answers.get("q_" + dimension)
            if _present(answer):
                clarified[dimension] = answer
    return clarified


def applied_rows(context: RetrievalResult) -> list[dict[str, Any]]:
    rows = {row["id"]: row for row in context.selected + context.never}
    return [rows[id] for id in sorted(rows)]


def coverage_for(context: RetrievalResult, task_ctx: dict[str, Any]) -> dict[str, list[str]]:
    """A category by itself cannot fill a task-specific requirement."""
    rows = applied_rows(context)
    irrelevant = task_ctx.get("not_applicable", [])
    excluded = {value for value in irrelevant if isinstance(value, str)} if isinstance(irrelevant, list) else set()
    relevant = [dimension for dimension in DIMENSIONS if dimension not in excluded]
    resolved = {dimension for dimension in DIMENSIONS
                if dimension_value(task_ctx, dimension) is not None}
    for row in rows:
        meta = row.get("variant_meta", {})
        declared = meta.get("coverage", []) if isinstance(meta, dict) else []
        if isinstance(declared, dict):
            declared = declared.get("resolved", [])
        if isinstance(declared, list):
            resolved.update(item for item in declared if isinstance(item, str) and item in DIMENSIONS)
        # These explicit declarations carry information that a broad category lacks.
        for dimension, aliases in ALIASES.items():
            if any(re.match(rf"^{re.escape(alias.replace('_', ' '))}\s*:\s*\S", row["directive"], re.I)
                   for alias in aliases):
                resolved.add(dimension)
        if row["polarity"] == "never":
            resolved.add("constraints")
    ids = {row["id"] for row in rows}
    conflicts: set[tuple[str, str]] = set()
    for edge in context.snapshot.edges:
        if edge.get("edge_type", edge.get("type")) == "contradicts" and edge["src"] in ids and edge["dst"] in ids:
            conflicts.add(tuple(sorted((edge["src"], edge["dst"]))))
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            if {left["polarity"], right["polarity"]} == {"do", "never"} and tokens(left["directive"]) and tokens(left["directive"]) == tokens(right["directive"]):
                conflicts.add((left["id"], right["id"]))
    gaps = [f"conflict:{left}:{right}" for left, right in sorted(conflicts)]
    gaps.extend(dimension for dimension in relevant if dimension not in resolved)
    return {"resolved": [dimension for dimension in relevant if dimension in resolved], "gaps": gaps}


def interview(engine: StandardsEngine, req: InterviewRequest) -> InterviewResponse:
    context = engine.context(req.profile_id, req.task_ctx)
    coverage = coverage_for(context, req.task_ctx)
    questions = []
    for gap in coverage["gaps"][:3]:
        text, options = _QUESTIONS.get(gap, (
            "How should the conflicting approved requirements be reconciled?",
            ["Review both requirements", "Keep the hard constraint"],
        ))
        questions.append(InterviewQuestion(id="q_" + gap, text=text, options=options, resolves=gap))
    return InterviewResponse(questions=questions, coverage=coverage)
