"""Intent echo and elimination evidence, with durable unattended review."""

from __future__ import annotations

import json
from typing import Any

from taste.config import RULE_COVERAGE_THRESHOLD
from taste.core import StandardsEngine
from taste.errors import DomainError
from taste.interview import clarified_context, dimension_value
from taste.models import MirrorRequest, MirrorResponse
from taste.score import rule_match


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def mirror(engine: StandardsEngine, req: MirrorRequest) -> MirrorResponse:
    attended = req.task_ctx.get("attended", True)
    if not isinstance(attended, bool):
        raise DomainError("invalid_request", "Attendance must be a boolean.")
    context = engine.context(req.profile_id, req.task_ctx)
    clarified = clarified_context(req.task_ctx, req.interview_answers)
    goal = dimension_value(clarified, "goal")
    shape = dimension_value(clarified, "output_shape")
    assumptions = []
    if goal is None:
        goal = "Clarify the task goal"
        assumptions.append("The task goal has not been specified.")
    if shape is None:
        shape = str(req.task_ctx.get("type", "draft")) + " output"
        assumptions.append("The output shape is inferred from the task type and needs owner review.")
    echo = f"Goal: {_text(goal)}. Output shape: {_text(shape)}."
    for row in context.never:
        if rule_match(row["directive"], echo) >= RULE_COVERAGE_THRESHOLD:
            assumptions.append(f"Conflict with approved never-rule {row['id']}: {row['directive']}")
    if not attended:
        engine.store.queue_review(req.profile_id, req.task_ctx, echo, assumptions)
    return MirrorResponse(mirror=echo, assumptions=assumptions, inferred=not attended,
                          gate="required" if attended else "queued")
