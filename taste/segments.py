"""Fixed J1 segmentation and shared, deterministic task identity."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from taste.errors import DomainError

_SEGMENTS = {
    "write_thread": ("hook", "body", "close"),
    "write_doc": ("lead", "structure", "detail", "close"),
    "security_review": ("triage", "findings", "verdict"),
    "daily_digest": ("numbers", "flags", "recommendation"),
    "decision_brief": ("verdict", "tradeoffs", "recommendation"),
}


def segments_for(task_type: str | None) -> list[str]:
    return list(_SEGMENTS.get(task_type or "", ("draft", "refine", "finalize")))


def task_key(ctx: dict[str, Any]) -> str:
    """A segment and attendance mode never reset a task's shared budget."""
    for field in ("task_ref", "ref"):
        value = ctx.get(field)
        if isinstance(value, str) and value.strip():
            return value
    relevant = {key: value for key, value in ctx.items() if key not in {"segment", "attended"}}
    try:
        canonical = json.dumps(
            relevant, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    except (TypeError, ValueError):
        raise DomainError(
            "invalid_request", "Task context must contain valid JSON values."
        ) from None
    return "task_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
