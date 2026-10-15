"""Best-of-three selection with irreversible, task-wide SQLite reservations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .config import HONE_LIMIT, HONE_STOP_SCORE
from .errors import DomainError
from .models import Candidate, HoneResponse, Packet
from .providers import Generator, Judge
from .score import score_packet
from .segments import task_key
from .store import Store


def hone_packet(
    store: Store,
    packet: Packet,
    segment_ctx: dict[str, Any],
    draft: str | None,
    generator: Generator,
    judge: Judge | None = None,
) -> HoneResponse:
    return hone_task(
        store, packet.profile_id, segment_ctx, lambda: packet, draft, generator, judge
    )


def hone_task(
    store: Store,
    profile_id: str,
    segment_ctx: dict[str, Any],
    compile_packet: Callable[[], Packet],
    draft: str | None,
    generator: Generator,
    judge: Judge | None = None,
) -> HoneResponse:
    """Reserve before compilation so dependency failures also consume a slot."""
    identity = task_key(segment_ctx)
    reservation = store.reserve_hone(profile_id, identity, HONE_LIMIT)
    packet = compile_packet()
    try:
        texts = generator.generate(packet, draft, n=3)
        if len(texts) != 3 or not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError("Invalid generation")
    except DomainError:
        raise
    except Exception as exc:
        raise DomainError(
            "dependency_unavailable", "Candidate generation unavailable", 503
        ) from exc
    candidates = store.register_candidates(
        packet.profile_id,
        identity,
        [
            Candidate(label=label, content=text)
            for label, text in zip(("A", "B", "C"), texts, strict=True)
        ],
    )
    scores = score_packet(packet, candidates, judge)
    eligible = [
        (candidate, score)
        for candidate, score in zip(candidates, scores, strict=True)
        if not score.violations
    ]
    if not eligible:
        raise DomainError(
            "no_eligible_candidate", "No generated candidate satisfies hard rules", 409
        )
    best, score = min(eligible, key=lambda pair: (-pair[1].score, pair[0].label))
    return HoneResponse(
        candidates=candidates,
        pick=best.label,
        score=score.score,
        iterate=reservation < HONE_LIMIT and score.score < HONE_STOP_SCORE,
    )
