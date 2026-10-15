"""Seed-inclusive one-hop retrieval over one immutable corpus revision."""

from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from taste.config import EDGE_MIN_WEIGHT, RETRIEVAL_LIMIT, SEED_COUNT
from taste.embed import Embedder, MiniLMEmbedder, checked_vector, cosine
from taste.errors import DomainError
from taste.temporal import effective_standard, parse_utc

if TYPE_CHECKING:
    from taste.store import Snapshot, Store


@dataclass(frozen=True)
class RetrievalResult:
    snapshot: Snapshot
    as_of: datetime
    selected: list[dict[str, Any]]
    never: list[dict[str, Any]]
    model_name: str


def context_query(task_ctx: dict[str, Any]) -> str:
    relevant = {
        key: value
        for key, value in task_ctx.items()
        if key not in {"attended", "as_of", "task_ref", "ref"}
    }
    try:
        return json.dumps(
            relevant, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError):
        raise DomainError(
            "invalid_request", "Task context must contain valid JSON values."
        ) from None


class StandardsEngine:
    def __init__(self, store: Store, embedder: Embedder | None = None) -> None:
        self.store = store
        self.embedder = embedder if embedder is not None else MiniLMEmbedder()

    def context(self, profile_id: str, task_ctx: dict[str, Any]) -> RetrievalResult:
        snapshot = self.store.snapshot(profile_id)
        as_of = parse_utc(task_ctx.get("as_of", snapshot.evaluated_at))
        eligible = {
            row["id"]: row
            for row in snapshot.standards
            if effective_standard(row, profile_id, as_of)
        }
        never = [
            dict(row) for id, row in sorted(eligible.items()) if row.get("polarity") == "never"
        ]
        if not eligible:
            return RetrievalResult(snapshot, as_of, [], never, self.embedder.model_name)
        ids = sorted(eligible)
        query = context_query(task_ctx)
        try:
            vectors = self.embedder.embed([query] + [eligible[id]["directive"] for id in ids])
            if len(vectors) != len(ids) + 1:
                raise ValueError("Embedding output count mismatch.")
            query_vector = checked_vector(vectors[0])
            similarities = {
                id: cosine(query_vector, vector)
                for id, vector in zip(ids, vectors[1:], strict=True)
            }
        except Exception:
            raise DomainError(
                "dependency_unavailable", "Semantic embedding is unavailable.", 503
            ) from None
        seeds = heapq.nsmallest(SEED_COUNT, similarities, key=lambda id: (-similarities[id], id))
        scores = {id: similarities[id] for id in seeds}
        seed_set = set(seeds)
        for edge in snapshot.edges:
            if (
                edge.get("profile_id") != profile_id
                or edge.get("src") not in seed_set
                or edge.get("dst") not in eligible
            ):
                continue
            if edge.get("edge_type", edge.get("type")) not in {
                "relates",
                "contradicts",
                "applies_to",
            }:
                continue
            weight = edge.get("weight")
            if (
                not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or not EDGE_MIN_WEIGHT <= weight <= 1
            ):
                continue
            id = edge["dst"]
            score = similarities[edge["src"]] * weight
            scores[id] = max(scores.get(id, -math.inf), score)
        ranked = heapq.nsmallest(RETRIEVAL_LIMIT, scores, key=lambda id: (-scores[id], id))
        selected = [dict(eligible[id], _retrieval_score=scores[id]) for id in ranked]
        return RetrievalResult(snapshot, as_of, selected, never, self.embedder.model_name)
