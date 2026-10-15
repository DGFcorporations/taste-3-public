"""UTC validity policy and read-only temporal views over coherent snapshots."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from taste.config import DRIFT_DAYS
from taste.errors import DomainError

if TYPE_CHECKING:
    from taste.models import Standard
    from taste.store import Store


def parse_utc(value: Any) -> datetime:
    """Reject naive instants instead of silently assigning a local timezone."""
    try:
        instant = (
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            if isinstance(value, str)
            else value
        )
        if (
            not isinstance(instant, datetime)
            or instant.tzinfo is None
            or instant.utcoffset() is None
        ):
            raise ValueError
        return instant.astimezone(UTC)
    except (TypeError, ValueError, OverflowError):
        raise DomainError(
            "invalid_request", "A valid timezone-aware timestamp is required."
        ) from None


def in_window(standard: dict[str, Any], as_of: datetime) -> bool:
    instant = parse_utc(as_of)
    start, end = standard.get("valid_from"), standard.get("valid_to")
    return (start is None or parse_utc(start) <= instant) and (
        end is None or instant < parse_utc(end)
    )


def effective_standard(standard: dict[str, Any], profile_id: str, as_of: datetime) -> bool:
    return (
        standard.get("profile_id") == profile_id
        and standard.get("state") == "active"
        and standard.get("_local_approved") is True
        and in_window(standard, as_of)
    )


def approve_standard(
    store: Store,
    profile_id: str,
    standard_id: str,
    rationale: str,
    owner_confirmed: bool,
    expected_updated: datetime | str | None = None,
    supersedes: list[str] | None = None,
    expected_targets: dict[str, datetime | str] | None = None,
) -> Standard:
    """Store alone owns approval checks, clocks, and the atomic transition."""
    return store.approve(
        profile_id,
        standard_id,
        rationale,
        owner_confirmed=owner_confirmed,
        expected_updated=expected_updated,
        supersedes=supersedes,
        expected_targets=expected_targets,
    )


def history_for(store: Store, profile_id: str, standard_id: str | None = None) -> dict[str, Any]:
    snapshot = store.snapshot(profile_id)
    rows = {row["id"]: row for row in snapshot.standards if row.get("profile_id") == profile_id}
    edges = [
        edge
        for edge in snapshot.edges
        if edge.get("profile_id") == profile_id
        and edge.get("edge_type", edge.get("type")) == "supersedes"
    ]
    ids = set(rows) if standard_id is None else {standard_id} & rows.keys()
    if standard_id is not None:
        # Walk only retained supersession history, not production applicability.
        adjacency: dict[str, set[str]] = {}
        for edge in edges:
            adjacency.setdefault(edge["src"], set()).add(edge["dst"])
            adjacency.setdefault(edge["dst"], set()).add(edge["src"])
        pending = list(ids)
        while pending:
            for related in adjacency.get(pending.pop(), set()):
                if related in rows and related not in ids:
                    ids.add(related)
                    pending.append(related)
    return {
        "profile_id": profile_id,
        "revision": snapshot.revision,
        "standards": [dict(rows[id]) for id in sorted(ids)],
        "edges": [dict(edge) for edge in edges if edge["src"] in ids and edge["dst"] in ids],
    }


def drift_report(store: Store, profile_id: str, *, now: datetime) -> dict[str, Any]:
    """An explicit report never retires or rewrites historical nodes."""
    instant = parse_utc(now)
    snapshot = store.snapshot(profile_id)
    cutoff = instant - timedelta(days=DRIFT_DAYS)
    stale = [
        dict(row)
        for row in snapshot.standards
        if row.get("profile_id") == profile_id
        and row.get("valid_to") is not None
        and parse_utc(row["valid_to"]) < cutoff
    ]
    return {
        "profile_id": profile_id,
        "revision": snapshot.revision,
        "as_of": instant.isoformat(),
        "stale": sorted(stale, key=lambda row: row["id"]),
    }
