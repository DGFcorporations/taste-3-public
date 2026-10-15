import importlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest


def module(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pytest.fail(f"Missing required temporal implementation: {name}")


def test_validity_intervals_start_inclusive_end_exclusive():
    temporal = module("taste.temporal")
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = start + timedelta(days=1)
    assert temporal.in_window({"valid_from": start.isoformat(), "valid_to": end.isoformat()}, start)
    assert not temporal.in_window(
        {"valid_from": start.isoformat(), "valid_to": end.isoformat()}, end
    )
    assert not temporal.in_window({"valid_from": start.isoformat()}, start - timedelta(seconds=1))
    assert temporal.in_window({}, start)


@pytest.mark.parametrize("value", ["invalid", "2026-01-01T00:00:00", datetime(2026, 1, 1), None, 5])
def test_naive_and_invalid_instants_raise_422(value):
    with pytest.raises(Exception) as failure:
        module("taste.temporal").parse_utc(value)
    assert failure.value.code == "invalid_request"
    assert failure.value.status_code == 422


def test_timezone_offset_is_normalized_to_utc():
    assert module("taste.temporal").parse_utc("2026-01-01T01:00:00+01:00") == datetime(
        2026, 1, 1, tzinfo=UTC
    )


def test_drift_reporting_is_explicit_strictly_thirty_days_and_read_only():
    now = datetime(2026, 3, 1, tzinfo=UTC)
    rows = [
        {"id": "stale", "profile_id": "p", "valid_to": (now - timedelta(days=31)).isoformat()},
        {"id": "boundary", "profile_id": "p", "valid_to": (now - timedelta(days=30)).isoformat()},
        {"id": "live", "profile_id": "p", "valid_to": None},
    ]
    store = SimpleNamespace(
        snapshot=lambda profile: SimpleNamespace(
            profile_id=profile, standards=rows, edges=[], revision=2, evaluated_at=now, exemplars=[]
        )
    )
    result = module("taste.temporal").drift_report(store, "p", now=now)
    assert [row["id"] for row in result["stale"]] == ["stale"]
    assert rows[0]["valid_to"] == "2026-01-29T00:00:00+00:00"


def test_history_retains_superseded_nodes_and_relations():
    rows = [
        {"id": "old", "profile_id": "p", "state": "superseded"},
        {"id": "new", "profile_id": "p", "state": "active"},
        {"id": "other", "profile_id": "p", "state": "retired"},
    ]
    edges = [{"profile_id": "p", "src": "new", "dst": "old", "type": "supersedes", "weight": 1.0}]
    store = SimpleNamespace(
        snapshot=lambda profile: SimpleNamespace(
            profile_id=profile,
            standards=rows,
            edges=edges,
            revision=2,
            evaluated_at=datetime(2026, 1, 1, tzinfo=UTC),
            exemplars=[],
        )
    )
    result = module("taste.temporal").history_for(store, "p", "old")
    assert [row["id"] for row in result["standards"]] == ["new", "old"]
    assert result["edges"] == edges


@pytest.mark.parametrize(
    "task_type,expected",
    [
        ("write_thread", ["hook", "body", "close"]),
        ("write_doc", ["lead", "structure", "detail", "close"]),
        ("security_review", ["triage", "findings", "verdict"]),
        ("daily_digest", ["numbers", "flags", "recommendation"]),
        ("decision_brief", ["verdict", "tradeoffs", "recommendation"]),
        ("unknown", ["draft", "refine", "finalize"]),
    ],
)
def test_segment_freedoms_cover_fixed_j1_order(task_type, expected):
    assert module("taste.segments").segments_for(task_type) == expected


def test_task_identity_shared_across_segments_and_attendance():
    key = module("taste.segments").task_key
    assert key({"task_ref": "specific", "ref": "less-specific"}) == "specific"
    assert key({"ref": "reference"}) == "reference"
    assert key({"goal": "Synthetic", "segment": "hook", "attended": True}) == key(
        {"attended": False, "segment": "body", "goal": "Synthetic"}
    )
    assert key({"goal": "Synthetic"}) != key({"goal": "Different"})


def test_local_supersession_closes_windows_and_history_remains_queryable():
    from taste.models import Standard
    from taste.store import Store

    temporal = module("taste.temporal")
    clock = [datetime(2026, 1, 1, tzinfo=UTC)]
    with Store(":memory:", clock=lambda: clock[0]) as store:

        def propose(id, directive):
            return store.propose(
                Standard(
                    id=id,
                    profile_id="p",
                    category="structure",
                    polarity="do",
                    directive=directive,
                    created=clock[0],
                    updated=clock[0],
                )
            )

        old = propose("old", "Begin with a synthetic overview")
        temporal.approve_standard(
            store, "p", old.id, "Synthetic owner reason", True, expected_updated=old.updated
        )
        clock[0] += timedelta(days=1)
        new = propose("new", "Begin with a synthetic finding")
        temporal.approve_standard(
            store,
            "p",
            new.id,
            "Synthetic replacement reason",
            True,
            expected_updated=new.updated,
            supersedes=["old"],
            expected_targets={"old": store.get_standard("p", "old").updated},
        )
        snapshot = store.snapshot("p")
        rows = {row["id"]: row for row in snapshot.standards}
        assert rows["old"]["state"] == "superseded"
        assert rows["old"]["valid_to"] == clock[0].isoformat()
        assert rows["new"]["valid_from"] == clock[0].isoformat()
        assert not temporal.effective_standard(rows["old"], "p", clock[0] - timedelta(hours=1))
        assert temporal.in_window(rows["old"], clock[0] - timedelta(hours=1))
        assert not temporal.in_window(rows["old"], clock[0])
        history = temporal.history_for(store, "p", "old")
        assert [row["id"] for row in history["standards"]] == ["new", "old"]
        assert [(edge["src"], edge["dst"]) for edge in history["edges"]] == [("new", "old")]


def test_approval_without_owner_confirmation_keeps_proposal_ineligible():
    from taste.models import Standard
    from taste.store import Store

    temporal = module("taste.temporal")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    with Store(":memory:", clock=lambda: now) as store:
        proposal = store.propose(
            Standard(
                id="proposal",
                profile_id="p",
                category="structure",
                polarity="do",
                directive="Synthetic directive",
                created=now,
                updated=now,
            )
        )
        revision = store.snapshot("p").revision
        with pytest.raises(Exception) as failure:
            temporal.approve_standard(
                store,
                "p",
                proposal.id,
                "Synthetic reason",
                False,
                expected_updated=proposal.updated,
            )
        assert failure.value.status_code == 409
        snapshot = store.snapshot("p")
        assert snapshot.revision == revision
        assert not temporal.effective_standard(snapshot.standards[0], "p", now)
