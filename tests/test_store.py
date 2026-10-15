"""Synthetic persistence boundary tests; no owner corpus is accessed."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from importlib import import_module, util

import pytest

from taste.errors import DomainError

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def store_class():
    assert util.find_spec("taste.store") is not None, "persistence module is missing"
    return import_module("taste.store").Store


def standard(id="s1", **changes):
    from taste.models import Standard

    data = dict(
        id=id,
        profile_id="owner",
        category="structure",
        polarity="do",
        directive="Use clear headings",
        rationale="Readable structure",
        confidence=0.8,
        provenance="inference",
        state="provisional",
        supersedes=[],
        created=NOW.isoformat(),
        updated=NOW.isoformat(),
    )
    data.update(changes)
    return Standard.model_validate(data)


def approve(store, item):
    return store.approve(
        item.profile_id,
        item.id,
        "Synthetic owner rationale",
        owner_confirmed=True,
        expected_updated=item.updated,
    )


def test_proposal_cannot_activate_and_snapshot_epoch_is_stable():
    with store_class()(clock=lambda: NOW) as store:
        assert store.snapshot("owner").evaluated_at == datetime(1970, 1, 1, tzinfo=UTC)
        result = store.propose(standard(state="active"))
        assert result.state == "provisional"
        snapshot = store.snapshot("owner")
        assert snapshot.revision == 1
        assert snapshot.evaluated_at == NOW
        assert snapshot.standards[0]["_local_approved"] is False
        assert store.get_standard("other", "s1") is None


def test_approval_requires_confirmation_nonblank_rationale_and_current_revision():
    with store_class()(clock=lambda: NOW) as store:
        item = store.propose(standard())
        for kwargs in [
            dict(),
            dict(owner_confirmed=True),
            dict(owner_confirmed=True, expected_updated=NOW - timedelta(days=1)),
        ]:
            with pytest.raises(DomainError):
                store.approve("owner", "s1", "Synthetic reason", **kwargs)
        with pytest.raises(DomainError):
            store.approve("owner", "s1", " ", owner_confirmed=True, expected_updated=item.updated)
        assert store.snapshot("owner").standards[0]["_local_approved"] is False
        assert approve(store, item).state == "active"
        assert store.snapshot("owner").standards[0]["_local_approved"] is True


def test_supersession_is_atomic_retains_history_and_rejects_cycle():
    ticks = iter(
        [
            NOW,
            NOW + timedelta(seconds=1),
            NOW + timedelta(seconds=2),
            NOW + timedelta(seconds=3),
            NOW + timedelta(seconds=4),
        ]
    )
    with store_class()(clock=lambda: next(ticks)) as store:
        approve(store, store.propose(standard()))
        new = store.propose(standard("s2", directive="Use concise headings"))
        with pytest.raises(DomainError):
            store.approve(
                "owner",
                "s2",
                "reason",
                owner_confirmed=True,
                expected_updated=new.updated,
                supersedes=["missing"],
            )
        assert store.get_standard("owner", "s1").state == "active"
        assert store.get_standard("owner", "s2").state == "provisional"
        active = store.approve(
            "owner",
            "s2",
            "replacement reason",
            owner_confirmed=True,
            expected_updated=new.updated,
            supersedes=["s1"],
            expected_targets={"s1": store.get_standard("owner", "s1").updated},
        )
        assert store.get_standard("owner", "s1").state == "superseded"
        assert active.state == "active"
        rows = store.snapshot("owner").standards
        old_row = next(r for r in rows if r["id"] == "s1")
        new_row = next(r for r in rows if r["id"] == "s2")
        assert old_row["valid_to"] == new_row["valid_from"]
        with pytest.raises(DomainError):
            store.add_edge("owner", "s1", "s2", "supersedes")
        assert len(store.snapshot("owner").edges) == 1


def test_supersession_requires_reviewed_target_revisions_and_rolls_back():
    with store_class()() as store:
        approve(store, store.propose(standard()))
        new = store.propose(standard("s2", directive="Use concise headings"))
        before = store.snapshot("owner")
        with pytest.raises(DomainError) as failure:
            store.approve("owner", "s2", "Synthetic reason", owner_confirmed=True,
                          expected_updated=new.updated, supersedes=["s1"])
        assert failure.value.code == "state_conflict"
        assert store.snapshot("owner") == before


def test_edges_are_profile_scoped_and_weight_is_bounded():
    with store_class()() as store:
        store.propose(standard())
        store.propose(standard("s2", profile_id="other", directive="Other profile"))
        for destination, weight in [("s2", 1), ("s1", float("nan")), ("s1", 2)]:
            with pytest.raises(DomainError):
                store.add_edge("owner", "s1", destination, "relates", weight)
        assert store.snapshot("owner").edges == []


def test_database_checks_reject_blank_decision_rationale(tmp_path):
    path = tmp_path / "store.db"
    with store_class()(path) as store:
        from taste.models import DecisionRecord

        record = DecisionRecord.model_construct(
            id="d1",
            profile_id="owner",
            task_ctx={},
            candidates=[],
            pick="A",
            rationale="  ",
            agent_role="test",
            provenance="owner_pick",
            ts=NOW,
        )
        with pytest.raises(DomainError):
            store.put_decision(record)
        assert store.snapshot("owner").exemplars == []
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] >= 1
        conn.execute("INSERT INTO nodes VALUES ('owner','invalid','decision')")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO decisions VALUES ('owner','invalid',char(9)||char(10),'{}')")


def test_candidates_are_task_scoped_opaque_and_reused_labels_are_ambiguous():
    from taste.models import Candidate

    with store_class()(clock=lambda: NOW) as store:
        one = store.register_candidates("owner", "task", [Candidate(label="A", content="First")])[0]
        again = store.register_candidates("owner", "task", [Candidate(label="A", content="First")])[
            0
        ]
        assert one.content_ref == again.content_ref
        assert one.content_ref != "A"
        store.register_candidates("owner", "task", [Candidate(label="A", content="Second")])
        assert store.resolve_candidate("owner", "task", "A") is None
        assert store.resolve_candidate("owner", "task", one.content_ref).content == "First"
        assert store.resolve_candidate("other", "task", one.content_ref) is None
        assert store.resolve_candidate("owner", "elsewhere", one.content_ref) is None
        assert store.snapshot("owner").revision == 0


def test_feedback_freezes_content_and_never_fills_missing_choice_later():
    from taste.models import Candidate

    with store_class()(clock=lambda: NOW) as store:
        store.register_candidates(
            "owner",
            "task",
            [Candidate(label="A", content="Chosen"), Candidate(label="B", content="Rejected")],
        )
        record, proposals = store.record_feedback(
            "owner", {"task_ref": "task"}, "A", "Prefer clarity", ["B"]
        )
        assert record.rationale == "Prefer clarity"
        assert all(p.state == "provisional" for p in proposals)
        example = store.snapshot("owner").exemplars[0]
        assert example["text"] == "Chosen"
        assert example["rejected"] == ["Rejected"]
        assert example["why"] == "Prefer clarity"
        store.record_feedback("owner", {"task_ref": "missing"}, "Z", "Prefer brevity", [])
        before = store.snapshot("owner")
        store.register_candidates("owner", "missing", [Candidate(label="Z", content="Later")])
        after = store.snapshot("owner")
        assert before.revision == after.revision
        assert before.exemplars == after.exemplars


def test_queue_and_budget_do_not_advance_corpus_and_budget_is_cross_store(tmp_path):
    path = tmp_path / "budget.db"
    with store_class()(path, clock=lambda: NOW) as store:
        store.queue_review("owner", {"goal": "Synthetic"}, "Synthetic mirror", [])
        assert len(store.list_reviews("owner")) == 1
        assert store.list_reviews("other") == []
        assert store.snapshot("owner").revision == 0

    def reserve(_):
        with store_class()(path, clock=lambda: NOW) as connection:
            try:
                return connection.reserve_hone("owner", "task")
            except DomainError as error:
                return error

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(reserve, range(20)))
    assert sorted(r for r in results if isinstance(r, int)) == list(range(1, 16))
    errors = [r for r in results if not isinstance(r, int)]
    assert len(errors) == 5
    assert all(e.code == "hone_budget_exhausted" for e in errors)
    with store_class()(path) as store:
        assert store.snapshot("owner").revision == 0


def test_invalid_feedback_and_duplicate_candidate_labels_are_atomic():
    from taste.models import Candidate

    with store_class()() as store:
        with pytest.raises(DomainError) as caught:
            store.record_feedback("owner", {}, "A", "\t\n", [])
        assert caught.value.status_code == 422
        with pytest.raises(DomainError):
            store.register_candidates(
                "owner",
                "t",
                [Candidate(label="A", content="First"), Candidate(label="A", content="Second")],
            )
        assert store.resolve_candidate("owner", "t", "A") is None
        assert store.snapshot("owner").revision == 0
        assert store.snapshot("owner").exemplars == []


def test_database_refuses_repository_escape(tmp_path):
    with pytest.raises(DomainError) as caught:
        store_class()(tmp_path.parent / ".." / ".." / ".." / "escape.db")
    assert caught.value.status_code == 422


def test_standard_model_construct_does_not_bypass_finite_validation():
    item = standard().model_copy(update={"confidence": float("nan")})
    with store_class()() as store:
        with pytest.raises(DomainError) as caught:
            store.propose(item)
        assert caught.value.status_code == 422
        assert store.list_standards() == []


def test_edge_identity_is_stable_across_updates_snapshot_and_reopen(tmp_path):
    path = tmp_path / "edge-ids.db"
    with store_class()(path, clock=lambda: NOW) as store:
        store.propose(standard())
        store.propose(standard("s2", directive="Other rule"))
        original = store.add_edge("owner", "s1", "s2", "relates", 0.6)
        updated = store.add_edge("owner", "s1", "s2", "relates", 0.9)
        assert updated.id == original.id
        assert store.snapshot("owner").edges[0]["id"] == original.id
    with store_class()(path) as store:
        assert store.snapshot("owner").edges[0]["id"] == original.id


def test_snapshot_sql_query_count_is_bounded_independent_of_standard_count():
    def read_query_count(store):
        queries = []
        store._connection.set_trace_callback(queries.append)
        try:
            store.snapshot("owner")
        finally:
            store._connection.set_trace_callback(None)
        return sum(q.lstrip().upper().startswith("SELECT") for q in queries)

    with store_class()() as store:
        store.propose(standard())
        one = read_query_count(store)
        for index in range(2, 32):
            store.propose(standard(f"s{index}", directive=f"Synthetic rule {index}"))
        many = read_query_count(store)
        assert many == one
        assert many <= 8


@pytest.mark.parametrize(
    "rationale",
    ["\v", "\f", "\u00a0", "\u2000\u2001", "\u2028\u2029", "\u3000", "\x1c\x1d\x1e\x1f"],
)
def test_database_rejects_all_unicode_whitespace_only_decision_rationale(tmp_path, rationale):
    path = tmp_path / "unicode-rationale.db"
    with store_class()(path):
        pass
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO nodes VALUES ('owner','invalid','decision')")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO decisions VALUES ('owner','invalid',?,'{}')", (rationale,))


def test_frozen_clock_does_not_allow_stale_approval_revision():
    with store_class()(clock=lambda: NOW) as store:
        item = store.propose(standard())
        current = approve(store, item)
        with pytest.raises(DomainError) as caught:
            store.approve(
                "owner",
                item.id,
                "Stale confirmation",
                owner_confirmed=True,
                expected_updated=item.updated,
            )
        assert caught.value.status_code == 409
        assert current.updated > item.updated
        row = store.snapshot("owner").standards[0]
        assert row["valid_from"] == NOW.isoformat()
        assert store.snapshot("owner").evaluated_at == NOW


def test_repeated_rationale_links_new_frozen_evidence_to_canonical_existing_rule():
    from taste.models import Candidate

    with store_class()(clock=lambda: NOW) as store:
        store.register_candidates(
            "owner",
            "task",
            [
                Candidate(label="A", content="Chosen text"),
                Candidate(label="B", content="Rejected text"),
            ],
        )
        _, proposals = store.record_feedback(
            "owner", {"task_ref": "task"}, "A", "Prefer clarity", ["B"]
        )
        canonical = proposals[0]
        approve(store, canonical)
        record, proposals = store.record_feedback(
            "owner", {"task_ref": "task"}, "A", "Prefer clarity", ["B"]
        )
        assert proposals == []
        snapshot = store.snapshot("owner")
        evidence = next(e for e in snapshot.exemplars if e["decision_id"] == record.id)
        assert evidence["standard_ids"] == [canonical.id]
        assert evidence["text"] == "Chosen text"
        assert evidence["rejected"] == ["Rejected text"]
        assert any(
            e["src"] == evidence["id"]
            and e["dst"] == canonical.id
            and e["edge_type"] == "exemplar_of"
            for e in snapshot.edges
        )


def test_pre_release_schema_migration_keeps_edge_identity_and_graph(tmp_path):
    path = tmp_path / "legacy-schema.db"
    with store_class()(path, clock=lambda: NOW) as store:
        store.propose(standard())
        store.propose(standard("s2", directive="Other rule"))
        edge = store.add_edge("owner", "s1", "s2", "relates", 0.75)
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE edges DROP COLUMN id")
        conn.execute("PRAGMA user_version=1")
    with store_class()(path) as store:
        migrated = store.snapshot("owner").edges[0]
        assert migrated["id"] == edge.id
        assert migrated["weight"] == 0.75
        assert store._connection.execute("PRAGMA user_version").fetchone()[0] == 2
