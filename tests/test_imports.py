"""Portable synthetic records retain declarations without importing authority."""

import pytest

from taste.errors import DomainError
from tests.test_store import NOW, approve, standard, store_class


def portable(id="s1", **changes):
    data = standard(id, **changes).model_dump(mode="json")
    data["created"] = "2026-01-01T00:00:00+00:00"
    data["updated"] = "2026-01-01T00:00:00+00:00"
    data["vendor_extension"] = {"nested": [1, "synthetic"]}
    return data


def test_import_preserves_source_json_and_quarantines_claimed_approval():
    source = portable(state="active")
    source["variant_meta"] = {"taste3": {"local_approved": True}, "other": {"keep": 7}}
    with store_class()(clock=lambda: NOW) as store:
        result = store.import_standards("owner", [source])
        assert result["imported"] == 1
        exported = store.list_standards()[0]
        for key, value in source.items():
            if key != "variant_meta":
                assert exported[key] == value
        assert exported["variant_meta"]["other"] == {"keep": 7}
        assert exported["variant_meta"]["taste3"]["local_approved"] is False
        assert store.snapshot("owner").standards[0]["_local_approved"] is False
        assert store.list_standards(state="active")[0]["id"] == "s1"


def test_unchanged_local_export_reimport_keeps_approval_and_revision():
    with store_class()(clock=lambda: NOW) as store:
        approve(store, store.propose(standard()))
        before = store.snapshot("owner")
        result = store.import_standards("owner", store.list_standards())
        assert result["unchanged"] == 1
        after = store.snapshot("owner")
        assert after.revision == before.revision
        assert after.standards[0]["_local_approved"] is True


def test_semantic_alias_preserves_full_source_without_overwriting_canonical():
    first = portable()
    alias = portable(
        "alias",
        directive="  USE\u00a0CLEAR HEADINGS  ",
        rationale="Alias reason",
        provenance="persona",
    )
    with store_class()(clock=lambda: NOW) as store:
        store.import_standards("owner", [first])
        result = store.import_standards("owner", [alias])
        assert result["aliases"] == 1
        rows = store.list_standards()
        assert len(rows) == 1
        assert rows[0]["rationale"] == "Readable structure"
        assert rows[0]["provenance"] == "inference"
        assert alias in rows[0]["variant_meta"]["taste3"]["aliases"]


def test_import_conflict_or_mixed_profile_rolls_back_entire_batch():
    with store_class()(clock=lambda: NOW) as store:
        store.import_standards("owner", [portable()])
        before = store.snapshot("owner")
        for invalid in [
            portable(directive="Conflicting directive"),
            portable("other", profile_id="elsewhere"),
        ]:
            with pytest.raises(DomainError):
                store.import_standards("owner", [portable("new", directive="New rule"), invalid])
            assert store.snapshot("owner") == before


def test_unresolved_supersedes_edges_remain_pending_until_same_profile_target_import():
    with store_class()(clock=lambda: NOW) as store:
        store.import_standards("owner", [portable("new", supersedes=["old"])])
        assert store.snapshot("owner").edges == []
        store.import_standards("other", [portable("old", profile_id="other")])
        assert store.snapshot("owner").edges == []
        store.import_standards("owner", [portable("old", directive="Old directive")])
        edges = store.snapshot("owner").edges
        assert [(e["src"], e["dst"], e["edge_type"]) for e in edges] == [
            ("new", "old", "supersedes")
        ]


def test_malformed_import_missing_portable_fields_is_422_and_atomic():
    with store_class()() as store:
        malformed = portable()
        del malformed["updated"]
        with pytest.raises(DomainError) as caught:
            store.import_standards("owner", [portable("new", directive="New rule"), malformed])
        assert caught.value.status_code == 422
        assert store.list_standards() == []


def test_imported_supersession_cycle_rolls_back_nodes_and_pending_edges():
    with store_class()() as store:
        with pytest.raises(DomainError) as caught:
            store.import_standards(
                "owner",
                [
                    portable("a", supersedes=["b"]),
                    portable("b", directive="Different rule", supersedes=["a"]),
                ],
            )
        assert caught.value.status_code == 409
        assert store.snapshot("owner").revision == 0
        assert store.snapshot("owner").edges == []
        assert store.list_standards() == []


def test_nested_taste3_extensions_survive_roundtrip_and_changed_same_id_conflicts():
    source = portable()
    source["variant_meta"] = {
        "taste3": {"extension": {"nested": ["original", 7]}, "local_approved": True}
    }
    with store_class()() as first, store_class()() as second:
        first.import_standards("owner", [source])
        exported = first.list_standards()
        assert exported[0]["variant_meta"]["taste3"]["extension"] == {"nested": ["original", 7]}
        second.import_standards("owner", exported)
        assert second.list_standards()[0]["variant_meta"]["taste3"]["extension"] == {
            "nested": ["original", 7]
        }
        assert second.snapshot("owner").standards[0]["_local_approved"] is False
        changed = first.list_standards()[0]
        changed["variant_meta"]["taste3"]["extension"]["nested"] = ["changed"]
        with pytest.raises(DomainError) as caught:
            first.import_standards("owner", [changed])
        assert caught.value.status_code == 409


def test_typed_outgoing_graph_topology_roundtrips_with_validity_metadata():
    with store_class()(clock=lambda: NOW) as first, store_class()(clock=lambda: NOW) as second:
        first.propose(standard())
        first.propose(standard("s2", directive="Related rule"))
        approve(first, first.get_standard("owner", "s1"))
        for kind, weight in [("relates", 0.8), ("contradicts", 0.6), ("applies_to", 0.7)]:
            first.add_edge("owner", "s1", "s2", kind, weight)
        exported = first.list_standards()
        outgoing = exported[0]["variant_meta"]["edges"]
        assert {(e["type"], e["target_id"], e["weight"]) for e in outgoing} == {
            ("relates", "s2", 0.8),
            ("contradicts", "s2", 0.6),
            ("applies_to", "s2", 0.7),
        }
        assert first.import_standards("owner", exported)["unchanged"] == 2
        second.import_standards("owner", exported)
        assert second.list_standards()[0]["variant_meta"]["edges"] == outgoing
        assert second.snapshot("owner").standards[0]["valid_from"] == NOW.isoformat()
        assert {
            (e["src"], e["dst"], e["edge_type"], e["weight"])
            for e in second.snapshot("owner").edges
        } == {
            ("s1", "s2", "relates", 0.8),
            ("s1", "s2", "contradicts", 0.6),
            ("s1", "s2", "applies_to", 0.7),
        }


def test_imported_typed_edges_pending_until_target_exists_and_invalid_batch_rolls_back():
    source = portable()
    source["variant_meta"] = {
        "edges": [{"type": "relates", "target_id": "missing", "weight": 0.75}]
    }
    with store_class()() as store:
        store.import_standards("owner", [source])
        assert store.snapshot("owner").edges == []
        assert store.list_standards()[0]["variant_meta"]["edges"] == source["variant_meta"]["edges"]
        store.import_standards("owner", [portable("missing", directive="Target rule")])
        edge = store.snapshot("owner").edges[0]
        assert (edge["src"], edge["dst"], edge["edge_type"], edge["weight"]) == (
            "s1",
            "missing",
            "relates",
            0.75,
        )
        bad = portable("bad", directive="Invalid edge source")
        bad["variant_meta"] = {"edges": [{"type": "relates", "target_id": "s1", "weight": 1.1}]}
        before = store.snapshot("owner")
        with pytest.raises(DomainError) as caught:
            store.import_standards("owner", [portable("new", directive="New rule"), bad])
        assert caught.value.status_code == 422
        assert store.snapshot("owner") == before


def test_reimport_original_supersession_source_is_noop_after_topology_materializes():
    source = [portable("new", supersedes=["old"]), portable("old", directive="Old rule")]
    with store_class()() as store:
        store.import_standards("owner", source)
        before = store.snapshot("owner")
        assert store.import_standards("owner", source)["unchanged"] == 2
        assert store.snapshot("owner") == before
