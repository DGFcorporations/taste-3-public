"""HTTP/domain boundaries use synthetic private stores and no live provider."""

import importlib
import sqlite3
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from taste.models import Candidate, Standard
from taste.segments import task_key
from taste.store import Store


def api():
    try:
        return importlib.import_module("taste.api")
    except ModuleNotFoundError:
        pytest.fail("Missing HTTP adapter implementation")


def test_metadata_and_export_on_fresh_corpus():
    with Store(":memory:") as store:
        with TestClient(api().create_app(store=store)) as client:
            assert client.get("/meta").json()["contract"] == "v1"
            assert client.get("/meta").json()["permutation"]["C"] == "C1"
            assert client.get("/standards", params={"profile_id": "synthetic"}).json() == {
                "standards": []
            }
            assert client.post("/activate", json={}).status_code == 404
        assert store.snapshot("synthetic").revision == 0


@pytest.mark.parametrize("attempts,status,code", [
    (0, 503, "dependency_unavailable"), (15, 429, "hone_budget_exhausted")
])
def test_hone_reserves_before_embedding_and_exhaustion_takes_precedence(attempts, status, code):
    class BrokenEmbedder:
        model_name = "synthetic-unavailable"

        def embed(self, texts):
            raise RuntimeError("Synthetic model failure")

    with Store(":memory:") as store:
        proposed = store.propose(Standard(id="s", profile_id="p", category="structure",
                                          polarity="do", directive="Synthetic rule"))
        store.approve("p", "s", "Synthetic reason", owner_confirmed=True,
                      expected_updated=proposed.updated)
        for _ in range(attempts):
            store.reserve_hone("p", "shared-task")
        with TestClient(api().create_app(store=store, embedder=BrokenEmbedder())) as client:
            response = client.post("/hone", json={"profile_id": "p",
                                   "segment_ctx": {"task_ref": "shared-task"}})
        assert response.status_code == status
        assert response.json()["error"]["code"] == code
        if attempts == 0:
            assert store.reserve_hone("p", "shared-task") == 2


@pytest.mark.parametrize("field", ["created", "updated", "valid_from", "valid_to"])
@pytest.mark.parametrize("instant", ["0001-01-01T00:00:00+01:00", "9999-12-31T23:59:59-01:00"])
def test_import_utc_overflow_is_422_and_atomic(field, instant):
    row = Standard(id="s", profile_id="p", category="structure", polarity="do",
                   directive="Synthetic rule").model_dump(mode="json")
    if field in {"created", "updated"}:
        row[field] = instant
    else:
        row["variant_meta"] = {"taste3": {field: instant}}
    with Store(":memory:") as store, TestClient(api().create_app(store=store)) as client:
        response = client.post("/standards/import", json={"profile_id": "p", "standards": [row]})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_request"
        assert store.snapshot("p").standards == []
        assert store.snapshot("p").revision == 0


@pytest.mark.parametrize("rationale", [None, "", "   "])
def test_rationale_validation_has_no_decision_write_or_raw_input(rationale):
    body = {"profile_id": "synthetic", "task_ctx": {"goal": "PRIVATE-GOAL"}, "pick": "PRIVATE-PICK"}
    if rationale is not None:
        body["rationale"] = rationale
    with Store(":memory:") as store, TestClient(api().create_app(store=store)) as client:
        before = store.snapshot("synthetic")
        response = client.post("/feedback", json=body)
        assert response.status_code == 422
        assert set(response.json()) == {"error"}
        assert "PRIVATE" not in response.text
        assert store.snapshot("synthetic").revision == before.revision


def test_invalid_json_and_state_use_common_sanitized_envelope():
    with Store(":memory:") as store, TestClient(api().create_app(store=store)) as client:
        response = client.post(
            "/packet", content='{ "PRIVATE": ', headers={"content-type": "application/json"}
        )
        assert response.status_code == 422
        assert "PRIVATE" not in response.text
        response = client.get("/standards", params={"state": "PRIVATE-INVALID"})
        assert response.status_code == 422
        assert "PRIVATE" not in response.text


def test_streaming_body_limit_applies_without_content_length():
    with (
        Store(":memory:") as store,
        TestClient(api().create_app(store=store, max_body_bytes=128)) as client,
    ):
        response = client.post(
            "/packet",
            content=iter([b'{"task_ctx":{"goal":"', b"x" * 256, b'"}}']),
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "request_too_large"


def test_all_v1_operations_and_snapshot_stability():
    with Store(":memory:") as store, TestClient(api().create_app(store=store)) as client:
        ctx = {"type": "write_thread", "goal": "Synthetic concise result"}
        assert len(client.post("/interview", json={"task_ctx": {}}).json()["questions"]) == 3
        mirror = client.post("/mirror", json={"task_ctx": {"goal": "Synthetic", "attended": False}})
        assert mirror.json()["gate"] == "queued"
        assert len(store.list_reviews("owner")) == 1
        one = client.post("/packet", json={"task_ctx": ctx}).json()
        assert one == client.post("/packet", json={"task_ctx": ctx}).json()
        assert len(one["free_degrees"]) >= 3
        scores = client.post(
            "/score",
            json={
                "task_ctx": ctx,
                "candidates": [{"label": "A", "content": "Synthetic concise result"}],
            },
        ).json()
        assert scores["scores"][0]["breakdown"]["positive"] >= 0
        feedback = client.post(
            "/feedback",
            json={
                "task_ctx": ctx,
                "pick": "A",
                "rationale": "Prefer synthetic brevity",
                "rejected": ["Synthetic excessive text"],
            },
        )
        assert feedback.json()["accepted"] is True
        hone = client.post("/hone", json={"segment_ctx": ctx, "draft": "Synthetic result"})
        assert hone.status_code == 200
        assert len(hone.json()["candidates"]) == 3
        ingest = client.post(
            "/ingest",
            json={
                "docs": [
                    {
                        "id": "s",
                        "text": "Always begin with the result. Never use hype.",
                        "format": "md",
                    }
                ]
            },
        )
        assert ingest.status_code == 200
        assert all(row["state"] == "provisional" for row in ingest.json()["candidates"])


def test_import_preserves_original_timestamp_spelling_and_unknown_extensions():
    row = {
        "id": "portable",
        "profile_id": "p",
        "category": "structure",
        "polarity": "do",
        "directive": "Synthetic portable directive",
        "rationale": "Synthetic reason",
        "confidence": 0.5,
        "provenance": "persona",
        "state": "active",
        "supersedes": [],
        "created": "2026-01-01T01:00:00+01:00",
        "updated": "2026-01-01T01:00:00+01:00",
        "unknown": {"portable": 7},
    }
    with Store(":memory:") as store, TestClient(api().create_app(store=store)) as client:
        assert (
            client.post(
                "/standards/import", json={"profile_id": "p", "standards": [row]}
            ).status_code
            == 200
        )
        exported = client.get("/standards", params={"profile_id": "p"}).json()["standards"][0]
        for key, value in row.items():
            assert exported[key] == value
        assert exported["variant_meta"]["taste3"]["local_approved"] is False


def test_score_content_references_cannot_resolve_foreign_tasks():
    with Store(":memory:") as store, TestClient(api().create_app(store=store)) as client:
        foreign = store.register_candidates(
            "p", task_key({"task_ref": "other"}), [Candidate(label="A", content="PRIVATE-FOREIGN")]
        )[0]
        response = client.post(
            "/score",
            json={
                "profile_id": "p",
                "task_ctx": {"task_ref": "current"},
                "candidates": [{"label": "A", "content_ref": foreign.content_ref}],
            },
        )
        assert response.status_code == 422
        assert "PRIVATE" not in response.text


def test_factory_uses_disposable_environment_and_closes_owned_store(monkeypatch, tmp_path):
    path = tmp_path / "factory.db"
    monkeypatch.setenv("TASTE3_DB", str(path))
    app = api().create_app()
    with TestClient(app) as client:
        assert client.get("/meta").status_code == 200
    assert path.exists()
    with pytest.raises(sqlite3.ProgrammingError):
        app.state.service.store.snapshot("owner")


def test_unexpected_dependency_failure_has_no_private_exception_text(monkeypatch):
    with Store(":memory:") as store:

        def fail(*args):
            raise RuntimeError("PRIVATE-PATH /secret SQL credential")

        monkeypatch.setattr(store, "list_standards", fail)
        with TestClient(api().create_app(store=store), raise_server_exceptions=False) as client:
            response = client.get("/standards")
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "internal_error"
        assert "PRIVATE" not in response.text


def test_unknown_route_and_method_errors_share_v1_envelope():
    with Store(":memory:") as store, TestClient(api().create_app(store=store)) as client:
        for method, path, status in [("get", "/PRIVATE-MISSING", 404), ("post", "/meta", 405)]:
            response = getattr(client, method)(path)
            assert response.status_code == status
            assert set(response.json()) == {"error"}
            assert "PRIVATE" not in response.text


def test_constructor_failure_closes_created_store(monkeypatch, tmp_path):
    service_module = importlib.import_module("taste.service")
    instances = []

    class ObservedStore(Store):
        def __init__(self, path):
            super().__init__(path)
            instances.append(self)

    def unavailable_provider(settings):
        raise RuntimeError("PRIVATE provider initialization")

    monkeypatch.setattr(service_module, "Store", ObservedStore)
    monkeypatch.setattr(service_module, "CloudProvider", unavailable_provider)
    with pytest.raises(service_module.DomainError) as failure:
        service_module.TasteService(
            settings=service_module.Settings(
                db_path=tmp_path / "init.db",
                api_base="https://synthetic.example",
                api_model="synthetic",
            )
        )
    assert "PRIVATE" not in str(failure.value)
    with pytest.raises(sqlite3.ProgrammingError):
        instances[0].snapshot("owner")


def test_candidate_batch_limit_rejects_before_artifact_writes():
    with Store(":memory:") as store, TestClient(api().create_app(store=store)) as client:
        candidates = [{"label": f"A{i}", "content": "Synthetic"} for i in range(129)]
        response = client.post(
            "/score",
            json={"profile_id": "p", "task_ctx": {"task_ref": "bounded"}, "candidates": candidates},
        )
        assert response.status_code == 422
        assert store.resolve_candidate("p", "bounded", "A0") is None


def test_successful_reference_scoring_preserves_task_scoped_content():
    with Store(":memory:") as store, TestClient(api().create_app(store=store)) as client:
        saved = store.register_candidates(
            "p", "same", [Candidate(label="A", content="Synthetic saved content")]
        )[0]
        response = client.post(
            "/score",
            json={
                "profile_id": "p",
                "task_ctx": {"task_ref": "same"},
                "candidates": [{"label": "A", "content_ref": saved.content_ref}],
            },
        )
        assert response.status_code == 200
        assert response.json()["scores"][0]["ref"] == saved.content_ref
        assert (
            store.resolve_candidate("p", "same", saved.content_ref).content
            == "Synthetic saved content"
        )


def test_hone_budget_domain_failure_retains_429_http_status():
    with Store(":memory:") as store, TestClient(api().create_app(store=store)) as client:
        for _unused in range(15):
            store.reserve_hone("p", "shared")
        response = client.post(
            "/hone",
            json={"profile_id": "p", "segment_ctx": {"task_ref": "shared", "segment": "hook"}},
        )
        assert response.status_code == 429
        assert response.json()["error"]["code"] == "hone_budget_exhausted"


def test_cyclic_direct_payload_is_rejected_without_hanging():
    code = "from taste.service import bounded_payload; from taste.errors import DomainError; value={}; value['self']=value\ntry:\n bounded_payload(value)\nexcept DomainError as failure:\n assert failure.code == 'invalid_request'\nelse:\n raise AssertionError('Invalid JSON cycle was accepted')"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=3)
    assert result.returncode == 0


def test_excessive_json_integer_is_invalid_request_not_internal_error():
    with (
        Store(":memory:") as store,
        TestClient(api().create_app(store=store), raise_server_exceptions=False) as client,
    ):
        body = '{"task_ctx":{"goal":' + "1" * 5000 + "}}"
        response = client.post(
            "/packet", content=body, headers={"content-type": "application/json"}
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_request"


def test_confirm_unknown_standard_returns_404():
    with Store(":memory:") as store:
        with TestClient(api().create_app(store=store)) as client:
            response = client.post("/standards/std_missing/confirm", json={})
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"


def test_confirm_without_any_rationale_returns_422():
    with Store(":memory:") as store:
        store.propose(Standard(id="s", profile_id="p", category="structure",
                               polarity="do", directive="Synthetic rule"))
        with TestClient(api().create_app(store=store)) as client:
            for body in ({}, {"rationale": ""}, {"rationale": "   "}):
                response = client.post("/standards/s/confirm",
                                       json={**body, "profile_id": "p"})
                assert response.status_code == 422
                assert response.json()["error"]["code"] == "invalid_rationale"
        assert store.get_standard("p", "s").state.value == "provisional"


def test_confirm_activates_with_body_or_stored_rationale():
    with Store(":memory:") as store:
        proposed = store.propose(Standard(id="s1", profile_id="p", category="structure",
                                          polarity="do", directive="Synthetic rule"))
        store.propose(Standard(id="s2", profile_id="p", category="structure",
                               polarity="do", directive="Other rule",
                               rationale="Stored reason"))
        with TestClient(api().create_app(store=store)) as client:
            response = client.post("/standards/s1/confirm",
                                   json={"profile_id": "p",
                                         "rationale": "Body reason"})
            assert response.status_code == 200
            assert response.json()["state"] == "active"
            assert response.json()["updated"] != proposed.updated.isoformat()
            response = client.post("/standards/s2/confirm", json={"profile_id": "p"})
            assert response.status_code == 200
        approved = store.get_standard("p", "s2")
        assert approved.state.value == "active"
        assert approved.rationale == "Stored reason"
