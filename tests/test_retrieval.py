"""B4 policy tests use disclosed synthetic vectors, never a semantic model."""

import importlib
import math
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest


def module(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pytest.fail(f"Missing required graph implementation: {name}")


STAMP = datetime(2026, 1, 1, tzinfo=UTC)


def standard(id, vector=None, **changes):
    row = {
        "id": id,
        "profile_id": "p",
        "category": "voice",
        "polarity": "do",
        "directive": id,
        "rationale": "Synthetic reason",
        "confidence": 1.0,
        "provenance": "synthetic",
        "state": "active",
        "supersedes": [],
        "created": STAMP.isoformat(),
        "updated": STAMP.isoformat(),
        "_local_approved": True,
        "valid_from": STAMP.isoformat(),
        "valid_to": None,
    }
    row.update(changes)
    return row


class VectorEmbedder:
    model_name = "test-only:deterministic-vectors"

    def __init__(self, vectors=None):
        self.vectors = vectors or {}
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [self.vectors.get(text, [1.0, 0.0]) for text in texts]


def engine(rows, edges=(), embedder=None):
    snapshot = SimpleNamespace(
        profile_id="p",
        revision=7,
        evaluated_at=STAMP,
        standards=rows,
        edges=list(edges),
        exemplars=[],
    )
    store = SimpleNamespace(snapshot=lambda profile: snapshot)
    return module("taste.core").StandardsEngine(store, embedder=embedder or VectorEmbedder())


def edge(src, dst, type="relates", weight=0.5, profile="p"):
    return {"profile_id": profile, "src": src, "dst": dst, "type": type, "weight": weight}


def test_seeds_emit_and_stable_id_breaks_ties():
    result = engine([standard("z"), standard("a")]).context("p", {"goal": "synthetic"})
    assert [row["id"] for row in result.selected] == ["a", "z"]
    assert result.as_of == STAMP
    assert result.snapshot.revision == 7
    assert result.model_name == "test-only:deterministic-vectors"


def test_one_outgoing_hop_uses_max_path_and_excludes_wrong_edges():
    vectors = {f"s{i}": [1.0, 0.0] for i in range(5)}
    vectors.update(
        {name: [0.0, 1.0] for name in ["x", "y", "z", "bad", "reverse", "cross", "second"]}
    )
    rows = [standard(id) for id in vectors]
    edges = [
        edge("s0", "x", weight=0.6),
        edge("s1", "x", weight=0.9),
        edge("s0", "y", "contradicts"),
        edge("s0", "z", "applies_to"),
        edge("s0", "bad", weight=0.49),
        edge("reverse", "s0"),
        edge("s0", "cross", profile="other"),
        edge("x", "second"),
        edge("s0", "bad", "supersedes", 1.0),
    ]
    result = engine(rows, edges, VectorEmbedder(vectors)).context("p", {})
    assert [row["id"] for row in result.selected] == ["s0", "s1", "s2", "s3", "s4", "x", "y", "z"]
    assert result.selected[5]["_retrieval_score"] == pytest.approx(0.9)


def test_local_approval_state_windows_and_profile_gate_both_seeds_and_targets():
    rows = [
        standard("good"),
        standard("unapproved", _local_approved=False),
        standard("old", state="superseded"),
        standard("future", valid_from="2026-02-01T00:00:00Z"),
        standard("ended", valid_to=STAMP.isoformat()),
        standard("foreign", profile_id="q"),
    ]
    result = engine(rows, [edge("good", row["id"], weight=1.0) for row in rows]).context("p", {})
    assert [row["id"] for row in result.selected] == ["good"]
    historical = engine(rows).context("p", {"as_of": "2026-01-01T00:00:00Z"})
    assert [row["id"] for row in historical.selected] == ["good"]


def test_low_relevance_never_rules_survive_twelve_result_cap():
    rows = [standard(f"seed{i}") for i in range(5)] + [standard(f"target{i}") for i in range(15)]
    rows.append(standard("always", polarity="never"))
    vectors = {row["id"]: [0.0, 1.0] for row in rows if not row["id"].startswith("seed")}
    result = engine(
        rows, [edge("seed0", f"target{i}") for i in range(15)], VectorEmbedder(vectors)
    ).context("p", {})
    assert len(result.selected) == 12
    assert [row["id"] for row in result.never] == ["always"]


def test_context_canonicalization_excludes_attendance_and_evaluation_time():
    embedder = VectorEmbedder()
    policy = engine([standard("one")], embedder=embedder)
    first = policy.context("p", {"goal": "Synthetic", "attended": True, "as_of": STAMP.isoformat()})
    second = policy.context(
        "p", {"as_of": "2026-01-02T00:00:00Z", "attended": False, "goal": "Synthetic"}
    )
    assert first.selected == second.selected
    assert embedder.calls[0][0] == embedder.calls[1][0]


def test_empty_eligible_corpus_avoids_embedding():
    class ForbiddenEmbedder(VectorEmbedder):
        def embed(self, texts):
            pytest.fail("Empty eligible corpus must not invoke model inference")

    assert (
        engine([standard("unapproved", _local_approved=False)], embedder=ForbiddenEmbedder())
        .context("p", {})
        .selected
        == []
    )


@pytest.mark.parametrize("vectors", [[[math.nan, 0]], [[1]], [[0, 0]], []])
def test_invalid_embedding_output_is_sanitized_dependency_failure(vectors):
    class BrokenEmbedder(VectorEmbedder):
        def embed(self, texts):
            return vectors

    with pytest.raises(Exception) as failure:
        engine([standard("one")], embedder=BrokenEmbedder()).context("p", {})
    assert failure.value.code == "dependency_unavailable"
    assert failure.value.status_code == 503


def test_cosine_normalizes_vectors_and_rejects_invalid_numbers():
    cosine = module("taste.embed").cosine
    assert cosine([3, 4], [6, 8]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == 0.0
    for left, right in [([1], [1, 2]), ([0, 0], [1, 0]), ([math.inf], [1])]:
        with pytest.raises(ValueError):
            cosine(left, right)
