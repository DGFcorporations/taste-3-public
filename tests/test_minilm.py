"""Real C1 acceptance: actual CPU ONNX MiniLM, synthetic text only."""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from taste.config import MINILM_MODEL, ROOT
from taste.core import StandardsEngine
from taste.embed import MiniLMEmbedder, cosine


def test_actual_minilm_dimensions_determinism_and_synthetic_retrieval(monkeypatch):
    cache = ROOT / "_agent-scratch" / "models"
    monkeypatch.setenv("HF_HOME", str(cache / "huggingface"))
    monkeypatch.setenv("HF_XET_CACHE", str(cache / "xet"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache / "xdg"))
    model = MiniLMEmbedder(cache)
    texts = [
        "A cat sleeps on a sofa.",
        "A feline rests on the couch.",
        "An aircraft engine uses aviation fuel.",
    ]
    first = model.embed(texts)
    model.close()
    second = model.embed(texts)
    assert model.model_name == MINILM_MODEL
    assert len(first) == 3
    assert all(len(vector) == 384 for vector in first)
    for left, right in zip(first, second, strict=True):
        assert left == pytest.approx(right, abs=1e-6)
    assert cosine(first[0], first[1]) > cosine(first[0], first[2])
    now = datetime(2026, 1, 1, tzinfo=UTC)
    directives = [
        "Use gentle language about a sleeping cat on a couch.",
        "Explain the aviation fuel pressure of an aircraft engine.",
        "Report quarterly profit margins.",
        "List database migration commands.",
        "Describe a volcanic eruption.",
        "Summarize train timetables.",
    ]
    rows = [
        {
            "id": f"std{i}",
            "profile_id": "synthetic",
            "directive": text,
            "state": "active",
            "_local_approved": True,
            "polarity": "do",
            "valid_from": now.isoformat(),
            "valid_to": None,
        }
        for i, text in enumerate(directives)
    ]
    snapshot = SimpleNamespace(
        profile_id="synthetic", revision=1, evaluated_at=now, standards=rows, edges=[], exemplars=[]
    )
    result = StandardsEngine(SimpleNamespace(snapshot=lambda profile: snapshot), model).context(
        "synthetic", {"goal": "Describe a sleepy cat resting on a sofa"}
    )
    assert result.selected[0]["id"] == "std0"
    assert len(result.selected) == 5
    assert any(path.suffix == ".onnx" for path in cache.rglob("*.onnx"))
    model.close()


def test_model_load_failure_has_no_hash_fallback_or_private_message(monkeypatch):
    model = MiniLMEmbedder(ROOT / "_agent-scratch" / "models")

    def broken():
        raise RuntimeError("PRIVATE /owner/path credential=secret")

    monkeypatch.setattr(model, "_load_model", broken)
    with pytest.raises(Exception) as failure:
        model.embed(["Synthetic local text"])
    assert failure.value.code == "dependency_unavailable"
    assert failure.value.status_code == 503
    assert "PRIVATE" not in str(failure.value)
    assert "secret" not in str(failure.value)


def test_cache_path_escape_rejected_without_creating_directory():
    with pytest.raises(ValueError):
        MiniLMEmbedder(Path("/tmp/forbidden-taste-model-cache"))


def test_vector_cache_is_bounded_and_does_not_return_mutable_cached_values(monkeypatch):
    class SyntheticBackend:
        def embed(self, texts, batch_size):
            return [[float(i + 1)] + [0.0] * 383 for i, text in enumerate(texts)]

    model = MiniLMEmbedder(ROOT / "_agent-scratch" / "models", cache_size=2)
    monkeypatch.setattr(model, "_load_model", lambda: SyntheticBackend())
    vectors = model.embed(["a", "b", "c", "a"])
    assert vectors[0] == vectors[3]
    vectors[3][0] = -100
    assert model.embed(["a"])[0][0] == 1
    assert len(model._vectors) <= 2


def test_lazy_model_initialization_confines_xet_cache_even_if_hub_already_imported(monkeypatch):
    import fastembed
    from huggingface_hub import constants

    monkeypatch.setenv("HF_XET_CACHE", "/tmp/unapproved-xet-cache")
    monkeypatch.setattr(constants, "HF_XET_CACHE", "/tmp/unapproved-xet-cache")
    observed = []

    def backend(**kwargs):
        observed.append(Path(constants.HF_XET_CACHE))
        return object()

    monkeypatch.setattr(fastembed, "TextEmbedding", backend)
    model = MiniLMEmbedder(ROOT / "_agent-scratch" / "models")
    model._load_model()
    assert observed[0].resolve().is_relative_to(ROOT)
    assert constants.HF_XET_CACHE == "/tmp/unapproved-xet-cache"
