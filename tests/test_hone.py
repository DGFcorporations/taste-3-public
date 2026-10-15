"""Honing tests protect task-wide reservations and reject violating winners."""

import importlib

import pytest


def modules():
    try:
        return tuple(
            importlib.import_module(f"taste.{name}")
            for name in ("models", "hone", "providers", "store", "errors", "segments")
        )
    except ImportError as exc:
        pytest.fail(f"Hone boundary is not implemented: {exc.name}")


def test_hone_emits_three_registered_candidates_and_stops_at_task_budget():
    m, honing, providers, storage, errors, _ = modules()
    with storage.Store() as store:
        packet = m.Packet()
        for i in range(15):
            response = honing.hone_packet(
                store,
                packet,
                {"task_ref": "one-task", "segment": str(i)},
                "A draft with details",
                providers.BaselineGenerator(),
            )
            assert len(response.candidates) == 3
            assert response.pick in {c.label for c in response.candidates}
            assert all(c.content_ref for c in response.candidates)
        assert response.iterate is False
        with pytest.raises(errors.DomainError) as exc:
            honing.hone_packet(
                store,
                packet,
                {"task_ref": "one-task", "segment": "new"},
                "other",
                providers.BaselineGenerator(),
            )
        assert exc.value.code == "hone_budget_exhausted"
        assert exc.value.status_code == 429


def test_all_violating_candidates_retain_evidence_and_consume_slot():
    m, honing, _, storage, errors, segments = modules()

    class Generator:
        model_name = "synthetic"

        def generate(self, packet, draft, n=3):
            return ["forbidden phrase"] * n

    packet = m.Packet(
        directives=[m.Directive(polarity="never", text='Never use "forbidden phrase"')]
    )
    with storage.Store() as store:
        with pytest.raises(errors.DomainError) as exc:
            honing.hone_packet(store, packet, {"task_ref": "bad"}, "draft", Generator())
        assert exc.value.code == "no_eligible_candidate"
        assert store.reserve_hone("owner", segments.task_key({"task_ref": "bad"})) == 2
        assert (
            store.resolve_candidate("owner", segments.task_key({"task_ref": "bad"}), "A").content
            == "forbidden phrase"
        )


def test_generation_failure_does_not_return_reserved_budget():
    m, honing, _, storage, errors, segments = modules()

    class Generator:
        model_name = "synthetic"

        def generate(self, packet, draft, n=3):
            raise errors.DomainError("dependency_unavailable", "Generation unavailable", 503)

    with storage.Store() as store:
        with pytest.raises(errors.DomainError):
            honing.hone_packet(store, m.Packet(), {"task_ref": "crash"}, "draft", Generator())
        assert store.reserve_hone("owner", segments.task_key({"task_ref": "crash"})) == 2
