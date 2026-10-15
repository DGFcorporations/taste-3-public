"""Mirror gates describe intent and durable review state using synthetic data."""

import pytest

from taste.errors import DomainError
from taste.models import MirrorRequest
from taste.store import Store
from tests.test_interview import NOW, approved_rule, engine_for, module


def mirror(store, ctx, **kwargs):
    return module("taste.mirror").mirror(engine_for(store), MirrorRequest(task_ctx=ctx, **kwargs))


def test_attended_and_default_mirrors_preserve_goal_and_shape_without_review():
    with Store(clock=lambda: NOW) as store:
        for attended in ({}, {"attended": True}):
            response = mirror(store, {"goal": "Explain synthetic result", "output_shape": "One-page memo", **attended})
            assert "Explain synthetic result" in response.mirror
            assert "One-page memo" in response.mirror
            assert response.inferred is False and response.gate == "required"
        assert store.list_reviews("owner") == []


def test_unattended_review_is_committed_and_survives_reopen(tmp_path):
    path = tmp_path / "reviews.db"
    with Store(path, clock=lambda: NOW) as store:
        before = store.snapshot("owner").revision
        response = mirror(store, {"goal": "Synthetic goal", "attended": False})
        assert response.inferred is True and response.gate == "queued"
        assert store.snapshot("owner").revision == before
    with Store(path) as store:
        reviews = store.list_reviews("owner")
        assert len(reviews) == 1
        assert reviews[0]["mirror"] == response.mirror
        assert reviews[0]["assumptions"] == response.assumptions
        assert store.list_reviews("other") == []


def test_failed_review_write_cannot_claim_queued(monkeypatch):
    with Store(clock=lambda: NOW) as store:
        def unavailable(*args):
            raise DomainError("store_busy", "Synthetic store unavailable", 503)
        monkeypatch.setattr(store, "queue_review", unavailable)
        with pytest.raises(DomainError) as caught:
            mirror(store, {"attended": False})
        assert caught.value.code == "store_busy"
        assert store.list_reviews("owner") == []


def test_answers_can_explain_missing_goal_and_shape():
    with Store(clock=lambda: NOW) as store:
        response = mirror(store, {}, interview_answers={"goal": "Synthetic answer", "output_shape": "Checklist"})
        assert "Synthetic answer" in response.mirror and "Checklist" in response.mirror


def test_question_id_answers_fill_blank_context_without_overriding_explicit_intent():
    with Store() as store:
        response = mirror(store, {"goal": "", "output_shape": "Memo"},
                          interview_answers={"q_goal": "Explain the result", "q_output_shape": "Checklist"})
        assert "Explain the result" in response.mirror and "Memo" in response.mirror
        assert "Checklist" not in response.mirror


def test_elimination_reports_only_effective_never_rules():
    with Store(clock=lambda: NOW) as store:
        approved_rule(store, "ban", polarity="never", directive='Never use "generic opener"')
        response = mirror(store, {"goal": "Start with generic opener", "attended": False})
        assert any("ban" in item and "generic opener" in item for item in response.assumptions)
        assert store.list_reviews("owner")[0]["assumptions"] == response.assumptions
        assert not any("ban" in item for item in mirror(store, {"goal": "Lead with result"}).assumptions)


@pytest.mark.parametrize("attended", [None, "false", 0])
def test_attendance_requires_an_explicit_boolean(attended):
    with Store() as store:
        with pytest.raises(DomainError) as caught:
            mirror(store, {"attended": attended})
        assert caught.value.status_code == 422
        assert store.list_reviews("owner") == []
