"""Synthetic boundary tests: malformed data must never enter graph policy."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from taste.models import (
    Candidate,
    CandidateScore,
    DecisionRecord,
    Edge,
    Exemplar,
    FeedbackRequest,
    HoneRequest,
    HoneResponse,
    InterviewRequest,
    LayerState,
    Packet,
    ScoreRequest,
    Standard,
)


def standard_data(**changes: object) -> dict:
    return {
        "category": "structure",
        "polarity": "do",
        "directive": "Lead with the result",
        **changes,
    }


@pytest.mark.parametrize("timestamp", [0, 0.0, "1234"])
def test_timestamp_requires_iso_representation_not_coerced_unix_epoch(timestamp):
    with pytest.raises(ValidationError):
        Standard.model_validate(standard_data(created=timestamp, updated=timestamp))


def test_standard_preserves_portable_unknown_extensions() -> None:
    source = standard_data(
        id="std_synthetic",
        profile_id="synthetic",
        confidence=0.75,
        provenance="correction",
        state="retired",
        rationale="Easy to scan",
        supersedes=["std_previous"],
        created="2026-01-01T00:00:00Z",
        updated="2026-02-01T00:00:00Z",
        variant_meta={"other_variant": [1, True, None]},
        future_extension={"format": "v2", "values": ["a", 2]},
    )
    portable = Standard.model_validate(source).model_dump(mode="json")
    assert portable == source
    assert Standard.model_validate(portable).model_dump(mode="json") == source


def test_standard_defaults_remain_provisional_with_independent_metadata() -> None:
    first = Standard(**standard_data())
    second = Standard(**standard_data())
    first.variant_meta["synthetic"] = 1
    first.supersedes.append("old")
    assert second.variant_meta == {}
    assert second.supersedes == []
    assert first.state is LayerState.PROVISIONAL
    assert first.created.tzinfo is UTC
    assert first.id != second.id


@pytest.mark.parametrize(
    "field,value",
    [
        ("category", "made_up"),
        ("polarity", "must"),
        ("provenance", "agent_approved"),
        ("state", "confirmed"),
    ],
)
def test_standard_rejects_values_outside_v1_enums(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        Standard(**standard_data(**{field: value}))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -0.001, 1.001])
@pytest.mark.parametrize("kind", ["standard", "candidate", "candidate_score", "hone", "edge"])
def test_numeric_boundaries_reject_nonfinite_or_out_of_range(kind: str, value: float) -> None:
    with pytest.raises(ValidationError):
        if kind == "standard":
            Standard(**standard_data(confidence=value))
        elif kind == "candidate":
            Candidate(label="A", score=value)
        elif kind == "candidate_score":
            CandidateScore(ref="A", score=value)
        elif kind == "hone":
            HoneResponse(score=value)
        else:
            Edge(profile_id="synthetic", id="edge", src="a", dst="b", type="relates", weight=value)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_score_and_confidence_bounds_are_inclusive(value: float) -> None:
    assert Standard(**standard_data(confidence=value)).confidence == value
    assert Candidate(label="A", score=value).score == value
    assert CandidateScore(ref="A", score=value).score == value


@pytest.mark.parametrize("model", [DecisionRecord, FeedbackRequest])
@pytest.mark.parametrize("rationale", [None, "", " \t\n", "\u2003"])
def test_rationale_is_required_and_nonblank(model: type, rationale: object) -> None:
    with pytest.raises(ValidationError):
        model(pick="A", rationale=rationale)


@pytest.mark.parametrize("model", [DecisionRecord, FeedbackRequest])
def test_missing_rationale_is_rejected(model: type) -> None:
    with pytest.raises(ValidationError):
        model(pick="A")


def test_decision_preserves_owner_rationale_and_real_candidate_content() -> None:
    record = DecisionRecord(
        pick="A",
        rationale="  Result comes first.  ",
        candidates=[Candidate(label="A", content="The result")],
    )
    assert record.rationale == "  Result comes first.  "
    assert record.candidates[0].content == "The result"
    assert record.ts.tzinfo is UTC


@pytest.mark.parametrize("field", ["created", "updated"])
def test_standard_rejects_naive_timestamps(field: str) -> None:
    with pytest.raises(ValidationError):
        Standard(**standard_data(**{field: "2026-01-01T00:00:00"}))


def test_aware_timestamps_are_normalized_to_utc() -> None:
    local = datetime(2026, 1, 1, 3, 0, tzinfo=timezone(timedelta(hours=3)))
    standard = Standard(**standard_data(created=local, updated=local))
    assert standard.created == datetime(2026, 1, 1, tzinfo=UTC)
    assert standard.created.tzinfo is UTC
    assert standard.model_dump(mode="json")["created"] == "2026-01-01T00:00:00Z"


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_ctx", {"nested": [float("nan")]}),
        ("task_ctx", {"nested": {"bad": float("inf")}}),
        ("task_ctx", {"tuple": ("a", "b")}),
        ("task_ctx", {"date": datetime(2026, 1, 1)}),
        ("task_ctx", {1: "nonstring key"}),
        ("task_ctx", {"set": {"a"}}),
    ],
)
def test_context_rejects_non_json_values_recursively(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        InterviewRequest(**{field: value})


def test_context_preserves_nested_json_without_coercion() -> None:
    value = {"nested": [None, False, 3, 0.25, {"text": "synthetic"}]}
    assert HoneRequest(segment_ctx=value).model_dump()["segment_ctx"] == value
    assert Packet(task_ctx=value).model_dump()["task_ctx"] == value


@pytest.mark.parametrize("field", ["variant_meta", "future_extension"])
def test_standard_rejects_non_json_metadata_and_extensions(field: str) -> None:
    with pytest.raises(ValidationError):
        Standard(**standard_data(**{field: {"bad": [float("nan")]}}))


def test_duplicate_candidate_labels_cannot_be_scored_as_distinct_artifacts() -> None:
    with pytest.raises(ValidationError):
        ScoreRequest(
            candidates=[
                Candidate(label="A", content="first"),
                Candidate(label="A", content="second"),
            ]
        )
    request = ScoreRequest(candidates=[Candidate(label="A"), Candidate(label="B")])
    assert [candidate.label for candidate in request.candidates] == ["A", "B"]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_score_breakdown_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ValidationError):
        CandidateScore(ref="A", score=0.5, breakdown={"positive": value, "negative": 0})


def test_edge_rejects_unknown_relation_and_naive_creation_time() -> None:
    base = dict(profile_id="synthetic", id="edge", src="a", dst="b", weight=0.5)
    with pytest.raises(ValidationError):
        Edge(**base, type="activates")
    with pytest.raises(ValidationError):
        Edge(**base, type="relates", created="2026-01-01T00:00:00")


def test_candidate_keeps_optional_content_and_default_score() -> None:
    assert Candidate(label="A").model_dump() == {
        "label": "A",
        "content_ref": "",
        "content": None,
        "score": 0.0,
    }


def test_contrastive_exemplar_keeps_real_rejected_content() -> None:
    exemplar = Exemplar(text="Result first", why="Easier to scan", rejected="Long opening")
    assert exemplar.model_dump() == {
        "text": "Result first",
        "why": "Easier to scan",
        "rejected": "Long opening",
    }


@pytest.mark.parametrize("value", [True, "0.5"])
def test_scores_require_json_numbers_instead_of_boolean_or_text(value: object) -> None:
    with pytest.raises(ValidationError):
        Candidate(label="A", score=value)
    with pytest.raises(ValidationError):
        CandidateScore(ref="A", score=0.5, breakdown={"positive": value})


def test_cyclic_python_context_is_rejected_as_non_json() -> None:
    context: dict = {}
    context["cycle"] = context
    with pytest.raises(ValidationError):
        InterviewRequest(task_ctx=context)


def test_contrastive_exemplar_extensions_remain_finite_json() -> None:
    with pytest.raises(ValidationError):
        Exemplar(text="Result", source_details={"value": float("inf")})
