"""Independent TASTE v1 wire models and temporal graph atoms."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


def _now() -> datetime:
    return datetime.now(UTC)


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must include a UTC offset")
    try:
        return value.astimezone(UTC)
    except OverflowError:
        raise ValueError("Timestamp is outside the supported UTC range") from None


def _datetime_input(value: Any) -> datetime:
    """v1 dates are ISO strings or internal datetimes, not coerced epoch numbers."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    raise ValueError("Timestamp must be an ISO8601 datetime")


def _json_value(value: Any) -> Any:
    """Validate without coercion so context hashes reflect actual JSON values."""
    try:
        _check_json(value, set())
    except RecursionError as exc:
        raise ValueError("JSON context is nested too deeply") from exc
    return value


def _check_json(value: Any, ancestors: set[int]) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    if isinstance(value, (list, dict)):
        if id(value) in ancestors:
            raise ValueError("Cyclic context is not JSON")
        ancestors.add(id(value))
        try:
            if isinstance(value, dict) and not all(isinstance(key, str) for key in value):
                raise ValueError("JSON object keys must be strings")
            for child in value.values() if isinstance(value, dict) else value:
                _check_json(child, ancestors)
        finally:
            ancestors.remove(id(value))
        return
    raise ValueError("Context and metadata must contain finite JSON values")


UnitFloat = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False, strict=True)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False, strict=True)]
UTCDateTime = Annotated[datetime, BeforeValidator(_datetime_input), AfterValidator(_utc)]


class WireModel(BaseModel):
    """Shared validation for arbitrary JSON at the v1 extension boundaries."""

    @field_validator("task_ctx", "segment_ctx", "variant_meta", mode="before", check_fields=False)
    @classmethod
    def json_context(cls, value: Any) -> Any:
        return _json_value(value)


class PortableModel(WireModel):
    """Preserve extension fields while requiring them to remain valid JSON."""

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def json_extensions(cls, value: Any) -> Any:
        if isinstance(value, dict):
            for key, extension in value.items():
                if key not in cls.model_fields:
                    _json_value(extension)
        return value


class Polarity(StrEnum):
    DO = "do"
    NEVER = "never"
    WHY = "why"


class Category(StrEnum):
    STRUCTURE = "structure"
    AESTHETIC = "aesthetic"
    PROCEDURAL = "procedural"
    DOCTRINE = "doctrine"
    ANTI = "anti"


class Provenance(StrEnum):
    PERSONA = "persona"
    DECISION = "decision"
    CORRECTION = "correction"
    INFERENCE = "inference"
    AMENDMENT = "amendment"


class LayerState(StrEnum):
    PROVISIONAL = "provisional"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETIRED = "retired"


class Standard(PortableModel):
    # Declared state is portable input, never evidence of local approval.

    id: str = Field(default_factory=lambda: _id("std"))
    profile_id: str = "owner"
    category: Category
    polarity: Polarity
    directive: str
    rationale: str = ""
    confidence: UnitFloat = 0.5
    provenance: Provenance = Provenance.PERSONA
    state: LayerState = LayerState.PROVISIONAL
    supersedes: list[str] = Field(default_factory=list)
    created: UTCDateTime = Field(default_factory=_now)
    updated: UTCDateTime = Field(default_factory=_now)
    variant_meta: dict[str, Any] = Field(default_factory=dict)


class Directive(BaseModel):
    polarity: Polarity
    text: str
    standard_id: str | None = None


class Exemplar(PortableModel):
    text: str
    why: str = ""
    rejected: str | None = None


class Constraint(BaseModel):
    hard: bool = True
    text: str


class Packet(WireModel):
    profile_id: str = "owner"
    task_ctx: dict[str, Any] = Field(default_factory=dict)
    directives: list[Directive] = Field(default_factory=list)
    exemplars: list[Exemplar] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    free_degrees: list[str] = Field(default_factory=list)
    variant_meta: dict[str, Any] = Field(default_factory=dict)


class Candidate(WireModel):
    label: str
    content_ref: str = ""
    content: str | None = None
    score: UnitFloat = 0.0


class DecisionRecord(WireModel):
    id: str = Field(default_factory=lambda: _id("dec"))
    profile_id: str = "owner"
    task_ctx: dict[str, Any] = Field(default_factory=dict)
    candidates: list[Candidate] = Field(default_factory=list)
    pick: str
    rationale: str
    agent_role: str = ""
    provenance: str = "owner_pick"
    ts: UTCDateTime = Field(default_factory=_now)
    variant_meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("rationale")
    @classmethod
    def nonblank_rationale(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Decision rationale is required")
        return value


class InterviewRequest(WireModel):
    profile_id: str = "owner"
    task_ctx: dict[str, Any] = Field(default_factory=dict)


class InterviewQuestion(BaseModel):
    id: str
    text: str
    options: list[str] = Field(default_factory=list)
    resolves: str = ""


def _coverage() -> dict[str, list[str]]:
    return {"resolved": [], "gaps": []}


class InterviewResponse(BaseModel):
    questions: list[InterviewQuestion] = Field(default_factory=list)
    coverage: dict[str, list[str]] = Field(default_factory=_coverage)


class MirrorRequest(WireModel):
    profile_id: str = "owner"
    task_ctx: dict[str, Any] = Field(default_factory=dict)
    interview_answers: dict[str, str] = Field(default_factory=dict)


class MirrorResponse(BaseModel):
    mirror: str
    assumptions: list[str] = Field(default_factory=list)
    inferred: bool = False
    gate: Literal["required", "queued"] = "required"


class PacketRequest(WireModel):
    profile_id: str = "owner"
    task_ctx: dict[str, Any] = Field(default_factory=dict)
    answers: dict[str, str] = Field(default_factory=dict)
    guidelines: list[str] = Field(default_factory=list)


class ScoreRequest(WireModel):
    profile_id: str = "owner"
    task_ctx: dict[str, Any] = Field(default_factory=dict)
    candidates: list[Candidate]

    @field_validator("candidates")
    @classmethod
    def unique_labels(cls, candidates: list[Candidate]) -> list[Candidate]:
        labels = [candidate.label for candidate in candidates]
        if len(labels) != len(set(labels)):
            raise ValueError("Candidate labels must be unique within a batch")
        return candidates


class CandidateScore(BaseModel):
    ref: str
    score: UnitFloat
    breakdown: dict[str, FiniteFloat] = Field(default_factory=dict)
    violations: list[str] = Field(default_factory=list)


class ScoreResult(BaseModel):
    scores: list[CandidateScore] = Field(default_factory=list)


class FeedbackRequest(WireModel):
    profile_id: str = "owner"
    task_ctx: dict[str, Any] = Field(default_factory=dict)
    pick: str
    rationale: str
    rejected: list[str] = Field(default_factory=list)

    @field_validator("rationale")
    @classmethod
    def nonblank_rationale(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Feedback rationale is required")
        return value


class FeedbackResponse(BaseModel):
    accepted: bool
    record_id: str | None = None
    extracted: dict[str, int] = Field(default_factory=dict)


class HoneRequest(WireModel):
    profile_id: str = "owner"
    segment_ctx: dict[str, Any] = Field(default_factory=dict)
    draft: str | None = None


class HoneResponse(BaseModel):
    candidates: list[Candidate] = Field(default_factory=list)
    pick: str = ""
    score: UnitFloat = 0.0
    iterate: bool = False


class IngestRequest(BaseModel):
    profile_id: str = "owner"
    docs: list[dict[str, str]] = Field(default_factory=list)


class Restatement(BaseModel):
    layer: Standard
    implications: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    candidates: list[Standard] = Field(default_factory=list)
    restatements: list[Restatement] = Field(default_factory=list)


class ImportRequest(BaseModel):
    profile_id: str = "owner"
    standards: list[Standard] = Field(default_factory=list)


class MetaResponse(BaseModel):
    name: str
    version: str
    contract: str = "v1"
    permutation: dict[str, str] = Field(default_factory=dict)


class ErrorBody(BaseModel):
    error: dict[str, str]


EdgeType = Literal["supersedes", "contradicts", "exemplar_of", "applies_to", "relates"]


class Edge(BaseModel):
    profile_id: str
    id: str = Field(default_factory=lambda: _id("edge"))
    src: str
    dst: str
    type: EdgeType
    weight: UnitFloat = 1.0
    created: UTCDateTime = Field(default_factory=_now)
