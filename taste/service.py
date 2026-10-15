"""Shared use cases, validation, dependency ownership, and safe diagnostics."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from taste.config import Settings
from taste.core import StandardsEngine
from taste.embed import Embedder, MiniLMEmbedder
from taste.errors import DomainError
from taste.feedback import feedback
from taste.hone import hone_task
from taste.ingest import ingest
from taste.inject import AXES, compile_packet
from taste.interview import interview
from taste.mirror import mirror
from taste.models import (
    FeedbackRequest,
    FeedbackResponse,
    HoneRequest,
    HoneResponse,
    ImportRequest,
    IngestRequest,
    IngestResponse,
    InterviewRequest,
    InterviewResponse,
    LayerState,
    MirrorRequest,
    MirrorResponse,
    Packet,
    PacketRequest,
    ScoreRequest,
    ScoreResult,
)
from taste.providers import BaselineGenerator, CloudProvider, Generator, Judge
from taste.score import score_packet
from taste.segments import task_key
from taste.store import Store

MAX_BODY_BYTES = 10 * 1024 * 1024
MAX_LIST_ITEMS = 10000
MAX_CANDIDATES = 128
MAX_DOCUMENTS = 128
MAX_TEXT_CHARS = 1_000_000
_LOG = logging.getLogger("taste.operations")

REQUEST_MODELS: dict[str, type[BaseModel]] = {
    "interview": InterviewRequest,
    "mirror": MirrorRequest,
    "packet": PacketRequest,
    "score": ScoreRequest,
    "feedback": FeedbackRequest,
    "hone": HoneRequest,
    "ingest": IngestRequest,
    "import_standards": ImportRequest,
}


def bounded_payload(value: Any) -> None:
    """The same finite JSON/count/text boundaries serve HTTP and MCP."""
    if isinstance(value, dict):
        for field, limit in (("candidates", MAX_CANDIDATES), ("docs", MAX_DOCUMENTS)):
            items = value.get(field)
            if isinstance(items, list) and len(items) > limit:
                raise DomainError("invalid_request", "Request batch exceeds the supported limit.")
    pending = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if isinstance(current, (dict, list)):
            if id(current) in seen:
                continue
            seen.add(id(current))
        if isinstance(current, str) and len(current) > MAX_TEXT_CHARS:
            raise DomainError("invalid_request", "Request text exceeds the supported limit.")
        if isinstance(current, list):
            if len(current) > MAX_LIST_ITEMS:
                raise DomainError(
                    "invalid_request", "Request collection exceeds the supported limit."
                )
            pending.extend(current)
        elif isinstance(current, dict):
            if len(current) > MAX_LIST_ITEMS:
                raise DomainError(
                    "invalid_request", "Request collection exceeds the supported limit."
                )
            pending.extend(current.values())
    try:
        if len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode()) > MAX_BODY_BYTES:
            raise DomainError("request_too_large", "Request body exceeds the supported limit.", 413)
    except (TypeError, ValueError, RecursionError):
        raise DomainError("invalid_request", "Request must contain finite JSON values.") from None


class TasteService:
    def __init__(
        self,
        store: Store | None = None,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        judge: Judge | None = None,
        generator: Generator | None = None,
    ) -> None:
        try:
            self.settings = settings if settings is not None else Settings.from_env()
        except ValidationError:
            raise DomainError("invalid_request", "Runtime configuration is invalid.") from None
        self._owned_close: list[Callable[[], None]] = []
        try:
            self.store = store if store is not None else Store(self.settings.db_path)
            if store is None:
                self._owned_close.append(self.store.close)
            if embedder is None:
                actual_embedder = MiniLMEmbedder(self.settings.model_cache)
                self._owned_close.append(actual_embedder.close)
                embedder = actual_embedder
            self.engine = StandardsEngine(self.store, embedder)
            provider = None
            if (self.settings.api_base or self.settings.api_model) and (
                judge is None or generator is None
            ):
                provider = CloudProvider(self.settings)
                self._owned_close.append(provider.close)
            self.judge = judge if judge is not None else provider
            self.generator = (
                generator
                if generator is not None
                else (provider if provider is not None else BaselineGenerator())
            )
        except DomainError:
            self.close()
            raise
        except Exception:
            self.close()
            raise DomainError(
                "dependency_unavailable", "Runtime dependencies are unavailable.", 503
            ) from None

    def close(self) -> None:
        while self._owned_close:
            try:
                self._owned_close.pop()()
            except Exception:
                _LOG.error(json.dumps({"event": "dependency_close", "status": "failed"}))

    def runtime(self) -> dict[str, Any]:
        return {
            "generator": self.generator.model_name,
            "judge_configured": self.judge is not None,
            "embedding": self.engine.embedder.model_name,
        }

    def meta(self) -> dict[str, Any]:
        return {
            "name": "taste-3",
            "version": "0.1.0",
            "contract": "v1",
            "permutation": dict(AXES),
            "runtime": self.runtime(),
        }

    def interview(self, req: InterviewRequest) -> InterviewResponse:
        bounded_payload(req.model_dump(mode="json"))
        return interview(self.engine, req)

    def mirror(self, req: MirrorRequest) -> MirrorResponse:
        bounded_payload(req.model_dump(mode="json"))
        return mirror(self.engine, req)

    def packet(self, req: PacketRequest) -> Packet:
        bounded_payload(req.model_dump(mode="json"))
        packet = compile_packet(self.engine, req)
        packet.variant_meta["runtime"] = self.runtime()
        return packet

    def score(self, req: ScoreRequest) -> ScoreResult:
        bounded_payload(req.model_dump(mode="json"))
        identity = task_key(req.task_ctx)
        resolved = []
        for candidate in req.candidates:
            if candidate.content is not None:
                resolved.append(candidate)
                continue
            prior = (
                self.store.resolve_candidate(req.profile_id, identity, candidate.content_ref)
                if candidate.content_ref
                else None
            )
            if prior is None or prior.content is None:
                raise DomainError(
                    "invalid_request", "Candidate content is unavailable for this task."
                )
            resolved.append(candidate.model_copy(update={"content": prior.content}))
        packet = self.packet(PacketRequest(profile_id=req.profile_id, task_ctx=req.task_ctx))
        registered = self.store.register_candidates(req.profile_id, identity, resolved)
        return ScoreResult(scores=score_packet(packet, registered, self.judge))

    def feedback(self, req: FeedbackRequest) -> FeedbackResponse:
        bounded_payload(req.model_dump(mode="json"))
        return feedback(self.store, req)

    def hone(self, req: HoneRequest) -> HoneResponse:
        bounded_payload(req.model_dump(mode="json"))
        return hone_task(
            self.store,
            req.profile_id,
            req.segment_ctx,
            lambda: self.packet(PacketRequest(profile_id=req.profile_id, task_ctx=req.segment_ctx)),
            req.draft,
            self.generator,
            self.judge,
        )

    def ingest(self, req: IngestRequest) -> IngestResponse:
        bounded_payload(req.model_dump(mode="json"))
        return ingest(self.store, req)

    def standards(self, profile_id: str = "owner", state: str | None = None) -> dict[str, Any]:
        if state is not None and state not in {member.value for member in LayerState}:
            raise DomainError("invalid_request", "Invalid standard state filter.")
        return {"standards": self.store.list_standards(profile_id, state)}

    def check(self, profile_id: str = "owner") -> dict[str, Any]:
        active = self.store.list_standards(profile_id, state="active")
        all_stds = self.store.list_standards(profile_id)
        cats: dict[str, int] = {}
        pols: dict[str, int] = {}
        for s in active:
            c = s.get("category", "unknown")
            p = s.get("polarity", "unknown")
            cats[c] = cats.get(c, 0) + 1
            pols[p] = pols.get(p, 0) + 1
        return {
            "status": "healthy",
            "name": "taste-3",
            "version": "0.1.0",
            "active_standards": len(active),
            "total_standards": len(all_stds),
            "categories": cats,
            "polarities": pols,
        }

    def import_standards(self, req: ImportRequest) -> dict[str, int]:
        bounded_payload(req.model_dump(mode="json"))
        return self.store.import_standards(req.profile_id, list(req.standards))

    def dispatch(self, operation: str, payload: Any) -> dict[str, Any]:
        """Transport-independent error envelopes never include dependency input."""
        try:
            bounded_payload(payload)
            if not isinstance(payload, dict):
                raise DomainError("invalid_request", "Request must be a JSON object.")
            if operation == "standards":
                profile_id, state = payload.get("profile_id", "owner"), payload.get("state")
                category = payload.get("category")
                if category == "check":
                    return self.check(profile_id if isinstance(profile_id, str) else "owner")
                if not isinstance(profile_id, str) or (
                    state is not None and not isinstance(state, str)
                ):
                    raise DomainError("invalid_request", "Invalid standards request.")
                result: Any = self.standards(profile_id, state)
            elif operation == "check":
                profile_id = payload.get("profile_id", "owner")
                result = self.check(profile_id if isinstance(profile_id, str) else "owner")

            else:
                model = REQUEST_MODELS.get(operation)
                if model is None:
                    raise DomainError("invalid_request", "Unknown operation.")
                request = model.model_validate(payload)
                if operation == "import_standards":
                    assert isinstance(request, ImportRequest)
                    result = self.store.import_standards(
                        request.profile_id, payload.get("standards", [])
                    )
                else:
                    result = getattr(self, operation)(request)
            _LOG.info(
                json.dumps({"event": "operation", "operation": operation, "status": "success"})
            )
            return result.model_dump(mode="json") if isinstance(result, BaseModel) else result
        except ValidationError:
            error = DomainError("invalid_request", "Request validation failed.")
        except DomainError as failure:
            _LOG.warning(
                json.dumps(
                    {
                        "event": "operation",
                        "operation": operation,
                        "status": "failed",
                        "code": failure.code,
                    }
                )
            )
            raise
        except Exception:
            error = DomainError("internal_error", "The operation could not be completed.", 500)
        _LOG.warning(
            json.dumps(
                {
                    "event": "operation",
                    "operation": operation,
                    "status": "failed",
                    "code": error.code,
                }
            )
        )
        raise error from None
