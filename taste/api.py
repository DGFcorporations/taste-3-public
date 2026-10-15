"""Private HTTP v1 adapter with bounded bodies and sanitized failures."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from taste.config import Settings
from taste.embed import Embedder
from taste.errors import DomainError
from taste.providers import Generator, Judge
from taste.service import MAX_BODY_BYTES, TasteService
from taste.store import Store
from taste.temporal import approve_standard


class BodyLimit:
    def __init__(self, app: ASGIApp, limit: int) -> None:
        self.app, self.limit = app, limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        chunks: list[bytes] = []
        count = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body = message.get("body", b"")
            count += len(body)
            if count > self.limit:
                response = JSONResponse(
                    DomainError(
                        "request_too_large", "Request body exceeds the supported limit.", 413
                    ).payload(),
                    status_code=413,
                )
                await response(scope, receive, send)
                return
            chunks.append(body)
            if not message.get("more_body", False):
                break
        consumed = False

        async def replay() -> Message:
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def create_app(
    service: TasteService | None = None,
    *,
    store: Store | None = None,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    judge: Judge | None = None,
    generator: Generator | None = None,
    max_body_bytes: int = MAX_BODY_BYTES,
) -> FastAPI:
    actual = (
        service
        if service is not None
        else TasteService(
            store=store, settings=settings, embedder=embedder, judge=judge, generator=generator
        )
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if service is None:
                actual.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.service = actual
    app.add_middleware(BodyLimit, limit=max_body_bytes)

    @app.exception_handler(DomainError)
    async def domain_failure(request: Request, error: DomainError) -> JSONResponse:
        return JSONResponse(error.payload(), status_code=error.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_failure(request: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            DomainError("invalid_request", "Request validation failed.").payload(), status_code=422
        )

    @app.exception_handler(Exception)
    async def unexpected_failure(request: Request, error: Exception) -> JSONResponse:
        return JSONResponse(
            DomainError("internal_error", "The operation could not be completed.", 500).payload(),
            status_code=500,
        )

    @app.exception_handler(HTTPException)
    async def routing_failure(request: Request, error: HTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(error.status_code, "http_error")
        payload = DomainError(
            code, "HTTP request could not be completed.", error.status_code
        ).payload()
        return JSONResponse(payload, status_code=error.status_code, headers=error.headers)

    @app.get("/meta")
    def meta() -> dict[str, Any]:
        return actual.meta()

    @app.get("/check")
    def check(profile_id: str = "owner") -> dict[str, Any]:
        return actual.check(profile_id)

    @app.get("/standards")
    def standards(
        profile_id: str = "owner", state: str | None = None, category: str | None = None
    ) -> dict[str, Any]:
        return actual.dispatch(
            "standards", {"profile_id": profile_id, "state": state, "category": category}
        )


    @app.post("/standards/{standard_id}/confirm")
    async def confirm(standard_id: str, request: Request) -> dict[str, Any]:
        """Owner-authority activation: confirm the current revision of one standard.

        The body may carry an optional ``rationale`` (and ``profile_id``); when
        absent the stored rationale stands. Unknown ids answer 404 and a still
        blank rationale answers 422 — no decision is activated without one.
        """
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError, RecursionError):
            body = {}
        if body is None:
            body = {}
        if not isinstance(body, dict):
            raise DomainError("invalid_request", "Request body must be a JSON object.", 422)
        profile_id = body.get("profile_id", "owner")
        if not isinstance(profile_id, str):
            raise DomainError("invalid_request", "Invalid profile id.", 422)

        def activate() -> dict[str, Any]:
            standard = actual.store.get_standard(profile_id, standard_id)
            if standard is None:
                raise DomainError(
                    "not_found", "The requested standard does not exist.", 404
                )
            rationale = body.get("rationale")
            if rationale is None:
                rationale = standard.rationale
            if not isinstance(rationale, str) or not rationale.strip():
                raise DomainError(
                    "invalid_rationale", "A nonblank rationale is required.", 422
                )
            target_revisions: dict[str, Any] = {}
            for target_id in standard.supersedes:
                target = actual.store.get_standard(profile_id, target_id)
                if target is None:
                    raise DomainError(
                        "state_conflict", "The superseded standard does not exist.", 409
                    )
                target_revisions[target.id] = target.updated
            approved = approve_standard(
                actual.store,
                profile_id,
                standard.id,
                rationale.strip(),
                True,
                expected_updated=standard.updated,
                supersedes=list(standard.supersedes) or None,
                expected_targets=target_revisions or None,
            )
            return approved.model_dump(mode="json")

        return await run_in_threadpool(activate)

    def endpoint_for(operation: str) -> Callable[[Request], Awaitable[dict[str, Any]]]:
        async def endpoint(request: Request) -> dict[str, Any]:
            try:
                body = await request.json()
            except (ValueError, UnicodeDecodeError, RecursionError):
                raise DomainError("invalid_request", "Request must contain valid JSON.") from None
            return await run_in_threadpool(actual.dispatch, operation, body)

        return endpoint

    for operation in (
        "interview",
        "mirror",
        "packet",
        "score",
        "feedback",
        "hone",
        "ingest",
        "import_standards",
    ):
        path = "/standards/import" if operation == "import_standards" else "/" + operation
        app.add_api_route(path, endpoint_for(operation), methods=["POST"], name=operation)
    return app
