"""Exactly four MCP tools with the HTTP service's business payloads."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import anyio
from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from mcp.types import (
    INVALID_PARAMS,
    CallToolResult,
    ListToolsResult,
    RequestParams,
    TextContent,
    Tool,
)
from pydantic import Field, ValidationError

from taste.errors import DomainError
from taste.models import FeedbackRequest, PacketRequest, ScoreRequest
from taste.service import TasteService

TOOL_MODELS = {
    "taste_packet": PacketRequest,
    "taste_score": ScoreRequest,
    "taste_feedback": FeedbackRequest,
    "taste_recall": None,
}
_OPERATIONS = {
    "taste_packet": "packet",
    "taste_score": "score",
    "taste_feedback": "feedback",
    "taste_recall": "standards",
}




class RawToolParams(RequestParams):
    """Business arguments are validated by the common service boundary."""

    name: Any = None
    arguments: Any = Field(default_factory=dict)


def invoke_tool(service: TasteService, name: Any, arguments: Any) -> CallToolResult:
    try:
        if not isinstance(name, str):
            raise DomainError("invalid_request", "Invalid tool name.")
        operation = _OPERATIONS.get(name)
        if operation is None:
            raise DomainError("invalid_request", "Unknown tool.")
        payload = service.dispatch(operation, arguments)
        failed = False
    except DomainError as error:
        payload, failed = error.payload(), True
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
        structured_content=payload,
        is_error=failed,
    )


def create_server(service: TasteService) -> Server[Any]:
    async def list_tools(ctx: Any, params: Any) -> ListToolsResult:
        tools = []
        for name, model in TOOL_MODELS.items():
            schema = (
                model.model_json_schema()
                if model is not None
                else {
                    "type": "object",
                    "properties": {
                        "profile_id": {"type": "string", "default": "owner"},
                        "state": {
                            "type": "string",
                            "enum": ["provisional", "active", "superseded", "retired"],
                        },
                    },
                }
            )
            tools.append(
                Tool(
                    name=name,
                    description="Private TASTE v1 " + _OPERATIONS[name],
                    input_schema=schema,
                )
            )
        return ListToolsResult(tools=tools)

    async def call_tool(ctx: Any, params: Any) -> CallToolResult:
        return await anyio.to_thread.run_sync(invoke_tool, service, params.name, params.arguments)

    async def safe_validation(
        ctx: ServerRequestContext[Any, Any], call_next: CallNext
    ) -> HandlerResult:
        try:
            return await call_next(ctx)
        except (MCPError, ValidationError) as failure:
            if ctx.method != "tools/call" or (
                isinstance(failure, MCPError) and failure.code != INVALID_PARAMS
            ):
                raise
            # Public middleware covers spec-level validation that runs before
            # the registered handler, including a non-object arguments value.
            payload = DomainError("invalid_request", "Request validation failed.").payload()
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(payload))],
                structured_content=payload,
                is_error=True,
            )

    # No SDK input-schema callback: our service performs common validation and
    # returns sanitized domain errors instead of raw SDK argument excerpts.
    server: Server[Any] = Server("taste-3", version="0.1.0", on_list_tools=list_tools)
    server.add_request_handler("tools/call", RawToolParams, call_tool)
    server.middleware.insert(0, safe_validation)
    return server


async def run_stdio(service: TasteService | None = None) -> None:
    actual = service if service is not None else TasteService()
    try:
        server = create_server(actual)
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())
    finally:
        if service is None:
            actual.close()


def main() -> int:
    try:
        asyncio.run(run_stdio())
        return 0
    except DomainError as error:
        print(json.dumps(error.payload()), file=sys.stderr)
    except Exception:
        print(
            json.dumps(
                DomainError("internal_error", "MCP startup could not be completed.", 500).payload()
            ),
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
