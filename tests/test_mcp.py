"""Actual stdio MCP initialization, tool calls, and HTTP payload parity."""

import asyncio
import importlib
import os
import sys

import pytest
from fastapi.testclient import TestClient
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolResult, Request

from taste.config import ROOT
from taste.store import Store


def module(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pytest.fail(f"Missing transport implementation: {name}")


def test_exact_four_mcp_tools_and_business_payload_parity():
    service_class = module("taste.service").TasteService
    with Store(":memory:") as store:
        service = service_class(store=store)
        mcp = module("taste.mcp_server")
        assert set(mcp.TOOL_MODELS) == {
            "taste_packet",
            "taste_score",
            "taste_feedback",
            "taste_recall",
        }
        with TestClient(module("taste.api").create_app(service=service)) as client:
            for name, route, arguments in [
                ("taste_packet", "/packet", {"task_ctx": {"goal": "Synthetic"}}),
                ("taste_score", "/score", {"candidates": [{"label": "A", "content": "Synthetic"}]}),
                ("taste_recall", "/standards", {"profile_id": "owner", "state": "retired"}),
                (
                    "taste_feedback",
                    "/feedback",
                    {"pick": "A", "rationale": "   ", "task_ctx": {"goal": "PRIVATE"}},
                ),
            ]:
                result = mcp.invoke_tool(service, name, arguments)
                response = (
                    client.get(route, params=arguments)
                    if name == "taste_recall"
                    else client.post(route, json=arguments)
                )
                assert result.structured_content == response.json()
                assert result.is_error == (response.status_code >= 400)
        service.close()


def test_actual_stdio_protocol_and_sanitized_validation(tmp_path):
    module("taste.mcp_server")
    env = dict(os.environ)
    env["TASTE3_DB"] = str(tmp_path / "stdio.db")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TMPDIR"] = str(ROOT / "_agent-scratch" / "tmp")
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "taste.mcp_server"], env=env, cwd=ROOT
    )

    async def exercise():
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write, read_timeout_seconds=15) as session,
        ):
            initialized = await session.initialize()
            assert initialized.server_info.name == "taste-3"
            assert {tool.name for tool in (await session.list_tools()).tools} == {
                "taste_packet",
                "taste_score",
                "taste_feedback",
                "taste_recall",
            }
            packet = await session.call_tool(
                "taste_packet", {"profile_id": "synthetic", "task_ctx": {"type": "write_thread"}}
            )
            assert packet.structured_content["profile_id"] == "synthetic"
            scored = await session.call_tool(
                "taste_score",
                {
                    "profile_id": "synthetic",
                    "task_ctx": {"task_ref": "stdio-task"},
                    "candidates": [
                        {"label": "A", "content": "Synthetic chosen text"},
                        {"label": "B", "content": "Synthetic rejected text"},
                    ],
                },
            )
            assert len(scored.structured_content["scores"]) == 2
            assert all(
                "positive" in row["breakdown"] for row in scored.structured_content["scores"]
            )
            accepted = await session.call_tool(
                "taste_feedback",
                {
                    "profile_id": "synthetic",
                    "task_ctx": {"task_ref": "stdio-task"},
                    "pick": "A",
                    "rationale": "Prefer synthetic clarity",
                    "rejected": ["B"],
                },
            )
            assert accepted.structured_content["accepted"] is True
            assert accepted.structured_content["record_id"]
            failure = await session.call_tool(
                "taste_feedback", {"pick": "PRIVATE-PICK", "rationale": "   "}
            )
            assert failure.is_error
            assert failure.structured_content["error"]["code"] == "invalid_request"
            assert "PRIVATE" not in str(failure.content)
            recall = await session.call_tool("taste_recall", {"state": "retired"})
            assert recall.structured_content == {"standards": []}
            malformed = await session.send_request(
                Request(
                    method="tools/call",
                    params={"name": "taste_packet", "arguments": ["PRIVATE-ARGUMENT"]},
                ),
                CallToolResult,
            )
            assert malformed.is_error
            assert malformed.structured_content["error"]["code"] == "invalid_request"
            assert "PRIVATE" not in str(malformed.content)

    asyncio.run(exercise())
