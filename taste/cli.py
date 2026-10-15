"""Private operator commands; activation requires an exact interactive review."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from taste.config import ROOT, Settings, confined_path
from taste.errors import DomainError
from taste.mcp_server import run_stdio
from taste.service import MAX_BODY_BYTES, TasteService
from taste.temporal import approve_standard, drift_report, history_for, parse_utc


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="taste")
    result.add_argument("--db")
    commands = result.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--socket", default=str(ROOT / "data" / "taste-3.sock"))
    commands.add_parser("mcp")
    for name in ("export", "import", "reviews", "approve", "supersede", "history", "drift"):
        command = commands.add_parser(name)
        command.add_argument("--profile", default="owner")
        if name == "export":
            command.add_argument(
                "--state", choices=["provisional", "active", "superseded", "retired"]
            )
            command.add_argument("--output")
        elif name == "import":
            command.add_argument("file")
        elif name in {"approve", "supersede"}:
            command.add_argument("standard_id")
            command.add_argument("--rationale", required=True)
            if name == "supersede":
                command.add_argument("old_ids", nargs="+")
        elif name == "history":
            command.add_argument("standard_id", nargs="?")
        elif name == "drift":
            command.add_argument("--as-of")
    return result


def confirm_standard(service: TasteService, args: argparse.Namespace) -> dict[str, Any] | None:
    standard = service.store.get_standard(args.profile, args.standard_id)
    if standard is None:
        raise DomainError("state_conflict", "The requested standard does not exist.", 409)
    replacements = args.old_ids if args.command == "supersede" else standard.supersedes
    print("Standard: " + standard.id)
    print("Polarity: " + standard.polarity.value)
    print("Category: " + standard.category.value)
    print("Revision: " + standard.updated.isoformat())
    print("Directive: " + standard.directive)
    print("Stored rationale: " + standard.rationale)
    print("Approval rationale: " + args.rationale)
    print(
        "Comprehension: Apply this "
        + standard.polarity.value
        + " directive in the "
        + standard.category.value
        + " layer within its approved validity window."
    )
    target_revisions: dict[str, datetime | str] = {}
    if replacements:
        for id in replacements:
            old = service.store.get_standard(args.profile, id)
            if old is None:
                raise DomainError("state_conflict", "The superseded standard does not exist.", 409)
            print("Replacement target: " + old.id + " — " + old.directive)
            print("Target revision: " + old.updated.isoformat())
            print("Target polarity: " + old.polarity.value)
            print("Target category: " + old.category.value)
            print("Target rationale: " + old.rationale)
            target_revisions[old.id] = old.updated
    if input("Approve this exact revision? Type yes: ").strip().casefold() != "yes":
        return None
    approved = approve_standard(
        service.store,
        args.profile,
        standard.id,
        args.rationale,
        True,
        expected_updated=standard.updated,
        supersedes=replacements,
        expected_targets=target_revisions,
    )
    return approved.model_dump(mode="json")


def atomic_export(path: Path, encoded: str) -> None:
    """Keep an existing export intact until its complete replacement is ready."""
    path = confined_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".taste-export-", delete=False
        ) as output:
            temporary = Path(output.name)
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        settings = Settings.from_env()
        if args.db is not None:
            settings = Settings.model_validate({**settings.model_dump(), "db_path": args.db})
        if args.command == "serve":
            from granian import Granian
            from granian.constants import Interfaces

            socket = confined_path(args.socket)
            socket.parent.mkdir(parents=True, exist_ok=True)
            os.environ["TASTE3_DB"] = str(settings.db_path)
            Granian(
                "taste.api:create_app",
                uds=socket,
                uds_permissions=0o600,
                interface=Interfaces.ASGI,
                factory=True,
                log_enabled=False,
                log_access=False,
            ).serve()
            return 0
        service = TasteService(settings=settings)
        try:
            if args.command == "mcp":
                asyncio.run(run_stdio(service))
                return 0
            if args.command == "export":
                output: Any = service.standards(args.profile, args.state)
            elif args.command == "import":
                path = confined_path(args.file)
                with path.open("rb") as source:
                    content = source.read(MAX_BODY_BYTES + 1)
                if len(content) > MAX_BODY_BYTES:
                    raise DomainError(
                        "request_too_large", "Import file exceeds the supported limit.", 413
                    )
                value = json.loads(content)
                rows = value.get("standards") if isinstance(value, dict) else value
                output = service.dispatch(
                    "import_standards", {"profile_id": args.profile, "standards": rows}
                )
            elif args.command == "reviews":
                output = {"reviews": service.store.list_reviews(args.profile)}
            elif args.command in {"approve", "supersede"}:
                output = confirm_standard(service, args)
                if output is None:
                    return 1
            elif args.command == "history":
                output = history_for(service.store, args.profile, args.standard_id)
            else:
                instant = parse_utc(args.as_of) if args.as_of else datetime.now(UTC)
                output = drift_report(service.store, args.profile, now=instant)
            encoded = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
            if args.command == "export" and args.output:
                atomic_export(confined_path(args.output), encoded)
            else:
                print(encoded, end="")
            return 0
        finally:
            service.close()
    except DomainError as error:
        print(json.dumps(error.payload()), file=sys.stderr)
    except (ValidationError, ValueError, OSError, EOFError):
        print(
            json.dumps(
                DomainError(
                    "invalid_request", "The operator request could not be completed."
                ).payload()
            ),
            file=sys.stderr,
        )
    except Exception:
        print(
            json.dumps(
                DomainError(
                    "internal_error", "The operation could not be completed.", 500
                ).payload()
            ),
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
