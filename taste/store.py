"""SQLite persistence owns all durable taste transitions and their invariants."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import unicodedata
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from taste.errors import DomainError
from taste.models import Candidate, DecisionRecord, Edge, Standard

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
ROOT = Path(__file__).resolve().parent.parent
_ENGINE_METADATA = {"local_approved", "approval_pending", "valid_from", "valid_to", "aliases"}


def utcnow() -> datetime:
    return datetime.now(UTC)


def _json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def _instant(value: datetime | str) -> datetime:
    try:
        result = (
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            if isinstance(value, str)
            else value
        )
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError
        return result.astimezone(UTC)
    except (ValueError, TypeError, AttributeError, OverflowError):
        raise DomainError("invalid_request", "An aware UTC timestamp is required.", 422) from None


def _identity(data: dict[str, Any]) -> str:
    directive = " ".join(unicodedata.normalize("NFKC", data["directive"]).casefold().split())
    return _json([data["profile_id"], data["category"], data["polarity"], directive])


def _portable(data: dict[str, Any]) -> dict[str, Any]:
    """Remove exported internal evidence before comparing portable content."""
    result: dict[str, Any] = json.loads(_json(data))
    result.pop("_local_approved", None)
    meta = result.get("variant_meta")
    if isinstance(meta, dict):
        local = meta.get("taste3")
        if isinstance(local, dict):
            for field in _ENGINE_METADATA:
                local.pop(field, None)
            if not local:
                meta.pop("taste3", None)
        if not meta:
            result.pop("variant_meta", None)
    return result


def _token(data: dict[str, Any]) -> str:
    return hashlib.sha256(_json(_portable(data)).encode()).hexdigest()


def _edge_id(profile_id: str, src: str, dst: str, edge_type: str) -> str:
    return "edge_" + hashlib.sha256(_json([profile_id, src, dst, edge_type]).encode()).hexdigest()


def _validate_import_edges(profile_id: str, source_id: str, declared_edges: list[Any]) -> None:
    """Validate portable graph declarations without changing their source representation."""
    known_edges: dict[tuple[str, str], float] = {}
    for edge in declared_edges:
        if (
            not isinstance(edge, dict)
            or not isinstance(edge.get("target_id"), str)
            or not edge["target_id"].strip()
        ):
            raise ValueError
        if (
            edge.get("profile_id", profile_id) != profile_id
            or edge.get("target_profile_id", profile_id) != profile_id
        ):
            raise ValueError
        relation = Edge.model_validate(
            dict(
                profile_id=profile_id,
                src=source_id,
                dst=edge["target_id"],
                type=edge.get("type"),
                weight=edge.get("weight", 1),
                created=EPOCH,
            )
        )
        edge_key = (relation.type, relation.dst)
        if edge_key in known_edges and known_edges[edge_key] != relation.weight:
            raise ValueError
        known_edges[edge_key] = relation.weight


def _prepare_import_standard(
    profile_id: str, source: dict[str, Any] | Standard
) -> dict[str, Any]:
    """Copy and validate one record before the batch opens a write transaction."""
    data = (
        source.model_dump(mode="json")
        if isinstance(source, Standard)
        else json.loads(_json(source))
    )
    required = {
        "id",
        "profile_id",
        "category",
        "polarity",
        "directive",
        "rationale",
        "confidence",
        "provenance",
        "state",
        "supersedes",
        "created",
        "updated",
    }
    if not isinstance(data, dict) or not required.issubset(data):
        raise ValueError
    validated = Standard.model_validate(data)
    if validated.profile_id != profile_id:
        raise ValueError
    meta = data.get("variant_meta", {})
    if not isinstance(meta.get("taste3", {}), dict):
        raise ValueError
    for field in ("valid_from", "valid_to"):
        value = data.get(field, meta.get("taste3", {}).get(field))
        if value is not None:
            _instant(value)
    declared_edges = meta.get("edges", [])
    if not isinstance(declared_edges, list):
        raise ValueError
    _validate_import_edges(profile_id, data["id"], declared_edges)
    return data


def _render_standard(
    row: sqlite3.Row,
    aliases: list[dict[str, Any]],
    outgoing: dict[tuple[str, str], dict[str, Any]],
    *,
    internal: bool,
) -> dict[str, Any]:
    """Overlay stored evidence while retaining portable extensions and edge ordering."""
    data: dict[str, Any] = json.loads(row["payload"])
    approved = row["approval_token"] is not None and row["approval_token"] == _token(data)
    meta = data.setdefault("variant_meta", {})
    declared = {
        (edge["type"], edge["target_id"]): dict(edge) for edge in meta.get("edges", [])
    }
    for key, edge in outgoing.items():
        declared.setdefault(key, {}).update(edge)
    if declared or "edges" in meta:
        meta["edges"] = [declared[key] for key in sorted(declared)]
    meta.setdefault("taste3", {}).update(
        local_approved=approved,
        approval_pending=data["state"] == "active" and not approved,
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        aliases=aliases,
    )
    if internal:
        data["_local_approved"] = approved
        data["valid_from"] = row["valid_from"]
        data["valid_to"] = row["valid_to"]
    return data


@dataclass(frozen=True)
class Snapshot:
    profile_id: str
    revision: int
    evaluated_at: datetime
    standards: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    exemplars: list[dict[str, Any]]


_SCHEMA = """
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS profiles (
 profile_id TEXT PRIMARY KEY, revision INTEGER NOT NULL DEFAULT 0,
 evaluated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS nodes (
 profile_id TEXT NOT NULL, id TEXT NOT NULL, kind TEXT NOT NULL
 CHECK(kind IN ('standard','decision','exemplar')), PRIMARY KEY(profile_id,id));
CREATE TABLE IF NOT EXISTS standards (
 profile_id TEXT NOT NULL, id TEXT NOT NULL, semantic_identity TEXT NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('provisional','active','superseded','retired')),
 payload TEXT NOT NULL, approval_token TEXT, valid_from TEXT, valid_to TEXT,
 PRIMARY KEY(profile_id,id), UNIQUE(profile_id,semantic_identity),
 FOREIGN KEY(profile_id,id) REFERENCES nodes(profile_id,id));
CREATE INDEX IF NOT EXISTS standards_state ON standards(profile_id,state,id);
CREATE TABLE IF NOT EXISTS edges (
 profile_id TEXT NOT NULL, src TEXT NOT NULL, dst TEXT NOT NULL,
 edge_type TEXT NOT NULL CHECK(edge_type IN ('supersedes','contradicts','exemplar_of','applies_to','relates')),
 weight REAL NOT NULL CHECK(weight >= 0 AND weight <= 1), created TEXT NOT NULL,
 id TEXT NOT NULL,
 PRIMARY KEY(profile_id,src,dst,edge_type),
 CHECK(edge_type != 'supersedes' OR src != dst),
 FOREIGN KEY(profile_id,src) REFERENCES nodes(profile_id,id),
 FOREIGN KEY(profile_id,dst) REFERENCES nodes(profile_id,id));
CREATE INDEX IF NOT EXISTS edges_out ON edges(profile_id,src,edge_type);
CREATE INDEX IF NOT EXISTS edges_in ON edges(profile_id,dst,edge_type);
CREATE TABLE IF NOT EXISTS approvals (
 id INTEGER PRIMARY KEY, profile_id TEXT NOT NULL, standard_id TEXT NOT NULL,
 revision_token TEXT NOT NULL, rationale TEXT NOT NULL CHECK(length(trim(rationale)) > 0),
 created TEXT NOT NULL, FOREIGN KEY(profile_id,standard_id) REFERENCES nodes(profile_id,id));
CREATE TABLE IF NOT EXISTS aliases (
 profile_id TEXT NOT NULL, alias_id TEXT NOT NULL, canonical_id TEXT NOT NULL,
 payload TEXT NOT NULL, PRIMARY KEY(profile_id,alias_id),
 FOREIGN KEY(profile_id,canonical_id) REFERENCES standards(profile_id,id));
CREATE TABLE IF NOT EXISTS pending_edges (
 profile_id TEXT NOT NULL, src TEXT NOT NULL, dst TEXT NOT NULL, edge_type TEXT NOT NULL,
 weight REAL NOT NULL, PRIMARY KEY(profile_id,src,dst,edge_type));
CREATE TABLE IF NOT EXISTS decisions (
 profile_id TEXT NOT NULL, id TEXT NOT NULL, rationale TEXT NOT NULL
 CHECK(length(trim(rationale, char(9,10,11,12,13,28,29,30,31,32,133,160,5760,8192,8193,8194,8195,8196,8197,8198,8199,8200,8201,8202,8232,8233,8239,8287,12288))) > 0), payload TEXT NOT NULL,
 PRIMARY KEY(profile_id,id), FOREIGN KEY(profile_id,id) REFERENCES nodes(profile_id,id));
CREATE TABLE IF NOT EXISTS exemplars (
 profile_id TEXT NOT NULL, id TEXT NOT NULL, decision_id TEXT NOT NULL, payload TEXT NOT NULL,
 PRIMARY KEY(profile_id,id), FOREIGN KEY(profile_id,id) REFERENCES nodes(profile_id,id),
 FOREIGN KEY(profile_id,decision_id) REFERENCES decisions(profile_id,id));
CREATE TABLE IF NOT EXISTS candidates (
 profile_id TEXT NOT NULL, task_key TEXT NOT NULL, ref TEXT NOT NULL, label TEXT NOT NULL,
 payload TEXT NOT NULL, PRIMARY KEY(profile_id,task_key,ref));
CREATE INDEX IF NOT EXISTS candidate_labels ON candidates(profile_id,task_key,label);
CREATE TABLE IF NOT EXISTS reviews (
 id TEXT PRIMARY KEY, profile_id TEXT NOT NULL, payload TEXT NOT NULL, created TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS review_profiles ON reviews(profile_id,id);
CREATE TABLE IF NOT EXISTS hone_budgets (
 profile_id TEXT NOT NULL, task_key TEXT NOT NULL, reservations INTEGER NOT NULL CHECK(reservations >= 0),
 PRIMARY KEY(profile_id,task_key));
PRAGMA user_version=2;
COMMIT;
"""


class Store:
    def __init__(self, path: Path | str = ":memory:", clock: Callable[[], datetime] = utcnow):
        if str(path) != ":memory:":
            location = Path(path)
            if not location.is_absolute():
                location = ROOT / location
            location = location.resolve()
            if not location.is_relative_to(ROOT):
                raise DomainError(
                    "invalid_request", "Database path must stay inside the repository.", 422
                )
            location.parent.mkdir(parents=True, exist_ok=True)
            path = location
        self.clock = clock
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(path), timeout=5, isolation_level=None, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA busy_timeout=5000")
        try:
            self._connection.execute("PRAGMA journal_mode=WAL")
            version = self._connection.execute("PRAGMA user_version").fetchone()[0]
            if version > 2:
                raise DomainError("store_unavailable", "Unsupported database schema.", 503)
            if version == 0:
                self._connection.executescript(_SCHEMA)
            elif version == 1:
                self._migrate_v1()
        except (sqlite3.Error, DomainError):
            self._connection.close()
            raise

    def _migrate_v1(self) -> None:
        """Preserve pre-release scratch databases while tightening durable boundaries."""
        self._connection.execute("PRAGMA foreign_keys=OFF")
        try:
            with self._transaction() as conn:
                columns = {row["name"] for row in conn.execute("PRAGMA table_info(edges)")}
                if "id" not in columns:
                    conn.execute("ALTER TABLE edges ADD COLUMN id TEXT NOT NULL DEFAULT ''")
                for edge in conn.execute(
                    "SELECT profile_id,src,dst,edge_type FROM edges"
                ).fetchall():
                    conn.execute(
                        "UPDATE edges SET id=? WHERE profile_id=? AND src=? AND dst=? AND edge_type=?",
                        (
                            _edge_id(
                                edge["profile_id"], edge["src"], edge["dst"], edge["edge_type"]
                            ),
                            edge["profile_id"],
                            edge["src"],
                            edge["dst"],
                            edge["edge_type"],
                        ),
                    )
                conn.execute("""CREATE TABLE decisions_v2 (
                    profile_id TEXT NOT NULL, id TEXT NOT NULL, rationale TEXT NOT NULL
                    CHECK(length(trim(rationale, char(9,10,11,12,13,28,29,30,31,32,133,160,5760,8192,8193,8194,8195,8196,8197,8198,8199,8200,8201,8202,8232,8233,8239,8287,12288))) > 0), payload TEXT NOT NULL,
                    PRIMARY KEY(profile_id,id), FOREIGN KEY(profile_id,id) REFERENCES nodes(profile_id,id))""")
                conn.execute("INSERT INTO decisions_v2 SELECT * FROM decisions")
                conn.execute("DROP TABLE decisions")
                conn.execute("ALTER TABLE decisions_v2 RENAME TO decisions")
                conn.execute("PRAGMA user_version=2")
        finally:
            self._connection.execute("PRAGMA foreign_keys=ON")

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @contextmanager
    def _transaction(self, *, write: bool = True) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                yield self._connection
                self._connection.execute("COMMIT")
            except Exception as error:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                if isinstance(error, sqlite3.OperationalError):
                    raise DomainError(
                        "store_busy", "The store is temporarily unavailable.", 503
                    ) from None
                if isinstance(error, sqlite3.IntegrityError):
                    raise DomainError(
                        "identity_conflict", "The requested storage transition conflicts.", 409
                    ) from None
                raise

    def _now(self) -> str:
        return _instant(self.clock()).isoformat()

    def _bump(self, conn: sqlite3.Connection, profile_id: str, instant: str) -> None:
        conn.execute(
            "INSERT INTO profiles VALUES (?,1,?) ON CONFLICT(profile_id) DO UPDATE SET revision=revision+1,evaluated_at=excluded.evaluated_at",
            (profile_id, instant),
        )

    def _insert_standard(self, conn: sqlite3.Connection, data: dict[str, Any]) -> None:
        local = data.get("variant_meta", {}).get("taste3", {})
        valid_from = data.get("valid_from", local.get("valid_from"))
        valid_to = data.get("valid_to", local.get("valid_to"))
        conn.execute("INSERT INTO nodes VALUES (?,?,'standard')", (data["profile_id"], data["id"]))
        conn.execute(
            "INSERT INTO standards(profile_id,id,semantic_identity,state,payload,valid_from,valid_to) VALUES (?,?,?,?,?,?,?)",
            (
                data["profile_id"],
                data["id"],
                _identity(data),
                data["state"],
                _json(data),
                valid_from,
                valid_to,
            ),
        )

    def propose(self, standard: Standard) -> Standard:
        try:
            data = Standard.model_validate(standard.model_dump(mode="json")).model_dump(mode="json")
        except ValidationError:
            raise DomainError("invalid_request", "Invalid standard.", 422) from None
        data["state"] = "provisional"
        data = _portable(data)
        with self._transaction() as conn:
            self._insert_standard(conn, data)
            self._bump(conn, data["profile_id"], self._now())
        return Standard.model_validate(data)

    def get_standard(self, profile_id: str, id: str) -> Standard | None:
        with self._transaction(write=False) as conn:
            row = conn.execute(
                "SELECT payload FROM standards WHERE profile_id=? AND id=?", (profile_id, id)
            ).fetchone()
            return Standard.model_validate(json.loads(row[0])) if row else None

    def _standard_rows(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
        state: str | None = None,
        *,
        internal: bool = False,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM standards WHERE profile_id=?"
        parameters: tuple[Any, ...] = (profile_id,)
        if state is not None:
            query += " AND state=?"
            parameters += (state,)
        rows = conn.execute(query + " ORDER BY id", parameters).fetchall()
        aliases_by_standard: dict[str, list[dict[str, Any]]] = {}
        for alias in conn.execute(
            "SELECT canonical_id,payload FROM aliases WHERE profile_id=? ORDER BY alias_id",
            (profile_id,),
        ):
            aliases_by_standard.setdefault(alias["canonical_id"], []).append(
                json.loads(alias["payload"])
            )
        outgoing: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
        for edge in conn.execute(
            "SELECT src,dst,edge_type,weight FROM edges WHERE profile_id=? UNION ALL SELECT src,dst,edge_type,weight FROM pending_edges WHERE profile_id=? ORDER BY src,edge_type,dst",
            (profile_id, profile_id),
        ):
            outgoing.setdefault(edge["src"], {})[(edge["edge_type"], edge["dst"])] = dict(
                type=edge["edge_type"], target_id=edge["dst"], weight=edge["weight"]
            )
        result = []
        for row in rows:
            result.append(
                _render_standard(
                    row,
                    aliases_by_standard.get(row["id"], []),
                    outgoing.get(row["id"], {}),
                    internal=internal,
                )
            )
        return result

    def list_standards(
        self, profile_id: str = "owner", state: str | None = None
    ) -> list[dict[str, Any]]:
        if state not in (None, "provisional", "active", "superseded", "retired"):
            raise DomainError("invalid_request", "Invalid standard state.", 422)
        with self._transaction(write=False) as conn:
            return self._standard_rows(conn, profile_id, state)

    def _edge(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
        src: str,
        dst: str,
        edge_type: str,
        weight: float,
        instant: str,
    ) -> Edge:
        edge = Edge.model_validate(
            dict(
                id=_edge_id(profile_id, src, dst, edge_type),
                profile_id=profile_id,
                src=src,
                dst=dst,
                type=edge_type,
                weight=weight,
                created=instant,
            )
        )
        for id in (src, dst):
            if not conn.execute(
                "SELECT 1 FROM nodes WHERE profile_id=? AND id=?", (profile_id, id)
            ).fetchone():
                raise DomainError(
                    "invalid_request", "Edge endpoints must exist in the same profile.", 422
                )
        if edge_type == "supersedes":
            cycle = conn.execute(
                "WITH RECURSIVE reach(id) AS (SELECT ? UNION SELECT e.dst FROM edges e JOIN reach r ON e.src=r.id WHERE e.profile_id=? AND e.edge_type='supersedes') SELECT 1 FROM reach WHERE id=?",
                (dst, profile_id, src),
            ).fetchone()
            if cycle:
                raise DomainError("state_conflict", "Supersession must be acyclic.", 409)
        conn.execute(
            "INSERT INTO edges(profile_id,src,dst,edge_type,weight,created,id) VALUES (?,?,?,?,?,?,?) ON CONFLICT(profile_id,src,dst,edge_type) DO UPDATE SET weight=excluded.weight",
            (profile_id, src, dst, edge_type, weight, instant, edge.id),
        )
        stored = conn.execute(
            "SELECT id,created FROM edges WHERE profile_id=? AND src=? AND dst=? AND edge_type=?",
            (profile_id, src, dst, edge_type),
        ).fetchone()
        return edge.model_copy(update={"id": stored["id"], "created": _instant(stored["created"])})

    def add_edge(
        self, profile_id: str, src: str, dst: str, edge_type: str, weight: float = 1
    ) -> Edge:
        with self._transaction() as conn:
            instant = self._now()
            try:
                edge = self._edge(conn, profile_id, src, dst, edge_type, weight, instant)
            except ValidationError:
                raise DomainError("invalid_request", "Invalid graph edge.", 422) from None
            self._bump(conn, profile_id, instant)
            return edge

    def _resolve_edges(self, conn: sqlite3.Connection, profile_id: str, instant: str) -> None:
        for row in conn.execute(
            "SELECT * FROM pending_edges WHERE profile_id=? ORDER BY src,dst", (profile_id,)
        ).fetchall():
            endpoints = []
            for id in (row["src"], row["dst"]):
                alias = conn.execute(
                    "SELECT canonical_id FROM aliases WHERE profile_id=? AND alias_id=?",
                    (profile_id, id),
                ).fetchone()
                endpoints.append(alias[0] if alias else id)
            if all(
                conn.execute(
                    "SELECT 1 FROM nodes WHERE profile_id=? AND id=?", (profile_id, id)
                ).fetchone()
                for id in endpoints
            ):
                self._edge(
                    conn,
                    profile_id,
                    endpoints[0],
                    endpoints[1],
                    row["edge_type"],
                    row["weight"],
                    instant,
                )
                conn.execute(
                    "DELETE FROM pending_edges WHERE profile_id=? AND src=? AND dst=? AND edge_type=?",
                    (profile_id, row["src"], row["dst"], row["edge_type"]),
                )

    def import_standards(
        self, profile_id: str, standards: list[dict[str, Any] | Standard]
    ) -> dict[str, int]:
        prepared = []
        try:
            for source in standards:
                prepared.append(_prepare_import_standard(profile_id, source))
        except (ValidationError, ValueError, TypeError):
            raise DomainError(
                "invalid_request", "Invalid or mixed-profile import batch.", 422
            ) from None
        summary = dict(imported=0, aliases=0, unchanged=0)
        # Intentionally cohesive: aliases, pending relations, and the revision
        # must commit with the whole batch.
        with self._transaction() as conn:
            existing = {row["id"]: row for row in self._standard_rows(conn, profile_id)}
            for data in prepared:
                same = conn.execute(
                    "SELECT payload FROM standards WHERE profile_id=? AND id=?",
                    (profile_id, data["id"]),
                ).fetchone()
                alias = conn.execute(
                    "SELECT payload FROM aliases WHERE profile_id=? AND alias_id=?",
                    (profile_id, data["id"]),
                ).fetchone()
                if same or alias:
                    source = json.loads((same or alias)[0])
                    stored = existing.get(data["id"], source)
                    if _portable(source) != _portable(data) and _portable(stored) != _portable(
                        data
                    ):
                        raise DomainError(
                            "identity_conflict",
                            "An imported identifier conflicts with stored content.",
                            409,
                        )
                    summary["unchanged"] += 1
                    continue
                equivalent = conn.execute(
                    "SELECT id FROM standards WHERE profile_id=? AND semantic_identity=?",
                    (profile_id, _identity(data)),
                ).fetchone()
                if equivalent:
                    conn.execute(
                        "INSERT INTO aliases VALUES (?,?,?,?)",
                        (profile_id, data["id"], equivalent[0], _json(data)),
                    )
                    summary["aliases"] += 1
                else:
                    self._insert_standard(conn, data)
                    summary["imported"] += 1
                for dst in data.get("supersedes", []):
                    conn.execute(
                        "INSERT OR IGNORE INTO pending_edges VALUES (?,?,?,'supersedes',1)",
                        (profile_id, data["id"], dst),
                    )
                for edge in data.get("variant_meta", {}).get("edges", []):
                    conn.execute(
                        "INSERT INTO pending_edges VALUES (?,?,?,?,?) ON CONFLICT(profile_id,src,dst,edge_type) DO UPDATE SET weight=excluded.weight",
                        (
                            profile_id,
                            data["id"],
                            edge["target_id"],
                            edge["type"],
                            edge.get("weight", 1),
                        ),
                    )
            if summary["imported"] or summary["aliases"]:
                instant = self._now()
                self._resolve_edges(conn, profile_id, instant)
                self._bump(conn, profile_id, instant)
        return summary

    def approve(
        self,
        profile_id: str,
        id: str,
        rationale: str,
        *,
        owner_confirmed: bool = False,
        expected_updated: datetime | str | None = None,
        supersedes: list[str] | None = None,
        expected_targets: dict[str, datetime | str] | None = None,
    ) -> Standard:
        if owner_confirmed is not True or expected_updated is None:
            raise DomainError(
                "state_conflict",
                "Explicit owner confirmation of the current revision is required.",
                409,
            )
        if not isinstance(rationale, str) or not rationale.strip():
            raise DomainError("invalid_rationale", "A nonblank rationale is required.", 422)
        # Intentionally cohesive: approval evidence and every superseded target
        # form one atomic transition.
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM standards WHERE profile_id=? AND id=?", (profile_id, id)
            ).fetchone()
            if not row:
                raise DomainError("state_conflict", "The requested standard does not exist.", 409)
            data = json.loads(row["payload"])
            if _instant(expected_updated) != _instant(data["updated"]):
                raise DomainError(
                    "state_conflict", "The approved revision is no longer current.", 409
                )
            if data["state"] in ("superseded", "retired"):
                raise DomainError(
                    "state_conflict", "Historical standards cannot be reactivated.", 409
                )
            old_ids = supersedes if supersedes is not None else data.get("supersedes", [])
            old_rows = []
            if len(set(old_ids)) != len(old_ids):
                raise DomainError("state_conflict", "Duplicate supersession targets.", 409)
            if old_ids and (expected_targets is None or set(expected_targets) != set(old_ids)):
                raise DomainError(
                    "state_conflict", "Exact supersession target revisions require review.", 409
                )
            for old_id in old_ids:
                old = conn.execute(
                    "SELECT * FROM standards WHERE profile_id=? AND id=?", (profile_id, old_id)
                ).fetchone()
                if (
                    old_id == id
                    or not old
                    or old["state"] != "active"
                    or old["approval_token"] != _token(json.loads(old["payload"]))
                ):
                    raise DomainError(
                        "state_conflict", "Supersession requires an approved active target.", 409
                    )
                if expected_targets is not None and _instant(expected_targets[old_id]) != _instant(
                    json.loads(old["payload"])["updated"]
                ):
                    raise DomainError(
                        "state_conflict", "A reviewed supersession target has changed.", 409
                    )
                old_rows.append(old)
            instant = self._now()
            updated = max(
                _instant(instant), _instant(data["updated"]) + timedelta(microseconds=1)
            ).isoformat()
            data.update(state="active", rationale=rationale, updated=updated, supersedes=old_ids)
            data["valid_from"] = instant
            data["valid_to"] = None
            token = _token(data)
            conn.execute(
                "UPDATE standards SET state='active',payload=?,approval_token=?,valid_from=?,valid_to=NULL WHERE profile_id=? AND id=?",
                (_json(data), token, instant, profile_id, id),
            )
            for old in old_rows:
                self._edge(conn, profile_id, id, old["id"], "supersedes", 1, instant)
                old_data = json.loads(old["payload"])
                old_updated = max(
                    _instant(instant), _instant(old_data["updated"]) + timedelta(microseconds=1)
                ).isoformat()
                old_data.update(state="superseded", updated=old_updated, valid_to=instant)
                conn.execute(
                    "UPDATE standards SET state='superseded',payload=?,valid_to=? WHERE profile_id=? AND id=?",
                    (_json(old_data), instant, profile_id, old["id"]),
                )
            conn.execute(
                "INSERT INTO approvals(profile_id,standard_id,revision_token,rationale,created) VALUES (?,?,?,?,?)",
                (profile_id, id, token, rationale, instant),
            )
            self._bump(conn, profile_id, instant)
            return Standard.model_validate(data)

    def _put_decision(self, conn: sqlite3.Connection, record: DecisionRecord) -> None:
        conn.execute("INSERT INTO nodes VALUES (?,?,'decision')", (record.profile_id, record.id))
        conn.execute(
            "INSERT INTO decisions VALUES (?,?,?,?)",
            (record.profile_id, record.id, record.rationale, _json(record.model_dump(mode="json"))),
        )

    def put_decision(self, record: DecisionRecord) -> None:
        try:
            record = DecisionRecord.model_validate(record.model_dump(mode="json"))
        except ValidationError:
            raise DomainError(
                "invalid_rationale", "A valid decision with nonblank rationale is required.", 422
            ) from None
        with self._transaction() as conn:
            self._put_decision(conn, record)
            self._bump(conn, record.profile_id, self._now())

    def register_candidates(
        self, profile_id: str, task_key: str, candidates: list[Candidate]
    ) -> list[Candidate]:
        labels = [item.label for item in candidates]
        if len(set(labels)) != len(labels):
            raise DomainError(
                "invalid_request", "Candidate labels must be unique within a batch.", 422
            )
        registered = []
        with self._transaction() as conn:
            for candidate in candidates:
                data = Candidate.model_validate(candidate.model_dump(mode="json")).model_dump(
                    mode="json"
                )
                ref = (
                    "art_"
                    + hashlib.sha256(
                        _json(
                            [
                                profile_id,
                                task_key,
                                data["label"],
                                data["content"],
                                data["content_ref"] if data["content"] is None else "",
                            ]
                        ).encode()
                    ).hexdigest()
                )
                data["content_ref"] = ref
                conn.execute(
                    "INSERT OR IGNORE INTO candidates VALUES (?,?,?,?,?)",
                    (profile_id, task_key, ref, data["label"], _json(data)),
                )
                registered.append(Candidate.model_validate(data))
        return registered

    def _resolve_candidate(
        self, conn: sqlite3.Connection, profile_id: str, task_key: str, ref_or_label: str
    ) -> Candidate | None:
        row = conn.execute(
            "SELECT payload FROM candidates WHERE profile_id=? AND task_key=? AND ref=?",
            (profile_id, task_key, ref_or_label),
        ).fetchone()
        if row:
            return Candidate.model_validate(json.loads(row[0]))
        rows = conn.execute(
            "SELECT payload FROM candidates WHERE profile_id=? AND task_key=? AND label=?",
            (profile_id, task_key, ref_or_label),
        ).fetchall()
        return Candidate.model_validate(json.loads(rows[0][0])) if len(rows) == 1 else None

    def resolve_candidate(
        self, profile_id: str, task_key: str, ref_or_label: str
    ) -> Candidate | None:
        with self._transaction(write=False) as conn:
            return self._resolve_candidate(conn, profile_id, task_key, ref_or_label)

    def record_feedback(
        self,
        profile_id: str,
        task_ctx: dict[str, Any],
        pick: str,
        rationale: str,
        rejected: list[str],
    ) -> tuple[DecisionRecord, list[Standard]]:
        from taste.segments import task_key

        if not isinstance(rationale, str) or not rationale.strip():
            raise DomainError("invalid_rationale", "A nonblank rationale is required.", 422)
        # Intentionally cohesive: freeze content, proposals, and exemplar evidence
        # together at this revision.
        with self._transaction() as conn:
            instant = self._now()
            key = task_key(task_ctx)
            chosen = self._resolve_candidate(conn, profile_id, key, pick)
            rejects = []
            reject_refs: list[str | None] = []
            unresolved = []
            for rejection in rejected:
                candidate = self._resolve_candidate(conn, profile_id, key, rejection)
                if candidate and candidate.content is not None:
                    rejects.append(candidate.content)
                    reject_refs.append(candidate.content_ref)
                elif conn.execute(
                    "SELECT 1 FROM candidates WHERE profile_id=? AND task_key=? AND label=?",
                    (profile_id, key, rejection),
                ).fetchone() or rejection.startswith("art_"):
                    unresolved.append(rejection)
                else:
                    rejects.append(rejection)
                    reject_refs.append(None)
            record = DecisionRecord.model_validate(
                dict(
                    id="dec_" + uuid4().hex,
                    profile_id=profile_id,
                    task_ctx=task_ctx,
                    candidates=[
                        dict(
                            label=chosen.label,
                            content_ref=chosen.content_ref,
                            score=chosen.score or 0,
                        )
                    ]
                    if chosen
                    else [],
                    pick=pick,
                    rationale=rationale,
                    agent_role=str(task_ctx.get("role", "owner")),
                    provenance="owner_pick",
                    ts=instant,
                )
            )
            self._put_decision(conn, record)
            proposal_data = dict(
                id="std_" + uuid4().hex,
                profile_id=profile_id,
                category="doctrine",
                polarity="why",
                directive=" ".join(rationale.split()),
                rationale=rationale,
                confidence=0.5,
                provenance="decision",
                state="provisional",
                supersedes=[],
                created=instant,
                updated=instant,
            )
            equivalent = conn.execute(
                "SELECT payload FROM standards WHERE profile_id=? AND semantic_identity=?",
                (profile_id, _identity(proposal_data)),
            ).fetchone()
            proposals = []
            if not equivalent:
                self._insert_standard(conn, proposal_data)
                proposals.append(Standard.model_validate(proposal_data))
            standard_ids: list[str] = (
                [json.loads(equivalent["payload"])["id"]]
                if equivalent
                else [proposal.id for proposal in proposals]
            )
            exemplar_id = "ex_" + uuid4().hex
            evidence = dict(
                id=exemplar_id,
                decision_id=record.id,
                text=chosen.content if chosen else None,
                rejected=rejects,
                why=rationale,
                source_ref=chosen.content_ref if chosen else None,
                rejected_refs=reject_refs,
                unresolved_pick=None if chosen else pick,
                unresolved_rejected=unresolved,
                standard_ids=standard_ids,
            )
            conn.execute("INSERT INTO nodes VALUES (?,?,'exemplar')", (profile_id, exemplar_id))
            conn.execute(
                "INSERT INTO exemplars VALUES (?,?,?,?)",
                (profile_id, exemplar_id, record.id, _json(evidence)),
            )
            for standard_id in standard_ids:
                self._edge(conn, profile_id, exemplar_id, standard_id, "exemplar_of", 1, instant)
            self._bump(conn, profile_id, instant)
            return record, proposals

    def queue_review(
        self, profile_id: str, task_ctx: dict[str, Any], mirror: str, assumptions: list[Any]
    ) -> str:
        id = "review_" + uuid4().hex
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO reviews VALUES (?,?,?,?)",
                (
                    id,
                    profile_id,
                    _json(dict(id=id, task_ctx=task_ctx, mirror=mirror, assumptions=assumptions)),
                    self._now(),
                ),
            )
        return id

    def list_reviews(self, profile_id: str) -> list[dict[str, Any]]:
        with self._transaction(write=False) as conn:
            return [
                dict(json.loads(row["payload"]), created=row["created"])
                for row in conn.execute(
                    "SELECT payload,created FROM reviews WHERE profile_id=? ORDER BY created,id",
                    (profile_id,),
                )
            ]

    def reserve_hone(self, profile_id: str, task_key: str, limit: int = 15) -> int:
        if not isinstance(limit, int) or limit < 1 or limit > 15:
            raise DomainError("invalid_request", "Invalid task reservation limit.", 422)
        with self._transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO hone_budgets VALUES (?,?,0)", (profile_id, task_key)
            )
            row = conn.execute(
                "SELECT reservations FROM hone_budgets WHERE profile_id=? AND task_key=?",
                (profile_id, task_key),
            ).fetchone()
            if row[0] >= limit:
                raise DomainError(
                    "hone_budget_exhausted", "The task hone budget is exhausted.", 429
                )
            reservation = int(row[0]) + 1
            conn.execute(
                "UPDATE hone_budgets SET reservations=? WHERE profile_id=? AND task_key=?",
                (reservation, profile_id, task_key),
            )
            return reservation

    def snapshot(self, profile_id: str) -> Snapshot:
        with self._transaction(write=False) as conn:
            profile = conn.execute(
                "SELECT revision,evaluated_at FROM profiles WHERE profile_id=?", (profile_id,)
            ).fetchone()
            return Snapshot(
                profile_id=profile_id,
                revision=profile[0] if profile else 0,
                evaluated_at=_instant(profile[1]) if profile else EPOCH,
                standards=self._standard_rows(conn, profile_id, internal=True),
                edges=[
                    dict(row)
                    for row in conn.execute(
                        "SELECT * FROM edges WHERE profile_id=? ORDER BY src,dst,edge_type",
                        (profile_id,),
                    )
                ],
                exemplars=[
                    json.loads(row[0])
                    for row in conn.execute(
                        "SELECT payload FROM exemplars WHERE profile_id=? ORDER BY id",
                        (profile_id,),
                    )
                ],
            )
