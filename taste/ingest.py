"""Conservative, text-only extraction creates proposals, never approval."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from taste.errors import DomainError
from taste.models import IngestRequest, IngestResponse, Restatement, Standard
from taste.store import Store

_FORMATS = frozenset({"md", "markdown", "txt", "text", "doc"})
_IMPERATIVE = re.compile(r"^(never|avoid|do not|use|always|prefer|keep|lead|include|choose|ensure|return|write)\b", re.I)


def _identity(data: dict[str, Any]) -> tuple[str, str, str]:
    return data["category"], data["polarity"], " ".join(unicodedata.normalize("NFKC", data["directive"]).casefold().split())


def _rules(text: str, *, markdown: bool) -> list[tuple[str, str]]:
    result = []
    fenced = False
    for line in text.splitlines():
        if markdown and line.startswith(("    ", "\t")):
            continue
        line = line.strip()
        if line.startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if fenced or line.startswith(("#", "<", "!", ">")):
            continue
        line = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", line)
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            if not _IMPERATIVE.match(sentence):
                continue
            parts = re.split(r"\s+because\s+", sentence, maxsplit=1, flags=re.I)
            result.append((parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""))
    return result


def _category(directive: str, polarity: str) -> str:
    if polarity == "never":
        return "anti"
    text = directive.casefold()
    if any(word in text for word in ("heading", "section", "lead", "opening", "format", "paragraph")):
        return "structure"
    if any(word in text for word in ("tone", "color", "style", "font", "sentence")):
        return "aesthetic"
    if any(word in text for word in ("review", "confirm", "step", "before", "after")):
        return "procedural"
    return "doctrine"


def ingest(store: Store, req: IngestRequest) -> IngestResponse:
    # Validate the whole document list before any proposal is stored.
    for doc in req.docs:
        if not doc.get("id", "").strip() or "text" not in doc or doc.get("format", "txt").casefold() not in _FORMATS:
            raise DomainError("invalid_request", "Documents require an ID, text, and a supported text format.")
    seen = {_identity(row) for row in store.list_standards(req.profile_id)}
    proposals = []
    restatements = []
    for doc in req.docs:
        for directive, rationale in _rules(doc["text"], markdown=doc.get("format", "txt").casefold() in {"md", "markdown"}):
            polarity = "never" if re.match(r"^(never|avoid|do not)\b", directive, re.I) else "do"
            data = dict(profile_id=req.profile_id, category=_category(directive, polarity),
                        polarity=polarity, directive=directive, rationale=rationale,
                        provenance="persona", state="provisional", confidence=0.5,
                        variant_meta={"source_doc_id": doc["id"], "extraction": "conservative-text"})
            key = _identity(data)
            if key in seen:
                continue
            proposal = store.propose(Standard.model_validate(data))
            seen.add(key)
            proposals.append(proposal)
            implication = ("Reject outputs matching this proposed ban" if polarity == "never"
                           else "Prefer outputs following this proposed instruction")
            restatements.append(Restatement(layer=proposal, implications=[
                f"{implication}: {directive}", "This remains provisional until the owner confirms it.",
            ]))
    return IngestResponse(candidates=proposals, restatements=restatements)
