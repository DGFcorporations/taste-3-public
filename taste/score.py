"""Auditable rules gate with cost-shaped, validated borderline judging."""

from __future__ import annotations

import math
import re
import unicodedata
from decimal import Decimal

from .config import JUDGE_GAP, RULE_COVERAGE_THRESHOLD
from .models import Candidate, CandidateScore, Packet
from .providers import Judge

_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "to",
        "of",
        "and",
        "or",
        "never",
        "do",
        "not",
        "use",
        "avoid",
        "always",
        "must",
        "should",
        "be",
        "with",
        "that",
    }
)


def tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())) - _STOP


def rule_match(rule: str, content: str) -> float:
    """Quoted literal bans are exact; prose uses documented term coverage.

    This heuristic is intentionally inspectable. It does not pretend to solve
    arbitrary instruction semantics or sentiment/negation understanding.
    """
    phrases = re.findall(r'"([^"]+)"|“([^”]+)”|\'([^\']+)\'', unicodedata.normalize("NFKC", rule))
    if phrases:
        normalized = " ".join(unicodedata.normalize("NFKC", content).casefold().split())
        return float(
            any(
                " ".join(next(p for p in group if p).casefold().split()) in normalized
                for group in phrases
            )
        )
    terms = tokens(rule)
    return len(terms & tokens(content)) / len(terms) if terms else 0.0


def rules_score(packet: Packet, candidate: Candidate) -> CandidateScore:
    content = candidate.content or ""
    positive_rules = [d.text for d in packet.directives if d.polarity != "never"]
    positive = (
        sum(rule_match(rule, content) for rule in positive_rules) / len(positive_rules)
        if positive_rules
        else 0.5
    )
    # Polarity is the authoritative contract field. A prohibition does not
    # become a requirement because its imperative lacks an English prefix.
    bans = list(dict.fromkeys(d.text for d in packet.directives if d.polarity == "never"))
    constraints = list(
        dict.fromkeys(c.text for c in packet.constraints if c.hard and c.text not in bans)
    )
    hard = bans + constraints
    violations = [rule for rule in bans if rule_match(rule, content) >= RULE_COVERAGE_THRESHOLD]
    violations.extend(rule for rule in constraints if constraint_violated(rule, content))
    negative = len(violations) / len(hard) if hard else 0.0
    return CandidateScore(
        ref=candidate.content_ref or candidate.label,
        score=0.0 if violations else positive,
        breakdown={
            "positive": positive,
            "negative": negative,
            "judge_used": 0.0,
            "judge_degraded": 0.0,
        },
        violations=violations,
    )


def constraint_violated(rule: str, content: str) -> bool:
    normalized = rule.casefold().strip()
    maximum = re.search(
        r"\b(at most|maximum(?: of)?|max|no more than|under)\s+(\d+)\s+words\b", normalized
    )
    if maximum:
        limit = int(maximum.group(2)) - int(maximum.group(1) == "under")
        return len(content.split()) > limit
    minimum = re.search(r"\b(?:at least|minimum(?: of)?|min)\s+(\d+)\s+words\b", normalized)
    if minimum:
        return len(content.split()) < int(minimum.group(1))
    forbidden = re.match(r"(?:never|avoid|do not|don't|must not|no\b)", normalized)
    coverage = rule_match(rule, content)
    return coverage >= RULE_COVERAGE_THRESHOLD if forbidden else coverage < RULE_COVERAGE_THRESHOLD


def score_packet(
    packet: Packet, candidates: list[Candidate], judge: Judge | None = None
) -> list[CandidateScore]:
    results = [rules_score(packet, candidate) for candidate in candidates]
    eligible = sorted(
        (index for index, score in enumerate(results) if not score.violations),
        key=lambda index: (-results[index].score, results[index].ref),
    )
    if len(eligible) < 2:
        return results
    top, second = eligible[:2]
    # Decimal subtraction respects the declared threshold for finite decimal
    # scores such as 0.35 and 0.20, whose binary subtraction rounds below 0.15.
    gap = Decimal(str(results[top].score)) - Decimal(str(results[second].score))
    if gap >= Decimal(str(JUDGE_GAP)):
        return results
    try:
        if judge is None:
            raise ValueError("Unconfigured judge")
        selected = [candidates[index] for index in eligible]
        judged = judge.judge(packet, selected)
        refs = {results[index].ref for index in eligible}
        if set(judged) != refs or any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in judged.values()
        ):
            raise ValueError("Invalid judge score set")
        for index in eligible:
            results[index].score = float(judged[results[index].ref])
            results[index].breakdown.update(judge_used=1.0, judge_score=results[index].score)
    except Exception:
        # Provider failures cannot promote a hard violation or masquerade as judging.
        for index in eligible:
            results[index].breakdown["judge_degraded"] = 1.0
    return results
