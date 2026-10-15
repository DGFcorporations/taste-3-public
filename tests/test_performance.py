"""Policy-scale evidence uses a disclosed synthetic embedder, not C1 throughput."""

import json
import tracemalloc
from datetime import UTC, datetime
from time import perf_counter
from types import SimpleNamespace

import pytest

from taste.config import ROOT
from taste.core import StandardsEngine


class SyntheticScaleEmbedder:
    model_name = "test-only:synthetic-scale-vectors"

    def embed(self, texts):
        return [[1.0, 0.0] for text in texts]


@pytest.mark.parametrize("count", [100, 1000, 10000])
def test_synthetic_scale_retains_bounded_selection_and_records_measurement(count):
    now = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        {
            "id": f"std{i:05}",
            "profile_id": "scale",
            "directive": f"Synthetic directive {i}",
            "state": "active",
            "_local_approved": True,
            "polarity": "do",
            "valid_from": now.isoformat(),
            "valid_to": None,
        }
        for i in range(count)
    ]
    edges = [
        {
            "profile_id": "scale",
            "src": "std00000",
            "dst": f"std{i:05}",
            "edge_type": "relates",
            "weight": 0.5,
        }
        for i in range(5, count)
    ]
    snapshot = SimpleNamespace(
        profile_id="scale", revision=1, evaluated_at=now, standards=rows, edges=edges, exemplars=[]
    )
    policy = StandardsEngine(
        SimpleNamespace(snapshot=lambda profile: snapshot), SyntheticScaleEmbedder()
    )
    tracemalloc.start()
    started = perf_counter()
    result = policy.context("scale", {"goal": "Synthetic scale test"})
    elapsed = perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(result.selected) == 12
    assert [row["id"] for row in result.selected[:5]] == [
        "std00000",
        "std00001",
        "std00002",
        "std00003",
        "std00004",
    ]
    evidence = {
        "nodes": count,
        "edges": len(edges),
        "elapsed_seconds": elapsed,
        "peak_retrieval_bytes": peak,
        "model_name": result.model_name,
        "claim": "Synthetic policy scan and one-hop selection only; excludes SQLite and actual MiniLM inference.",
    }
    output = ROOT / "_agent-scratch" / f"graph-scale-{count}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + "\n")
