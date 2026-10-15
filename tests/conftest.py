"""Repository-local, synthetic test isolation with no application imports."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def pytest_configure(config: pytest.Config) -> None:
    (ROOT / "_agent-scratch").mkdir(parents=True, exist_ok=True)
    if config.option.basetemp is None:
        config.option.basetemp = str(ROOT / "_agent-scratch" / f"pytest-{os.getpid()}")
    resolved = Path(config.option.basetemp).resolve()
    if not resolved.is_relative_to(ROOT / "_agent-scratch"):
        raise pytest.UsageError("Test temporary files must stay in repository scratch")



@pytest.fixture(autouse=True)
def synthetic_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep no-argument factories away from the owner's corpus and live APIs."""
    monkeypatch.setenv("TASTE3_DB", str(tmp_path / "synthetic.db"))
    monkeypatch.setenv("TASTE3_MODEL_CACHE", str(ROOT / "_agent-scratch" / "models"))
    for name in ("TASTE3_API_BASE", "TASTE3_API_MODEL", "TASTE3_API_KEY", "TASTE3_API_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
