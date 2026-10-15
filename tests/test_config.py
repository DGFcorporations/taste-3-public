"""Path and secret boundaries using synthetic repository-local fixtures."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from taste.config import ROOT, Settings, confined_path
from taste.errors import DomainError


def test_relative_runtime_paths_resolve_under_repository() -> None:
    assert confined_path("_agent-scratch/synthetic.db") == ROOT / "_agent-scratch" / "synthetic.db"
    settings = Settings(db_path="_agent-scratch/synthetic.db", model_cache="_agent-scratch/models")
    assert settings.db_path == ROOT / "_agent-scratch" / "synthetic.db"
    assert settings.model_cache == ROOT / "_agent-scratch" / "models"


def test_memory_database_does_not_become_a_literal_path() -> None:
    assert Settings(db_path=":memory:").db_path == ":memory:"


@pytest.mark.parametrize("value", ["/tmp/taste-synthetic.db", "../taste-synthetic.db"])
def test_paths_outside_repository_are_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        confined_path(value)
    with pytest.raises(ValidationError):
        Settings(db_path=value)
    with pytest.raises(ValidationError):
        Settings(model_cache=value)


def test_nonexistent_child_of_external_symlink_is_rejected(tmp_path: Path) -> None:
    link = tmp_path / "escape"
    link.symlink_to(ROOT.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        confined_path(link / "nonexistent-synthetic" / "database.db")


def test_symlink_remaining_inside_repository_is_allowed(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "internal"
    link.symlink_to(target, target_is_directory=True)
    assert confined_path(link / "db.sqlite") == target / "db.sqlite"


def test_settings_reads_only_explicit_runtime_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TASTE3_DB", str(tmp_path / "env.db"))
    monkeypatch.setenv("TASTE3_MODEL_CACHE", str(tmp_path / "models"))
    monkeypatch.setenv("TASTE3_API_BASE", "https://synthetic.invalid/v1")
    monkeypatch.setenv("TASTE3_API_MODEL", "synthetic-flash")
    monkeypatch.setenv("TASTE3_API_KEY", "synthetic-secret")
    monkeypatch.setenv("TASTE3_API_TIMEOUT", "12.5")
    settings = Settings.from_env()
    assert settings.db_path == tmp_path / "env.db"
    assert settings.model_cache == tmp_path / "models"
    assert settings.api_base == "https://synthetic.invalid/v1"
    assert settings.api_model == "synthetic-flash"
    assert settings.api_key.get_secret_value() == "synthetic-secret"
    assert settings.timeout_s == 12.5
    assert "synthetic-secret" not in repr(settings)
    assert "synthetic-secret" not in settings.model_dump_json()


@pytest.mark.parametrize("timeout", [0, -1, 121, float("nan"), float("inf")])
def test_provider_timeout_is_finite_positive_and_bounded(timeout: float) -> None:
    with pytest.raises(ValidationError):
        Settings(timeout_s=timeout)


def test_configuration_is_frozen_and_does_not_create_runtime_directories(tmp_path: Path) -> None:
    cache = tmp_path / "new-model-cache"
    settings = Settings(db_path=tmp_path / "new-db" / "db.sqlite", model_cache=cache)
    assert not cache.exists()
    assert not (tmp_path / "new-db").exists()
    with pytest.raises(ValidationError):
        settings.timeout_s = 3


def test_domain_error_has_only_safe_wire_envelope() -> None:
    error = DomainError("store_busy", "Store is temporarily busy", status_code=503)
    assert error.status_code == 503
    assert error.payload() == {"error": {"code": "store_busy", "message": "Store is temporarily busy"}}
    assert str(error) == "Store is temporarily busy"


@pytest.mark.parametrize("base", ["http://synthetic.invalid/v1", "file:///synthetic", "not-a-url", "https://user:password@synthetic.invalid/v1"])
def test_provider_endpoint_requires_https_without_embedded_credentials(base: str) -> None:
    with pytest.raises(ValidationError):
        Settings(api_base=base)
