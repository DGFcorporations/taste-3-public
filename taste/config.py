"""Trusted runtime settings; all writable paths belong to this repository."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

ROOT = Path(__file__).resolve().parents[1]
SEED_COUNT = 5
RETRIEVAL_LIMIT = 12
EDGE_MIN_WEIGHT = 0.5
JUDGE_GAP = 0.15
RULE_COVERAGE_THRESHOLD = 0.6
HONE_LIMIT = 15
HONE_STOP_SCORE = 0.8
DRIFT_DAYS = 30
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def confined_path(path: str | Path) -> Path:
    raw = Path(path)
    try:
        resolved = (raw if raw.is_absolute() else ROOT / raw).resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError("Runtime path cannot be resolved safely") from exc
    if not resolved.is_relative_to(ROOT):
        raise ValueError("Runtime path must stay inside the repository")
    return resolved


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, validate_default=True)

    db_path: Path | str = ROOT / "data" / "taste-3.db"
    model_cache: Path = ROOT / "_agent-scratch" / "models"
    api_base: str | None = None
    api_model: str | None = None
    api_key: SecretStr | None = None
    timeout_s: float = Field(default=30.0, gt=0.0, le=120.0, allow_inf_nan=False)

    @field_validator("db_path", mode="before")
    @classmethod
    def confined_database(cls, value: str | Path) -> str | Path:
        return ":memory:" if value == ":memory:" else confined_path(value)

    @field_validator("model_cache", mode="before")
    @classmethod
    def confined_cache(cls, value: str | Path) -> Path:
        return confined_path(value)

    @field_validator("api_base")
    @classmethod
    def https_endpoint(cls, value: str | None) -> str | None:
        if value is not None:
            parsed = urlsplit(value)
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("Provider endpoint requires HTTPS without embedded credentials")
        return value

    @classmethod
    def from_env(cls) -> Settings:
        names = {
            "db_path": "TASTE3_DB", "model_cache": "TASTE3_MODEL_CACHE",
            "api_base": "TASTE3_API_BASE", "api_model": "TASTE3_API_MODEL",
            "api_key": "TASTE3_API_KEY", "timeout_s": "TASTE3_API_TIMEOUT",
        }
        return cls.model_validate({field: os.environ[name] for field, name in names.items() if name in os.environ})
