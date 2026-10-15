"""Lazy CPU MiniLM inference with repository-local downloads and bounded vectors."""

from __future__ import annotations

import math
import os
import unicodedata
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

from taste.config import MINILM_MODEL, ROOT, confined_path
from taste.errors import DomainError

_MODEL_SETUP_LOCK = RLock()


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def checked_vector(values: Any) -> list[float]:
    vector = [float(value) for value in values]
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("Embedding vectors must be nonempty and finite.")
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Embedding vectors must have a finite nonzero norm.")
    return vector


def cosine(left: list[float], right: list[float]) -> float:
    a, b = checked_vector(left), checked_vector(right)
    if len(a) != len(b):
        raise ValueError("Embedding dimensions must match.")
    a_norm, b_norm = math.hypot(*a), math.hypot(*b)
    value = math.fsum((x / a_norm) * (y / b_norm) for x, y in zip(a, b, strict=True))
    return max(-1.0, min(1.0, value))


class MiniLMEmbedder:
    model_name = MINILM_MODEL

    def __init__(self, cache_dir: str | Path | None = None, *, cache_size: int = 4096) -> None:
        self.cache_dir = confined_path(
            cache_dir if cache_dir is not None else ROOT / "_agent-scratch" / "models"
        )
        if not self.cache_dir.is_relative_to(ROOT):
            raise ValueError("Model cache must belong to the repository.")
        if cache_size < 1:
            raise ValueError("Vector cache size must be positive.")
        self.cache_size = cache_size
        self._vectors: OrderedDict[tuple[str, str], tuple[float, ...]] = OrderedDict()
        self._model: Any = None
        self._lock = RLock()
        self._fingerprint = self.model_name + ":fastembed:cpu:onnx"

    def _load_model(self) -> Any:
        if self._model is None:
            # Hub imports snapshot environment constants. Confine both the
            # initial import and an already-imported optional Xet downloader.
            with _MODEL_SETUP_LOCK:
                self.cache_dir = confined_path(self.cache_dir)
                xet_cache = str(confined_path(self.cache_dir / "xet"))
                previous_env = os.environ.get("HF_XET_CACHE")
                os.environ["HF_XET_CACHE"] = xet_cache
                try:
                    from huggingface_hub import constants

                    previous_constant = constants.HF_XET_CACHE
                    constants.HF_XET_CACHE = xet_cache
                    try:
                        from fastembed import TextEmbedding
                        from loguru import logger

                        self.cache_dir.mkdir(parents=True, exist_ok=True)
                        # Library download failures may otherwise log raw paths.
                        logger.disable("fastembed")
                        try:
                            self._model = TextEmbedding(
                                model_name=self.model_name,
                                cache_dir=str(self.cache_dir),
                                providers=["CPUExecutionProvider"],
                                cuda=False,
                                threads=2,
                            )
                        finally:
                            logger.enable("fastembed")
                    finally:
                        constants.HF_XET_CACHE = previous_constant
                finally:
                    if previous_env is None:
                        os.environ.pop("HF_XET_CACHE", None)
                    else:
                        os.environ["HF_XET_CACHE"] = previous_env
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        normalized = [" ".join(unicodedata.normalize("NFKC", text).split()) for text in texts]
        with self._lock:
            try:
                keys = [(self._fingerprint, text) for text in normalized]
                missing = list(dict.fromkeys(key for key in keys if key not in self._vectors))
                current = {key: self._vectors[key] for key in keys if key in self._vectors}
                if missing:
                    vectors = list(
                        self._load_model().embed([key[1] for key in missing], batch_size=64)
                    )
                    if len(vectors) != len(missing):
                        raise ValueError("Embedding output count mismatch.")
                    for key, vector in zip(missing, vectors, strict=True):
                        checked = tuple(checked_vector(vector))
                        if len(checked) != 384:
                            raise ValueError("MiniLM embedding dimensions must be 384.")
                        current[key] = checked
                for key in keys:
                    self._vectors[key] = current[key]
                    self._vectors.move_to_end(key)
                    while len(self._vectors) > self.cache_size:
                        self._vectors.popitem(last=False)
                return [list(current[key]) for key in keys]
            except Exception:
                raise DomainError(
                    "dependency_unavailable", "Semantic embedding is unavailable.", 503
                ) from None

    def close(self) -> None:
        with self._lock:
            self._vectors.clear()
            self._model = None
