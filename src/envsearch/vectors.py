"""Optional local multilingual embeddings and exact cosine search over a small corpus."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .corpus import Chunk
from .textnorm import normalize

DEFAULT_MODEL = "intfloat/multilingual-e5-small"


class E5Embedder:
    """Load lazily: BM25 and metadata operations never download or load a model."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        if os.environ.get('ENVSEARCH_EMBEDDING_BACKEND') == 'onnx':
            if self.model_name != DEFAULT_MODEL:
                raise ValueError('The hosted ONNX backend supports multilingual-e5-small only.')
            from .onnx_embeddings import encode
            return encode([prefix + normalize(text) for text in texts])
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError('Vector search needs optional dependencies: pip install -e ".[vectors]"') from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model.encode(
            [prefix + normalize(text) for text in texts],
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        )

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "passage: ")

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "query: ")


def _fingerprint(chunks: list[Chunk]) -> str:
    # Content as well as IDs: rebuilding can preserve IDs while changing the text.
    payload = json.dumps([(c.id, c.text) for c in chunks], ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _unit_rows(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("Embeddings must be a finite, nonempty-dimensional matrix. Rebuild with `envsearch build --vectors`.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Embeddings contain a zero vector. Rebuild with `envsearch build --vectors`.")
    return values / norms


class DenseIndex:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embedder):
        self.embeddings = _unit_rows(embeddings)
        if len(self.embeddings) != len(chunks):
            raise ValueError("Embedding count differs from chunks. Rebuild with `envsearch build --vectors`.")
        self.chunk_ids = [c.id for c in chunks]
        self.fingerprint = _fingerprint(chunks)
        self.embedder = embedder

    @classmethod
    def build(cls, chunks: list[Chunk], embedder=None) -> "DenseIndex":
        if not chunks:
            raise ValueError("Cannot build vectors for an empty corpus. Fetch documents first.")
        embedder = embedder or E5Embedder()
        return cls(chunks, embedder.embed_documents([c.text for c in chunks]), embedder)

    def search(self, query: str, candidates: list[int], k: int) -> list[tuple[int, float]]:
        if not candidates or k <= 0:
            return []
        query_vectors = _unit_rows(self.embedder.embed_queries([query]))
        if query_vectors.shape != (1, self.embeddings.shape[1]):
            raise ValueError("Query embedding dimensions differ from index. Rebuild with `envsearch build --vectors`.")
        scores = self.embeddings[candidates] @ query_vectors[0]
        order = np.argsort(-scores, kind="stable")[:k]
        return [(candidates[i], float(scores[i])) for i in order]

    def save(self, path: Path) -> None:
        np.save(path / "embeddings.npy", self.embeddings, allow_pickle=False)
        (path / "vectors.json").write_text(json.dumps({
            "version": 1, "model": self.embedder.model_name,
            "chunk_ids": self.chunk_ids, "fingerprint": self.fingerprint,
        }), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, chunks: list[Chunk]) -> "DenseIndex | None":
        meta_path, array_path = path / "vectors.json", path / "embeddings.npy"
        if not meta_path.exists() and not array_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (meta["version"] != 1 or meta["chunk_ids"] != [c.id for c in chunks]
                    or meta["fingerprint"] != _fingerprint(chunks)):
                raise ValueError("Stale vector index")
            array = np.load(array_path, allow_pickle=False)
            return cls(chunks, array, E5Embedder(meta["model"]))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError("Invalid or stale vector index. Rebuild with `envsearch build --vectors`.") from exc
