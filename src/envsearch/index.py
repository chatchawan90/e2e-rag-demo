"""BM25 + optional multilingual vectors, metadata prefilters, and reciprocal-rank fusion."""
from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi
from filelock import FileLock

from .corpus import Chunk, Document, chunk_document, extract_pages, raw_path
from .textnorm import tokenize
from .vectors import DenseIndex


@dataclass
class Hit:
    chunk: Chunk
    score: float        # fused score (RRF) or raw BM25 for single-query search
    rank: int


class Index:
    def __init__(self, chunks: list[Chunk], docs: dict[str, Document]):
        self.chunks = chunks
        self.docs = docs
        self.by_id = {c.id: c for c in chunks}
        self._bm25 = BM25Okapi([c.tokens or ["_"] for c in chunks]) if chunks else None
        if self._bm25 is not None:
            # Positive BM25 IDF keeps common terms searchable, including a one-PDF library.
            frequencies = Counter(token for c in chunks for token in set(c.tokens or ["_"]))
            self._bm25.idf = {token: math.log1p((len(chunks) - count + 0.5) / (count + 0.5))
                              for token, count in frequencies.items()}
        self.vectors: DenseIndex | None = None

    def add_vectors(self, embedder=None) -> None:
        self.vectors = DenseIndex.build(self.chunks, embedder)

    # ------------------------------------------------------------------ build / io
    @classmethod
    def build(cls, docs: list[Document], raw_dir: Path, log=print) -> "Index":
        chunks: list[Chunk] = []
        for doc in docs:
            p = raw_path(doc, raw_dir)
            if not p.exists():
                log(f"  skip {doc.id}: not downloaded ({p})")
                continue
            pages = extract_pages(p)
            doc_chunks = chunk_document(doc, pages)
            chars = sum(len(pg.strip()) for pg in pages)
            if chars < 20 * max(1, len(pages)):  # < ~20 chars/page means image-only pages
                log(f"  WARN {doc.id}: very little text extracted — scanned PDF? Needs OCR.")
            for c in doc_chunks:
                c.tokens = tokenize(c.text)
            chunks.extend(doc_chunks)
            log(f"  {doc.id}: {len(pages)} pages -> {len(doc_chunks)} chunks")
        return cls(chunks, {d.id: d for d in docs})

    def save(self, index_dir: Path) -> None:
        index_dir.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(index_dir) + ".lock"):
            self._recover(index_dir)
            stage = Path(tempfile.mkdtemp(prefix=f".{index_dir.name}-", dir=index_dir.parent))
            backup = index_dir.with_name(index_dir.name + ".backup")
            try:
                self._write(stage)
                if index_dir.exists():
                    index_dir.replace(backup)
                try:
                    stage.replace(index_dir)
                except Exception:
                    if backup.exists():
                        backup.replace(index_dir)
                    raise
                if backup.exists():
                    shutil.rmtree(backup, ignore_errors=True)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)

    @staticmethod
    def _recover(index_dir: Path) -> None:
        backup = index_dir.with_name(index_dir.name + ".backup")
        if backup.exists():
            if index_dir.exists():
                shutil.rmtree(backup)
            else:
                backup.replace(index_dir)

    def _write(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        (index_dir / "docs.json").write_text(
            json.dumps({k: asdict(v) for k, v in self.docs.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
        if self.vectors is not None:
            self.vectors.save(index_dir)
        else:
            # A BM25 rebuild must not accidentally reuse embeddings from an older corpus.
            for name in ("embeddings.npy", "vectors.json"):
                (index_dir / name).unlink(missing_ok=True)

    @classmethod
    def load(cls, index_dir: Path, *, load_vectors: bool = True) -> "Index":
        index_dir.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(index_dir) + ".lock"):
            cls._recover(index_dir)
            return cls._read(index_dir, load_vectors=load_vectors)

    @classmethod
    def _read(cls, index_dir: Path, *, load_vectors: bool) -> "Index":
        f = index_dir / "chunks.jsonl"
        if not f.exists():
            raise FileNotFoundError(f"No index at {index_dir}. Run `envsearch fetch` then `envsearch build`.")
        chunks = [Chunk(**json.loads(line)) for line in f.open(encoding="utf-8")]
        docs = {k: Document(**v) for k, v in json.loads((index_dir / "docs.json").read_text(encoding="utf-8")).items()}
        index = cls(chunks, docs)
        if load_vectors:
            index.vectors = DenseIndex.load(index_dir, chunks)
        return index

    # ------------------------------------------------------------------ search
    def _candidates(self, lang=None, doc_ids=None, jurisdiction=None, metadata=None) -> list[int]:
        return [i for i, c in enumerate(self.chunks)
                if (lang is None or c.lang == lang)
                and (doc_ids is None or c.doc_id in doc_ids)
                and (jurisdiction is None or self.docs[c.doc_id].jurisdiction.casefold() == jurisdiction.casefold())
                and all(self.docs[c.doc_id].metadata.get(key) == value for key, value in (metadata or {}).items())]

    def search_one(self, query: str, k: int = 20, lang: str | None = None, doc_ids: set[str] | None = None,
                   *, jurisdiction: str | None = None, metadata: dict[str, str] | None = None) -> list[Hit]:
        return self._keyword_search(query, k, self._candidates(lang, doc_ids, jurisdiction, metadata))

    def _keyword_search(self, query: str, k: int, candidates: list[int]) -> list[Hit]:
        if not self._bm25 or k <= 0 or not candidates:
            return []
        q = tokenize(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        order = sorted(candidates, key=lambda i: scores[i], reverse=True)
        hits: list[Hit] = []
        for i in order:
            if scores[i] <= 0:
                break
            c = self.chunks[i]
            hits.append(Hit(c, float(scores[i]), len(hits) + 1))
            if len(hits) >= k:
                break
        return hits

    def search(self, queries: list[str], k: int = 8, lang: str | None = None,
               doc_ids: set[str] | None = None, rrf_k: int = 60, per_doc_cap: int = 3, *,
               mode: str = "auto", jurisdiction: str | None = None,
               metadata: dict[str, str] | None = None, candidate_k: int | None = None,
               rerank: bool = False, reranker=None, trace: dict | None = None) -> list[Hit]:
        """Prefilter both branches, fuse ranks, then cap hits per document.

        auto uses hybrid if vectors exist, otherwise BM25. Explicit vector/hybrid
        requests never silently fall back. Metadata filters are ANDed exact matches.
        """
        started = time.perf_counter()
        mode = self.resolve_mode(mode)
        limit = max(k * 4, 20) if candidate_k is None else candidate_k
        if limit < max(k, 1):
            raise ValueError("candidate_k must be at least k and positive")
        queries = list(dict.fromkeys(q.strip() for q in queries if q and q.strip()))
        if trace is not None:
            trace.update(total_chunks=len(self.chunks), eligible_chunks=0, candidate_k=limit,
                         fused_candidates=0, reranked_candidates=0, returned=0, mode=mode, rerank=rerank)
        if k <= 0:
            return []
        if rrf_k < 0 or per_doc_cap < 1:
            raise ValueError("rrf_k must be nonnegative and per_doc_cap must be positive")
        candidates = self._candidates(lang, doc_ids, jurisdiction, metadata)
        if trace is not None:
            trace["eligible_chunks"] = len(candidates)
        if not candidates or not queries:
            return []
        fused: dict[str, float] = {}
        for q in queries:
            rankings = []
            if mode in {"bm25", "hybrid"}:
                rankings.append([h.chunk.id for h in self._keyword_search(q, limit, candidates)])
            if mode in {"vector", "hybrid"}:
                rankings.append([self.chunks[i].id for i, _ in self.vectors.search(q, candidates, limit)])
            for ranking in rankings:
                for rank, cid in enumerate(ranking, 1):
                    fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        if trace is not None:
            trace.update(fused_candidates=len(ranked), retrieval_seconds=round(time.perf_counter() - started, 4))
        if rerank and ranked:
            from .rerank import get_reranker
            rerank_started = time.perf_counter()
            pool = ranked[:limit]
            scores = (reranker or get_reranker()).score(queries[0], [self.by_id[cid].text for cid, _ in pool])
            if len(scores) != len(pool) or any(not math.isfinite(float(score)) for score in scores):
                raise ValueError("Reranker returned invalid scores")
            ranked = sorted(((cid, float(score)) for (cid, _), score in zip(pool, scores)),
                            key=lambda pair: pair[1], reverse=True)
            if trace is not None:
                trace.update(reranked_candidates=len(pool), rerank_seconds=round(time.perf_counter() - rerank_started, 4))
        out: list[Hit] = []
        per_doc: dict[str, int] = {}
        for cid, s in ranked:
            c = self.by_id[cid]
            if per_doc.get(c.doc_id, 0) >= per_doc_cap:
                continue
            per_doc[c.doc_id] = per_doc.get(c.doc_id, 0) + 1
            out.append(Hit(c, s, len(out) + 1))
            if len(out) >= k:
                break
        if trace is not None:
            trace.update(returned=len(out), total_seconds=round(time.perf_counter() - started, 4))
        return out

    def resolve_mode(self, mode: str) -> str:
        if mode not in {"auto", "bm25", "vector", "hybrid"}:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        if mode == "auto":
            return "hybrid" if self.vectors is not None else "bm25"
        if mode in {"vector", "hybrid"} and self.vectors is None:
            raise ValueError("No vector index. Run `envsearch build --vectors` first.")
        return mode

    def neighbors(self, chunk_id: str, span: int = 1) -> list[Chunk]:
        """The chunk plus adjacent chunks from the same document (for context expansion)."""
        i = next((n for n, c in enumerate(self.chunks) if c.id == chunk_id), None)
        if i is None:
            return []
        doc = self.chunks[i].doc_id
        return [c for c in self.chunks[max(0, i - span): i + span + 1] if c.doc_id == doc]

    # ------------------------------------------------------------------ helpers
    def cite_url(self, chunk: Chunk) -> str:
        doc = self.docs[chunk.doc_id]
        return f"{doc.url}#page={chunk.page}" if doc.url.lower().split("?")[0].endswith(".pdf") else doc.url
