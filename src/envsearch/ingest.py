"""Ingest local PDFs without discarding existing chunks, vectors, or document metadata."""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pymupdf
from filelock import FileLock

from .config import Settings
from .corpus import Document, chunk_document, load_manifest
from .index import Index
from .textnorm import is_thai, tokenize
from .vectors import DenseIndex, E5Embedder

MAX_PDF_BYTES = 50 * 1024 * 1024


@dataclass
class IngestResult:
    doc_id: str
    title: str
    pages: int
    chunks: int
    duplicate: bool = False


def uploaded_documents(settings: Settings) -> list[Document]:
    return [Document(**json.loads(path.read_text(encoding="utf-8")))
            for path in sorted((settings.home / "uploads").glob("*/document.json"))]


def all_documents(settings: Settings) -> list[Document]:
    docs = load_manifest(settings.manifest) if settings.manifest.exists() else []
    docs.extend(uploaded_documents(settings))
    if len({doc.id for doc in docs}) != len(docs):
        raise ValueError("Duplicate document IDs across the manifest and uploaded PDFs")
    return docs


def ingest_pdf(data: bytes, filename: str, settings: Settings, *, title: str = "",
               jurisdiction: str = "", metadata: dict[str, str] | None = None,
               with_vectors: bool = True, embedder=None, progress=lambda _: None) -> IngestResult:
    if not data or len(data) > MAX_PDF_BYTES:
        raise ValueError("Choose a nonempty PDF smaller than 50 MB.")
    if not data.lstrip().startswith(b"%PDF-"):
        raise ValueError("This file is not a PDF. Export it as PDF and try again.")
    settings.home.mkdir(parents=True, exist_ok=True)
    # Serialize the whole read/append/commit, including independent CLI uploads.
    with FileLock(str(settings.home / ".ingest.lock")):
        try:
            index = Index.load(settings.index_dir)
        except FileNotFoundError:
            if settings.index_dir.exists():
                raise  # Do not replace an incomplete/corrupt existing index with an empty one.
            index = Index([], {})
        doc_id = "upload-" + hashlib.sha256(data).hexdigest()[:24]
        if doc_id in index.docs:
            old = [c for c in index.chunks if c.doc_id == doc_id]
            return IngestResult(doc_id, index.docs[doc_id].title, len({c.page for c in old}), len(old), True)

        progress("Extracting page text")
        try:
            with pymupdf.open(stream=data, filetype="pdf") as pdf:
                if pdf.needs_pass:
                    raise ValueError("This PDF is password protected. Upload an unlocked copy.")
                if len(pdf) > 1000:
                    raise ValueError("Split PDFs longer than 1,000 pages before uploading.")
                pages = [page.get_text("text", sort=True) for page in pdf]
        except pymupdf.FileDataError as exc:
            raise ValueError("The PDF could not be read. Export a new copy and try again.") from exc

        destination = settings.home / "uploads" / doc_id
        doc = Document(id=doc_id, title=title.strip() or Path(filename).stem,
                       url=f"local://{doc_id}.pdf", path=str((destination / "source.pdf").resolve()),
                       lang="th" if is_thai("".join(pages)) else "en",
                       jurisdiction=jurisdiction.strip().upper(), metadata=metadata or {})
        progress("Splitting pages into searchable passages")
        chunks = chunk_document(doc, pages)
        if not chunks:
            raise ValueError("No usable text was found. This PDF may be scanned; run OCR before uploading.")
        for chunk in chunks:
            chunk.tokens = tokenize(chunk.text)
        combined = Index([*index.chunks, *chunks], {**index.docs, doc.id: doc})
        if index.vectors is not None:
            progress("Embedding new passages")
            encoder = embedder or index.vectors.embedder
            if encoder.model_name != index.vectors.embedder.model_name:
                raise ValueError("The embedding model must match the existing index.")
            new_vectors = encoder.embed_documents([c.text for c in chunks])
            combined.vectors = DenseIndex(combined.chunks,
                                          np.vstack([index.vectors.embeddings, new_vectors]), encoder)
        elif with_vectors:
            progress("Building multilingual embeddings")
            combined.add_vectors(embedder or E5Embedder())

        progress("Saving the PDF and updating the index")
        # Only persist the upload after extraction and embedding have succeeded.
        destination.mkdir(parents=True, exist_ok=False)
        try:
            (destination / "source.pdf").write_bytes(data)
            (destination / "document.json").write_text(json.dumps(asdict(doc), ensure_ascii=False), encoding="utf-8")
            combined.save(settings.index_dir)
        except Exception:
            shutil.rmtree(destination)
            raise
        return IngestResult(doc_id, doc.title, len(pages), len(chunks))
