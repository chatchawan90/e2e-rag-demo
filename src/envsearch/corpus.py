"""Manifest loading, downloading, page extraction and chunking.

Chunks never cross a page boundary, so every citation can point at an exact page.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

import httpx
import yaml

from .textnorm import is_thai, normalize


@dataclass
class Document:
    id: str
    title: str
    url: str
    lang: str = "en"
    jurisdiction: str = ""
    title_en: str | None = None
    path: str | None = None  # local file; overrides url for fetching (used by tests / private docs)
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.metadata, dict) or any(
            not isinstance(k, str) or not k or not isinstance(v, str) for k, v in self.metadata.items()
        ):
            raise ValueError(f"{self.id}: metadata must map nonempty string keys to string values; quote dates/numbers in YAML")


@dataclass
class Chunk:
    id: str
    doc_id: str
    page: int                # 1-based
    text: str                # normalised text shown to Claude
    lang: str
    tokens: list[str] = field(default_factory=list)


def load_manifest(path: Path) -> list[Document]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    docs = [Document(**d) for d in data["documents"]]
    for d in docs:  # local paths are relative to the manifest file
        if d.path and not Path(d.path).is_absolute():
            d.path = str((Path(path).parent / d.path).resolve())
    ids = [d.id for d in docs]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate doc ids in manifest: {sorted(dupes)}")
    return docs


# ----------------------------------------------------------------------------- fetch

def raw_path(doc: Document, raw_dir: Path) -> Path:
    if doc.path:
        return Path(doc.path)
    ext = Path(doc.url.split("?")[0]).suffix.lower() or ".html"
    return raw_dir / f"{doc.id}{ext if ext in {'.pdf', '.txt', '.html', '.htm'} else '.html'}"


def fetch(docs: list[Document], raw_dir: Path, force: bool = False) -> list[tuple[str, str]]:
    """Download every document. Returns (doc_id, status) pairs; never raises for one bad URL."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    results = []
    headers = {"User-Agent": "envsearch/0.1 (+research; public regulatory documents)"}
    with httpx.Client(follow_redirects=True, timeout=60, headers=headers) as client:
        for doc in docs:
            dest = raw_path(doc, raw_dir)
            if doc.path:
                results.append((doc.id, "local file"))
                continue
            if dest.exists() and not force:
                results.append((doc.id, "cached"))
                continue
            try:
                r = client.get(doc.url)
                r.raise_for_status()
                dest.write_bytes(r.content)
                results.append((doc.id, f"ok {len(r.content) // 1024} KB"))
            except Exception as e:  # noqa: BLE001 — report and continue
                results.append((doc.id, f"FAILED: {e}"))
    return results


# ----------------------------------------------------------------------------- extract

class _TextOnly(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "nav", "header", "footer"}:
            self._skip += 1
        if tag in {"p", "br", "li", "h1", "h2", "h3", "h4", "tr", "div"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "nav", "header", "footer"} and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def extract_pages(path: Path) -> list[str]:
    """Return one string per page (HTML and TXT count as a single page; TXT may use \\f)."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import pymupdf
        with pymupdf.open(path) as pdf:
            return [page.get_text("text", sort=True) for page in pdf]
    raw = path.read_text(encoding="utf-8", errors="replace")
    if suffix in {".html", ".htm"}:
        p = _TextOnly()
        p.feed(raw)
        return ["".join(p.parts)]
    return raw.split("\f")


# ----------------------------------------------------------------------------- chunk

_HEADER_NOISE = re.compile(r"^\s*(\d+|-\s*\d+\s*-|หน้า\s*\d+.*|เล่ม\s*\d+.*ราชกิจจานุเบกษา.*)\s*$")


def _clean_lines(page_text: str) -> list[str]:
    lines = [normalize(l).strip() for l in page_text.splitlines()]
    return [l for l in lines if l and not _HEADER_NOISE.match(l)]


def _paragraphs(lines: list[str]) -> list[str]:
    """Re-join hard-wrapped PDF lines into paragraphs.

    English: a line that doesn't end a sentence is continued by the next one.
    Thai: lines starting with a clause marker (ข้อ N, (N), N.N) start a new paragraph.
    """
    paras: list[str] = []
    starts_new = re.compile(r"^(ข้อ\s*\d+|\(\d+\)|\d+(\.\d+)+\s|§|•|[A-Z][A-Za-z ]{0,40}:|Question:|Answer:)")
    for line in lines:
        if paras and not starts_new.match(line) and not re.search(r"[.:;?!]$", paras[-1]):
            sep = "" if is_thai(paras[-1][-1:] + line[:1], 0.5) else " "
            paras[-1] = paras[-1] + sep + line
        else:
            paras.append(line)
    return paras


def chunk_document(doc: Document, pages: list[str], target_chars: int = 1100, overlap_paras: int = 1) -> list[Chunk]:
    chunks: list[Chunk] = []
    # Thai carries more meaning per character; smaller windows keep chunks focused.
    for page_no, page_text in enumerate(pages, start=1):
        paras = _paragraphs(_clean_lines(page_text))
        if not paras:
            continue
        limit = int(target_chars * 0.7) if is_thai("".join(paras[:5])) else target_chars
        buf: list[str] = []
        idx = 0

        def flush():
            nonlocal idx
            text = "\n".join(buf).strip()
            if len(text) >= 40:
                chunks.append(Chunk(
                    id=f"{doc.id}:p{page_no}:c{idx}", doc_id=doc.id, page=page_no, text=text,
                    lang="th" if is_thai(text) else "en",
                ))
                idx += 1

        pieces: list[str] = []
        for para in paras:  # split paragraphs that alone exceed the window
            while len(para) > limit:
                cut = max(para.rfind(". ", 0, limit) + 1, para.rfind(" ", 0, limit))
                cut = cut if cut > limit // 3 else limit
                pieces.append(para[:cut].strip())
                para = para[cut:].strip()
            if para:
                pieces.append(para)

        for para in pieces:
            if buf and sum(len(p) for p in buf) + len(para) > limit:
                flush()
                buf = buf[-overlap_paras:] if overlap_paras else []
            buf.append(para)
        flush()
    return chunks


def fingerprint(docs: list[Document], raw_dir: Path) -> str:
    h = hashlib.sha256()
    for d in docs:
        p = raw_path(d, raw_dir)
        h.update(d.id.encode())
        if p.exists():
            h.update(str(p.stat().st_size).encode())
    return h.hexdigest()[:12]
