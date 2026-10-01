"""envsearch command line.

  envsearch fetch                 download everything in corpus/manifest.yaml
  envsearch build                 extract, chunk and index
  envsearch search "query"        retrieval only (no API key needed)
  envsearch ask "question"        cited answer from Claude
  envsearch eval verify           check gold labels against the corpus
  envsearch eval run [--retrieval-only] [--no-expand]
  envsearch eval draft --doc th-moi-effluent-2560 -n 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .config import REPO_ROOT, get_client, get_settings


def _index(mode="auto"):
    from .index import Index
    return Index.load(get_settings().index_dir, load_vectors=mode != "bm25")


def _metadata_pair(value):
    key, sep, val = value.partition("=")
    if not sep or not key.strip():
        raise argparse.ArgumentTypeError("metadata filters must use KEY=VALUE")
    return key.strip(), val


def _retrieval_options(a):
    return dict(mode=a.mode, lang=a.lang, doc_ids=set(a.doc) if a.doc else None,
                jurisdiction=a.jurisdiction, metadata=dict(a.filter or []),
                candidate_k=a.candidates, per_doc_cap=a.per_doc_cap, rerank=a.rerank)


def _retrieval_args(parser, *, add_doc=True):
    parser.add_argument("--mode", choices=["auto", "bm25", "vector", "hybrid"], default="auto",
                        help="auto: hybrid when vectors exist, otherwise BM25")
    parser.add_argument("--lang", choices=["en", "th"], help="passage language (not jurisdiction)")
    parser.add_argument("--jurisdiction", help="jurisdiction, e.g. US or TH (case insensitive)")
    if add_doc:
        parser.add_argument("--doc", action="append", help="allowed document ID (repeatable)")
    parser.add_argument("--filter", action="append", type=_metadata_pair, metavar="KEY=VALUE",
                        help="exact document metadata match; repeated filters are ANDed")
    parser.add_argument("--candidates", type=int, help="candidates per branch/query and maximum reranker pool (default max(4*k,20))")
    parser.add_argument("--per-doc-cap", type=int, default=3, help="maximum returned passages per document")
    parser.add_argument("--rerank", action="store_true", help="rerank candidates with a local multilingual cross-encoder")


def cmd_fetch(a):
    from .corpus import fetch, load_manifest
    s = get_settings()
    for doc_id, status in fetch(load_manifest(s.manifest), s.raw_dir, force=a.force):
        print(f"{doc_id:28s} {status}")


def cmd_build(a):
    from .ingest import all_documents
    from .index import Index
    from filelock import FileLock
    s = get_settings()
    s.home.mkdir(parents=True, exist_ok=True)
    with FileLock(str(s.home / ".ingest.lock")):
        idx = Index.build(all_documents(s), s.raw_dir)
        if a.vectors:
            print("embedding chunks with multilingual-e5-small (first run downloads the model)...")
            idx.add_vectors()
        idx.save(s.index_dir)
    print(f"indexed {len(idx.chunks)} chunks from {len({c.doc_id for c in idx.chunks})} documents -> {s.index_dir}")


def cmd_ingest(a):
    from .ingest import ingest_pdf, MAX_PDF_BYTES
    path = Path(a.pdf)
    if path.stat().st_size > MAX_PDF_BYTES:
        raise ValueError("Choose a PDF smaller than 50 MB.")
    result = ingest_pdf(path.read_bytes(), path.name, get_settings(), title=a.title or "",
                        jurisdiction=a.jurisdiction or "", metadata=dict(a.filter or []),
                        with_vectors=not a.no_vectors, progress=print)
    action = "already indexed" if result.duplicate else "indexed"
    print(f"{action}: {result.title} ({result.chunks} passages), id={result.doc_id}")


def cmd_serve(a):
    import importlib.util
    import subprocess
    if importlib.util.find_spec("streamlit") is None:
        raise RuntimeError('Install the web interface with: pip install -e ".[web,vectors]"')
    app = Path(__file__).with_name("web_app.py")
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app),
                     "--server.address", "127.0.0.1", "--server.port", str(a.port),
                     "--server.headless", "true", "--server.maxUploadSize", "50",
                     "--browser.gatherUsageStats", "false",
                     "--theme.base", "light", "--theme.primaryColor", "#2459A6",
                     "--theme.backgroundColor", "#FFFFFF", "--theme.secondaryBackgroundColor", "#F0F4F9",
                     "--theme.textColor", "#203047"]))


def cmd_search(a):
    from .rewrite import expand_query
    s = get_settings()
    idx = _index(a.mode)
    idx.resolve_mode(a.mode)
    queries = expand_query(a.query, get_client() if a.expand else None, s.rewrite_model)
    if len(queries) > 1:
        print("queries:", " | ".join(queries), "\n")
    for h in idx.search(queries, k=a.k, **_retrieval_options(a)):
        d = idx.docs[h.chunk.doc_id]
        snippet = h.chunk.text.replace("\n", " ")
        print(f"[{h.rank}] {d.id} p.{h.chunk.page}  ({h.score:.4f})\n    {snippet[:300]}\n")


def cmd_ask(a):
    from .answer import answer
    s = get_settings()
    ans = answer(a.question, _index(a.mode), get_client(required=True), model=a.model or s.answer_model,
                 rewrite_model=s.rewrite_model, k=a.k, expand=not a.no_expand, **_retrieval_options(a))
    print(ans.render())
    if a.debug:
        print("\nqueries:", ans.queries)
        print("retrieved:", [f"{h.chunk.doc_id}:p{h.chunk.page}" for h in ans.hits])
        print("usage:", ans.usage)


def cmd_eval(a):
    from . import evaluate as ev
    s = get_settings()
    items = ev.load_items(Path(a.questions))
    if a.only:
        items = [i for i in items if any(t in i.tags or t == i.id for t in a.only.split(","))]
    idx = _index(a.mode if a.action == "run" else "bm25")

    if a.action == "verify":
        problems = ev.verify(items, idx)
        for p in problems:
            print(f"  {p['id']}: {p['problem']}")
        print(f"{len(items) - len(problems)}/{len(items)} items verified against the corpus")
        sys.exit(1 if problems else 0)

    if a.action == "draft":
        drafts = ev.draft(idx, get_client(required=True), model=s.answer_model,
                          doc_ids=a.doc or None, n=a.n)
        out = Path(a.out)
        out.write_text(yaml.safe_dump({"items": drafts}, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"wrote {len(drafts)} draft items to {out} — review them before merging into questions.yaml")
        return

    client = get_client(required=not a.retrieval_only)
    result = ev.run(items, idx, client, model=a.model or s.answer_model, rewrite_model=s.rewrite_model,
                    k=a.k, retrieval_only=a.retrieval_only, expand=not a.no_expand, **_retrieval_options(a))
    path = ev.write_report(result, Path(a.report_dir))
    print()
    print(path.read_text(encoding="utf-8"))


def main(argv=None):
    s = get_settings()
    p = argparse.ArgumentParser(prog="envsearch")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch"); f.add_argument("--force", action="store_true"); f.set_defaults(fn=cmd_fetch)
    b = sub.add_parser("build"); b.set_defaults(fn=cmd_build)
    b.add_argument("--vectors", action="store_true", help="also build local multilingual embeddings (requires .[vectors])")

    ing = sub.add_parser("ingest", help="add a local PDF to the existing index")
    ing.add_argument("pdf"); ing.add_argument("--title"); ing.add_argument("--jurisdiction")
    ing.add_argument("--filter", action="append", type=_metadata_pair, metavar="KEY=VALUE",
                     help="metadata to attach to this PDF")
    ing.add_argument("--no-vectors", action="store_true", help="keep a BM25-only index if it has no vectors yet")
    ing.set_defaults(fn=cmd_ingest)
    serve = sub.add_parser("serve", help="open the local chat and PDF upload interface")
    serve.add_argument("--port", type=int, default=8501); serve.set_defaults(fn=cmd_serve)

    se = sub.add_parser("search"); se.add_argument("query"); se.add_argument("-k", type=int, default=8)
    _retrieval_args(se)
    se.add_argument("--expand", action="store_true",
                                                                     help="bilingual query expansion (needs API key)")
    se.set_defaults(fn=cmd_search)

    ak = sub.add_parser("ask"); ak.add_argument("question"); ak.add_argument("-k", type=int, default=s.top_k)
    ak.add_argument("--model"); ak.add_argument("--no-expand", action="store_true"); ak.add_argument("--debug", action="store_true")
    _retrieval_args(ak)
    ak.set_defaults(fn=cmd_ask)

    e = sub.add_parser("eval"); e.add_argument("action", choices=["run", "verify", "draft"])
    e.add_argument("--questions", default=str(REPO_ROOT / "evals" / "questions.yaml"))
    e.add_argument("--report-dir", default=str(REPO_ROOT / "evals" / "reports"))
    e.add_argument("-k", type=int, default=s.top_k); e.add_argument("--model")
    e.add_argument("--retrieval-only", action="store_true"); e.add_argument("--no-expand", action="store_true")
    e.add_argument("--only", help="comma-separated tags or ids")
    e.add_argument("--doc", action="append", help="run/draft: restrict to doc id (repeatable)")
    _retrieval_args(e, add_doc=False)
    e.add_argument("-n", type=int, default=5); e.add_argument("--out", default=str(REPO_ROOT / "evals" / "drafts.yaml"))
    e.set_defaults(fn=cmd_eval)

    a = p.parse_args(argv)
    if hasattr(a, "filter") and len(dict(a.filter or [])) != len(a.filter or []):
        p.error("each --filter metadata key may appear only once")
    try:
        a.fn(a)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        p.exit(1, f"envsearch: {exc}\n")


if __name__ == "__main__":
    main()
