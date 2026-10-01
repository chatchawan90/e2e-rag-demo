"""Local operational metrics. Never persist prompts, answers, PDF text, or API keys."""
import json
import sqlite3
import time
from pathlib import Path

FIELDS = {"provider", "model", "mode", "k", "candidate_k", "per_doc_cap", "rerank",
          "returned", "citations", "found", "doc_id", "chunks", "pages", "error_type", "eval_items"}
TRACE_FIELDS = {"total_chunks", "eligible_chunks", "candidate_k", "fused_candidates", "reranked_candidates",
                "returned", "mode", "rerank", "retrieval_seconds", "rerank_seconds", "total_seconds",
                "scope_decision", "scope_seconds", "scope_input_tokens", "scope_output_tokens"}


def _connect(home):
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(home / "monitoring.sqlite3", timeout=10)
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, created REAL, payload TEXT)")
    except sqlite3.Error:
        connection.close()
        raise
    return connection


def record_event(home, kind: str, status: str, seconds: float, **values):
    row = {key: values[key] for key in FIELDS if key in values}
    row.update(kind=kind, status=status, seconds=round(seconds, 3))
    row["usage"] = {key: values.get("usage", {}).get(key, 0) for key in ("input_tokens", "output_tokens")}
    row["trace"] = {key: val for key, val in values.get("trace", {}).items() if key in TRACE_FIELDS}
    connection = _connect(home)
    try:
        with connection:
            connection.execute("INSERT INTO events(created,payload) VALUES (?,?)", (time.time(), json.dumps(row)))
    finally:
        connection.close()


def recent_events(home, limit=500):
    connection = _connect(home)
    try:
        rows = connection.execute("SELECT id,created,payload FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": row[0], "created": row[1], **json.loads(row[2])} for row in rows]
    finally:
        connection.close()
