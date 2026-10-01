"""User-authored gold questions, stored separately from the shipped seed dataset."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import uuid

from filelock import FileLock

from . import evaluate as ev
from .config import REPO_ROOT
from .textnorm import normalize

SEED_PATH = REPO_ROOT / "evals/questions.yaml"


def local_path(home):
    return Path(home) / "gold" / "questions.json"


def _read_local(path):
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("items"), list):
            raise ValueError
        items = [ev.Item(**row) for row in data["items"]]
        for item in items:
            if not isinstance(item.id, str) or not isinstance(item.question, str):
                raise ValueError
        return items
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("The saved local gold dataset is invalid. Repair it before adding more questions.") from exc


def load_dataset(home, *, seed_path=SEED_PATH):
    # Atomic replacement makes a concurrent reader see one complete version.
    items = [*ev.load_items(seed_path), *_read_local(local_path(home))]
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Gold question IDs must be unique across seed and local datasets.")
    return items


def validate_item(item, index):
    if not item.id.strip() or not item.question.strip():
        raise ValueError("Enter a question.")
    if len(item.question) > 6000:
        raise ValueError("Keep the question to 6,000 characters or fewer.")
    if item.expected_scope not in {"in_scope", "out_of_scope", "clarify"}:
        raise ValueError("Choose a valid expected scope.")
    if item.answerable:
        if item.expected_scope != "in_scope":
            raise ValueError("Only an in-scope question can expect a document answer.")
        if not item.gold_docs:
            raise ValueError("Select at least one expected source document.")
        if not item.must_include or any(not group or any(not word.strip() for word in group) for group in item.must_include):
            raise ValueError("Enter at least one nonempty answer keyword group.")
    elif item.gold_docs or item.gold_pages or item.must_include:
        raise ValueError("Refusal and clarification cases should not contain answer/source labels.")
    indexed_docs = {c.doc_id for c in index.chunks}
    if any(did not in indexed_docs for did in item.gold_docs):
        raise ValueError("Expected source documents must have indexed passages.")
    if item.gold_pages:
        if len(item.gold_docs) != 1:
            raise ValueError("Page labels currently support one source document. Leave pages blank for multiple sources.")
        pages = {c.page for c in index.chunks if c.doc_id == item.gold_docs[0]}
        if any(type(page) is not int or page <= 0 or page not in pages for page in item.gold_pages):
            raise ValueError("Each expected page must have indexed passages in the selected document.")


def save_item(item, home, index, *, seed_path=SEED_PATH):
    """Validate and append atomically under a lock; never overwrite an existing question."""
    validate_item(item, index)
    path = local_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock", timeout=10):
        existing = load_dataset(home, seed_path=seed_path)
        canonical = lambda q: " ".join(normalize(q).casefold().split())
        if any(i.id == item.id or canonical(i.question) == canonical(item.question) for i in existing):
            raise ValueError("This question or ID is already in the gold dataset.")
        local = _read_local(path)
        data = json.dumps({"items": [asdict(i) for i in [*local, item]]}, ensure_ascii=False, indent=2)
        stage = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with stage.open("w", encoding="utf-8") as output:
                output.write(data + "\n")
                output.flush()
                os.fsync(output.fileno())
            stage.replace(path)
        finally:
            stage.unlink(missing_ok=True)
    return item
