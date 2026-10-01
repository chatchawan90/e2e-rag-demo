"""Paths and model settings. Everything is overridable with environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    home: Path                 # data directory (raw downloads + index)
    manifest: Path             # corpus/manifest.yaml
    answer_model: str          # model that writes cited answers
    rewrite_model: str         # cheap model for bilingual query expansion
    top_k: int                 # passages handed to Claude per question

    @property
    def raw_dir(self) -> Path:
        return self.home / "raw"

    @property
    def index_dir(self) -> Path:
        return self.home / "index"


def get_client(required: bool = False):
    """Anthropic client, or None when no API key is set (retrieval-only mode)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        if required:
            raise SystemExit("ANTHROPIC_API_KEY is not set. Retrieval works without it; answers and eval of answers don't.")
        return None
    import anthropic
    return anthropic.Anthropic()


def get_settings() -> Settings:
    return Settings(
        home=Path(os.environ.get("ENVSEARCH_HOME", REPO_ROOT / "data")).expanduser(),
        manifest=Path(os.environ.get("ENVSEARCH_MANIFEST", REPO_ROOT / "corpus" / "manifest.yaml")).expanduser(),
        answer_model=os.environ.get("ENVSEARCH_ANSWER_MODEL", "claude-sonnet-5-5"),
        rewrite_model=os.environ.get("ENVSEARCH_REWRITE_MODEL", "claude-haiku-4-5"),
        top_k=int(os.environ.get("ENVSEARCH_TOP_K", "8")),
    )
