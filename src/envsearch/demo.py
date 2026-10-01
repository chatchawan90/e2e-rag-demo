"""Create an isolated demo snapshot containing only public manifest documents."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from urllib.parse import urlparse

from .config import REPO_ROOT
from .corpus import load_manifest, raw_path
from .index import Index
from .vectors import DenseIndex


def prepare_demo(source_home: Path, destination: Path, *, manifest: Path = REPO_ROOT / 'corpus/manifest.yaml'):
    source_home, destination = source_home.resolve(), destination.resolve()
    if destination.exists():
        raise ValueError('Choose a new demo folder; existing folders are never overwritten.')
    source = Index.load(source_home / 'index')
    # Only the public manifest is eligible, never the upload registry, gold additions or reports.
    docs = {d.id: d for d in load_manifest(manifest)
            if not d.path and urlparse(d.url).scheme in {'http', 'https'}
            and d.id in source.docs and source.docs[d.id].url == d.url and not source.docs[d.id].path}
    positions = [i for i, c in enumerate(source.chunks) if c.doc_id in docs]
    chunks = [source.chunks[i] for i in positions]
    if not chunks:
        raise ValueError('No indexed public manifest documents are available for the demo.')
    exported = Index(chunks, docs)
    if source.vectors is not None:
        exported.vectors = DenseIndex(chunks, source.vectors.embeddings[positions], source.vectors.embedder)
    destination.mkdir(parents=True)
    try:
        for doc in docs.values():
            original = raw_path(doc, source_home / 'raw')
            if original.is_file():
                target = raw_path(doc, destination / 'raw')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, target)
        exported.save(destination / 'index')
    except Exception:
        shutil.rmtree(destination)
        raise
    return exported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=REPO_ROOT / 'data')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    index = prepare_demo(args.source, args.output)
    print(f'Prepared {len(index.docs)} public documents / {len(index.chunks)} passages at {args.output.resolve()}')


if __name__ == '__main__':
    main()
