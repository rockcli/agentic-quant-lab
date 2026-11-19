"""
Index build/update utilities
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agentic_quant.ai.rag.retriever import HybridRetriever


def build_index(docs_dir: str | Path, index_dir: str | Path, force: bool = False) -> HybridRetriever:
    """
    Build or incrementally update RAG index.

    Args:
        docs_dir: Markdown file directory
        index_dir: Index persistence directory
        force: Force rebuild
    """
    docs_dir = Path(docs_dir)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Calculate file fingerprint
    md_files = sorted(docs_dir.glob("*.md"))
    fingerprint = hashlib.sha256(
        "\n".join(str(f.stat().st_mtime) for f in md_files).encode()
    ).hexdigest()

    fingerprint_path = index_dir / "fingerprint.txt"
    if not force and fingerprint_path.exists():
        old = fingerprint_path.read_text().strip()
        if old == fingerprint:
            # File unchanged, load directly (simplified: rebuild here, can expand to serialize later)
            pass

    retriever = HybridRetriever(docs_dir)
    retriever.build()

    fingerprint_path.write_text(fingerprint, encoding="utf-8")
    (index_dir / "stats.json").write_text(
        json.dumps(
            {"chunks": retriever.count(), "files": len(md_files)},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return retriever
