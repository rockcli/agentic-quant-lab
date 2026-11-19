"""
Text chunking strategy

Supports chunking by Markdown heading level and sliding window chunking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    chunk_id: str
    source_path: str
    title: str
    text: str
    parent_titles: str = ""  # Parent title chain, used for context resumption


class MarkdownChunker:
    """
    Markdown chunker
    - Chunks by heading level (# -> ## -> ###)
    - Each chunk preserves parent heading context
    - Chunk size approximately 512 tokens (estimated by Chinese characters)
    """

    def __init__(self, max_chars: int = 1500, overlap_chars: int = 200):
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk_file(self, file_path: Path) -> list[Chunk]:
        """Chunk a single markdown file."""
        text = file_path.read_text(encoding="utf-8")
        title = file_path.stem

        # Split by headers
        sections = self._split_by_headers(text)
        chunks: list[Chunk] = []

        for sec_idx, (headers, sec_text) in enumerate(sections):
            parent = " > ".join(headers[:-1]) if len(headers) > 1 else ""
            # If paragraph is too long, split further by length
            sub_chunks = self._chunk_text(sec_text)
            for sub_idx, sub_text in enumerate(sub_chunks):
                chunks.append(
                    Chunk(
                        chunk_id=f"{file_path.stem}-{sec_idx}-{sub_idx}",
                        source_path=str(file_path),
                        title=headers[-1] if headers else title,
                        text=sub_text,
                        parent_titles=parent,
                    )
                )

        return chunks

    def _split_by_headers(self, text: str) -> list[tuple[list[str], str]]:
        """Split text by Markdown headings."""
        pattern = re.compile(r"^(#{1,3}\s+.+)$", re.MULTILINE)
        parts = pattern.split(text)

        if len(parts) <= 1:
            return [([], text)]

        results: list[tuple[list[str], str]] = []
        current_headers: list[str] = []

        for i in range(1, len(parts), 2):
            header = parts[i].strip()
            level = len(header) - len(header.lstrip("#"))
            content = parts[i + 1] if i + 1 < len(parts) else ""

            # Update current heading level
            header_text = header.lstrip("#").strip()
            while len(current_headers) >= level:
                current_headers.pop()
            current_headers.append(header_text)

            results.append((list(current_headers), content.strip()))

        return results

    def _chunk_text(self, text: str) -> list[str]:
        """Split text by max length, preserving overlap."""
        if len(text) <= self.max_chars:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + self.max_chars
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.overlap_chars

        return chunks
