"""
Local版书籍Retrieval器（轻量 BM25 风格）
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


@dataclass
class Chunk:
    chunk_id: str
    source_path: str
    title: str
    text: str


@dataclass
class SearchResult:
    chunk_id: str
    source_path: str
    title: str
    score: float
    snippet: str


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_PATTERN.findall(text)]


class BookRetriever:
    def __init__(self, docs_dir: str | Path):
        self.docs_dir = Path(docs_dir)
        self.chunks: list[Chunk] = []
        self.doc_freq: dict[str, int] = {}
        self.term_freqs: list[dict[str, int]] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_len: float = 0.0

    def build(self) -> None:
        self.chunks = list(self._iter_chunks())
        self.term_freqs = []
        self.doc_freq = {}
        self.doc_lengths = []

        for chunk in self.chunks:
            tokens = tokenize(chunk.text)
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.term_freqs.append(tf)
            self.doc_lengths.append(len(tokens))
            for token in tf:
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

        self.avg_doc_len = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 1.0

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not self.chunks:
            self.build()
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores: list[tuple[int, float]] = []
        for idx, tf in enumerate(self.term_freqs):
            score = self._score_chunk(tf, self.doc_lengths[idx], q_tokens)
            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results: list[SearchResult] = []
        for idx, score in scores[:top_k]:
            chunk = self.chunks[idx]
            snippet = chunk.text[:220].replace("\n", " ")
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    source_path=chunk.source_path,
                    title=chunk.title,
                    score=score,
                    snippet=snippet,
                )
            )
        return results

    def _score_chunk(self, tf: dict[str, int], doc_len: int, q_tokens: list[str]) -> float:
        k1 = 1.5
        b = 0.75
        n_docs = len(self.chunks)
        score = 0.0
        for token in q_tokens:
            if token not in tf:
                continue
            df = self.doc_freq.get(token, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            freq = tf[token]
            denom = freq + k1 * (1 - b + b * doc_len / self.avg_doc_len)
            score += idf * (freq * (k1 + 1)) / denom
        return score

    def _iter_chunks(self) -> Iterable[Chunk]:
        md_files = sorted(self.docs_dir.glob("*.md"))
        for file_path in md_files:
            text = file_path.read_text(encoding="utf-8")
            title = file_path.stem
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            buffer: list[str] = []
            current_len = 0
            chunk_idx = 0
            for p in paragraphs:
                p_len = len(p)
                if current_len + p_len > 1000 and buffer:
                    chunk_text = "\n\n".join(buffer)
                    yield Chunk(
                        chunk_id=f"{file_path.stem}-{chunk_idx}",
                        source_path=str(file_path),
                        title=title,
                        text=chunk_text,
                    )
                    chunk_idx += 1
                    buffer = []
                    current_len = 0
                buffer.append(p)
                current_len += p_len

            if buffer:
                yield Chunk(
                    chunk_id=f"{file_path.stem}-{chunk_idx}",
                    source_path=str(file_path),
                    title=title,
                    text="\n\n".join(buffer),
                )
