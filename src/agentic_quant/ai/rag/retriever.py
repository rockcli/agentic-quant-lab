"""
Hybrid retriever: BM25 + Vector retrieval

Supports filter conditions (by book, chapter), can fall back to pure BM25.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from agentic_quant.ai.rag.chunker import Chunk, MarkdownChunker
from agentic_quant.ai.rag.embedder import TfidfEmbedder
from agentic_quant.ai.rag.vector_store import InMemoryVectorStore


def _tokenize(text: str) -> list[str]:
    import re

    token_pattern = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
    return [t.lower() for t in token_pattern.findall(text)]


class HybridRetriever:
    """
    Hybrid retriever: BM25 keyword + TF-IDF vector semantics
    """

    def __init__(
        self,
        docs_dir: str | Path,
        use_vector: bool = True,
        top_k: int = 5,
    ):
        self.docs_dir = Path(docs_dir)
        self.use_vector = use_vector
        self.top_k = top_k
        self.chunks: list[Chunk] = []
        self._bm25_ready = False
        self._vector_ready = False

        # BM25 related
        self.doc_freq: dict[str, int] = {}
        self.term_freqs: list[dict[str, int]] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_len: float = 1.0

        # Vector related
        self.embedder = TfidfEmbedder(max_features=2000)
        self.vector_store = InMemoryVectorStore()

    def build(self) -> None:
        """BuildIndex（BM25 + Vector）。"""
        self.chunks = list(self._iter_chunks())
        if not self.chunks:
            return

        self._build_bm25()
        if self.use_vector:
            self._build_vector()

    def _iter_chunks(self) -> Iterable[Chunk]:
        chunker = MarkdownChunker()
        for file_path in sorted(self.docs_dir.glob("*.md")):
            yield from chunker.chunk_file(file_path)

    def _build_bm25(self) -> None:
        self.term_freqs = []
        self.doc_freq = {}
        self.doc_lengths = []

        for chunk in self.chunks:
            tokens = _tokenize(chunk.text)
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.term_freqs.append(tf)
            self.doc_lengths.append(len(tokens))
            for token in tf:
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

        self.avg_doc_len = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 1.0
        self._bm25_ready = True

    def _build_vector(self) -> None:
        texts = [c.text for c in self.chunks]
        self.embedder.fit(texts)
        vectors = self.embedder.embed(texts)
        metadatas = [{"index": i, "chunk_id": c.chunk_id} for i, c in enumerate(self.chunks)]
        self.vector_store.add(vectors, metadatas)
        self._vector_ready = True

    def search(self, query: str, top_k: int | None = None, filter_books: list[str] | None = None) -> list[dict]:
        """
        BlendingRetrieval。
        Returns:
            [{"chunk": Chunk, "bm25_score": float, "vector_score": float, "hybrid_score": float}, ...]
        """
        k = top_k or self.top_k

        # Determine candidate range
        candidates = list(range(len(self.chunks)))
        if filter_books:
            candidates = [
                i for i in candidates
                if any(b in self.chunks[i].source_path for b in filter_books)
            ]

        # BM25 scoring
        bm25_scores = {i: self._bm25_score(i, query) for i in candidates}

        # Vector scoring
        vector_scores: dict[int, float] = {}
        if self._vector_ready and self.use_vector:
            q_vec = self.embedder.embed([query])[0]
            vec_results = self.vector_store.search(q_vec, top_k=max(k * 2, 10))
            for r in vec_results:
                idx = r["metadata"]["index"]
                if idx in candidates:
                    vector_scores[idx] = r["score"]

        # Hybrid sorting (BM25 primary, vector auxiliary)
        results = []
        for idx in candidates:
            b_score = bm25_scores.get(idx, 0.0)
            v_score = vector_scores.get(idx, 0.0)
            # Simple weighting: BM25 weight 0.7, vector weight 0.3
            hybrid = b_score * 0.7 + v_score * 0.3 * 10  # Vector scores are usually smaller, amplify
            results.append(
                {
                    "chunk": self.chunks[idx],
                    "bm25_score": b_score,
                    "vector_score": v_score,
                    "hybrid_score": hybrid,
                }
            )

        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:k]

    def _bm25_score(self, idx: int, query: str) -> float:
        if not self._bm25_ready:
            return 0.0
        k1 = 1.5
        b = 0.75
        n_docs = len(self.chunks)
        tf = self.term_freqs[idx]
        doc_len = self.doc_lengths[idx]
        score = 0.0
        for token in _tokenize(query):
            if token not in tf:
                continue
            df = self.doc_freq.get(token, 0)
            idf = __import__("math").log(1 + (n_docs - df + 0.5) / (df + 0.5))
            freq = tf[token]
            denom = freq + k1 * (1 - b + b * doc_len / self.avg_doc_len)
            score += idf * (freq * (k1 + 1)) / denom
        return score

    def count(self) -> int:
        return len(self.chunks)
