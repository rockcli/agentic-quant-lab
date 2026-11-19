"""
Vector storage module

In-memory implementation, with reserved upgrade path for ChromaDB/FAISS.
"""

from __future__ import annotations

import numpy as np


class InMemoryVectorStore:
    """
    In-memory vector storage (numpy implementation)
    Supports cosine similarity search.
    """

    def __init__(self):
        self.vectors: np.ndarray | None = None
        self.metadatas: list[dict] = []

    def add(self, vectors: np.ndarray, metadatas: list[dict]) -> None:
        """Add vectors."""
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.metadatas.extend(metadatas)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """Cosine similarity search."""
        if self.vectors is None or len(self.metadatas) == 0:
            return []

        # Cosine similarity = dot(a, b) / (|a|*|b|), if vectors are normalized use dot directly
        similarities = self.vectors @ query_vector.flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append(
                    {
                        "metadata": self.metadatas[idx],
                        "score": float(similarities[idx]),
                        "index": int(idx),
                    }
                )
        return results

    def count(self) -> int:
        return len(self.metadatas)
