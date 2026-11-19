"""
Text embedding module

Reserves upgrade path for sentence-transformers / OpenAI embedding,
currently provides TF-IDF vectors as a lightweight fallback.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_PATTERN.findall(text)]


class Embedder:
    """Base embedder interface"""

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class TfidfEmbedder(Embedder):
    """
    TF-IDF vector embedding (no external dependencies)
    Suitable for short text / keyword matching scenarios.
    """

    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self._idf: dict[str, float] = {}
        self._vocab: list[str] = []

    def fit(self, texts: list[str]) -> None:
        """Based OnCorpusCalculate IDF。"""
        tokenized = [tokenize(t) for t in texts]
        df: Counter[str] = Counter()
        for tokens in tokenized:
            for token in set(tokens):
                df[token] += 1

        n = len(texts)
        self._idf = {
            token: math.log((1 + n) / (1 + freq)) + 1.0
            for token, freq in df.items()
        }
        # Take the most frequent max_features terms as vocabulary
        self._vocab = sorted(
            self._idf.keys(),
            key=lambda t: self._idf[t],
            reverse=True,
        )[: self.max_features]
        self._idf = {t: self._idf[t] for t in self._vocab}

    def embed(self, texts: list[str]) -> np.ndarray:
        """Convert text list to TF-IDF vector matrix."""
        if not self._vocab:
            raise RuntimeError("Please call fit() first to build vocabulary")

        tokenized = [tokenize(t) for t in texts]
        vectors = []
        for tokens in tokenized:
            tf = Counter(tokens)
            vec = np.zeros(len(self._vocab), dtype=np.float32)
            for idx, term in enumerate(self._vocab):
                if term in tf:
                    vec[idx] = tf[term] * self._idf[term]
            # L2 Normalization
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec)
        return np.vstack(vectors)


class SentenceTransformersEmbedder(Embedder):
    """
    sentence-transformers embedder (requires additional installation)
    pip install sentence-transformers
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError as exc:
            raise RuntimeError(
                "SentenceTransformersEmbedder requires sentence-transformers, "
                "please install: pip install sentence-transformers"
            ) from exc

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)  # type: ignore[no-any-return]
