import tempfile
import unittest
from pathlib import Path

from agentic_quant.ai.rag.book_retriever import BookRetriever, tokenize


class BookRetrieverTest(unittest.TestCase):
    def test_tokenize_mixed_language(self):
        tokens = tokenize("MACD 金叉 RSI 30-50")
        self.assertIn("macd", tokens)
        self.assertIn("金", tokens)
        self.assertIn("叉", tokens)
        self.assertIn("rsi", tokens)

    def test_search_returns_ranked_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir)
            (docs_dir / "book1.md").write_text(
                "Risk Management\n\nStop loss and position size control are key to strategy survival.",
                encoding="utf-8",
            )
            (docs_dir / "book2.md").write_text(
                "Technical Indicators\n\nMACD golden cross and RSI recovery can be combined.",
                encoding="utf-8",
            )
            retriever = BookRetriever(docs_dir)
            results = retriever.search("MACD RSI", top_k=2)
            self.assertGreaterEqual(len(results), 1)
            self.assertIn("book2", results[0].source_path)


if __name__ == "__main__":
    unittest.main()
