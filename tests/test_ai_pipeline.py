"""
AI Agent Pipeline Test (Using mock LLM, no real API Key required)
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic_quant.ai.llm_client import LLMClient
from agentic_quant.ai.pipeline.generate import GeneratedStrategy, StrategyGenerationPipeline
from agentic_quant.ai.rag.indexer import build_index


class MockLLM(LLMClient):
    """Mock LLM, used for offline testing"""

    def __init__(self, response_text: str = ""):
        # Do not call parent class __init__, avoid API key check
        self.provider = "mock"
        self.model = "mock"
        self.response_text = response_text
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

    def chat(self, system_prompt: str, user_prompt: str, **kwargs) -> object:
        from agentic_quant.ai.llm_client import LLMResponse

        return LLMResponse(
            content=self.response_text,
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            model="mock",
        )


class LLMClientTest(unittest.TestCase):
    def test_mock_llm_chat(self):
        mock = MockLLM("hello")
        resp = mock.chat("sys", "user")
        self.assertEqual(resp.content, "hello")

    def test_cost_summary_tracks_tokens(self):
        mock = MockLLM("x")
        mock.chat("s", "u")
        summary = mock.cost_summary()
        self.assertEqual(summary["provider"], "mock")


class StrategyGenerationTest(unittest.TestCase):
    def _make_docs(self) -> Path:
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = Path(tmpdir) / "docs"
            docs.mkdir()
            (docs / "risk.md").write_text(
                "# Risk Management\n\nStop Loss and Position Size control are key to strategy survival.\n\n"
                "# Position Management\n\nPosition sizing method based on ATR.\n",
                encoding="utf-8",
            )
            (docs / "tech.md").write_text(
                "# Technical Indicators\n\nMACD Golden Cross and RSI recovery can be combined.\n",
                encoding="utf-8",
            )
            return docs

    def test_pipeline_validates_code(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = Path(tmpdir) / "docs"
            docs.mkdir()
            (docs / "tech.md").write_text("# Technical\n\nMACD Golden Cross Buy.\n", encoding="utf-8")

            retriever = build_index(docs, Path(tmpdir) / "index")
            mock_llm = MockLLM(
                response_text='```python\nfrom quant.backtest.strategy.base import Strategy, Signal\nfrom quant.backtest.data.structures import Bar\n\nclass MyStrategy(Strategy):\n    def on_bar(self, bar: Bar) -> Signal | None:\n        return None\n```'
            )
            pipeline = StrategyGenerationPipeline(retriever, llm=mock_llm)
            result = pipeline.run("Write a MACD Golden Cross Strategy")

            self.assertTrue(result.code_valid)
            self.assertIn("class MyStrategy", result.code)
            self.assertGreater(len(result.references), 0)

    def test_pipeline_detects_invalid_code(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = Path(tmpdir) / "docs"
            docs.mkdir()
            (docs / "tech.md").write_text("# Technical\n\ncontent。\n", encoding="utf-8")

            retriever = build_index(docs, Path(tmpdir) / "index")
            mock_llm = MockLLM(response_text="This is not Code")
            pipeline = StrategyGenerationPipeline(retriever, llm=mock_llm)
            result = pipeline.run("Write a Strategy")

            self.assertFalse(result.code_valid)
            self.assertIn("No class inheriting Strategy found", result.error)

    def test_extract_code_from_markdown(self):
        text = "Some explanation\n```python\nclass A: pass\n```\nEnd"
        code = StrategyGenerationPipeline._extract_code(text)
        self.assertEqual(code, "class A: pass")

    def test_save_generated_strategy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = GeneratedStrategy(
                description="test strategy",
                code="class S(Strategy): pass",
                code_valid=True,
                references=[],
            )
            pipeline = StrategyGenerationPipeline.__new__(StrategyGenerationPipeline)
            path = pipeline.save(result, Path(tmpdir))
            self.assertTrue(path.exists())
            self.assertTrue(path.with_suffix(".meta.json").exists())


class HybridRetrieverTest(unittest.TestCase):
    def test_search_returns_ranked_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = Path(tmpdir) / "docs"
            docs.mkdir()
            (docs / "book1.md").write_text(
                "# Risk Management\n\nStop Loss and Position Size control are key to strategy survival.\n",
                encoding="utf-8",
            )
            (docs / "book2.md").write_text(
                "# Technical Indicators\n\nMACD Golden Cross and RSI recovery can be combined.\n",
                encoding="utf-8",
            )

            retriever = build_index(docs, Path(tmpdir) / "index")
            results = retriever.search("MACD RSI", top_k=2)
            self.assertGreaterEqual(len(results), 1)
            # book2 should be prioritized in recall
            self.assertIn("book2", results[0]["chunk"].source_path)

    def test_filter_books(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = Path(tmpdir) / "docs"
            docs.mkdir()
            (docs / "a.md").write_text("# A\n\ncontentA。\n", encoding="utf-8")
            (docs / "b.md").write_text("# B\n\ncontentB。\n", encoding="utf-8")

            retriever = build_index(docs, Path(tmpdir) / "index")
            results = retriever.search("content", filter_books=["a.md"])
            for r in results:
                self.assertIn("a.md", r["chunk"].source_path)


if __name__ == "__main__":
    unittest.main()
