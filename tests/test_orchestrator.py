"""
End-to-end orchestrator test (mock LLM, no real API / document index needed)
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic_quant.ai.llm_client import LLMClient
from agentic_quant.ai.pipeline.generate import GeneratedStrategy
from agentic_quant.ai.pipeline.orchestrator import ResearchOrchestrator
from agentic_quant.backtest.core.engine import BacktestEngine
from agentic_quant.backtest.data.structures import Bar
from agentic_quant.strategy.examples import SmaCrossStrategy


class MockLLM(LLMClient):
    """Mock LLM, does not depend on real API"""

    def __init__(self, response_text: str = ""):
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


def _make_bars() -> list[Bar]:
    import datetime

    base = datetime.datetime(2024, 1, 1)
    bars = []
    for i in range(30):
        price = 100.0 + i * 0.5
        bars.append(
            Bar(
                timestamp=base + datetime.timedelta(days=i),
                open=price,
                high=price + 1.0,
                low=price - 0.5,
                close=price + 0.3,
                volume=1000 + i * 10,
            )
        )
    return bars


class OrchestratorTest(unittest.TestCase):
    def test_quick_generate_returns_strategy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = Path(tmpdir) / "docs"
            docs.mkdir()
            (docs / "tech.md").write_text("# Technical\n\nMACD Golden Cross Buy。\n", encoding="utf-8")

            mock_llm = MockLLM(
                response_text='```python\nfrom quant.backtest.strategy.base import Strategy, Signal\nfrom quant.backtest.data.structures import Bar\n\nclass MyStrategy(Strategy):\n    def on_bar(self, bar: Bar) -> Signal | None:\n        return None\n```'
            )
            orch = ResearchOrchestrator(
                docs_dir=docs,
                index_dir=Path(tmpdir) / "index",
                llm=mock_llm,
            )
            result = orch.quick_generate("MACD Golden CrossStrategy")
            self.assertIsInstance(result, GeneratedStrategy)
            self.assertTrue(result.code_valid)
            self.assertIn("class MyStrategy", result.code)

    def test_quick_diagnose_returns_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = Path(tmpdir) / "docs"
            docs.mkdir()
            (docs / "risk.md").write_text("# Risk\n\nStop loss control.\n", encoding="utf-8")

            mock_llm = MockLLM(
                response_text='{\n  "summary": "mediocre performance",\n  "root_causes": [],\n  "suggestions": [],\n  "priority": [],\n  "experiment_design": {}\n}'
            )
            orch = ResearchOrchestrator(
                docs_dir=docs,
                index_dir=Path(tmpdir) / "index",
                llm=mock_llm,
            )
            bars = _make_bars()
            result = BacktestEngine().run(bars, SmaCrossStrategy(fast=5, slow=10))
            report = orch.quick_diagnose(result, strategy_code="class S(Strategy): pass")
            self.assertEqual(report.summary, "mediocre performance")

    def test_run_full_pipeline_no_iteration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = Path(tmpdir) / "docs"
            docs.mkdir()
            (docs / "tech.md").write_text("# Technical\n\nTrend Following。\n", encoding="utf-8")

            mock_llm = MockLLM(
                response_text='```python\nfrom quant.backtest.strategy.base import Strategy, Signal\nfrom quant.backtest.data.structures import Bar\n\nclass MyStrategy(Strategy):\n    def on_bar(self, bar: Bar) -> Signal | None:\n        return None\n```'
            )
            orch = ResearchOrchestrator(
                docs_dir=docs,
                index_dir=Path(tmpdir) / "index",
                experiment_base_dir=Path(tmpdir) / "experiments",
                llm=mock_llm,
            )
            bars = _make_bars()
            session = orch.run(
                description="Trend FollowingStrategy",
                bars=bars,
                session_name="test_session",
                max_iterations=0,
            )
            self.assertEqual(session.session_id, "test_session")
            self.assertIsNotNone(session.generated_strategy)
            self.assertTrue(session.generated_strategy.code_valid)
            self.assertIsNotNone(session.initial_backtest)
            self.assertEqual(len(session.iterations), 0)


if __name__ == "__main__":
    unittest.main()
