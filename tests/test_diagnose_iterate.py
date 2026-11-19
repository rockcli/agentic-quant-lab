"""
Diagnose Loop + Automatic Iterative + Experiment Tracking Test (Using Mock LLM)
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic_quant.ai.experiment.tracker import ExperimentTracker
from agentic_quant.ai.llm_client import LLMClient
from agentic_quant.ai.pipeline.diagnose import DiagnosisPipeline, DiagnosisReport
from agentic_quant.ai.pipeline.iterate import AutoIterationPipeline
from agentic_quant.backtest.core.engine import BacktestEngine
from agentic_quant.backtest.core.result import BacktestResult
from agentic_quant.backtest.data.structures import Bar
from agentic_quant.backtest.strategy.base import Signal, Strategy


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


class SimpleStrategy(Strategy):
    """Simple strategy: Buy on 10th bar, Sell on 20th bar"""

    def __init__(self):
        super().__init__()
        self.threshold = 10

    def on_bar(self, bar) -> Signal | None:
        if len(self.bars) == self.threshold:
            return Signal(timestamp=bar.timestamp, action="buy", size=1.0)
        if len(self.bars) == self.threshold + 10:
            return Signal(timestamp=bar.timestamp, action="sell", size=0.0)
        return None


class DiagnosisPipelineTest(unittest.TestCase):
    def test_build_snapshot_extracts_metrics(self):
        bars = _make_bars()
        engine = BacktestEngine()
        result = engine.run(bars, SimpleStrategy())

        snapshot = DiagnosisPipeline._build_snapshot(result, "Simple")
        self.assertEqual(snapshot["strategy"], "Simple")
        self.assertIn("metrics", snapshot)
        self.assertIn("sharpe_ratio", snapshot["metrics"])

    def test_diagnosis_with_mock_llm(self):
        mock_llm = MockLLM(
            response_text='{\n  "summary": "Strategy performance is mediocre",\n  "root_causes": [{"cause": "Insufficient samples", "evidence": "Only 10 trades"}],\n  "suggestions": [{"action": "Add parameters", "target": "threshold", "rationale": "Increase trading frequency"}],\n  "priority": ["P0: Increase trading frequency"],\n  "experiment_design": {"variables": ["threshold"], "metrics": ["sharpe"], "stop_condition": "sharpe>1"}\n}'
        )
        pipeline = DiagnosisPipeline(llm=mock_llm)
        bars = _make_bars()
        result = BacktestEngine().run(bars, SimpleStrategy())
        report = pipeline.run(result, strategy_name="Simple")

        self.assertEqual(report.summary, "Strategy performance is mediocre")
        self.assertEqual(len(report.root_causes), 1)
        self.assertEqual(report.root_causes[0]["cause"], "Insufficient samples")
        self.assertEqual(len(report.suggestions), 1)
        self.assertEqual(report.priority[0], "P0: Increase trading frequency")

    def test_diagnosis_fallback_on_json_error(self):
        mock_llm = MockLLM(response_text="This is not JSON")
        pipeline = DiagnosisPipeline(llm=mock_llm)
        bars = _make_bars()
        result = BacktestEngine().run(bars, SimpleStrategy())
        report = pipeline.run(result)

        self.assertIn("JSON parse failed", report.summary)

    def test_save_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = DiagnosisReport(
                summary="Test",
                root_causes=[{"cause": "c1", "evidence": "e1"}],
                suggestions=[],
                priority=["P0"],
                experiment_design={},
            )
            path = DiagnosisPipeline.save(report, Path(tmpdir))
            self.assertTrue(path.exists())
            import json

            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["summary"], "Test")


class ExperimentTrackerTest(unittest.TestCase):
    def test_log_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker("test_exp", base_dir=tmpdir)
            bars = _make_bars()
            result = BacktestEngine().run(bars, SimpleStrategy())
            tracker.log(
                iteration=1,
                strategy_name="Simple",
                strategy_code="class Simple(Strategy): pass",
                params={"threshold": 10},
                result=result,
            )
            self.assertEqual(len(tracker.records), 1)
            self.assertEqual(tracker.records[0].iteration, 1)

    def test_best_by_metric(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker("test_exp", base_dir=tmpdir)
            for i in range(3):
                result = BacktestResult(
                    equity_curve=None,  # type: ignore[arg-type]
                    fills=[],
                    signals=[],
                    total_return=float(i),
                    sharpe_ratio=float(i),
                    max_drawdown=-0.1 * i,
                    win_rate=0.5,
                    profit_factor=1.0,
                    num_trades=10,
                )
                tracker.log(
                    iteration=i + 1,
                    strategy_name="S",
                    strategy_code="",
                    params={"x": i},
                    result=result,
                )
            best = tracker.best_by("sharpe_ratio", maximize=True)
            self.assertIsNotNone(best)
            assert best is not None
            self.assertEqual(best.iteration, 3)

    def test_compare_and_lineage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker("test_exp", base_dir=tmpdir)
            for i in range(3):
                result = BacktestResult(
                    equity_curve=None,  # type: ignore[arg-type]
                    fills=[],
                    signals=[],
                    total_return=float(i),
                    sharpe_ratio=float(i),
                    max_drawdown=-0.1 * i,
                    win_rate=0.5,
                    profit_factor=1.0,
                    num_trades=10,
                )
                tracker.log(
                    iteration=i + 1,
                    strategy_name="S",
                    strategy_code="",
                    params={"x": i},
                    result=result,
                    parent_iteration=i if i > 0 else None,
                )
            comp = tracker.compare("sharpe_ratio")
            self.assertEqual(len(comp), 3)
            lineage = tracker.lineage(3)
            self.assertEqual(len(lineage), 3)


class AutoIterationTest(unittest.TestCase):
    def test_iteration_stops_at_max(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker("iter_test", base_dir=tmpdir)
            mock_llm = MockLLM(
                response_text='{\n  "summary": "Mediocre",\n  "root_causes": [],\n  "suggestions": [{"action": "Adjust", "target": "threshold", "rationale": "Test"}],\n  "priority": [],\n  "experiment_design": {}\n}'
            )
            pipeline = AutoIterationPipeline(tracker=tracker, llm=mock_llm)
            bars = _make_bars()

            def factory(params):
                s = SimpleStrategy()
                s.threshold = params.get("threshold", 10)
                return s

            results = pipeline.run(
                bars=bars,
                strategy_factory=factory,
                initial_params={"threshold": 10},
                max_iterations=2,
            )
            self.assertEqual(len(results), 2)
            self.assertEqual(len(tracker.records), 2)

    def test_should_stop_on_high_sharpe(self):
        result = BacktestResult(
            equity_curve=None,  # type: ignore[arg-type]
            fills=[],
            signals=[],
            total_return=0.5,
            sharpe_ratio=3.0,
            max_drawdown=-0.05,
            win_rate=0.8,
            profit_factor=2.0,
            num_trades=20,
        )
        diag = DiagnosisReport(summary="Very good", root_causes=[], suggestions=[], priority=[], experiment_design={})
        self.assertTrue(AutoIterationPipeline._should_stop(result, diag, 1, 5))

    def test_param_mutation_changes_numeric(self):
        params = {"threshold": 10, "name": "test"}
        diag = DiagnosisReport(
            summary="",
            root_causes=[],
            suggestions=[{"action": "Adjust", "target": "threshold", "rationale": "Test"}],
            priority=[],
            experiment_design={},
        )
        mutated = AutoIterationPipeline._mutate_params(params, diag)
        self.assertNotEqual(mutated["threshold"], 10)
        self.assertEqual(mutated["name"], "test")  # Non-numeric value unchanged

    def test_generate_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker("report_test", base_dir=tmpdir)
            for i in range(2):
                result = BacktestResult(
                    equity_curve=None,  # type: ignore[arg-type]
                    fills=[],
                    signals=[],
                    total_return=float(i),
                    sharpe_ratio=float(i),
                    max_drawdown=-0.1,
                    win_rate=0.5,
                    profit_factor=1.0,
                    num_trades=10,
                )
                tracker.log(
                    iteration=i + 1,
                    strategy_name="S",
                    strategy_code="",
                    params={"x": i},
                    result=result,
                )
            pipeline = AutoIterationPipeline(tracker=tracker)
            report = pipeline.generate_report()
            self.assertIn("ExperimentReport", report)
            self.assertIn("Iteration 1", report)


if __name__ == "__main__":
    unittest.main()
