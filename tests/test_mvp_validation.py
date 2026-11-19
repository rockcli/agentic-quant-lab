import tempfile
import unittest
from pathlib import Path

from agentic_quant.ai.mvp_validation import (
    BacktestSnapshot,
    build_diagnosis_input,
    format_snapshot,
)
from agentic_quant.ai.rag.book_retriever import BookRetriever


class MvpValidationTest(unittest.TestCase):
    def test_format_snapshot(self):
        snapshot = BacktestSnapshot(
            strategy_name="Demo",
            total_return=0.1,
            max_drawdown=-0.05,
            sharpe_ratio=1.2,
            trade_count=10,
            bad_cases=[],
        )
        text = format_snapshot(snapshot)
        self.assertIn("Demo", text)
        self.assertIn("Sharpe=1.20", text)

    def test_build_diagnosis_input_contains_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir)
            (docs_dir / "risk.md").write_text(
                "Risk management\n\nPosition sizing and stop loss are critical.",
                encoding="utf-8",
            )
            retriever = BookRetriever(docs_dir)
            snapshot = BacktestSnapshot(
                strategy_name="Demo",
                total_return=0.02,
                max_drawdown=-0.1,
                sharpe_ratio=0.5,
                trade_count=12,
                bad_cases=[{"date": "2024-02-01", "event": "consecutive stop loss"}],
            )
            data = build_diagnosis_input(snapshot, retriever)
            self.assertIn("prompt", data)
            self.assertIn("root cause", data["prompt"])
            self.assertGreaterEqual(len(data["references"]), 1)


if __name__ == "__main__":
    unittest.main()
