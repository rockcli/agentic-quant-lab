import unittest

from agentic_quant.ai.mvp_validation import BacktestSnapshot, build_diagnosis_input


class DummyRetriever:
    """Test double for validating retrieval fallback logic."""

    def __init__(self):
        self.calls = []

    def search(self, query: str, top_k: int = 3):
        self.calls.append((query, top_k))
        if query == "risk control position sizing stop loss drawdown":
            return []
        return [
            type(
                "SearchResult",
                (),
                {
                    "title": "Risk Book",
                    "snippet": "position sizing and stop loss",
                    "source_path": "/tmp/risk.md",
                },
            )()
        ]


class SystemContractsTest(unittest.TestCase):
    def test_mvp_diagnosis_has_retrieval_fallback(self):
        snapshot = BacktestSnapshot(
            strategy_name="Demo",
            total_return=0.01,
            max_drawdown=-0.08,
            sharpe_ratio=0.4,
            trade_count=20,
            bad_cases=[{"date": "2024-02-01", "event": "consecutive stop loss"}],
        )
        retriever = DummyRetriever()
        result = build_diagnosis_input(snapshot, retriever)

        self.assertEqual(len(retriever.calls), 2)
        self.assertEqual(retriever.calls[0][0], "risk control position sizing stop loss drawdown")
        self.assertEqual(retriever.calls[1][0], "trading risk management")
        self.assertEqual(len(result["references"]), 1)
        self.assertIn("Risk Book", result["prompt"])
        self.assertIn("consecutive stop loss", result["prompt"])


if __name__ == "__main__":
    unittest.main()
