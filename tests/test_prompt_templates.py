import unittest

from agentic_quant.ai.rag.prompt_templates import (
    build_risk_review_prompt,
    build_strategy_generation_prompt,
)


class PromptTemplatesTest(unittest.TestCase):
    def test_strategy_prompt_contains_required_sections(self):
        prompt = build_strategy_generation_prompt(
            strategy_goal="MACD Golden Cross + RSI Reversal",
            retrieved_context="Retrieved content from books",
        )
        self.assertIn("You are a vn.py CTA strategy engineer", prompt)
        self.assertIn("MACD Golden Cross + RSI Reversal", prompt)
        self.assertIn("Retrieved content from books", prompt)
        self.assertIn("Output complete Python code", prompt)

    def test_risk_prompt_contains_required_sections(self):
        prompt = build_risk_review_prompt(
            strategy_code="class Demo: pass",
            backtest_summary="Total Return 8%",
            bad_case_records="Consecutive Stop Loss",
            retrieved_context="Risk Management book snippet",
        )
        self.assertIn("You are a quantitative risk control reviewer", prompt)
        self.assertIn("class Demo: pass", prompt)
        self.assertIn("Total Return 8%", prompt)
        self.assertIn("Consecutive Stop Loss", prompt)
        self.assertIn("P0/P1/P2", prompt)


if __name__ == "__main__":
    unittest.main()
