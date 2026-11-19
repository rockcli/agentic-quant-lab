"""
Demo: Book retrieval + prompt assembly
"""

from __future__ import annotations

from pathlib import Path

from agentic_quant.ai.rag.book_retriever import BookRetriever
from agentic_quant.ai.rag.prompt_templates import (
    build_risk_review_prompt,
    build_strategy_generation_prompt,
)


def main() -> None:
    docs_dir = Path(__file__).resolve().parents[3] / "docs" / "knowledge_base"
    retriever = BookRetriever(docs_dir=docs_dir)
    query = "MACD golden cross RSI oversold rebound position size risk control"
    results = retriever.search(query, top_k=3)

    context = "\n\n".join(
        f"[{idx+1}] {r.title}\nSource: {r.source_path}\nsnippet: {r.snippet}"
        for idx, r in enumerate(results)
    )

    strategy_prompt = build_strategy_generation_prompt(
        "MACD golden cross buy, RSI between 30-50 before allowing open position, set 3% fixed stop loss.",
        context,
    )
    risk_prompt = build_risk_review_prompt(
        strategy_code="# Put strategy code here",
        backtest_summary="2024-02 drawdown 15%, Sharpe dropped to 0.2.",
        bad_case_records="2024-02-05 to 2024-02-21 consecutive stop loss 7 times.",
        retrieved_context=context,
    )

    print("=== RetrievalResult ===")
    for item in results:
        print(f"- score={item.score:.3f} | {item.title} | {item.snippet[:80]}...")

    print("\n=== StrategyGenerate Prompt ===")
    print(strategy_prompt[:600] + "...\n")

    print("=== Risk Review Prompt ===")
    print(risk_prompt[:600] + "...\n")


if __name__ == "__main__":
    main()
