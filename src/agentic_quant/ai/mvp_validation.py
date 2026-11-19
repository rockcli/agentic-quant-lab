"""
Phase 4: Minimal validation pipeline (backtest result output + risk diagnosis loop)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agentic_quant.ai.rag.book_retriever import BookRetriever
from agentic_quant.ai.rag.prompt_templates import build_risk_review_prompt


@dataclass
class BacktestSnapshot:
    strategy_name: str
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    trade_count: int
    bad_cases: list[dict]


def format_snapshot(snapshot: BacktestSnapshot) -> str:
    return (
        f"Strategy={snapshot.strategy_name}, Total Return={snapshot.total_return:.2%}, "
        f"Max Drawdown={snapshot.max_drawdown:.2%}, Sharpe={snapshot.sharpe_ratio:.2f}, "
        f"Trade Count={snapshot.trade_count}"
    )


def build_diagnosis_input(snapshot: BacktestSnapshot, retriever: BookRetriever) -> dict:
    query = "risk control position sizing stop loss drawdown"
    refs = retriever.search(query, top_k=3)
    if not refs:
        refs = retriever.search("trading risk management", top_k=3)
    context = "\n\n".join(
        f"[{i+1}] {item.title}\n{item.snippet}\nSource: {item.source_path}"
        for i, item in enumerate(refs)
    )
    prompt = build_risk_review_prompt(
        strategy_code="# Paste this round's backtest strategy code here",
        backtest_summary=format_snapshot(snapshot),
        bad_case_records=json.dumps(snapshot.bad_cases, ensure_ascii=False, indent=2),
        retrieved_context=context,
    )
    return {"query": query, "references": refs, "prompt": prompt}


def run_demo() -> None:
    docs_dir = Path(__file__).resolve().parents[2] / "docs" / "knowledge_base"
    retriever = BookRetriever(docs_dir)
    snapshot = BacktestSnapshot(
        strategy_name="MACD_RSI_A-Share_Example",
        total_return=0.082,
        max_drawdown=-0.137,
        sharpe_ratio=0.91,
        trade_count=46,
        bad_cases=[
            {"date": "2024-02-06", "event": "Stop loss triggered by gap-down after chasing rally"},
            {"date": "2024-02-14", "event": "Consecutive false breakouts in choppy market"},
        ],
    )
    diagnosis_input = build_diagnosis_input(snapshot, retriever)

    print("=== MVP ValidateOutput ===")
    print(format_snapshot(snapshot))
    print(f"Retrieval Query: {diagnosis_input['query']}")
    print(f"Reference Snippet Count: {len(diagnosis_input['references'])}")
    print("Diagnose Prompt (first 500 characters):")
    print(diagnosis_input["prompt"][:500] + "...")


if __name__ == "__main__":
    run_demo()
