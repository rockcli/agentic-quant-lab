"""
Phase 3: Strategy generation and risk review prompt templates
"""

from __future__ import annotations

from textwrap import dedent


def build_strategy_generation_prompt(strategy_goal: str, retrieved_context: str) -> str:
    return dedent(
        f"""
        You are a vn.py CTA strategy engineer. Please generate runnable strategy code based on knowledge base snippets.

        【Strategy Objective】
        {strategy_goal}

        【Knowledge BaseRetrievalsnippet】
        {retrieved_context}

        [Output Requirements]
        1) Output complete Python code, class must inherit CtaTemplate.
        2) Must include parameter area, variable area, on_init/on_start/on_stop/on_bar.
        3) Must include explicit risk control: stop loss, position size management.
        4) Do not output pseudo-code, code must be directly usable for vn.py backtesting.
        5) Append 3-5 entries of "consistency explanation with knowledge base" after the code, citing key logic per entry.
        """
    ).strip()


def build_risk_review_prompt(
    strategy_code: str,
    backtest_summary: str,
    bad_case_records: str,
    retrieved_context: str,
) -> str:
    return dedent(
        f"""
        You are a quantitative risk control reviewer. Please diagnose the strategy failure periods in conjunction with book theory.

        【Strategy Code】
        {strategy_code}

        [Backtest Summary]
        {backtest_summary}

        [Failed Samples (Bad Cases)]
        {bad_case_records}

        【Knowledge BaseRetrievalsnippet】
        {retrieved_context}

        [Output Format]
        - Risk root causes (at least 3 entries)
        - Corresponding evidence (backtest observation + knowledge base theory)
        - Actionable optimization suggestions (parameter, filter, position size, stop loss, trading frequency)
        - Priority level sorting (P0/P1/P2)
        - Next round experiment design (control groups, observation metrics, termination condition)
        """
    ).strip()
