"""
Diagnose pipeline: backtest result -> structured diagnosis input -> LLM diagnosis -> improvement suggestions

Ref: Advances in Financial Machine Learning (Lopez de Prado), Ch.16
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_quant.ai.llm_client import LLMClient
from agentic_quant.ai.rag.retriever import HybridRetriever
from agentic_quant.backtest.core.result import BacktestResult


@dataclass
class DiagnosisReport:
    """LLM DiagnoseReport"""

    summary: str
    root_causes: list[dict[str, str]]
    suggestions: list[dict[str, Any]]
    priority: list[str]
    experiment_design: dict[str, Any]
    raw_response: str = ""


class DiagnosisPipeline:
    """
    Backtest result diagnosis pipeline

    Pipeline:
    1. Extract key metrics, trade samples, drawdown periods from backtest result
    2. Retrieve relevant theory from knowledge base (overfitting, stop loss, position sizing)
    3. Assemble diagnosis prompt and call LLM
    4. Parse structured diagnosis report
    """

    SYSTEM_PROMPT = """You are a senior quantitative risk control analyst, skilled at identifying strategy flaws from backtest results.

Diagnosis principles (ref AFML Ch.16):
- Distinguish in-sample overfitting vs out-of-sample decay
- Focus on P&L asymmetry (left tail risk)
- Check trading frequency vs cost coverage ability
- Evaluate parameter sensitivity

Output must strictly follow JSON format, do not output other content."""

    def __init__(self, retriever: HybridRetriever | None = None, llm: LLMClient | None = None):
        self.retriever = retriever
        self.llm = llm or LLMClient()

    def run(self, result: BacktestResult, strategy_code: str = "", strategy_name: str = "") -> DiagnosisReport:
        """ExecuteDiagnose"""
        snapshot = self._build_snapshot(result, strategy_name)

        # Retrieve relevant knowledge
        context = ""
        if self.retriever:
            queries = [
                "overfitting backtest",
                "drawdown recovery risk management",
                "position sizing stop loss",
            ]
            chunks = []
            for q in queries:
                chunks.extend(self.retriever.search(q, top_k=2))
            seen = set()
            filtered = []
            for c in chunks:
                cid = c["chunk"].chunk_id
                if cid not in seen:
                    seen.add(cid)
                    filtered.append(c)
            context = "\n\n".join(
                f"[{i+1}] {r['chunk'].title}\n{r['chunk'].text[:250]}..."
                for i, r in enumerate(filtered[:5])
            )

        user_prompt = self._build_prompt(snapshot, strategy_code, context)

        try:
            response = self.llm.chat(self.SYSTEM_PROMPT, user_prompt)
            report = self._parse_response(response.content)
            report.raw_response = response.content
        except Exception as exc:
            report = DiagnosisReport(
                summary=f"DiagnoseFailed: {exc}",
                root_causes=[],
                suggestions=[],
                priority=[],
                experiment_design={},
                raw_response=str(exc),
            )

        return report

    @staticmethod
    def _build_snapshot(result: BacktestResult, strategy_name: str = "") -> dict[str, Any]:
        """Convert backtest result into diagnosis snapshot"""
        snapshot = {
            "strategy": strategy_name,
            "metrics": result.to_dict(),
        }

        # Sample trade details
        if result.fills:
            fills_sorted = sorted(result.fills, key=lambda f: f.pnl if hasattr(f, "pnl") else 0)
            worst = fills_sorted[:3]
            best = fills_sorted[-3:] if len(fills_sorted) >= 3 else fills_sorted
            snapshot["worst_trades"] = [
                {"time": str(f.timestamp), "pnl": getattr(f, "pnl", None)} for f in worst
            ]
            snapshot["best_trades"] = [
                {"time": str(f.timestamp), "pnl": getattr(f, "pnl", None)} for f in best
            ]
        else:
            snapshot["worst_trades"] = []
            snapshot["best_trades"] = []

        # Equitycurve feature
        if result.equity_curve is not None and len(result.equity_curve) > 0:
            equity = result.equity_curve["equity"].to_list()
            snapshot["equity_curve_length"] = len(equity)
            snapshot["final_equity"] = equity[-1] if equity else None
        else:
            snapshot["equity_curve_length"] = 0
            snapshot["final_equity"] = None

        return snapshot

    @staticmethod
    def _build_prompt(snapshot: dict[str, Any], strategy_code: str, context: str) -> str:
        return f"""[Backtest Snapshot]
{json.dumps(snapshot, ensure_ascii=False, indent=2)}

[Strategy Code]
{strategy_code[:1500] if strategy_code else "(Not provided)"}

[Knowledge Base Snippet]
{context}

[Output Format]
Please output strictly in JSON format as follows:
{{
  "summary": "One-sentence diagnosis conclusion",
  "root_causes": [
    {{"cause": "Root cause description", "evidence": "Backtest data evidence"}},
  ],
  "suggestions": [
    {{"action": "Optimization action", "target": "Parameter/logic/risk control", "rationale": "Rationale"}},
  ],
  "priority": ["P0: ...", "P1: ...", "P2: ..."],
  "experiment_design": {{
    "variables": ["Variables to adjust"],
    "metrics": ["Observation metrics"],
    "stop_condition": "Termination condition"
  }}
}}"""

    @staticmethod
    def _parse_response(content: str) -> DiagnosisReport:
        """Parse JSON diagnosis report from LLM output"""
        # Try to extract ```json code block
        text = content.strip()
        if "```json" in text:
            start = text.index("```json") + len("```json")
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + len("```")
            end = text.index("```", start)
            text = text[start:end].strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Degrade: construct minimal report
            return DiagnosisReport(
                summary="JSON parse failed, degraded to raw output",
                root_causes=[],
                suggestions=[],
                priority=[],
                experiment_design={},
            )

        return DiagnosisReport(
            summary=data.get("summary", ""),
            root_causes=data.get("root_causes", []),
            suggestions=data.get("suggestions", []),
            priority=data.get("priority", []),
            experiment_design=data.get("experiment_design", {}),
        )

    @staticmethod
    def save(report: DiagnosisReport, output_dir: Path) -> Path:
        """SaveDiagnoseReport"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "diagnosis_report.json"
        path.write_text(
            json.dumps(
                {
                    "summary": report.summary,
                    "root_causes": report.root_causes,
                    "suggestions": report.suggestions,
                    "priority": report.priority,
                    "experiment_design": report.experiment_design,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return path
