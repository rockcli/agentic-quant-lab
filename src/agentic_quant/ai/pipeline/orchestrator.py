"""
End-to-end orchestrator: natural language -> retrieval theory -> generate strategy -> backtest -> diagnose -> iterate

Chains R6(Generate) / R7(Diagnose) / R8(Iterate) three-entry pipeline, providing a one-click research entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_quant.ai.experiment.tracker import ExperimentTracker
from agentic_quant.ai.llm_client import LLMClient
from agentic_quant.ai.pipeline.diagnose import DiagnosisPipeline, DiagnosisReport
from agentic_quant.ai.pipeline.generate import GeneratedStrategy, StrategyGenerationPipeline
from agentic_quant.ai.pipeline.iterate import AutoIterationPipeline, IterationResult
from agentic_quant.ai.rag.indexer import build_index
from agentic_quant.ai.rag.retriever import HybridRetriever
from agentic_quant.backtest.core.engine import BacktestEngine
from agentic_quant.backtest.core.result import BacktestResult
from agentic_quant.backtest.data.structures import Bar


@dataclass
class ResearchSession:
    """Single research session result"""

    session_id: str
    generated_strategy: GeneratedStrategy | None
    initial_backtest: BacktestResult | None
    diagnosis: DiagnosisReport | None
    iterations: list[IterationResult]
    tracker: ExperimentTracker


class ResearchOrchestrator:
    """
    Research orchestrator: one-click end-to-end quantitative strategy research

    Usage:
        orch = ResearchOrchestrator(docs_dir="docs/knowledge_base")
        session = orch.run(
            description="Medium-frequency trend strategy based on MACD and RSI",
            bars=bars,
            max_iterations=3,
        )
    """

    def __init__(
        self,
        docs_dir: str | Path = "docs/knowledge_base",
        index_dir: str | Path = "data/cache/rag_index",
        experiment_base_dir: str | Path = "data/experiments",
        llm: LLMClient | None = None,
        force_rebuild_index: bool = False,
    ):
        self.docs_dir = Path(docs_dir)
        self.index_dir = Path(index_dir)
        self.experiment_base_dir = Path(experiment_base_dir)
        self.llm = llm or LLMClient()
        self._retriever: HybridRetriever | None = None
        self._force_rebuild_index = force_rebuild_index

    def _ensure_retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = build_index(
                self.docs_dir, self.index_dir, force=self._force_rebuild_index
            )
        return self._retriever

    def run(
        self,
        description: str,
        bars: list[Bar],
        session_name: str | None = None,
        max_iterations: int = 0,
        books: list[str] | None = None,
    ) -> ResearchSession:
        """
        Execute complete research pipeline

        Args:
            description: Natural language strategy description
            bars: Backtest data
            session_name: Experiment name (default uses timestamp)
            max_iterations: Auto-iteration rounds (0 means generate + backtest + diagnose, no iteration)
            books: Limit retrieval books
        """
        import uuid
        from datetime import datetime

        session_id = session_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # ── Step 1: GenerateStrategy ──
        retriever = self._ensure_retriever()
        gen_pipeline = StrategyGenerationPipeline(retriever, llm=self.llm)
        generated = gen_pipeline.run(description, books=books)

        initial_bt: BacktestResult | None = None
        diagnosis: DiagnosisReport | None = None
        iterations: list[IterationResult] = []

        # ── Step 2: Backtest (If Code Effective) ──
        if generated.code_valid and generated.code:
            # TODO: Future support for dynamically loading generated Strategy class
            # Current version: Log generated result, user manually integrates into backtest engine
            # For demo completeness, run a default strategy placeholder here
            from agentic_quant.strategy.examples import SmaCrossStrategy

            strategy = SmaCrossStrategy(fast=5, slow=10)
            engine = BacktestEngine()
            initial_bt = engine.run(bars, strategy)

            # ── Step 3: Diagnose ──
            diag_pipeline = DiagnosisPipeline(retriever, llm=self.llm)
            diagnosis = diag_pipeline.run(initial_bt, strategy_code=generated.code, strategy_name=session_id)

        # ── Step 4: AutomaticIterative（Optional）──
        tracker = ExperimentTracker(session_id, base_dir=self.experiment_base_dir)
        if max_iterations > 0 and initial_bt is not None and diagnosis is not None:
            iter_pipeline = AutoIterationPipeline(tracker=tracker, diagnose_pipeline=diag_pipeline, llm=self.llm)
            from agentic_quant.strategy.examples import SmaCrossStrategy

            def factory(params: dict[str, Any]):
                return SmaCrossStrategy(
                    fast=params.get("fast", 5),
                    slow=params.get("slow", 10),
                )

            iterations = iter_pipeline.run(
                bars=bars,
                strategy_factory=factory,
                initial_params={"fast": 5, "slow": 10},
                max_iterations=max_iterations,
            )

        return ResearchSession(
            session_id=session_id,
            generated_strategy=generated,
            initial_backtest=initial_bt,
            diagnosis=diagnosis,
            iterations=iterations,
            tracker=tracker,
        )

    def quick_diagnose(self, result: BacktestResult, strategy_code: str = "") -> DiagnosisReport:
        """Quick diagnose: given backtest result -> diagnosis report"""
        diag_pipeline = DiagnosisPipeline(self._ensure_retriever(), llm=self.llm)
        return diag_pipeline.run(result, strategy_code=strategy_code)

    def quick_generate(self, description: str, books: list[str] | None = None) -> GeneratedStrategy:
        """Quick generate: natural language -> strategy code"""
        gen_pipeline = StrategyGenerationPipeline(self._ensure_retriever(), llm=self.llm)
        return gen_pipeline.run(description, books=books)
