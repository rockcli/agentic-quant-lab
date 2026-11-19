"""
Auto-iteration pipeline: diagnose -> parameter adjustment -> re-backtest -> log experiment

Ref: Experimentation for Engineers (Kohavi et al.), Ch.5
"""

from __future__ import annotations

import copy
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_quant.ai.experiment.tracker import ExperimentTracker
from agentic_quant.ai.llm_client import LLMClient
from agentic_quant.ai.pipeline.diagnose import DiagnosisPipeline, DiagnosisReport
from agentic_quant.backtest.core.engine import BacktestEngine
from agentic_quant.backtest.core.result import BacktestResult
from agentic_quant.backtest.data.structures import Bar
from agentic_quant.backtest.strategy.base import Strategy


@dataclass
class IterationResult:
    """Single iteration result"""

    iteration: int
    strategy: Strategy
    backtest_result: BacktestResult
    diagnosis: DiagnosisReport
    params: dict[str, Any]


class AutoIterationPipeline:
    """
    Auto-iteration pipeline

    Pipeline:
    1. Run initial strategy backtest
    2. Diagnose backtest result
    3. Adjust parameters according to diagnosis suggestion (or regenerate strategy)
    4. Re-run backtest
    5. Log experiment and compare
    6. Repeat until termination condition is reached
    """

    def __init__(
        self,
        tracker: ExperimentTracker,
        diagnose_pipeline: DiagnosisPipeline | None = None,
        llm: LLMClient | None = None,
    ):
        self.tracker = tracker
        self.diagnose = diagnose_pipeline or DiagnosisPipeline(llm=llm)
        self.llm = llm

    def run(
        self,
        bars: list[Bar],
        strategy_factory: Callable[[dict[str, Any]], Strategy],
        initial_params: dict[str, Any],
        max_iterations: int = 3,
        param_mutation_rules: dict[str, tuple[float, float]] | None = None,
    ) -> list[IterationResult]:
        """
        Execute auto-iteration

        Args:
            bars: Backtest data
            strategy_factory: Strategy factory function, receives params and returns strategy instance
            initial_params: Initial parameters
            max_iterations: Maximum iteration rounds
            param_mutation_rules: Parameter mutation rules {param_name: (min_ratio, max_ratio)}
        """
        results: list[IterationResult] = []
        current_params = copy.deepcopy(initial_params)
        parent_iteration: int | None = None

        for i in range(max_iterations):
            iteration = i + 1

            # 1. Build strategy and backtest
            strategy = strategy_factory(current_params)
            engine = BacktestEngine()
            bt_result = engine.run(bars, strategy)

            # 2. Diagnose
            diagnosis = self.diagnose.run(bt_result, strategy_name=strategy.__class__.__name__)

            # 3. LogExperiment
            self.tracker.log(
                iteration=iteration,
                strategy_name=strategy.__class__.__name__,
                strategy_code=strategy.__class__.__name__,
                params=current_params,
                result=bt_result,
                diagnosis=diagnosis,
                parent_iteration=parent_iteration,
            )

            result = IterationResult(
                iteration=iteration,
                strategy=strategy,
                backtest_result=bt_result,
                diagnosis=diagnosis,
                params=copy.deepcopy(current_params),
            )
            results.append(result)

            # 4. CheckTerminateCondition
            if self._should_stop(bt_result, diagnosis, iteration, max_iterations):
                break

            # 5. Parameter mutation
            current_params = self._mutate_params(
                current_params, diagnosis, param_mutation_rules
            )
            parent_iteration = iteration

        return results

    @staticmethod
    def _should_stop(
        result: BacktestResult,
        diagnosis: DiagnosisReport,
        iteration: int,
        max_iterations: int,
    ) -> bool:
        """TerminateConditionCheck"""
        return (
            iteration >= max_iterations
            or any("None to optimize" in s for s in diagnosis.priority)
            or (result.sharpe_ratio > 2.0 and result.num_trades > 10)
        )

    @staticmethod
    def _mutate_params(
        params: dict[str, Any],
        diagnosis: DiagnosisReport,
        rules: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, Any]:
        """
        Mutate parameters based on diagnosis suggestions

        Default rule: numeric parameters randomly perturbed within +/-20% range
        """
        new_params = copy.deepcopy(params)
        rules = rules or {}

        # Prioritize adjustment based on diagnosis suggestions
        for sugg in diagnosis.suggestions:
            target = sugg.get("target", "")
            if target in new_params and isinstance(new_params[target], (int, float)):
                min_r, max_r = rules.get(target, (0.8, 1.2))
                factor = random.uniform(min_r, max_r)
                new_params[target] = new_params[target] * factor
                if isinstance(new_params[target], int):
                    new_params[target] = int(round(new_params[target]))

        # If diagnosis does not give specific parameter suggestions, randomly perturb all numeric parameters
        if not diagnosis.suggestions:
            for key, val in new_params.items():
                if isinstance(val, (int, float)):
                    min_r, max_r = rules.get(key, (0.8, 1.2))
                    factor = random.uniform(min_r, max_r)
                    new_params[key] = val * factor
                    if isinstance(new_params[key], int):
                        new_params[key] = int(round(new_params[key]))

        return new_params

    def generate_report(self, output_path: Path | None = None) -> str:
        """GenerateIterativeComparisonReport"""
        summary = self.tracker.summary()
        lines = [
            f"# ExperimentReport: {summary['experiment']}",
            f"Total iterations: {summary['total_iterations']}",
            "",
            "## Per-round Metrics Comparison",
        ]
        for rec in summary["records"]:
            lines.append(
                f"- Iteration {rec['iteration']} ({rec['strategy']}): "
                f"Sharpe={rec.get('sharpe_ratio', 'N/A')}, Return={rec.get('total_return', 'N/A')}"
            )

        best = self.tracker.best_by("sharpe_ratio", maximize=True)
        if best:
            lines.extend([
                "",
                "## optimalExperiment",
                f"- Iteration {best.iteration}: Sharpe={best.result_metrics.get('sharpe_ratio')}",
                f"- Parameter: {best.params}",
            ])

        report = "\n".join(lines)
        if output_path:
            Path(output_path).write_text(report, encoding="utf-8")
        return report
