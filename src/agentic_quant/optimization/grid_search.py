"""
Parameter Optimization Framework: Grid Search + Bayesian Optimization

Reference: Testing and Tuning Market Trading Systems (Masters)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import logging

from agentic_quant.backtest.core.engine import BacktestEngine
from agentic_quant.backtest.data.structures import Bar
from agentic_quant.backtest.strategy.base import Strategy


@dataclass
class OptimizationResult:
    """OptimizeResult"""

    best_params: dict[str, Any]
    best_metric: float
    metric_name: str
    all_results: list[tuple[dict[str, Any], float]]


class Optimizer(ABC):
    """Optimizer Abstract Base Class"""

    def __init__(self, metric: str = "sharpe_ratio"):
        self.metric = metric

    @abstractmethod
    def optimize(
        self,
        strategy_cls: type[Strategy],
        bars: list[Bar],
        **kwargs: Any,
    ) -> OptimizationResult:
        """Execute optimization"""
        ...

    def _evaluate(self, strategy_cls: type[Strategy], bars: list[Bar], params: dict[str, Any]) -> float:
        """Evaluate a single group of parameters."""
        try:
            engine = BacktestEngine()
            strategy = strategy_cls(**params)
            result = engine.run(bars, strategy)
            if not hasattr(result, self.metric):
                raise AttributeError(f"BacktestResult has no metric {self.metric!r}")
            return float(getattr(result, self.metric))
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "GridSearch parameter evaluation failed; params=%s error=%s", params, exc
            )
            return float("-inf")


class GridSearchOptimizer(Optimizer):
    """
    Exhaustive Grid Search
    Suitable for small parameter space scenarios.
    """

    def optimize(  # type: ignore[override]
        self,
        strategy_cls: type[Strategy],
        bars: list[Bar],
        param_grid: list[dict[str, Any]],
    ) -> OptimizationResult:
        """
        Args:
            param_grid: Parameter combination list, each item is a complete parameter dictionary
        """
        if not param_grid:
            raise ValueError("param_grid cannot be empty")

        all_results: list[tuple[dict[str, Any], float]] = []
        best_params = None
        best_metric = float("-inf")

        for params in param_grid:
            metric_val = self._evaluate(strategy_cls, bars, params)
            all_results.append((params, metric_val))
            if metric_val > best_metric:
                best_metric = metric_val
                best_params = params

        if best_params is None:
            raise RuntimeError(
                f"GridSearchOptimizer: all {len(param_grid)} parameter sets failed for metric {self.metric!r}"
            )

        return OptimizationResult(
            best_params=best_params,
            best_metric=best_metric,
            metric_name=self.metric,
            all_results=all_results,
        )


class BayesianOptimizer(Optimizer):
    """
    Bayesian Optimization (reserved interface)
    Suitable for high-dimensional parameter space, requires optuna or scikit-optimize.
    """

    def optimize(  # type: ignore[override]
        self,
        strategy_cls: type[Strategy],
        bars: list[Bar],
        param_space: dict[str, tuple[float, float]],
        n_trials: int = 50,
    ) -> OptimizationResult:
        """
        Args:
            param_space: Parameter range dictionary, e.g. {"fast": (5, 20), "slow": (20, 60)}
            n_trials: Number of search trials
        """
        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError(
                "BayesianOptimizer requires optuna, install: pip install optuna"
            ) from exc

        def objective(trial: optuna.Trial) -> float:
            params = {}
            for name, (low, high) in param_space.items():
                # float parameters use suggest_float, int parameters use suggest_int
                if isinstance(low, float) or isinstance(high, float):
                    params[name] = trial.suggest_float(name, float(low), float(high))
                else:
                    params[name] = trial.suggest_int(name, int(low), int(high))
            return self._evaluate(strategy_cls, bars, params)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        all_results = [
            (trial.params, trial.value or float("-inf")) for trial in study.trials
        ]

        return OptimizationResult(
            best_params=study.best_params,
            best_metric=study.best_value or float("-inf"),
            metric_name=self.metric,
            all_results=all_results,
        )
