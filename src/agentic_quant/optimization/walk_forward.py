"""
Walk-Forward Analysis — Complete Purged WFA Implement

References:
- Testing and Tuning Market Trading Systems (Masters)
- Advances in Financial Machine Learning (Lopez de Prado), Ch.7, 12
- AlgoXpert Alpha Research Framework (arXiv:2603.09219)

Key Enhancements:
- Purge Gap: Add time gap between training and test sets, prevent information leakage
- Majority Rule: Passed only if >=70% OOS Windows have positive return
- Catastrophic Veto: Any OOS Window with drawdown exceeding threshold is directly eliminated
- Regime Analysis: Evaluate performance segmented by Volatility/Trend
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from agentic_quant.backtest.core.engine import BacktestEngine
from agentic_quant.backtest.core.result import BacktestResult
from agentic_quant.backtest.data.structures import Bar
from agentic_quant.backtest.strategy.base import Strategy
from agentic_quant.optimization.grid_search import GridSearchOptimizer
from agentic_quant.validation.regime import MarketRegime

logger = logging.getLogger(__name__)


# ──────────────────────── Data Structure ──────────────────────────

@dataclass
class WFOSplit:
    """WFO Train/Test Split"""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    purge_start: datetime | None = None
    purge_end: datetime | None = None


@dataclass
class WFOSplitResult:
    """Single Window WFO Result"""
    split: WFOSplit
    best_params: dict[str, Any]
    train_metric: float
    test_metric: float
    train_result: BacktestResult | None = None
    test_result: BacktestResult | None = None


@dataclass
class WFAReport:
    """WFA Complete Report"""
    strategy_name: str
    total_windows: int
    passed_windows: int
    oos_positive_rate: float
    majority_pass: bool
    catastrophic_veto: bool
    catastrophic_details: list[str]
    avg_oos_sharpe: float
    avg_oos_max_dd: float
    avg_oos_total_return: float
    param_stability: float
    robustness_ratio: float
    metric_decay: float
    grade: str
    window_results: list[dict[str, Any]] = field(default_factory=list)
    regime_analysis: dict[str, Any] = field(default_factory=dict)
    regime_breakdown: dict[str, Any] = field(default_factory=dict)
    summary_lines: list[str] = field(default_factory=list)


# ──────────────────────── Walk-Forward Optimizer ──────────────────────

class WalkForwardOptimizer:
    """
    Purged Walk-Forward Optimizer

    Split the total time range into N non-overlapping (train, purge, test) triple groups,
    each pair optimizes parameters on the training set and validates on the test set.
    """

    def __init__(
        self,
        total_start: datetime,
        total_end: datetime,
        train_rolls: int = 5,
        train_pct: float = 0.6,
        purge_days: int = 5,
    ):
        """
        Args:
            total_start: Total backtest start time
            total_end: Total backtest end time
            train_rolls: Number of rolling windows
            train_pct: Training set proportion (Test Set = 1-train_pct)
            purge_days: Purge interval days between train/test (prevents information leakage)
        """
        if not (0.5 <= train_pct < 0.95):
            raise ValueError("train_pct should be in [0.5, 0.95) range")
        if total_end <= total_start:
            raise ValueError("End time must be greater than start time")
        self.total_start = total_start
        self.total_end = total_end
        self.train_rolls = train_rolls
        self.train_pct = train_pct
        self.purge_days = purge_days

    def splits(self) -> list[WFOSplit]:
        """Generate train/test split list with purge gap"""
        total_seconds = (self.total_end - self.total_start).total_seconds()
        window_seconds = total_seconds / self.train_rolls
        train_seconds = window_seconds * self.train_pct
        test_seconds = window_seconds * (1 - self.train_pct)
        purge = timedelta(days=self.purge_days)

        result: list[WFOSplit] = []
        for i in range(self.train_rolls):
            window_start = self.total_start + timedelta(seconds=i * window_seconds)
            train_end = window_start + timedelta(seconds=train_seconds)
            purge_start = train_end
            purge_end = train_end + purge
            test_start = purge_end
            test_end = min(window_start + timedelta(seconds=train_seconds + test_seconds), self.total_end)

            result.append(WFOSplit(
                train_start=window_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                purge_start=purge_start,
                purge_end=purge_end,
            ))
        return result

    def optimize(
        self,
        strategy_cls: type[Strategy],
        bars: list[Bar],
        param_grid: list[dict[str, Any]],
        metric: str = "sharpe_ratio",
    ) -> list[WFOSplitResult]:
        """Execute Walk-Forward Optimize"""
        splits = self.splits()
        results: list[WFOSplitResult] = []

        for split in splits:
            train_bars = [b for b in bars if split.train_start <= b.timestamp <= split.train_end]
            test_bars = [b for b in bars if split.test_start <= b.timestamp <= split.test_end]

            if not train_bars or not test_bars:
                continue

            best_params = None
            best_train_metric = float("-inf")
            best_train_result: BacktestResult | None = None

            for params in param_grid:
                try:
                    engine = BacktestEngine()
                    strategy = strategy_cls(**params)
                    result = engine.run(train_bars, strategy)
                    if not hasattr(result, metric):
                        raise AttributeError(f"BacktestResult has no metric {metric!r}")
                    val = float(getattr(result, metric))
                    if val > best_train_metric:
                        best_train_metric = val
                        best_params = params
                        best_train_result = result
                except Exception as exc:
                    logger.warning(
                        "WFA optimize: param grid evaluation failed; params=%s error=%s",
                        params,
                        exc,
                    )
                    continue

            if best_params is None:
                logger.warning("WFA optimize: no parameter set succeeded for split %s", split)
                continue

            engine = BacktestEngine()
            strategy = strategy_cls(**best_params)
            test_result = engine.run(test_bars, strategy)
            if not hasattr(test_result, metric):
                raise AttributeError(f"BacktestResult has no metric {metric!r}")
            test_metric = float(getattr(test_result, metric))

            results.append(WFOSplitResult(
                split=split,
                best_params=best_params,
                train_metric=best_train_metric,
                test_metric=test_metric,
                train_result=best_train_result,
                test_result=test_result,
            ))

        return results

    def stability_report(self, results: list[WFOSplitResult]) -> dict[str, float]:
        """ParameterStabilityReport"""
        if not results:
            return {}
        train_vals = [r.train_metric for r in results]
        test_vals = [r.test_metric for r in results]
        deltas = [t - e for t, e in zip(train_vals, test_vals, strict=False)]
        return {
            "train_mean": statistics.mean(train_vals),
            "test_mean": statistics.mean(test_vals),
            "delta_mean": statistics.mean(deltas),
            "delta_std": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "rolls": len(results),
        }


# ──────────────────────── WFA Analyzer (Enhanced) ──────────────────────

class WalkForwardAnalyzer:
    """
    Walk-Forward Comprehensive Analyzer

    Complete WFA Validation Pipeline:
    1. Rolling Window Train/Test
    2. Majority Rule Check
    3. Catastrophic Veto Check
    4. Regime-dependent Analysis
    5. Comprehensive Rating
    """

    def __init__(
        self,
        total_start: datetime,
        total_end: datetime,
        train_rolls: int = 5,
        train_pct: float = 0.6,
        purge_days: int = 5,
        metric: str = "sharpe_ratio",
        majority_threshold: float = 0.70,
        max_drawdown_veto: float = 0.30,
    ):
        """
        Args:
            total_start: Total backtest start time
            total_end: Total backtest end time
            train_rolls: Number of rolling windows (Suggestion ≥8)
            train_pct: Training set proportion
            purge_days: Train/Test purge interval
            metric: Optimization target metric
            majority_threshold: Majority passed threshold (default 70%)
            max_drawdown_veto: Catastrophic veto drawdown threshold (default 30%)
        """
        self.wfo = WalkForwardOptimizer(
            total_start=total_start,
            total_end=total_end,
            train_rolls=train_rolls,
            train_pct=train_pct,
            purge_days=purge_days,
        )
        self.metric = metric
        self.majority_threshold = majority_threshold
        self.max_drawdown_veto = max_drawdown_veto
        self.total_start = total_start
        self.total_end = total_end

    def _compute_regime_breakdown(
        self,
        window_results: list[WFOSplitResult],
        close_prices: np.ndarray,
    ) -> dict[str, Any]:
        """Use MarketRegime to calculate 4-quadrant regime decomposition for WFA windows"""
        if len(window_results) < 2 or len(close_prices) < 200:
            return {}
        return MarketRegime.wfa_breakdown(  # type: ignore[no-any-return]
            window_results=window_results,
            close_prices=close_prices,
            total_start=self.total_start,
            total_end=self.total_end,
        )

    def analyze(  # type: ignore[no-redef]
        self,
        strategy_cls: type[Strategy],
        bars: list[Bar],
        param_grid: list[dict[str, Any]],
        strategy_name: str = "",
    ) -> WFAReport:
        """ExecuteComplete Walk-Forward Analyze"""
        splits = self.wfo.splits()
        window_results: list[WFOSplitResult] = []

        for split in splits:
            train_bars = [b for b in bars if split.train_start <= b.timestamp <= split.train_end]
            test_bars = [b for b in bars if split.test_start <= b.timestamp <= split.test_end]

            if not train_bars or not test_bars:
                continue

            optimizer = GridSearchOptimizer(metric=self.metric)
            opt_result = optimizer.optimize(strategy_cls, train_bars, param_grid=param_grid)
            best_params = opt_result.best_params

            if not best_params:
                continue

            engine_train = BacktestEngine()
            train_bt = engine_train.run(train_bars, strategy_cls(**best_params))
            train_metric = getattr(train_bt, self.metric, 0.0)

            engine_test = BacktestEngine()
            test_bt = engine_test.run(test_bars, strategy_cls(**best_params))
            test_metric = getattr(test_bt, self.metric, 0.0)

            window_results.append(WFOSplitResult(
                split=split,
                best_params=best_params,
                train_metric=train_metric,
                test_metric=test_metric,
                train_result=train_bt,
                test_result=test_bt,
            ))

        report = self._build_report(window_results, strategy_name)
        # Regime Decomposition (Based on actual price data)
        if bars and len(window_results) >= 2:
            close_prices = np.array([b.close for b in bars], dtype=float)
            report.regime_breakdown = self._compute_regime_breakdown(window_results, close_prices)
        return report

    def analyze_vectorized(
        self,
        close_prices: np.ndarray,
        param_grid: list[dict[str, Any]],
        strategy_fn,
        strategy_name: str = "",
    ) -> WFAReport:
        """
        Vectorized WFA: Calculate directly using numpy arrays, avoid per-bar backtest overhead.

        Args:
            close_prices: 1-D CloseArray
            param_grid: Parameter Grid
            strategy_fn: (close, **params) -> (entries, exits) Signal GenerationFunction
        """
        window_results: list[WFOSplitResult] = []
        splits = self.wfo.splits()

        # Map timestamps to indices proportionally to the actual data length.
        # Do not blindly use calendar-day counts as array indices.
        total_days = len(close_prices)
        total_span = (self.total_end - self.total_start).total_seconds() or 1.0

        def _rel_index(dt):
            return int(round((dt - self.total_start).total_seconds() / total_span * (total_days - 1)))

        for split in splits:
            train_start_idx = max(0, _rel_index(split.train_start))
            train_end_idx = min(total_days - 1, _rel_index(split.train_end))
            test_start_idx = max(0, _rel_index(split.test_start))
            test_end_idx = min(total_days - 1, _rel_index(split.test_end))

            # Each fold window at least 80 days
            if train_end_idx - train_start_idx < 80 or test_end_idx - test_start_idx < 80:
                continue

            train_close = close_prices[train_start_idx:train_end_idx + 1]
            test_close = close_prices[test_start_idx:test_end_idx + 1]

            best_params = None
            best_train_sharpe = float("-inf")

            for params in param_grid:
                try:
                    entries, exits = strategy_fn(train_close, **params)
                    sharpe = _compute_sharpe(train_close, entries, exits)
                    if sharpe > best_train_sharpe:
                        best_train_sharpe = sharpe
                        best_params = params
                except Exception as exc:
                    logger.warning(
                        "WFA date-index optimize: param evaluation failed; params=%s error=%s",
                        params,
                        exc,
                    )
                    continue

            if best_params is None:
                logger.warning("WFA date-index optimize: no params succeeded for split %s", split)
                continue

            entries, exits = strategy_fn(test_close, **best_params)
            test_sharpe = _compute_sharpe(test_close, entries, exits)
            test_return = _compute_total_return(test_close, entries, exits)
            test_max_dd = _compute_max_drawdown(test_close, entries, exits)

            window_results.append(WFOSplitResult(
                split=split,
                best_params=best_params,
                train_metric=best_train_sharpe,
                test_metric=test_sharpe,
                test_result=BacktestResult(
                    equity_curve=__import__("polars").DataFrame(),
                    fills=[], signals=[],
                    total_return=test_return,
                    sharpe_ratio=test_sharpe,
                    max_drawdown=test_max_dd,
                    win_rate=0.0, profit_factor=0.0, num_trades=0,
                ),
            ))

        report = self._build_report(window_results, strategy_name)
        # Regime DecompositionAnalyze（Based OnCompletePricesequence）
        if len(close_prices) >= 200 and len(window_results) >= 2:
            report.regime_breakdown = self._compute_regime_breakdown(window_results, close_prices)
        return report

    def _build_report(
        self, window_results: list[WFOSplitResult], strategy_name: str = ""
    ) -> WFAReport:
        """Build WFA Report from Window Results"""
        if not window_results:
            return WFAReport(
                strategy_name=strategy_name,
                total_windows=0, passed_windows=0,
                oos_positive_rate=0.0, majority_pass=False,
                catastrophic_veto=False, catastrophic_details=[],
                avg_oos_sharpe=0.0, avg_oos_max_dd=0.0,
                avg_oos_total_return=0.0, param_stability=0.0,
                robustness_ratio=0.0, metric_decay=0.0, grade="NoneData",
            )

        n = len(window_results)

        # Majority passed check
        oos_returns = [r.test_result.total_return if r.test_result else 0.0 for r in window_results]
        positive_windows = sum(1 for r in oos_returns if r > 0)
        oos_positive_rate = positive_windows / n
        majority_pass = oos_positive_rate >= self.majority_threshold

        # Catastrophic Veto
        catastrophic_details: list[str] = []
        catastrophic_veto = False
        for i, r in enumerate(window_results):
            if r.test_result:
                dd = abs(r.test_result.max_drawdown)
                if dd > self.max_drawdown_veto:
                    catastrophic_veto = True
                    catastrophic_details.append(
                        f"Window{i+1}: Max Drawdown {dd:.1%} > Threshold {self.max_drawdown_veto:.0%}"
                    )
                # Single window return < -20% also considered catastrophic
                if r.test_result.total_return < -0.20:
                    catastrophic_veto = True
                    catastrophic_details.append(
                        f"Window{i+1}: Return {r.test_result.total_return:.1%} < -20%"
                    )

        # Statistics
        oos_sharpes = [r.test_result.sharpe_ratio if r.test_result else 0.0 for r in window_results]
        oos_max_dds = [r.test_result.max_drawdown if r.test_result else 0.0 for r in window_results]

        avg_oos_sharpe = statistics.mean(oos_sharpes)
        avg_oos_max_dd = statistics.mean(oos_max_dds)
        avg_oos_return = statistics.mean(oos_returns)

        # ParameterStability
        param_stability = WalkForwardAnalyzer._compute_param_stability(
            [r.best_params for r in window_results]
        )

        # Robustness ratio: proportion of windows where OOS Sharpe >= IS Sharpe x 0.6
        robust_count = sum(
            1 for r in window_results
            if r.test_metric >= r.train_metric * 0.6
        )
        robustness_ratio = robust_count / n

        # Performance Decay
        train_vals = [r.train_metric for r in window_results]
        test_vals = [r.test_metric for r in window_results]
        metric_decay = max(0.0, 1.0 - (
            statistics.mean(test_vals) / (statistics.mean(train_vals) + 1e-9)
        ))

        # Comprehensive Rating
        grade = _compute_wfa_grade(
            majority_pass, catastrophic_veto, avg_oos_sharpe,
            robustness_ratio, param_stability,
        )

        # WindowDetail
        window_detail: list[dict[str, Any]] = []
        for i, r in enumerate(window_results):
            detail = {
                "window": i + 1,
                "train_period": f"{r.split.train_start.date()} → {r.split.train_end.date()}",
                "test_period": f"{r.split.test_start.date()} → {r.split.test_end.date()}",
                "best_params": r.best_params,
                "train_sharpe": round(r.train_metric, 4),
                "test_sharpe": round(r.test_metric, 4),
            }
            if r.test_result:
                detail["test_return"] = round(r.test_result.total_return, 4)
                detail["test_max_dd"] = round(r.test_result.max_drawdown, 4)
            window_detail.append(detail)

        # Regime Analysis
        regime_analysis = _regime_analysis(window_results)

        # Summary
        summary = [
            f"## WFA Acceptance Conclusion: {grade}",
            f"- Total Windows: {n}",
            f"- OOS Positive Return Windows: {positive_windows}/{n} ({oos_positive_rate:.0%})",
            f"- {'✅ Majority Passed' if majority_pass else '❌ Majority Failed'} (Threshold {self.majority_threshold:.0%})",
            f"- {'❌ TriggerCatastrophic Veto' if catastrophic_veto else '✅ NoneCatastrophic Veto'}",
            f"- Averaging OOS Sharpe: {avg_oos_sharpe:.2f}",
            f"- Average OOS Drawdown: {avg_oos_max_dd:.1%}",
            f"- Parameter Stability: {param_stability:.1%}",
            f"- Robustness Ratio: {robustness_ratio:.1%}",
            f"- Performance Decay Rate: {metric_decay:.1%}",
        ]

        return WFAReport(
            strategy_name=strategy_name,
            total_windows=n,
            passed_windows=positive_windows,
            oos_positive_rate=oos_positive_rate,
            majority_pass=majority_pass,
            catastrophic_veto=catastrophic_veto,
            catastrophic_details=catastrophic_details,
            avg_oos_sharpe=avg_oos_sharpe,
            avg_oos_max_dd=avg_oos_max_dd,
            avg_oos_total_return=avg_oos_return,
            param_stability=param_stability,
            robustness_ratio=robustness_ratio,
            metric_decay=metric_decay,
            grade=grade,
            window_results=window_detail,
            regime_analysis=regime_analysis,
            summary_lines=summary,
        )

    @staticmethod
    def _compute_param_stability(param_history: list[dict[str, Any]]) -> float:
        """CalculateParameter Stability Score [0, 1]"""
        if not param_history or len(param_history) <= 1:
            return 0.0

        numeric_params: dict[str, list[float]] = {}
        for params in param_history:
            for k, v in params.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    numeric_params.setdefault(k, []).append(float(v))

        if not numeric_params:
            return 0.0

        scores: list[float] = []
        for values in numeric_params.values():
            if len(values) < 2:
                continue
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values)
            if abs(mean_val) < 1e-9:
                scores.append(0.0)
                continue
            cv = std_val / abs(mean_val)
            scores.append(max(0.0, 1.0 - min(cv, 1.0)))

        return statistics.mean(scores) if scores else 0.0


# ──────────────────────── Report Generation ──────────────────────────

def generate_wfa_report(report: WFAReport, output_dir: str | Path | None = None) -> str:
    """Generate WFA Markdown Report"""
    lines = [
        f"# 🔄 WFA Walk-Forward ValidationReport{f' — {report.strategy_name}' if report.strategy_name else ''}",
        "",
        f"**testTime**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. Acceptance Conclusion",
        "",
        f"**Comprehensive Rating: {report.grade}**",
        "",
    ] + report.summary_lines + [
        "",
        "## 2. Per-Window Details",
        "",
        "| Window | Train Period | Test Period | Parameter | TrainSharpe | TestSharpe | TestReturn | TestDrawdown |",
        "|------|--------|--------|------|------------|------------|----------|----------|",
    ]

    for w in report.window_results:
        params_str = ",".join(f"{k}={v}" for k, v in w["best_params"].items())
        lines.append(
            f"| {w['window']} | {w['train_period']} | {w['test_period']} "
            f"| {params_str} | {w['train_sharpe']} | {w['test_sharpe']} "
            f"| {w.get('test_return', 'N/A')} | {w.get('test_max_dd', 'N/A')} |"
        )

    if report.catastrophic_details:
        lines.extend([
            "",
            "## 3. Catastrophic Veto Details",
            "",
        ])
        for d in report.catastrophic_details:
            lines.append(f"- ❌ {d}")

    if report.regime_analysis:
        lines.extend([
            "",
            "## 4. Regime Analysis (Traditional)",
            "",
        ])
        ra = report.regime_analysis
        if "high_vol" in ra:
            lines.append(f"- High Volatility Period OOS Sharpe: {ra['high_vol']:.3f}")
        if "low_vol" in ra:
            lines.append(f"- Low Volatility Period OOS Sharpe: {ra['low_vol']:.3f}")
        if "bull" in ra:
            lines.append(f"- Bull Market Period OOS Sharpe: {ra['bull']:.3f}")
        if "bear" in ra:
            lines.append(f"- Bear Market Period OOS Sharpe: {ra['bear']:.3f}")

    # Regime Decomposition 4 QuadrantReport（Based On MA200 + VolatilityMedian）
    if report.regime_breakdown and report.regime_breakdown.get("regimes"):
        rb = report.regime_breakdown
        lines.extend([
            "",
            "## 4-B. Regime Decomposition 4-Quadrant Performance Comparison",
            "",
            "Based on MA(200) Trend + 20-day Volatility Median Classification:",
            "",
            "| Regime | Windows | Proportion | OOS Sharpe | Win Rate | Avg Return | Avg Drawdown |",
            "|------|--------|------|------------|------|----------|----------|",
        ])
        for regime_name in ["bull_high_vol", "bull_low_vol", "bear_high_vol", "bear_low_vol"]:
            rdata = rb["regimes"].get(regime_name)
            if rdata:
                lines.append(
                    f"| {regime_name} | {rdata['window_count']} | {rdata['pct']:.0%} "
                    f"| {rdata['oos_sharpe']:.2f} | {rdata['oos_win_rate']:.0%} "
                    f"| {rdata['oos_return']:.2%} | {rdata['oos_max_dd']:.2%} |"
                )
            else:
                lines.append(f"| {regime_name} | 0 | 0% | - | - | - | - |")

        lines.extend([
            "",
            f"**OptimalRegime**: {rb.get('best_regime', 'N/A')}",
            f"**Worst Regime**: {rb.get('worst_regime', 'N/A')}",
            f"**Performance Spread**: {rb.get('spread', 0):.2f}",
            "",
        ])

    lines.extend([
        "",
        "## 5. Acceptance Criteria",
        "",
        "| Criteria | Threshold | Actual | Passed? |",
        "|------|------|------|--------|",
        f"| Majority Passed (OOS Positive Return) | ≥70% | {report.oos_positive_rate:.0%} | {'✅' if report.majority_pass else '❌'} |",
        f"| Catastrophic Veto (Single Window Drawdown) | <30% | {'Trigger' if report.catastrophic_veto else 'Not Triggered'} | {'❌' if report.catastrophic_veto else '✅'} |",
        f"| OOS Sharpe | ≥0.5 | {report.avg_oos_sharpe:.2f} | {'✅' if report.avg_oos_sharpe >= 0.5 else '❌'} |",
        f"| Performance Decay Rate | <50% | {report.metric_decay:.1%} | {'✅' if report.metric_decay < 0.5 else '❌'} |",
        f"| ParameterStability | ≥50% | {report.param_stability:.1%} | {'✅' if report.param_stability >= 0.5 else '❌'} |",
        "",
        "> References: De Prado, AFML Ch.7,12; AlgoXpert Framework (arXiv:2603.09219)",
    ])

    md = "\n".join(lines)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        (out / f"wfa_report_{ts}.md").write_text(md, encoding="utf-8")
        json_data = {
            "strategy": report.strategy_name,
            "grade": report.grade,
            "total_windows": report.total_windows,
            "oos_positive_rate": report.oos_positive_rate,
            "avg_oos_sharpe": report.avg_oos_sharpe,
            "avg_oos_max_dd": report.avg_oos_max_dd,
            "majority_pass": report.majority_pass,
            "catastrophic_veto": report.catastrophic_veto,
            "metric_decay": report.metric_decay,
            "param_stability": report.param_stability,
            "window_results": report.window_results,
            "regime_breakdown": report.regime_breakdown,
            "timestamp": datetime.now().isoformat(),
        }
        (out / f"wfa_report_{ts}.json").write_text(
            json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return md


# ──────────────────────── Helper Functions ──────────────────────────

def _compute_wfa_grade(
    majority_pass: bool,
    catastrophic_veto: bool,
    avg_oos_sharpe: float,
    robustness_ratio: float,
    param_stability: float,
) -> str:
    """Comprehensive Rating"""
    if catastrophic_veto:
        return "D (Catastrophic Veto)"
    if not majority_pass:
        return "C (Majority Failed)"
    if avg_oos_sharpe < 0.5:
        return "C (Insufficient OOS Sharpe)"
    if avg_oos_sharpe >= 1.0 and robustness_ratio >= 0.8 and param_stability >= 0.7:
        return "A (excellent)"
    if avg_oos_sharpe >= 0.7 and robustness_ratio >= 0.7 and param_stability >= 0.5:
        return "B (Good)"
    return "B- (Acceptable)"


def _compute_sharpe(close: np.ndarray, entries: np.ndarray, exits: np.ndarray) -> float:
    """Calculate Sharpe ratio from signal sequences (T+1 execution, exit bar retained)."""
    if len(entries) < 2:
        return 0.0
    rets = np.diff(close) / (close[:-1] + 1e-9)
    position = np.zeros(len(close), dtype=float)
    in_pos = False
    entry_bar = -1
    for i in range(len(entries)):
        if entries[i] and not in_pos and i + 1 < len(close):
            in_pos = True
            entry_bar = i + 1
        if exits[i] and in_pos:
            position[i] = 1.0
            in_pos = False
        elif in_pos and i >= entry_bar:
            position[i] = 1.0
    strategy_rets = rets * position[1:]
    if np.std(strategy_rets) < 1e-9:
        return 0.0
    return float(np.mean(strategy_rets) / np.std(strategy_rets) * np.sqrt(252))


def _compute_total_return(close: np.ndarray, entries: np.ndarray, exits: np.ndarray) -> float:
    """Calculate total return from signal sequences (T+1 execution, exit bar retained)."""
    if len(entries) < 2:
        return 0.0
    rets = np.diff(close) / (close[:-1] + 1e-9)
    position = np.zeros(len(close), dtype=float)
    in_pos = False
    entry_bar = -1
    for i in range(len(entries)):
        if entries[i] and not in_pos and i + 1 < len(close):
            in_pos = True
            entry_bar = i + 1
        if exits[i] and in_pos:
            position[i] = 1.0
            in_pos = False
        elif in_pos and i >= entry_bar:
            position[i] = 1.0
    return float(np.prod(1 + rets * position[1:]) - 1)


def _compute_max_drawdown(close: np.ndarray, entries: np.ndarray, exits: np.ndarray) -> float:
    """Calculate max drawdown from signal sequences (T+1 execution, exit bar retained)."""
    rets = np.diff(close) / (close[:-1] + 1e-9)
    position = np.zeros(len(close), dtype=float)
    in_pos = False
    entry_bar = -1
    for i in range(len(entries)):
        if entries[i] and not in_pos and i + 1 < len(close):
            in_pos = True
            entry_bar = i + 1
        if exits[i] and in_pos:
            position[i] = 1.0
            in_pos = False
        elif in_pos and i >= entry_bar:
            position[i] = 1.0
    strategy_rets = rets * position[1:]
    equity = np.cumprod(1 + strategy_rets)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-9)
    return float(abs(np.min(dd)))


# ═══════════════════════════ ExitParameter WFA Heatmap ═══════════════════

@dataclass
class ExitWFAHeatmap:
    """ExitParameter WFA Heatmap — Entry×ExitParameterCross ValidationResult

    Attributes:
        entry_labels: Entry parameter tag list (used for table row headers)
        exit_labels: Exit parameter tag list (used for table column headers)
        heatmap_matrix: (n_entry, n_exit) OOS Sharpe Matrix
        best_exit_idx: Global optimal exit parameter index (highest average OOS Sharpe)
        best_entry_per_exit: Optimal entry parameter index for each exit parameter
        best_overall_sharpe: Global optimal OOS Sharpe
    """
    entry_labels: list[str]
    exit_labels: list[str]
    heatmap_matrix: np.ndarray
    best_exit_idx: int
    best_entry_per_exit: list[int]
    best_overall_sharpe: float


def _format_exit_label(ep: dict[str, float | int]) -> str:
    """Format as short tag, e.g. PT2_SL1_H10"""
    pt = ep.get("pt_mult", 0)
    sl = ep.get("sl_mult", 0)
    hold = ep.get("max_hold", 0)
    return f"PT{pt:.0f}_SL{sl:.1f}_H{hold:.0f}".replace(".0_", "_").replace(".0", "")


def _format_entry_label(p: dict[str, Any]) -> str:
    """Format as short tag, e.g. F5_S20"""
    return f"F{p['fast']}_S{p['slow']}"


def run_exit_wfa(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    entry_param_grid: list[dict[str, Any]],
    exit_param_grid: list[dict[str, float | int]],
    train_rolls: int = 5,
    train_pct: float = 0.66,
    purge: int = 5,
    enable_vpa: bool = False,
) -> ExitWFAHeatmap:
    """
    ExitParameter WFA Heatmap — Cross ValidationEntry×ExitParameter

    For each exit parameter group, execute the complete rolling WFA pipeline:
    1. Evaluate all entry parameter combinations in each train window, select optimal
    2. Evaluate that entry parameter combination in the corresponding test window (with current exit parameter)
    3. Log OOS Sharpe for all entry parameters in the test window (used for heatmap)
    4. Average across windows -> fill into heatmap

    Args:
        opens/highs/lows/closes/volumes: OHLCV DataArray
        entry_param_grid: Entry parameter grid (fast/slow combinations)
        exit_param_grid: Exit parameter grid (pt_mult/sl_mult/max_hold combinations)
        train_rolls: WFA Number of rolling windows（Default 5 fold）
        train_pct: Training set proportion（Default 0.66）
        purge: Number of purge bars between train/test (default 5)
        enable_vpa: Whether to enable VPA filter (off by default for WFA, keep primary model independent)
    Returns:
        ExitWFAHeatmap HeatmapData
    """
    from agentic_quant.validation.signal_engine import run_strategy

    n = len(closes)
    if n < 200:
        raise ValueError(f"Insufficient data: need 200, got {n}")

    window_size = n // train_rolls
    n_entry = len(entry_param_grid)
    n_exit = len(exit_param_grid)

    # shape = (n_exit, train_rolls, n_entry) Log each exit parameter OOS Sharpe across windows for all entry params
    exit_window_entry_sharpes: list[np.ndarray] = []
    best_entry_per_exit = [-1] * n_exit

    for exit_idx, exit_params in enumerate(exit_param_grid):
        pt = exit_params["pt_mult"]
        sl = exit_params["sl_mult"]
        hold = exit_params["max_hold"]

        entry_sharpes = np.full((train_rolls, n_entry), np.nan)

        for w in range(train_rolls):
            ts = w * window_size
            train_end = ts + int(window_size * train_pct)
            vs = train_end + purge
            ve = ts + window_size

            if ve >= n:
                ve = n - 1
            if vs >= n:
                continue
            # At least 80 effective days per fold
            if ve - vs < 80 or train_end - ts < 80:
                continue

            train_s = slice(ts, train_end + 1)
            test_s = slice(vs, ve + 1)

            best_entry_idx = -1
            best_train_sr = float("-inf")

            for entry_idx, entry_params in enumerate(entry_param_grid):
                try:
                    train_result = run_strategy(
                        opens[train_s], highs[train_s], lows[train_s],
                        closes[train_s], volumes[train_s],
                        fast=entry_params["fast"], slow=entry_params["slow"],
                        enable_vpa=enable_vpa,
                        tb_pt_mult=pt, tb_sl_mult=sl, tb_max_hold=hold,
                    )
                    test_result = run_strategy(
                        opens[test_s], highs[test_s], lows[test_s],
                        closes[test_s], volumes[test_s],
                        fast=entry_params["fast"], slow=entry_params["slow"],
                        enable_vpa=enable_vpa,
                        tb_pt_mult=pt, tb_sl_mult=sl, tb_max_hold=hold,
                    )
                    entry_sharpes[w, entry_idx] = test_result.sharpe_ratio

                    if train_result.sharpe_ratio > best_train_sr:
                        best_train_sr = train_result.sharpe_ratio
                        best_entry_idx = entry_idx
                except Exception as exc:
                    logger.warning(
                        "Exit WFA: run_strategy failed for entry=%s exit=%s error=%s",
                        entry_params,
                        exit_params,
                        exc,
                    )
                    continue

            if best_entry_idx >= 0:
                best_entry_per_exit[exit_idx] = best_entry_idx

        exit_window_entry_sharpes.append(entry_sharpes)

    # Build heatmap: each exit parameter column = averaged OOS Sharpe vector across windows
    heatmap = np.full((n_entry, n_exit), np.nan)
    for exit_idx in range(n_exit):
        entry_sharpes = exit_window_entry_sharpes[exit_idx]
        heatmap[:, exit_idx] = np.nanmean(entry_sharpes, axis=0)

    # Select globally optimal exit parameter (highest mean OOS Sharpe across all entry parameters)
    exit_mean_sharpes = np.nanmean(heatmap, axis=0)
    best_exit_idx = int(np.nanargmax(exit_mean_sharpes))
    best_overall_sharpe = float(exit_mean_sharpes[best_exit_idx])

    entry_labels = [_format_entry_label(p) for p in entry_param_grid]
    exit_labels = [_format_exit_label(ep) for ep in exit_param_grid]

    return ExitWFAHeatmap(
        entry_labels=entry_labels,
        exit_labels=exit_labels,
        heatmap_matrix=heatmap,
        best_exit_idx=best_exit_idx,
        best_entry_per_exit=best_entry_per_exit,
        best_overall_sharpe=best_overall_sharpe,
    )


def generate_exit_wfa_heatmap_report(
    heatmap: ExitWFAHeatmap,
    entry_param_grid: list[dict[str, Any]],
    exit_param_grid: list[dict[str, float | int]],
    strategy_name: str = "",
    output_dir: str | Path | None = None,
) -> str:
    """
    GenerateExitParameter WFA HeatmapReport（Markdown + JSON）

    Args:
        heatmap: ExitWFAHeatmap HeatmapData
        entry_param_grid: EntryParameter Grid
        exit_param_grid: ExitParameter Grid
        strategy_name: Strategy Name
        output_dir: Output directory (None = do not write file)
    Returns:
        Markdown report text
    """
    from datetime import datetime

    hm = heatmap.heatmap_matrix
    n_entry, n_exit = hm.shape
    best_exit = heatmap.exit_labels[heatmap.best_exit_idx]
    best_exit_params = exit_param_grid[heatmap.best_exit_idx]

    # Find optimal exit parameter for each entry parameter
    best_exit_per_entry = [int(np.nanargmax(hm[e_idx, :])) for e_idx in range(n_entry)]

    lines = [
        f"# Exit WFA HeatmapReport{f' — {strategy_name}' if strategy_name else ''}",
        "",
        f"**GenerateTime**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. Optimal Exit Parameter",
        "",
        f"**Global Optimal**: {best_exit}",
        f"  - pt_mult (Take Profitmultiple): {best_exit_params['pt_mult']}",
        f"  - sl_mult (Stop Lossmultiple): {best_exit_params['sl_mult']}",
        f"  - max_hold (Max Holding Period): {best_exit_params['max_hold']}",
        f"  - Averaging OOS Sharpe: {heatmap.best_overall_sharpe:.4f}",
        "",
        "## 2. Heatmap Details",
        "",
        "### 2.1 DataPreview",
        "",
        f"- Entry Parameter Combinations: {n_entry} (fast×slow)",
        f"- Exit Parameter Combinations: {n_exit} (pt_mult×sl_mult×max_hold)",
        f"- Matrix Shape: {n_entry}×{n_exit}",
        "",
        "### 2.2 Entry x Exit Heatmap by Position Maturity×ExitHeatmap",
        "",
    ]

    # Display sub-tables grouped by max_hold
    hold_values = sorted(set(ep["max_hold"] for ep in exit_param_grid))
    for hold_val in hold_values:
        # Exit parameter index for current max_hold
        hold_indices = [i for i, ep in enumerate(exit_param_grid) if ep["max_hold"] == hold_val]
        hold_exit_labels = [heatmap.exit_labels[i] for i in hold_indices]
        hold_hm = hm[:, hold_indices]

        lines.extend([
            f"**max_hold = {hold_val}d**",
            "",
            "| EntryParameter↓ \\ ExitParameter→ | " + " | ".join(hold_exit_labels) + " |",
            "|" + "---|" * (len(hold_exit_labels) + 1),
        ])

        for e_idx in range(n_entry):
            row_vals = hold_hm[e_idx, :]
            row_str = " | ".join(
                f"{v:.3f}" if not np.isnan(v) else "N/A" for v in row_vals
            )
            lines.append(f"| {heatmap.entry_labels[e_idx]} | {row_str} |")

        lines.append("")

    # Optimal entry parameter for each column (exit parameter)
    lines.extend([
        "",
        "### 2.3 Optimal Entry Parameter per Exit Parameter",
        "",
        "| ExitParameter | optimalEntryParameter | OOS Sharpe |",
        "|----------|-------------|------------|",
    ])
    for exit_idx in range(n_exit):
        best_entry_idx = heatmap.best_entry_per_exit[exit_idx]
        if best_entry_idx >= 0:
            oos_sharpe = hm[best_entry_idx, exit_idx]
            lines.append(
                f"| {heatmap.exit_labels[exit_idx]} | {heatmap.entry_labels[best_entry_idx]} | {oos_sharpe:.4f} |"
            )
        else:
            lines.append(f"| {heatmap.exit_labels[exit_idx]} | N/A | N/A |")

    lines.extend([
        "",
        "### 2.4 Optimal Exit Parameter per Entry Parameter",
        "",
        "| EntryParameter | optimalExitParameter | OOS Sharpe |",
        "|----------|-------------|------------|",
    ])
    for e_idx in range(n_entry):
        be_idx = best_exit_per_entry[e_idx]
        oos_sharpe = hm[e_idx, be_idx]
        lines.append(
            f"| {heatmap.entry_labels[e_idx]} | {heatmap.exit_labels[be_idx]} | {oos_sharpe:.4f} |"
        )

    # Complete matrix (JSON format)
    lines.extend([
        "",
        "---",
        "",
        "## 3. Complete Matrix (Across All Exit Parameters)",
        "",
        "| EntryParameter↓ \\ ExitParameter→ | " + " | ".join(heatmap.exit_labels) + " |",
        "|" + "---|" * (n_exit + 1),
    ])
    for e_idx in range(n_entry):
        row_vals = hm[e_idx, :]
        row_str = " | ".join(
            f"{v:.3f}" if not np.isnan(v) else "N/A" for v in row_vals
        )
        lines.append(f"| {heatmap.entry_labels[e_idx]} | {row_str} |")

    lines.append("")

    md = "\n".join(lines)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_path = out / f"exit_wfa_heatmap_{ts}.md"
        md_path.write_text(md, encoding="utf-8")

        # JSON Output
        full_matrix: list[list[float]] = []
        for e_idx in range(n_entry):
            row = []
            for x_idx in range(n_exit):
                v = hm[e_idx, x_idx]
                row.append(float(v) if not np.isnan(v) else None)
            full_matrix.append(row)

        json_data = {
            "strategy": strategy_name,
            "timestamp": datetime.now().isoformat(),
            "entry_params": entry_param_grid,
            "exit_params": exit_param_grid,
            "entry_labels": heatmap.entry_labels,
            "exit_labels": heatmap.exit_labels,
            "heatmap_matrix": full_matrix,
            "best_exit_idx": heatmap.best_exit_idx,
            "best_entry_per_exit": heatmap.best_entry_per_exit,
            "best_exit_per_entry": best_exit_per_entry,
            "best_exit_params": best_exit_params,
            "best_overall_sharpe": heatmap.best_overall_sharpe,
        }
        json_path = out / f"exit_wfa_heatmap_{ts}.json"
        json_path.write_text(
            json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return md


def _regime_analysis(window_results: list[WFOSplitResult]) -> dict[str, float]:
    """
    Window OOS performance distribution analysis (not market regime classification, but bucketing by strategy performance).

    Note: bull/bear/high_vol/low_vol tags refer to the strategy relative performance in each window,
    not market regime classification. See WalkForwardAnalyzer._compute_regime_breakdown for true regime classification.
    """
    if len(window_results) < 4:
        return {}

    test_rets = [
        r.test_result.total_return if r.test_result else 0.0
        for r in window_results
    ]
    if not test_rets:
        return {}

    median_ret = statistics.median(test_rets)
    # Split into upper/lower halves by strategy performance
    upper_half = [r for r in window_results if (r.test_result and r.test_result.total_return > median_ret)]
    lower_half = [r for r in window_results if (r.test_result and r.test_result.total_return <= median_ret)]

    result: dict[str, float] = {}
    if upper_half:
        result["upper_half"] = statistics.mean(
            r.test_result.sharpe_ratio if r.test_result else 0.0 for r in upper_half
        )
    if lower_half:
        result["lower_half"] = statistics.mean(
            r.test_result.sharpe_ratio if r.test_result else 0.0 for r in lower_half
        )

    # Split into high/low by performance
    if len(test_rets) >= 4:
        hi_perf = [r for r in window_results if r.test_metric > statistics.mean([t.test_metric for t in window_results])]
        lo_perf = [r for r in window_results if r.test_metric <= statistics.mean([t.test_metric for t in window_results])]
        if hi_perf:
            result["high_perf"] = statistics.mean(
                r.test_result.sharpe_ratio if r.test_result else 0.0 for r in hi_perf
            )
        if lo_perf:
            result["low_perf"] = statistics.mean(
                r.test_result.sharpe_ratio if r.test_result else 0.0 for r in lo_perf
            )

    return result
