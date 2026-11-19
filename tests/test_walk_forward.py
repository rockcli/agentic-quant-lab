"""
Walk-Forward Optimization Test

Include:
- basic/foundation WFO partitioning and stability report
- Purge/Embargo Boundary Test
- Catastrophic veto rule test
- RegimeDecompositionTest
- WFA Report GenerationTest
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import numpy as np
import pytest

from agentic_quant.backtest.data.structures import Bar
from agentic_quant.backtest.strategy.base import Signal, Strategy
from agentic_quant.optimization.walk_forward import (
    WalkForwardAnalyzer,
    WalkForwardOptimizer,
    WFAReport,
    _compute_wfa_grade,
    _regime_analysis,
    generate_wfa_report,
)
from agentic_quant.strategy.examples import SmaCrossStrategy

# ──────────────────────── TestHelper Functions ────────────────────────


class AlwaysBuy(Strategy):
    """Test strategy: always buy and hold"""

    def __init__(self, threshold: float = 0.0):
        super().__init__("AlwaysBuy")
        self.threshold = threshold
        self._done = False

    def on_bar(self, bar: Bar) -> Signal | None:
        if not self._done and bar.close > self.threshold:
            self._done = True
            return Signal(bar.timestamp, "buy", 1.0, bar.close)
        return None


def _make_trending_bars(start: datetime, n: int, trend: float = 0.5) -> list[Bar]:
    """Generate bar data with clear trends so the strategy passes WFO."""
    np.random.seed(42)
    prices = 100.0 + np.cumsum(np.ones(n) * trend + np.random.randn(n) * 0.1)
    return [
        Bar(
            timestamp=start + timedelta(hours=i),
            open=float(prices[i]),
            high=float(prices[i] + abs(np.random.randn()) * 0.3 + 0.1),
            low=float(prices[i] - abs(np.random.randn()) * 0.3 - 0.1),
            close=float(prices[i]),
            volume=float(np.random.randint(1000, 10000)),
        )
        for i in range(n)
    ]


def _make_mixed_regime_bars(start: datetime, n: int) -> list[Bar]:
    """Generate bar data with different trends across long segments, used for WFO."""
    np.random.seed(42)
    segment = n // 5
    prices = []
    for i in range(5):
        trend = 0.3 if i % 2 == 0 else -0.2
        seg = np.cumsum(np.ones(segment) * trend + np.random.randn(segment) * 0.15)
        prices.extend(seg)
    prices = np.array(prices) + 100.0
    return [
        Bar(
            timestamp=start + timedelta(hours=i),
            open=float(prices[i]),
            high=float(prices[i] + abs(np.random.randn()) * 0.3 + 0.1),
            low=float(prices[i] - abs(np.random.randn()) * 0.3 - 0.1),
            close=float(prices[i]),
            volume=float(np.random.randint(1000, 10000)),
        )
        for i in range(n)
    ]


# ──────────────────────── Original unittest compatibleTest ────────────────────────
# (Fixed to be compatible with current WFAReport dataclass return type)


class WalkForwardTest(unittest.TestCase):
    def _make_bars(self, start: datetime, n: int) -> list[Bar]:
        return [
            Bar(
                timestamp=start.replace(day=start.day + i),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0 + i,
                volume=1000.0,
            )
            for i in range(n)
        ]

    def test_splits_count(self):
        """Validate WFO split count is correct"""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 1)
        wfo = WalkForwardOptimizer(start, end, train_rolls=3, train_pct=0.6)
        splits = wfo.splits()
        self.assertEqual(len(splits), 3)
        for s in splits:
            self.assertTrue(s.train_start < s.train_end <= s.test_start < s.test_end)

    def test_splits_purge_gap(self):
        """Validate that purge interval in split exists and is correct"""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 1)
        wfo = WalkForwardOptimizer(start, end, train_rolls=2, train_pct=0.6, purge_days=5)
        splits = wfo.splits()
        for s in splits:
            self.assertIsNotNone(s.purge_start)
            self.assertIsNotNone(s.purge_end)
            purge_duration = (s.purge_end - s.purge_start).days
            self.assertEqual(purge_duration, 5, f"purge gap should be 5 days, got {purge_duration}")
            # purge is after train end and before test start
            self.assertGreaterEqual(s.purge_start, s.train_end)
            self.assertLessEqual(s.purge_end, s.test_start)

    def test_stability_report_empty(self):
        """Short result list should return short dictionary"""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 1)
        wfo = WalkForwardOptimizer(start, end, train_rolls=3)
        report = wfo.stability_report([])
        self.assertEqual(report, {})

    def test_n_rolls_at_least_1(self):
        """At Least 1 itemsRollingWindow"""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 1)
        wfo = WalkForwardOptimizer(start, end, train_rolls=1)
        splits = wfo.splits()
        self.assertGreaterEqual(len(splits), 1)


class WalkForwardAnalyzerTest(unittest.TestCase):
    """Access WFAReport version using attribute"""

    def test_analyzer_empty_data(self):
        """Short bars BackDefaultReport"""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 2)
        analyzer = WalkForwardAnalyzer(start, end, train_rolls=2)
        bars: list[Bar] = []
        grid = [{"fast": 5, "slow": 10}]
        report = analyzer.analyze(SmaCrossStrategy, bars, grid)
        self.assertEqual(report.avg_oos_sharpe, 0.0)
        self.assertEqual(report.window_results, [])

    def test_analyzer_produces_results(self):
        """Generate window result when data available"""
        start = datetime(2024, 1, 1)
        bars = _make_mixed_regime_bars(start, 500)
        end = bars[-1].timestamp
        # purge_days=0 because data spans ~21 days, default purge_days=5 would exceed window length
        analyzer = WalkForwardAnalyzer(start, end, train_rolls=3, train_pct=0.6, purge_days=0)
        grid = [
            {"fast": 5, "slow": 20},
            {"fast": 10, "slow": 30},
            {"fast": 15, "slow": 40},
        ]
        report = analyzer.analyze(SmaCrossStrategy, bars, grid)
        self.assertGreater(len(report.window_results), 0)
        self.assertIn("avg_oos_sharpe", dir(report))
        self.assertIn("avg_oos_max_dd", dir(report))
        self.assertIn("param_stability", dir(report))
        self.assertIn("robustness_ratio", dir(report))

    def test_param_stability_score_range(self):
        """Parameter stability score should be in [0, 1] range."""
        start = datetime(2024, 1, 1)
        bars = _make_mixed_regime_bars(start, 500)
        end = bars[-1].timestamp
        analyzer = WalkForwardAnalyzer(start, end, train_rolls=3, train_pct=0.6)
        grid = [
            {"fast": 5, "slow": 20},
            {"fast": 10, "slow": 30},
            {"fast": 15, "slow": 40},
        ]
        report = analyzer.analyze(SmaCrossStrategy, bars, grid)
        stability = report.param_stability
        self.assertGreaterEqual(stability, 0.0)
        self.assertLessEqual(stability, 1.0)

    def test_wfa_report_grade_field(self):
        """Report includes rating when results available"""
        start = datetime(2024, 1, 1)
        bars = _make_mixed_regime_bars(start, 500)
        end = bars[-1].timestamp
        analyzer = WalkForwardAnalyzer(start, end, train_rolls=3, train_pct=0.6)
        grid = [
            {"fast": 5, "slow": 20},
            {"fast": 10, "slow": 30},
        ]
        report = analyzer.analyze(SmaCrossStrategy, bars, grid)
        self.assertIsInstance(report.grade, str)

    def test_wfa_report_contains_catastrophic_flag(self):
        """Report includes catastrophic_veto flag"""
        start = datetime(2024, 1, 1)
        bars = _make_mixed_regime_bars(start, 500)
        end = bars[-1].timestamp
        analyzer = WalkForwardAnalyzer(start, end, train_rolls=3, train_pct=0.6)
        grid = [{"fast": 5, "slow": 20}]
        report = analyzer.analyze(SmaCrossStrategy, bars, grid)
        self.assertIsInstance(report.catastrophic_veto, bool)


# ═══════════════════════ Supplementary Test: Purge/Embargo Boundary ═══════════════════════


class TestPurgeEmbargoBoundary:
    """Purge/Embargo Boundary Test"""

    def test_purge_days_zero(self):
        """Can still generate effective split with purge_days=0"""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 1)
        wfo = WalkForwardOptimizer(start, end, train_rolls=3, train_pct=0.6, purge_days=0)
        splits = wfo.splits()
        assert len(splits) == 3
        for s in splits:
            # purge_start == purge_end when purge_days=0
            assert s.purge_start == s.purge_end

    def test_purge_days_large_value(self):
        """Large purge_days still generates effective split"""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 1)
        wfo = WalkForwardOptimizer(start, end, train_rolls=3, train_pct=0.6, purge_days=30)
        splits = wfo.splits()
        for s in splits:
            purge_seconds = (s.purge_end - s.purge_start).total_seconds()
            assert purge_seconds >= 30 * 86400 - 1  # may have float precision error

    def test_purge_gap_prevents_overlap(self):
        """Purge interval ensures training set and test set do not overlap"""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 1)
        wfo = WalkForwardOptimizer(start, end, train_rolls=4, train_pct=0.6, purge_days=5)
        splits = wfo.splits()
        for s in splits:
            assert s.train_end <= s.purge_start
            assert s.purge_end <= s.test_start
            assert s.test_start > s.train_end  # strictly greater than

    def test_train_pct_invalid_low(self):
        """train_pct < 0.5 raises exception"""
        with pytest.raises(ValueError):
            WalkForwardOptimizer(
                datetime(2024, 1, 1), datetime(2024, 6, 1),
                train_pct=0.4,
            )

    def test_train_pct_invalid_high(self):
        """train_pct >= 0.95 raises exception"""
        with pytest.raises(ValueError):
            WalkForwardOptimizer(
                datetime(2024, 1, 1), datetime(2024, 6, 1),
                train_pct=0.96,
            )

    def test_end_before_start_raises(self):
        """Raise exception if end time is before start time"""
        with pytest.raises(ValueError):
            WalkForwardOptimizer(
                datetime(2024, 6, 1), datetime(2024, 1, 1),
            )

    def test_split_boundary_values(self):
        """Split boundary validation: train_start <= train_end <= test_start <= test_end"""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 1)
        wfo = WalkForwardOptimizer(start, end, train_rolls=5, train_pct=0.6, purge_days=5)
        splits = wfo.splits()
        for i, s in enumerate(splits):
            assert s.train_start >= start, f"split {i}: train_start before total_start"
            assert s.test_end <= end, f"split {i}: test_end after total_end"
            assert s.train_end <= s.test_start, f"split {i}: train_end after test_start"

    def test_multiple_purge_values(self):
        """Different purge_days values produce different intervals"""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 1)
        wfo1 = WalkForwardOptimizer(start, end, train_rolls=2, train_pct=0.6, purge_days=5)
        wfo2 = WalkForwardOptimizer(start, end, train_rolls=2, train_pct=0.6, purge_days=10)
        splits1 = wfo1.splits()
        splits2 = wfo2.splits()
        # Validate purge intervals differ
        d1 = (splits1[0].purge_end - splits1[0].purge_start).days
        d2 = (splits2[0].purge_end - splits2[0].purge_start).days
        assert d1 == 5
        assert d2 == 10


# ═══════════════════════ Supplementary Test: Catastrophic Veto ═══════════════════════


class TestCatastrophicVeto:
    """Catastrophic veto rule test"""

    def test_veto_when_drawdown_exceeds_threshold(self):
        """Drawdown exceeding threshold triggers veto"""
        grade = _compute_wfa_grade(
            majority_pass=True,
            catastrophic_veto=True,
            avg_oos_sharpe=0.8,
            robustness_ratio=0.9,
            param_stability=0.8,
        )
        assert "D" in grade, f"Catastrophic veto should get D rating, but got {grade}"

    def test_no_veto_when_drawdown_within_threshold(self):
        """Drawdown not exceeding threshold does not trigger veto"""
        grade = _compute_wfa_grade(
            majority_pass=True,
            catastrophic_veto=False,
            avg_oos_sharpe=0.8,
            robustness_ratio=0.9,
            param_stability=0.8,
        )
        assert "A" in grade or "B" in grade

    def test_veto_overrides_other_metrics(self):
        """Veto takes priority over other metrics"""
        grade = _compute_wfa_grade(
            majority_pass=True,
            catastrophic_veto=True,
            avg_oos_sharpe=2.0,  # even with high Sharpe
            robustness_ratio=1.0,
            param_stability=1.0,
        )
        assert "D" in grade, "Veto status should cover other high metrics"

    def test_majority_fail_results_in_grade_c(self):
        """Majority failed produces C rating"""
        grade = _compute_wfa_grade(
            majority_pass=False,
            catastrophic_veto=False,
            avg_oos_sharpe=0.8,
            robustness_ratio=0.9,
            param_stability=0.8,
        )
        assert "C" in grade

    def test_low_sharpe_results_in_grade_c(self):
        """Insufficient OOS Sharpe produces C rating"""
        grade = _compute_wfa_grade(
            majority_pass=True,
            catastrophic_veto=False,
            avg_oos_sharpe=0.3,  # < 0.5
            robustness_ratio=0.9,
            param_stability=0.8,
        )
        assert "C" in grade

    def test_grade_a_criteria(self):
        """A level condition: Sharpe >= 1.0, robustness >= 0.8, stability >= 0.7"""
        grade = _compute_wfa_grade(
            majority_pass=True,
            catastrophic_veto=False,
            avg_oos_sharpe=1.2,
            robustness_ratio=0.85,
            param_stability=0.75,
        )
        assert "A" in grade

    def test_grade_b_criteria(self):
        """B level condition: Sharpe >= 0.7, robustness >= 0.7, stability >= 0.5"""
        grade = _compute_wfa_grade(
            majority_pass=True,
            catastrophic_veto=False,
            avg_oos_sharpe=0.8,
            robustness_ratio=0.75,
            param_stability=0.6,
        )
        assert "B" in grade

    def test_grade_b_minus_fallback(self):
        """All other passed cases default to B-"""
        grade = _compute_wfa_grade(
            majority_pass=True,
            catastrophic_veto=False,
            avg_oos_sharpe=0.6,
            robustness_ratio=0.5,
            param_stability=0.3,
        )
        assert "B-" in grade


# ═══════════════════════ Supplementary Test: Regime Decomposition ═══════════════════════


class TestRegimeAnalysis:
    """RegimeDecompositionTest"""

    def test_insufficient_windows_returns_empty(self):
        """Returns short when fewer than 4 windows"""
        result = _regime_analysis([])
        assert result == {}

    def test_empty_result_definition(self):
        """Directly test _regime_analysis return type"""

        # < 4 windows
        splits = [None] * 3
        result = _regime_analysis(splits)
        assert result == {}


# ═══════════════════════ Supplementary Test: Parameter Stability ═══════════════════════


class TestParamStabilityComputation:
    """ParameterStabilityCalculateTest"""

    def test_empty_history_returns_zero(self):
        """ShortHistoricalBack 0"""
        stability = WalkForwardAnalyzer._compute_param_stability([])
        assert stability == 0.0

    def test_single_param_set_returns_zero(self):
        """Returns 0 with only one parameter group"""
        stability = WalkForwardAnalyzer._compute_param_stability([
            {"fast": 10, "slow": 20},
        ])
        assert stability == 0.0

    def test_stable_params(self):
        """Stable parameters should get high score (close to 1.0)"""
        stability = WalkForwardAnalyzer._compute_param_stability([
            {"fast": 10, "slow": 20},
            {"fast": 10, "slow": 21},
            {"fast": 11, "slow": 20},
        ])
        assert stability > 0.5, f"stable parameter should score high, but got {stability}"

    def test_unstable_params(self):
        """High parameter volatility should get low score"""
        stability = WalkForwardAnalyzer._compute_param_stability([
            {"fast": 5, "slow": 20},
            {"fast": 50, "slow": 100},
            {"fast": 100, "slow": 200},
        ])
        assert stability < 0.5, f"unstable parameter should score low, but got {stability}"

    def test_non_numeric_params_ignored(self):
        """Non-value parameters should be skipped"""
        stability = WalkForwardAnalyzer._compute_param_stability([
            {"fast": 10, "mode": "conservative"},
            {"fast": 12, "mode": "aggressive"},
        ])
        # mode is skipped, fast is relatively stable
        assert stability > 0.0

    def test_all_non_numeric_returns_zero(self):
        """All non-value parameters return 0"""
        stability = WalkForwardAnalyzer._compute_param_stability([
            {"mode": "a"},
            {"mode": "b"},
        ])
        assert stability == 0.0

    def test_mixed_numeric_types(self):
        """Integer and float parameter blending"""
        stability = WalkForwardAnalyzer._compute_param_stability([
            {"fast": 10, "slow": 20.0},
            {"fast": 11, "slow": 21.0},
            {"fast": 10, "slow": 22.0},
        ])
        assert isinstance(stability, float)
        assert 0.0 <= stability <= 1.0


# ═══════════════════════ Supplementary Test: Report Generation ═══════════════════════


class TestGenerateWfaReport:
    """WFA Report GenerationTest"""

    def _make_minimal_report(self) -> WFAReport:
        return WFAReport(
            strategy_name="TestStrategy",
            total_windows=3,
            passed_windows=2,
            oos_positive_rate=0.67,
            majority_pass=True,
            catastrophic_veto=False,
            catastrophic_details=[],
            avg_oos_sharpe=0.85,
            avg_oos_max_dd=0.15,
            avg_oos_total_return=0.12,
            param_stability=0.75,
            robustness_ratio=0.67,
            metric_decay=0.2,
            grade="B",
            window_results=[
                {
                    "window": 1,
                    "train_period": "2024-01-01 → 2024-02-01",
                    "test_period": "2024-02-06 → 2024-03-01",
                    "best_params": {"fast": 10, "slow": 20},
                    "train_sharpe": 1.5,
                    "test_sharpe": 0.9,
                    "test_return": 0.05,
                    "test_max_dd": 0.1,
                },
            ],
        )

    def test_generate_basic_report(self):
        """GeneratebasicReport"""
        report = self._make_minimal_report()
        output = generate_wfa_report(report)
        assert isinstance(output, str)
        assert "TestStrategy" in output
        assert "WFA" in output

    def test_generate_report_with_catastrophic_details(self):
        """Report includes catastrophic veto details"""
        report = self._make_minimal_report()
        report.catastrophic_veto = True
        report.catastrophic_details = ["Window1: Max Drawdown 35% > Threshold 30%"]
        output = generate_wfa_report(report)
        assert "Catastrophic Veto" in output or "Window1" in output

    def test_generate_report_with_output_dir(self, tmp_path):
        """Specifying output_dir should generate file"""
        report = self._make_minimal_report()
        generate_wfa_report(report, output_dir=str(tmp_path))
        files = list(tmp_path.iterdir())
        md_files = [f for f in files if f.suffix == ".md"]
        json_files = [f for f in files if f.suffix == ".json"]
        assert len(md_files) >= 1
        assert len(json_files) >= 1

    def test_generate_report_with_regime_analysis(self):
        """Report includes regime analysis"""
        report = self._make_minimal_report()
        report.regime_analysis = {"high_vol": 0.5, "low_vol": 1.2, "bull": 1.0, "bear": 0.3}
        output = generate_wfa_report(report)
        assert "High Volatility Period" in output or "RegimeAnalyze" in output

    def test_report_with_summary_lines(self):
        """Report includes summary line"""
        report = self._make_minimal_report()
        report.summary_lines = [
            "## WFA Acceptance Conclusion: B",
            "- Total windows: 3",
        ]
        output = generate_wfa_report(report)
        assert "Acceptance" in output or "B" in output

    def test_empty_report(self):
        """ShortReport Generation"""
        report = WFAReport(
            strategy_name="",
            total_windows=0, passed_windows=0,
            oos_positive_rate=0.0, majority_pass=False,
            catastrophic_veto=False, catastrophic_details=[],
            avg_oos_sharpe=0.0, avg_oos_max_dd=0.0,
            avg_oos_total_return=0.0, param_stability=0.0,
            robustness_ratio=0.0, metric_decay=0.0, grade="NoneData",
        )
        output = generate_wfa_report(report)
        assert isinstance(output, str)


if __name__ == "__main__":
    unittest.main()
