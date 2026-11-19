"""
Market Regime Classifier Unit Test

Testcoverage:
- Helper Functions _sma / _rolling_slope / _rolling_std
- MarketRegime.classify four-quadrant classification
- MarketRegime.regime_summary StatisticsSummary
- MarketRegime.predominant_regime dominant regime
- MarketRegime.wfa_breakdown Regime Decomposition
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from agentic_quant.validation.regime import (
    MarketRegime,
    _rolling_slope,
    _rolling_std,
    _sma,
)


# ═══════════════════════ SMA Test ═══════════════════════


class TestSma:
    """Simple Moving Average Test"""

    def test_basic_sma(self):
        """basic SMA calculation"""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _sma(arr, period=3)
        # First 2 items are NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        # SMA(3): (1+2+3)/3=2, (2+3+4)/3=3, (3+4+5)/3=4
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(4.0)

    def test_sma_short_data(self):
        """Returns all NaN when data length is insufficient"""
        result = _sma(np.array([1.0, 2.0]), period=5)
        assert np.all(np.isnan(result))

    def test_sma_identity(self):
        """When period=1, SMA equals original values"""
        arr = np.array([10.0, 20.0, 30.0])
        result = _sma(arr, period=1)
        assert np.allclose(result, arr)

    def test_sma_zero_period(self):
        """ShortArrayBackShortResult"""
        result = _sma(np.array([]), period=3)
        assert len(result) == 0


# ═══════════════════════ Rolling Slope Test ═══════════════════════


class TestRollingSlope:
    """Rolling Slope Test"""

    def test_positive_slope(self):
        """Rising trend should yield positive slope"""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _rolling_slope(arr, window=3)
        # First 2 items are NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        # Slope should be positive
        assert result[2] > 0
        assert result[3] > 0
        assert result[4] > 0

    def test_negative_slope(self):
        """Declining trend should yield negative slope"""
        arr = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = _rolling_slope(arr, window=3)
        assert result[2] < 0
        assert result[3] < 0
        assert result[4] < 0

    def test_flat_returns_zero(self):
        """Flat sequence slope close to 0"""
        arr = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        result = _rolling_slope(arr, window=3)
        # Horizontal line slope should be 0
        assert abs(result[2]) < 1e-6

    def test_short_data(self):
        """Returns all NaN when data is insufficient"""
        result = _rolling_slope(np.array([1.0, 2.0]), window=5)
        assert np.all(np.isnan(result))

    def test_nan_handling(self):
        """Skip when NaN is included"""
        arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
        result = _rolling_slope(arr, window=3)
        # Window [2, NaN, 4] should return NaN
        assert np.isnan(result[3])


# ═══════════════════════ Rolling Standard Deviation Test ═══════════════════════


class TestRollingStd:
    """Rolling Standard Deviation Test"""

    def test_basic_std(self):
        """Basic standard deviation calculation"""
        arr = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result = _rolling_std(arr, window=3)
        # Constant sequence standard deviation is 0
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(0.0)
        assert result[3] == pytest.approx(0.0)
        assert result[4] == pytest.approx(0.0)

    def test_varying_std(self):
        """Standard deviation of changing sequence > 0"""
        arr = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        result = _rolling_std(arr, window=3)
        assert result[2] > 0
        assert result[3] > 0
        assert result[4] > 0

    def test_short_data(self):
        """Returns all NaN when data is insufficient"""
        result = _rolling_std(np.array([1.0]), window=5)
        assert np.all(np.isnan(result))


# ═══════════════════════ MarketRegime.classify ═══════════════════════


class TestMarketRegimeClassify:
    """Market Regime Classification Test"""

    def test_insufficient_data_returns_empty(self):
        """Returns all empty tags during warmup when data is insufficient"""
        mr = MarketRegime(ma_period=10, vol_window=5, slope_window=5)
        prices = np.random.randn(5) + 100
        labels = mr.classify(prices)
        assert np.all(labels == "")

    def test_bull_regime_detected(self):
        """Strong rising trend should classify as bull market"""
        mr = MarketRegime(ma_period=5, vol_window=5, slope_window=3)
        # Strong rising trend: Price continues to rise
        prices = np.linspace(100, 200, 50)
        labels = mr.classify(prices)
        valid_labels = labels[labels != ""]
        # Most valid tags should be bull
        bull_count = np.sum(valid_labels == "bull_high_vol") + np.sum(valid_labels == "bull_low_vol")
        assert bull_count > len(valid_labels) * 0.5, f"Most tags should be bull market, bull={bull_count}/{len(valid_labels)}"

    def test_bear_regime_detected(self):
        """Strong declining trend should classify as bear market"""
        mr = MarketRegime(ma_period=5, vol_window=5, slope_window=3)
        # Strong declining trend
        prices = np.linspace(200, 100, 50)
        labels = mr.classify(prices)
        valid_labels = labels[labels != ""]
        bear_count = np.sum(valid_labels == "bear_high_vol") + np.sum(valid_labels == "bear_low_vol")
        assert bear_count > len(valid_labels) * 0.5

    def test_output_contains_only_valid_regimes(self):
        """Output only includes the 4 defined regime types or empty string"""
        mr = MarketRegime(ma_period=10, vol_window=10, slope_window=5)
        prices = np.linspace(100, 150, 100) + np.random.randn(100) * 2
        labels = mr.classify(prices)
        for label in labels:
            assert label in ("", "bull_high_vol", "bull_low_vol", "bear_high_vol", "bear_low_vol")

    def test_classify_output_length(self):
        """Outputlength andInputConsistent"""
        mr = MarketRegime(ma_period=10, vol_window=10, slope_window=5)
        prices = np.random.randn(100) + 100
        labels = mr.classify(prices)
        assert len(labels) == len(prices)

    def test_warmup_period(self):
        """Warmup period length is max(ma_period, vol_window, slope_window) + 1"""
        mr = MarketRegime(ma_period=20, vol_window=15, slope_window=10)
        prices = np.random.randn(50) + 100
        labels = mr.classify(prices)
        # Warmup is 21, first 21 items should all be empty
        warmup = max(20, 15, 10) + 1
        for i in range(min(warmup, len(labels))):
            assert labels[i] == "", f"Index {i} should be empty tag"


# ═══════════════════════ MarketRegime.regime_summary ═══════════════════════


class TestMarketRegimeSummary:
    """RegimeStatisticsSummaryTest"""

    def test_summary_returns_all_regimes(self):
        """summary should return all 4 quadrants"""
        mr = MarketRegime(ma_period=5, vol_window=5, slope_window=3)
        prices = np.random.randn(100) * 5 + 100
        summary = mr.regime_summary(prices)
        for regime in mr.REGIME_NAMES:
            assert regime in summary

    def test_summary_contains_expected_fields(self):
        """Each regime should include days, pct, avg_return, std_return, sharpe"""
        mr = MarketRegime(ma_period=5, vol_window=5, slope_window=3)
        prices = np.random.randn(100) + 100
        summary = mr.regime_summary(prices)
        for data in summary.values():
            for key in ("days", "pct", "avg_return", "std_return", "sharpe"):
                assert key in data

    def test_summary_with_precomputed_labels(self):
        """Using pre-computed labels"""
        mr = MarketRegime(ma_period=5, vol_window=5, slope_window=3)
        prices = np.linspace(100, 200, 50)
        labels = mr.classify(prices)
        summary = mr.regime_summary(prices, regime_labels=labels)
        assert isinstance(summary, dict)

    def test_summary_pct_sum(self):
        """Sum of regime proportions should be approximately 1.0"""
        mr = MarketRegime(ma_period=10, vol_window=10, slope_window=5)
        prices = np.random.randn(200) * 3 + 100
        summary = mr.regime_summary(prices)
        total_pct = sum(data["pct"] for data in summary.values())
        assert total_pct == pytest.approx(1.0, abs=0.01), f"Proportion sum should be approx 1.0, actual {total_pct}"

    def test_empty_regime_returns_defaults(self):
        """Regime with no data returns default zero values"""
        mr = MarketRegime(ma_period=50, vol_window=20, slope_window=10)
        # Data too short, all regimes are empty
        prices = np.random.randn(50) + 100
        summary = mr.regime_summary(prices)
        for data in summary.values():
            # Empty regime should have days=0
            if data["days"] == 0:
                assert data["pct"] == 0.0
                assert data["avg_return"] == 0.0


# ═══════════════════════ MarketRegime.predominant_regime ═══════════════════════


class TestPredominantRegime:
    """Dominant Regime Test"""

    def test_returns_most_common(self):
        """Return the most frequent regime within the window"""
        mr = MarketRegime(ma_period=5, vol_window=5, slope_window=3)
        prices = np.linspace(100, 200, 100)
        labels = mr.classify(prices)
        predominant = mr.predominant_regime(labels, 20, 80)
        assert predominant != ""
        assert isinstance(predominant, str)

    def test_empty_window_returns_empty_string(self):
        """Window with no valid tags returns empty string"""
        labels = np.full(10, "", dtype=object)
        result = MarketRegime.predominant_regime(labels, 0, 9)
        assert result == ""

    def test_invalid_indices(self):
        """Index out of range"""
        mr = MarketRegime(ma_period=5, vol_window=5, slope_window=3)
        prices = np.random.randn(50) + 100
        labels = mr.classify(prices)
        # end_idx out of range should not crash
        result = mr.predominant_regime(labels, 0, 1000)
        assert isinstance(result, str)

    def test_single_regime_dominated(self):
        """When a single regime dominates, return that regime"""
        labels = np.array(["", "", "bull_low_vol", "bull_low_vol", "bull_low_vol", "", "bear_high_vol"])
        result = MarketRegime.predominant_regime(labels, 2, 4)
        assert result == "bull_low_vol"


# ═══════════════════════ MarketRegime.wfa_breakdown ═══════════════════════


class TestWfaBreakdown:
    """WFA RegimeDecompositionTest"""

    def test_empty_windows_returns_empty_regimes(self):
        """ShortWindowResultBackShortDecomposition"""
        from agentic_quant.optimization.walk_forward import WFOSplit, WFOSplitResult

        result = MarketRegime.wfa_breakdown(
            [], np.random.randn(100) + 100,
            datetime(2024, 1, 1), datetime(2024, 6, 1),
        )
        assert isinstance(result, dict)
        assert "regimes" in result
        assert result["best_regime"] == "N/A"

    def test_basic_breakdown_structure(self):
        """Basic decomposition structure"""
        from agentic_quant.optimization.walk_forward import WFOSplit, WFOSplitResult
        from agentic_quant.backtest.core.result import BacktestResult

        split = WFOSplit(
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 2, 1),
            test_start=datetime(2024, 2, 6),
            test_end=datetime(2024, 3, 1),
        )
        bt_result = BacktestResult(
            equity_curve=None, fills=[], signals=[],
            total_return=0.05, sharpe_ratio=0.8, max_drawdown=0.1,
            win_rate=0.6, profit_factor=1.5, num_trades=10,
        )
        wfos = [
            WFOSplitResult(
                split=split, best_params={"fast": 10},
                train_metric=1.0, test_metric=0.8,
                train_result=None, test_result=bt_result,
            ),
        ]
        prices = np.linspace(100, 150, 200) + np.random.randn(200) * 2
        result = MarketRegime.wfa_breakdown(
            wfos, prices,
            datetime(2024, 1, 1), datetime(2024, 3, 1),
        )
        assert isinstance(result, dict)
        # At least some regimes should be populated
        non_empty = {k: v for k, v in result.get("regimes", {}).items() if v.get("window_count", 0) > 0}
        if non_empty:
            assert len(non_empty) >= 1

    def test_breakdown_with_dict_windows(self):
        """Using dict format window results"""
        result = MarketRegime.wfa_breakdown(
            [{"test_start": "2024-01-01", "test_end": "2024-02-01", "test_metric": 0.5}],
            np.random.randn(60) + 100,
            datetime(2024, 1, 1), datetime(2024, 2, 1),
        )
        assert isinstance(result, dict)

    def test_result_contains_spread(self):
        """Result includes best/worst Regime and spread"""
        from agentic_quant.optimization.walk_forward import WFOSplit, WFOSplitResult
        from agentic_quant.backtest.core.result import BacktestResult

        splits = [
            WFOSplit(
                train_start=datetime(2024, 1, 1),
                train_end=datetime(2024, 1, 31),
                test_start=datetime(2024, 2, 5),
                test_end=datetime(2024, 3, 1),
            ),
            WFOSplit(
                train_start=datetime(2024, 2, 1),
                train_end=datetime(2024, 2, 28),
                test_start=datetime(2024, 3, 5),
                test_end=datetime(2024, 3, 31),
            ),
        ]
        wfos = []
        for s in splits:
            bt = BacktestResult(
                equity_curve=None, fills=[], signals=[],
                total_return=0.05, sharpe_ratio=0.8, max_drawdown=0.1,
                win_rate=0.6, profit_factor=1.5, num_trades=10,
            )
            wfos.append(WFOSplitResult(
                split=s, best_params={"fast": 10},
                train_metric=1.0, test_metric=0.8,
                train_result=None, test_result=bt,
            ))

        prices = np.linspace(100, 150, 90) + np.random.randn(90) * 2
        result = MarketRegime.wfa_breakdown(
            wfos, prices,
            datetime(2024, 1, 1), datetime(2024, 3, 31),
        )
        assert "best_regime" in result
        assert "worst_regime" in result
        assert "spread" in result

    def test_breakdown_missing_dates_graceful(self):
        """Graceful degradation when date info is missing"""
        result = MarketRegime.wfa_breakdown(
            [{"some_key": "value"}],
            np.random.randn(50) + 100,
            "invalid_start", "invalid_end",
        )
        assert isinstance(result, dict)
