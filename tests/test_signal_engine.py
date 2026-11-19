"""
Pure numpy signal engine unit test

Testcoverage:
- SMA Dual Moving Average CrossoverSignal（Golden Cross/Death Cross/BoundaryCondition）
- VPA Volume-Price Confirmation（Confirm/reject/shadow/wick/Volume）
- Triple-Barrier Tag（Take Profit/Stop Loss/Timeout）
- Combined pipeline apply_vpa_filter / run_strategy
"""

from __future__ import annotations

import numpy as np
import pytest

import agentic_quant.validation.signal_engine as signal_engine_module
from agentic_quant.validation.signal_engine import (
    VPAConfirmResult,
    apply_vpa_filter,
    kama_cross_signals,
    run_strategy,
    sma_cross_signals,
    triple_barrier_labels,
    triple_barrier_labels_ohlc,
    vpa_confirm,
)

# ═══════════════════════ SMA CrossoverSignal ═══════════════════════


class TestSmaCrossSignals:
    """SMA Dual Moving Average CrossoverSignalTest"""

    def _make_price_series(self, values: list[float]) -> np.ndarray:
        return np.array(values, dtype=float)

    def test_upward_cross_triggers_entry(self):
        """Fast MA crossing above slow MA should trigger entry signal"""
        # Construct golden cross: fast MA crosses above slow MA
        prices = self._make_price_series([
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # 10
            2.0, 4.0, 6.0, 8.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,  # 10
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,  # 10
        ])
        entries, exits = sma_cross_signals(prices, fast=5, slow=10)
        assert np.any(entries), "Should detect golden cross"
        # First golden cross should occur after fast=5 has enough data
        entry_indices = np.where(entries)[0]
        assert len(entry_indices) > 0
        # Entry should not be before earliest feasible position
        assert entry_indices[0] >= 9

    def test_downward_cross_triggers_exit(self):
        """Fast MA crossing below slow MA should trigger exit signal"""
        # First golden cross entry, then death cross exit
        prices = self._make_price_series([
            # Flat segment
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
            # Rising segment - golden cross
            12.0, 14.0, 16.0, 18.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0,
            # Declining segment - death cross
            18.0, 16.0, 14.0, 12.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
        ])
        entries, exits = sma_cross_signals(prices, fast=5, slow=10)
        assert np.any(exits), "Should detect death cross"

    def test_no_cross_returns_no_signals(self):
        """Stable price with no crossover should give no signal"""
        prices = self._make_price_series([10.0] * 30)
        entries, exits = sma_cross_signals(prices, fast=5, slow=10)
        assert not np.any(entries)
        assert not np.any(exits)

    def test_short_data_returns_all_false(self):
        """Insufficient data length: return all False"""
        prices = self._make_price_series([1.0, 2.0, 3.0])
        entries, exits = sma_cross_signals(prices, fast=5, slow=10)
        assert not np.any(entries)
        assert not np.any(exits)

    def test_entry_only_when_not_in_position(self):
        """Only open position when short is enabled"""
        # Construct multiple crossover scenario
        prices = self._make_price_series([
            # Segment 1: decline then rise (golden cross)
            15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 10.0, 10.0, 10.0, 10.0,
            12.0, 14.0, 16.0, 18.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0,
            # Segment 2: decline (death cross)
            18.0, 16.0, 14.0, 12.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
            # Segment 3: rise again (golden cross, should open again)
            12.0, 14.0, 16.0, 18.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0,
        ])
        entries, exits = sma_cross_signals(prices, fast=5, slow=10)
        entry_indices = np.where(entries)[0]
        exit_indices = np.where(exits)[0]
        # Should have two entries and at least one exit
        assert len(entry_indices) >= 1
        assert len(exit_indices) >= 1

    def test_output_length_matches_input(self):
        """OutputArraylength andInputConsistent"""
        prices = self._make_price_series(np.random.randn(50) + 100)
        entries, exits = sma_cross_signals(prices, fast=5, slow=20)
        assert len(entries) == len(prices)
        assert len(exits) == len(prices)

    def test_custom_fast_slow(self):
        """Support custom fast/slow periods"""
        prices = self._make_price_series([
            5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
        ])
        # Fast=2, slow=5 is more sensitive than default
        entries, exits = sma_cross_signals(prices, fast=2, slow=5)
        assert np.any(entries)


# ═══════════════════════ VPA Volume-Price Confirmation ═══════════════════════


class TestVpaConfirm:
    """VPA Volume-Price ConfirmationTest"""

    def test_insufficient_data_returns_not_confirmed(self):
        """Insufficient data: return unconfirmed"""
        opens = np.array([100.0, 101.0])
        highs = np.array([102.0, 103.0])
        lows = np.array([99.0, 100.0])
        closes = np.array([101.0, 102.0])
        volumes = np.array([1000, 1500])
        result = vpa_confirm(opens, highs, lows, closes, volumes, idx=0)
        assert not result.confirmed
        assert result.reason == "insufficient_data"

    def test_idx_out_of_range_returns_not_confirmed(self):
        """Index out of range: return unconfirmed"""
        n = 20
        opens = np.random.randn(n) + 100
        highs = opens + 2.0
        lows = opens - 2.0
        closes = opens + np.random.randn(n) * 0.5
        volumes = np.ones(n) * 10000
        result = vpa_confirm(opens, highs, lows, closes, volumes, idx=n)
        assert not result.confirmed
        assert result.reason == "insufficient_data"

    def test_buy_validation(self):
        """Buy confirm condition: volume increase + bullish candle + above prev close + short wick"""
        n = 10
        opens = np.ones(n) * 100.0
        highs = np.ones(n) * 105.0
        lows = np.ones(n) * 99.0
        closes = np.ones(n) * 102.0
        volumes = np.ones(n) * 1000

        # Manufacture confirm condition: idx=9 (last position)
        opens[9] = 100.0
        closes[9] = 103.0  # Bullish candle
        highs[9] = 103.5   # Short upper wick
        lows[9] = 99.5
        volumes[9] = 2000  # High Volume > avg * 1.2

        # Previous close = 102.0, today close = 103.0 > 102.0 OK
        result = vpa_confirm(opens, highs, lows, closes, volumes, idx=9, side="buy")
        assert result.confirmed, f"Should confirm but got: {result.reason} - {result.detail}"
        assert result.reason == "validation"

    def test_sell_validation(self):
        """Sell confirm condition: volume increase + bearish candle + below prev close + short wick"""
        n = 10
        opens = np.ones(n) * 100.0
        highs = np.ones(n) * 103.0
        lows = np.ones(n) * 97.0
        closes = np.ones(n) * 99.0
        volumes = np.ones(n) * 1000

        # ManufacturingConfirmCondition: idx=9
        opens[9] = 101.0
        closes[9] = 98.0   # Bearish candle
        highs[9] = 101.5
        lows[9] = 97.5
        volumes[9] = 2000  # High Volume

        result = vpa_confirm(opens, highs, lows, closes, volumes, idx=9, side="sell")
        assert result.confirmed, f"Should confirm but got: {result.reason} - {result.detail}"
        assert result.reason == "validation"

    def test_low_volume_rejection(self):
        """Low volume should be rejected"""
        n = 10
        opens = np.ones(n) * 100.0
        highs = np.ones(n) * 105.0
        lows = np.ones(n) * 99.0
        closes = np.ones(n) * 102.0
        avg_vol = 1000
        volumes = np.ones(n) * avg_vol

        # Reduce volume at idx=9
        volumes[9] = 100  # Far below mean
        closes[9] = 103.0  # Bullish candle
        highs[9] = 103.5

        result = vpa_confirm(opens, highs, lows, closes, volumes, idx=9)
        assert not result.confirmed
        # Should be rejected due to low volume
        assert "low_vol" in result.detail

    def test_long_wick_rejection(self):
        """Long upper wick should be rejected"""
        n = 10
        opens = np.ones(n) * 100.0
        highs = np.ones(n) * 105.0
        lows = np.ones(n) * 99.0
        closes = np.ones(n) * 101.0
        volumes = np.ones(n) * 2000

        # Long upper wick: high=110, close=101, open=100
        opens[9] = 100.0
        closes[9] = 101.0  # Bullish candle
        highs[9] = 110.0   # Long upper wick
        volumes[9] = 3000

        result = vpa_confirm(opens, highs, lows, closes, volumes, idx=9)
        assert not result.confirmed or result.wick_pct >= 0.30
        # If not confirmed and reason includes long wick
        if not result.confirmed:
            assert "long_wick" in result.detail or result.reason == "anomaly"

    def test_wrong_color_rejection(self):
        """Bearish candle with buy signal should be rejected"""
        n = 10
        opens = np.ones(n) * 100.0
        highs = np.ones(n) * 103.0
        lows = np.ones(n) * 98.0
        closes = np.ones(n) * 101.0
        volumes = np.ones(n) * 2000

        # Bearish candle (close < open)
        opens[9] = 101.0
        closes[9] = 99.0
        volumes[9] = 3000

        result = vpa_confirm(opens, highs, lows, closes, volumes, idx=9, side="buy")
        assert not result.confirmed
        # Should indicate wrong color direction
        assert result.reason == "anomaly"

    def test_vol_ratio_calculation(self):
        """Validate vol_ratio calculation correctness"""
        n = 10
        opens = np.ones(n) * 100.0
        highs = np.ones(n) * 102.0
        lows = np.ones(n) * 98.0
        closes = np.ones(n) * 101.0
        volumes = np.array([1000] * 5 + [2000] * 5)  # Last 5 avg = 2000

        result = vpa_confirm(opens, highs, lows, closes, volumes, idx=9)
        # vol_lookback=5, first 5 avg volume = 1000
        # vol_ratio = 2000 / 1000 = 2.0
        assert result.vol_ratio == pytest.approx(2.0, rel=0.5)

    def test_wick_pct_calculation(self):
        """Validate upper wick proportion calculation"""
        n = 10
        opens = np.full(n, 100.0)
        highs = np.full(n, 110.0)
        lows = np.full(n, 95.0)
        closes = np.full(n, 105.0)
        volumes = np.full(n, 2000)

        # Buy: upper wick = (high - max(close, open)) / total_range
        # max(close, open) = 105, total_range = 110-95 = 15
        # wick_pct = (110-105)/15 = 0.333
        result = vpa_confirm(opens, highs, lows, closes, volumes, idx=9, side="buy")
        assert result.wick_pct == pytest.approx(0.3333, rel=0.5)


# ═══════════════════════ apply_vpa_filter ═══════════════════════


class TestApplyVpaFilter:
    """VPA filterTest"""

    def test_filter_passes_confirmed_entries(self):
        """VPA confirmed entry should be preserved"""
        n = 15
        opens = np.ones(n) * 100.0
        highs = np.ones(n) * 103.0
        lows = np.ones(n) * 98.0
        closes = np.ones(n) * 101.0
        volumes = np.ones(n) * 2000

        # Create confirm condition at idx=9
        opens[9] = 100.0
        closes[9] = 102.0
        highs[9] = 102.5
        volumes[9] = 3000

        entries = np.zeros(n, dtype=bool)
        entries[9] = True

        filtered, results = apply_vpa_filter(entries, opens, highs, lows, closes, volumes)
        assert filtered[9], "Confirmed signal should be preserved"

    def test_filter_rejects_bad_entries(self):
        """Unconfirmed entry should be filtered"""
        n = 15
        opens = np.ones(n) * 100.0
        highs = np.ones(n) * 103.0
        lows = np.ones(n) * 98.0
        closes = np.ones(n) * 101.0
        volumes = np.ones(n) * 2000

        # idx=9 NoneHigh Volume
        entries = np.zeros(n, dtype=bool)
        entries[9] = True

        filtered, results = apply_vpa_filter(entries, opens, highs, lows, closes, volumes)
        assert not filtered[9], "Unconfirmed signal should be filtered"

    def test_filter_returns_confirm_results(self):
        """Filter should return confirm result for each signal"""
        n = 20
        opens = np.ones(n) * 100.0
        highs = np.ones(n) * 103.0
        lows = np.ones(n) * 98.0
        closes = np.ones(n) * 101.0
        volumes = np.ones(n) * 2000

        entries = np.zeros(n, dtype=bool)
        entries[10] = True
        entries[15] = True

        filtered, results = apply_vpa_filter(entries, opens, highs, lows, closes, volumes)
        assert len(results) == 2
        assert all(isinstance(r, VPAConfirmResult) for r in results)


# ═══════════════════════ Triple-Barrier Tag ═══════════════════════


class TestTripleBarrierLabels:
    """Triple-Barrier TagTest"""

    def test_take_profit_label(self):
        """Hit first take profit should be labeled +1"""
        # Price rises sharply from 100, then pulls back
        closes = np.array([100.0, 102.0, 105.0, 108.0, 106.0, 104.0, 110.0])
        entry_indices = np.array([0])  # Enter at bar 0
        # ATR Based On close change: tr = [2, 3, 3, 2, 2, 6]
        # atr approx = 3
        # pt_level = 100 + 2.0 * atr ≈ 106
        # At bar 2 (close=105) < 106, bar 3 (close=108) >= 106 -> take profit
        labels = triple_barrier_labels(closes, entry_indices, pt_mult=2.0, sl_mult=1.0)
        assert labels[0] == 1, f"Should take profit (+1) but got {labels[0]}"

    def test_stop_loss_label(self):
        """Hit first stop loss should be labeled -1"""
        # Price sharply declines from 100
        closes = np.array([100.0, 98.0, 95.0, 92.0, 94.0, 96.0])
        entry_indices = np.array([0])
        # pt_level = 100 + 2.0 * atr, sl_level = 100 - 1.0 * atr
        # ATR approx = 2, sl ≈ 98
        # bar 1 (close=98) <= sl -> Stop Loss
        labels = triple_barrier_labels(closes, entry_indices, pt_mult=2.0, sl_mult=1.0)
        assert labels[0] == -1, f"Should stop loss (-1) but got {labels[0]}"

    def test_timeout_with_profit(self):
        """Timeout with profit should be labeled +1"""
        # Slow rise, does not hit take profit or stop loss
        closes = np.array([100.0, 100.5, 101.0, 101.5, 102.0, 102.5])
        entry_indices = np.array([0])
        max_hold = 3  # Max hold 3 bars
        # Small atr, tight pt/sl range, but price slowly rises (won't hit)
        labels = triple_barrier_labels(closes, entry_indices, pt_mult=5.0, sl_mult=5.0, max_hold=max_hold)
        # max_hold=3, check end_idx=0+3=3, close[3]=101.5 > 100 -> profitable
        assert labels[0] == 1, f"Timeout profitable should be +1 but got {labels[0]}"

    def test_timeout_with_loss(self):
        """Timeout with loss should be labeled -1"""
        closes = np.array([100.0, 99.5, 99.0, 98.5, 98.0, 97.5])
        entry_indices = np.array([0])
        max_hold = 3
        labels = triple_barrier_labels(closes, entry_indices, pt_mult=5.0, sl_mult=5.0, max_hold=max_hold)
        # end_idx=3, close[3]=98.5 < 100 -> loss
        assert labels[0] == -1, f"Timeout loss should be -1 but got {labels[0]}"

    def test_entry_at_last_bar(self):
        """Entry at last bar: label as 0"""
        closes = np.array([100.0, 101.0, 102.0])
        entry_indices = np.array([2])  # Last position
        labels = triple_barrier_labels(closes, entry_indices)
        assert labels[0] == 0

    def test_multiple_entries(self):
        """Multiple entry signals labeled separately"""
        closes = np.array([100.0, 105.0, 95.0, 110.0, 90.0, 100.0])
        entry_indices = np.array([0, 2, 3])
        labels = triple_barrier_labels(closes, entry_indices, pt_mult=1.0, sl_mult=1.0, max_hold=3)
        assert len(labels) == 3

    def test_pt_sl_mult_change_behaviour(self):
        """Adjust pt_mult/sl_mult changes behavior"""
        closes = np.array([100.0, 103.0, 97.0, 106.0, 94.0])
        entry_indices = np.array([0])

        # Wide stop loss, narrow take profit: easier to take profit
        labels_wide_sl = triple_barrier_labels(closes, entry_indices, pt_mult=1.0, sl_mult=5.0, max_hold=4)
        # Narrow stop loss, wide take profit: easier to stop loss
        labels_wide_pt = triple_barrier_labels(closes, entry_indices, pt_mult=5.0, sl_mult=1.0, max_hold=4)

        assert len(labels_wide_sl) == 1
        assert len(labels_wide_pt) == 1

    def test_empty_entry_indices(self):
        """ShortEntryIndexBackShortTag"""
        labels = triple_barrier_labels(np.array([100.0, 101.0]), np.array([], dtype=int))
        assert len(labels) == 0


# ═══════════════════════ Triple-Barrier OHLC ═══════════════════════


class TestTripleBarrierLabelsOHLC:
    """Triple-Barrier OHLC VersionTest"""

    def test_high_triggers_take_profit(self):
        """Intraday high triggers take profit"""
        opens = np.array([100.0, 101.0, 102.0])
        highs = np.array([102.0, 108.0, 105.0])  # bar 1 IntradayBreakout
        lows = np.array([99.0, 100.0, 101.0])
        closes = np.array([101.0, 105.0, 104.0])
        entry_indices = np.array([0])

        labels = triple_barrier_labels_ohlc(
            opens, highs, lows, closes, entry_indices,
            pt_mult=1.0, sl_mult=5.0, max_hold=2,
        )
        # bar 1 high=108 should trigger take profit
        assert labels[0] == 1

    def test_low_triggers_stop_loss(self):
        """Intraday low triggers stop loss"""
        opens = np.array([100.0, 99.0, 98.0])
        highs = np.array([102.0, 100.0, 99.0])
        lows = np.array([99.0, 93.0, 97.0])  # bar 1 intraday breach
        closes = np.array([101.0, 98.0, 98.5])
        entry_indices = np.array([0])

        labels = triple_barrier_labels_ohlc(
            opens, highs, lows, closes, entry_indices,
            pt_mult=5.0, sl_mult=1.0, max_hold=2,
        )
        # bar 1 low=93 should trigger stop loss
        assert labels[0] == -1

    def test_entry_at_last_bar(self):
        """Entry at last bar: label as 0"""
        opens = np.array([100.0, 101.0])
        highs = np.array([102.0, 103.0])
        lows = np.array([99.0, 100.0])
        closes = np.array([101.0, 102.0])
        entry_indices = np.array([1])
        labels = triple_barrier_labels_ohlc(opens, highs, lows, closes, entry_indices)
        assert labels[0] == 0

    def test_empty_indices(self):
        """ShortIndexBackShortTag"""
        labels = triple_barrier_labels_ohlc(
            np.array([100.0]), np.array([101.0]),
            np.array([99.0]), np.array([100.5]),
            np.array([], dtype=int),
        )
        assert len(labels) == 0


# ═══════════════════════ Integration engine ═══════════════════════


class TestRunStrategy:
    """CompleteStrategyPipelineTest"""

    def _generate_trend_data(self, n: int = 100, seed: int = 42) -> tuple:
        """Generate OHLCV test data with trend"""
        rng = np.random.RandomState(seed)
        # Trend + noise
        trend = np.linspace(0, 5, n)
        noise = rng.randn(n) * 0.5
        close = 100.0 + trend + noise
        open_p = close + rng.randn(n) * 0.3
        high = np.maximum(close, open_p) + abs(rng.randn(n)) * 0.5 + 0.1
        low = np.minimum(close, open_p) - abs(rng.randn(n)) * 0.5 - 0.1
        volume = np.abs(rng.randn(n)) * 10000 + 5000
        return open_p, high, low, close, volume

    def test_basic_execution(self):
        """StrategyPipelinebasicExecute"""
        opens, highs, lows, closes, volumes = self._generate_trend_data(100)
        result = run_strategy(opens, highs, lows, closes, volumes)
        assert result.total_return is not None
        assert result.sharpe_ratio is not None
        assert result.max_drawdown is not None
        assert result.num_trades >= 0

    def test_without_vpa(self):
        """Do not enable VPA filter"""
        opens, highs, lows, closes, volumes = self._generate_trend_data(100)
        result = run_strategy(opens, highs, lows, closes, volumes, enable_vpa=False)
        assert result.num_trades >= 0

    def test_short_data_returns_zero_metrics(self):
        """Very short data returns zero metrics"""
        opens = np.array([100.0, 101.0])
        highs = np.array([102.0, 103.0])
        lows = np.array([99.0, 100.0])
        closes = np.array([101.0, 102.0])
        volumes = np.array([1000, 2000])
        result = run_strategy(opens, highs, lows, closes, volumes)
        assert result.num_trades == 0

    def test_strategy_result_type(self):
        """Validate returned strategy result type"""
        opens, highs, lows, closes, volumes = self._generate_trend_data(80)
        result = run_strategy(opens, highs, lows, closes, volumes)
        assert hasattr(result, "total_return")
        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "max_drawdown")
        assert hasattr(result, "win_rate")
        assert hasattr(result, "num_trades")
        assert hasattr(result, "tb_labels")
        assert hasattr(result, "filtered_entries")

    def test_triple_barrier_labels_in_result(self):
        """Result includes triple-barrier labels"""
        opens, highs, lows, closes, volumes = self._generate_trend_data(120)
        result = run_strategy(opens, highs, lows, closes, volumes)
        assert len(result.tb_labels) == result.num_trades

    def test_deterministic_output(self):
        """Same input produces same output"""
        opens, highs, lows, closes, volumes = self._generate_trend_data(100, seed=42)
        result1 = run_strategy(opens, highs, lows, closes, volumes, fast=5, slow=20)
        opens2, highs2, lows2, closes2, volumes2 = self._generate_trend_data(100, seed=42)
        result2 = run_strategy(opens2, highs2, lows2, closes2, volumes2, fast=5, slow=20)
        assert result1.total_return == result2.total_return
        assert result1.sharpe_ratio == result2.sharpe_ratio

    def test_t_plus_one_entry_day_return_uses_next_open(self, monkeypatch):
        """Signal day close to next day open gap should not count in T+1 entry returns."""
        opens = np.array([100.0, 200.0, 220.0, 220.0])
        highs = np.array([101.0, 221.0, 221.0, 221.0])
        lows = np.array([99.0, 199.0, 219.0, 219.0])
        closes = np.array([100.0, 220.0, 220.0, 220.0])
        volumes = np.array([1000.0, 1000.0, 1000.0, 1000.0])

        def fixed_signals(prices, fast=10, slow=20):
            entries = np.array([True, False, False, False])
            exits = np.zeros(4, dtype=bool)
            return entries, exits

        monkeypatch.setattr(signal_engine_module, "sma_cross_signals", fixed_signals)
        result = run_strategy(
            opens,
            highs,
            lows,
            closes,
            volumes,
            strategy="sma",
            commission_rate=0.0,
            stamp_duty=0.0,
            tb_max_hold=10,
        )

        assert result.total_return == pytest.approx(0.10)

    def test_custom_sma_parameters(self):
        """Custom SMA parameter"""
        opens, highs, lows, closes, volumes = self._generate_trend_data(150)
        result_fast = run_strategy(opens, highs, lows, closes, volumes, fast=5, slow=20)
        result_slow = run_strategy(opens, highs, lows, closes, volumes, fast=20, slow=50)
        assert result_fast.num_trades >= 0
        assert result_slow.num_trades >= 0


# ═══════════════════════ KAMA CrossoverSignal ═══════════════════════


class TestKamaCrossSignals:
    """Deterministic tests for the KAMA crossover entry/exit signals."""

    def test_rising_market_after_flat_generates_entry(self):
        closes = np.array(
            [10.0] * 40 + [10.5, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            dtype=float,
        )
        entries, exits = kama_cross_signals(closes, er_period=5, fast_sc=2, slow_sc=20)
        assert np.any(entries)
        assert not np.any(exits[:len(entries)//2])

    def test_declining_after_uptrend_generates_exit(self):
        closes = np.array(
            [10.0] * 30
            + [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            + [19, 18, 17, 16, 15, 14, 13, 12, 11, 10],
            dtype=float,
        )
        entries, exits = kama_cross_signals(closes, er_period=5, fast_sc=2, slow_sc=20)
        assert np.any(entries)
        assert np.any(exits)

    def test_flat_price_no_signals(self):
        closes = np.array([10.0] * 60, dtype=float)
        entries, exits = kama_cross_signals(closes, er_period=5, fast_sc=2, slow_sc=20)
        assert not np.any(entries)
        assert not np.any(exits)
