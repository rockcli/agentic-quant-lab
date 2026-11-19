"""
Example Strategy Library

Reference: Trading Systems and Methods (Kaufman), Ch.8-9
      Quantitative Portfolio Management (Isichenko)
      Volume Profile + VPA
"""

from __future__ import annotations

from agentic_quant.backtest.data.structures import Bar
from agentic_quant.backtest.features.engine import FeatureEngine
from agentic_quant.backtest.strategy.base import Signal, Strategy
from agentic_quant.strategy.registry import register_strategy


@register_strategy("sma_cross")
class SmaCrossStrategy(Strategy):
    """
    Dual Moving Average Crossover Strategy (example)
    Uses incremental SMA calculation, avoids O(n^2) series rebuild
    Reference: Trading Systems and Methods, Ch.8
    """

    def __init__(self, fast: int = 10, slow: int = 30):
        super().__init__(name=f"SMA_Cross_{fast}_{slow}")
        self.fast = fast
        self.slow = slow
        self._prev_position = 0  # 0=flat, 1=long
        self._closes: list[float] = []

    def on_bar(self, bar: Bar) -> Signal | None:
        self._closes.append(bar.close)
        if len(self._closes) < self.slow:
            return None

        fast_sma = sum(self._closes[-self.fast :]) / self.fast
        slow_sma = sum(self._closes[-self.slow :]) / self.slow

        position = 1 if fast_sma > slow_sma else 0
        signal = None

        if position == 1 and self._prev_position == 0:
            signal = Signal(
                timestamp=bar.timestamp, action="buy", size=1.0, price=bar.close
            )
        elif position == 0 and self._prev_position == 1:
            signal = Signal(
                timestamp=bar.timestamp, action="close", size=0.0, price=bar.close
            )

        self._prev_position = position
        return signal


@register_strategy("donchian_breakout")
class DonchianBreakout(Strategy):
    """
    Donchian Channel Breakout Strategy (Trend Following)
    Reference: Trading Systems and Methods, Ch.8
    """

    def __init__(self, channel_period: int = 20, atr_stop_mult: float = 2.0):
        super().__init__(name=f"Donchian_{channel_period}")
        self.channel_period = channel_period
        self.atr_stop_mult = atr_stop_mult
        self._closes: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._in_position = False
        self._stop_price = 0.0

    def on_bar(self, bar: Bar) -> Signal | None:
        self._closes.append(bar.close)
        self._highs.append(bar.high)
        self._lows.append(bar.low)

        if len(self._closes) < self.channel_period:
            return None

        upper = max(self._highs[-self.channel_period :])
        lower = min(self._lows[-self.channel_period :])

        if not self._in_position:
            if bar.close >= upper:
                self._in_position = True
                self._stop_price = bar.close - self.atr_stop_mult * self._approx_atr()
                return Signal(timestamp=bar.timestamp, action="buy", size=1.0, price=bar.close)
        else:
            if bar.close <= lower or bar.close <= self._stop_price:
                self._in_position = False
                return Signal(timestamp=bar.timestamp, action="close", size=0.0, price=bar.close)
            # Trailing stop-loss move up
            new_stop = bar.close - self.atr_stop_mult * self._approx_atr()
            if new_stop > self._stop_price:
                self._stop_price = new_stop

        return None

    def _approx_atr(self) -> float:
        """Approximate ATR using recent bars' high-low"""
        if len(self._highs) < 2:
            return 1.0
        recent = [
            h - lo for h, lo in zip(self._highs[-5:], self._lows[-5:], strict=False)
        ]
        return sum(recent) / len(recent) if recent else 1.0


@register_strategy("bollinger_mean_reversion")
class BollingerMeanReversion(Strategy):
    """
    Bollinger Bands Mean Reversion Strategy
    Buy when price touches lower band and RSI is oversold, close when price touches upper band or reverts to middle band.
    Reference: Trading Systems and Methods, Ch.9
    """

    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
        rsi_window: int = 14,
        rsi_lower: int = 30,
        rsi_upper: int = 70,
    ):
        super().__init__(name=f"BB_MeanRev_{window}")
        self.window = window
        self.num_std = num_std
        self.rsi_window = rsi_window
        self.rsi_lower = rsi_lower
        self.rsi_upper = rsi_upper
        self._closes: list[float] = []
        self._in_position = False

    def on_bar(self, bar: Bar) -> Signal | None:
        self._closes.append(bar.close)
        if len(self._closes) < max(self.window, self.rsi_window):
            return None

        series = __import__("polars").Series("close", self._closes)
        upper, middle, lower = FeatureEngine.bollinger_bands(
            series, window=self.window, num_std=self.num_std
        )
        rsi = FeatureEngine.rsi(series, window=self.rsi_window)

        u = float(upper[-1])
        m = float(middle[-1])
        lo = float(lower[-1])
        rsi_last = rsi[-1]
        if rsi_last is None:
            return None
        r = float(rsi_last)

        if not self._in_position:
            if bar.close <= lo and r <= self.rsi_lower:
                self._in_position = True
                return Signal(timestamp=bar.timestamp, action="buy", size=1.0, price=bar.close)
        else:
            if bar.close >= u or bar.close >= m or r >= self.rsi_upper:
                self._in_position = False
                return Signal(timestamp=bar.timestamp, action="close", size=0.0, price=bar.close)

        return None


@register_strategy("pairs_trading")
class PairsTrading(Strategy):
    """
    Statistical Arbitrage: Pair Trading (Simplified Z-Score Strategy)
    Reference: Quantitative Portfolio Management (Isichenko)
    """

    def __init__(self, lookback: int = 60, z_entry: float = 2.0, z_exit: float = 0.5):
        super().__init__(name=f"Pairs_{lookback}")
        self.lookback = lookback
        self.z_entry = z_entry
        self.z_exit = z_exit
        self._spreads: list[float] = []
        self._position = 0  # +1 long spread / -1 short spread

    def on_bar(self, bar: Bar) -> Signal | None:
        # Simplified: assume bar.close is already the spread series
        self._spreads.append(bar.close)
        if len(self._spreads) < self.lookback:
            return None

        window = self._spreads[-self.lookback :]
        mean = sum(window) / len(window)
        variance = sum((x - mean) ** 2 for x in window) / len(window)
        std = variance**0.5 if variance > 0 else 1.0
        z = (bar.close - mean) / std

        if self._position == 0:
            if z <= -self.z_entry:
                self._position = 1
                return Signal(timestamp=bar.timestamp, action="buy", size=1.0, price=bar.close)
            if z >= self.z_entry:
                self._position = -1
                return Signal(timestamp=bar.timestamp, action="sell", size=-1.0, price=bar.close)
        elif self._position == 1 and z >= self.z_exit or self._position == -1 and z <= -self.z_exit:
            self._position = 0
            return Signal(timestamp=bar.timestamp, action="close", size=0.0, price=bar.close)

        return None


@register_strategy("volume_profile")
class VolumeProfileStrategy(Strategy):
    """
    Volume Distribution Strategy (Simplified POC Bounce)
    Reference: Volume Profile (Insider's Guide), VPA
    """

    def __init__(self, va_percent: float = 0.7, lookback: int = 30):
        super().__init__(name=f"VolProfile_{lookback}")
        self.va_percent = va_percent
        self.lookback = lookback
        self._closes: list[float] = []
        self._volumes: list[float] = []
        self._in_position = False

    def on_bar(self, bar: Bar) -> Signal | None:
        self._closes.append(bar.close)
        self._volumes.append(bar.volume)
        if len(self._closes) < self.lookback:
            return None

        # Calculate value area (simplified: bucketing by price to find high-volume zones)
        window_closes = self._closes[-self.lookback :]
        window_volumes = self._volumes[-self.lookback :]

        # POC = price with highest volume
        poc = self._weighted_mode(window_closes, window_volumes)
        # VAH/VAL approximation: centered on POC, covering va_percent of volume range
        # Simplified as POC +/- 1 standard deviation
        mean = sum(window_closes) / len(window_closes)
        std = (sum((x - mean) ** 2 for x in window_closes) / len(window_closes)) ** 0.5
        vah = poc + std
        val = poc - std

        if not self._in_position:
            if val <= bar.close <= vah and bar.close > poc:
                # Within value area and near POC above, considered as having support
                self._in_position = True
                return Signal(timestamp=bar.timestamp, action="buy", size=1.0, price=bar.close)
        else:
            if bar.close < val or bar.close > vah * 1.02:
                self._in_position = False
                return Signal(timestamp=bar.timestamp, action="close", size=0.0, price=bar.close)

        return None

    @staticmethod
    def _weighted_mode(prices: list[float], volumes: list[float]) -> float:
        """Weighted mode approximation: VWAP"""
        if not prices or not volumes:
            return 0.0
        total = sum(p * v for p, v in zip(prices, volumes, strict=False))
        vol_sum = sum(volumes)
        return total / vol_sum if vol_sum > 0 else prices[-1]


@register_strategy("meta_labeling_vpa")
class MetaLabelingVpaStrategy(Strategy):
    """
    Meta-Labeling + VPA Confirmation Strategy

    Primary Model (Kaufman): SMA Crossover (fast=5, slow=20) determines direction (side)
    Secondary Model (Lopez de Prado): VPA Volume-Price Confirmation determines magnitude (size)

    VPA Rules based on Anna Coulling's Volume Price Analysis:
    - Volume validates price (effort vs result)
    - Validation = Volume increase confirming uptrend
    - Anomaly = volume-price divergence filtering false breakouts
    - Wick analysis avoids false breakout

    Reference: Trading Systems and Methods Ch.22
          Advances in Financial Machine Learning Ch.3 (Meta-Labeling)
          A Complete Guide To Volume Price Analysis (Coulling)
    """

    def __init__(
        self,
        fast: int = 5,
        slow: int = 20,
        vol_lookback: int = 5,
        vol_mult: float = 1.2,
        wick_threshold: float = 0.30,
    ):
        super().__init__(name=f"MetaVPA_{fast}_{slow}")
        self.fast = fast
        self.slow = slow
        self.vol_lookback = vol_lookback
        self.vol_mult = vol_mult
        self.wick_threshold = wick_threshold
        self._prev_sma_long = False
        self._in_position = False

    def on_bar(self, bar: Bar) -> Signal | None:
        n = len(self.bars)
        if n < self.slow:
            return None

        closes = [b.close for b in self.bars]
        sma_fast = sum(closes[-self.fast :]) / self.fast
        sma_slow = sum(closes[-self.slow :]) / self.slow
        curr_long = sma_fast > sma_slow

        # Primary model: SMA cross detection
        primary_buy = curr_long and not self._prev_sma_long
        primary_sell = not curr_long and self._prev_sma_long
        self._prev_sma_long = curr_long

        # Secondary model: VPA confirmation (only when primary signals)
        if primary_buy and n >= self.vol_lookback + 1:
            confirmed = self._vpa_confirm(n - 1, side="buy")
            if confirmed and not self._in_position:
                self._in_position = True
                return Signal(
                    timestamp=bar.timestamp,
                    action="buy",
                    size=1.0,
                    price=bar.close,
                    meta={"source": "meta_vpa", "wick": self._last_wick, "vol_ratio": self._last_vol_ratio},
                )

        if primary_sell and self._in_position:
            self._in_position = False
            return Signal(
                timestamp=bar.timestamp,
                action="close",
                size=0.0,
                price=bar.close,
                meta={"source": "sma_dead_cross"},
            )

        return None

    def _vpa_confirm(self, idx: int, side: str) -> bool:
        """VPA secondary model."""
        if idx < self.vol_lookback:
            return False

        bar = self.bars[idx]
        prev_bars = self.bars[idx - self.vol_lookback : idx]
        avg_amount = sum(b.volume for b in prev_bars) / len(prev_bars)

        # Rule 1: Volume increase (effort)
        volume_increase = bar.volume > avg_amount * self.vol_mult
        self._last_vol_ratio = bar.volume / avg_amount if avg_amount > 0 else 0

        # Rule 2: Price validates direction (result)
        if side == "buy":
            price_up = bar.close > bar.open
            higher_close = bar.close > self.bars[idx - 1].close
        else:
            price_up = bar.close < bar.open
            higher_close = bar.close < self.bars[idx - 1].close

        # Rule 3: Wick analysis
        total_range = bar.high - bar.low
        if total_range > 0:
            if side == "buy":
                upper_wick = (bar.high - max(bar.close, bar.open)) / total_range
                wick_ok = upper_wick < self.wick_threshold
                self._last_wick = upper_wick
            else:
                lower_wick = (min(bar.close, bar.open) - bar.low) / total_range
                wick_ok = lower_wick < self.wick_threshold
                self._last_wick = lower_wick
        else:
            wick_ok = True
            self._last_wick = 0.0

        return volume_increase and price_up and higher_close and wick_ok  # type: ignore[no-any-return]
