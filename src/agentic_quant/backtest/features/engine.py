"""
Feature EngineeringModule

Include常用Technical Indicators与Financials特征
参考: Trading Systems and Methods (Kaufman), Ch.7-9
"""

from __future__ import annotations

import polars as pl


class FeatureEngine:
    """Feature EngineeringModule"""

    @staticmethod
    def sma(series: pl.Series, window: int) -> pl.Series:
        """简单Moving Average (SMA)"""
        return series.rolling_mean(window)

    @staticmethod
    def ema(series: pl.Series, span: int) -> pl.Series:
        """Exponential Moving Average (EMA)"""
        return series.ewm_mean(span=span)

    @staticmethod
    def rsi(series: pl.Series, window: int = 14) -> pl.Series:
        """
        相对强弱Index (RSI)
        参考: Trading Systems and Methods, Ch.9
        """
        delta = series.diff()
        gain = delta.clip(lower_bound=0)
        loss = (-delta).clip(lower_bound=0)
        avg_gain = gain.rolling_mean(window)
        avg_loss = loss.rolling_mean(window)
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    @staticmethod
    def bollinger_bands(
        series: pl.Series, window: int = 20, num_std: float = 2.0
    ) -> tuple[pl.Series, pl.Series, pl.Series]:
        """
        Bollinger Bands
        Returns: (upper, middle, lower)
        """
        middle = series.rolling_mean(window)
        std = series.rolling_std(window)
        upper = middle + num_std * std
        lower = middle - num_std * std
        return upper, middle, lower

    @staticmethod
    def atr(
        high: pl.Series, low: pl.Series, close: pl.Series, window: int = 14
    ) -> pl.Series:
        """
        Average True Range (ATR)
        Used ForStop Loss与头寸Calculate，参考: Trading Systems and Methods, Ch.23
        """
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        # pl.max_horizontal 对 Series Back Expr，需用 DataFrame 上下文求值
        df = pl.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3})
        tr = df.select(pl.max_horizontal("tr1", "tr2", "tr3")).to_series()
        return tr.rolling_mean(window)

    @staticmethod
    def macd(
        series: pl.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple[pl.Series, pl.Series, pl.Series]:
        """
        MACD Metrics
        Returns: (macd_line, signal_line, histogram)
        """
        ema_fast = series.ewm_mean(span=fast)
        ema_slow = series.ewm_mean(span=slow)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm_mean(span=signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
