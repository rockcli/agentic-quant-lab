"""
Market regime classifier -- 4-quadrant decomposition by trend/volatility

Divides the market into 4 regime quadrants:
  bull_high_vol  |  bear_high_vol
  bull_low_vol   |  bear_low_vol

Trend dimension: Price > MA(200) and MA(200) slope positive = bull, else bear
Volatility dimension: 20d return std > full historical median = high vol, else low vol

Reference:
- Lopez de Prado, Advances in Financial Machine Learning, Ch.7
- Marcos Lopez de Prado, "A Data Science Solution to the Multiple-Testing Crisis"
- https://en.wikipedia.org/wiki/Market_regime
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

# ──────────────────────── Helper Functions ──────────────────────────


def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average, O(n) cumulative sum implementation, front-filled NaN"""
    n = len(arr)
    result = np.full(n, np.nan)
    if n < period:
        return result
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    for i in range(period - 1, n):
        result[i] = (cumsum[i + 1] - cumsum[i - period + 1]) / period
    return result


def _rolling_slope(arr: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Rolling window linear regression slope.

    For each valid position i, fit y = a + b*x using arr[i-window+1 : i+1],
    Returns slope b (i.e. tangent direction of MA).
    Front-filled with NaN.

    Least squares formula:
        b = sum((x - x_mean)(y - y_mean)) / sum((x - x_mean)^2)
    """
    n = len(arr)
    result = np.full(n, np.nan)
    if n < window:
        return result
    x = np.arange(window, dtype=float)
    x_mean = np.mean(x)
    x_ss = np.sum((x - x_mean) ** 2)
    if x_ss < 1e-12:
        return result

    for i in range(window - 1, n):
        y = arr[i - window + 1 : i + 1]
        if np.any(np.isnan(y)):
            continue
        y_mean = np.mean(y)
        slope = np.sum((x - x_mean) * (y - y_mean)) / x_ss
        result[i] = slope

    return result


def _rolling_std(arr: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling standard deviation (population std ddof=0), front-filled NaN"""
    n = len(arr)
    result = np.full(n, np.nan)
    if n < window:
        return result
    for i in range(window - 1, n):
        segment = arr[i - window + 1 : i + 1]
        if np.all(np.isfinite(segment)):
            result[i] = float(np.std(segment, ddof=0))
    return result


# ──────────────────────── Market regime classifier ──────────────────────────


def classify_regime(closes: np.ndarray, window: int = 20) -> str:
    """
    Three-class regime classification based on efficiency ratio (ER).

    ER = |close[-1] - close[-window]| / sum(|close[i] - close[i-1]|), i in [-window+1..-1]

    Returns:
        "trend":    ER > 0.40, trending market, normal trading
        "ranging":  0.15 < ER <= 0.40, ranging market, close positions only, no new entries
        "noise":    ER <= 0.15, noisy market, liquidate all and wait
    """
    if len(closes) < window + 1:
        return "ranging"  # Insufficient data, be conservative
    window_closes = closes[-window:]
    net_change = abs(float(window_closes[-1]) - float(window_closes[0]))
    sum_abs = float(np.sum(np.abs(np.diff(window_closes))))
    er = net_change / (sum_abs + 1e-9)
    if er > 0.40:
        return "trend"
    elif er > 0.15:
        return "ranging"
    else:
        return "noise"


def classify_regime_percentile(
    closes: np.ndarray,
    er_window: int = 20,
    rank_window: int = 100,
    trend_pct: float = 70.0,
    noise_pct: float = 30.0,
    min_history: int = 60,
) -> str:
    """
    Adaptive regime classification using ER percentile rank (2026 approach).

    Instead of fixed thresholds (0.40/0.15), ranks the current ER against its
    own rolling historical distribution. This self-normalizes across different
    assets and market conditions — a "trending" ER for CSI 800 might be 0.25
    while for gold it might be 0.45.

    Pipeline:
      1. Compute ER over `er_window` for the entire series
      2. Rank the latest ER within the trailing `rank_window` ER values
      3. Classify: > trend_pct% → trend, < noise_pct% → noise, else → ranging

    Reference: NovaLens Kaufman ER Gate (2025), TSL Adaptive Regime Detector (2025)

    Args:
        closes: close price array
        er_window: ER calculation window (default 20, same as fixed-threshold)
        rank_window: lookback window for percentile ranking (default 100)
        trend_pct: percentile threshold for "trend" (default 70)
        noise_pct: percentile threshold for "noise" (default 30)
        min_history: minimum bars needed before producing a valid label

    Returns:
        "trend", "ranging", or "noise"
    """
    n = len(closes)
    if n < er_window + 1 or n < min_history:
        return "ranging"

    # Compute rolling ER for the whole series
    er_series = np.full(n, np.nan)
    for i in range(er_window, n):
        w = closes[i - er_window : i + 1]
        net = abs(float(w[-1]) - float(w[0]))
        path = float(np.sum(np.abs(np.diff(w))))
        er_series[i] = net / (path + 1e-9)

    # Get valid ER history for ranking
    valid_er = er_series[-rank_window:] if n >= rank_window else er_series[er_window:]
    valid_er = valid_er[np.isfinite(valid_er)]

    if len(valid_er) < min_history // 2:
        return "ranging"

    current_er = valid_er[-1]
    # Percentile rank: what fraction of historical ER is below current?
    pct_rank = float(np.mean(valid_er <= current_er)) * 100.0

    if pct_rank >= trend_pct:
        return "trend"
    elif pct_rank <= noise_pct:
        return "noise"
    else:
        return "ranging"


# ──────────────────── ADX directionalitymovementMetrics ────────────────────────


def _adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
         period: int = 14) -> np.ndarray:
    """
    Vectorized ADX (Average Directional Index), returns array same length as input, front-filled NaN.

    ADX measures trend strength (regardless of direction), 0-100.
    - ADX < 20: weak trend / ranging
    - 20 < ADX < 25: trend forming
    - ADX > 25: strong trend
    - ADX > 40: extremely strong trend

    Reference: Wilder, "New Concepts in Technical Trading Systems", 1978
    """
    n = len(highs)
    adx_series = np.full(n, np.nan)
    if n < period + 2:
        return adx_series

    # 1. ±DM
    up_move = np.diff(highs)
    down_move = -np.diff(lows)
    pos_dm = np.zeros(n - 1)
    neg_dm = np.zeros(n - 1)
    for i in range(n - 1):
        if up_move[i] > down_move[i] and up_move[i] > 0:
            pos_dm[i] = up_move[i]
        if down_move[i] > up_move[i] and down_move[i] > 0:
            neg_dm[i] = down_move[i]

    # 2. ATR (Wilder's smoothed, period)
    tr = np.zeros(n)
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)
    atr = np.full(n, np.nan)
    atr[period] = float(np.mean(tr[1:period + 1]))
    alpha = 1.0 / period
    for i in range(period + 1, n):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

    # 3. Smoothed +DI, -DI (Wilder's smoothing)
    smoothed_pos = np.full(n, np.nan)
    smoothed_neg = np.full(n, np.nan)
    smoothed_pos[period] = float(np.mean(pos_dm[period - 1])) * 1  # First valid value uses mean
    smoothed_neg[period] = float(np.mean(neg_dm[period - 1])) * 1
    # Use Wilder smoothing
    sp = sum(pos_dm[:period])
    sn = sum(neg_dm[:period])
    for i in range(period, n - 1):
        if i == period:
            sp = sum(pos_dm[:period])
            sn = sum(neg_dm[:period])
        else:
            sp = sp - sp / period + pos_dm[i - 1]
            sn = sn - sn / period + neg_dm[i - 1]
        if atr[i] > 0 and np.isfinite(atr[i]):
            smoothed_pos[i] = sp / period / atr[i] * 100
            smoothed_neg[i] = sn / period / atr[i] * 100
        else:
            pass  # leave as NaN

    # Fill last day
    if n - 1 > period:
        sp = sp - sp / period + (pos_dm[n - 2] if n - 2 < len(pos_dm) else 0)
        sn = sn - sn / period + (neg_dm[n - 2] if n - 2 < len(neg_dm) else 0)
        if atr[n - 1] > 0 and np.isfinite(atr[n - 1]):
            smoothed_pos[n - 1] = sp / period / atr[n - 1] * 100
            smoothed_neg[n - 1] = sn / period / atr[n - 1] * 100

    # 4. DX = 100 * |+DI - -DI| / (+DI + -DI)
    dx_series = np.full(n, np.nan)
    for i in range(period, n):
        pdi = smoothed_pos[i]
        ndi = smoothed_neg[i]
        if np.isfinite(pdi) and np.isfinite(ndi) and (pdi + ndi) > 0:
            dx_series[i] = abs(pdi - ndi) / (pdi + ndi) * 100

    # 5. ADX = smoothed DX
    adx_series[period] = float(np.nanmean(dx_series[period:period + period])) if period < n else np.nan
    for i in range(period + period, n):
        prev = adx_series[i - 1]
        if np.isfinite(prev) and np.isfinite(dx_series[i]):
            adx_series[i] = (prev * (period - 1) + dx_series[i]) / period
        elif np.isfinite(dx_series[i]):
            # Cold start
            window = dx_series[max(0, i - period + 1):i + 1]
            valid = window[np.isfinite(window)]
            if len(valid) > 0:
                adx_series[i] = float(np.mean(valid))

    return adx_series


def classify_regime_enhanced(highs: np.ndarray, lows: np.ndarray,
                             closes: np.ndarray, er_window: int = 20,
                             adx_period: int = 14,
                             adx_threshold: float = 25) -> str:
    """
    Dual-confirmation regime classification with ER + ADX (replaces classify_regime).

    Strategy:
    - ER > 0.40 AND ADX > adx_threshold → "trend" (strong trend, normal trading)
    - ER > 0.40 AND ADX <= adx_threshold → "ranging" (fake trend, no entries)
    - 0.15 < ER <= 0.40 → "ranging" (ranging, no new positions)
    - ER <= 0.15 → "noise" (noise, no new positions)

    Reference: Wilder ADX (1978) + Kaufman ER (1995)
    Combined effect: ER detects directionality, ADX confirms trend strength.
    A single ER > 0.40 could come from one extreme day of high volatility,
    Adding ADX > 25 filters out false trend signals.
    """
    n = len(closes)
    if n < max(er_window, adx_period * 2) + 1:
        return "ranging"

    # ER Calculate
    window_closes = closes[-er_window:]
    net_change = abs(float(window_closes[-1]) - float(window_closes[0]))
    sum_abs = float(np.sum(np.abs(np.diff(window_closes))))
    er = net_change / (sum_abs + 1e-9)

    # ADX value (take latest valid value)
    try:
        adx_vals = _adx(highs, lows, closes, period=adx_period)
        last_adx = None
        for val in reversed(adx_vals):
            if np.isfinite(val):
                last_adx = float(val)
                break
        adx_valid = last_adx is not None and last_adx > adx_threshold
    except Exception:
        adx_valid = False

    # ER basic/foundationjudgment
    if er <= 0.15:
        return "noise"
    elif 0.15 < er <= 0.40:
        return "ranging"
    else:
        # ER > 0.40: use ADX to confirm whether it is a real trend
        return "trend" if adx_valid else "ranging"


def is_bear_market(closes: np.ndarray, ma_period: int = 200) -> bool:
    """
    Determine if latest day is in bear market: close below MA(ma_period).

    Used for global regime filter -- when index below 200MA, all strategy positions are closed, no buys.

    Args:
        closes: Close price series (index 0 earliest, index -1 latest)
        ma_period: long termMoving AverageCyclical，Default 200

    Returns:
        True indicates bear market (latest close < MA), should pause new positions
    """
    if len(closes) < ma_period:
        return False  # Default no restriction when data insufficient
    ma = _sma(closes, ma_period)
    return bool(closes[-1] < ma[-1])


@dataclass(frozen=True)
class MarketRegime:
    """
    Market regime classifier (purely functional, no side effects)

    Decomposes market regime into 2x2=4 quadrants:
    - Trend: bull / bear
    - Volatility: high_vol / low_vol

    Usage:
        regime = MarketRegime(ma_period=200, vol_window=20)
        labels = regime.classify(close_prices)
        summary = regime.regime_summary(close_prices)

    Why MA(200) + slope:
    - MA(200) is a widely used long-term trend line by institutions
    - Slope identifies trend strength, avoids false signals when price just crosses MA
    - Combined use reduces false signals

    Why return std median as threshold:
    - Median is robust to extreme values
    - Automatically adapts to different underlying volatility levels
    """

    ma_period: int = 200
    """Long-term MA period, used for bull/bear trend direction"""

    vol_window: int = 20
    """Volatility calculation window (trading days), corresponds to ~1 calendar month"""

    slope_window: int = 20
    """MA slope regression window (trading days), used for trend strength judgment"""

    # Regime names defined as module-level constant below
    # REGIME_NAMES is set after class definition

    def classify(self, close_prices: np.ndarray) -> np.ndarray:
        """
        Classify market regime at each time point。

        Two-step classification pipeline:
        1. Trenddimension:
           - close > MA(ma_period) and MA slope > 0 -> bull
           - Otherwise -> bear
        2. Volatilitydimension:
           - vol_window daily return std > full historical median -> high_vol
           - Otherwise -> low_vol

        Args:
            close_prices: Close price series, 1D numpy array, index 0 is earliest

        Returns:
            regime_labels: string array with same length as close_prices.
                First max(ma_period, vol_window, slope_window) points are warmup, filled with "".
                Valid values: "bull_high_vol", "bull_low_vol", "bear_high_vol", "bear_low_vol"
        """
        n = len(close_prices)
        labels = np.full(n, "", dtype=object)

        warmup = max(self.ma_period, self.vol_window, self.slope_window) + 1
        if n < warmup:
            return labels

        # -- Step 1: Calculate MA and slope --
        ma = _sma(close_prices, self.ma_period)
        ma_slope = _rolling_slope(ma, self.slope_window)

        # -- Step 2: Trend judgment (independent per point) --
        # Bull condition: close above MA and MA itself trending up
        trend_bull = (close_prices > ma) & (ma_slope > 0)

        # -- Step 3: VolatilityCalculate --
        returns = np.diff(close_prices) / (close_prices[:-1] + 1e-9)
        returns = np.insert(returns, 0, 0.0)  # Align length, index 0 = 0
        rolling_vol = _rolling_std(returns, self.vol_window)

        # Use median of all valid volatility as high/low vol boundary
        # Median is more robust than mean, not affected by extreme quotes
        valid_vol = rolling_vol[np.isfinite(rolling_vol)]
        if len(valid_vol) == 0:
            return labels
        vol_threshold = float(np.median(valid_vol))
        high_vol = rolling_vol > vol_threshold

        # -- Step 4: Combine 4 quadrants --
        for i in range(warmup - 1, n):
            if (
                not np.isfinite(rolling_vol[i])
                or not np.isfinite(ma[i])
                or not np.isfinite(ma_slope[i])
            ):
                continue

            bull = trend_bull[i]
            hvol = high_vol[i]

            if bull and hvol:
                labels[i] = "bull_high_vol"
            elif bull and not hvol:
                labels[i] = "bull_low_vol"
            elif not bull and hvol:
                labels[i] = "bear_high_vol"
            else:
                labels[i] = "bear_low_vol"

        return labels

    def regime_summary(
        self,
        close_prices: np.ndarray,
        regime_labels: np.ndarray | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Calculate statistics summary for each regime.

        Outputs each regime time proportion, daily return mean, std dev, and annualized Sharpe,
        Used to determine if strategy has regime dependency (e.g. works in bull low-vol, fails in bear high-vol).

        Args:
            close_prices: Close Price Series
            regime_labels: pre-calculated regime tags, None triggers auto classify()

        Returns:
            {regime_name: {days, pct, avg_return, std_return, sharpe}}
        """
        if regime_labels is None:
            regime_labels = self.classify(close_prices)

        returns = np.diff(close_prices) / (close_prices[:-1] + 1e-9)
        returns = np.insert(returns, 0, 0.0)

        valid_mask = regime_labels != ""
        total_valid = max(1, int(np.sum(valid_mask)))

        result: dict[str, dict[str, Any]] = {}
        for regime in self.REGIME_NAMES:
            mask = regime_labels == regime
            count = int(np.sum(mask))
            if count == 0:
                result[regime] = {
                    "days": 0,
                    "pct": 0.0,
                    "avg_return": 0.0,
                    "std_return": 0.0,
                    "sharpe": 0.0,
                }
                continue

            regime_rets = returns[mask]
            avg_ret = float(np.mean(regime_rets))
            std_ret = float(np.std(regime_rets, ddof=1))

            result[regime] = {
                "days": count,
                "pct": round(count / total_valid, 4),
                "avg_return": round(avg_ret, 6),
                "std_return": round(std_ret, 6),
                "sharpe": round(
                    avg_ret / (std_ret + 1e-9) * np.sqrt(252), 4
                ),
            }

        return result

    @staticmethod
    def predominant_regime(
        regime_labels: np.ndarray,
        start_idx: int,
        end_idx: int,
    ) -> str:
        """
        Returns the most frequent regime in [start_idx, end_idx] window.

        Used in WFA scenario: for each test window, find the dominant market regime,
        then classify the window into the corresponding quadrant for group analysis.

        Args:
            regime_labels: tag array output by classify()
            start_idx: starting index (inclusive)
            end_idx: ending index (inclusive)

        Returns:
            Dominant regime name. Returns "" if window has no valid tags.
        """
        window_labels = regime_labels[start_idx : end_idx + 1]
        valid = window_labels[window_labels != ""]
        if len(valid) == 0:
            return ""
        values, counts = np.unique(valid, return_counts=True)
        return str(values[int(np.argmax(counts))])

    @staticmethod
    def wfa_breakdown(
        window_results: list,
        close_prices: np.ndarray,
        total_start: Any,
        total_end: Any,
        mr: MarketRegime | None = None,
        sharpe_key: str = "test_metric",
        return_key: str = "test_return",
        dd_key: str = "test_max_dd",
    ) -> dict[str, Any]:
        """
        Calculate 4-quadrant regime decomposition for WFA windows.

        For each WFA test window, use predominant_regime to determine its regime,
        then summarize OOS performance metrics by regime, outputting Sharpe, win rate, drawdown per quadrant.

        Args:
            window_results: list of WFA window results (WFOSplitResult or dict)
            close_prices: CompleteClose Price Series
            total_start: backtest start time
            total_end: backtest end time
            mr: MarketRegime instance, uses default params if None
            sharpe_key: field name to read Sharpe from window result
            return_key: field name to read return from window result
            dd_key: field name to read drawdown from window result

        Returns:
            {
                "regimes": {regime_name: {windows, window_count, oos_sharpe, ...}},
                "best_regime": str,
                "worst_regime": str,
                "spread": float,
            }
        """
        if mr is None:
            mr = MarketRegime()

        regime_labels = mr.classify(close_prices)
        total_days = max(1, (total_end - total_start).days) if hasattr(total_start, "timestamp") else len(close_prices)

        # Collect window metrics grouped by dominant regime
        regime_groups: dict[str, dict[str, list]] = {
            r: {"windows": [], "sharpe": [], "ret": [], "dd": []}
            for r in mr.REGIME_NAMES
        }

        for i, r in enumerate(window_results):
            # Compatible with both WFOSplitResult and dict formats
            if hasattr(r, "split"):
                split = r.split
                test_start = split.test_start
                test_end = split.test_end
                # Try both property access and dict access
                sh = getattr(r, sharpe_key, None)
                if sh is None and isinstance(r, dict):
                    sh = r.get(sharpe_key, 0.0)
                ret = getattr(r.test_result, "total_return", 0.0) if hasattr(r, "test_result") and r.test_result else 0.0
                dd = getattr(r.test_result, "max_drawdown", 0.0) if hasattr(r, "test_result") and r.test_result else 0.0
            elif isinstance(r, dict):
                test_start = r.get("test_start")
                test_end = r.get("test_end")
                sh = r.get(sharpe_key, 0.0)
                ret = r.get(return_key, 0.0)
                dd = r.get(dd_key, 0.0)
            else:
                continue

            # Estimate index based on date
            if hasattr(test_start, "timestamp") and total_days > 1:
                start_idx = max(0, int((test_start - total_start).days))
                end_idx = min(total_days - 1, int((test_end - total_start).days))
            else:
                # If date unavailable, skip this window
                continue

            predominant = MarketRegime.predominant_regime(
                regime_labels, start_idx, end_idx
            )
            if not predominant or predominant not in regime_groups:
                continue

            regime_groups[predominant]["windows"].append(i + 1)
            regime_groups[predominant]["sharpe"].append(float(sh))  # type: ignore[arg-type]
            regime_groups[predominant]["ret"].append(float(ret))
            regime_groups[predominant]["dd"].append(float(dd))

        # SummarizeStatistics
        result: dict[str, Any] = {}
        best_sharpe = float("-inf")
        worst_sharpe = float("inf")
        best_regime = "N/A"
        worst_regime = "N/A"

        for regime, data in regime_groups.items():
            if not data["windows"]:
                continue
            n_wins = len(data["windows"])
            avg_sharpe = float(np.mean(data["sharpe"]))
            avg_ret = float(np.mean(data["ret"]))
            avg_dd = float(np.mean([abs(v) for v in data["dd"]]))
            win_rate = sum(1 for v in data["ret"] if v > 0) / max(n_wins, 1)

            result[regime] = {
                "windows": data["windows"],
                "window_count": n_wins,
                "pct": round(n_wins / max(sum(len(g["windows"]) for g in regime_groups.values()), 1), 4),
                "oos_sharpe": round(avg_sharpe, 4),
                "oos_win_rate": round(win_rate, 4),
                "oos_return": round(avg_ret, 4),
                "oos_max_dd": round(avg_dd, 4),
            }

            if avg_sharpe > best_sharpe:
                best_sharpe = avg_sharpe
                best_regime = regime
            if avg_sharpe < worst_sharpe and avg_sharpe != 0:
                worst_sharpe = avg_sharpe
                worst_regime = regime

        return {
            "regimes": result,
            "best_regime": best_regime,
            "worst_regime": worst_regime,
            "spread": round(
                best_sharpe - worst_sharpe
                if best_sharpe != float("-inf") and worst_sharpe != float("inf")
                else 0.0,
                4,
            ),
        }


# Module-level constant for frozen dataclass
MarketRegime.REGIME_NAMES = [
    "bull_high_vol",
    "bull_low_vol",
    "bear_high_vol",
    "bear_low_vol",
]
