"""Signal-generation indicators extracted from the monolithic signal engine.

This module contains the pure NumPy strategy signal functions: SMA, KAMA,
Donchian, ADX, Bollinger, and numeric Sharpe helpers.
"""

from __future__ import annotations

import numpy as np

def _safe_sharpe(rets: np.ndarray, periods: int = 252) -> float:
    """
    Numerically safe Sharpe calculation.
    Fewer than 2 sample items or zero volatility -> return 0.0, prevent NaN/inf from contaminating downstream sorting.
    """
    if len(rets) < 2:
        return 0.0
    std = float(np.std(rets, ddof=1))
    if std < 1e-12:
        return 0.0
    sharpe = float(np.mean(rets) / std * np.sqrt(periods))
    return sharpe if np.isfinite(sharpe) else 0.0


# ═══════════════════════════ KAMA AdaptiveMoving Average ═══════════════════════════

def kama(
    closes: np.ndarray,
    er_period: int = 10,
    fast_sc: int = 2,
    slow_sc: int = 30,
) -> np.ndarray:
    """
    Kaufman Adaptive Moving Average — Pure numpy vectorized implementation

    Core idea: use Efficiency Ratio to dynamically adjust the smoothing coefficient,
    Follows sensitively in trending markets, automatically dulls in range-bound markets to reduce false breakouts.

    Algorithm:
    1. directionality = |Close[t] - Close[t-er_period]|
    2. Volatility = sum(|Close[i] - Close[i-1]|) over er_period
    3. ER = directionality / Volatility (0=pure noise, 1=pure trend)
    4. SC = [ER * (fastSC - slowSC) + slowSC]^2
    5. KAMA[t] = KAMA[t-1] + SC * (Close[t] - KAMA[t-1])

    Theoretical Basis:Kaufman《Trading Systems and Methods》, Ch.17
    """
    n = len(closes)
    kama_line = np.full(n, np.nan)
    if n < er_period + 2:
        return kama_line

    fastest = 2.0 / (fast_sc + 1.0)
    slowest = 2.0 / (slow_sc + 1.0)
    sc_delta = fastest - slowest

    direction = np.abs(closes[er_period:] - closes[:-er_period])
    abs_diff = np.abs(np.diff(closes))
    cumsum_abs = np.cumsum(np.insert(abs_diff, 0, 0))
    volatility = np.zeros(n - er_period)
    for i in range(er_period, n):
        volatility[i - er_period] = cumsum_abs[i] - cumsum_abs[i - er_period]

    er = np.clip(direction / (volatility + 1e-9), 0.0, 1.0)
    sc = (er * sc_delta + slowest) ** 2

    kama_line[er_period] = closes[er_period]
    for i in range(er_period + 1, n):
        kama_line[i] = kama_line[i - 1] + sc[i - er_period - 1] * (closes[i] - kama_line[i - 1])

    return kama_line


def kama_cross_signals(
    closes: np.ndarray,
    er_period: int = 10,
    fast_sc: int = 2,
    slow_sc: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    KAMA price crossover signal: Price crosses above KAMA -> Buy, crosses below -> Sell

    Advantages compared to SMA crossover:
    - Trending market (ER->1): KAMA follows price quickly, signals are timely
    - Range-bound market (ER->0): KAMA almost flattens, no false signals generated
    """
    n = len(closes)
    kama_line = kama(closes, er_period=er_period, fast_sc=fast_sc, slow_sc=slow_sc)

    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    in_pos = False

    for i in range(1, n):
        if np.isnan(kama_line[i]) or np.isnan(kama_line[i - 1]):
            continue
        curr_long = closes[i] > kama_line[i]
        prev_long = closes[i - 1] > kama_line[i - 1]
        if curr_long and not prev_long and not in_pos:
            entries[i] = True
            in_pos = True
        elif not curr_long and prev_long and in_pos:
            exits[i] = True
            in_pos = False

    return entries, exits


# ═══════════════════════════ SMA CrossoverSignal ═══════════════════════════

def sma_cross_signals(
    closes: np.ndarray,
    fast: int = 10,
    slow: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    SMA dual moving average crossover signal (vectorized implementation)

    Args:
        closes: Close Price Series
        fast: Fast line period (default 10)
        slow: Slow line period (default 20)
    Returns:
        (entries, exits): Boolean array, True indicates golden cross/death cross trigger
    """
    n = len(closes)
    sma_f = np.full(n, np.nan)
    sma_s = np.full(n, np.nan)

    # Cumulative sum method for SMA, O(n) complexity
    cumsum = np.cumsum(np.insert(closes, 0, 0))
    for i in range(fast - 1, n):
        sma_f[i] = (cumsum[i + 1] - cumsum[i - fast + 1]) / fast
    for i in range(slow - 1, n):
        sma_s[i] = (cumsum[i + 1] - cumsum[i - slow + 1]) / slow

    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    in_pos = False

    for i in range(1, n):
        f_valid = not np.isnan(sma_f[i]) and not np.isnan(sma_f[i - 1])
        s_valid = not np.isnan(sma_s[i]) and not np.isnan(sma_s[i - 1])
        if f_valid and s_valid:
            # Golden cross: fast line crosses above slow line from below
            if sma_f[i] > sma_s[i] and sma_f[i - 1] <= sma_s[i - 1]:
                if not in_pos:
                    entries[i] = True
                    in_pos = True
            # Death cross: fast line crosses below slow line from above
            elif sma_f[i] < sma_s[i] and sma_f[i - 1] >= sma_s[i - 1] and in_pos:
                exits[i] = True
                in_pos = False

    return entries, exits


# ═══════════════════════════ Donchian Channel Signal ═══════════════════════════

def donchian_signals(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Donchian Channel BreakoutSignal

    Tradinglogic：
    - Buy: Close breaks out above N-day high (channel upper band)
    - Sell: Close breaks down below N-day low (channel lower band)

    Theoretical Basis:
    - Richard Donchian weekly rule breakout system
    - Trend following strategy, suitable for strong trending markets

    Args:
        highs: Highsequence
        lows: Lowsequence
        closes: Close Price Series
        period: Channel calculation period (default 20)
    Returns:
        (entries, exits): Boolean array, True indicates breakout trigger
    """
    n = len(closes)
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    in_pos = False

    for i in range(period, n):
        upper = np.max(highs[i - period:i])
        lower = np.min(lows[i - period:i])

        if closes[i] > upper and not in_pos:
            entries[i] = True
            in_pos = True
        elif closes[i] < lower and in_pos:
            exits[i] = True
            in_pos = False

    return entries, exits


# ═══════════════════════ ADX TrendstrengthFilter ═══════════════════════

def compute_adx(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """
    Average Directional Index (ADX) — Wilder ClassicImplement

    ADX measures trend strength: ADX > 25 = strong trend, 20-25 = trend forming, <20 = avoid range-bound markets
    Theoretical Basis:Wilder《New Concepts in Technical Trading Systems》
    """
    n = len(highs)
    adx = np.full(n, np.nan)
    if n < period + 2:
        return adx

    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))

    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    alpha = 1.0 / period
    atr = np.zeros(n)
    atr[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, n):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

    smoothed_plus = np.zeros(n)
    smoothed_minus = np.zeros(n)
    smoothed_plus[period] = np.sum(plus_dm[1:period + 1])
    smoothed_minus[period] = np.sum(minus_dm[1:period + 1])
    for i in range(period + 1, n):
        smoothed_plus[i] = alpha * plus_dm[i] + (1 - alpha) * smoothed_plus[i - 1]
        smoothed_minus[i] = alpha * minus_dm[i] + (1 - alpha) * smoothed_minus[i - 1]

    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    for i in range(period, n):
        if atr[i] > 1e-9:
            plus_di[i] = 100.0 * smoothed_plus[i] / atr[i]
            minus_di[i] = 100.0 * smoothed_minus[i] / atr[i]

    dx = np.zeros(n)
    for i in range(period, n):
        denom = plus_di[i] + minus_di[i]
        if denom > 1e-9:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / denom

    dx_valid = dx[period:n]
    adx_start = period * 2 - 1
    if adx_start < n:
        adx[adx_start] = np.mean(dx_valid[:period])
        for i in range(adx_start + 1, n):
            adx[i] = alpha * dx[i] + (1 - alpha) * adx[i - 1]

    return adx


def apply_adx_filter(
    entries: np.ndarray, exits: np.ndarray,
    adx: np.ndarray, threshold: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """ADX trend strength filter: only allow trading when ADX > threshold"""
    filtered_entries = entries.copy()
    for i in range(len(entries)):
        if np.isnan(adx[i]) or adx[i] <= threshold:
            filtered_entries[i] = False
    return filtered_entries, exits


def apply_ma_direction_filter(
    entries: np.ndarray, closes: np.ndarray, ma_period: int = 200,
) -> np.ndarray:
    """long termMoving Averagedirectional filter：only whenPrice > SMA(ma_period) allowed whenBuy"""
    n = len(closes)
    sma = np.full(n, np.nan)
    if n >= ma_period:
        cumsum = np.cumsum(np.insert(closes, 0, 0))
        for i in range(ma_period - 1, n):
            sma[i] = (cumsum[i + 1] - cumsum[i - ma_period + 1]) / ma_period
    filtered = entries.copy()
    for i in range(n):
        if entries[i] and not np.isnan(sma[i]) and closes[i] <= sma[i]:
            filtered[i] = False
    return filtered


def donchian_regime_signals(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 20,
    adx_threshold: float = 20.0,
    ma_period: int = 200,
    adx_window: int = 14,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Donchian ChannelBreakout + ADX Trendstrength + SMA directional filter（three-in-oneSignal）

    Three filters each serve their own purpose:
    - Donchian = capturing breakouts (sensitivity)
    - ADX = confirming trend strength (avoid false breakouts in range-bound markets)
    - 200MA = confirming major trend direction (avoid counter-trend trading)
    """
    entries, exits = donchian_signals(highs, lows, closes, period=period)
    adx = compute_adx(highs, lows, closes, period=adx_window)
    entries, exits = apply_adx_filter(entries, exits, adx, threshold=adx_threshold)
    entries = apply_ma_direction_filter(entries, closes, ma_period=ma_period)
    return entries, exits


# ═══════════════════════════ Bollinger Band Signal ═══════════════════════════

def bollinger_signals(
    closes: np.ndarray,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bollinger Band Mean ReversionSignal

    Tradinglogic：
    - Buy: Close breaks below lower band (oversold price, expecting rebound)
    - Sell: Close breaks above upper band (overbought price, expecting pullback)

    Theoretical Basis:
    - John Bollinger《Bollinger on Bollinger Bands》
    - Mean reversion strategy, suitable for range-bound markets

    Args:
        closes: Close Price Series
        period: Bollinger BandsCyclical（Default 20）
        num_std: Standard Deviationmultiple（Default 2.0）
    Returns:
        (entries, exits): BooleanArray
    """
    n = len(closes)

    # Vectorized calculation of SMA and standard deviation
    sma = np.full(n, np.nan)
    std = np.full(n, np.nan)

    for i in range(period - 1, n):
        window = closes[i - period + 1:i + 1]
        sma[i] = np.mean(window)
        std[i] = np.std(window, ddof=1)

    upper = sma + num_std * std
    lower = sma - num_std * std

    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    in_pos = False

    for i in range(1, n):
        if np.isnan(lower[i]) or np.isnan(upper[i]):
            continue
        if closes[i] < lower[i] and not in_pos:
            entries[i] = True
            in_pos = True
        elif closes[i] > upper[i] and in_pos:
            exits[i] = True
            in_pos = False

    return entries, exits


# ═══════════════════════════ VPA Volume-Price Confirmation ═══════════════════════════

def compute_adaptive_atr_multipliers(
    closes: np.ndarray,
    lookback: int = 252,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate adaptive ATR multiple adjustment coefficient (based on rolling volatility percentile)

    Strategy Logic：
    - High volatility (>80th percentile): take profit multiple x 0.8, stop loss multiple x 1.2
        (High-vol environment has poor trend persistence; tighten TP for quick entries/exits, relax SL for larger noise)
    - Low volatility (<20th percentile): take profit multiple x 1.2, stop loss multiple x 0.8
        (Low-vol environment has good trend persistence; let profits run by relaxing TP, strictly control downside by tightening SL)
    - Normal volatility (20-80th percentile): keep unchanged

    Theoretical Basis:
    - Chande《Beyond Technical Analysis》VolatilityAdaptiveExit
    - Volatility clustering effect: high vol follows high vol, low vol follows low vol
    - Using expanding window percentile, adaptive ability accumulates over time

    Args:
        closes: Close Price Series
        lookback: Volatility calculation window (default 252 trading days ~ 1 year)
    Returns:
        (pt_mult_adj, sl_mult_adj): Adjustment coefficient arrays, same length as closes
            Effective values start from index lookback; before that, 1.0 (no adjustment)
    """
    n = len(closes)
    pt_adj = np.ones(n, dtype=float)
    sl_adj = np.ones(n, dtype=float)

    if n < lookback + 5:
        return pt_adj, sl_adj

    # Daily simple return
    daily_rets = np.diff(closes) / (closes[:-1] + 1e-9)

    # Rolling annualized volatility (lookback-day window)
    rolling_vol = np.full(n, np.nan)
    for i in range(lookback, n):
        window = daily_rets[i - lookback : i]
        rolling_vol[i] = float(np.std(window, ddof=1)) * np.sqrt(252)

    # Expanding window percentile: effective from day lookback onward
    for i in range(lookback, n):
        current_vol = rolling_vol[i]
        if np.isnan(current_vol):
            continue

        # Fetch all historical volatility values up to current (expanding window)
        hist = rolling_vol[lookback : i + 1]
        if len(hist) < 10:
            continue

        # Percentile: rank of current value among historical values
        percentile = float(np.mean(hist <= current_vol))

        if percentile > 0.80:
            pt_adj[i] = 0.8   # High vol: tighten take profit
            sl_adj[i] = 1.2   # High vol: relax stop loss
        elif percentile < 0.20:
            pt_adj[i] = 1.2   # Low vol: relax take profit
            sl_adj[i] = 0.8   # Low vol: tighten stop loss
        # 20%-80% percentile: keep unchanged (pt_adj=sl_adj=1.0)

    return pt_adj, sl_adj


# ═══════════════════════════ Triple-Barrier Tag ═══════════════════

