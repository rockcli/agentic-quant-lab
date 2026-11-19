"""Triple-barrier labeling extracted from the monolithic signal engine."""

from __future__ import annotations

import numpy as np

from agentic_quant.indicators.atr import wilder_atr

def triple_barrier_labels(
    closes: np.ndarray,
    entry_indices: np.ndarray,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_hold: int = 20,
    atr_window: int = 14,
    adaptive_pt_adj: np.ndarray | None = None,
    adaptive_sl_adj: np.ndarray | None = None,
) -> np.ndarray:
    """
    Triple-Barrier TagGenerate

    For each entry signal, monitor subsequent price path:
    - hit firstTake Profit → label = +1
    - hit firstStop Loss → label = -1
    - Hold to expiry without touching -> label = 0 (determined by final P&L)

    Args:
        closes: Close Price Series
        entry_indices: Bar index where entry signal occurs
        pt_mult: Take Profit ATR multiple（Default 2.0）
        sl_mult: Stop Loss ATR multiple（Default 1.0）
        max_hold: Maximum hold bars (default 20)
        atr_window: ATR CalculateWindow
        adaptive_pt_adj: Adaptive take profit adjustment coefficient array (same length as closes),
            When None, no adjustment; use fixed pt_mult
        adaptive_sl_adj: Adaptive stop loss adjustment coefficient array (same length as closes)
    Returns:
        labels: Label array same length as entry_indices
    """
    n = len(closes)
    labels = np.zeros(len(entry_indices), dtype=int)

    # Calculate ATR
    atr = np.zeros(n)
    if n > 1:
        tr = np.zeros(n)
        # Simplified ATR: approximated using high-low (when single sequence without OHLC)
        # Pass true_range when using actual data
        for i in range(1, n):
            tr[i] = abs(closes[i] - closes[i - 1])
        tr[0] = tr[1] if n > 1 else 0.01
        alpha = 1.0 / atr_window
        atr[0] = tr[0]
        for i in range(1, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

    for j, entry_idx in enumerate(entry_indices):
        if entry_idx >= n - 1:
            labels[j] = 0
            continue

        entry_price = closes[entry_idx]
        current_atr = atr[entry_idx]
        if np.isnan(current_atr) or current_atr <= 0:
            current_atr = entry_price * 0.01  # Fallback 1%

        # Adaptive adjustment: adjust take profit/stop loss multiples based on volatility percentile at entry
        eff_pt_mult = pt_mult
        eff_sl_mult = sl_mult
        if adaptive_pt_adj is not None and entry_idx < len(adaptive_pt_adj):
            eff_pt_mult = pt_mult * adaptive_pt_adj[entry_idx]
        if adaptive_sl_adj is not None and entry_idx < len(adaptive_sl_adj):
            eff_sl_mult = sl_mult * adaptive_sl_adj[entry_idx]

        pt_level = entry_price + eff_pt_mult * current_atr
        sl_level = entry_price - eff_sl_mult * current_atr

        end_idx = min(entry_idx + max_hold, n - 1)
        label = 0

        for t in range(entry_idx + 1, end_idx + 1):
            price = closes[t]
            if price >= pt_level:
                label = 1   # Take Profit
                break
            if price <= sl_level:
                label = -1  # Stop Loss
                break
        else:
            # Time expiry, label by final P&L (>= 0 to avoid break-even being misjudged as loss)
            label = 1 if closes[end_idx] >= entry_price else -1

        labels[j] = label

    return labels


def triple_barrier_labels_ohlc(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    entry_indices: np.ndarray,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_hold: int = 20,
    atr_window: int = 14,
    adaptive_pt_adj: np.ndarray | None = None,
    adaptive_sl_adj: np.ndarray | None = None,
) -> np.ndarray:
    """
    Triple-barrier labeling (using OHLC data, based on actual high/low points)

    More accurate than close-only version, because intraday may touch barriers before retracing.

    Args:
        opens/highs/lows/closes: OHLC Data
        entry_indices: Bar index where entry signal occurs
        pt_mult/sl_mult: Fixed Stop/Stop Loss ATR multiple
        max_hold: maximum hold bars
        atr_window: ATR CalculateWindow
        adaptive_pt_adj: Adaptive take profit adjustment coefficient array (same length as closes),
            when None, no adjustment
        adaptive_sl_adj: Adaptive stop loss adjustment coefficient array (same length as closes)
    Returns:
        labels: Label array same length as entry_indices
    """
    n = len(closes)
    labels = np.zeros(len(entry_indices), dtype=int)

    # Calculate ATR using high-low range (shared Wilder implementation)
    atr = wilder_atr(highs, lows, closes, window=atr_window)

    for j, entry_idx in enumerate(entry_indices):
        if entry_idx >= n - 1:
            labels[j] = 0
            continue

        entry_price = closes[entry_idx]
        current_atr = atr[entry_idx]
        if np.isnan(current_atr) or current_atr <= 0:
            current_atr = entry_price * 0.01

        # Adaptive adjustment: adjust take profit/stop loss multiples based on volatility percentile at entry
        eff_pt_mult = pt_mult
        eff_sl_mult = sl_mult
        if adaptive_pt_adj is not None and entry_idx < len(adaptive_pt_adj):
            eff_pt_mult = pt_mult * adaptive_pt_adj[entry_idx]
        if adaptive_sl_adj is not None and entry_idx < len(adaptive_sl_adj):
            eff_sl_mult = sl_mult * adaptive_sl_adj[entry_idx]

        pt_level = entry_price + eff_pt_mult * current_atr
        sl_level = entry_price - eff_sl_mult * current_atr

        end_idx = min(entry_idx + max_hold, n - 1)
        label = 0

        for t in range(entry_idx + 1, end_idx + 1):
            # Use high/low to judge whether intraday hits first.
            # Conservative tie-break: if both barriers are touched on the same bar,
            # record the stop loss first.
            if lows[t] <= sl_level:
                label = -1
                break
            if highs[t] >= pt_level:
                label = 1
                break
        else:
            label = 1 if closes[end_idx] >= entry_price else -1

        labels[j] = label

    return labels


# ═══════════════════════════ combination engine ═══════════════════════════
