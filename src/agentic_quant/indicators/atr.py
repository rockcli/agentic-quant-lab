"""Shared Wilder ATR implementation.

This is the single ATR implementation used by live trading and validation so
that TP/SL distances cannot drift between the real-time path and backtests.
"""

from __future__ import annotations

import numpy as np


def wilder_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    window: int = 14,
) -> np.ndarray:
    """Return Wilder-smoothed average true range.

    The first valid value is seeded at index 1 with the first true range;
    subsequent values use alpha = 1 / window.
    """
    n = len(closes)
    atr = np.zeros(n, dtype=float)
    if n < 2:
        return atr

    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ),
    )
    alpha = 1.0 / max(window, 1)
    if len(tr) > 0:
        atr[1] = tr[0]
        for i in range(1, min(n - 1, len(tr))):
            atr[i + 1] = alpha * tr[i] + (1.0 - alpha) * atr[i]
    return atr
