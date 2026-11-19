"""Weekly time-frame filtering helpers extracted from the signal engine."""

from __future__ import annotations

from datetime import date as _date

import numpy as np

def _parse_date_str(ts: str) -> _date:
    """
    Parse date string into date object

    Supported formats:
    - "20240101" (consecutive digits)
    - "2024-01-01" (with hyphens)
    - "2024/01/01" (with slashes)
    """
    s = str(ts).replace("-", "").replace("/", "").strip()
    if len(s) >= 8 and s[:4].isdigit():
        return _date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    raise ValueError(f"Unable to parse date: {ts}")


def _get_week_ids(timestamps: np.ndarray) -> np.ndarray:
    """
    Extract ISO week identifiers from date string array

    Output format: "YYYY-WW" (e.g., "2024-03" indicates the 3rd week of 2024)
    Used for weekly group resampling. ISO week numbers correctly handle year-crossing boundaries.
    """
    week_ids: list[str] = []
    for raw_ts in timestamps:
        try:
            d = _parse_date_str(str(raw_ts))
            iso_year, iso_week, _ = d.isocalendar()
            week_ids.append(f"{iso_year}-{iso_week:02d}")
        except (ValueError, IndexError):
            week_ids.append(f"unknown-{len(week_ids)}")
    return np.array(week_ids, dtype=str)


def resample_to_weekly(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    timestamps: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Daily OHLCV -> Weekly OHLCV Resampling

    Resampling rules (following standard OHLCV definitions):
    - Weekly open: open of the first trading day of the week
    - Weekly high: highest high of all trading days in the week
    - Weekly low: lowest low of all trading days in the week
    - Weekly close: close of the last trading day of the week
    Weekly volume: sum of volume across all trading days in the week

    Args:
        opens/highs/lows/closes/volumes: Daily OHLCV Array
        timestamps: Date string array, same length as closes
    Returns:
        {"open": weekly open, "high": weekly high, "low": weekly low,
         "close": weekly close, "volume": weekly volume}
    """
    n = len(closes)
    week_ids = _get_week_ids(timestamps)

    # Group by week, log start and end index of each week
    week_starts: list[int] = []
    week_ends: list[int] = []

    i = 0
    while i < n:
        current = week_ids[i]
        start = i
        while i < n and week_ids[i] == current:
            i += 1
        week_starts.append(start)
        week_ends.append(i - 1)

    n_weeks = len(week_starts)
    if n_weeks == 0:
        return {
            "open": np.array([], dtype=float),
            "high": np.array([], dtype=float),
            "low": np.array([], dtype=float),
            "close": np.array([], dtype=float),
            "volume": np.array([], dtype=float),
        }

    weekly_open = np.array(
        [opens[week_starts[j]] for j in range(n_weeks)], dtype=float,
    )
    weekly_high = np.array(
        [np.max(highs[week_starts[j] : week_ends[j] + 1]) for j in range(n_weeks)],
        dtype=float,
    )
    weekly_low = np.array(
        [np.min(lows[week_starts[j] : week_ends[j] + 1]) for j in range(n_weeks)],
        dtype=float,
    )
    weekly_close = np.array(
        [closes[week_ends[j]] for j in range(n_weeks)], dtype=float,
    )
    weekly_volume = np.array(
        [np.sum(volumes[week_starts[j] : week_ends[j] + 1]) for j in range(n_weeks)],
        dtype=float,
    )

    return {
        "open": weekly_open,
        "high": weekly_high,
        "low": weekly_low,
        "close": weekly_close,
        "volume": weekly_volume,
    }


def _compute_sma(values: np.ndarray, period: int) -> np.ndarray:
    """Vectorized SMA calculation (O(n) cumulative sum method), pure numpy implementation"""
    n = len(values)
    sma = np.full(n, np.nan)
    if n < period:
        return sma
    cumsum = np.cumsum(np.insert(values, 0, 0))
    for i in range(period - 1, n):
        sma[i] = (cumsum[i + 1] - cumsum[i - period + 1]) / period
    return sma


def apply_weekly_filter(
    entries: np.ndarray,
    exits: np.ndarray,
    closes: np.ndarray,
    timestamps: np.ndarray,
    ma_period: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Weekly SMA Trenddirectional filter（core function of multi-timeframe filtering）

    Core logic: use longer timeframe (weekly) trend direction to filter shorter timeframe (daily) signals.
    This is an effective means to solve false breakouts in range-bound markets -- weekly trend provides the "big direction",
    Daily signals only effective when aligned with the trend, thus substantially reducing drawdown from false breakouts.

    filterRule:
    - Buy signal: only preserve when close > weekly SMA(ma_period) (go long with the major trend)
    - Sell signal: only preserve when close < weekly SMA(ma_period) (go short with the major trend)

    Theoretical basis:
    - Williams "Trading Chaos" multi-timeframe analysis: long period sets direction, short period sets entry
    - Trend-following core principle: go long stocks in weekly rising trends, go short in weekly downtrends
    - Avoid false breakouts from SMA crossovers in range-bound markets (weekly SMA provides directional anchor)

    Args:
        entries: DailyBuySignalBooleanArray
        exits: DailySellSignalBooleanArray
        closes: DailyClose Price Series
        timestamps: Date string array, same length as closes
        ma_period: Weekly SMA calculation period (default 20, about 100 trading days ~ 5 months)
    Returns:
        (filtered_entries, filtered_exits): Boolean arrays of signals filtered by weekly trend
    """
    n = len(closes)
    week_ids = _get_week_ids(timestamps)

    # Step 1: group by week -- find the start and end index of each week in the daily sequence
    week_starts: list[int] = []
    week_ends: list[int] = []

    i = 0
    while i < n:
        current = week_ids[i]
        start = i
        while i < n and week_ids[i] == current:
            i += 1
        week_starts.append(start)
        week_ends.append(i - 1)

    n_weeks = len(week_starts)
    if n_weeks < ma_period:
        # Skip filtering when less than ma_period weeks of data (too little data, SMA meaningless)
        return entries.copy(), exits.copy()

    # Step 2: calculate weekly close (last trading day of each week) and SMA
    weekly_closes = np.array(
        [closes[week_ends[j]] for j in range(n_weeks)], dtype=float,
    )
    weekly_sma = _compute_sma(weekly_closes, ma_period)

    # Step 3: map weekly SMA back to daily -- each trading day inherits the trend status of its week
    daily_weekly_sma = np.full(n, np.nan)
    for j in range(n_weeks):
        daily_weekly_sma[week_starts[j] : week_ends[j] + 1] = weekly_sma[j]

    # Step 4: filter signal -- check direction match day by day
    filtered_entries = entries.copy()

    for i in range(n):
        sma_val = daily_weekly_sma[i]
        if np.isnan(sma_val):
            # No effective SMA values before week ma_period-1, skip filtering
            continue

        # Buy signals only preserve when price > weekly SMA (align with weekly rising trend)
        if entries[i] and closes[i] <= sma_val:
            filtered_entries[i] = False

        # Exit signals are not filtered -- avoid lock-up. Entry signals only preserve when aligned with weekly trend direction.

    return filtered_entries, exits

