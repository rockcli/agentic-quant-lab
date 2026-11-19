"""Volume-price-analysis confirmation filters extracted from the signal engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass
class VPAConfirmResult:
    """VPA ConfirmResult"""
    confirmed: bool
    reason: str             # "validation" | "anomaly"
    detail: str             # Detailed metrics
    vol_ratio: float        # Volume ratio (current volume / recent 5-day average volume)
    wick_pct: float         # shadow/wickProportion


def vpa_confirm(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    idx: int,
    side: str = "buy",
    vol_lookback: int = 5,
    vol_mult: float = 1.0,          # Fix: 1.2->1.0 relax volume increase threshold, improve VPA pass rate
    wick_threshold: float = 0.50,   # Fix: 0.30→0.50 relax shadow tolerance
) -> VPAConfirmResult:
    """
    VPA (Volume Price Analysis) Volume-Price ConfirmationFilter

    Based on Coulling's "Volume Price Analysis" three-entry core rules:
    1. Volume IncreaseValidatePricemovement（effort vs result）
    2. price-volume alignment = confirm (validation), price-volume divergence = reject (anomaly)
    3. shadow/wick analysis to avoid false breakouts

    Args:
        opens/highs/lows/closes/volumes: OHLCV Array
        idx: Current bar Index
        side: "buy" or "sell"
        vol_lookback: Volume mean reference window
        vol_mult: Volume IncreasemultipleThreshold
        wick_threshold: shadow/wickProportionUpper Limit
    Returns:
        VPAConfirmResult
    """
    if idx < vol_lookback or idx >= len(closes):
        return VPAConfirmResult(
            confirmed=False, reason="insufficient_data",
            detail="insufficient data", vol_ratio=0, wick_pct=0,
        )

    bar_close = closes[idx]
    bar_open = opens[idx]
    bar_high = highs[idx]
    bar_low = lows[idx]
    bar_vol = volumes[idx]

    prev_vols = volumes[idx - vol_lookback : idx]
    avg_vol = np.mean(prev_vols) if len(prev_vols) > 0 else 1.0
    vol_ratio = bar_vol / avg_vol if avg_vol > 0 else 0

    # Rule 1: Volume expansion confirms price movement direction
    volume_increase = bar_vol > avg_vol * vol_mult

    # Rule 2: PricemovementValidatedirection
    if side == "buy":
        price_up = bar_close > bar_open           # bullish candle
        higher_close = bar_close > closes[idx - 1]  # higher than previous day
    else:
        price_up = bar_close < bar_open
        higher_close = bar_close < closes[idx - 1]

    # Rule 3: shadow/wickAnalyze
    total_range = bar_high - bar_low
    wick_pct = 0.0
    if total_range > 0:
        if side == "buy":
            wick_pct = (bar_high - max(bar_close, bar_open)) / total_range
        else:
            wick_pct = (min(bar_close, bar_open) - bar_low) / total_range
    wick_ok = wick_pct < wick_threshold

    # Coulling's core principle
    if volume_increase and price_up and higher_close and wick_ok:
        return VPAConfirmResult(
            confirmed=True, reason="validation",
            detail=f"vol={bar_vol:.0f}(avg={avg_vol:.0f}),wick={wick_pct:.1%}",
            vol_ratio=vol_ratio, wick_pct=wick_pct,
        )

    # DiagnoserejectReason
    reasons = []
    if not volume_increase:
        reasons.append(f"low_vol({vol_ratio:.1f}x)")
    if not price_up:
        reasons.append("wrong_color")
    if not higher_close:
        reasons.append("no_higher_close")
    if not wick_ok:
        reasons.append("long_wick")

    return VPAConfirmResult(
        confirmed=False, reason="anomaly",
        detail=",".join(reasons),
        vol_ratio=vol_ratio, wick_pct=wick_pct,
    )


def apply_vpa_filter(
    entries: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    vol_mult: float = 1.0,          # Fix: 1.2→1.0 relaxVolume IncreaseThreshold
    wick_threshold: float = 0.50,   # Fix: 0.30→0.50 relax shadow tolerance
) -> tuple[np.ndarray, list[VPAConfirmResult]]:
    """
    Apply VPA double-confirmation filter to SMA crossover signals

    Only entry signals confirmed as valid by VPA are preserved,
    Entry signals rejected as anomalies are cancelled.

    Args:
        entries: SMA Crossovergenerated entry SignalArray
        opens/highs/lows/closes/volumes: OHLCV
        vol_mult: Volume IncreasemultipleThreshold
        wick_threshold: shadow/wickProportionUpper Limit
    Returns:
        (filtered_entries, confirm_results): filtered entry + confirmation info for each signal
    """
    entry_indices = np.where(entries)[0]
    filtered = np.zeros_like(entries)
    confirm_results: list[VPAConfirmResult] = []

    for idx in entry_indices:
        result = vpa_confirm(
            opens, highs, lows, closes, volumes,
            idx=idx, side="buy",
            vol_mult=vol_mult, wick_threshold=wick_threshold,
        )
        confirm_results.append(result)
        if result.confirmed:
            filtered[idx] = True

    return filtered, confirm_results


# ═══════════════════════════ Adaptive ATR multiple ═══════════════════════

def _vpa_confirm_upper_third(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    idx: int,
    side: str = "buy",
    vol_lookback: int = 20,
) -> bool:
    """
    Simplified VPA volume-price confirmation (for incremental evaluation, does not modify existing vpa_confirm)

    Simplified implementation of Coulling volume-price analysis core principles:
    1. Volume validation: volume > 20-day average (volume confirms price action)
    2. Price level validation: long signal close in upper third of day's range (bulls control the close)

    Args:
        opens/highs/lows/closes/volumes: OHLCV Array
        idx: Current bar Index
        side: "buy" | "sell"
        vol_lookback: Volume reference window (default 20)

    Returns:
        True indicates volume-price confirmation passed
    """
    if idx < vol_lookback or idx >= len(closes):
        return False

    # Condition 1: Volume > average volume (20-day moving average)
    avg_vol = float(np.mean(volumes[idx - vol_lookback : idx]))
    vol_ok = volumes[idx] > avg_vol

    # Condition 2: Price in the favorable third of today's range
    daily_range = highs[idx] - lows[idx]
    if daily_range > 0:
        if side == "buy":
            # upper third: Close above open + 2/3 of amplitude
            upper_third = lows[idx] + daily_range * 2.0 / 3.0
            price_ok = closes[idx] >= upper_third
        else:
            # lower third: Close below open - 2/3 of amplitude
            lower_third = lows[idx] + daily_range / 3.0
            price_ok = closes[idx] <= lower_third
    else:
        price_ok = False

    return vol_ok and price_ok


def _apply_vpa_filter_upper_third(
    entries: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
) -> np.ndarray:
    """
    Apply simplified VPA filter (volume > 20-day average + close in upper third)

    Only preserve entry signals that satisfy both volume and price conditions; the rest are filtered out.

    Args:
        entries: SMA Crossovergenerated entry SignalBooleanArray
        opens/highs/lows/closes/volumes: OHLCV

    Returns:
        filtered_entries: Entry signals filtered by VPA confirmation
    """
    entry_indices = np.where(entries)[0]
    filtered = np.zeros_like(entries)

    for idx in entry_indices:
        if _vpa_confirm_upper_third(
            opens, highs, lows, closes, volumes, idx, side="buy",
        ):
            filtered[idx] = True

    return filtered

