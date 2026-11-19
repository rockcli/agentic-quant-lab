"""
ExpandingPerformanceMetrics

Reference:
- Trading Systems and Methods (Kaufman), Ch.21
- Advances in Financial Machine Learning (Lopez de Prado)
"""

from __future__ import annotations

import math

import polars as pl


def calmar_ratio(returns: pl.Series, max_drawdown: float, ann_factor: int = 252) -> float:
    """
    Calmar Ratio = Annualized Return / |Max Drawdown|
    """
    if max_drawdown >= 0 or returns.is_empty():
        return 0.0
    ann_return = returns.mean() * ann_factor  # type: ignore[operator]
    return ann_return / abs(max_drawdown)  # type: ignore[operator,return-value]


def sortino_ratio(returns: pl.Series, ann_factor: int = 252) -> float:
    """
    Sortino Ratio = Annualized Return / Downside Standard Deviation
    Only penalizes downside volatility.
    """
    if returns.is_empty():
        return 0.0
    downside = returns.filter(returns < 0)
    if downside.is_empty() or downside.std() == 0:
        return 0.0
    return (returns.mean() * ann_factor) / (downside.std() * math.sqrt(ann_factor))  # type: ignore[operator,return-value]


def omega_ratio(returns: pl.Series, threshold: float = 0.0) -> float:
    """
    Omega Ratio = Gains above threshold / Losses below threshold
    Risk-adjusted return metric under non-normal assumptions.
    """
    if returns.is_empty():
        return 0.0
    excess = returns - threshold
    gains = excess.filter(excess > 0).sum()
    losses = abs(excess.filter(excess < 0).sum())
    return gains / losses if losses > 1e-9 else float("inf")  # type: ignore[operator,return-value]


def deflated_sharpe(
    sharpe: float,
    skewness: float,
    kurtosis: float,
    n_trials: int,
    n_samples: int = 252,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado).

    The returned value is a probability-like confidence in [0, 1]: the chance
    that the observed Sharpe is greater than the expected maximum Sharpe from
    ``n_trials`` random trials, after accounting for non-normal return
    skewness/kurtosis.

    Args:
        sharpe: observed (annualized) Sharpe
        skewness: return skewness
        kurtosis: return kurtosis (excess; normal = 0)
        n_trials: number of independent strategy/parameter trials
        n_samples: number of observations used to compute the Sharpe
    Returns:
        Deflated Sharpe confidence in [0, 1]
    """
    if n_trials <= 1 or n_samples <= 1:
        return 1.0 if sharpe > 0 else 0.5

    non_excess_kurtosis = kurtosis + 3.0
    sr_var = (
        1.0
        - skewness * sharpe
        + (non_excess_kurtosis - 1.0) / 4.0 * sharpe**2
    ) / (n_samples - 1)
    if sr_var <= 0:
        # Degenerate variance: the distribution is too narrow to reject H0.
        return 1.0 if sharpe > 0 else 0.5

    sr_se = math.sqrt(sr_var)
    # Expected maximum Sharpe under N independent standard-normal trials
    # (extreme-value approximation used by Bailey & Lopez de Prado).
    expected_max = sr_se * math.sqrt(2.0 * math.log(float(n_trials)))
    z = (sharpe - expected_max) / sr_se
    dsr = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return max(0.0, min(1.0, float(dsr)))


def probabilistic_sharpe(
    sharpe: float, benchmark: float, skewness: float, kurtosis: float, n: int
) -> float:
    """
    Probabilistic Sharpe Ratio (PSR)
    Probability of observing Sharpe > benchmark.
    Reference: Advances in Financial Machine Learning, Ch.14

    Args:
        sharpe: observed Sharpe
        benchmark: benchmark Sharpe (e.g., 0 or 1)
        skewness: Skewness
        kurtosis: Kurtosis
        n: number of samples
    Returns:
        probability value (0~1)
    """
    if n <= 1 or sharpe < benchmark:
        return 0.0
    # kurtosis passed here is excess kurtosis (normal = 0); PSR formula needs
    # non-excess kurtosis, so add 3 before using it.
    non_excess_kurtosis = kurtosis + 3.0
    var = (
        1
        - skewness * sharpe
        + (non_excess_kurtosis - 1) / 4 * sharpe**2
    ) / (n - 1)
    if var <= 0:
        return 1.0 if sharpe > benchmark else 0.5
    z = (sharpe - benchmark) * math.sqrt(n - 1) / math.sqrt(var)
    # Standard normal CDF approximation
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def skewness_kurtosis(returns: pl.Series) -> tuple[float, float]:
    """Return skewness and kurtosis."""
    if returns.is_empty():
        return 0.0, 3.0
    return float(returns.skew()), float(returns.kurtosis())  # type: ignore[arg-type]
