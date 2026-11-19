"""
Overfitting Detection — De Prado CSCV CompleteImplement

References:
- Advances in Financial Machine Learning (Lopez de Prado), Ch.11-14
- Testing and Tuning Market Trading Systems (Masters)
- AlgoXpert Alpha Research Framework (arXiv:2603.09219)

CoreConcept:
- PBO (Probability of Backtest Overfitting): probability that the IS optimal strategy performs below median in OOS
- CSCV (Combinatorially Symmetric Cross-Validation): estimate PBO through repeated random IS/OOS splits
- Parameter Plateau: performance does not collapse after small parameter perturbations, indicating strategy stability
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────── basic/foundation PBO ────────────────────────────


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
) -> float:
    """
    Simplified PBO: single IS/OOS 50/50 split.

    Args:
        returns_matrix: (n_strategies, n_periods) return matrix for each strategy and period
    Returns:
        PBO value (0~1), higher = greater overfitting risk
    """
    if returns_matrix.ndim != 2:
        raise ValueError("returns_matrix must be a 2D array")
    n_strategies, n_periods = returns_matrix.shape
    if n_strategies < 2 or n_periods < 2:
        return 0.0

    mid = n_periods // 2
    is_returns = returns_matrix[:, :mid]
    oos_returns = returns_matrix[:, mid:]

    is_sharpe = np.mean(is_returns, axis=1) / (np.std(is_returns, axis=1) + 1e-9)
    best_is_idx = int(np.argmax(is_sharpe))

    oos_sharpe = np.mean(oos_returns, axis=1) / (np.std(oos_returns, axis=1) + 1e-9)
    rank = np.sum(oos_sharpe > oos_sharpe[best_is_idx])
    return rank / n_strategies  # type: ignore[no-any-return]


# ──────────────────────── CSCV CompleteImplement ──────────────────────────


@dataclass
class PBOResult:
    """PBO testCompleteResult"""

    pbo: float
    """Overfitting probability (0~1), <0.10 is excellent, >0.30 needs attention"""
    cscv_combinations: int
    """Number of CSCV combinations"""
    logit_scores: list[float]
    """Logit score for each combination group"""
    is_best_rank_quantiles: list[float]
    """Rank quantile of IS optimum in OOS"""
    performance_degradation: float
    """Median IS-to-OOS performance decay rate (based on CSCV combinations)"""
    is_degraded: bool
    """Whether significant performance degradation exists"""
    grade: str
    """Comprehensive rating: A(excellent)/B(acceptable)/C(needs improvement)/D(overfitting)"""
    n_blocks: int = 0
    """Number of time blocks (S groups), used by new CSCV"""
    dsr: float = 0.0
    """Deflated Sharpe Ratio (Bailey & De Prado): multiple testing adjustment for significance"""


def cscv_pbo(
    returns_matrix: np.ndarray,
    n_combinations: int = 100,
    metric_fn: str = "sharpe",
    random_seed: int = 42,
    n_blocks: int = 16,
) -> PBOResult:
    """
    CSCV (Combinatorially Symmetric Cross-Validation) -- standard implementation from De Prado AFML Ch.11/14

    Algorithm steps (De Prado, AFML Ch.11 "Strategy Selection"):
    1. Split T observations chronologically into S equal-length subsequences (blocks), each preserving internal order
    2. Each time randomly select S/2 blocks as IS (concatenated and sorted by time), remaining S/2 blocks as OOS
    3. Find the optimal strategy in IS (highest Sharpe), log its rank quantile in OOS
    4. Repeat n_combinations times, PBO = frequency of IS optimal ranking below median in OOS

    Key correction (2026-05-15):
    - Old version used rng.shuffle(individual days), destroying temporal structure, violating De Prado original
      "The observations in each subset preserve the original time sequence.
       The random sampling is done on the relatively uncorrelated subsets,
       rather than on the observations."
    - New version uses time block random sampling, blocks preserve original order

    Args:
        returns_matrix: (n_strategies, n_periods) ReturnMatrix
        n_combinations: number of combination samples, recommend >= 100
        metric_fn: PerformanceMetrics ("sharpe" | "mean_return" | "calmar")
        random_seed: random seed
        n_blocks: Number of time blocks S (default 16), recommended 10~20
    """
    rng = np.random.RandomState(random_seed)
    n_strategies, n_periods = returns_matrix.shape

    if n_strategies < 2 or n_periods < 4:
        return PBOResult(
            pbo=0.0,
            cscv_combinations=0,
            n_blocks=0,
            logit_scores=[],
            is_best_rank_quantiles=[],
            performance_degradation=0.0,
            dsr=0.0,
            is_degraded=False,
            grade="N/A",
        )

    # ─Time block partitioning─
    # Each block at least 5 days to preserve autocorrelation, S is even
    n_blocks = min(n_blocks, n_periods // 5)
    if n_blocks < 4:
        n_blocks = max(2, n_periods // 5)
    n_blocks = n_blocks - (n_blocks % 2)  # Ensure S is even
    block_size = n_periods // n_blocks

    blocks = []
    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size if b < n_blocks - 1 else n_periods
        blocks.append(np.arange(start, end))
    half_blocks = n_blocks // 2

    # ── MetricsFunction ──
    def _metric(mat: np.ndarray) -> np.ndarray:
        mean = np.mean(mat, axis=1)
        std = np.std(mat, axis=1) + 1e-9
        if metric_fn == "mean_return":
            return mean  # type: ignore[no-any-return]
        if metric_fn == "calmar":
            cumulative = np.cumsum(mat, axis=1)
            peak = np.maximum.accumulate(cumulative, axis=1)
            dd = peak - cumulative
            max_dd = np.max(dd, axis=1) + 1e-9
            return mean / max_dd  # type: ignore[no-any-return]
        return mean / std  # type: ignore[no-any-return]  # sharpe

    # ─CSCV main loop─
    logit_scores: list[float] = []
    rank_quantiles: list[float] = []
    degradations: list[float] = []

    for _ in range(n_combinations):
        # Randomly shuffle block order, first S/2 blocks = IS, last S/2 blocks = OOS
        rng.shuffle(blocks)
        is_idx = np.concatenate(blocks[:half_blocks])
        oos_idx = np.concatenate(blocks[half_blocks:])

        # Sort by time (blocks already have internal order, sorting resumes global order direction)
        is_idx = np.sort(is_idx)
        oos_idx = np.sort(oos_idx)

        is_mat = returns_matrix[:, is_idx]
        oos_mat = returns_matrix[:, oos_idx]

        is_metric = _metric(is_mat)
        oos_metric = _metric(oos_mat)

        best_is = int(np.argmax(is_metric))

        # Rank quantile of IS optimal in OOS
        rank = np.sum(oos_metric > oos_metric[best_is])
        quantile = rank / n_strategies
        rank_quantiles.append(quantile)

        # Logit: λ = ln(ω/(1-ω)), ω ∈ (0,1)
        lam = np.clip(quantile, 1e-9, 1 - 1e-9)
        logit = float(np.log(lam / (1 - lam)))
        logit_scores.append(logit)

        # Performance decay for this combination
        d = max(0.0, 1.0 - (oos_metric[best_is] / (is_metric[best_is] + 1e-9)))
        degradations.append(d)

    # ── PBO ──
    pbo = sum(1.0 for q in rank_quantiles if q > 0.5) / len(rank_quantiles)

    # ─Performance decay (median, more robust than single split)─
    degradation = float(np.median(degradations)) if degradations else 0.0

    # ── Deflated Sharpe Ratio (Bailey & De Prado, 2017) ──
    dsr = _compute_deflated_sharpe(rank_quantiles, logit_scores, n_strategies)

    # ── ComprehensiveRating ──
    # PBO is the primary metric, DSR is auxiliary confidence
    if pbo < 0.10:
        grade = "A" if dsr >= 0.50 else "B"  # PBO<10% is excellent, DSR does fine-tuning
    elif pbo < 0.20:
        grade = "B" if dsr >= 0.30 else "C"
    elif pbo < 0.30:
        grade = "C"
    else:
        grade = "D"

    return PBOResult(
        pbo=round(pbo, 4),
        cscv_combinations=n_combinations,
        n_blocks=n_blocks,
        logit_scores=[round(v, 4) for v in logit_scores],
        is_best_rank_quantiles=[round(v, 4) for v in rank_quantiles],
        performance_degradation=round(degradation, 4),
        dsr=round(dsr, 4),
        is_degraded=degradation > 0.5,
        grade=grade,
    )


def _compute_deflated_sharpe(
    rank_quantiles: list[float],
    logit_scores: list[float],
    n_strategies: int,
) -> float:
    """
    Calculate Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2017)

    Controls for multiple testing bias: given N strategy trials, pure random search also has probability of producing a "good-looking"
    optimal Sharpe. DSR uses extreme value theory to estimate this false positive threshold, calculating if the observed IS optimal strategy
    is statistically significantly better than random expectation.

    Simplified implementation (suitable for small N scenarios):
    1. Calculate standardized mean of CSCV logit distribution (t-like statistic)
    2. |logit_mean| / (logit_std / sqrt(M)) > threshold -> strategy performance significantly deviates from random
    3. DSR in [0,1], higher indicates less likely to be pure overfitting

    References:
    - Bailey, D. and Lopez de Prado, M. (2017): "The Deflated Sharpe Ratio:
      Correcting for Selection Bias, Backtest Overfitting and Non-Normality"
    - De Prado, AFML Ch.14

    Args:
        rank_quantiles: ranking quantiles for each CSCV combination
        logit_scores: logit values for each CSCV combination
        n_strategies: total number of strategy parameter combinations
    Returns:
        DSR value (0~1)
    """
    if not logit_scores or n_strategies < 2:
        return 0.0

    n_combinations = len(logit_scores)
    logit_std = float(np.std(logit_scores))
    mean_logit = float(np.mean(logit_scores))

    if logit_std < 1e-9:
        # All CSCV combination results consistent: non-zero logit indicates stable extreme ranking, direction explained by PBO alone.
        return 1.0 if abs(mean_logit) > 1e-9 else 0.0

    # Standardized effect size (using absolute value)
    # |mean_logit| measures extremity of strategy performance: regardless of sign, larger |logit| means less likely to be random result
    # PBO separately measures overfitting degree (ratio of quantile > 0.5)
    # DSR measures confidence that "strategy has real alpha", using |logit| rather than sign(logit)
    se = logit_std / np.sqrt(n_combinations)
    z_score = abs(mean_logit) / se

    # Expected maximum z-score (expected maximum of N independent standard normal trials)
    # E[max(Z_i)] ≈ √(2·ln(N))
    expected_max = np.sqrt(2.0 * np.log(float(n_strategies)))

    from math import erf, sqrt

    def norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    # DSR = P( effect > E[max] | H0 )
    adjusted_z = z_score - expected_max
    dsr = norm_cdf(adjusted_z)

    return round(max(0.0, min(1.0, float(dsr))), 4)


# ──────────────────────── Parameter Stability Analysis ──────────────────────────


@dataclass
class ParamStabilityResult:
    """Parameter Stability AnalysisResult"""

    param_name: str
    """Parameter name"""
    values: list[float]
    """Parameter value list"""
    scores: list[float]
    """Corresponding performance metrics"""
    best_value: float
    """Optimal parameter value"""
    best_score: float
    """optimalPerformance"""
    stability_score: float
    """Stability score (0~1), higher = more stable"""
    plateau_width: float
    """Parameter plateau width (ratio of parameter range where performance >= optimal x 0.9)"""
    has_cliff: bool
    """Whether a cliff exists (small parameter change causes performance collapse)"""


class ParameterStabilityAnalyzer:
    """
    Parameter stability analyzer

    Grid-scan strategy parameters to evaluate:
    1. Parameter plateau width -- wide range of parameters can produce near-optimal performance
    2. Cliff detection -- whether small parameter perturbations cause performance collapse
    3. Stability score -- comprehensive parameter sensitivity
    """

    @staticmethod
    def analyze(
        param_values: list[float],
        performance_scores: list[float],
        param_name: str = "param",
    ) -> ParamStabilityResult:
        """Analyze single parameter stability"""
        if len(param_values) < 3 or len(performance_scores) < 3:
            return ParamStabilityResult(
                param_name=param_name,
                values=param_values,
                scores=performance_scores,
                best_value=0,
                best_score=0,
                stability_score=0,
                plateau_width=0,
                has_cliff=False,
            )

        arr_param = np.array(param_values)
        arr_score = np.array(performance_scores)

        best_idx = int(np.argmax(arr_score))
        best_value = arr_param[best_idx]
        best_score = arr_score[best_idx]

        # Plateau threshold: 90% of optimal for positive performance, 10% absolute gap for negative performance to avoid misjudgment.
        threshold = best_score * 0.9 if best_score >= 0 else best_score - 0.1 * abs(best_score)

        # Plateau width: ratio of parameter range where performance >= threshold
        plateau_mask = arr_score >= threshold
        if np.sum(plateau_mask) > 0:
            plateau_range = np.max(arr_param[plateau_mask]) - np.min(arr_param[plateau_mask])
            total_range = np.max(arr_param) - np.min(arr_param)
            plateau_width = plateau_range / total_range if total_range > 0 else 0
        else:
            plateau_width = 0.0

        # Cliff detection: adjacent parameter points where performance change exceeds 50%
        diffs = np.abs(np.diff(arr_score) / (np.abs(arr_score[:-1]) + 1e-9))
        has_cliff = bool(np.any(diffs > 0.5))

        # Stability score: plateau width x no cliff penalty
        stability = plateau_width * (0.5 if has_cliff else 1.0)

        return ParamStabilityResult(
            param_name=param_name,
            values=[float(v) for v in arr_param],
            scores=[float(v) for v in arr_score],
            best_value=float(best_value),
            best_score=float(best_score),
            stability_score=round(float(stability), 4),
            plateau_width=round(float(plateau_width), 4),
            has_cliff=has_cliff,
        )

    @staticmethod
    def analyze_grid(
        param_grid: dict[str, list[float]],
        score_matrix: np.ndarray,
    ) -> list[ParamStabilityResult]:
        """
        Analyze multi-dimensional parameter grid stability.

        Args:
            param_grid: {"fast": [5,10,15], "slow": [20,30,40]}
            score_matrix: (len(param1), len(param2), ...) PerformanceMatrix
        Returns:
            Stability analysis result for each parameter (via marginalizing other dimensions)
        """
        results: list[ParamStabilityResult] = []
        dims = list(param_grid.keys())

        for axis, name in enumerate(dims):
            # Average along other axes to get marginal performance for this parameter
            other_axes = tuple(i for i in range(len(dims)) if i != axis)
            marginal = np.max(score_matrix, axis=other_axes) if other_axes else score_matrix

            vals = param_grid[name]
            scores = list(marginal.flatten()[: len(vals)])
            results.append(ParameterStabilityAnalyzer.analyze(vals, scores, param_name=name))

        return results


# ──────────────────────── ComprehensiveReport ──────────────────────────


def generate_pbo_report(
    pbo_result: PBOResult,
    param_stability: list[ParamStabilityResult],
    strategy_name: str = "",
    output_dir: str | Path | None = None,
) -> str:
    """Generate PBO + ParameterStabilityComprehensive Markdown Report"""
    lines = [
        f"# 🔬 PBO Overfitting TestReport{f' — {strategy_name}' if strategy_name else ''}",
        "",
        f"**testTime**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. PBO Test (CSCV Method)",
        "",
        "| Metric | Value | Standard |",
        "|------|------|------|",
        f"| PBO | {pbo_result.pbo:.2%} | <10% excellent, <20% acceptable, >30% caution |",
        f"| CSCV combinations | {pbo_result.cscv_combinations} | >=100 recommended |",
        f"| Time blocks S | {pbo_result.n_blocks} | Recommended 10~20 |",
        f"| IS→OOS PerformanceDecay | {pbo_result.performance_degradation:.2%} | <50% normal |",
        f"| DSR (Deflated SR) | {pbo_result.dsr:.4f} | >=0.95 significant |",
        f"| Rating | **{pbo_result.grade}** | A=excellent B=acceptable C=needs improvement D=overfitting |",
        "",
    ]

    if pbo_result.is_degraded:
        lines.append(
            "> **Warning**: in-sample to out-of-sample performance decay exceeds 50%, serious overfitting risk!"
        )
    elif pbo_result.pbo < 0.10:
        lines.append(
            "> PBO < 10%, strategy overfitting probability low, can proceed to next phase validation."
        )
    elif pbo_result.pbo < 0.20:
        lines.append(
            "> PBO between 10%-20%, recommend adding OOS validation window before deciding."
        )
    else:
        lines.append(
            "> PBO > 20%, strategy has significant overfitting, needs redesign or additional regularization."
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Parameter Stability Analysis",
            "",
            "| Parameter | Optimal value | Optimal performance | Plateau width | Stability | Cliff? |",
            "|------|--------|----------|----------|--------|--------|",
        ]
    )

    for ps in param_stability:
        cliff = "⚠️ Yes" if ps.has_cliff else "✅ No"
        lines.append(
            f"| {ps.param_name} | {ps.best_value:.1f} | {ps.best_score:.4f} "
            f"| {ps.plateau_width:.1%} | {ps.stability_score:.2%} | {cliff} |"
        )

    lines.append("")
    lines.append("**Interpretation**:")
    lines.append("- Wider plateau width means more robust parameter selection")
    lines.append(
        "- Cliff risk indicates small parameter changes cause performance collapse -- needs to be eliminated"
    )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Acceptance Conclusion",
            "",
        ]
    )

    all_stable = all(not ps.has_cliff for ps in param_stability)
    if pbo_result.grade in ("A", "B") and all_stable:
        lines.append(
            "**Passed**: PBO within acceptable range and parameters stable, suggest proceeding to WFA phase."
        )
    elif pbo_result.grade == "D":
        lines.append(
            "**Not passed**: Overfitting probability too high, need to redesign strategy or add OOS validation."
        )
    else:
        lines.append("**Needs improvement**: risk points exist, suggest fixing and retesting.")

    lines.extend(
        [
            "",
            "> References: De Prado, AFML Ch.14; AlgoXpert Framework (arXiv:2603.09219)",
        ]
    )

    report = "\n".join(lines)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        (out / f"pbo_report_{ts}.md").write_text(report, encoding="utf-8")
        # Also save JSON
        json_data = {
            "pbo": pbo_result.pbo,
            "grade": pbo_result.grade,
            "degradation": pbo_result.performance_degradation,
            "dsr": pbo_result.dsr,
            "n_blocks": pbo_result.n_blocks,
            "param_stability": [
                {
                    "name": ps.param_name,
                    "stability": ps.stability_score,
                    "plateau_width": ps.plateau_width,
                    "has_cliff": ps.has_cliff,
                }
                for ps in param_stability
            ],
            "timestamp": datetime.now().isoformat(),
        }
        (out / f"pbo_report_{ts}.json").write_text(
            json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return report


# ──────────────────────── Batch PBO test ──────────────────────────


@dataclass
class PBOSummary:
    """Batch PBO testSummarizeResult"""

    n_total: int
    """Total underlyings"""
    n_success: int
    """Number of underlyings with successful PBO completion"""
    n_passed: int
    """Number of underlyings with PBO < 0.20"""
    pass_rate: float
    """Pass rate (PBO < 0.20)"""
    pbo_values: list[float]
    """PBO values for all underlyings"""
    pbo_mean: float
    """PBO Mean"""
    pbo_median: float
    """PBO Median"""
    pbo_std: float
    """PBO Standard Deviation"""
    grade_counts: dict[str, int]
    """Count per rating (A/B/C/D/NA)"""
    results: list[tuple[str, PBOResult]]
    """(UnderlyingCode, PBOResult) List"""


def _compute_daily_strategy_returns(
    closes: np.ndarray,
    entries: np.ndarray,
    exits: np.ndarray,
    commission_rate: float = 0.0,
    stamp_duty: float = 0.0,
) -> np.ndarray:
    """
    Rebuild strategy daily return sequence based on close position signal.

    Consistent with run_strategy() internal calculation logic, ensuring that in the MSCV matrix,
    each row (parameter combination) return sequence is consistent with actual strategy position returns.

    Args:
        closes: Close Price Series (n,)
        entries: EntrySignalBooleanArray (n,)
        exits: ExitSignalBooleanArray (n,)
        commission_rate: commission rate (0.025 per 10k), 0 means not included
        stamp_duty: stamp duty rate (0.1%, sell only), 0 means not included
    Returns:
        Strategy daily return sequence (n-1,), length is 1 less than closes
    """
    rets = np.diff(closes) / (closes[:-1] + 1e-9)
    n = len(closes)
    pos = np.zeros(n, dtype=float)
    in_pos = False
    for i in range(n):
        if entries[i] and not in_pos:
            in_pos = True
        elif exits[i] and in_pos:
            in_pos = False
        pos[i] = 1.0 if in_pos else 0.0

    strat_rets = rets * pos[1:]

    # Transaction CostAdjust
    if (commission_rate > 0 or stamp_duty > 0) and len(strat_rets) > 0:
        pos_shift = np.diff(np.concatenate([[0], pos]))
        buy_count = int(np.sum(pos_shift > 0))
        sell_count = int(np.sum(pos_shift < 0))
        total_cost = buy_count * commission_rate + sell_count * (commission_rate + stamp_duty)
        cost_per_day = total_cost / len(strat_rets)
        strat_rets = strat_rets - cost_per_day

    return strat_rets  # type: ignore[no-any-return]


def run_pbo_single(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    param_grid: list[dict[str, int]],
    enable_vpa: bool = False,
    n_combinations: int = 100,
    random_seed: int = 42,
    strategy: str = "sma",
    commission_rate: float = 0.0,
    stamp_duty: float = 0.0,
) -> PBOResult:
    """
    Run PBO overfitting test for a single stock underlying.

    Construct strategy daily returns for each parameter combination into an (n_params, n_days) matrix,
    Passed CSCV Method for testing parameter overfitting across different time windows。

    Args:
        opens/highs/lows/closes/volumes: OHLCV Array
        param_grid: parameter grid, keys depend on strategy
        enable_vpa: whether to enable VPA volume-price filter (default: off)
        n_combinations: number of CSCV combination samples
        random_seed: random seed
        strategy: Strategy Type (sma/kama/donchian/donchian_regime/bollinger)
        commission_rate: commission rate, default 0 (no adjustment), set to 0.00025 to enable 0.025 per 10k fee deduction
        stamp_duty: stamp duty rate, default 0 (no adjustment), set to 0.001 to enable 0.1% fee deduction
    """
    from agentic_quant.validation.signal_engine import run_strategy

    n_strategies = len(param_grid)
    n_days = len(closes)
    if n_strategies < 2 or n_days < 20:
        return PBOResult(
            pbo=0.0,
            cscv_combinations=0,
            n_blocks=0,
            logit_scores=[],
            is_best_rank_quantiles=[],
            performance_degradation=0.0,
            dsr=0.0,
            is_degraded=False,
            grade="N/A",
        )

    # Calculate strategy daily return for each parameter combination
    daily_returns_list: list[np.ndarray] = []
    for p in param_grid:
        try:
            result = run_strategy(
                opens,
                highs,
                lows,
                closes,
                volumes,
                strategy=strategy,
                fast=p.get("fast", 10),
                slow=p.get("slow", 20),
                period=p.get("period", 20),
                num_std=p.get("num_std", p.get("std", 2.0)),
                er_period=p.get("er_period", 10),
                fast_sc=p.get("fast_sc", 2),
                slow_sc=p.get("slow_sc", 30),
                enable_vpa=enable_vpa,
                commission_rate=0.0,  # PBO matrix deducts fees separately, not counted here
                stamp_duty=0.0,
            )
            daily_rets = _compute_daily_strategy_returns(
                closes,
                result.filtered_entries,
                result.exits,
                commission_rate=commission_rate,
                stamp_duty=stamp_duty,
            )
            daily_returns_list.append(daily_rets)
        except Exception as e:
            logger.warning("Parameter %s RunException: %s", p, e)
            daily_returns_list.append(np.zeros(n_days - 1))

    # Build CSCV ReturnMatrix (n_params, n_days-1)
    returns_matrix = np.stack(daily_returns_list)
    n_periods = returns_matrix.shape[1]
    if n_periods < 10:
        logger.warning("Insufficient data period (%d days), skip PBO", n_periods)
        return PBOResult(
            pbo=0.0,
            cscv_combinations=0,
            n_blocks=0,
            logit_scores=[],
            is_best_rank_quantiles=[],
            performance_degradation=0.0,
            dsr=0.0,
            is_degraded=False,
            grade="N/A",
        )

    return cscv_pbo(
        returns_matrix,
        n_combinations=n_combinations,
        metric_fn="sharpe",
        random_seed=random_seed,
    )


def _default_fetch_ohlcv(symbol: str) -> dict[str, np.ndarray] | None:
    """
    Placeholder default data fetcher.

    Live market data fetching is intentionally decoupled from this research library.
    Pass an explicit ``fetch_func`` callable to :func:`run_pbo_batch`.
    """
    raise ValueError("Pass fetch_func explicitly; live data fetching is decoupled")


def run_pbo_batch(
    symbols: list[str],
    param_grid: list[dict[str, int]],
    fetch_func: Callable[[str], dict[str, np.ndarray] | None] | None = None,
    enable_vpa: bool = True,
    n_combinations: int = 100,
    random_seed: int = 42,
    show_progress: bool = True,
    request_delay: float = 0.15,
) -> PBOSummary:
    """
    Run PBO test in batch for multiple underlyings.

    Fetch data via a caller-supplied ``fetch_func`` and calculate PBO for each underlying;
    failure of one does not affect others.

    Args:
        symbols: Underlying code list.
        param_grid: Parameter grid.
        fetch_func: Data fetching callable taking ``symbol`` and returning an OHLCV dict
            (``open``, ``high``, ``low``, ``close``, ``volume`` numpy arrays) or ``None``.
        enable_vpa: Whether to enable VPA filter.
        n_combinations: Number of CSCV combinations.
        random_seed: Random seed start value (increments offset per underlying).
        show_progress: Whether to show progress.
        request_delay: Interval in seconds between two requests.
    """
    if fetch_func is None:
        raise ValueError("Pass fetch_func explicitly; live data fetching is decoupled")

    results: list[tuple[str, PBOResult]] = []
    failed: list[str] = []

    # Progress display: prefer tqdm, fallback to simple print
    pbar = None
    try:
        if show_progress:
            from tqdm import tqdm

            pbar = tqdm(total=len(symbols), desc="PBO Batchtest", unit="Underlying")
    except ImportError:
        pbar = None

    for idx, symbol in enumerate(symbols):
        if pbar:
            pbar.update(1)
            pbar.set_postfix_str(symbol)

        try:
            data = fetch_func(symbol)
            if data is None:
                failed.append(symbol)
                if show_progress and pbar is None:
                    print(f"  [{idx + 1}/{len(symbols)}] {symbol}: Data Fetch Failed")
                continue

            pbo_result = run_pbo_single(
                data["open"],
                data["high"],
                data["low"],
                data["close"],
                data["volume"],
                param_grid=param_grid,
                enable_vpa=enable_vpa,
                n_combinations=n_combinations,
                random_seed=random_seed + idx,
            )
            results.append((symbol, pbo_result))

            if show_progress and pbar is None:
                print(
                    f"  [{idx + 1}/{len(symbols)}] {symbol}: PBO={pbo_result.pbo:.2%} "
                    f"grade={pbo_result.grade}"
                )

            time.sleep(request_delay)

        except Exception as e:
            logger.warning("PBO testException %s: %s", symbol, e)
            failed.append(symbol)

    if pbar:
        pbar.close()

    # SummarizeStatistics
    pbo_values = [r.pbo for _, r in results] if results else []
    grades = Counter(r.grade for _, r in results)
    n_passed = sum(1 for v in pbo_values if v < 0.20)

    return PBOSummary(
        n_total=len(symbols),
        n_success=len(results),
        n_passed=n_passed,
        pass_rate=n_passed / max(len(results), 1),
        pbo_values=pbo_values,
        pbo_mean=float(np.mean(pbo_values)) if pbo_values else 0.0,
        pbo_median=float(np.median(pbo_values)) if pbo_values else 0.0,
        pbo_std=float(np.std(pbo_values)) if pbo_values else 0.0,
        grade_counts=dict(grades),
        results=results,
    )


def generate_batch_pbo_report(
    summary: PBOSummary,
    strategy_name: str = "",
    output_dir: str | Path | None = None,
) -> str:
    """
    GenerateBatch PBO testSummarizeReport（Markdown）。

    Includes PBO distribution histogram (ASCII), pass rate, per-stock details.

    Args:
    summary: SummarizeResult returned by run_pbo_batch()
        strategy_name: Strategy Name（Optional）
    output_dir: output directory, saves both .md and .json when provided
    Returns:
    Markdown format report string
    """
    lines = [
        f"# PBO BatchtestReport{f' — {strategy_name}' if strategy_name else ''}",
        "",
        f"**testTime**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Parameter Grid**: {len(summary.results[0][1].logit_scores) if summary.results else 0} combinations",
        "",
        "---",
        "",
        "## 1. Summary Statistics",
        "",
        "| Metrics | Value |",
        "|------|------|",
        f"| Total underlyings | {summary.n_total} |",
        f"| Successtest | {summary.n_success} |",
        f"| Meeting Criteria (PBO<20%) | {summary.n_passed} / {summary.n_success} ({summary.pass_rate:.1%}) |",
        f"| PBO Mean | {summary.pbo_mean:.2%} |",
        f"| PBO Median | {summary.pbo_median:.2%} |",
        f"| PBO Standard Deviation | {summary.pbo_std:.2%} |",
        "",
    ]

    # Rating distribution
    lines.extend(
        [
            "## 2. Rating Distribution",
            "",
            "| Rating | Count | Proportion | Meaning |",
            "|------|------|------|------|",
        ]
    )
    grade_meanings = {
        "A": "excellent (PBO<10%)",
        "B": "Acceptable (PBO<20%)",
        "C": "Needs improvement (PBO<30%)",
        "D": "Overfitting (PBO>=30%)",
        "N/A": "Insufficient data",
    }
    for grade in ["A", "B", "C", "D", "N/A"]:
        count = summary.grade_counts.get(grade, 0)
        pct = count / max(summary.n_success, 1)
        meaning = grade_meanings.get(grade, "")
        lines.append(f"| **{grade}** | {count} | {pct:.1%} | {meaning} |")

    lines.append("")

    # PBO distribution histogram (ASCII entries)
    lines.extend(
        [
            "## 3. PBO Distribution Histogram",
            "",
            "```",
        ]
    )
    pbo_values = summary.pbo_values
    if pbo_values:
        bins = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        hist, edges = np.histogram(pbo_values, bins=bins)
        max_count = max(hist) if max(hist) > 0 else 1
        bar_width = 40
        for i in range(len(hist)):
            count = hist[i]
            bar_len = max(1, int(count / max_count * bar_width))
            bar = "█" * bar_len
            label = f"[{edges[i]:.1f}, {edges[i + 1]:.1f})"
            lines.append(f"  {label:>12} |{bar} {count}")
    lines.append("```")
    lines.append("")

    # Per-stock PBO details (sorted by PBO descending)
    lines.extend(
        [
            "## 4. Underlying Details",
            "",
            "| Underlying | PBO | PerformanceDecay | Rating |",
            "|------|-----|----------|------|",
        ]
    )
    for symbol, result in sorted(summary.results, key=lambda x: x[1].pbo, reverse=True):
        lines.append(
            f"| {symbol} | {result.pbo:.2%} | {result.performance_degradation:.1%} | "
            f"{result.grade} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    # Conclusion
    if summary.n_success == 0:
        lines.append(
            "**Failed**: All underlying PBO tests failed, please check data source or parameter configuration."
        )
    elif summary.pass_rate >= 0.80:
        lines.append(
            f"**Passed**: pass rate {summary.pass_rate:.1%} >= 80%, "
            f"strategy has PBO < 20% on {summary.n_passed}/{summary.n_success} underlyings, "
            f"overall overfitting risk is low."
        )
    elif summary.pass_rate >= 0.60:
        lines.append(
            f"**Needs monitoring**: pass rate {summary.pass_rate:.1%}, "
            f"PartialUnderlyingOverfitting risk exists（{summary.n_success - summary.n_passed} stocks PBO >= 20%）。"
        )
    else:
        lines.append(
            f"**Failed**: pass rate {summary.pass_rate:.1%} < 60%, "
            f"overall strategy overfitting risk is high (Overfitting)."
        )

    lines.append("")
    lines.append(
        "> References: De Prado, AFML Ch.14; Lopez de Prado, CSCV; Masters, Testing and Tuning"
    )

    report = "\n".join(lines)

    # WriteFile
    md_path = None
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_path = out / f"pbo_batch_{ts}.md"
        md_path.write_text(report, encoding="utf-8")
        logger.info("Batch PBO report saved: %s", md_path)

    # Also save JSON summary for programmatic reading
    if output_dir:
        json_data = {
            "n_total": summary.n_total,
            "n_success": summary.n_success,
            "n_passed": summary.n_passed,
            "pass_rate": summary.pass_rate,
            "pbo_mean": summary.pbo_mean,
            "pbo_median": summary.pbo_median,
            "pbo_std": summary.pbo_std,
            "grade_counts": summary.grade_counts,
            "details": [
                {
                    "symbol": s,
                    "pbo": r.pbo,
                    "grade": r.grade,
                    "degradation": r.performance_degradation,
                }
                for s, r in summary.results
            ],
            "timestamp": datetime.now().isoformat(),
        }
        json_path = out / f"pbo_batch_{ts}.json"
        json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Batch PBO data saved: %s", json_path)

    return report
