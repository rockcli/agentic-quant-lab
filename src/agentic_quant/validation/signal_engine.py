"""
Pure NumPy Signal Engine: SMA Crossover + VPA Volume-Price Confirmation + Triple-Barrier Labeling

Design Principles:
- Zero Dependency: NumPy only, no polars/vectorbt dependency
- Vectorized: Use NumPy batch operations instead of Python loops where possible
- Composable: Three independent modules, usable standalone or combined

Theoretical Basis:
- De Prado《Advances in Financial Machine Learning》Ch.3 Triple-Barrier, Ch.6 Ensemble
- Masters《Testing and Tuning Market Trading Systems》Parameter Plateau Theory
- Coulling《Volume Price Analysis》Volume-Price Validation Principles
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date
from typing import Any

import numpy as np

from agentic_quant.indicators.atr import wilder_atr
from .indicators import (
    _safe_sharpe,
    apply_adx_filter,
    apply_ma_direction_filter,
    bollinger_signals,
    compute_adx,
    donchian_regime_signals,
    donchian_signals,
    kama,
    kama_cross_signals,
    sma_cross_signals,
)
from .labeling import triple_barrier_labels, triple_barrier_labels_ohlc
from .weekly import _compute_sma, _get_week_ids, _parse_date_str, apply_weekly_filter, resample_to_weekly
from .vpa import (
    VPAConfirmResult,
    _apply_vpa_filter_upper_third,
    _vpa_confirm_upper_third,
    apply_vpa_filter,
    vpa_confirm,
)

# ═══════════════════════════ Helper Functions ═══════════════════════════

@dataclass
class StrategyResult:
    """StrategyBacktest Result"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    avg_hold_days: float
    profit_factor: float
    tb_labels: np.ndarray      # Triple-Barrier Tag
    filtered_entries: np.ndarray  # VPA filteredEntrySignal
    exits: np.ndarray            # ExitSignal
    cost_adjusted_sharpe: float = 0.0  # Annualized Sharpe after deducting commission + stamp duty
    total_cost: float = 0.0            # Total transaction cost (commission + stamp duty)


def run_strategy(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    fast: int = 10,
    slow: int = 20,
    strategy: str = "sma",
    period: int = 20,
    num_std: float = 2.0,
    er_period: int = 10,
    fast_sc: int = 2,
    slow_sc: int = 30,
    adx_threshold: float = 20.0,
    ma_period: int = 200,
    adx_window: int = 14,
    enable_vpa: bool = False,
    vol_mult: float = 1.0,
    enable_weekly_filter: bool = False,
    weekly_ma: int = 20,
    timestamps: np.ndarray | None = None,
    tb_pt_mult: float = 2.0,
    tb_sl_mult: float = 1.0,
    tb_max_hold: int = 20,
    adaptive_exit: bool = False,
    vol_lookback: int = 252,
    market_closes: np.ndarray | None = None,
    commission_rate: float = 0.00025,
    stamp_duty: float = 0.001,
) -> StrategyResult:
    """
    RunCompleteSignalPipeline：StrategySignal Generation → Regimefilter → Triple-Barrier Tag → PerformanceStatistics

    Supports five strategy types:
    - kama: KAMA AdaptiveMoving AveragePriceCrossover
    - sma: SMA Dual Moving Average Crossover
    - donchian: Donchian Channel Breakout
    - donchian_regime: Donchian + ADXTrend + 200MAdirection three-in-one
    - bollinger: Bollinger Band Mean Reversion

    Args:
        opens/highs/lows/closes/volumes: OHLCV
        strategy: Strategy Type
        period: Donchian/Bollinger/Regime Cyclical
        er_period/fast_sc/slow_sc: KAMA Parameter
        adx_threshold: ADX trend threshold (donchian_regime only)
        ma_period: Long-term moving average period (donchian_regime only)
        adx_window: ADX calculation window (donchian_regime only)
        period: Donchian/Bollinger CyclicalParameter
        num_std: Bollinger Standard Deviationmultiple
        er_period: KAMA efficiency ratio window (only takes effect when strategy="kama")
        fast_sc: KAMA fast SC constant (only takes effect when strategy="kama")
        slow_sc: KAMA slow SC constant (only takes effect when strategy="kama")
        enable_vpa: YesNoEnable VPA filter（DefaultClose）
        vol_mult: VPA Volume IncreaseThreshold
        enable_weekly_filter: YesNoEnableWeeklydirectional filter
        weekly_ma: Weekly MA Cyclical
        timestamps: Timestamp array (same length as closes, for weekly filter)
        tb_pt_mult/tb_sl_mult/tb_max_hold: Triple-Barrier Parameter
        adaptive_exit: YesNoEnableAdaptive ATR multipleExit
        vol_lookback: Volatility calculation window (default 252 days)
        commission_rate: Commission rate (0.025 per 10k), charged on both buy and sell, included in cost_adjusted_sharpe
        stamp_duty: Stamp duty rate (0.1%), charged only on sell, included in cost_adjusted_sharpe
    Returns:
        StrategyResult: Full backtest performance, including post-cost cost_adjusted_sharpe and total_cost
    """
    n = len(closes)

    # Step 1: Strategy signal generation (dispatch by strategy type)
    if strategy == "kama":
        entries, exits = kama_cross_signals(
            closes, er_period=er_period, fast_sc=fast_sc, slow_sc=slow_sc,
        )
        # Step 1.1: Global market regime filter (index > 200MA?)
        from agentic_quant.validation.regime import is_bear_market
        if market_closes is not None and len(market_closes) >= 200 and is_bear_market(market_closes):
            entries = np.zeros_like(entries)  # Bear market: block new buys, preserve sells
    elif strategy == "sma":
        entries, exits = sma_cross_signals(closes, fast=fast, slow=slow)
    elif strategy == "donchian":
        entries, exits = donchian_signals(highs, lows, closes, period=period)
    elif strategy == "donchian_regime":
        entries, exits = donchian_regime_signals(
            highs, lows, closes, period=period,
            adx_threshold=adx_threshold, ma_period=ma_period, adx_window=adx_window,
        )
    elif strategy == "bollinger":
        entries, exits = bollinger_signals(closes, period=period, num_std=num_std)
    else:
        raise ValueError(f"Unknown strategy type: {strategy}")

    # Step 1.5: WeeklyTrenddirectional filter（Multi-Timeframe）
    # Use weekly SMA trend direction to filter daily signals:
    # - Buy signals only take effect when price > weekly SMA (align with weekly rising trend)
    # - Sell signals only take effect when price < weekly SMA (align with weekly downtrend)
    # Theoretical basis: Williams multi-timeframe analysis, weekly sets direction, daily sets entry
    if enable_weekly_filter and timestamps is not None and len(timestamps) == n:
        entries, exits = apply_weekly_filter(
            entries, exits, closes, timestamps, ma_period=weekly_ma,
        )

    # Step 2: VPA filter
    if enable_vpa:
        entries, vpa_results = apply_vpa_filter(
            entries, opens, highs, lows, closes, volumes,
            vol_mult=vol_mult,
        )

    # Step 2.5: Adaptive ATR multipleAdjustCoefficient
    # Pre-calculate take profit/stop loss adjustment coefficients for each time point, used in steps 3 and 4
    pt_mult_adj = np.ones(n, dtype=float)
    sl_mult_adj = np.ones(n, dtype=float)
    if adaptive_exit:
        pt_mult_adj, sl_mult_adj = compute_adaptive_atr_multipliers(
            closes, lookback=vol_lookback,
        )

    # Step 3: Triple-Barrier Tag
    entry_indices = np.where(entries)[0]
    if len(entry_indices) > 0:
        tb_labels = triple_barrier_labels_ohlc(
            opens, highs, lows, closes, entry_indices,
            pt_mult=tb_pt_mult, sl_mult=tb_sl_mult, max_hold=tb_max_hold,
            adaptive_pt_adj=pt_mult_adj if adaptive_exit else None,
            adaptive_sl_adj=sl_mult_adj if adaptive_exit else None,
        )
    else:
        tb_labels = np.array([], dtype=int)

    # Step 4: Performance statistics (triple-barrier driven exit)
    # T+1 execution: positions start from entry_bar, entry execution day return uses open->close;
    # Subsequent position days continue using close-to-close, avoid including overnight gap from signal day close to next day open in returns.
    rets = np.diff(closes) / (closes[:-1] + 1e-9)

    # ATR Used For TB Exitjudgment (shared Wilder implementation)
    atr_vals = wilder_atr(highs, lows, closes, window=15)

    pos = np.zeros(n, dtype=float)
    entry_exec_prices = np.zeros(n, dtype=float)
    in_pos = False
    entry_bar = -1
    entry_price = 0.0
    # Log actual return for each recorded trade (used to calculate real profit factor)
    trade_returns: list[float] = []

    for i in range(n):
        # ─First check exit (already in positions)─
        if in_pos:
            # ─Triple-barrier exit (ATR already incorporates volatility info)─
            eff_pt_mult = tb_pt_mult
            eff_sl_mult = tb_sl_mult
            eff_max_hold = tb_max_hold

            base_atr = atr_vals[i] if atr_vals[i] > 0 else entry_price * 0.01
            stop_dist = eff_sl_mult * base_atr
            profit_dist = eff_pt_mult * stop_dist
            pt_level = entry_price + profit_dist
            sl_level = entry_price - stop_dist
            time_up = (i - entry_bar) >= eff_max_hold

            exited = False
            exit_price_est = closes[i]
            if lows[i] <= sl_level:
                in_pos = False
                exit_price_est = sl_level
                exited = True
            elif highs[i] >= pt_level:
                in_pos = False
                exit_price_est = pt_level
                exited = True
            elif time_up or exits[i]:
                in_pos = False
                exited = True

            if exited:
                pos[i] = 1.0  # Include the exit bar's return in the equity curve.
                trade_ret = (exit_price_est - entry_price) / max(entry_price, 1e-9)
                trade_returns.append(trade_ret)

        # ─Entry: signal generated at bar i, T+1 opens at bar i+1 open─
        entered_now = False
        if entries[i] and not in_pos and (i + 1) < n:
            in_pos = True
            entry_bar = i + 1  # T+1 Execute bar
            entry_price = opens[i + 1]  # Next bar open
            entry_exec_prices[entry_bar] = entry_price
            entered_now = True
            # Mark positions starting from execution bar
            pos[entry_bar] = 1.0

        # Position continuation: in_pos=True and not a new entry this bar -> continue marking positions
        if in_pos and not entered_now:
            pos[i] = 1.0

    strat_rets = rets * pos[1:]
    entry_bars = np.where(entry_exec_prices[1:] > 0)[0] + 1
    for bar_idx in entry_bars:
        ret_idx = bar_idx - 1
        strat_rets[ret_idx] = (
            closes[bar_idx] - entry_exec_prices[bar_idx]
        ) / max(entry_exec_prices[bar_idx], 1e-9)

    # Total return (compounded, not simple sum)
    total_return = float(np.prod(1 + strat_rets) - 1)

    # Sharpe (annualized, sample standard deviation ddof=1)
    sharpe = _safe_sharpe(strat_rets, periods=252)

    # Max Drawdown
    equity = np.cumprod(1 + strat_rets)
    peak = np.maximum.accumulate(equity)
    max_dd = float(abs(np.min((equity - peak) / (peak + 1e-9))))

    # Win Rate（Based On Triple-Barrier Tag）
    win_rate = float(np.mean(tb_labels == 1)) if len(tb_labels) > 0 else 0.0

    # Number of trades: count using actual executed trade_returns (filter out entries not executed on last bar)
    num_trades = len(trade_returns)

    # ─Real profit factor = total profitable amount / total loss amount─
    if len(trade_returns) > 0:
        trade_rets_arr = np.array(trade_returns)
        total_wins = float(np.sum(trade_rets_arr[trade_rets_arr > 0]))
        total_losses = float(abs(np.sum(trade_rets_arr[trade_rets_arr < 0])))
        profit_factor = total_wins / max(total_losses, 1e-9)
    else:
        profit_factor = 0.0

    # Average position holding days
    avg_hold = float(np.sum(pos)) / max(num_trades, 1)

    # ─Transaction cost: deducted on occurrence day, participates in compounding─
    total_cost = 0.0
    cost_strat_rets = strat_rets.copy()
    if len(strat_rets) > 0 and (commission_rate > 0 or stamp_duty > 0):
        pos_padded = np.concatenate([[0], pos])
        pos_diff = np.diff(pos_padded)  # pos_diff[i] = pos[i] - pos[i-1] (aligned with bar i)
        # pos_diff effective index 0..n-1, corresponding to bar 0..n-1
        # strat_rets index 0..n-2, corresponding to bar 0..n-2
        buy_count = int(np.sum(pos_diff[:-1] > 0))   # only bars within strat_rets range
        sell_count = int(np.sum(pos_diff[:-1] < 0))
        total_cost = buy_count * commission_rate + sell_count * (commission_rate + stamp_duty)

        # Deduct cost on the day trading occurs, so it participates in compounding
        for k in range(min(len(cost_strat_rets), len(pos_diff) - 1)):
            if pos_diff[k] > 0:
                cost_strat_rets[k] -= commission_rate  # Deduct commission on buy day
            elif pos_diff[k] < 0:
                cost_strat_rets[k] -= (commission_rate + stamp_duty)  # Deduct commission + stamp duty on sell day

    cost_sharpe = _safe_sharpe(cost_strat_rets, periods=252)

    return StrategyResult(
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        num_trades=num_trades,
        avg_hold_days=avg_hold,
        profit_factor=profit_factor,
        tb_labels=tb_labels,
        filtered_entries=entries,
        exits=exits,
        cost_adjusted_sharpe=cost_sharpe,
        total_cost=total_cost,
    )


# ═══════════════════════════ StrategySignalCorrelation ═══════════════════════════

def evaluate_strategy_correlation(
    signals_dict: dict[str, np.ndarray],
) -> dict[str, Any]:
    """
    Calculate signal correlation matrix and overlap statistics across multiple strategies

    Used for combination de-redundancy analysis:
    - Correlation < 0.3: low correlation, good diversification effect
    - Correlation 0.3~0.7: moderate correlation
    - Correlation > 0.7: high correlation, limited diversification effect

    Uses dual metrics: Pearson correlation coefficient + Jaccard similarity,
    Jaccard measures the actual overlap ratio of signal events, more robust for sparse signals.

    Args:
        signals_dict: {"strategy_name": entry_bool_array}
    Returns:
        {
            "strategy_names": [NameList],
            "correlation_matrix": {name1: {name2: corr}},
            "pairwise_overlap": {pair: {jaccard, n_i, n_j, n_both}}
        }
    """
    names = list(signals_dict.keys())
    k = len(names)
    corr_mat = np.zeros((k, k))
    overlap: dict[str, dict[str, float | int]] = {}

    for i in range(k):
        for j in range(k):
            si = signals_dict[names[i]].astype(float)
            sj = signals_dict[names[j]].astype(float)
            if np.std(si) > 1e-9 and np.std(sj) > 1e-9:
                corr_mat[i, j] = float(np.corrcoef(si, sj)[0, 1])
            else:
                corr_mat[i, j] = 0.0

    # Pairwise Jaccard similarity (measures signal event overlap ratio)
    for i in range(k):
        for j in range(i + 1, k):
            pair = f"{names[i]}_vs_{names[j]}"
            si = signals_dict[names[i]]
            sj = signals_dict[names[j]]
            both = int(np.sum(si & sj))
            union = int(np.sum(si | sj))
            overlap[pair] = {
                "n_i": int(np.sum(si)),
                "n_j": int(np.sum(sj)),
                "n_both": both,
                "jaccard": both / max(union, 1),
            }

    # Convert to nested dictionary for easy serialization
    corr_dict: dict[str, dict[str, float]] = {}
    for i in range(k):
        corr_dict[names[i]] = {}
        for j in range(k):
            corr_dict[names[i]][names[j]] = round(float(corr_mat[i, j]), 4)

    return {
        "strategy_names": names,
        "correlation_matrix": corr_dict,
        "pairwise_overlap": overlap,
    }


# ═══════════════════════════ VPA IncrementalEvaluate ═══════════════════════════

@dataclass
class VPAIncrementReport:
    """
    VPA incremental evaluation report -- comparing full performance difference between bare SMA and SMA+VPA

    Report structure:
    - Overall metrics: performance across dimensions for bare SMA vs VPA
    - Incremental metrics: Sharpe delta, win rate delta, drawdown delta, trade reduction rate
    - Group decomposition: by year, by market regime (4 quadrant)

    Positive incremental sign = VPA improvement, negative = VPA degradation.
    """
    symbol: str
    # Bare SMA baseline
    bare_win_rate: float
    bare_sharpe: float
    bare_max_dd: float
    bare_total_return: float
    bare_num_trades: int
    # After VPA filter
    vpa_win_rate: float
    vpa_sharpe: float
    vpa_max_dd: float
    vpa_total_return: float
    vpa_num_trades: int
    # Incremental（VPA minus bare）
    sharpe_delta: float
    win_rate_delta: float
    max_dd_delta: float            # Negative = VPA reduced drawdown (improvement)
    trade_reduction_pct: float     # Ratio of trades reduced by VPA
    # Group statistics
    yearly_breakdown: dict[int, dict]   # year -> {bare_*, vpa_*, *_delta}
    regime_breakdown: dict[str, dict]   # regime -> {bare_*, vpa_*, *_delta}


def _compute_strat_metrics(
    entries: np.ndarray,
    exits: np.ndarray,
    closes: np.ndarray,
    tb_labels: np.ndarray,
) -> dict:
    """
    Calculate strategy performance metrics (shared calculation logic extracted from run_strategy)

    Used for independent evaluation of bare SMA and VPA-filtered versions in VPA incremental comparison.

    Args:
        entries: EntrySignalBooleanArray
        exits: ExitSignalBooleanArray
        closes: Close Price Series
        tb_labels: Triple-barrier labels (aligned with entry)

    Returns:
        {total_return, sharpe, max_dd, win_rate, num_trades}
    """
    n = len(closes)
    num_trades = int(np.sum(entries))

    if num_trades == 0:
        return {
            "total_return": 0.0, "sharpe": 0.0, "max_dd": 0.0,
            "win_rate": 0.0, "num_trades": 0,
        }

    # Daily return
    rets = np.diff(closes) / (closes[:-1] + 1e-9)

    # PositionsMock: T+1 execution (enter next bar), exit bar return is retained.
    pos = np.zeros(n, dtype=float)
    in_pos = False
    entry_bar = -1
    for i in range(n):
        if entries[i] and not in_pos and i + 1 < n:
            in_pos = True
            entry_bar = i + 1
        if exits[i] and in_pos:
            pos[i] = 1.0  # Include the exit bar's close-to-close return.
            in_pos = False
        elif in_pos and i >= entry_bar:
            pos[i] = 1.0

    # Strategy daily return = positions * underlying daily return
    strat_rets = rets * pos[1:]

    # Total return (compounded)
    total_return = float(np.prod(1 + strat_rets) - 1)

    # Annualized Sharpe (sample standard deviation)
    sharpe = _safe_sharpe(strat_rets, periods=252)

    # Max Drawdown
    equity = np.cumprod(1 + strat_rets)
    peak = np.maximum.accumulate(equity)
    max_dd = float(abs(np.min((equity - peak) / (peak + 1e-9))))

    # Win Rate（Based On Triple-Barrier Tag）
    win_rate = float(np.mean(tb_labels == 1)) if len(tb_labels) > 0 else 0.0

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "num_trades": num_trades,
    }


def evaluate_vpa_increment(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    symbol: str = "",
    fast: int = 10,
    slow: int = 20,
    tb_pt_mult: float = 2.0,
    tb_sl_mult: float = 1.0,
    tb_max_hold: int = 20,
    timestamps: np.ndarray | None = None,
) -> VPAIncrementReport:
    """
    VPA incremental value evaluation -- quantifying VPA's incremental boost over bare SMA

    ExecuteCompleteEvaluatePipeline：
      Bare SMA signal -> triple-barrier exit -> performance metrics
      SMA+VPA Signal -> Triple-Barrier Exit -> PerformanceMetrics
      Compare two sets of metrics -> incremental analysis -> group decomposition by year/regime

    VPA ConfirmCondition：
    1. Volume > 20-day average (volume expansion)
    2. Long close in today's upper third (price validation)

    Theoretical Basis:
    - Coulling《Volume Price Analysis》price-volume alignmentConfirmTrend
    - De Prado "Advances in Financial ML" triple-barrier exit system
    - Masters "Testing and Tuning" parameter plateau and incremental evaluation

    Args:
        opens/highs/lows/closes/volumes: OHLCV Array
        symbol: Underlying code (used for report identification)
        fast/slow: SMA dual moving average parameters
        tb_pt_mult/tb_sl_mult/tb_max_hold: Triple-Barrier Parameter
        timestamps: Optional timestamp array (length consistent with closes), used for yearly grouping

    Returns:
        VPAIncrementReport: Complete incremental evaluation report
    """
    n = len(closes)

    # === Step 1: SMA crossover signal (bare signal) ===
    entries, exits = sma_cross_signals(closes, fast=fast, slow=slow)

    # === Step 2: VPA filter (simplified: 20-day average volume + upper third) ===
    vpa_entries = _apply_vpa_filter_upper_third(
        entries, opens, highs, lows, closes, volumes,
    )

    # === Step 3: Triple-barrier labeling (two independent calculations) ===
    bare_entry_indices = np.where(entries)[0]
    vpa_entry_indices = np.where(vpa_entries)[0]

    bare_tb = triple_barrier_labels_ohlc(
        opens, highs, lows, closes, bare_entry_indices,
        pt_mult=tb_pt_mult, sl_mult=tb_sl_mult, max_hold=tb_max_hold,
    )
    vpa_tb = triple_barrier_labels_ohlc(
        opens, highs, lows, closes, vpa_entry_indices,
        pt_mult=tb_pt_mult, sl_mult=tb_sl_mult, max_hold=tb_max_hold,
    )

    # === Step 4: overallPerformanceCalculate ===
    bare_m = _compute_strat_metrics(entries, exits, closes, bare_tb)
    vpa_m = _compute_strat_metrics(vpa_entries, exits, closes, vpa_tb)

    # Incremental: VPA minus bare, positive sign indicates VPA improvement
    sharpe_delta = vpa_m["sharpe"] - bare_m["sharpe"]
    win_rate_delta = vpa_m["win_rate"] - bare_m["win_rate"]
    max_dd_delta = vpa_m["max_dd"] - bare_m["max_dd"]
    trade_reduction_pct = (
        (bare_m["num_trades"] - vpa_m["num_trades"])
        / max(bare_m["num_trades"], 1)
    )

    # === Step 5: Group by year ===
    yearly_breakdown: dict[int, dict] = {}
    if timestamps is not None and len(timestamps) == n:
        year_strs: list[str] = []
        for t in timestamps:
            s = str(t)
            year_strs.append(s[:4] if len(s) >= 4 and s[:4].isdigit() else "")
        years = np.array(year_strs)

        for yr_str in sorted({y for y in years if y}):
            yr = int(yr_str)
            yr_mask = years == yr_str

            yr_bare_entries = entries & yr_mask
            yr_vpa_entries = vpa_entries & yr_mask
            yr_bare_indices = np.where(yr_bare_entries)[0]
            yr_vpa_indices = np.where(yr_vpa_entries)[0]

            if len(yr_bare_indices) == 0 and len(yr_vpa_indices) == 0:
                continue

            yr_bare_tb = (
                triple_barrier_labels_ohlc(
                    opens, highs, lows, closes, yr_bare_indices,
                    pt_mult=tb_pt_mult, sl_mult=tb_sl_mult,
                    max_hold=tb_max_hold,
                )
                if len(yr_bare_indices) > 0 else np.array([], dtype=int)
            )
            yr_vpa_tb = (
                triple_barrier_labels_ohlc(
                    opens, highs, lows, closes, yr_vpa_indices,
                    pt_mult=tb_pt_mult, sl_mult=tb_sl_mult,
                    max_hold=tb_max_hold,
                )
                if len(yr_vpa_indices) > 0 else np.array([], dtype=int)
            )

            yr_bare = _compute_strat_metrics(
                yr_bare_entries, exits, closes, yr_bare_tb,
            )
            yr_vpa = _compute_strat_metrics(
                yr_vpa_entries, exits, closes, yr_vpa_tb,
            )

            yearly_breakdown[yr] = {
                "bare_win_rate": yr_bare["win_rate"],
                "bare_sharpe": yr_bare["sharpe"],
                "bare_num_trades": yr_bare["num_trades"],
                "vpa_win_rate": yr_vpa["win_rate"],
                "vpa_sharpe": yr_vpa["sharpe"],
                "vpa_num_trades": yr_vpa["num_trades"],
                "win_rate_delta": yr_vpa["win_rate"] - yr_bare["win_rate"],
                "sharpe_delta": yr_vpa["sharpe"] - yr_bare["sharpe"],
            }

    # === Step 6: Group by market regime (4 quadrant) ===
    regime_breakdown: dict[str, dict] = {}
    try:
        from agentic_quant.validation.regime import MarketRegime  # noqa: PLC0415

        mr = MarketRegime()
        regime_labels = mr.classify(closes)

        for regime_name in mr.REGIME_NAMES:
            regime_mask = np.array(
                [r == regime_name for r in regime_labels],
                dtype=bool,
            )

            reg_bare_entries = entries & regime_mask
            reg_vpa_entries = vpa_entries & regime_mask
            reg_bare_indices = np.where(reg_bare_entries)[0]
            reg_vpa_indices = np.where(reg_vpa_entries)[0]

            if len(reg_bare_indices) == 0 and len(reg_vpa_indices) == 0:
                continue

            reg_bare_tb = (
                triple_barrier_labels_ohlc(
                    opens, highs, lows, closes, reg_bare_indices,
                    pt_mult=tb_pt_mult, sl_mult=tb_sl_mult,
                    max_hold=tb_max_hold,
                )
                if len(reg_bare_indices) > 0 else np.array([], dtype=int)
            )
            reg_vpa_tb = (
                triple_barrier_labels_ohlc(
                    opens, highs, lows, closes, reg_vpa_indices,
                    pt_mult=tb_pt_mult, sl_mult=tb_sl_mult,
                    max_hold=tb_max_hold,
                )
                if len(reg_vpa_indices) > 0 else np.array([], dtype=int)
            )

            reg_bare = _compute_strat_metrics(
                reg_bare_entries, exits, closes, reg_bare_tb,
            )
            reg_vpa = _compute_strat_metrics(
                reg_vpa_entries, exits, closes, reg_vpa_tb,
            )

            regime_breakdown[regime_name] = {
                "bare_win_rate": reg_bare["win_rate"],
                "bare_sharpe": reg_bare["sharpe"],
                "bare_num_trades": reg_bare["num_trades"],
                "vpa_win_rate": reg_vpa["win_rate"],
                "vpa_sharpe": reg_vpa["sharpe"],
                "vpa_num_trades": reg_vpa["num_trades"],
                "win_rate_delta": reg_vpa["win_rate"] - reg_bare["win_rate"],
                "sharpe_delta": reg_vpa["sharpe"] - reg_bare["sharpe"],
            }
    except ImportError:
        # Skip regime decomposition when market regime is unavailable
        pass

    return VPAIncrementReport(
        symbol=symbol,
        bare_win_rate=bare_m["win_rate"],
        bare_sharpe=bare_m["sharpe"],
        bare_max_dd=bare_m["max_dd"],
        bare_total_return=bare_m["total_return"],
        bare_num_trades=bare_m["num_trades"],
        vpa_win_rate=vpa_m["win_rate"],
        vpa_sharpe=vpa_m["sharpe"],
        vpa_max_dd=vpa_m["max_dd"],
        vpa_total_return=vpa_m["total_return"],
        vpa_num_trades=vpa_m["num_trades"],
        sharpe_delta=sharpe_delta,
        win_rate_delta=win_rate_delta,
        max_dd_delta=max_dd_delta,
        trade_reduction_pct=trade_reduction_pct,
        yearly_breakdown=yearly_breakdown,
        regime_breakdown=regime_breakdown,
    )


# ═══════════════════════════ WeeklyTrenddirectional filter ═══════════════════


@dataclass
class WeeklyFilterReport:
    """
    Weekly filter incremental evaluation report -- comparing full performance difference between bare SMA and SMA+weekly filter

    Report structure:
    - Overall metrics: performance across dimensions for bare SMA vs weekly filter
    - Incremental metrics: Sharpe delta, win rate delta, drawdown delta, trade reduction rate

    Positive incremental sign = weekly filter improvement, negative = weekly filter degradation.
    """
    symbol: str
    weekly_ma: int                     # Weekly SMA Cyclical
    # Bare SMA baseline
    bare_win_rate: float
    bare_sharpe: float
    bare_max_dd: float
    bare_total_return: float
    bare_num_trades: int
    # After weekly filter
    wf_win_rate: float
    wf_sharpe: float
    wf_max_dd: float
    wf_total_return: float
    wf_num_trades: int
    # Incremental（WF minus bare）
    sharpe_delta: float
    win_rate_delta: float
    max_dd_delta: float
    trade_reduction_pct: float         # Ratio of trades reduced by weekly filter


def evaluate_weekly_filter(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    timestamps: np.ndarray,
    symbol: str = "",
    fast: int = 10,
    slow: int = 20,
    weekly_ma: int = 20,
    tb_pt_mult: float = 2.0,
    tb_sl_mult: float = 1.0,
    tb_max_hold: int = 20,
) -> WeeklyFilterReport:
    """
    Weekly filter incremental value evaluation -- quantifying weekly SMA's incremental boost over bare SMA strategy

    ExecuteCompleteEvaluatePipeline:
      Bare SMA signal -> triple-barrier exit -> performance metrics
      SMA + Weeklyfilter -> Triple-Barrier Exit -> PerformanceMetrics
      Compare two sets of metrics -> incremental analysis

    Theoretical basis:
    - Multi-timeframe analysis: weekly sets trend direction, daily sets entry timing
    - De Prado "Advances in Financial ML" triple-barrier exit system

    Args:
        opens/highs/lows/closes/volumes: OHLCV Array
        timestamps: Date string array, same length as closes
        symbol: Underlying code (used for report identification)
        fast/slow: SMA dual moving average parameters
        weekly_ma: Weekly SMA Cyclical（Default 20）
        tb_pt_mult/tb_sl_mult/tb_max_hold: Triple-Barrier Parameter
    Returns:
        WeeklyFilterReport: Complete incremental evaluation report
    """
    # === Step 1: SMA crossover signal (bare signal) ===
    bare_entries, bare_exits = sma_cross_signals(closes, fast=fast, slow=slow)

    # === Step 2: WeeklyTrenddirectional filter ===
    wf_entries, wf_exits = apply_weekly_filter(
        bare_entries, bare_exits, closes, timestamps, ma_period=weekly_ma,
    )

    # === Step 3: Triple-barrier labeling (two independent calculations) ===
    bare_entry_indices = np.where(bare_entries)[0]
    wf_entry_indices = np.where(wf_entries)[0]

    bare_tb = triple_barrier_labels_ohlc(
        opens, highs, lows, closes, bare_entry_indices,
        pt_mult=tb_pt_mult, sl_mult=tb_sl_mult, max_hold=tb_max_hold,
    )
    wf_tb = triple_barrier_labels_ohlc(
        opens, highs, lows, closes, wf_entry_indices,
        pt_mult=tb_pt_mult, sl_mult=tb_sl_mult, max_hold=tb_max_hold,
    )

    # === Step 4: overallPerformanceCalculate ===
    bare_m = _compute_strat_metrics(bare_entries, bare_exits, closes, bare_tb)
    wf_m = _compute_strat_metrics(wf_entries, wf_exits, closes, wf_tb)

    # Incremental: WF minus bare, positive sign indicates weekly filter improvement
    sharpe_delta = wf_m["sharpe"] - bare_m["sharpe"]
    win_rate_delta = wf_m["win_rate"] - bare_m["win_rate"]
    max_dd_delta = wf_m["max_dd"] - bare_m["max_dd"]
    trade_reduction_pct = (
        (bare_m["num_trades"] - wf_m["num_trades"])
        / max(bare_m["num_trades"], 1)
    )

    return WeeklyFilterReport(
        symbol=symbol,
        weekly_ma=weekly_ma,
        bare_win_rate=bare_m["win_rate"],
        bare_sharpe=bare_m["sharpe"],
        bare_max_dd=bare_m["max_dd"],
        bare_total_return=bare_m["total_return"],
        bare_num_trades=bare_m["num_trades"],
        wf_win_rate=wf_m["win_rate"],
        wf_sharpe=wf_m["sharpe"],
        wf_max_dd=wf_m["max_dd"],
        wf_total_return=wf_m["total_return"],
        wf_num_trades=wf_m["num_trades"],
        sharpe_delta=sharpe_delta,
        win_rate_delta=win_rate_delta,
        max_dd_delta=max_dd_delta,
        trade_reduction_pct=trade_reduction_pct,
    )
