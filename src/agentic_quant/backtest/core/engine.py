"""
Event-Driven Backtest Engine

Core logic: bar-by-bar, process Signal -> Mock Trade -> Update Positions -> Calculate PnL
Reference: Trading Systems and Methods (Kaufman), Ch.21-22
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from agentic_quant.backtest.core.portfolio import Fill, Position
from agentic_quant.backtest.core.result import BacktestResult
from agentic_quant.backtest.data.structures import Bar
from agentic_quant.backtest.strategy.base import Signal, Strategy

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Event-Driven Backtest Engine
    Core logic: bar-by-bar, process Signal -> Mock Trade -> Update Positions -> Calculate PnL
    Reference: Trading Systems and Methods, Ch.21-22
    """

    VALID_EXECUTION_TIMINGS = {"close", "next_open"}

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        commission_rate: float = 0.0003,  # 0.03% commission
        slippage: float = 0.0,  # Slippage (in absolute price terms)
        allow_short: bool = False,
        execution_timing: str = "close",  # "close"=bar close | "next_open"=next bar open (default close for backward compatibility)
    ):
        if execution_timing not in self.VALID_EXECUTION_TIMINGS:
            valid = ", ".join(sorted(self.VALID_EXECUTION_TIMINGS))
            raise ValueError(f"execution_timing must be one of: {valid}")

        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.allow_short = allow_short
        self.execution_timing = execution_timing

        self._cash: float = initial_capital
        self._position = Position()
        self._equity_history: list[dict] = []
        self._fills: list[Fill] = []
        self._signals: list[Signal] = []
        self._pending_signal: Signal | None = None  # Pending signal queue for execution

    @property
    def equity(self) -> float:
        """Current total equity = cash + position market value"""
        return self._cash + self._position.size * self._position.market_price  # type: ignore[no-any-return]

    def reset(self) -> None:
        """Reset engine state for reuse across multiple backtests"""
        self._cash = self.initial_capital
        self._position = Position()
        self._equity_history = []
        self._fills = []
        self._signals = []
        self._pending_signal = None  # type: ignore[assignment]

    def _execute(self, bar: Bar, signal: Signal) -> Fill | None:
        """
        Execute signal, generate trade log
        execution_timing:
          - "close": Market order executes at current bar close (original behavior)
          - "next_open": Market order executes at next bar open (daily default, avoids look-ahead bias)
        """
        if signal.action == "hold":
            return None

        # Determine execution price based on execution_timing
        if self.execution_timing == "next_open":
            fill_price = bar.open  # Next bar open
        else:  # "close" or default
            # Price includes slippage: buy adds, sell reduces
            if signal.action in ("buy",):
                fill_price = bar.close + self.slippage
            elif signal.action in ("sell", "close"):
                fill_price = bar.close - self.slippage
            else:
                fill_price = bar.close

        # next_open mode: add slippage on top of open
        # Slippage direction depends on trade direction (buy adds slippage, sell subtracts slippage)
        if self.execution_timing == "next_open" and self.slippage > 0:
            target_size = signal.size
            if signal.action == "close":
                target_size = 0.0
            trade_size = target_size - self._position.size
            if abs(trade_size) >= 1e-9:
                if trade_size > 0:
                    fill_price += self.slippage  # Buy: slippage up
                else:
                    fill_price -= self.slippage  # Sell: slippage down

        # CalculatetargetPosition Size
        target_size = signal.size
        if signal.action == "close":
            target_size = 0.0

        # If short selling is not allowed, limit position size
        if not self.allow_short and target_size < 0:
            target_size = 0.0

        trade_size = target_size - self._position.size
        if abs(trade_size) < 1e-9:
            return None

        notional = abs(trade_size * fill_price)
        commission = notional * self.commission_rate

        # Update cash (buy deducts cash, sell adds cash)
        if trade_size > 0:
            self._cash -= trade_size * fill_price + commission
        else:
            self._cash += abs(trade_size) * fill_price - commission

        # Update position average price
        old_size = self._position.size
        new_size = old_size + trade_size
        if abs(new_size) < 1e-9:
            self._position.avg_price = 0.0
        elif old_size == 0:
            self._position.avg_price = fill_price
        elif (old_size > 0 and trade_size > 0) or (old_size < 0 and trade_size < 0):
            # Add to position, weighted averaging
            self._position.avg_price = (
                old_size * self._position.avg_price + trade_size * fill_price
            ) / new_size
        else:
            # Reduce or reverse position, remaining portion avg price unchanged (if any remains)
            if abs(new_size) > 1e-9:
                pass  # Average price held unchanged
            else:
                self._position.avg_price = 0.0

        self._position.size = new_size

        fill = Fill(
            timestamp=bar.timestamp if self.execution_timing == "next_open" else signal.timestamp,
            action=signal.action,
            price=fill_price,
            size=trade_size,
            commission=commission,
            slippage=self.slippage,
        )
        self._fills.append(fill)
        return fill

    def _update_pnl(self, bar: Bar) -> None:
        """Update unrealized P&L and mark price based on latest price"""
        self._position.market_price = bar.close
        if self._position.size != 0:
            self._position.unrealized_pnl = (
                bar.close - self._position.avg_price
            ) * self._position.size
        else:
            self._position.unrealized_pnl = 0.0

    def run(self, bars: list[Bar], strategy: Strategy) -> BacktestResult:
        """
        Run backtest
        Args:
            bars: Bar data sorted by timestamp
            strategy: Strategy instance
        Returns:
            BacktestResult
        """
        self.reset()
        logger.info(f"Start backtest: Strategy={strategy.name}, Initial Capital={self.initial_capital:,.2f}, Execution Time={self.execution_timing}")

        for bar in bars:
            # ── 1. First execute signal from previous bar (next_open mode) --
            if self._pending_signal is not None:
                self._signals.append(self._pending_signal)
                self._execute(bar, self._pending_signal)
                self._pending_signal = None

            # ── 2. Current bar generates new signal ──
            signal = strategy.feed(bar)
            if signal:
                if self.execution_timing == "next_open":
                    # Deferred to next bar open for execution
                    self._pending_signal = signal
                else:
                    # close mode: execute immediately
                    self._signals.append(signal)
                    self._execute(bar, signal)

            # ── 3. Update PnL ──
            self._update_pnl(bar)
            self._equity_history.append(
                {
                    "timestamp": bar.timestamp,
                    "equity": self.equity,
                    "cash": self._cash,
                    "position_size": self._position.size,
                    "unrealized_pnl": self._position.unrealized_pnl,
                    "close": bar.close,
                }
            )

        # Last bar: next_open mode has no next bar, pending signal is discarded
        # (close mode pending signals are handled inside the loop)
        if self._pending_signal is not None and self.execution_timing == "next_open":
            logger.warning("Pending signal on last bar discarded (next_open mode has no next bar)")
            self._pending_signal = None

        result = self._build_result()
        logger.info(
            f"Backtest end: Total Return={result.total_return:.2%}, "
            f"Sharpe={result.sharpe_ratio:.2f}, MaxDD={result.max_drawdown:.2%}, "
            f"Trade Count={result.num_trades}"
        )
        return result

    def _build_result(self) -> BacktestResult:
        """Build backtest result object, calculate all performance metrics"""
        if not self._equity_history:
            raise ValueError("Backtest produced no equity log")

        df = pl.DataFrame(self._equity_history)
        returns = df["equity"].pct_change().drop_nulls()

        total_return = (df["equity"].tail(1).item() / self.initial_capital) - 1.0

        # Infer annualization factor from bar time interval (avoid hardcoding 252)
        ann_factor = self._infer_annualization_factor(df)
        if len(returns) > 1 and returns.std() > 0:  # type: ignore[operator]
            sharpe = (returns.mean() / returns.std()) * np.sqrt(ann_factor)  # type: ignore[operator]
        else:
            sharpe = 0.0

        # Max Drawdown
        cumulative = df["equity"]
        running_max = cumulative.cum_max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        # Trade P&L statistics: segment by position zero-crossing for complete trades
        trade_pnls = self._compute_trade_pnls()
        num_trades = len(trade_pnls)
        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p <= 0]
        win_rate = len(wins) / num_trades if num_trades > 0 else 0.0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 1e-9 else float("inf")

        return BacktestResult(
            equity_curve=df,
            fills=self._fills,
            signals=self._signals,
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=num_trades,
        )

    def _infer_annualization_factor(self, df: pl.DataFrame) -> int:
        """Infer annualization factor from bar time interval"""
        if len(df) < 2:
            return 252
        # Take median of first few intervals, avoid outliers
        ts = df["timestamp"]
        deltas = [(ts[i] - ts[i - 1]).total_seconds() for i in range(1, min(10, len(ts)))]
        median_seconds = sorted(deltas)[len(deltas) // 2]
        bars_per_year = 365.25 * 24 * 3600 / max(median_seconds, 1)
        return int(round(bars_per_year))

    def _compute_trade_pnls(self) -> list[float]:
        """
        Segment complete trades by position zero-crossing, calculate each trade P&L
        Logic: simulate position changes, each time position goes from non-zero to zero generates a trade PnL
        """
        trade_pnls: list[float] = []
        pos_size = 0.0
        pos_cost = 0.0  # Total position cost (with direction)

        for fill in self._fills:
            if fill.action == "hold":
                continue
            old_size = pos_size
            new_size = old_size + fill.size

            if old_size == 0:
                # New open position
                pos_cost = fill.size * fill.price
            elif (old_size > 0 and fill.size > 0) or (old_size < 0 and fill.size < 0):
                # Same-direction add to position
                pos_cost += fill.size * fill.price
            else:
                # Opposite-direction operation (reduce or close position)
                close_size = min(abs(old_size), abs(fill.size))
                avg_cost = pos_cost / old_size if old_size != 0 else 0.0
                pnl = (fill.price - avg_cost) * close_size * (1 if old_size > 0 else -1)
                if abs(new_size) < 1e-9:
                    # Fully close position, log P&L
                    trade_pnls.append(pnl - fill.commission)
                    pos_cost = 0.0
                else:
                    # Partial close or reverse open position
                    if abs(fill.size) >= abs(old_size):
                        # Reverse exceeds original size, first close then reverse open
                        trade_pnls.append(pnl - fill.commission * (close_size / abs(fill.size)))
                        pos_cost = new_size * fill.price
                    else:
                        # Partial close position
                        pos_cost = new_size * avg_cost

            pos_size = new_size

        # Final: if there are still unclosed positions, settle at last mark price
        if abs(pos_size) > 1e-9 and self._equity_history:
            last_price = self._equity_history[-1]["close"]
            avg_cost = pos_cost / pos_size if pos_size != 0 else 0.0
            pnl = (last_price - avg_cost) * pos_size
            trade_pnls.append(pnl)

        return trade_pnls
