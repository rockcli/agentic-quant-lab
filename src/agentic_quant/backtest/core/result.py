"""
Backtest Result与PerformanceMetricsCalculate

参考: Trading Systems and Methods (Kaufman), Ch.21-22
      Advances in Financial Machine Learning (Lopez de Prado)
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from agentic_quant.backtest.core.portfolio import Fill
from agentic_quant.backtest.strategy.base import Signal


@dataclass
class BacktestResult:
    """Backtest Result"""

    equity_curve: pl.DataFrame  # Equitycurve
    fills: list[Fill]  # TransactionList
    signals: list[Signal]  # Signal List
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int

    def to_dict(self) -> dict:
        """将CoreMetrics转为Dictionary，便于Serialize和Compare。"""
        return {
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "num_trades": self.num_trades,
        }
