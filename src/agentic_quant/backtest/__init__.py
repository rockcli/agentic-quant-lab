"""
自研Backtest EngineCorePackage

ProvideEvent-Driven回测、Data Structure、Feature Engineering与Strategy基Class。
"""

from __future__ import annotations

from agentic_quant.backtest.core.engine import BacktestEngine
from agentic_quant.backtest.core.portfolio import Fill, Position
from agentic_quant.backtest.core.result import BacktestResult
from agentic_quant.backtest.data.structures import Bar, BarSampler, Tick
from agentic_quant.backtest.features.engine import FeatureEngine
from agentic_quant.backtest.strategy.base import Signal, Strategy

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "Bar",
    "BarSampler",
    "FeatureEngine",
    "Fill",
    "Position",
    "Signal",
    "Strategy",
    "Tick",
]
