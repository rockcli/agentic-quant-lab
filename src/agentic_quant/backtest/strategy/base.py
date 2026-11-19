"""
Strategy base class and Signal definition

Reference: Trading Systems and Methods (Kaufman), Ch.21-22
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from agentic_quant.backtest.data.structures import Bar


@dataclass
class Signal:
    """StrategySignal"""

    timestamp: datetime
    action: str  # "buy" | "sell" | "hold" | "close"
    size: float = 1.0  # Position size (positive = long direction, negative = short direction)
    price: float | None = None  # Target price, None means market price
    meta: dict = field(default_factory=dict)  # Strategy additional info (e.g. stop loss price, take profit price)


class Strategy(ABC):
    """
    Strategy Abstract Base Class
    All concrete strategies must inherit this class and implement on_bar method
    """

    def __init__(self, name: str = " unnamed"):
        self.name = name
        self.bars: list[Bar] = []
        self.signals: list[Signal] = []

    def feed(self, bar: Bar) -> Signal | None:
        """
        Receive new bar data, drive strategy to generate signal
        Subclasses generally do not need to override this method
        """
        self.bars.append(bar)
        signal = self.on_bar(bar)
        if signal:
            self.signals.append(signal)
        return signal

    @abstractmethod
    def on_bar(self, bar: Bar) -> Signal | None:
        """
        Subclass implements: generate trading signal based on latest bar
        Args:
            bar: Latest bar data
        Returns:
            Signal or None (no signal)
        """
        ...
