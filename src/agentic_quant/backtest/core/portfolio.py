"""
Transaction与PositionsData Structure
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Fill:
    """TransactionLog"""

    timestamp: datetime
    action: str
    price: float
    size: float
    commission: float
    slippage: float


@dataclass
class Position:
    """PositionsStatus"""

    size: float = 0.0  # 正Long / 负Short
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    market_price: float = 0.0  # Current标记Price，Used For equity Calculate
