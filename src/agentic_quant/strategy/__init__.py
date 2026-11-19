"""
StrategyModule：基Class、示例Strategy、Registry
"""

from agentic_quant.strategy.examples import SmaCrossStrategy
from agentic_quant.strategy.registry import (
    StrategyRegistry,
    get_global_registry,
    register_strategy,
)

__all__ = [
    "SmaCrossStrategy",
    "StrategyRegistry",
    "get_global_registry",
    "register_strategy",
]
