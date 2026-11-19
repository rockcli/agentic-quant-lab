"""
Strategy Registry

Supports instantiating strategies by name, enabling AI to dynamically load generated code and run configuration-driven backtests.
"""

from __future__ import annotations

from collections.abc import Callable

from agentic_quant.backtest.strategy.base import Strategy


class StrategyRegistry:
    """
    Strategy registry, supports instantiating strategies by name.

    Usage:
        registry = StrategyRegistry()
        registry.register("sma_cross", SmaCrossStrategy)
        strategy = registry.create("sma_cross", fast=10, slow=30)
    """

    def __init__(self):
        self._strategies: dict[str, type[Strategy]] = {}

    def register(self, name: str, cls: type[Strategy]) -> None:
        """Register a strategy class."""
        if not issubclass(cls, Strategy):
            raise TypeError(f"{cls.__name__} must inherit Strategy base class")
        self._strategies[name] = cls

    def get(self, name: str) -> type[Strategy]:
        """Fetch strategy class by name."""
        if name not in self._strategies:
            raise KeyError(f"Strategy not registered: {name}. Registered: {list(self._strategies.keys())}")
        return self._strategies[name]

    def list(self) -> list[str]:
        """Return all registered strategy names."""
        return sorted(self._strategies.keys())

    def create(self, name: str, **kwargs) -> Strategy:
        """Instantiate strategy by name and parameters."""
        cls = self.get(name)
        return cls(**kwargs)

    def unregister(self, name: str) -> None:
        """Unregister a strategy."""
        self._strategies.pop(name, None)


def register_strategy(name: str) -> Callable[[type[Strategy]], type[Strategy]]:
    """
    Decorator: register strategy class in the global registry.

    Usage:
        @register_strategy("sma_cross")
        class SmaCrossStrategy(Strategy): ...
    """

    def decorator(cls: type[Strategy]) -> type[Strategy]:
        _global_registry.register(name, cls)
        return cls

    return decorator


# Global registry instance
_global_registry = StrategyRegistry()


def get_global_registry() -> StrategyRegistry:
    """Fetch the global strategy registry."""
    return _global_registry
