"""
回测数据结构：Tick、Bar、BarSampler

参考: Advances in Financial Machine Learning, Ch.2
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Tick:
    """单条 tick 数据结构"""

    timestamp: datetime
    price: float
    volume: float
    side: str = "buy"  # buy / sell


@dataclass
class Bar:
    """
    单根 K 线（Bar）数据结构
    支持 Time Bar / Volume Bar / Dollar Bar（参考 AFML 第2章）
    """

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    dollar_volume: float = 0.0  # close * volume，用于 Dollar Bar 采样


class BarSampler:
    """
    Bar 采样器
    实现标准 Time Bar 与信息驱动 Bar（Volume Bar / Dollar Bar）
    参考: Advances in Financial Machine Learning, Ch.2
    """

    def __init__(self, bar_type: str = "time", threshold: float = 1000.0):
        """
        Args:
            bar_type: "time" | "volume" | "dollar"
            threshold: Volume/Dollar Bar 阈值；Time Bar 时传 0 即可
        """
        if bar_type not in ("time", "volume", "dollar"):
            raise ValueError(f"不支持的 bar_type: {bar_type}")
        self.bar_type = bar_type
        self.threshold = threshold

    def sample(self, ticks: list[Tick]) -> list[Bar]:
        """将 tick 列表按指定规则采样为 Bar 列表。"""
        if not ticks:
            return []

        bars: list[Bar] = []
        cur_open = ticks[0].price
        cur_high = ticks[0].price
        cur_low = ticks[0].price
        cur_close = ticks[0].price
        cur_volume = 0.0
        cur_dollar = 0.0
        cur_time = ticks[0].timestamp

        for t in ticks:
            cur_high = max(cur_high, t.price)
            cur_low = min(cur_low, t.price)
            cur_close = t.price
            cur_volume += t.volume
            cur_dollar += t.price * t.volume

            should_close = False
            if (self.bar_type == "volume" and cur_volume >= self.threshold) or (
                self.bar_type == "dollar" and cur_dollar >= self.threshold
            ):
                should_close = True
            elif self.bar_type == "time":
                # time bar 由调用方负责按时间分桶，这里只负责收集
                pass

            if should_close:
                bars.append(
                    Bar(
                        timestamp=cur_time,
                        open=cur_open,
                        high=cur_high,
                        low=cur_low,
                        close=cur_close,
                        volume=cur_volume,
                        dollar_volume=cur_dollar,
                    )
                )
                cur_open = t.price
                cur_high = t.price
                cur_low = t.price
                cur_volume = 0.0
                cur_dollar = 0.0
                cur_time = t.timestamp

        # 收尾未闭合的 bar；最后一笔 tick 恰好闭合时，计数器已归零。
        if cur_volume > 0 or cur_dollar > 0 or self.bar_type == "time":
            bars.append(
                Bar(
                    timestamp=cur_time,
                    open=cur_open,
                    high=cur_high,
                    low=cur_low,
                    close=cur_close,
                    volume=cur_volume,
                    dollar_volume=cur_dollar,
                )
            )
        return bars
