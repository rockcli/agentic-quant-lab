"""
Embargo Embargo Validation System

Reference: Advances in Financial Machine Learning (Lopez de Prado), Ch.1.3.1.6
      "Once a strategy is deployed, it follows a cursus honorum..."

Embargo Phase Definition:
- After backtest ends, strategy runs on real data after backtest end date, but does not trade
- This data is not visible during backtest, it is true out of sample
- If embargo period performance is consistent with backtest, the strategy enters paper trading

Core Features:
1. Log daily strategy signals
2. Track next day actual price changes
3. Cumulative P&L statistics
4. Circuit breaker / alert mechanism
5. Daily report generation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────── Data Structure ──────────────────────────

@dataclass
class EmbargoSignal:
    """Single entry embargo signal log"""
    symbol: str
    name: str
    signal_date: str          # Signal date
    action: str               # buy / sell / hold
    signal_price: float       # Price at signal time
    target_shares: int        # Target shares
    allocated_capital: float  # Allocated capital
    strategy: str             # Strategy source
    score: float = 0.0        # Strategy score


@dataclass
class EmbargoVerification:
    """Daily validation result"""
    signal: EmbargoSignal
    verify_date: str
    open_price: float
    close_price: float
    high_price: float
    low_price: float
    daily_return: float        # Daily return
    cumulative_return: float   # Cumulative return
    pnl: float                 # Daily P&L
    cumulative_pnl: float      # Cumulative P&L
    hit_stop_loss: bool        # Whether stop loss is triggered
    hit_take_profit: bool      # Whether take profit is triggered
    max_adverse: float         # Maximum adverse intraday volatility
    volume_ok: bool            # Whether liquidity is sufficient


@dataclass
class EmbargoTracker:
    """Embargo tracker status"""
    signals: list[EmbargoSignal] = field(default_factory=list)
    verifications: list[EmbargoVerification] = field(default_factory=list)
    cumulative_pnl: float = 0.0
    total_capital: float = 1_000_000.0
    start_date: str = ""
    last_update: str = ""

    def to_dict(self) -> dict:
        return {
            "start_date": self.start_date,
            "last_update": self.last_update,
            "total_capital": self.total_capital,
            "cumulative_pnl": round(self.cumulative_pnl, 2),
            "cumulative_return": round(self.cumulative_pnl / self.total_capital, 6),
            "active_signals": len(self.signals),
            "total_verifications": len(self.verifications),
        }


# ──────────────────────── Embargo Manager ──────────────────────

class EmbargoManager:
    """
    Embargo Embargo Validation Manager

    Usage:
        mgr = EmbargoManager(state_dir="reports/embargo")
        mgr.load_state()

        # Run daily:
        signals = mgr.load_today_signals()
        mgr.verify_day(signals)
        mgr.check_alerts()
        mgr.save_state()
    """

    def __init__(
        self,
        state_dir: str | Path = "reports/embargo",
        total_capital: float = 1_000_000.0,
        stop_loss_pct: float = 0.05,
        circuit_breaker_daily: float = 0.03,
        circuit_breaker_cumulative: float = 0.20,
    ):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = EmbargoTracker(total_capital=total_capital)
        self.stop_loss_pct = stop_loss_pct
        self.circuit_breaker_daily = circuit_breaker_daily
        self.circuit_breaker_cumulative = circuit_breaker_cumulative
        self.alerts: list[str] = []

    # ─── State persistence ───

    @property
    def state_path(self) -> Path:
        return self.state_dir / "embargo_state.json"

    @property
    def history_path(self) -> Path:
        return self.state_dir / "embargo_history.json"

    def load_state(self) -> bool:
        """Load historical tracking status"""
        if self.state_path.exists():
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.tracker.start_date = data.get("start_date", "")
            self.tracker.last_update = data.get("last_update", "")
            self.tracker.cumulative_pnl = data.get("cumulative_pnl", 0.0)
            self.tracker.total_capital = data.get("total_capital", 1_000_000.0)
            logger.info("Embargo state loaded，Cumulative P&L: ¥%.2f", self.tracker.cumulative_pnl)
            return True
        # New tracking
        self.tracker.start_date = datetime.now().strftime("%Y-%m-%d")
        self.tracker.last_update = datetime.now().strftime("%Y-%m-%d")
        self.save_state()
        return False

    def save_state(self) -> None:
        """Save current state"""
        self.tracker.last_update = datetime.now().strftime("%Y-%m-%d")
        self.state_path.write_text(
            json.dumps(self.tracker.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save_history(self, verifications: list[EmbargoVerification]) -> None:
        """Append validation history"""
        records: list[dict] = []
        if self.history_path.exists():
            records = json.loads(self.history_path.read_text(encoding="utf-8"))

        for v in verifications:
            records.append({
                "signal_date": v.signal.signal_date,
                "verify_date": v.verify_date,
                "symbol": v.signal.symbol,
                "action": v.signal.action,
                "signal_price": v.signal.signal_price,
                "close_price": v.close_price,
                "daily_return": round(v.daily_return, 6),
                "cumulative_return": round(v.cumulative_return, 6),
                "pnl": round(v.pnl, 2),
                "cumulative_pnl": round(v.cumulative_pnl, 2),
                "hit_stop_loss": v.hit_stop_loss,
                "max_adverse": round(v.max_adverse, 6),
            })

        self.history_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ─── Daily validation ───

    def load_signals_from_plan(self, plan_path: str | Path) -> list[EmbargoSignal]:
        """Load signals from trading plan JSON"""
        plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
        signals: list[EmbargoSignal] = []

        for p in plan.get("positions", []):
            if p.get("action") == "BUY":
                signals.append(EmbargoSignal(
                    symbol=p["symbol"],
                    name=p.get("name", ""),
                    signal_date=plan.get("report_date", ""),
                    action="buy",
                    signal_price=p.get("price", 0),
                    target_shares=p.get("shares", 0),
                    allocated_capital=p.get("allocated", 0),
                    strategy=p.get("signal", ""),
                    score=p.get("score", 0),
                ))

        self.tracker.signals = signals
        logger.info("Loaded %d buy signals from plan", len(signals))
        return signals

    def verify_day(
        self,
        signals: list[EmbargoSignal],
        price_data: dict[str, dict] | None = None,
        verify_date: str | None = None,
    ) -> list[EmbargoVerification]:
        """
        Validate today signal performance.

        Args:
            signals: List of signals to validate
            price_data: {symbol: {open, close, high, low, volume}} TodayPriceData
            verify_date: Validation date
        Returns:
            List of validation results
        """
        if verify_date is None:
            verify_date = datetime.now().strftime("%Y-%m-%d")

        if price_data is None:
            price_data = {}

        verifications: list[EmbargoVerification] = []
        self.alerts = []

        for sig in signals:
            prices = price_data.get(sig.symbol, {})

            if not prices:
                logger.warning("Missing price data for %s, skipping validation", sig.symbol)
                continue

            open_p = float(prices.get("open", sig.signal_price))
            close_p = float(prices.get("close", sig.signal_price))
            high_p = float(prices.get("high", sig.signal_price))
            low_p = float(prices.get("low", sig.signal_price))
            volume = float(prices.get("volume", 0))

            # Daily return
            daily_ret = (close_p - sig.signal_price) / sig.signal_price
            pnl = daily_ret * sig.allocated_capital

            # Cumulative
            self.tracker.cumulative_pnl += pnl
            cum_ret = self.tracker.cumulative_pnl / self.tracker.total_capital

            # Stop loss / take profit detection
            max_adverse = (low_p - sig.signal_price) / sig.signal_price
            hit_stop = max_adverse <= -self.stop_loss_pct
            hit_tp = (high_p - sig.signal_price) / sig.signal_price >= self.stop_loss_pct * 3

            # Liquidity check (simplified: volume > 0)
            volume_ok = volume > 0

            v = EmbargoVerification(
                signal=sig,
                verify_date=verify_date,
                open_price=open_p,
                close_price=close_p,
                high_price=high_p,
                low_price=low_p,
                daily_return=round(daily_ret, 6),
                cumulative_return=round(cum_ret, 6),
                pnl=round(pnl, 2),
                cumulative_pnl=round(self.tracker.cumulative_pnl, 2),
                hit_stop_loss=hit_stop,
                hit_take_profit=hit_tp,
                max_adverse=round(max_adverse, 6),
                volume_ok=volume_ok,
            )
            verifications.append(v)

        self.tracker.verifications = verifications
        return verifications

    # --- Alert Detection ---

    def check_alerts(self) -> list[str]:
        """Check whether circuit breaker / alert conditions are triggered"""
        self.alerts = []

        # Daily circuit breaker: only triggered on losing days, not false-alarmed on profitable days
        if not self.tracker.verifications:
            return self.alerts
        daily_pnl = sum(v.pnl for v in self.tracker.verifications)
        if daily_pnl < 0:
            daily_return = -daily_pnl / self.tracker.total_capital
            if daily_return > self.circuit_breaker_daily:
                self.alerts.append(
                    f"🔴 Daily circuit breaker triggered: daily loss {daily_return:.2%} > {self.circuit_breaker_daily:.1%}"
                )

        # Cumulative circuit breaker
        if self.tracker.cumulative_pnl < 0:
            cum_return = -self.tracker.cumulative_pnl / self.tracker.total_capital
            if cum_return > self.circuit_breaker_cumulative:
                self.alerts.append(
                    f"🔴 Cumulative circuit breaker triggered: cumulative loss {cum_return:.2%} > {self.circuit_breaker_cumulative:.0%}"
                )

        # Single stop loss
        for v in self.tracker.verifications:
            if v.hit_stop_loss:
                self.alerts.append(
                    f"⚠️ {v.signal.symbol} TriggerStop Loss: IntradayMax Drawdown {v.max_adverse:.2%}"
                )

        if self.alerts:
            logger.warning("Embargo Alert: %s", "; ".join(self.alerts))

        return self.alerts

    # ─── StatisticsMetrics ───

    def compute_metrics(self) -> dict[str, Any]:
        """Calculate embargo period statistics/metrics"""
        if self.history_path.exists():
            history = json.loads(self.history_path.read_text(encoding="utf-8"))
        else:
            history = []

        if not history:
            return {"status": "No historical data"}

        daily_rets = [h["daily_return"] for h in history]
        pnls = [h["pnl"] for h in history]

        wins = sum(1 for r in daily_rets if r > 0)
        total = len(daily_rets)
        win_rate = wins / total if total > 0 else 0

        import statistics
        mean_ret = statistics.mean(daily_rets) if daily_rets else 0
        std_ret = statistics.stdev(daily_rets) if len(daily_rets) > 1 else 1e-9
        sharpe = mean_ret / std_ret * (252 ** 0.5)

        equity_curve = [1 + h["cumulative_return"] for h in history]
        peak = equity_curve[0] if equity_curve else 0
        max_dd = 0.0
        for v in equity_curve:
            peak = max(peak, v)
            dd = (v - peak) / peak if peak != 0 else 0.0
            if dd < max_dd:
                max_dd = dd

        return {
            "total_signals": total,
            "win_count": wins,
            "win_rate": round(win_rate, 4),
            "mean_daily_return": round(mean_ret, 6),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(abs(max_dd), 6),
            "cumulative_pnl": round(self.tracker.cumulative_pnl, 2),
            "cumulative_return": round(
                self.tracker.cumulative_pnl / self.tracker.total_capital, 6
            ),
            "days_tracked": total,
        }

    # ─── Daily Report ───

    def generate_daily_report(
        self,
        verifications: list[EmbargoVerification],
        metrics: dict[str, Any] | None = None,
        output_dir: str | Path | None = None,
    ) -> str:
        """Generate Embargo Daily Report"""
        if metrics is None:
            metrics = self.compute_metrics()

        today = datetime.now().strftime("%Y-%m-%d")
        lines = [
            f"# 📋 Embargo Embargo ValidationDaily Report — {today}",
            "",
            f"**Tracking start**: {self.tracker.start_date}",
            f"**Total Capital**: ¥{self.tracker.total_capital:,.0f}",
            f"**Cumulative P&L**: ¥{self.tracker.cumulative_pnl:+,.2f}",
            f"**Cumulative return**: {self.tracker.cumulative_pnl/self.tracker.total_capital:+.4%}",
            "",
            "---",
            "",
            "## 📊 TodaySignalValidate",
            "",
            "| Code | Name | Signal Price | Close | Daily Return | P&L | Stop Loss？ | Take Profit？ |",
            "|------|------|--------|--------|--------|-----|--------|--------|",
        ]

        for v in verifications:
            stop = "⚠️ Yes" if v.hit_stop_loss else "No"
            tp = "✅ Yes" if v.hit_take_profit else "No"
            lines.append(
                f"| {v.signal.symbol} | {v.signal.name} | ¥{v.signal.signal_price:.2f} "
                f"| ¥{v.close_price:.2f} | {v.daily_return:+.2%} "
                f"| ¥{v.pnl:+,.0f} | {stop} | {tp} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 📈 Cumulative Statistics",
            "",
            "| Metrics | Value |",
            "|------|------|",
            f"| Total Signals | {metrics.get('total_signals', 0)} |",
            f"| Win Rate | {metrics.get('win_rate', 0):.1%} |",
            f"| Avg Daily Return | {metrics.get('mean_daily_return', 0):.4%} |",
            f"| Sharpe | {metrics.get('sharpe_ratio', 0):.2f} |",
            f"| Max Drawdown | {metrics.get('max_drawdown', 0):.2%} |",
            f"| Cumulative P&L | ¥{metrics.get('cumulative_pnl', 0):+,.0f} |",
            "",
        ])

        # Alert
        if self.alerts:
            lines.extend([
                "## 🚨 Alert",
                "",
            ])
            for alert in self.alerts:
                lines.append(f"- {alert}")
            lines.append("")

        # Acceptance Progress
        lines.extend([
            "---",
            "",
            "## 🎯 Embargo Acceptance Progress",
            "",
        ])

        days = metrics.get("days_tracked", 0)
        embargo_target_days = 60  # Recommend at least 2 months
        lines.extend([
            f"- Tracked: {days} days / target ≥{embargo_target_days} days ({min(100, days/embargo_target_days*100):.0f}%)",
            f"- PassedCondition: Win Rate ≥ {0.55:.0%} | OOS ReturnDecay < 50% | Max Drawdown <= backtest x 1.5",
        ])

        lines.extend([
            "",
            "> Reference: De Prado, AFML Ch.1.3.1.6 - Embargo -> Paper Trading -> Graduation -> Re-allocation",
        ])

        md = "\n".join(lines)

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            (out / f"embargo_daily_{ts}.md").write_text(md, encoding="utf-8")
            (out / f"embargo_daily_{ts}.json").write_text(
                json.dumps({
                    "date": today,
                    "verifications": [
                        {
                            "symbol": v.signal.symbol,
                            "signal_price": v.signal.signal_price,
                            "close_price": v.close_price,
                            "daily_return": v.daily_return,
                            "pnl": v.pnl,
                            "hit_stop_loss": v.hit_stop_loss,
                        }
                        for v in verifications
                    ],
                    "metrics": metrics,
                    "alerts": self.alerts,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return md
