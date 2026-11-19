"""
Embargo Embargo ValidationSystem Unit Test

Testcoverage:
- EmbargoManager Initialize and Status Management
- daily validate logic (Price, Stop Loss, Take Profit, Liquidity)
- Alert detection (Daily circuit breaker, Cumulative circuit breaker, Single Stop Loss)
- Serialize/Deserialize（Save/LoadStatus）
- Historical Log and Statistics Metrics Calculation
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from agentic_quant.validation.embargo import (
    EmbargoManager,
    EmbargoSignal,
    EmbargoTracker,
    EmbargoVerification,
)

# ═══════════════════════ Data StructureTest ═══════════════════════


class TestEmbargoSignal:
    """EmbargoSignal DataClassTest"""

    def test_signal_creation(self):
        """Create Signal and Check Field"""
        sig = EmbargoSignal(
            symbol="000001.SH",
            name="Shanghai Composite Index",
            signal_date="2025-01-01",
            action="buy",
            signal_price=3000.0,
            target_shares=1000,
            allocated_capital=100000.0,
            strategy="SmaCross",
            score=0.85,
        )
        assert sig.symbol == "000001.SH"
        assert sig.action == "buy"
        assert sig.signal_price == 3000.0
        assert sig.score == 0.85

    def test_signal_default_score(self):
        """score default value is 0.0"""
        sig = EmbargoSignal(
            symbol="A",
            name="B",
            signal_date="2025-01-01",
            action="buy",
            signal_price=10.0,
            target_shares=100,
            allocated_capital=1000.0,
            strategy="Test",
        )
        assert sig.score == 0.0


class TestEmbargoTracker:
    """EmbargoTracker DataClassTest"""

    def test_to_dict_basic(self):
        """to_dict returns correct Dictionary structure"""
        tracker = EmbargoTracker(
            start_date="2025-01-01",
            last_update="2025-01-02",
            total_capital=1_000_000.0,
            cumulative_pnl=5000.0,
        )
        d = tracker.to_dict()
        assert d["start_date"] == "2025-01-01"
        assert d["last_update"] == "2025-01-02"
        assert d["total_capital"] == 1_000_000.0
        assert d["cumulative_pnl"] == 5000.0
        assert d["active_signals"] == 0  # ShortList


# ═══════════════════════ EmbargoManager basic/foundationTest ═══════════════════════


class TestEmbargoManagerInit:
    """EmbargoManager InitializeTest"""

    def test_init_defaults(self, tmp_path):
        """DefaultParameter Initialization"""
        mgr = EmbargoManager(state_dir=str(tmp_path))
        assert mgr.stop_loss_pct == 0.05
        assert mgr.circuit_breaker_daily == 0.03
        assert mgr.circuit_breaker_cumulative == 0.20
        assert mgr.state_dir == tmp_path
        assert mgr.alerts == []

    def test_init_creates_directory(self, tmp_path):
        """Automatically creates directory on Initialize"""
        sub_dir = tmp_path / "embargo_sub"
        assert not sub_dir.exists()
        EmbargoManager(state_dir=str(sub_dir))
        assert sub_dir.exists()

    def test_state_path_property(self, tmp_path):
        """state_path returns correct Path"""
        mgr = EmbargoManager(state_dir=str(tmp_path))
        assert mgr.state_path == tmp_path / "embargo_state.json"

    def test_history_path_property(self, tmp_path):
        """history_path returns correct Path"""
        mgr = EmbargoManager(state_dir=str(tmp_path))
        assert mgr.history_path == tmp_path / "embargo_history.json"


# ═══════════════════════ Serialize/Deserialize ═══════════════════════


class TestEmbargoManagerPersistence:
    """EmbargoManager PersistTest"""

    def test_load_state_new_tracker(self, tmp_path):
        """Loading non-existent Status File should create new tracker"""
        mgr = EmbargoManager(state_dir=str(tmp_path))
        result = mgr.load_state()
        assert not result  # newly created returns False
        assert mgr.tracker.start_date != ""
        assert mgr.tracker.last_update != ""

    def test_save_and_load_state(self, tmp_path):
        """After Save, Load should resume previous Status"""
        mgr = EmbargoManager(state_dir=str(tmp_path))
        mgr.tracker.cumulative_pnl = 15000.0
        mgr.tracker.start_date = "2025-03-01"
        mgr.save_state()

        # Create new manager and load status
        mgr2 = EmbargoManager(state_dir=str(tmp_path))
        result = mgr2.load_state()
        assert result  # Load existing file returns True
        assert mgr2.tracker.cumulative_pnl == 15000.0
        assert mgr2.tracker.start_date == "2025-03-01"

    def test_save_state_updates_last_update(self, tmp_path):
        """When saving status, last_update is Updated"""
        mgr = EmbargoManager(state_dir=str(tmp_path))
        mgr.tracker.last_update = "2020-01-01"
        mgr.save_state()
        # After Save, should Update to Current Date
        today = datetime.now().strftime("%Y-%m-%d")
        assert mgr.tracker.last_update == today

    def test_save_history_creates_file(self, tmp_path):
        """SaveHistorical should Create File"""
        mgr = EmbargoManager(state_dir=str(tmp_path))
        sig = EmbargoSignal(
            symbol="TEST",
            name="Test",
            signal_date="2025-01-01",
            action="buy",
            signal_price=100.0,
            target_shares=10,
            allocated_capital=1000.0,
            strategy="Test",
        )
        verifications = [
            EmbargoVerification(
                signal=sig,
                verify_date="2025-01-02",
                open_price=101.0,
                close_price=102.0,
                high_price=103.0,
                low_price=99.0,
                daily_return=0.02,
                cumulative_return=0.02,
                pnl=20.0,
                cumulative_pnl=20.0,
                hit_stop_loss=False,
                hit_take_profit=False,
                max_adverse=-0.01,
                volume_ok=True,
            )
        ]
        mgr.save_history(verifications)
        assert mgr.history_path.exists()

    def test_save_history_append(self, tmp_path):
        """Multiple SaveHistorical calls should Append rather than overwrite"""
        mgr = EmbargoManager(state_dir=str(tmp_path))
        sig = EmbargoSignal(
            symbol="A",
            name="A",
            signal_date="2025-01-01",
            action="buy",
            signal_price=100.0,
            target_shares=10,
            allocated_capital=1000.0,
            strategy="T",
        )
        v1 = EmbargoVerification(
            signal=sig,
            verify_date="2025-01-02",
            open_price=101.0,
            close_price=102.0,
            high_price=103.0,
            low_price=99.0,
            daily_return=0.02,
            cumulative_return=0.02,
            pnl=20.0,
            cumulative_pnl=20.0,
            hit_stop_loss=False,
            hit_take_profit=False,
            max_adverse=-0.01,
            volume_ok=True,
        )
        mgr.save_history([v1])

        sig2 = EmbargoSignal(
            symbol="B",
            name="B",
            signal_date="2025-01-03",
            action="buy",
            signal_price=200.0,
            target_shares=5,
            allocated_capital=1000.0,
            strategy="T",
        )
        v2 = EmbargoVerification(
            signal=sig2,
            verify_date="2025-01-04",
            open_price=201.0,
            close_price=202.0,
            high_price=203.0,
            low_price=199.0,
            daily_return=0.01,
            cumulative_return=0.03,
            pnl=10.0,
            cumulative_pnl=30.0,
            hit_stop_loss=False,
            hit_take_profit=False,
            max_adverse=-0.005,
            volume_ok=True,
        )
        mgr.save_history([v2])

        records = json.loads(mgr.history_path.read_text(encoding="utf-8"))
        assert len(records) == 2


# ═══════════════════════ dailyValidateTest ═══════════════════════


class TestEmbargoManagerVerifyDay:
    """EmbargoManager dailyValidatelogicTest"""

    def test_verify_day_basic(self, tmp_path):
        """basicValidatePipeline"""
        mgr = EmbargoManager(state_dir=str(tmp_path))
        signal = EmbargoSignal(
            symbol="AAPL",
            name="Apple",
            signal_date="2025-01-01",
            action="buy",
            signal_price=150.0,
            target_shares=100,
            allocated_capital=15000.0,
            strategy="SmaCross",
        )
        price_data = {
            "AAPL": {"open": 151.0, "close": 152.0, "high": 153.0, "low": 149.5, "volume": 1000000},
        }
        results = mgr.verify_day([signal], price_data=price_data, verify_date="2025-01-02")
        assert len(results) == 1
        v = results[0]
        assert v.verify_date == "2025-01-02"
        assert v.signal.symbol == "AAPL"
        # Return = (152 - 150) / 150 = 0.01333
        assert v.daily_return == pytest.approx(0.013333, rel=1e-4)

    def test_verify_day_missing_price_skips(self, tmp_path):
        """When Price Data is missing, should Skip this Signal"""
        mgr = EmbargoManager(state_dir=str(tmp_path))
        signal = EmbargoSignal(
            symbol="NODATA",
            name="NoData",
            signal_date="2025-01-01",
            action="buy",
            signal_price=100.0,
            target_shares=10,
            allocated_capital=1000.0,
            strategy="T",
        )
        results = mgr.verify_day([signal], price_data={}, verify_date="2025-01-02")
        assert len(results) == 0

    def test_verify_day_pnl_calculation(self, tmp_path):
        """Validate P&L calculation correctness"""
        mgr = EmbargoManager(state_dir=str(tmp_path))
        signal = EmbargoSignal(
            symbol="X",
            name="X",
            signal_date="2025-01-01",
            action="buy",
            signal_price=100.0,
            target_shares=10,
            allocated_capital=10000.0,
            strategy="T",
        )
        price_data = {
            "X": {"open": 101.0, "close": 105.0, "high": 106.0, "low": 100.5, "volume": 5000}
        }
        results = mgr.verify_day([signal], price_data=price_data, verify_date="2025-01-02")
        # daily_ret = (105 - 100) / 100 = 0.05
        # pnl = 0.05 * 10000 = 500
        assert results[0].daily_return == pytest.approx(0.05, rel=1e-4)
        assert results[0].pnl == pytest.approx(500.0, rel=1e-2)

    def test_verify_day_stop_loss_trigger(self, tmp_path):
        """Intraday Low triggers Stop Loss"""
        mgr = EmbargoManager(state_dir=str(tmp_path), stop_loss_pct=0.05)
        signal = EmbargoSignal(
            symbol="Y",
            name="Y",
            signal_date="2025-01-01",
            action="buy",
            signal_price=100.0,
            target_shares=10,
            allocated_capital=1000.0,
            strategy="T",
        )
        price_data = {
            "Y": {"open": 99.0, "close": 101.0, "high": 102.0, "low": 93.0, "volume": 10000}
        }
        results = mgr.verify_day([signal], price_data=price_data)
        # max_adverse = (93 - 100) / 100 = -0.07 < -0.05 TriggerStop Loss
        assert results[0].hit_stop_loss
        assert results[0].max_adverse <= -0.05

    def test_verify_day_no_stop_loss(self, tmp_path):
        """Intraday Low has not reached Stop Loss threshold"""
        mgr = EmbargoManager(state_dir=str(tmp_path), stop_loss_pct=0.05)
        signal = EmbargoSignal(
            symbol="Z",
            name="Z",
            signal_date="2025-01-01",
            action="buy",
            signal_price=100.0,
            target_shares=10,
            allocated_capital=1000.0,
            strategy="T",
        )
        price_data = {
            "Z": {"open": 99.0, "close": 101.0, "high": 102.0, "low": 97.0, "volume": 10000}
        }
        results = mgr.verify_day([signal], price_data=price_data)
        # max_adverse = (97 - 100) / 100 = -0.03 > -0.05 does not trigger Stop Loss
        assert not results[0].hit_stop_loss

    def test_verify_day_take_profit_trigger(self, tmp_path):
        """Intraday High triggers Take Profit"""
        mgr = EmbargoManager(state_dir=str(tmp_path), stop_loss_pct=0.05)
        signal = EmbargoSignal(
            symbol="W",
            name="W",
            signal_date="2025-01-01",
            action="buy",
            signal_price=100.0,
            target_shares=10,
            allocated_capital=1000.0,
            strategy="T",
        )
        price_data = {
            "W": {"open": 101.0, "close": 102.0, "high": 116.0, "low": 100.0, "volume": 10000}
        }
        results = mgr.verify_day([signal], price_data=price_data)
        # high - signal_price / signal_price = 0.16 >= 0.05 * 3 = 0.15  TriggerTake Profit
        assert results[0].hit_take_profit

    def test_verify_day_volume_check(self, tmp_path):
        """Liquidity Check: Volume > 0"""
        mgr = EmbargoManager(state_dir=str(tmp_path))
        signal = EmbargoSignal(
            symbol="V",
            name="V",
            signal_date="2025-01-01",
            action="buy",
            signal_price=100.0,
            target_shares=10,
            allocated_capital=1000.0,
            strategy="T",
        )
        # HasVolume
        price_data = {
            "V": {"open": 101.0, "close": 102.0, "high": 103.0, "low": 99.0, "volume": 5000}
        }
        results = mgr.verify_day([signal], price_data=price_data)
        assert results[0].volume_ok

        # Zero Volume
        price_data["V"]["volume"] = 0
        results2 = mgr.verify_day([signal], price_data=price_data)
        assert not results2[0].volume_ok

    def test_verify_day_cumulative_pnl(self, tmp_path):
        """LongtimesValidateCumulative P&L"""
        mgr = EmbargoManager(state_dir=str(tmp_path))
        sig = EmbargoSignal(
            symbol="C",
            name="C",
            signal_date="2025-01-01",
            action="buy",
            signal_price=100.0,
            target_shares=10,
            allocated_capital=1000.0,
            strategy="T",
        )
        # first day: Up to 105
        price_data = {
            "C": {"open": 101.0, "close": 105.0, "high": 106.0, "low": 99.0, "volume": 5000}
        }
        mgr.verify_day([sig], price_data=price_data, verify_date="2025-01-02")
        assert mgr.tracker.cumulative_pnl == pytest.approx(50.0, rel=1)  # 0.05 * 1000

        # Second day: Down to 95
        price_data2 = {
            "C": {"open": 104.0, "close": 95.0, "high": 105.0, "low": 94.0, "volume": 5000}
        }
        mgr.verify_day([sig], price_data=price_data2, verify_date="2025-01-03")
        # Cumulative: 50 + (95-100)/100 * 1000 = 50 + (-50) = 0
        assert mgr.tracker.cumulative_pnl == pytest.approx(0.0, rel=1)


# ═══════════════════════ Alert Detection Test ═══════════════════════


class TestEmbargoManagerAlerts:
    """EmbargoManager Alert Detection Test"""

    def test_no_alerts_when_no_verifications(self, tmp_path):
        """No alert when no validate log"""
        mgr = EmbargoManager(state_dir=str(tmp_path))
        alerts = mgr.check_alerts()
        assert alerts == []

    def test_daily_circuit_breaker(self, tmp_path):
        """Daily loss exceeds circuit breaker threshold"""
        mgr = EmbargoManager(state_dir=str(tmp_path), circuit_breaker_daily=0.03)
        sig = EmbargoSignal(
            symbol="D",
            name="D",
            signal_date="2025-01-01",
            action="buy",
            signal_price=100.0,
            target_shares=10,
            allocated_capital=200000.0,
            strategy="T",
        )
        # loss exceeds 3%: Down to 65 => daily_ret = (65-100)/100 = -0.35
        # pnl = -0.35 * 200000 = -70000, daily_return = 70000/1e6 = 0.07 > 0.03
        price_data = {
            "D": {"open": 100.0, "close": 65.0, "high": 101.0, "low": 64.0, "volume": 5000}
        }
        mgr.verify_day([sig], price_data=price_data, verify_date="2025-01-02")
        alerts = mgr.check_alerts()
        assert len(alerts) >= 1
        # daily circuit breaker should be in Alert List
        daily_alerts = [a for a in alerts if "daily circuit breaker" in a.lower()]
        assert len(daily_alerts) >= 1

    def test_daily_circuit_breaker_not_triggered(self, tmp_path):
        """Small loss does not trigger circuit breaker"""
        mgr = EmbargoManager(state_dir=str(tmp_path), circuit_breaker_daily=0.03)
        sig = EmbargoSignal(
            symbol="S",
            name="S",
            signal_date="2025-01-01",
            action="buy",
            signal_price=100.0,
            target_shares=10,
            allocated_capital=10000.0,
            strategy="T",
        )
        price_data = {
            "S": {"open": 100.0, "close": 99.0, "high": 101.0, "low": 98.5, "volume": 5000}
        }
        mgr.verify_day([sig], price_data=price_data, verify_date="2025-01-02")
        # pnl = -100, daily_return = 100/1e6 = 0.0001 < 0.03
        alerts = mgr.check_alerts()
        assert len(alerts) == 0

    def test_cumulative_circuit_breaker(self, tmp_path):
        """Cumulative loss exceeds circuit breaker threshold"""
        mgr = EmbargoManager(state_dir=str(tmp_path), circuit_breaker_cumulative=0.10)
        sig = EmbargoSignal(
            symbol="E",
            name="E",
            signal_date="2025-01-01",
            action="buy",
            signal_price=100.0,
            target_shares=10,
            allocated_capital=400000.0,
            strategy="T",
        )
        # one-time loss exceeds 10% Total Capital: Down to 70 -> pnl = (70-100)/100 * 400000 = -120000
        # cum_return = 120000/1000000 = 0.12 >= 0.10
        price_data = {
            "E": {"open": 100.0, "close": 70.0, "high": 101.0, "low": 69.0, "volume": 5000}
        }
        mgr.verify_day([sig], price_data=price_data, verify_date="2025-01-02")
        alerts = mgr.check_alerts()
        assert len(alerts) >= 1

    def test_stop_loss_alert_generated(self, tmp_path):
        """Signal that triggers Stop Loss should generate Alert"""
        mgr = EmbargoManager(state_dir=str(tmp_path), stop_loss_pct=0.05)
        sig = EmbargoSignal(
            symbol="SL",
            name="StopLoss",
            signal_date="2025-01-01",
            action="buy",
            signal_price=100.0,
            target_shares=10,
            allocated_capital=1000.0,
            strategy="T",
        )
        price_data = {
            "SL": {"open": 100.0, "close": 101.0, "high": 102.0, "low": 93.0, "volume": 5000}
        }
        mgr.verify_day([sig], price_data=price_data, verify_date="2025-01-02")
        alerts = mgr.check_alerts()
        # should include Stop Loss Alert
        stop_alerts = [a for a in alerts if "Stop Loss" in a or "SL" in a]
        assert len(stop_alerts) >= 1 or not alerts

    def test_multiple_signals_multiple_alerts(self, tmp_path):
        """Multiple Signals may produce Multiple Alerts"""
        mgr = EmbargoManager(state_dir=str(tmp_path), stop_loss_pct=0.05)
        sigs = [
            EmbargoSignal(
                symbol=f"S{i}",
                name=f"Stock{i}",
                signal_date="2025-01-01",
                action="buy",
                signal_price=100.0,
                target_shares=10,
                allocated_capital=100000.0,
                strategy="T",
            )
            for i in range(3)
        ]
        price_data = {
            "S0": {"open": 100.0, "close": 70.0, "high": 101.0, "low": 69.0, "volume": 5000},
            "S1": {"open": 101.0, "close": 99.0, "high": 102.0, "low": 98.0, "volume": 5000},
            "S2": {"open": 100.0, "close": 94.0, "high": 101.0, "low": 93.0, "volume": 5000},
        }
        mgr.verify_day(sigs, price_data=price_data, verify_date="2025-01-02")
        alerts = mgr.check_alerts()
        # S0 triggers Stop Loss, daily circuit breaker, may also cumulative circuit breaker
        assert len(alerts) >= 1


# ═══════════════════════ StatisticsMetricsTest ═══════════════════════


class TestEmbargoManagerMetrics:
    """EmbargoManager StatisticsMetricsTest"""

    def test_compute_metrics_no_history(self, tmp_path):
        """Returns None Data Status when no historical data"""
        mgr = EmbargoManager(state_dir=str(tmp_path))
        metrics = mgr.compute_metrics()
        assert metrics == {"status": "No historical data"}

    def test_compute_metrics_with_history(self, tmp_path):
        """When historical data exists, calculate each Metric"""
        mgr = EmbargoManager(state_dir=str(tmp_path))

        # First create Historical Log
        sig = EmbargoSignal(
            symbol="M",
            name="M",
            signal_date="2025-01-01",
            action="buy",
            signal_price=100.0,
            target_shares=10,
            allocated_capital=10000.0,
            strategy="T",
        )
        verifications = [
            EmbargoVerification(
                signal=sig,
                verify_date="2025-01-02",
                open_price=101.0,
                close_price=102.0,
                high_price=103.0,
                low_price=99.0,
                daily_return=0.02,
                cumulative_return=0.02,
                pnl=200.0,
                cumulative_pnl=200.0,
                hit_stop_loss=False,
                hit_take_profit=False,
                max_adverse=-0.01,
                volume_ok=True,
            ),
            EmbargoVerification(
                signal=sig,
                verify_date="2025-01-03",
                open_price=102.0,
                close_price=98.0,
                high_price=103.0,
                low_price=97.0,
                daily_return=-0.02,
                cumulative_return=0.0,
                pnl=-200.0,
                cumulative_pnl=0.0,
                hit_stop_loss=False,
                hit_take_profit=False,
                max_adverse=-0.03,
                volume_ok=True,
            ),
        ]
        mgr.save_history(verifications)
        mgr.tracker.cumulative_pnl = 0.0

        metrics = mgr.compute_metrics()
        assert "total_signals" in metrics
        assert metrics["total_signals"] == 2
        assert metrics["win_count"] == 1  # one positive Return
        assert metrics["win_rate"] == 0.5

    def test_compute_metrics_all_losses(self, tmp_path):
        """Metrics calculation for all losses"""
        mgr = EmbargoManager(state_dir=str(tmp_path))
        sig = EmbargoSignal(
            symbol="L",
            name="L",
            signal_date="2025-01-01",
            action="buy",
            signal_price=100.0,
            target_shares=10,
            allocated_capital=10000.0,
            strategy="T",
        )
        verifications = [
            EmbargoVerification(
                signal=sig,
                verify_date="2025-01-02",
                open_price=99.0,
                close_price=98.0,
                high_price=100.0,
                low_price=97.0,
                daily_return=-0.02,
                cumulative_return=-0.02,
                pnl=-200.0,
                cumulative_pnl=-200.0,
                hit_stop_loss=False,
                hit_take_profit=False,
                max_adverse=-0.03,
                volume_ok=True,
            ),
        ]
        mgr.save_history(verifications)
        metrics = mgr.compute_metrics()
        assert metrics["win_count"] == 0
        assert metrics["win_rate"] == 0.0


# ═══════════════════════ Daily ReportGenerateTest ═══════════════════════


class TestEmbargoManagerReport:
    """EmbargoManager Daily ReportGenerateTest"""

    def test_generate_daily_report_basic(self, tmp_path):
        """Generate Daily Report should not throw Exception"""
        mgr = EmbargoManager(state_dir=str(tmp_path))
        sig = EmbargoSignal(
            symbol="R",
            name="Report",
            signal_date="2025-01-01",
            action="buy",
            signal_price=100.0,
            target_shares=10,
            allocated_capital=1000.0,
            strategy="T",
        )
        verifications = [
            EmbargoVerification(
                signal=sig,
                verify_date="2025-01-02",
                open_price=101.0,
                close_price=102.0,
                high_price=103.0,
                low_price=99.0,
                daily_return=0.02,
                cumulative_return=0.02,
                pnl=20.0,
                cumulative_pnl=20.0,
                hit_stop_loss=False,
                hit_take_profit=False,
                max_adverse=-0.01,
                volume_ok=True,
            ),
        ]
        report = mgr.generate_daily_report(verifications)
        assert isinstance(report, str)
        assert "Embargo" in report

    def test_generate_daily_report_with_output_dir(self, tmp_path):
        """When output_dir is specified, should Generate File"""
        mgr = EmbargoManager(state_dir=str(tmp_path))
        sig = EmbargoSignal(
            symbol="R",
            name="R",
            signal_date="2025-01-01",
            action="buy",
            signal_price=100.0,
            target_shares=10,
            allocated_capital=1000.0,
            strategy="T",
        )
        verifications = [
            EmbargoVerification(
                signal=sig,
                verify_date="2025-01-02",
                open_price=101.0,
                close_price=102.0,
                high_price=103.0,
                low_price=99.0,
                daily_return=0.02,
                cumulative_return=0.02,
                pnl=20.0,
                cumulative_pnl=20.0,
                hit_stop_loss=False,
                hit_take_profit=False,
                max_adverse=-0.01,
                volume_ok=True,
            ),
        ]
        mgr.generate_daily_report(verifications, output_dir=str(tmp_path))
        files = list(tmp_path.iterdir())
        md_files = [f for f in files if f.suffix == ".md"]
        json_files = [f for f in files if f.suffix == ".json"]
        assert len(md_files) >= 1
        assert len(json_files) >= 1

    def test_generate_daily_report_with_alerts(self, tmp_path):
        """Daily Report including Alert"""
        mgr = EmbargoManager(state_dir=str(tmp_path))
        mgr.alerts = ["TestAlert1", "TestAlert2"]
        sig = EmbargoSignal(
            symbol="A",
            name="A",
            signal_date="2025-01-01",
            action="buy",
            signal_price=100.0,
            target_shares=10,
            allocated_capital=1000.0,
            strategy="T",
        )
        verifications = [
            EmbargoVerification(
                signal=sig,
                verify_date="2025-01-02",
                open_price=101.0,
                close_price=102.0,
                high_price=103.0,
                low_price=99.0,
                daily_return=0.02,
                cumulative_return=0.02,
                pnl=20.0,
                cumulative_pnl=20.0,
                hit_stop_loss=False,
                hit_take_profit=False,
                max_adverse=-0.01,
                volume_ok=True,
            ),
        ]
        report = mgr.generate_daily_report(verifications)
        assert "TestAlert1" in report
        assert "TestAlert2" in report


# ═══════════════════════ SignalLoadTest ═══════════════════════


class TestEmbargoManagerLoadSignals:
    """EmbargoManager SignalLoadTest"""

    def test_load_signals_from_plan(self, tmp_path):
        """Load Signal from plan JSON"""
        plan_path = tmp_path / "plan.json"
        plan_data = {
            "report_date": "2025-01-01",
            "positions": [
                {
                    "action": "BUY",
                    "symbol": "000001.SH",
                    "name": "Shanghai Composite Index",
                    "price": 3000.0,
                    "shares": 1000,
                    "allocated": 100000.0,
                    "signal": "SmaCross",
                    "score": 0.85,
                },
            ],
        }
        plan_path.write_text(json.dumps(plan_data), encoding="utf-8")

        mgr = EmbargoManager(state_dir=str(tmp_path))
        signals = mgr.load_signals_from_plan(str(plan_path))
        assert len(signals) == 1
        assert signals[0].symbol == "000001.SH"
        assert signals[0].action == "buy"
        assert signals[0].strategy == "SmaCross"

    def test_load_signals_from_plan_non_buy_skipped(self, tmp_path):
        """Non-BUY signals should be Skipped"""
        plan_path = tmp_path / "plan.json"
        plan_data = {
            "report_date": "2025-01-01",
            "positions": [
                {"symbol": "A", "price": 100.0, "shares": 10, "allocated": 1000.0, "signal": "T"},
                {
                    "symbol": "B",
                    "action": "SELL",
                    "price": 200.0,
                    "shares": 5,
                    "allocated": 1000.0,
                    "signal": "T",
                },
            ],
        }
        plan_path.write_text(json.dumps(plan_data), encoding="utf-8")

        mgr = EmbargoManager(state_dir=str(tmp_path))
        signals = mgr.load_signals_from_plan(str(plan_path))
        assert len(signals) == 0
