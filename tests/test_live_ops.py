"""Live-ops safety machinery: LiveMonitor kill switches (daily loss halt +
new-day resume, drawdown halt persistence), signal dedup expiry, and the
engine's catastrophe breaker hysteresis + halt_max_bars deadlock fix."""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from config import DRAWDOWN_CONTROLS, STRATEGY
from src.backtest.engine import BacktestEngine
from src.live.monitor import LiveMonitor


@pytest.fixture
def monitor(tmp_path, monkeypatch):
    """Monitor with state isolated from the real data/monitor_state.json."""
    monkeypatch.setattr(LiveMonitor, "STATE_FILE", tmp_path / "state.json")
    return LiveMonitor(initial_balance=500.0, daily_loss_limit_pct=0.008,
                       max_account_dd_pct=0.30, max_trades_per_day=3)


class TestLiveMonitorKillSwitches:
    def test_daily_loss_halts_and_new_day_resumes(self, monitor):
        day1 = datetime(2026, 7, 10, 15, 0)
        ok, _ = monitor.can_trade(day1)
        assert ok
        monitor.record_trade_result(-5.0, day1)  # limit is 500*0.8% = $4
        ok, reason = monitor.can_trade(day1)
        assert not ok and reason == "daily_loss_exceeded"
        # the daily halt clears itself on the next trading day
        ok, reason = monitor.can_trade(day1 + timedelta(days=1))
        assert ok, reason

    def test_drawdown_halt_requires_manual_resume(self, monitor):
        ts = datetime(2026, 7, 10, 15, 0)
        monitor.update_balance(340.0)  # 32% below the 500 peak
        ok, reason = monitor.can_trade(ts)
        assert not ok and reason == "account_drawdown_exceeded"
        # recovery alone does NOT clear it (manual review by design)
        monitor.update_balance(500.0)
        ok, reason = monitor.can_trade(ts)
        assert not ok and "account_drawdown" in reason
        monitor.force_resume()
        ok, _ = monitor.can_trade(ts)
        assert ok

    def test_max_trades_per_day_gate(self, monitor):
        ts = datetime(2026, 7, 10, 15, 0)
        for _ in range(3):
            monitor.record_trade_result(+1.0, ts)
        ok, reason = monitor.can_trade(ts)
        assert not ok and reason == "max_trades_per_day"


class TestSignalDedupExpiry:
    def _scanning_setup(self, monkeypatch):
        """Minimal live-scan harness (same shape as TestLiveScannerBOSParity)."""
        from src.live import signals as live_signals
        from src.strategy.manipulation import ManipulationResult
        from src.strategy.distribution import DistributionResult
        from src.strategy.market_structure import StructureBreak

        monkeypatch.setitem(STRATEGY, "entry_mode", "RETEST_ONLY")
        monkeypatch.setitem(STRATEGY, "bos_required", True)
        monkeypatch.setitem(STRATEGY, "min_confluence_score", 1)
        monkeypatch.setitem(STRATEGY, "short_min_confluence_score", 1)
        monkeypatch.setitem(STRATEGY, "retest_tolerance_atr_mult", 0.5)
        monkeypatch.setitem(STRATEGY, "rejection_wick_ratio", 1.5)

        scanner = live_signals.LiveSignalScanner(symbol="XAUUSD",
                                                 account_balance=500.0)
        n = scanner.min_bars + 40
        ts = pd.date_range("2026-01-05 09:00", periods=n, freq="5min")
        df = pd.DataFrame({
            "timestamp": ts, "open": 2000.0, "high": 2001.0,
            "low": 1999.0, "close": 2000.0, "volume": 100,
            "tick_volume": 100,
        })
        cur = n - 1
        df.loc[cur, ["open", "high", "low", "close"]] = [2001.6, 2002.0,
                                                         2000.8, 2001.9]
        manip = ManipulationResult(valid=True, direction="DOWN",
                                   extreme_price=1997.0,
                                   return_candle_idx=cur - 25, atr=1.0)
        dist = DistributionResult(valid=True, direction="UP",
                                  break_price=2003.0, break_distance=1.0,
                                  body_expansion=2.0,
                                  break_candle_idx=cur - 5, atr=1.0)
        monkeypatch.setattr(scanner, "_is_consolidation", lambda *a, **k: True)
        monkeypatch.setattr(scanner, "_find_manipulation", lambda *a, **k: manip)
        monkeypatch.setattr(scanner, "_find_distribution", lambda *a, **k: dist)
        monkeypatch.setattr(live_signals, "validate_distribution_strength",
                            lambda *a, **k: True)
        monkeypatch.setattr(
            live_signals, "find_bos_after_manipulation",
            lambda df_, ri, ed, **k: StructureBreak(
                valid=True, direction=ed, broken_level=2001.0,
                break_price=2003.0, break_candle_idx=len(df_) - 6,
                swing_idx=len(df_) - 20))
        monkeypatch.setattr(scanner.time_filter, "can_enter_trade",
                            lambda *a, **k: (True, ""))
        monkeypatch.setattr(scanner.news_filter, "can_enter_trade",
                            lambda *a, **k: (True, ""))
        monkeypatch.setattr(scanner.htf_bias, "can_enter_trade",
                            lambda *a, **k: (True, "", ""))
        monkeypatch.setattr(scanner.volume_filter, "can_enter_trade",
                            lambda *a, **k: (True, "", ""))
        return scanner, df

    def test_same_setup_not_reemitted_until_expiry(self, monkeypatch):
        scanner, df = self._scanning_setup(monkeypatch)
        assert len(scanner.scan(df)) == 1
        # identical rescan inside the expiry window -> suppressed
        assert len(scanner.scan(df)) == 0
        # age the dedup entries past the window -> re-emits
        aged = datetime.now() - timedelta(
            minutes=scanner._signal_expiry_minutes + 1)
        for k in scanner._recent_signals:
            scanner._recent_signals[k] = aged
        assert len(scanner.scan(df)) == 1


class TestBreakerHysteresis:
    @pytest.fixture(autouse=True)
    def _breaker_cfg(self, monkeypatch):
        monkeypatch.setitem(DRAWDOWN_CONTROLS, "enabled", True)
        monkeypatch.setitem(DRAWDOWN_CONTROLS, "circuit_breaker_enabled", True)
        monkeypatch.setitem(DRAWDOWN_CONTROLS, "max_account_dd_pct", 0.30)
        monkeypatch.setitem(DRAWDOWN_CONTROLS, "resume_dd_pct", 0.15)
        monkeypatch.setitem(DRAWDOWN_CONTROLS, "halt_max_bars", 100)

    def _engine(self):
        e = BacktestEngine(
            enable_session_filter=False, enable_news_filter=False,
            enable_htf_bias=False, enable_key_levels=False,
            enable_volume_filter=False, enable_fundamentals=False,
        )
        e.current_balance = 500.0
        e.equity_peak = 500.0
        return e

    def test_halts_at_threshold_resumes_on_recovery(self):
        e = self._engine()
        assert not e._drawdown_halted(100)
        e.current_balance = 340.0  # 32% DD
        assert e._drawdown_halted(101)
        e.current_balance = 430.0  # 14% DD <= resume 15%
        assert not e._drawdown_halted(102)

    def test_halt_max_bars_breaks_the_deadlock(self):
        """Flat + halted = frozen equity: without the timeout the breaker
        could latch forever (the deadlock found in the 2026-07-02 session)."""
        e = self._engine()
        e.current_balance = 340.0
        assert e._drawdown_halted(1000)      # trips
        assert e._drawdown_halted(1050)      # still deep, still halted
        assert not e._drawdown_halted(1100)  # 100 bars -> timed-out resume
        assert e.dd_halt_start_idx is None
