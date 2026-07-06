"""Tests for E-adaptive rolling confidence recalibration (engine method level)."""

import pytest

from src.backtest.engine import BacktestEngine, TradeRecord
from config import CONFIDENCE_SIZING


from datetime import datetime


def _trade(label, net_pnl, r):
    t = TradeRecord(entry_time=datetime(2025, 1, 6, 12, 0))
    t.confidence_label = label
    t.net_pnl = net_pnl
    t.r_multiple = r
    return t


def _engine_with_state():
    eng = BacktestEngine.__new__(BacktestEngine)  # skip full init; test the method only
    eng.trades = []
    eng._adaptive_gated = {}
    eng.adaptive_events = []
    return eng


@pytest.fixture(autouse=True)
def _adaptive_cfg():
    saved = dict(CONFIDENCE_SIZING.get("adaptive", {}))
    CONFIDENCE_SIZING["adaptive"] = {
        "enabled": True,
        "window_trades": 60,
        "recalib_every": 20,
        "min_bucket_n": 10,
        "gate_timeout_trades": 40,
    }
    yield
    CONFIDENCE_SIZING["adaptive"] = saved


class TestAdaptiveRecalibration:
    def test_losing_bucket_gets_gated(self):
        eng = _engine_with_state()
        # 20 trades: LOW bucket 2W/10L with 1R payoffs (WR 17% < breakeven 50%)
        for i in range(10):
            eng.trades.append(_trade("LOW", -10.0, -1.0))
        eng.trades.append(_trade("LOW", 10.0, 1.0))
        eng.trades.append(_trade("LOW", 10.0, 1.0))
        for i in range(8):
            eng.trades.append(_trade("HIGH", 10.0, 1.0))
        eng._maybe_recalibrate_confidence()
        assert "LOW" in eng._adaptive_gated
        assert eng.adaptive_events[-1]["action"] == "GATED"

    def test_profitable_bucket_not_gated(self):
        eng = _engine_with_state()
        # HIGH bucket: 6W/4L with 2.5R winners (WR 60% >> breakeven ~29%)
        for i in range(6):
            eng.trades.append(_trade("HIGH", 25.0, 2.5))
        for i in range(4):
            eng.trades.append(_trade("HIGH", -10.0, -1.0))
        for i in range(10):
            eng.trades.append(_trade("GOOD", 10.0, 1.0))
        eng._maybe_recalibrate_confidence()
        assert "HIGH" not in eng._adaptive_gated

    def test_small_bucket_ignored(self):
        eng = _engine_with_state()
        # Only 5 LOW trades (< min_bucket_n=10): never gated on thin evidence
        for i in range(5):
            eng.trades.append(_trade("LOW", -10.0, -1.0))
        for i in range(15):
            eng.trades.append(_trade("HIGH", 10.0, 1.0))
        eng._maybe_recalibrate_confidence()
        assert "LOW" not in eng._adaptive_gated

    def test_only_recalibrates_on_interval(self):
        eng = _engine_with_state()
        for i in range(19):  # 19 % 20 != 0 -> no recalibration
            eng.trades.append(_trade("LOW", -10.0, -1.0))
        eng._maybe_recalibrate_confidence()
        assert "LOW" not in eng._adaptive_gated

    def test_timeout_readmission(self):
        eng = _engine_with_state()
        eng._adaptive_gated["LOW"] = 20  # gated at trade 20
        for i in range(60):
            eng.trades.append(_trade("HIGH", 10.0, 1.0))
        eng._maybe_recalibrate_confidence()  # now at trade 60: 60-20 >= 40
        assert "LOW" not in eng._adaptive_gated
        assert any(e["action"] == "READMITTED_TIMEOUT" for e in eng.adaptive_events)

    def test_disabled_is_noop(self):
        CONFIDENCE_SIZING["adaptive"]["enabled"] = False
        eng = _engine_with_state()
        for i in range(20):
            eng.trades.append(_trade("LOW", -10.0, -1.0))
        eng._maybe_recalibrate_confidence()
        assert eng._adaptive_gated == {}
