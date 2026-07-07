"""NY_IB stream tests: BRACKET exit semantics, EOD flat, live producer."""
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from config import NY_IB_MODEL, RISK_MODEL, STRATEGY
from src.backtest.engine import BacktestEngine, TradeRecord
from src.backtest.execution import ExecutionEngine


# ---------------------------------------------------------------- engine side

def _bare_engine():
    """Minimal engine instance for _check_exit unit tests (skip full init)."""
    eng = BacktestEngine.__new__(BacktestEngine)
    eng.execution = ExecutionEngine()
    eng.time_filter = SimpleNamespace(should_close_for_rollover=lambda ts: False)
    eng.max_trade_duration = 240
    eng._nyib_mins = (990, 1050, 1320, 1380)  # 16:30/17:30/22:00/23:00
    eng.state = SimpleNamespace(in_position=True, current_trade=None,
                                position_entry_idx=0)
    eng._exits = []
    eng._exit_position = lambda price, reason, candle, verbose: eng._exits.append(
        (price, reason))
    return eng


def _trade(exit_style="BRACKET", entry_model="NY_IB"):
    t = TradeRecord(entry_time=datetime(2025, 1, 6, 18, 0))
    t.direction = "LONG"
    t.entry_price = 100.0
    t.sl_price = 95.0
    t.original_sl = 95.0
    t.tp_price = 110.0
    t.exit_style = exit_style
    t.entry_model = entry_model
    t.best_price_in_favor = 100.0
    t.worst_price_against = 100.0
    t.trailing_active = False
    t.sl_moved_to_be = False
    t.exit_tier = ""
    return t


def _candle(ts, high, low, close, atr=2.0):
    return pd.Series({"timestamp": pd.Timestamp(ts), "open": close,
                      "high": high, "low": low, "close": close, "atr": atr})


def test_bracket_never_moves_to_breakeven():
    eng = _bare_engine()
    eng.state.current_trade = _trade("BRACKET")
    eng.state.position_entry_idx = 10
    # +3R excursion; global move_sl_to_be_at_r would normally fire
    candle = _candle("2025-01-06 18:00", high=106.0, low=99.0, close=105.5)
    eng._check_exit(pd.DataFrame(), 12, candle, verbose=False)
    assert eng.state.current_trade.sl_price == 95.0
    assert not eng.state.current_trade.sl_moved_to_be
    assert eng._exits == []


def test_non_bracket_does_move_to_breakeven():
    if STRATEGY.get("move_sl_to_be_at_r", 0) <= 0:
        pytest.skip("BE move disabled in config")
    eng = _bare_engine()
    eng.state.current_trade = _trade(exit_style="", entry_model="AMD")
    eng.state.position_entry_idx = 10
    candle = _candle("2025-01-06 18:00", high=106.0, low=99.0, close=105.5)
    eng._check_exit(pd.DataFrame(), 12, candle, verbose=False)
    assert eng.state.current_trade.sl_price > 95.0  # BE (or trail) moved it


def test_bracket_never_trails():
    eng = _bare_engine()
    eng.state.current_trade = _trade("BRACKET")
    eng.state.position_entry_idx = 10
    # Huge excursion that would activate any trailing config
    candle = _candle("2025-01-06 18:00", high=109.5, low=99.5, close=109.0)
    eng._check_exit(pd.DataFrame(), 12, candle, verbose=False)
    assert eng.state.current_trade.sl_price == 95.0
    assert not eng.state.current_trade.trailing_active


def test_eod_flat_fires_at_2300():
    eng = _bare_engine()
    eng.state.current_trade = _trade("BRACKET")
    eng.state.position_entry_idx = 10
    candle = _candle("2025-01-06 23:00", high=100.5, low=99.5, close=100.2)
    eng._check_exit(pd.DataFrame(), 12, candle, verbose=False)
    assert eng._exits and eng._exits[0][1] == "EOD_FLAT"


def test_eod_flat_only_for_ny_ib():
    eng = _bare_engine()
    eng.state.current_trade = _trade(exit_style="", entry_model="AMD")
    eng.state.position_entry_idx = 10
    candle = _candle("2025-01-06 23:00", high=100.5, low=99.5, close=100.2)
    eng._check_exit(pd.DataFrame(), 12, candle, verbose=False)
    assert all(reason != "EOD_FLAT" for _, reason in eng._exits)


# ---------------------------------------------------------------- live side

@pytest.fixture
def nyib_enabled():
    saved = dict(NY_IB_MODEL)
    NY_IB_MODEL["enabled"] = True
    yield
    NY_IB_MODEL.clear()
    NY_IB_MODEL.update(saved)


def _live_day_df(price=2400.0, ib_range_pct=0.006, breakout=True,
                 breakout_at_end=True):
    """Synthetic broker-time M5 day: IB 16:30-17:25 + post-IB bars."""
    rows = []
    ib_size = price * ib_range_pct
    lo, hi = price - ib_size / 2, price + ib_size / 2
    ts = pd.date_range("2025-01-06 16:30", periods=12, freq="5min")
    for i, t in enumerate(ts):  # IB hour: oscillate within the range
        rows.append({"timestamp": t, "open": lo + 1, "high": hi, "low": lo,
                     "close": lo + (ib_size if i % 2 else 1), "volume": 100})
    post = pd.date_range("2025-01-06 17:30", periods=24, freq="5min")
    for t in post[:-1]:
        rows.append({"timestamp": t, "open": price, "high": hi - 0.5,
                     "low": lo + 0.5, "close": price, "volume": 100})
    last_close = hi + 2.0 if breakout else price  # close beyond IB high
    rows.append({"timestamp": post[-1], "open": price, "high": last_close + 0.5,
                 "low": price - 1, "close": last_close, "volume": 100})
    df = pd.DataFrame(rows).reset_index(drop=True)
    if breakout and not breakout_at_end:
        # move the breakout bar earlier so it is stale (>3 bars back)
        df.loc[len(df) - 8, "close"] = hi + 2.0
        df.loc[len(df) - 8, "high"] = hi + 2.5
        df.loc[len(df) - 1, "close"] = price
        df.loc[len(df) - 1, "high"] = hi - 0.5
    return df


def _scanner():
    from src.live.signals import LiveSignalScanner
    return LiveSignalScanner(symbol="XAUUSD", account_balance=500.0)


def test_live_nyib_emits_signal(nyib_enabled):
    sc = _scanner()
    sigs = sc.scan_ny_ib(_live_day_df())
    assert len(sigs) == 1
    s = sigs[0]
    assert s.entry_mode == "NY_IB"
    assert s.direction == "LONG"
    ib_size = s.consolidation_high - s.consolidation_low
    assert s.entry_price == pytest.approx(
        s.consolidation_high - NY_IB_MODEL["retrace_frac"] * ib_size, abs=0.01)
    assert s.stop_loss < s.entry_price < s.take_profit
    assert s.position_size_lots >= RISK_MODEL.get("min_lot", 0.01)


def test_live_nyib_day_scoped_dedup(nyib_enabled):
    sc = _scanner()
    df = _live_day_df()
    assert len(sc.scan_ny_ib(df)) == 1
    assert sc.scan_ny_ib(df) == []  # same day: attempt consumed


def test_live_nyib_stale_breakout_skipped(nyib_enabled):
    sc = _scanner()
    sigs = sc.scan_ny_ib(_live_day_df(breakout_at_end=False))
    assert sigs == []


def test_live_nyib_no_breakout_no_signal(nyib_enabled):
    sc = _scanner()
    assert sc.scan_ny_ib(_live_day_df(breakout=False)) == []


def test_live_nyib_ib_size_gate(nyib_enabled):
    sc = _scanner()
    # 0.05% range — below ib_min_pct 0.4%
    assert sc.scan_ny_ib(_live_day_df(ib_range_pct=0.0005)) == []


def test_live_nyib_respects_disable_flag():
    """When explicitly disabled, the producer emits nothing regardless of setup."""
    sc = _scanner()
    saved = NY_IB_MODEL.get("enabled")
    NY_IB_MODEL["enabled"] = False
    try:
        assert sc.scan_ny_ib(_live_day_df()) == []
    finally:
        NY_IB_MODEL["enabled"] = saved


def test_live_nyib_telegram_format(nyib_enabled):
    from src.live.telegram_notifier import TelegramNotifier
    sc = _scanner()
    sigs = sc.scan_ny_ib(_live_day_df())
    tn = TelegramNotifier.__new__(TelegramNotifier)
    msg = tn._format_signal(sigs[0])
    assert "NY-IB" in msg and "LIMIT" in msg and "23:00" in msg
    assert "★" not in msg  # no AMD confidence stars
