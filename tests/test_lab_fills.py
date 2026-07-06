"""Fill/exit rule tests: WORST_CASE parity with ExecutionEngine.check_exit,
gap-honest stops, exclusivity, EOD/timeout, trailing ratchet."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.backtest.execution import ExecutionEngine
from src.research.lab import CostModel, simulate
from src.research.strategies.base import (ENTRY_LIMIT, ENTRY_MARKET,
                                          ENTRY_STOP, MTFContext, make_signals)


def mk_df(bars, start="2025-01-06 10:00", freq_min=5, atr=2.0):
    """bars: list of (o, h, l, c)."""
    ts = pd.date_range(start, periods=len(bars), freq=f"{freq_min}min")
    df = pd.DataFrame(bars, columns=["open", "high", "low", "close"])
    df["volume"] = 100
    df["timestamp"] = ts
    df["atr"] = atr
    return df


def ctx_of(df):
    return MTFContext(tf="M5", df=df, htf={})


COSTS = CostModel.from_config()


def run_one(df, **sig):
    signals = make_signals([sig])
    trades, stats = simulate(ctx_of(df), signals, COSTS)
    return trades, stats


def test_market_fills_next_open():
    df = mk_df([(100, 101, 99, 100), (102, 103, 101, 102.5), (102.5, 104, 102, 103)])
    trades, _ = run_one(df, signal_idx=0, direction=1, entry_type=ENTRY_MARKET, sl=95.0)
    assert len(trades) == 1
    assert trades["entry"].iloc[0] == 102  # open of bar 1, not close of bar 0


def test_limit_not_touched_no_trade():
    df = mk_df([(100, 101, 99, 100), (101, 102, 100.5, 101), (101, 102, 100.5, 101)])
    trades, stats = run_one(df, signal_idx=0, direction=1, entry_type=ENTRY_LIMIT,
                            entry_price=99.0, sl=95.0, ttl_bars=2)
    assert trades.empty
    assert stats["unfilled"] == 1


def test_limit_fills_at_limit_price():
    df = mk_df([(100, 101, 99, 100), (100, 101, 98.5, 100.6), (101, 102, 100, 101)])
    trades, _ = run_one(df, signal_idx=0, direction=1, entry_type=ENTRY_LIMIT,
                        entry_price=99.0, sl=95.0, ttl_bars=2)
    assert trades["entry"].iloc[0] == 99.0


def test_stop_gap_through_fills_at_open():
    # long stop at 101, next bar opens at 103 (gap) -> fill at 103, not 101
    df = mk_df([(100, 100.5, 99, 100), (103, 104, 102.5, 103.5), (103, 104, 102, 103)])
    trades, _ = run_one(df, signal_idx=0, direction=1, entry_type=ENTRY_STOP,
                        entry_price=101.0, sl=95.0, ttl_bars=2)
    assert trades["entry"].iloc[0] == 103.0


def test_worst_case_sl_first_matches_engine():
    """Bar touches both SL and TP -> lab must exit at SL with slippage,
    bit-for-bit equal to ExecutionEngine.check_exit."""
    atr = 2.0
    df = mk_df([(100, 100.5, 99.5, 100),
                (100, 100.2, 99.8, 100),          # entry bar (market @ open 100)
                (100, 106, 94, 100)],             # touches SL 96 AND TP 105
               atr=atr)
    trades, _ = run_one(df, signal_idx=0, direction=1, entry_type=ENTRY_MARKET,
                        sl=96.0, tp=105.0)
    engine = ExecutionEngine()
    fake_trade = SimpleNamespace(direction="LONG", sl_price=96.0, tp_price=105.0)
    candle = pd.Series({"open": 100, "high": 106, "low": 94, "close": 100})
    decision = engine.check_exit(fake_trade, candle, atr)
    assert decision.exit_reason == "SL"
    assert trades["exit_reason"].iloc[0] == "SL"
    assert trades["exit"].iloc[0] == pytest.approx(decision.exit_price)


def test_worst_case_short_side():
    atr = 2.0
    df = mk_df([(100, 100.5, 99.5, 100),
                (100, 100.2, 99.8, 100),
                (100, 106, 94, 100)], atr=atr)
    trades, _ = run_one(df, signal_idx=0, direction=-1, entry_type=ENTRY_MARKET,
                        sl=104.0, tp=95.0)
    engine = ExecutionEngine()
    fake_trade = SimpleNamespace(direction="SHORT", sl_price=104.0, tp_price=95.0)
    candle = pd.Series({"open": 100, "high": 106, "low": 94, "close": 100})
    decision = engine.check_exit(fake_trade, candle, atr)
    assert trades["exit"].iloc[0] == pytest.approx(decision.exit_price)
    assert trades["exit_reason"].iloc[0] == "SL"


def test_entry_bar_sl_hit_exits_same_bar():
    df = mk_df([(100, 100.5, 99.5, 100), (100, 100.2, 95.5, 96), (96, 97, 95, 96)])
    trades, _ = run_one(df, signal_idx=0, direction=1, entry_type=ENTRY_MARKET,
                        sl=96.0, tp=110.0)
    assert trades["exit_reason"].iloc[0] == "SL"
    assert trades["bars_held"].iloc[0] == 0


def test_timeout_exits_at_close():
    bars = [(100, 100.5, 99.5, 100)] * 6
    df = mk_df(bars)
    trades, _ = run_one(df, signal_idx=0, direction=1, entry_type=ENTRY_MARKET,
                        sl=90.0, max_bars=3)
    assert trades["exit_reason"].iloc[0] == "TIMEOUT"
    assert trades["bars_held"].iloc[0] == 3
    assert trades["exit"].iloc[0] == 100  # close


def test_eod_flat_exits_last_bar_before_cutoff():
    # bars at 23:10..23:40; eod 2330 -> exit at close of the 23:25 bar
    df = mk_df([(100, 101, 99, 100)] * 7, start="2025-01-06 23:10")
    trades, _ = run_one(df, signal_idx=0, direction=1, entry_type=ENTRY_MARKET,
                        sl=90.0, eod_hhmm=2330)
    assert trades["exit_reason"].iloc[0] == "EOD"
    assert trades["exit_time"].iloc[0] == pd.Timestamp("2025-01-06 23:25")


def test_be_move_applies_next_bar_not_same_bar():
    # bar1 entry@100; bar1 hits +1R (105) -> BE armed for bar2.
    # bar2 dips to 99.9 (below entry, above orig SL 95) -> BE stop at 100 fires.
    df = mk_df([(100, 100.5, 99.5, 100),
                (100, 105.5, 99.8, 105),
                (105, 105.5, 99.9, 100.2),
                (100, 101, 99, 100)])
    trades, _ = run_one(df, signal_idx=0, direction=1, entry_type=ENTRY_MARKET,
                        sl=95.0, be_at_r=1.0)
    assert trades["exit_reason"].iloc[0] == "SL"
    slip = COSTS.slippage_atr_mult * 2.0
    assert trades["exit"].iloc[0] == pytest.approx(100.0 - slip)


def test_trailing_ratchet_never_widens():
    # trail activates, price rises then falls; stop must hold the high ratchet
    df = mk_df([(100, 100.5, 99.5, 100),
                (100, 108, 99.9, 107),    # entry bar, mfe 8 > act 2 -> trail on
                (107, 110, 106.5, 109),   # ratchet to 110-4=106
                (109, 109.5, 105.9, 106),  # low 105.9 <= 106 -> SL
                (106, 107, 105, 106)])
    trades, _ = run_one(df, signal_idx=0, direction=1, entry_type=ENTRY_MARKET,
                        sl=98.0, trail_atr_mult=2.0, trail_act_r=1.0)
    assert trades["exit_reason"].iloc[0] == "SL"
    slip = COSTS.slippage_atr_mult * 2.0
    assert trades["exit"].iloc[0] == pytest.approx(106.0 - slip)


def test_position_exclusivity_one_trade():
    bars = [(100, 100.5, 99.5, 100)] * 8
    df = mk_df(bars)
    signals = make_signals([
        dict(signal_idx=0, direction=1, entry_type=ENTRY_MARKET, sl=90.0, max_bars=4),
        dict(signal_idx=2, direction=1, entry_type=ENTRY_MARKET, sl=90.0, max_bars=4),
    ])
    trades, stats = simulate(ctx_of(df), signals, COSTS)
    assert len(trades) == 1
    assert stats["skipped_busy"] == 1


def test_indicator_exit_at_close():
    df = mk_df([(100, 100.5, 99.5, 100)] * 5)
    flags = np.array([False, False, True, False, False])
    signals = make_signals([dict(signal_idx=0, direction=1,
                                 entry_type=ENTRY_MARKET, sl=90.0)])
    trades, _ = simulate(ctx_of(df), signals, COSTS, exit_flags_long=flags)
    assert trades["exit_reason"].iloc[0] == "IND"
    assert trades["bars_held"].iloc[0] == 1


def test_r_net_includes_all_costs():
    df = mk_df([(100, 100.5, 99.5, 100), (100, 100.2, 99.8, 100),
                (100, 106, 104.9, 105), (105, 105.5, 104, 105)])
    trades, _ = run_one(df, signal_idx=0, direction=1, entry_type=ENTRY_MARKET,
                        sl=95.0, tp=105.0)
    t = trades.iloc[0]
    expected_cost = (COSTS.spread_usd_oz + COSTS.slippage_atr_mult * 2.0
                     + COSTS.commission_usd_oz)  # 0 nights
    assert t["cost_per_oz"] == pytest.approx(expected_cost)
    assert t["r_net"] == pytest.approx(t["r_price"] - expected_cost / t["stop_dist"])
