"""No-lookahead HTF alignment, split parity, sizing floor, session masks."""
import numpy as np
import pandas as pd
import pytest

from src.research.mtf import align_htf, prepare_frame, split_boundary
from src.research.sizing import replay_equity
from src.research.strategies.base import session_mask


def m5_frame(start, periods):
    ts = pd.date_range(start, periods=periods, freq="5min")
    df = pd.DataFrame({
        "timestamp": ts,
        "open": np.arange(periods, dtype=float) + 100,
        "high": np.arange(periods, dtype=float) + 101,
        "low": np.arange(periods, dtype=float) + 99,
        "close": np.arange(periods, dtype=float) + 100.5,
        "volume": 100,
    })
    return df


def test_htf_alignment_no_lookahead():
    m5 = m5_frame("2025-01-06 09:00", 40)  # 09:00 .. 12:15
    entry = prepare_frame(m5, "M5")
    h1 = align_htf(entry, m5, "H1")

    # entry bar 10:05 must see the H1 bar labeled 09:00 (closed 10:00)
    i = entry.index[entry["timestamp"] == "2025-01-06 10:05"][0]
    assert h1.loc[i, "h1_timestamp"] == pd.Timestamp("2025-01-06 09:00")
    # entry bar exactly 10:00 also sees the 09:00 bar (it just closed)
    i = entry.index[entry["timestamp"] == "2025-01-06 10:00"][0]
    assert h1.loc[i, "h1_timestamp"] == pd.Timestamp("2025-01-06 09:00")
    # entry bar 09:55 must see NOTHING (09:00 bar still forming)
    i = entry.index[entry["timestamp"] == "2025-01-06 09:55"][0]
    assert pd.isna(h1.loc[i, "h1_timestamp"])
    # entry bar 11:20 sees the 10:00 bar, never the forming 11:00 bar
    i = entry.index[entry["timestamp"] == "2025-01-06 11:20"][0]
    assert h1.loc[i, "h1_timestamp"] == pd.Timestamp("2025-01-06 10:00")


def test_htf_alignment_weekend_gap():
    fri = m5_frame("2025-01-03 20:00", 24)   # Friday 20:00-21:55
    mon = m5_frame("2025-01-06 01:00", 24)   # Monday 01:00-02:55
    m5 = pd.concat([fri, mon], ignore_index=True)
    entry = prepare_frame(m5, "M5")
    h1 = align_htf(entry, m5, "H1")
    i = entry.index[entry["timestamp"] == "2025-01-06 01:00"][0]
    # Monday's first bar sees Friday's last COMPLETED H1 bar (21:00, closed 22:00)
    assert h1.loc[i, "h1_timestamp"] == pd.Timestamp("2025-01-03 21:00")


def test_htf_indicator_fn_applied():
    m5 = m5_frame("2025-01-06 09:00", 60)
    entry = prepare_frame(m5, "M5")

    def add_flag(df):
        df = df.copy()
        df["flag"] = df["close"] * 2
        return df

    h1 = align_htf(entry, m5, "H1", add_flag)
    assert "h1_flag" in h1.columns


def test_split_boundary_parity():
    m5 = m5_frame("2025-01-06 09:00", 100)
    assert split_boundary(m5, 0.70) == m5.iloc[70]["timestamp"]


def test_sizing_floor_engages():
    trades = pd.DataFrame({
        "entry_time": [pd.Timestamp("2025-01-06 12:00")],
        "direction": ["LONG"],
        "entry": [2000.0], "exit": [2010.0],
        "stop_dist": [5.0], "cost_per_oz": [0.4], "mae_r": [0.2],
    })
    # $500 * 0.5% = $2.5 risk; $5 stop -> raw 0.005 lots -> floored, min-lot 0.01
    res = replay_equity(trades, initial_capital=500.0, risk_pct=0.005)
    lots_implied = (res["final_equity"] - 500.0) / ((2010 - 2000) * 100 - 0.4 * 100)
    assert lots_implied == pytest.approx(0.01)


def test_sizing_floors_not_rounds():
    trades = pd.DataFrame({
        "entry_time": [pd.Timestamp("2025-01-06 12:00")],
        "direction": ["LONG"],
        "entry": [2000.0], "exit": [2010.0],
        "stop_dist": [5.0], "cost_per_oz": [0.0], "mae_r": [0.0],
    })
    # equity 1990 * 0.5% = 9.95 -> 9.95/500 = 0.0199 lots -> FLOOR to 0.01 (not 0.02)
    res = replay_equity(trades, initial_capital=1990.0, risk_pct=0.005)
    assert res["final_equity"] == pytest.approx(1990.0 + 10.0 * 100 * 0.01)


def test_session_mask_normal_and_wrap():
    ts = pd.Series(pd.date_range("2025-01-06 00:00", periods=288, freq="5min"))
    day = session_mask(ts, "10:00", "18:00")
    assert day[ts.dt.strftime("%H:%M") == "10:00"].all()
    assert not day[ts.dt.strftime("%H:%M") == "18:00"].any()
    wrap = session_mask(ts, "22:00", "02:00")
    assert wrap[ts.dt.strftime("%H:%M") == "23:00"].all()
    assert wrap[ts.dt.strftime("%H:%M") == "01:00"].all()
    assert not wrap[ts.dt.strftime("%H:%M") == "12:00"].any()
