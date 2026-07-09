"""Planted data defects must be flagged; clean data must audit clean."""
import numpy as np
import pandas as pd

from src.research.audit import audit_m5, format_report


def clean_week():
    """Mon-Fri broker week: bars 01:00-23:55 each day (settlement break 0-1h)."""
    frames = []
    for d in range(6, 11):  # 2025-01-06 (Mon) .. 2025-01-10 (Fri)
        ts = pd.date_range(f"2025-01-{d:02d} 01:00", f"2025-01-{d:02d} 23:55",
                           freq="5min")
        frames.append(pd.DataFrame({"timestamp": ts}))
    df = pd.concat(frames, ignore_index=True)
    n = len(df)
    rng = np.random.default_rng(1)
    close = 100 + rng.normal(0, 0.1, n).cumsum()
    df["open"] = np.roll(close, 1)
    df.loc[0, "open"] = 100.0
    df["high"] = np.maximum(df["open"], close) + 0.05
    df["low"] = np.minimum(df["open"], close) - 0.05
    df["close"] = close
    df["volume"] = 100
    return df


def test_clean_frame_audits_clean():
    a = audit_m5(clean_week())
    assert a["duplicate_timestamps"] == 0
    assert a["non_monotonic"] == 0
    assert a["off_grid"] == 0
    assert a["ohlc_violations"] == 0
    assert a["gaps"]["anomalous"] == 0
    assert a["gaps"]["settlement"] == 4  # four overnight breaks Mon..Fri
    assert a["bars_per_day"]["n_days"] == 5


def test_duplicate_and_off_grid_flagged():
    df = clean_week()
    df = pd.concat([df, df.iloc[[10]]], ignore_index=True).sort_values(
        "timestamp").reset_index(drop=True)
    df.loc[5, "timestamp"] += pd.Timedelta(minutes=2)
    a = audit_m5(df)
    assert a["duplicate_timestamps"] == 1
    assert a["off_grid"] == 1


def test_ohlc_corruption_flagged():
    df = clean_week()
    df.loc[20, "high"] = df.loc[20, "low"] - 1.0  # high < low
    df.loc[30, "low"] = df.loc[30, "close"] + 0.5  # low above close
    a = audit_m5(df)
    assert a["ohlc_violations"] == 2


def test_anomalous_gap_vs_weekend():
    df = clean_week()
    # remove 3 hours mid-Wednesday -> anomalous intra-session hole
    hole = (df["timestamp"] >= "2025-01-08 12:00") & (df["timestamp"] < "2025-01-08 15:00")
    df = df[~hole].reset_index(drop=True)
    # append Monday of next week -> weekend gap
    nxt = clean_week()
    nxt["timestamp"] += pd.Timedelta(days=7)
    df = pd.concat([df, nxt.iloc[:50]], ignore_index=True)
    a = audit_m5(df)
    assert a["gaps"]["anomalous"] == 1
    assert a["gaps"]["weekend"] == 1
    top = a["gaps"]["top_anomalous"][0]
    assert top["minutes"] == 185.0  # 11:55 -> 15:00


def test_return_outlier_flagged():
    df = clean_week()
    df.loc[100, "close"] = df.loc[100, "close"] * 1.10  # 10% M5 jump
    df.loc[100, "high"] = max(df.loc[100, "high"], df.loc[100, "close"])
    a = audit_m5(df)
    assert a["return_outliers"]["count"] >= 1


def test_format_report_runs():
    txt = format_report(audit_m5(clean_week()))
    assert "M5 CACHE AUDIT" in txt and "gaps:" in txt
    assert format_report(audit_m5(pd.DataFrame(columns=["timestamp"]))) \
        .endswith("EMPTY FRAME")
