"""
Strategy plug-in contract for the research lab.

Each strategy module exposes:
    NAME: str
    TIMEFRAMES: list[str]            # entry TFs it makes sense on
    HTF_NEEDS: list[str]             # e.g. ["H1", "D1"]; lab builds ctx.htf
    HTF_INDICATORS: dict[str, callable] (optional)  # tf -> fn(df)->df adding columns
    DEFAULTS: dict                   # screening params
    PARAM_GRID: dict[str, list]      # small tuning grid (survivors only)
    generate_signals(ctx, params) -> signals DataFrame
        or -> (signals, extras) where extras may contain
        {"exit_flags_long": bool array, "exit_flags_short": bool array}
        (per-bar indicator exits, evaluated at bar close).

NO LOOKAHEAD: a signal row with signal_idx=i may use only data with
positional index <= i on ctx.df / ctx.htf. Orders go live at bar i+1.

Sessions are BROKER time (~NY+7, DST-locked to NY): NY open 09:30 ET = 16:30,
NY close 17:00 ET = 00:00 (daily gap at hour 0; trading day restarts ~01:00),
London open ~10:00 (drifts +-1h vs UK DST), London/NY overlap ~15:30-19:00.
"""
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

ENTRY_MARKET = 0
ENTRY_LIMIT = 1
ENTRY_STOP = 2

SIGNAL_DEFAULTS = {
    "entry_price": np.nan,   # required for LIMIT/STOP
    "tp": np.nan,            # NaN -> trail/indicator/timeout managed
    "trail_atr_mult": 0.0,   # 0 = no trailing
    "trail_act_r": 0.0,      # R at which trailing activates
    "be_at_r": 0.0,          # move SL to entry at this R (0 = never)
    "ttl_bars": 1,           # pending-order expiry (bars after signal)
    "max_bars": 100000,      # timeout exit at close
    "eod_hhmm": 0,           # broker-time force-flat, e.g. 2330 (0 = none)
    "tag": 0,                # sub-setup diagnostics
}

REQUIRED_COLS = ["signal_idx", "direction", "entry_type", "sl"]


@dataclass
class MTFContext:
    tf: str
    df: pd.DataFrame                       # entry-TF bars + atr, positional index
    htf: Dict[str, pd.DataFrame] = field(default_factory=dict)  # aligned row-for-row


def make_signals(rows: list) -> pd.DataFrame:
    """Build a signal frame from dicts, filling schema defaults."""
    if not rows:
        cols = REQUIRED_COLS + list(SIGNAL_DEFAULTS.keys())
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    for col, default in SIGNAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default) if col != "entry_price" and col != "tp" else df[col]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"signal frame missing columns: {missing}")
    return df.sort_values("signal_idx").reset_index(drop=True)


def minutes_of_day(ts: pd.Series) -> pd.Series:
    return ts.dt.hour * 60 + ts.dt.minute


def session_mask(ts: pd.Series, start_hhmm: str, end_hhmm: str) -> pd.Series:
    """Broker-time-of-day window [start, end); supports overnight wrap."""
    sh, sm = (int(x) for x in start_hhmm.split(":"))
    eh, em = (int(x) for x in end_hhmm.split(":"))
    start_min, end_min = sh * 60 + sm, eh * 60 + em
    mod = minutes_of_day(ts)
    if start_min <= end_min:
        return (mod >= start_min) & (mod < end_min)
    return (mod >= start_min) | (mod < end_min)


def rsi(series: pd.Series, period: int) -> pd.Series:
    """Wilder-smoothed RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss  # div-by-zero -> inf -> RSI 100; 0/0 -> NaN -> 50
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(50.0)


def anchored_vwap(df: pd.DataFrame) -> pd.Series:
    """Daily-anchored VWAP from typical price x tick volume (broker calendar day).

    Tick volume is a liquidity proxy only — treat VWAP levels as soft zones.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    day = df["timestamp"].dt.normalize()
    pv = (tp * df["volume"]).groupby(day).cumsum()
    vv = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    return pv / vv


def prior_day_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bar prior-calendar-day high/low/range (no lookahead: prior day only)."""
    day = df["timestamp"].dt.normalize()
    daily = df.groupby(day).agg(d_high=("high", "max"), d_low=("low", "min"))
    daily["d_range"] = daily["d_high"] - daily["d_low"]
    prior = daily.shift(1)
    out = prior.reindex(day.values)
    out.index = df.index
    return out


def adr(df: pd.DataFrame, days: int = 20) -> pd.Series:
    """Rolling average daily range mapped to each bar, prior days only."""
    day = df["timestamp"].dt.normalize()
    daily_range = df.groupby(day).agg(h=("high", "max"), l=("low", "min"))
    r = (daily_range["h"] - daily_range["l"]).rolling(days, min_periods=5).mean().shift(1)
    out = r.reindex(day.values)
    out.index = df.index
    return out
