"""
Data layer for the research lab: load M5, resample, indicators,
and no-lookahead higher-timeframe alignment.

All timestamps are BROKER time (IC Markets ~= NY+7). Sessions must be
expressed as broker-time-of-day masks, never bar offsets (weekend and
daily-settlement gaps make bar counting unsafe).
"""
import logging
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from src.data.resample import resample_ohlcv, TIMEFRAME_MINUTES
from src.strategy.indicators import calculate_atr

logger = logging.getLogger(__name__)

# Daily bars for regime gates (not in the engine's resample map).
DAILY_MINUTES = 1440

_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "lab_m5_cache.csv"


def load_m5(start=None, end=None, use_cache: bool = True,
            cache_path=None) -> pd.DataFrame:
    """Load the full XAUUSD M5 frame, with a local CSV cache to skip the DB.

    cache_path: explicit cache file (.csv or .csv.gz) — lets scripts and
    tests point at fixtures/exports; default is data/lab_m5_cache.csv with a
    .csv.gz fallback."""
    df = None
    if cache_path is not None:
        df = pd.read_csv(cache_path, parse_dates=["timestamp"])
    elif use_cache:
        for path in (_CACHE_PATH, _CACHE_PATH.with_suffix(".csv.gz")):
            if path.exists():
                df = pd.read_csv(path, parse_dates=["timestamp"])
                break
    if df is None or df.empty:
        from src.data.db import Database
        db = Database()
        df = db.get_candles("XAUUSD", "M5", None, None)
        if df.empty:
            raise RuntimeError("No M5 candles in database")
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        if use_cache:
            _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(_CACHE_PATH, index=False)
    df = df.sort_values("timestamp").reset_index(drop=True)
    if start is not None:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["timestamp"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


def tf_minutes(tf: str) -> int:
    if tf == "D1":
        return DAILY_MINUTES
    return TIMEFRAME_MINUTES[tf]


def resample_any(m5: pd.DataFrame, tf: str) -> pd.DataFrame:
    """resample_ohlcv plus D1 support (1440-min buckets, broker midnight)."""
    if tf == "M5":
        return m5.copy()
    if tf == "D1":
        work = m5.set_index("timestamp")
        agg = {"open": "first", "high": "max", "low": "min",
               "close": "last", "volume": "sum"}
        out = work.resample("1440min", label="left", closed="left").agg(agg)
        out = out.dropna(subset=["open", "high", "low", "close"]).reset_index()
        return out
    return resample_ohlcv(m5, tf)


def prepare_frame(m5: pd.DataFrame, tf: str, atr_period: int = 14) -> pd.DataFrame:
    """Entry-TF frame with positional index and the engine's EMA-smoothed ATR."""
    df = resample_any(m5, tf)
    df = df.reset_index(drop=True)
    df["atr"] = calculate_atr(df, period=atr_period)
    return df


def align_htf(
    entry_df: pd.DataFrame,
    m5: pd.DataFrame,
    htf: str,
    indicator_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Row-for-row HTF context for each entry bar, lookahead-safe.

    An HTF bar labeled T (open time) covers [T, T+delta) and becomes
    knowable only at T+delta. merge_asof(backward) on available_at
    guarantees each entry bar sees only COMPLETED HTF bars; a bar opening
    exactly at T+delta does see the bar that just closed. Never ffill HTF
    closes and never shift positionally — weekend/settlement gaps break both.
    """
    h = resample_any(m5, htf)
    h = h.reset_index(drop=True)
    if indicator_fn is not None:
        h = indicator_fn(h)
    h = h.copy()
    h["available_at"] = h["timestamp"] + pd.Timedelta(minutes=tf_minutes(htf))
    prefix = htf.lower() + "_"
    renames = {c: prefix + c for c in h.columns if c != "available_at"}
    h = h.rename(columns=renames).sort_values("available_at")

    out = pd.merge_asof(
        entry_df[["timestamp"]].copy(),
        h,
        left_on="timestamp",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    out.index = entry_df.index
    return out.drop(columns=["available_at"])


def split_boundary(m5: pd.DataFrame, split: float = 0.70) -> pd.Timestamp:
    """Same convention as walk_forward.py: row-count split on the M5 frame."""
    return pd.Timestamp(m5.iloc[int(len(m5) * split)]["timestamp"])
