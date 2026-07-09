"""
Cheap categorical context features for conditional pattern mining.

Every column is categorical with <= 5 levels (keeps the event x context grid
small and every cell's sample count large) and lookahead-safe: values at bar i
use only information available at bar-i close. HTF context goes through
mtf.align_htf (completed bars only); rolling percentiles are shift(1)-ed.
"""
import numpy as np
import pandas as pd

from .mtf import align_htf
from .strategies.base import anchored_vwap, prior_day_stats, session_mask

# 20 trading days of M5 bars — window for the ATR-regime percentile.
ATR_REGIME_WINDOW = 20 * 288

CONTEXT_COLUMNS = [
    "session", "dow", "atr_regime", "h1_trend", "pd_side",
    "pdh_dist", "pdl_dist", "vwap_side",
]


def _h1_ema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    return df


def context_frame(df: pd.DataFrame, m5: pd.DataFrame) -> pd.DataFrame:
    """Per-bar context labels for the entry frame `df` (a prepare_frame
    output). `m5` is the raw M5 frame used to build HTF context."""
    ts = df["timestamp"]
    out = pd.DataFrame(index=df.index)

    sess = np.full(len(df), "off", dtype=object)
    sess[session_mask(ts, "02:00", "10:00")] = "asia"
    sess[session_mask(ts, "10:00", "16:30")] = "london"
    sess[session_mask(ts, "16:30", "23:59")] = "ny"
    out["session"] = sess

    out["dow"] = ts.dt.dayofweek.astype(str)

    pct = df["atr"].rolling(ATR_REGIME_WINDOW, min_periods=288).rank(pct=True).shift(1)
    out["atr_regime"] = pd.cut(pct, [0, 1 / 3, 2 / 3, 1.0],
                               labels=["low", "mid", "high"]).astype(object)
    out.loc[pct.isna(), "atr_regime"] = "na"

    h1 = align_htf(df, m5, "H1", indicator_fn=_h1_ema)
    trend = np.where(h1["h1_close"] > h1["h1_ema50"], "up", "down")
    trend = np.where(h1["h1_close"].isna(), "na", trend)
    out["h1_trend"] = trend

    pdst = prior_day_stats(df)
    mid = (pdst["d_high"] + pdst["d_low"]) / 2.0
    out["pd_side"] = np.where(mid.isna(), "na",
                              np.where(df["close"] > mid, "above", "below"))

    for col, lvl in (("pdh_dist", pdst["d_high"]), ("pdl_dist", pdst["d_low"])):
        dist = (lvl - df["close"]).abs() / df["atr"]
        out[col] = np.where(dist.isna(), "na",
                            np.where(dist < 1.0, "near",
                                     np.where(dist < 3.0, "mid", "far")))

    vwap = anchored_vwap(df)
    out["vwap_side"] = np.where(vwap.isna(), "na",
                                np.where(df["close"] > vwap, "above", "below"))
    return out
