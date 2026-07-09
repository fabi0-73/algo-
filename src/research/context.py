"""
Cheap categorical context features for conditional pattern mining.

Every column is categorical with <= 5 levels (keeps the event x context grid
small and every cell's sample count large) and lookahead-safe: values at bar i
use only information available at bar-i close. HTF context goes through
mtf.align_htf (completed bars only); rolling percentiles are shift(1)-ed.
"""
import numpy as np
import pandas as pd

from .mtf import align_htf, prepare_frame
from .smt import load_corr
from .strategies.base import anchored_vwap, prior_day_stats, session_mask

# 20 trading days of M5 bars — window for the ATR-regime percentile.
ATR_REGIME_WINDOW = 20 * 288

CONTEXT_COLUMNS = [
    "session", "dow", "atr_regime", "h1_trend", "pd_side",
    "pdh_dist", "pdl_dist", "vwap_side", "xag_mom", "eur_mom",
]

# Cross-asset momentum context: a correlated asset's trailing momentum as
# context for gold (xag = silver, eur = EURUSD). Silver's own M5 edge is real
# but untradeable at its cost floor (see memory/PR notes) — its value is as a
# leading context. Conservative one-bar lag: a bar stamped s is usable from
# s+5min (its close), so a gold bar at t only sees correlated bars stamped
# <= t-5min even though both close simultaneously.
XAG_MOM_BARS = 12          # trailing 1h on M5
XAG_MOM_THRESH_ATR = 0.5   # |momentum| >= 0.5 asset-ATR -> up/dn
XAG_MOM_TOLERANCE_MIN = 30  # stale feed (gap/halt) -> na, never carried


def _corr_mom(ts: pd.Series, corr: pd.DataFrame = None,
              cache_name: str = "xagusd") -> np.ndarray:
    """Bucket {up, flat, dn, na} of a correlated asset's last-1h return in
    its own ATR units at each gold bar. corr=None auto-loads
    data/lab_<cache_name>_cache.csv; missing/short/empty -> all 'na' (tests
    and fresh clones pass). Train-isolation note: values at gold bar t read
    only correlated bars closed before t, so mining on a truncated gold frame
    never sees post-boundary information even when the file spans further."""
    if corr is None:
        corr = load_corr(cache_name)
    if corr is None or len(corr) < XAG_MOM_BARS + 20:
        return np.full(len(ts), "na", dtype=object)
    c = prepare_frame(corr.copy(), "M5")
    mom = (c["close"] - c["close"].shift(XAG_MOM_BARS)) / c["atr"]
    right = pd.DataFrame({
        "available_at": c["timestamp"] + pd.Timedelta(minutes=5),
        "mom": mom.to_numpy(float),
    }).dropna().sort_values("available_at")
    if right.empty:
        return np.full(len(ts), "na", dtype=object)
    merged = pd.merge_asof(
        pd.DataFrame({"ts": ts.to_numpy()}), right,
        left_on="ts", right_on="available_at", direction="backward",
        tolerance=pd.Timedelta(minutes=XAG_MOM_TOLERANCE_MIN))
    m = merged["mom"]
    return np.where(m.isna(), "na",
                    np.where(m >= XAG_MOM_THRESH_ATR, "up",
                             np.where(m <= -XAG_MOM_THRESH_ATR, "dn",
                                      "flat")))


def _h1_ema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    return df


def context_frame(df: pd.DataFrame, m5: pd.DataFrame,
                  xag: pd.DataFrame = None,
                  eur: pd.DataFrame = None) -> pd.DataFrame:
    """Per-bar context labels for the entry frame `df` (a prepare_frame
    output). `m5` is the raw M5 frame used to build HTF context. `xag`/`eur`
    are optional raw correlated M5 frames (tests); default auto-loads the
    caches."""
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

    out["xag_mom"] = _corr_mom(df["timestamp"], xag, "xagusd")
    out["eur_mom"] = _corr_mom(df["timestamp"], eur, "eurusd")
    return out
