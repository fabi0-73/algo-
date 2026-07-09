"""
Forward-return measurement for event studies.

Reference price is open[i+1]: an event completing at bar-i close is actionable
at the next bar's open — the same convention as lab.py's MARKET fill. Returns
are in ATR-at-event units, so under a fixed 1-ATR-stop convention a forward
return IS an R multiple and net_r() just subtracts the round-trip cost.

Cost per round trip (from lab.py CostModel, per-oz): spread crossed once +
commission + entry slippage (slippage_atr_mult x ATR). Horizon exits are
closes, not stop-outs, so no SL-side slippage is charged — matches lab.py's
market-out path and keeps the floor conservative but honest.
"""
import numpy as np
import pandas as pd

from .events import HORIZONS
from .lab import CostModel


def forward_outcomes(df: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Per bar i and horizon h (long convention, sign with event direction):
        fr_h  = (close[i+h] - open[i+1]) / atr[i]
        mfe_h = (max(high[i+1..i+h]) - open[i+1]) / atr[i]
        mae_h = (min(low[i+1..i+h]) - open[i+1]) / atr[i]
    Tail bars without a full horizon are NaN. Pure shift/rolling — O(n)."""
    ref = df["open"].shift(-1)
    atr = df["atr"]
    out = pd.DataFrame(index=df.index)
    for h in horizons:
        # rolling window of h bars ending at i+h == shift(-h) of a right-
        # aligned rolling over [i+1 .. i+h]
        hi = df["high"].rolling(h).max().shift(-h)
        lo = df["low"].rolling(h).min().shift(-h)
        out[f"fr_{h}"] = (df["close"].shift(-h) - ref) / atr
        out[f"mfe_{h}"] = (hi - ref) / atr
        out[f"mae_{h}"] = (lo - ref) / atr
    return out


def cost_in_atr(df: pd.DataFrame, cost: CostModel = None) -> pd.Series:
    """Round-trip cost per bar in ATR units: (spread + commission)/ATR +
    entry slippage fraction. This is the floor any mined edge must clear."""
    cost = cost or CostModel.from_config()
    fixed = cost.spread_usd_oz + cost.commission_usd_oz
    return fixed / df["atr"] + cost.slippage_atr_mult


def net_r(fr: pd.Series, cost_atr: pd.Series) -> pd.Series:
    """Net R under the fixed 1-ATR-stop convention: signed forward return in
    ATR units minus the round-trip cost in ATR units."""
    return fr - cost_atr


def tod_bucket(ts: pd.Series, bucket_minutes: int = 120) -> pd.Series:
    """Time-of-day bucket id (broker time) — the stratification key for
    baselines and permutation nulls. 120-min buckets -> 12 per day."""
    mod = ts.dt.hour * 60 + ts.dt.minute
    return (mod // bucket_minutes).astype(int)


def day_ids(ts: pd.Series) -> pd.Series:
    """Trading-day integer codes for day-block bootstrapping."""
    return ts.dt.normalize().astype("int64")


def baseline_pool(fwd: pd.DataFrame, buckets: pd.Series, horizon: int) -> dict:
    """Unconditional forward returns (all bars) with their TOD buckets —
    the pool the permutation null draws from. NaN tails dropped."""
    fr = fwd[f"fr_{horizon}"]
    ok = fr.notna()
    return {
        "values": fr[ok].to_numpy(float),
        "buckets": buckets[ok].to_numpy(),
    }
