"""
Liquidity Levels — OHLCV-derived stop-cluster proxies.

Real "liquidation data" does not exist for spot gold (heatmaps are crypto-only;
OANDA's order-book API was discontinued Sep 2024), so liquidity points are
computed from price itself. Level types, ranked by evidence (Osler NY Fed
SR125/SR150 on stop clustering; equity limit-order clustering literature):

  - Equal highs/lows   — clustered swing points within a tolerance (canonical pools)
  - PDH/PDL, PWH/PWL   — previous day/week extremes (reuses key_levels columns)
  - Asian session H/L  — the overnight range gold sweeps at London/NY
  - Round numbers      — $25 increments (stops cluster just beyond round levels)

Sides: a level formed by HIGHS has buy-stops resting ABOVE it; a level formed
by LOWS has sell-stops BELOW. A sweep runs those stops, then (per the model)
price snaps back — the sweep_entry module fades that.

NOTE on time frame: timestamps are in the BROKER frame (IC Markets ≈ NY+7; see
TIME_CONFIG in config.py). The Asian session window comes from SESSION_FILTER
("asian_start" 23:00 → "asian_end" 08:00) which is already in that frame.
"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import SESSION_FILTER, SWEEP_MODEL


@dataclass
class LiquidityLevel:
    """A price level where resting stops likely cluster."""
    price: float
    kind: str   # "PDH", "PDL", "PWH", "PWL", "ASIAN_H", "ASIAN_L", "ROUND", "EQ_H", "EQ_L"
    side: str   # "ABOVE" = buy-stops above (sweep up -> fade SHORT); "BELOW" = sell-stops below


def add_asian_range(df: pd.DataFrame) -> pd.DataFrame:
    """Add asian_high / asian_low columns: the completed overnight range.

    Session spans midnight (default 23:00 -> 08:00 broker time). Candles at
    hour >= start belong to the NEXT calendar day's session. Rows still inside
    the forming session get NaN (an incomplete range is not a valid level).
    """
    df = df.copy()
    start_h = int(str(SESSION_FILTER.get("asian_start", "23:00")).split(":")[0])
    end_h = int(str(SESSION_FILTER.get("asian_end", "08:00")).split(":")[0])

    ts = pd.to_datetime(df["timestamp"])
    hour = ts.dt.hour
    in_asian = (hour >= start_h) | (hour < end_h)
    session_date = ts.dt.normalize() + pd.to_timedelta((hour >= start_h).astype(int), unit="D")

    asian = (
        df.loc[in_asian]
        .groupby(session_date[in_asian])
        .agg(asian_high=("high", "max"), asian_low=("low", "min"))
    )
    df["asian_high"] = session_date.map(asian["asian_high"])
    df["asian_low"] = session_date.map(asian["asian_low"])
    # Range only valid once the session has completed
    df.loc[in_asian, ["asian_high", "asian_low"]] = np.nan
    return df


def find_swing_points(
    highs: np.ndarray, lows: np.ndarray, strength: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """Indices of swing highs / swing lows (strictly higher/lower than
    `strength` neighbours on each side). Vectorized; O(n * strength)."""
    n = len(highs)
    if n < 2 * strength + 1:
        return np.array([], dtype=int), np.array([], dtype=int)
    is_sh = np.ones(n, dtype=bool)
    is_sl = np.ones(n, dtype=bool)
    is_sh[:strength] = is_sh[-strength:] = False
    is_sl[:strength] = is_sl[-strength:] = False
    for k in range(1, strength + 1):
        c = slice(strength, n - strength)
        is_sh[c] &= (highs[c] > highs[strength - k:n - strength - k]) & (
            highs[c] > highs[strength + k:n - strength + k])
        is_sl[c] &= (lows[c] < lows[strength - k:n - strength - k]) & (
            lows[c] < lows[strength + k:n - strength + k])
    return np.where(is_sh)[0], np.where(is_sl)[0]


def _cluster_levels(prices: np.ndarray, tolerance: float, min_touches: int) -> List[float]:
    """Group sorted prices into clusters within tolerance; return the mean of
    each cluster with >= min_touches members."""
    if len(prices) == 0:
        return []
    prices = np.sort(prices)
    levels, cluster = [], [prices[0]]
    for p in prices[1:]:
        if p - cluster[-1] <= tolerance:
            cluster.append(p)
        else:
            if len(cluster) >= min_touches:
                levels.append(float(np.mean(cluster)))
            cluster = [p]
    if len(cluster) >= min_touches:
        levels.append(float(np.mean(cluster)))
    return levels


def get_active_levels(
    current_idx: int,
    close: float,
    atr: float,
    row_levels: dict,
    swing_high_idx: np.ndarray = None,
    swing_low_idx: np.ndarray = None,
    highs: np.ndarray = None,
    lows: np.ndarray = None,
    cfg: dict = None,
) -> List[LiquidityLevel]:
    """Assemble the liquidity levels active at this bar, deduped within
    0.25*ATR (kinds concatenated with '+' so reports show confluence of levels).

    row_levels: dict of the per-row column values (prev_day_high, ..., asian_low).
    swing/high/low arrays: pass to include rolling equal-high/low clusters.
    """
    cfg = cfg or SWEEP_MODEL
    out: List[LiquidityLevel] = []

    def _add(price, kind, side):
        if price is not None and not pd.isna(price) and price > 0:
            out.append(LiquidityLevel(float(price), kind, side))

    if cfg.get("use_pdh_pdl", True):
        _add(row_levels.get("prev_day_high"), "PDH", "ABOVE")
        _add(row_levels.get("prev_day_low"), "PDL", "BELOW")
    if cfg.get("use_weekly", True):
        _add(row_levels.get("prev_week_high"), "PWH", "ABOVE")
        _add(row_levels.get("prev_week_low"), "PWL", "BELOW")
    if cfg.get("use_asian_range", True):
        _add(row_levels.get("asian_high"), "ASIAN_H", "ABOVE")
        _add(row_levels.get("asian_low"), "ASIAN_L", "BELOW")
    if cfg.get("use_round_numbers", True):
        step = float(cfg.get("round_step", 25.0))
        below = np.floor(close / step) * step
        _add(below, "ROUND", "BELOW")
        _add(below + step, "ROUND", "ABOVE")

    if (cfg.get("use_equal_levels", True) and swing_high_idx is not None
            and atr and atr > 0):
        lookback = int(cfg.get("level_lookback", 200))
        strength = int(cfg.get("swing_strength", 3))
        tol = float(cfg.get("equal_tolerance_atr_mult", 0.10)) * atr
        min_touch = int(cfg.get("equal_min_touches", 2))
        lo_i, hi_i = current_idx - lookback, current_idx - strength
        sh = swing_high_idx[(swing_high_idx >= lo_i) & (swing_high_idx <= hi_i)]
        sl = swing_low_idx[(swing_low_idx >= lo_i) & (swing_low_idx <= hi_i)]
        for lvl in _cluster_levels(highs[sh], tol, min_touch):
            _add(lvl, "EQ_H", "ABOVE")
        for lvl in _cluster_levels(lows[sl], tol, min_touch):
            _add(lvl, "EQ_L", "BELOW")

    # Dedupe levels that coincide (e.g. PDH == equal high): merge kinds
    if not out:
        return out
    dedupe_tol = 0.25 * atr if atr and atr > 0 else 0.0
    out.sort(key=lambda l: l.price)
    merged = [out[0]]
    for lvl in out[1:]:
        last = merged[-1]
        if lvl.side == last.side and abs(lvl.price - last.price) <= dedupe_tol:
            last.kind = f"{last.kind}+{lvl.kind}"
        else:
            merged.append(lvl)
    return merged
