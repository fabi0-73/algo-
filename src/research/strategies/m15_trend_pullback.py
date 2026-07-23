"""
M15 trend-pullback with H4 confirmation — the LOW-CAPITAL variant of the
validated htf_trend_pullback stream (new hypothesis family, own OOS budget).

Pre-registered rationale, 2026-07-23: htf_trend_pullback's R-edge is
validated (train+OOS both TFs) but its 2.93-ATR M30/H1 stops cost $28
median per 0.01 lot — unaffordable at $500. Dropping the entry frame to
M15 shrinks ATR (and the dollar stop) roughly in half while keeping the
same continuation mechanism; the user's requested shape is explicitly
"1m/5m/15m entries with 4H confirmation". M1/M5 are EXCLUDED up front:
the cost floor (0.10-0.17 ATR/trade) ate every prior M5-entry family in
this repo, and static M1/M5 scalping is a falsified family in the
neighbor lab too. M15 is the shortest frame with a fighting chance.

VERDICT 2026-07-23: KILLED at pre-registered defaults (lab_3d311384):
689 trades / 43.6 per month — and expR +0.019, PF 1.03, path DD 81%.
The M15 frame produces exactly the requested frequency and the cost
floor consumes exactly all of it. No grid run (defaults failed), no OOS
spent. Boundary now measured three ways in this repo: M5 dead (many
families), M15 dead (this), M30 alive (+0.18R parent). The frequency/
cost sweet spot for trend-continuation on this feed is M30, full stop.

Rules (same skeleton as the parent, H4 bias instead of H1):
  H4 bias = EMA(fast) vs EMA(slow) on completed H4 bars (25/93 default —
  the parent's tuned smoothing, one frame up: slower, fewer regime flips).
  Entry on M15: prior bar pulls back to/through EMA(pullback=41), current
  bar CLOSES back on the trend side -> MARKET next bar, trend direction
  only. SL sl_atr * ATR(M15); TP rr * risk; trail/BE/timeout as parent.
  max_bars 185 M15 bars ~= 2 days.
"""
import numpy as np
import pandas as pd

from .base import MTFContext, make_signals, ENTRY_MARKET

NAME = "m15_trend_pullback"
TIMEFRAMES = ["M15"]
HTF_NEEDS = ["H4"]


def _h4_emas(h4: pd.DataFrame) -> pd.DataFrame:
    h4 = h4.copy()
    h4["emaf"] = h4["close"].ewm(span=25, adjust=False).mean()
    h4["emas"] = h4["close"].ewm(span=93, adjust=False).mean()
    return h4


HTF_INDICATORS = {"H4": _h4_emas}

DEFAULTS = {
    "pullback_ema": 41,
    "sl_atr": 2.0,          # tighter than parent (2.93): M15 wicks are smaller
    "rr": 6.0,
    "trail_atr_mult": 3.0,
    "trail_act_r": 1.5,
    "be_at_r": 1.5,
    "max_bars": 185,
}
PARAM_GRID = {
    "sl_atr": [1.5, 2.0, 2.93],
    "rr": [3.0, 6.0],
}


def generate_signals(ctx: MTFContext, params: dict):
    df = ctx.df
    h4 = ctx.htf.get("H4")
    if h4 is None or "h4_emaf" not in h4.columns:
        return make_signals([])

    close = df["close"].to_numpy(float)
    low = df["low"].to_numpy(float)
    high = df["high"].to_numpy(float)
    atr = df["atr"].to_numpy(float)
    ema_pb = df["close"].ewm(span=int(params["pullback_ema"]),
                             adjust=False).mean().to_numpy(float)
    bias = np.sign(h4["h4_emaf"].to_numpy(float) - h4["h4_emas"].to_numpy(float))

    dipped = np.roll(low <= ema_pb, 1)
    spiked = np.roll(high >= ema_pb, 1)
    dipped[0] = spiked[0] = False
    warm = np.arange(len(df)) >= 93

    long_f = (bias > 0) & dipped & (close > ema_pb) & warm
    short_f = (bias < 0) & spiked & (close < ema_pb) & warm

    rows = []
    for i in np.flatnonzero(long_f | short_f):
        i = int(i)
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        d = 1 if long_f[i] else -1
        sl = close[i] - d * params["sl_atr"] * a
        risk = abs(close[i] - sl)
        rows.append({"signal_idx": i, "direction": d,
                     "entry_type": ENTRY_MARKET, "sl": float(sl),
                     "tp": float(close[i] + d * params["rr"] * risk),
                     "trail_atr_mult": float(params["trail_atr_mult"]),
                     "trail_act_r": float(params["trail_act_r"]),
                     "be_at_r": float(params["be_at_r"]),
                     "max_bars": int(params["max_bars"])})
    return make_signals(rows)
