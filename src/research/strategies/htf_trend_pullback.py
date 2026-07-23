"""
HTF trend-pullback continuation stream (independent second-stream candidate,
2026-07-23). NOT an AMD variant: continuation at M30/H1, multi-hour holds.

VERDICT 2026-07-23: TRAIN PASS at pre-registered DEFAULTS, no grid run
(M30 215tr/+0.205R/PF 1.34; H1 105tr/+0.263R/PF 1.44; boundary 2025-12-30).
Spread-stress $0.45/oz: pass (0.196/0.257R). SINGLE OOS LOOK SPENT
(lab_d8e63ab5): M30 92tr/+0.121R/PF 1.20 (59% retention); H1 43tr/+0.208R/
PF 1.34 (79%). R-STREAM VALIDATED — first new lab keeper since ny_ib.
CAPITAL GEOMETRY CAVEAT: 2.93-ATR stops at the 0.01 min-lot floor =
$12-25 realized risk/trade; $500 equity replay shows 68%/94% path DD —
viable as a signal stream or at >=$1.5-2k capital, NOT co-executed on a
small shared account. Live integration requires the scanner's M30
resampling gap (design doc 5b.2) + lab_portfolio overlap check first.

Pre-registered provenance: verbatim transfer of the neighbor lab's
h1_ema_pullback — the ONLY strategy in their 14-slot roster to pass 4/4
walk-forward folds on 2022-2025 (783 trades, PF 1.21, DD $280 at 0.01 lots)
— with their WFO-tuned live parameters as DEFAULTS (htf 25/93 EMA bias,
pullback EMA 41, sl 2.93 ATR, tp 7.34R, trail 3.43 ATR, max_bars 185).
Academic support for the slow-gate grid axis: long-lookback trend filters
are the only gold-timing family surviving Hansen-SPA data-snooping tests
(1990-2015); their value is avoiding bear regimes.

Rules: H1 bias = EMA(fast) vs EMA(slow) on completed H1 bars. Entry on the
entry frame (M30/H1): prior bar pulls back to/through EMA(pullback) and the
current bar CLOSES back on the trend side -> MARKET next bar, trend
direction only. SL = sl_atr * ATR from close; TP = rr * risk; ATR trail
activates at trail_act_r; BE at be_at_r; unconditional timeout max_bars.
Optional D1 slow gate (grid): longs only above the D1 SMA(100), shorts
only below.
"""
import numpy as np
import pandas as pd

from .base import MTFContext, make_signals, ENTRY_MARKET

NAME = "htf_trend_pullback"
TIMEFRAMES = ["M30", "H1"]
HTF_NEEDS = ["H1", "D1"]


def _h1_emas(h1: pd.DataFrame) -> pd.DataFrame:
    h1 = h1.copy()
    h1["emaf"] = h1["close"].ewm(span=25, adjust=False).mean()
    h1["emas"] = h1["close"].ewm(span=93, adjust=False).mean()
    return h1


def _d1_sma(d1: pd.DataFrame) -> pd.DataFrame:
    d1 = d1.copy()
    d1["sma_slow"] = d1["close"].rolling(100).mean()
    return d1


HTF_INDICATORS = {"H1": _h1_emas, "D1": _d1_sma}

DEFAULTS = {
    "pullback_ema": 41,
    "sl_atr": 2.93,
    "rr": 7.34,
    "trail_atr_mult": 3.43,
    "trail_act_r": 1.72,
    "be_at_r": 1.72,
    "max_bars": 185,
    "use_d1_gate": False,
    "long_only": False,
}
PARAM_GRID = {
    "use_d1_gate": [False, True],
    "long_only": [False, True],
    "rr": [3.0, 7.34],
}


def generate_signals(ctx: MTFContext, params: dict):
    df = ctx.df
    h1 = ctx.htf.get("H1")
    d1 = ctx.htf.get("D1")
    if h1 is None or "h1_emaf" not in h1.columns:
        return make_signals([])

    close = df["close"].to_numpy(float)
    low = df["low"].to_numpy(float)
    high = df["high"].to_numpy(float)
    atr = df["atr"].to_numpy(float)
    ema_pb = df["close"].ewm(span=int(params["pullback_ema"]),
                             adjust=False).mean().to_numpy(float)

    bias = np.sign(h1["h1_emaf"].to_numpy(float) - h1["h1_emas"].to_numpy(float))
    if params["use_d1_gate"] and d1 is not None and "d1_sma_slow" in d1.columns:
        sma = d1["d1_sma_slow"].to_numpy(float)
        d1_ok_long = close > sma
        d1_ok_short = close < sma
    else:
        d1_ok_long = np.ones(len(df), bool)
        d1_ok_short = np.ones(len(df), bool)

    dipped = np.roll(low <= ema_pb, 1)
    spiked = np.roll(high >= ema_pb, 1)
    dipped[0] = spiked[0] = False
    warm = np.arange(len(df)) >= 93  # let EMAs mature on the entry frame too

    long_f = (bias > 0) & dipped & (close > ema_pb) & d1_ok_long & warm
    short_f = (bias < 0) & spiked & (close < ema_pb) & d1_ok_short & warm
    if params["long_only"]:
        short_f[:] = False

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
