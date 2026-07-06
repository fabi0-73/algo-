"""
Session-VWAP 2-sigma fade back to VWAP (re-entry confirmation).

Evidence is equities-grade C for intraday VWAP mean reversion; gold
trend days will produce losing streaks — the consecutive-close
standdown filter is the risk control, not the edge itself.

SHORT: at least one prior bar closed above vwap + band_mult*sigma and
this bar closes back below the band (still above VWAP) -> market entry
next open, TP = VWAP at signal time, SL at the (band_mult+1)-sigma
level. LONG mirror at the lower band. Skip when the last standdown_bars
consecutive bars all closed beyond the band (trend day). Entries only
broker 10:00-22:00, force-flat 23:30, max 3 trades/day per direction.
Optional trend_filter: longs only above SMA(200), shorts mirror.

VERDICT 2026-07-06: KILLED on train (run 65fecd3b) — all combos negative (best -0.011R); trend_filter variants n<=7. DD 37-67% unfiltered.
"""
import numpy as np

from .base import MTFContext, anchored_vwap, make_signals, session_mask, ENTRY_MARKET

NAME = "vwap_reversion"
TIMEFRAMES = ["M5", "M15"]
HTF_NEEDS = []
DEFAULTS = {
    "band_mult": 2.0,        # sigma multiple for the fade band
    "sigma_bars": 60,        # rolling std window of (close - vwap)
    "sigma_min_periods": 30,
    "standdown_bars": 6,     # skip if this many consecutive closes beyond band
    "trend_filter": False,   # longs only above SMA(sma_trend), shorts mirror
    "sma_trend": 200,
    "session_start": "10:00",
    "session_end": "22:00",
    "eod_hhmm": 2330,
    "max_per_day_side": 3,
}
PARAM_GRID = {
    "band_mult": [2.0, 2.5],
    "trend_filter": [False, True],
}


def generate_signals(ctx: MTFContext, params: dict):
    df = ctx.df
    close = df["close"]
    ts = df["timestamp"]
    vwap = anchored_vwap(df)
    dev = close - vwap
    sigma = dev.rolling(int(params["sigma_bars"]),
                        min_periods=int(params["sigma_min_periods"])).std()
    bm = float(params["band_mult"])
    upper = vwap + bm * sigma
    lower = vwap - bm * sigma

    # consecutive-close run length beyond each band, ending at each bar
    beyond_up = close > upper
    beyond_dn = close < lower
    run_up = beyond_up.groupby((~beyond_up).cumsum()).cumsum()
    run_dn = beyond_dn.groupby((~beyond_dn).cumsum()).cumsum()
    prev_run_up = run_up.shift(1).fillna(0)
    prev_run_dn = run_dn.shift(1).fillna(0)

    in_sess = session_mask(ts, params["session_start"], params["session_end"])
    valid = sigma.notna() & (sigma > 0) & vwap.notna() & df["atr"].notna()
    sd = int(params["standdown_bars"])

    # re-entry confirmation: prior bar(s) closed beyond, this bar back inside
    # (still on the fade side of VWAP so TP=VWAP is in the profit direction)
    short_sig = (~beyond_up) & (prev_run_up >= 1) & (prev_run_up < sd) & \
        (close > vwap) & in_sess & valid
    long_sig = (~beyond_dn) & (prev_run_dn >= 1) & (prev_run_dn < sd) & \
        (close < vwap) & in_sess & valid

    if params["trend_filter"]:
        sma_t = close.rolling(int(params["sma_trend"])).mean()
        long_sig &= close > sma_t
        short_sig &= close < sma_t

    day = ts.dt.normalize()
    eod = int(params["eod_hhmm"])
    cap = int(params["max_per_day_side"])

    cands = [(int(i), 1) for i in np.flatnonzero(long_sig.to_numpy())]
    cands += [(int(i), -1) for i in np.flatnonzero(short_sig.to_numpy())]
    cands.sort()

    rows = []
    counts = {}
    for i, direction in cands:
        key = (day.iloc[i], direction)
        c = counts.get(key, 0)
        if c >= cap:
            continue
        counts[key] = c + 1
        s = float(sigma.iloc[i])
        v = float(vwap.iloc[i])
        sl = v - (bm + 1.0) * s if direction > 0 else v + (bm + 1.0) * s
        rows.append({"signal_idx": i, "direction": direction,
                     "entry_type": ENTRY_MARKET, "sl": sl, "tp": v,
                     "eod_hhmm": eod})

    return make_signals(rows)
