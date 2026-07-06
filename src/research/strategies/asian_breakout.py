"""
Asian-range breakout with load-bearing quality filters (graded research:
the NAIVE Asian breakout tested NEGATIVE on gold — the filters are the edge
candidate, not the breakout itself).

Asian range = high/low of the broker 02:00-09:00 window, usable only after
the last window bar has CLOSED. Range must be "respected" (top and bottom
each touched >= touch_count times) and neither a vertical trend night
(range > max_range_adr * ADR20) nor dead (range < min_range_adr * ADR20).

REDESIGN 2026-07-06 (one pre-registered shot, train-only): the original
gate scaled the 7-hour session range by a single-BAR ATR (max_range_atr=4.0),
which rejected 93% of days (funnel: too_wide 356/382) — a unit mismatch,
not a market verdict. Gate now scales by ADR(20), the intended unit.

Entry: first candle CLOSE beyond the range inside the entry window
(default broker 10:00-19:00) -> MARKET next bar. One breakout attempt per
day per direction (the first confirming close consumes the attempt even
when a guard rejects it — no chasing later closes). Guards: skip if the
confirming close is already > chase_max_atr * ATR beyond the range edge,
or if the day's high-low travel so far exceeds adr_frac * ADR(20).

SL: sl_atr * ATR beyond the breakout candle's entry-side extreme
(long: candle low, short: candle high). BE at 1R, TP at rr * risk
(risk proxied from signal-bar close), force-flat 23:30 broker.

VERDICT 2026-07-06: KILLED on train (runs 65fecd3b + ADR-gate redesign) — with the corrected ADR-scaled gate delivering 49-101 trades, ALL 16 combos negative (best -0.012R). Confirms published negative ORB-on-gold result; earlier positive rows were n<=17 noise.
"""
import numpy as np

from .base import MTFContext, make_signals, session_mask, adr, ENTRY_MARKET

NAME = "asian_breakout"
TIMEFRAMES = ["M15", "M30"]
HTF_NEEDS = []
DEFAULTS = {
    "asian_start": "02:00",
    "asian_end": "09:00",
    "entry_start": "10:00",   # research grid also suggests 09:00; kept fixed
    "entry_end": "19:00",     # to hold PARAM_GRID at <= 8 combos
    "touch_count": 2,         # min touches of BOTH range edges
    "touch_tol_atr": 0.1,     # touch = within this * ATR of the edge
    "min_range_adr": 0.15,    # dead night guard (fraction of ADR20)
    "max_range_adr": 0.60,    # vertical trend night guard (fraction of ADR20)
    "chase_max_atr": 1.5,     # late-chase guard at confirmation close
    "adr_frac": 0.5,          # day travel so far vs ADR(20)
    "adr_days": 20,
    "min_asian_frac": 0.6,    # require this share of expected window bars
    "sl_atr": 1.5,
    "rr": 2.0,
    "be_at_r": 1.0,
    "eod_hhmm": 2330,
    "long_only": False,
}
PARAM_GRID = {
    "touch_count": [1, 2],
    "rr": [2.0, 3.0],
    "long_only": [False, True],
}


def generate_signals(ctx: MTFContext, params: dict):
    df = ctx.df
    ts = df["timestamp"]
    o_hi = df["high"].to_numpy(float)
    o_lo = df["low"].to_numpy(float)
    o_cl = df["close"].to_numpy(float)
    atr = df["atr"].to_numpy(float)
    adr20 = adr(df, int(params["adr_days"])).to_numpy(float)

    day = ts.dt.normalize()
    asian = session_mask(ts, params["asian_start"], params["asian_end"]).to_numpy()
    entry_win = session_mask(ts, params["entry_start"], params["entry_end"]).to_numpy()

    # day-cumulative travel: bar i sees only bars <= i of its own day
    day_hi = df["high"].groupby(day.values).cummax().to_numpy(float)
    day_lo = df["low"].groupby(day.values).cummin().to_numpy(float)

    # expected Asian bar count from the window length and the entry TF
    bar_min = int(ts.diff().dt.total_seconds().div(60).median())
    win_min = 7 * 60
    min_bars = max(4, int(params["min_asian_frac"] * win_min / max(bar_min, 1)))

    long_only = bool(params["long_only"])
    rows = []
    for _, pos in df.groupby(day.values).indices.items():
        pos = np.sort(np.asarray(pos))
        a_pos = pos[asian[pos]]
        if len(a_pos) < min_bars:
            continue
        i_end = int(a_pos.max())          # last Asian bar; closed before any entry bar
        a_end = atr[i_end]
        d_end = adr20[i_end]
        if not np.isfinite(a_end) or a_end <= 0 or not np.isfinite(d_end) or d_end <= 0:
            continue
        rng_hi = float(o_hi[a_pos].max())
        rng_lo = float(o_lo[a_pos].min())
        rng = rng_hi - rng_lo
        if rng > params["max_range_adr"] * d_end or rng < params["min_range_adr"] * d_end:
            continue
        tol = params["touch_tol_atr"] * a_end
        if (o_hi[a_pos] >= rng_hi - tol).sum() < int(params["touch_count"]):
            continue
        if (o_lo[a_pos] <= rng_lo + tol).sum() < int(params["touch_count"]):
            continue

        e_pos = pos[entry_win[pos] & (pos > i_end)]
        long_done, short_done = False, long_only
        for i in e_pos:
            i = int(i)
            if long_done and short_done:
                break
            if not np.isfinite(atr[i]) or not np.isfinite(adr20[i]):
                continue
            if not long_done and o_cl[i] > rng_hi:
                long_done = True  # first confirming close consumes the attempt
                if (o_cl[i] - rng_hi <= params["chase_max_atr"] * atr[i]
                        and day_hi[i] - day_lo[i] <= params["adr_frac"] * adr20[i]):
                    sl = o_lo[i] - params["sl_atr"] * atr[i]
                    risk = o_cl[i] - sl
                    rows.append({"signal_idx": i, "direction": 1,
                                 "entry_type": ENTRY_MARKET, "sl": float(sl),
                                 "tp": float(o_cl[i] + params["rr"] * risk),
                                 "be_at_r": float(params["be_at_r"]),
                                 "eod_hhmm": int(params["eod_hhmm"])})
            if not short_done and o_cl[i] < rng_lo:
                short_done = True
                if (rng_lo - o_cl[i] <= params["chase_max_atr"] * atr[i]
                        and day_hi[i] - day_lo[i] <= params["adr_frac"] * adr20[i]):
                    sl = o_hi[i] + params["sl_atr"] * atr[i]
                    risk = sl - o_cl[i]
                    rows.append({"signal_idx": i, "direction": -1,
                                 "entry_type": ENTRY_MARKET, "sl": float(sl),
                                 "tp": float(o_cl[i] - params["rr"] * risk),
                                 "be_at_r": float(params["be_at_r"]),
                                 "eod_hhmm": int(params["eod_hhmm"])})

    return make_signals(rows)
