"""
Noise-area breakout (Zarattini/Barbon/Aziz "Beat the Market", gold-adapted).

For each intraday time slot t of the NY session (broker 16:30-23:00), the
noise sigma(t) is the mean of |close(t) - session open| over the prior
lookback_days sessions — per-slot rolling mean SHIFTED one session, so today
never sees its own aggregate. A close above session_open + band_mult*sigma(t)
is a breakout beyond noise -> market long next bar; mirror short. First
trigger per day per direction (max 1 long + 1 short attempt/day).

Hard SL at the opposite band at entry time; indicator exit when close crosses
back through the session-anchored VWAP (anchored 16:30); force-flat 23:00.
Days with fewer than lookback_days/2 sigma observations are skipped.

Evidence: grade B with replications on SPY/ES/NQ; untested on gold — this is
the falsification run. vol_gate optionally trades only days whose prior-day
range exceeded the 20-day ADR.

VERDICT 2026-07-06: KILLED on train (run 65fecd3b) — all combos -0.009..-0.081R, PF<=0.97. Regime dependence: worked on ES 2018+, not gold 2024-26.
"""
import numpy as np
import pandas as pd

from .base import (MTFContext, make_signals, minutes_of_day, session_mask,
                   prior_day_stats, adr, ENTRY_MARKET)

NAME = "noise_area"
TIMEFRAMES = ["M15"]
HTF_NEEDS = []
DEFAULTS = {
    "lookback_days": 14,
    "band_mult": 1.0,
    "vol_gate": False,
}
PARAM_GRID = {
    "band_mult": [1.0, 1.5],
    "vol_gate": [False, True],
}

SESSION_START = "16:30"   # NY 09:30 ET in broker time
SESSION_END = "23:00"     # NY 16:00 ET; force-flat here
LAST_SIGNAL = "22:45"     # signal bars end here so the next-bar fill stays in-session
EOD_HHMM = 2300


def _session_vwap(df, in_sess, day):
    """Cumulative VWAP anchored at the 16:30 session start, NaN off-session."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    grp = day[in_sess]
    pv = (tp[in_sess] * df["volume"][in_sess]).groupby(grp).cumsum()
    vv = df["volume"][in_sess].groupby(grp).cumsum().replace(0, np.nan)
    out = pd.Series(np.nan, index=df.index)
    out[in_sess] = pv / vv
    return out


def generate_signals(ctx: MTFContext, params: dict):
    df = ctx.df
    ts = df["timestamp"]
    close = df["close"]
    day = ts.dt.normalize()
    slot = minutes_of_day(ts)
    in_sess = session_mask(ts, SESSION_START, SESSION_END)
    sig_window = session_mask(ts, SESSION_START, LAST_SIGNAL)
    lookback = int(params["lookback_days"])
    min_obs = max(2, lookback // 2)

    # Today's session open: open of the first 16:30-23:00 bar of the day,
    # known at that bar's open, so usable at any in-session bar close.
    sess_open_by_day = df["open"][in_sess].groupby(day[in_sess]).first()
    sess_open = pd.Series(np.nan, index=df.index)
    sess_open[in_sess] = day[in_sess].map(sess_open_by_day).to_numpy()

    # Per-slot noise from COMPLETED prior sessions only: pivot day x slot,
    # rolling mean over lookback sessions, shifted one session down.
    dev = (close - sess_open).abs()
    piv = pd.DataFrame({"day": day[in_sess], "slot": slot[in_sess],
                        "dev": dev[in_sess]}).pivot(
        index="day", columns="slot", values="dev")
    sig_piv = piv.rolling(lookback, min_periods=min_obs).mean().shift(1)
    key = pd.MultiIndex.from_arrays([day[in_sess], slot[in_sess]])
    sigma = pd.Series(np.nan, index=df.index)
    sigma[in_sess] = sig_piv.stack().reindex(key).to_numpy()

    upper = sess_open + params["band_mult"] * sigma
    lower = sess_open - params["band_mult"] * sigma

    long_trig = sig_window & sigma.notna() & (close > upper)
    short_trig = sig_window & sigma.notna() & (close < lower)

    if params["vol_gate"]:
        prior = prior_day_stats(df)
        gate = (prior["d_range"] > adr(df, 20)).fillna(False)
        long_trig &= gate
        short_trig &= gate

    svwap = _session_vwap(df, in_sess, day)
    exit_flags_long = ((close < svwap) & in_sess).to_numpy()
    exit_flags_short = ((close > svwap) & in_sess).to_numpy()

    n = len(df)
    day_np = day.to_numpy()
    slot_np = slot.to_numpy()
    in_sess_np = in_sess.to_numpy()
    rows = []
    for trig, direction in ((long_trig, 1), (short_trig, -1)):
        seen = set()
        for i in np.flatnonzero(trig.to_numpy()):
            d = day_np[i]
            if d in seen:
                continue
            seen.add(d)  # first trigger per day per direction, used or not
            j = i + 1
            if j >= n or day_np[j] != d or not in_sess_np[j]:
                continue
            # SL at the opposite band at entry time (entry-bar slot; its
            # sigma comes from prior sessions, so it is known at bar i).
            band = lower.iloc[j] if direction > 0 else upper.iloc[j]
            if not np.isfinite(band):
                band = lower.iloc[i] if direction > 0 else upper.iloc[i]
            if not np.isfinite(band):
                continue
            rows.append({"signal_idx": int(i), "direction": direction,
                         "entry_type": ENTRY_MARKET, "sl": float(band),
                         "eod_hhmm": EOD_HHMM, "tag": int(slot_np[i])})

    return make_signals(rows), {"exit_flags_long": exit_flags_long,
                                "exit_flags_short": exit_flags_short}
