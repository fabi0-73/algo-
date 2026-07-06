"""
Intraday momentum: late-morning half-hour return predicts the 15:30-16:00 ET
half-hour (peer-reviewed GLD r5->r13; ~53% WR thin edge, coefficient ~7x
larger on high-volatility days, hence the vol gate).

Broker time (~NY+7): predictor is the M30 bar labeled 18:30 (11:30-12:00 ET,
close known 19:00). Signal fires on the bar labeled 22:00 (closes 22:30);
MARKET entry at next bar open = the 22:30 bar's open (15:30 ET). Direction =
sign(predictor close - open). max_bars=0 exits at the entry bar's CLOSE
(simulator TIMEOUT at k - fill_bar >= max_bars) = 23:00 broker = 16:00 ET.
Hard SL sl_atr*ATR. vol_gate: prior-day range > 20-day ADR (both prior-day
shifted, base helpers). Days with no 18:30 bar (holiday) or a doji predictor
are skipped by actual timestamp lookup, never bar offsets.

VERDICT 2026-07-06: KILLED on train (run 65fecd3b) — all combos -0.044..-0.081R. Peer-reviewed edge (~53% WR near-1:1) confirmed too thin for CFD costs.
"""
import numpy as np

from .base import (MTFContext, make_signals, minutes_of_day, prior_day_stats,
                   adr, ENTRY_MARKET)

NAME = "intraday_momentum"
TIMEFRAMES = ["M30"]
HTF_NEEDS = []
DEFAULTS = {
    "predictor_hhmm": 1830,  # M30 bar covering 18:30-19:00 broker
    "signal_hhmm": 2200,     # bar closing 22:30; entry at 22:30 open
    "sl_atr": 1.5,
    "vol_gate": True,
    "adr_days": 20,
}
PARAM_GRID = {
    "sl_atr": [1.0, 1.5],
    "vol_gate": [False, True],
}


def _to_minutes(hhmm):
    return (int(hhmm) // 100) * 60 + int(hhmm) % 100


def generate_signals(ctx: MTFContext, params: dict):
    df = ctx.df
    ts = df["timestamp"]
    mod = minutes_of_day(ts)
    day = ts.dt.normalize()
    close = df["close"]
    atr = df["atr"]

    p_min = _to_minutes(params["predictor_hhmm"])
    s_min = _to_minutes(params["signal_hhmm"])

    # Predictor sign keyed by broker calendar day. The 18:30 bar closes at
    # 19:00, hours before the 22:00 signal bar closes, and both share one
    # broker day (trading day runs ~01:00-24:00), so the predictor's
    # positional index is always < signal_idx. Missing key = holiday -> skip.
    pred_sign = {}
    for i in np.flatnonzero((mod == p_min).to_numpy()):
        pred_sign[day.iloc[i]] = np.sign(close.iloc[i] - df["open"].iloc[i])

    if params["vol_gate"]:
        pstats = prior_day_stats(df)
        a = adr(df, int(params["adr_days"]))
        gate = ((pstats["d_range"] > a) & a.notna()).to_numpy()
    else:
        gate = np.ones(len(df), dtype=bool)

    sig_idx = np.flatnonzero(((mod == s_min) & atr.notna()).to_numpy())
    rows = []
    for i in sig_idx:
        d = pred_sign.get(day.iloc[i], 0.0)
        if d == 0 or not gate[i]:
            continue
        direction = 1 if d > 0 else -1
        sl = float(close.iloc[i] - direction * params["sl_atr"] * atr.iloc[i])
        rows.append({"signal_idx": int(i), "direction": direction,
                     "entry_type": ENTRY_MARKET, "sl": sl,
                     "max_bars": 0})
    return make_signals(rows)
