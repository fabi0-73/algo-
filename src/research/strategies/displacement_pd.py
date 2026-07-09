"""
Displacement continuation SHORT below the prior-day midpoint — the sole
survivor of the 2026-07-09 conditional mining run (mining_2cce45e7, 23-event x
24-context x 6-horizon grid, TRAIN < 2025-09-18 only): cell = displacement
dir=-1 h=3 context=pd_side=below, n=343 declustered, excess +0.31 ATR over the
TOD-matched in-context baseline, p_adj=0.013 (BH-FDR q=0.10 over 1,335 cells),
netR +0.094 after costs, sign-consistent across both train halves. Economic
read: downside momentum persists ~15 minutes when price is already below value.

Spec faithful to the mined cell: bearish displacement bar (|body| >= body_atr x
ATR, body >= body_frac of range, close < open) that closes below the prior-day
midpoint -> MARKET short at next open, exit at the close 3 bars after the
signal (max_bars = hold_bars - 1 because the fill bar is bar 1 of the window).
Mining had no stop (returns normalized by 1 ATR); the hard SL at sl_atr x ATR
makes it tradeable and strictly more conservative (WORST_CASE fills).

Promotion discipline: screen with the mining boundary PINNED
(--split 0.5634 on the 130,401-bar cache reproduces 2025-09-18), then ONE
--oos look (2025-09-18 -> 2026-07-09, includes the hostile Mar-Jul window),
then scripts/monte_carlo.py. Bar: expR_net >= 0.10 & n >= 30 & PF >= 1.15,
or the high-WR exception (>= 0.05 / n >= 60 / WR >= 55%).

VERDICT 2026-07-09: KILLED at the train screen (runs ee40ba7e DEFAULTS,
6d837c88 grid; boundary pinned 2025-09-18; OOS look NEVER spent). DEFAULTS
(1-ATR stop): n=346, 39.6% WR, expR_net +0.030, PF 1.05. Grid is monotone and
instructive — wider stop recovers more of the mined drift (1.0/1.5/2.0 ATR ->
+0.030/+0.042/+0.058), hold 6 always worse than hold 3 (the edge really is
~15min) — but the best cell (sl 2.0: +0.058R, PF 1.20, 44.8% WR, n=344) fails
the main bar (0.10) and the WR leg of the exception (55%). The mined drift is
REAL (the mining stats stand) but too thin to trade through any stop + costs
standalone. Salvage idea (untested): confluence/sizing input for AMD shorts
when a bearish displacement fires below the prior-day mid.
"""
import numpy as np

from ..events import detect_displacement
from .base import ENTRY_MARKET, MTFContext, make_signals, prior_day_stats

NAME = "displacement_pd"
TIMEFRAMES = ["M5"]
HTF_NEEDS = []
DEFAULTS = {
    "body_atr": 1.5,     # mining used detect_displacement defaults
    "body_frac": 0.7,
    "hold_bars": 3,      # the mined h=3 horizon
    "sl_atr": 1.0,       # mining's R-normalization unit as a real stop
}
PARAM_GRID = {
    "sl_atr": [1.0, 1.5, 2.0],
    "hold_bars": [3, 6],
}


def generate_signals(ctx: MTFContext, params: dict):
    df = ctx.df
    disp = detect_displacement(df, {"min_body_atr": params["body_atr"],
                                    "min_body_frac": params["body_frac"]})
    pdst = prior_day_stats(df)
    pd_mid = (pdst["d_high"] + pdst["d_low"]) / 2.0

    fired = (disp["fired"] & (disp["direction"] == -1)
             & (df["close"] < pd_mid) & pd_mid.notna()
             & df["atr"].notna()).to_numpy()

    close = df["close"].to_numpy(float)
    atr = df["atr"].to_numpy(float)
    rows = []
    for i in np.flatnonzero(fired):
        rows.append({
            "signal_idx": int(i),
            "direction": -1,
            "entry_type": ENTRY_MARKET,
            "sl": float(close[i] + params["sl_atr"] * atr[i]),
            "max_bars": int(params["hold_bars"]) - 1,
        })
    return make_signals(rows)
