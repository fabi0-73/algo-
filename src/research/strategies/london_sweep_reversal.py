"""
London sweep-and-reclaim REVERSAL of the Asian range (the reversal twin of
the KILLED asian_breakout continuation module).

VERDICT 2026-07-23: KILLED at the train screen (run ebed0a62) — M5: 142
trades, 19.7% WR, expR −0.272, PF 0.60; M15: 104 trades, 24.0% WR, expR
−0.275, PF 0.55. Both far below the 0.10R bar; no grid run, no OOS spent.
The mean-reversion drift the event study found on Asia-high breaks is real
but the reclaim-entry/box-target/session-boxed harvest loses it to entry
timing + costs. Together with asian_breakout (continuation twin, also
killed), the Asia-range family is closed in BOTH directions on this feed.

Pre-registered geometry, 2026-07-23 — evidence stack before any outcome:
  * Own data: the only FDR-PASS cell in the 8.5y B2 event study says London
    breaks of the Asia high MEAN-REVERT (excess -0.176 ATR @ h=3, p_adj
    0.006) — raw drift doesn't clear costs, so the bet is that trade
    management (structural stop past the sweep, box-anchored target,
    time-boxed session) harvests what fixed-horizon drift cannot, exactly
    as AMD's own drift-null atoms do.
  * Osler (FRBNY 150): stop-cluster penetration CASCADES; reversals are
    reliable only after reclaim confirmation -> we NEVER fade the sweep
    bar itself; entry requires a CLOSE back inside the box.
  * Reddit consensus (2026-07 sweep): Asia = liquidity box; London sweeps
    it; enter on reclaim/MSS, stop past the manipulation extreme, target
    opposite liquidity BUT "opposite side is rarely hit in London" -> box
    midpoint default target; close before NY ("NY reverses London").

Rules:
  Asian box = high/low of broker 02:00-10:00 (complete before London).
  Box sanity: range within [min_range_adr, max_range_adr] * ADR(20).
  Sweep: first London-window (10:00-16:00) bar whose extreme pokes
  >= poke_min_atr * ATR beyond a box edge. One attempt/day/side; the first
  sweep consumes the side's attempt.
  Reclaim: within reclaim_max_bars after the sweep bar, a bar CLOSES back
  inside the box -> signal at that bar, MARKET next bar, direction =
  reversal. No reclaim in window = cascade day = no trade (Osler).
  SL: beyond the sweep-leg extreme +/- sl_buf_atr * ATR (double-sweep
  protection). TP: box midpoint ("mid", default) or opposite edge ("far");
  skip if implied RR < min_rr. BE at 1R. Force-flat before NY (16:30) by
  default — grid tests holding to 23:30.
"""
import numpy as np

from .base import MTFContext, make_signals, session_mask, adr, ENTRY_MARKET

NAME = "london_sweep_reversal"
TIMEFRAMES = ["M5", "M15"]
HTF_NEEDS = []
DEFAULTS = {
    "asian_start": "02:00",
    "asian_end": "10:00",
    "sweep_start": "10:00",
    "sweep_end": "16:00",
    "poke_min_atr": 0.05,
    "reclaim_max_bars": 6,
    "min_range_adr": 0.15,
    "max_range_adr": 0.60,
    "adr_days": 20,
    "min_asian_frac": 0.6,
    "sl_buf_atr": 0.25,
    "tp_mode": "mid",        # "mid" = box midpoint, "far" = opposite edge
    "min_rr": 1.0,
    "be_at_r": 1.0,
    "eod_hhmm": 1630,        # flat before NY; grid tests 2330
}
PARAM_GRID = {
    "reclaim_max_bars": [3, 6],
    "tp_mode": ["mid", "far"],
    "eod_hhmm": [1630, 2330],
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
    sweep_win = session_mask(ts, params["sweep_start"], params["sweep_end"]).to_numpy()

    bar_min = int(ts.diff().dt.total_seconds().div(60).median())
    win_min = 8 * 60
    min_bars = max(4, int(params["min_asian_frac"] * win_min / max(bar_min, 1)))
    reclaim_max = int(params["reclaim_max_bars"])
    tp_mid = params["tp_mode"] == "mid"

    rows = []
    for _, pos in df.groupby(day.values).indices.items():
        pos = np.sort(np.asarray(pos))
        a_pos = pos[asian[pos]]
        if len(a_pos) < min_bars:
            continue
        i_end = int(a_pos.max())
        d_end = adr20[i_end]
        if not np.isfinite(d_end) or d_end <= 0:
            continue
        box_hi = float(o_hi[a_pos].max())
        box_lo = float(o_lo[a_pos].min())
        rng = box_hi - box_lo
        if rng > params["max_range_adr"] * d_end or rng < params["min_range_adr"] * d_end:
            continue
        box_mid = box_lo + rng / 2.0

        s_pos = pos[sweep_win[pos] & (pos > i_end)]
        for side in (1, -1):                       # 1 = high sweep -> SHORT
            swept_at = None
            for i in s_pos:
                i = int(i)
                a = atr[i]
                if not np.isfinite(a) or a <= 0:
                    continue
                if swept_at is None:
                    poked = (o_hi[i] >= box_hi + params["poke_min_atr"] * a
                             if side == 1 else
                             o_lo[i] <= box_lo - params["poke_min_atr"] * a)
                    if poked:
                        swept_at = i               # first sweep consumes the side
                    continue
                # reclaim window after the sweep bar
                if i - swept_at > reclaim_max:
                    break                          # cascade day: stand aside
                reclaimed = (o_cl[i] < box_hi) if side == 1 else (o_cl[i] > box_lo)
                if not reclaimed:
                    continue
                leg = slice(swept_at, i + 1)
                if side == 1:
                    sl = float(o_hi[leg].max() + params["sl_buf_atr"] * a)
                    tp = float(box_mid if tp_mid else box_lo)
                    risk = sl - o_cl[i]
                    reward = o_cl[i] - tp
                else:
                    sl = float(o_lo[leg].min() - params["sl_buf_atr"] * a)
                    tp = float(box_mid if tp_mid else box_hi)
                    risk = o_cl[i] - sl
                    reward = tp - o_cl[i]
                if risk > 0 and reward / risk >= params["min_rr"]:
                    rows.append({"signal_idx": i, "direction": -side,
                                 "entry_type": ENTRY_MARKET, "sl": sl,
                                 "tp": tp,
                                 "be_at_r": float(params["be_at_r"]),
                                 "eod_hhmm": int(params["eod_hhmm"])})
                break                              # one attempt per side per day

    return make_signals(rows)
