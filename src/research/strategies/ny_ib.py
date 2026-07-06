"""
NY Initial Balance retracement (Trade That Swing IB pullback family).

IB = high/low of the first NY hour, broker 16:30-17:30 (= 09:30-10:30 ET),
built from COMPLETED M5 bars only (the last IB bar opens 17:25 and closes
17:30, so any signal bar at/after 17:30 sits strictly later positionally).
Filter: IB size within [ib_min_pct, ib_max_pct] of price. After a bar
CLOSES beyond the IB within broker 17:30-22:00, place a LIMIT back inside
the range at retrace_frac of IB size from the broken edge (the pullback
entry is what separates this from naive ORB market entries). One trade per
day; pending order expires before 23:00; all positions flat 23:00 broker.

Exit styles (SL/TP are prices from IB geometry):
  classic: SL 0.35*IB beyond entry away from break dir, TP 0.30*IB past edge
  high_wr: SL 1.00*IB, TP 0.20*IB past edge (small target, high win rate)

VERDICT 2026-07-06: KEEPER (runs 65fecd3b/6ef3f2fb) — high_wr retrace 0.10: train 103tr 72.8% WR +0.057R PF 1.45; OOS 71tr 83.1% WR +0.121R PF 2.15 DD 14.6% (deg 2.1). Survives $0.45 spread. classic+long_only also passed (train +0.141R, OOS +0.258R). Lab-validated only — engine integration pending.
"""
import numpy as np

from .base import MTFContext, make_signals, session_mask, ENTRY_LIMIT

NAME = "ny_ib"
TIMEFRAMES = ["M5"]
HTF_NEEDS = []
DEFAULTS = {
    "ib_start": "16:30",     # broker 16:30 = 09:30 ET (NY open)
    "ib_end": "17:30",       # broker 17:30 = 10:30 ET
    "scan_end": "22:00",     # breakout close must occur before this
    "flat_hhmm": 2300,       # force-flat and pending-order cutoff
    "min_ib_bars": 10,       # of 12 possible M5 bars in the IB hour
    "ib_min_pct": 0.004,
    "ib_max_pct": 0.02,
    "retrace_frac": 0.25,
    "exit_style": "classic",
    "long_only": False,
    "max_bars": 70,          # backstop only; EOD flat is the real exit
}
PARAM_GRID = {
    "retrace_frac": [0.10, 0.25],
    "exit_style": ["classic", "high_wr"],
    "long_only": [False, True],
}

_EXITS = {"classic": (0.35, 0.30), "high_wr": (1.00, 0.20)}  # (sl_frac, tp_frac)


def generate_signals(ctx: MTFContext, params: dict):
    df = ctx.df
    ts = df["timestamp"]
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    atr = df["atr"].to_numpy(float)
    mod = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()

    ib_mask = session_mask(ts, params["ib_start"], params["ib_end"]).to_numpy()
    scan_mask = session_mask(ts, params["ib_end"], params["scan_end"]).to_numpy()
    flat = int(params["flat_hhmm"])
    flat_min = (flat // 100) * 60 + flat % 100

    sl_frac, tp_frac = _EXITS[params["exit_style"]]
    retrace = float(params["retrace_frac"])
    long_only = bool(params["long_only"])

    rows = []
    for _, g in df.groupby(ts.dt.normalize()):
        gi = g.index.to_numpy()  # positional indices, in time order
        ib_idx = gi[ib_mask[gi]]
        if len(ib_idx) < int(params["min_ib_bars"]):
            continue
        ib_high = h[ib_idx].max()
        ib_low = l[ib_idx].min()
        ib_size = ib_high - ib_low
        ref = c[ib_idx[-1]]  # last completed IB bar's close
        if not (params["ib_min_pct"] * ref <= ib_size <= params["ib_max_pct"] * ref):
            continue
        fill_ok = gi[mod[gi] < flat_min]
        if len(fill_ok) == 0:
            continue
        last_fill = fill_ok[-1]  # last bar of this day before the flat cutoff

        for i in gi[scan_mask[gi]]:
            if not np.isfinite(atr[i]):
                continue
            if c[i] > ib_high:
                direction = 1
            elif not long_only and c[i] < ib_low:
                direction = -1
            else:
                continue
            ttl = int(last_fill - i)
            if ttl < 1:
                break  # breakout too late in a shortened day
            if direction > 0:
                entry = ib_high - retrace * ib_size
                sl = entry - sl_frac * ib_size
                tp = ib_high + tp_frac * ib_size
            else:
                entry = ib_low + retrace * ib_size
                sl = entry + sl_frac * ib_size
                tp = ib_low - tp_frac * ib_size
            rows.append({"signal_idx": int(i), "direction": direction,
                         "entry_type": ENTRY_LIMIT, "entry_price": float(entry),
                         "sl": float(sl), "tp": float(tp), "ttl_bars": ttl,
                         "max_bars": int(params["max_bars"]),
                         "eod_hhmm": flat, "tag": 1 if direction > 0 else 2})
            break  # one trade per day

    return make_signals(rows)
