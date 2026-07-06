"""
Pullback-window trend entry (open-source 'Sunrise' XAUUSD M5 port, grade B;
claimed 55.4% WR / PF 1.64 / ~3 trades-mo — published code was long-only).

State machine: ARM when close (EMA-1) crosses above ALL of EMA(14/18/24)
with EMA(14) rising over the last slope_bars (mirror for shorts). Then wait
for a pullback of 1-3 consecutive counter-trend candles (close < open for
longs); after each pullback candle a 2-bar WINDOW opens — a STOP order at
the pullback extreme +/- 0.1*ATR (the resting-order semantics of the
original EA, which modifies the pending stop as the pullback extends).
A 4th consecutive counter-trend candle invalidates the setup.

SL 2.5*ATR from the entry level, no TP — managed by BE at 1R and a 2*ATR
trail activating at 1R, timeout 240 M5 bars (scaled /3 on M15). Extreme-vol
block (skip when ATR > 0.25% of price) and broker 10:00-22:00 session gate
on the signal bar; trades may run past the session (eod off).

VERDICT 2026-07-06: KILLED on train (run 65fecd3b) — best (long_only) +0.030R < 0.10R bar, PF 1.06. Port trades 60x the published frequency; 55% WR claim not reproduced.
"""
import numpy as np

from .base import MTFContext, make_signals, session_mask, ENTRY_STOP

NAME = "pullback_window"
TIMEFRAMES = ["M5", "M15"]
HTF_NEEDS = []
DEFAULTS = {
    "ema_fast": 14,
    "ema_mid": 18,
    "ema_slow": 24,
    "slope_bars": 3,
    "atr_max_pct": 0.0025,   # skip signals when ATR > this fraction of price
    "max_pullback": 3,       # 1-3 counter-trend candles; 4th invalidates
    "entry_offset_atr": 0.1,
    "sl_atr": 2.5,
    "trail_atr_mult": 2.0,
    "trail_act_r": 1.0,
    "be_at_r": 1.0,
    "ttl_bars": 2,           # the 2-bar entry window
    "max_bars_m5": 240,      # scaled by TF minutes (M15 -> 80)
    "session_start": "10:00",
    "session_end": "22:00",
    "long_only": False,
}
PARAM_GRID = {
    "long_only": [False, True],
    "sl_atr": [2.0, 2.5],
}

_TF_SCALE = {"M5": 1, "M15": 3}


def generate_signals(ctx: MTFContext, params: dict):
    df = ctx.df
    close = df["close"]
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = close.to_numpy(float)
    atr = df["atr"].to_numpy(float)

    ema_f = close.ewm(span=int(params["ema_fast"]), adjust=False).mean()
    ema_m = close.ewm(span=int(params["ema_mid"]), adjust=False).mean()
    ema_s = close.ewm(span=int(params["ema_slow"]), adjust=False).mean()

    above = (close > ema_f) & (close > ema_m) & (close > ema_s)
    below = (close < ema_f) & (close < ema_m) & (close < ema_s)
    slope_bars = int(params["slope_bars"])
    cross_up = (above & ~above.shift(1, fill_value=False)
                & (ema_f.diff(slope_bars) > 0)).to_numpy()
    cross_dn = (below & ~below.shift(1, fill_value=False)
                & (ema_f.diff(slope_bars) < 0)).to_numpy()

    in_session = session_mask(df["timestamp"], params["session_start"],
                              params["session_end"]).to_numpy()
    vol_ok = atr <= float(params["atr_max_pct"]) * c

    long_only = bool(params["long_only"])
    max_pull = int(params["max_pullback"])
    off = float(params["entry_offset_atr"])
    sl_atr = float(params["sl_atr"])
    max_bars = max(1, int(params["max_bars_m5"]) // _TF_SCALE.get(ctx.tf, 1))
    warm = max(int(params["ema_slow"]), slope_bars) + 1

    rows = []
    state = 0       # 0 = idle, 1 = armed (crossover seen, tracking pullback)
    direction = 0
    pull_count = 0
    pull_ext = np.nan

    for i in range(len(df)):
        if i < warm or not np.isfinite(atr[i]):
            state = 0
            continue
        if state == 0:
            if cross_up[i]:
                state, direction, pull_count, pull_ext = 1, 1, 0, -np.inf
            elif (not long_only) and cross_dn[i]:
                state, direction, pull_count, pull_ext = 1, -1, 0, np.inf
            continue

        # armed: opposite full cross flips (or disarms when long-only);
        # a fresh same-direction cross restarts the pullback tracking
        if direction > 0 and cross_dn[i]:
            if long_only:
                state = 0
            else:
                direction, pull_count, pull_ext = -1, 0, np.inf
            continue
        if direction < 0 and cross_up[i]:
            direction, pull_count, pull_ext = 1, 0, -np.inf
            continue
        if direction > 0 and cross_up[i]:
            pull_count, pull_ext = 0, -np.inf
            continue
        if direction < 0 and cross_dn[i]:
            pull_count, pull_ext = 0, np.inf
            continue

        counter = c[i] < o[i] if direction > 0 else c[i] > o[i]
        if counter:
            pull_count += 1
            if pull_count > max_pull:
                state = 0
                continue
            if direction > 0:
                pull_ext = max(pull_ext, h[i])
                level = pull_ext + off * atr[i]
                sl = level - sl_atr * atr[i]
            else:
                pull_ext = min(pull_ext, l[i])
                level = pull_ext - off * atr[i]
                sl = level + sl_atr * atr[i]
            if in_session[i] and vol_ok[i]:
                rows.append({
                    "signal_idx": int(i), "direction": direction,
                    "entry_type": ENTRY_STOP,
                    "entry_price": float(level), "sl": float(sl),
                    "trail_atr_mult": float(params["trail_atr_mult"]),
                    "trail_act_r": float(params["trail_act_r"]),
                    "be_at_r": float(params["be_at_r"]),
                    "ttl_bars": int(params["ttl_bars"]),
                    "max_bars": max_bars,
                    "tag": pull_count,
                })
        elif pull_count >= 1:
            # pullback ended on a trend candle: setup consumed; the last
            # emitted window (ttl 2 bars) still covers the breakout entry
            state = 0

    return make_signals(rows)
