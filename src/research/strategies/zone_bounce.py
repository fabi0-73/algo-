"""
S/R zone bounce with a 1:1 bracket in the NY session (Choon Chiat
FTMO-verified style — the only verified ~70% WR family in the roster).

Zones come from M30: any rolling zone_bars-bar window whose total range is
under zone_max_atr * M30-ATR marks a consolidation; its hi/lo is carried
forward so each M5 bar sees the LAST COMPLETED zone (align_htf keeps this
lookahead-safe). LONG: during broker 16:30-22:00, the bar's low touches
zone_lo, the bar closes back above it with a bullish body, the low pierced
the Bollinger(20,2) lower band this bar or the previous one, and RSI(14)
< rsi_buy -> market entry next open. SL sits sl_frac * (close - zone_lo)
below zone_lo (stop distance floored at min_sl_atr * ATR), TP is exactly
rr * stop distance, both anchored to the signal-bar close. SHORT mirror at
zone_hi. Max 3 trades per day, force-flat 23:30 broker.

VERDICT 2026-07-06: KILLED on train (run 65fecd3b) — all combos negative (best -0.121R), n<=14. The verified-trader 70% WR did not reproduce as codeable rules.
"""
import numpy as np

from src.strategy.indicators import calculate_atr
from .base import MTFContext, make_signals, rsi, session_mask, ENTRY_MARKET

NAME = "zone_bounce"
TIMEFRAMES = ["M5"]
HTF_NEEDS = ["M30"]
DEFAULTS = {
    "zone_bars": 12,       # M30 bars in the consolidation window (6h)
    "zone_max_atr": 2.0,   # window range must be < this * M30 ATR
    "bb_period": 20,
    "bb_std": 2.0,
    "rsi_period": 14,
    "rsi_buy": 40.0,       # long when RSI < this
    "rsi_sell": 60.0,      # short when RSI > this
    "sl_frac": 1.0,        # SL this fraction of (entry_est - zone edge) beyond it
    "min_sl_atr": 0.8,     # stop-distance floor in entry-TF ATR
    "rr": 1.0,             # TP at exactly this many R
    "max_trades_day": 3,
    "eod_hhmm": 2330,
    "long_only": False,
}
PARAM_GRID = {
    "rr": [1.0, 1.5],
    "long_only": [False, True],
}


def _m30_zones(h):
    """M30 ATR + last completed consolidation zone hi/lo (past values only)."""
    h = h.copy()
    h["atr"] = calculate_atr(h, period=14)
    zb = int(DEFAULTS["zone_bars"])
    roll_hi = h["high"].rolling(zb).max()
    roll_lo = h["low"].rolling(zb).min()
    is_zone = (roll_hi - roll_lo) < DEFAULTS["zone_max_atr"] * h["atr"]
    h["zone_hi"] = roll_hi.where(is_zone).ffill()
    h["zone_lo"] = roll_lo.where(is_zone).ffill()
    return h


HTF_INDICATORS = {"M30": _m30_zones}


def generate_signals(ctx: MTFContext, params: dict):
    df = ctx.df
    op = df["open"]
    hi = df["high"]
    lo = df["low"]
    close = df["close"]
    atr = df["atr"]
    zone_lo = ctx.htf["M30"]["m30_zone_lo"]
    zone_hi = ctx.htf["M30"]["m30_zone_hi"]

    mid = close.rolling(int(params["bb_period"])).mean()
    sd = close.rolling(int(params["bb_period"])).std(ddof=0)
    bb_lo = mid - params["bb_std"] * sd
    bb_up = mid + params["bb_std"] * sd
    r = rsi(close, int(params["rsi_period"]))

    in_sess = session_mask(df["timestamp"], "16:30", "22:00")
    pierced_lo = (lo <= bb_lo) | (lo.shift(1) <= bb_lo.shift(1))
    pierced_hi = (hi >= bb_up) | (hi.shift(1) >= bb_up.shift(1))

    long_sig = (in_sess & atr.notna() & bb_lo.notna() & zone_lo.notna()
                & (lo <= zone_lo) & (close > zone_lo) & (close > op)
                & pierced_lo & (r < params["rsi_buy"]))
    short_sig = (in_sess & atr.notna() & bb_up.notna() & zone_hi.notna()
                 & (hi >= zone_hi) & (close < zone_hi) & (close < op)
                 & pierced_hi & (r > params["rsi_sell"]))
    if params["long_only"]:
        short_sig &= False

    day = df["timestamp"].dt.normalize()
    cand = [(int(i), 1) for i in np.flatnonzero(long_sig.to_numpy())]
    cand += [(int(i), -1) for i in np.flatnonzero(short_sig.to_numpy())]
    cand.sort()

    rows = []
    day_count = {}
    for i, direction in cand:
        d = day.iloc[i]
        if day_count.get(d, 0) >= int(params["max_trades_day"]):
            continue
        entry_est = float(close.iloc[i])
        a = float(atr.iloc[i])
        if direction > 0:
            edge = float(zone_lo.iloc[i])
            sl = edge - params["sl_frac"] * (entry_est - edge)
            sl = min(sl, entry_est - params["min_sl_atr"] * a)
            stop_dist = entry_est - sl
            tp = entry_est + params["rr"] * stop_dist
        else:
            edge = float(zone_hi.iloc[i])
            sl = edge + params["sl_frac"] * (edge - entry_est)
            sl = max(sl, entry_est + params["min_sl_atr"] * a)
            stop_dist = sl - entry_est
            tp = entry_est - params["rr"] * stop_dist
        rows.append({"signal_idx": i, "direction": direction,
                     "entry_type": ENTRY_MARKET, "sl": float(sl),
                     "tp": float(tp), "eod_hhmm": int(params["eod_hhmm"])})
        day_count[d] = day_count.get(d, 0) + 1

    return make_signals(rows)
