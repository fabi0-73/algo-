"""
RSI(2) pullback with SMA(200) trend filter (Connors family, intraday adaptation).

Evidence: strong on US equity dailies (75-91% WR claims), ~50% WR in the one
independent intraday index test, and gold-specific evidence is negative
(RSI(14) fade on XAUUSD lost before costs). Included as a cheap falsification
test — the reference implementation for the lab interface.

LONG: close > SMA200 and RSI(2) < rsi_buy -> market entry next open.
Exit: close > SMA(5) or RSI(2) > rsi_exit (indicator exit), hard SL sl_atr*ATR.
SHORT mirror. All computed on the entry TF.

VERDICT 2026-07-06: KILLED on train (run 65fecd3b) — best combo -0.002R net, PF<=1.00. 57-65% WR is real but costs eat it; matches negative gold prior.
"""
import numpy as np

from .base import MTFContext, make_signals, rsi, ENTRY_MARKET

NAME = "rsi2_trend"
TIMEFRAMES = ["M30", "H1"]
HTF_NEEDS = []
DEFAULTS = {
    "rsi_period": 2,
    "rsi_buy": 5.0,        # long when RSI < this
    "rsi_sell": 95.0,      # short when RSI > this
    "rsi_exit_hi": 70.0,   # long indicator exit
    "rsi_exit_lo": 30.0,   # short indicator exit
    "sma_trend": 200,
    "sma_exit": 5,
    "sl_atr": 1.5,
    "max_bars": 96,
    "both_sides": True,
}
PARAM_GRID = {
    "rsi_buy": [5.0, 10.0],
    "sl_atr": [1.5, 2.5],
}


def generate_signals(ctx: MTFContext, params: dict):
    df = ctx.df
    close = df["close"]
    r = rsi(close, int(params["rsi_period"]))
    sma_t = close.rolling(int(params["sma_trend"])).mean()
    sma_x = close.rolling(int(params["sma_exit"])).mean()
    atr = df["atr"]

    long_sig = (close > sma_t) & (r < params["rsi_buy"])
    short_sig = (close < sma_t) & (r > params["rsi_sell"]) if params["both_sides"] else \
        np.zeros(len(df), dtype=bool)

    exit_flags_long = ((close > sma_x) | (r > params["rsi_exit_hi"])).to_numpy()
    exit_flags_short = ((close < sma_x) | (r < params["rsi_exit_lo"])).to_numpy()

    valid = atr.notna() & sma_t.notna()
    rows = []
    for i in np.flatnonzero((long_sig & valid).to_numpy()):
        rows.append({"signal_idx": int(i), "direction": 1, "entry_type": ENTRY_MARKET,
                     "sl": float(close.iloc[i] - params["sl_atr"] * atr.iloc[i]),
                     "max_bars": int(params["max_bars"])})
    for i in np.flatnonzero((short_sig & valid).to_numpy()):
        rows.append({"signal_idx": int(i), "direction": -1, "entry_type": ENTRY_MARKET,
                     "sl": float(close.iloc[i] + params["sl_atr"] * atr.iloc[i]),
                     "max_bars": int(params["max_bars"])})

    return make_signals(rows), {"exit_flags_long": exit_flags_long,
                                "exit_flags_short": exit_flags_short}
