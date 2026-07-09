"""
Trade simulator with engine-parity costs and honest fills.

Parity with src/backtest/execution.py (verified against source):
  - spread cost   = spread_points * $0.01 * contract_size * lots (crossed once)
  - slippage      = slippage_atr_mult * ATR in $/oz, charged at entry AND
                    embedded unfavorably in SL exit prices
  - commission    = commission_per_lot_round_turn * lots
  - swap          = swap_fee_per_lot_per_day * lots * rollover crossings
                    (first rollover strictly after entry; while roll <= exit)
  - WORST_CASE    = SL before TP when a bar touches both

Documented intentional divergences from the engine (all conservative or more
honest): MARKET fills at next-bar OPEN (engine: signal-bar close); STOP fills
gap-honest at max(open, stop); entry-bar exits ARE checked (engine starts at
bar+1); BE/trail updates computed on bar k take effect at bar k+1 (engine:
same bar); sizing replay uses floor() (engine risk.py rounds).

R accounting: r_price = dir*(exit-entry)/stop_dist (engine-comparable,
SL slippage already in exit). r_net = r_price - cost_per_oz/stop_dist.
SCREENING DECISIONS USE r_net.
"""
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from config import EXECUTION, RISK_MODEL
from .strategies.base import ENTRY_MARKET, ENTRY_LIMIT, ENTRY_STOP, MTFContext


@dataclass
class CostModel:
    spread_usd_oz: float = 0.30
    commission_usd_oz: float = 0.07
    slippage_atr_mult: float = 0.02
    swap_usd_oz_per_night: float = 0.05
    rollover_hh: int = 21
    rollover_mm: int = 59
    contract_size: float = 100.0

    @classmethod
    def from_config(cls, spread_points: float = None,
                    commission_usd_oz: float = None,
                    contract_size: float = None) -> "CostModel":
        """Config defaults are XAUUSD; pass overrides for other symbols
        (e.g. XAGUSD: spread_points 7 = $0.07/oz, commission $7/1000oz lot
        = 0.007, contract_size 1000)."""
        contract = float(contract_size if contract_size is not None
                         else RISK_MODEL.get("contract_size", 100))
        pts = spread_points if spread_points is not None else float(
            EXECUTION.get("fixed_spread_points", 30.0))
        swap = 0.0
        if EXECUTION.get("enable_swap_fees", False):
            swap = float(EXECUTION.get("swap_fee_per_lot_per_day", 0.0) or 0.0) / contract
        try:
            hh, mm = (int(x) for x in str(EXECUTION.get("rollover_time_utc", "21:59")).split(":"))
        except Exception:
            hh, mm = 21, 59
        return cls(
            spread_usd_oz=pts * 0.01,
            commission_usd_oz=(commission_usd_oz if commission_usd_oz is not None
                               else float(EXECUTION.get("commission_per_lot_round_turn", 7.0)) / contract),
            slippage_atr_mult=float(EXECUTION.get("slippage_atr_mult", 0.02)),
            swap_usd_oz_per_night=swap,
            rollover_hh=hh,
            rollover_mm=mm,
            contract_size=contract,
        )


def count_rollovers(entry_time: pd.Timestamp, exit_time: pd.Timestamp,
                    hh: int = 21, mm: int = 59) -> int:
    """Engine convention: first rollover strictly after entry; count while <= exit."""
    roll = entry_time.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if roll <= entry_time:
        roll += timedelta(days=1)
    nights = 0
    while roll <= exit_time:
        nights += 1
        roll += timedelta(days=1)
    return nights


def simulate(
    ctx: MTFContext,
    signals: pd.DataFrame,
    costs: CostModel,
    exit_flags_long: np.ndarray = None,
    exit_flags_short: np.ndarray = None,
):
    """
    Run all signals through the fill/exit rules. One position per stream.

    Returns (trades_df, stats_dict).
    """
    df = ctx.df
    n = len(df)
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    atr = df["atr"].to_numpy(float)
    ts = df["timestamp"]
    mod = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
    day = ts.dt.normalize().to_numpy()

    stats = {"signals": len(signals), "skipped_busy": 0, "unfilled": 0,
             "bad": 0, "taken": 0}
    trades = []
    busy_until = -1  # last exit bar; a signal ON the exit bar is allowed

    for row in signals.itertuples(index=False):
        si = int(row.signal_idx)
        if si < busy_until:
            stats["skipped_busy"] += 1
            continue
        if si + 1 >= n or not np.isfinite(atr[si]):
            stats["bad"] += 1
            continue
        direction = int(row.direction)
        etype = int(row.entry_type)
        j0 = si + 1

        # ---- fill ----
        fill_bar, entry = -1, np.nan
        if etype == ENTRY_MARKET:
            fill_bar, entry = j0, o[j0]
        else:
            level = float(row.entry_price)
            if not np.isfinite(level):
                stats["bad"] += 1
                continue
            j_end = min(si + max(1, int(row.ttl_bars)), n - 1)
            for j in range(j0, j_end + 1):
                if etype == ENTRY_LIMIT:
                    touched = l[j] <= level if direction > 0 else h[j] >= level
                    if touched:
                        fill_bar, entry = j, level  # engine parity: fill at limit
                        break
                else:  # STOP: gap-honest
                    touched = h[j] >= level if direction > 0 else l[j] <= level
                    if touched:
                        entry = max(o[j], level) if direction > 0 else min(o[j], level)
                        fill_bar = j
                        break
            if fill_bar < 0:
                stats["unfilled"] += 1
                continue

        sl = float(row.sl)
        stop_dist = (entry - sl) if direction > 0 else (sl - entry)
        if not np.isfinite(stop_dist) or stop_dist <= 0:
            stats["bad"] += 1
            continue
        tp = float(row.tp) if np.isfinite(row.tp) else np.nan
        has_tp = np.isfinite(tp)
        trail_mult = float(row.trail_atr_mult)
        trail_act = float(row.trail_act_r) * stop_dist
        be_at = float(row.be_at_r) * stop_dist
        eod = int(row.eod_hhmm)
        eod_min = (eod // 100) * 60 + (eod % 100) if eod > 0 else -1
        max_bars = int(row.max_bars)
        atr_entry = atr[fill_bar] if np.isfinite(atr[fill_bar]) else 0.0
        slip_entry = costs.slippage_atr_mult * atr_entry

        cur_sl = sl
        be_done = False
        trailing = False
        mfe = 0.0
        mae = 0.0
        exit_bar, exit_px, reason = -1, np.nan, ""

        for k in range(fill_bar, n):
            a_k = atr[k] if np.isfinite(atr[k]) else atr_entry
            slip_k = costs.slippage_atr_mult * a_k
            if direction > 0:
                sl_hit = l[k] <= cur_sl
                tp_hit = has_tp and h[k] >= tp
            else:
                sl_hit = h[k] >= cur_sl
                tp_hit = has_tp and l[k] <= tp

            if sl_hit:  # WORST_CASE: SL first when both touched
                exit_px = cur_sl - slip_k if direction > 0 else cur_sl + slip_k
                exit_bar, reason = k, "SL"
                break
            if tp_hit:
                exit_bar, exit_px, reason = k, tp, "TP"
                break

            # indicator exit at close (evaluated on completed bar)
            flags = exit_flags_long if direction > 0 else exit_flags_short
            if flags is not None and flags[k]:
                exit_bar, exit_px, reason = k, c[k], "IND"
                break

            # EOD flat: last bar of this trading day before the cutoff
            if eod_min >= 0 and mod[k] < eod_min:
                if k == n - 1 or day[k + 1] != day[k] or mod[k + 1] >= eod_min:
                    exit_bar, exit_px, reason = k, c[k], "EOD"
                    break

            if k - fill_bar >= max_bars:
                exit_bar, exit_px, reason = k, c[k], "TIMEOUT"
                break
            if k == n - 1:
                exit_bar, exit_px, reason = k, c[k], "EOD_DATA"
                break

            # ---- update excursions and next-bar stops (effective k+1) ----
            if direction > 0:
                mfe = max(mfe, h[k] - entry)
                mae = max(mae, entry - l[k])
                if be_at > 0 and not be_done and h[k] >= entry + be_at:
                    cur_sl = max(cur_sl, entry)
                    be_done = True
                if trail_mult > 0:
                    if not trailing and mfe >= trail_act:
                        trailing = True
                    if trailing:
                        cur_sl = max(cur_sl, h[k] - trail_mult * a_k)
            else:
                mfe = max(mfe, entry - l[k])
                mae = max(mae, h[k] - entry)
                if be_at > 0 and not be_done and l[k] <= entry - be_at:
                    cur_sl = min(cur_sl, entry)
                    be_done = True
                if trail_mult > 0:
                    if not trailing and mfe >= trail_act:
                        trailing = True
                    if trailing:
                        cur_sl = min(cur_sl, l[k] + trail_mult * a_k)

        # excursions on the exit bar too (for MAE-based DD replay)
        if exit_bar >= 0:
            if direction > 0:
                mae = max(mae, entry - l[exit_bar])
                mfe = max(mfe, h[exit_bar] - entry)
            else:
                mae = max(mae, h[exit_bar] - entry)
                mfe = max(mfe, entry - l[exit_bar])

        entry_time = ts.iloc[fill_bar]
        exit_time = ts.iloc[exit_bar]
        nights = count_rollovers(entry_time, exit_time,
                                 costs.rollover_hh, costs.rollover_mm) \
            if costs.swap_usd_oz_per_night > 0 else 0
        cost_per_oz = (costs.spread_usd_oz + slip_entry + costs.commission_usd_oz
                       + costs.swap_usd_oz_per_night * nights)
        r_price = direction * (exit_px - entry) / stop_dist
        r_net = r_price - cost_per_oz / stop_dist

        trades.append({
            "signal_time": ts.iloc[si],
            "entry_time": entry_time,
            "exit_time": exit_time,
            "direction": "LONG" if direction > 0 else "SHORT",
            "entry": entry,
            "exit": exit_px,
            "sl0": sl,
            "stop_dist": stop_dist,
            "exit_reason": reason,
            "bars_held": exit_bar - fill_bar,
            "nights": nights,
            "mfe_r": mfe / stop_dist,
            "mae_r": mae / stop_dist,
            "r_price": r_price,
            "cost_per_oz": cost_per_oz,
            "r_net": r_net,
            "atr_entry": atr_entry,
            "tag": int(row.tag),
        })
        stats["taken"] += 1
        busy_until = exit_bar

    return pd.DataFrame(trades), stats
