"""
Portfolio: champion AMD (engine trades) + ny_ib lab stream on one $500 ledger,
with position-overlap analysis and a bootstrap sizing frontier.

Champion R is net-of-costs: net_pnl / (stop_dist * contract * lots).
Lab R is r_net (costs already in R). Both streams replayed chronologically
through the real sizing rule (floor + min-lot clamp). MC frontier resamples
the combined pool jointly (net_r, stop_dist, mae_r).

Caveat (documented): lab fills use close-confirmation/limit conventions,
engine fills use the full execution model — the merge is an approximation
pending engine integration of the ny_ib stream.

Usage:
    python scripts/lab_portfolio.py --report reports/backtest_124d15ef --sims 10000
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RISK_MODEL
from src.research import mtf
from src.research.lab import CostModel, simulate
from src.research.strategies import rsi2_trend  # noqa: F401 (registry warm)
from src.research.strategies.base import MTFContext
from src.research.strategies import ny_ib

CONTRACT = float(RISK_MODEL.get("contract_size", 100))
LOT_STEP = float(RISK_MODEL.get("lot_step", 0.01))
MAX_LOT = float(RISK_MODEL.get("max_lot", 1.0))
INITIAL = 500.0
RUIN = 100.0

NY_IB_CONFIG = dict(ny_ib.DEFAULTS, retrace_frac=0.10, exit_style="high_wr")


def load_champion(report_dir):
    with open(Path(report_dir) / "results.json") as f:
        trades = json.load(f)["trades"]
    rows = []
    for t in trades:
        entry = float(t.get("actual_fill_price") or t["entry_price"])
        sl = float(t.get("original_sl") or t["sl_price"])
        stop = abs(entry - sl)
        lots = float(t["position_size"])
        if stop <= 0 or lots <= 0:
            continue
        rows.append({
            "stream": "amd",
            "entry_time": datetime.strptime(t["entry_time"], "%Y-%m-%d %H:%M:%S"),
            "exit_time": datetime.strptime(t["exit_time"], "%Y-%m-%d %H:%M:%S"),
            "stop_dist": stop,
            "net_r": float(t["net_pnl"]) / (stop * CONTRACT * lots),
            "mae_r": float(t.get("mae_r") or 0.0),
        })
    return pd.DataFrame(rows)


def build_ny_ib():
    m5 = mtf.load_m5()
    df = mtf.prepare_frame(m5, "M5")
    ctx = MTFContext(tf="M5", df=df, htf={})
    signals = ny_ib.generate_signals(ctx, NY_IB_CONFIG)
    trades, _ = simulate(ctx, signals, CostModel.from_config())
    out = pd.DataFrame({
        "stream": "ny_ib",
        "entry_time": trades["entry_time"],
        "exit_time": trades["exit_time"],
        "stop_dist": trades["stop_dist"],
        "net_r": trades["r_net"],
        "mae_r": trades["mae_r"],
    })
    return out


def overlap_stats(a: pd.DataFrame, b: pd.DataFrame):
    """Count trades in b whose holding interval intersects any interval in a."""
    ints = list(zip(a["entry_time"], a["exit_time"]))
    hits = 0
    for e, x in zip(b["entry_time"], b["exit_time"]):
        if any(e < ax and x > ae for ae, ax in ints):
            hits += 1
    return hits, len(b)


def replay(trades: pd.DataFrame, min_lot: float, risk_pct: float = 0.005,
           with_trough: bool = True):
    t = trades.sort_values("entry_time").reset_index(drop=True)
    equity, peak, max_dd = INITIAL, INITIAL, 0.0
    wins = 0
    monthly = {}
    for row in t.itertuples(index=False):
        risk_per_lot = row.stop_dist * CONTRACT
        lots = np.floor((equity * risk_pct) / risk_per_lot / LOT_STEP) * LOT_STEP
        lots = min(max(lots, min_lot), MAX_LOT)
        pnl = row.net_r * risk_per_lot * lots
        points = []
        if with_trough:
            points.append(equity - row.mae_r * risk_per_lot * lots)
        points.append(equity + pnl)
        for p in points:
            peak = max(peak, p)
            if peak > 0:
                max_dd = max(max_dd, (peak - p) / peak)
        equity += pnl
        wins += pnl > 0
        mk = pd.Timestamp(row.entry_time).strftime("%Y-%m")
        monthly[mk] = monthly.get(mk, 0.0) + pnl
    prof = sum(1 for v in monthly.values() if v > 0)
    return {"final": equity, "max_dd": max_dd, "wr": wins / len(t),
            "n": len(t), "prof_months": prof / len(monthly) if monthly else 0}


def frontier(pool: pd.DataFrame, min_lot: float, n_sims: int, rng,
             risk_pct: float = 0.005):
    arr = pool[["net_r", "stop_dist", "mae_r"]].to_numpy(float)
    n = len(arr)
    finals = np.empty(n_sims)
    ruined = np.zeros(n_sims, bool)
    hit3k = np.zeros(n_sims, bool)
    dds = np.empty(n_sims)
    for s in range(n_sims):
        seq = arr[rng.integers(0, n, n)]
        eq, peak, mdd = INITIAL, INITIAL, 0.0
        for r, stop, mae in seq:
            rpl = stop * CONTRACT
            lots = np.floor((eq * risk_pct) / rpl / LOT_STEP) * LOT_STEP
            lots = min(max(lots, min_lot), MAX_LOT)
            trough = eq - mae * rpl * lots
            eq += r * rpl * lots
            for p in (trough, eq):
                peak = max(peak, p)
                if peak > 0:
                    mdd = max(mdd, (peak - p) / peak)
            if eq >= 3000:
                hit3k[s] = True
            if eq <= RUIN:
                ruined[s] = True
                break
        finals[s] = eq
        dds[s] = mdd
    mult = finals / INITIAL
    return {"median": float(np.median(mult)), "p5": float(np.percentile(mult, 5)),
            "p95": float(np.percentile(mult, 95)), "ruin": float(ruined.mean()),
            "p3k": float(hit3k.mean()), "dd_med": float(np.median(dds))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="reports/backtest_124d15ef")
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--floors", default="0.01,0.02,0.03")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    amd = load_champion(args.report)
    nyib = build_ny_ib()
    months = 17.9  # 2024-09-05 .. 2026-02-27

    print(f"AMD champion: {len(amd)} trades ({len(amd)/months:.1f}/mo), "
          f"WR {(amd['net_r'] > 0).mean():.1%}, avg net R {amd['net_r'].mean():+.3f}")
    print(f"ny_ib stream: {len(nyib)} trades ({len(nyib)/months:.1f}/mo), "
          f"WR {(nyib['net_r'] > 0).mean():.1%}, avg net R {nyib['net_r'].mean():+.3f}")

    h, n = overlap_stats(amd, nyib)
    h2, n2 = overlap_stats(nyib, amd)
    print(f"Overlap: {h}/{n} ny_ib trades open while an AMD position is on; "
          f"{h2}/{n2} AMD trades vice versa")

    combined = pd.concat([amd, nyib], ignore_index=True)
    print(f"\nCombined: {len(combined)} trades ({len(combined)/months:.1f}/mo), "
          f"blended WR {(combined['net_r'] > 0).mean():.1%}")

    print("\nChronological replay at $500 (floor 0.01):")
    for label, pool in (("AMD only", amd), ("ny_ib only", nyib),
                        ("combined", combined)):
        r = replay(pool, 0.01)
        print(f"  {label:<10} n={r['n']:>3} WR {r['wr']:.1%} final ${r['final']:.0f} "
              f"maxDD {r['max_dd']:.1%} profitable months {r['prof_months']:.0%}")

    rng = np.random.default_rng(args.seed)
    floors = [float(x) for x in args.floors.split(",")]
    print(f"\nBootstrap frontier ({args.sims} sims, 17.9mo horizon, ruin <= $100):")
    hdr = (f"{'pool':<10} {'floor':>5} | {'median':>7} | {'5th':>6} | {'95th':>7} | "
           f"{'P(ruin)':>7} | {'P($3k)':>7} | {'medDD':>6}")
    print(hdr)
    print("-" * len(hdr))
    for label, pool in (("AMD", amd), ("combined", combined)):
        for f in floors:
            m = frontier(pool, f, args.sims, rng)
            print(f"{label:<10} {f:>5.2f} | {m['median']:>6.2f}x | {m['p5']:>5.2f}x | "
                  f"{m['p95']:>6.2f}x | {m['ruin']:>6.1%} | {m['p3k']:>6.1%} | "
                  f"{m['dd_med']:>6.1%}")


if __name__ == "__main__":
    main()
