"""
E6 — Sizing frontier via bootstrap Monte Carlo on a validated trade list.

Resamples the champion's trades (r_multiple + initial stop distance + intended
risk pct) and replays them sequentially through the real sizing rule
(risk-% with a min-lot floor), at several floor levels. Reports the honest
frontier: median growth, P(reach $3,000 in month 1), P(15x in 18mo), P(ruin).

Usage:
    python scripts/sizing_frontier.py --report reports/backtest_124d15ef
    python scripts/sizing_frontier.py --report reports/backtest_124d15ef --sims 20000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RISK_MODEL

CONTRACT_SIZE = RISK_MODEL.get("contract_size", 100)
LOT_STEP = RISK_MODEL.get("lot_step", 0.01)
MAX_LOT = RISK_MODEL.get("max_lot", 1.0)
INITIAL_CAPITAL = 500.0
RUIN_EQUITY = INITIAL_CAPITAL * 0.20          # 80% loss of initial = practical ruin
TARGET_MONTH1 = 3000.0
TARGET_18MO_MULT = 15.0                       # user's benchmark trajectory


def load_trades(report_dir: str):
    with open(Path(report_dir) / "results.json") as f:
        results = json.load(f)
    trades = results["trades"]
    samples = []
    for t in trades:
        entry = t.get("actual_fill_price") or t.get("entry_price")
        sl = t.get("original_sl") or t.get("sl_price")
        stop_dist = abs(float(entry) - float(sl))
        if stop_dist <= 0:
            continue
        samples.append((
            float(t["r_multiple"]),
            stop_dist,
            float(t.get("risk_pct_used") or 0.003),
        ))
    return samples, results


def simulate(samples, min_lot, n_sims, rng, trades_per_month):
    n = len(samples)
    arr = np.array(samples)  # columns: r, stop_dist, risk_pct
    month1_trades = max(1, int(round(trades_per_month)))

    finals = np.empty(n_sims)
    ruined = np.zeros(n_sims, dtype=bool)
    hit_3k_month1 = np.zeros(n_sims, dtype=bool)
    hit_3k_ever = np.zeros(n_sims, dtype=bool)
    hit_15x = np.zeros(n_sims, dtype=bool)
    max_dds = np.empty(n_sims)

    for s in range(n_sims):
        idx = rng.integers(0, n, size=n)
        seq = arr[idx]
        equity = INITIAL_CAPITAL
        peak = equity
        max_dd = 0.0
        for i in range(n):
            r, stop_dist, risk_pct = seq[i]
            risk_per_lot = stop_dist * CONTRACT_SIZE
            lots = (equity * risk_pct) / risk_per_lot
            lots = np.floor(lots / LOT_STEP) * LOT_STEP
            lots = min(max(lots, min_lot), MAX_LOT)
            pnl = r * risk_per_lot * lots
            equity += pnl
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 1.0
            if dd > max_dd:
                max_dd = dd
            if equity >= TARGET_MONTH1:
                hit_3k_ever[s] = True
                if i < month1_trades:
                    hit_3k_month1[s] = True
            if equity <= RUIN_EQUITY:
                ruined[s] = True
                break
        finals[s] = equity
        max_dds[s] = max_dd

    mult = finals / INITIAL_CAPITAL
    hit_15x = mult >= TARGET_18MO_MULT
    return {
        "median_mult": float(np.median(mult)),
        "p5_mult": float(np.percentile(mult, 5)),
        "p95_mult": float(np.percentile(mult, 95)),
        "p_ruin": float(ruined.mean()),
        "p_3k_month1": float(hit_3k_month1.mean()),
        "p_3k_ever": float(hit_3k_ever.mean()),
        "p_15x": float(hit_15x.mean()),
        "median_max_dd": float(np.median(max_dds)),
        "p90_max_dd": float(np.percentile(max_dds, 90)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="Report dir with results.json")
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--floors", type=str, default="0.01,0.02,0.03")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    samples, results = load_trades(args.report)
    n = len(samples)
    months = 18.0
    tpm = n / months
    rng = np.random.default_rng(args.seed)

    print(f"Loaded {n} trades from {args.report} (~{tpm:.1f}/month)")
    print(f"Bootstrap: {args.sims} sims x {n} trades, ruin = equity <= ${RUIN_EQUITY:.0f}")
    print()
    hdr = (f"{'floor':>6} | {'median 18mo':>11} | {'5th pct':>8} | {'95th pct':>8} | "
           f"{'P(ruin)':>8} | {'P($3k m1)':>9} | {'P($3k ever)':>11} | {'P(15x)':>7} | "
           f"{'med maxDD':>9} | {'p90 maxDD':>9}")
    print(hdr)
    print("-" * len(hdr))
    for f_str in args.floors.split(","):
        floor = float(f_str)
        m = simulate(samples, floor, args.sims, rng, tpm)
        print(f"{floor:>6.2f} | {m['median_mult']:>10.2f}x | {m['p5_mult']:>7.2f}x | "
              f"{m['p95_mult']:>7.2f}x | {m['p_ruin']:>7.1%} | {m['p_3k_month1']:>8.1%} | "
              f"{m['p_3k_ever']:>10.1%} | {m['p_15x']:>6.1%} | "
              f"{m['median_max_dd']:>8.1%} | {m['p90_max_dd']:>8.1%}")
    print()
    print("floor = minimum lot size; 'P($3k m1)' = probability equity reaches $3,000")
    print(f"within the first month (~{max(1, int(round(tpm)))} trades). 18mo horizon = full resample.")


if __name__ == "__main__":
    main()
