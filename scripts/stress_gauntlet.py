"""Severe-cost / DSR / fresh-start-ruin / concentration gauntlet.

Ports the four kill-gates from the quant-trading-lab methodology audit
(2026-07-23) onto a saved backtest report's trade list:

  1. Execution-cost stress tiers, RE-DERIVED for our execution model
     (2026-07-23 user decision): their tiers charge quote-drift delay on
     BOTH sides because their bot uses market orders. Our entries are
     resting limits at the retest (LIMIT_AT_RETEST) — a limit doesn't chase
     the quote; its failure mode is a missed fill, which the engine models
     natively. Only our exits (SL/BE/trail = stop-market) take delay. So:
     delay charged ONCE, spread + $0.07 commission unchanged:
         base:    11 + 32  pts + $0.07 = $0.50  (== engine's measured cost)
         adverse: 12 + 129 pts + $0.07 = $1.48  -> delta +$0.98/0.01 lot
         severe:  23 + 284 pts + $0.07 = $3.14  -> delta +$2.64/0.01 lot
     --market-order-tiers restores their original both-sides deltas
     (+$1.95 / +$5.16) for cross-lab comparison.
  2. Deflated Sharpe (Bailey & Lopez de Prado): PSR against the expected
     max Sharpe of N independent trials; reported across an N grid because
     the honest trial count is uncertain.
  3. Rolling fresh-start windows: every monthly-start 12-month window is
     replayed from a fresh $500 at fixed 0.01 lots (the floor a fresh $500
     account actually trades); worst min-equity, worst DD, ruin count.
  4. Single-year concentration of positive profit (flag > 60%).

Usage:
    python scripts/stress_gauntlet.py --report reports/backtest_187cca99 [--start-balance 500]
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np

# Limit-entry tiers (delay charged once — our execution model). Deltas are
# tier-minus-base within the same schedule, applied on top of the engine's
# already-modeled costs (which match the $0.50 limit-entry base tier).
TIERS_LIMIT_ENTRY = {
    "base(model)": 0.0,
    "adverse": 0.98,
    "severe": 2.64,
}
# Their original market-order tiers (delay charged both sides), kept for
# cross-lab comparison via --market-order-tiers.
TIERS_MARKET_ORDER = {
    "base(model)": 0.0,
    "adverse": 2.77 - 0.82,
    "severe": 5.98 - 0.82,
}
TIERS = TIERS_LIMIT_ENTRY
EULER_GAMMA = 0.5772156649015329
_N = NormalDist()


def load_trades(report_dir: Path):
    j = json.loads((report_dir / "results.json").read_text())
    trades = [t for t in j["trades"] if t.get("r_multiple") is not None]
    return j, trades


def stressed_pnls(trades, extra_per_001: float):
    """Net P&L per trade after deducting extra cost scaled by lots."""
    out = []
    for t in trades:
        lots = t.get("position_size") or 0.01
        out.append(t["net_pnl"] - extra_per_001 * (lots / 0.01))
    return np.array(out)


def pnl_per_001(trades, extra_per_001: float):
    """Each trade normalised to fixed 0.01-lot execution, cost-stressed."""
    out = []
    for t in trades:
        lots = t.get("position_size") or 0.01
        out.append(t["net_pnl"] / (lots / 0.01) - extra_per_001)
    return np.array(out)


def basic_stats(pnls: np.ndarray):
    gross_win = pnls[pnls > 0].sum()
    gross_loss = -pnls[pnls < 0].sum()
    return {
        "net": round(float(pnls.sum()), 2),
        "pf": round(float(gross_win / gross_loss), 3) if gross_loss > 0 else float("inf"),
        "win_rate": round(float((pnls > 0).mean() * 100), 1),
        "n": int(len(pnls)),
    }


def max_drawdown(equity: np.ndarray):
    peak = np.maximum.accumulate(equity)
    return float((peak - equity).max()), float(((peak - equity) / peak).max())


def monthly_series(trades, pnls):
    buckets = defaultdict(float)
    for t, p in zip(trades, pnls):
        buckets[t["exit_time"][:7]] += p
    months = sorted(buckets)
    return months, np.array([buckets[m] for m in months])


def deflated_sharpe(monthly: np.ndarray, n_trials_grid):
    """PSR vs expected-max-SR threshold (Bailey & Lopez de Prado)."""
    n = len(monthly)
    if n < 6 or monthly.std(ddof=1) == 0:
        return {"error": f"too few months (n={n})"}
    sr = monthly.mean() / monthly.std(ddof=1)          # monthly, non-annualised
    diffs = monthly - monthly.mean()
    m2 = (diffs ** 2).mean()
    skew = (diffs ** 3).mean() / m2 ** 1.5
    kurt = (diffs ** 4).mean() / m2 ** 2               # Pearson (normal = 3)
    var_sr = (1 + 0.5 * sr * sr) / n
    out = {"sr_monthly": round(sr, 4), "sr_annualized": round(sr * math.sqrt(12), 3),
           "skew": round(skew, 3), "kurtosis": round(kurt, 3), "n_months": n, "dsr_by_trials": {}}
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr * sr
    if denom <= 0:
        out["error"] = "PSR denominator <= 0 (extreme higher moments)"
        return out
    for N in n_trials_grid:
        sr0 = math.sqrt(var_sr) * ((1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / N)
                                   + EULER_GAMMA * _N.inv_cdf(1 - 1 / (N * math.e)))
        z = (sr - sr0) * math.sqrt(n - 1) / math.sqrt(denom)
        out["dsr_by_trials"][N] = round(_N.cdf(z), 4)
    return out


def fresh_start_windows(trades, per001: np.ndarray, start_balance: float):
    """Fixed-0.01-lot replay of every monthly-start 12-month window."""
    months = sorted({t["entry_time"][:7] for t in trades})
    results = []
    for i, m0 in enumerate(months):
        # 12-calendar-month window
        y, mo = int(m0[:4]), int(m0[5:7])
        end = f"{y + (mo + 11) // 12}-{(mo + 11) % 12 + 1:02d}"
        window = [(t, p) for t, p in zip(trades, per001)
                  if m0 <= t["entry_time"][:7] < end]
        if len(window) < 5:
            continue
        eq = start_balance + np.cumsum([p for _, p in window])
        eq = np.concatenate([[start_balance], eq])
        dd_usd, _ = max_drawdown(eq)
        results.append({"start": m0, "trades": len(window),
                        "final": round(float(eq[-1]), 2),
                        "min_equity": round(float(eq.min()), 2),
                        "max_dd_usd": round(dd_usd, 2)})
    if not results:
        return {"windows": 0}
    ruin = sum(1 for r in results if r["min_equity"] <= 0)
    return {
        "windows": len(results),
        "ruin_count": ruin,
        "worst_min_equity": min(r["min_equity"] for r in results),
        "worst_dd_usd": max(r["max_dd_usd"] for r in results),
        "negative_windows": sum(1 for r in results if r["final"] < start_balance),
        "worst_window": min(results, key=lambda r: r["min_equity"]),
    }


def concentration(trades, pnls):
    per_year = defaultdict(float)
    for t, p in zip(trades, pnls):
        per_year[t["exit_time"][:4]] += p
    pos = {y: v for y, v in per_year.items() if v > 0}
    total_pos = sum(pos.values())
    share = max(pos.values()) / total_pos if total_pos > 0 else float("nan")
    return {"per_year": {y: round(v, 2) for y, v in sorted(per_year.items())},
            "largest_positive_year_share": round(share, 3) if pos else None,
            "flag_gt_60pct": bool(pos and share > 0.60)}


def block_bootstrap_joint(monthly: np.ndarray, start_balance: float,
                          n_boot=10_000, block=3, seed=7):
    """Joint pass: net>0 AND maxDD <= 50% of start. Circular 3-month blocks."""
    rng = np.random.default_rng(seed)
    n = len(monthly)
    if n < 6:
        return {"error": "too few months"}
    n_blocks = math.ceil(n / block)
    hits = 0
    for _ in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        path = monthly[idx][:n]
        eq = start_balance + np.concatenate([[0], np.cumsum(path)])
        dd_usd, _ = max_drawdown(eq)
        if path.sum() > 0 and dd_usd <= 0.5 * start_balance:
            hits += 1
    return {"joint_pass_fraction": round(hits / n_boot, 4), "gate": 0.60}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="reports/backtest_187cca99")
    ap.add_argument("--start-balance", type=float, default=500.0)
    ap.add_argument("--trials-grid", default="10,50,100,300")
    ap.add_argument("--market-order-tiers", action="store_true",
                    help="Use the neighbor lab's both-sides-delay tiers "
                         "(market-order execution) instead of our limit-entry tiers")
    args = ap.parse_args()

    global TIERS
    if args.market_order_tiers:
        TIERS = TIERS_MARKET_ORDER

    report_dir = Path(args.report)
    j, trades = load_trades(report_dir)
    grid = [int(x) for x in args.trials_grid.split(",")]

    print("=" * 68)
    print(f"STRESS GAUNTLET — {report_dir.name}  ({len(trades)} trades)")
    print("=" * 68)

    # Control lane: reproduce recorded base numbers
    base = stressed_pnls(trades, 0.0)
    rec_net, got_net = j["net_pnl_usd"], base.sum()
    ctrl = abs(rec_net - got_net) < 0.01
    print(f"\n[control] recorded net ${rec_net:.2f} vs trade-sum ${got_net:.2f}"
          f"  -> {'OK' if ctrl else 'MISMATCH — do not trust the rest'}")

    print("\n--- 1. COST-STRESS TIERS (actual lot sizes, compounded arc) ---")
    for name, extra in TIERS.items():
        pnls = stressed_pnls(trades, extra)
        s = basic_stats(pnls)
        eq = args.start_balance + np.concatenate([[0], np.cumsum(pnls)])
        dd_usd, dd_pct = max_drawdown(eq)
        print(f"{name:12s} extra=${extra:4.2f}/0.01lot | net ${s['net']:8.2f} | "
              f"PF {s['pf']:5.3f} | WR {s['win_rate']:4.1f}% | "
              f"maxDD ${dd_usd:7.2f} ({dd_pct*100:4.1f}%)")

    print("\n--- 2. DEFLATED SHARPE (monthly, per-0.01-lot stream) ---")
    for name, extra in TIERS.items():
        _, monthly = monthly_series(trades, pnl_per_001(trades, extra))
        d = deflated_sharpe(monthly, grid)
        if "error" in d and "dsr_by_trials" not in d:
            print(f"{name:12s} {d['error']}")
            continue
        dsr = "  ".join(f"N={k}:{v:.3f}" for k, v in d["dsr_by_trials"].items())
        print(f"{name:12s} SRann {d['sr_annualized']:5.2f} skew {d['skew']:6.2f} "
              f"kurt {d['kurtosis']:5.1f} | DSR {dsr}")

    print("\n--- 3. FRESH-START 12-MONTH WINDOWS ($%.0f, fixed 0.01 lot) ---"
          % args.start_balance)
    for name, extra in TIERS.items():
        f = fresh_start_windows(trades, pnl_per_001(trades, extra), args.start_balance)
        if f.get("windows", 0) == 0:
            print(f"{name:12s} no windows")
            continue
        print(f"{name:12s} windows {f['windows']:2d} | ruin {f['ruin_count']} | "
              f"worst min-eq ${f['worst_min_equity']:7.2f} | "
              f"worst DD ${f['worst_dd_usd']:7.2f} | "
              f"losing windows {f['negative_windows']}")

    print("\n--- 4. YEAR CONCENTRATION (stressed tiers) ---")
    for name, extra in TIERS.items():
        c = concentration(trades, stressed_pnls(trades, extra))
        print(f"{name:12s} {c['per_year']} | largest share "
              f"{c['largest_positive_year_share']} | flag>60%: {c['flag_gt_60pct']}")

    print("\n--- 5. BLOCK-BOOTSTRAP JOINT PASS (10k, 3-mo blocks, per-0.01-lot) ---")
    for name, extra in TIERS.items():
        _, monthly = monthly_series(trades, pnl_per_001(trades, extra))
        b = block_bootstrap_joint(monthly, args.start_balance)
        if "error" in b:
            print(f"{name:12s} {b['error']}")
        else:
            print(f"{name:12s} joint-pass {b['joint_pass_fraction']*100:5.1f}% "
                  f"(their gate: >=60%)")

    print("\nDone. Gates are informational — binding tier is a user decision "
          "(we trade 0.01-0.05 lots on $500, not their $100 fixed-lot).")


if __name__ == "__main__":
    main()
