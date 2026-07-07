"""
SMT bucket analysis: annotate the engine's AMD trades with SMT-divergence
present/absent at their manipulation, then compare the two buckets'
expectancy / win-rate / profit-factor. Decisive screen BEFORE any engine work.

Usage:
    python scripts/smt_bucket.py --report reports/backtest_3f3ea9d1 --corr xagusd
    python scripts/smt_bucket.py --report reports/backtest_3f3ea9d1 --corr eurusd
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.smt import smt_divergence, load_corr


def bucket_stats(rows):
    if not rows:
        return None
    r = np.array([x["r"] for x in rows])
    net = np.array([x["net"] for x in rows])
    wins = r[r > 0]
    losses = r[r <= 0]
    pf = (net[net > 0].sum() / abs(net[net < 0].sum())
          if (net < 0).any() else float("inf"))
    return {"n": len(rows), "wr": (r > 0).mean(), "exp_r": r.mean(),
            "avg_net": net.mean(), "pf": pf, "total_net": net.sum()}


def line(label, s):
    if not s:
        return f"  {label:<12} (empty)"
    return (f"  {label:<12} n={s['n']:>3}  WR {s['wr']:>5.1%}  "
            f"expR {s['exp_r']:>+.3f}  PF {min(s['pf'],9.99):>4.2f}  "
            f"avg${s['avg_net']:>+6.2f}  tot${s['total_net']:>+8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="reports/backtest_3f3ea9d1")
    ap.add_argument("--corr", default="xagusd", help="cached corr asset name")
    ap.add_argument("--kind", default="positive", choices=["positive", "inverse"])
    args = ap.parse_args()

    corr = load_corr(args.corr)
    if corr is None:
        print(f"No cached data for {args.corr}")
        return
    c0, c1 = corr["timestamp"].iloc[0], corr["timestamp"].iloc[-1]
    print(f"Correlated: {args.corr} ({args.kind})  {c0} -> {c1}  {len(corr)} bars")

    trades = json.load(open(Path(args.report) / "results.json"))["trades"]
    amd = [t for t in trades if (t.get("entry_model") or "AMD") == "AMD"]
    print(f"AMD trades in report: {len(amd)}")

    smt_pos, smt_neg, skipped = [], [], 0
    for t in amd:
        mt = pd.Timestamp(t["entry_time"])
        if not (c0 <= mt <= c1):
            skipped += 1
            continue
        present, _ = smt_divergence(corr, mt, t["direction"], args.kind)
        if present is None:
            skipped += 1
            continue
        rec = {"r": float(t["r_multiple"]), "net": float(t["net_pnl"])}
        (smt_pos if present else smt_neg).append(rec)

    covered = len(smt_pos) + len(smt_neg)
    print(f"Covered by corr window: {covered}  (skipped {skipped} outside coverage)\n")
    print("Bucket comparison (SMT divergence present vs absent at the sweep):")
    sp, sn = bucket_stats(smt_pos), bucket_stats(smt_neg)
    print(line("SMT PRESENT", sp))
    print(line("SMT ABSENT", sn))
    print(line("ALL COVERED", bucket_stats(smt_pos + smt_neg)))

    if sp and sn:
        d_exp = sp["exp_r"] - sn["exp_r"]
        d_wr = sp["wr"] - sn["wr"]
        print(f"\n  Delta (present - absent): expR {d_exp:+.3f}  WR {d_wr:+.1%}")
        gate = (d_exp >= 0.15 or (d_wr >= 0.10 and sp["exp_r"] > sn["exp_r"])) \
            and sp["n"] >= 40 and sn["n"] >= 40
        print(f"  Pre-registered gate (>=0.15R or >=10% WR sep, >=40/bucket): "
              f"{'PASS -> build it' if gate else 'FAIL -> kill'}")


if __name__ == "__main__":
    main()
