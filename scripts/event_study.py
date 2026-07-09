"""
Event study: standalone statistical edge of each AMD atom.

For every (event x direction x horizon) cell: direction-signed forward
returns in ATR units vs a time-of-day-matched unconditional baseline,
day-block bootstrap CI, permutation p-value, BH-FDR across the WHOLE grid.
Train window only by default; --oos is a single look (mtf_lab.py discipline).

Usage:
    python scripts/event_study.py                          # all events, train
    python scripts/event_study.py --events sweep_high,judas --cache data/lab_m5_cache.csv.gz
    python scripts/event_study.py --oos                    # ONE look at OOS
"""
import argparse
import json
import sys
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research import mtf
from src.research.events import EVENT_REGISTRY, HORIZONS, SESSIONS, run_all
from src.research.forward import (baseline_pool, cost_in_atr, day_ids,
                                  forward_outcomes, tod_bucket)
from src.research.lab import CostModel
from src.research.stats import bh_fdr
from src.research.strategies.base import session_mask
from src.research.study import directional_indices, evaluate_cell

HDR = (f"{'event':<20} {'dir':>3} {'h':>3} {'split':<5} {'n':>5} "
       f"{'mean':>7} {'excess':>7} {'ci_lo':>7} {'ci_hi':>7} "
       f"{'p':>7} {'p_adj':>7} {'netR':>7} {'fdr':>4}")


def session_split(df, idx, vals):
    out = {}
    ts = df["timestamp"].iloc[idx]
    for name, (s, e) in SESSIONS.items():
        m = session_mask(ts, s, e).to_numpy()
        out[name] = {"n": int(m.sum()),
                     "mean": float(vals[m].mean()) if m.any() else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="all")
    ap.add_argument("--timeframe", default="M5",
                    help="entry TF (M5/M15/M30/H1/D1); HORIZONS stay in bars "
                         "of this TF — declare the TF upfront, never per-event")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--split", type=float, default=0.70)
    ap.add_argument("--min-n", type=int, default=300)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fdr-q", type=float, default=0.10)
    ap.add_argument("--oos", action="store_true",
                    help="ALSO evaluate the OOS window (single look!)")
    ap.add_argument("--spread-points", type=float, default=None)
    ap.add_argument("--commission-usd-oz", type=float, default=None,
                    help="override $/oz commission (XAGUSD: 0.007)")
    ap.add_argument("--out", default="reports/research")
    ap.add_argument("--no-json", action="store_true")
    args = ap.parse_args()

    names = list(EVENT_REGISTRY) if args.events == "all" else args.events.split(",")
    unknown = set(names) - set(EVENT_REGISTRY)
    if unknown:
        print(f"unknown events: {sorted(unknown)}")
        sys.exit(1)

    m5 = mtf.load_m5(args.start, args.end, cache_path=args.cache)
    df = mtf.prepare_frame(m5, args.timeframe)
    boundary = mtf.split_boundary(m5, args.split)
    costs = CostModel.from_config(spread_points=args.spread_points,
                                  commission_usd_oz=args.commission_usd_oz)
    run_id = uuid.uuid4().hex[:8]

    fwd = forward_outcomes(df)
    cost_atr = cost_in_atr(df, costs).to_numpy(float)
    buckets = tod_bucket(df["timestamp"]).to_numpy()
    days = day_ids(df["timestamp"]).to_numpy()
    train_mask = (df["timestamp"] < boundary).to_numpy()
    events = run_all(df, names)

    splits = [("train", train_mask)]
    if args.oos:
        splits.append(("oos", ~train_mask))

    print(f"Event study {run_id}: {len(df)} {args.timeframe} bars, "
          f"boundary {boundary}, {len(names)} events x {len(HORIZONS)} horizons")
    rows = []
    for split_name, mask in splits:
        pools = {h: baseline_pool(fwd[mask], tod_bucket(df.loc[mask, 'timestamp']), h)
                 for h in HORIZONS}
        for name in names:
            by_dir = directional_indices(events[name], mask)
            for d, idx in by_dir.items():
                for h in HORIZONS:
                    fr = fwd[f"fr_{h}"].to_numpy(float)
                    cell = evaluate_cell(
                        idx, d, h, fr, cost_atr, days, buckets,
                        pools[h]["values"], pools[h]["buckets"],
                        args.min_n, args.n_boot, args.n_perm, args.seed)
                    cell.update({"event": name, "direction": d, "horizon": h,
                                 "split": split_name})
                    if not cell["skipped"]:
                        sign = d if d != 0 else 1
                        cell["by_session"] = session_split(
                            df, cell["event_idx"], sign * fr[cell["event_idx"]])
                        del cell["event_idx"]
                    rows.append(cell)

    # FDR per split, across the full grid — the honest multiple-testing unit.
    for split_name, _ in splits:
        srows = [r for r in rows if r["split"] == split_name and not r["skipped"]]
        pvals = np.array([r["p_value"] for r in srows])
        fdr = bh_fdr(pvals, q=args.fdr_q)
        for r, adj, rej in zip(srows, fdr["p_adj"], fdr["reject"]):
            r["p_adj"] = float(adj) if not np.isnan(adj) else None
            r["fdr_pass"] = bool(rej)
        n_tested = fdr["n_tests"]
        n_skip = sum(1 for r in rows if r["split"] == split_name and r["skipped"])
        print(f"\n[{split_name}] {n_tested} tests (FDR q={args.fdr_q}, "
              f"BH-corrected), {n_skip} cells below min_n={args.min_n} "
              f"after declustering")
        print(HDR)
        print("-" * len(HDR))
        for r in sorted(srows, key=lambda r: (not r["fdr_pass"], r["p_value"])):
            print(f"{r['event']:<20} {r['direction']:>3} {r['horizon']:>3} "
                  f"{r['split']:<5} {r['n']:>5} {r['mean']:>7.3f} "
                  f"{r['excess']:>7.3f} {r['ci_lo']:>7.3f} {r['ci_hi']:>7.3f} "
                  f"{r['p_value']:>7.4f} {r['p_adj']:>7.4f} "
                  f"{r['net_r_mean']:>7.3f} {'PASS' if r['fdr_pass'] else '':>4}")

    if not args.no_json:
        out_dir = Path(args.out) / f"events_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {"run_id": run_id, "boundary": str(boundary),
                "horizons": list(HORIZONS), "min_n": args.min_n,
                "fdr_q": args.fdr_q, "seed": args.seed,
                "n_boot": args.n_boot, "n_perm": args.n_perm,
                "spread_usd_oz": costs.spread_usd_oz,
                "events": names, "rows": rows}
        with open(out_dir / "edge_table.json", "w") as f:
            json.dump(meta, f, indent=2, default=float)
        print(f"\nedge table: {out_dir / 'edge_table.json'}")


if __name__ == "__main__":
    main()
