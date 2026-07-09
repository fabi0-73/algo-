"""
Conditional pattern mining: event x context cells, TRAIN WINDOW ONLY.

The OOS window is never loaded here — survivors are promoted by hand into
src/research/strategies/ modules and face the existing gauntlet
(mtf_lab.py -> --oos single look -> monte_carlo.py). Guardrails live in code,
not in the operator: fixed horizon set, min sample size after declustering,
effect floor above round-trip cost, BH-FDR over the full grid, sign
consistency across both train halves.

Usage:
    python scripts/mine_patterns.py                        # full grid, depth 1
    python scripts/mine_patterns.py --events sweep_low --contexts session,h1_trend
    python scripts/mine_patterns.py --depth 2 --min-n 400  # pairs (expensive)
"""
import argparse
import itertools
import json
import sys
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research import mtf
from src.research.context import CONTEXT_COLUMNS, context_frame
from src.research.events import EVENT_REGISTRY, HORIZONS, run_all
from src.research.forward import (baseline_pool, cost_in_atr, day_ids,
                                  forward_outcomes, tod_bucket)
from src.research.lab import CostModel
from src.research.stats import bh_fdr
from src.research.study import directional_indices, evaluate_cell, sign_consistent

HDR = (f"{'event':<20} {'dir':>3} {'h':>3} {'context':<28} {'n':>5} "
       f"{'excess':>7} {'p_adj':>7} {'netR':>7} {'cost':>6}")


def context_cells(ctx, columns, depth):
    """Yield (label, mask) for single context values and, at depth 2, pairs
    from different columns. 'na' levels are never mined."""
    singles = []
    for col in columns:
        for val in sorted(ctx[col].dropna().unique()):
            if val == "na":
                continue
            singles.append((f"{col}={val}", (ctx[col] == val).to_numpy()))
    yield from singles
    if depth >= 2:
        for (la, ma), (lb, mb) in itertools.combinations(singles, 2):
            if la.split("=")[0] != lb.split("=")[0]:
                yield f"{la} & {lb}", ma & mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="all")
    ap.add_argument("--contexts", default="all",
                    help=f"comma list from {CONTEXT_COLUMNS}")
    ap.add_argument("--depth", type=int, default=1, choices=(1, 2))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--split", type=float, default=0.70)
    ap.add_argument("--min-n", type=int, default=300)
    ap.add_argument("--min-net-r", type=float, default=0.05)
    ap.add_argument("--cost-mult", type=float, default=1.5,
                    help="gross excess must exceed this multiple of cost")
    ap.add_argument("--fdr-q", type=float, default=0.10)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--spread-points", type=float, default=None)
    ap.add_argument("--out", default="reports/research")
    ap.add_argument("--no-json", action="store_true")
    args = ap.parse_args()

    names = list(EVENT_REGISTRY) if args.events == "all" else args.events.split(",")
    columns = CONTEXT_COLUMNS if args.contexts == "all" else args.contexts.split(",")

    # Train slice only. Detectors/context are causal, so computing them on
    # the truncated frame equals computing on the full frame then masking —
    # and this way OOS bars are never even in memory.
    m5_full = mtf.load_m5(args.start, args.end, cache_path=args.cache)
    boundary = mtf.split_boundary(m5_full, args.split)
    m5 = m5_full[m5_full["timestamp"] < boundary].reset_index(drop=True)
    del m5_full
    df = mtf.prepare_frame(m5, "M5")
    costs = CostModel.from_config(spread_points=args.spread_points)
    run_id = uuid.uuid4().hex[:8]

    fwd = forward_outcomes(df)
    cost_atr = cost_in_atr(df, costs).to_numpy(float)
    buckets = tod_bucket(df["timestamp"]).to_numpy()
    days = day_ids(df["timestamp"]).to_numpy()
    events = run_all(df, names)
    ctx = context_frame(df, m5)
    half_pos = len(df) // 2

    cells = list(context_cells(ctx, columns, args.depth))
    print(f"Mining run {run_id}: {len(df)} TRAIN bars (< {boundary}), "
          f"{len(names)} events x {len(cells)} contexts x {len(HORIZONS)} horizons")

    rows = []
    n_skipped = 0
    fr_by_h = {h: fwd[f"fr_{h}"].to_numpy(float) for h in HORIZONS}
    for label, cmask in cells:
        pools = {h: baseline_pool(fwd[cmask], tod_bucket(df.loc[cmask, "timestamp"]), h)
                 for h in HORIZONS}
        for name in names:
            by_dir = directional_indices(events[name], cmask)
            for d, idx in by_dir.items():
                for h in HORIZONS:
                    if pools[h]["values"].size == 0:
                        continue
                    cell = evaluate_cell(
                        idx, d, h, fr_by_h[h], cost_atr, days, buckets,
                        pools[h]["values"], pools[h]["buckets"],
                        args.min_n, args.n_boot, args.n_perm, args.seed)
                    if cell["skipped"]:
                        n_skipped += 1
                        continue
                    sign = d if d != 0 else 1
                    eidx = cell.pop("event_idx")
                    cell["consistent"] = sign_consistent(
                        eidx, sign * fr_by_h[h][eidx], half_pos)
                    cell.update({"event": name, "direction": d, "horizon": h,
                                 "context": label})
                    rows.append(cell)

    pvals = np.array([r["p_value"] for r in rows]) if rows else np.array([])
    fdr = bh_fdr(pvals, q=args.fdr_q)
    for r, adj, rej in zip(rows, fdr["p_adj"], fdr["reject"]):
        r["p_adj"] = float(adj) if not np.isnan(adj) else None
        r["fdr_pass"] = bool(rej)
        r["effect_pass"] = (r["net_r_mean"] >= args.min_net_r
                            and r["excess"] >= args.cost_mult * r["cost_mean"])
        r["candidate"] = r["fdr_pass"] and r["effect_pass"] and r["consistent"]

    candidates = [r for r in rows if r["candidate"]]
    candidates.sort(key=lambda r: -r["net_r_mean"])
    n_fdr = sum(1 for r in rows if r["fdr_pass"])
    print(f"\n{fdr['n_tests']} tests run ({n_skipped} cells below min_n dropped); "
          f"{n_fdr} pass FDR q={args.fdr_q} "
          f"(expect ~{args.fdr_q * max(n_fdr, 1):.1f} false); "
          f"{len(candidates)} clear ALL gates "
          f"(netR>={args.min_net_r}, excess>={args.cost_mult}x cost, sign-consistent)")
    if candidates:
        print(HDR)
        print("-" * len(HDR))
        for r in candidates[:40]:
            print(f"{r['event']:<20} {r['direction']:>3} {r['horizon']:>3} "
                  f"{r['context']:<28} {r['n']:>5} {r['excess']:>7.3f} "
                  f"{r['p_adj']:>7.4f} {r['net_r_mean']:>7.3f} {r['cost_mean']:>6.3f}")
        print("\nnext: promote to src/research/strategies/<name>.py and run "
              "scripts/mtf_lab.py (then --oos ONCE, then monte_carlo.py)")

    if not args.no_json:
        out_dir = Path(args.out) / f"mining_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {"run_id": run_id, "boundary": str(boundary),
                "train_only": True, "horizons": list(HORIZONS),
                "min_n": args.min_n, "min_net_r": args.min_net_r,
                "cost_mult": args.cost_mult, "fdr_q": args.fdr_q,
                "seed": args.seed, "n_tests": int(fdr["n_tests"]),
                "n_skipped_min_n": n_skipped,
                "candidates": candidates, "all_rows": rows}
        with open(out_dir / "candidates.json", "w") as f:
            json.dump(meta, f, indent=2, default=float)
        print(f"\ncandidates: {out_dir / 'candidates.json'}")


if __name__ == "__main__":
    main()
