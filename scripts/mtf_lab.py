"""
MTF research lab CLI — screen strategies across timeframes with
engine-parity costs, 70/30 train/OOS split, $500 equity replay.

Usage:
    python scripts/mtf_lab.py --strategies all --timeframes M5,M15,M30,H1
    python scripts/mtf_lab.py --strategies rsi2_trend --grid
    python scripts/mtf_lab.py --strategies asian_breakout --spread-points 45
"""
import argparse
import itertools
import sys
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research import mtf
from src.research.lab import CostModel, simulate
from src.research.sizing import replay_equity
from src.research.report import (HDR, format_row, summarize,
                                 trades_to_records, write_results)
from src.research.strategies import load_registry
from src.research.strategies.base import MTFContext


def param_sets(module, use_grid: bool):
    if not use_grid or not getattr(module, "PARAM_GRID", None):
        return [dict(module.DEFAULTS)]
    keys = list(module.PARAM_GRID.keys())
    sets = []
    for combo in itertools.product(*(module.PARAM_GRID[k] for k in keys)):
        p = dict(module.DEFAULTS)
        p.update(dict(zip(keys, combo)))
        sets.append(p)
    return sets


def months_between(a: pd.Timestamp, b: pd.Timestamp) -> float:
    return max((b - a).total_seconds() / (30.44 * 86400), 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default="all")
    ap.add_argument("--timeframes", default="M5,M15,M30,H1")
    ap.add_argument("--split", type=float, default=0.70)
    ap.add_argument("--grid", action="store_true", help="run PARAM_GRID (survivors only)")
    ap.add_argument("--oos", action="store_true",
                    help="ALSO print OOS rows (single look per survivor!)")
    ap.add_argument("--spread-points", type=float, default=None,
                    help="spread sensitivity override (e.g. 45 = $0.45/oz)")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default="reports/lab")
    ap.add_argument("--no-json", action="store_true")
    args = ap.parse_args()

    registry = load_registry()
    if args.strategies != "all":
        wanted = set(args.strategies.split(","))
        registry = {k: v for k, v in registry.items() if k in wanted}
        missing = wanted - set(registry)
        if missing:
            print(f"WARNING: strategies not found: {sorted(missing)}")
    if not registry:
        print("No strategies loaded.")
        return

    tfs = args.timeframes.split(",")
    m5 = mtf.load_m5(args.start, args.end)
    boundary = mtf.split_boundary(m5, args.split)
    t0, t1 = m5["timestamp"].iloc[0], m5["timestamp"].iloc[-1]
    costs = CostModel.from_config(spread_points=args.spread_points)
    run_id = uuid.uuid4().hex[:8]

    print(f"Lab run {run_id}: {len(m5)} M5 bars {t0} -> {t1}")
    print(f"Train/OOS boundary: {boundary}  |  spread ${costs.spread_usd_oz:.2f}/oz, "
          f"comm ${costs.commission_usd_oz:.2f}/oz, slip {costs.slippage_atr_mult:.0%} ATR, "
          f"swap ${costs.swap_usd_oz_per_night:.2f}/oz/night")
    print()
    print(HDR)
    print("-" * len(HDR))

    frames = {}
    summary_rows, trade_records = [], []

    for name, module in sorted(registry.items()):
        run_tfs = [tf for tf in tfs if tf in module.TIMEFRAMES]
        for tf in run_tfs:
            if tf not in frames:
                frames[tf] = mtf.prepare_frame(m5, tf)
            df = frames[tf]
            htf = {}
            for h in getattr(module, "HTF_NEEDS", []):
                ind = getattr(module, "HTF_INDICATORS", {}).get(h)
                htf[h] = mtf.align_htf(df, m5, h, ind)
            ctx = MTFContext(tf=tf, df=df, htf=htf)

            for params in param_sets(module, args.grid):
                result = module.generate_signals(ctx, params)
                signals, extras = result if isinstance(result, tuple) else (result, {})
                trades, stats = simulate(ctx, signals, costs, **extras)

                if trades.empty:
                    print(f"{name:<22} {tf:>4}   ZERO TRADES "
                          f"(signals={stats['signals']}, unfilled={stats['unfilled']}, "
                          f"busy={stats['skipped_busy']})")
                    continue

                is_train = trades["entry_time"] < boundary
                splits = [("train", trades[is_train],
                           months_between(t0, boundary))]
                if args.oos:
                    splits.append(("oos", trades[~is_train],
                                   months_between(boundary, t1)))

                diff = {k: v for k, v in params.items()
                        if module.DEFAULTS.get(k) != v}
                suffix = f"  {diff}" if diff else ""
                for split_name, tr, months in splits:
                    replay = replay_equity(tr)
                    s = summarize(tr, months, replay)
                    s.update({"strategy": name, "tf": tf, "split": split_name,
                              "params": params, "fill_stats": stats})
                    summary_rows.append(s)
                    print(format_row(name, tf, split_name, s) + suffix)
                    if split_name == "train":
                        trade_records.extend(
                            trades_to_records(tr, name, tf, params))
                    elif args.oos:
                        trade_records.extend(
                            trades_to_records(tr, name, tf, params))

    if not args.no_json and summary_rows:
        out = write_results(Path(args.out) / f"lab_{run_id}", run_id, boundary,
                            costs, summary_rows, trade_records)
        print(f"\nresults: {out}")


if __name__ == "__main__":
    main()
