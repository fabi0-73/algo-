"""
Lab reporting: per strategy x TF x split summary rows, console table,
and results.json compatible with portfolio merging (entry/exit timestamps
in the same string format as the engine's results.json).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

TS_FMT = "%Y-%m-%d %H:%M:%S"


def summarize(trades: pd.DataFrame, months: float, replay: dict = None) -> dict:
    """Metric conventions follow src/backtest/metrics.py where comparable:
    winner = r_net > 0; PF on net R sums; expectancy in net R."""
    if trades.empty:
        return {"trades": 0, "tr_mo": 0.0, "wr": 0.0, "avg_w": 0.0, "avg_l": 0.0,
                "exp_r_net": 0.0, "exp_r_price": 0.0, "r_mo": 0.0, "pf": 0.0,
                "max_dd": 0.0, "final_eq": 500.0}
    r = trades["r_net"].to_numpy(float)
    wins = r[r > 0]
    losses = r[r <= 0]
    pf = float(wins.sum() / abs(losses.sum())) if losses.size and abs(losses.sum()) > 0 else float("inf")
    return {
        "trades": int(len(trades)),
        "tr_mo": len(trades) / months if months > 0 else 0.0,
        "wr": float(len(wins) / len(r)),
        "avg_w": float(wins.mean()) if wins.size else 0.0,
        "avg_l": float(losses.mean()) if losses.size else 0.0,
        "exp_r_net": float(r.mean()),
        "exp_r_price": float(trades["r_price"].mean()),
        "r_mo": float(r.sum() / months) if months > 0 else 0.0,
        "pf": pf,
        "max_dd": replay["max_dd_pct"] if replay else 0.0,
        "final_eq": replay["final_equity"] if replay else 0.0,
    }


ROW_FMT = ("{name:<22} {tf:>4} {split:>6} {trades:>5} {tr_mo:>6.1f} {wr:>6.1%} "
           "{avg_w:>6.2f} {avg_l:>6.2f} {exp:>7.3f} {r_mo:>6.2f} {pf:>5.2f} "
           "{dd:>6.1%} {eq:>7.0f}")
HDR = ("{:<22} {:>4} {:>6} {:>5} {:>6} {:>6} {:>6} {:>6} {:>7} {:>6} {:>5} "
       "{:>6} {:>7}").format("strategy", "tf", "split", "n", "tr/mo", "WR",
                             "avgW", "avgL", "expRn", "R/mo", "PF", "maxDD", "eq$")


def format_row(name: str, tf: str, split: str, s: dict) -> str:
    return ROW_FMT.format(name=name, tf=tf, split=split, trades=s["trades"],
                          tr_mo=s["tr_mo"], wr=s["wr"], avg_w=s["avg_w"],
                          avg_l=s["avg_l"], exp=s["exp_r_net"], r_mo=s["r_mo"],
                          pf=min(s["pf"], 99.99), dd=s["max_dd"], eq=s["final_eq"])


def trades_to_records(trades: pd.DataFrame, strategy: str, tf: str, params: dict) -> list:
    out = []
    for row in trades.itertuples(index=False):
        out.append({
            "strategy": strategy,
            "tf": tf,
            "params": params,
            "direction": row.direction,
            "signal_time": pd.Timestamp(row.signal_time).strftime(TS_FMT),
            "entry_time": pd.Timestamp(row.entry_time).strftime(TS_FMT),
            "exit_time": pd.Timestamp(row.exit_time).strftime(TS_FMT),
            "entry": round(float(row.entry), 3),
            "exit": round(float(row.exit), 3),
            "sl": round(float(row.sl0), 3),
            "stop_dist": round(float(row.stop_dist), 4),
            "exit_reason": row.exit_reason,
            "bars_held": int(row.bars_held),
            "nights": int(row.nights),
            "r_price": round(float(row.r_price), 4),
            "r_net": round(float(row.r_net), 4),
            "mae_r": round(float(row.mae_r), 3),
            "mfe_r": round(float(row.mfe_r), 3),
            "tag": int(row.tag),
        })
    return out


def write_results(out_dir: Path, run_id: str, boundary_ts, cost_model,
                  summary_rows: list, trade_records: list) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "boundary_ts": str(boundary_ts),
        "cost_model": {k: getattr(cost_model, k) for k in
                       ("spread_usd_oz", "commission_usd_oz", "slippage_atr_mult",
                        "swap_usd_oz_per_night", "contract_size")},
        "summary": summary_rows,
        "trades": trade_records,
    }
    path = out_dir / "results.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=str)
    return path
