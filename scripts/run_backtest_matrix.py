"""
Run Backtest Matrix Script
Sweeps entry modes and SMC parameters over a 6-month window.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from copy import deepcopy
from datetime import datetime
from itertools import product
import logging
import json

import pandas as pd

from config import STRATEGY, BACKTEST, VALIDATION
from src.data.db import Database
from src.backtest.engine import BacktestEngine
from src.strategy.entry import (
    ENTRY_MODE_RETEST_ONLY,
    ENTRY_MODE_RETEST_WITH_FVG,
    ENTRY_MODE_ORDER_BLOCK,
    ENTRY_MODE_PEAK_LOW,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


ENTRY_MODES = [
    ENTRY_MODE_RETEST_ONLY,
    ENTRY_MODE_RETEST_WITH_FVG,
    ENTRY_MODE_ORDER_BLOCK,
    ENTRY_MODE_PEAK_LOW,
]

FVG_GRID = [0.06, 0.10, 0.14]
OB_BODY_GRID = [0.10, 0.15, 0.20]
OB_DISPLACEMENT_GRID = [1.2, 1.5, 1.8]


def _apply_strategy(base_strategy: dict, overrides: dict) -> None:
    """Reset STRATEGY to base, then apply overrides."""
    STRATEGY.clear()
    STRATEGY.update(base_strategy)
    STRATEGY.update(overrides)


def _load_data(
    symbol: str,
    timeframe: str,
    months: int,
    end_date: str = None,
    allow_short_data: bool = False,
):
    db = Database()
    df = db.get_candles(symbol, timeframe)

    if df.empty:
        logger.error("No candle data found in database. Run fetch_data.py first.")
        return None, None, None

    available_start = df["timestamp"].min()
    available_end = df["timestamp"].max()
    available_months = (available_end - available_start).days / 30

    logger.info(
        "Available data: %s to %s (~%.1f months)",
        available_start,
        available_end,
        available_months,
    )

    if end_date:
        end_ts = pd.to_datetime(end_date)
    else:
        end_ts = available_end

    start_ts = end_ts - pd.DateOffset(months=months)

    if available_months < months and not allow_short_data:
        logger.error(
            "Not enough data for %d months. Fetch more data (e.g., python scripts/fetch_data.py --months %d).",
            months,
            months,
        )
        return None, None, None

    df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)].copy()
    df = df.reset_index(drop=True)

    logger.info("Filtered to %d candles from %s to %s", len(df), start_ts, end_ts)
    return df, start_ts, end_ts


def _run_single(
    df: pd.DataFrame,
    base_strategy: dict,
    overrides: dict,
    stage: str,
    config_id: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    max_trade_duration: int,
):
    _apply_strategy(base_strategy, overrides)

    engine = BacktestEngine(
        initial_capital=BACKTEST["initial_capital"],
        max_trade_duration=max_trade_duration,
    )
    results = engine.run(df, verbose=False)

    confluence = results.get("confluence_stats", {})
    validation = results.get("validation", {})
    funnel = results.get("funnel_stats", {})

    passes_validation = all(validation.values()) if validation else False

    row = {
        "stage": stage,
        "config_id": config_id,
        "backtest_id": results.get("backtest_id"),
        "symbol": STRATEGY["symbol"],
        "timeframe": STRATEGY["timeframe"],
        "start_date": start_ts.isoformat(),
        "end_date": end_ts.isoformat(),
        "entry_mode": STRATEGY["entry_mode"],
        "bos_required": STRATEGY["bos_required"],
        "fvg_min_size_atr_mult": STRATEGY["fvg_min_size_atr_mult"],
        "ob_min_body_atr_mult": STRATEGY["ob_min_body_atr_mult"],
        "ob_displacement_mult": STRATEGY["ob_displacement_mult"],
        "total_trades": results.get("total_trades", 0),
        "win_rate": results.get("win_rate"),
        "expectancy_r": results.get("expectancy_r"),
        "profit_factor": results.get("profit_factor"),
        "max_drawdown_pct": results.get("max_drawdown_pct"),
        "passes_validation": passes_validation,
        "entries_with_fvg": confluence.get("entries_with_fvg", 0),
        "entries_with_ob": confluence.get("entries_with_ob", 0),
        "entries_with_bos": confluence.get("entries_with_bos", 0),
        "avg_confluence_score": confluence.get("avg_confluence_score", 0.0),
        "consolidations_found": funnel.get("consolidations_found", 0),
        "entries_executed": funnel.get("entries_executed", 0),
        "error": results.get("error"),
    }

    return row


def _sort_key(row: dict) -> tuple:
    return (
        -(row.get("expectancy_r") or 0),
        -(row.get("total_trades") or 0),
        (row.get("max_drawdown_pct") or 0),
    )


def _select_top_modes(stage1_rows: list, top_n: int = 2):
    best_by_mode = {}

    for mode in ENTRY_MODES:
        candidates = [r for r in stage1_rows if r["entry_mode"] == mode and not r.get("error")]
        if not candidates:
            continue
        candidates.sort(key=_sort_key)
        best_by_mode[mode] = candidates[0]

    ranked = sorted(best_by_mode.values(), key=_sort_key)
    top_modes = ranked[:top_n]

    return best_by_mode, [r["entry_mode"] for r in top_modes]


def main():
    parser = argparse.ArgumentParser(description="Run AMD+SMC backtest matrix")
    parser.add_argument("--symbol", type=str, default=STRATEGY["symbol"], help="Trading symbol")
    parser.add_argument("--timeframe", type=str, default=STRATEGY["timeframe"], help="Timeframe")
    parser.add_argument("--months", type=int, default=6, help="Months of data to use")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD), default latest")
    parser.add_argument("--allow-short-data", action="store_true", help="Run even if less data")
    parser.add_argument("--max-trade-duration", type=int, default=200, help="Max bars per trade")
    parser.add_argument("--stage1-only", action="store_true", help="Run baseline + stage-1 only")
    parser.add_argument("--baseline-only", action="store_true", help="Run baseline only")
    args = parser.parse_args()

    base_strategy = deepcopy(STRATEGY)

    df, start_ts, end_ts = _load_data(
        args.symbol,
        args.timeframe,
        months=args.months,
        end_date=args.end,
        allow_short_data=args.allow_short_data,
    )

    if df is None:
        return

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join("reports", f"matrix_{run_id}")
    os.makedirs(report_dir, exist_ok=True)

    rows = []

    # Baseline run
    logger.info("Running baseline with current defaults")
    rows.append(
        _run_single(
            df=df,
            base_strategy=base_strategy,
            overrides={},
            stage="baseline",
            config_id="baseline_defaults",
            start_ts=start_ts,
            end_ts=end_ts,
            max_trade_duration=args.max_trade_duration,
        )
    )

    stage1_rows = []
    if not args.baseline_only:
        # Stage 1 sweep: entry_mode x bos_required
        for entry_mode, bos_required in product(ENTRY_MODES, [False, True]):
            overrides = {
                "entry_mode": entry_mode,
                "bos_required": bos_required,
            }
            config_id = f"stage1_{entry_mode}_bos_{str(bos_required).lower()}"
            logger.info("Stage1: %s", config_id)
            row = _run_single(
                df=df,
                base_strategy=base_strategy,
                overrides=overrides,
                stage="stage1",
                config_id=config_id,
                start_ts=start_ts,
                end_ts=end_ts,
                max_trade_duration=args.max_trade_duration,
            )
            stage1_rows.append(row)
            rows.append(row)

    best_by_mode = {}
    top_modes = []
    stage2_rows = []

    if stage1_rows:
        best_by_mode, top_modes = _select_top_modes(stage1_rows, top_n=2)

    if not args.baseline_only and not args.stage1_only and top_modes:
        # Stage 2 sweep for top modes
        for entry_mode in top_modes:
            best_row = best_by_mode.get(entry_mode)
            if not best_row:
                continue

            bos_required = best_row["bos_required"]

            for fvg_min, ob_body, ob_disp in product(FVG_GRID, OB_BODY_GRID, OB_DISPLACEMENT_GRID):
                overrides = {
                    "entry_mode": entry_mode,
                    "bos_required": bos_required,
                    "fvg_min_size_atr_mult": fvg_min,
                    "ob_min_body_atr_mult": ob_body,
                    "ob_displacement_mult": ob_disp,
                }
                config_id = (
                    f"stage2_{entry_mode}_bos_{str(bos_required).lower()}"
                    f"_fvg_{fvg_min}_obbody_{ob_body}_obdisp_{ob_disp}"
                )
                logger.info("Stage2: %s", config_id)
                row = _run_single(
                    df=df,
                    base_strategy=base_strategy,
                    overrides=overrides,
                    stage="stage2",
                    config_id=config_id,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    max_trade_duration=args.max_trade_duration,
                )
                stage2_rows.append(row)
                rows.append(row)

    # Aggregate results
    results_df = pd.DataFrame(rows)
    results_path = os.path.join(report_dir, "matrix_results.csv")
    results_df.to_csv(results_path, index=False)

    # Rank configs vs validation targets
    ranked = results_df.sort_values(
        by=["expectancy_r", "total_trades", "max_drawdown_pct"],
        ascending=[False, False, True],
        na_position="last",
    )
    top3 = ranked.head(3).to_dict("records")

    summary = {
        "run_id": run_id,
        "data": {
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "months": args.months,
            "start_date": start_ts.isoformat(),
            "end_date": end_ts.isoformat(),
            "candles": len(df),
        },
        "stage1_best_by_mode": best_by_mode,
        "top_modes": top_modes,
        "top3_configs": top3,
        "validation_targets": VALIDATION,
    }

    summary_path = os.path.join(report_dir, "matrix_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Matrix complete. Results: %s", results_path)
    logger.info("Summary: %s", summary_path)


if __name__ == "__main__":
    main()
