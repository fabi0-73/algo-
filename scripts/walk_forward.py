"""
Walk-Forward Validation

Anchored walk-forward: train on first N% of data, test on remaining.
Compares train vs test metrics and reports degradation ratio.

Usage:
    python scripts/walk_forward.py
    python scripts/walk_forward.py --split 0.7
    python scripts/walk_forward.py --split 0.7 --monte-carlo
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from src.data.db import Database
from src.backtest.engine import BacktestEngine
from config import STRATEGY, BACKTEST, RISK_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_engine(df, initial_capital: float, verbose: bool = False) -> Dict[str, Any]:
    """Run backtest engine on a DataFrame slice."""
    # Use the configured trade duration (was hardcoded to 200, which disagreed with
    # STRATEGY["max_trade_duration"] and made walk-forward inconsistent with run_backtest).
    engine = BacktestEngine(initial_capital=initial_capital,
                            max_trade_duration=STRATEGY.get("max_trade_duration", 240))
    return engine.run(df, verbose=verbose)


def print_comparison(train_res: Dict, test_res: Dict, split_ratio: float) -> None:
    """Print train vs test comparison table with degradation ratios."""
    print("\n" + "=" * 72)
    print("WALK-FORWARD VALIDATION")
    print("=" * 72)
    print(f"Split ratio: {split_ratio:.0%} train / {1 - split_ratio:.0%} test")
    print()

    header = f"{'Metric':<28} {'Train':>14} {'Test':>14} {'Degradation':>14}"
    print(header)
    print("-" * 72)

    metrics = [
        ("Total Trades", "total_trades", "int"),
        ("Win Rate (%)", "win_rate", "pct"),
        ("Expectancy (R)", "expectancy_r", "float"),
        ("Profit Factor", "profit_factor", "float"),
        ("Max Drawdown (%)", "max_drawdown_pct", "pct_raw"),
        ("Net P&L ($)", "net_pnl_usd", "usd"),
        ("Avg R-Multiple", "avg_r_multiple", "float"),
        ("Avg Win R", "avg_win_r", "float"),
        ("Avg Loss R", "avg_loss_r", "float"),
    ]

    for label, key, fmt in metrics:
        train_val = train_res.get(key, 0)
        test_val = test_res.get(key, 0)

        if fmt == "int":
            train_str = f"{train_val}"
            test_str = f"{test_val}"
        elif fmt == "pct":
            train_str = f"{train_val:.1f}%"
            test_str = f"{test_val:.1f}%"
        elif fmt == "pct_raw":
            train_str = f"{train_val:.1f}%"
            test_str = f"{test_val:.1f}%"
        elif fmt == "usd":
            train_str = f"${train_val:,.2f}"
            test_str = f"${test_val:,.2f}"
        else:
            train_str = f"{train_val:.3f}"
            test_str = f"{test_val:.3f}"

        if train_val and train_val != 0 and key not in ("max_drawdown_pct",):
            ratio = test_val / train_val if train_val != 0 else 0
            deg_str = f"{ratio:.2f}x"
        elif key == "max_drawdown_pct":
            deg_str = "lower=better" if test_val <= train_val else "WORSE"
        else:
            deg_str = "N/A"

        print(f"{label:<28} {train_str:>14} {test_str:>14} {deg_str:>14}")

    print("-" * 72)

    train_exp = train_res.get("expectancy_r", 0)
    test_exp = test_res.get("expectancy_r", 0)
    if train_exp > 0:
        deg_ratio = test_exp / train_exp
        status = "PASS" if deg_ratio >= 0.6 else "FAIL"
        print(f"\nDegradation Ratio (expectancy): {deg_ratio:.2f} (target >= 0.60) -> {status}")
    else:
        print("\nDegradation Ratio: N/A (train expectancy <= 0)")

    train_pf = train_res.get("profit_factor", 0)
    test_pf = test_res.get("profit_factor", 0)
    if train_pf > 1:
        pf_ratio = test_pf / train_pf
        pf_status = "PASS" if pf_ratio >= 0.5 else "FAIL"
        print(f"Degradation Ratio (PF):         {pf_ratio:.2f} (target >= 0.50) -> {pf_status}")

    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Walk-forward validation")
    parser.add_argument("--split", type=float, default=0.7, help="Train/test split ratio (default 0.7)")
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--timeframe", type=str, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--capital", type=float, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--monte-carlo", action="store_true", help="Run Monte Carlo on test set")
    parser.add_argument("--mc-simulations", type=int, default=10000)
    args = parser.parse_args()

    symbol = args.symbol or STRATEGY["symbol"]
    timeframe = args.timeframe or STRATEGY["timeframe"]
    capital = args.capital or BACKTEST["initial_capital"]

    start_date = datetime.strptime(args.start, "%Y-%m-%d") if args.start else None
    end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else None

    logger.info(f"Loading data for {symbol} {timeframe}")
    db = Database()
    df = db.get_candles(symbol, timeframe, start_date, end_date)

    if df.empty:
        logger.error("No data found. Run fetch_data.py first.")
        return

    df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * args.split)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    logger.info(f"Total: {len(df)} candles | Train: {len(train_df)} | Test: {len(test_df)}")
    logger.info(f"Train: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    logger.info(f"Test:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

    logger.info("Running TRAIN backtest...")
    train_res = run_engine(train_df, capital, verbose=args.verbose)

    logger.info("Running TEST backtest...")
    test_res = run_engine(test_df, capital, verbose=args.verbose)

    if "error" in train_res:
        logger.error(f"Train failed: {train_res['error']}")
    if "error" in test_res:
        logger.error(f"Test failed: {test_res['error']}")

    if "error" not in train_res and "error" not in test_res:
        print_comparison(train_res, test_res, args.split)

    if args.monte_carlo and "error" not in test_res:
        logger.info("Running Monte Carlo on test set...")
        from scripts.monte_carlo import extract_r_multiples, extract_risk_pcts, run_simulation, print_results

        trades = test_res.get("trades", [])
        if trades:
            r_vals = extract_r_multiples(trades)
            risk_pcts = extract_risk_pcts(trades)  # per-trade confidence-tier sizing if recorded
            risk_pct = RISK_MODEL.get("risk_pct_per_trade_default", 0.005)
            mc_results = run_simulation(r_vals, risk_pct, capital, args.mc_simulations,
                                        risk_pcts=risk_pcts, method="bootstrap")
            print_results(mc_results)
        else:
            logger.warning("No test trades for Monte Carlo analysis")


if __name__ == "__main__":
    main()
