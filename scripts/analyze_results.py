"""
Analyze Results Script
Deep analysis of backtest results and trade statistics.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime
import logging
import pandas as pd

from src.data.db import Database
from src.backtest.metrics import (
    calculate_metrics,
    calculate_monthly_returns,
    calculate_session_performance,
    generate_report,
    BacktestMetrics,
)
from src.visualization.charts import (
    plot_equity_curve,
    plot_r_distribution,
    plot_monthly_performance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def analyze_backtest(
    backtest_id: str = None,
    show_charts: bool = True,
    save_report: bool = True,
):
    """
    Analyze backtest results from database.
    
    Args:
        backtest_id: Specific backtest ID to analyze (latest if None)
        show_charts: Whether to display charts
        save_report: Whether to save report to file
    """
    db = Database()
    
    # Get trades
    trades_df = db.get_trades(backtest_id=backtest_id)
    
    if trades_df.empty:
        logger.error("No trades found in database")
        return
    
    # Get unique backtest IDs
    backtest_ids = trades_df["backtest_id"].unique()
    logger.info(f"Found {len(backtest_ids)} backtest(s) in database")
    
    # Use latest if not specified
    if backtest_id is None:
        backtest_id = backtest_ids[-1]
        trades_df = trades_df[trades_df["backtest_id"] == backtest_id]
    
    logger.info(f"Analyzing backtest: {backtest_id}")
    logger.info(f"Total trades: {len(trades_df)}")
    
    trades = trades_df.to_dict("records")
    
    # Calculate metrics
    metrics = calculate_metrics(trades)
    
    # Print report
    report = generate_report(metrics)
    print("\n" + report)
    
    # Monthly analysis
    monthly = calculate_monthly_returns(trades)
    if not monthly.empty:
        print("\nMONTHLY BREAKDOWN:")
        print("-" * 60)
        print(monthly.to_string())
    
    # Session analysis
    sessions = calculate_session_performance(trades)
    if sessions:
        print("\nPERFORMANCE BY SESSION:")
        print("-" * 60)
        for session, stats in sessions.items():
            print(f"  {session:12} - Trades: {stats['trades']:3} | "
                  f"Win Rate: {stats['win_rate']:5.1f}% | "
                  f"Avg R: {stats['avg_r']:6.3f} | "
                  f"P&L: ${stats['total_pnl']:,.0f}")
    
    # Direction analysis
    print("\nPERFORMANCE BY DIRECTION:")
    print("-" * 60)
    print(f"  LONG:  {metrics.long_trades} trades | Win Rate: {metrics.long_win_rate:.1%}")
    print(f"  SHORT: {metrics.short_trades} trades | Win Rate: {metrics.short_win_rate:.1%}")
    
    # Manipulation direction analysis
    manip_up = trades_df[trades_df["manipulation_direction"] == "UP"]
    manip_down = trades_df[trades_df["manipulation_direction"] == "DOWN"]
    
    print("\nPERFORMANCE BY MANIPULATION TYPE:")
    print("-" * 60)
    if len(manip_up) > 0:
        up_wins = len(manip_up[manip_up["pnl_usd"] > 0])
        print(f"  Fakeout UP (Short setups):   {len(manip_up)} trades | "
              f"Win Rate: {up_wins/len(manip_up):.1%} | "
              f"Avg R: {manip_up['r_multiple'].mean():.3f}")
    if len(manip_down) > 0:
        down_wins = len(manip_down[manip_down["pnl_usd"] > 0])
        print(f"  Fakeout DOWN (Long setups):  {len(manip_down)} trades | "
              f"Win Rate: {down_wins/len(manip_down):.1%} | "
              f"Avg R: {manip_down['r_multiple'].mean():.3f}")
    
    # Show charts
    if show_charts:
        # R-distribution
        plot_r_distribution(trades, show=True)
        
        # Monthly performance
        plot_monthly_performance(trades, show=True)
    
    # Save report
    if save_report:
        report_path = f"reports/analysis_{backtest_id}.txt"
        os.makedirs("reports", exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report)
            f.write("\n\nMONTHLY BREAKDOWN:\n")
            f.write(monthly.to_string())
        logger.info(f"Report saved to {report_path}")


def compare_backtests(backtest_ids: list):
    """
    Compare multiple backtests side by side.
    
    Args:
        backtest_ids: List of backtest IDs to compare
    """
    db = Database()
    
    results = []
    for bt_id in backtest_ids:
        trades_df = db.get_trades(backtest_id=bt_id)
        if not trades_df.empty:
            trades = trades_df.to_dict("records")
            metrics = calculate_metrics(trades)
            results.append({
                "backtest_id": bt_id,
                "trades": metrics.total_trades,
                "win_rate": f"{metrics.win_rate:.1%}",
                "expectancy": f"{metrics.expectancy:.3f}R",
                "profit_factor": f"{metrics.profit_factor:.2f}",
                "max_dd": f"{metrics.max_drawdown_pct:.1%}",
                "total_pnl": f"${metrics.total_pnl:,.0f}",
                "valid": "PASS" if metrics.passes_validation else "FAIL",
            })
    
    if results:
        comparison = pd.DataFrame(results)
        print("\nBACKTEST COMPARISON:")
        print("=" * 80)
        print(comparison.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Analyze backtest results")
    parser.add_argument("--id", type=str, help="Specific backtest ID to analyze")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart display")
    parser.add_argument("--no-save", action="store_true", help="Don't save report")
    parser.add_argument("--compare", type=str, nargs="+", help="Compare multiple backtest IDs")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_backtests(args.compare)
    else:
        analyze_backtest(
            backtest_id=args.id,
            show_charts=not args.no_charts,
            save_report=not args.no_save,
        )


if __name__ == "__main__":
    main()
