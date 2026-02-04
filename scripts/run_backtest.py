"""
Run Backtest Script
Execute AMD strategy backtest on historical data.
Supports all realism filters and execution model options.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime, timedelta
import logging

from src.data.db import Database
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import calculate_metrics, generate_report, generate_funnel_report
from src.visualization.charts import generate_report as generate_visual_report
from config import STRATEGY, BACKTEST, VALIDATION, EXECUTION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_backtest(
    symbol: str = None,
    timeframe: str = None,
    start_date: datetime = None,
    end_date: datetime = None,
    initial_capital: float = None,
    save_trades: bool = False,
    generate_charts: bool = True,
    verbose: bool = False,
    # Execution model options
    fill_model: str = None,
    intrabar_assumption: str = None,
    spread_points: float = None,
    slippage_model: str = None,
    commission_per_lot: float = None,
    # Filter toggles
    enable_session_filter: bool = None,
    enable_news_filter: bool = None,
    enable_htf_bias: bool = None,
    enable_key_levels: bool = None,
    enable_volume_filter: bool = None,
    enable_fundamentals: bool = None,
):
    """
    Run backtest on historical data from database.

    Args:
        symbol: Trading symbol (default from config)
        timeframe: Timeframe (default from config)
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_capital: Starting capital
        save_trades: Whether to save trades to database
        generate_charts: Whether to generate visual report
        verbose: Whether to log each trade
        fill_model: Entry fill model (CLOSE, LIMIT_AT_RETEST)
        intrabar_assumption: Intrabar ambiguity (WORST_CASE, BEST_CASE, RANDOM)
        spread_points: Bid/ask spread in points
        slippage_model: Slippage model (NONE, FIXED, ATR_MULT)
        commission_per_lot: Commission per lot round-trip
        enable_session_filter: Enable kill zone / session filter
        enable_news_filter: Enable news blackout filter
        enable_htf_bias: Enable HTF bias alignment
        enable_key_levels: Enable key level scoring
        enable_volume_filter: Enable volume confirmation
        enable_fundamentals: Enable fundamentals filter (DXY/yields)
    """
    symbol = symbol or STRATEGY["symbol"]
    timeframe = timeframe or STRATEGY["timeframe"]
    initial_capital = initial_capital or BACKTEST["initial_capital"]

    logger.info(f"Loading data for {symbol} {timeframe}")

    # Load data from database
    db = Database()
    df = db.get_candles(symbol, timeframe, start_date, end_date)

    if df.empty:
        logger.error("No data found in database. Run fetch_data.py first.")
        return None

    logger.info(f"Loaded {len(df)} candles")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Calculate months of data
    days = (df["timestamp"].max() - df["timestamp"].min()).days
    months = days / 30
    logger.info(f"Data spans approximately {months:.1f} months")

    if months < VALIDATION["min_months"]:
        logger.warning(f"Less than {VALIDATION['min_months']} months of data. Results may not be reliable.")

    # Log execution settings
    logger.info("Execution settings:")
    logger.info(f"  Fill model: {fill_model or EXECUTION.get('entry_fill_model', 'LIMIT_AT_RETEST')}")
    logger.info(f"  Intrabar: {intrabar_assumption or EXECUTION.get('intrabar_fill_rule', 'WORST_CASE')}")
    logger.info(f"  Spread: {spread_points or EXECUTION.get('fixed_spread_points', 0.25)} pts")

    # Log filter settings
    logger.info("Filter settings:")
    logger.info(f"  Session filter: {enable_session_filter if enable_session_filter is not None else True}")
    logger.info(f"  News filter: {enable_news_filter if enable_news_filter is not None else True}")
    logger.info(f"  HTF bias: {enable_htf_bias if enable_htf_bias is not None else True}")
    logger.info(f"  Key levels: {enable_key_levels if enable_key_levels is not None else True}")
    logger.info(f"  Volume filter: {enable_volume_filter if enable_volume_filter is not None else True}")
    logger.info(f"  Fundamentals: {enable_fundamentals if enable_fundamentals is not None else False}")

    # Run backtest with new options
    engine = BacktestEngine(
        initial_capital=initial_capital,
        max_trade_duration=200,
        # Execution model
        fill_model=fill_model,
        intrabar_assumption=intrabar_assumption,
        spread_points=spread_points,
        slippage_model=slippage_model,
        commission_per_lot=commission_per_lot,
        # Filters
        enable_session_filter=enable_session_filter,
        enable_news_filter=enable_news_filter,
        enable_htf_bias=enable_htf_bias,
        enable_key_levels=enable_key_levels,
        enable_volume_filter=enable_volume_filter,
        enable_fundamentals=enable_fundamentals,
    )
    results = engine.run(df, verbose=verbose)

    if "error" in results:
        logger.error(f"Backtest failed: {results['error']}")
        return results

    # Print summary
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Backtest ID:      {results['backtest_id']}")
    print(f"Total Trades:     {results['total_trades']}")
    print(f"Win Rate:         {results['win_rate']:.1f}%")
    print(f"Expectancy:       {results['expectancy_r']:.3f}R")
    print(f"Profit Factor:    {results['profit_factor']:.2f}")
    print(f"Max Drawdown:     {results['max_drawdown_pct']:.1f}%")
    print("=" * 60)

    # P&L breakdown with costs
    print("\nP&L BREAKDOWN:")
    print(f"  Gross P&L:      ${results.get('gross_pnl_usd', 0):,.2f}")
    print(f"  Net P&L:        ${results.get('net_pnl_usd', 0):,.2f}")
    print(f"  Final Capital:  ${results['final_capital']:,.2f}")

    # Cost breakdown
    cost_stats = results.get("cost_stats", {})
    if cost_stats:
        print("\nEXECUTION COSTS:")
        print(f"  Spread:         ${cost_stats.get('total_spread_cost', 0):,.2f}")
        print(f"  Slippage:       ${cost_stats.get('total_slippage_cost', 0):,.2f}")
        print(f"  Commission:     ${cost_stats.get('total_commission_cost', 0):,.2f}")
        print(f"  Total Costs:    ${cost_stats.get('total_costs', 0):,.2f}")
    print("=" * 60)

    # Exit reason breakdown
    print("\nEXIT REASONS:")
    print(f"  Stop Loss:      {results.get('sl_exits', 0)}")
    print(f"  Take Profit:    {results.get('tp_exits', 0)}")
    print(f"  Timeout:        {results.get('timeout_exits', 0)}")
    print(f"  Rollover:       {results.get('rollover_exits', 0)}")
    print("=" * 60)

    # Validation
    validation = results["validation"]
    print("\nVALIDATION:")
    print(f"  Min 500 trades: {'PASS' if validation['meets_min_trades'] else 'FAIL'} ({results['total_trades']})")
    print(f"  Expectancy > 0.2R: {'PASS' if validation['meets_expectancy'] else 'FAIL'} ({results['expectancy_r']:.3f})")
    print(f"  Max DD < 15%: {'PASS' if validation['meets_drawdown'] else 'FAIL'} ({results['max_drawdown_pct']:.1f}%)")

    all_pass = all(validation.values())
    print(f"\n  OVERALL: {'PASS - Ready for live testing' if all_pass else 'FAIL - Needs optimization'}")
    print("=" * 60)

    # Funnel stats (expanded)
    if "funnel_stats" in results:
        funnel = results["funnel_stats"]
        print("\nSIGNAL FUNNEL:")
        print(f"  Consolidations found:    {funnel.get('consolidations_found', 0)}")
        print(f"  -> No manipulation:      {funnel.get('no_manipulation', 0)}")
        print(f"  -> No distribution:      {funnel.get('no_distribution', 0)}")
        print(f"  -> No BOS:               {funnel.get('no_bos', 0)}")
        print(f"  -> No entry/retest:      {funnel.get('no_entry_retest', 0)}")
        print(f"  -> Short filtered:       {funnel.get('short_filtered', 0)}")
        print(f"  -> Risk invalid:         {funnel.get('risk_invalid', 0)}")
        print(f"  -> Pattern duplicates:   {funnel.get('pattern_duplicates', 0)}")
        print("\n  FILTER REJECTIONS:")
        print(f"  -> Session/Killzone:     {funnel.get('filtered_session', 0)}")
        print(f"  -> News blackout:        {funnel.get('filtered_news', 0)}")
        print(f"  -> HTF bias:             {funnel.get('filtered_htf_bias', 0)}")
        print(f"  -> Key levels:           {funnel.get('filtered_key_levels', 0)}")
        print(f"  -> Volume:               {funnel.get('filtered_volume', 0)}")
        print(f"  -> Fundamentals:         {funnel.get('filtered_fundamentals', 0)}")
        print(f"  -> Daily limit:          {funnel.get('filtered_daily_limit', 0)}")
        print(f"  -> Cooldown:             {funnel.get('filtered_cooldown', 0)}")
        print(f"  -> Near rollover:        {funnel.get('filtered_rollover', 0)}")
        print(f"  -> Fill not triggered:   {funnel.get('fill_not_triggered', 0)}")
        print(f"\n  ENTRIES EXECUTED:        {funnel.get('entries_executed', 0)}")
        print("=" * 60)

    # Confluence stats
    if "confluence_stats" in results:
        conf = results["confluence_stats"]
        print("\nCONFLUENCE STATS:")
        print(f"  Avg confluence score:    {conf.get('avg_confluence_score', 0):.2f}")
        print(f"  Entries with FVG:        {conf.get('entries_with_fvg', 0)}")
        print(f"  Entries with OB:         {conf.get('entries_with_ob', 0)}")
        print(f"  Entries with BOS:        {conf.get('entries_with_bos', 0)}")
        print("=" * 60)

    # Save trades if requested
    if save_trades and results["trades"]:
        engine.save_trades_to_db(db)
        logger.info("Trades saved to database")

    # Generate charts
    if generate_charts and results["trades"]:
        report_dir = generate_visual_report(results, show_charts=False)
        logger.info(f"Visual report saved to {report_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run AMD strategy backtest with realism filters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic backtest
  python run_backtest.py

  # Backtest with specific date range
  python run_backtest.py --start 2024-01-01 --end 2024-06-30

  # Disable all filters (raw signals)
  python run_backtest.py --no-session --no-news --no-htf --no-keylevel --no-volume

  # Use optimistic execution model
  python run_backtest.py --fill-model CLOSE --intrabar BEST_CASE --spread 0

  # Full realism with fundamentals
  python run_backtest.py --enable-fundamentals --intrabar WORST_CASE
        """
    )

    # Data options
    parser.add_argument("--symbol", type=str, help="Trading symbol (default: XAUUSD)")
    parser.add_argument("--timeframe", type=str, help="Timeframe (default: M5)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, help="Initial capital (default: 10000)")

    # Output options
    parser.add_argument("--save", action="store_true", help="Save trades to database")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Log each trade")

    # Execution model options
    parser.add_argument(
        "--fill-model", type=str, choices=["CLOSE", "LIMIT_AT_RETEST"],
        help="Entry fill model (default: LIMIT_AT_RETEST)"
    )
    parser.add_argument(
        "--intrabar", type=str, choices=["WORST_CASE", "BEST_CASE", "RANDOM"],
        help="Intrabar ambiguity assumption (default: WORST_CASE)"
    )
    parser.add_argument("--spread", type=float, help="Bid/ask spread in points (default: 30)")
    parser.add_argument(
        "--slippage", type=str, choices=["NONE", "FIXED", "ATR_MULT"],
        help="Slippage model (default: ATR_MULT)"
    )
    parser.add_argument("--commission", type=float, help="Commission per lot round-trip (default: 7.0)")

    # Filter toggles (disable)
    parser.add_argument("--no-session", action="store_true", help="Disable session/kill zone filter")
    parser.add_argument("--no-news", action="store_true", help="Disable news blackout filter")
    parser.add_argument("--no-htf", action="store_true", help="Disable HTF bias filter")
    parser.add_argument("--no-keylevel", action="store_true", help="Disable key level filter")
    parser.add_argument("--no-volume", action="store_true", help="Disable volume filter")

    # Filter toggles (enable)
    parser.add_argument("--enable-fundamentals", action="store_true", help="Enable fundamentals filter (DXY/yields)")

    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d") if args.start else None
    end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else None

    run_backtest(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
        save_trades=args.save,
        generate_charts=not args.no_charts,
        verbose=args.verbose,
        # Execution model
        fill_model=args.fill_model,
        intrabar_assumption=args.intrabar,
        spread_points=args.spread,
        slippage_model=args.slippage,
        commission_per_lot=args.commission,
        # Filters
        enable_session_filter=not args.no_session if args.no_session else None,
        enable_news_filter=not args.no_news if args.no_news else None,
        enable_htf_bias=not args.no_htf if args.no_htf else None,
        enable_key_levels=not args.no_keylevel if args.no_keylevel else None,
        enable_volume_filter=not args.no_volume if args.no_volume else None,
        enable_fundamentals=args.enable_fundamentals if args.enable_fundamentals else None,
    )


if __name__ == "__main__":
    main()
