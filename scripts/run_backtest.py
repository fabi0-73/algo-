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
from src.backtest.metrics import (
    calculate_metrics,
    generate_report,
    generate_funnel_report,
)
from src.visualization.charts import generate_report as generate_visual_report
from config import STRATEGY, BACKTEST, VALIDATION, EXECUTION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _print_train_test_comparison(train_res: dict, test_res: dict, split_ratio: float) -> None:
    """Print train vs test comparison table."""
    print("\n" + "=" * 72)
    print("TRAIN vs TEST COMPARISON")
    print("=" * 72)
    print(f"Split: {split_ratio:.0%} train / {1 - split_ratio:.0%} test\n")

    header = f"{'Metric':<26} {'Train':>14} {'Test':>14} {'Ratio':>12}"
    print(header)
    print("-" * 68)

    rows = [
        ("Trades", "total_trades"),
        ("Win Rate (%)", "win_rate"),
        ("Expectancy (R)", "expectancy_r"),
        ("Profit Factor", "profit_factor"),
        ("Max Drawdown (%)", "max_drawdown_pct"),
        ("Net P&L ($)", "net_pnl_usd"),
    ]
    for label, key in rows:
        tv = train_res.get(key, 0)
        ev = test_res.get(key, 0)
        if "usd" in key or "pnl" in key.lower():
            ts, es = f"${tv:,.2f}", f"${ev:,.2f}"
        elif "pct" in key or "rate" in key.lower():
            ts, es = f"{tv:.1f}%", f"{ev:.1f}%"
        elif isinstance(tv, int):
            ts, es = str(tv), str(ev)
        else:
            ts, es = f"{tv:.3f}", f"{ev:.3f}"
        ratio_str = f"{ev / tv:.2f}x" if tv and tv != 0 else "N/A"
        print(f"{label:<26} {ts:>14} {es:>14} {ratio_str:>12}")

    train_exp = train_res.get("expectancy_r", 0)
    test_exp = test_res.get("expectancy_r", 0)
    if train_exp > 0:
        deg = test_exp / train_exp
        print(f"\nDegradation ratio: {deg:.2f} (target >= 0.60) -> {'PASS' if deg >= 0.6 else 'FAIL'}")
    print("=" * 72)


def _run_monte_carlo_on_results(results: dict, initial_capital: float) -> None:
    """Run a quick Monte Carlo on the test set trades."""
    trades = results.get("trades", [])
    if not trades:
        return

    try:
        import numpy as np
        from scripts.monte_carlo import run_simulation, print_results as mc_print
        from config import RISK_MODEL

        r_vals = np.array([t.get("r_multiple", 0) for t in trades], dtype=np.float64)
        risk_pct = RISK_MODEL.get("risk_pct_per_trade_default", 0.005)
        mc = run_simulation(r_vals, risk_pct, initial_capital, num_simulations=5000)
        mc_print(mc)
    except Exception as e:
        logger.warning(f"Monte Carlo skipped: {e}")


def run_backtest(
    symbol: str = None,
    timeframe: str = None,
    start_date: datetime = None,
    end_date: datetime = None,
    initial_capital: float = None,
    save_trades: bool = False,
    generate_charts: bool = True,
    save_report: bool = True,
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
    split_ratio: float = None,
    train_only: bool = False,
    test_only: bool = False,
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
        generate_charts: Whether to generate chart images in the report
        save_report: Whether to save backtest report artifacts
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
        split_ratio: Train/test split ratio (time-based)
        train_only: Run only the training split
        test_only: Run only the test split
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
    account_type = (EXECUTION.get("account_type") or "").upper()
    preset = EXECUTION.get("account_presets", {}).get(account_type, {})
    effective_spread = spread_points
    if effective_spread is None:
        effective_spread = preset.get("fixed_spread_points")
    if effective_spread is None:
        effective_spread = EXECUTION.get("fixed_spread_points", 0.25)
    effective_commission = commission_per_lot
    if effective_commission is None:
        effective_commission = preset.get("commission_per_lot_round_turn")
    if effective_commission is None:
        effective_commission = EXECUTION.get("commission_per_lot_round_turn", 7.0)
    logger.info("Execution settings:")
    logger.info(f"  Fill model: {fill_model or EXECUTION.get('entry_fill_model', 'LIMIT_AT_RETEST')}")
    logger.info(f"  Intrabar: {intrabar_assumption or EXECUTION.get('intrabar_fill_rule', 'WORST_CASE')}")
    if account_type:
        logger.info(f"  Account: {account_type}")
    logger.info(f"  Spread: {effective_spread} pts")
    logger.info(f"  Commission: {effective_commission} per lot RT")

    # Log filter settings
    logger.info("Filter settings:")
    logger.info(f"  Session filter: {enable_session_filter if enable_session_filter is not None else True}")
    logger.info(f"  News filter: {enable_news_filter if enable_news_filter is not None else True}")
    logger.info(f"  HTF bias: {enable_htf_bias if enable_htf_bias is not None else True}")
    logger.info(f"  Key levels: {enable_key_levels if enable_key_levels is not None else True}")
    logger.info(f"  Volume filter: {enable_volume_filter if enable_volume_filter is not None else True}")
    logger.info(f"  Fundamentals: {enable_fundamentals if enable_fundamentals is not None else False}")

    def _run_single(df_slice):
        engine = BacktestEngine(
            initial_capital=initial_capital,
            max_trade_duration=STRATEGY.get("max_trade_duration", 200),
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
        return engine, engine.run(df_slice, verbose=verbose)

    def _print_summary(res, label):
        if res is None:
            return
        if "error" in res:
            logger.error(f"{label} backtest failed: {res['error']}")
            # Still show funnel stats to help diagnose 0-trade runs
            funnel = res.get("funnel_stats")
            if funnel:
                print("\n" + "=" * 60)
                print(f"SIGNAL FUNNEL ({label.upper()})")
                print("=" * 60)
                print(f"  Consolidations found:    {funnel.get('consolidations_found', 0)}")
                print(f"  -> No manipulation:      {funnel.get('no_manipulation', 0)}")
                print(f"  -> No distribution:      {funnel.get('no_distribution', 0)}")
                print(f"  -> Weak distribution:    {funnel.get('no_distribution_follow_through', 0)}")
                print(f"  -> No BOS:               {funnel.get('no_bos', 0)}")
                print(f"  -> No entry/retest:      {funnel.get('no_entry_retest', 0)}")
                print(f"  -> Entry too late:       {funnel.get('entry_too_late', 0)}")
                print("\n  FILTER REJECTIONS:")
                print(f"  -> Session/Killzone:     {funnel.get('filtered_session', 0)}")
                print(f"  -> Blackout hour:        {funnel.get('filtered_blackout_hour', 0)}")
                print(f"  -> News blackout:        {funnel.get('filtered_news', 0)}")
                print(f"  -> HTF bias:             {funnel.get('filtered_htf_bias', 0)}")
                print(f"  -> Key levels:           {funnel.get('filtered_key_levels', 0)}")
                print(f"  -> Volume:               {funnel.get('filtered_volume', 0)}")
                print(f"  -> Fundamentals:         {funnel.get('filtered_fundamentals', 0)}")
                print(f"  -> Fill not triggered:   {funnel.get('fill_not_triggered', 0)}")
                print("=" * 60)
            return
            return

        print("\n" + "=" * 60)
        print(f"BACKTEST RESULTS ({label.upper()})")
        print("=" * 60)
        print(f"Backtest ID:      {res['backtest_id']}")
        print(f"Total Trades:     {res['total_trades']}")
        print(f"Win Rate:         {res['win_rate']:.1f}%")
        print(f"Expectancy:       {res['expectancy_r']:.3f}R")
        print(f"Profit Factor:    {res['profit_factor']:.2f}")
        print(f"Max Drawdown:     {res['max_drawdown_pct']:.1f}%")
        print("=" * 60)

        print("\nP&L BREAKDOWN:")
        print(f"  Gross P&L:      ${res.get('gross_pnl_usd', 0):,.2f}")
        print(f"  Net P&L:        ${res.get('net_pnl_usd', 0):,.2f}")
        print(f"  Final Capital:  ${res['final_capital']:,.2f}")

        cost_stats = res.get("cost_stats", {})
        if cost_stats:
            print("\nEXECUTION COSTS:")
            print(f"  Spread:         ${cost_stats.get('total_spread_cost', 0):,.2f}")
            print(f"  Slippage:       ${cost_stats.get('total_slippage_cost', 0):,.2f}")
            print(f"  Commission:     ${cost_stats.get('total_commission_cost', 0):,.2f}")
            print(f"  Total Costs:    ${cost_stats.get('total_costs', 0):,.2f}")
        print("=" * 60)

    results = None
    results_train = None
    results_test = None
    engine_full = None
    engine_train = None
    engine_test = None

    if split_ratio is not None:
        df = df.sort_values("timestamp").reset_index(drop=True)
        split_idx = int(len(df) * split_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        logger.info(f"Split ratio: {split_ratio:.2f} (train {len(train_df)} / test {len(test_df)} candles)")

        if not test_only:
            engine_train, results_train = _run_single(train_df)
            _print_summary(results_train, "train")
        if not train_only:
            engine_test, results_test = _run_single(test_df)
            _print_summary(results_test, "test")

        results = results_test or results_train

        # Train vs test comparison table
        if results_train and results_test and "error" not in results_train and "error" not in results_test:
            _print_train_test_comparison(results_train, results_test, split_ratio)

            # Auto-run Monte Carlo on test set
            _run_monte_carlo_on_results(results_test, initial_capital)
    else:
        engine_full, results = _run_single(df)
        _print_summary(results, "full")

    if results is None:
        return None
    if "error" in results:
        return results

    # Exit reason breakdown
    print("\nEXIT REASONS:")
    print(f"  Stop Loss:      {results.get('sl_exits', 0)}")
    print(f"  Take Profit:    {results.get('tp_exits', 0)}")
    print(f"  Timeout:        {results.get('timeout_exits', 0)}")
    print(f"  Rollover:       {results.get('rollover_exits', 0)}")
    print("=" * 60)

    # Validation
    validation = results["validation"]
    min_trades_target = VALIDATION.get("min_trades", 200)
    min_exp_target = VALIDATION.get("min_expectancy_r", 0.25)
    max_dd_target = VALIDATION.get("max_drawdown_pct", 0.20)
    print("\nVALIDATION:")
    print(f"  Min {min_trades_target} trades: {'PASS' if validation['meets_min_trades'] else 'FAIL'} ({results['total_trades']})")
    print(f"  Expectancy > {min_exp_target}R: {'PASS' if validation['meets_expectancy'] else 'FAIL'} ({results['expectancy_r']:.3f})")
    print(f"  Max DD < {max_dd_target*100:.0f}%: {'PASS' if validation['meets_drawdown'] else 'FAIL'} ({results['max_drawdown_pct']:.1f}%)")

    all_pass = (
        validation.get("meets_min_trades", False)
        and validation.get("meets_expectancy", False)
        and validation.get("meets_drawdown", False)
    )
    print(f"\n  OVERALL: {'PASS - Ready for live testing' if all_pass else 'FAIL - Needs optimization'}")
    print("=" * 60)

    # Objective profile validation for $100 branch (300-500 trades + net positive)
    print("\nOBJECTIVE PROFILE (300-500 + NET+):")
    print(
        f"  Trade Band 300-500: {'PASS' if validation.get('meets_trade_band_300_500', False) else 'FAIL'} "
        f"({results['total_trades']})"
    )
    print(
        f"  Net P&L Positive:   {'PASS' if validation.get('meets_net_positive', False) else 'FAIL'} "
        f"(${results.get('net_pnl_usd', 0):,.2f})"
    )
    print(
        f"  Objective Pass:     {'PASS' if validation.get('objective_pass', False) else 'FAIL'}"
    )
    print("=" * 60)

    # Funnel stats (expanded)
    if "funnel_stats" in results:
        funnel = results["funnel_stats"]
        print("\nSIGNAL FUNNEL:")
        print(f"  Consolidations found:    {funnel.get('consolidations_found', 0)}")
        print(f"  -> No manipulation:      {funnel.get('no_manipulation', 0)}")
        print(f"  -> No distribution:      {funnel.get('no_distribution', 0)}")
        print(f"  -> Weak distribution:    {funnel.get('no_distribution_follow_through', 0)}")
        print(f"  -> No BOS:               {funnel.get('no_bos', 0)}")
        print(f"  -> No entry/retest:      {funnel.get('no_entry_retest', 0)}")
        print(f"  -> Entry too late:       {funnel.get('entry_too_late', 0)}")
        print(f"  -> Short filtered:       {funnel.get('short_filtered', 0)}")
        print(f"  -> Risk invalid:         {funnel.get('risk_invalid', 0)}")
        print(f"  -> Pattern duplicates:   {funnel.get('pattern_duplicates', 0)}")
        print("\n  FILTER REJECTIONS:")
        print(f"  -> Session/Killzone:     {funnel.get('filtered_session', 0)}")
        print(f"  -> Blackout hour:        {funnel.get('filtered_blackout_hour', 0)}")
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

    # Confidence tier stats
    if "confidence_tier_stats" in results:
        tier_stats = results["confidence_tier_stats"]
        print("\nCONFIDENCE TIER BREAKDOWN:")
        print(f"  {'Tier':<12} {'Count':>6} {'Wins':>6} {'WR%':>8} {'PnL':>10}")
        print("  " + "-" * 46)
        for tier_name in ["high", "medium", "standard", "base"]:
            if tier_name in tier_stats:
                ts = tier_stats[tier_name]
                print(f"  {tier_name:<12} {ts['count']:>6} {ts['wins']:>6} {ts['win_rate']:>7.1f}% ${ts['pnl']:>9.2f}")
        print("=" * 60)

    # Trailing stop stats
    if results.get("trades"):
        import pandas as _pd
        _tdf = _pd.DataFrame(results["trades"])
        if "trailing_active" in _tdf.columns:
            trail_count = _tdf["trailing_active"].sum()
            be_count = _tdf.get("sl_moved_to_be", _pd.Series(dtype=bool)).sum() if "sl_moved_to_be" in _tdf.columns else 0
            big_winners = len(_tdf[_tdf["r_multiple"] >= 3.0])
            print(f"\nEXIT MANAGEMENT:")
            print(f"  BE activations:          {be_count}")
            print(f"  Trailing activations:    {trail_count}")
            print(f"  Big winners (>=3R):      {big_winners}")
            print("=" * 60)

    run_sets = []
    if engine_train and results_train:
        run_sets.append(("train", engine_train, results_train))
    if engine_test and results_test:
        run_sets.append(("test", engine_test, results_test))
    if engine_full and results:
        run_sets.append(("full", engine_full, results))

    # Save trades if requested
    if save_trades:
        for label, eng, res in run_sets:
            if res.get("trades"):
                eng.save_trades_to_db(db)
                logger.info("Trades saved to database (%s)", label)

    # Generate report artifacts (text/json always; charts optional)
    if save_report:
        for label, _, res in run_sets:
            if res.get("trades"):
                report_dir = generate_visual_report(
                    res,
                    show_charts=False,
                    render_charts=generate_charts,
                )
                logger.info(
                    "Report saved to %s (%s) [charts=%s]",
                    report_dir, label, "on" if generate_charts else "off"
                )

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
    parser.add_argument("--capital", type=float, help=f"Initial capital (default: {BACKTEST['initial_capital']})")

    # Output options
    parser.add_argument("--save", action="store_true", help="Save trades to database")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart image generation (report text/json still saved)")
    parser.add_argument("--no-report", action="store_true", help="Skip writing report artifacts to reports/")
    parser.add_argument("--verbose", "-v", action="store_true", help="Log each trade")
    parser.add_argument("--split", type=float, help="Train/test split ratio (e.g., 0.7)")
    parser.add_argument("--train-only", action="store_true", help="Run only the training split")
    parser.add_argument("--test-only", action="store_true", help="Run only the test split")

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
        save_report=not args.no_report,
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
        split_ratio=args.split,
        train_only=args.train_only,
        test_only=args.test_only,
    )


if __name__ == "__main__":
    main()
