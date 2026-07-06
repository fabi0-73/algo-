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
    enable_phantom_fills: bool = None,
    enable_market_chase: bool = None,
    split_ratio: float = None,
    train_only: bool = False,
    test_only: bool = False,
    run_timeframe: str = None,
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

    # Higher-TF run: resample base M5 bars and scale wall-clock plumbing.
    # Pattern params intentionally stay in bars (bigger structures on higher TFs).
    if run_timeframe and run_timeframe != timeframe:
        from src.data.resample import resample_ohlcv
        from config import apply_run_timeframe
        df = resample_ohlcv(df, run_timeframe)
        apply_run_timeframe(run_timeframe)
        logger.info(f"Resampled to {run_timeframe}: {len(df)} candles")

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
            enable_phantom_fills=enable_phantom_fills,
            enable_market_chase=enable_market_chase,
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
            print(f"  Swap:           ${cost_stats.get('total_swap_cost', 0):,.2f}")
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

    # Exit reason breakdown (honest taxonomy: true losses vs protected/profit stops)
    print("\nEXIT REASONS:")
    print(f"  Stop Loss (loss): {results.get('sl_exits', 0)}")
    print(f"  Breakeven Stop:   {results.get('be_exits', 0)}")
    print(f"  Trailing Stop:    {results.get('trail_exits', 0)}")
    print(f"  Take Profit:      {results.get('tp_exits', 0)}")
    print(f"  Timeout:          {results.get('timeout_exits', 0)}")
    print(f"  Rollover:         {results.get('rollover_exits', 0)}")
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
        print(f"  -> Monthly limit:        {funnel.get('filtered_monthly_limit', 0)}")
        print(f"  -> Drawdown halt:        {funnel.get('filtered_drawdown_halt', 0)}")
        print(f"  -> Cooldown:             {funnel.get('filtered_cooldown', 0)}")
        print(f"  -> Near rollover:        {funnel.get('filtered_rollover', 0)}")
        print(f"  -> Fill not triggered:   {funnel.get('fill_not_triggered', 0)}")
        if funnel.get("sweep_signals", 0) > 0 or funnel.get("sweep_entries", 0) > 0:
            print("\n  SWEEP MODEL:")
            print(f"  -> Sweep signals:        {funnel.get('sweep_signals', 0)}")
            print(f"  -> Sweep filtered:       {funnel.get('sweep_filtered', 0)}")
            print(f"  -> Sweep risk invalid:   {funnel.get('sweep_risk_invalid', 0)}")
            print(f"  -> Sweep entries:        {funnel.get('sweep_entries', 0)}")
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

    # Entry model breakdown (AMD vs SWEEP)
    if results.get("trades"):
        import pandas as _pd
        _mdf = _pd.DataFrame(results["trades"])
        if "entry_model" in _mdf.columns and _mdf["entry_model"].nunique() > 1:
            print("\nENTRY MODEL BREAKDOWN:")
            print(f"  {'Model':<8} {'Count':>6} {'Wins':>6} {'WR%':>8} {'AvgR':>8} {'MaxDDCtb':>9} {'PnL':>10}")
            print("  " + "-" * 60)
            for model in ["AMD", "SWEEP"]:
                subset = _mdf[_mdf["entry_model"] == model]
                if len(subset) == 0:
                    continue
                wins = len(subset[subset["net_pnl"] > 0])
                wr = wins / len(subset) * 100
                worst = subset["net_pnl"].min()
                print(f"  {model:<8} {len(subset):>6} {wins:>6} {wr:>7.1f}% {subset['r_multiple'].mean():>7.3f}R {worst:>9.2f} ${subset['net_pnl'].sum():>9.2f}")
            # Sweep level-kind detail
            _sdf = _mdf[_mdf["entry_model"] == "SWEEP"]
            if len(_sdf) > 0 and "sweep_level_kind" in _sdf.columns:
                print("\n  SWEEP BY LEVEL KIND:")
                for kind, grp in _sdf.groupby("sweep_level_kind"):
                    w = len(grp[grp["net_pnl"] > 0])
                    print(f"    {str(kind):<16} n={len(grp):<4} WR {w/len(grp)*100:5.1f}%  avgR {grp['r_multiple'].mean():+.3f}  PnL ${grp['net_pnl'].sum():+.2f}")
            print("=" * 60)

    # Signal confidence breakdown (empirical LOW/MODERATE/GOOD/HIGH)
    if results.get("trades"):
        import pandas as _pd
        _cdf = _pd.DataFrame(results["trades"])
        if "confidence_label" in _cdf.columns and _cdf["confidence_label"].any():
            print("\nSIGNAL CONFIDENCE BREAKDOWN:")
            print(f"  {'Level':<10} {'Count':>6} {'Wins':>6} {'WR%':>8} {'AvgR':>8} {'AvgLots':>8} {'PnL':>10}")
            print("  " + "-" * 60)
            for label in ["HIGH", "GOOD", "MODERATE", "LOW"]:
                subset = _cdf[_cdf["confidence_label"] == label]
                if len(subset) == 0:
                    continue
                wins = len(subset[subset["net_pnl"] > 0])
                wr = wins / len(subset) * 100
                lots = subset["position_size"].mean() if "position_size" in subset.columns else 0
                print(f"  {label:<10} {len(subset):>6} {wins:>6} {wr:>7.1f}% {subset['r_multiple'].mean():>7.3f}R {lots:>8.3f} ${subset['net_pnl'].sum():>9.2f}")
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
            # Adaptive exit tier breakdown
            if "exit_tier" in _tdf.columns and _tdf["exit_tier"].any():
                print(f"\n  EXIT TIER BREAKDOWN:")
                print(f"  {'Tier':<12} {'Count':>6} {'Wins':>6} {'WR%':>8} {'AvgR':>8} {'AvgMFE':>8} {'PnL':>10}")
                print("  " + "-" * 62)
                for tier_name in _tdf["exit_tier"].unique():
                    if not tier_name:
                        tier_name_display = "(default)"
                    else:
                        tier_name_display = tier_name
                    subset = _tdf[_tdf["exit_tier"] == tier_name]
                    t_wins = len(subset[subset["r_multiple"] > 0])
                    t_wr = t_wins / len(subset) * 100 if len(subset) > 0 else 0
                    t_mfe = subset["mfe_r"].mean() if "mfe_r" in subset.columns else 0
                    print(f"  {tier_name_display:<12} {len(subset):>6} {t_wins:>6} {t_wr:>7.1f}% {subset['r_multiple'].mean():>7.3f}R {t_mfe:>7.2f}R ${subset['net_pnl'].sum():>9.2f}")
            print("=" * 60)

    # MFE/MAE analysis
    mfe = results.get("mfe_stats", {})
    if mfe:
        print(f"\nMFE/MAE ANALYSIS:")
        print(f"  Avg MFE:                 {mfe.get('avg_mfe_r', 0):.2f}R")
        print(f"  Median MFE:              {mfe.get('median_mfe_r', 0):.2f}R")
        print(f"  Avg MAE:                 {mfe.get('avg_mae_r', 0):.2f}R")
        print(f"  Winners Avg MFE:         {mfe.get('winners_avg_mfe_r', 0):.2f}R")
        print(f"  MFE Capture:             {mfe.get('mfe_capture_pct', 0):.1f}%")
        if "by_confluence" in mfe:
            print(f"\n  MFE BY CONFLUENCE SCORE:")
            print(f"  {'Score':<8} {'Count':>6} {'AvgMFE':>8} {'AvgR':>8} {'Capture%':>10}")
            print("  " + "-" * 44)
            for score, data in sorted(mfe["by_confluence"].items(), key=lambda x: int(x[0])):
                print(f"  {score:<8} {data['count']:>6} {data['avg_mfe_r']:>7.2f}R {data['avg_r']:>7.2f}R {data['capture_pct']:>9.1f}%")
        print("=" * 60)

    # Move potential analysis
    mp_stats = results.get("move_potential_stats", {})
    if mp_stats:
        print(f"\nMOVE POTENTIAL ANALYSIS:")
        print(f"  {'Score':<8} {'Count':>6} {'Wins':>6} {'WR%':>8} {'AvgR':>8} {'AvgMFE':>8} {'PnL':>10}")
        print("  " + "-" * 60)
        for score, data in sorted(mp_stats.items(), key=lambda x: int(x[0])):
            print(f"  {score:<8} {data['count']:>6} {data['wins']:>6} {data['win_rate']:>7.1f}% {data['avg_r']:>7.3f}R {data['avg_mfe_r']:>7.2f}R ${data['pnl']:>9.2f}")
        print("=" * 60)

    # Phantom fills analysis
    pf = results.get("phantom_fills", {})
    if pf and pf.get("total_phantom", 0) > 0:
        print("\n" + "=" * 60)
        print("PHANTOM FILLS ANALYSIS (Missed Fill Simulation)")
        print("=" * 60)
        print(f"  Total Missed Fills:     {pf['total_phantom']}")
        print(f"  Would-Be Winners:       {pf['phantom_wins']}")
        print(f"  Would-Be Losers:        {pf['phantom_losses']}")
        print(f"  Phantom Win Rate:       {pf['phantom_win_rate']:.1f}%")
        print(f"  Phantom Avg R:          {pf['phantom_avg_r']:+.3f}")
        print(f"  Phantom Median R:       {pf['phantom_median_r']:+.3f}")
        print(f"  Phantom Net P&L:        ${pf['phantom_net_pnl']:,.2f}")
        print(f"  Phantom Profit Factor:  {pf['phantom_profit_factor']:.2f}")
        print(f"  Avg Entry Slippage:     ${pf['avg_entry_slippage']:.2f}")
        print(f"  Avg Bars to Exit:       {pf['avg_bars_to_resolution']:.0f}")
        print(f"\n  Exit Reasons:")
        for reason, count in sorted(pf.get("by_exit_reason", {}).items()):
            print(f"    {reason:<20} {count}")
        print(f"\n  BY CONFLUENCE SCORE:")
        print(f"  {'Score':<8} {'Count':>6} {'Wins':>6} {'WR%':>7} {'AvgR':>8} {'PF':>6} {'PnL':>10}")
        print("  " + "-" * 55)
        for score, stats in sorted(pf.get("by_confluence_score", {}).items()):
            print(f"  {score:<8} {stats['count']:>6} {stats['wins']:>6} {stats['win_rate']:>6.1f}% {stats['avg_r']:>+7.3f} {stats['profit_factor']:>5.2f} ${stats['net_pnl']:>+9.2f}")
        print(f"\n  BY DIRECTION:")
        for d, stats in pf.get("by_direction", {}).items():
            print(f"    {d:<6} {stats['count']:>4} trades | WR: {stats['win_rate']:.1f}% | Avg R: {stats['avg_r']:+.3f} | PnL: ${stats['net_pnl']:+.2f}")
        print("=" * 60)

    # Market chase stats
    cs = results.get("chase_stats", {})
    if cs and cs.get("chase_attempts", 0) > 0:
        print("\n" + "=" * 60)
        print("MARKET CHASE STATS (Real Trades from Missed Fills)")
        print("=" * 60)
        print(f"  Chase Attempts:         {cs['chase_attempts']}")
        print(f"  Chase Executed:         {cs['chase_executed']}")
        print(f"  Rejected (direction):   {cs['chase_rejected_direction']}")
        print(f"  Rejected (confluence):  {cs['chase_rejected_confluence']}")
        print(f"  Rejected (slippage):    {cs['chase_rejected_slippage']}")
        print(f"  Rejected (risk):        {cs['chase_rejected_risk_invalid']}")
        print("=" * 60)

    # Add chase entries to funnel display
    if "funnel_stats" in results and cs and cs.get("chase_executed", 0) > 0:
        print(f"\n  CHASE ENTRIES:           {cs['chase_executed']}")

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

    # Analysis
    parser.add_argument("--phantom-fills", action="store_true", help="Simulate missed fill trades at candle close (phantom analysis)")
    parser.add_argument("--market-chase", action="store_true", help="Enter at close for high-quality missed fills (real trades)")
    parser.add_argument("--adaptive-exits", action="store_true", help="Enable adaptive exit tiers by move potential score")
    parser.add_argument("--no-drawdown-controls", action="store_true", help="Disable Phase-2 drawdown controls (circuit breaker, risk scaling, loss brake)")
    parser.add_argument("--strategy-mode", type=str, choices=["AMD", "SWEEP", "BOTH"],
                        help="Entry model(s) to run (default from SWEEP_MODEL config)")
    parser.add_argument("--sweep-exit", type=str, choices=["FIXED_RR", "HYBRID"],
                        help="Exit style for SWEEP trades (default from SWEEP_MODEL config)")
    parser.add_argument("--enable-ny-ib", action="store_true",
                        help="Run the NY_IB stream alongside AMD (NY_IB_MODEL)")
    parser.add_argument("--ny-ib-only", action="store_true",
                        help="Run ONLY the NY_IB stream (suppresses AMD scanning)")
    parser.add_argument("--min-confidence", type=int, choices=[0, 1, 2, 3, 4],
                        help="Skip entries with signal confidence below this score (0 = off)")
    parser.add_argument("--run-timeframe", type=str, choices=["M5", "M15", "M30", "H1", "H4"],
                        help="Resample base M5 data to this timeframe for the run")
    parser.add_argument("--long-runner-duration", type=int,
                        help="Max bars for LONG trades with active trailing stop (E4 overlay, 0 = off)")
    parser.add_argument("--entry-price-mode", type=str, choices=["RETEST", "OTE"],
                        help="Entry limit placement: RETEST level or OTE deep-pullback band (E3)")
    parser.add_argument("--adaptive-confidence", action="store_true",
                        help="Enable rolling confidence-bucket recalibration (E-adaptive)")

    args = parser.parse_args()

    # Confidence entry gate override (E-conf-gate experiment)
    if args.min_confidence is not None:
        from config import SIGNAL_CONFIDENCE
        SIGNAL_CONFIDENCE["min_confidence_to_trade"] = args.min_confidence

    # E4 overnight-drift overlay override
    if args.long_runner_duration is not None:
        STRATEGY["long_runner_max_duration"] = args.long_runner_duration

    # E3 OTE entry-price mode override
    if args.entry_price_mode:
        STRATEGY["entry_price_mode"] = args.entry_price_mode

    # E-adaptive override
    if args.adaptive_confidence:
        from config import CONFIDENCE_SIZING
        CONFIDENCE_SIZING["adaptive"]["enabled"] = True

    # Enable adaptive exits if requested
    if args.adaptive_exits:
        from config import ADAPTIVE_EXITS
        ADAPTIVE_EXITS["enabled"] = True

    # Disable Phase-2 drawdown controls if requested (for isolating their effect)
    if args.no_drawdown_controls:
        from config import DRAWDOWN_CONTROLS
        DRAWDOWN_CONTROLS["enabled"] = False

    # Strategy mode / sweep exit style overrides
    if args.strategy_mode or args.sweep_exit:
        from config import SWEEP_MODEL
        if args.strategy_mode:
            SWEEP_MODEL["strategy_mode"] = args.strategy_mode
        if args.sweep_exit:
            SWEEP_MODEL["exit_style"] = args.sweep_exit

    # NY_IB stream overrides
    if args.enable_ny_ib or args.ny_ib_only:
        from config import NY_IB_MODEL
        NY_IB_MODEL["enabled"] = True
        NY_IB_MODEL["only"] = bool(args.ny_ib_only)

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
        enable_phantom_fills=args.phantom_fills if args.phantom_fills else None,
        enable_market_chase=args.market_chase if args.market_chase else None,
        split_ratio=args.split,
        train_only=args.train_only,
        test_only=args.test_only,
        run_timeframe=args.run_timeframe,
    )


if __name__ == "__main__":
    main()
