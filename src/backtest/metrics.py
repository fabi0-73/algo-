"""
Performance Metrics
Calculate and analyze backtest performance statistics.
Supports gross/net P&L breakdown and execution cost tracking.
"""
from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

from config import VALIDATION


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics."""

    # Trade counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0

    # Win rate
    win_rate: float = 0.0

    # R-multiples
    avg_r_multiple: float = 0.0
    median_r_multiple: float = 0.0
    max_r_multiple: float = 0.0
    min_r_multiple: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0

    # Expectancy
    expectancy: float = 0.0

    # P&L (with gross/net breakdown)
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    total_pnl: float = 0.0  # Alias for net_pnl (backwards compat)
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    avg_pnl_per_trade: float = 0.0

    # Execution costs
    total_spread_cost: float = 0.0
    total_slippage_cost: float = 0.0
    total_commission_cost: float = 0.0
    total_costs: float = 0.0
    cost_per_trade: float = 0.0

    # Risk metrics
    max_drawdown_pct: float = 0.0
    max_drawdown_usd: float = 0.0
    calmar_ratio: float = 0.0
    sharpe_ratio: float = 0.0

    # Consecutive trades
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Time metrics
    avg_trade_duration: float = 0.0  # in candles/bars
    avg_win_duration: float = 0.0
    avg_loss_duration: float = 0.0

    # By direction
    long_trades: int = 0
    short_trades: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0

    # By exit reason
    sl_exits: int = 0
    tp_exits: int = 0
    timeout_exits: int = 0
    rollover_exits: int = 0

    # By confluence
    avg_confluence_score: float = 0.0
    trades_with_fvg: int = 0
    trades_with_ob: int = 0
    trades_with_bos: int = 0

    # Validation
    passes_validation: bool = False


def calculate_metrics(
    trades: List[Dict[str, Any]],
    equity_curve: List[float] = None,
    mtm_equity_curve: List[float] = None,
    initial_capital: float = 10000.0,
    cost_stats: Dict[str, float] = None,
) -> BacktestMetrics:
    """
    Calculate comprehensive performance metrics from trade list.

    Args:
        trades: List of trade dictionaries
        equity_curve: List of equity values over time
        mtm_equity_curve: Mark-to-market equity curve
        initial_capital: Starting capital
        cost_stats: Optional cost breakdown from engine

    Returns:
        BacktestMetrics object with all calculations
    """
    metrics = BacktestMetrics()

    if not trades:
        return metrics

    df = pd.DataFrame(trades)

    # Basic counts
    metrics.total_trades = len(df)

    # Use net_pnl if available, fall back to pnl_usd
    pnl_col = "net_pnl" if "net_pnl" in df.columns else "pnl_usd"

    metrics.winning_trades = len(df[df[pnl_col] > 0])
    metrics.losing_trades = len(df[df[pnl_col] < 0])
    metrics.breakeven_trades = len(df[df[pnl_col] == 0])

    # Win rate
    metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0

    # R-multiples
    r_multiples = df["r_multiple"]
    metrics.avg_r_multiple = r_multiples.mean()
    metrics.median_r_multiple = r_multiples.median()
    metrics.max_r_multiple = r_multiples.max()
    metrics.min_r_multiple = r_multiples.min()

    winners = df[df["r_multiple"] > 0]
    losers = df[df["r_multiple"] <= 0]
    metrics.avg_win_r = winners["r_multiple"].mean() if len(winners) > 0 else 0
    metrics.avg_loss_r = losers["r_multiple"].mean() if len(losers) > 0 else 0

    # Expectancy (same as avg R-multiple)
    metrics.expectancy = metrics.avg_r_multiple

    # P&L with gross/net breakdown
    if "gross_pnl" in df.columns:
        metrics.gross_pnl = df["gross_pnl"].sum()
    if "net_pnl" in df.columns:
        metrics.net_pnl = df["net_pnl"].sum()
    else:
        metrics.net_pnl = df["pnl_usd"].sum()

    metrics.total_pnl = metrics.net_pnl  # Backwards compat

    metrics.gross_profit = df[df[pnl_col] > 0][pnl_col].sum()
    metrics.gross_loss = abs(df[df[pnl_col] < 0][pnl_col].sum())
    metrics.profit_factor = metrics.gross_profit / metrics.gross_loss if metrics.gross_loss > 0 else 999.0
    metrics.avg_pnl_per_trade = metrics.net_pnl / metrics.total_trades if metrics.total_trades > 0 else 0

    # Execution costs
    if "spread_cost" in df.columns:
        metrics.total_spread_cost = df["spread_cost"].sum()
    if "slippage_cost" in df.columns:
        metrics.total_slippage_cost = df["slippage_cost"].sum()
    if "commission_cost" in df.columns:
        metrics.total_commission_cost = df["commission_cost"].sum()
    if "total_costs" in df.columns:
        metrics.total_costs = df["total_costs"].sum()
    else:
        metrics.total_costs = metrics.total_spread_cost + metrics.total_slippage_cost + metrics.total_commission_cost

    # Override with cost_stats if provided (more accurate)
    if cost_stats:
        metrics.total_spread_cost = cost_stats.get("total_spread_cost", metrics.total_spread_cost)
        metrics.total_slippage_cost = cost_stats.get("total_slippage_cost", metrics.total_slippage_cost)
        metrics.total_commission_cost = cost_stats.get("total_commission_cost", metrics.total_commission_cost)
        metrics.total_costs = cost_stats.get("total_costs", metrics.total_costs)
        if "gross_pnl" in cost_stats:
            metrics.gross_pnl = cost_stats["gross_pnl"]
        if "net_pnl" in cost_stats:
            metrics.net_pnl = cost_stats["net_pnl"]

    metrics.cost_per_trade = metrics.total_costs / metrics.total_trades if metrics.total_trades > 0 else 0

    # Compute trading days for ratio annualization
    if 'timestamp' in df.columns and len(df) > 1:
        trading_days = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days or 1
    else:
        trading_days = max(len(df), 1)

    # Drawdown - prefer MTM equity curve
    dd_curve = mtm_equity_curve if mtm_equity_curve else equity_curve
    if dd_curve:
        equity = pd.Series(dd_curve)
        rolling_max = equity.cummax()
        drawdown_usd = rolling_max - equity
        drawdown_pct = drawdown_usd / rolling_max

        metrics.max_drawdown_pct = drawdown_pct.max()
        metrics.max_drawdown_usd = drawdown_usd.max()

        # Calmar ratio (annualized return / max drawdown)
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        annualized_return = (1 + total_return) ** (252 / trading_days) - 1
        metrics.calmar_ratio = annualized_return / metrics.max_drawdown_pct if metrics.max_drawdown_pct > 0 else 0

    # Sharpe ratio from per-trade USD returns (annualized)
    if dd_curve and pnl_col in df.columns and len(df) > 1:
        trade_returns = df[pnl_col] / equity.iloc[0]
        avg_ret = trade_returns.mean()
        std_ret = trade_returns.std()
        if std_ret > 0:
            trades_per_year = len(df) / max(trading_days / 252, 1/252)
            metrics.sharpe_ratio = (avg_ret / std_ret) * (trades_per_year ** 0.5)
        else:
            metrics.sharpe_ratio = 0

    # Consecutive wins/losses
    metrics.max_consecutive_wins, metrics.max_consecutive_losses = _calculate_consecutive(df, pnl_col)

    # By direction
    long_trades = df[df["direction"] == "LONG"]
    short_trades = df[df["direction"] == "SHORT"]

    metrics.long_trades = len(long_trades)
    metrics.short_trades = len(short_trades)

    if len(long_trades) > 0:
        metrics.long_win_rate = len(long_trades[long_trades[pnl_col] > 0]) / len(long_trades)
    if len(short_trades) > 0:
        metrics.short_win_rate = len(short_trades[short_trades[pnl_col] > 0]) / len(short_trades)

    # By exit reason
    if "exit_reason" in df.columns:
        metrics.sl_exits = len(df[df["exit_reason"] == "SL"])
        metrics.tp_exits = len(df[df["exit_reason"] == "TP"])
        metrics.timeout_exits = len(df[df["exit_reason"] == "TIMEOUT"])
        metrics.rollover_exits = len(df[df["exit_reason"] == "ROLLOVER"])

    # By confluence
    if "confluence_score" in df.columns:
        metrics.avg_confluence_score = df["confluence_score"].mean()
    if "fvg_confluence" in df.columns:
        metrics.trades_with_fvg = df["fvg_confluence"].sum()
    if "ob_confluence" in df.columns:
        metrics.trades_with_ob = df["ob_confluence"].sum()
    if "bos_confirmed" in df.columns:
        metrics.trades_with_bos = df["bos_confirmed"].sum()

    # Validation
    metrics.passes_validation = (
        metrics.total_trades >= VALIDATION["min_trades"] and
        metrics.expectancy >= VALIDATION["min_expectancy_r"] and
        metrics.max_drawdown_pct <= VALIDATION["max_drawdown_pct"]
    )

    return metrics


def _calculate_consecutive(df: pd.DataFrame, pnl_col: str = "pnl_usd") -> tuple:
    """Calculate max consecutive wins and losses."""
    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0

    for _, trade in df.iterrows():
        if trade[pnl_col] > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)

    return max_wins, max_losses


def calculate_monthly_returns(
    trades: List[Dict[str, Any]],
    initial_capital: float = 10000.0,
) -> pd.DataFrame:
    """
    Calculate monthly return statistics.

    Args:
        trades: List of trade dictionaries
        initial_capital: Starting capital

    Returns:
        DataFrame with monthly statistics
    """
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["month"] = df["entry_time"].dt.to_period("M")

    # Use net_pnl if available
    pnl_col = "net_pnl" if "net_pnl" in df.columns else "pnl_usd"

    agg_dict = {
        pnl_col: ["sum", "count", "mean"],
        "r_multiple": "mean",
    }

    # Add cost column if available
    if "total_costs" in df.columns:
        agg_dict["total_costs"] = "sum"

    monthly = df.groupby("month").agg(agg_dict).round(2)

    # Flatten column names
    monthly.columns = ["_".join(col).strip("_") for col in monthly.columns.values]
    monthly = monthly.rename(columns={
        f"{pnl_col}_sum": "total_pnl",
        f"{pnl_col}_count": "trade_count",
        f"{pnl_col}_mean": "avg_pnl",
        "r_multiple_mean": "avg_r",
    })

    # Calculate monthly returns
    cumulative = [initial_capital]
    for pnl in monthly["total_pnl"]:
        cumulative.append(cumulative[-1] + pnl)

    monthly["starting_equity"] = cumulative[:-1]
    monthly["ending_equity"] = cumulative[1:]
    monthly["return_pct"] = ((monthly["ending_equity"] - monthly["starting_equity"]) / monthly["starting_equity"] * 100).round(2)

    return monthly


def calculate_session_performance(
    trades: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """
    Calculate performance by trading session.

    Sessions:
    - Asian: 00:00-08:00 UTC
    - London: 08:00-16:00 UTC
    - NY: 13:00-21:00 UTC

    Args:
        trades: List of trade dictionaries

    Returns:
        Dictionary with session statistics
    """
    if not trades:
        return {}

    df = pd.DataFrame(trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["hour"] = df["entry_time"].dt.hour

    # Use net_pnl if available
    pnl_col = "net_pnl" if "net_pnl" in df.columns else "pnl_usd"

    def get_session(hour):
        if 0 <= hour < 8:
            return "Asian"
        elif 8 <= hour < 13:
            return "London"
        elif 13 <= hour < 21:
            return "NY"
        else:
            return "Off-hours"

    df["session"] = df["hour"].apply(get_session)

    sessions = {}
    for session in ["Asian", "London", "NY", "Off-hours"]:
        session_trades = df[df["session"] == session]
        if len(session_trades) > 0:
            wins = len(session_trades[session_trades[pnl_col] > 0])
            sessions[session] = {
                "trades": len(session_trades),
                "win_rate": round(wins / len(session_trades) * 100, 1),
                "avg_r": round(session_trades["r_multiple"].mean(), 3),
                "total_pnl": round(session_trades[pnl_col].sum(), 2),
            }

    return sessions


def calculate_htf_bias_performance(
    trades: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """
    Calculate performance by HTF bias alignment.

    Args:
        trades: List of trade dictionaries

    Returns:
        Dictionary with bias statistics
    """
    if not trades:
        return {}

    df = pd.DataFrame(trades)

    if "htf_bias_primary" not in df.columns:
        return {}

    # Use net_pnl if available
    pnl_col = "net_pnl" if "net_pnl" in df.columns else "pnl_usd"

    biases = {}
    for bias in df["htf_bias_primary"].unique():
        if pd.isna(bias) or bias == "":
            continue
        bias_trades = df[df["htf_bias_primary"] == bias]
        if len(bias_trades) > 0:
            wins = len(bias_trades[bias_trades[pnl_col] > 0])
            biases[bias] = {
                "trades": len(bias_trades),
                "win_rate": round(wins / len(bias_trades) * 100, 1),
                "avg_r": round(bias_trades["r_multiple"].mean(), 3),
                "total_pnl": round(bias_trades[pnl_col].sum(), 2),
            }

    return biases


def generate_report(metrics: BacktestMetrics) -> str:
    """Generate a text report of backtest results."""
    lines = [
        "=" * 60,
        "AMD STRATEGY BACKTEST REPORT",
        "=" * 60,
        "",
        "TRADE STATISTICS",
        "-" * 40,
        f"Total Trades:     {metrics.total_trades}",
        f"Winning Trades:   {metrics.winning_trades}",
        f"Losing Trades:    {metrics.losing_trades}",
        f"Win Rate:         {metrics.win_rate:.1%}",
        "",
        "R-MULTIPLE ANALYSIS",
        "-" * 40,
        f"Avg R-Multiple:   {metrics.avg_r_multiple:.3f}",
        f"Median R:         {metrics.median_r_multiple:.3f}",
        f"Best Trade:       {metrics.max_r_multiple:.2f}R",
        f"Worst Trade:      {metrics.min_r_multiple:.2f}R",
        f"Avg Win:          {metrics.avg_win_r:.2f}R",
        f"Avg Loss:         {metrics.avg_loss_r:.2f}R",
        f"EXPECTANCY:       {metrics.expectancy:.3f}R",
        "",
        "PROFIT & LOSS",
        "-" * 40,
        f"Gross P&L:        ${metrics.gross_pnl:,.2f}",
        f"Net P&L:          ${metrics.net_pnl:,.2f}",
        f"Gross Profit:     ${metrics.gross_profit:,.2f}",
        f"Gross Loss:       ${metrics.gross_loss:,.2f}",
        f"Profit Factor:    {metrics.profit_factor:.2f}",
        f"Avg P&L/Trade:    ${metrics.avg_pnl_per_trade:.2f}",
        "",
        "EXECUTION COSTS",
        "-" * 40,
        f"Total Spread:     ${metrics.total_spread_cost:,.2f}",
        f"Total Slippage:   ${metrics.total_slippage_cost:,.2f}",
        f"Total Commission: ${metrics.total_commission_cost:,.2f}",
        f"Total Costs:      ${metrics.total_costs:,.2f}",
        f"Cost/Trade:       ${metrics.cost_per_trade:.2f}",
        "",
        "RISK METRICS",
        "-" * 40,
        f"Max Drawdown:     {metrics.max_drawdown_pct:.2%}",
        f"Max DD (USD):     ${metrics.max_drawdown_usd:,.2f}",
        f"Sharpe Ratio:     {metrics.sharpe_ratio:.2f}",
        "",
        "STREAKS",
        "-" * 40,
        f"Max Consec Wins:  {metrics.max_consecutive_wins}",
        f"Max Consec Losses:{metrics.max_consecutive_losses}",
        "",
        "BY DIRECTION",
        "-" * 40,
        f"Long Trades:      {metrics.long_trades} ({metrics.long_win_rate:.1%} win)",
        f"Short Trades:     {metrics.short_trades} ({metrics.short_win_rate:.1%} win)",
        "",
        "EXIT REASONS",
        "-" * 40,
        f"Stop Loss:        {metrics.sl_exits}",
        f"Take Profit:      {metrics.tp_exits}",
        f"Timeout:          {metrics.timeout_exits}",
        f"Rollover:         {metrics.rollover_exits}",
        "",
        "CONFLUENCE",
        "-" * 40,
        f"Avg Score:        {metrics.avg_confluence_score:.2f}",
        f"With FVG:         {metrics.trades_with_fvg}",
        f"With OB:          {metrics.trades_with_ob}",
        f"With BOS:         {metrics.trades_with_bos}",
        "",
        "VALIDATION",
        "-" * 40,
        f"Min Trades (500): {'PASS' if metrics.total_trades >= 500 else 'FAIL'}",
        f"Expectancy (0.2R):{'PASS' if metrics.expectancy >= 0.2 else 'FAIL'}",
        f"Max DD (15%):     {'PASS' if metrics.max_drawdown_pct <= 0.15 else 'FAIL'}",
        "",
        f"OVERALL: {'PASS' if metrics.passes_validation else 'FAIL'}",
        "=" * 60,
    ]

    return "\n".join(lines)


def generate_funnel_report(funnel_stats: Dict[str, int]) -> str:
    """Generate a funnel analysis report showing pattern filtering."""
    total_consol = funnel_stats.get("consolidations_found", 0)

    lines = [
        "",
        "=" * 60,
        "PATTERN FUNNEL ANALYSIS",
        "=" * 60,
        "",
        f"Consolidations Found:    {total_consol}",
        "-" * 40,
        f"  -> No Manipulation:    {funnel_stats.get('no_manipulation', 0)}",
        f"  -> No Distribution:    {funnel_stats.get('no_distribution', 0)}",
        f"  -> Weak Distribution:  {funnel_stats.get('no_distribution_follow_through', 0)}",
        f"  -> No BOS:             {funnel_stats.get('no_bos', 0)}",
        f"  -> No Entry/Retest:    {funnel_stats.get('no_entry_retest', 0)}",
        f"  -> Entry Too Late:     {funnel_stats.get('entry_too_late', 0)}",
        f"  -> Short Filtered:     {funnel_stats.get('short_filtered', 0)}",
        f"  -> Risk Invalid:       {funnel_stats.get('risk_invalid', 0)}",
        f"  -> Pattern Duplicates: {funnel_stats.get('pattern_duplicates', 0)}",
        "",
        "FILTER REJECTIONS",
        "-" * 40,
        f"  -> Session/Killzone:   {funnel_stats.get('filtered_session', 0)}",
        f"  -> News Blackout:      {funnel_stats.get('filtered_news', 0)}",
        f"  -> HTF Bias:           {funnel_stats.get('filtered_htf_bias', 0)}",
        f"  -> Key Levels:         {funnel_stats.get('filtered_key_levels', 0)}",
        f"  -> Volume:             {funnel_stats.get('filtered_volume', 0)}",
        f"  -> Fundamentals:       {funnel_stats.get('filtered_fundamentals', 0)}",
        f"  -> Daily Limit:        {funnel_stats.get('filtered_daily_limit', 0)}",
        f"  -> Cooldown:           {funnel_stats.get('filtered_cooldown', 0)}",
        f"  -> Near Rollover:      {funnel_stats.get('filtered_rollover', 0)}",
        f"  -> Fill Not Triggered: {funnel_stats.get('fill_not_triggered', 0)}",
        "",
        f"ENTRIES EXECUTED:        {funnel_stats.get('entries_executed', 0)}",
        "=" * 60,
    ]

    return "\n".join(lines)


def generate_amd_conformity_report(amd_conformity: Dict[str, float]) -> str:
    """Generate a report of AMD pattern conformity statistics."""
    if not amd_conformity:
        return ""

    lines = [
        "",
        "=" * 60,
        "AMD CONFORMITY",
        "=" * 60,
        "",
        f"Consolidations Found:              {amd_conformity.get('consolidations_found', 0)}",
        f"Manipulations Found:               {amd_conformity.get('manipulations_found', 0)}",
        f"Distributions Found:               {amd_conformity.get('distributions_found', 0)}",
        f"Distributions w/ BOS+Confluence:   {amd_conformity.get('distributions_with_bos_confluence', 0)}",
        "",
        f"Consolidation -> Manipulation:     {amd_conformity.get('consolidation_to_manipulation_pct', 0.0):.2f}%",
        f"Manipulation -> Distribution:      {amd_conformity.get('manipulation_to_distribution_pct', 0.0):.2f}%",
        f"Distribution w/ BOS+Confluence:    {amd_conformity.get('distribution_with_bos_confluence_pct', 0.0):.2f}%",
        f"Avg Bars Manipulation->Distribution: {amd_conformity.get('avg_bars_manipulation_to_distribution', 0.0):.2f}",
        "=" * 60,
    ]

    return "\n".join(lines)
