"""
Visualization Module
Charts and visual reports for backtest analysis.
Supports MTM equity curves, cost breakdown, and filter funnel analysis.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

from src.backtest.metrics import BacktestMetrics, calculate_metrics, generate_funnel_report


def plot_equity_curve(
    equity_curve: List[float],
    mtm_equity_curve: List[float] = None,
    trades: List[Dict[str, Any]] = None,
    title: str = "AMD Strategy Equity Curve",
    save_path: str = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot equity curve with optional MTM overlay and trade markers.

    Args:
        equity_curve: List of equity values over time (closed trades only)
        mtm_equity_curve: Mark-to-market equity curve (includes unrealized)
        trades: Optional list of trades to mark on chart
        title: Chart title
        save_path: Path to save the figure
        show: Whether to display the figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    # Main equity curve
    ax1 = axes[0]
    equity = pd.Series(equity_curve)
    ax1.plot(equity, linewidth=1.5, color="#2E86AB", label="Realized Equity")

    # Plot MTM curve if available
    if mtm_equity_curve:
        mtm = pd.Series(mtm_equity_curve)
        ax1.plot(mtm, linewidth=1, color="#A23B72", alpha=0.7, label="MTM Equity")

    # Use MTM for drawdown if available, otherwise realized
    dd_equity = pd.Series(mtm_equity_curve) if mtm_equity_curve else equity
    rolling_max = dd_equity.cummax()
    drawdown = (rolling_max - dd_equity) / rolling_max * 100

    ax1.fill_between(range(len(dd_equity)), dd_equity, rolling_max,
                     alpha=0.3, color="#E94F37", label="Drawdown")

    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.set_xlabel("Bar")
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Starting and ending equity
    ax1.axhline(y=equity.iloc[0], color="gray", linestyle="--", alpha=0.5)
    ax1.annotate(f"Start: ${equity.iloc[0]:,.0f}",
                 xy=(0, equity.iloc[0]), fontsize=9)
    ax1.annotate(f"End: ${equity.iloc[-1]:,.0f}",
                 xy=(len(equity)-1, equity.iloc[-1]), fontsize=9,
                 ha="right")

    # Drawdown subplot
    ax2 = axes[1]
    ax2.fill_between(range(len(drawdown)), 0, drawdown,
                     color="#E94F37", alpha=0.5)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Bar")
    ax2.set_ylim(0, drawdown.max() * 1.1 if drawdown.max() > 0 else 10)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)

    # Max drawdown annotation
    max_dd_idx = drawdown.idxmax()
    max_dd_val = drawdown.max()
    ax2.annotate(f"Max DD: {max_dd_val:.1f}%",
                 xy=(max_dd_idx, max_dd_val),
                 xytext=(max_dd_idx + len(equity)*0.05, max_dd_val * 0.8),
                 fontsize=9, color="#E94F37",
                 arrowprops=dict(arrowstyle="->", color="#E94F37"))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_trades_on_chart(
    df: pd.DataFrame,
    trades: List[Dict[str, Any]],
    start_idx: int = 0,
    end_idx: int = None,
    title: str = "AMD Strategy Trades",
    save_path: str = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot candlestick-style chart with trade markers.

    Args:
        df: DataFrame with OHLC data
        trades: List of trade dictionaries
        start_idx: Starting index to plot
        end_idx: Ending index to plot
        title: Chart title
        save_path: Path to save the figure
        show: Whether to display the figure

    Returns:
        Matplotlib figure
    """
    if end_idx is None:
        end_idx = len(df)

    plot_df = df.iloc[start_idx:end_idx].copy()

    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot candlesticks (simplified as bars)
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        color = "#26A69A" if row["close"] >= row["open"] else "#EF5350"

        # High-low line
        ax.plot([i, i], [row["low"], row["high"]], color=color, linewidth=0.5)

        # Body
        body_bottom = min(row["open"], row["close"])
        body_height = abs(row["close"] - row["open"])
        ax.bar(i, body_height, bottom=body_bottom, width=0.6, color=color)

    # Plot trades
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])

        # Use net_pnl if available
        pnl_col = "net_pnl" if "net_pnl" in trades_df.columns else "pnl_usd"

        for _, trade in trades_df.iterrows():
            # Find matching candle indices
            entry_matches = plot_df[plot_df["timestamp"] == trade["entry_time"]]
            if not entry_matches.empty:
                entry_idx = plot_df.index.get_loc(entry_matches.index[0])

                # Entry marker
                marker = "^" if trade["direction"] == "LONG" else "v"
                color = "#26A69A" if trade[pnl_col] > 0 else "#EF5350"
                ax.scatter(entry_idx, trade["entry_price"],
                          marker=marker, s=100, c=color,
                          edgecolors="black", linewidths=0.5, zorder=5)

                # SL and TP lines
                if "exit_time" in trade and pd.notna(trade.get("exit_time")):
                    exit_matches = plot_df[plot_df["timestamp"] == trade["exit_time"]]
                    if not exit_matches.empty:
                        exit_idx = plot_df.index.get_loc(exit_matches.index[0])

                        # Draw SL line
                        ax.hlines(y=trade["sl_price"], xmin=entry_idx, xmax=exit_idx,
                                 colors="#EF5350", linestyles="dashed", alpha=0.5)

                        # Draw TP line
                        ax.hlines(y=trade["tp_price"], xmin=entry_idx, xmax=exit_idx,
                                 colors="#26A69A", linestyles="dashed", alpha=0.5)

                        # Exit marker
                        ax.scatter(exit_idx, trade["exit_price"],
                                  marker="x", s=80, c=color, linewidths=2, zorder=5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Bar")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_r_distribution(
    trades: List[Dict[str, Any]],
    title: str = "R-Multiple Distribution",
    save_path: str = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot histogram of R-multiples.

    Args:
        trades: List of trade dictionaries
        title: Chart title
        save_path: Path to save the figure
        show: Whether to display the figure

    Returns:
        Matplotlib figure
    """
    if not trades:
        return None

    df = pd.DataFrame(trades)
    r_multiples = df["r_multiple"]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Color by win/loss
    wins = r_multiples[r_multiples > 0]
    losses = r_multiples[r_multiples <= 0]

    bins = np.linspace(r_multiples.min() - 0.5, r_multiples.max() + 0.5, 30)

    ax.hist(wins, bins=bins, alpha=0.7, color="#26A69A", label=f"Wins ({len(wins)})")
    ax.hist(losses, bins=bins, alpha=0.7, color="#EF5350", label=f"Losses ({len(losses)})")

    # Mean line
    mean_r = r_multiples.mean()
    ax.axvline(x=mean_r, color="black", linestyle="--", linewidth=2,
               label=f"Mean: {mean_r:.2f}R")

    # Zero line
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("R-Multiple")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_monthly_performance(
    trades: List[Dict[str, Any]],
    title: str = "Monthly Performance",
    save_path: str = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot monthly P&L as bar chart.

    Args:
        trades: List of trade dictionaries
        title: Chart title
        save_path: Path to save the figure
        show: Whether to display the figure

    Returns:
        Matplotlib figure
    """
    if not trades:
        return None

    df = pd.DataFrame(trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["month"] = df["entry_time"].dt.to_period("M")

    # Use net_pnl if available
    pnl_col = "net_pnl" if "net_pnl" in df.columns else "pnl_usd"

    monthly = df.groupby("month").agg({
        pnl_col: "sum",
        "r_multiple": "mean",
    }).reset_index()

    monthly["month_str"] = monthly["month"].astype(str)

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ["#26A69A" if p > 0 else "#EF5350" for p in monthly[pnl_col]]
    bars = ax.bar(monthly["month_str"], monthly[pnl_col], color=colors)

    # Add value labels
    for bar, val in zip(bars, monthly[pnl_col]):
        height = bar.get_height()
        ax.annotate(f"${val:,.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -15),
                    textcoords="offset points",
                    ha="center", va="bottom" if height >= 0 else "top",
                    fontsize=8)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Net P&L ($)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_cost_breakdown(
    cost_stats: Dict[str, float],
    title: str = "Execution Cost Breakdown",
    save_path: str = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot execution cost breakdown as pie and bar charts.

    Args:
        cost_stats: Dictionary with cost statistics
        title: Chart title
        save_path: Path to save the figure
        show: Whether to display the figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Pie chart of cost components
    ax1 = axes[0]
    costs = {
        "Spread": cost_stats.get("total_spread_cost", 0),
        "Slippage": cost_stats.get("total_slippage_cost", 0),
        "Commission": cost_stats.get("total_commission_cost", 0),
    }

    # Filter out zero values
    costs = {k: v for k, v in costs.items() if v > 0}

    if costs:
        colors = ["#E94F37", "#F6BD60", "#84A59D"]
        ax1.pie(costs.values(), labels=costs.keys(), autopct='%1.1f%%',
                colors=colors[:len(costs)], startangle=90)
        ax1.set_title("Cost Distribution")
    else:
        ax1.text(0.5, 0.5, "No costs recorded", ha="center", va="center")
        ax1.axis("off")

    # Right: Gross vs Net P&L
    ax2 = axes[1]
    gross = cost_stats.get("gross_pnl", 0)
    net = cost_stats.get("net_pnl", 0)
    total_costs = cost_stats.get("total_costs", 0)

    categories = ["Gross P&L", "Costs", "Net P&L"]
    values = [gross, -total_costs, net]
    colors = ["#26A69A" if v > 0 else "#E94F37" for v in values]

    bars = ax2.bar(categories, values, color=colors)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.annotate(f"${val:,.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -15),
                    textcoords="offset points",
                    ha="center", va="bottom" if height >= 0 else "top",
                    fontsize=10, fontweight="bold")

    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.set_title("P&L Impact of Costs")
    ax2.set_ylabel("USD")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_funnel_analysis(
    funnel_stats: Dict[str, int],
    title: str = "Pattern Funnel Analysis",
    save_path: str = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot funnel chart showing pattern filtering stages.

    Args:
        funnel_stats: Dictionary with funnel statistics
        title: Chart title
        save_path: Path to save the figure
        show: Whether to display the figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Build funnel stages
    total_consol = funnel_stats.get("consolidations_found", 0)
    after_manip = total_consol - funnel_stats.get("no_manipulation", 0)
    after_dist = after_manip - funnel_stats.get("no_distribution", 0)
    after_bos = after_dist - funnel_stats.get("no_bos", 0)
    after_entry = after_bos - funnel_stats.get("no_entry_retest", 0)
    after_short = after_entry - funnel_stats.get("short_filtered", 0)
    after_risk = after_short - funnel_stats.get("risk_invalid", 0)
    after_dedup = after_risk - funnel_stats.get("pattern_duplicates", 0)

    # Filter stages
    filter_rej = (
        funnel_stats.get("filtered_session", 0) +
        funnel_stats.get("filtered_news", 0) +
        funnel_stats.get("filtered_htf_bias", 0) +
        funnel_stats.get("filtered_key_levels", 0) +
        funnel_stats.get("filtered_volume", 0) +
        funnel_stats.get("filtered_fundamentals", 0) +
        funnel_stats.get("filtered_daily_limit", 0) +
        funnel_stats.get("filtered_cooldown", 0) +
        funnel_stats.get("filtered_rollover", 0)
    )
    after_filters = after_dedup - filter_rej

    fill_rej = funnel_stats.get("fill_not_triggered", 0)
    executed = funnel_stats.get("entries_executed", 0)

    stages = [
        ("Consolidations", total_consol),
        ("After Manipulation", after_manip),
        ("After Distribution", after_dist),
        ("After BOS", after_bos),
        ("After Entry Check", after_entry),
        ("After Risk Check", after_risk),
        ("After Dedup", after_dedup),
        ("After Filters", after_filters),
        ("Executed", executed),
    ]

    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]

    # Create horizontal bar chart (funnel style)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(stages)))

    y_pos = np.arange(len(stages))[::-1]  # Reverse for funnel view
    bars = ax.barh(y_pos, values, color=colors)

    # Add labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)

    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax.annotate(f"{val:,}",
                    xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha="left", va="center",
                    fontsize=10, fontweight="bold")

    # Calculate and show conversion rates
    if total_consol > 0:
        conv_rate = executed / total_consol * 100
        ax.set_xlabel(f"Count (Conversion: {conv_rate:.2f}%)")
    else:
        ax.set_xlabel("Count")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def generate_report(
    results: Dict[str, Any],
    output_dir: str = "reports",
    show_charts: bool = False,
) -> str:
    """
    Generate full visual report with all charts.

    Args:
        results: Backtest results dictionary
        output_dir: Directory to save reports
        show_charts: Whether to display charts

    Returns:
        Path to report directory
    """
    os.makedirs(output_dir, exist_ok=True)

    backtest_id = results.get("backtest_id", "unknown")
    report_dir = os.path.join(output_dir, f"backtest_{backtest_id}")
    os.makedirs(report_dir, exist_ok=True)

    trades = results.get("trades", [])
    equity_curve = results.get("equity_curve", [])
    mtm_equity_curve = results.get("mtm_equity_curve", [])
    cost_stats = results.get("cost_stats", {})
    funnel_stats = results.get("funnel_stats", {})

    # Generate charts
    if equity_curve:
        plot_equity_curve(
            equity_curve,
            mtm_equity_curve=mtm_equity_curve,
            title=f"Equity Curve - Backtest {backtest_id}",
            save_path=os.path.join(report_dir, "equity_curve.png"),
            show=show_charts,
        )

    if trades:
        plot_r_distribution(
            trades,
            title=f"R-Multiple Distribution - Backtest {backtest_id}",
            save_path=os.path.join(report_dir, "r_distribution.png"),
            show=show_charts,
        )

        plot_monthly_performance(
            trades,
            title=f"Monthly Performance - Backtest {backtest_id}",
            save_path=os.path.join(report_dir, "monthly_performance.png"),
            show=show_charts,
        )

    if cost_stats:
        plot_cost_breakdown(
            cost_stats,
            title=f"Cost Breakdown - Backtest {backtest_id}",
            save_path=os.path.join(report_dir, "cost_breakdown.png"),
            show=show_charts,
        )

    if funnel_stats:
        plot_funnel_analysis(
            funnel_stats,
            title=f"Pattern Funnel - Backtest {backtest_id}",
            save_path=os.path.join(report_dir, "funnel_analysis.png"),
            show=show_charts,
        )

    # Generate text report
    metrics = calculate_metrics(
        trades,
        equity_curve,
        mtm_equity_curve,
        results.get("initial_capital", 10000),
        cost_stats
    )
    from src.backtest.metrics import generate_report as gen_text_report
    report_text = gen_text_report(metrics)

    # Add funnel report
    if funnel_stats:
        report_text += generate_funnel_report(funnel_stats)

    report_path = os.path.join(report_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"Report generated: {report_dir}")
    return report_dir
