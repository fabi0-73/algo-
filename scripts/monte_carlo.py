"""
Monte Carlo Ruin Simulation

Reshuffles the trade sequence from a completed backtest to estimate:
- Drawdown distribution (P5, P25, P50, P75, P95)
- Ruin probability at various thresholds
- Expected time to double vs time to ruin
- Consecutive loss streak probabilities

Usage:
    python scripts/monte_carlo.py --report reports/backtest_aae991eb
    python scripts/monte_carlo.py --report reports/backtest_aae991eb --simulations 20000
    python scripts/monte_carlo.py --report reports/backtest_aae991eb --charts
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import logging
import numpy as np
from typing import List, Dict, Any, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_trades_from_report(report_dir: str) -> List[Dict[str, Any]]:
    """Load trade list from a backtest results.json file."""
    results_path = os.path.join(report_dir, "results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"results.json not found in {report_dir}")

    with open(results_path, "r") as f:
        data = json.load(f)

    trades = data.get("trades", [])
    if not trades:
        raise ValueError("No trades found in results.json")

    return trades


def extract_r_multiples(trades: List[Dict[str, Any]]) -> np.ndarray:
    """Extract R-multiple array from trade list."""
    r_values = []
    for t in trades:
        r = t.get("r_multiple", 0.0)
        r_values.append(r)
    return np.array(r_values, dtype=np.float64)


def run_simulation(
    r_multiples: np.ndarray,
    risk_pct: float,
    initial_capital: float,
    num_simulations: int = 10000,
    ruin_thresholds: List[float] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation by reshuffling trade sequence.

    Args:
        r_multiples: Array of R-multiples from actual trades
        risk_pct: Risk per trade as decimal (e.g. 0.005 for 0.5%)
        initial_capital: Starting account balance
        num_simulations: Number of reshuffled runs
        ruin_thresholds: Drawdown thresholds to check ruin probability
        seed: Random seed for reproducibility

    Returns:
        Dictionary with simulation results
    """
    if ruin_thresholds is None:
        ruin_thresholds = [0.20, 0.30, 0.50, 0.80]

    rng = np.random.default_rng(seed)
    n_trades = len(r_multiples)

    max_drawdowns = np.zeros(num_simulations)
    final_capitals = np.zeros(num_simulations)
    ruin_counts = {t: 0 for t in ruin_thresholds}
    time_to_double = []
    time_to_ruin = {t: [] for t in ruin_thresholds}

    all_equity_curves = np.zeros((num_simulations, n_trades + 1))
    all_equity_curves[:, 0] = initial_capital

    max_consec_losses_all = np.zeros(num_simulations, dtype=int)

    for sim in range(num_simulations):
        shuffled = rng.permutation(r_multiples)
        equity = initial_capital
        peak = initial_capital
        max_dd = 0.0
        doubled = False
        consec_losses = 0
        max_consec = 0

        for j, r in enumerate(shuffled):
            pnl = equity * risk_pct * r
            equity += pnl
            all_equity_curves[sim, j + 1] = equity

            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

            if r <= 0:
                consec_losses += 1
                if consec_losses > max_consec:
                    max_consec = consec_losses
            else:
                consec_losses = 0

            if not doubled and equity >= initial_capital * 2:
                doubled = True
                time_to_double.append(j + 1)

            for threshold in ruin_thresholds:
                if dd >= threshold and len(time_to_ruin[threshold]) < sim + 1:
                    time_to_ruin[threshold].append(j + 1)

        max_drawdowns[sim] = max_dd
        final_capitals[sim] = equity
        max_consec_losses_all[sim] = max_consec

        for threshold in ruin_thresholds:
            if max_dd >= threshold:
                ruin_counts[threshold] += 1

    dd_percentiles = {
        "P5": float(np.percentile(max_drawdowns, 5)),
        "P25": float(np.percentile(max_drawdowns, 25)),
        "P50": float(np.percentile(max_drawdowns, 50)),
        "P75": float(np.percentile(max_drawdowns, 75)),
        "P95": float(np.percentile(max_drawdowns, 95)),
    }

    ruin_probabilities = {
        f"{int(t * 100)}%": round(ruin_counts[t] / num_simulations * 100, 2)
        for t in ruin_thresholds
    }

    equity_percentiles = {}
    for pct_label, pct_val in [("P5", 5), ("P25", 25), ("P50", 50), ("P75", 75), ("P95", 95)]:
        equity_percentiles[pct_label] = np.percentile(all_equity_curves, pct_val, axis=0).tolist()

    avg_time_to_double = float(np.mean(time_to_double)) if time_to_double else float("inf")
    pct_doubled = len(time_to_double) / num_simulations * 100

    avg_time_to_ruin = {}
    for threshold in ruin_thresholds:
        times = time_to_ruin[threshold]
        key = f"{int(threshold * 100)}%"
        avg_time_to_ruin[key] = float(np.mean(times)) if times else float("inf")

    consec_loss_dist = {}
    for n in range(1, 16):
        pct = float(np.sum(max_consec_losses_all >= n) / num_simulations * 100)
        consec_loss_dist[f"{n}+"] = round(pct, 2)

    return {
        "num_simulations": num_simulations,
        "num_trades": n_trades,
        "risk_pct": risk_pct,
        "initial_capital": initial_capital,
        "drawdown_distribution": dd_percentiles,
        "ruin_probability": ruin_probabilities,
        "final_capital_distribution": {
            "P5": float(np.percentile(final_capitals, 5)),
            "P25": float(np.percentile(final_capitals, 25)),
            "P50": float(np.percentile(final_capitals, 50)),
            "P75": float(np.percentile(final_capitals, 75)),
            "P95": float(np.percentile(final_capitals, 95)),
        },
        "time_to_double": {
            "avg_trades": round(avg_time_to_double, 1),
            "pct_doubled": round(pct_doubled, 2),
        },
        "avg_time_to_ruin": avg_time_to_ruin,
        "consecutive_loss_probability": consec_loss_dist,
        "equity_percentile_curves": equity_percentiles,
        "max_drawdowns": max_drawdowns.tolist(),
    }


def print_results(results: Dict[str, Any]) -> None:
    """Print formatted Monte Carlo results."""
    print("\n" + "=" * 60)
    print("MONTE CARLO SIMULATION RESULTS")
    print("=" * 60)
    print(f"Simulations:      {results['num_simulations']:,}")
    print(f"Trades per sim:   {results['num_trades']}")
    print(f"Risk per trade:   {results['risk_pct'] * 100:.1f}%")
    print(f"Initial capital:  ${results['initial_capital']:,.2f}")

    print("\nDRAWDOWN DISTRIBUTION:")
    for k, v in results["drawdown_distribution"].items():
        print(f"  {k}: {v * 100:.1f}%")

    print("\nRUIN PROBABILITY:")
    for k, v in results["ruin_probability"].items():
        print(f"  DD >= {k}: {v:.1f}%")

    print("\nFINAL CAPITAL DISTRIBUTION:")
    for k, v in results["final_capital_distribution"].items():
        print(f"  {k}: ${v:,.2f}")

    print("\nTIME TO DOUBLE:")
    ttd = results["time_to_double"]
    if ttd["avg_trades"] < float("inf"):
        print(f"  Avg trades:     {ttd['avg_trades']:.0f}")
    else:
        print(f"  Avg trades:     Never")
    print(f"  % that doubled: {ttd['pct_doubled']:.1f}%")

    print("\nAVG TIME TO RUIN (trades):")
    for k, v in results["avg_time_to_ruin"].items():
        if v < float("inf"):
            print(f"  DD >= {k}: {v:.0f} trades")
        else:
            print(f"  DD >= {k}: Never")

    print("\nCONSECUTIVE LOSS PROBABILITY:")
    for k, v in results["consecutive_loss_probability"].items():
        if v > 0.1:
            print(f"  {k} losses: {v:.1f}%")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo ruin simulation")
    parser.add_argument(
        "--report", type=str, required=True,
        help="Path to backtest report directory (e.g. reports/backtest_aae991eb)",
    )
    parser.add_argument("--simulations", type=int, default=10000, help="Number of simulations")
    parser.add_argument("--risk-pct", type=float, default=None, help="Risk per trade (default from config)")
    parser.add_argument("--capital", type=float, default=None, help="Initial capital (default from config)")
    parser.add_argument("--charts", action="store_true", help="Generate chart images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")

    args = parser.parse_args()

    from config import RISK_MODEL, BACKTEST

    risk_pct = args.risk_pct or RISK_MODEL.get("risk_pct_per_trade_default", 0.005)
    capital = args.capital or BACKTEST.get("initial_capital", 100.0)

    logger.info(f"Loading trades from {args.report}")
    trades = load_trades_from_report(args.report)
    r_multiples = extract_r_multiples(trades)
    logger.info(f"Loaded {len(r_multiples)} trades, avg R = {r_multiples.mean():.3f}")

    logger.info(f"Running {args.simulations:,} simulations...")
    results = run_simulation(
        r_multiples,
        risk_pct=risk_pct,
        initial_capital=capital,
        num_simulations=args.simulations,
        seed=args.seed,
    )

    print_results(results)

    if args.save:
        save_data = {k: v for k, v in results.items() if k not in ("equity_percentile_curves", "max_drawdowns")}
        out_path = os.path.join(args.report, "monte_carlo.json")
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2)
        logger.info(f"Results saved to {out_path}")

    if args.charts:
        try:
            from src.visualization.charts import (
                plot_monte_carlo_fan,
                plot_drawdown_distribution,
                plot_ruin_probability,
            )
            save_dir = args.report
            plot_monte_carlo_fan(
                results["equity_percentile_curves"],
                save_path=os.path.join(save_dir, "monte_carlo_fan.png"),
                show=False,
            )
            plot_drawdown_distribution(
                results["max_drawdowns"],
                save_path=os.path.join(save_dir, "monte_carlo_dd_distribution.png"),
                show=False,
            )
            plot_ruin_probability(
                results["ruin_probability"],
                save_path=os.path.join(save_dir, "monte_carlo_ruin.png"),
                show=False,
            )
            logger.info(f"Charts saved to {save_dir}")
        except ImportError as e:
            logger.warning(f"Could not generate charts: {e}")


if __name__ == "__main__":
    main()
