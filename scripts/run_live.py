"""
Live Trading Runner

Main entry point for live AMD signal scanning.
Default mode is signal-only (no execution).

Usage:
    # Signal-only mode (default) -- prints signals to console as JSON
    python scripts/run_live.py

    # Signal-only with custom interval
    python scripts/run_live.py --interval 300

    # Single scan (no loop)
    python scripts/run_live.py --once

    # With custom balance for position sizing
    python scripts/run_live.py --balance 500
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import logging
import time
from datetime import datetime

from config import STRATEGY, BACKTEST, MT5_CONFIG
from src.live.signals import LiveSignalScanner
from src.live.monitor import LiveMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_latest_candles(symbol: str, timeframe: str, count: int = 500):
    """
    Fetch the latest candles from MT5 or database.

    Tries MT5 first; falls back to database if MT5 is not configured.
    """
    # Try MT5
    if MT5_CONFIG.get("login") and MT5_CONFIG["login"] != 0:
        try:
            from src.data.mt5_client import MT5Client
            client = MT5Client()
            df = client.get_candles(symbol, timeframe, count=count)
            if df is not None and not df.empty:
                logger.info(f"MT5: fetched {len(df)} candles for {symbol} {timeframe}")
                return df
        except Exception as e:
            logger.warning(f"MT5 fetch failed: {e}")

    # Fallback to database
    try:
        from src.data.db import Database
        db = Database()
        df = db.get_candles(symbol, timeframe)
        if df is not None and not df.empty:
            df = df.sort_values("timestamp").tail(count).reset_index(drop=True)
            logger.info(f"DB: fetched {len(df)} candles for {symbol} {timeframe}")
            return df
    except Exception as e:
        logger.warning(f"DB fetch failed: {e}")

    return None


def run_signal_loop(
    symbol: str,
    timeframe: str,
    balance: float,
    interval_seconds: int,
    once: bool = False,
    output_file: str = None,
):
    """
    Continuous signal scanning loop.

    Args:
        symbol: Trading instrument
        timeframe: Candle timeframe (e.g. "M5")
        balance: Account balance for position sizing
        interval_seconds: Seconds between scans
        once: If True, scan once and exit
        output_file: Optional file path to append signals as JSON lines
    """
    scanner = LiveSignalScanner(symbol=symbol, account_balance=balance)
    monitor = LiveMonitor(
        initial_balance=balance,
        daily_loss_limit_pct=0.01,
        max_account_dd_pct=0.15,
        max_trades_per_day=3,
    )

    logger.info("=" * 60)
    logger.info("AMD LIVE SIGNAL SCANNER")
    logger.info("=" * 60)
    logger.info(f"Symbol:    {symbol}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Balance:   ${balance:,.2f}")
    logger.info(f"Mode:      SIGNAL ONLY")
    logger.info(f"Interval:  {interval_seconds}s")
    logger.info("=" * 60)

    scan_count = 0

    while True:
        scan_count += 1
        now = datetime.utcnow()
        logger.info(f"Scan #{scan_count} at {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")

        can_trade, reason = monitor.can_trade(now)
        if not can_trade:
            logger.warning(f"Trading blocked: {reason}")
            status = monitor.status_summary()
            logger.info(f"Monitor: {json.dumps(status)}")
            if once:
                break
            time.sleep(interval_seconds)
            continue

        df = fetch_latest_candles(symbol, timeframe, count=500)
        if df is None or df.empty:
            logger.error("No candle data available")
            if once:
                break
            time.sleep(interval_seconds)
            continue

        signals = scanner.scan(df)

        if signals:
            for sig in signals:
                sig_dict = sig.to_dict()
                sig_json = json.dumps(sig_dict, indent=2)

                print("\n" + "=" * 60)
                print("SIGNAL DETECTED")
                print("=" * 60)
                print(sig_json)
                print("=" * 60)

                if output_file:
                    with open(output_file, "a") as f:
                        f.write(json.dumps(sig_dict) + "\n")

                logger.info(
                    f"SIGNAL: {sig.direction} {sig.symbol} @ {sig.entry_price} | "
                    f"SL: {sig.stop_loss} | TP: {sig.take_profit} | "
                    f"RR: {sig.risk_reward} | Conf: {sig.confluence_score} | "
                    f"Quality: {sig.confidence}"
                )
        else:
            logger.info("No signals detected")

        if once:
            break

        logger.info(f"Next scan in {interval_seconds}s...")
        time.sleep(interval_seconds)


def main():
    parser = argparse.ArgumentParser(
        description="AMD Live Signal Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Signal-only with default settings
  python scripts/run_live.py

  # Single scan, no loop
  python scripts/run_live.py --once

  # Custom balance and faster scanning
  python scripts/run_live.py --balance 500 --interval 60

  # Log signals to file
  python scripts/run_live.py --output signals.jsonl
        """,
    )

    parser.add_argument("--symbol", type=str, default=None, help="Symbol (default from config)")
    parser.add_argument("--timeframe", type=str, default=None, help="Timeframe (default from config)")
    parser.add_argument("--balance", type=float, default=None, help="Account balance")
    parser.add_argument("--interval", type=int, default=300, help="Scan interval in seconds (default 300)")
    parser.add_argument("--once", action="store_true", help="Single scan, no loop")
    parser.add_argument("--output", type=str, default=None, help="Output file for signal JSON lines")

    args = parser.parse_args()

    symbol = args.symbol or STRATEGY["symbol"]
    timeframe = args.timeframe or STRATEGY["timeframe"]
    balance = args.balance or BACKTEST["initial_capital"]

    run_signal_loop(
        symbol=symbol,
        timeframe=timeframe,
        balance=balance,
        interval_seconds=args.interval,
        once=args.once,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
