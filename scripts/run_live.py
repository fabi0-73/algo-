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

from config import (STRATEGY, BACKTEST, MT5_CONFIG, TELEGRAM,
                    SESSION_FILTER, DRAWDOWN_CONTROLS)
from src.live.signals import LiveSignalScanner
from src.live.monitor import LiveMonitor
from src.live.telegram_notifier import TelegramNotifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# One MT5 client for the whole scanner lifetime: re-initializing the terminal
# connection every poll is wasteful and bypasses the client's own reconnect
# logic. Reset to None on any fetch error so the next scan reconnects fresh.
_mt5_client = None


def _get_mt5_client():
    global _mt5_client
    if _mt5_client is None:
        from src.data.mt5_client import MT5Client
        _mt5_client = MT5Client()
    return _mt5_client


def fetch_latest_candles(symbol: str, timeframe: str, count: int = 500):
    """
    Fetch the latest candles from MT5 or database.

    Tries MT5 first; falls back to database if MT5 is not configured.
    """
    global _mt5_client
    # Try MT5
    if MT5_CONFIG.get("login") and MT5_CONFIG["login"] != 0:
        try:
            client = _get_mt5_client()
            df = client.get_candles(symbol, timeframe, count=count)
            if df is not None and not df.empty:
                # Closed-candle guard: MT5 position-0 bar is the currently
                # FORMING bar. Drop it unconditionally so every downstream
                # consumer (AMD scan-back and NY_IB close-confirmation) sees
                # completed bars only. Robust to local-vs-broker clock skew;
                # costs at most one bar of latency.
                df = df.iloc[:-1].reset_index(drop=True)
                logger.info(f"MT5: fetched {len(df)} closed candles for {symbol} {timeframe}")
                return df
        except Exception as e:
            logger.warning(f"MT5 fetch failed: {e} — will reconnect next scan")
            _mt5_client = None

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
    use_telegram: bool = None,
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
        use_telegram: If True, send signals to Telegram. If None, use TELEGRAM["enabled"] from config.
    """
    scanner = LiveSignalScanner(symbol=symbol, account_balance=balance)
    # Advisory in signal-only mode (no fills are recorded), but keep the
    # thresholds identical to the VALIDATED config — the old hardcoded
    # 1%/15% literals would halt at the backtest's RESUME threshold if
    # execution were ever wired in.
    monitor = LiveMonitor(
        initial_balance=balance,
        daily_loss_limit_pct=SESSION_FILTER.get("daily_loss_limit_pct", 0.008),
        max_account_dd_pct=DRAWDOWN_CONTROLS.get("max_account_dd_pct", 0.30),
        max_trades_per_day=SESSION_FILTER.get("max_trades_per_day", 3),
    )

    telegram_notifier = None
    if use_telegram if use_telegram is not None else TELEGRAM.get("enabled", False):
        token = TELEGRAM.get("bot_token", "").strip()
        chat_id = TELEGRAM.get("chat_id", "").strip()
        if token and chat_id:
            telegram_notifier = TelegramNotifier(bot_token=token, chat_id=chat_id)
            logger.info("Telegram: enabled")
        else:
            logger.warning("Telegram requested but TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing; skipping Telegram")

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
    last_bar_ts = None

    def do_scan() -> str:
        """One poll cycle. Returns 'ok' (keep looping) or 'stop' (fatal)."""
        nonlocal last_bar_ts
        now = datetime.utcnow()
        logger.info(f"Scan #{scan_count} at {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")

        can_trade, reason = monitor.can_trade(now)
        if not can_trade:
            logger.warning(f"Trading blocked: {reason}")
            status = monitor.status_summary()
            logger.info(f"Monitor: {json.dumps(status)}")
            return "ok"

        df = fetch_latest_candles(symbol, timeframe, count=500)
        if df is None or df.empty:
            logger.error("No candle data available")
            return "ok"

        newest_ts = df["timestamp"].iloc[-1]

        # News-coverage guard: a calendar that ends before the data means the
        # blackout filter approves everything silently (this ran unprotected
        # Feb->Jul 2026). Better no scanner than a blind one — stop loudly.
        if not scanner.news_filter.coverage_ok(newest_ts):
            msg = (f"news calendar coverage ends before newest candle "
                   f"({newest_ts}) — news filter is BLIND. Regenerate with: "
                   f"python scripts/generate_news_events.py, then restart.")
            logger.critical(msg)
            if telegram_notifier:
                telegram_notifier.send_alert("Scanner STOPPED: " + msg)
            return "stop"

        # Freshness guard: decisions are closed-candle; if no new bar closed
        # since the last scan (weekend, holiday, stalled feed) there is
        # nothing new to decide — and rescanning stale bars would re-emit
        # the same signal every 30 min once dedup expires.
        if last_bar_ts is not None and newest_ts == last_bar_ts:
            logger.info(f"No new closed bar since {newest_ts} — skipping scan")
            return "ok"
        last_bar_ts = newest_ts

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

                if telegram_notifier:
                    telegram_notifier.send_signal(sig)

                exit_info = ""
                if sig.exit_tier:
                    tp_str = f"TP:{sig.suggested_tp}" if sig.suggested_tp > 0 else "TRAIL"
                    exit_info = f" | Exit:{sig.exit_tier}({tp_str}) MP:{sig.move_potential}"
                logger.info(
                    f"SIGNAL: {sig.direction} {sig.symbol} @ {sig.entry_price} | "
                    f"SL: {sig.stop_loss} | TP: {sig.take_profit} | "
                    f"RR: {sig.risk_reward} | Conf: {sig.confluence_score} | "
                    f"Tier: {sig.confidence.upper()} ({sig.risk_pct*100:.1f}% risk)"
                    f"{exit_info}"
                )
        else:
            logger.info("No signals detected")
        return "ok"

    while True:
        scan_count += 1
        try:
            status = do_scan()
        except Exception:
            # One bad scan must not kill the scanner (there is no supervisor
            # by choice) — log with traceback and retry next interval.
            logger.exception("Scan loop error — retrying next interval")
            status = "ok"
        if status == "stop" or once:
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

  # Send signals to Telegram
  python scripts/run_live.py --telegram

  # Test Telegram connection (sends one message and exits)
  python scripts/run_live.py --telegram-test
        """,
    )

    parser.add_argument("--telegram-test", action="store_true", help="Send a test message to Telegram and exit (verify bot token and chat_id)")
    parser.add_argument("--symbol", type=str, default=None, help="Symbol (default from config)")
    parser.add_argument("--timeframe", type=str, default=None, help="Timeframe (default from config)")
    parser.add_argument("--balance", type=float, default=None, help="Account balance")
    parser.add_argument("--interval", type=int, default=300, help="Scan interval in seconds (default 300)")
    parser.add_argument("--once", action="store_true", help="Single scan, no loop")
    parser.add_argument("--output", type=str, default=None, help="Output file for signal JSON lines")
    parser.add_argument("--telegram", action="store_true", help="Send signals to Telegram (uses TELEGRAM_* env vars)")

    args = parser.parse_args()

    if args.telegram_test:
        token = TELEGRAM.get("bot_token", "").strip()
        chat_id = TELEGRAM.get("chat_id", "").strip()
        if not token or not chat_id:
            logger.error("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env for --telegram-test")
            sys.exit(1)
        notifier = TelegramNotifier(bot_token=token, chat_id=chat_id)
        ok = notifier.send_test_message()
        logger.info("Telegram test: %s", "OK" if ok else "FAILED")
        sys.exit(0 if ok else 1)

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
        use_telegram=True if args.telegram else None,
    )


if __name__ == "__main__":
    main()
