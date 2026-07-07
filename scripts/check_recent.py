"""Replay the most recent MT5 candles through the full engine (AMD + NY_IB,
confidence sizing) to see what trades the live config would have generated.
Signal-only research — no orders. Usage: python scripts/check_recent.py [bars]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import STRATEGY, BACKTEST, NY_IB_MODEL, CONFIDENCE_SIZING
from src.data.mt5_client import MT5Client
from src.backtest.engine import BacktestEngine

bars = int(sys.argv[1]) if len(sys.argv) > 1 else 4000

client = MT5Client()
df = client.get_candles("XAUUSD", "M5", count=bars)
df = df.iloc[:-1].reset_index(drop=True)  # drop forming bar (live guard)
t0, t1 = df["timestamp"].iloc[0], df["timestamp"].iloc[-1]
print(f"MT5: {len(df)} closed M5 candles  {t0} -> {t1}  (broker time)")
print(f"Config: NY_IB={NY_IB_MODEL.get('enabled')}  "
      f"confidence_sizing={CONFIDENCE_SIZING.get('enabled')}")
print("=" * 78)

engine = BacktestEngine(initial_capital=500.0,
                        max_trade_duration=STRATEGY.get("max_trade_duration", 200))
res = engine.run(df, verbose=False)

trades = engine.trades
if not trades:
    print("No trades generated in this window.")
else:
    print(f"{len(trades)} trade(s) would have fired (SL/TP = the ORIGINAL signal levels):\n")
    hdr = (f"{'#':>2} {'model':<6} {'dir':<5} {'entry_time':<16} {'ENTRY':>9} "
           f"{'SL':>9} {'TP':>9} {'SLdist$':>7} {'TPdist$':>7} {'R:R':>5} "
           f"{'lots':>5} {'conf':>4} {'exit':<9} {'R':>6}")
    print(hdr)
    print("-" * len(hdr))
    for i, t in enumerate(trades, 1):
        osl = t.original_sl if getattr(t, "original_sl", 0) else t.sl_price
        otp = t.original_tp if getattr(t, "original_tp", 0) else t.tp_price
        sl_d = abs(t.entry_price - osl)
        tp_d = abs(otp - t.entry_price)
        rr = tp_d / sl_d if sl_d > 0 else 0.0
        conf = f"{t.signal_confidence}" if t.entry_model == "AMD" else "-"
        rmult = f"{t.r_multiple:+.2f}" if t.r_multiple is not None else " open"
        exitr = (t.exit_reason or "OPEN")[:9]
        print(f"{i:>2} {t.entry_model:<6} {t.direction:<5} "
              f"{str(t.entry_time)[:16]:<16} {t.entry_price:>9.2f} {osl:>9.2f} "
              f"{otp:>9.2f} {sl_d:>7.2f} {tp_d:>7.2f} {rr:>5.2f} "
              f"{t.position_size:>5.2f} {conf:>4} {exitr:<9} {rmult:>6}")

# Open position at end of window (a live signal you'd be acting on now)
if engine.state.in_position and engine.state.current_trade:
    ct = engine.state.current_trade
    print(f"\n>>> OPEN at window end: {ct.entry_model} {ct.direction} @ {ct.entry_price:.2f} "
          f"SL {ct.sl_price:.2f} TP {ct.tp_price:.2f} ({ct.position_size} lots)")

# Rejection funnel — why non-trades were skipped
rs = engine.rejection_stats
interesting = {k: v for k, v in rs.items() if v and (
    k.startswith("nyib") or "filtered" in k or "entries" in k
    or k in ("no_manipulation", "no_distribution", "no_bos", "entry_too_late"))}
print("\nFunnel (nonzero):")
for k in sorted(interesting):
    print(f"  {k:<28} {interesting[k]}")
