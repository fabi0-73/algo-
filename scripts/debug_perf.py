"""Debug performance bottleneck in backtest."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pandas as pd
import numpy as np
from src.data.db import Database
from src.strategy.indicators import add_indicators
from src.strategy.consolidation import detect_consolidation
from src.strategy.fvg import find_fvgs_in_range
from src.strategy.order_blocks import find_order_blocks_in_range
from config import STRATEGY

def main():
    print("Loading data...")
    db = Database()
    df = db.get_candles('XAUUSD', 'M5')

    # Use only 200 candles for quick test
    df = df.tail(200).reset_index(drop=True)
    print(f"Using {len(df)} candles")

    # Add indicators
    print("Adding indicators...")
    df = add_indicators(df.copy(), atr_period=STRATEGY["atr_period"])

    # Test individual operations
    atr = df['atr'].iloc[100]
    lookback = STRATEGY["consolidation_lookback"]

    print("\n=== Timing Individual Operations ===")

    # 1. Test is_consolidation
    window = df.iloc[50:62]  # 12 candles
    start = time.time()
    for _ in range(1000):
        range_high = window['high'].max()
        range_low = window['low'].min()
        range_size = range_high - range_low
        max_range = STRATEGY["consolidation_range_atr_mult"] * atr
        closes = window['close']
        closes_inside = ((closes >= range_low) & (closes <= range_high)).sum()
    print(f"Consolidation check (x1000): {(time.time() - start)*1000:.2f}ms")

    # 2. Test FVG search
    start = time.time()
    for _ in range(100):
        fvgs = find_fvgs_in_range(df, 30, 80, direction="BULLISH", atr=atr)
    print(f"FVG search (x100): {(time.time() - start)*1000:.2f}ms")

    # 3. Test Order Block search
    start = time.time()
    for _ in range(100):
        obs = find_order_blocks_in_range(df, 30, 80, direction="BULLISH", atr=atr)
    print(f"Order Block search (x100): {(time.time() - start)*1000:.2f}ms")

    # 4. Test DataFrame iloc slicing
    start = time.time()
    for _ in range(10000):
        window = df.iloc[50:62]
    print(f"DataFrame iloc slice (x10000): {(time.time() - start)*1000:.2f}ms")

    # 5. Test getting single row
    start = time.time()
    for _ in range(10000):
        row = df.iloc[100]
    print(f"Single row iloc (x10000): {(time.time() - start)*1000:.2f}ms")

    # 6. Test nested loop similar to _scan_for_patterns
    print("\n=== Simulating _scan_for_patterns loop ===")
    start = time.time()
    for current_idx in range(46, 100):  # 54 bars
        for offset in range(10, 50):  # 40 iterations
            consol_end_idx = current_idx - offset
            if consol_end_idx < lookback:
                continue
            consol_start_idx = consol_end_idx - lookback
            if consol_start_idx < 0:
                continue
            # Slice window
            window = df.iloc[consol_start_idx:consol_end_idx + 1]
            # Simple calculations
            range_high = window['high'].max()
            range_low = window['low'].min()
    elapsed = time.time() - start
    print(f"54 bars × 40 offsets with slicing: {elapsed*1000:.2f}ms ({elapsed/54*1000:.2f}ms per bar)")

    # 7. Test same loop but with numpy arrays instead of DataFrame
    print("\n=== Testing with NumPy arrays ===")
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    start = time.time()
    for current_idx in range(46, 100):
        for offset in range(10, 50):
            consol_end_idx = current_idx - offset
            if consol_end_idx < lookback:
                continue
            consol_start_idx = consol_end_idx - lookback
            if consol_start_idx < 0:
                continue
            # NumPy slicing
            h = highs[consol_start_idx:consol_end_idx + 1]
            l = lows[consol_start_idx:consol_end_idx + 1]
            range_high = h.max()
            range_low = l.min()
    elapsed = time.time() - start
    print(f"Same loop with NumPy: {elapsed*1000:.2f}ms ({elapsed/54*1000:.2f}ms per bar)")

if __name__ == "__main__":
    main()
