"""
Debug Script - Diagnose AMD pattern detection
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from src.data.db import Database
from src.strategy.indicators import add_indicators, calculate_atr
from src.strategy.consolidation import detect_consolidation
from src.strategy.manipulation import detect_manipulation
from src.strategy.distribution import detect_distribution
from src.strategy.entry import check_entry, check_immediate_entry
from config import STRATEGY


def main():
    print("=" * 60)
    print("AMD Strategy Debug")
    print("=" * 60)
    
    # Load data
    db = Database()
    df = db.get_candles("XAUUSD", "M5")
    print(f"\nLoaded {len(df)} candles")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Add indicators
    df = add_indicators(df)
    print(f"\nATR range: {df['atr'].min():.2f} - {df['atr'].max():.2f}")
    print(f"Avg ATR: {df['atr'].mean():.2f}")
    print(f"Avg body size: {df['body_size'].mean():.2f}")
    
    # Test consolidation detection at various points
    print("\n" + "=" * 60)
    print("PHASE 1: Consolidation Detection")
    print("=" * 60)
    
    consolidation_count = 0
    manipulation_count = 0
    distribution_count = 0
    entry_count = 0
    
    min_lookback = STRATEGY["consolidation_lookback"] + STRATEGY["atr_period"] + 10
    
    # Sample every 100 candles to speed up debug
    sample_points = range(min_lookback, len(df), 100)
    
    first_consol_shown = False
    for i in sample_points:
        test_df = df.iloc[:i+1]
        
        consol = detect_consolidation(test_df)
        if consol.valid:
            consolidation_count += 1
            
            # Show first consolidation details
            if not first_consol_shown:
                print(f"\n  First consolidation at index {i}:")
                print(f"    Range: {consol.range_low:.2f} - {consol.range_high:.2f}")
                print(f"    ATR: {consol.atr:.2f}")
                first_consol_shown = True
            
            # Check manipulation
            manip = detect_manipulation(test_df, consol)
            if manip.valid:
                manipulation_count += 1
                
                # Check distribution
                dist = detect_distribution(test_df, consol, manip)
                if dist.valid:
                    distribution_count += 1
                    
                    # Check entry
                    entry = check_entry(test_df, consol, manip, dist)
                    if not entry.valid:
                        entry = check_immediate_entry(test_df, consol, manip, dist)
                    
                    if entry.valid:
                        entry_count += 1
    
    print(f"\nSampled {len(list(sample_points))} points")
    print(f"Consolidations found: {consolidation_count}")
    print(f"Manipulations found: {manipulation_count}")
    print(f"Distributions found: {distribution_count}")
    print(f"Entry signals: {entry_count}")
    
    # Show a detailed example
    print("\n" + "=" * 60)
    print("DETAILED EXAMPLE - Testing at candle 5000")
    print("=" * 60)
    
    test_idx = min(5000, len(df) - 1)
    test_df = df.iloc[:test_idx+1]
    
    current_atr = test_df['atr'].iloc[-1]
    print(f"\nCurrent ATR: {current_atr:.2f}")
    
    # Get last 20 candles
    window = test_df.iloc[-20:]
    range_high = window['high'].max()
    range_low = window['low'].min()
    range_size = range_high - range_low
    
    print(f"Last 20 candles range: {range_low:.2f} - {range_high:.2f}")
    print(f"Range size: {range_size:.2f}")
    print(f"Max allowed range ({STRATEGY['consolidation_range_atr_mult']} * ATR): {STRATEGY['consolidation_range_atr_mult'] * current_atr:.2f}")
    print(f"Range meets criteria: {range_size <= 0.8 * current_atr}")
    
    # Check closes inside range
    closes = window['close']
    closes_inside = ((closes >= range_low) & (closes <= range_high)).sum()
    close_pct = closes_inside / len(window)
    print(f"Closes inside range: {closes_inside}/20 ({close_pct:.1%})")
    
    consol = detect_consolidation(test_df)
    print(f"\nConsolidation valid: {consol.valid}")
    if consol.valid:
        print(f"  Range: {consol.range_low:.2f} - {consol.range_high:.2f}")
        print(f"  Range size: {consol.range_size:.2f}")
    
    print("\n" + "=" * 60)
    print("PARAMETER CHECK")
    print("=" * 60)
    print(f"\nCurrent settings:")
    print(f"  consolidation_range_atr_mult: {STRATEGY['consolidation_range_atr_mult']}")
    print(f"  consolidation_close_pct: {STRATEGY['consolidation_close_pct']}")
    print(f"  manipulation_break_atr_mult: {STRATEGY['manipulation_break_atr_mult']}")
    print(f"  manipulation_return_candles: {STRATEGY['manipulation_return_candles']}")
    
    # Calculate what range would be needed
    avg_atr = df['atr'].mean()
    print(f"\nWith avg ATR of {avg_atr:.2f}:")
    print(f"  Max consolidation range: {STRATEGY['consolidation_range_atr_mult'] * avg_atr:.2f}")
    print(f"  Min manipulation break: {STRATEGY['manipulation_break_atr_mult'] * avg_atr:.2f}")
    
    # Check typical range sizes
    print("\n" + "=" * 60)
    print("TYPICAL RANGE SIZES (20-candle windows)")
    print("=" * 60)
    
    ranges = []
    for i in range(20, min(1000, len(df))):
        w = df.iloc[i-20:i]
        r = w['high'].max() - w['low'].min()
        ranges.append(r)
    
    ranges = pd.Series(ranges)
    print(f"Min range: {ranges.min():.2f}")
    print(f"Max range: {ranges.max():.2f}")
    print(f"Mean range: {ranges.mean():.2f}")
    print(f"Median range: {ranges.median():.2f}")
    print(f"10th percentile: {ranges.quantile(0.10):.2f}")
    print(f"25th percentile: {ranges.quantile(0.25):.2f}")
    
    print(f"\nFor consolidation detection to work, range_atr_mult should be:")
    print(f"  Current: {STRATEGY['consolidation_range_atr_mult']}")
    print(f"  Suggested (to capture 10% of ranges): {ranges.quantile(0.10) / avg_atr:.2f}")
    print(f"  Suggested (to capture 25% of ranges): {ranges.quantile(0.25) / avg_atr:.2f}")


if __name__ == "__main__":
    main()
