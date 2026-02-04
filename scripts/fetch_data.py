"""
Data Fetching Script
Downloads historical candle data from MT5 and stores in PostgreSQL.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime, timedelta
import logging

from src.data.mt5_client import MT5Client
from src.data.db import Database
from config import STRATEGY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fetch_and_store(
    symbol: str = None,
    timeframe: str = None,
    months: int = 6,
    start_date: datetime = None,
    end_date: datetime = None,
):
    """
    Fetch historical data from MT5 and store in database.
    
    Args:
        symbol: Trading symbol (default from config)
        timeframe: Timeframe (default from config)
        months: Number of months of history to fetch
        start_date: Override start date
        end_date: Override end date
    """
    symbol = symbol or STRATEGY["symbol"]
    timeframe = timeframe or STRATEGY["timeframe"]
    
    # Calculate date range
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=months * 30)
    
    logger.info(f"Fetching {symbol} {timeframe} from {start_date} to {end_date}")
    
    # Initialize database
    db = Database()
    db.create_tables()
    
    # Check existing data
    existing_range = db.get_date_range(symbol, timeframe)
    if existing_range[0] is not None:
        logger.info(f"Existing data: {existing_range[0]} to {existing_range[1]}")
    
    # Fetch from MT5
    with MT5Client() as client:
        # Get symbol info
        info = client.get_symbol_info(symbol)
        if info:
            logger.info(f"Symbol info: spread={info['spread']}, digits={info['digits']}")
        
        # Fetch candles
        df = client.get_candles(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        
        if df is None or df.empty:
            logger.error("No data fetched")
            return
        
        logger.info(f"Fetched {len(df)} candles from MT5")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Store in database
    rows = db.insert_candles(df)
    logger.info(f"Inserted/updated {rows} rows in database")
    
    # Verify
    count = db.get_candle_count(symbol, timeframe)
    date_range = db.get_date_range(symbol, timeframe)
    logger.info(f"Total candles in DB: {count}")
    logger.info(f"DB date range: {date_range[0]} to {date_range[1]}")


def main():
    parser = argparse.ArgumentParser(description="Fetch historical data from MT5")
    parser.add_argument("--symbol", type=str, help="Trading symbol (default: XAUUSD)")
    parser.add_argument("--timeframe", type=str, help="Timeframe (default: M5)")
    parser.add_argument("--months", type=int, default=6, help="Months of history (default: 6)")
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD)",
    )
    
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start, "%Y-%m-%d") if args.start else None
    end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else None
    
    fetch_and_store(
        symbol=args.symbol,
        timeframe=args.timeframe,
        months=args.months,
        start_date=start_date,
        end_date=end_date,
    )


if __name__ == "__main__":
    main()
