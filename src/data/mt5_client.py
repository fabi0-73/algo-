"""
MetaTrader 5 Client
Handles connection and data fetching from MT5 terminal.
"""
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import logging

from config import MT5_CONFIG, STRATEGY

logger = logging.getLogger(__name__)


class MT5Client:
    """Client for interacting with MetaTrader 5 terminal."""
    
    # Timeframe mapping
    TIMEFRAMES = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }
    
    def __init__(self):
        self._connected = False
    
    def _check_connection(self) -> bool:
        """Verify MT5 connection is still alive. Reset flag if dead."""
        if not self._connected:
            return False
        info = mt5.terminal_info()
        if info is None:
            logger.warning("MT5 connection lost — resetting")
            self._connected = False
            return False
        return True

    def connect(self) -> bool:
        """
        Initialize connection to MT5 terminal.
        Returns True if successful.
        """
        if self._check_connection():
            return True
        
        # Initialize MT5
        init_params = {}
        if MT5_CONFIG["path"]:
            init_params["path"] = MT5_CONFIG["path"]
        if MT5_CONFIG["login"]:
            init_params["login"] = MT5_CONFIG["login"]
        if MT5_CONFIG["password"]:
            init_params["password"] = MT5_CONFIG["password"]
        if MT5_CONFIG["server"]:
            init_params["server"] = MT5_CONFIG["server"]
        
        if not mt5.initialize(**init_params):
            error = mt5.last_error()
            logger.error(f"MT5 initialization failed: {error}")
            return False
        
        self._connected = True
        terminal_info = mt5.terminal_info()
        logger.info(f"Connected to MT5: {terminal_info.name}")
        return True
    
    def disconnect(self):
        """Shutdown MT5 connection."""
        if self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("Disconnected from MT5")
    
    def get_candles(
        self,
        symbol: str = None,
        timeframe: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        count: int = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical candle data from MT5.
        
        Args:
            symbol: Trading symbol (default from config)
            timeframe: Timeframe string like "M5" (default from config)
            start_date: Start datetime for data
            end_date: End datetime for data
            count: Number of candles to fetch (alternative to date range)
        
        Returns:
            DataFrame with OHLCV data or None on error
        """
        if not self._connected:
            if not self.connect():
                return None
        
        symbol = symbol or STRATEGY["symbol"]
        timeframe = timeframe or STRATEGY["timeframe"]
        tf = self.TIMEFRAMES.get(timeframe)
        
        if tf is None:
            logger.error(f"Invalid timeframe: {timeframe}")
            return None
        
        # Check if symbol is available
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol {symbol} not found")
            return None
        
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select symbol {symbol}")
                return None
        
        # Fetch data
        if count is not None:
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        elif start_date is not None and end_date is not None:
            rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
        elif start_date is not None:
            end_date = datetime.now()
            rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
        else:
            # Default: last 1000 candles
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, 1000)
        
        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            logger.error(f"Failed to fetch candles: {error}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={
            "time": "timestamp",
        })
        # Keep tick_volume (used by engine/strategy) and add volume alias
        df["volume"] = df["tick_volume"]
        
        # Add symbol and timeframe columns
        df["symbol"] = symbol
        df["timeframe"] = timeframe
        
        # Reorder columns (keep tick_volume for engine compatibility)
        df = df[["symbol", "timeframe", "timestamp", "open", "high", "low", "close", "tick_volume", "volume"]]
        
        logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe}")
        return df
    
    def get_latest_candles(self, symbol: str = None, timeframe: str = None, count: int = 100) -> Optional[pd.DataFrame]:
        """Convenience method to get the most recent candles."""
        return self.get_candles(symbol=symbol, timeframe=timeframe, count=count)
    
    def get_symbol_info(self, symbol: str = None) -> Optional[dict]:
        """Get symbol information including spread, digits, etc."""
        if not self._connected:
            if not self.connect():
                return None
        
        symbol = symbol or STRATEGY["symbol"]
        info = mt5.symbol_info(symbol)
        
        if info is None:
            return None
        
        return {
            "symbol": info.name,
            "digits": info.digits,
            "point": info.point,
            "spread": info.spread,
            "trade_contract_size": info.trade_contract_size,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
        }
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
