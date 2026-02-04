"""Data layer - MT5 client and database operations"""
from .db import Database, Candle, Trade

try:  # Optional dependency for live MT5 data
    from .mt5_client import MT5Client
except Exception:  # pragma: no cover - optional dependency
    MT5Client = None

__all__ = ["MT5Client", "Database", "Candle", "Trade"]
