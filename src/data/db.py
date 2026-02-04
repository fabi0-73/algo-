"""
Database Layer
PostgreSQL models and operations for candles and trades.
"""
from datetime import datetime
from decimal import Decimal
from typing import Optional, List
import logging

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Numeric,
    BigInteger,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.dialects.postgresql import insert
import pandas as pd

from config import DATABASE_URL

logger = logging.getLogger(__name__)

Base = declarative_base()


class Candle(Base):
    """OHLCV candle data model."""
    
    __tablename__ = "candles"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Numeric(12, 5), nullable=False)
    high = Column(Numeric(12, 5), nullable=False)
    low = Column(Numeric(12, 5), nullable=False)
    close = Column(Numeric(12, 5), nullable=False)
    volume = Column(BigInteger, nullable=False)
    
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_candle"),
        Index("idx_candle_symbol_tf_time", "symbol", "timeframe", "timestamp"),
    )
    
    def __repr__(self):
        return f"<Candle {self.symbol} {self.timeframe} {self.timestamp}>"


class Trade(Base):
    """Trade record model."""
    
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime, nullable=True)
    direction = Column(String(10), nullable=False)  # "LONG" or "SHORT"
    entry_price = Column(Numeric(12, 5), nullable=False)
    exit_price = Column(Numeric(12, 5), nullable=True)
    sl_price = Column(Numeric(12, 5), nullable=False)
    tp_price = Column(Numeric(12, 5), nullable=False)
    position_size = Column(Numeric(10, 4), nullable=False)
    r_multiple = Column(Numeric(8, 4), nullable=True)
    pnl_pips = Column(Numeric(10, 2), nullable=True)
    pnl_usd = Column(Numeric(12, 2), nullable=True)
    
    # AMD context
    consolidation_high = Column(Numeric(12, 5), nullable=False)
    consolidation_low = Column(Numeric(12, 5), nullable=False)
    manipulation_extreme = Column(Numeric(12, 5), nullable=False)
    manipulation_direction = Column(String(10), nullable=False)  # "UP" or "DOWN"
    
    # Metadata
    backtest_id = Column(String(50), nullable=True)
    notes = Column(String(500), nullable=True)
    
    __table_args__ = (
        Index("idx_trade_entry_time", "entry_time"),
        Index("idx_trade_backtest", "backtest_id"),
    )
    
    def __repr__(self):
        return f"<Trade {self.direction} @ {self.entry_price} ({self.entry_time})>"


class Database:
    """Database operations handler."""
    
    def __init__(self, url: str = None):
        self.url = url or DATABASE_URL
        self.engine = create_engine(self.url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def create_tables(self):
        """Create all tables if they don't exist."""
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created")
    
    def drop_tables(self):
        """Drop all tables (use with caution)."""
        Base.metadata.drop_all(self.engine)
        logger.info("Database tables dropped")
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    # =========================================================================
    # Candle Operations
    # =========================================================================
    
    def insert_candles(self, df: pd.DataFrame) -> int:
        """
        Insert candles from DataFrame. Uses upsert to handle duplicates.
        Returns number of rows affected.
        """
        if df.empty:
            return 0
        
        records = df.to_dict("records")
        
        with self.get_session() as session:
            stmt = insert(Candle).values(records)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_candle",
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "volume": stmt.excluded.volume,
                }
            )
            result = session.execute(stmt)
            session.commit()
            return result.rowcount
    
    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = None,
    ) -> pd.DataFrame:
        """
        Fetch candles from database as DataFrame.
        """
        with self.get_session() as session:
            query = session.query(Candle).filter(
                Candle.symbol == symbol,
                Candle.timeframe == timeframe,
            )
            
            if start_date:
                query = query.filter(Candle.timestamp >= start_date)
            if end_date:
                query = query.filter(Candle.timestamp <= end_date)
            
            query = query.order_by(Candle.timestamp)
            
            if limit:
                query = query.limit(limit)
            
            candles = query.all()
        
        if not candles:
            return pd.DataFrame()
        
        df = pd.DataFrame([{
            "symbol": c.symbol,
            "timeframe": c.timeframe,
            "timestamp": c.timestamp,
            "open": float(c.open),
            "high": float(c.high),
            "low": float(c.low),
            "close": float(c.close),
            "volume": c.volume,
        } for c in candles])
        
        return df
    
    def get_candle_count(self, symbol: str, timeframe: str) -> int:
        """Get count of candles in database."""
        with self.get_session() as session:
            return session.query(Candle).filter(
                Candle.symbol == symbol,
                Candle.timeframe == timeframe,
            ).count()
    
    def get_date_range(self, symbol: str, timeframe: str) -> tuple:
        """Get min and max dates for candles."""
        with self.get_session() as session:
            from sqlalchemy import func
            result = session.query(
                func.min(Candle.timestamp),
                func.max(Candle.timestamp),
            ).filter(
                Candle.symbol == symbol,
                Candle.timeframe == timeframe,
            ).first()
            return result
    
    # =========================================================================
    # Trade Operations
    # =========================================================================
    
    def insert_trade(self, trade: Trade) -> int:
        """Insert a single trade record. Returns trade ID."""
        with self.get_session() as session:
            session.add(trade)
            session.commit()
            return trade.id
    
    def insert_trades(self, trades: List[Trade]) -> int:
        """Insert multiple trade records. Returns count inserted."""
        with self.get_session() as session:
            session.add_all(trades)
            session.commit()
            return len(trades)
    
    def get_trades(
        self,
        backtest_id: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> pd.DataFrame:
        """Fetch trades from database as DataFrame."""
        with self.get_session() as session:
            query = session.query(Trade)
            
            if backtest_id:
                query = query.filter(Trade.backtest_id == backtest_id)
            if start_date:
                query = query.filter(Trade.entry_time >= start_date)
            if end_date:
                query = query.filter(Trade.entry_time <= end_date)
            
            query = query.order_by(Trade.entry_time)
            trades = query.all()
        
        if not trades:
            return pd.DataFrame()
        
        df = pd.DataFrame([{
            "id": t.id,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "direction": t.direction,
            "entry_price": float(t.entry_price),
            "exit_price": float(t.exit_price) if t.exit_price else None,
            "sl_price": float(t.sl_price),
            "tp_price": float(t.tp_price),
            "position_size": float(t.position_size),
            "r_multiple": float(t.r_multiple) if t.r_multiple else None,
            "pnl_pips": float(t.pnl_pips) if t.pnl_pips else None,
            "pnl_usd": float(t.pnl_usd) if t.pnl_usd else None,
            "consolidation_high": float(t.consolidation_high),
            "consolidation_low": float(t.consolidation_low),
            "manipulation_extreme": float(t.manipulation_extreme),
            "manipulation_direction": t.manipulation_direction,
            "backtest_id": t.backtest_id,
        } for t in trades])
        
        return df
    
    def clear_trades(self, backtest_id: str = None):
        """Clear trades from database. If backtest_id provided, only clear those."""
        with self.get_session() as session:
            query = session.query(Trade)
            if backtest_id:
                query = query.filter(Trade.backtest_id == backtest_id)
            query.delete()
            session.commit()
