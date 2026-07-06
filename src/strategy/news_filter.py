"""
News Filter
Avoid trading around high-impact economic news events.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional
import os
import logging
import pandas as pd
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from config import NEWS_FILTER, TIME_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class NewsEvent:
    """Economic news event."""
    timestamp: datetime
    currency: str = ""
    impact: str = ""
    title: str = ""


class NewsFilterEngine:
    """
    News-based trade filtering.

    Blocks entries within a configurable window around high-impact news events
    (e.g., NFP, CPI, FOMC for USD).
    """

    def __init__(
        self,
        enabled: bool = None,
        csv_path: str = None,
        pre_minutes: int = None,
        post_minutes: int = None,
        impact_filter: List[str] = None,
        require_csv: bool = None,
    ):
        """
        Initialize news filter.

        Args:
            enabled: Whether news filtering is enabled
            csv_path: Path to news events CSV
            pre_minutes: Minutes before event to block
            post_minutes: Minutes after event to block
            impact_filter: List of impact levels to block (e.g., ["HIGH"])
        """
        self.enabled = enabled if enabled is not None else NEWS_FILTER["enabled"]
        self.csv_path = csv_path or NEWS_FILTER["csv_path"]
        self.pre_minutes = pre_minutes if pre_minutes is not None else NEWS_FILTER["pre_minutes"]
        self.post_minutes = post_minutes if post_minutes is not None else NEWS_FILTER["post_minutes"]
        self.impact_filter = impact_filter or NEWS_FILTER.get("impact_filter", ["HIGH"])
        self.require_csv = require_csv if require_csv is not None else NEWS_FILTER.get("require_csv", False)

        self.events: List[NewsEvent] = []
        self._loaded = False

        # Auto-load if enabled and file exists
        if self.enabled:
            self._try_load_events()

    def _try_load_events(self):
        """Try to load news events from CSV."""
        if not os.path.exists(self.csv_path):
            msg = (
                f"News CSV not found: {self.csv_path}. "
                f"Generate it with: python scripts/generate_news_events.py"
            )
            if self.require_csv:
                raise RuntimeError(
                    msg + " (NEWS_FILTER['require_csv']=True — refusing to run with news "
                    "filtering silently disabled.)"
                )
            logger.error(msg + " News filter DISABLED.")
            self.enabled = False
            return

        try:
            self._load_events()
            self._loaded = True
            logger.info(f"Loaded {len(self.events)} news events from {self.csv_path}")
        except Exception as e:
            if self.require_csv:
                raise RuntimeError(f"Failed to load news CSV {self.csv_path}: {e}")
            logger.error(f"Failed to load news CSV: {e}. News filter DISABLED.")
            self.enabled = False

    def _load_events(self):
        """Load events from CSV file."""
        df = pd.read_csv(self.csv_path)

        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()

        # Parse timestamp
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        elif "datetime" in df.columns:
            df["timestamp"] = pd.to_datetime(df["datetime"], utc=True)
        elif "date" in df.columns and "time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"], utc=True)
        else:
            raise ValueError("News CSV must have 'timestamp', 'datetime', or 'date'+'time' columns")

        # Filter by impact if column exists
        if "impact" in df.columns and self.impact_filter:
            df = df[df["impact"].str.upper().isin([i.upper() for i in self.impact_filter])]

        # Convert to NewsEvent objects
        self.events = []
        for _, row in df.iterrows():
            event = NewsEvent(
                timestamp=row["timestamp"].to_pydatetime(),
                currency=row.get("currency", ""),
                impact=row.get("impact", ""),
                title=row.get("title", ""),
            )
            self.events.append(event)

        # Sort by timestamp
        self.events.sort(key=lambda e: e.timestamp)

    def is_in_blackout(self, ts: datetime) -> tuple:
        """
        Check if timestamp is in news blackout window.

        Args:
            ts: Timestamp to check (should be UTC or timezone-aware)

        Returns:
            Tuple of (in_blackout, event_title)
        """
        if not self.enabled or not self.events:
            return False, ""

        # Ensure timestamp is timezone-aware
        if ts.tzinfo is None:
            data_tz = ZoneInfo(TIME_CONFIG["data_timezone"])
            ts = ts.replace(tzinfo=data_tz)

        ts_utc = ts.astimezone(ZoneInfo("UTC"))

        for event in self.events:
            event_start = event.timestamp - timedelta(minutes=self.pre_minutes)
            event_end = event.timestamp + timedelta(minutes=self.post_minutes)

            if event_start <= ts_utc <= event_end:
                return True, event.title or f"{event.currency} {event.impact}"

        return False, ""

    def can_enter_trade(self, ts: datetime) -> tuple:
        """
        Check if trade entry is allowed (not in news blackout).

        Args:
            ts: Current timestamp

        Returns:
            Tuple of (can_enter, reason)
        """
        if not self.enabled:
            return True, ""

        in_blackout, event_info = self.is_in_blackout(ts)
        if in_blackout:
            return False, f"news_blackout:{event_info}"

        return True, ""

    def get_upcoming_events(self, ts: datetime, hours_ahead: int = 24) -> List[NewsEvent]:
        """
        Get upcoming news events within time window.

        Args:
            ts: Current timestamp
            hours_ahead: Hours to look ahead

        Returns:
            List of upcoming events
        """
        if not self.enabled or not self.events:
            return []

        if ts.tzinfo is None:
            data_tz = ZoneInfo(TIME_CONFIG["data_timezone"])
            ts = ts.replace(tzinfo=data_tz)

        ts_utc = ts.astimezone(ZoneInfo("UTC"))
        cutoff = ts_utc + timedelta(hours=hours_ahead)

        upcoming = [e for e in self.events if ts_utc <= e.timestamp <= cutoff]
        return upcoming

    def add_blackout_column(self, df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """
        Add blackout indicator column to DataFrame.

        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column

        Returns:
            DataFrame with 'in_news_blackout' column
        """
        df = df.copy()

        if not self.enabled or not self.events:
            df["in_news_blackout"] = False
            return df

        df["in_news_blackout"] = df[timestamp_col].apply(lambda ts: self.is_in_blackout(ts)[0])
        return df


def load_news_filter(**kwargs) -> NewsFilterEngine:
    """Factory function to create news filter."""
    return NewsFilterEngine(**kwargs)
