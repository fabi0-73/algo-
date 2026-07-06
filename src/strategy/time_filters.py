"""
Time and Session Filters
Kill zone, Asian session, daily trade limits, and cooldown management.
"""
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Optional, Dict
import pandas as pd
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from config import TIME_CONFIG, SESSION_FILTER, EXECUTION


@dataclass
class SessionState:
    """Current session state for a trading day."""
    trading_day: str  # YYYY-MM-DD in session timezone
    trades_today: int = 0
    daily_pnl: float = 0.0
    last_trade_time: Optional[datetime] = None


class TimeFilterEngine:
    """
    Time-based filtering for trade entries.

    Handles:
    - Kill zone (London/NY overlap) filtering
    - Asian session avoidance
    - Daily trade limits
    - Cooldown after trades
    - Rollover window avoidance
    """

    def __init__(
        self,
        enabled: bool = None,
        kill_zone_start: str = None,
        kill_zone_end: str = None,
        avoid_asian: bool = None,
        asian_start: str = None,
        asian_end: str = None,
        max_trades_per_day: int = None,
        daily_loss_limit_pct: float = None,
        cooldown_minutes: int = None,
        close_before_rollover: bool = None,
        close_before_rollover_minutes: int = None,
        no_new_entries_before_rollover_minutes: int = None,
        data_timezone: str = None,
        session_timezone: str = None,
    ):
        """
        Initialize time filter engine.

        Args:
            enabled: Whether session filtering is enabled
            kill_zone_start: Kill zone start time (HH:MM)
            kill_zone_end: Kill zone end time (HH:MM)
            avoid_asian: Whether to avoid Asian session
            asian_start: Asian session start (HH:MM)
            asian_end: Asian session end (HH:MM)
            max_trades_per_day: Maximum trades per day
            daily_loss_limit_pct: Daily loss limit as decimal
            cooldown_minutes: Minutes to wait after each trade
            close_before_rollover: Whether to close before rollover
            close_before_rollover_minutes: Minutes before rollover to close
            no_new_entries_before_rollover_minutes: Minutes before rollover to block entries
            data_timezone: Timezone for OHLC data
            session_timezone: Timezone for session rules
        """
        self.enabled = enabled if enabled is not None else SESSION_FILTER["enabled"]
        self.kill_zone_start = self._parse_time(kill_zone_start or SESSION_FILTER["kill_zone_start"])
        self.kill_zone_end = self._parse_time(kill_zone_end or SESSION_FILTER["kill_zone_end"])
        self.avoid_asian = avoid_asian if avoid_asian is not None else SESSION_FILTER["avoid_asian"]
        self.asian_start = self._parse_time(asian_start or SESSION_FILTER["asian_start"])
        self.asian_end = self._parse_time(asian_end or SESSION_FILTER["asian_end"])
        self.max_trades_per_day = max_trades_per_day if max_trades_per_day is not None else SESSION_FILTER["max_trades_per_day"]
        self.daily_loss_limit_pct = daily_loss_limit_pct if daily_loss_limit_pct is not None else SESSION_FILTER["daily_loss_limit_pct"]
        self.cooldown_minutes = cooldown_minutes if cooldown_minutes is not None else SESSION_FILTER["cooldown_minutes_after_trade"]
        self.close_before_rollover = close_before_rollover if close_before_rollover is not None else SESSION_FILTER["close_before_rollover"]
        self.close_before_rollover_minutes = close_before_rollover_minutes if close_before_rollover_minutes is not None else SESSION_FILTER["close_before_rollover_minutes"]
        self.no_new_entries_before_rollover_minutes = no_new_entries_before_rollover_minutes if no_new_entries_before_rollover_minutes is not None else SESSION_FILTER["no_new_entries_before_rollover_minutes"]

        self.monthly_loss_limit_pct = SESSION_FILTER.get("monthly_loss_limit_pct", 0.0)

        self.blackout_hours = SESSION_FILTER.get("blackout_hours_utc", [])
        self.blackout_weekdays = SESSION_FILTER.get("blackout_weekdays", [])

        self.data_tz = ZoneInfo(data_timezone or TIME_CONFIG["data_timezone"])
        self.session_tz = ZoneInfo(session_timezone or TIME_CONFIG["session_timezone"])

        # Parse rollover time
        rollover_str = EXECUTION.get("rollover_time_utc", "21:59")
        self.rollover_time = self._parse_time(rollover_str)

        # State tracking per trading day
        self.session_states: Dict[str, SessionState] = {}

        # Monthly P&L tracking for circuit breaker
        self._monthly_pnl: Dict[str, float] = {}

    def _parse_time(self, time_str: str) -> time:
        """Parse time string HH:MM to time object."""
        parts = time_str.split(":")
        return time(int(parts[0]), int(parts[1]))

    def convert_to_session_tz(self, ts: datetime) -> datetime:
        """
        Convert timestamp to session timezone.

        Args:
            ts: Timestamp (may be naive or aware)

        Returns:
            Timezone-aware datetime in session timezone
        """
        if ts.tzinfo is None:
            # Assume data timezone for naive timestamps
            ts = ts.replace(tzinfo=self.data_tz)
        return ts.astimezone(self.session_tz)

    def get_trading_day_key(self, ts: datetime) -> str:
        """
        Get trading day key for a timestamp.

        Trading day starts at Asian session end (08:00) and ends next day.

        Args:
            ts: Timestamp in session timezone

        Returns:
            Trading day key (YYYY-MM-DD)
        """
        session_ts = self.convert_to_session_tz(ts)
        # If before Asian end (08:00), belongs to previous trading day
        if session_ts.time() < self.asian_end:
            day = session_ts.date() - timedelta(days=1)
        else:
            day = session_ts.date()
        return day.strftime("%Y-%m-%d")

    def get_session_state(self, ts: datetime) -> SessionState:
        """Get or create session state for trading day."""
        key = self.get_trading_day_key(ts)
        if key not in self.session_states:
            self.session_states[key] = SessionState(trading_day=key)
        return self.session_states[key]

    def is_in_kill_zone(self, ts: datetime) -> bool:
        """
        Check if timestamp is in kill zone.

        Kill zone: 12:00-16:00 UTC (London/NY overlap)

        Args:
            ts: Timestamp to check

        Returns:
            True if in kill zone
        """
        if not self.enabled:
            return True

        session_ts = self.convert_to_session_tz(ts)
        current_time = session_ts.time()

        return self.kill_zone_start <= current_time <= self.kill_zone_end

    def is_in_asian_session(self, ts: datetime) -> bool:
        """
        Check if timestamp is in Asian session.

        Asian session spans midnight: 23:00-08:00 UTC

        Args:
            ts: Timestamp to check

        Returns:
            True if in Asian session
        """
        if not self.enabled:
            return False

        session_ts = self.convert_to_session_tz(ts)
        current_time = session_ts.time()

        # Asian session spans midnight (23:00 -> 08:00)
        if self.asian_start > self.asian_end:
            # Spans midnight
            return current_time >= self.asian_start or current_time < self.asian_end
        else:
            return self.asian_start <= current_time < self.asian_end

    def is_in_london_ny_session(self, ts: datetime) -> bool:
        """
        Check if timestamp is in London/NY session (08:00-16:00 UTC).

        Args:
            ts: Timestamp to check

        Returns:
            True if in London/NY session
        """
        session_ts = self.convert_to_session_tz(ts)
        current_time = session_ts.time()

        london_ny_start = time(8, 0)
        london_ny_end = time(16, 0)

        return london_ny_start <= current_time <= london_ny_end

    def consolidation_formed_in_asian(
        self,
        consol_start_ts: datetime,
        consol_end_ts: datetime,
    ) -> bool:
        """
        Check if consolidation formed during Asian session (23:00-08:00 UTC).

        Returns True if at least part of the consolidation was during Asian session.

        Args:
            consol_start_ts: Start timestamp of consolidation
            consol_end_ts: End timestamp of consolidation

        Returns:
            True if consolidation formed during Asian session
        """
        if consol_start_ts is None or consol_end_ts is None:
            return False

        # Check if either end is in Asian session, or if consolidation spans it
        start_in_asian = self.is_in_asian_session(consol_start_ts)
        end_in_asian = self.is_in_asian_session(consol_end_ts)

        return start_in_asian or end_in_asian

    def distribution_in_london_ny(self, dist_ts: datetime) -> bool:
        """
        Check if distribution occurred during London/NY session (08:00-16:00 UTC).

        Args:
            dist_ts: Timestamp of distribution breakout

        Returns:
            True if distribution was during London/NY session
        """
        if dist_ts is None:
            return False

        return self.is_in_london_ny_session(dist_ts)

    def is_in_specific_kill_zone(self, ts: datetime) -> tuple:
        """
        Check if timestamp is in a specific kill zone (London Open or NY Open).

        Kill zones from config:
        - london_open_kz: Default ("08:00", "10:00")
        - ny_open_kz: Default ("13:00", "15:00")

        Args:
            ts: Timestamp to check

        Returns:
            Tuple of (in_kz: bool, kz_name: str)
            kz_name is "london_open", "ny_open", or "" if not in any KZ
        """
        session_ts = self.convert_to_session_tz(ts)
        current_time = session_ts.time()

        # Check London Open KZ
        london_kz = SESSION_FILTER.get("london_open_kz", ("08:00", "10:00"))
        london_start = self._parse_time(london_kz[0])
        london_end = self._parse_time(london_kz[1])
        if london_start <= current_time <= london_end:
            return True, "london_open"

        # Check NY Open KZ
        ny_kz = SESSION_FILTER.get("ny_open_kz", ("13:00", "15:00"))
        ny_start = self._parse_time(ny_kz[0])
        ny_end = self._parse_time(ny_kz[1])
        if ny_start <= current_time <= ny_end:
            return True, "ny_open"

        return False, ""

    def is_near_rollover(self, ts: datetime, minutes_before: int = None) -> bool:
        """
        Check if timestamp is near rollover time.

        Args:
            ts: Timestamp to check
            minutes_before: Minutes before rollover to consider "near"

        Returns:
            True if within window before rollover
        """
        if minutes_before is None:
            minutes_before = self.no_new_entries_before_rollover_minutes

        session_ts = self.convert_to_session_tz(ts)
        current_time = session_ts.time()

        # Calculate rollover window start
        rollover_dt = datetime.combine(session_ts.date(), self.rollover_time)
        window_start = (rollover_dt - timedelta(minutes=minutes_before)).time()

        # Handle window crossing midnight
        if window_start > self.rollover_time:
            # Window starts before midnight
            return current_time >= window_start or current_time <= self.rollover_time
        else:
            return window_start <= current_time <= self.rollover_time

    def should_close_for_rollover(self, ts: datetime) -> bool:
        """
        Check if position should be closed before rollover.

        Args:
            ts: Current timestamp

        Returns:
            True if should close
        """
        if not self.enabled or not self.close_before_rollover:
            return False

        return self.is_near_rollover(ts, self.close_before_rollover_minutes)

    def is_in_cooldown(self, ts: datetime) -> bool:
        """
        Check if in cooldown period after last trade.

        Args:
            ts: Current timestamp

        Returns:
            True if in cooldown
        """
        if not self.enabled or self.cooldown_minutes <= 0:
            return False

        state = self.get_session_state(ts)
        if state.last_trade_time is None:
            return False

        session_ts = self.convert_to_session_tz(ts)
        last_trade_ts = self.convert_to_session_tz(state.last_trade_time)

        time_since_trade = session_ts - last_trade_ts
        return time_since_trade < timedelta(minutes=self.cooldown_minutes)

    def has_reached_daily_limit(self, ts: datetime) -> bool:
        """
        Check if daily trade limit reached.

        Args:
            ts: Current timestamp

        Returns:
            True if limit reached
        """
        if not self.enabled:
            return False

        state = self.get_session_state(ts)
        return state.trades_today >= self.max_trades_per_day

    def has_exceeded_daily_loss_limit(self, ts: datetime, starting_capital: float) -> bool:
        """
        Check if daily loss limit exceeded.

        Args:
            ts: Current timestamp
            starting_capital: Starting capital for percentage calculation

        Returns:
            True if loss limit exceeded
        """
        if not self.enabled or self.daily_loss_limit_pct <= 0:
            return False

        state = self.get_session_state(ts)
        loss_pct = abs(state.daily_pnl) / starting_capital if state.daily_pnl < 0 else 0
        return loss_pct >= self.daily_loss_limit_pct

    def _get_month_key(self, ts: datetime) -> str:
        """Get month key (YYYY-MM) from timestamp in session timezone."""
        session_ts = self.convert_to_session_tz(ts)
        return session_ts.strftime("%Y-%m")

    def has_exceeded_monthly_loss_limit(self, ts: datetime, starting_capital: float) -> bool:
        """
        Check if monthly loss limit exceeded.

        Args:
            ts: Current timestamp
            starting_capital: Starting capital for percentage calculation

        Returns:
            True if monthly loss limit exceeded
        """
        if not self.enabled or self.monthly_loss_limit_pct <= 0:
            return False

        month_key = self._get_month_key(ts)
        monthly_pnl = self._monthly_pnl.get(month_key, 0.0)
        loss_pct = abs(monthly_pnl) / starting_capital if monthly_pnl < 0 else 0
        return loss_pct >= self.monthly_loss_limit_pct

    def can_enter_trade(
        self,
        ts: datetime,
        starting_capital: float = 10000.0,
    ) -> tuple:
        """
        Check if trade entry is allowed at this time.

        Args:
            ts: Current timestamp
            starting_capital: Starting capital for loss limit check

        Returns:
            Tuple of (can_enter, reason)
        """
        if not self.enabled:
            return True, ""

        # Check kill zone (optional)
        if SESSION_FILTER.get("use_kill_zone", True):
            if not self.is_in_kill_zone(ts):
                return False, "outside_killzone"

            # If enabled, only allow entries during London open or NY open kill zones
            if SESSION_FILTER.get("only_trade_in_kz", False):
                in_specific_kz, _ = self.is_in_specific_kill_zone(ts)
                if not in_specific_kz:
                    return False, "outside_specific_killzone"

        # If enabled, only allow entries during London open or NY open kill zones
        if SESSION_FILTER.get("only_trade_in_kz", False):
            in_specific_kz, _ = self.is_in_specific_kill_zone(ts)
            if not in_specific_kz:
                return False, "outside_specific_killzone"

        # Check blackout hours
        if self.blackout_hours:
            session_ts = self.convert_to_session_tz(ts)
            if session_ts.hour in self.blackout_hours:
                return False, "blackout_hour"

        # Check blackout weekdays
        if self.blackout_weekdays:
            session_ts = self.convert_to_session_tz(ts)
            if session_ts.weekday() in self.blackout_weekdays:
                return False, "blackout_weekday"

        # Check Asian session
        if self.avoid_asian and self.is_in_asian_session(ts):
            return False, "asian_session"

        # Check rollover window
        if self.is_near_rollover(ts):
            return False, "near_rollover"

        # Check cooldown
        if self.is_in_cooldown(ts):
            return False, "cooldown"

        # Check daily trade limit
        if self.has_reached_daily_limit(ts):
            return False, "daily_limit"

        # Check daily loss limit
        if self.has_exceeded_daily_loss_limit(ts, starting_capital):
            return False, "daily_loss_limit"

        # Check monthly loss limit
        if self.has_exceeded_monthly_loss_limit(ts, starting_capital):
            return False, "monthly_loss_limit"

        return True, ""

    def record_trade(self, ts: datetime, pnl: float = 0.0):
        """
        Record a trade for session state tracking.

        Args:
            ts: Trade timestamp
            pnl: Trade PnL (for daily loss tracking)
        """
        state = self.get_session_state(ts)
        state.trades_today += 1
        state.daily_pnl += pnl
        state.last_trade_time = ts

    def record_trade_exit(self, ts: datetime, pnl: float):
        """
        Record trade exit PnL.

        Args:
            ts: Exit timestamp
            pnl: Trade PnL
        """
        state = self.get_session_state(ts)
        state.daily_pnl += pnl

        # Track monthly P&L for circuit breaker
        month_key = self._get_month_key(ts)
        self._monthly_pnl[month_key] = self._monthly_pnl.get(month_key, 0.0) + pnl

    def reset_daily_state(self, trading_day: str = None):
        """
        Reset state for a trading day or all days.

        Args:
            trading_day: Specific day to reset (None = all)
        """
        if trading_day:
            if trading_day in self.session_states:
                del self.session_states[trading_day]
        else:
            self.session_states.clear()


def localize_dataframe_timestamps(
    df: pd.DataFrame,
    data_timezone: str = None,
    session_timezone: str = None,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Add timezone-aware timestamp columns to DataFrame.

    Args:
        df: DataFrame with timestamp column
        data_timezone: Timezone for data interpretation
        session_timezone: Timezone for session evaluation
        timestamp_col: Name of timestamp column

    Returns:
        DataFrame with additional columns:
        - timestamp_tz: Timezone-aware timestamp in data timezone
        - timestamp_session_tz: Timestamp in session timezone
    """
    data_tz = ZoneInfo(data_timezone or TIME_CONFIG["data_timezone"])
    session_tz = ZoneInfo(session_timezone or TIME_CONFIG["session_timezone"])

    df = df.copy()

    # Localize to data timezone
    if df[timestamp_col].dt.tz is None:
        df["timestamp_tz"] = df[timestamp_col].dt.tz_localize(data_tz)
    else:
        df["timestamp_tz"] = df[timestamp_col]

    # Convert to session timezone
    df["timestamp_session_tz"] = df["timestamp_tz"].dt.tz_convert(session_tz)

    # Add convenience columns
    df["session_time"] = df["timestamp_session_tz"].dt.time
    df["session_hour"] = df["timestamp_session_tz"].dt.hour
    df["session_date"] = df["timestamp_session_tz"].dt.date

    return df
