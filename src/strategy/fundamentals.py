"""
Fundamentals Filter
DXY (US Dollar Index) and Real Yields for gold trade filtering.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import os
import logging
import pandas as pd
import numpy as np

from config import FUNDAMENTALS, TIME_CONFIG

logger = logging.getLogger(__name__)


class Trend(Enum):
    """Trend direction."""
    UP = "UP"
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"


@dataclass
class FundamentalsResult:
    """Fundamentals analysis result."""
    dxy_trend: Trend = Trend.NEUTRAL
    yields_trend: Trend = Trend.NEUTRAL
    dxy_value: float = 0.0
    dxy_ma: float = 0.0
    yields_value: float = 0.0
    yields_ma: float = 0.0


class FundamentalsEngine:
    """
    Fundamentals-based trade filtering.

    Gold fundamentals logic:
    - Long Gold: DXY down AND real yields down (weak dollar, negative real rates)
    - Short Gold: DXY up AND real yields up (strong dollar, positive real rates)

    If data files are missing, filter is auto-disabled.
    """

    def __init__(
        self,
        enabled: bool = None,
        dxy_csv_path: str = None,
        real_yields_csv_path: str = None,
        resample_rule: str = None,
        dxy_ma_period: int = None,
        yields_ma_period: int = None,
        require_dxy_down_for_long: bool = None,
        require_yields_down_for_long: bool = None,
        require_dxy_up_for_short: bool = None,
        require_yields_up_for_short: bool = None,
        safe_haven_override: bool = None,
    ):
        """
        Initialize fundamentals engine.

        Args:
            enabled: Whether fundamentals filtering is enabled
            dxy_csv_path: Path to DXY CSV
            real_yields_csv_path: Path to real yields CSV
            resample_rule: Resample rule for alignment
            dxy_ma_period: MA period for DXY trend
            yields_ma_period: MA period for yields trend
            require_dxy_down_for_long: Require DXY down for long gold
            require_yields_down_for_long: Require yields down for long gold
            require_dxy_up_for_short: Require DXY up for short gold
            require_yields_up_for_short: Require yields up for short gold
            safe_haven_override: Bypass checks (crisis mode)
        """
        self.enabled = enabled if enabled is not None else FUNDAMENTALS["enabled"]
        self.dxy_csv_path = dxy_csv_path or FUNDAMENTALS["dxy_csv_path"]
        self.yields_csv_path = real_yields_csv_path or FUNDAMENTALS["real_yields_csv_path"]
        self.resample_rule = resample_rule or FUNDAMENTALS["resample_rule"]
        self.dxy_ma_period = dxy_ma_period if dxy_ma_period is not None else FUNDAMENTALS["dxy_ma_period"]
        self.yields_ma_period = yields_ma_period if yields_ma_period is not None else FUNDAMENTALS["yields_ma_period"]
        self.require_dxy_down_for_long = require_dxy_down_for_long if require_dxy_down_for_long is not None else FUNDAMENTALS["require_dxy_down_for_long"]
        self.require_yields_down_for_long = require_yields_down_for_long if require_yields_down_for_long is not None else FUNDAMENTALS["require_yields_down_for_long"]
        self.require_dxy_up_for_short = require_dxy_up_for_short if require_dxy_up_for_short is not None else FUNDAMENTALS["require_dxy_up_for_short"]
        self.require_yields_up_for_short = require_yields_up_for_short if require_yields_up_for_short is not None else FUNDAMENTALS["require_yields_up_for_short"]
        self.safe_haven_override = safe_haven_override if safe_haven_override is not None else FUNDAMENTALS["safe_haven_override"]

        self.dxy_df: Optional[pd.DataFrame] = None
        self.yields_df: Optional[pd.DataFrame] = None
        self._loaded = False

        if self.enabled:
            self._try_load_data()

    def _try_load_data(self):
        """Try to load fundamentals data from CSVs."""
        dxy_exists = os.path.exists(self.dxy_csv_path)
        yields_exists = os.path.exists(self.yields_csv_path)

        if not dxy_exists and not yields_exists:
            logger.warning("Fundamentals CSVs not found. Filter disabled.")
            self.enabled = False
            return

        try:
            if dxy_exists:
                self.dxy_df = self._load_csv(self.dxy_csv_path, "dxy")
                logger.info(f"Loaded {len(self.dxy_df)} DXY records")

            if yields_exists:
                self.yields_df = self._load_csv(self.yields_csv_path, "yields")
                logger.info(f"Loaded {len(self.yields_df)} yields records")

            self._loaded = True

        except Exception as e:
            logger.warning(f"Failed to load fundamentals data: {e}. Filter disabled.")
            self.enabled = False

    def _load_csv(self, path: str, name: str) -> pd.DataFrame:
        """Load and prepare a fundamentals CSV."""
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower().str.strip()

        # Parse timestamp
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        elif "datetime" in df.columns:
            df["timestamp"] = pd.to_datetime(df["datetime"], utc=True)
        elif "date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"], utc=True)
        else:
            raise ValueError(f"{name} CSV must have timestamp/datetime/date column")

        # Get value column
        if "value" in df.columns:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
        elif "close" in df.columns:
            df["value"] = pd.to_numeric(df["close"], errors="coerce")
        elif "price" in df.columns:
            df["value"] = pd.to_numeric(df["price"], errors="coerce")
        else:
            raise ValueError(f"{name} CSV must have value/close/price column")

        df = df.dropna(subset=["timestamp", "value"])
        df = df.sort_values("timestamp")

        return df[["timestamp", "value"]]

    def _calculate_trend(self, df: pd.DataFrame, ma_period: int) -> pd.DataFrame:
        """Calculate trend from value series."""
        df = df.copy()
        df["ma"] = df["value"].rolling(window=ma_period).mean()

        # Trend = UP if value > MA, DOWN if value < MA, NEUTRAL otherwise
        df["trend"] = Trend.NEUTRAL.value
        df.loc[df["value"] > df["ma"], "trend"] = Trend.UP.value
        df.loc[df["value"] < df["ma"], "trend"] = Trend.DOWN.value

        return df

    def add_fundamentals(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Add fundamentals columns to OHLC DataFrame.

        Args:
            df: OHLC DataFrame
            timestamp_col: Timestamp column name

        Returns:
            DataFrame with fundamentals columns
        """
        df = df.copy()

        if not self.enabled or not self._loaded:
            df["dxy_trend"] = Trend.NEUTRAL.value
            df["dxy_value"] = np.nan
            df["dxy_ma"] = np.nan
            df["yields_trend"] = Trend.NEUTRAL.value
            df["yields_value"] = np.nan
            df["yields_ma"] = np.nan
            return df

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Add timezone if missing
        if df[timestamp_col].dt.tz is None:
            try:
                from zoneinfo import ZoneInfo
            except ImportError:
                from backports.zoneinfo import ZoneInfo
            data_tz = ZoneInfo(TIME_CONFIG["data_timezone"])
            df[timestamp_col] = df[timestamp_col].dt.tz_localize(data_tz)

        df = df.sort_values(timestamp_col)

        # Add DXY trend
        if self.dxy_df is not None and len(self.dxy_df) > 0:
            dxy = self._calculate_trend(self.dxy_df, self.dxy_ma_period)
            dxy = dxy.rename(columns={
                "value": "dxy_value",
                "ma": "dxy_ma",
                "trend": "dxy_trend"
            })
            df = pd.merge_asof(
                df,
                dxy[["timestamp", "dxy_value", "dxy_ma", "dxy_trend"]],
                on=timestamp_col,
                direction="backward",
            )
        else:
            df["dxy_trend"] = Trend.NEUTRAL.value
            df["dxy_value"] = np.nan
            df["dxy_ma"] = np.nan

        # Add yields trend
        if self.yields_df is not None and len(self.yields_df) > 0:
            yields = self._calculate_trend(self.yields_df, self.yields_ma_period)
            yields = yields.rename(columns={
                "value": "yields_value",
                "ma": "yields_ma",
                "trend": "yields_trend"
            })
            df = pd.merge_asof(
                df,
                yields[["timestamp", "yields_value", "yields_ma", "yields_trend"]],
                on=timestamp_col,
                direction="backward",
            )
        else:
            df["yields_trend"] = Trend.NEUTRAL.value
            df["yields_value"] = np.nan
            df["yields_ma"] = np.nan

        # Fill NaN trends with NEUTRAL
        df["dxy_trend"] = df["dxy_trend"].fillna(Trend.NEUTRAL.value)
        df["yields_trend"] = df["yields_trend"].fillna(Trend.NEUTRAL.value)

        return df

    def get_fundamentals_at_index(self, df: pd.DataFrame, idx: int) -> FundamentalsResult:
        """
        Get fundamentals at specific index.

        Args:
            df: DataFrame with fundamentals columns
            idx: Row index

        Returns:
            FundamentalsResult with trend values
        """
        row = df.iloc[idx]

        return FundamentalsResult(
            dxy_trend=Trend(row.get("dxy_trend", Trend.NEUTRAL.value)),
            yields_trend=Trend(row.get("yields_trend", Trend.NEUTRAL.value)),
            dxy_value=row.get("dxy_value", 0.0),
            dxy_ma=row.get("dxy_ma", 0.0),
            yields_value=row.get("yields_value", 0.0),
            yields_ma=row.get("yields_ma", 0.0),
        )

    def can_enter_trade(
        self,
        direction: str,
        df: pd.DataFrame,
        idx: int,
    ) -> tuple:
        """
        Check if entry is allowed based on fundamentals.

        Args:
            direction: Trade direction ("LONG" or "SHORT")
            df: DataFrame with fundamentals columns
            idx: Current row index

        Returns:
            Tuple of (can_enter, fundamentals_result, reason)
        """
        if not self.enabled or self.safe_haven_override:
            return True, FundamentalsResult(), ""

        fund = self.get_fundamentals_at_index(df, idx)

        if direction == "LONG":
            # Long gold: DXY down AND yields down
            if self.require_dxy_down_for_long and fund.dxy_trend != Trend.DOWN:
                return False, fund, f"dxy_not_down:{fund.dxy_trend.value}"

            if self.require_yields_down_for_long and fund.yields_trend != Trend.DOWN:
                return False, fund, f"yields_not_down:{fund.yields_trend.value}"

        else:  # SHORT
            # Short gold: DXY up AND yields up
            if self.require_dxy_up_for_short and fund.dxy_trend != Trend.UP:
                return False, fund, f"dxy_not_up:{fund.dxy_trend.value}"

            if self.require_yields_up_for_short and fund.yields_trend != Trend.UP:
                return False, fund, f"yields_not_up:{fund.yields_trend.value}"

        return True, fund, ""


def create_fundamentals_engine(**kwargs) -> FundamentalsEngine:
    """Factory function to create fundamentals engine."""
    return FundamentalsEngine(**kwargs)
