"""
HTF Bias
Higher Timeframe trend detection for trade filtering.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd
import numpy as np

from config import HTF_BIAS, TIME_CONFIG


class Bias(Enum):
    """Trend bias direction."""
    BULL = "BULL"
    BEAR = "BEAR"
    NEUTRAL = "NEUTRAL"


@dataclass
class HTFBiasResult:
    """HTF bias analysis result."""
    primary_bias: Bias = Bias.NEUTRAL
    secondary_bias: Bias = Bias.NEUTRAL
    primary_ema_fast: float = 0.0
    primary_ema_slow: float = 0.0
    secondary_ema_fast: float = 0.0
    secondary_ema_slow: float = 0.0


class HTFBiasEngine:
    """
    Higher Timeframe Bias calculator.

    Resamples M5 data to H4 and D1, calculates EMA cross trend,
    and maps bias back to M5 rows for filtering.
    """

    def __init__(
        self,
        enabled: bool = None,
        primary_timeframe: str = None,
        secondary_timeframe: str = None,
        method: str = None,
        ema_fast: int = None,
        ema_slow: int = None,
        require_primary_alignment: bool = None,
        require_secondary_alignment: bool = None,
        neutral_policy: str = None,
    ):
        """
        Initialize HTF bias engine.

        Args:
            enabled: Whether HTF bias filtering is enabled
            primary_timeframe: Primary HTF (default H4)
            secondary_timeframe: Secondary HTF (default D1)
            method: Bias detection method ("EMA_CROSS")
            ema_fast: Fast EMA period
            ema_slow: Slow EMA period
            require_primary_alignment: Require primary alignment
            require_secondary_alignment: Require secondary alignment
            neutral_policy: "BLOCK" or "ALLOW" when neutral
        """
        self.enabled = enabled if enabled is not None else HTF_BIAS["enabled"]
        self.primary_tf = primary_timeframe or HTF_BIAS["primary_timeframe"]
        self.secondary_tf = secondary_timeframe or HTF_BIAS["secondary_timeframe"]
        self.method = method or HTF_BIAS["method"]
        self.ema_fast = ema_fast if ema_fast is not None else HTF_BIAS["ema_fast"]
        self.ema_slow = ema_slow if ema_slow is not None else HTF_BIAS["ema_slow"]
        self.require_primary = require_primary_alignment if require_primary_alignment is not None else HTF_BIAS["require_primary_alignment"]
        self.require_secondary = require_secondary_alignment if require_secondary_alignment is not None else HTF_BIAS["require_secondary_alignment"]
        self.neutral_policy = neutral_policy or HTF_BIAS["neutral_policy"]

    def _get_resample_rule(self, timeframe: str) -> str:
        """Convert timeframe string to pandas resample rule."""
        tf_map = {
            "M5": "5min",
            "M15": "15min",
            "M30": "30min",
            "H1": "1h",
            "H4": "4h",
            "D1": "1D",
            "W1": "1W",
        }
        return tf_map.get(timeframe.upper(), "4h")

    def _resample_to_htf(
        self,
        df: pd.DataFrame,
        timeframe: str,
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Resample M5 data to higher timeframe.

        Args:
            df: M5 DataFrame with OHLC
            timeframe: Target timeframe (H4, D1)
            timestamp_col: Timestamp column name

        Returns:
            Resampled DataFrame
        """
        rule = self._get_resample_rule(timeframe)

        # Set timestamp as index
        df_temp = df.set_index(timestamp_col)

        # Convert to UTC for consistent bar boundaries, then convert back
        data_tz = TIME_CONFIG.get("data_timezone", "UTC")
        needs_conversion = data_tz != "UTC"
        if needs_conversion and df_temp.index.tz is None:
            df_temp.index = df_temp.index.tz_localize(data_tz).tz_convert("UTC")
        elif needs_conversion and df_temp.index.tz is not None:
            df_temp.index = df_temp.index.tz_convert("UTC")

        # Resample OHLC on UTC-aligned boundaries
        htf = df_temp.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum" if "volume" in df_temp.columns else "first",
        }).dropna()

        # Convert back to data timezone
        if needs_conversion:
            htf.index = htf.index.tz_convert(data_tz).tz_localize(None)

        htf = htf.reset_index()
        return htf

    def _calculate_ema_bias(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate EMA cross bias for a DataFrame.

        Args:
            df: OHLC DataFrame

        Returns:
            Series with bias values
        """
        ema_fast = df["close"].ewm(span=self.ema_fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.ema_slow, adjust=False).mean()

        bias = pd.Series(index=df.index, dtype=object)
        bias[ema_fast > ema_slow] = Bias.BULL.value
        bias[ema_fast < ema_slow] = Bias.BEAR.value
        bias[ema_fast == ema_slow] = Bias.NEUTRAL.value
        bias = bias.fillna(Bias.NEUTRAL.value)

        return bias, ema_fast, ema_slow

    def add_htf_bias(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Add HTF bias columns to M5 DataFrame.

        Args:
            df: M5 DataFrame with OHLC
            timestamp_col: Timestamp column name

        Returns:
            DataFrame with htf_bias_primary and htf_bias_secondary columns
        """
        df = df.copy()

        if not self.enabled:
            df["htf_bias_primary"] = Bias.NEUTRAL.value
            df["htf_bias_secondary"] = Bias.NEUTRAL.value
            return df

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Calculate primary HTF bias (H4)
        htf_primary = self._resample_to_htf(df, self.primary_tf, timestamp_col)
        if len(htf_primary) > self.ema_slow:
            bias_primary, ema_fast_p, ema_slow_p = self._calculate_ema_bias(htf_primary)
            htf_primary["htf_bias_primary"] = bias_primary
            htf_primary["primary_ema_fast"] = ema_fast_p
            htf_primary["primary_ema_slow"] = ema_slow_p
        else:
            htf_primary["htf_bias_primary"] = Bias.NEUTRAL.value
            htf_primary["primary_ema_fast"] = np.nan
            htf_primary["primary_ema_slow"] = np.nan

        # Calculate secondary HTF bias (D1)
        htf_secondary = self._resample_to_htf(df, self.secondary_tf, timestamp_col)
        if len(htf_secondary) > self.ema_slow:
            bias_secondary, ema_fast_s, ema_slow_s = self._calculate_ema_bias(htf_secondary)
            htf_secondary["htf_bias_secondary"] = bias_secondary
            htf_secondary["secondary_ema_fast"] = ema_fast_s
            htf_secondary["secondary_ema_slow"] = ema_slow_s
        else:
            htf_secondary["htf_bias_secondary"] = Bias.NEUTRAL.value
            htf_secondary["secondary_ema_fast"] = np.nan
            htf_secondary["secondary_ema_slow"] = np.nan

        # Map back to M5 using forward fill (asof merge)
        df = df.sort_values(timestamp_col)

        # Merge primary bias
        htf_primary = htf_primary.sort_values(timestamp_col)
        df = pd.merge_asof(
            df,
            htf_primary[[timestamp_col, "htf_bias_primary", "primary_ema_fast", "primary_ema_slow"]],
            on=timestamp_col,
            direction="backward",
        )

        # Merge secondary bias
        htf_secondary = htf_secondary.sort_values(timestamp_col)
        df = pd.merge_asof(
            df,
            htf_secondary[[timestamp_col, "htf_bias_secondary", "secondary_ema_fast", "secondary_ema_slow"]],
            on=timestamp_col,
            direction="backward",
        )

        # Fill any remaining NaN
        df["htf_bias_primary"] = df["htf_bias_primary"].fillna(Bias.NEUTRAL.value)
        df["htf_bias_secondary"] = df["htf_bias_secondary"].fillna(Bias.NEUTRAL.value)

        return df

    def get_bias_at_index(self, df: pd.DataFrame, idx: int) -> HTFBiasResult:
        """
        Get HTF bias at specific index.

        Args:
            df: DataFrame with HTF bias columns
            idx: Row index

        Returns:
            HTFBiasResult with bias values
        """
        row = df.iloc[idx]

        return HTFBiasResult(
            primary_bias=Bias(row.get("htf_bias_primary", Bias.NEUTRAL.value)),
            secondary_bias=Bias(row.get("htf_bias_secondary", Bias.NEUTRAL.value)),
            primary_ema_fast=row.get("primary_ema_fast", 0.0),
            primary_ema_slow=row.get("primary_ema_slow", 0.0),
            secondary_ema_fast=row.get("secondary_ema_fast", 0.0),
            secondary_ema_slow=row.get("secondary_ema_slow", 0.0),
        )

    def can_enter_trade(
        self,
        direction: str,
        df: pd.DataFrame,
        idx: int,
    ) -> tuple:
        """
        Check if entry is allowed based on HTF bias alignment.

        Args:
            direction: Trade direction ("LONG" or "SHORT")
            df: DataFrame with HTF bias columns
            idx: Current row index

        Returns:
            Tuple of (can_enter, bias_result, reason)
        """
        if not self.enabled:
            return True, HTFBiasResult(), ""

        bias = self.get_bias_at_index(df, idx)
        # Determine required bias for direction
        if direction == "LONG":
            required_bias = Bias.BULL
        else:
            required_bias = Bias.BEAR

        # Check primary alignment
        if self.require_primary:
            if bias.primary_bias == Bias.NEUTRAL:
                if self.neutral_policy == "BLOCK":
                    return False, bias, "htf_primary_neutral"
            elif bias.primary_bias != required_bias:
                return False, bias, f"htf_primary_{bias.primary_bias.value}_vs_{required_bias.value}"

        # Check secondary alignment
        if self.require_secondary:
            if bias.secondary_bias == Bias.NEUTRAL:
                if self.neutral_policy == "BLOCK":
                    return False, bias, "htf_secondary_neutral"
            elif bias.secondary_bias != required_bias:
                return False, bias, f"htf_secondary_{bias.secondary_bias.value}_vs_{required_bias.value}"

        return True, bias, ""


def create_htf_bias_engine(**kwargs) -> HTFBiasEngine:
    """Factory function to create HTF bias engine."""
    return HTFBiasEngine(**kwargs)
