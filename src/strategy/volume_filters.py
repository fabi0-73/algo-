"""
Volume Filters
Volume confirmation for distribution candles.
"""
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

from config import VOLUME_FILTER


@dataclass
class VolumeAnalysis:
    """Volume analysis result."""
    candle_volume: float = 0.0
    volume_ma: float = 0.0
    volume_ratio: float = 0.0
    consolidation_avg_volume: float = 0.0
    meets_ratio_threshold: bool = False
    meets_min_volume: bool = False
    valid: bool = False


class VolumeFilterEngine:
    """
    Volume-based trade filtering.

    Requires:
    - Distribution candle volume >= distribution_volume_ratio_min * consolidation avg
    - Minimum tick volume threshold
    """

    def __init__(
        self,
        enabled: bool = None,
        volume_ma_period: int = None,
        distribution_volume_ratio_min: float = None,
        min_tick_volume: int = None,
    ):
        """
        Initialize volume filter engine.

        Args:
            enabled: Whether volume filtering is enabled
            volume_ma_period: Period for volume MA
            distribution_volume_ratio_min: Min ratio vs consolidation avg
            min_tick_volume: Minimum tick volume threshold
        """
        self.enabled = enabled if enabled is not None else VOLUME_FILTER["enabled"]
        self.volume_ma_period = volume_ma_period if volume_ma_period is not None else VOLUME_FILTER["volume_ma_period"]
        self.distribution_volume_ratio_min = distribution_volume_ratio_min if distribution_volume_ratio_min is not None else VOLUME_FILTER["distribution_volume_ratio_min"]
        self.min_tick_volume = min_tick_volume if min_tick_volume is not None else VOLUME_FILTER["min_tick_volume"]

    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume indicator columns to DataFrame.

        Args:
            df: DataFrame with 'volume' column

        Returns:
            DataFrame with volume_ma column
        """
        df = df.copy()

        if "volume" not in df.columns:
            df["volume"] = 0
            df["volume_ma"] = 0
            return df

        df["volume_ma"] = df["volume"].rolling(window=self.volume_ma_period).mean()

        return df

    def analyze_distribution_volume(
        self,
        df: pd.DataFrame,
        distribution_idx: int,
        consolidation_start_idx: int,
        consolidation_end_idx: int,
    ) -> VolumeAnalysis:
        """
        Analyze volume at distribution candle vs consolidation.

        Args:
            df: DataFrame with volume column
            distribution_idx: Index of distribution candle
            consolidation_start_idx: Start of consolidation window
            consolidation_end_idx: End of consolidation window

        Returns:
            VolumeAnalysis with results
        """
        result = VolumeAnalysis()

        if "volume" not in df.columns:
            result.valid = True  # No volume data = pass filter
            return result

        # Get distribution candle volume
        dist_volume = df.iloc[distribution_idx]["volume"]
        result.candle_volume = dist_volume

        # Get volume MA
        if "volume_ma" in df.columns:
            result.volume_ma = df.iloc[distribution_idx]["volume_ma"]
        else:
            result.volume_ma = df["volume"].iloc[max(0, distribution_idx - self.volume_ma_period):distribution_idx].mean()

        # Calculate consolidation average volume
        consol_window = df.iloc[consolidation_start_idx:consolidation_end_idx + 1]
        if len(consol_window) > 0:
            result.consolidation_avg_volume = consol_window["volume"].mean()
        else:
            result.consolidation_avg_volume = result.volume_ma

        # Calculate ratio
        if result.consolidation_avg_volume > 0:
            result.volume_ratio = dist_volume / result.consolidation_avg_volume
        else:
            result.volume_ratio = 1.0

        # Check thresholds
        result.meets_ratio_threshold = result.volume_ratio >= self.distribution_volume_ratio_min
        result.meets_min_volume = dist_volume >= self.min_tick_volume
        result.valid = result.meets_ratio_threshold and result.meets_min_volume

        return result

    def is_volume_valid(
        self,
        df: pd.DataFrame,
        candle_idx: int,
    ) -> tuple:
        """
        Quick check if candle has minimum valid volume.

        Args:
            df: DataFrame with volume column
            candle_idx: Index of candle to check

        Returns:
            Tuple of (valid, volume)
        """
        if not self.enabled:
            return True, 0

        if "volume" not in df.columns:
            return True, 0

        volume = df.iloc[candle_idx]["volume"]
        valid = volume >= self.min_tick_volume

        return valid, volume

    def can_enter_trade(
        self,
        df: pd.DataFrame,
        distribution_idx: int,
        consolidation_start_idx: int,
        consolidation_end_idx: int,
    ) -> tuple:
        """
        Check if volume confirms distribution for trade entry.

        Args:
            df: DataFrame with volume column
            distribution_idx: Index of distribution candle
            consolidation_start_idx: Start of consolidation
            consolidation_end_idx: End of consolidation

        Returns:
            Tuple of (can_enter, volume_analysis, reason)
        """
        if not self.enabled:
            return True, VolumeAnalysis(valid=True), ""

        analysis = self.analyze_distribution_volume(
            df, distribution_idx,
            consolidation_start_idx, consolidation_end_idx
        )

        if not analysis.meets_min_volume:
            return False, analysis, f"low_volume:{analysis.candle_volume}<{self.min_tick_volume}"

        if not analysis.meets_ratio_threshold:
            return False, analysis, f"low_volume_ratio:{analysis.volume_ratio:.2f}<{self.distribution_volume_ratio_min}"

        return True, analysis, ""


def create_volume_filter_engine(**kwargs) -> VolumeFilterEngine:
    """Factory function to create volume filter engine."""
    return VolumeFilterEngine(**kwargs)
