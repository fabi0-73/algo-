"""
Key Levels
Previous Day High/Low, Weekly High/Low, Monthly High/Low for confluence scoring.
"""
from dataclasses import dataclass
from typing import Optional, Dict, List
import pandas as pd
import numpy as np

from config import KEY_LEVELS


@dataclass
class KeyLevelScore:
    """Score breakdown for key level proximity."""
    pdh_proximity: bool = False  # Near Previous Day High
    pdl_proximity: bool = False  # Near Previous Day Low
    weekly_high_proximity: bool = False
    weekly_low_proximity: bool = False
    monthly_high_proximity: bool = False
    monthly_low_proximity: bool = False
    total_score: int = 0


class KeyLevelsEngine:
    """
    Key levels calculator and scorer.

    Computes:
    - Previous Day High/Low (PDH/PDL)
    - Previous Week High/Low
    - Previous Month High/Low

    Scores entries based on proximity to key levels.
    """

    def __init__(
        self,
        enabled: bool = None,
        use_pdh_pdl: bool = None,
        use_weekly_high_low: bool = None,
        use_monthly_high_low: bool = None,
        tolerance_atr_mult: float = None,
        mode: str = None,
        min_score_required: int = None,
        score_weights: Dict[str, int] = None,
    ):
        """
        Initialize key levels engine.

        Args:
            enabled: Whether key levels filtering is enabled
            use_pdh_pdl: Use Previous Day High/Low
            use_weekly_high_low: Use Previous Week High/Low
            use_monthly_high_low: Use Previous Month High/Low
            tolerance_atr_mult: Tolerance for proximity (ATR multiple)
            mode: "SCORE" or "REQUIRE"
            min_score_required: Minimum score if mode is REQUIRE
            score_weights: Weights for each level type
        """
        self.enabled = enabled if enabled is not None else KEY_LEVELS["enabled"]
        self.use_pdh_pdl = use_pdh_pdl if use_pdh_pdl is not None else KEY_LEVELS["use_pdh_pdl"]
        self.use_weekly_high_low = use_weekly_high_low if use_weekly_high_low is not None else KEY_LEVELS["use_weekly_high_low"]
        self.use_monthly_high_low = use_monthly_high_low if use_monthly_high_low is not None else KEY_LEVELS["use_monthly_high_low"]
        self.tolerance_atr_mult = tolerance_atr_mult if tolerance_atr_mult is not None else KEY_LEVELS["tolerance_atr_mult"]
        self.mode = mode or KEY_LEVELS["mode"]
        self.min_score_required = min_score_required if min_score_required is not None else KEY_LEVELS["min_keylevel_score_required"]
        self.score_weights = score_weights or KEY_LEVELS["score_weights"]

    def add_key_levels(self, df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """
        Add key level columns to DataFrame.

        Args:
            df: DataFrame with OHLC and timestamp
            timestamp_col: Name of timestamp column

        Returns:
            DataFrame with key level columns added
        """
        df = df.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Add date column for grouping
        df["_date"] = df[timestamp_col].dt.date
        df["_week"] = df[timestamp_col].dt.isocalendar().week.astype(int)
        df["_year"] = df[timestamp_col].dt.year
        df["_month"] = df[timestamp_col].dt.to_period("M")

        # Calculate daily high/low
        daily_hl = df.groupby("_date").agg({
            "high": "max",
            "low": "min"
        }).rename(columns={"high": "day_high", "low": "day_low"})

        # Calculate weekly high/low
        weekly_hl = df.groupby(["_year", "_week"]).agg({
            "high": "max",
            "low": "min"
        }).rename(columns={"high": "week_high", "low": "week_low"})

        # Calculate monthly high/low
        monthly_hl = df.groupby("_month").agg({
            "high": "max",
            "low": "min"
        }).rename(columns={"high": "month_high", "low": "month_low"})

        # Add Previous Day High/Low
        if self.use_pdh_pdl:
            daily_hl["prev_day_high"] = daily_hl["day_high"].shift(1)
            daily_hl["prev_day_low"] = daily_hl["day_low"].shift(1)
            df = df.merge(
                daily_hl[["prev_day_high", "prev_day_low"]],
                left_on="_date",
                right_index=True,
                how="left"
            )
        else:
            df["prev_day_high"] = np.nan
            df["prev_day_low"] = np.nan

        # Add Previous Week High/Low
        if self.use_weekly_high_low:
            weekly_hl["prev_week_high"] = weekly_hl["week_high"].shift(1)
            weekly_hl["prev_week_low"] = weekly_hl["week_low"].shift(1)
            df = df.merge(
                weekly_hl[["prev_week_high", "prev_week_low"]],
                left_on=["_year", "_week"],
                right_index=True,
                how="left"
            )
        else:
            df["prev_week_high"] = np.nan
            df["prev_week_low"] = np.nan

        # Add Previous Month High/Low
        if self.use_monthly_high_low:
            monthly_hl["prev_month_high"] = monthly_hl["month_high"].shift(1)
            monthly_hl["prev_month_low"] = monthly_hl["month_low"].shift(1)
            df = df.merge(
                monthly_hl[["prev_month_high", "prev_month_low"]],
                left_on="_month",
                right_index=True,
                how="left"
            )
        else:
            df["prev_month_high"] = np.nan
            df["prev_month_low"] = np.nan

        # Clean up temporary columns
        df.drop(columns=["_date", "_week", "_year", "_month"], inplace=True)

        return df

    def calculate_score(
        self,
        price: float,
        row: pd.Series,
        atr: float,
    ) -> KeyLevelScore:
        """
        Calculate key level proximity score for a price.

        Args:
            price: Price to check (e.g., entry price or retest level)
            row: DataFrame row with key level columns
            atr: Current ATR for tolerance calculation

        Returns:
            KeyLevelScore with proximity flags and total score
        """
        if not self.enabled:
            return KeyLevelScore()

        tolerance = atr * self.tolerance_atr_mult
        score = KeyLevelScore()

        # Check PDH/PDL
        if self.use_pdh_pdl:
            pdh = row.get("prev_day_high", np.nan)
            pdl = row.get("prev_day_low", np.nan)

            if pd.notna(pdh) and abs(price - pdh) <= tolerance:
                score.pdh_proximity = True
                score.total_score += self.score_weights.get("pdh_pdl", 1)
            elif pd.notna(pdl) and abs(price - pdl) <= tolerance:
                score.pdl_proximity = True
                score.total_score += self.score_weights.get("pdh_pdl", 1)

        # Check Weekly High/Low
        if self.use_weekly_high_low:
            wh = row.get("prev_week_high", np.nan)
            wl = row.get("prev_week_low", np.nan)

            if pd.notna(wh) and abs(price - wh) <= tolerance:
                score.weekly_high_proximity = True
                score.total_score += self.score_weights.get("weekly", 1)
            elif pd.notna(wl) and abs(price - wl) <= tolerance:
                score.weekly_low_proximity = True
                score.total_score += self.score_weights.get("weekly", 1)

        # Check Monthly High/Low
        if self.use_monthly_high_low:
            mh = row.get("prev_month_high", np.nan)
            ml = row.get("prev_month_low", np.nan)

            if pd.notna(mh) and abs(price - mh) <= tolerance:
                score.monthly_high_proximity = True
                score.total_score += self.score_weights.get("monthly", 1)
            elif pd.notna(ml) and abs(price - ml) <= tolerance:
                score.monthly_low_proximity = True
                score.total_score += self.score_weights.get("monthly", 1)

        return score

    def can_enter_trade(
        self,
        price: float,
        row: pd.Series,
        atr: float,
    ) -> tuple:
        """
        Check if entry is allowed based on key level score.

        Args:
            price: Entry price to check
            row: DataFrame row with key level columns
            atr: Current ATR

        Returns:
            Tuple of (can_enter, score, reason)
        """
        if not self.enabled:
            return True, KeyLevelScore(), ""

        score = self.calculate_score(price, row, atr)

        if self.mode == "REQUIRE":
            if score.total_score < self.min_score_required:
                return False, score, f"key_level_score:{score.total_score}<{self.min_score_required}"

        return True, score, ""

    def get_nearby_levels(
        self,
        price: float,
        row: pd.Series,
        atr: float,
    ) -> List[tuple]:
        """
        Get list of nearby key levels.

        Args:
            price: Reference price
            row: DataFrame row with key level columns
            atr: Current ATR for tolerance

        Returns:
            List of (level_name, level_price, distance) tuples
        """
        levels = []
        tolerance = atr * self.tolerance_atr_mult * 2  # Wider for display

        level_cols = [
            ("PDH", "prev_day_high"),
            ("PDL", "prev_day_low"),
            ("Weekly High", "prev_week_high"),
            ("Weekly Low", "prev_week_low"),
            ("Monthly High", "prev_month_high"),
            ("Monthly Low", "prev_month_low"),
        ]

        for name, col in level_cols:
            level = row.get(col, np.nan)
            if pd.notna(level):
                distance = abs(price - level)
                if distance <= tolerance:
                    levels.append((name, level, distance))

        return sorted(levels, key=lambda x: x[2])


def create_key_levels_engine(**kwargs) -> KeyLevelsEngine:
    """Factory function to create key levels engine."""
    return KeyLevelsEngine(**kwargs)
