"""Utility helpers for rolling calculations and safe operations."""
from typing import Iterable
import numpy as np
import pandas as pd


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling percentile rank of the latest value within the window.

    The percentile is calculated as the percentage of window values that are
    less than or equal to the current value. Results are expressed on a 0-100
    scale. NaN is returned when there are insufficient observations.
    """

    def percentile_window(values: Iterable[float]) -> float:
        arr = np.asarray(list(values), dtype=float)
        if arr.size == 0 or np.isnan(arr[-1]):
            return np.nan
        current = arr[-1]
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return np.nan
        rank = np.sum(valid <= current)
        return (rank / valid.size) * 100

    return series.rolling(window=window, min_periods=1).apply(percentile_window, raw=False)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Elementwise division that guards against division by zero."""
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average with minimum periods equal to window."""
    return series.rolling(window=window, min_periods=window).mean()
