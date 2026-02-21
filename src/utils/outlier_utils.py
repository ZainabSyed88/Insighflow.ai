"""Shared outlier detection utilities used by cleaning and anomaly agents."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def iqr_outlier_mask(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Return a boolean mask where True indicates an outlier via IQR method.

    Args:
        series: Numeric pandas Series.
        multiplier: IQR fence multiplier (default 1.5).

    Returns:
        Boolean Series — True for outlier rows.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (series < lower) | (series > upper)


def iqr_bounds(series: pd.Series, multiplier: float = 1.5) -> Tuple[float, float]:
    """Return (lower_bound, upper_bound) for a numeric series using IQR."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - multiplier * iqr, q3 + multiplier * iqr


def zscore_outlier_mask(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Return a boolean mask where True indicates a Z-score outlier.

    Args:
        series: Numeric pandas Series.
        threshold: Z-score threshold (default 3.0).

    Returns:
        Boolean Series — True for outlier rows.
    """
    from scipy import stats

    z_scores = np.abs(stats.zscore(series.dropna()))
    mask = pd.Series(False, index=series.index)
    mask.loc[series.dropna().index] = z_scores > threshold
    return mask


def detect_outliers_all_methods(
    df: pd.DataFrame,
    numeric_cols: list | None = None,
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0,
) -> Dict[str, Dict[str, pd.Series]]:
    """Run IQR and Z-score outlier detection on all numeric columns.

    Returns:
        Dict mapping column name → { "iqr": bool_mask, "zscore": bool_mask }
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    results: Dict[str, Dict[str, pd.Series]] = {}
    for col in numeric_cols:
        if df[col].dropna().empty:
            continue
        results[col] = {
            "iqr": iqr_outlier_mask(df[col], iqr_multiplier),
            "zscore": zscore_outlier_mask(df[col], zscore_threshold),
        }
    return results
