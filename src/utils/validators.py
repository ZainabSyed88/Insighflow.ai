"""Data validation utilities."""

import pandas as pd
import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate DataFrame structure and content.
    
    Returns:
        Tuple of (is_valid, messages)
    """
    messages = []
    
    if df is None:
        return False, ["DataFrame is None"]
    
    if len(df) == 0:
        return False, ["DataFrame is empty"]
    
    if len(df.columns) == 0:
        return False, ["DataFrame has no columns"]
    
    return True, messages


def validate_columns(df: pd.DataFrame, required_columns: List[str] = None) -> Tuple[bool, List[str]]:
    """
    Validate that required columns exist.
    
    Returns:
        Tuple of (is_valid, messages)
    """
    messages = []
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            messages.append(f"Missing required columns: {missing_cols}")
            return False, messages
    
    return True, messages


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Check overall data quality metrics.
    
    Returns:
        Dictionary with quality metrics
    """
    metrics = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values_total": int(df.isnull().sum().sum()),
        "missing_percentage": float((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": len(df.select_dtypes(include=['object']).columns),
        "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 ** 2),
    }
    
    return metrics
