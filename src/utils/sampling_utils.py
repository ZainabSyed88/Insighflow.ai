"""Stratified and uniform sampling utilities for large datasets."""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_ROWS = 10_000


def stratified_sample(
    df: pd.DataFrame,
    max_rows: int = DEFAULT_MAX_ROWS,
    strata_col: Optional[str] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a stratified sample of *df* with at most *max_rows* rows.

    If *strata_col* is given and is categorical, proportional stratified
    sampling is used.  Otherwise, a uniform random sample is taken.

    The original DataFrame is returned unchanged if it has ≤ max_rows rows.
    """
    if len(df) <= max_rows:
        return df

    if strata_col and strata_col in df.columns:
        try:
            frac = max_rows / len(df)
            sampled = df.groupby(strata_col, group_keys=False).apply(
                lambda x: x.sample(frac=frac, random_state=random_state)
            )
            if len(sampled) > max_rows:
                sampled = sampled.sample(n=max_rows, random_state=random_state)
            logger.info(f"Stratified sample on '{strata_col}': {len(df)} → {len(sampled)} rows")
            return sampled.reset_index(drop=True)
        except Exception:
            logger.warning(f"Stratified sampling failed on '{strata_col}', falling back to uniform")

    sampled = df.sample(n=max_rows, random_state=random_state)
    logger.info(f"Uniform sample: {len(df)} → {len(sampled)} rows")
    return sampled.reset_index(drop=True)


def auto_detect_strata_column(df: pd.DataFrame, max_cardinality: int = 50) -> Optional[str]:
    """Heuristic: pick the first low-cardinality object/category column as strata."""
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if 2 <= df[col].nunique() <= max_cardinality:
            return col
    return None
