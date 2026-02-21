"""File handling utilities for CSV and Excel files."""

import pandas as pd
import os
import hashlib
import tempfile
from pathlib import Path
from src.config import settings
import logging

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file for cache lookups.

    Args:
        file_path: Path to the file.

    Returns:
        Hex-digest string.
    """
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def lookup_cache(file_hash: str):
    """Check if an analysis cache entry exists for the given hash.

    Returns the AnalysisCache row or None.
    """
    try:
        from src.database.db_init import get_session_factory
        from src.database.models import AnalysisCache
        session = get_session_factory()()
        row = session.query(AnalysisCache).filter_by(file_hash=file_hash).first()
        session.close()
        return row
    except Exception as e:
        logger.warning(f"Cache lookup failed: {e}")
        return None


def save_cache(file_hash: str, state) -> None:
    """Persist analysis results to cache, keyed by file hash."""
    try:
        from src.database.db_init import get_session_factory
        from src.database.models import AnalysisCache
        session = get_session_factory()()
        entry = AnalysisCache(
            file_hash=file_hash,
            filename=state.filename,
            trend_results=state.trend_results if state.trend_results else None,
            anomaly_results=state.anomaly_results if state.anomaly_results else None,
            correlation_results=state.correlation_results if state.correlation_results else None,
            insights_text=state.insights_result.insights_text if state.insights_result else None,
            report_synthesis=state.report_synthesis.model_dump() if state.report_synthesis else None,
            pdf_path=state.pdf_path,
        )
        session.merge(entry)
        session.commit()
        session.close()
        logger.info(f"Cached analysis for hash {file_hash[:12]}…")
    except Exception as e:
        logger.warning(f"Cache save failed: {e}")


def validate_file(file_path: str) -> tuple[bool, str]:
    """
    Validate that file exists and is a supported format.
    
    Returns:
        Tuple of (is_valid, message)
    """
    if not os.path.exists(file_path):
        return False, "File not found"
    
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        return False, f"File size exceeds {settings.max_upload_size_mb}MB limit"
    
    ext = Path(file_path).suffix.lower()
    if ext not in ['.csv', '.xlsx', '.xls']:
        return False, "File must be CSV or Excel format"
    
    return True, "File valid"


def load_file(file_path: str) -> tuple[pd.DataFrame, str]:
    """
    Load CSV or Excel file into pandas DataFrame.
    
    Returns:
        Tuple of (DataFrame, message)
    """
    try:
        ext = Path(file_path).suffix.lower()
        
        if ext == '.csv':
            df = pd.read_csv(file_path)
            logger.info(f"Loaded CSV file: {file_path} ({len(df)} rows, {len(df.columns)} columns)")
        
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            logger.info(f"Loaded Excel file: {file_path} ({len(df)} rows, {len(df.columns)} columns)")
        
        else:
            return None, "Unsupported file format"
        
        return df, "File loaded successfully"
    
    except Exception as e:
        error_msg = f"Error loading file: {str(e)}"
        logger.error(error_msg)
        return None, error_msg


def save_dataframe(df: pd.DataFrame, filename: str) -> str:
    """
    Save DataFrame to CSV file.
    
    Returns:
        Path to saved file
    """
    try:
        data_dir = settings.database_dir / "cleaned_data"
        data_dir.mkdir(exist_ok=True, parents=True)
        
        file_path = data_dir / filename
        df.to_csv(file_path, index=False)
        logger.info(f"Saved DataFrame to {file_path}")
        
        return str(file_path)
    
    except Exception as e:
        logger.error(f"Error saving DataFrame: {str(e)}")
        return None


def create_temp_file(df: pd.DataFrame, suffix: str = ".csv") -> str:
    """
    Create temporary file with DataFrame.
    
    Returns:
        Path to temp file
    """
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            if suffix == '.csv':
                df.to_csv(f.name, index=False)
            elif suffix in ['.xlsx', '.xls']:
                df.to_excel(f.name, index=False)
            
            logger.info(f"Created temp file: {f.name}")
            return f.name
    
    except Exception as e:
        logger.error(f"Error creating temp file: {str(e)}")
        return None
