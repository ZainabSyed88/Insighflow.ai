"""LangGraph state schema for multi-agent workflow."""

from typing import Annotated, Optional, Dict, List, Any
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
import pandas as pd
import tempfile
import os
import logging

logger = logging.getLogger(__name__)

# Directory for temporary data files
DATA_DIR = os.path.join(tempfile.gettempdir(), "ai_insights_data")
os.makedirs(DATA_DIR, exist_ok=True)

VIZ_DIR = os.path.join("data", "viz")
os.makedirs(VIZ_DIR, exist_ok=True)


def save_dataframe(df: pd.DataFrame, prefix: str = "data") -> str:
    """Save a DataFrame to a temporary Parquet file and return the path."""
    path = os.path.join(DATA_DIR, f"{prefix}_{id(df)}.parquet")
    df.to_parquet(path, index=False)
    logger.info(f"Saved DataFrame ({len(df)} rows) to {path}")
    return path


def load_dataframe(path: str) -> pd.DataFrame:
    """Load a DataFrame from a Parquet file path."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded DataFrame ({len(df)} rows) from {path}")
    return df


class WorkflowStage(str, Enum):
    """Workflow stages."""
    UPLOAD = "upload"
    CLEANING = "cleaning"
    CLEANING_APPROVAL = "cleaning_approval"
    VERIFICATION = "verification"
    VERIFICATION_APPROVAL = "verification_approval"
    ANALYSIS = "analysis"           # parallel fan-out stage
    INSIGHTS = "insights"
    VISUALIZATION = "visualization"
    REPORT_SYNTHESIS = "report_synthesis"
    COMPLETED = "completed"
    FAILED = "failed"


class ApprovalStatus(str, Enum):
    """User approval status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class CleaningAction(BaseModel):
    """A single, auditable cleaning operation with justification."""
    action: str = ""                 # e.g. "drop_rows", "impute", "drop_column", "type_cast"
    target: str = ""                 # column name or "all"
    rows_affected: int = 0
    detail: str = ""                 # human-readable explanation
    justification: str = ""          # why this action was taken
    before_snapshot: Dict[str, Any] = Field(default_factory=dict)   # stats before
    after_snapshot: Dict[str, Any] = Field(default_factory=dict)    # stats after


class CleaningResult(BaseModel):
    """Result of data cleaning operation."""
    raw_data_path: str = ""          # Path to original data Parquet
    cleaned_data_path: str = ""      # Path to cleaned data Parquet
    removed_rows: int = 0
    missing_handled: int = 0
    outliers_removed: int = 0
    data_quality_score: float = 0.0
    cleaning_summary: str = ""
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    cleaning_actions: List[CleaningAction] = Field(default_factory=list)  # per-step audit log


class VerificationResult(BaseModel):
    """Result of verification operation."""
    is_approved: bool = False
    percentage_data_removed: float = 0.0
    integrity_issues: List[str] = Field(default_factory=list)
    feedback_summary: str = ""
    recommendations: List[str] = Field(default_factory=list)
    severity: str = "low"  # low, medium, high
    column_checks: Dict[str, Any] = Field(default_factory=dict)     # per-column before/after
    action_verdicts: List[Dict[str, Any]] = Field(default_factory=list)  # verdict on each cleaning action


class InsightsResult(BaseModel):
    """Result of insights analysis."""
    patterns: Dict[str, Any] = Field(default_factory=dict)
    anomalies: List[Dict[str, Any]] = Field(default_factory=list)
    statistical_summary: Dict[str, Any] = Field(default_factory=dict)
    insights_text: str = ""
    key_findings: List[str] = Field(default_factory=list)


class VisualizationConfig(BaseModel):
    """Configuration for a single visualization."""
    chart_type: str = ""
    title: str = ""
    x_axis: str = ""
    y_axis: str = ""
    html_path: str = ""              # Path to saved Plotly HTML
    png_path: str = ""               # Path to static PNG (for PDF embedding)
    config: Dict[str, Any] = Field(default_factory=dict)


class TrendResult(BaseModel):
    """Result of trend analysis for a single column or overall."""
    column: str = ""
    trend_direction: str = ""        # "increasing", "decreasing", "no trend"
    is_significant: bool = False
    p_value: float = 1.0
    forecast_df_path: str = ""       # Parquet path if Prophet forecast was run
    decomposition_components: Dict[str, Any] = Field(default_factory=dict)
    narrative: str = ""


class AnomalyResult(BaseModel):
    """Result of anomaly / outlier detection."""
    total_anomalies: int = 0
    anomaly_rate: float = 0.0
    column_details: Dict[str, Any] = Field(default_factory=dict)
    flagged_rows_path: str = ""      # Parquet with anomaly flags
    reason_codes: List[str] = Field(default_factory=list)
    narrative: str = ""


class CorrelationResult(BaseModel):
    """Result of correlation analysis."""
    pearson_matrix_path: str = ""
    spearman_matrix_path: str = ""
    top_positive: List[Dict[str, Any]] = Field(default_factory=list)
    top_negative: List[Dict[str, Any]] = Field(default_factory=list)
    narrative: str = ""


class ReportSynthesis(BaseModel):
    """Merged report output from the synthesis agent."""
    executive_summary: str = ""
    data_quality_score: float = 0.0
    trend_findings: str = ""
    anomaly_findings: str = ""
    correlation_findings: str = ""
    recommendations: List[str] = Field(default_factory=list)
    all_figure_paths: List[str] = Field(default_factory=list)


class AppState(BaseModel):
    """Complete application state for LangGraph.

    DataFrames are stored as Parquet files on disk. Only file paths
    are kept in this schema so that LangGraph's MemorySaver can
    serialize the state without errors.
    """

    # File information
    upload_id: Optional[int] = None
    filename: str = ""
    file_path: str = ""              # Original uploaded file

    # Data paths (Parquet)
    raw_data_path: str = ""          # Path to raw data Parquet
    cleaned_data_path: str = ""      # Path to cleaned data Parquet

    # Cleaning stage
    cleaning_result: Optional[CleaningResult] = None
    cleaning_attempt: int = 0
    cleaning_approval_status: ApprovalStatus = ApprovalStatus.PENDING
    cleaning_approval_feedback: str = ""

    # Verification stage
    verification_result: Optional[VerificationResult] = None
    verification_attempts: int = 0
    verification_approval_status: ApprovalStatus = ApprovalStatus.PENDING
    verification_approval_feedback: str = ""

    # Analysis stage
    insights_result: Optional[InsightsResult] = None

    # New parallel analysis results
    trend_results: Dict[str, Any] = Field(default_factory=dict)
    anomaly_results: Dict[str, Any] = Field(default_factory=dict)
    correlation_results: Dict[str, Any] = Field(default_factory=dict)

    # Report synthesis
    report_synthesis: Optional[ReportSynthesis] = None

    # Visualization stage
    visualizations: List[VisualizationConfig] = Field(default_factory=list)
    visualization_data: Dict[str, Any] = Field(default_factory=dict)

    # PDF report
    pdf_path: str = ""
    pdf_ready: bool = False

    # Workflow control
    current_stage: WorkflowStage = WorkflowStage.UPLOAD
    last_error: Optional[str] = None
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)

    # Latency & caching
    agent_timings: Dict[str, float] = Field(default_factory=dict)
    file_hash: str = ""

    # Metadata
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
