"""SQLAlchemy models for data persistence."""

from sqlalchemy import Column, Integer, String, DateTime, Float, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import json

Base = declarative_base()


class Upload(Base):
    """Track uploaded files and their metadata."""
    
    __tablename__ = "uploads"
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    row_count = Column(Integer)
    column_count = Column(Integer)
    columns = Column(JSON)  # Store column names and types
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    file_size_bytes = Column(Integer)


class CleaningHistory(Base):
    """Track data cleaning attempts and actions."""
    
    __tablename__ = "cleaning_history"
    
    id = Column(Integer, primary_key=True)
    upload_id = Column(Integer, nullable=False)  # Foreign key to uploads
    attempt_number = Column(Integer, default=1)
    outliers_removed = Column(Integer)
    missing_handled = Column(Integer)
    data_quality_score = Column(Float)  # 0-1 score
    cleaning_summary = Column(Text)
    cleaned_data_path = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)


class VerificationFeedback(Base):
    """Track verification agent feedback and outcomes."""
    
    __tablename__ = "verification_feedback"
    
    id = Column(Integer, primary_key=True)
    upload_id = Column(Integer, nullable=False)
    cleaning_attempt_id = Column(Integer, nullable=False)
    is_approved = Column(Boolean, nullable=False)
    percentage_data_removed = Column(Float)
    integrity_issues = Column(JSON)  # Store list of issues found
    feedback_summary = Column(Text)
    recommendations = Column(JSON)  # Store list of recommendations
    created_at = Column(DateTime, default=datetime.utcnow)


class FinalInsights(Base):
    """Store final analysis insights and results."""
    
    __tablename__ = "final_insights"
    
    id = Column(Integer, primary_key=True)
    upload_id = Column(Integer, nullable=False)
    patterns = Column(JSON)  # Store correlation analysis, trends
    anomalies = Column(JSON)  # Store detected anomalies
    statistical_summary = Column(JSON)  # Store summary statistics
    insights_text = Column(Text)
    visualizations_config = Column(JSON)  # Store chart configurations
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class GraphCheckpoint(Base):
    """Store LangGraph state checkpoints for resumability."""
    
    __tablename__ = "graph_checkpoints"
    
    id = Column(Integer, primary_key=True)
    upload_id = Column(Integer, nullable=False)
    checkpoint_id = Column(String(100), nullable=False)
    stage = Column(String(50))  # cleaning, verification, insights, visualization
    state_data = Column(JSON)  # Full state snapshot
    created_at = Column(DateTime, default=datetime.utcnow)


class AnalysisCache(Base):
    """Cache analysis results keyed by file content hash for instant re-use."""

    __tablename__ = "analysis_cache"

    id = Column(Integer, primary_key=True)
    file_hash = Column(String(64), nullable=False, unique=True, index=True)
    filename = Column(String(255))
    row_count = Column(Integer)
    column_count = Column(Integer)
    trend_results = Column(JSON)
    anomaly_results = Column(JSON)
    correlation_results = Column(JSON)
    insights_text = Column(Text)
    report_synthesis = Column(JSON)
    pdf_path = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)  # optional TTL
