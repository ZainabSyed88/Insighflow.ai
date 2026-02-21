"""Tests for new agents and utilities in the Analyst-in-a-Box extension."""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.state import (
    AppState, CleaningResult, VerificationResult, InsightsResult,
    save_dataframe, load_dataframe,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Create a small test DataFrame with datetime and numeric columns."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "sales": np.random.normal(1000, 200, n).astype(int),
        "revenue": np.random.normal(5000, 800, n),
        "units": np.random.poisson(50, n),
        "category": np.random.choice(["A", "B", "C"], n),
    })
    # Inject a few outliers
    df.loc[5, "sales"] = 99999
    df.loc[10, "revenue"] = -50000
    return df


@pytest.fixture
def sample_state(sample_df, tmp_path):
    """Create a minimal AppState with cleaned data on disk."""
    path = str(tmp_path / "test_cleaned.parquet")
    sample_df.to_parquet(path, index=False)
    state = AppState()
    state.cleaned_data_path = path
    state.filename = "test_data.csv"
    state.file_path = path
    state.cleaning_result = CleaningResult(
        data_quality_score=0.95, removed_rows=2, missing_handled=1,
    )
    state.verification_result = VerificationResult(is_approved=True)
    return state


# ---------------------------------------------------------------------------
# 1. Outlier Utilities
# ---------------------------------------------------------------------------

class TestOutlierUtils:
    def test_iqr_outlier_mask(self, sample_df):
        from src.utils.outlier_utils import iqr_outlier_mask
        mask = iqr_outlier_mask(sample_df["sales"], multiplier=1.5)
        assert mask.dtype == bool
        assert mask.sum() > 0  # should catch the 99999 outlier

    def test_zscore_outlier_mask(self, sample_df):
        from src.utils.outlier_utils import zscore_outlier_mask
        mask = zscore_outlier_mask(sample_df["revenue"], threshold=3.0)
        assert mask.dtype == bool

    def test_detect_outliers_all_methods(self, sample_df):
        from src.utils.outlier_utils import detect_outliers_all_methods
        results = detect_outliers_all_methods(sample_df, numeric_cols=["sales", "revenue"])
        assert "sales" in results
        assert "iqr" in results["sales"]
        assert "zscore" in results["sales"]


# ---------------------------------------------------------------------------
# 2. Sampling Utilities
# ---------------------------------------------------------------------------

class TestSamplingUtils:
    def test_no_sampling_small_df(self, sample_df):
        from src.utils.sampling_utils import stratified_sample
        result = stratified_sample(sample_df, max_rows=200)
        assert len(result) == len(sample_df)

    def test_uniform_sampling(self, sample_df):
        from src.utils.sampling_utils import stratified_sample
        big_df = pd.concat([sample_df] * 200, ignore_index=True)
        result = stratified_sample(big_df, max_rows=500)
        assert len(result) == 500


# ---------------------------------------------------------------------------
# 3. Chart Utilities
# ---------------------------------------------------------------------------

class TestChartUtils:
    def test_correlation_heatmap(self, sample_df):
        from src.utils.chart_utils import create_correlation_heatmap
        corr = sample_df[["sales", "revenue", "units"]].corr()
        fig = create_correlation_heatmap(corr)
        assert fig is not None

    def test_moving_average_traces(self, sample_df):
        from src.utils.chart_utils import create_moving_average_traces
        fig = create_moving_average_traces(sample_df, "sales", x_col="date")
        assert fig is not None


# ---------------------------------------------------------------------------
# 4. TrendAgent
# ---------------------------------------------------------------------------

class TestTrendAgent:
    @patch("src.agents.trend_agent.call_llm", return_value="Trend looks positive.")
    def test_execute(self, mock_llm, sample_state):
        from src.agents.trend_agent import TrendAgent
        agent = TrendAgent.__new__(TrendAgent)
        agent.name = "TrendAgent"
        agent.model = "test"
        agent.llm = MagicMock()
        agent._log_action = MagicMock()

        state = agent.execute(sample_state)
        assert "columns" in state.trend_results
        assert len(state.trend_results["columns"]) > 0
        assert "TrendAgent" in state.agent_timings


# ---------------------------------------------------------------------------
# 5. AnomalyAgent
# ---------------------------------------------------------------------------

class TestAnomalyAgent:
    @patch("src.agents.anomaly_agent.call_llm", return_value="2 anomalies found.")
    def test_execute(self, mock_llm, sample_state):
        from src.agents.anomaly_agent import AnomalyAgent
        agent = AnomalyAgent.__new__(AnomalyAgent)
        agent.name = "AnomalyAgent"
        agent.model = "test"
        agent.llm = MagicMock()
        agent._log_action = MagicMock()

        state = agent.execute(sample_state)
        assert "total_anomalies" in state.anomaly_results
        assert state.anomaly_results["total_anomalies"] >= 0
        assert "AnomalyAgent" in state.agent_timings


# ---------------------------------------------------------------------------
# 6. CorrelationAgent
# ---------------------------------------------------------------------------

class TestCorrelationAgent:
    @patch("src.agents.correlation_agent.call_llm", return_value="Sales and revenue are correlated.")
    def test_execute(self, mock_llm, sample_state):
        from src.agents.correlation_agent import CorrelationAgent
        agent = CorrelationAgent.__new__(CorrelationAgent)
        agent.name = "CorrelationAgent"
        agent.model = "test"
        agent.llm = MagicMock()
        agent._log_action = MagicMock()

        state = agent.execute(sample_state)
        assert "top_positive" in state.correlation_results or "skipped" in state.correlation_results
        assert "CorrelationAgent" in state.agent_timings


# ---------------------------------------------------------------------------
# 7. ReportSynthesisAgent
# ---------------------------------------------------------------------------

class TestReportSynthesisAgent:
    @patch("src.agents.report_synthesis_agent.call_llm")
    def test_execute(self, mock_llm, sample_state):
        mock_llm.side_effect = [
            "Executive summary here.",
            '["Rec 1", "Rec 2", "Rec 3"]',
        ]
        sample_state.trend_results = {"overall_narrative": "Up trend.", "columns": []}
        sample_state.anomaly_results = {"narrative": "Few anomalies.", "anomaly_rate": 0.02}
        sample_state.correlation_results = {"narrative": "Sales correlates with revenue."}
        sample_state.insights_result = InsightsResult(insights_text="Good data.")

        from src.agents.report_synthesis_agent import ReportSynthesisAgent
        agent = ReportSynthesisAgent.__new__(ReportSynthesisAgent)
        agent.name = "ReportSynthesisAgent"
        agent.model = "test"
        agent.llm = MagicMock()
        agent._log_action = MagicMock()

        # Patch out background PDF generation
        with patch.object(agent, "_start_pdf_generation"):
            state = agent.execute(sample_state)

        assert state.report_synthesis is not None
        assert state.report_synthesis.data_quality_score > 0
        assert "ReportSynthesisAgent" in state.agent_timings


# ---------------------------------------------------------------------------
# 8. PDF Generator (template rendering only — no WeasyPrint needed)
# ---------------------------------------------------------------------------

class TestPDFGenerator:
    def test_render_html(self, sample_state):
        sample_state.trend_results = {"overall_narrative": "Up.", "columns": []}
        sample_state.anomaly_results = {"narrative": "Clean.", "anomaly_rate": 0.01}
        sample_state.correlation_results = {"narrative": "Correlated."}
        sample_state.insights_result = InsightsResult(
            insights_text="Key insight.", statistical_summary={"total_rows": 100},
        )
        from src.graph.state import ReportSynthesis
        sample_state.report_synthesis = ReportSynthesis(
            executive_summary="All good.",
            data_quality_score=88.0,
            recommendations=["Do X", "Do Y"],
        )

        from src.utils.pdf_generator import _render_html
        html = _render_html(sample_state)
        assert "All good." in html
        assert "88" in html


# ---------------------------------------------------------------------------
# 9. File handler hashing
# ---------------------------------------------------------------------------

class TestFileHandlers:
    def test_compute_file_hash(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("a,b\n1,2\n3,4")
        from src.utils.file_handlers import compute_file_hash
        h = compute_file_hash(str(f))
        assert len(h) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# 10. Integration smoke test (mocked LLM)
# ---------------------------------------------------------------------------

class TestWorkflowIntegration:
    """Light integration test — runs parallel analysis with LLM mocked."""

    @patch("src.agents.trend_agent.call_llm", return_value="Up trend noted.")
    @patch("src.agents.anomaly_agent.call_llm", return_value="1 anomaly.")
    @patch("src.agents.correlation_agent.call_llm", return_value="Correlated.")
    @patch("src.agents.report_synthesis_agent.call_llm")
    def test_parallel_analysis_smoke(self, mock_synth_llm, mock_corr, mock_anom, mock_trend, sample_state):
        mock_synth_llm.side_effect = [
            "Summary here.",
            '["Rec A"]',
        ]
        # Also patch the agents that use REPL (Insights, Visualization)
        with patch("src.agents.insights_agent.InsightsAgent.execute", return_value=sample_state):
            with patch("src.agents.visualization_agent.VisualizationAgent.execute", return_value=sample_state):
                with patch("src.agents.report_synthesis_agent.ReportSynthesisAgent._start_pdf_generation"):
                    from src.graph.workflow import WorkflowOrchestrator
                    orch = WorkflowOrchestrator()
                    result = orch.run_parallel_analysis(sample_state)

        assert result.current_stage == "completed"
        assert "_total_analysis" in result.agent_timings
