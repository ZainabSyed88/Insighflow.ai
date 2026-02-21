"""Report synthesis agent — merges all parallel agent outputs into a unified report."""

import time
import logging
import threading
from typing import Any, Dict, Optional

from src.agents.base_agent import Agent
from src.graph.state import AppState, ReportSynthesis, load_dataframe
from src.utils.llm_utils import call_llm
from src.utils.logger import audit
from src.config import settings
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

EXEC_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior data analyst writing an executive summary for a "
     "business stakeholder. Synthesize the following analysis sections into "
     "a coherent 5-7 sentence executive summary. Highlight key takeaways, "
     "risks, and recommended actions. Be specific with numbers."),
    ("user", "{synthesis_context}"),
])

RECOMMENDATIONS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a data-driven consultant. Based on the following analysis of a "
     "dataset (trends, anomalies, correlations, data quality), produce exactly "
     "5 actionable business recommendations as a JSON array of strings."),
    ("user", "{rec_context}"),
])


class ReportSynthesisAgent(Agent):
    """Agent that merges outputs from parallel analysis agents and generates
    the final executive summary, quality score, and PDF."""

    def __init__(self):
        super().__init__(name="ReportSynthesisAgent")

    def execute(self, state: AppState, user_context: dict | None = None) -> AppState:
        audit("ReportSynthesisAgent.execute START", filename=state.filename)
        from src.utils.logger import StepTimer
        timer = StepTimer("ReportSynthesisAgent", logger)
        t0 = time.time()

        try:
            # --- Compute Data Quality Score (0-100) ---
            quality_score = self._compute_quality_score(state)

            # --- Gather section narratives ---
            trend_text = state.trend_results.get("overall_narrative", "No trend analysis available.")
            anomaly_text = state.anomaly_results.get("narrative", "No anomaly analysis available.")
            corr_text = state.correlation_results.get("narrative", "No correlation analysis available.")
            insights_text = state.insights_result.insights_text if state.insights_result else ""

            synthesis_context = (
                f"DATA QUALITY SCORE: {quality_score}/100\n\n"
                f"TREND ANALYSIS:\n{trend_text}\n\n"
                f"ANOMALY DETECTION:\n{anomaly_text}\n\n"
                f"CORRELATION ANALYSIS:\n{corr_text}\n\n"
                f"GENERAL INSIGHTS:\n{insights_text}\n"
            )

            # --- Executive Summary (use primary model for best quality) ---
            try:
                executive_summary = call_llm(
                    EXEC_SUMMARY_PROMPT,
                    {"synthesis_context": synthesis_context},
                    temperature=0.5,
                )
            except Exception as e:
                logger.warning(f"Executive summary LLM failed: {e}")
                executive_summary = (
                    f"Analysis complete. Data quality score: {quality_score}/100. "
                    "See individual sections for details."
                )

            # --- Recommendations ---
            try:
                import json
                recs_raw = call_llm(
                    RECOMMENDATIONS_PROMPT,
                    {"rec_context": synthesis_context},
                    temperature=0.4,
                )
                recs_text = recs_raw.strip()
                if recs_text.startswith("```"):
                    recs_text = recs_text.strip("`").strip("json").strip()
                recommendations = json.loads(recs_text)
                if not isinstance(recommendations, list):
                    recommendations = [str(recommendations)]
            except Exception as e:
                logger.warning(f"Recommendations LLM failed: {e}")
                recommendations = ["Review flagged anomalies", "Monitor trend columns", "Investigate strong correlations"]

            # --- Collect all figure paths ---
            all_fig_paths = [v.html_path for v in state.visualizations if v.html_path]

            state.report_synthesis = ReportSynthesis(
                executive_summary=executive_summary,
                data_quality_score=quality_score,
                trend_findings=trend_text,
                anomaly_findings=anomaly_text,
                correlation_findings=corr_text,
                recommendations=recommendations,
                all_figure_paths=all_fig_paths,
            )

            # --- Kick off PDF generation in background thread ---
            self._start_pdf_generation(state)

        except Exception as e:
            logger.error(f"ReportSynthesisAgent error: {e}")
            state.last_error = f"Report synthesis error: {e}"

        state.agent_timings["ReportSynthesisAgent"] = round(time.time() - t0, 2)
        timer.summary()
        audit("ReportSynthesisAgent.execute END", duration=state.agent_timings["ReportSynthesisAgent"])
        return state

    # ------------------------------------------------------------------
    def _compute_quality_score(self, state: AppState) -> float:
        """Compute a 0-100 data health score from cleaning + anomaly results."""
        score = 100.0

        # Deduction for missing values remaining
        if state.cleaning_result:
            cr = state.cleaning_result
            if cr.data_quality_score < 1.0:
                score -= (1.0 - cr.data_quality_score) * 30  # up to -30

        # Deduction for anomaly rate
        anomaly_rate = state.anomaly_results.get("anomaly_rate", 0.0)
        score -= min(anomaly_rate * 200, 30)  # up to -30

        # Deduction for verification issues
        if state.verification_result and state.verification_result.integrity_issues:
            score -= min(len(state.verification_result.integrity_issues) * 5, 20)

        return round(max(0.0, min(100.0, score)), 1)

    # ------------------------------------------------------------------
    def _start_pdf_generation(self, state: AppState):
        """Launch PDF generation in a background thread.

        Note: Streamlit reruns can lose the thread reference, but the file
        will be written to disk regardless.  The UI polls pdf_ready / pdf_path.
        """
        try:
            from src.utils.pdf_generator import generate_pdf

            def _generate():
                try:
                    pdf_path = generate_pdf(state)
                    state.pdf_path = pdf_path
                    state.pdf_ready = True
                    logger.info(f"PDF generated: {pdf_path}")
                except Exception as e:
                    logger.error(f"PDF generation failed: {e}")

            thread = threading.Thread(target=_generate, daemon=True)
            thread.start()
        except ImportError:
            logger.warning("PDF generator not available — skipping PDF creation")
