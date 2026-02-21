"""Workflow orchestrator with database persistence and parallel analysis agents."""

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from src.graph.state import AppState, WorkflowStage, InsightsResult, ReportSynthesis
from src.agents.cleaning_agent import DataCleaningAgent
from src.agents.verification_agent import VerificationAgent
from src.agents.insights_agent import InsightsAgent
from src.agents.visualization_agent import VisualizationAgent
from src.agents.trend_agent import TrendAgent
from src.agents.anomaly_agent import AnomalyAgent
from src.agents.correlation_agent import CorrelationAgent
from src.agents.report_synthesis_agent import ReportSynthesisAgent
from src.database.db_init import get_session_factory
from src.database.models import Upload, CleaningHistory, VerificationFeedback, FinalInsights
from src.utils.file_handlers import compute_file_hash, lookup_cache, save_cache

logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """Orchestrates the multi-agent workflow with database persistence.
    
    Each agent internally uses LangGraph (e.g., create_react_agent) for
    LLM-powered tool calling. This orchestrator calls them in sequence,
    commits results to the DB, and lets the Streamlit UI manage approval gates.
    """
    
    def __init__(self):
        self._session_factory = None
    
    @property
    def session_factory(self):
        if self._session_factory is None:
            self._session_factory = get_session_factory()
        return self._session_factory
    
    def _get_session(self):
        return self.session_factory()
    
    def save_upload(self, state: AppState) -> int:
        """Save upload record and return the upload_id."""
        session = self._get_session()
        try:
            from src.graph.state import load_dataframe
            df = load_dataframe(state.raw_data_path)
            upload = Upload(
                filename=state.filename,
                file_path=state.file_path,
                row_count=len(df),
                column_count=len(df.columns),
                columns={col: str(dtype) for col, dtype in df.dtypes.items()},
                uploaded_at=datetime.utcnow(),
                file_size_bytes=0,
            )
            session.add(upload)
            session.commit()
            upload_id = upload.id
            session.close()
            return upload_id
        except Exception as e:
            session.rollback()
            session.close()
            logger.error(f"Failed to save upload: {e}")
            return 0
    
    def run_cleaning(self, state: AppState) -> AppState:
        """Run the data cleaning agent."""
        logger.info(f"Executing data cleaning (attempt {state.cleaning_attempt + 1})")
        state.current_stage = WorkflowStage.CLEANING
        state.cleaning_attempt += 1
        
        cleaning_agent = DataCleaningAgent()
        state = cleaning_agent.execute(state)
        
        # Save to DB
        if state.cleaning_result and state.upload_id:
            session = self._get_session()
            try:
                record = CleaningHistory(
                    upload_id=state.upload_id,
                    attempt_number=state.cleaning_attempt,
                    outliers_removed=state.cleaning_result.outliers_removed,
                    missing_handled=state.cleaning_result.missing_handled,
                    data_quality_score=state.cleaning_result.data_quality_score,
                    cleaning_summary=state.cleaning_result.cleaning_summary,
                    cleaned_data_path=state.cleaned_data_path,
                )
                session.add(record)
                session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to save cleaning history: {e}")
            finally:
                session.close()
        
        state.current_stage = WorkflowStage.CLEANING_APPROVAL
        return state
    
    def run_verification(self, state: AppState) -> AppState:
        """Run the verification agent."""
        logger.info(f"Executing verification (attempt {state.verification_attempts + 1})")
        state.current_stage = WorkflowStage.VERIFICATION
        state.verification_attempts += 1
        
        verification_agent = VerificationAgent()
        state = verification_agent.execute(state)
        
        # Save to DB
        if state.verification_result and state.upload_id:
            session = self._get_session()
            try:
                record = VerificationFeedback(
                    upload_id=state.upload_id,
                    cleaning_attempt_id=state.cleaning_attempt,
                    is_approved=state.verification_result.is_approved,
                    percentage_data_removed=state.verification_result.percentage_data_removed,
                    integrity_issues=state.verification_result.integrity_issues,
                    feedback_summary=state.verification_result.feedback_summary,
                    recommendations=state.verification_result.recommendations,
                )
                session.add(record)
                session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to save verification: {e}")
            finally:
                session.close()
        
        state.current_stage = WorkflowStage.VERIFICATION_APPROVAL
        return state
    
    def run_insights(self, state: AppState) -> AppState:
        """Run the insights analysis agent."""
        logger.info("Executing insights analysis")
        state.current_stage = WorkflowStage.INSIGHTS
        
        insights_agent = InsightsAgent()
        state = insights_agent.execute(state)
        
        # Save to DB
        if state.insights_result and state.upload_id:
            session = self._get_session()
            try:
                record = FinalInsights(
                    upload_id=state.upload_id,
                    patterns=state.insights_result.patterns,
                    anomalies=state.insights_result.anomalies,
                    statistical_summary=state.insights_result.statistical_summary,
                    insights_text=state.insights_result.insights_text,
                )
                session.add(record)
                session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to save insights: {e}")
            finally:
                session.close()
        
        return state
    
    def run_visualization(self, state: AppState) -> AppState:
        """Run the visualization generation agent."""
        logger.info("Generating visualizations")
        state.current_stage = WorkflowStage.VISUALIZATION
        
        viz_agent = VisualizationAgent()
        state = viz_agent.execute(state)
        
        state.current_stage = WorkflowStage.COMPLETED
        return state

    # ------------------------------------------------------------------
    # NEW: Parallel analysis pipeline
    # ------------------------------------------------------------------

    def run_parallel_analysis(self, state: AppState, progress_callback=None) -> AppState:
        """Fan out TrendAgent, AnomalyAgent, CorrelationAgent, InsightsAgent,
        and VisualizationAgent in parallel, then merge results in
        ReportSynthesisAgent.

        Args:
            state: Post-verification AppState.
            progress_callback: Optional callable(agent_name, status) for UI updates.

        Returns:
            Fully populated AppState with all analysis results.
        """
        state.current_stage = WorkflowStage.ANALYSIS
        t0 = time.time()

        # Check cache first
        if state.file_hash:
            cached = lookup_cache(state.file_hash)
            if cached:
                logger.info("Cache hit — returning cached results")
                if progress_callback:
                    progress_callback("Cache", "hit")
                state.trend_results = cached.trend_results or {}
                state.anomaly_results = cached.anomaly_results or {}
                state.correlation_results = cached.correlation_results or {}
                if cached.insights_text:
                    state.insights_result = InsightsResult(
                        insights_text=cached.insights_text,
                    )
                if cached.report_synthesis:
                    state.report_synthesis = ReportSynthesis(**cached.report_synthesis)
                if cached.pdf_path:
                    state.pdf_path = cached.pdf_path
                    state.pdf_ready = True
                state.current_stage = WorkflowStage.COMPLETED
                return state

        # Per-agent timeout (seconds). LLM calls can hang indefinitely;
        # this prevents any single agent from blocking the pipeline forever.
        AGENT_TIMEOUT = 120  # 2 minutes per agent

        # Instantiate agents
        trend_agent = TrendAgent()
        anomaly_agent = AnomalyAgent()
        correlation_agent = CorrelationAgent()
        insights_agent = InsightsAgent()
        viz_agent = VisualizationAgent()

        # The agents that can run in parallel (independent of each other)
        parallel_agents = [
            ("TrendAgent", trend_agent),
            ("AnomalyAgent", anomaly_agent),
            ("CorrelationAgent", correlation_agent),
            ("InsightsAgent", insights_agent),
            ("VisualizationAgent", viz_agent),
        ]

        def _run_agent(name, agent):
            logger.info(f"[Orchestrator] ▸ {name} STARTING")
            if progress_callback:
                progress_callback(name, "running")
            agent_t0 = time.time()
            try:
                agent.execute(state)
                elapsed = round(time.time() - agent_t0, 2)
                logger.info(f"[Orchestrator] ✓ {name} COMPLETED in {elapsed}s")
                state.agent_timings[name] = elapsed
            except Exception as e:
                elapsed = round(time.time() - agent_t0, 2)
                logger.error(f"[Orchestrator] ✗ {name} FAILED after {elapsed}s: {e}")
                state.agent_timings[name] = elapsed
            if progress_callback:
                progress_callback(name, "done")
            return name

        logger.info(f"[Orchestrator] Launching {len(parallel_agents)} agents in parallel")
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(_run_agent, n, a): n for n, a in parallel_agents}
            completed_agents = []
            for future in as_completed(futures):
                agent_name = futures[future]
                try:
                    future.result(timeout=AGENT_TIMEOUT)
                    completed_agents.append(agent_name)
                except TimeoutError:
                    logger.error(
                        f"[Orchestrator] ⏰ {agent_name} TIMED OUT after {AGENT_TIMEOUT}s — skipping"
                    )
                    state.agent_timings[agent_name] = f"TIMEOUT({AGENT_TIMEOUT}s)"
                    if progress_callback:
                        progress_callback(agent_name, "timeout")
                except Exception as e:
                    logger.error(f"[Orchestrator] ✗ {agent_name} unhandled error: {e}")
                    if progress_callback:
                        progress_callback(agent_name, "error")

        logger.info(
            f"[Orchestrator] Parallel phase done — {len(completed_agents)}/{len(parallel_agents)} "
            f"agents completed in {round(time.time() - t0, 2)}s"
        )

        # --- Merge / Report Synthesis ---
        logger.info("[Orchestrator] ▸ ReportSynthesisAgent STARTING")
        if progress_callback:
            progress_callback("ReportSynthesisAgent", "running")

        synthesis_agent = ReportSynthesisAgent()
        try:
            state = synthesis_agent.execute(state)
            logger.info(
                f"[Orchestrator] ✓ ReportSynthesisAgent COMPLETED in "
                f"{state.agent_timings.get('ReportSynthesisAgent', '?')}s"
            )
        except Exception as e:
            logger.error(f"[Orchestrator] ✗ ReportSynthesisAgent FAILED: {e}")

        if progress_callback:
            progress_callback("ReportSynthesisAgent", "done")

        # Save insights to DB
        if state.insights_result and state.upload_id:
            session = self._get_session()
            try:
                record = FinalInsights(
                    upload_id=state.upload_id,
                    patterns=state.insights_result.patterns,
                    anomalies=state.insights_result.anomalies,
                    statistical_summary=state.insights_result.statistical_summary,
                    insights_text=state.insights_result.insights_text,
                )
                session.add(record)
                session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to save insights: {e}")
            finally:
                session.close()

        # Save to cache
        if state.file_hash:
            save_cache(state.file_hash, state)

        total_time = round(time.time() - t0, 2)
        state.agent_timings["_total_analysis"] = total_time
        logger.info(
            f"[Orchestrator] ══ PIPELINE COMPLETE in {total_time}s  timings={state.agent_timings}"
        )
        state.current_stage = WorkflowStage.COMPLETED
        return state
    
    def get_past_analyses(self) -> list:
        """Query past upload records from the database."""
        session = self._get_session()
        try:
            uploads = session.query(Upload).order_by(Upload.uploaded_at.desc()).limit(20).all()
            results = []
            for u in uploads:
                insights = session.query(FinalInsights).filter_by(upload_id=u.id).first()
                results.append({
                    "id": u.id,
                    "filename": u.filename,
                    "rows": u.row_count,
                    "columns": u.column_count,
                    "uploaded_at": u.uploaded_at.strftime("%Y-%m-%d %H:%M") if u.uploaded_at else "",
                    "has_insights": insights is not None,
                    "insights_text": insights.insights_text if insights else "",
                })
            return results
        except Exception as e:
            logger.error(f"Failed to query past analyses: {e}")
            return []
        finally:
            session.close()


# Module-level orchestrator
orchestrator = WorkflowOrchestrator()
