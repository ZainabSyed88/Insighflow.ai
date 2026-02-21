"""Trend analysis agent — STL decomposition, Mann-Kendall test, Prophet forecast."""

import pandas as pd
import numpy as np
import os
import time
import logging
from typing import Any, Dict, Optional

from src.agents.base_agent import Agent
from src.graph.state import (
    AppState, TrendResult, VisualizationConfig,
    VIZ_DIR, save_dataframe, load_dataframe,
)
from src.utils.llm_utils import call_llm, get_llm
from src.utils.chart_utils import (
    create_moving_average_traces, create_line_chart,
    apply_dark_theme, save_figure_html, save_figure_png, DARK_LAYOUT,
)
from src.utils.sampling_utils import stratified_sample
from src.utils.logger import audit
from src.config import settings
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

TREND_NARRATIVE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a senior business analyst. Given the following statistical trend "
        "results for a dataset column, write 2-3 concise sentences describing the "
        "trend in plain business English. Be specific about direction, significance, "
        "and potential business implications."
    )),
    ("user", "{trend_context}"),
])


class TrendAgent(Agent):
    """Agent for time-series trend analysis on every numeric column."""

    def __init__(self):
        super().__init__(name="TrendAgent")

    # ------------------------------------------------------------------
    def execute(self, state: AppState, user_context: dict | None = None) -> AppState:
        audit("TrendAgent.execute START", filename=state.filename)
        from src.utils.logger import StepTimer
        timer = StepTimer("TrendAgent", logger)
        t0 = time.time()

        if not state.cleaned_data_path:
            state.last_error = "No cleaned data for trend analysis"
            return state

        try:
            with timer.step("load & sample data"):
                df = load_dataframe(state.cleaned_data_path)
                df_sample = stratified_sample(df, max_rows=10_000)
                numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()

            if not numeric_cols:
                logger.warning("No numeric columns — skipping trend analysis")
                state.trend_results = {"skipped": True, "reason": "no numeric columns"}
                return state

            # Detect datetime column
            with timer.step("detect datetime col"):
                dt_col = self._detect_datetime_col(df_sample)
                if dt_col:
                    df_sample = df_sample.sort_values(dt_col)
                    df_sample[dt_col] = pd.to_datetime(df_sample[dt_col], errors="coerce")
                    df_sample = df_sample.dropna(subset=[dt_col])

            column_results: list[dict] = []
            figures: list[VisualizationConfig] = []
            chart_idx = len(state.visualizations) + 1

            with timer.step(f"analyze {min(len(numeric_cols), 8)} columns"):
                for col in numeric_cols[:8]:  # cap at 8 columns for latency
                    result = self._analyze_column(df_sample, col, dt_col, chart_idx)
                    column_results.append(result["summary"])
                    figures.extend(result["figures"])
                    chart_idx += len(result["figures"])

            # Aggregate narratives
            combined_context = "\n".join(
                f"Column '{r['column']}': direction={r['trend_direction']}, "
                f"p_value={r['p_value']:.4f}, significant={r['is_significant']}"
                for r in column_results
            )
            with timer.step("LLM narrative"):
                try:
                    overall_narrative = call_llm(
                        TREND_NARRATIVE_PROMPT,
                        {"trend_context": combined_context},
                        model=settings.llm_fast_model or None,
                        temperature=0.4,
                    )
                except Exception as e:
                    logger.warning(f"LLM narration failed: {e}")
                    overall_narrative = "Trend analysis complete. See individual column results."

            state.trend_results = {
                "columns": column_results,
                "overall_narrative": overall_narrative,
                "datetime_column": dt_col,
            }
            state.visualizations.extend(figures)

        except Exception as e:
            logger.error(f"TrendAgent error: {e}")
            state.last_error = f"Trend analysis error: {e}"

        state.agent_timings["TrendAgent"] = round(time.time() - t0, 2)
        timer.summary()
        audit("TrendAgent.execute END", duration=state.agent_timings["TrendAgent"])
        return state

    # ------------------------------------------------------------------
    def _detect_datetime_col(self, df: pd.DataFrame) -> Optional[str]:
        """Auto-detect the most likely datetime column."""
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        for col in df.select_dtypes(include=["object"]).columns:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() / len(df) > 0.8:
                    return col
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    def _analyze_column(
        self, df: pd.DataFrame, col: str, dt_col: Optional[str], chart_idx: int
    ) -> dict:
        """Run STL, Mann-Kendall, optional Prophet, and create charts for one column."""
        summary: Dict[str, Any] = {
            "column": col,
            "trend_direction": "no trend",
            "is_significant": False,
            "p_value": 1.0,
        }
        figures: list[VisualizationConfig] = []
        series = df[col].dropna()
        if len(series) < 6:
            return {"summary": summary, "figures": figures}

        # --- Mann-Kendall ---
        try:
            import pymannkendall as mk
            mk_result = mk.original_test(series)
            summary["trend_direction"] = mk_result.trend
            summary["is_significant"] = mk_result.h
            summary["p_value"] = float(mk_result.p)
        except Exception as e:
            logger.warning(f"Mann-Kendall failed for {col}: {e}")

        # --- STL decomposition ---
        try:
            from statsmodels.tsa.seasonal import STL
            period = min(max(2, len(series) // 4), 13)
            if len(series) >= 2 * period:
                stl = STL(series.values, period=period, robust=True)
                res = stl.fit()
                summary["decomposition_components"] = {
                    "trend_mean": float(np.nanmean(res.trend)),
                    "seasonal_strength": float(np.nanstd(res.seasonal) / (np.nanstd(res.seasonal) + np.nanstd(res.resid) + 1e-9)),
                }
        except Exception as e:
            logger.warning(f"STL failed for {col}: {e}")

        # --- Prophet forecast (only if datetime column and 30+ rows) ---
        if dt_col and len(df) >= 30:
            try:
                from prophet import Prophet
                prophet_df = df[[dt_col, col]].rename(columns={dt_col: "ds", col: "y"}).dropna()
                if len(prophet_df) >= 10:
                    m = Prophet(yearly_seasonality="auto", weekly_seasonality="auto",
                                daily_seasonality=False, verbosity=0)
                    m.fit(prophet_df)
                    future = m.make_future_dataframe(periods=30)
                    forecast = m.predict(future)
                    fc_path = save_dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], prefix=f"forecast_{col}")
                    summary["forecast_path"] = fc_path
            except Exception as e:
                logger.warning(f"Prophet failed for {col}: {e}")

        # --- Moving average chart ---
        try:
            fig = create_moving_average_traces(
                df, col, x_col=dt_col,
                title=f"Trend — {col}",
            )
            html_path = os.path.join(VIZ_DIR, f"trend_{chart_idx}.html")
            save_figure_html(fig, html_path)
            figures.append(VisualizationConfig(
                chart_type="trend_line", title=f"Trend — {col}", html_path=html_path,
            ))
        except Exception as e:
            logger.warning(f"Chart creation failed for {col}: {e}")

        return {"summary": summary, "figures": figures}
