"""Correlation analysis agent — Pearson, Spearman, heatmaps, top pairs."""

import pandas as pd
import numpy as np
import os
import time
import logging
from typing import Any, Dict, List

from src.agents.base_agent import Agent
from src.graph.state import (
    AppState, CorrelationResult, VisualizationConfig,
    VIZ_DIR, save_dataframe, load_dataframe,
)
from src.utils.chart_utils import (
    create_correlation_heatmap, create_scatter_pair,
    save_figure_html, DARK_LAYOUT,
)
from src.utils.llm_utils import call_llm
from src.utils.sampling_utils import stratified_sample
from src.utils.logger import audit
from src.config import settings
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

CORRELATION_NARRATIVE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a business intelligence analyst. Given the correlation analysis "
     "results below, write a concise narrative (3-5 sentences) explaining the "
     "most business-relevant correlations, any surprising findings, and "
     "potential causal hypotheses."),
    ("user", "{corr_context}"),
])


class CorrelationAgent(Agent):
    """Agent for Pearson + Spearman correlation analysis."""

    def __init__(self):
        super().__init__(name="CorrelationAgent")

    def execute(self, state: AppState, user_context: dict | None = None) -> AppState:
        audit("CorrelationAgent.execute START", filename=state.filename)
        from src.utils.logger import StepTimer
        timer = StepTimer("CorrelationAgent", logger)
        t0 = time.time()

        if not state.cleaned_data_path:
            state.last_error = "No cleaned data for correlation analysis"
            return state

        try:
            df = load_dataframe(state.cleaned_data_path)
            df_sample = stratified_sample(df, max_rows=10_000)
            numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()

            if len(numeric_cols) < 2:
                state.correlation_results = {"skipped": True, "reason": "fewer than 2 numeric columns"}
                return state

            # --- Correlation matrices ---
            pearson_corr = df_sample[numeric_cols].corr(method="pearson")
            spearman_corr = df_sample[numeric_cols].corr(method="spearman")

            pearson_path = save_dataframe(pearson_corr.reset_index(), prefix="pearson_corr")
            spearman_path = save_dataframe(spearman_corr.reset_index(), prefix="spearman_corr")

            # --- Top correlations ---
            top_pos, top_neg = self._top_correlations(pearson_corr, n=5)

            # --- Charts ---
            figures: list[VisualizationConfig] = []
            chart_idx = len(state.visualizations) + 1

            # Heatmap
            try:
                fig_heat = create_correlation_heatmap(pearson_corr, title="Pearson Correlation Matrix")
                html_path = os.path.join(VIZ_DIR, f"corr_heatmap_{chart_idx}.html")
                save_figure_html(fig_heat, html_path)
                figures.append(VisualizationConfig(
                    chart_type="heatmap", title="Correlation Heatmap", html_path=html_path,
                ))
                chart_idx += 1
            except Exception as e:
                logger.warning(f"Heatmap failed: {e}")

            # Scatter plots for top 3 absolute-value pairs
            all_pairs = sorted(top_pos + top_neg, key=lambda x: abs(x["correlation"]), reverse=True)
            for pair in all_pairs[:3]:
                try:
                    fig_sc = create_scatter_pair(
                        df_sample, pair["col_a"], pair["col_b"],
                        title=f"{pair['col_a']} vs {pair['col_b']} (r={pair['correlation']:.2f})",
                    )
                    html_path = os.path.join(VIZ_DIR, f"corr_scatter_{chart_idx}.html")
                    save_figure_html(fig_sc, html_path)
                    figures.append(VisualizationConfig(
                        chart_type="scatter", title=f"Correlation — {pair['col_a']} vs {pair['col_b']}",
                        html_path=html_path,
                    ))
                    chart_idx += 1
                except Exception as e:
                    logger.warning(f"Scatter pair plot failed: {e}")

            # --- LLM narrative ---
            context_str = (
                f"Numeric columns: {numeric_cols}\n"
                f"Top 5 positive correlations: {top_pos}\n"
                f"Top 5 negative correlations: {top_neg}\n"
            )
            try:
                narrative = call_llm(
                    CORRELATION_NARRATIVE_PROMPT, {"corr_context": context_str},
                    model=settings.llm_fast_model or None, temperature=0.4,
                )
            except Exception as e:
                logger.warning(f"LLM correlation narrative failed: {e}")
                narrative = "Correlation analysis complete. Review the heatmap and scatter plots above."

            state.correlation_results = {
                "pearson_matrix_path": pearson_path,
                "spearman_matrix_path": spearman_path,
                "top_positive": top_pos,
                "top_negative": top_neg,
                "narrative": narrative,
            }
            state.visualizations.extend(figures)

        except Exception as e:
            logger.error(f"CorrelationAgent error: {e}")
            state.last_error = f"Correlation analysis error: {e}"

        state.agent_timings["CorrelationAgent"] = round(time.time() - t0, 2)
        timer.summary()
        audit("CorrelationAgent.execute END", duration=state.agent_timings["CorrelationAgent"])
        return state

    # ------------------------------------------------------------------
    @staticmethod
    def _top_correlations(corr: pd.DataFrame, n: int = 5):
        """Extract top-n positive and top-n negative pairs from a corr matrix."""
        pairs: List[Dict[str, Any]] = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs.append({
                    "col_a": cols[i],
                    "col_b": cols[j],
                    "correlation": round(float(corr.iloc[i, j]), 4),
                })
        sorted_pairs = sorted(pairs, key=lambda x: x["correlation"], reverse=True)
        top_pos = [p for p in sorted_pairs if p["correlation"] > 0][:n]
        top_neg = [p for p in sorted_pairs if p["correlation"] < 0][-n:]
        return top_pos, top_neg
