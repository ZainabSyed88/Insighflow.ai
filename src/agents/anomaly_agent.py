"""Anomaly / outlier detection agent — IQR, Z-score, Isolation Forest."""

import pandas as pd
import numpy as np
import os
import time
import logging
from typing import Any, Dict

from src.agents.base_agent import Agent
from src.graph.state import (
    AppState, AnomalyResult, VisualizationConfig,
    VIZ_DIR, save_dataframe, load_dataframe,
)
from src.utils.outlier_utils import detect_outliers_all_methods, iqr_outlier_mask
from src.utils.chart_utils import (
    create_box_plot, create_scatter_with_outliers,
    save_figure_html, DARK_LAYOUT,
)
from src.utils.llm_utils import call_llm
from src.utils.sampling_utils import stratified_sample
from src.utils.logger import audit
from src.config import settings
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

ANOMALY_NARRATIVE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a data quality expert. Given the anomaly detection results below, "
     "write a concise business-readable summary (3-5 sentences) covering: how many "
     "anomalies were found, which columns are most affected, and what actions a "
     "business user should consider."),
    ("user", "{anomaly_context}"),
])


class AnomalyAgent(Agent):
    """Agent for multi-method anomaly and outlier detection."""

    def __init__(self):
        super().__init__(name="AnomalyAgent")

    def execute(self, state: AppState, user_context: dict | None = None) -> AppState:
        audit("AnomalyAgent.execute START", filename=state.filename)
        from src.utils.logger import StepTimer
        timer = StepTimer("AnomalyAgent", logger)
        t0 = time.time()

        if not state.cleaned_data_path:
            state.last_error = "No cleaned data for anomaly detection"
            return state

        try:
            df = load_dataframe(state.cleaned_data_path)
            df_sample = stratified_sample(df, max_rows=10_000)
            numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()

            if not numeric_cols:
                state.anomaly_results = {"skipped": True, "reason": "no numeric columns"}
                return state

            # --- IQR + Zscore per column ---
            outlier_results = detect_outliers_all_methods(
                df_sample, numeric_cols,
                iqr_multiplier=settings.outlier_iqr_multiplier,
            )

            column_details: Dict[str, Any] = {}
            reason_codes: list[str] = []
            combined_mask = pd.Series(False, index=df_sample.index)

            for col, masks in outlier_results.items():
                iqr_count = int(masks["iqr"].sum())
                zs_count = int(masks["zscore"].sum())
                col_mask = masks["iqr"] | masks["zscore"]
                combined_mask |= col_mask
                column_details[col] = {
                    "iqr_outliers": iqr_count,
                    "zscore_outliers": zs_count,
                    "total": int(col_mask.sum()),
                }
                if iqr_count:
                    reason_codes.append(f"{col}: {iqr_count} IQR outliers")
                if zs_count:
                    reason_codes.append(f"{col}: {zs_count} Z-score outliers")

            # --- Isolation Forest (multivariate) ---
            iso_labels = None
            if len(numeric_cols) >= 2:
                try:
                    from sklearn.ensemble import IsolationForest
                    X = df_sample[numeric_cols].dropna()
                    if len(X) >= 20:
                        iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
                        preds = iso.fit_predict(X)
                        iso_mask = pd.Series(False, index=df_sample.index)
                        iso_mask.loc[X.index] = preds == -1
                        combined_mask |= iso_mask
                        iso_count = int(iso_mask.sum())
                        column_details["_isolation_forest"] = {"anomalies": iso_count}
                        reason_codes.append(f"Isolation Forest: {iso_count} multivariate anomalies")
                except Exception as e:
                    logger.warning(f"Isolation Forest failed: {e}")

            total_anomalies = int(combined_mask.sum())
            anomaly_rate = total_anomalies / len(df_sample) if len(df_sample) else 0.0

            # Save flagged rows
            flagged_df = df_sample.copy()
            flagged_df["_is_anomaly"] = combined_mask
            flagged_path = save_dataframe(flagged_df, prefix="anomaly_flagged")

            # --- Charts ---
            figures: list[VisualizationConfig] = []
            chart_idx = len(state.visualizations) + 1

            # Box plots
            try:
                box_cols = numeric_cols[:6]
                fig_box = create_box_plot(df_sample, box_cols, title="Outlier Distribution — Box Plots")
                html_path = os.path.join(VIZ_DIR, f"anomaly_box_{chart_idx}.html")
                save_figure_html(fig_box, html_path)
                figures.append(VisualizationConfig(
                    chart_type="box", title="Outlier Box Plots", html_path=html_path,
                ))
                chart_idx += 1
            except Exception as e:
                logger.warning(f"Box plot failed: {e}")

            # Scatter with outliers (first two numeric cols)
            if len(numeric_cols) >= 2:
                try:
                    fig_scatter = create_scatter_with_outliers(
                        df_sample, numeric_cols[0], numeric_cols[1],
                        combined_mask,
                        title=f"Anomalies — {numeric_cols[0]} vs {numeric_cols[1]}",
                    )
                    html_path = os.path.join(VIZ_DIR, f"anomaly_scatter_{chart_idx}.html")
                    save_figure_html(fig_scatter, html_path)
                    figures.append(VisualizationConfig(
                        chart_type="scatter", title="Anomaly Scatter", html_path=html_path,
                    ))
                    chart_idx += 1
                except Exception as e:
                    logger.warning(f"Scatter plot failed: {e}")

            # --- LLM narrative ---
            context_str = (
                f"Total rows: {len(df_sample)}, anomalies: {total_anomalies} "
                f"({anomaly_rate:.1%})\n"
                f"Per-column details: {column_details}\n"
                f"Reason codes: {reason_codes}"
            )
            try:
                narrative = call_llm(
                    ANOMALY_NARRATIVE_PROMPT, {"anomaly_context": context_str},
                    model=settings.llm_fast_model or None, temperature=0.4,
                )
            except Exception as e:
                logger.warning(f"LLM anomaly narrative failed: {e}")
                narrative = f"Detected {total_anomalies} anomalies ({anomaly_rate:.1%}) across {len(numeric_cols)} numeric columns."

            state.anomaly_results = {
                "total_anomalies": total_anomalies,
                "anomaly_rate": round(anomaly_rate, 4),
                "column_details": column_details,
                "reason_codes": reason_codes,
                "flagged_rows_path": flagged_path,
                "narrative": narrative,
            }
            state.visualizations.extend(figures)

        except Exception as e:
            logger.error(f"AnomalyAgent error: {e}")
            state.last_error = f"Anomaly detection error: {e}"

        state.agent_timings["AnomalyAgent"] = round(time.time() - t0, 2)
        timer.summary()
        audit("AnomalyAgent.execute END", duration=state.agent_timings["AnomalyAgent"])
        return state
