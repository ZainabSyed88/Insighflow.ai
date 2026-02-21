"""Data cleaning agent with per-step justification tracking and guardrails."""

import pandas as pd
import numpy as np
from src.agents.base_agent import Agent
from src.graph.state import (
    AppState, CleaningResult, CleaningAction,
    save_dataframe, load_dataframe,
)
from src.config import settings
import logging
from typing import Dict, Any, List
from langgraph.prebuilt import create_react_agent
from langchain_experimental.tools.python.tool import PythonAstREPLTool

logger = logging.getLogger(__name__)

# Maximum percentage of rows the LLM is allowed to remove in one go.
MAX_ROW_REMOVAL_PCT = 10.0


def _col_snapshot(series: pd.Series) -> dict:
    """Return a compact statistical snapshot for a single column."""
    if pd.api.types.is_numeric_dtype(series):
        return {
            "count": int(series.count()),
            "nulls": int(series.isnull().sum()),
            "mean": round(float(series.mean()), 4) if series.count() > 0 else None,
            "std": round(float(series.std()), 4) if series.count() > 1 else None,
            "min": round(float(series.min()), 4) if series.count() > 0 else None,
            "max": round(float(series.max()), 4) if series.count() > 0 else None,
        }
    return {
        "count": int(series.count()),
        "nulls": int(series.isnull().sum()),
        "unique": int(series.nunique()),
    }


def _df_snapshot(df: pd.DataFrame) -> dict:
    """Whole-frame summary (row count + per-column snapshots)."""
    return {
        "rows": len(df),
        "columns": {col: _col_snapshot(df[col]) for col in df.columns},
    }


class DataCleaningAgent(Agent):
    """Agent responsible for data cleaning with auditable, justified steps."""

    def __init__(self):
        """Initialize the cleaning agent."""
        super().__init__(name="DataCleaningAgent")

        # Python REPL tool
        self.python_repl = PythonAstREPLTool()

        # Missing-value threshold for drop-vs-impute decision
        self._missing_drop_threshold = 0.05  # 5%

        # System prompt  — instructs the LLM to clean *step-by-step*
        system_prompt = f"""You are an expert Data Cleaning AI Agent.
Your job is to clean the pandas DataFrame named `df` loaded in the environment.
You have access to a Python REPL tool. You MUST use it to execute code.

IMPORTANT RULES — follow these strictly:

1. Perform cleaning ONE STEP AT A TIME. After each step, call the
   `record_action(action, target, detail, justification)` function that is
   pre-loaded in the environment. Example:
     record_action("drop_rows", "Price", "Dropped 12 rows with negative Price", "Negative prices are invalid for a sales dataset")
   Valid action types: "drop_rows", "drop_column", "impute_knn", "impute_iterative",
     "impute_mean", "impute_median", "impute_mode", "impute_zero", "impute_ffill",
     "type_cast", "deduplicate", "clip_outliers", "other".

2. MISSING VALUE STRATEGY (mandatory — follow exactly):
   For EACH column with missing values, calculate the missing percentage.
   a) If missing% < 5%  → DROP those rows. This is acceptable data loss.
      Justification must mention the exact missing% and that it is below 5%.
   b) If missing% >= 5% → Use MACHINE LEARNING imputation:
      • For NUMERIC columns: use `sklearn.impute.KNNImputer(n_neighbors=5)`
        or `sklearn.experimental.enable_iterative_imputer` + `sklearn.impute.IterativeImputer()`.
        KNNImputer is preferred; fall back to IterativeImputer if KNN is too slow.
      • For CATEGORICAL columns: use `sklearn.impute.KNNImputer` after
        `sklearn.preprocessing.OrdinalEncoder`, then decode back.
        If that fails, fall back to mode imputation.
      • After ML imputation, CLIP imputed values to the column's original
        [min, max] range so the imputer does not extrapolate.
      Justification must mention the exact missing%, the ML method used,
      and that clipping was applied.
   c) If missing% > 50% → DROP the entire column.
      Justification must explain the column is too sparse to be reliable.
   sklearn is already importable — you can `from sklearn.impute import KNNImputer`.

3. NEVER remove more than {MAX_ROW_REMOVAL_PCT}% of total rows in a single step.

4. For outliers, use IQR with multiplier {settings.outlier_iqr_multiplier}.
   CLIP outliers to the fence values instead of dropping rows,
   unless the values are clearly data-entry errors (e.g. negative prices, ages > 150).

5. After each operation, update `df` in the REPL environment.
6. Do NOT calculate quality scores — the system does that automatically.
7. Always provide a RECOMMENDATION in the justification (e.g. "Recommend monitoring
   this column in future data pipelines for systematic missingness").
8. When finished, say: "I have finished cleaning the dataframe."
"""

        self.tools = [self.python_repl]
        self.agent_executor = create_react_agent(
            self.llm,
            tools=self.tools,
            prompt=system_prompt,
        )

    # ------------------------------------------------------------------
    def execute(self, state: AppState, user_context: dict | None = None) -> AppState:
        from src.utils.logger import audit
        audit("DataCleaningAgent.execute START", filename=state.filename)
        self._log_action("Starting data cleaning with justification tracking")

        if not state.raw_data_path:
            state.last_error = "No data provided for cleaning"
            return state

        try:
            # Load data
            df = load_dataframe(state.raw_data_path)
            original_row_count = len(df)
            original_col_count = len(df.columns)

            # Capture BEFORE snapshot
            before_snap = _df_snapshot(df)

            # Mutable list the REPL callback will append to
            action_log: List[dict] = []

            def record_action(action: str, target: str, detail: str, justification: str):
                """Callback injected into the REPL so the LLM can log each step."""
                current_df = self.python_repl.locals.get("df")
                rows_now = len(current_df) if current_df is not None else -1
                prev_rows = action_log[-1]["rows_after"] if action_log else original_row_count
                action_log.append({
                    "action": action,
                    "target": target,
                    "detail": detail,
                    "justification": justification,
                    "rows_before": prev_rows,
                    "rows_after": rows_now,
                    "rows_affected": abs(prev_rows - rows_now),
                })
                logger.info(
                    f"[CleaningAction] {action} on '{target}': {detail} "
                    f"(rows {prev_rows} → {rows_now}) | justification: {justification}"
                )

            # Inject into REPL namespace
            self.python_repl.locals = {
                "df": df,
                "pd": pd,
                "np": np,
                "record_action": record_action,
            }

            # Pre-import sklearn imputers into REPL so LLM can use them directly
            try:
                from sklearn.impute import KNNImputer
                self.python_repl.locals["KNNImputer"] = KNNImputer
            except ImportError:
                pass
            try:
                from sklearn.experimental import enable_iterative_imputer  # noqa: F401
                from sklearn.impute import IterativeImputer
                self.python_repl.locals["IterativeImputer"] = IterativeImputer
            except ImportError:
                pass
            try:
                from sklearn.preprocessing import OrdinalEncoder
                self.python_repl.locals["OrdinalEncoder"] = OrdinalEncoder
            except ImportError:
                pass

            # Build minimal dataset context
            # Include per-column missing% so LLM can apply the 5% rule
            missing_info = {}
            for c in df.columns:
                n_null = int(df[c].isnull().sum())
                pct = round(n_null / len(df) * 100, 2) if len(df) > 0 else 0
                if n_null > 0:
                    missing_info[c] = f"{n_null} ({pct}%)"

            df_info = (
                f"DataFrame shape: {df.shape}\n"
                f"Columns: {df.columns.tolist()}\n"
                f"Dtypes:\n{df.dtypes.to_string()}\n"
                f"Missing values per column (count + %):\n{missing_info}\n"
                f"Describe:\n{df.describe().to_json()}\n"
            )

            # Execute the LLM agent
            result = self.agent_executor.invoke({
                "messages": [("user",
                    f"Please clean the dataset step by step. "
                    f"Remember to call record_action() after every cleaning operation.\n\n"
                    f"Dataset summary:\n{df_info}"
                )]
            })

            output_msg = result["messages"][-1].content if "messages" in result else "Completed"
            logger.info(f"Agent reasoning: {output_msg}")

            # Retrieve the modified dataframe
            cleaned_df = self.python_repl.locals.get("df", df)

            # ---------- POST-CLEANING GUARDRAILS ----------
            rows_removed = original_row_count - len(cleaned_df)
            removal_pct = (rows_removed / original_row_count * 100) if original_row_count else 0

            # Guard: if LLM removed too many rows, revert to capped version
            if removal_pct > MAX_ROW_REMOVAL_PCT * 2:
                logger.warning(
                    f"LLM removed {removal_pct:.1f}% rows — exceeds safety limit. "
                    f"Reverting to conservative cleaning."
                )
                cleaned_df, action_log = self._conservative_clean(df)
                rows_removed = original_row_count - len(cleaned_df)

            # Capture AFTER snapshot
            after_snap = _df_snapshot(cleaned_df)

            # Build CleaningAction objects
            cleaning_actions: List[CleaningAction] = []
            for entry in action_log:
                # Per-column snapshots where applicable
                target_col = entry["target"]
                before_col = before_snap["columns"].get(target_col, {})
                after_col = after_snap["columns"].get(target_col, {})
                cleaning_actions.append(CleaningAction(
                    action=entry["action"],
                    target=target_col,
                    rows_affected=entry["rows_affected"],
                    detail=entry["detail"],
                    justification=entry["justification"],
                    before_snapshot=before_col,
                    after_snapshot=after_col,
                ))

            # If the LLM produced zero actions (forgot to call record_action),
            # synthesise them from the diff.
            if not cleaning_actions and rows_removed > 0:
                cleaning_actions.append(CleaningAction(
                    action="drop_rows",
                    target="all",
                    rows_affected=rows_removed,
                    detail=f"Agent removed {rows_removed} rows (no per-step log captured).",
                    justification="Automatic — LLM did not log individual steps.",
                ))

            missing_handled = max(
                0, df.isnull().sum().sum() - cleaned_df.isnull().sum().sum()
            )

            # Save cleaned data
            cleaned_path = save_dataframe(cleaned_df, prefix="cleaned")

            data_quality_score = self._calculate_quality_score(
                original_row_count, len(cleaned_df), missing_handled
            )

            summary_lines = [
                f"Cleaned data: {original_row_count} → {len(cleaned_df)} rows "
                f"({rows_removed} removed, {removal_pct:.1f}%).",
            ]
            for ca in cleaning_actions:
                summary_lines.append(
                    f"  • [{ca.action}] {ca.target}: {ca.detail} — {ca.justification}"
                )

            state.cleaning_result = CleaningResult(
                raw_data_path=state.raw_data_path,
                cleaned_data_path=cleaned_path,
                removed_rows=rows_removed,
                missing_handled=missing_handled,
                outliers_removed=rows_removed,
                data_quality_score=data_quality_score,
                cleaning_summary="\n".join(summary_lines),
                parameters_used={
                    "iqr_multiplier": settings.outlier_iqr_multiplier,
                    "missing_threshold": settings.missing_value_threshold,
                    "max_removal_pct": MAX_ROW_REMOVAL_PCT,
                },
                cleaning_actions=cleaning_actions,
            )

            state.cleaned_data_path = cleaned_path
            self._log_action("Data cleaning completed", {
                "rows_removed": rows_removed,
                "quality_score": data_quality_score,
                "actions_logged": len(cleaning_actions),
            })

        except Exception as e:
            logger.error(f"Error during data cleaning: {str(e)}")
            state.last_error = f"Cleaning error: {str(e)}"

        from src.utils.logger import audit
        audit("DataCleaningAgent.execute END", filename=state.filename)
        return state

    # ------------------------------------------------------------------
    def _conservative_clean(self, df: pd.DataFrame):
        """Fallback cleaning when the LLM is too aggressive.

        Strategy:
          1. Drop duplicate rows
          2. Drop columns with >50% nulls
          3. Missing < 5%  → drop those rows
             Missing >= 5% → ML imputation (KNN / Iterative) + clip to original range
          4. Categorical nulls  → same 5% rule with encoded KNN or mode fallback
          5. Clip numeric outliers to IQR fences (no row drops)
        """
        actions: List[dict] = []
        original_rows = len(df)
        missing_drop_thresh = getattr(self, "_missing_drop_threshold", 0.05)

        # 1. Duplicates
        n_dup = df.duplicated().sum()
        if n_dup:
            df = df.drop_duplicates()
            actions.append({
                "action": "deduplicate", "target": "all",
                "detail": f"Removed {n_dup} duplicate rows",
                "justification": "Exact duplicate rows carry no additional information. "
                                 "Recommend checking upstream ETL for dedup logic.",
                "rows_before": original_rows, "rows_after": len(df),
                "rows_affected": n_dup,
            })

        # 2. High-null columns (>50%)
        threshold = settings.missing_value_threshold
        null_pct = df.isnull().mean()
        drop_cols = null_pct[null_pct > threshold].index.tolist()
        if drop_cols:
            df = df.drop(columns=drop_cols)
            actions.append({
                "action": "drop_column", "target": ", ".join(drop_cols),
                "detail": f"Dropped {len(drop_cols)} columns with >{threshold*100:.0f}% nulls",
                "justification": "Columns with majority missing values are unreliable for analysis. "
                                 "Recommend investigating data source for these columns.",
                "rows_before": len(df), "rows_after": len(df), "rows_affected": 0,
            })

        # 3. Numeric missing values — 5% threshold
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Separate into < 5% (drop rows) and >= 5% (ML impute)
        cols_to_drop_rows: List[str] = []
        cols_to_ml_impute: List[str] = []

        for col in num_cols:
            if col not in df.columns:
                continue
            n_null = int(df[col].isnull().sum())
            if n_null == 0:
                continue
            pct = n_null / len(df)
            if pct < missing_drop_thresh:
                cols_to_drop_rows.append(col)
            else:
                cols_to_ml_impute.append(col)

        # 3a. Drop rows for low-missing columns
        for col in cols_to_drop_rows:
            n_null = int(df[col].isnull().sum())
            pct = n_null / len(df) * 100
            rows_before = len(df)
            df = df.dropna(subset=[col])
            actions.append({
                "action": "drop_rows", "target": col,
                "detail": f"Dropped {n_null} rows with missing '{col}' ({pct:.1f}% missing)",
                "justification": f"Missing% ({pct:.1f}%) is below the 5% threshold — "
                                 f"safe to drop; minimal data loss. "
                                 f"Recommend monitoring this column for increasing missingness.",
                "rows_before": rows_before, "rows_after": len(df),
                "rows_affected": rows_before - len(df),
            })

        # 3b. ML imputation for high-missing numeric columns
        if cols_to_ml_impute:
            ml_cols_in_df = [c for c in cols_to_ml_impute if c in df.columns]
            if ml_cols_in_df:
                df, ml_actions = self._ml_impute_numeric(df, ml_cols_in_df)
                actions.extend(ml_actions)

        # 4. Categorical missing values — same 5% rule
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            if col not in df.columns:
                continue
            n_null = int(df[col].isnull().sum())
            if n_null == 0:
                continue
            pct = n_null / len(df) * 100
            if pct / 100 < missing_drop_thresh:
                rows_before = len(df)
                df = df.dropna(subset=[col])
                actions.append({
                    "action": "drop_rows", "target": col,
                    "detail": f"Dropped {n_null} rows with missing '{col}' ({pct:.1f}% missing)",
                    "justification": f"Missing% ({pct:.1f}%) below 5% threshold — safe to drop. "
                                     f"Recommend reviewing data entry process for this field.",
                    "rows_before": rows_before, "rows_after": len(df),
                    "rows_affected": rows_before - len(df),
                })
            else:
                # Try KNN on encoded categories, fall back to mode
                df, cat_action = self._ml_impute_categorical(df, col, n_null, pct)
                actions.append(cat_action)

        # 5. Clip outliers
        iqr_mult = settings.outlier_iqr_multiplier
        # Re-detect numeric cols (some may have been dropped)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols:
            if col not in df.columns:
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
            n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
            if n_outliers:
                df[col] = df[col].clip(lower, upper)
                actions.append({
                    "action": "clip_outliers", "target": col,
                    "detail": f"Clipped {n_outliers} outliers to [{lower:.4g}, {upper:.4g}]",
                    "justification": f"IQR×{iqr_mult} fencing preserves all rows while bounding "
                                     f"extreme values. Recommend investigating root cause of outliers.",
                    "rows_before": len(df), "rows_after": len(df), "rows_affected": n_outliers,
                })

        return df, actions

    # ------------------------------------------------------------------
    def _ml_impute_numeric(
        self, df: pd.DataFrame, cols: List[str]
    ) -> tuple:
        """Impute numeric columns with >=5% missing values using ML (KNN → Iterative fallback).

        After imputation, values are clipped to the original [min, max] of each column.
        """
        actions: List[dict] = []

        # Capture original ranges for clipping
        original_min = {c: df[c].min() for c in cols}
        original_max = {c: df[c].max() for c in cols}
        null_counts = {c: int(df[c].isnull().sum()) for c in cols}
        null_pcts = {c: null_counts[c] / len(df) * 100 for c in cols}

        method_used = "KNNImputer"
        try:
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=5, weights="uniform")
            df[cols] = imputer.fit_transform(df[cols])
        except Exception as e:
            logger.warning(f"KNNImputer failed: {e} — falling back to IterativeImputer")
            method_used = "IterativeImputer"
            try:
                from sklearn.experimental import enable_iterative_imputer  # noqa: F401
                from sklearn.impute import IterativeImputer
                imputer = IterativeImputer(max_iter=10, random_state=42)
                df[cols] = imputer.fit_transform(df[cols])
            except Exception as e2:
                logger.warning(f"IterativeImputer also failed: {e2} — using median")
                method_used = "median (ML fallback failed)"
                for c in cols:
                    df[c] = df[c].fillna(df[c].median())

        # Clip imputed values to original range
        for c in cols:
            if c in df.columns:
                df[c] = df[c].clip(original_min[c], original_max[c])

        for c in cols:
            actions.append({
                "action": "impute_knn" if "KNN" in method_used else "impute_iterative",
                "target": c,
                "detail": (
                    f"Imputed {null_counts[c]} nulls ({null_pcts[c]:.1f}% missing) "
                    f"using {method_used}, then clipped to original range "
                    f"[{original_min[c]:.4g}, {original_max[c]:.4g}]"
                ),
                "justification": (
                    f"Missing% ({null_pcts[c]:.1f}%) is ≥5% — ML imputation ({method_used}) "
                    f"leverages inter-column relationships for more accurate fill values "
                    f"than simple median/mean. Values clipped to column's original range "
                    f"to prevent extrapolation. Recommend investigating why this column "
                    f"has significant missingness."
                ),
                "rows_before": len(df), "rows_after": len(df),
                "rows_affected": null_counts[c],
            })

        return df, actions

    # ------------------------------------------------------------------
    def _ml_impute_categorical(
        self, df: pd.DataFrame, col: str, n_null: int, pct: float
    ) -> tuple:
        """Impute a single categorical column with >=5% missing using KNN on encoded values,
        falling back to mode."""
        method = "KNNImputer (encoded)"
        try:
            from sklearn.impute import KNNImputer
            from sklearn.preprocessing import OrdinalEncoder

            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            non_null_mask = df[col].notna()
            encoded = encoder.fit_transform(df[[col]].fillna("__MISSING__"))
            encoded[~non_null_mask.values] = np.nan
            imputer = KNNImputer(n_neighbors=5)
            imputed = imputer.fit_transform(encoded)
            imputed_rounded = np.clip(np.round(imputed), 0, len(encoder.categories_[0]) - 1).astype(int)
            df[col] = encoder.categories_[0][imputed_rounded.flatten()]
        except Exception as e:
            logger.warning(f"Categorical KNN imputation failed for '{col}': {e} — using mode")
            method = "mode (KNN fallback failed)"
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(mode_val)

        action = {
            "action": "impute_knn" if "KNN" in method else "impute_mode",
            "target": col,
            "detail": (
                f"Imputed {n_null} nulls ({pct:.1f}% missing) in categorical column '{col}' "
                f"using {method}"
            ),
            "justification": (
                f"Missing% ({pct:.1f}%) is ≥5% — ML imputation preserves category relationships "
                f"better than simple mode fill. Recommend reviewing data collection "
                f"for this field to reduce future missingness."
            ),
            "rows_before": len(df), "rows_after": len(df),
            "rows_affected": n_null,
        }
        return df, action

    # ------------------------------------------------------------------
    def _calculate_quality_score(self, original_rows: int, cleaned_rows: int,
                                missing_handled: int) -> float:
        """Calculate data quality score (0-1)."""
        if original_rows == 0:
            return 0.0

        retention_ratio = cleaned_rows / original_rows
        quality_score = retention_ratio * 0.8 + (1 - min(missing_handled / original_rows, 1)) * 0.2

        return round(quality_score, 3)
