"""Verification agent — validates each cleaning action with justification review."""

import pandas as pd
import numpy as np
from src.agents.base_agent import Agent
from src.graph.state import AppState, VerificationResult, load_dataframe
from src.config import settings
import logging
from typing import List, Dict, Any
import json
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


class VerificationAgent(Agent):
    """Agent responsible for verifying data quality and reviewing each cleaning action."""

    def __init__(self):
        """Initialize the verification agent."""
        super().__init__(name="VerificationAgent")

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Data Quality Assurance AI Agent.
You are given:
  • A list of individual cleaning ACTIONS, each with its justification and before/after stats.
  • Aggregated before/after summaries.
  • Automated warnings from sanity checks.

For EACH cleaning action, decide whether it was appropriate. Return your
analysis strictly as a JSON object matching this schema:
{{
  "is_approved": true/false,
  "action_verdicts": [
    {{
      "action": "<action type>",
      "target": "<column>",
      "verdict": "approved" | "questionable" | "rejected",
      "reason": "Why you approved or flagged this action"
    }}
  ],
  "integrity_issues": ["issue 1", ...],
  "recommendations": ["recommendation 1", ...],
  "severity": "low" | "medium" | "high",
  "feedback_summary": "1-2 sentence overall conclusion"
}}

Rules for verdicts:
- "approved": The action is clearly beneficial (e.g. removing true duplicates,
  imputing a small number of nulls, clipping obvious outliers).
- "questionable": The action is defensible but aggressive (e.g. dropping >5%
  rows, removing outliers without clipping first).
- "rejected": The action harms data integrity (e.g. dropping a useful column,
  removing too many rows without justification).

Set is_approved=false if ANY action is "rejected" or if total data removed >15%.
Ensure valid JSON and nothing else.
"""),
            ("user", "{verification_context}")
        ])

    def execute(self, state: AppState, user_context: dict | None = None) -> AppState:
        from src.utils.logger import audit
        audit("VerificationAgent.execute START", filename=state.filename)
        self._log_action("Starting data verification with per-action review")

        if not state.cleaned_data_path or state.cleaning_result is None:
            state.last_error = "No data to verify"
            return state

        try:
            df = load_dataframe(state.cleaned_data_path)
            cleaning = state.cleaning_result

            # Automated sanity checks
            integrity_ok, integrity_issues_auto = self._check_data_integrity(df)
            stats_ok, stats_warnings_auto = self._check_statistical_consistency(df)

            original_len = len(df) + cleaning.removed_rows
            percentage_removed = (
                (cleaning.removed_rows / original_len * 100) if original_len > 0 else 0
            )

            # --- Per-column before / after comparison ---
            column_checks: Dict[str, Any] = {}
            if cleaning.raw_data_path:
                try:
                    raw_df = load_dataframe(cleaning.raw_data_path)
                    for col in raw_df.columns:
                        before_nulls = int(raw_df[col].isnull().sum())
                        after_nulls = int(df[col].isnull().sum()) if col in df.columns else None
                        entry: Dict[str, Any] = {
                            "before_nulls": before_nulls,
                            "after_nulls": after_nulls,
                            "column_dropped": col not in df.columns,
                        }
                        if pd.api.types.is_numeric_dtype(raw_df[col]) and col in df.columns:
                            entry["before_mean"] = round(float(raw_df[col].mean()), 4)
                            entry["after_mean"] = round(float(df[col].mean()), 4)
                            entry["before_std"] = round(float(raw_df[col].std()), 4)
                            entry["after_std"] = round(float(df[col].std()), 4)
                            # Flag large distribution shifts
                            if entry["before_std"] > 0:
                                mean_shift_pct = abs(entry["after_mean"] - entry["before_mean"]) / (entry["before_std"] + 1e-9) * 100
                                if mean_shift_pct > 30:
                                    stats_warnings_auto.append(
                                        f"Column '{col}' mean shifted by {mean_shift_pct:.1f}% of original std"
                                    )
                        column_checks[col] = entry
                except Exception as e:
                    logger.warning(f"Column comparison failed: {e}")

            # --- Build per-action summaries for LLM ---
            actions_desc = []
            for ca in cleaning.cleaning_actions:
                actions_desc.append({
                    "action": ca.action,
                    "target": ca.target,
                    "rows_affected": ca.rows_affected,
                    "detail": ca.detail,
                    "justification": ca.justification,
                    "before": ca.before_snapshot,
                    "after": ca.after_snapshot,
                })

            context = {
                "cleaning_actions": actions_desc,
                "cleaning_summary": cleaning.cleaning_summary,
                "percentage_data_removed": round(percentage_removed, 2),
                "automated_integrity_warnings": integrity_issues_auto,
                "automated_statistical_warnings": stats_warnings_auto,
                "current_row_count": len(df),
                "original_row_count": original_len,
                "columns_before": list(column_checks.keys()),
                "columns_after": df.columns.tolist(),
            }

            context_str = json.dumps(context, indent=2, default=str)

            # LLM call
            chain = self.prompt | self.llm
            response = chain.invoke({"verification_context": context_str})

            content = response.content
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]

            llm_result = json.loads(content.strip())

            issues = llm_result.get("integrity_issues", [])
            all_issues = list(set(issues + integrity_issues_auto + stats_warnings_auto))

            action_verdicts = llm_result.get("action_verdicts", [])

            is_approved = llm_result.get("is_approved", False)

            # Hard overrides
            if percentage_removed > 30:
                is_approved = False
                all_issues.append(f"Auto-Fail: High data removal ({percentage_removed:.1f}%)")
            if any(v.get("verdict") == "rejected" for v in action_verdicts):
                is_approved = False
                all_issues.append("Auto-Fail: One or more cleaning actions were rejected")

            state.verification_result = VerificationResult(
                is_approved=is_approved,
                percentage_data_removed=percentage_removed,
                integrity_issues=all_issues,
                feedback_summary=llm_result.get("feedback_summary", "Verification complete."),
                recommendations=llm_result.get("recommendations", []),
                severity=llm_result.get("severity", "medium"),
                column_checks=column_checks,
                action_verdicts=action_verdicts,
            )

            self._log_action("Verification completed", {
                "approved": is_approved,
                "issues": len(all_issues),
                "removed_percentage": f"{percentage_removed:.1f}%",
                "actions_reviewed": len(action_verdicts),
            })

        except Exception as e:
            logger.error(f"Error during verification: {str(e)}")
            state.last_error = f"Verification error: {str(e)}"

        from src.utils.logger import audit
        audit("VerificationAgent.execute END", filename=state.filename)
        return state

    def _check_data_integrity(self, df: pd.DataFrame) -> tuple[bool, List[str]]:
        """Check for data type consistency and format issues."""
        issues = []

        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate rows")

        nulls = df.isnull().sum().sum()
        if nulls > 0:
            issues.append(f"Found {nulls} remaining null values")

        return len(issues) == 0, issues

    def _check_statistical_consistency(self, df: pd.DataFrame) -> tuple[bool, List[str]]:
        """Check for statistical anomalies."""
        warnings = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if len(df[col]) > 10:
                skewness = df[col].skew()
                if abs(skewness) > 3:
                    warnings.append(f"Column '{col}' has extreme skewness ({skewness:.2f})")

            if df[col].std() == 0:
                warnings.append(f"Column '{col}' has zero variance (constant values)")

        return len(warnings) == 0, warnings
