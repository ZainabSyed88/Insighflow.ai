"""Insights agent for pattern detection and analysis."""

import pandas as pd
import numpy as np
import time
from src.agents.base_agent import Agent
from src.graph.state import AppState, InsightsResult, load_dataframe
from src.config import settings
import logging
from typing import Dict, List, Any
import re
import json
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


class InsightsAgent(Agent):
    """Agent responsible for analyzing patterns and generating insights using LLM."""
    
    def __init__(self):
        """Initialize the insights agent."""
        super().__init__(name="InsightsAgent")
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Data Analyst AI Agent.
Your job is to deeply analyze the provided dataset summary and formulate insightful patterns, anomalies, and findings.
The user provides statistical context and column summaries of a cleaned dataset.

Return your analysis strictly as a JSON object matching this schema:
{{
  "patterns": {{"description_of_pattern_type": "specific_finding", ...}},
  "anomalies": [{{"column": "...", "type": "...", "description": "..."}}, ...],
  "key_findings": ["finding 1", "finding 2", ...],
  "insights_text": "A comprehensive markdown-formatted summary of your analysis, suitable for business stakeholders."
}}

Ensure valid JSON structure and nothing else.
"""),
            ("user", "{data_context}")
        ])
    
    def execute(self, state: AppState, user_context: dict | None = None) -> AppState:
        """
        Execute analysis on cleaned data.
        
        Args:
            state: Current state with cleaned_data_path
            user_context: Optional dict for future auth / RBAC injection.
            
        Returns:
            Updated state with insights_result
        """
        from src.utils.logger import audit, StepTimer
        from src.utils.llm_utils import _invoke_with_timeout, LLM_CALL_TIMEOUT
        audit("InsightsAgent.execute START", filename=state.filename)
        timer = StepTimer("InsightsAgent", logger)
        t0 = time.time()
        self._log_action("Starting insights analysis with LLM")
        df = None  # Will be set after loading; used in fallback
        
        if not state.cleaned_data_path:
            logger.error("No cleaned data available for analysis")
            state.last_error = "No data for analysis"
            return state
        
        try:
            with timer.step("load data"):
                df = load_dataframe(state.cleaned_data_path)
            
            with timer.step("generate summary"):
                # Generate rich statistical summary
                statistical_summary = self._generate_statistical_summary(df)
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                cat_cols = df.select_dtypes(include=["object", "category"]).columns
                
                # Use dtypes + describe (no df.head()) to save tokens
                context = f"Columns & Dtypes:\n{df.dtypes.to_string()}\n\n"
                context += f"Statistical Summary:\n{json.dumps(statistical_summary, indent=2)}\n\n"
                context += f"Numerical Describe:\n{df[numeric_cols].describe().to_json()}\n\n"
                
                if len(cat_cols) > 0:
                    context += f"Categorical Describe:\n{df[cat_cols].describe().to_json()}\n\n"
                
                if len(numeric_cols) > 1:
                    context += f"Correlation Matrix:\n{df[numeric_cols].corr().to_json()}\n\n"
            
            with timer.step("LLM invoke"):
                chain = self.prompt | self.llm
                response = _invoke_with_timeout(chain, {"data_context": context}, timeout=LLM_CALL_TIMEOUT)
            
            with timer.step("parse response"):
                content = response.content
                # Strip <think>...</think> tags (Qwen/DeepSeek models)
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                # Strip markdown code fences
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                # Try to find JSON object if there's surrounding text
                if not content.startswith("{"):
                    match = re.search(r"\{.*\}", content, re.DOTALL)
                    if match:
                        content = match.group(0)
                    
                llm_result = json.loads(content)
                
                state.insights_result = InsightsResult(
                    patterns=llm_result.get("patterns", {}),
                    anomalies=llm_result.get("anomalies", []),
                    statistical_summary=statistical_summary,
                    insights_text=llm_result.get("insights_text", "No insights generated."),
                    key_findings=llm_result.get("key_findings", [])
                )
            
            self._log_action("Insights analysis completed", {
                "patterns_found": len(llm_result.get("patterns", {})),
                "anomalies_detected": len(llm_result.get("anomalies", [])),
                "key_findings": len(llm_result.get("key_findings", []))
            })
            
        except TimeoutError:
            logger.error("InsightsAgent LLM call timed out")
            state.last_error = "Insights LLM call timed out"
            # Fallback: produce a basic statistical insights result
            state.insights_result = self._fallback_insights(df)
        except Exception as e:
            logger.error(f"Error during insights analysis: {str(e)}")
            state.last_error = f"Analysis error: {str(e)}"
            # Fallback: produce a basic statistical insights result
            state.insights_result = self._fallback_insights(df)
        
        state.agent_timings["InsightsAgent"] = round(time.time() - t0, 2)
        timer.summary()
        audit("InsightsAgent.execute END", filename=state.filename,
              duration=state.agent_timings["InsightsAgent"])
        return state
        
    def _fallback_insights(self, df) -> InsightsResult:
        """Produce a basic statistical insights result when LLM fails."""
        logger.info("Generating fallback insights from raw statistics")
        if df is None:
            return InsightsResult(
                insights_text="Insights generation failed and no data was available for fallback analysis.",
                key_findings=["Insights agent encountered an error. Please re-run the analysis."],
            )

        findings: List[str] = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        findings.append(f"Dataset contains {len(df):,} rows and {len(df.columns)} columns "
                        f"({len(numeric_cols)} numeric, {len(cat_cols)} categorical).")

        # Missing values
        missing = df.isnull().sum()
        cols_with_missing = missing[missing > 0]
        if len(cols_with_missing):
            worst = cols_with_missing.idxmax()
            findings.append(f"{len(cols_with_missing)} column(s) still have missing values; "
                            f"worst is '{worst}' with {cols_with_missing[worst]:,} missing.")
        else:
            findings.append("No missing values detected after cleaning.")

        # Numeric highlights
        if len(numeric_cols):
            desc = df[numeric_cols].describe()
            for col in numeric_cols[:5]:  # top 5
                findings.append(
                    f"'{col}': mean={desc.at['mean', col]:.2f}, "
                    f"std={desc.at['std', col]:.2f}, "
                    f"range=[{desc.at['min', col]:.2f}, {desc.at['max', col]:.2f}]"
                )

        # Categorical highlights
        for col in cat_cols[:3]:
            n_unique = df[col].nunique()
            top = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
            findings.append(f"'{col}': {n_unique} unique values, most common = '{top}'")

        text = "## Fallback Statistical Summary\n\n"
        text += "*LLM-based insights were unavailable; showing auto-generated statistics.*\n\n"
        for i, f in enumerate(findings, 1):
            text += f"{i}. {f}\n"

        return InsightsResult(
            patterns={},
            anomalies=[],
            statistical_summary=self._generate_statistical_summary(df),
            insights_text=text,
            key_findings=findings,
        )

    def _generate_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics natively to pass to LLM."""
        numeric_count = len(df.select_dtypes(include=[np.number]).columns)
        cat_count = len(df.select_dtypes(include=['object', 'category']).columns)
        
        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": numeric_count,
            "categorical_columns": cat_count,
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 ** 2),
        }
        
        return summary
