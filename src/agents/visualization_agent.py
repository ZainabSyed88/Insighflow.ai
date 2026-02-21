"""Visualization agent for generating charts via LLM-driven Python REPL."""

import pandas as pd
import numpy as np
import os
import time
import threading
from src.agents.base_agent import Agent
from src.graph.state import AppState, VisualizationConfig, VIZ_DIR, load_dataframe
from src.config import settings
import logging
from typing import List
import json
from langgraph.prebuilt import create_react_agent
from langchain_experimental.tools.python.tool import PythonAstREPLTool

logger = logging.getLogger(__name__)

# Maximum time for the REPL agent loop (seconds)
VIZ_AGENT_TIMEOUT = 120


class VisualizationAgent(Agent):
    """Agent that uses a Python REPL to generate diverse Plotly visualizations."""
    
    def __init__(self):
        """Initialize the visualization agent."""
        super().__init__(name="VisualizationAgent")
        
        self.python_repl = PythonAstREPLTool()
        
        system_prompt = f"""You are an expert Data Visualization AI Agent.
You have access to a Python REPL with pandas, numpy, and plotly pre-loaded.
A DataFrame named `df` is available in the environment.
A variable `VIZ_DIR` contains the directory path where you must save charts.

YOUR TASK: Generate AT LEAST 4-5 diverse, POWER BI-QUALITY Plotly visualizations.
You MUST include a MIX of the following chart types:
1. A **histogram** or **distribution plot** for an important numeric column
2. A **bar chart** (horizontal or vertical) showing top categories, value counts, or aggregated metrics (e.g. average by category)
3. A **grouped or stacked bar chart** comparing a metric across two categorical dimensions
4. A **box plot** or **violin plot** showing distribution across categories
5. A **scatter plot** or **bubble chart** showing relationships between two numeric columns
6. (BONUS) A **line chart**, **pie chart**, **donut**, **treemap**, or **KPI card** if relevant

DO NOT generate a correlation heatmap — that is handled separately by the Correlation Agent.

CRITICAL STYLING — POWER BI LOOK (follow exactly):
- ALWAYS apply this base layout to EVERY figure:
  ```python
  fig.update_layout(
      template='plotly_dark',
      paper_bgcolor='#0e1117',
      plot_bgcolor='#111827',
      font=dict(family='Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif', color='#e2e8f0', size=13),
      title=dict(font=dict(size=20, color='#f8fafc'), x=0.5, xanchor='center'),
      margin=dict(l=60, r=30, t=75, b=55),
      hoverlabel=dict(bgcolor='#1e293b', font_size=13, font_color='#f1f5f9', bordercolor='#334155'),
      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, bgcolor='rgba(0,0,0,0)'),
  )
  ```
- ALWAYS add subtle gridlines: `fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.06)', showline=True, linewidth=1, linecolor='#334155')`
  Same for yaxes.
- Use these colors for categorical charts: `['#636EFA','#EF553B','#00CC96','#AB63FA','#FFA15A','#19D3F3','#FF6692','#B6E880']`
- For continuous scales use `'Viridis'`, `'Plasma'`, or `'Turbo'`
- For heatmaps use `colorscale='RdYlBu'`
- Add hover templates: `hovertemplate='Column: %{{y:,.2f}}<extra></extra>'`
- NEVER use default grey bars/markers — every trace MUST have explicit bright colors
- Titles MUST be centered
- Save to HTML: `fig.write_html(os.path.join(VIZ_DIR, 'chart_N.html'))`
- After saving, print a JSON line like: {{"chart_type": "histogram", "title": "...", "file": "chart_N.html"}}

IMPORTANT:
- Do NOT show figures, only save them
- Save at least 4 different charts
- Use `os.path.join(VIZ_DIR, ...)` for all file paths
- When done, say "All visualizations generated."
"""
        
        self.tools = [self.python_repl]
        
        self.agent_executor = create_react_agent(
            self.llm,
            tools=self.tools,
            prompt=system_prompt
        )
    
    def execute(self, state: AppState, user_context: dict | None = None) -> AppState:
        """
        Execute visualization generation using LLM-driven REPL.
        Uses a thread-based timeout to prevent indefinite LLM loops.
        """
        from src.utils.logger import audit, StepTimer
        audit("VisualizationAgent.execute START", filename=state.filename)
        timer = StepTimer("VisualizationAgent", logger)
        t0 = time.time()
        self._log_action("Starting visualization generation with LLM REPL")
        
        if not state.cleaned_data_path:
            logger.error("No data available for visualization")
            state.last_error = "No data for visualization"
            return state
        
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            
            with timer.step("load data"):
                df = load_dataframe(state.cleaned_data_path)
            
            with timer.step("clean viz dir"):
                # Clean the viz directory
                for f in os.listdir(VIZ_DIR):
                    if f.endswith('.html'):
                        os.remove(os.path.join(VIZ_DIR, f))
            
            # Inject into REPL environment
            self.python_repl.locals = {
                "df": df, "pd": pd, "np": np,
                "go": go, "px": px, "os": os,
                "VIZ_DIR": VIZ_DIR,
            }
            
            # Build context for the LLM
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            context = f"DataFrame shape: {df.shape}\n"
            context += f"Columns & dtypes:\n{df.dtypes.to_string()}\n"
            context += f"Numeric columns: {numeric_cols}\n"
            context += f"Categorical columns: {categorical_cols}\n"
            context += f"Describe:\n{df.describe().to_json()}\n"
            
            if state.insights_result and state.insights_result.key_findings:
                context += f"\nKey findings from analysis:\n"
                for f_item in state.insights_result.key_findings[:5]:
                    context += f"- {f_item}\n"
            
            # Execute the agent with a timeout
            with timer.step("LLM REPL agent"):
                result_holder = []
                error_holder = []

                def _run_agent():
                    try:
                        res = self.agent_executor.invoke({
                            "messages": [("user", f"Generate diverse visualizations for this dataset:\n{context}")]
                        })
                        result_holder.append(res)
                    except Exception as e:
                        error_holder.append(e)

                agent_thread = threading.Thread(target=_run_agent, daemon=True)
                agent_thread.start()
                agent_thread.join(timeout=VIZ_AGENT_TIMEOUT)

                if agent_thread.is_alive():
                    logger.warning(
                        f"VisualizationAgent REPL loop timed out after {VIZ_AGENT_TIMEOUT}s — "
                        "collecting whatever charts were saved so far"
                    )
                elif error_holder:
                    logger.error(f"VisualizationAgent REPL error: {error_holder[0]}")
                elif result_holder:
                    result = result_holder[0]
                    output_msg = result["messages"][-1].content if "messages" in result else "Completed"
                    logger.info(f"Viz agent output: {output_msg[:200]}")
            
            with timer.step("collect charts"):
                # Collect generated HTML files
                visualizations = []
                for fname in sorted(os.listdir(VIZ_DIR)):
                    if fname.endswith('.html'):
                        html_path = os.path.join(VIZ_DIR, fname)
                        chart_idx = len(visualizations) + 1
                        viz_config = VisualizationConfig(
                            chart_type="plotly",
                            title=f"Visualization {chart_idx}",
                            html_path=html_path,
                        )
                        visualizations.append(viz_config)
                
                # Fallback: if agent produced nothing, generate one basic chart
                if not visualizations and numeric_cols:
                    fig = px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}", template="plotly_dark")
                    fallback_path = os.path.join(VIZ_DIR, "chart_fallback.html")
                    fig.write_html(fallback_path)
                    visualizations.append(VisualizationConfig(
                        chart_type="histogram",
                        title=f"Distribution of {numeric_cols[0]}",
                        html_path=fallback_path,
                    ))
            
            state.visualizations = visualizations
            state.visualization_data = {
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "visualization_count": len(visualizations),
            }
            
            self._log_action("Visualization generation completed", {
                "charts_created": len(visualizations)
            })
            
        except Exception as e:
            logger.error(f"Error during visualization generation: {str(e)}")
            state.last_error = f"Visualization error: {str(e)}"
        
        state.agent_timings["VisualizationAgent"] = round(time.time() - t0, 2)
        timer.summary()
        audit("VisualizationAgent.execute END", filename=state.filename,
              duration=state.agent_timings.get("VisualizationAgent"))
        return state
