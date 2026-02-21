"""Main Streamlit application for AI Data Insights Analyst — Production UI."""

import streamlit as st
import streamlit.components.v1 as components
import logging
import os
import re
from src.utils.logger import get_logger
from src.graph.state import (
    AppState, WorkflowStage, ApprovalStatus,
    save_dataframe, load_dataframe, VIZ_DIR,
)
from src.graph.workflow import orchestrator
from src.utils.file_handlers import load_file, validate_file, compute_file_hash
from ui.styles import (
    inject_global_css, hero_banner, section_header, kpi_card,
    status_badge,
    BRAND_PRIMARY, BRAND_SECONDARY, BRAND_ACCENT,
    BRAND_SUCCESS, BRAND_WARNING, BRAND_DANGER,
    BRAND_BG_CARD, BRAND_BG_SURFACE, BRAND_BG_DARK,
    BRAND_TEXT, BRAND_TEXT_MUTED, BRAND_BORDER,
)
from ui.components import (
    data_health_badge, agent_progress_status,
    trend_results_tab, anomaly_results_tab, correlation_results_tab,
    full_report_tab, pdf_download_section,
)


# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="InsightFlow",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session state ──────────────────────────────────────────────
def init_session_state():
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "show_landing" not in st.session_state:
        st.session_state.show_landing = True


# ───────────────────────────────────────────────────────────────
#  SIDEBAR
# ───────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        # ── Brand logo ──
        st.markdown("""<div style="text-align:center;padding:16px 0 8px 0;">
<div style="font-size:40px;">🔬</div>
<div style="font-size:20px;font-weight:800;color:#1a1a2e;letter-spacing:-0.02em;">InsightFlow</div>
</div>""", unsafe_allow_html=True)

        st.divider()

        # ── Pipeline stepper (native Streamlit) ──
        state = st.session_state.app_state
        stages = [
            ("Data Upload", "📁"),
            ("AI Cleaning", "🧹"),
            ("Verification", "✓"),
            ("Analysis", "🧠"),
            ("Visualization", "📊"),
            ("Report", "📋"),
        ]
        _stage_map = {
            WorkflowStage.UPLOAD: 0,
            WorkflowStage.CLEANING: 1,
            WorkflowStage.CLEANING_APPROVAL: 1,
            WorkflowStage.VERIFICATION: 2,
            WorkflowStage.VERIFICATION_APPROVAL: 2,
            WorkflowStage.ANALYSIS: 3,
            WorkflowStage.INSIGHTS: 3,
            WorkflowStage.VISUALIZATION: 4,
            WorkflowStage.REPORT_SYNTHESIS: 5,
            WorkflowStage.COMPLETED: 6,
        }
        cur_idx = _stage_map.get(state.current_stage, 0)

        for i, (name, icon) in enumerate(stages):
            if i < cur_idx:
                st.write(f"✅ {name}")
            elif i == cur_idx:
                st.write(f"🔵 **{icon} {name}**")
            else:
                st.write(f"⏳ {name}")

        if state.current_stage == WorkflowStage.COMPLETED:
            st.success("🎉 Pipeline Complete!")

        st.divider()

        # ── File info ──
        if state.filename:
            st.caption("ACTIVE DATASET")
            st.write(f"📄 **{state.filename}**")
            st.write(f"Stage: {state.current_stage.value.replace('_', ' ').title()}")

        # ── Error display ──
        if state.last_error:
            st.error(f"⚠️ {state.last_error[:200]}")

        st.divider()

        with st.expander("📂 Previous Analyses", expanded=False):
            past = orchestrator.get_past_analyses()
            if past:
                for item in past:
                    status_icon = "✅" if item["has_insights"] else "⏳"
                    st.write(f"{status_icon} **{item['filename']}**")
                    st.caption(f"{item['rows']}×{item['columns']} — {item['uploaded_at']}")
            else:
                st.caption("No previous analyses yet.")

        # ── Footer ──
        st.markdown("""<div style="position:fixed;bottom:0;left:0;width:inherit;padding:10px 16px;background:linear-gradient(0deg,#FFFFFF 60%,transparent);text-align:center;">
<div style="font-size:12px;color:#6b7280;">Built with LangGraph · Plotly · Streamlit</div>
<div style="font-size:11px;color:#9CA3AF;margin-top:2px;">v2.0 · InsightFlow</div>
</div>""", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
#  UPLOAD SECTION
# ───────────────────────────────────────────────────────────────
def upload_section():
    st.markdown(hero_banner(
        "InsightFlow",
        ""
    ), unsafe_allow_html=True)

    # What happens next — right below the name
    st.markdown("##### ✨ What happens next?")
    st.markdown("""
1. **AI Cleaning** — smart imputation & outlier removal  
2. **Verification** — quality audit & recommendations  
3. **Analysis** — trends, anomalies, correlations  
4. **Visualizations** — beautiful Plotly charts  
5. **Report** — exportable executive summary
""")

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

    # Upload area
    st.markdown(f"""
    <div style="
        background: #F8F9FC;
        border: 2px dashed {BRAND_BORDER};
        border-radius: 20px;
        padding: 12px 24px 4px 24px;
        text-align: center;
        margin-bottom: 8px;
    ">
        <div style="font-size:36px; margin-bottom:4px;">📂</div>
        <div style="font-size:15px; font-weight:600; color:#1a1a2e;">
            Drop your file here or click to browse
        </div>
        <div style="font-size:13px; color:#6b7280; margin-bottom:12px;">
            Supports CSV, XLSX, XLS · Max 50 MB
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload dataset",
        type=["csv", "xlsx", "xls"],
        help="Maximum file size: 50MB",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        import tempfile

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
        ) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name

        is_valid, message = validate_file(tmp_path)

        if is_valid:
            df, load_msg = load_file(tmp_path)

            if df is not None:
                state = st.session_state.app_state
                raw_path = save_dataframe(df, prefix="raw")
                state.raw_data_path = raw_path
                state.filename = uploaded_file.name
                state.file_path = tmp_path

                st.success(f"✅ {load_msg}")

                # KPI row
                missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
                dup_count = df.duplicated().sum()
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(kpi_card("Total Rows", f"{len(df):,}", "📋", BRAND_PRIMARY), unsafe_allow_html=True)
                with c2:
                    st.markdown(kpi_card("Columns", len(df.columns), "📊", BRAND_ACCENT), unsafe_allow_html=True)
                with c3:
                    m_color = BRAND_DANGER if missing_pct > 10 else BRAND_WARNING if missing_pct > 2 else BRAND_SUCCESS
                    st.markdown(kpi_card("Missing", f"{missing_pct:.1f}%", "⚠️", m_color), unsafe_allow_html=True)
                with c4:
                    st.markdown(kpi_card("Duplicates", f"{dup_count:,}", "♻️",
                                         BRAND_WARNING if dup_count > 0 else BRAND_SUCCESS), unsafe_allow_html=True)

                # Data preview
                st.markdown("")
                with st.expander("📋 Data Preview (first 10 rows)", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

                # Column types summary
                with st.expander("📐 Column Types", expanded=False):
                    import pandas as pd
                    import numpy as np
                    dtype_summary = df.dtypes.value_counts().reset_index()
                    dtype_summary.columns = ["Type", "Count"]
                    dtype_summary["Type"] = dtype_summary["Type"].astype(str)
                    st.dataframe(dtype_summary, use_container_width=True, hide_index=True)

                st.markdown("")
                if st.button("🚀 Start AI Pipeline", key="start_cleaning", use_container_width=True):
                    upload_id = orchestrator.save_upload(state)
                    state.upload_id = upload_id
                    state.current_stage = WorkflowStage.CLEANING
                    st.rerun()
            else:
                st.error(f"❌ {load_msg}")
        else:
            st.error(f"❌ {message}")


# ───────────────────────────────────────────────────────────────
#  CLEANING SECTION
# ───────────────────────────────────────────────────────────────
def _render_cleaning_summary(result):
    """Render a visually rich cleaning summary with action cards."""
    actions = getattr(result, "cleaning_actions", None) or []
    summary_text = result.cleaning_summary.split("\n")[0] if result.cleaning_summary else ""

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(99,102,241,.1),rgba(139,92,246,.08));
                border:1px solid {BRAND_BORDER}; border-radius:16px;
                padding:20px 24px; margin-bottom:16px;">
        <div style="font-size:15px; font-weight:600; color:#1a1a2e; margin-bottom:4px;">
            📊 Cleaning Overview
        </div>
        <div style="font-size:14px; color:#4a4a68; line-height:1.6;">
            {summary_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not actions:
        st.caption("No detailed action log available.")
        return

    action_styles = {
        "drop_rows":       ("🗑️", BRAND_DANGER,   "Rows Dropped"),
        "drop_column":     ("❌", "#FF6692",        "Column Dropped"),
        "impute_knn":      ("🧠", "#AB63FA",        "ML Imputation (KNN)"),
        "impute_iterative":("🧠", "#AB63FA",        "ML Imputation (Iterative)"),
        "impute_mean":     ("📊", BRAND_ACCENT,     "Mean Imputation"),
        "impute_median":   ("📊", BRAND_ACCENT,     "Median Imputation"),
        "impute_mode":     ("📊", BRAND_ACCENT,     "Mode Imputation"),
        "impute_zero":     ("📊", BRAND_ACCENT,     "Zero Fill"),
        "impute_ffill":    ("📊", BRAND_ACCENT,     "Forward Fill"),
        "type_cast":       ("🔄", BRAND_WARNING,    "Type Conversion"),
        "deduplicate":     ("♻️", BRAND_SUCCESS,    "Deduplication"),
        "clip_outliers":   ("✂️", BRAND_PRIMARY,    "Outlier Clipping"),
    }

    for i, ca in enumerate(actions, 1):
        icon, color, label = action_styles.get(
            ca.action, ("⚙️", BRAND_TEXT_MUTED, ca.action.replace("_", " ").title())
        )
        st.markdown(f"""
        <div style="background:#FFFFFF; border:1px solid #E5E7EB; border-radius:14px; padding:16px 20px;
                    margin-bottom:10px; border-left:4px solid {color};
                    transition:all .2s ease; box-shadow:0 1px 3px rgba(0,0,0,.06);"
             onmouseover="this.style.background='#F8F9FC'"
             onmouseout="this.style.background='#FFFFFF'">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-size:14px; font-weight:600; color:#1a1a2e;">
                    {icon} Step {i}: {label}
                </span>
                <span style="background:{color}18; color:{color}; padding:3px 12px;
                            border-radius:20px; font-size:12px; font-weight:600;">
                    {ca.rows_affected} rows
                </span>
            </div>
            <div style="font-size:13px; color:#4a4a68; margin-bottom:4px;">
                <strong>Target:</strong>
                <code style="background:rgba(99,102,241,.15); padding:2px 8px;
                            border-radius:6px; color:{BRAND_ACCENT}; font-size:12px;">{ca.target}</code>
            </div>
            <div style="font-size:13px; color:#4a4a68; margin-bottom:8px;">
                <strong>Action:</strong> {ca.detail}
            </div>
            <div style="background:#F1F5F9; border-radius:10px; padding:10px 14px;">
                <div style="font-size:11px; text-transform:uppercase; letter-spacing:.06em;
                            color:#6b7280; margin-bottom:3px; font-weight:600;">
                    💡 Justification
                </div>
                <div style="font-size:13px; color:#1a1a2e; line-height:1.5;">
                    {ca.justification}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def cleaning_section():
    state = st.session_state.app_state

    if not state.raw_data_path:
        st.warning("No data loaded. Please upload a file first.")
        return

    st.markdown(section_header("🧹", "AI Data Cleaning",
                               "Intelligent cleaning with ML imputation and per-step justifications"),
                unsafe_allow_html=True)

    result = state.cleaning_result

    if result is None:
        st.info(
            "🤖 **Ready to Clean** — The AI agent will analyze your data, handle missing values "
            "using ML imputation (KNN / Iterative), remove duplicates, fix data types, and clip "
            "outliers — all with step-by-step justifications."
        )

        if st.button("⚡ Execute AI Cleaning", key="execute_cleaning", use_container_width=True):
            with st.status("🤖 AI Cleaning Agent Running…", expanded=True) as status:
                st.write("⏳ Initializing cleaning agent…")
                st.write(f"📄 Reading data from `{os.path.basename(state.raw_data_path)}`")
                try:
                    st.write("🧠 LLM is analyzing data and generating cleaning code…")
                    st.write("⚙️ Executing cleaning operations via Python REPL…")
                    state = orchestrator.run_cleaning(state)
                    st.session_state.app_state = state
                    if state.last_error:
                        status.update(label="❌ Cleaning Failed", state="error")
                        st.error(state.last_error)
                    else:
                        status.update(label="✅ Cleaning Complete!", state="complete")
                except Exception as e:
                    status.update(label="❌ Cleaning Failed", state="error")
                    st.error(f"Error: {e}")
            st.rerun()
        return

    # ── Results KPIs ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Rows Removed", result.removed_rows, "🗑️", BRAND_DANGER), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Missing Handled", result.missing_handled, "🔧", BRAND_WARNING), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Outliers Removed", result.outliers_removed, "✂️", BRAND_ACCENT), unsafe_allow_html=True)
    with c4:
        q_color = BRAND_SUCCESS if result.data_quality_score >= 0.8 else BRAND_WARNING
        st.markdown(kpi_card("Quality Score", f"{result.data_quality_score:.0%}", "⭐", q_color), unsafe_allow_html=True)

    st.markdown("")

    # ── Per-action justification cards ──
    if getattr(result, "cleaning_actions", None):
        with st.expander("📝 Cleaning Actions & Justifications", expanded=True):
            for i, ca in enumerate(result.cleaning_actions, 1):
                icon = {"drop_rows": "🗑️", "drop_column": "❌", "impute_mean": "📊",
                        "impute_median": "📊", "impute_mode": "📊", "impute_zero": "📊",
                        "impute_ffill": "📊", "impute_knn": "🧠", "impute_iterative": "🧠",
                        "type_cast": "🔄", "deduplicate": "♻️",
                        "clip_outliers": "✂️"}.get(ca.action, "⚙️")
                with st.expander(f"{icon}  Step {i}: **{ca.action}** on `{ca.target}` — {ca.rows_affected} rows"):
                    st.markdown(f"**What:** {ca.detail}")
                    st.markdown(f"**Why:** {ca.justification}")
                    if ca.before_snapshot or ca.after_snapshot:
                        bcol, acol = st.columns(2)
                        with bcol:
                            st.caption("Before")
                            st.json(ca.before_snapshot)
                        with acol:
                            st.caption("After")
                            st.json(ca.after_snapshot)
    else:
        st.info("No per-step cleaning actions were logged.")

    # ── Before / After preview ──
    tab_before, tab_after = st.tabs(["📁 Before Cleaning", "✅ After Cleaning"])
    with tab_before:
        try:
            raw_df = load_dataframe(result.raw_data_path)
            st.dataframe(raw_df.head(10), use_container_width=True, hide_index=True)
        except Exception:
            st.info("Original data preview unavailable.")
    with tab_after:
        try:
            clean_df = load_dataframe(result.cleaned_data_path)
            st.dataframe(clean_df.head(10), use_container_width=True, hide_index=True)
        except Exception:
            st.info("Cleaned data preview unavailable.")

    with st.expander("📋 Full Cleaning Summary", expanded=False):
        _render_cleaning_summary(result)

    # ── Approve / Reject ──
    st.markdown("")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Approve Cleaning", key="approve_cleaning", use_container_width=True):
            state.cleaning_approval_status = ApprovalStatus.APPROVED
            state.current_stage = WorkflowStage.VERIFICATION
            st.rerun()
    with c2:
        if st.button("🔄 Reject & Retry", key="reject_cleaning", use_container_width=True):
            state.cleaning_approval_status = ApprovalStatus.REJECTED
            state.cleaning_result = None
            state.current_stage = WorkflowStage.CLEANING
            st.rerun()


# ───────────────────────────────────────────────────────────────
#  VERIFICATION SECTION
# ───────────────────────────────────────────────────────────────
def verification_section():
    state = st.session_state.app_state

    if not state.cleaned_data_path:
        st.warning("No cleaned data. Please complete cleaning first.")
        return

    st.markdown(section_header("✓", "Data Verification",
                               "AI-powered quality audit with recommendations"),
                unsafe_allow_html=True)

    result = state.verification_result

    if result is None:
        st.info(
            "🔍 **Ready to Verify** — The verification agent will audit every cleaning action, "
            "check column distributions, and provide actionable recommendations."
        )

        if st.button("🔍 Execute Verification", key="execute_verification", use_container_width=True):
            with st.status("🔍 Verification Agent Running…", expanded=True) as status:
                st.write("⏳ Loading cleaned data…")
                st.write("🧠 LLM is evaluating data quality…")
                try:
                    state = orchestrator.run_verification(state)
                    st.session_state.app_state = state
                    if state.last_error:
                        status.update(label="❌ Verification Error", state="error")
                    else:
                        status.update(label="✅ Verification Complete!", state="complete")
                except Exception as e:
                    status.update(label="❌ Verification Failed", state="error")
                    st.error(f"Error: {e}")
            st.rerun()
        return

    # ── Status badge ──
    if result.is_approved:
        st.markdown(f"""
        <div style="background:rgba(16,185,129,.1); border:1px solid rgba(16,185,129,.3);
                    border-radius:14px; padding:16px 20px; display:flex; align-items:center; gap:12px;">
            <span style="font-size:24px;">✅</span>
            <div>
                <div style="font-size:15px; font-weight:600; color:{BRAND_SUCCESS};">Verification Passed</div>
                <div style="font-size:13px; color:#4a4a68;">Data quality meets all thresholds</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:rgba(245,158,11,.08); border:1px solid rgba(245,158,11,.25);
                    border-radius:14px; padding:16px 20px; display:flex; align-items:center; gap:12px;">
            <span style="font-size:24px;">⚠️</span>
            <div>
                <div style="font-size:15px; font-weight:600; color:{BRAND_WARNING};">Issues Found</div>
                <div style="font-size:13px; color:#4a4a68;">Severity: {result.severity}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Recommendations ──
    if result.recommendations:
        st.markdown(f'<div style="font-size:15px; font-weight:700; color:#1a1a2e; margin-bottom:12px;">💡 Recommendations</div>', unsafe_allow_html=True)
        for i, rec in enumerate(result.recommendations, 1):
            st.markdown(f"""
            <div style="background:#F8F9FC; border:1px solid #E5E7EB; border-left:3px solid {BRAND_PRIMARY};
                        border-radius:0 12px 12px 0; padding:12px 16px; margin-bottom:6px;
                        font-size:14px; color:#1a1a2e; line-height:1.6;">
                <span style="color:{BRAND_PRIMARY}; font-weight:600;">#{i}</span> {rec}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("🎯 No recommendations — data looks great!")

    st.markdown(f'<div style="height:1px; background:{BRAND_BORDER}; margin:20px 0;"></div>', unsafe_allow_html=True)

    # ── Feature Engineering ──
    st.markdown(section_header("🧩", "Feature Engineering",
                               "Let AI create new features from your existing columns"),
                unsafe_allow_html=True)

    if "fe_approved" not in st.session_state:
        st.session_state.fe_approved = None

    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Yes, Engineer Features", key="approve_fe", use_container_width=True):
            st.session_state.fe_approved = True
            st.rerun()
    with c2:
        if st.button("⏩ Skip", key="skip_fe", use_container_width=True):
            st.session_state.fe_approved = False
            st.rerun()

    if st.session_state.fe_approved is True:
        _run_feature_engineering(state)
    elif st.session_state.fe_approved is False:
        st.caption("Feature engineering skipped.")

    st.markdown(f'<div style="height:1px; background:{BRAND_BORDER}; margin:20px 0;"></div>', unsafe_allow_html=True)

    # ── Approve / Retry ──
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Approve Data", key="approve_verification", use_container_width=True):
            state.verification_approval_status = ApprovalStatus.APPROVED
            state.current_stage = WorkflowStage.ANALYSIS
            st.session_state.fe_approved = None
            st.rerun()
    with c2:
        if state.verification_attempts < 3 and st.button("🔄 Retry Cleaning", key="retry_cleaning", use_container_width=True):
            state.verification_result = None
            state.cleaning_result = None
            state.current_stage = WorkflowStage.CLEANING
            st.session_state.fe_approved = None
            st.rerun()


# ───────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING
# ───────────────────────────────────────────────────────────────
def _run_feature_engineering(state):
    if getattr(state, "_fe_done", False):
        st.success("✅ Feature engineering already applied!")
        return

    with st.status("🧩 Running Feature Engineering…", expanded=True) as fe_status:
        st.write("🧠 LLM is analyzing columns for useful combinations…")
        try:
            from langchain_experimental.tools.python.tool import PythonAstREPLTool
            from langgraph.prebuilt import create_react_agent
            import numpy as np

            df = load_dataframe(state.cleaned_data_path)
            repl = PythonAstREPLTool()
            repl.locals = {"df": df, "pd": __import__("pandas"), "np": np}

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            dt_cols = [c for c in df.columns if __import__("pandas").api.types.is_datetime64_any_dtype(df[c])]

            fe_prompt = f"""You are a Feature Engineering AI Agent.
A DataFrame `df` is loaded. You MUST use the Python REPL to add new engineered columns.

Available columns:
  Numeric: {numeric_cols}
  Categorical: {cat_cols}
  Datetime: {dt_cols}

CREATE 3-6 useful new features. Ideas:
  - Ratios between numeric pairs
  - Interaction terms
  - Date parts (year, month, day_of_week, is_weekend)
  - Binned versions of skewed numeric cols (pd.qcut)
  - Log transforms of highly skewed columns
  - Category frequency encoding

RULES:
  - Update `df` in the REPL after each new column.
  - Handle division by zero with .replace([np.inf, -np.inf], np.nan).fillna(0)
  - Print each new column name when created.
  - When done say "Feature engineering complete."
"""
            from src.config import settings
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            llm = ChatNVIDIA(
                model=settings.llm_model,
                api_key=settings.nvidia_api_key,
                base_url=settings.openai_base_url,
                max_tokens=4096,
                temperature=0.3,
            )
            agent = create_react_agent(llm, tools=[repl], prompt=fe_prompt)
            agent.invoke({"messages": [("user", "Engineer useful features for this dataset.")]})

            new_df = repl.locals.get("df", df)
            new_cols = [c for c in new_df.columns if c not in df.columns]

            if new_cols:
                new_path = save_dataframe(new_df, prefix="fe_cleaned")
                state.cleaned_data_path = new_path
                st.session_state.app_state = state
                st.write(f"✅ Created **{len(new_cols)}** new features: `{'`, `'.join(new_cols)}`")
                fe_status.update(label=f"✅ Added {len(new_cols)} engineered features!", state="complete")
            else:
                st.write("No new features were created.")
                fe_status.update(label="ℹ️ No features added", state="complete")

            state._fe_done = True
        except Exception as e:
            fe_status.update(label="❌ Feature Engineering Failed", state="error")
            st.error(f"Error: {e}")


# ───────────────────────────────────────────────────────────────
#  ANALYSIS / INSIGHTS SECTION
# ───────────────────────────────────────────────────────────────
def insights_section():
    state = st.session_state.app_state

    if not state.cleaned_data_path:
        st.warning("No data for analysis. Please complete verification first.")
        return

    st.markdown(section_header("🧠", "Full Analysis Pipeline",
                               "5 AI agents run in parallel: Trends · Anomalies · Correlations · Insights · Visualization"),
                unsafe_allow_html=True)

    # ── Pipeline not yet run ──
    if state.insights_result is None and not state.trend_results:
        agent_cols = st.columns(5)
        for col, (icon, name) in zip(agent_cols, [
            ("📈","Trends"), ("🔍","Anomalies"), ("🔗","Correlations"),
            ("💡","Insights"), ("📊","Viz")
        ]):
            with col:
                st.markdown(f"<div style='text-align:center;padding:16px 0;'>"
                            f"<div style='font-size:28px;'>{icon}</div>"
                            f"<div style='font-size:13px;color:#6b7280;margin-top:4px;font-weight:500;'>{name}</div>"
                            f"</div>", unsafe_allow_html=True)

        st.markdown("")
        if st.button("🚀 Run Full Analysis", key="execute_parallel", use_container_width=True):
            if state.file_path and os.path.exists(state.file_path):
                state.file_hash = compute_file_hash(state.file_path)

            with st.status("🔄 Running parallel analysis agents…", expanded=True) as run_status:
                def _progress(name, status):
                    import time as _t
                    ts = _t.strftime("%H:%M:%S")
                    if status == "running":
                        st.write(f"▸ `{ts}` — **{name}** starting…")
                    elif status == "done":
                        timing = state.agent_timings.get(name, "")
                        suffix = f" ({timing}s)" if timing else ""
                        st.write(f"✓ `{ts}` — **{name}** done{suffix}")
                    elif status == "timeout":
                        st.write(f"⏰ `{ts}` — **{name}** timed out")
                    elif status == "error":
                        st.write(f"✗ `{ts}` — **{name}** error")

                st.write("⏳ Initializing 5 parallel agents…")
                try:
                    state = orchestrator.run_parallel_analysis(state, progress_callback=_progress)
                    st.session_state.app_state = state

                    total = state.agent_timings.get("_total_analysis", "?")
                    if state.last_error:
                        run_status.update(label=f"⚠️ Completed with warnings ({total}s)", state="error")
                    else:
                        run_status.update(label=f"✅ Analysis Complete! ({total}s)", state="complete")
                except Exception as e:
                    run_status.update(label="❌ Analysis Failed", state="error")
                    st.error(f"Error: {e}")
            st.rerun()
        return

    # ── Results available — multi-tab dashboard ──
    if state.report_synthesis:
        data_health_badge(state.report_synthesis.data_quality_score)
        st.markdown("")

    tabs = st.tabs([
        "💡 Insights", "📈 Trends", "🔍 Anomalies",
        "🔗 Correlations", "📊 Visualizations", "📋 Full Report",
    ])

    with tabs[0]:
        _show_insights(state)
    with tabs[1]:
        trend_results_tab(state.trend_results)
        _show_charts_for_type(state, "trend")
    with tabs[2]:
        anomaly_results_tab(state.anomaly_results)
        _show_charts_for_type(state, "anomaly")
    with tabs[3]:
        correlation_results_tab(state.correlation_results)
        _show_charts_for_type(state, "corr")
    with tabs[4]:
        visualization_section()
    with tabs[5]:
        full_report_tab(state)
        pdf_download_section(state)


# ───────────────────────────────────────────────────────────────
#  INSIGHTS TAB
# ───────────────────────────────────────────────────────────────
def _show_insights(state):
    result = state.insights_result

    # Fallback to report synthesis
    if result is None and state.report_synthesis:
        st.markdown(section_header("🔑", "Key Findings"), unsafe_allow_html=True)
        if state.report_synthesis.executive_summary:
            st.markdown(state.report_synthesis.executive_summary)
        if state.report_synthesis.recommendations:
            st.markdown(f'<div style="font-size:15px; font-weight:700; color:{BRAND_TEXT}; margin:16px 0 8px 0;">💡 Recommendations</div>', unsafe_allow_html=True)
            for i, rec in enumerate(state.report_synthesis.recommendations, 1):
                st.write(f"{i}. {rec}")
        return

    if result is None:
        if state.last_error and "insight" in state.last_error.lower():
            st.warning(f"Insights generation failed: {state.last_error}")
        else:
            st.info("Insights not yet available. Run the full analysis pipeline first.")
        return

    st.markdown(section_header("🔑", "Key Findings"), unsafe_allow_html=True)

    if result.key_findings:
        for i, finding in enumerate(result.key_findings, 1):
            colors = [BRAND_PRIMARY, BRAND_ACCENT, BRAND_SUCCESS, BRAND_WARNING, BRAND_SECONDARY]
            c = colors[(i - 1) % len(colors)]
            st.markdown(f"""
            <div style="background:#F8F9FC; border:1px solid #E5E7EB; border-left:4px solid {c};
                        border-radius:0 14px 14px 0; padding:14px 18px; margin-bottom:8px;
                        font-size:14px; color:#1a1a2e; line-height:1.6;">
                <span style="color:{c}; font-weight:700; margin-right:6px;">#{i}</span>
                {finding}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No key findings were extracted.")

    with st.expander("📝 Full Analysis", expanded=True):
        st.markdown(result.insights_text)

    # Stats card
    if result.statistical_summary:
        st.markdown("")
        cols = st.columns(min(len(result.statistical_summary), 4))
        for idx, (key, val) in enumerate(result.statistical_summary.items()):
            with cols[idx % len(cols)]:
                label = key.replace("_", " ").title()
                display_val = f"{val:,.2f}" if isinstance(val, float) else f"{val:,}" if isinstance(val, int) else str(val)
                st.markdown(kpi_card(label, display_val, "📉", BRAND_ACCENT), unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
#  CHARTS HELPER
# ───────────────────────────────────────────────────────────────
def _show_charts_for_type(state, chart_type_prefix: str):
    matching = [v for v in state.visualizations if chart_type_prefix in (v.html_path or "")]
    for viz in matching:
        if viz.html_path and os.path.exists(viz.html_path):
            st.markdown(f'<div style="font-size:15px; font-weight:600; color:{BRAND_TEXT}; margin:16px 0 8px 0;">{viz.title}</div>', unsafe_allow_html=True)
            with open(viz.html_path, "r", encoding="utf-8") as f:
                components.html(f.read(), height=500, scrolling=True)


# ───────────────────────────────────────────────────────────────
#  VISUALIZATION SECTION
# ───────────────────────────────────────────────────────────────
def visualization_section():
    state = st.session_state.app_state

    if not state.cleaned_data_path:
        st.warning("No data for visualization.")
        return

    st.markdown(section_header("📊", "Visualizations",
                               "AI-generated Plotly charts with Power BI styling"),
                unsafe_allow_html=True)

    visualizations = state.visualizations

    if not visualizations:
        if st.button("🎨 Generate Visualizations", key="execute_viz", use_container_width=True):
            with st.status("📊 Visualization Agent Running…", expanded=True) as status:
                st.write("⏳ Loading cleaned data into REPL environment…")
                st.write("🧠 LLM is designing diverse chart types…")
                st.write("🎨 Generating Plotly visualizations…")
                try:
                    state = orchestrator.run_visualization(state)
                    st.session_state.app_state = state
                    if state.last_error:
                        status.update(label="❌ Visualization Error", state="error")
                    else:
                        status.update(label=f"✅ Generated {len(state.visualizations)} charts!", state="complete")
                except Exception as e:
                    status.update(label="❌ Visualization Failed", state="error")
                    st.error(f"Error: {e}")
            st.rerun()
        return

    st.markdown(f"""
    <div style="background:rgba(16,185,129,.08); border:1px solid rgba(16,185,129,.2);
                border-radius:12px; padding:12px 18px; margin-bottom:16px;
                font-size:13px; color:{BRAND_SUCCESS}; font-weight:500;">
        ✅ Generated {len(visualizations)} visualizations
    </div>
    """, unsafe_allow_html=True)

    for i, viz in enumerate(visualizations, 1):
        st.markdown(f"""
        <div style="font-size:15px; font-weight:600; color:{BRAND_TEXT}; margin:20px 0 8px 0;
                    padding-bottom:6px; border-bottom:1px solid {BRAND_BORDER};">
            {i}. {viz.title}
        </div>
        """, unsafe_allow_html=True)

        if viz.html_path and os.path.exists(viz.html_path):
            with open(viz.html_path, "r", encoding="utf-8") as f:
                components.html(f.read(), height=500, scrolling=True)
        else:
            st.info(f"Chart file not found: {viz.html_path}")

    st.markdown(f'<div style="height:1px; background:{BRAND_BORDER}; margin:24px 0;"></div>', unsafe_allow_html=True)

    # Export
    st.markdown(f'<div style="font-size:15px; font-weight:700; color:{BRAND_TEXT}; margin-bottom:12px;">📥 Export Report</div>', unsafe_allow_html=True)
    if st.button("📄 Generate HTML Report", key="export_report"):
        report_html = _generate_report(state)
        st.download_button(
            label="⬇️ Download Report",
            data=report_html,
            file_name=f"report_{state.filename.replace('.', '_')}.html",
            mime="text/html",
            key="download_report",
        )


# ───────────────────────────────────────────────────────────────
#  CHAT WITH DATA
# ───────────────────────────────────────────────────────────────
def _build_chat_context(state) -> str:
    """Build a rich context string from analysis results for the chat LLM."""
    context_parts = []

    # Report synthesis
    if state.report_synthesis:
        synth = state.report_synthesis
        context_parts.append(f"EXECUTIVE SUMMARY:\n{synth.executive_summary}")
        if synth.recommendations:
            context_parts.append("RECOMMENDATIONS:\n" + "\n".join(f"- {r}" for r in synth.recommendations))
        context_parts.append(f"Data Quality Score: {synth.data_quality_score}/100")

    # Insights
    if state.insights_result:
        ir = state.insights_result
        if ir.key_findings:
            context_parts.append("KEY FINDINGS:\n" + "\n".join(f"- {f}" for f in ir.key_findings))
        if ir.insights_text:
            context_parts.append(f"DETAILED ANALYSIS:\n{ir.insights_text[:2000]}")

    # Trends
    if state.trend_results and not state.trend_results.get("skipped"):
        narrative = state.trend_results.get("overall_narrative", "")
        if narrative:
            context_parts.append(f"TREND ANALYSIS:\n{narrative[:1000]}")

    # Anomalies
    if state.anomaly_results and not state.anomaly_results.get("skipped"):
        anom_narr = state.anomaly_results.get("narrative", "")
        total_anom = state.anomaly_results.get("total_anomalies", 0)
        anom_rate = state.anomaly_results.get("anomaly_rate", 0)
        context_parts.append(f"ANOMALY DETECTION:\nTotal anomalies: {total_anom}, Rate: {anom_rate:.2%}\n{anom_narr[:800]}")

    # Correlations
    if state.correlation_results and not state.correlation_results.get("skipped"):
        corr_narr = state.correlation_results.get("narrative", "")
        if corr_narr:
            context_parts.append(f"CORRELATION ANALYSIS:\n{corr_narr[:800]}")

    # Visualization list
    if state.visualizations:
        viz_list = "\n".join(f"- {v.title}" for v in state.visualizations)
        context_parts.append(f"GENERATED VISUALIZATIONS:\n{viz_list}")

    return "\n\n---\n\n".join(context_parts) if context_parts else ""


def chat_section():
    state = st.session_state.app_state

    st.markdown(section_header("💬", "Chat with Your Data & Reports",
                               "Ask questions about your dataset, visualizations, trends, anomalies, and insights"),
                unsafe_allow_html=True)

    if not state.cleaned_data_path:
        st.info("🔒 **Chat Locked** — Complete the analysis pipeline to unlock the chat feature.")
        return

    # Context pills showing what the AI knows about
    available = []
    if state.report_synthesis:
        available.append("📋 Report")
    if state.insights_result:
        available.append("💡 Insights")
    if state.trend_results and not state.trend_results.get("skipped"):
        available.append("📈 Trends")
    if state.anomaly_results and not state.anomaly_results.get("skipped"):
        available.append("🔍 Anomalies")
    if state.correlation_results and not state.correlation_results.get("skipped"):
        available.append("🔗 Correlations")
    if state.visualizations:
        available.append(f"📊 {len(state.visualizations)} Charts")

    if available:
        pills_html = " ".join(
            f'<span style="background:rgba(99,102,241,.1); color:#6366F1; '
            f'padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600; '
            f'margin-right:4px;">{a}</span>'
            for a in available
        )
        st.markdown(
            f'<div style="margin-bottom:16px;"><span style="font-size:13px; color:#6b7280; '
            f'margin-right:8px;">AI has context from:</span>{pills_html}</div>',
            unsafe_allow_html=True,
        )

    # Quick-ask suggestion chips
    if not st.session_state.chat_history:
        st.markdown(
            f'<div style="font-size:13px; color:#6b7280; margin-bottom:8px;">💡 Try asking:</div>',
            unsafe_allow_html=True,
        )
        suggestions = [
            "What are the top 3 insights from the data?",
            "Explain the anomalies found",
            "Which columns are most correlated?",
            "Summarize the key trends",
            "What should I focus on to improve data quality?",
        ]
        chip_cols = st.columns(len(suggestions))
        for col, suggestion in zip(chip_cols, suggestions):
            with col:
                if st.button(suggestion, key=f"suggest_{suggestion[:20]}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": suggestion})
                    st.rerun()

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_question = st.chat_input("Ask about your data, charts, trends, anomalies, report…")

    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing…"):
                try:
                    from langchain_experimental.agents import create_pandas_dataframe_agent
                    from src.config import settings
                    from langchain_nvidia_ai_endpoints import ChatNVIDIA

                    df = load_dataframe(state.cleaned_data_path)

                    # Build analysis context
                    analysis_context = _build_chat_context(state)

                    llm = ChatNVIDIA(
                        model=settings.llm_model,
                        api_key=settings.nvidia_api_key,
                        base_url=settings.openai_base_url,
                        max_tokens=4096,
                        temperature=0.3,
                    )

                    prefix = (
                        "You are an expert data analyst assistant. The user has already run a full AI analysis pipeline "
                        "on their dataset. Below is the analysis context including insights, trends, anomalies, "
                        "correlations, and report summary. Use this context AND the DataFrame to answer questions.\n\n"
                        f"=== ANALYSIS CONTEXT ===\n{analysis_context}\n=== END CONTEXT ===\n\n"
                        "When answering:\n"
                        "- Reference specific findings from the analysis context when relevant\n"
                        "- Use the DataFrame for any numeric computations or data lookups\n"
                        "- Be concise but thorough\n"
                        "- If asked about visualizations, describe what they show based on the analysis\n"
                    ) if analysis_context else None

                    agent = create_pandas_dataframe_agent(
                        llm, df,
                        verbose=False,
                        allow_dangerous_code=True,
                        handle_parsing_errors=True,
                        prefix=prefix,
                    )
                    response = agent.invoke({"input": user_question})
                    answer = response.get("output", "I couldn't generate an answer.")
                    st.write(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"Error: {e}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})


# ───────────────────────────────────────────────────────────────
#  HTML REPORT GENERATOR
# ───────────────────────────────────────────────────────────────
def _generate_report(state: AppState) -> str:
    html_parts = [f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Data Insights Report — {state.filename}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        font-family: 'Inter', -apple-system, sans-serif;
        background: linear-gradient(160deg, #0F172A 0%, #1a1035 50%, #0F172A 100%);
        color: #F1F5F9;
        min-height: 100vh;
        padding: 40px 20px;
    }}
    .container {{ max-width: 1100px; margin: 0 auto; }}
    .hero {{
        background: linear-gradient(135deg, rgba(99,102,241,.12), rgba(34,211,238,.08));
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        margin-bottom: 32px;
    }}
    .hero h1 {{
        font-size: 2rem;
        background: linear-gradient(135deg, #6366F1, #22D3EE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }}
    .hero p {{ color: #CBD5E1; margin-top: 8px; }}
    h2 {{
        font-size: 1.4rem;
        font-weight: 700;
        margin: 32px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #334155;
    }}
    .kpi-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }}
    .kpi {{
        flex: 1;
        min-width: 140px;
        background: rgba(30,41,59,.7);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }}
    .kpi .value {{ font-size: 28px; font-weight: 700; }}
    .kpi .label {{ font-size: 12px; color: #CBD5E1; text-transform: uppercase; letter-spacing: .06em; margin-top: 4px; font-weight: 600; }}
    .finding {{
        background: rgba(30,41,59,.5);
        border-left: 4px solid #6366F1;
        border-radius: 0 12px 12px 0;
        padding: 14px 18px;
        margin: 6px 0;
        line-height: 1.6;
    }}
    .chart-frame {{
        border: 1px solid #334155;
        border-radius: 14px;
        margin: 16px 0;
        overflow: hidden;
    }}
    .chart-frame iframe {{ width: 100%; height: 500px; border: none; }}
    .footer {{
        text-align: center;
        padding: 32px 0;
        font-size: 12px;
        color: #64748B;
        border-top: 1px solid #334155;
        margin-top: 40px;
    }}
</style>
</head><body>
<div class="container">
    <div class="hero">
        <h1>🔬 AI Data Insights Report</h1>
        <p>{state.filename}</p>
    </div>
"""]

    if state.insights_result:
        html_parts.append('<h2>🔑 Key Findings</h2>')
        for i, f in enumerate(state.insights_result.key_findings, 1):
            html_parts.append(f'<div class="finding"><strong>#{i}</strong> {f}</div>')

        html_parts.append('<h2>📝 Full Analysis</h2>')
        text = state.insights_result.insights_text
        text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
        text = text.replace("\n", "<br>")
        html_parts.append(f'<div style="line-height:1.8;">{text}</div>')

        stats = state.insights_result.statistical_summary
        if stats:
            html_parts.append('<h2>📉 Statistics</h2><div class="kpi-row">')
            for key, val in stats.items():
                label = key.replace("_", " ").title()
                display_val = f"{val:,.2f}" if isinstance(val, float) else f"{val:,}" if isinstance(val, int) else str(val)
                html_parts.append(f'<div class="kpi"><div class="value">{display_val}</div><div class="label">{label}</div></div>')
            html_parts.append("</div>")

    if state.visualizations:
        html_parts.append('<h2>📊 Visualizations</h2>')
        for viz in state.visualizations:
            if viz.html_path and os.path.exists(viz.html_path):
                with open(viz.html_path, "r", encoding="utf-8") as f:
                    chart_html = f.read()
                escaped = chart_html.replace('"', "&quot;")
                html_parts.append(f'<h3 style="margin:20px 0 8px 0;">{viz.title}</h3>')
                html_parts.append(f'<div class="chart-frame"><iframe srcdoc="{escaped}"></iframe></div>')

    html_parts.append(f"""
    <div class="footer">
        Generated by AI Data Insights Analyst · Built with LangGraph, Plotly &amp; Streamlit
    </div>
</div>
</body></html>""")

    return "\n".join(html_parts)


# ───────────────────────────────────────────────────────────────
#  FLOATING CHAT FAB
# ───────────────────────────────────────────────────────────────
def _render_chat_fab():
    """Render a floating action button that toggles the chat panel."""
    state = st.session_state.app_state
    if not state.cleaned_data_path:
        return  # Only show after pipeline has data

    # Initialize chat panel state
    if "chat_panel_open" not in st.session_state:
        st.session_state.chat_panel_open = False

    msg_count = len(st.session_state.chat_history)
    badge_html = f'<span class="badge">{msg_count}</span>' if msg_count > 0 else ""

    st.markdown(f"""
    <style>
    .chat-fab-wrapper {{
        position: fixed;
        bottom: 32px;
        right: 32px;
        z-index: 9999;
    }}
    </style>
    """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
#  LANDING PAGE (inline – no iframe)
# ───────────────────────────────────────────────────────────────
def _render_landing_page():
    """Render the landing page directly in Streamlit (no iframe)."""

    # Hide Streamlit chrome for landing page
    st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    .stApp > header { display: none !important; }
    [data-testid="stHeader"] { display: none !important; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    [data-testid="stAppViewContainer"] { background: #FFFFFF !important; }
    .stApp { background: #FFFFFF !important; }
    footer { display: none !important; }
    #MainMenu { display: none !important; }
    /* Hide the Streamlit button label text but keep it clickable */
    .landing-cta-wrap .stButton button {
        background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
        color: #FFFFFF !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        padding: 18px 48px !important;
        border-radius: 16px !important;
        border: none !important;
        box-shadow: 0 8px 30px rgba(99,102,241,0.4) !important;
        letter-spacing: -0.01em !important;
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
        cursor: pointer !important;
    }
    .landing-cta-wrap .stButton button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 40px rgba(99,102,241,0.55) !important;
    }
    .landing-cta-wrap2 .stButton button {
        background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
        color: #FFFFFF !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 16px 40px !important;
        border-radius: 16px !important;
        border: none !important;
        box-shadow: 0 8px 30px rgba(99,102,241,0.4) !important;
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
    }
    .landing-cta-wrap2 .stButton button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 40px rgba(99,102,241,0.55) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Read landing HTML and extract only the <style> block ──
    landing_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "landing", "index.html",
    )
    with open(landing_path, "r", encoding="utf-8") as f:
        raw_html = f.read()

    # Extract CSS from <style>...</style>
    import re as _re
    style_match = _re.search(r"<style>(.*?)</style>", raw_html, _re.DOTALL)
    landing_css = style_match.group(1) if style_match else ""

    # Inject the landing CSS
    st.markdown(f"<style>{landing_css}</style>", unsafe_allow_html=True)

    # ── Navbar ──
    st.markdown("""
    <nav class="navbar" style="position:relative;">
        <a href="#" class="nav-brand" style="text-decoration:none;">
            <div class="nav-logo">🔬</div>
            <span class="nav-title">AI Data Insights</span>
        </a>
        <ul class="nav-links" style="list-style:none;display:flex;align-items:center;gap:8px;">
            <li><a href="#features" style="text-decoration:none;color:#4a4a68;font-size:15px;font-weight:500;padding:8px 18px;border-radius:10px;">Features</a></li>
            <li><a href="#how-it-works" style="text-decoration:none;color:#4a4a68;font-size:15px;font-weight:500;padding:8px 18px;border-radius:10px;">How It Works</a></li>
            <li><a href="#tech-stack" style="text-decoration:none;color:#4a4a68;font-size:15px;font-weight:500;padding:8px 18px;border-radius:10px;">Tech Stack</a></li>
        </ul>
    </nav>
    """, unsafe_allow_html=True)

    # ── Hero Section ──
    st.markdown("""
    <section class="hero" style="padding-top:80px;">
        <h1 style="font-size:clamp(2.8rem,5.5vw,4.2rem);font-weight:800;color:#1a1a2e;line-height:1.1;
                    letter-spacing:-0.03em;margin-bottom:24px;
                    background:none;-webkit-text-fill-color:#1a1a2e;">
            Analyze your data <span style="background:linear-gradient(135deg,#6366F1,#8B5CF6);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">in seconds</span>
        </h1>
        <p style="font-size:clamp(1rem,2vw,1.25rem);color:#4a4a68;max-width:640px;margin:0 auto 40px;
                  font-weight:400;line-height:1.7;-webkit-text-fill-color:#4a4a68;">
            Upload your files and uncover valuable insights using AI-powered
            multi-agent analysis. Automated cleaning, visualization, and reporting — all in one platform.
        </p>
    </section>
    """, unsafe_allow_html=True)

    # ── Get Started Button (native Streamlit — guaranteed to work) ──
    _c1, c2, _c3 = st.columns([1, 1, 1])
    with c2:
        st.markdown('<div class="landing-cta-wrap">', unsafe_allow_html=True)
        if st.button("🚀  Get started", key="_hero_get_started", use_container_width=True):
            st.session_state.show_landing = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Social Proof ──
    st.markdown("""
    <div class="social-proof" style="text-align:center;padding:30px 48px 10px;">
        <p class="label" style="text-transform:uppercase;font-size:12px;font-weight:700;
           letter-spacing:0.12em;color:#6b7280;margin-bottom:8px;">Trusted by data analysts and teams</p>
        <div class="stars" style="display:flex;justify-content:center;gap:4px;margin-bottom:12px;">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="#FBBF24"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="#FBBF24"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="#FBBF24"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="#FBBF24"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="#FBBF24"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature Tags ──
    st.markdown("""
    <div class="feature-tags" style="display:flex;flex-wrap:wrap;justify-content:center;gap:12px;padding:20px 48px 60px;max-width:900px;margin:0 auto;">
        <div class="feature-tag"><span class="tag-icon">✨</span> Quick data visualizations</div>
        <div class="feature-tag"><span class="tag-icon">💡</span> Accurate statistical analysis</div>
        <div class="feature-tag"><span class="tag-icon">📊</span> Excel &amp; CSV support</div>
        <div class="feature-tag"><span class="tag-icon">🔍</span> Data cleaning</div>
        <div class="feature-tag"><span class="tag-icon">🤖</span> Multi-agent AI pipeline</div>
        <div class="feature-tag"><span class="tag-icon">🔒</span> Local data processing</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Features Section ──
    st.markdown("""
    <section class="features-section" id="features" style="padding:80px 48px;max-width:1200px;margin:0 auto;">
        <p class="section-label" style="text-align:center;text-transform:uppercase;font-size:13px;font-weight:700;letter-spacing:0.1em;color:#6366F1;margin-bottom:12px;">Capabilities</p>
        <h2 class="section-title" style="text-align:center;font-size:clamp(1.8rem,3.5vw,2.5rem);font-weight:800;color:#1a1a2e;letter-spacing:-0.02em;margin-bottom:16px;">Everything you need for data analysis</h2>
        <p class="section-subtitle" style="text-align:center;font-size:1.05rem;color:#4a4a68;max-width:600px;margin:0 auto 60px;">
            Powered by specialized AI agents that work together to deliver comprehensive insights from your raw data.
        </p>
        <div class="features-grid" style="display:grid;grid-template-columns:repeat(3,1fr);gap:32px;">
            <div class="feature-card" style="background:#fff;border:1px solid #E5E7EB;border-radius:16px;padding:36px 32px;">
                <div class="feature-icon viz" style="width:52px;height:52px;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:24px;margin-bottom:20px;background:rgba(99,102,241,0.1);color:#6366F1;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                </div>
                <h3 style="font-size:1.2rem;font-weight:700;color:#1a1a2e;margin-bottom:10px;">Visualizations</h3>
                <p style="font-size:15px;color:#4a4a68;line-height:1.65;">Automatically create interactive Plotly charts — histograms, scatter plots, bar charts, and more.</p>
            </div>
            <div class="feature-card" style="background:#fff;border:1px solid #E5E7EB;border-radius:16px;padding:36px 32px;">
                <div class="feature-icon insights" style="width:52px;height:52px;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:24px;margin-bottom:20px;background:rgba(139,92,246,0.1);color:#8B5CF6;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                </div>
                <h3 style="font-size:1.2rem;font-weight:700;color:#1a1a2e;margin-bottom:10px;">Insights</h3>
                <p style="font-size:15px;color:#4a4a68;line-height:1.65;">Chat with your data while AI agents uncover actionable trends, anomalies, and correlations.</p>
            </div>
            <div class="feature-card" style="background:#fff;border:1px solid #E5E7EB;border-radius:16px;padding:36px 32px;">
                <div class="feature-icon analysis" style="width:52px;height:52px;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:24px;margin-bottom:20px;background:rgba(34,211,238,0.1);color:#06B6D4;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>
                </div>
                <h3 style="font-size:1.2rem;font-weight:700;color:#1a1a2e;margin-bottom:10px;">Analysis</h3>
                <p style="font-size:15px;color:#4a4a68;line-height:1.65;">Perform complex statistical analysis and generate predictive insights with LLM-driven agents.</p>
            </div>
            <div class="feature-card" style="background:#fff;border:1px solid #E5E7EB;border-radius:16px;padding:36px 32px;">
                <div class="feature-icon cleaning" style="width:52px;height:52px;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:24px;margin-bottom:20px;background:rgba(16,185,129,0.1);color:#10B981;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48 2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48 2.83-2.83"/></svg>
                </div>
                <h3 style="font-size:1.2rem;font-weight:700;color:#1a1a2e;margin-bottom:10px;">Smart Cleaning</h3>
                <p style="font-size:15px;color:#4a4a68;line-height:1.65;">AI-powered data cleaning handles missing values, outliers, and data type validation automatically.</p>
            </div>
            <div class="feature-card" style="background:#fff;border:1px solid #E5E7EB;border-radius:16px;padding:36px 32px;">
                <div class="feature-icon report" style="width:52px;height:52px;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:24px;margin-bottom:20px;background:rgba(245,158,11,0.1);color:#F59E0B;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
                </div>
                <h3 style="font-size:1.2rem;font-weight:700;color:#1a1a2e;margin-bottom:10px;">Executive Reports</h3>
                <p style="font-size:15px;color:#4a4a68;line-height:1.65;">Generate comprehensive HTML and PDF reports with executive summaries and key findings.</p>
            </div>
            <div class="feature-card" style="background:#fff;border:1px solid #E5E7EB;border-radius:16px;padding:36px 32px;">
                <div class="feature-icon agents" style="width:52px;height:52px;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:24px;margin-bottom:20px;background:rgba(239,68,68,0.1);color:#EF4444;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
                </div>
                <h3 style="font-size:1.2rem;font-weight:700;color:#1a1a2e;margin-bottom:10px;">Quality Verification</h3>
                <p style="font-size:15px;color:#4a4a68;line-height:1.65;">Built-in verification agent audits data quality and ensures your analysis is built on reliable data.</p>
            </div>
        </div>
    </section>
    """, unsafe_allow_html=True)

    # ── How It Works ──
    st.markdown("""
    <section class="how-it-works" id="how-it-works" style="background:#F8F9FC;padding:100px 48px;">
        <p class="section-label" style="text-align:center;text-transform:uppercase;font-size:13px;font-weight:700;letter-spacing:0.1em;color:#6366F1;margin-bottom:12px;">Workflow</p>
        <h2 class="section-title" style="text-align:center;font-size:clamp(1.8rem,3.5vw,2.5rem);font-weight:800;color:#1a1a2e;letter-spacing:-0.02em;margin-bottom:16px;">How it works</h2>
        <p class="section-subtitle" style="text-align:center;font-size:1.05rem;color:#4a4a68;max-width:600px;margin:0 auto 64px;">
            A simple four-step pipeline powered by specialized AI agents working in concert.
        </p>
        <div style="max-width:1000px;margin:0 auto;display:grid;grid-template-columns:repeat(4,1fr);gap:0;position:relative;">
            <div style="text-align:center;position:relative;z-index:1;">
                <div style="width:80px;height:80px;border-radius:50%;background:#fff;border:3px solid #E5E7EB;display:flex;align-items:center;justify-content:center;margin:0 auto 20px;font-size:28px;">📁</div>
                <h4 style="font-size:1rem;font-weight:700;color:#1a1a2e;margin-bottom:8px;">Upload</h4>
                <p style="font-size:14px;color:#6b7280;max-width:200px;margin:0 auto;">Drop your CSV or Excel file — up to 50MB supported.</p>
            </div>
            <div style="text-align:center;position:relative;z-index:1;">
                <div style="width:80px;height:80px;border-radius:50%;background:#fff;border:3px solid #E5E7EB;display:flex;align-items:center;justify-content:center;margin:0 auto 20px;font-size:28px;">🧹</div>
                <h4 style="font-size:1rem;font-weight:700;color:#1a1a2e;margin-bottom:8px;">Clean &amp; Verify</h4>
                <p style="font-size:14px;color:#6b7280;max-width:200px;margin:0 auto;">AI agents clean, impute, and verify your data quality automatically.</p>
            </div>
            <div style="text-align:center;position:relative;z-index:1;">
                <div style="width:80px;height:80px;border-radius:50%;background:#fff;border:3px solid #E5E7EB;display:flex;align-items:center;justify-content:center;margin:0 auto 20px;font-size:28px;">🧠</div>
                <h4 style="font-size:1rem;font-weight:700;color:#1a1a2e;margin-bottom:8px;">Analyze</h4>
                <p style="font-size:14px;color:#6b7280;max-width:200px;margin:0 auto;">Trend, anomaly, and correlation agents extract deep insights.</p>
            </div>
            <div style="text-align:center;position:relative;z-index:1;">
                <div style="width:80px;height:80px;border-radius:50%;background:#fff;border:3px solid #E5E7EB;display:flex;align-items:center;justify-content:center;margin:0 auto 20px;font-size:28px;">📊</div>
                <h4 style="font-size:1rem;font-weight:700;color:#1a1a2e;margin-bottom:8px;">Visualize &amp; Report</h4>
                <p style="font-size:14px;color:#6b7280;max-width:200px;margin:0 auto;">Get interactive charts and downloadable executive reports.</p>
            </div>
        </div>
    </section>
    """, unsafe_allow_html=True)

    # ── Tech Stack ──
    st.markdown("""
    <section class="tech-section" id="tech-stack" style="padding:100px 48px;max-width:1200px;margin:0 auto;">
        <p class="section-label" style="text-align:center;text-transform:uppercase;font-size:13px;font-weight:700;letter-spacing:0.1em;color:#6366F1;margin-bottom:12px;">Built with the best</p>
        <h2 class="section-title" style="text-align:center;font-size:clamp(1.8rem,3.5vw,2.5rem);font-weight:800;color:#1a1a2e;letter-spacing:-0.02em;margin-bottom:60px;">Powered by modern technology</h2>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:24px;">
            <div class="tech-card" style="text-align:center;padding:32px 24px;border:1px solid #E5E7EB;border-radius:16px;background:#fff;">
                <div style="font-size:36px;margin-bottom:14px;">🔗</div>
                <h4 style="font-size:1rem;font-weight:700;color:#1a1a2e;margin-bottom:6px;">LangGraph</h4>
                <p style="font-size:13px;color:#6b7280;">Graph-based multi-agent orchestration</p>
            </div>
            <div class="tech-card" style="text-align:center;padding:32px 24px;border:1px solid #E5E7EB;border-radius:16px;background:#fff;">
                <div style="font-size:36px;margin-bottom:14px;">🤖</div>
                <h4 style="font-size:1rem;font-weight:700;color:#1a1a2e;margin-bottom:6px;">LLM Powered</h4>
                <p style="font-size:13px;color:#6b7280;">GPT OSS 120B via NVIDIA API</p>
            </div>
            <div class="tech-card" style="text-align:center;padding:32px 24px;border:1px solid #E5E7EB;border-radius:16px;background:#fff;">
                <div style="font-size:36px;margin-bottom:14px;">📊</div>
                <h4 style="font-size:1rem;font-weight:700;color:#1a1a2e;margin-bottom:6px;">Plotly</h4>
                <p style="font-size:13px;color:#6b7280;">Interactive data visualizations</p>
            </div>
            <div class="tech-card" style="text-align:center;padding:32px 24px;border:1px solid #E5E7EB;border-radius:16px;background:#fff;">
                <div style="font-size:36px;margin-bottom:14px;">⚡</div>
                <h4 style="font-size:1rem;font-weight:700;color:#1a1a2e;margin-bottom:6px;">Streamlit</h4>
                <p style="font-size:13px;color:#6b7280;">Fast, responsive user interface</p>
            </div>
        </div>
    </section>
    """, unsafe_allow_html=True)

    # ── Final CTA ──
    st.markdown("""
    <section style="padding:80px 48px;text-align:center;background:linear-gradient(135deg,rgba(99,102,241,0.04),rgba(139,92,246,0.04));">
        <h2 style="font-size:clamp(1.8rem,3.5vw,2.5rem);font-weight:800;color:#1a1a2e;letter-spacing:-0.02em;margin-bottom:16px;">
            Ready to unlock your data's potential?
        </h2>
        <p style="font-size:1.1rem;color:#4a4a68;max-width:500px;margin:0 auto 36px;">
            Start analyzing your datasets in seconds with AI-powered agents. No setup required.
        </p>
    </section>
    """, unsafe_allow_html=True)

    _c4, c5, _c6 = st.columns([1, 1, 1])
    with c5:
        st.markdown('<div class="landing-cta-wrap2">', unsafe_allow_html=True)
        if st.button("🚀  Get started — it's free", key="_cta_get_started", use_container_width=True):
            st.session_state.show_landing = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Footer ──
    st.markdown("""
    <footer style="background:#1a1a2e;color:#94A3B8;padding:60px 48px 40px;margin-top:40px;">
        <div style="max-width:1200px;margin:0 auto;display:grid;grid-template-columns:2fr 1fr 1fr 1fr;gap:48px;">
            <div>
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
                    <div style="width:36px;height:36px;background:linear-gradient(135deg,#6366F1,#8B5CF6);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:17px;color:white;">🔬</div>
                    <span style="font-size:18px;font-weight:700;color:#FFFFFF;">AI Data Insights</span>
                </div>
                <p style="font-size:14px;line-height:1.7;color:#94A3B8;max-width:300px;">
                    A sophisticated multi-agent system that automates data cleaning, verification, analysis, and visualization using LangGraph and LLMs.
                </p>
            </div>
            <div>
                <h5 style="font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#FFFFFF;margin-bottom:20px;">Product</h5>
                <ul style="list-style:none;padding:0;">
                    <li style="margin-bottom:12px;"><a href="#features" style="text-decoration:none;color:#94A3B8;font-size:14px;">Features</a></li>
                    <li style="margin-bottom:12px;"><a href="#how-it-works" style="text-decoration:none;color:#94A3B8;font-size:14px;">How It Works</a></li>
                    <li style="margin-bottom:12px;"><a href="#tech-stack" style="text-decoration:none;color:#94A3B8;font-size:14px;">Tech Stack</a></li>
                </ul>
            </div>
            <div>
                <h5 style="font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#FFFFFF;margin-bottom:20px;">Agents</h5>
                <ul style="list-style:none;padding:0;">
                    <li style="margin-bottom:12px;"><span style="color:#94A3B8;font-size:14px;">Cleaning Agent</span></li>
                    <li style="margin-bottom:12px;"><span style="color:#94A3B8;font-size:14px;">Verification Agent</span></li>
                    <li style="margin-bottom:12px;"><span style="color:#94A3B8;font-size:14px;">Insights Agent</span></li>
                    <li style="margin-bottom:12px;"><span style="color:#94A3B8;font-size:14px;">Visualization Agent</span></li>
                </ul>
            </div>
            <div>
                <h5 style="font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#FFFFFF;margin-bottom:20px;">Resources</h5>
                <ul style="list-style:none;padding:0;">
                    <li style="margin-bottom:12px;"><span style="color:#94A3B8;font-size:14px;">Documentation</span></li>
                    <li style="margin-bottom:12px;"><span style="color:#94A3B8;font-size:14px;">GitHub</span></li>
                </ul>
            </div>
        </div>
        <div style="max-width:1200px;margin:40px auto 0;padding-top:24px;border-top:1px solid rgba(255,255,255,0.08);display:flex;justify-content:space-between;align-items:center;font-size:13px;color:#64748B;">
            <span>&copy; 2026 AI Data Insights Analyst. All rights reserved.</span>
            <span>Built with LangGraph · Plotly · Streamlit</span>
        </div>
    </footer>
    """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
#  MAIN
# ───────────────────────────────────────────────────────────────
def main():
    init_session_state()

    # ── Landing page gate ──
    if st.session_state.show_landing:
        _render_landing_page()
        return

    # ── Main application ──
    inject_global_css()
    sidebar()

    state = st.session_state.app_state
    current_stage = state.current_stage

    # Initialize chat panel toggle
    if "chat_panel_open" not in st.session_state:
        st.session_state.chat_panel_open = False

    if current_stage == WorkflowStage.COMPLETED:
        tab_results, tab_chat = st.tabs(["📈 Results Dashboard", "💬 Chat with Data"])
        with tab_results:
            insights_section()
        with tab_chat:
            chat_section()
    elif current_stage == WorkflowStage.UPLOAD:
        upload_section()
    elif current_stage in (WorkflowStage.CLEANING, WorkflowStage.CLEANING_APPROVAL):
        cleaning_section()
    elif current_stage in (WorkflowStage.VERIFICATION, WorkflowStage.VERIFICATION_APPROVAL):
        verification_section()
    elif current_stage in (WorkflowStage.INSIGHTS, WorkflowStage.ANALYSIS, WorkflowStage.REPORT_SYNTHESIS):
        insights_section()
    elif current_stage == WorkflowStage.VISUALIZATION:
        visualization_section()

    # Floating chat button — visible after data is cleaned
    if state.cleaned_data_path and current_stage != WorkflowStage.COMPLETED:
        st.markdown("")
        st.divider()
        with st.expander("💬 Chat with Your Data", expanded=st.session_state.chat_panel_open):
            st.session_state.chat_panel_open = True
            chat_section()


if __name__ == "__main__":
    main()
