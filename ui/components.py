"""Reusable Streamlit UI components for the AI Data Insights Analyst."""

import streamlit as st
import streamlit.components.v1 as html_components
import os
from typing import Optional
from ui.styles import (
    BRAND_PRIMARY, BRAND_SECONDARY, BRAND_ACCENT,
    BRAND_SUCCESS, BRAND_WARNING, BRAND_DANGER,
    BRAND_BG_CARD, BRAND_BG_SURFACE, BRAND_BG_DARK,
    BRAND_TEXT, BRAND_TEXT_MUTED, BRAND_BORDER,
    kpi_card, glass_card, status_badge, section_header,
)


# ───────────────────────────────────────────────────────────────
# Data Health Badge
# ───────────────────────────────────────────────────────────────
def data_health_badge(score: float) -> None:
    """Render a premium Data Health Score gauge."""
    if score >= 80:
        colour, label, glow = BRAND_SUCCESS, "Excellent", f"{BRAND_SUCCESS}40"
    elif score >= 60:
        colour, label, glow = "#22D3EE", "Good", "#22D3EE40"
    elif score >= 40:
        colour, label, glow = BRAND_WARNING, "Fair", f"{BRAND_WARNING}40"
    else:
        colour, label, glow = BRAND_DANGER, "At Risk", f"{BRAND_DANGER}40"

    pct = min(score, 100)
    st.markdown(f"""
    <div style="
        display: flex;
        align-items: center;
        gap: 20px;
        background: #FFFFFF;
        border: 1px solid {BRAND_BORDER};
        border-radius: 16px;
        padding: 20px 28px;
        box-shadow: 0 1px 3px rgba(0,0,0,.06);
        animation: fadeInUp .5s ease-out;
    ">
        <!-- Circular gauge -->
        <div style="position:relative; width:80px; height:80px; flex-shrink:0;">
            <svg viewBox="0 0 36 36" style="width:80px; height:80px; transform:rotate(-90deg);">
                <circle cx="18" cy="18" r="15.9" fill="none"
                        stroke="{BRAND_BORDER}" stroke-width="2.5"/>
                <circle cx="18" cy="18" r="15.9" fill="none"
                        stroke="{colour}" stroke-width="2.5"
                        stroke-dasharray="{pct} {100-pct}"
                        stroke-linecap="round"
                        style="filter:drop-shadow(0 0 4px {glow}); transition:stroke-dasharray 1s ease;"/>
            </svg>
            <div style="position:absolute; inset:0; display:flex; align-items:center;
                        justify-content:center; font-size:18px; font-weight:700; color:{colour};">
                {score:.0f}
            </div>
        </div>
        <!-- Label -->
        <div>
            <div style="font-size:12px; text-transform:uppercase; letter-spacing:.08em;
                        color:#6b7280; font-weight:600; margin-bottom:2px;">
                Data Health Score
            </div>
            <div style="font-size:22px; font-weight:700; color:{colour};">
                {label}
            </div>
            <div style="font-size:13px; color:#6b7280; margin-top:2px;">
                {score:.0f} / 100 quality points
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
# Agent Progress
# ───────────────────────────────────────────────────────────────
def agent_progress_status(agent_statuses: dict) -> None:
    """Show per-agent status indicators with styled badges."""
    for name, status in agent_statuses.items():
        if status == "done":
            badge = status_badge("Completed", "success")
        elif status == "running":
            badge = status_badge("Running…", "info")
        elif status == "hit":
            badge = status_badge("Cache Hit", "info")
        elif status == "timeout":
            badge = status_badge("Timed Out", "warning")
        elif status == "error":
            badge = status_badge("Error", "danger")
        else:
            badge = status_badge("Pending", "info")
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;padding:6px 0;">'
            f'<span style="font-size:14px;color:{BRAND_TEXT};font-weight:500;">{name}</span>'
            f'{badge}'
            f'</div>',
            unsafe_allow_html=True,
        )


# ───────────────────────────────────────────────────────────────
# Trend Results
# ───────────────────────────────────────────────────────────────
def trend_results_tab(trend_results: dict) -> None:
    """Render trend analysis results inside a styled container."""
    if not trend_results or trend_results.get("skipped"):
        st.info("Trend analysis was skipped (no suitable numeric columns).")
        return

    st.markdown(section_header("📈", "Trend Analysis"), unsafe_allow_html=True)

    # Narrative card
    if trend_results.get("overall_narrative"):
        st.markdown(trend_results["overall_narrative"])

    if trend_results.get("datetime_column"):
        st.caption(f"Datetime column detected: **{trend_results['datetime_column']}**")

    columns = trend_results.get("columns", [])
    if columns:
        import pandas as pd
        df = pd.DataFrame(columns)
        display_cols = [c for c in ["column", "trend_direction", "is_significant", "p_value"]
                        if c in df.columns]
        if display_cols:
            st.dataframe(df[display_cols], use_container_width=True, hide_index=True)


# ───────────────────────────────────────────────────────────────
# Anomaly Results
# ───────────────────────────────────────────────────────────────
def anomaly_results_tab(anomaly_results: dict) -> None:
    """Render anomaly detection results."""
    if not anomaly_results or anomaly_results.get("skipped"):
        st.info("Anomaly detection was skipped.")
        return

    st.markdown(section_header("🔍", "Anomaly Detection"), unsafe_allow_html=True)

    # KPI cards row
    total = anomaly_results.get("total_anomalies", 0)
    rate = anomaly_results.get("anomaly_rate", 0)
    cols = st.columns(3)
    with cols[0]:
        st.markdown(kpi_card("Total Anomalies", total, "⚠️", BRAND_WARNING), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(kpi_card("Anomaly Rate", f"{rate*100:.1f}%", "📊",
                             BRAND_DANGER if rate > 0.05 else BRAND_SUCCESS), unsafe_allow_html=True)
    with cols[2]:
        severity = "High" if rate > 0.1 else "Medium" if rate > 0.03 else "Low"
        s_color = BRAND_DANGER if severity == "High" else BRAND_WARNING if severity == "Medium" else BRAND_SUCCESS
        st.markdown(kpi_card("Severity", severity, "🎯", s_color), unsafe_allow_html=True)

    if anomaly_results.get("narrative"):
        st.markdown("")
        st.markdown(anomaly_results["narrative"])


# ───────────────────────────────────────────────────────────────
# Correlation Results
# ───────────────────────────────────────────────────────────────
def correlation_results_tab(correlation_results: dict) -> None:
    """Render correlation analysis results."""
    if not correlation_results or correlation_results.get("skipped"):
        st.info("Correlation analysis was skipped.")
        return

    st.markdown(section_header("🔗", "Correlation Analysis"), unsafe_allow_html=True)

    if correlation_results.get("narrative"):
        st.markdown(correlation_results["narrative"])

    import pandas as pd
    for label, key, color in [
        ("Top Positive Correlations", "top_positive", BRAND_SUCCESS),
        ("Top Negative Correlations", "top_negative", BRAND_DANGER),
    ]:
        pairs = correlation_results.get(key, [])
        if pairs:
            st.markdown(
                f'<div style="font-size:14px; font-weight:600; color:{color}; margin:16px 0 8px 0;">'
                f'{label}</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(pd.DataFrame(pairs), use_container_width=True, hide_index=True)


# ───────────────────────────────────────────────────────────────
# Full Report Tab
# ───────────────────────────────────────────────────────────────
def full_report_tab(state) -> None:
    """Render the Full Report / Executive Summary tab."""
    synth = state.report_synthesis
    if synth is None:
        st.info("Report not yet generated. Run the full analysis pipeline first.")
        return

    st.markdown(section_header("📋", "Executive Summary"), unsafe_allow_html=True)
    data_health_badge(synth.data_quality_score)

    st.markdown("")
    st.markdown(synth.executive_summary)

    if synth.recommendations:
        st.markdown(
            f'<div style="font-size:16px;font-weight:700;color:{BRAND_TEXT};margin:20px 0 12px 0;">'
            f'💡 Recommendations</div>',
            unsafe_allow_html=True,
        )
        for i, rec in enumerate(synth.recommendations, 1):
            priority_colors = [BRAND_DANGER, BRAND_WARNING, BRAND_PRIMARY, BRAND_SUCCESS, BRAND_ACCENT]
            pc = priority_colors[(i - 1) % len(priority_colors)]
            st.markdown(f"""
            <div style="
                background: #F8F9FC;
                border: 1px solid #E5E7EB;
                border-left: 4px solid {pc};
                border-radius: 0 12px 12px 0;
                padding: 14px 18px;
                margin-bottom: 8px;
                font-size: 14px;
                color: #1a1a2e;
                line-height: 1.6;
            ">
                <span style="color:{pc}; font-weight:700; margin-right:8px;">#{i}</span>
                {rec}
            </div>
            """, unsafe_allow_html=True)

    # Agent timings
    if state.agent_timings:
        with st.expander("⏱ Agent Performance", expanded=False):
            import pandas as pd
            timing_data = [
                {"Agent": k.replace("Agent", ""), "Duration": f"{v}s" if isinstance(v, (int, float)) else str(v)}
                for k, v in state.agent_timings.items()
                if not k.startswith("_")
            ]
            if timing_data:
                st.dataframe(pd.DataFrame(timing_data), use_container_width=True, hide_index=True)


# ───────────────────────────────────────────────────────────────
# PDF Download
# ───────────────────────────────────────────────────────────────
def pdf_download_section(state) -> None:
    """Show PDF download with styled container."""
    if state.pdf_ready and state.pdf_path and os.path.exists(state.pdf_path):
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(16,185,129,.1), rgba(5,150,105,.08));
            border: 1px solid rgba(16,185,129,.25);
            border-radius: 16px;
            padding: 20px 24px;
            margin: 16px 0;
            display: flex;
            align-items: center;
            gap: 16px;
        ">
            <div style="font-size:32px;">📄</div>
            <div>
                <div style="font-size:15px; font-weight:600; color:#1a1a2e;">
                    PDF Report Ready
                </div>
                <div style="font-size:13px; color:#6b7280;">
                    Download the full analysis as a formatted PDF document
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with open(state.pdf_path, "rb") as f:
            st.download_button(
                label="⬇️ Download PDF Report",
                data=f.read(),
                file_name=os.path.basename(state.pdf_path),
                mime="application/pdf",
                key="download_pdf_report",
            )
    elif state.report_synthesis is not None:
        st.info("📄 Generating PDF… it will be available shortly.")
