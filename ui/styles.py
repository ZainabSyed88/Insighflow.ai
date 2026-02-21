"""Centralized CSS theme for the AI Data Insights Analyst UI."""

# ── Brand colours ──────────────────────────────────────────────
BRAND_PRIMARY = "#6366F1"      # Indigo-500
BRAND_SECONDARY = "#8B5CF6"    # Violet-500
BRAND_ACCENT = "#22D3EE"       # Cyan-400
BRAND_SUCCESS = "#10B981"      # Emerald-500
BRAND_WARNING = "#F59E0B"      # Amber-500
BRAND_DANGER = "#EF4444"       # Red-500
BRAND_BG_DARK = "#F8F9FC"      # Light background
BRAND_BG_CARD = "#FFFFFF"      # White cards
BRAND_BG_SURFACE = "#F1F5F9"   # Slate-100
BRAND_TEXT = "#1a1a2e"         # Dark text
BRAND_TEXT_MUTED = "#6b7280"   # Gray-500
BRAND_BORDER = "#E5E7EB"       # Gray-200


def inject_global_css():
    """Inject the global CSS theme into Streamlit via st.markdown."""
    import streamlit as st

    st.markdown(f"""
    <style>
    /* ─── Google Fonts ─── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ─── Root / Global ─── */
    :root {{
        --brand-primary: {BRAND_PRIMARY};
        --brand-secondary: {BRAND_SECONDARY};
        --brand-accent: {BRAND_ACCENT};
        --brand-success: {BRAND_SUCCESS};
        --brand-warning: {BRAND_WARNING};
        --brand-danger: {BRAND_DANGER};
        --bg-dark: {BRAND_BG_DARK};
        --bg-card: {BRAND_BG_CARD};
        --bg-surface: {BRAND_BG_SURFACE};
        --text-primary: {BRAND_TEXT};
        --text-muted: {BRAND_TEXT_MUTED};
        --border-color: {BRAND_BORDER};
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;
        --shadow-sm: 0 1px 3px rgba(0,0,0,.08);
        --shadow-md: 0 4px 16px rgba(0,0,0,.08);
        --shadow-lg: 0 8px 32px rgba(0,0,0,.12);
        --transition: all .25s cubic-bezier(.4,0,.2,1);
    }}

    html, body, [data-testid="stAppViewContainer"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }}

    /* ─── Main container ─── */
    .stApp {{
        background: #FFFFFF !important;
    }}

    /* ─── Header bar (top bar) ─── */
    [data-testid="stHeader"] {{
        background: rgba(255,255,255,0.92) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-bottom: 1px solid {BRAND_BORDER} !important;
    }}
    header {{
        background: transparent !important;
    }}

    /* ─── Sidebar ─── */
    [data-testid="stSidebar"] {{
        background: #FFFFFF !important;
        border-right: 1px solid {BRAND_BORDER} !important;
    }}
    /* ─── Sidebar collapse/expand button ─── */
    [data-testid="stSidebarCollapsedControl"] {{
        background: rgba(255,255,255,0.9) !important;
        border: 1px solid {BRAND_BORDER} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stSidebarCollapsedControl"] button {{
        color: #1a1a2e !important;
    }}
    [data-testid="stSidebarCollapsedControl"] svg {{
        fill: #1a1a2e !important;
        stroke: #1a1a2e !important;
    }}
    [data-testid="collapsedControl"] {{
        background: rgba(255,255,255,0.9) !important;
        border: 1px solid {BRAND_BORDER} !important;
        border-radius: 8px !important;
        color: #1a1a2e !important;
    }}
    [data-testid="collapsedControl"] svg {{
        fill: #1a1a2e !important;
        stroke: #1a1a2e !important;
    }}
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li {{
        font-size: 14px;
        color: #4a4a68 !important;
    }}

    /* ─── Global text visibility ─── */
    p, span, li, label, .stMarkdown {{
        color: #4a4a68 !important;
    }}
    .stCaption, [data-testid="stCaptionContainer"] {{
        color: #6b7280 !important;
        font-size: 13px !important;
    }}

    /* ─── Headers ─── */
    h1 {{
        background: linear-gradient(135deg, var(--brand-primary), var(--brand-secondary)) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
    }}
    h2 {{
        color: #1a1a2e !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
    }}
    h3 {{
        color: #1a1a2e !important;
        font-weight: 600 !important;
    }}

    /* ─── Buttons ─── */
    .stButton > button {{
        background: linear-gradient(135deg, var(--brand-primary), var(--brand-secondary)) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-md) !important;
        padding: 10px 28px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        letter-spacing: 0.02em !important;
        transition: var(--transition) !important;
        box-shadow: 0 4px 14px rgba(99,102,241,.25) !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99,102,241,.4) !important;
    }}
    .stButton > button:active {{
        transform: translateY(0) !important;
    }}

    /* ─── Download button ─── */
    .stDownloadButton > button {{
        background: linear-gradient(135deg, var(--brand-success), #059669) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-md) !important;
        padding: 10px 28px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(16,185,129,.3) !important;
    }}
    .stDownloadButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(16,185,129,.45) !important;
    }}

    /* ─── Tabs ─── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: {BRAND_BG_SURFACE};
        border-radius: var(--radius-lg);
        padding: 4px;
        border: 1px solid var(--border-color);
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: var(--radius-md) !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
        color: #4a4a68 !important;
        transition: var(--transition) !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, var(--brand-primary), var(--brand-secondary)) !important;
        color: white !important;
        font-weight: 600 !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        display: none !important;
    }}
    .stTabs [data-baseweb="tab-border"] {{
        display: none !important;
    }}

    /* ─── Metrics ─── */
    [data-testid="stMetric"] {{
        background: #FFFFFF !important;
        border: 1px solid {BRAND_BORDER} !important;
        border-radius: var(--radius-md) !important;
        padding: 16px 20px !important;
        transition: var(--transition) !important;
    }}
    [data-testid="stMetric"]:hover {{
        border-color: var(--brand-primary) !important;
        box-shadow: 0 0 20px rgba(99,102,241,.1) !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: #6b7280 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }}
    [data-testid="stMetricValue"] {{
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #1a1a2e !important;
    }}

    /* ─── Expanders ─── */
    .streamlit-expanderHeader {{
        background: #FFFFFF !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-md) !important;
        font-weight: 600 !important;
        color: #1a1a2e !important;
    }}
    .streamlit-expanderContent {{
        background: {BRAND_BG_SURFACE} !important;
        border: 1px solid var(--border-color) !important;
        border-top: none !important;
        border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
    }}
    /* New Streamlit expander selectors */
    [data-testid="stExpander"] {{
        background: #FFFFFF !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-md) !important;
        overflow: hidden !important;
    }}
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] [data-testid="stExpanderToggleDetails"],
    [data-testid="stExpander"] details > summary {{
        background: #FFFFFF !important;
        color: #1a1a2e !important;
        font-weight: 600 !important;
        padding: 12px 16px !important;
    }}
    [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] details > summary:hover {{
        background: #F8F9FC !important;
    }}
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] [data-testid="stExpanderToggleDetails"] span,
    [data-testid="stExpander"] [data-testid="stExpanderToggleDetails"] p {{
        color: #1a1a2e !important;
    }}
    [data-testid="stExpander"] [data-testid="stExpanderDetails"],
    [data-testid="stExpander"] details > div {{
        background: #FAFBFC !important;
        color: #4a4a68 !important;
        border-top: 1px solid var(--border-color) !important;
        padding: 16px !important;
    }}
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] p,
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] span,
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] strong,
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] li {{
        color: #4a4a68 !important;
    }}
    /* Nested expanders (cleaning step cards) */
    [data-testid="stExpander"] [data-testid="stExpander"] {{
        background: #F8F9FC !important;
        border: 1px solid {BRAND_BORDER} !important;
        margin-bottom: 6px !important;
    }}
    [data-testid="stExpander"] [data-testid="stExpander"] summary,
    [data-testid="stExpander"] [data-testid="stExpander"] details > summary {{
        background: #F8F9FC !important;
        color: #4a4a68 !important;
    }}
    [data-testid="stExpander"] [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] [data-testid="stExpander"] details > summary:hover {{
        background: #F1F5F9 !important;
        color: #1a1a2e !important;
    }}
    /* Expander toggle arrow / icon */
    [data-testid="stExpander"] svg {{
        fill: #6b7280 !important;
        color: #6b7280 !important;
    }}

    /* ─── DataFrames ─── */
    [data-testid="stDataFrame"] {{
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-md) !important;
        overflow: hidden !important;
    }}

    /* ─── File uploader ─── */
    [data-testid="stFileUploader"] {{
        background: var(--bg-card) !important;
        border: 2px dashed var(--border-color) !important;
        border-radius: var(--radius-lg) !important;
        padding: 20px !important;
        transition: var(--transition) !important;
    }}
    [data-testid="stFileUploader"]:hover {{
        border-color: var(--brand-primary) !important;
        background: rgba(99,102,241,.06) !important;
    }}
    /* Browse button inside file uploader */
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploader"] [data-testid="baseButton-secondary"] {{
        background: linear-gradient(135deg, var(--brand-primary), var(--brand-secondary)) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        padding: 8px 24px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        cursor: pointer !important;
        box-shadow: 0 2px 8px rgba(99,102,241,.3) !important;
    }}
    [data-testid="stFileUploader"] button:hover,
    [data-testid="stFileUploader"] [data-testid="baseButton-secondary"]:hover {{
        box-shadow: 0 4px 14px rgba(99,102,241,.5) !important;
        transform: translateY(-1px) !important;
    }}
    /* File uploader label & drag text */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] div {{
        color: #4a4a68 !important;
    }}
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] {{
        color: #6b7280 !important;
    }}
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] span {{
        color: #6b7280 !important;
        font-size: 14px !important;
    }}
    /* File size limit text */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] small {{
        color: #9CA3AF !important;
        font-size: 12px !important;
    }}
    /* Uploaded file name pill */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {{
        background: rgba(99,102,241,.08) !important;
        border: 1px solid rgba(99,102,241,.2) !important;
        border-radius: var(--radius-sm) !important;
        color: #1a1a2e !important;
    }}
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span {{
        color: #1a1a2e !important;
    }}
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] {{
        color: #EF4444 !important;
    }}

    /* ─── Status widget ─── */
    [data-testid="stStatusWidget"] {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-md) !important;
    }}

    /* ─── Chat ─── */
    [data-testid="stChatMessage"] {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-lg) !important;
        padding: 16px !important;
        margin-bottom: 8px !important;
    }}
    .stChatInput {{
        border-color: var(--border-color) !important;
    }}
    .stChatInput input,
    .stChatInput textarea {{
        background: #FFFFFF !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-md) !important;
        color: #1a1a2e !important;
        font-size: 14px !important;
    }}
    .stChatInput input::placeholder,
    .stChatInput textarea::placeholder {{
        color: #9CA3AF !important;
    }}

    /* ─── Floating Chat FAB ─── */
    .chat-fab {{
        position: fixed;
        bottom: 32px;
        right: 32px;
        z-index: 9999;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg, {BRAND_PRIMARY}, {BRAND_SECONDARY});
        color: #FFFFFF;
        border: none;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 26px;
        box-shadow: 0 6px 24px rgba(99,102,241,.45);
        transition: all .3s cubic-bezier(.4,0,.2,1);
    }}
    .chat-fab:hover {{
        transform: scale(1.1) translateY(-2px);
        box-shadow: 0 8px 32px rgba(99,102,241,.6);
    }}
    .chat-fab .badge {{
        position: absolute;
        top: -2px;
        right: -2px;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: {BRAND_DANGER};
        font-size: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
    }}

    /* ─── Info / Warning / Error boxes ─── */
    .stAlert {{
        border-radius: var(--radius-md) !important;
        border: none !important;
    }}

    /* ─── Scrollbar ─── */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: transparent;
    }}
    ::-webkit-scrollbar-thumb {{
        background: var(--border-color);
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: var(--text-muted);
    }}

    /* ─── Animations ─── */
    @keyframes fadeInUp {{
        from {{ opacity:0; transform:translateY(16px); }}
        to   {{ opacity:1; transform:translateY(0); }}
    }}
    @keyframes pulse-glow {{
        0%,100% {{ box-shadow: 0 0 8px rgba(99,102,241,.3); }}
        50%     {{ box-shadow: 0 0 20px rgba(99,102,241,.6); }}
    }}
    .fade-in {{
        animation: fadeInUp .5s ease-out;
    }}

    /* ─── Divider override ─── */
    hr {{
        border-color: var(--border-color) !important;
        opacity: 0.5;
    }}

    /* ─── Toast / success messages ─── */
    .stSuccess {{
        background: rgba(16,185,129,.12) !important;
        color: var(--brand-success) !important;
        border: 1px solid rgba(16,185,129,.25) !important;
        border-radius: var(--radius-md) !important;
    }}
    </style>
    """, unsafe_allow_html=True)


# ── Reusable HTML fragments ────────────────────────────────────

def hero_banner(title: str, subtitle: str) -> str:
    """Full-width gradient hero banner."""
    sub_html = f"""<p style="
            color: {BRAND_TEXT_MUTED};
            font-size: 1.05rem;
            margin: 0;
            font-weight: 400;
        ">{subtitle}</p>""" if subtitle else ""
    return f"""
    <div style="
        background: linear-gradient(135deg, rgba(99,102,241,.15), rgba(139,92,246,.1), rgba(34,211,238,.08));
        border: 1px solid {BRAND_BORDER};
        border-radius: 20px;
        padding: 40px 36px;
        margin-bottom: 28px;
        text-align: center;
        animation: fadeInUp .6s ease-out;
    ">
        <h1 style="
            font-size: 2.4rem;
            margin: 0 0 8px 0;
            background: linear-gradient(135deg, {BRAND_PRIMARY}, {BRAND_ACCENT});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            letter-spacing: -0.03em;
        ">{title}</h1>
        {sub_html}
    </div>
    """


def section_header(icon: str, title: str, subtitle: str = "") -> str:
    """Styled section header with icon."""
    sub_html = f'<p style="color:{BRAND_TEXT_MUTED}; font-size:14px; margin:4px 0 0 0;">{subtitle}</p>' if subtitle else ""
    return f"""
    <div style="margin-bottom: 20px; animation: fadeInUp .4s ease-out;">
        <h2 style="
            font-size: 1.6rem;
            font-weight: 700;
            color: {BRAND_TEXT};
            margin: 0;
            display: flex;
            align-items: center;
            gap: 10px;
        ">
            <span style="font-size: 1.5rem;">{icon}</span>
            {title}
        </h2>
        {sub_html}
    </div>
    """


def kpi_card(label: str, value, icon: str = "📊", color: str = BRAND_PRIMARY, delta: str = "") -> str:
    """Glassmorphism KPI card."""
    delta_html = ""
    if delta:
        is_positive = not delta.startswith("-")
        d_color = BRAND_SUCCESS if is_positive else BRAND_DANGER
        d_icon = "↑" if is_positive else "↓"
        delta_html = f'<div style="font-size:12px; color:{d_color}; margin-top:4px; font-weight:500;">{d_icon} {delta}</div>'

    return f"""
    <div style="
        background: #FFFFFF;
        border: 1px solid {BRAND_BORDER};
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        transition: all .25s ease;
        position: relative;
        overflow: hidden;
    " onmouseover="this.style.borderColor='{color}'; this.style.boxShadow='0 0 24px {color}20'"
       onmouseout="this.style.borderColor='{BRAND_BORDER}'; this.style.boxShadow='none'">
        <div style="
            position: absolute;
            top: -20px;
            right: -20px;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: {color}0A;
        "></div>
        <div style="font-size: 24px; margin-bottom: 6px;">{icon}</div>
        <div style="font-size: 28px; font-weight: 700; color: #1a1a2e; line-height: 1.2;">
            {value}
        </div>
        <div style="font-size: 13px; color: #6b7280; text-transform: uppercase;
                    letter-spacing: 0.06em; margin-top: 6px; font-weight: 600;">
            {label}
        </div>
        {delta_html}
    </div>
    """


def status_badge(text: str, variant: str = "info") -> str:
    """Inline status badge. variant: info | success | warning | danger."""
    colours = {
        "info":    (BRAND_PRIMARY, f"{BRAND_PRIMARY}18"),
        "success": (BRAND_SUCCESS, f"{BRAND_SUCCESS}18"),
        "warning": (BRAND_WARNING, f"{BRAND_WARNING}18"),
        "danger":  (BRAND_DANGER,  f"{BRAND_DANGER}18"),
    }
    fg, bg = colours.get(variant, colours["info"])
    return (f'<span style="background:{bg}; color:{fg}; padding:4px 14px; border-radius:20px;'
            f' font-size:13px; font-weight:600; letter-spacing:.03em;">{text}</span>')


def glass_card(content_html: str, border_color: str = BRAND_BORDER) -> str:
    """Wrap arbitrary HTML in a glass card."""
    return f"""
    <div style="
        background: #FFFFFF;
        border: 1px solid {border_color};
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        animation: fadeInUp .45s ease-out;
    ">
        {content_html}
    </div>
    """


def pipeline_stepper(stages: list, current_idx: int) -> str:
    """Vertical pipeline stepper for sidebar.

    stages: list of (name, icon) tuples.
    current_idx: 0-based index of the active stage (-1 = none).
    """
    items_html = ""
    for i, (name, icon) in enumerate(stages):
        if i < current_idx:
            # completed
            dot = f'<div style="width:28px;height:28px;border-radius:50%;background:{BRAND_SUCCESS};display:flex;align-items:center;justify-content:center;font-size:13px;color:#fff;flex-shrink:0;">✓</div>'
            label_color = BRAND_TEXT_MUTED
            line_color = BRAND_SUCCESS
        elif i == current_idx:
            # active
            dot = f'<div style="width:28px;height:28px;border-radius:50%;background:linear-gradient(135deg,{BRAND_PRIMARY},{BRAND_SECONDARY});display:flex;align-items:center;justify-content:center;font-size:13px;color:#fff;flex-shrink:0;animation:pulse-glow 2s ease infinite;">{icon}</div>'
            label_color = BRAND_TEXT
            line_color = BRAND_BORDER
        else:
            # pending
            dot = f'<div style="width:28px;height:28px;border-radius:50%;background:{BRAND_BG_CARD};border:2px solid {BRAND_BORDER};display:flex;align-items:center;justify-content:center;font-size:12px;color:{BRAND_TEXT_MUTED};flex-shrink:0;">{icon}</div>'
            label_color = BRAND_TEXT_MUTED
            line_color = BRAND_BORDER

        # connector line (skip for last)
        connector = ""
        if i < len(stages) - 1:
            connector = f'<div style="width:2px;height:20px;background:{line_color};margin-left:13px;"></div>'

        items_html += f"""
        <div style="display:flex; align-items:center; gap:12px;">
            {dot}
            <span style="font-size:13px; font-weight:{'600' if i == current_idx else '400'}; color:{label_color};">{name}</span>
        </div>
        {connector}
        """

    return f'<div style="padding:4px 0;">{items_html}</div>'
