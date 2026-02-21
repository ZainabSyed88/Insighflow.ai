"""Common Plotly figure factory functions shared across agents."""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# ── Power BI-inspired theme ──────────────────────────────────────────
# Clean white-card look, Segoe UI font, subtle grid, rounded feel.
_PBI_FONT = "Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"

DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#111827",
    font=dict(family=_PBI_FONT, color="#e2e8f0", size=13),
    title_font=dict(family=_PBI_FONT, size=20, color="#f8fafc"),
    margin=dict(l=60, r=30, t=75, b=55),
    hoverlabel=dict(
        bgcolor="#1e293b",
        font_size=13,
        font_family=_PBI_FONT,
        font_color="#f1f5f9",
        bordercolor="#334155",
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="center", x=0.5,
        font=dict(size=12),
        bgcolor="rgba(0,0,0,0)",
    ),
)

# Curated palettes (Power BI-esque)
PBI_CATEGORICAL = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA",
                   "#FFA15A", "#19D3F3", "#FF6692", "#B6E880"]
PBI_SEQUENTIAL = "Viridis"
PBI_DIVERGING = "RdYlBu"


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Apply the Power BI-inspired dark theme to any Plotly figure."""
    fig.update_layout(**DARK_LAYOUT)
    fig.update_xaxes(
        showgrid=True, gridcolor="rgba(255,255,255,0.06)",
        zeroline=False, showline=True, linewidth=1, linecolor="#334155",
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="rgba(255,255,255,0.06)",
        zeroline=False, showline=True, linewidth=1, linecolor="#334155",
    )
    return fig


def create_line_chart(
    df: pd.DataFrame,
    x: str,
    y_cols: list[str],
    title: str = "Line Chart",
) -> go.Figure:
    """Create a multi-line chart with Power BI styling."""
    fig = go.Figure()
    for i, col in enumerate(y_cols):
        fig.add_trace(go.Scatter(
            x=df[x] if x in df.columns else df.index,
            y=df[col],
            mode="lines",
            name=col,
            line=dict(color=PBI_CATEGORICAL[i % len(PBI_CATEGORICAL)], width=2.5),
            hovertemplate=f"{col}: %{{y:,.2f}}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        hovermode="x unified",
        **DARK_LAYOUT,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                     showline=True, linewidth=1, linecolor="#334155")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                     showline=True, linewidth=1, linecolor="#334155")
    return fig


def create_moving_average_traces(
    df: pd.DataFrame,
    col: str,
    x_col: Optional[str] = None,
    windows: list[int] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Create a figure with the original series and rolling moving averages."""
    if windows is None:
        windows = [7, 30]
    x = df[x_col] if x_col and x_col in df.columns else df.index
    is_datetime = x_col and x_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[x_col])
    x_label = x_col if x_col else "Index"

    fig = go.Figure()

    # Original series — visible but subtle so MAs stand out
    fig.add_trace(go.Scatter(
        x=x, y=df[col], mode="lines", name=f"{col} (raw)",
        opacity=0.35,
        line=dict(color="#636EFA", width=1),
        hovertemplate=f"{col}: %{{y:,.2f}}<extra>raw</extra>",
    ))

    # Moving average lines — bold and distinct
    ma_colors = ["#EF553B", "#00CC96", "#FFA15A"]
    ma_dashes = ["solid", "dash", "dot"]
    for i, w in enumerate(windows):
        if len(df) >= w:
            ma = df[col].rolling(window=w, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=x, y=ma, mode="lines",
                name=f"{w}-period MA",
                line=dict(
                    color=ma_colors[i % len(ma_colors)],
                    width=2.5,
                    dash=ma_dashes[i % len(ma_dashes)],
                ),
                hovertemplate=f"{w}-MA: %{{y:,.2f}}<extra></extra>",
            ))

    # Layout: gridlines, axis labels, legend, sizing
    layout_kwargs = dict(
        title=dict(
            text=title or f"{col} — Moving Averages",
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title=x_label,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            gridwidth=1,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="rgba(255,255,255,0.3)",
            spikethickness=1,
        ),
        yaxis=dict(
            title=col,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.15)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            font=dict(size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
        height=520,
        margin=dict(l=60, r=30, t=80, b=60),
        **DARK_LAYOUT,
    )

    # Range slider & buttons for datetime axes
    if is_datetime:
        layout_kwargs["xaxis"]["rangeslider"] = dict(visible=True, thickness=0.06)
        layout_kwargs["xaxis"]["rangeselector"] = dict(
            buttons=[
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            font=dict(size=11),
        )

    fig.update_layout(**layout_kwargs)
    return fig


def create_box_plot(
    df: pd.DataFrame,
    cols: list[str],
    title: str = "Box Plot",
    highlight_mask: pd.Series | None = None,
) -> go.Figure:
    """Create Power BI-styled box plots."""
    fig = go.Figure()
    for i, col in enumerate(cols):
        fig.add_trace(go.Box(
            y=df[col], name=col,
            marker_color=PBI_CATEGORICAL[i % len(PBI_CATEGORICAL)],
            line_color=PBI_CATEGORICAL[i % len(PBI_CATEGORICAL)],
            boxpoints="outliers",
            fillcolor=f"rgba({int(PBI_CATEGORICAL[i % len(PBI_CATEGORICAL)][1:3],16)},"
                      f"{int(PBI_CATEGORICAL[i % len(PBI_CATEGORICAL)][3:5],16)},"
                      f"{int(PBI_CATEGORICAL[i % len(PBI_CATEGORICAL)][5:7],16)},0.3)",
        ))
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        **DARK_LAYOUT,
    )
    fig.update_xaxes(showgrid=False, showline=True, linewidth=1, linecolor="#334155")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                     showline=True, linewidth=1, linecolor="#334155")
    return fig


def create_scatter_with_outliers(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    outlier_mask: pd.Series,
    title: str = "Scatter — Outliers Highlighted",
) -> go.Figure:
    """Scatter plot with Power BI styling — normal in blue, outliers in red."""
    fig = go.Figure()
    normal = df[~outlier_mask]
    outliers = df[outlier_mask]
    fig.add_trace(go.Scatter(
        x=normal[x_col], y=normal[y_col], mode="markers",
        name="Normal",
        marker=dict(color="#636EFA", size=6, opacity=0.7,
                    line=dict(width=0.5, color="#334155")),
        hovertemplate=f"{x_col}: %{{x:,.2f}}<br>{y_col}: %{{y:,.2f}}<extra>Normal</extra>",
    ))
    if len(outliers) > 0:
        fig.add_trace(go.Scatter(
            x=outliers[x_col], y=outliers[y_col], mode="markers",
            name="Outlier",
            marker=dict(color="#EF553B", size=9, symbol="diamond",
                        line=dict(width=1, color="#fff")),
            hovertemplate=f"{x_col}: %{{x:,.2f}}<br>{y_col}: %{{y:,.2f}}<extra>Outlier</extra>",
        ))
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        hovermode="closest",
        **DARK_LAYOUT,
    )
    fig.update_xaxes(title=x_col, showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                     showline=True, linewidth=1, linecolor="#334155")
    fig.update_yaxes(title=y_col, showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                     showline=True, linewidth=1, linecolor="#334155")
    return fig


def create_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
    colorscale: str = "RdYlBu",
) -> go.Figure:
    """Create a Power BI-styled heatmap from a correlation matrix."""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale=colorscale,
        zmin=-1, zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=11, family=_PBI_FONT),
        hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
        colorbar=dict(title="r", thickness=15, len=0.7),
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        height=600,
        **DARK_LAYOUT,
    )
    fig.update_xaxes(side="bottom", tickangle=45)
    return fig


def create_scatter_pair(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    title: str | None = None,
) -> go.Figure:
    """Power BI-styled scatter for a correlated pair."""
    fig = px.scatter(
        df, x=col_a, y=col_b,
        color_continuous_scale="Plasma",
        title=title or f"{col_a} vs {col_b}",
        opacity=0.7,
    )
    fig.update_traces(
        marker=dict(size=7, line=dict(width=0.5, color="#334155")),
        hovertemplate=f"{col_a}: %{{x:,.2f}}<br>{col_b}: %{{y:,.2f}}<extra></extra>",
    )
    fig.update_layout(
        title=dict(text=title or f"{col_a} vs {col_b}", x=0.5, xanchor="center"),
        **DARK_LAYOUT,
    )
    fig.update_xaxes(title=col_a, showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                     showline=True, linewidth=1, linecolor="#334155")
    fig.update_yaxes(title=col_b, showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                     showline=True, linewidth=1, linecolor="#334155")
    return fig


def save_figure_html(fig: go.Figure, path: str) -> str:
    """Save a Plotly figure to HTML and return the path."""
    fig.write_html(path)
    logger.info(f"Saved chart to {path}")
    return path


def save_figure_png(fig: go.Figure, path: str, width: int = 900, height: int = 500) -> str:
    """Save a Plotly figure as static PNG (requires kaleido)."""
    fig.write_image(path, width=width, height=height, scale=2)
    logger.info(f"Saved chart image to {path}")
    return path
