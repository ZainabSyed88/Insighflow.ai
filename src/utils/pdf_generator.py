"""PDF report generator — Jinja2 + WeasyPrint (fully offline)."""

import os
import logging
import tempfile
from datetime import datetime
from typing import Optional

from src.graph.state import AppState, VIZ_DIR

logger = logging.getLogger(__name__)

# Directory for generated reports
REPORT_DIR = os.path.join("data", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "templates")


def _render_html(state: AppState) -> str:
    """Render the Jinja2 report template to an HTML string."""
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader(os.path.abspath(TEMPLATE_DIR)))
    template = env.get_template("report_template.html")

    # Convert Plotly HTML charts to static PNGs for PDF embedding
    chart_images = _export_chart_images(state)

    # Gather data for template
    context = {
        "filename": state.filename,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "executive_summary": state.report_synthesis.executive_summary if state.report_synthesis else "",
        "data_quality_score": state.report_synthesis.data_quality_score if state.report_synthesis else 0,
        "cleaning_result": state.cleaning_result,
        "verification_result": state.verification_result,
        "trend_results": state.trend_results,
        "anomaly_results": state.anomaly_results,
        "correlation_results": state.correlation_results,
        "insights_result": state.insights_result,
        "recommendations": state.report_synthesis.recommendations if state.report_synthesis else [],
        "chart_images": chart_images,
        "agent_timings": state.agent_timings,
    }

    return template.render(**context)


def _export_chart_images(state: AppState) -> list[dict]:
    """Export Plotly HTML charts to PNG images for PDF embedding.

    Returns list of dicts: [{"title": ..., "png_path": ...}, ...]
    """
    images = []
    for viz in state.visualizations:
        if not viz.html_path or not os.path.exists(viz.html_path):
            continue
        try:
            import plotly.io as pio

            # Read the HTML file, extract the figure, re-export as PNG
            # Simpler approach: if we have the fig objects stored we'd use them,
            # but since we only have HTML, we create a minimal PNG via kaleido
            # by re-reading the HTML as a go.Figure is not trivial.
            # Instead, generate a placeholder — the charts are best viewed in HTML.
            png_path = viz.html_path.replace(".html", ".png")
            if os.path.exists(png_path):
                images.append({"title": viz.title, "png_path": os.path.abspath(png_path)})
                continue

            # Try using kaleido with a simple re-creation approach
            # For the POC, embed the chart title as placeholder if PNG export fails
            images.append({"title": viz.title, "png_path": ""})
        except Exception as e:
            logger.warning(f"Chart image export failed for {viz.title}: {e}")
            images.append({"title": viz.title, "png_path": ""})
    return images


def generate_pdf(state: AppState) -> str:
    """Generate a styled PDF report and return the file path.

    Uses WeasyPrint for HTML → PDF conversion (fully offline).
    Falls back to raw HTML file if WeasyPrint is not installed.
    """
    html_content = _render_html(state)

    # Write HTML for debugging / fallback
    html_path = os.path.join(REPORT_DIR, f"report_{state.filename.replace('.', '_')}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Report HTML saved to {html_path}")

    # Attempt PDF conversion
    pdf_path = html_path.replace(".html", ".pdf")
    try:
        from weasyprint import HTML
        HTML(string=html_content, base_url=os.path.abspath(REPORT_DIR)).write_pdf(pdf_path)
        logger.info(f"PDF report saved to {pdf_path}")
        return pdf_path
    except ImportError:
        logger.warning("WeasyPrint not installed — returning HTML report instead")
        return html_path
    except Exception as e:
        logger.error(f"WeasyPrint PDF generation failed: {e}")
        return html_path
