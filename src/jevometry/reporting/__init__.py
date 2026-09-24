"""Offline reporting: a single JSON payload rendered to HTML and Markdown."""

from jevometry.reporting.html import render_html
from jevometry.reporting.markdown import render_markdown
from jevometry.reporting.model import ReportModel, build_report_model

__all__ = ["ReportModel", "build_report_model", "render_html", "render_markdown"]
