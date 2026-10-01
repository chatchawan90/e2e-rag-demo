"""Evaluation, gold-label inspection, local metrics, and MCP setup panels."""
import sqlite3
import os
from datetime import datetime

import streamlit as st

from .config import get_settings
from .monitoring import recent_events, record_event


def monitor(home, kind, status, seconds, **values):
    if os.environ.get("ENVSEARCH_PUBLIC_DEMO") == "1":
        return
    try:
        record_event(home, kind, status, seconds, **values)
    except (OSError, sqlite3.Error):
        st.warning("This operation completed, but its monitoring record could not be saved.")


def gold_panel(index, show_pdf):
    from .gold_ui import gold_panel as render_gold
    return render_gold(index, show_pdf, get_settings().home, editable=os.environ.get("ENVSEARCH_PUBLIC_DEMO") != "1")


def evaluation_panel(*args, **kwargs):
    from .eval_ui import evaluation_panel as render_evaluation
    return render_evaluation(*args, monitor_fn=monitor, **kwargs)


def monitoring_panel(settings):
    st.subheader("Local activity")
    st.caption("Latest 500 operations from this web app. Prompts, answers, document text, and API keys are not logged. "
               "Answer tokens cover final generation; scope-check tokens appear separately in chat/search traces and per-question evaluation reports. "
               "Query expansion and provider retries are excluded.")
    with st.expander("Business outcomes to measure next"):
        st.write("These are not measured by the operational log yet. They need task outcomes or user feedback.")
        st.dataframe([
            {"Metric": "Verified task resolution", "Measure": "Reviewer-confirmed successful tasks / all attempted tasks"},
            {"Metric": "Time saved", "Measure": "Manual baseline time minus time to a verified result, including review/corrections"},
            {"Metric": "Escalation rate", "Measure": "Tasks requiring expert help / all attempted tasks; distinguish appropriate escalations"},
            {"Metric": "Cost per resolved task", "Measure": "All API, infrastructure, and review costs / verified resolved tasks"},
            {"Metric": "Material error rate", "Measure": "Audited answers with a consequential factual or jurisdiction error / audited answers"},
        ], hide_index=True)
        st.caption("Track feedback response rate as well as ratings; unanswered feedback is unknown, not success. "
                   "A citation click, no follow-up, or a high retrieval score does not establish task resolution.")
    if st.button("Refresh activity"):
        st.rerun()
    try:
        events = recent_events(settings.home)
    except (OSError, sqlite3.Error):
        st.warning("Local monitoring could not be read. Chat and document search remain available.")
        return
    if not events:
        st.info("Search, chat, upload a PDF, or run an evaluation to see activity here.")
        return
    a, b, c, d = st.columns(4)
    a.metric("Operations", len(events))
    b.metric("Errors", sum(e["status"] == "error" for e in events))
    c.metric("Mean duration", f"{sum(e['seconds'] for e in events) / len(events):.2f}s")
    d.metric("Answer tokens", sum(sum(e.get("usage", {}).values()) for e in events))
    table = [{"id": e["id"], "time": datetime.fromtimestamp(e["created"]).isoformat(timespec="seconds"),
              "scope": e.get("trace", {}).get("scope_decision"),
              **{key: e.get(key) for key in ("kind", "status", "seconds", "provider", "mode", "rerank", "returned", "error_type")}}
             for e in events]
    st.dataframe(table, hide_index=True, use_container_width=True)
    selected = st.selectbox("Inspect an operation", [e["id"] for e in events])
    st.json(next(e for e in events if e["id"] == selected))


def mcp_panel(settings):
    from .mcp_ui import mcp_panel as render_mcp
    return render_mcp(settings)
