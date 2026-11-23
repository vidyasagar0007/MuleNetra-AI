"""
MuleNetra AI - UI View Integration & Smoke Tests
Validates that all Streamlit page views render without exceptions,
handle data frames safely, and properly bind to risk and graph engines.
"""

import pytest
from unittest.mock import MagicMock
import streamlit as st
import pandas as pd

class MockColumn:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def __getattr__(self, name):
        return MagicMock()

def mock_columns(spec, *args, **kwargs):
    count = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
    return [MockColumn() for _ in range(count)]

class MockContext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

@pytest.fixture(scope="module")
def mock_streamlit_environment():
    """Set up mocked Streamlit UI context."""
    st.columns = mock_columns
    st.container = lambda *args, **kwargs: MockContext()
    st.spinner = lambda *args, **kwargs: MockContext()

    for attr in [
        'markdown', 'write', 'metric', 'dataframe', 'plotly_chart',
        'selectbox', 'slider', 'button', 'radio', 'text_input',
        'info', 'warning', 'error', 'success', 'download_button'
    ]:
        setattr(st, attr, MagicMock())

    st.selectbox.side_effect = lambda label, opts, *args, **kwargs: opts[0] if opts else None
    
    def mock_slider(label, *args, **kwargs):
        if 'Amount Range' in label:
            return (0.0, 50000.0)
        return 0.50
    st.slider.side_effect = mock_slider
    
    st.radio.return_value = 'Global Top Active Network'
    st.text_input.return_value = ''
    st.button.return_value = False
    return st

@pytest.fixture(scope="module")
def app_context():
    """Build shared data and intelligence context."""
    from src.data_loader import get_or_create_dataset
    from src.risk_engine import RiskScoringEngine
    
    df = get_or_create_dataset()
    engine = RiskScoringEngine()
    engine.initialize(df)
    scored = engine.score_transactions(df)
    profiles = engine.compute_account_risk_profiles(scored)
    clusters = engine.graph_engine.get_suspicious_clusters()
    
    return {
        "raw_df": df,
        "scored_tx": scored,
        "profiles": profiles,
        "clusters": clusters,
        "risk_engine": engine,
        "graph_engine": engine.graph_engine
    }

def test_page_dashboard_render(mock_streamlit_environment, app_context):
    from pages.dashboard import render_dashboard
    render_dashboard(app_context)
    assert st.markdown.called

def test_page_detection_render(mock_streamlit_environment, app_context):
    from pages.detection import render_detection
    render_detection(app_context)
    assert st.dataframe.called

def test_page_network_analysis_render(mock_streamlit_environment, app_context):
    from pages.network_analysis import render_network_analysis
    render_network_analysis(app_context)
    assert st.plotly_chart.called

def test_page_account_investigation_render(mock_streamlit_environment, app_context):
    from pages.account_investigation import render_account_investigation
    render_account_investigation(app_context)
    assert st.markdown.called

def test_page_ai_assistant_render(mock_streamlit_environment, app_context):
    from pages.ai_assistant import render_ai_assistant
    render_ai_assistant(app_context)
    assert st.write.called or st.markdown.called

def test_page_reports_render(mock_streamlit_environment, app_context):
    from pages.reports import render_reports
    render_reports(app_context)
    assert st.download_button.called

def test_page_model_performance_render(mock_streamlit_environment, app_context):
    from pages.model_performance import render_model_performance
    render_model_performance(app_context)
    assert st.dataframe.called
