"""
MuleTrace AI - Main Application Entrypoint
Hybrid AI-Powered Money Mule Network Detection and Financial Crime Investigation Platform.
"""

import sys
import os
import base64
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
import pandas as pd

from config import ASSETS_DIR, APP_TITLE, APP_SUBTITLE, APP_VERSION, ACADEMIC_TAG
from src.data_loader import get_or_create_dataset
from src.risk_engine import RiskScoringEngine
from src.graph_engine import GraphEngine

# Import Page Views
from pages.dashboard import render_dashboard
from pages.detection import render_detection
from pages.network_analysis import render_network_analysis
from pages.account_investigation import render_account_investigation
from pages.ai_assistant import render_ai_assistant
from pages.reports import render_reports
from pages.model_performance import render_model_performance

# Set Streamlit Page Configuration
st.set_page_config(
    page_title=f"{APP_TITLE} | Financial Crime Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and inject custom CSS
def load_css():
    css_path = ASSETS_DIR / "styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Session State & Cache Initialization
@st.cache_resource(show_spinner="Bootstrapping AI models and graph intelligence...")
def get_initialized_intelligence():
    """Load transaction data, initialize and execute scoring and graph engines."""
    df = get_or_create_dataset()
    engine = RiskScoringEngine()
    engine.initialize(df)
    scored_tx = engine.score_transactions(df)
    profiles = engine.compute_account_risk_profiles(scored_tx)
    clusters = engine.graph_engine.get_suspicious_clusters()
    return {
        "raw_df": df,
        "scored_tx": scored_tx,
        "profiles": profiles,
        "clusters": clusters,
        "risk_engine": engine,
        "graph_engine": engine.graph_engine
    }

# Main Application Controller
def main():
    # Sidebar Branding & Navigation
    with st.sidebar:
        # Logo SVG rendering
        logo_path = ASSETS_DIR / "logo.svg"
        if logo_path.exists():
            with open(logo_path, "rb") as f:
                svg_b64 = base64.b64encode(f.read()).decode("utf-8")
            st.markdown(f'<div style="margin-bottom: 0.5rem;"><img src="data:image/svg+xml;base64,{svg_b64}" style="width: 100%; max-width: 250px; display: block;" alt="MuleTrace AI" /></div>', unsafe_allow_html=True)
        else:
            st.markdown(f"### 🛡️ {APP_TITLE}")

        st.markdown(f"""
            <div style="font-size: 0.72rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 1.2rem; border-bottom: 1px solid #1E293B; padding-bottom: 0.5rem;">
                {ACADEMIC_TAG} • v{APP_VERSION}
            </div>
        """, unsafe_allow_html=True)

        # Primary Navigation Selector
        nav_options = [
            "Executive Dashboard",
            "AI Detection Engine",
            "Mule Network Intelligence",
            "Account Investigation",
            "GenAI Assistant",
            "Reports & Exports",
            "Model Performance"
        ]

        # Use radio for clean navigation
        selected_page = st.radio(
            "SURVEILLANCE MODULES",
            nav_options,
            index=0,
            label_visibility="visible"
        )

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Telemetry & System Status Sidebar Box
        st.markdown("""
            <div style="background-color: #0A0E17; border: 1px solid #1E293B; border-radius: 6px; padding: 0.85rem; font-size: 0.78rem;">
                <div style="color: #94A3B8; text-transform: uppercase; font-size: 0.68rem; margin-bottom: 0.4rem; font-weight: 600;">Engine Health</div>
                <div style="color: #10B981; font-weight: 600;">● Online • Zero-Leakage Mode</div>
                <div style="color: #64748B; margin-top: 0.4rem; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;">
                    XGBoost v3.2<br/>
                    PyTorch v2.15 CPU<br/>
                    NetworkX DiGraph
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # Dataset Reset Action
        if st.button("🔄 Regenerate Dataset & Retrain", key="btn_reset_data"):
            with st.spinner("Regenerating AML patterns and retraining models..."):
                st.cache_resource.clear()
                get_or_create_dataset(force_regenerate=True)
                st.rerun()

    # Load Intelligence Context
    context = get_initialized_intelligence()

    # Route Page
    if selected_page == "Executive Dashboard":
        render_dashboard(context)
    elif selected_page == "AI Detection Engine":
        render_detection(context)
    elif selected_page == "Mule Network Intelligence":
        render_network_analysis(context)
    elif selected_page == "Account Investigation":
        render_account_investigation(context)
    elif selected_page == "GenAI Assistant":
        render_ai_assistant(context)
    elif selected_page == "Reports & Exports":
        render_reports(context)
    elif selected_page == "Model Performance":
        render_model_performance(context)

if __name__ == "__main__":
    main()
