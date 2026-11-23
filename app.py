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
    # Load Intelligence Context upfront
    context = get_initialized_intelligence()
    n_tx = len(context["scored_tx"])
    n_nodes = len(context["profiles"])

    # Sidebar Branding & Navigation
    with st.sidebar:
        # Logo SVG rendering with base64 encoding to prevent markdown parser leaks
        logo_path = ASSETS_DIR / "logo.svg"
        if logo_path.exists():
            with open(logo_path, "rb") as f:
                svg_b64 = base64.b64encode(f.read()).decode("utf-8")
            st.markdown(
                f'''<div style="margin-bottom: 0.75rem; padding-top: 0.25rem;">
                    <img src="data:image/svg+xml;base64,{svg_b64}" style="width: 100%; max-width: 260px; display: block;" alt="MuleTrace AI" />
                </div>''',
                unsafe_allow_html=True
            )
        else:
            st.markdown(f"### 🛡️ {APP_TITLE}")
            st.markdown(f"<p style='color: #94A3B8; font-size: 0.75rem;'>{APP_SUBTITLE}</p>", unsafe_allow_html=True)

        st.markdown(f"""
            <div style="font-size: 0.7rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 1.25rem; border-bottom: 1px solid #23334B; padding-bottom: 0.6rem; display: flex; justify-content: space-between; align-items: center;">
                <span>SURVEILLANCE SUITE</span>
                <span style="color: #4C8DFF; font-family: 'JetBrains Mono', monospace;">v{APP_VERSION}</span>
            </div>
        """, unsafe_allow_html=True)

        # Primary Navigation Selector with Polished Icons
        nav_options = [
            "📊 Executive Dashboard",
            "⚡ AI Detection Center",
            "🕸️ Network Intelligence",
            "🔍 Account Investigation",
            "🤖 AI Investigation Assistant",
            "📑 Reports & Exports",
            "📈 Model Performance"
        ]

        selected_page = st.radio(
            "INTELLIGENCE MODULES",
            nav_options,
            index=0,
            label_visibility="visible"
        )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Guided Defense Mode Toggle
        show_guide = st.toggle(
            "🎓 Guided Defense Mode",
            value=st.session_state.get("show_demo_guide", True),
            key="toggle_demo_guide",
            help="Displays examiner talking points and step-by-step guidance for each module."
        )
        st.session_state["show_demo_guide"] = show_guide

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # System Status & Telemetry Card
        st.markdown("""
            <div style="background-color: #121C2E; border: 1px solid #23334B; border-radius: 8px; padding: 0.95rem; font-size: 0.78rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="color: #94A3B8; text-transform: uppercase; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.05em;">Engine Health</span>
                    <span class="badge badge-low" style="font-size: 0.64rem; padding: 0.15rem 0.45rem;">ONLINE</span>
                </div>
                <div style="color: #27D69B; font-weight: 600; font-size: 0.8rem; margin-bottom: 0.4rem;">
                    ● Active Surveillance • Zero-Leakage
                </div>
                <div style="color: #64748B; border-top: 1px solid #162238; padding-top: 0.45rem; margin-top: 0.45rem; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; line-height: 1.5;">
                    <span style="color: #94A3B8;">ML:</span> XGBoost v3.2 Champion<br/>
                    <span style="color: #94A3B8;">DL:</span> PyTorch Compact RiskNet<br/>
                    <span style="color: #94A3B8;">Graph:</span> NetworkX Directed Topology
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # Dataset & Environment Telemetry Card (Dynamic, grounded statistics)
        st.markdown(f"""
            <div style="background-color: #0D1423; border: 1px solid #23334B; border-radius: 6px; padding: 0.75rem 0.85rem; font-size: 0.72rem; color: #64748B;">
                <div style="color: #94A3B8; font-weight: 600; margin-bottom: 0.25rem;">ENVIRONMENT TELEMETRY</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #94A3B8;">
                    • {n_tx:,} Synthetic Txns<br/>
                    • {n_nodes:,} Banking Node Graph<br/>
                    • SQLite Local DB Persistence
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Dataset Reset Action
        if st.button("🔄 Regenerate Dataset & Retrain", key="btn_reset_data", use_container_width=True):
            with st.spinner("Regenerating AML patterns and retraining models..."):
                st.cache_resource.clear()
                get_or_create_dataset(force_regenerate=True)
                st.rerun()

        st.markdown(f"""
            <div style="text-align: center; margin-top: 1.2rem; font-size: 0.68rem; color: #475569; font-family: 'Inter', sans-serif;">
                {ACADEMIC_TAG} • Institutional Build<br/>
                For Forensic Research & Educational Defense
            </div>
        """, unsafe_allow_html=True)

    # Route Page cleanly based on substring
    if "Executive Dashboard" in selected_page:
        render_dashboard(context)
    elif "Detection" in selected_page:
        render_detection(context)
    elif "Network Intelligence" in selected_page:
        render_network_analysis(context)
    elif "Account Investigation" in selected_page:
        render_account_investigation(context)
    elif "Assistant" in selected_page:
        render_ai_assistant(context)
    elif "Reports" in selected_page:
        render_reports(context)
    elif "Model Performance" in selected_page:
        render_model_performance(context)

if __name__ == "__main__":
    main()
