"""
MuleTrace AI - GenAI Investigation Assistant View
Generates natural language forensic briefs, hypothesis formulations,
and instant courtroom-grade PDF dossiers.
"""

import streamlit as st
import json
from pathlib import Path
import pandas as pd
from src.genai_assistant import GenAIAssistant
from src.report_generator import ReportGenerator
from src.database import get_db, InvestigationCase
from datetime import datetime

def render_ai_assistant(context: dict):
    """Render the GenAI Investigation Assistant module."""
    scored_tx = context["scored_tx"]
    profiles = context["profiles"]
    graph_engine = context["graph_engine"]
    
    st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
            <div>
                <h1 style="font-size: 1.6rem; font-weight: 800; color: #F8FAFC; margin: 0; letter-spacing: -0.02em;">
                    GenAI Forensic Investigation Assistant
                </h1>
                <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 0.2rem;">
                    Automated, evidence-grounded intelligence briefing, hypothesis formulation, and SAR dossier compilation.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 1. Target Account & LLM Provider Setup
    pcol1, pcol2 = st.columns([2, 1])

    with pcol1:
        high_risk_accs = profiles[profiles["risk_score"] >= 0.35]["account_id"].tolist()
        if not high_risk_accs:
            high_risk_accs = profiles["account_id"].tolist()
            
        selected_account = st.selectbox(
            "Select Account for AI Forensic Audit",
            high_risk_accs,
            format_func=lambda acc: f"{acc} — Tier: {profiles[profiles['account_id']==acc]['risk_tier'].values[0]} ({profiles[profiles['account_id']==acc]['mule_role'].values[0]})"
        )

    with pcol2:
        provider_choice = st.selectbox(
            "GenAI Engine / Provider",
            ["Deterministic Offline Engine", "Local Ollama (Llama 3.2)", "Gemini API"],
            index=0
        )
        provider_key = "offline" if "Offline" in provider_choice else "ollama" if "Ollama" in provider_choice else "gemini"

    # Fetch data for account
    account_row = profiles[profiles["account_id"] == selected_account].iloc[0]
    profile_dict = account_row.to_dict()
    
    acc_tx = scored_tx[
        (scored_tx["sender_account"] == selected_account) | 
        (scored_tx["receiver_account"] == selected_account)
    ].sort_values(by="timestamp", ascending=False).reset_index(drop=True)

    g_meta = graph_engine.account_metrics.get(selected_account, {})

    # Trigger Generation
    assistant = GenAIAssistant(provider=provider_key)
    
    with st.spinner("Compiling multi-modal forensic evidence..."):
        brief = assistant.generate_investigation_brief(
            account_id=selected_account,
            account_meta=profile_dict,
            transactions=acc_tx.head(10).to_dict("records"),
            graph_data=g_meta
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 2. Forensic Brief Display Cards
    st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <span style="font-size: 0.82rem; color: #94A3B8;">AUDIT DOSSIER FOR <b style="color: #38BDF8;">{selected_account}</b></span>
            <span class="badge badge-medium" style="font-size: 0.72rem;">ENGINE: {brief.get('provider_used')}</span>
        </div>
    """, unsafe_allow_html=True)

    # Card 1: Executive Incident Summary
    st.markdown("""
        <div class="mt-card" style="border-left: 3px solid #38BDF8;">
            <h4 style="margin: 0 0 0.5rem 0; font-size: 1rem; color: #38BDF8;">1. Executive Incident Summary</h4>
    """, unsafe_allow_html=True)
    st.write(brief.get("executive_summary", ""))
    st.markdown("</div>", unsafe_allow_html=True)

    # Card 2 & 3: Observations and Hypotheses
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        st.markdown("""
            <div class="mt-card" style="border-left: 3px solid #6366F1; min-height: 220px;">
                <h4 style="margin: 0 0 0.5rem 0; font-size: 1rem; color: #818CF8;">2. Factual Forensic Observations</h4>
        """, unsafe_allow_html=True)
        st.markdown(brief.get("factual_observations", ""))
        st.markdown("</div>", unsafe_allow_html=True)

    with bcol2:
        st.markdown("""
            <div class="mt-card" style="border-left: 3px solid #A855F7; min-height: 220px;">
                <h4 style="margin: 0 0 0.5rem 0; font-size: 1rem; color: #C084FC;">3. Investigative Hypotheses & Topology</h4>
        """, unsafe_allow_html=True)
        st.markdown(brief.get("hypotheses", ""))
        st.markdown("</div>", unsafe_allow_html=True)

    # Card 4: Recommended Actions
    st.markdown("""
        <div class="mt-card" style="border-left: 3px solid #10B981;">
            <h4 style="margin: 0 0 0.5rem 0; font-size: 1rem; color: #34D399;">4. Recommended Compliance Actions (FIU Playbook)</h4>
    """, unsafe_allow_html=True)
    st.markdown(brief.get("recommended_actions", ""))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # 3. PDF Dossier Compilation & Download
    dcol1, dcol2 = st.columns([1.5, 1])
    with dcol1:
        st.markdown("""
            <div style="background-color: #131D31; border: 1px solid #1E293B; border-radius: 8px; padding: 1.25rem;">
                <h4 style="margin: 0 0 0.4rem 0; font-size: 1rem; color: #F8FAFC;">Courtroom & SAR-Ready PDF Dossier</h4>
                <p style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 1rem;">
                    Compile an official cryptographic forensic dossier complete with chain-of-custody metadata, tables, and officer sign-off lines.
                </p>
        """, unsafe_allow_html=True)

        rep_gen = ReportGenerator()
        if st.button("📄 Compile Official PDF Investigation Dossier", key="btn_pdf_gen"):
            with st.spinner("Compiling PDF with ReportLab..."):
                pdf_path = rep_gen.generate_account_pdf_report(
                    account_id=selected_account,
                    account_profile=profile_dict,
                    brief=brief,
                    transactions=acc_tx
                )
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()

                st.download_button(
                    label="⬇️ Download PDF Dossier",
                    data=pdf_bytes,
                    file_name=pdf_path.name,
                    mime="application/pdf",
                    key="btn_pdf_download"
                )
                st.success(f"Dossier successfully compiled: {pdf_path.name}")
        st.markdown("</div>", unsafe_allow_html=True)

    with dcol2:
        st.markdown("""
            <div style="background-color: #131D31; border: 1px solid #1E293B; border-radius: 8px; padding: 1.25rem;">
                <h4 style="margin: 0 0 0.4rem 0; font-size: 1rem; color: #F8FAFC;">Analyst Case Disposition</h4>
        """, unsafe_allow_html=True)
        
        case_status = st.selectbox(
            "Case Priority / Status",
            ["UNDER_INVESTIGATION", "ESCALATED_FOR_SAR", "CLOSED_BENIGN", "RESTRICTED_ACCOUNT"]
        )
        notes = st.text_area("Compliance Officer Case Notes", placeholder="Enter investigative observations...", height=80)
        
        if st.button("Save Case Disposition", key="btn_save_case"):
            db = get_db()
            try:
                case = InvestigationCase(
                    case_id=f"CASE_{selected_account}_{datetime.utcnow().strftime('%Y%m%d%H%M')}",
                    account_id=selected_account,
                    status=case_status,
                    priority="CRITICAL" if profile_dict["risk_tier"]=="CRITICAL" else "HIGH",
                    analyst_notes=notes,
                    ai_summary=brief.get("executive_summary", "")
                )
                db.add(case)
                db.commit()
                st.success("Case disposition saved to audit database.")
            except Exception as e:
                db.rollback()
                st.error(f"Error saving case: {e}")
            finally:
                db.close()
        st.markdown("</div>", unsafe_allow_html=True)
