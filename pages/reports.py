"""
MuleTrace AI - Reports, Exports & Regulatory Compliance View
Bulk CSV extractions, PDF dossier repository, and automated SAR filing packages.
"""

import streamlit as st
import json
from pathlib import Path
import pandas as pd
from config import REPORTS_DIR
from src.report_generator import ReportGenerator
from src.demo_guide import render_demo_guide

def render_reports(context: dict):
    """Render the Reports and Exports center."""
    scored_tx = context["scored_tx"]
    profiles = context["profiles"]
    clusters = context["clusters"]
    
    # Header Banner
    st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.5rem; border-bottom: 1px solid #23334B; padding-bottom: 1.25rem;">
            <div>
                <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.35rem;">
                    <span style="font-size: 1.2rem;">📑</span>
                    <h1 style="font-size: 1.65rem; font-weight: 800; color: #F8FAFC; margin: 0; letter-spacing: -0.025em; font-family: 'Manrope', sans-serif;">
                        Reports & Compliance Export Center
                    </h1>
                </div>
                <p style="font-size: 0.85rem; color: #94A3B8; margin: 0; font-family: 'Inter', sans-serif;">
                    Export verified surveillance records, download PDF forensic dossiers, and format automated SAR filing packages.
                </p>
            </div>
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span class="badge badge-low" style="font-size: 0.72rem; padding: 0.35rem 0.75rem;">
                    ● AUDITED EXTRACTIONS
                </span>
                <span class="badge badge-medium" style="font-size: 0.72rem; padding: 0.35rem 0.75rem;">
                    FINCEN / FIU READY
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Optional Collapsible Guided Demo Walkthrough
    render_demo_guide("reports")

    # 1. Bulk CSV Downloads Row
    st.markdown("<h4 style='color: #F8FAFC; font-size: 1rem; margin-bottom: 0.75rem; font-family: \"Manrope\", sans-serif;'>Surveillance Telemetry CSV Exports</h4>", unsafe_allow_html=True)

    ecol1, ecol2, ecol3 = st.columns(3)

    with ecol1:
        st.markdown("""
            <div class="mt-card" style="border-top: 3px solid #4C8DFF;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h5 style="margin: 0; color: #4C8DFF; font-size: 0.95rem; font-family: 'Manrope', sans-serif;">Scored Transactions</h5>
                    <span class="badge badge-medium" style="font-size: 0.65rem;">CSV FORMAT</span>
                </div>
                <p style="font-size: 0.78rem; color: #94A3B8; margin: 0.4rem 0 0.9rem 0; font-family: 'Inter', sans-serif;">
                    Complete transaction stream with ML, DL, and Hybrid calibrated risk scores.
                </p>
        """, unsafe_allow_html=True)
        csv_tx = scored_tx.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Transactions CSV",
            data=csv_tx,
            file_name="MuleTrace_Scored_Transactions.csv",
            mime="text/csv",
            key="dl_tx_csv",
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with ecol2:
        st.markdown("""
            <div class="mt-card" style="border-top: 3px solid #9270FF;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h5 style="margin: 0; color: #9270FF; font-size: 0.95rem; font-family: 'Manrope', sans-serif;">Account Risk Profiles</h5>
                    <span class="badge badge-violet" style="font-size: 0.65rem;">CSV FORMAT</span>
                </div>
                <p style="font-size: 0.78rem; color: #94A3B8; margin: 0.4rem 0 0.9rem 0; font-family: 'Inter', sans-serif;">
                    Account-level risk indices, mule roles, counterparties, and triggered heuristics.
                </p>
        """, unsafe_allow_html=True)
        csv_acc = profiles.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Profiles CSV",
            data=csv_acc,
            file_name="MuleTrace_Account_Risk_Profiles.csv",
            mime="text/csv",
            key="dl_acc_csv",
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with ecol3:
        st.markdown("""
            <div class="mt-card" style="border-top: 3px solid #F5B942;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h5 style="margin: 0; color: #F59E0B; font-size: 0.95rem; font-family: 'Manrope', sans-serif;">Suspicious Clusters</h5>
                    <span class="badge badge-warning" style="font-size: 0.65rem;">CSV FORMAT</span>
                </div>
                <p style="font-size: 0.78rem; color: #94A3B8; margin: 0.4rem 0 0.9rem 0; font-family: 'Inter', sans-serif;">
                    Flagged graph subgraphs, multi-node loops, and community structures.
                </p>
        """, unsafe_allow_html=True)
        clusters_df = pd.DataFrame(clusters)
        if len(clusters_df) > 0:
            csv_cl = clusters_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Clusters CSV",
                data=csv_cl,
                file_name="MuleTrace_Network_Clusters.csv",
                mime="text/csv",
                key="dl_cl_csv",
                use_container_width=True
            )
        else:
            st.button("No Clusters to Export", disabled=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 2. Existing Generated PDF Dossiers in repository
    st.markdown("<h4 style='color: #F8FAFC; font-size: 1rem; margin-bottom: 0.75rem; font-family: \"Manrope\", sans-serif;'>Generated PDF Forensic Dossiers Repository</h4>", unsafe_allow_html=True)

    pdf_files = list(REPORTS_DIR.glob("*.pdf"))
    if pdf_files:
        st.markdown("""<div class="mt-card" style="margin-bottom: 1.25rem;">""", unsafe_allow_html=True)
        pdf_rows = []
        for p in pdf_files:
            pdf_rows.append({
                "Document Name": p.name,
                "File Size": f"{p.stat().st_size / 1024:.1f} KB",
                "Last Modified": pd.to_datetime(p.stat().st_mtime, unit='s').strftime("%Y-%m-%d %H:%M UTC")
            })

        st.dataframe(pd.DataFrame(pdf_rows), use_container_width=True, hide_index=True)
        
        # Download selector for PDFs
        selected_pdf_name = st.selectbox("Select PDF Dossier to Download", [p.name for p in pdf_files])
        chosen_pdf_path = REPORTS_DIR / selected_pdf_name
        with open(chosen_pdf_path, "rb") as f:
            st.download_button(
                label=f"⬇️ Download Selected Dossier ({selected_pdf_name})",
                data=f.read(),
                file_name=selected_pdf_name,
                mime="application/pdf",
                key="btn_dl_existing_pdf"
            )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No PDF dossiers have been compiled yet. Visit the 'AI Assistant' to compile your first report.")

    # 3. Automated Regulatory SAR Package Generation
    st.markdown("""
        <div class="mt-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                <h4 style="margin: 0; font-size: 1rem; color: #F8FAFC; font-family: 'Manrope', sans-serif;">
                    Automated Regulatory SAR (Suspicious Activity Report) Generator
                </h4>
                <span class="badge badge-critical" style="font-size: 0.68rem;">FINCEN COMPLIANCE</span>
            </div>
            <p style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 1rem; font-family: 'Inter', sans-serif;">
                Standardized machine-readable FinCEN / FIU XML-JSON regulatory package for high-risk accounts.
            </p>
    """, unsafe_allow_html=True)

    sar_candidates = profiles[profiles["risk_tier"] == "CRITICAL"]["account_id"].tolist()
    if sar_candidates:
        sar_acc = st.selectbox("Select Target Account for SAR Package", sar_candidates)
        acc_info = profiles[profiles["account_id"] == sar_acc].iloc[0].to_dict()
        
        sar_payload = {
            "sar_header": {
                "system": "MuleTrace AI Regulatory Subsystem",
                "version": "1.0.0",
                "timestamp_utc": pd.Timestamp.now('UTC').isoformat(),
                "filing_institution": "FINANCIAL INTELLIGENCE UNIT (DEMO INSTITUTION)"
            },
            "subject_entity": {
                "account_id": sar_acc,
                "risk_tier": acc_info["risk_tier"],
                "risk_score": float(acc_info["risk_score"]),
                "mule_topology_role": acc_info["mule_role"],
                "cumulative_inflow_usd": float(acc_info["total_inflow"]),
                "cumulative_outflow_usd": float(acc_info["total_outflow"]),
                "in_degree": int(acc_info["in_degree"]),
                "out_degree": int(acc_info["out_degree"])
            },
            "suspicious_indicators": json.loads(acc_info["flagged_indicators"]) if isinstance(acc_info["flagged_indicators"], str) else acc_info["flagged_indicators"],
            "legal_disclaimer": "Academic research prototype sample export. Not an officially filed regulatory declaration. Risk scores indicate behavioral anomalies requiring human investigator review."
        }

        sar_json_str = json.dumps(sar_payload, indent=2)
        st.code(sar_json_str, language="json")

        st.download_button(
            label="⬇️ Download Regulatory SAR Package (JSON)",
            data=sar_json_str.encode("utf-8"),
            file_name=f"MuleTrace_SAR_{sar_acc}.json",
            mime="application/json",
            key="dl_sar_json"
        )
    else:
        st.info("No accounts currently qualify for Critical SAR threshold.")
    st.markdown("</div>", unsafe_allow_html=True)
