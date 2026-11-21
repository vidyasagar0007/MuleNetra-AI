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

def render_reports(context: dict):
    """Render the Reports and Exports view."""
    scored_tx = context["scored_tx"]
    profiles = context["profiles"]
    clusters = context["clusters"]
    
    st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
            <div>
                <h1 style="font-size: 1.6rem; font-weight: 800; color: #F8FAFC; margin: 0; letter-spacing: -0.02em;">
                    Forensic Reports & Data Export Center
                </h1>
                <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 0.2rem;">
                    Export verified surveillance records, download PDF forensic dossiers, and format automated SAR filing packages.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 1. Bulk CSV Downloads Row
    st.markdown("<h4 style='color: #F8FAFC; font-size: 1rem; margin-bottom: 0.75rem;'>Surveillance Telemetry CSV Exports</h4>", unsafe_allow_html=True)

    ecol1, ecol2, ecol3 = st.columns(3)

    with ecol1:
        st.markdown("""
            <div class="mt-card">
                <h5 style="margin: 0; color: #38BDF8; font-size: 0.95rem;">Scored Transactions</h5>
                <p style="font-size: 0.78rem; color: #94A3B8; margin: 0.3rem 0 0.8rem 0;">
                    Complete transaction stream with ML, DL, and Hybrid calibrated risk scores.
                </p>
        """, unsafe_allow_html=True)
        csv_tx = scored_tx.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Transactions CSV",
            data=csv_tx,
            file_name="MuleTrace_Scored_Transactions.csv",
            mime="text/csv",
            key="dl_tx_csv"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with ecol2:
        st.markdown("""
            <div class="mt-card">
                <h5 style="margin: 0; color: #818CF8; font-size: 0.95rem;">Account Risk Profiles</h5>
                <p style="font-size: 0.78rem; color: #94A3B8; margin: 0.3rem 0 0.8rem 0;">
                    Account-level risk indices, mule roles, counterparties, and triggered heuristics.
                </p>
        """, unsafe_allow_html=True)
        csv_acc = profiles.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Profiles CSV",
            data=csv_acc,
            file_name="MuleTrace_Account_Risk_Profiles.csv",
            mime="text/csv",
            key="dl_acc_csv"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with ecol3:
        st.markdown("""
            <div class="mt-card">
                <h5 style="margin: 0; color: #F59E0B; font-size: 0.95rem;">Suspicious Network Clusters</h5>
                <p style="font-size: 0.78rem; color: #94A3B8; margin: 0.3rem 0 0.8rem 0;">
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
                key="dl_cl_csv"
            )
        else:
            st.button("No Clusters to Export", disabled=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # 2. Existing Generated PDF Dossiers in repository
    st.markdown("<h4 style='color: #F8FAFC; font-size: 1rem; margin-bottom: 0.75rem;'>Generated PDF Forensic Dossiers Repository</h4>", unsafe_allow_html=True)

    pdf_files = list(REPORTS_DIR.glob("*.pdf"))
    if pdf_files:
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
                label=f"⬇️ Download {selected_pdf_name}",
                data=f.read(),
                file_name=selected_pdf_name,
                mime="application/pdf",
                key="btn_dl_existing_pdf"
            )
    else:
        st.info("No PDF dossiers have been compiled yet. Visit the 'GenAI Assistant' to compile your first report.")

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # 3. Automated Regulatory SAR Package Generation
    st.markdown("""
        <div style="background-color: #131D31; border: 1px solid #1E293B; border-radius: 8px; padding: 1.25rem;">
            <h4 style="margin: 0 0 0.4rem 0; font-size: 1rem; color: #F8FAFC;">Automated Regulatory SAR (Suspicious Activity Report) Generator</h4>
            <p style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 1rem;">
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
            "legal_disclaimer": "Research prototype generated payload. Not an official legal declaration."
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
