"""
MuleTrace AI - Deep Account Investigation Module
Deep-dive forensic interrogation for individual target accounts:
360-degree profile, counterparty ego-networks, risk waterfall, and transaction audits.
"""

import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.explainability import ExplainabilityEngine
from src.demo_guide import render_demo_guide

def render_account_investigation(context: dict):
    """Render the Deep Account Forensic Investigation view."""
    scored_tx = context["scored_tx"]
    profiles = context["profiles"]
    graph_engine = context["graph_engine"]
    
    # Header Banner
    st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.5rem; border-bottom: 1px solid #23334B; padding-bottom: 1.25rem;">
            <div>
                <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.35rem;">
                    <span style="font-size: 1.2rem;">🔍</span>
                    <h1 style="font-size: 1.65rem; font-weight: 800; color: #F8FAFC; margin: 0; letter-spacing: -0.025em; font-family: 'Manrope', sans-serif;">
                        Account Forensic Investigation
                    </h1>
                </div>
                <p style="font-size: 0.85rem; color: #94A3B8; margin: 0; font-family: 'Inter', sans-serif;">
                    Entity 360-degree risk attribution, counterparty ego-networks, and chronological transaction ledger.
                </p>
            </div>
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span class="badge badge-low" style="font-size: 0.72rem; padding: 0.35rem 0.75rem;">
                    ● ENTITY AUDIT TRAIL
                </span>
                <span class="badge badge-medium" style="font-size: 0.72rem; padding: 0.35rem 0.75rem;">
                    EXPLAINABLE AI
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Optional Collapsible Guided Demo Walkthrough
    render_demo_guide("account_investigation")

    # 1. Target Account Selector
    all_account_ids = profiles["account_id"].tolist()
    default_acc = all_account_ids[0] if all_account_ids else ""
    
    st.markdown("""
        <div class="mt-card" style="padding: 1rem 1.25rem; margin-bottom: 1.25rem;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.65rem;">
                Target Entity Selection
            </div>
    """, unsafe_allow_html=True)

    scol1, scol2 = st.columns([2.2, 1])
    with scol1:
        selected_account = st.selectbox(
            "Select Target Account (Ranked by Risk Index)",
            all_account_ids,
            format_func=lambda acc: f"{acc} — Tier: {profiles[profiles['account_id']==acc]['risk_tier'].values[0]} (Risk: {profiles[profiles['account_id']==acc]['risk_score'].values[0]:.2f})",
            index=0
        )
    with scol2:
        quick_search = st.text_input("Or Enter Specific Account ID", "").strip().upper()
        if quick_search and quick_search in all_account_ids:
            selected_account = quick_search

    st.markdown("</div>", unsafe_allow_html=True)

    # Retrieve account profile
    account_row = profiles[profiles["account_id"] == selected_account].iloc[0]
    profile_dict = account_row.to_dict()
    
    # Retrieve associated transactions
    acc_tx = scored_tx[
        (scored_tx["sender_account"] == selected_account) | 
        (scored_tx["receiver_account"] == selected_account)
    ].sort_values(by="timestamp", ascending=False).reset_index(drop=True)

    # 2. Forensic Profile Banner
    tier = profile_dict["risk_tier"]
    score = profile_dict["risk_score"]
    role = profile_dict["mule_role"]
    inflow = profile_dict["total_inflow"]
    outflow = profile_dict["total_outflow"]
    
    tier_badge_class = f"badge-{tier.lower()}"
    accent_bar_color = "#FF5B6E" if tier=="CRITICAL" else "#F5B942" if tier=="HIGH" else "#4C8DFF"

    st.markdown(f"""
        <div class="mt-card" style="border-left: 4px solid {accent_bar_color}; margin-bottom: 1.25rem;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <span style="font-size: 0.72rem; text-transform: uppercase; color: #94A3B8; font-weight: 700; letter-spacing: 0.06em;">TARGET ENTITY DOSSIER</span>
                    <h2 style="font-family: 'JetBrains Mono', monospace; font-size: 1.9rem; color: #4C8DFF; margin: 0.25rem 0 0.45rem 0; letter-spacing: -0.02em;">
                        {selected_account}
                    </h2>
                    <div style="display: flex; align-items: center; gap: 0.6rem;">
                        <span class="badge {tier_badge_class}" style="font-size: 0.76rem; padding: 0.28rem 0.65rem;">{tier} RISK TIER</span>
                        <span style="font-size: 0.82rem; color: #E2E8F0; font-family: 'Inter', sans-serif;">
                            Topology Role: <b style="color: #9270FF;">{role}</b>
                        </span>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Calibrated Risk Index</div>
                    <div style="font-size: 2.3rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: {'#FF5B6E' if score>=0.75 else '#F5B942' if score>=0.55 else '#27D69B'}; line-height: 1.1; margin-top: 0.2rem;">
                        {score:.3f}
                    </div>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1.2rem; padding-top: 1rem; border-top: 1px solid #162238;">
                <div>
                    <span style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Cumulative Inflow</span>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #27D69B; font-family: 'JetBrains Mono', monospace;">${inflow:,.2f}</div>
                </div>
                <div>
                    <span style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Cumulative Outflow</span>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #FF5B6E; font-family: 'JetBrains Mono', monospace;">${outflow:,.2f}</div>
                </div>
                <div>
                    <span style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Direct Counterparties</span>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #F8FAFC; font-family: 'JetBrains Mono', monospace;">{profile_dict['in_degree']} In / {profile_dict['out_degree']} Out</div>
                </div>
                <div>
                    <span style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Retention Ratio</span>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #4C8DFF; font-family: 'JetBrains Mono', monospace;">{(1 - min(outflow/max(inflow, 1.0), 1.0))*100:.1f}%</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 3. Two columns: Explainability Waterfall & Ego-Network
    col_w, col_g = st.columns([1.1, 0.9])

    with col_w:
        st.markdown("""
            <div class="mt-card" style="margin-bottom: 1rem;">
                <h5 style="color: #F8FAFC; margin: 0 0 0.5rem 0; font-family: 'Manrope', sans-serif; font-size: 0.95rem;">
                    Risk Attribution Waterfall Breakdown
                </h5>
        """, unsafe_allow_html=True)
        exp_engine = ExplainabilityEngine()
        fig_waterfall = exp_engine.plot_risk_waterfall(profile_dict)
        st.plotly_chart(fig_waterfall, use_container_width=True)

        # Triggered Indicators
        raw_inds = profile_dict.get("flagged_indicators", "[]")
        try:
            indicators = json.loads(raw_inds) if isinstance(raw_inds, str) else raw_inds
        except Exception:
            indicators = []

        st.markdown("<h5 style='color: #F8FAFC; margin: 0.6rem 0 0.4rem 0; font-family: \"Manrope\", sans-serif; font-size: 0.9rem;'>Triggered Forensic Rules & Indicators</h5>", unsafe_allow_html=True)
        if indicators:
            for ind in indicators:
                st.markdown(f"""
                    <div style="background-color: rgba(255, 91, 110, 0.08); border-left: 3px solid #FF5B6E; padding: 0.5rem 0.75rem; margin-bottom: 0.4rem; border-radius: 0 4px 4px 0; font-size: 0.82rem; color: #F8FAFC; font-family: 'Inter', sans-serif;">
                        ⚠️ {ind}
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("<div style='color: #27D69B; font-size: 0.85rem;'>✓ No heuristic risk rules violated.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_g:
        st.markdown("""
            <div class="mt-card" style="margin-bottom: 1rem;">
                <h5 style="color: #F8FAFC; margin: 0 0 0.5rem 0; font-family: 'Manrope', sans-serif; font-size: 0.95rem;">
                    Immediate Ego-Network Subgraph
                </h5>
        """, unsafe_allow_html=True)
        fig_ego = graph_engine.visualize_network(
            nodes_subset=[selected_account],
            title=f"Direct Linkages for {selected_account}"
        )
        st.plotly_chart(fig_ego, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 4. Associated Transactions Audit Trail
    st.markdown(f"""
        <div class="mt-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <h4 style="margin: 0; font-size: 1rem; color: #F8FAFC; font-family: 'Manrope', sans-serif;">
                    Audited Transaction History ({len(acc_tx)} records)
                </h4>
                <span style="font-size: 0.75rem; color: #94A3B8;">Factual ledger records chronologically sorted</span>
            </div>
    """, unsafe_allow_html=True)

    if len(acc_tx) > 0:
        display_tx = acc_tx.copy()
        display_tx["direction"] = np.where(display_tx["sender_account"] == selected_account, "OUTGOING ↗", "INCOMING ↙")
        display_tx["counterparty"] = np.where(
            display_tx["sender_account"] == selected_account,
            display_tx["receiver_account"],
            display_tx["sender_account"]
        )
        display_tx["formatted_amount"] = display_tx["amount"].apply(lambda v: f"${v:,.2f}")
        display_tx["risk"] = display_tx["hybrid_risk_score"].apply(lambda s: f"{s:.3f}")

        st.dataframe(
            display_tx[["tx_id", "timestamp", "direction", "counterparty", "formatted_amount", "tx_type", "pattern_type", "risk", "risk_tier"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "tx_id": st.column_config.TextColumn("TX ID"),
                "timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm"),
                "direction": st.column_config.TextColumn("Direction"),
                "counterparty": st.column_config.TextColumn("Counterparty"),
                "formatted_amount": st.column_config.TextColumn("Amount ($)"),
                "tx_type": st.column_config.TextColumn("Type"),
                "pattern_type": st.column_config.TextColumn("Topology Pattern"),
                "risk": st.column_config.TextColumn("Risk Index"),
                "risk_tier": st.column_config.TextColumn("Tier"),
            }
        )
    else:
        st.info("No transaction records found for this account.")
    st.markdown("</div>", unsafe_allow_html=True)
