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

def render_account_investigation(context: dict):
    """Render the Deep Account Investigation view."""
    scored_tx = context["scored_tx"]
    profiles = context["profiles"]
    graph_engine = context["graph_engine"]
    
    st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
            <div>
                <h1 style="font-size: 1.6rem; font-weight: 800; color: #F8FAFC; margin: 0; letter-spacing: -0.02em;">
                    Deep Account Forensic Investigation
                </h1>
                <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 0.2rem;">
                    Examine account risk attribution, counterparty ego-networks, and chronological audit trails.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 1. Target Account Selector
    all_account_ids = profiles["account_id"].tolist()
    
    # Pre-select highest risk account or query param
    default_acc = all_account_ids[0] if all_account_ids else ""
    
    scol1, scol2 = st.columns([2, 1])
    with scol1:
        selected_account = st.selectbox(
            "Select Target Account (Ranked by Risk Index)",
            all_account_ids,
            format_func=lambda acc: f"{acc} — Tier: {profiles[profiles['account_id']==acc]['risk_tier'].values[0]} (Score: {profiles[profiles['account_id']==acc]['risk_score'].values[0]:.2f})",
            index=0
        )
    with scol2:
        quick_search = st.text_input("Or Type Specific Account ID", "").strip().upper()
        if quick_search and quick_search in all_account_ids:
            selected_account = quick_search

    # Retrieve account profile
    account_row = profiles[profiles["account_id"] == selected_account].iloc[0]
    profile_dict = account_row.to_dict()
    
    # Retrieve associated transactions
    acc_tx = scored_tx[
        (scored_tx["sender_account"] == selected_account) | 
        (scored_tx["receiver_account"] == selected_account)
    ].sort_values(by="timestamp", ascending=False).reset_index(drop=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 2. Forensic Profile Banner
    tier = profile_dict["risk_tier"]
    score = profile_dict["risk_score"]
    role = profile_dict["mule_role"]
    inflow = profile_dict["total_inflow"]
    outflow = profile_dict["total_outflow"]
    
    tier_badge_class = f"badge-{tier.lower()}"

    st.markdown(f"""
        <div class="mt-card" style="border-left: 4px solid {'#F43F5E' if tier=='CRITICAL' else '#F59E0B' if tier=='HIGH' else '#38BDF8'};">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <span style="font-size: 0.75rem; text-transform: uppercase; color: #94A3B8; font-weight: 600;">TARGET PROFILE</span>
                    <h2 style="font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; color: #38BDF8; margin: 0.2rem 0;">{selected_account}</h2>
                    <span class="badge {tier_badge_class}" style="font-size: 0.8rem; padding: 0.25rem 0.6rem;">{tier} RISK TIER</span>
                    <span style="margin-left: 0.5rem; font-size: 0.85rem; color: #E2E8F0;">Topology Role: <b style="color: #A855F7;">{role}</b></span>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Calibrated Risk Index</div>
                    <div style="font-size: 2.2rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: {'#F43F5E' if score>=0.75 else '#F59E0B' if score>=0.55 else '#10B981'};">
                        {score:.3f}
                    </div>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1.2rem; padding-top: 1rem; border-top: 1px solid #1E293B;">
                <div>
                    <span style="font-size: 0.75rem; color: #94A3B8;">Cumulative Inflow</span>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #10B981;">${inflow:,.2f}</div>
                </div>
                <div>
                    <span style="font-size: 0.75rem; color: #94A3B8;">Cumulative Outflow</span>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #F43F5E;">${outflow:,.2f}</div>
                </div>
                <div>
                    <span style="font-size: 0.75rem; color: #94A3B8;">Direct Counterparties</span>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #F8FAFC;">{profile_dict['in_degree']} In / {profile_dict['out_degree']} Out</div>
                </div>
                <div>
                    <span style="font-size: 0.75rem; color: #94A3B8;">Retention Ratio</span>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #38BDF8;">{(1 - min(outflow/max(inflow, 1.0), 1.0))*100:.1f}%</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 3. Two columns: Explainability Waterfall & Ego-Network
    col_w, col_g = st.columns([1.1, 0.9])

    with col_w:
        exp_engine = ExplainabilityEngine()
        fig_waterfall = exp_engine.plot_risk_waterfall(profile_dict)
        st.plotly_chart(fig_waterfall, use_container_width=True)

        # Triggered Indicators
        raw_inds = profile_dict.get("flagged_indicators", "[]")
        try:
            indicators = json.loads(raw_inds) if isinstance(raw_inds, str) else raw_inds
        except Exception:
            indicators = []

        st.markdown("<h5 style='color: #F8FAFC; margin-bottom: 0.4rem;'>Triggered Forensic Rules & Indicators</h5>", unsafe_allow_html=True)
        if indicators:
            for ind in indicators:
                st.markdown(f"""
                    <div style="background-color: rgba(244, 63, 94, 0.08); border-left: 3px solid #F43F5E; padding: 0.5rem 0.75rem; margin-bottom: 0.4rem; border-radius: 0 4px 4px 0; font-size: 0.85rem; color: #F8FAFC;">
                        ⚠️ {ind}
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("<div style='color: #10B981; font-size: 0.85rem;'>✓ No heuristic risk rules violated.</div>", unsafe_allow_html=True)

    with col_g:
        st.markdown("<h5 style='color: #F8FAFC; margin-bottom: 0.4rem;'>Immediate Ego-Network Subgraph</h5>", unsafe_allow_html=True)
        fig_ego = graph_engine.visualize_network(
            nodes_subset=[selected_account],
            title=f"Direct Linkages for {selected_account}"
        )
        st.plotly_chart(fig_ego, use_container_width=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # 4. Associated Transactions Audit Trail
    st.markdown(f"""
        <div style="background-color: #131D31; border: 1px solid #1E293B; border-radius: 8px; padding: 1.25rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <h4 style="margin: 0; font-size: 1rem; color: #F8FAFC;">Audited Transaction History ({len(acc_tx)} records)</h4>
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
