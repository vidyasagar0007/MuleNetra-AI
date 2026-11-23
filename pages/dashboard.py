"""
MuleNetra AI - Executive Dashboard View
High-level telemetry, key risk indicators (KRIs), temporal volume trends,
risk distribution, and urgent investigation alerts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.demo_guide import render_demo_guide

def render_dashboard(context: dict):
    """Render the executive financial crime intelligence command center."""
    scored_tx = context["scored_tx"]
    profiles = context["profiles"]
    clusters = context["clusters"]
    
    # Header Banner with Live Operational Pill
    st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.5rem; border-bottom: 1px solid #23334B; padding-bottom: 1.25rem;">
            <div>
                <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.35rem;">
                    <span style="font-size: 1.2rem;">📊</span>
                    <h1 style="font-size: 1.65rem; font-weight: 800; color: #F8FAFC; margin: 0; letter-spacing: -0.025em; font-family: 'Manrope', sans-serif;">
                        Executive Command Center
                    </h1>
                </div>
                <p style="font-size: 0.85rem; color: #94A3B8; margin: 0; font-family: 'Inter', sans-serif;">
                    Real-time transaction surveillance, multi-hop mule network telemetry, and institutional risk distribution.
                </p>
            </div>
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span class="badge badge-low" style="font-size: 0.72rem; padding: 0.35rem 0.75rem;">
                    ● ACTIVE SURVEILLANCE • 100% HEALTH
                </span>
                <span class="badge badge-medium" style="font-size: 0.72rem; padding: 0.35rem 0.75rem;">
                    AUDITED ZERO-LEAKAGE
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Optional Collapsible Guided Demo Walkthrough
    render_demo_guide("dashboard")

    # 1. KPI Cards: 2 Rows x 3 Columns (Spacious, Aligned, Distinct Color Bands)
    total_tx = len(scored_tx)
    total_accounts = len(profiles)
    suspicious_tx = (scored_tx["hybrid_risk_score"] >= 0.55).sum()
    critical_accounts = (profiles["risk_tier"] == "CRITICAL").sum()
    high_risk_accounts = (profiles["risk_tier"] == "HIGH").sum()
    total_volume = scored_tx["amount"].sum()
    suspicious_volume = scored_tx[scored_tx["hybrid_risk_score"] >= 0.55]["amount"].sum()

    # Row 1: High-level volume and entity scope
    row1_c1, row1_c2, row1_c3 = st.columns(3)

    with row1_c1:
        st.markdown(f"""
            <div class="kpi-container info">
                <span class="kpi-label">Total Monitored Transactions</span>
                <span class="kpi-value">{total_tx:,}</span>
                <span class="kpi-delta">Sequential audit stream processed</span>
            </div>
        """, unsafe_allow_html=True)

    with row1_c2:
        st.markdown(f"""
            <div class="kpi-container violet">
                <span class="kpi-label">Active Account Entities</span>
                <span class="kpi-value">{total_accounts:,}</span>
                <span class="kpi-delta">Monitored banking graph nodes</span>
            </div>
        """, unsafe_allow_html=True)

    with row1_c3:
        st.markdown(f"""
            <div class="kpi-container info">
                <span class="kpi-label">Total Monitored Volume</span>
                <span class="kpi-value">${total_volume/1e6:.2f}M</span>
                <span class="kpi-delta">${suspicious_volume:,.0f} flagged for forensic review</span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Row 2: Anomaly and Risk Tier Telemetry
    row2_c1, row2_c2, row2_c3 = st.columns(3)

    with row2_c1:
        st.markdown(f"""
            <div class="kpi-container critical">
                <span class="kpi-label">Flagged Anomalous Txns</span>
                <span class="kpi-value">{suspicious_tx}</span>
                <span class="kpi-delta" style="color: #FF5B6E;">{(suspicious_tx/max(total_tx,1))*100:.2f}% anomaly rate across window</span>
            </div>
        """, unsafe_allow_html=True)

    with row2_c2:
        st.markdown(f"""
            <div class="kpi-container critical">
                <span class="kpi-label">Critical Money Mules</span>
                <span class="kpi-value">{critical_accounts}</span>
                <span class="kpi-delta" style="color: #FF5B6E;">Tier-1 Urgent FIU Escalation Required</span>
            </div>
        """, unsafe_allow_html=True)

    with row2_c3:
        st.markdown(f"""
            <div class="kpi-container warning">
                <span class="kpi-label">Flagged Network Clusters</span>
                <span class="kpi-value">{len(clusters)}</span>
                <span class="kpi-delta" style="color: #F5B942;">Multi-hop wash-trading subgraphs</span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # 2. Charts Row: Donut Chart & Activity Over Time
    chart_col1, chart_col2 = st.columns([1, 2])

    with chart_col1:
        st.markdown(f"""
            <div class="mt-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                    <h4 style="margin: 0; font-size: 0.95rem; color: #F8FAFC; font-family: 'Manrope', sans-serif;">
                        Account Risk Distribution
                    </h4>
                    <span style="font-size: 0.72rem; color: #94A3B8;">{total_accounts:,} Nodes</span>
                </div>
        """, unsafe_allow_html=True)
        
        tier_counts = profiles["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        
        # New Enterprise Palette
        color_map = {
            "CRITICAL": "#FF5B6E",
            "HIGH": "#F5B942",
            "MEDIUM": "#4C8DFF",
            "LOW": "#27D69B"
        }

        fig_donut = px.pie(
            tier_counts, 
            values="Count", 
            names="Tier",
            hole=0.68,
            color="Tier",
            color_discrete_map=color_map
        )
        fig_donut.update_layout(
            paper_bgcolor="#121C2E",
            plot_bgcolor="#121C2E",
            font=dict(color="#94A3B8", family="Inter"),
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=-0.22, 
                xanchor="center", 
                x=0.5,
                font=dict(size=11)
            ),
            margin=dict(l=10, r=10, t=10, b=10),
            height=270
        )
        fig_donut.update_traces(
            marker=dict(line=dict(color='#080D18', width=2))
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col2:
        st.markdown("""
            <div class="mt-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                    <h4 style="margin: 0; font-size: 0.95rem; color: #F8FAFC; font-family: 'Manrope', sans-serif;">
                        Transaction Activity & Volume Over Time
                    </h4>
                    <span style="font-size: 0.72rem; color: #94A3B8;">Daily Aggregated Stream</span>
                </div>
        """, unsafe_allow_html=True)
        
        scored_tx["date"] = pd.to_datetime(scored_tx["timestamp"]).dt.date
        daily_stats = scored_tx.groupby("date").agg(
            volume=("amount", "sum"),
            tx_count=("amount", "count"),
            flagged_count=("hybrid_risk_score", lambda s: (s >= 0.55).sum())
        ).reset_index()

        fig_time = go.Figure()
        fig_time.add_trace(go.Bar(
            x=daily_stats["date"],
            y=daily_stats["volume"],
            name="Daily Volume ($)",
            marker_color="#1E2E4A",
            marker_line=dict(color="#23334B", width=1),
            opacity=0.85
        ))
        fig_time.add_trace(go.Scatter(
            x=daily_stats["date"],
            y=daily_stats["flagged_count"] * 10000, # Scaled for overlay visualization
            name="Suspicious Spikes",
            line=dict(color="#FF5B6E", width=2.5),
            marker=dict(color="#FF5B6E", size=5),
            mode="lines+markers"
        ))
        fig_time.update_layout(
            paper_bgcolor="#121C2E",
            plot_bgcolor="#121C2E",
            font=dict(color="#94A3B8", family="Inter"),
            xaxis=dict(gridcolor="#162238", color="#64748B"),
            yaxis=dict(gridcolor="#162238", color="#64748B", title="Volume ($)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1, font=dict(size=11)),
            margin=dict(l=10, r=10, t=20, b=10),
            height=270
        )
        st.plotly_chart(fig_time, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # 3. Urgent Investigation Alerts Table
    st.markdown("""
        <div class="mt-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.85rem;">
                <div>
                    <h4 style="margin: 0; font-size: 1rem; color: #F8FAFC; font-family: 'Manrope', sans-serif;">
                        Priority Investigation Queue (Top Flagged Entities)
                    </h4>
                    <span style="font-size: 0.78rem; color: #94A3B8;">
                        Entities ranked by calibrated hybrid risk index (ML + Deep Learning + Graph Topology)
                    </span>
                </div>
                <div>
                    <span class="badge badge-critical" style="font-size: 0.72rem;">TIER-1 TRIAGE</span>
                </div>
            </div>
    """, unsafe_allow_html=True)

    high_risk_table = profiles.head(8)[[
        "account_id", "risk_score", "risk_tier", "mule_role", 
        "in_degree", "out_degree", "total_inflow", "total_outflow"
    ]].copy()

    high_risk_table["total_inflow"] = high_risk_table["total_inflow"].apply(lambda v: f"${v:,.2f}")
    high_risk_table["total_outflow"] = high_risk_table["total_outflow"].apply(lambda v: f"${v:,.2f}")
    high_risk_table["risk_score"] = high_risk_table["risk_score"].apply(lambda s: f"{s:.3f}")

    st.dataframe(
        high_risk_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "account_id": st.column_config.TextColumn("Account ID"),
            "risk_score": st.column_config.TextColumn("Risk Index"),
            "risk_tier": st.column_config.TextColumn("Risk Tier"),
            "mule_role": st.column_config.TextColumn("Graph Role"),
            "in_degree": st.column_config.NumberColumn("Inflow Edges"),
            "out_degree": st.column_config.NumberColumn("Outflow Edges"),
            "total_inflow": st.column_config.TextColumn("Total Inflow"),
            "total_outflow": st.column_config.TextColumn("Total Outflow"),
        }
    )
    st.markdown("</div>", unsafe_allow_html=True)
