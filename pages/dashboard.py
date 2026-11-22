"""
MuleTrace AI - Executive Dashboard View
High-level telemetry, key risk indicators (KRIs), temporal volume trends,
risk distribution, and urgent investigation alerts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def render_dashboard(context: dict):
    """Render the executive financial crime intelligence dashboard."""
    scored_tx = context["scored_tx"]
    profiles = context["profiles"]
    clusters = context["clusters"]
    
    # Header Banner
    st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
            <div>
                <h1 style="font-size: 1.6rem; font-weight: 800; color: #F8FAFC; margin: 0; letter-spacing: -0.02em;">
                    Financial Crime Executive Dashboard
                </h1>
                <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 0.2rem;">
                    Real-time transaction surveillance, mule network telemetry, and risk distribution metrics.
                </p>
            </div>
            <div>
                <span class="badge badge-low" style="font-size: 0.75rem; padding: 0.35rem 0.75rem;">
                    ● ACTIVE SURVEILLANCE • 100% ENGINE HEALTH
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 1. KPI Cards Row
    total_tx = len(scored_tx)
    total_accounts = len(profiles)
    suspicious_tx = (scored_tx["hybrid_risk_score"] >= 0.55).sum()
    critical_accounts = (profiles["risk_tier"] == "CRITICAL").sum()
    high_risk_accounts = (profiles["risk_tier"] == "HIGH").sum()
    total_volume = scored_tx["amount"].sum()
    suspicious_volume = scored_tx[scored_tx["hybrid_risk_score"] >= 0.55]["amount"].sum()

    # Row 1: High-level volume and entity scope (3 columns)
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
            <div class="kpi-container info">
                <span class="kpi-label">Monitored Account Nodes</span>
                <span class="kpi-value">{total_accounts:,}</span>
                <span class="kpi-delta">Active banking graph entities</span>
            </div>
        """, unsafe_allow_html=True)

    with row1_c3:
        st.markdown(f"""
            <div class="kpi-container info">
                <span class="kpi-label">Total Transaction Volume</span>
                <span class="kpi-value">${total_volume/1e6:.2f}M</span>
                <span class="kpi-delta">${suspicious_volume:,.0f} under forensic scrutiny</span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Row 2: Anomaly and Risk Tier Telemetry (3 columns)
    row2_c1, row2_c2, row2_c3 = st.columns(3)

    with row2_c1:
        st.markdown(f"""
            <div class="kpi-container critical">
                <span class="kpi-label">Flagged Anomalous Txns</span>
                <span class="kpi-value">{suspicious_tx}</span>
                <span class="kpi-delta" style="color: #FB7185;">{(suspicious_tx/max(total_tx,1))*100:.2f}% anomaly rate across window</span>
            </div>
        """, unsafe_allow_html=True)

    with row2_c2:
        st.markdown(f"""
            <div class="kpi-container critical">
                <span class="kpi-label">Critical Money Mules</span>
                <span class="kpi-value">{critical_accounts}</span>
                <span class="kpi-delta" style="color: #FB7185;">Tier-1 Urgent FIU Escalation</span>
            </div>
        """, unsafe_allow_html=True)

    with row2_c3:
        st.markdown(f"""
            <div class="kpi-container warning">
                <span class="kpi-label">Flagged Network Clusters</span>
                <span class="kpi-value">{len(clusters)}</span>
                <span class="kpi-delta">Multi-hop suspicious subgraphs</span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # 2. Visualizations Row
    chart_col1, chart_col2 = st.columns([1, 2])

    with chart_col1:
        st.markdown("""
            <div style="background-color: #131D31; border: 1px solid #1E293B; border-radius: 8px; padding: 1rem 1.25rem;">
                <h4 style="margin: 0 0 0.5rem 0; font-size: 0.95rem; color: #F8FAFC;">Account Risk Distribution</h4>
        """, unsafe_allow_html=True)
        
        tier_counts = profiles["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        
        color_map = {
            "CRITICAL": "#F43F5E",
            "HIGH": "#F59E0B",
            "MEDIUM": "#3B82F6",
            "LOW": "#10B981"
        }

        fig_donut = px.pie(
            tier_counts, 
            values="Count", 
            names="Tier",
            hole=0.65,
            color="Tier",
            color_discrete_map=color_map
        )
        fig_donut.update_layout(
            paper_bgcolor="#131D31",
            plot_bgcolor="#131D31",
            font=dict(color="#94A3B8", family="Inter"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(l=10, r=10, t=10, b=10),
            height=260
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col2:
        st.markdown("""
            <div style="background-color: #131D31; border: 1px solid #1E293B; border-radius: 8px; padding: 1rem 1.25rem;">
                <h4 style="margin: 0 0 0.5rem 0; font-size: 0.95rem; color: #F8FAFC;">Transaction Activity & Volume Over Time</h4>
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
            marker_color="#1E293B",
            opacity=0.8
        ))
        fig_time.add_trace(go.Scatter(
            x=daily_stats["date"],
            y=daily_stats["flagged_count"] * 10000, # Scaled for dual-axis visual
            name="Suspicious Spikes",
            line=dict(color="#F43F5E", width=2.5),
            mode="lines+markers"
        ))
        fig_time.update_layout(
            paper_bgcolor="#131D31",
            plot_bgcolor="#131D31",
            font=dict(color="#94A3B8", family="Inter"),
            xaxis=dict(gridcolor="#1E293B", color="#64748B"),
            yaxis=dict(gridcolor="#1E293B", color="#64748B", title="Volume ($)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=20, b=10),
            height=260
        )
        st.plotly_chart(fig_time, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # 3. Urgent Investigation Alerts Table
    st.markdown("""
        <div style="background-color: #131D31; border: 1px solid #1E293B; border-radius: 8px; padding: 1.25rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <h4 style="margin: 0; font-size: 1rem; color: #F8FAFC;">Priority Investigation Alerts (Top Flagged Entities)</h4>
                <span style="font-size: 0.8rem; color: #94A3B8;">Sorted by calibrated hybrid risk score</span>
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
            "in_degree": st.column_config.NumberColumn("Inflows Count"),
            "out_degree": st.column_config.NumberColumn("Outflows Count"),
            "total_inflow": st.column_config.TextColumn("Total Inflow"),
            "total_outflow": st.column_config.TextColumn("Total Outflow"),
        }
    )
    st.markdown("</div>", unsafe_allow_html=True)
