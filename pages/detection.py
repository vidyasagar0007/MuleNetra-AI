"""
MuleTrace AI - AI Detection Center View
Live multi-model inference, probability distributions, risk tier filtering,
operational threshold sensitivity controls, and transaction-level forensic inspection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def render_detection(context: dict):
    """Render the AI Detection Center workspace."""
    scored_tx = context["scored_tx"]
    risk_engine = context["risk_engine"]
    
    # Header Banner
    st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.5rem; border-bottom: 1px solid #23334B; padding-bottom: 1.25rem;">
            <div>
                <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.35rem;">
                    <span style="font-size: 1.2rem;">⚡</span>
                    <h1 style="font-size: 1.65rem; font-weight: 800; color: #F8FAFC; margin: 0; letter-spacing: -0.025em; font-family: 'Manrope', sans-serif;">
                        AI Detection Center
                    </h1>
                </div>
                <p style="font-size: 0.85rem; color: #94A3B8; margin: 0; font-family: 'Inter', sans-serif;">
                    Multi-model surveillance consensus: Blended Tree Ensemble (60% XGBoost, 40% Random Forest) fused with PyTorch Deep RiskNet.
                </p>
            </div>
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span class="badge badge-low" style="font-size: 0.72rem; padding: 0.35rem 0.75rem;">
                    ● CALIBRATED MULTI-MODEL
                </span>
                <span class="badge badge-medium" style="font-size: 0.72rem; padding: 0.35rem 0.75rem;">
                    CHRONOLOGICAL HOLD-OUT
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 1. Advanced Filter Toolbar
    st.markdown("""
        <div class="mt-card" style="padding: 1rem 1.25rem; margin-bottom: 1rem;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.75rem;">
                Surveillance Filter Toolbar
            </div>
    """, unsafe_allow_html=True)

    fcol1, fcol2, fcol3, fcol4 = st.columns([1.5, 1.5, 2, 1.2])
    
    with fcol1:
        tier_filter = st.selectbox(
            "Filter by Risk Tier",
            ["ALL", "CRITICAL (>=0.75)", "HIGH (0.55-0.74)", "MEDIUM (0.35-0.54)", "LOW (<0.35)"],
            index=0
        )

    with fcol2:
        patterns = ["ALL"] + sorted(scored_tx["pattern_type"].unique().tolist())
        pattern_filter = st.selectbox("Filter by AML Pattern", patterns, index=0)

    with fcol3:
        max_amt = float(scored_tx["amount"].max())
        amt_range = st.slider(
            "Transaction Amount Range ($)", 
            min_value=0.0, 
            max_value=max_amt, 
            value=(0.0, max_amt),
            step=500.0
        )

    with fcol4:
        search_acc = st.text_input("Search Account ID", "").strip().upper()

    st.markdown("</div>", unsafe_allow_html=True)

    # Apply filters
    filtered_df = scored_tx.copy()
    
    if tier_filter != "ALL":
        tier_key = tier_filter.split()[0]
        filtered_df = filtered_df[filtered_df["risk_tier"] == tier_key]

    if pattern_filter != "ALL":
        filtered_df = filtered_df[filtered_df["pattern_type"] == pattern_filter]

    if isinstance(amt_range, (list, tuple)) and len(amt_range) == 2:
        filtered_df = filtered_df[
            (filtered_df["amount"] >= amt_range[0]) & 
            (filtered_df["amount"] <= amt_range[1])
        ]
    elif isinstance(amt_range, (int, float)):
        filtered_df = filtered_df[filtered_df["amount"] <= amt_range]

    if search_acc:
        filtered_df = filtered_df[
            (filtered_df["sender_account"].str.contains(search_acc)) |
            (filtered_df["receiver_account"].str.contains(search_acc))
        ]

    # Metrics summary for current filter
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric("Transactions Displayed", f"{len(filtered_df):,}")
    with mcol2:
        st.metric("Total Filtered Volume", f"${filtered_df['amount'].sum():,.2f}")
    with mcol3:
        st.metric("Avg Hybrid Risk", f"{filtered_df['hybrid_risk_score'].mean():.3f}" if len(filtered_df)>0 else "0.000")
    with mcol4:
        st.metric("Anomalous Transactions", f"{(filtered_df['hybrid_risk_score'] >= 0.55).sum():,}")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 2. Interactive Operational Threshold Tuning Box
    st.markdown("""
        <div class="mt-card" style="border-left: 3px solid #4C8DFF; margin-bottom: 1.25rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <h5 style="margin: 0; color: #4C8DFF; font-size: 0.95rem; font-family: 'Manrope', sans-serif;">
                    Operational Alert Threshold Sensitivity Triage
                </h5>
                <span style="font-size: 0.72rem; color: #94A3B8;">Dynamic simulation of alert queue triage</span>
            </div>
    """, unsafe_allow_html=True)

    t_slider = st.slider("Select Cutoff Threshold for Alert Generation", 0.05, 0.95, 0.50, step=0.05)
    t_val = float(t_slider) if isinstance(t_slider, (int, float)) else float(t_slider[0]) if isinstance(t_slider, (list, tuple)) else 0.50
    alert_count = (filtered_df["hybrid_risk_score"] >= t_val).sum()
    st.markdown(f"""
            <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 0.35rem; font-family: 'Inter', sans-serif;">
                At threshold <b style="color: #F8FAFC;">{t_val:.2f}</b>: <b style="color: #FF5B6E;">{alert_count}</b> transactions trigger compliance alerts 
                ({(alert_count/max(len(filtered_df), 1))*100:.2f}% alert rate across filtered dataset).
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 3. Scored Transactions Data Table
    st.markdown("""
        <div class="mt-card" style="margin-bottom: 1.25rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <h4 style="margin: 0; font-size: 0.95rem; color: #F8FAFC; font-family: 'Manrope', sans-serif;">
                    Scored Transaction Stream
                </h4>
                <span style="font-size: 0.72rem; color: #94A3B8;">Top 100 Scored Records</span>
            </div>
    """, unsafe_allow_html=True)

    display_cols = [
        "tx_id", "timestamp", "sender_account", "receiver_account", "amount", 
        "tx_type", "pattern_type", "xgb_risk_prob", "rf_risk_prob", "dl_risk_prob", "hybrid_risk_score", "risk_tier"
    ]
    
    # Check if xgb_risk_prob and rf_risk_prob exist
    for c in ["xgb_risk_prob", "rf_risk_prob"]:
        if c not in filtered_df.columns:
            filtered_df[c] = filtered_df.get("ml_risk_prob", 0.0)

    view_table = filtered_df[display_cols].head(100).copy()
    view_table["amount"] = view_table["amount"].apply(lambda v: f"${v:,.2f}")
    view_table["xgb_risk_prob"] = view_table["xgb_risk_prob"].apply(lambda s: f"{s:.3f}")
    view_table["rf_risk_prob"] = view_table["rf_risk_prob"].apply(lambda s: f"{s:.3f}")
    view_table["dl_risk_prob"] = view_table["dl_risk_prob"].apply(lambda s: f"{s:.3f}")
    view_table["hybrid_risk_score"] = view_table["hybrid_risk_score"].apply(lambda s: f"{s:.3f}")

    st.dataframe(
        view_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "tx_id": st.column_config.TextColumn("TX ID"),
            "timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm"),
            "sender_account": st.column_config.TextColumn("Sender"),
            "receiver_account": st.column_config.TextColumn("Receiver"),
            "amount": st.column_config.TextColumn("Amount"),
            "tx_type": st.column_config.TextColumn("Type"),
            "pattern_type": st.column_config.TextColumn("Topology Pattern"),
            "xgb_risk_prob": st.column_config.TextColumn("XGBoost"),
            "rf_risk_prob": st.column_config.TextColumn("Random Forest"),
            "dl_risk_prob": st.column_config.TextColumn("PyTorch Net"),
            "hybrid_risk_score": st.column_config.TextColumn("Hybrid Score"),
            "risk_tier": st.column_config.TextColumn("Tier"),
        }
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # 4. Multi-Model Probability Comparison Scatter
    st.markdown("""
        <div class="mt-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                <div>
                    <h4 style="margin: 0; font-size: 0.95rem; color: #F8FAFC; font-family: 'Manrope', sans-serif;">
                        Multi-Model Consensus & Divergence Matrix
                    </h4>
                    <p style="font-size: 0.78rem; color: #94A3B8; margin: 0.2rem 0 0 0;">
                        Comparing XGBoost boosting probabilities against PyTorch deep representation risk scores across transactions.
                    </p>
                </div>
                <span class="badge badge-violet" style="font-size: 0.7rem;">ENSEMBLE ATTRIBUTION</span>
            </div>
    """, unsafe_allow_html=True)

    sample_for_plot = scored_tx.sample(min(800, len(scored_tx)), random_state=42)
    fig_scatter = px.scatter(
        sample_for_plot,
        x="xgb_risk_prob" if "xgb_risk_prob" in sample_for_plot.columns else "ml_risk_prob",
        y="dl_risk_prob",
        color="risk_tier",
        size="amount",
        hover_data=["tx_id", "sender_account", "receiver_account", "pattern_type"],
        color_discrete_map={
            "CRITICAL": "#FF5B6E", 
            "HIGH": "#F5B942", 
            "MEDIUM": "#4C8DFF", 
            "LOW": "#27D69B"
        },
        labels={
            "xgb_risk_prob": "XGBoost Anomaly Probability", 
            "ml_risk_prob": "XGBoost Anomaly Probability", 
            "dl_risk_prob": "PyTorch Neural Risk Probability"
        }
    )
    fig_scatter.update_layout(
        paper_bgcolor="#121C2E",
        plot_bgcolor="#121C2E",
        font=dict(color="#94A3B8", family="Inter"),
        xaxis=dict(gridcolor="#162238", color="#64748B"),
        yaxis=dict(gridcolor="#162238", color="#64748B"),
        margin=dict(l=20, r=20, t=20, b=20),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
