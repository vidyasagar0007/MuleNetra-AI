"""
MuleTrace AI - Money Mule Network Intelligence View
Interactive directed graph topology, community detection, mule role heuristics,
and circular wash-trading loop analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def render_network_analysis(context: dict):
    """Render the Network Intelligence view."""
    graph_engine = context["graph_engine"]
    profiles = context["profiles"]
    clusters = context["clusters"]
    
    st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
            <div>
                <h1 style="font-size: 1.6rem; font-weight: 800; color: #F8FAFC; margin: 0; letter-spacing: -0.02em;">
                    Money Mule Network Intelligence
                </h1>
                <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 0.2rem;">
                    Topological graph intelligence, flow-through centralities, and automated mule role heuristic detection.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 1. Summary Metrics
    mule_counts = profiles["mule_role"].value_counts()
    n_collectors = mule_counts.get("COLLECTOR", 0)
    n_distributors = mule_counts.get("DISTRIBUTOR", 0)
    n_intermediaries = mule_counts.get("INTERMEDIARY", 0)
    n_exit = mule_counts.get("EXIT_CASH_OUT", 0)

    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    with mcol1:
        st.metric("Graph Nodes", f"{len(graph_engine.graph.nodes()):,}")
    with mcol2:
        st.metric("Collectors (Fan-In)", f"{n_collectors}")
    with mcol3:
        st.metric("Distributors (Fan-Out)", f"{n_distributors}")
    with mcol4:
        st.metric("Intermediaries (Relay)", f"{n_intermediaries}")
    with mcol5:
        st.metric("Exit Mules (Cash-Out)", f"{n_exit}")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 2. Network View Selection (Global vs Cluster)
    tcol1, tcol2 = st.columns([1, 3])
    with tcol1:
        view_mode = st.radio(
            "Visualization Target",
            ["Global Top Active Network", "Filter by Suspicious Cluster", "Filter by Mule Role"],
            index=0
        )
        
        target_nodes = None
        if view_mode == "Filter by Suspicious Cluster" and len(clusters) > 0:
            cluster_opts = [f"{c['cluster_id']} ({len(c['flagged_accounts'])} flagged, ${c['total_volume']:,.0f})" for c in clusters]
            selected_cluster_str = st.selectbox("Select Cluster", cluster_opts)
            c_idx = cluster_opts.index(selected_cluster_str)
            target_nodes = clusters[c_idx]["all_accounts"]
        elif view_mode == "Filter by Mule Role":
            selected_role = st.selectbox("Select Role", ["COLLECTOR", "DISTRIBUTOR", "INTERMEDIARY", "EXIT_CASH_OUT"])
            role_accs = profiles[profiles["mule_role"] == selected_role]["account_id"].tolist()
            target_nodes = role_accs

    with tcol2:
        # Legend Header
        st.markdown("""
            <div style="display: flex; gap: 1rem; align-items: center; justify-content: flex-end; margin-bottom: 0.5rem; font-size: 0.78rem;">
                <span><span style="color: #F43F5E;">■</span> Collector (Fan-In)</span>
                <span><span style="color: #F59E0B;">◆</span> Distributor (Fan-Out)</span>
                <span><span style="color: #8B5CF6;">⬢</span> Intermediary</span>
                <span><span style="color: #EC4899;">★</span> Exit Cash-Out</span>
                <span><span style="color: #38BDF8;">●</span> Benign Peer</span>
            </div>
        """, unsafe_allow_html=True)

        fig_net = graph_engine.visualize_network(
            nodes_subset=target_nodes,
            title="Directed Money Movement & Account Linkage"
        )
        st.plotly_chart(fig_net, use_container_width=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # 3. Two columns: Suspicious Clusters & Circular Loops
    bcol1, bcol2 = st.columns(2)

    with bcol1:
        st.markdown("""
            <div style="background-color: #131D31; border: 1px solid #1E293B; border-radius: 8px; padding: 1.1rem 1.25rem;">
                <h4 style="margin: 0 0 0.5rem 0; font-size: 0.95rem; color: #F8FAFC;">Flagged Network Subgraphs & Clusters</h4>
        """, unsafe_allow_html=True)

        if len(clusters) > 0:
            cluster_rows = []
            for c in clusters[:6]:
                cluster_rows.append({
                    "Cluster ID": c["cluster_id"],
                    "Risk Tier": c["risk_tier"],
                    "Accounts": c["node_count"],
                    "Flagged Mules": len(c["flagged_accounts"]),
                    "Volume": f"${c['total_volume']:,.2f}",
                    "Has Cycles": "YES" if c["has_cycles"] else "NO"
                })
            st.dataframe(pd.DataFrame(cluster_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No suspicious multi-node clusters detected.")
        st.markdown("</div>", unsafe_allow_html=True)

    with bcol2:
        st.markdown("""
            <div style="background-color: #131D31; border: 1px solid #1E293B; border-radius: 8px; padding: 1.1rem 1.25rem;">
                <h4 style="margin: 0 0 0.5rem 0; font-size: 0.95rem; color: #F8FAFC;">Circular Wash-Trading Cycles Detected</h4>
        """, unsafe_allow_html=True)

        cycles = graph_engine.detect_circular_patterns(max_cycle_length=5)
        if len(cycles) > 0:
            cycle_rows = []
            for idx, cyc in enumerate(cycles[:6]):
                cycle_rows.append({
                    "Cycle ID": f"CYCLE_{idx+1:02d}",
                    "Hops": len(cyc),
                    "Path Structure": " → ".join(cyc) + f" → {cyc[0]}",
                    "Severity": "CRITICAL"
                })
            st.dataframe(pd.DataFrame(cycle_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No circular transaction cycles identified.")
        st.markdown("</div>", unsafe_allow_html=True)
