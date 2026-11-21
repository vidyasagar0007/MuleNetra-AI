"""
MuleTrace AI - Graph Intelligence Engine
Constructs directed transaction graphs, computes comprehensive network centralities
(degree, weighted degree, PageRank, clustering coefficient, flow ratios),
identifies mule topologies (Collectors, Intermediaries, Distributors, Exit Mules),
detects circular cycles, and builds interactive Plotly network visualizations.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import GRAPH_CONFIG

class GraphEngine:
    """
    Forensic Graph Analysis Engine for detecting money mule networks and computing
    advanced topological graph features.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.account_metrics: Dict[str, Dict[str, Any]] = {}
        self.subgraphs_cache: List[nx.DiGraph] = []

    def build_graph(self, df: pd.DataFrame, max_timestamp: Optional[Any] = None) -> nx.DiGraph:
        """
        Construct a directed weighted graph from transactions dataframe.
        If max_timestamp is provided, filters strictly to historical transactions
        to prevent forward-looking graph leakage.
        """
        self.graph.clear()
        
        df_filtered = df
        if max_timestamp is not None:
            max_ts = pd.to_datetime(max_timestamp)
            df_filtered = df[pd.to_datetime(df["timestamp"]) <= max_ts]

        edge_data = df_filtered.groupby(["sender_account", "receiver_account"]).agg(
            total_amount=("amount", "sum"),
            tx_count=("amount", "count"),
            is_laundering=("is_laundering", "max")
        ).reset_index()

        for _, row in edge_data.iterrows():
            u = str(row["sender_account"])
            v = str(row["receiver_account"])
            weight = float(row["total_amount"])
            count = int(row["tx_count"])
            is_suspicious = int(row["is_laundering"])

            self.graph.add_edge(
                u, v,
                weight=weight,
                tx_count=count,
                is_suspicious=is_suspicious
            )

        self.compute_graph_metrics()
        return self.graph

    def compute_graph_metrics(self):
        """
        Compute degree, weighted degree, PageRank, clustering coefficients,
        and transaction flow dynamics (retention, net flow ratio, flow-through velocity).
        """
        if len(self.graph) == 0:
            return

        in_degrees = dict(self.graph.in_degree())
        out_degrees = dict(self.graph.out_degree())
        deg_centrality = nx.degree_centrality(self.graph)

        # Weighted In-Degree & Out-Degree (Volume)
        in_flow = {n: 0.0 for n in self.graph.nodes()}
        out_flow = {n: 0.0 for n in self.graph.nodes()}

        for u, v, data in self.graph.edges(data=True):
            amt = data.get("weight", 0.0)
            out_flow[u] += amt
            in_flow[v] += amt

        # PageRank (Centrality as absorbing fund sinks or hubs)
        try:
            pagerank = nx.pagerank(self.graph, weight="weight", max_iter=150, alpha=0.85)
        except Exception:
            pagerank = {n: 0.0 for n in self.graph.nodes()}

        # Clustering Coefficient (Local triadic closure on undirected projection)
        undirected_g = self.graph.to_undirected()
        try:
            clustering = nx.clustering(undirected_g)
        except Exception:
            clustering = {n: 0.0 for n in self.graph.nodes()}

        metrics = {}
        for node in self.graph.nodes():
            in_d = in_degrees.get(node, 0)
            out_d = out_degrees.get(node, 0)
            tot_d = in_d + out_d
            tot_in = in_flow.get(node, 0.0)
            tot_out = out_flow.get(node, 0.0)
            tot_vol = tot_in + tot_out

            # Flow dynamics
            net_ratio = tot_out / max(tot_in, 1.0)
            retention_ratio = 1.0 - min(net_ratio, 1.0) if tot_in > 0 else 0.0

            # Mule role heuristics
            role = "BENIGN"
            is_flagged = False
            flag_reasons = []

            # 1. Collector Mule (Fan-in): multiple distinct senders, low out-degree
            if in_d >= GRAPH_CONFIG["fan_in_threshold"] and out_d <= 2:
                role = "COLLECTOR"
                is_flagged = True
                flag_reasons.append(f"Fan-In anomaly: {in_d} incoming counterparties -> 1 aggregator")

            # 2. Distributor Mule (Fan-out): low in-degree, high out-degree
            elif out_d >= GRAPH_CONFIG["fan_out_threshold"] and in_d <= 2:
                role = "DISTRIBUTOR"
                is_flagged = True
                flag_reasons.append(f"Fan-Out anomaly: Disbursing to {out_d} satellite accounts")

            # 3. Intermediary / Pass-Through Mule: high velocity balance (outflow ~= inflow)
            elif in_d >= 2 and out_d >= 2 and 0.82 <= net_ratio <= 1.18 and (tot_in > 9000.0):
                role = "INTERMEDIARY"
                is_flagged = True
                flag_reasons.append(f"Rapid Pass-Through: Inflow ${tot_in:,.0f} ~ Outflow ${tot_out:,.0f} (Ratio: {net_ratio:.2f})")

            # 4. Exit Mule / Cash Out: High inflow, zero or near-zero forward transfers
            elif in_d >= 1 and out_d == 0 and tot_in > 12000.0:
                role = "EXIT_CASH_OUT"
                is_flagged = True
                flag_reasons.append(f"Potential Cash-out Exit: High terminal inflow ${tot_in:,.0f}")

            # Calculate individual node graph risk score [0, 1]
            graph_risk = 0.0
            if is_flagged:
                graph_risk = min(0.95, 0.45 + (0.08 * min(tot_d, 6)))
            elif tot_d > 8 and net_ratio > 0.8:
                graph_risk = 0.35
            else:
                graph_risk = min(0.25, tot_d * 0.02)

            metrics[node] = {
                "in_degree": in_d,
                "out_degree": out_d,
                "total_degree": tot_d,
                "weighted_in_degree": tot_in,
                "weighted_out_degree": tot_out,
                "total_weighted_degree": tot_vol,
                "inflow": tot_in,
                "outflow": tot_out,
                "net_ratio": net_ratio,
                "retention_ratio": retention_ratio,
                "degree_centrality": deg_centrality.get(node, 0.0),
                "pagerank": pagerank.get(node, 0.0),
                "clustering_coefficient": clustering.get(node, 0.0),
                "mule_role": role,
                "is_flagged": is_flagged,
                "flag_reasons": flag_reasons,
                "graph_risk_score": float(graph_risk)
            }

        self.account_metrics = metrics

    def detect_circular_patterns(self, max_cycle_length: int = 5, max_results: int = 20) -> List[List[str]]:
        """
        Identify circular wash trading loops (e.g. A -> B -> C -> A) using a fast bounded DFS
        to avoid exponential complexity on large graphs.
        """
        found_cycles = []
        visited_cycles = set()

        candidate_nodes = [
            n for n in self.graph.nodes() 
            if self.graph.in_degree(n) >= 1 and self.graph.out_degree(n) >= 1
        ]
        
        # Sort candidates to check most active first
        candidate_nodes.sort(
            key=lambda n: self.account_metrics.get(n, {}).get("inflow", 0.0), 
            reverse=True
        )

        for start_node in candidate_nodes[:80]:
            stack = [(start_node, [start_node])]
            while stack and len(found_cycles) < max_results:
                curr, path = stack.pop()
                if len(path) > max_cycle_length:
                    continue

                for neighbor in self.graph.successors(curr):
                    if neighbor == start_node and len(path) >= 3:
                        min_idx = path.index(min(path))
                        canonical = tuple(path[min_idx:] + path[:min_idx])
                        if canonical not in visited_cycles:
                            visited_cycles.add(canonical)
                            found_cycles.append(list(canonical))
                            if len(found_cycles) >= max_results:
                                break
                    elif neighbor not in path and len(path) < max_cycle_length:
                        stack.append((neighbor, path + [neighbor]))

        return found_cycles

    def get_suspicious_clusters(self, min_size: int = 3) -> List[Dict[str, Any]]:
        """
        Decompose graph into weakly connected components and identify clusters
        with high concentration of flagged accounts or cyclic loops.
        """
        suspicious_clusters = []
        components = list(nx.weakly_connected_components(self.graph))
        
        for idx, comp in enumerate(components):
            subg = self.graph.subgraph(comp)
            if len(subg) < min_size:
                continue

            flagged_nodes = [n for n in subg.nodes() if self.account_metrics.get(n, {}).get("is_flagged", False)]
            total_volume = sum(data.get("weight", 0.0) for _, _, data in subg.edges(data=True))
            
            has_cycles = False
            try:
                has_cycles = not nx.is_directed_acyclic_graph(subg)
            except Exception:
                has_cycles = False

            if len(flagged_nodes) > 0 or has_cycles:
                suspicious_clusters.append({
                    "cluster_id": f"CLUSTER_{idx + 1:03d}",
                    "node_count": len(subg.nodes()),
                    "edge_count": len(subg.edges()),
                    "total_volume": float(total_volume),
                    "flagged_accounts": flagged_nodes,
                    "all_accounts": list(subg.nodes()),
                    "has_cycles": has_cycles,
                    "cycle_count": 1 if has_cycles else 0,
                    "risk_tier": "CRITICAL" if len(flagged_nodes) >= 2 or has_cycles else "HIGH"
                })

        suspicious_clusters.sort(key=lambda x: (len(x["flagged_accounts"]), x["total_volume"]), reverse=True)
        return suspicious_clusters

    def visualize_network(
        self, 
        nodes_subset: Optional[List[str]] = None,
        title: str = "Money Mule Transaction Network"
    ) -> go.Figure:
        """
        Generate a high-performance, dark-themed interactive Plotly 2D directed network graph.
        """
        if nodes_subset and len(nodes_subset) > 0:
            target_nodes = set(nodes_subset)
            extended_nodes = set(target_nodes)
            for n in target_nodes:
                if n in self.graph:
                    extended_nodes.update(self.graph.predecessors(n))
                    extended_nodes.update(self.graph.successors(n))
            subg = self.graph.subgraph(extended_nodes)
        else:
            top_nodes = sorted(self.graph.nodes(), key=lambda n: self.graph.degree(n), reverse=True)[:60]
            subg = self.graph.subgraph(top_nodes)

        if len(subg) == 0:
            fig = go.Figure()
            fig.update_layout(
                title="No network data to display",
                template="plotly_dark",
                paper_bgcolor="#0F172A",
                plot_bgcolor="#0F172A"
            )
            return fig

        pos = nx.spring_layout(subg, k=0.35, iterations=40, seed=42)

        edge_x, edge_y, edge_texts = [], [], []
        for u, v, data in subg.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_texts.append(f"{u} → {v}<br>Amount: ${data.get('weight', 0):,.2f}<br>Count: {data.get('tx_count', 1)}")

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.2, color="#475569"),
            hoverinfo="none",
            mode="lines"
        )

        node_x, node_y, node_colors, node_sizes, node_texts, node_symbols = [], [], [], [], [], []

        for node in subg.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            meta = self.account_metrics.get(node, {})
            role = meta.get("mule_role", "BENIGN")
            inflow = meta.get("inflow", 0.0)
            outflow = meta.get("outflow", 0.0)

            if role == "COLLECTOR":
                node_colors.append("#F43F5E")
                node_sizes.append(22)
                node_symbols.append("square")
            elif role == "DISTRIBUTOR":
                node_colors.append("#F59E0B")
                node_sizes.append(20)
                node_symbols.append("diamond")
            elif role == "INTERMEDIARY":
                node_colors.append("#8B5CF6")
                node_sizes.append(18)
                node_symbols.append("hexagon")
            elif role == "EXIT_CASH_OUT":
                node_colors.append("#EC4899")
                node_sizes.append(18)
                node_symbols.append("star")
            else:
                node_colors.append("#38BDF8")
                node_sizes.append(12)
                node_symbols.append("circle")

            hover_str = (
                f"<b>Account: {node}</b><br>"
                f"Role: <b>{role}</b><br>"
                f"In-Degree: {meta.get('in_degree', 0)} | Out-Degree: {meta.get('out_degree', 0)}<br>"
                f"PageRank: {meta.get('pagerank', 0):.4f}<br>"
                f"Clustering: {meta.get('clustering_coefficient', 0):.3f}<br>"
                f"Inflow: ${inflow:,.2f}<br>"
                f"Outflow: ${outflow:,.2f}<br>"
                f"Graph Risk: {meta.get('graph_risk_score', 0):.2f}"
            )
            node_texts.append(hover_str)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            hoverinfo="text",
            hovertext=node_texts,
            text=[n if subg.degree(n) >= 4 else "" for n in subg.nodes()],
            textposition="top center",
            textfont=dict(size=9, color="#94A3B8", family="JetBrains Mono"),
            marker=dict(
                size=node_sizes,
                color=node_colors,
                symbol=node_symbols,
                line=dict(width=1.5, color="#0F172A")
            )
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text=f"<b>{title}</b>",
                    font=dict(color="#F8FAFC", size=14, family="Inter")
                ),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=20, r=20, t=40),
                paper_bgcolor="#0F172A",
                plot_bgcolor="#0F172A",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        return fig

if __name__ == "__main__":
    from src.data_loader import get_or_create_dataset
    df = get_or_create_dataset()
    ge = GraphEngine()
    ge.build_graph(df)
    print(f"Graph constructed: {len(ge.graph.nodes())} nodes, {len(ge.graph.edges())} edges.")
    sample_node = list(ge.account_metrics.keys())[0]
    print(f"Sample node {sample_node} metrics:")
    for k, v in ge.account_metrics[sample_node].items():
        if k != "flag_reasons":
            print(f"  {k}: {v}")
    clusters = ge.get_suspicious_clusters()
    print(f"Detected {len(clusters)} suspicious clusters.")
    cycles = ge.detect_circular_patterns()
    print(f"Detected {len(cycles)} circular wash cycles.")
