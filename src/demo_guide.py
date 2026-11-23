"""
MuleNetra AI - Guided Demo & Viva Walkthrough Component
Provides structured, non-obtrusive examiner guidance, talking points,
and recommended control interactions across all seven surveillance modules.
"""

import streamlit as st

DEMO_GUIDES = {
    "dashboard": {
        "title": "Executive Command Center — Examiner Guide",
        "objective": "High-level operational overview providing real-time financial transaction surveillance, Key Risk Indicators (KRIs), temporal volume velocity, and urgent triage queue.",
        "controls": [
            "Point out the 6 spacious KPI summary cards across the 2-row grid (9,587 transactions, 500 accounts, $34.42M volume, anomaly rate).",
            "Highlight the Priority Investigation Queue table with top-ranked suspicious entities (e.g., ACC_0477).",
            "Point out the Active Surveillance and Audited Zero-Leakage compliance badges."
        ],
        "charts": [
            "Account Risk Distribution (Donut Chart): Visualizes account risk stratification (79.2% Low, 4.0% Medium, 8.4% High, 8.4% Critical).",
            "Transaction Activity & Volume Over Time: Dual-axis plot contrasting daily transaction volume (navy bars) with suspicious anomaly spikes (crimson markers)."
        ],
        "talking_points": [
            "Why is chronological splitting mandatory? Financial transactions are strictly temporal. Random train/test splits cause lookahead data leakage, artificially inflating metrics.",
            "Why is the anomaly rate ~4%? In real financial institutions, fraud is a severe needle-in-a-haystack problem. A balanced 50/50 dataset is unrealistic."
        ]
    },
    "detection": {
        "title": "AI Detection Center — Examiner Guide",
        "objective": "Transaction-level multi-model inference workspace allowing compliance officers to tune alert sensitivities and inspect ML/DL consensus.",
        "controls": [
            "Adjust the 'Operational Alert Threshold Sensitivity' slider from 0.50 to 0.60. Demonstrate how alert count immediately recalculates (92 -> 82 alerts).",
            "Filter by Risk Tier: select 'CRITICAL (>=0.75)' to view the highest-risk transaction stream.",
            "Search specific account 'ACC_0477' in the Search Account ID box to inspect its isolated transactions."
        ],
        "charts": [
            "Multi-Model Consensus & Divergence Matrix: Scatter plot comparing XGBoost anomaly probability (X-axis) against PyTorch neural risk (Y-axis). Points in the top-right quadrant represent high-confidence multi-model consensus.",
            "Scored Transaction Stream: Real-time ledger displaying individual XGBoost, Random Forest, PyTorch, and Hybrid risk scores."
        ],
        "talking_points": [
            "Why combine Tree Boosting with Neural Networks? XGBoost captures sharp non-linear decision boundaries and structuring thresholds ($8k-$10k), while PyTorch Deep RiskNet extracts continuous representation embeddings resistant to threshold manipulation.",
            "What is the operational alert threshold? It allows banks to calibrate compliance queue volume against investigator staffing capacity while monitoring False Positive Rates."
        ]
    },
    "network_analysis": {
        "title": "Mule Network Intelligence — Examiner Guide",
        "objective": "Graph-theoretic topological surveillance detecting money mule syndicates, structuring smurfs, collectors, and circular wash-trading loops.",
        "controls": [
            "Switch Visualization Target to 'Filter by Suspicious Cluster' -> select 'CLUSTER_001' to isolate the multi-node syndicate.",
            "Switch Visualization Target to 'Filter by Mule Role' -> select 'COLLECTOR' to highlight accounts exhibiting high fan-in pooling (e.g. ACC_0477).",
            "Inspect the 'Circular Wash-Trading Cycles Detected' table showing multi-hop closed loops."
        ],
        "charts": [
            "Directed Money Movement Network Graph: Interactive NetworkX digraph using spring layout. Node colors and shapes represent heuristic roles: Square = Collector (#FF5B6E), Diamond = Distributor (#F5B942), Hexagon = Intermediary (#9270FF), Star = Exit Cash-Out (#EC4899), Circle = Benign (#4C8DFF).",
            "Flagged Network Subgraphs Table: Lists weakly connected components ranked by mule concentration."
        ],
        "talking_points": [
            "How are circular loops identified? NetworkX depth-first cycle search identifies directed paths where funds return to origin (A -> B -> C -> A), a classic tax evasion and volume fabrication tactic.",
            "How are mule roles classified? Graph topology metrics: Collectors have high In-Degree and low Out-Degree; Distributors have high Out-Degree; Intermediaries have high betweenness centrality."
        ]
    },
    "account_investigation": {
        "title": "Account Forensic Investigation — Examiner Guide",
        "objective": "Deep-dive 360-degree forensic interrogation for a specific target account, providing full explainability, counterparty ego-network, and audit trail.",
        "controls": [
            "Select target account 'ACC_0477' (pre-ranked #1 Critical Collector Mule).",
            "Inspect the Forensic Profile Banner: Risk score 0.850, Cumulative Inflow $59,082, Cumulative Outflow $53,892, Counterparties (22 in / 1 out), Retention 8.8%.",
            "Show the 'Triggered Forensic Rules & Indicators' alert chips (Sub-threshold structuring, 22:1 Fan-in anomaly, rapid turnaround).",
            "Inspect the Audited Transaction History table with incoming smurf transfers and outgoing cash-out."
        ],
        "charts": [
            "Risk Attribution Waterfall Breakdown: Deconstructs calibrated risk into Baseline Risk (0.05), ML Behavior (+0.34), PyTorch Net (+0.25), and Graph Centrality (+0.23).",
            "Immediate Ego-Network Subgraph: Isolates direct 1-hop counterparties feeding capital into ACC_0477."
        ],
        "talking_points": [
            "Why is explainability (XAI) required? Under banking regulations (FATF, FinCEN), compliance officers cannot submit a Suspicious Activity Report (SAR) based on black-box predictions; explicit evidence factors are mandatory.",
            "What is an ego-network? A localized 1-hop subgraph isolating a specific entity and all its direct transaction counterparties."
        ]
    },
    "ai_assistant": {
        "title": "GenAI Investigation Assistant — Examiner Guide",
        "objective": "Automated, evidence-grounded intelligence briefing, hypothesis formulation, and forensic research PDF dossier generation.",
        "controls": [
            "Verify target ACC_0477 is selected with Deterministic Offline Engine (guaranteeing zero external API dependency and zero hallucinations).",
            "Review the structured 4-Part Brief: 1. Executive Summary, 2. Factual Observations, 3. Hypotheses (Smurf Aggregator), 4. Recommended Actions (FIU Playbook).",
            "Click '📄 Compile Official PDF Investigation Dossier' and demonstrate instant ReportLab PDF generation and download.",
            "In Analyst Case Disposition: set priority to 'ESCALATED_FOR_SAR', enter investigator notes, and click 'Save Case Disposition'."
        ],
        "charts": [
            "Structured Brief Cards: Visual evidence cards with distinct color accents (Blue, Violet, Amber, Green) organizing narrative intelligence.",
            "Case Disposition Module: Interactive database logging interface updating SQLite audit trail."
        ],
        "talking_points": [
            "How do you prevent GenAI hallucinations? The assistant relies on factual data injection where all numbers, counterparties, and timestamps are extracted directly from verified database records.",
            "What is the deterministic offline engine? A template-driven, rule-grounded forensic reasoning engine that operates without internet access or GPU overhead, ensuring complete privacy."
        ]
    },
    "reports": {
        "title": "Reports & Compliance Export Center — Examiner Guide",
        "objective": "Institutional reporting center for bulk surveillance data export, PDF dossier archives, and automated SAR filing.",
        "controls": [
            "Click '⬇️ Download Transactions CSV' to demonstrate full scored ledger export.",
            "Click '⬇️ Download Profiles CSV' to export account-level risk profiles.",
            "Inspect the Generated PDF Forensic Dossiers Repository table and download a previously compiled dossier.",
            "In Automated Regulatory SAR Generator: select ACC_0477, inspect the FinCEN XML-JSON structure in the code viewer, and click '⬇️ Download Regulatory SAR Package (JSON)'."
        ],
        "charts": [
            "Repository Data Table: Lists compiled PDF reports with filename, exact size in KB, and modification timestamp.",
            "Regulatory SAR Code Container: Displays structured machine-readable payload matching FinCEN/FIU electronic filing schema."
        ],
        "talking_points": [
            "What is a SAR? A Suspicious Activity Report filed by financial institutions under the Bank Secrecy Act to report suspected money laundering to law enforcement.",
            "Why export structured JSON/XML? Modern FIUs ingest machine-readable reports for automated cross-bank network correlation."
        ]
    },
    "model_performance": {
        "title": "AI Model Benchmarking — Examiner Guide",
        "objective": "Rigorous, audited academic evaluation of all four machine learning architectures under a leakage-free chronological protocol.",
        "controls": [
            "Review the Comparative Benchmark Table: Compare XGBoost Champion against Random Forest, PyTorch RiskNet, and Logistic Regression.",
            "Highlight the Operational Selection Rationale Box: Explain why XGBoost is Champion despite Logistic Regression's high linear baseline score.",
            "Inspect the Multi-Threshold Operational Performance table across cutoffs 0.10 to 0.80.",
            "Show the Test Split Confusion Matrix Heatmap (1,878 TN, 0 FP, 5 FN, 35 TP).",
            "Show the SHAP Global Feature Importance bar chart.",
            "Explain the Late-Fusion Formula Banner: R_hybrid = 0.40(Tree) + 0.30(PyTorch) + 0.30(Graph)."
        ],
        "charts": [
            "Confusion Matrix Heatmap: Plotly visual heatmap showing zero false positives (0 FP) for XGBoost at standard threshold.",
            "SHAP Feature Importance Bar Chart: Quantifies global impact of features (Transaction Amount, Sender Mean Amount, Flow Velocity, etc.) on model log-odds."
        ],
        "talking_points": [
            "Why is PR-AUC superior to ROC-AUC here? With severe class imbalance (1:49 fraud ratio), ROC-AUC is misleadingly high because the large true negative count suppresses FPR. PR-AUC measures genuine precision-recall performance.",
            "Why is XGBoost chosen as Champion? XGBoost achieves 0.9959 PR-AUC and 0.00000 False Positive Rate, saving banks immense compliance review overhead.",
            "What is late fusion? Combining calibrated probabilities from specialized models (tree boosting, deep representation, graph topology) to eliminate single-model bias."
        ]
    }
}

def render_demo_guide(module_key: str):
    """
    Renders an optional, collapsible, beautifully styled demo walkthrough guide.
    Does NOT obstruct normal workflow.
    """
    guide = DEMO_GUIDES.get(module_key)
    if not guide:
        return

    # Check if user has toggled demo guide
    show_guide = st.session_state.get("toggle_demo_guide", st.session_state.get("show_demo_guide", True))
    if not show_guide:
        return

    with st.expander(f"🎓 B.Tech Defense & Live Demo Guide — {guide['title']}", expanded=False):
        st.markdown(f"""
            <div style="background-color: #0D1423; border: 1px solid #23334B; border-radius: 6px; padding: 1rem 1.25rem; font-size: 0.82rem; color: #E2E8F0; margin-bottom: 0.75rem;">
                <div style="font-family: 'Manrope', sans-serif; font-size: 0.95rem; font-weight: 700; color: #4C8DFF; margin-bottom: 0.3rem;">
                    🎯 Module Objective & Core Concept
                </div>
                <div style="color: #94A3B8; line-height: 1.5; margin-bottom: 0.85rem;">
                    {guide['objective']}
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 0.85rem;">
                    <div style="background-color: #121C2E; border: 1px solid #1E293B; border-radius: 6px; padding: 0.85rem;">
                        <b style="color: #F5B942; font-family: 'Manrope', sans-serif; font-size: 0.85rem;">🖱️ Recommended Controls to Demonstrate:</b>
                        <ul style="margin: 0.4rem 0 0 1.1rem; padding: 0; color: #94A3B8; font-size: 0.78rem; line-height: 1.5;">
                            {"".join(f"<li>{c}</li>" for c in guide['controls'])}
                        </ul>
                    </div>
                    <div style="background-color: #121C2E; border: 1px solid #1E293B; border-radius: 6px; padding: 0.85rem;">
                        <b style="color: #9270FF; font-family: 'Manrope', sans-serif; font-size: 0.85rem;">📊 What Visual Analytics Represent:</b>
                        <ul style="margin: 0.4rem 0 0 1.1rem; padding: 0; color: #94A3B8; font-size: 0.78rem; line-height: 1.5;">
                            {"".join(f"<li>{c}</li>" for c in guide['charts'])}
                        </ul>
                    </div>
                </div>

                <div style="background-color: rgba(39, 214, 155, 0.08); border-left: 3px solid #27D69B; border-radius: 0 6px 6px 0; padding: 0.75rem 1rem;">
                    <b style="color: #27D69B; font-family: 'Manrope', sans-serif; font-size: 0.85rem;">💡 Key Examiner Questions & Model Answers:</b>
                    <ul style="margin: 0.4rem 0 0 1.1rem; padding: 0; color: #E2E8F0; font-size: 0.78rem; line-height: 1.55;">
                        {"".join(f"<li>{tp}</li>" for tp in guide['talking_points'])}
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)
