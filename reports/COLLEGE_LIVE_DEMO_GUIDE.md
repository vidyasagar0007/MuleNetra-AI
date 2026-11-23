# MuleNetra AI — College Live Demonstration & Defense Guide

## Executive Overview
This guide provides a comprehensive, step-by-step walkthrough for demonstrating **MuleNetra AI — Hybrid AI-Powered Money Mule Network Detection and Financial Crime Investigation Platform** during a formal B.Tech final-year project viva or academic presentation.

The platform runs 100% locally on `http://localhost:8501` without requiring internet connectivity, external GPU resources, or paid third-party API keys.

---

## 1. Dataset Consistency & Integrity Explanation

### 1.1 The Source of Truth: 9,587 Transactions and 500 Accounts
Examiners may inquire about the exact dataset scale. The platform displays:
- **Total Monitored Transactions:** `9,587`
- **Banking Entity Nodes:** `500`

### 1.2 Mathematical Formulation of the Synthetic Dataset
The dataset is generated via [`src/data_loader.py`](file:///C:/Users/vidya/.gemini/antigravity/scratch/MuleTrace-AI/src/data_loader.py) using `generate_synthetic_aml_data()` with fixed seed `RANDOM_SEED = 42`:
1. **Benign Retail & Corporate Baseline:** `9,500` transactions across retail users and merchant accounts (including an 8% high-value legitimate business transaction subset between $12,000 and $65,000 to prevent amount-only bias).
2. **Synthetic AML Laundering Topologies:** `87` transactions systematically distributed across the 60-day window:
   - Sub-threshold structuring / smurfing clusters ($8,000 – $9,900)
   - Fan-in aggregation to money mule collectors
   - Multi-hop circular wash-trading loops ($A \rightarrow B \rightarrow C \rightarrow A$)
   - Rapid pass-through dissipation (turnaround under 6 hours)
   - Dormant account sudden activation
   - Layered multi-hop transit hops
   - Exit cash-outs via ATM and wire withdrawals
3. **Total Ledger Count:** $9,500 + 87 = \mathbf{9,587}$ transactions.

> **Examiner Note:** The environment telemetry card in the sidebar dynamically queries `len(context["scored_tx"])` (9,587) and `len(context["profiles"])` (500), guaranteeing absolute mathematical consistency across the entire user interface.

---

## 2. Guided Demo Experience Feature

To support students during viva evaluation without cluttering the operational workspace:
1. **Sidebar Toggle:** `🎓 Guided Defense Mode` can be toggled on/off in the left sidebar.
2. **Non-Obtrusive Expanders:** At the top of every module, an expandable banner titled `🎓 B.Tech Defense & Live Demo Guide — Examiner Guide` is provided.
3. **Structured 4-Part Defense Breakdown on Every Page:**
   - **Target Objective:** Core scientific concept and operational purpose.
   - **Recommended Controls to Click:** Exact buttons, sliders, and filters to demonstrate.
   - **Visual Analytics Representation:** Explanation of charts, axes, and color schemes.
   - **Examiner Key Questions & Model Answers:** Rapid talking points grounded in machine learning theory.

---

## 3. End-to-End Investigation Scenario: Target `ACC_0477`

The optimal college viva scenario focuses on **`ACC_0477`**, a high-volume **Collector Mule** aggregating dirty funds from smurfing nodes and rapidly channeling them to an exit cash-out hub.

```mermaid
flowchart LR
    subgraph Smurfs ["22 Feeder Accounts (Smurfing Ring)"]
        S1["ACC_0012 ($2,680)"]
        S2["ACC_0045 ($2,950)"]
        S3["ACC_0188 ($2,740)"]
        SN["... 19 Others"]
    end

    Target["ACC_0477 (Collector Mule)<br/>Risk: 0.850 CRITICAL<br/>Inflow: $59,082 (22 txns)<br/>Outflow: $53,892 (1 txn)"]

    Exit["ACC_0099 (Cash-Out Exit)<br/>Outflow Target"]

    S1 -->|Inflow| Target
    S2 -->|Inflow| Target
    S3 -->|Inflow| Target
    SN -->|Inflow| Target
    Target -->|Rapid Pass-Through (91.2% dissipated)| Exit
```

---

### Step-by-Step Live Demonstration Sequence (10–12 Minutes)

#### Step 1: High-Level Surveillance in Executive Command Center
- **Where to Navigate:** Click `📊 Executive Dashboard`.
- **What to Show:**
  1. The **6 spacious KPI cards** across two rows (9,587 transactions, 500 accounts, $34.42M volume, 389 flagged transactions, 84 critical mules, 1 network cluster).
  2. The **Account Risk Distribution Donut Chart** (79.2% low risk, 8.4% critical).
  3. The **Activity & Volume Over Time** chart showing volume bars and crimson anomaly spikes.
  4. The **Priority Investigation Queue** table: Point to the top-ranked account **`ACC_0477`** with risk index **`0.850`**.
- **Talking Point for Examiners:** *"In a real Tier-1 financial institution, investigators triage alerts by descending risk index. `ACC_0477` stands out with anomalous in-degree and high-volume flow."*

#### Step 2: Multi-Model Inference & Threshold Triage in AI Detection Center
- **Where to Navigate:** Click `⚡ AI Detection Center`.
- **What to Demonstrate:**
  1. In **Search Account ID**, type `ACC_0477`. The ledger immediately filters down to transactions involving `ACC_0477`.
  2. Point out the model predictions in the table:
     - XGBoost Anomaly Probability: $\sim 0.970$
     - PyTorch Neural Net Probability: $\sim 0.850$
     - Calibrated Hybrid Risk Score: $\sim 0.850$ (CRITICAL)
  3. Clear the search bar, then drag the **Operational Alert Threshold Sensitivity** slider from `0.50` to `0.60`.
- **Talking Point for Examiners:** *"Our platform lets compliance officers simulate alert queue triage. Raising the cutoff from 0.50 to 0.60 reduces investigator alert volume while maintaining zero false positive rate under XGBoost."*

#### Step 3: Graph Topological Intelligence in Mule Network Intelligence
- **Where to Navigate:** Click `🕸️ Network Intelligence`.
- **What to Demonstrate:**
  1. Point out the **Graph Telemetry Strip** (500 nodes, 49 collectors, 35 intermediaries).
  2. In **Visualization Target**, choose `Filter by Mule Role` and select `COLLECTOR`. The interactive digraph renders accounts exhibiting funneling behavior, with `ACC_0477` highlighted as a red square node.
  3. Point out the **Circular Wash-Trading Cycles Detected** table below showing closed transaction loops (e.g., `ACC_0014 → ACC_0025 → ACC_0024 → ACC_0014`).
- **Talking Point for Examiners:** *"Traditional relational databases fail to detect multi-hop relationships. NetworkX models the transaction ledger as a directed multigraph, allowing us to compute PageRank, clustering coefficients, and detect cyclic wash-trading in polynomial time."*

#### Step 4: 360° Forensic Profile in Account Forensic Investigation
- **Where to Navigate:** Click `🔍 Account Investigation`.
- **What to Demonstrate:**
  1. Target `ACC_0477` is pre-selected.
  2. Show the **Forensic Profile Banner**:
     - Risk Score: `0.850` (CRITICAL RISK TIER)
     - Topology Role: `COLLECTOR`
     - Cumulative Inflow: `$59,082.83` (from 22 counterparties)
     - Cumulative Outflow: `$53,892.72` (to 1 counterparty)
     - Retention Ratio: `8.8%` (91.2% rapid pass-through dissipation)
  3. Show the **Risk Attribution Waterfall Chart**:
     - Baseline Risk: `0.05`
     - ML Behavioral Contribution: `+0.34`
     - PyTorch Representation: `+0.25`
     - Graph Centrality Anomaly: `+0.23`
     - Final Score: `0.85`
  4. Show the **Triggered Forensic Rules Chips**:
     - `⚠️ Sub-threshold structuring detected`
     - `⚠️ Extreme fan-in counterparty ratio (22:1)`
     - `⚠️ Rapid turnover / capital dissipation within hours`
  5. Inspect the **Immediate Ego-Network Subgraph** and the **Audited Transaction History** table.
- **Talking Point for Examiners:** *"This provides Explainable AI (XAI). The waterfall chart directly deconstructs why the machine learning models and graph engine flagged this entity."*

#### Step 5: Automated AI Briefing & PDF Dossier in GenAI Assistant
- **Where to Navigate:** Click `🤖 AI Investigation Assistant`.
- **What to Demonstrate:**
  1. `ACC_0477` is loaded with the **Deterministic Offline Engine** (guaranteeing 100% offline uptime and zero external API dependencies).
  2. Review the structured 4-part brief:
     - *1. Executive Incident Summary*
     - *2. Factual Forensic Observations*
     - *3. Investigative Hypotheses (Smurf Aggregator Mule)*
     - *4. Recommended Compliance Actions (Enhanced Due Diligence & SAR Filing)*
  3. Click **"📄 Compile Official PDF Investigation Dossier"**. Within seconds, ReportLab compiles the formal PDF document.
  4. Click **"⬇️ Download PDF Dossier"** and open the generated file.
  5. In **Analyst Case Disposition**, select `ESCALATED_FOR_SAR`, type `"Escalated to FIU based on 22-feeder fan-in anomaly."`, and click **"Save Case Disposition"**.
- **Talking Point for Examiners:** *"Our assistant uses grounded prompt templates with deterministic fallback. All facts, amounts, and dates are injected directly from SQLite, eliminating hallucination risks."*

#### Step 6: Regulatory Export in Reports & Compliance Center
- **Where to Navigate:** Click `📑 Reports & Exports`.
- **What to Demonstrate:**
  1. Click **"⬇️ Download Transactions CSV"** to demonstrate raw data export.
  2. Show the **Generated PDF Forensic Dossiers Repository** where the previously generated report is archived.
  3. Under **Automated Regulatory SAR Generator**, select `ACC_0477`.
  4. Inspect the syntax-highlighted **FinCEN XML-JSON Regulatory Package** and click **"⬇️ Download Regulatory SAR Package (JSON)"**.
- **Talking Point for Examiners:** *"In modern AML systems, machine-readable SAR packages are transmitted directly to regulatory portals such as FinCEN or the national FIU."*

#### Step 7: Academic Benchmark Defense in Model Performance
- **Where to Navigate:** Click `📈 Model Performance`.
- **What to Demonstrate:**
  1. The **Comparative Benchmark Table**:
     - **XGBoost (Champion):** Precision `1.0000`, Recall `0.8750`, F1 `0.9333`, PR-AUC `0.9959`, FPR `0.00000`.
     - Baselines: Random Forest, PyTorch RiskNet, Logistic Regression.
  2. The **Operational Selection Rationale Box**:
     - Emphasize why XGBoost is Champion despite Logistic Regression's high linear baseline score (0 false alarms vs 2 false alarms; 0.9959 PR-AUC vs 0.9536 PR-AUC).
  3. The **Confusion Matrix Heatmap** ($1,878$ True Negatives, $0$ False Positives, $5$ False Negatives, $35$ True Positives).
  4. The **SHAP Global Feature Importance** bar chart.
  5. The **Audited Late Fusion Formula Banner**:
     $$\text{Hybrid Risk} = 0.40 \cdot P_{\text{TreeEnsemble}} + 0.30 \cdot P_{\text{PyTorchNet}} + 0.30 \cdot G_{\text{TopologyAnomaly}}$$
- **Talking Point for Examiners:** *"We use late fusion rather than early feature concatenation. By allowing XGBoost to focus on velocity features, PyTorch on non-linear representation, and NetworkX on graph topology, the ensemble eliminates single-model bias."*

---

## 4. Key Viva Questions & Bulletproof Model Answers

### Q1: Why did you build a synthetic dataset instead of using real banking data?
> **Model Answer:** *"Real-world financial transaction data is protected by strict banking secrecy laws (GLBA, GDPR) and cannot be published or shared publicly. Public datasets like Kaggle's PaySim lack realistic graph network topologies and circular wash-trading loops. Our generator simulates 9,587 transactions across 500 accounts incorporating seven verified AML patterns, while preserving class imbalance (~4%) and full chronological ordering to guarantee zero lookahead leakage."*

### Q2: Why is PR-AUC more important than ROC-AUC in this project?
> **Model Answer:** *"In fraud and money laundering detection, benign transactions outnumber laundering cases by nearly 50 to 1. ROC-AUC is sensitive to True Negative Rate; when True Negatives are vast, the False Positive Rate remains tiny even when the model makes dozens of false alarms. PR-AUC evaluates Precision directly against Recall without being distorted by high True Negatives, making it the industry standard for imbalanced fraud surveillance."*

### Q3: What is the purpose of PyTorch RiskNet alongside XGBoost?
> **Model Answer:** *"XGBoost splits data along orthogonal axes, making it exceptionally good at detecting threshold rules like currency structuring under $10,000. However, sophisticated money launderers adapt by varying amounts. PyTorch RiskNet learns a smooth, continuous 64-dimensional latent embedding across balances, flow ratios, and temporal intervals, detecting subtle non-linear behavioral signatures that evade rigid decision trees."*

### Q4: How does the Graph Engine prevent future data leakage?
> **Model Answer:** *"During historical feature engineering, graph centrality metrics are computed only over past transaction windows. In addition, graph features represent topology anomaly signals that are combined at late fusion rather than fed into the train-set tabular features, ensuring complete temporal isolation."*

### Q5: Is a high risk score legal proof of money laundering?
> **Model Answer:** *"No. MuleNetra AI is a research intelligence prototype designed to assist human compliance investigators. As stated in our academic disclaimer, risk scores are behavioral anomaly indicators that triage high-risk accounts for human review, Enhanced Due Diligence (EDD), and formal SAR consideration."*

---

## 5. Emergency Troubleshooting & Offline Execution

| Scenario | Immediate Fix |
| :--- | :--- |
| **Streamlit server not opening** | Open Windows PowerShell: `cd "C:\Users\vidya\.gemini\antigravity\scratch\MuleTrace-AI"`, run `.\venv\Scripts\Activate.ps1`, then `streamlit run app.py`. |
| **Port 8501 busy** | Run `streamlit run app.py --server.port 8502` and open `http://localhost:8502`. |
| **Data out of sync** | Click the sidebar button `🔄 Regenerate Dataset & Retrain`. This regenerates the SQLite database and retrains models cleanly in ~25 seconds. |
| **No internet in presentation room** | The platform is 100% self-contained. The AI Assistant runs locally via the deterministic engine, ReportLab compiles PDFs locally, and PyTorch executes on CPU. |
