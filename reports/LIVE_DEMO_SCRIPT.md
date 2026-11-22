# MuleTrace AI — Live Demonstration Script & Viva Defense Guide

**Target Presentation Time:** 7 to 10 Minutes  
**Audience:** External Examiner, Project Review Committee, Academic Evaluators  
**Application URL:** `http://localhost:8501`  
**Prerequisites:** Streamlit server running via `streamlit run app.py`  
**Selected Target Account for Investigation:** `ACC_0477` (Critical Structuring & Collector Node)  

---

## Demonstration Timeline & Step-by-Step Walkthrough

```
[0:00 - 1:30] Phase 1: Executive Dashboard & Surveillance Overview
[1:30 - 3:00] Phase 2: AI Multi-Model Detection Center & Operational Threshold Tuning
[3:00 - 4:30] Phase 3: Mule Network Intelligence & Circular Wash Loop Detection
[4:30 - 6:30] Phase 4: Deep Account Forensic Interrogation (Target: ACC_0477)
[6:30 - 8:00] Phase 5: GenAI Forensic Briefing & Instant PDF Dossier Compilation
[8:00 - 9:00] Phase 6: Regulatory SAR Package & Telemetry Data Exports
[9:00 - 10:00] Phase 7: Audited Model Performance & Architectural Wrap-up
```

---

### Phase 1: Platform Introduction & Executive Dashboard (0:00 – 1:30)

#### Actions to Perform:
1. Open your browser at `http://localhost:8501`.
2. Ensure the top navigation radio button is selected on **"Executive Dashboard"**.
3. Point your cursor to the top banner and the six KPI cards across the screen.

#### What to Say to the Examiners:
> "Good morning, respected examiners. Welcome to **MuleTrace AI**, an enterprise-grade artificial intelligence platform developed for money mule detection and financial crime surveillance.
> 
> As you can see on the **Executive Dashboard**, our platform provides financial intelligence analysts with instant visibility across monitored transactions. 
> 
> Here in the KPI row, we are monitoring **9,587 transactions** across **500 accounts** totaling **\$21.68 Million in volume**. Our hybrid AI has flagged **87 transactions** as anomalous and escalated **20 accounts to the Critical risk tier**, with **75 in the High tier**. 
> 
> To the left, our Plotly donut chart visualizes the risk tier breakdown, demonstrating that over 80% of accounts remain low-risk benign baseline users. To the right, our dual-axis temporal chart plots transaction volume over our 60-day window, overlaid with red markers indicating coordinated suspicious bursts.
> 
> At the bottom, the **Priority Investigation Alerts** table automatically ranks the top flagged entities by their calibrated hybrid risk index. Notice our top candidate: account `ACC_0477`, flagged with an index of 0.852. We will investigate this account in detail shortly."

---

### Phase 2: AI Detection Center & Sensitivity Triage (1:30 – 3:00)

#### Actions to Perform:
1. In the left sidebar, click **"AI Detection Engine"**.
2. Show the multi-model stream columns: `xgb_risk_prob`, `rf_risk_prob`, `dl_risk_prob`, and `hybrid_risk_score`.
3. Locate the **"Operational Alert Threshold Sensitivity"** slider. Drag the slider from `0.50` down to `0.20`, then up to `0.70`.
4. Filter by Risk Tier: Select **"CRITICAL (>=0.75)"**.

#### What to Say to the Examiners:
> "Let us now navigate to the **AI Multi-Model Detection Center**. 
> 
> Unlike traditional monolithic models, MuleTrace AI displays calibrated predictions from our tree ensemble—XGBoost and Random Forest—alongside our PyTorch Deep RiskNet and the final fused hybrid score.
> 
> One of our core innovations is this **Operational Alert Threshold Sensitivity slider**. In real banking operations, compliance teams have varying alert capacities. 
> - At the standard cutoff of **0.50**, our model achieves **100% precision with zero false alarms**, flagging 35 frauds.
> - But watch what happens when I adjust the cutoff down to **0.20**: the alert queue expands to catch **95% of all frauds** (38 out of 40), while still maintaining **zero false positives** across the entire 1,878 non-fraud test transactions ($F_1 = 0.9744$).
> 
> When I filter by the **CRITICAL tier**, the data table isolates the most severe transactions, showing exact timestamps, structuring bands, and consensus probability scores."

---

### Phase 3: Mule Network Intelligence & Circular Loops (3:00 – 4:30)

#### Actions to Perform:
1. In the left sidebar, select **"Mule Network Intelligence"**.
2. Point out the summary metrics: **500 nodes**, **49 Collectors**, **35 Intermediaries**, and **20 circular cycles**.
3. Under *Visualization Target*, switch between **"Global Top Active Network"** and **"Filter by Mule Role"** (select **COLLECTOR**).
4. Scroll down to show the **Circular Wash-Trading Cycles Detected** table.

#### What to Say to the Examiners:
> "Now, let us examine the **Mule Network Intelligence** module. 
> 
> Traditional transaction monitoring fails because it cannot see network topology. Using NetworkX, MuleTrace AI constructs a directed multigraph of all 500 accounts and 9,032 transaction edges.
> 
> Our topological heuristic engine has automatically categorized accounts into specific mule roles:
> - **49 Collectors:** Nodes exhibiting high in-degree ($\ge 4$) and low out-degree ($\le 2$), consolidating small satellite deposits.
> - **35 Intermediaries:** Nodes where inflow and outflow closely balance with high throughput, acting as pass-through layering mules.
> 
> Notice this interactive Plotly 2D network diagram: Collectors are highlighted in red squares, Intermediaries in purple hexagons, and Distributors in orange diamonds.
> 
> Furthermore, below the network graph, our **Bounded-Depth DFS Algorithm** detected **20 circular wash trading cycles** ($A \rightarrow B \rightarrow C \rightarrow D \rightarrow A$). In money laundering, circular transfers are used to artificially inflate volume or disguise origin. Our bounded algorithm discovers these multi-hop loops in under 50 milliseconds without exponential recursion locks."

---

### Phase 4: Deep Account Forensic Interrogation — Case Study: `ACC_0477` (4:30 – 6:30)

#### Actions to Perform:
1. In the left sidebar, select **"Account Investigation"**.
2. In the *Select Target Account* dropdown (or search box), select or type **`ACC_0477`**.
3. Point out the **Target Profile Banner**, the **SHAP Waterfall Attribution Chart**, the **Triggered Indicators list**, and the **Ego-Network Subgraph**.
4. Scroll down to show the chronological transaction audit table for `ACC_0477`.

#### What to Say to the Examiners:
> "Now, let us conduct a deep forensic audit on a specific suspect: account **`ACC_0477`**.
> 
> Looking at the profile banner, `ACC_0477` has been assigned a **CRITICAL Risk Tier** with a calibrated score of **0.852**. The graph engine classifies it as a **Collector Mule**. Notice the financial footprint: it received **\$47,750 in cumulative inflow** across 5 counterparties with an almost 100% retention ratio.
> 
> To the left is our **TreeSHAP Risk Waterfall Chart**. This is our answer to the regulatory 'black-box' problem. The chart mathematically explains why this account was flagged:
> - The primary positive risk driver was `is_structuring_range` (+0.38), because this account received multiple transactions clustered between \$8,800 and \$9,950.
> - This was compounded by `sender_flow_velocity` (+0.24) and high incoming volume concentration (+0.18).
> 
> Below the waterfall, our rules engine flags: *'Sub-threshold structuring detected'* and *'Fan-In anomaly: Consolidating from multiple satellite accounts'*.
> 
> To the right, the **Ego-Network Subgraph** visualizes `ACC_0477` at the center, surrounded by its satellite feeder accounts. Below is the full chronological ledger of every associated transaction."

---

### Phase 5: GenAI Forensic Assistant & Instant PDF Dossier Compilation (6:30 – 8:00)

#### Actions to Perform:
1. In the left sidebar, select **"GenAI Assistant"**.
2. Ensure **`ACC_0477`** is selected. Keep the provider on **"Deterministic Offline Engine"**.
3. Point out the 4 structured cards:
   - *1. Executive Incident Summary*
   - *2. Factual Forensic Observations*
   - *3. Investigative Hypotheses & Topology*
   - *4. Recommended Compliance Actions*
4. Click the button: **"📄 Compile Official PDF Investigation Dossier"**.
5. Wait 2 seconds for compilation to complete, then click the green **"⬇️ Download PDF Investigation Dossier"** button.
6. Open the downloaded PDF in the browser or PDF viewer to show examiners the cover header, tables, and signature lines.

#### What to Say to the Examiners:
> "Now, let us proceed to the **GenAI Forensic Investigation Assistant**.
> 
> A common flaw in AI systems is LLM hallucination—inventing evidence that does not exist. MuleTrace AI prevents hallucinations by enforcing a **strictly grounded, retrieval-constrained reasoning architecture**.
> 
> The assistant extracts verified database records, SHAP feature weights, and graph centralities to assemble a 4-part forensic intelligence brief:
> 1. An **Executive Summary** synthesizing the incident.
> 2. **Factual Observations** listing verified inflow numbers and structuring counts.
> 3. **Investigative Hypotheses** identifying the suspected criminal pattern (in this case, smurfing consolidation).
> 4. **Recommended Compliance Actions**, such as freezing funds and issuing a FinCEN SAR.
> 
> Now, I will click **'Compile Official PDF Investigation Dossier'**. In just two seconds, ReportLab compiles an official audit-grade PDF document. 
> 
> Notice inside the PDF: it contains an institutional header, case tracking number, SHA-256 chain-of-custody verification hash, full transaction ledgers, SHAP attribution tables, and compliance officer signature blocks."

---

### Phase 6: Reports & Regulatory SAR Package Exports (8:00 – 9:00)

#### Actions to Perform:
1. In the left sidebar, select **"Reports & Exports"**.
2. Show the top three CSV download buttons (*Scored Transactions*, *Account Profiles*, *Network Clusters*).
3. Scroll down to the **Automated Regulatory SAR (Suspicious Activity Report) Generator**.
4. Select `ACC_0477` from the dropdown. Show the formatted JSON payload and click **"⬇️ Download Regulatory SAR Package (JSON)"**.

#### What to Say to the Examiners:
> "In the **Reports & Exports Center**, analysts can export all surveillance data with one click.
> 
> We provide bulk CSV exports for transactions, risk profiles, and graph clusters. 
> 
> Furthermore, we have built an **Automated Regulatory SAR Generator**. For high-risk entities like `ACC_0477`, it generates a standardized machine-readable JSON package formatted for submission to financial intelligence units like FinCEN. It packages subject entity identifiers, violation codes, cumulative inflows, and our academic research integrity disclaimer."

---

### Phase 7: Model Performance Benchmarking & Wrap-Up (9:00 – 10:00)

#### Actions to Perform:
1. In the left sidebar, select **"Model Performance"**.
2. Show the **Standard Comparative Benchmark** table.
3. Show the **Operational Selection Rationale Box**.
4. Show the **Confusion Matrix Heatmap** and **Global SHAP Feature Importances**.
5. Show the **Audited Multi-Modal Risk Fusion Architecture** box.

#### What to Say to the Examiners:
> "Finally, let us review our **Model Performance & Benchmarking** module.
> 
> All models were evaluated under a strict **chronological out-of-time test split** of 1,918 transactions:
> - Our champion **XGBoost Classifier** achieved **100% precision, 87.5% recall at default threshold ($F_1 = 0.9333$), zero false alarms ($FPR = 0.00000$), and a PR-AUC of 0.9959**.
> - At operational cutoff 0.20, it reaches **95% recall with 100% precision ($F_1 = 0.9744$)**.
> - Random Forest achieved identical precision and zero false positives.
> - PyTorch Deep RiskNet achieved **92.5% recall**, capturing deep representation interactions.
> 
> The confusion matrix confirms zero false positives for XGBoost, and the SHAP bar chart highlights transaction amount, historical mean, and flow velocity as the top predictive signals.
> 
> The platform is verified with a 23-test automated regression suite passing with zero warnings.
> 
> This concludes our demonstration of MuleTrace AI. Thank you, and I am now ready for your questions."

---

## Emergency Fallback Protocols During Live Viva

| Scenario / Issue | Instant Recovery Plan | What to Explain to Examiners |
| :--- | :--- | :--- |
| **No Internet Connection** | MuleTrace AI is 100% locally runnable. All models (XGBoost, PyTorch, NetworkX, SQLite) run entirely on offline CPU. In the GenAI Assistant, keep provider set to **"Deterministic Offline Engine"**. | *"Our platform was explicitly engineered to be cloud-independent and operational in secure, air-gapped financial intelligence enclaves."* |
| **Accidental Page Reload / Disconnect** | Simply refresh the browser tab (`F5`). All intelligence is cached via `@st.cache_resource` in memory and persists in SQLite. | *"Streamlit session caching preserves the initialized risk engine in memory, allowing instantaneous state recovery."* |
| **Examiner Asks for a Random Account** | In the Account Investigation module, use the search box to type any ID (e.g., `ACC_0099` for Intermediary layering, `ACC_0001` for benign retail). | *"The system indexes all 500 accounts dynamically; we can interrogate any entity instantly across the graph."* |
| **PDF Dossier Viewer Missing** | The compiled PDF is automatically stored on disk in `reports/` and can be opened in Adobe Acrobat or Chrome directly from the file explorer. | *"PDFs are compiled locally into the reports directory with simulated SHA-256 chain-of-custody hashes."* |

---
*End of Live Demonstration Script — MuleTrace AI B.Tech Capstone Defense (2026)*
