# MuleNetra AI — Final Year B.Tech Project Presentation Outline

**Project Title:** MuleNetra AI — Hybrid AI-Powered Money Mule Network Detection and Financial Crime Investigation Platform  
**Target Audience:** Academic Project Evaluation Committee, External Examiners, Faculty Guides  
**Format:** 15 Slides • Slide Titles, Key Bullet Points, Suggested Visuals, and Detailed Speaker Notes  

---

### Slide 1: Title & Project Overview
- **Title:** MuleNetra AI: Hybrid AI-Powered Money Mule Network Detection and Financial Crime Investigation Platform
- **Sub-Title:** A Unified Machine Learning, Deep Representation, and Graph Topological Framework for Financial Surveillance
- **Student Details:** Final Year B.Tech Project (Computer Science & Engineering / Artificial Intelligence)
- **Academic Year:** 2025–2026
- **Suggested Visuals:** High-tech fintech branding with the MuleNetra shield logo and dark navy background (`#0A0E17`).
- **Speaker Notes:**
  > "Good morning, respected members of the evaluation committee and external examiners. Today, I am proud to present 'MuleNetra AI'—an enterprise-grade artificial intelligence platform designed to detect complex money mule networks and automate financial crime investigations. In this presentation, I will walk you through our problem motivation, hybrid architecture, zero-leakage evaluation, and live demonstration."

---

### Slide 2: Problem Context — The Global Money Mule Crisis
- **UNODC Statistics:** \$800 Billion to \$2 Trillion laundered globally each year (2%–5% of global GDP).
- **Role of Money Mules:** Criminal syndicates recruit unwitting or complicit individuals (via job scams, phishing, romance fraud) to move illicit proceeds across banking rails.
- **Evading Detection:** Criminals deliberately bypass static thresholds using structuring, rapid pass-through, and circular laundering loops.
- **Suggested Visuals:** A diagram showing criminal origin -> money mule network -> upstream syndicate cash-out.
- **Speaker Notes:**
  > "Money mule networks are the lifeblood of modern financial crime, including cyber fraud, ransomware, and human trafficking. Criminal syndicates recruit intermediaries to receive and instantly move funds, breaking the audit trail. Because individual mule transactions frequently mimic retail behavior or stay just below reporting limits, traditional banking controls fail to detect them."

---

### Slide 3: Limitations of Existing Transaction Monitoring Systems (TMS)
- **High False-Positive Rates ($>95\%$):** Rule engines trigger on arbitrary amounts, creating severe alert fatigue and wasting compliance costs.
- **Isolated Transaction View:** Evaluates single records in relational databases with zero memory of account velocity or balance depletion.
- **Graph Topology Blindness:** Traditional databases cannot detect multi-hop wash cycles or satellite aggregation.
- **Black-Box Opacity:** Standalone deep learning models are rejected by regulators (FinCEN, FATF) due to a lack of defensible explanations.
- **Suggested Visuals:** Comparison graphic: Red 'Alert Fatigue' badge over stacked compliance logs vs. green 'Intelligent Prioritization'.
- **Speaker Notes:**
  > "Why do existing systems fail? Commercial banks rely on legacy rule engines that produce over 95% false positives. These systems look at transactions in total isolation—they cannot see account history or graph topology. Furthermore, when deep learning is proposed, compliance teams reject it because 'black-box' neural networks cannot provide the legally required explanation to regulators."

---

### Slide 4: Proposed Solution — The MuleNetra AI Architecture
- **Four Integrated Pillars:**
  1. *Supervised Tree Ensemble:* XGBoost Champion (`scale_pos_weight=108.9`) + Random Forest bagging baseline.
  2. *Deep Representation Learning:* PyTorch Compact RiskNet (BCE with pos_weight & batch normalization).
  3. *Graph Intelligence:* NetworkX directed topology, role heuristics, and bounded-depth cycle detection.
  4. *Explainable GenAI:* SHAP local/global attribution paired with grounded narrative briefs and PDF SAR packages.
- **Late-Fusion Hybrid Engine:** Mathematically calibrated risk index combining tabular, deep, and topological signals.
- **Suggested Visuals:** 4-pillar diagram converging into the Late-Fusion Hybrid Risk Engine.
- **Speaker Notes:**
  > "MuleNetra AI solves these challenges through a multi-modal hybrid architecture. Instead of relying on a single model, we combine tree ensembles for precise numerical boundaries, a PyTorch neural network for latent representations, a NetworkX graph engine for network topologies, and SHAP plus GenAI for regulatory explainability."

---

### Slide 5: System Architecture & Data Flow
- **5-Tier Operational Pipeline:**
  - *Tier 1: Data Ingestion & Sanitization* (Chronological 80/20 train/test separation).
  - *Tier 2: Sequential Feature Engineering* (22 domain features without lookahead).
  - *Tier 3: Multi-Model Inference & Graph Extraction* (XGBoost, RF, PyTorch, NetworkX).
  - *Tier 4: Late-Fusion Calibration & SHAP Attribution* (Transaction and account profiles).
  - *Tier 5: Presentation & Regulatory Reporting* (Streamlit dark UI, PDF dossier, SAR JSON).
- **Suggested Visuals:** Complete architectural flowchart from raw transactions to the 7 UI views.
- **Speaker Notes:**
  > "Here is our 5-tier architecture. Transactions flow sequentially through our pipeline. Crucially, our system maintains a strict chronological barrier: the training split occupies the first 80% time window, and the test split occupies the subsequent 20%. No test information is ever used in preprocessing, scaling, or graph construction."

---

### Slide 6: Dataset & Simulated Laundering Topologies
- **Realistic Dataset Parameters:** 9,587 transactions across 500 accounts over 60 days ($1,440\text{ hours}$). Natural prevalence: $0.91\%$ ($87$ laundering records).
- **Seven Generated Topologies:**
  1. *Structuring / Smurfing* (\$8.8k–\$9.95k bursts).
  2. *Fan-In (Collector Aggregation)*.
  3. *Fan-Out (Distributor Dispersal)*.
  4. *Rapid Pass-Through Movement* (95%–99% disbursed in 2–4 hours).
  5. *Circular Wash Trading* (Multi-node directed cycles).
  6. *Multi-Hop Layering Chains* (Sequential hops with fee deductions).
  7. *Dormant Account Activation* (>500 hours dormancy broken by high velocity).
- **Legitimacy Controls:** 8% high-value legitimate business wire transfers (\$12k–\$65k) to prevent trivial amount-based classification.
- **Suggested Visuals:** Network topology diagrams representing Fan-In, Fan-Out, and Circular Loops.
- **Speaker Notes:**
  > "To evaluate the system realistically, we generated 9,587 transactions modeled on IBM AMLSim distributions. We explicitly embedded seven distinct laundering topologies. Crucially, we also added legitimate high-value business transfers up to \$65,000. This ensures our AI cannot simply assume 'large amount equals fraud'—it must learn velocity, balance depletion, and structural relationships."

---

### Slide 7: Zero-Leakage Sequential Feature Engineering
- **22 Domain-Specific Features:**
  - *Transaction Signals:* `amount`, `log_amount`, `is_structuring_range`, `is_round_amount`, `hour_of_day`, `day_of_week`.
  - *Balance Dynamics:* `balance_diff_sender`, `balance_diff_receiver`, `balance_depletion_sender`.
  - *Sequential Historical Counters:* `sender_cum_count`, `sender_cum_amount`, `sender_mean_amount`, `sender_amount_ratio`.
  - *Velocity of Capital Flight:* `time_delta_sender_hours`, `sender_flow_velocity`, `time_delta_receiver_hours`.
  - *Transaction Types:* One-hot indicators for wire transfers, merchant payments, cash-outs, debits.
- **Leakage Prevention:** Cumulative metrics subtract the current transaction amount, preventing self-inclusion.
- **Suggested Visuals:** Table of key features with their mathematical formulas and fraud signals.
- **Speaker Notes:**
  > "We engineered 22 domain-specific features. A key contribution of our work is resolving 'historical amnesia'. Many academic projects calculate features separately on test data, wiping historical counters to zero. In MuleNetra AI, features are generated chronologically across the ordered stream, giving every test record its genuine prior history with strictly zero forward lookahead."

---

### Slide 8: Machine Learning & Deep Learning Implementation
- **Champion XGBoost Classifier:** Gradient boosted trees with `scale_pos_weight=108.9` for cost-sensitive minority gradient scaling.
- **Random Forest Baseline:** 120 bagging trees for variance reduction and out-of-distribution stability.
- **Logistic Regression Baseline:** Linear benchmark establishing the separability lower bound.
- **PyTorch Compact RiskNet:** 4-layer MLP ($22 \rightarrow 64 \rightarrow 32 \rightarrow 16 \rightarrow 1$) with `BatchNorm1d`, `LeakyReLU(0.1)`, `Dropout`, and `BCEWithLogitsLoss(pos_weight)`.
- **Suggested Visuals:** PyTorch network layer diagram alongside an XGBoost decision tree diagram.
- **Speaker Notes:**
  > "For model training, we implemented four architectures. XGBoost is our champion model, optimized with scale_pos_weight to penalize missed frauds 108 times more than false alarms. Alongside Random Forest and Logistic Regression baselines, we built a compact 4-layer PyTorch neural network that learns continuous latent representations of transaction velocity."

---

### Slide 9: Graph Intelligence & Mule Role Heuristics
- **NetworkX Directed Graph:** 500 nodes and 9,032 directed edges with transaction weights and frequencies.
- **Algorithmic Role Classification:**
  - *Collectors (Fan-In):* In-degree $\ge 4$, out-degree $\le 2$ (49 accounts detected).
  - *Distributors (Fan-Out):* Out-degree $\ge 4$, in-degree $\le 2$.
  - *Intermediaries (Relay):* In-degree $\ge 2$, out-degree $\ge 2$, outflow $\approx$ inflow (35 accounts detected).
  - *Exit Mules (Cash-Out):* In-degree $\ge 1$, out-degree $= 0$, terminal inflow $> \$12,000$.
- **Bounded-Depth DFS:** Identifies 20 circular wash loops ($\le 5$ hops) in under 50ms, avoiding exponential recursion locks.
- **Suggested Visuals:** Plotly 2D directed graph screenshot showing Collector, Intermediary, and Exit Mule nodes.
- **Speaker Notes:**
  > "Our graph intelligence engine constructs a directed multigraph of all transactions. Using topological heuristics, it automatically categorizes accounts into roles: Collectors consolidating funds, Intermediaries relaying money with near-zero retention, and Exit Mules cashing out. Furthermore, our bounded DFS algorithm detects 20 circular wash trading loops in milliseconds without freezing."

---

### Slide 10: Calibrated Hybrid Late-Fusion Risk Engine
- **Transaction-Level Late Fusion:**
  $$\text{Risk}_{\text{hybrid}}(\text{tx}) = 0.40 \cdot [0.60 \cdot P_{\text{XGBoost}} + 0.40 \cdot P_{\text{RandomForest}}] + 0.30 \cdot P_{\text{PyTorch}} + 0.30 \cdot \max(R_{\text{Graph}}(\text{sender}), R_{\text{Graph}}(\text{receiver}))$$
- **Account-Level Risk Profile:**
  $$\text{Risk}_{\text{account}} = 0.35 \cdot \text{PeakTxRisk} + 0.25 \cdot \text{MeanTxRisk} + 0.20 \cdot R_{\text{Graph}} + 0.20 \cdot \max(\text{PeakML}, \text{PeakDL})$$
- **Risk Tiers:** CRITICAL ($\ge 0.75$, 20 accounts), HIGH ($0.55–0.74$, 75 accounts), MEDIUM ($0.35–0.54$, 73 accounts), LOW ($< 0.35$, 332 accounts).
- **Suggested Visuals:** Visual formula breakdown showing weights: 40% Tree Ensemble, 30% PyTorch Neural Net, 30% Graph Topology.
- **Speaker Notes:**
  > "Rather than relying on one model, MuleNetra AI uses a late-fusion formula. 40% comes from the calibrated tree ensemble, 30% from PyTorch deep representations, and 30% from NetworkX topological risk. These signals are fused and clipped between 0.0 and 1.0, and aggregated into account-level risk profiles categorized into four actionable tiers."

---

### Slide 11: Quantitative Evaluation & Results
- **Evaluated on Unseen Test Split ($N=1,918$, 40 Positives, 1,878 Negatives):**
  - **XGBoost (Champion):** **1.0000 Precision**, **0.8750 Recall**, **0.9333 F1**, **0.9999 ROC-AUC**, **0.9959 PR-AUC**, **0.00000 FPR**.
  - **Random Forest:** **1.0000 Precision**, **0.8750 Recall**, **0.9333 F1**, **1.0000 PR-AUC**, **0.00000 FPR**.
  - **Logistic Regression:** 0.9487 Precision, **0.9250 Recall**, **0.9367 F1**, 0.9536 PR-AUC, 0.00106 FPR.
  - **PyTorch RiskNet:** 0.9250 Precision, **0.9250 Recall**, 0.9250 F1, 0.9503 PR-AUC, 0.00160 FPR.
- **Operational Matrix (XGBoost at Cutoff 0.20):** **1.0000 Precision**, **0.9500 Recall**, **0.9744 F1**, **0.00000 FPR** (38/40 Frauds caught, 0 false alarms).
- **Suggested Visuals:** Comparative metric bar chart and confusion matrix heatmap.
- **Speaker Notes:**
  > "Here are our audited out-of-time results. On 1,918 unseen test transactions, XGBoost achieved 100% precision and zero false alarms. At threshold 0.50, Logistic Regression has a slightly higher F1 because its balanced weighting gave it 92.5% recall at the cost of 2 false alarms. But look at PR-AUC: XGBoost achieves 0.9959 vs 0.9536. When tuned to cutoff 0.20, XGBoost achieves 95% recall with zero false positives, reaching an F1 of 0.9744."

---

### Slide 12: Explainable AI & GenAI Forensic Assistant
- **TreeSHAP Game-Theoretic Attribution:** Decomposes risk predictions into exact marginal feature contributions. Top global drivers: `amount`, `sender_mean_amount`, `log_amount`, `tx_type_TRANSFER`, `sender_flow_velocity`.
- **Local Waterfall Attributions:** Shows why individual accounts were flagged (e.g., structuring multiplier, balance depletion).
- **Grounded GenAI Briefing Engine:** Synthesizes a structured 4-part intelligence brief (Executive Summary, Observations, Hypotheses, Recommended Actions) using verified database facts with zero hallucination.
- **Regulatory Deliverables:** Compiles ReportLab PDF dossiers and FinCEN SAR JSON packages with simulated chain-of-custody hashes.
- **Suggested Visuals:** Screenshot of the Plotly SHAP waterfall chart and the generated PDF report page.
- **Speaker Notes:**
  > "To satisfy banking regulators, MuleNetra AI incorporates TreeSHAP explainability. Our waterfall charts show compliance officers the exact dollar and velocity factors that drove the risk score. Furthermore, our GenAI assistant synthesizes a grounded 4-part investigation brief and compiles an official PDF dossier and FinCEN SAR JSON package—with zero hallucinations."

---

### Slide 13: Live Demonstration of Platform Modules
- **Module 1:** Executive Dashboard (KPI metrics, risk donut, volume timeline, urgent alerts).
- **Module 2:** AI Detection Engine (Interactive threshold sensitivity slider, multi-model consensus stream).
- **Module 3:** Mule Network Intelligence (Plotly 2D directed graph, role filters, circular wash loops).
- **Module 4:** Account Investigation (Deep 360° audit, SHAP waterfall, counterparty ego-network).
- **Module 5:** GenAI Forensic Assistant (Narrative generator, offline fallback, PDF dossier compilation).
- **Module 6:** Reports & Exports (CSV telemetry downloads, FinCEN SAR JSON generator).
- **Module 7:** Model Performance (Multi-threshold matrix, confusion matrix, late fusion formula).
- **Suggested Visuals:** Multi-window grid showcasing screenshots of all seven platform modules.
- **Speaker Notes:**
  > "We have built a complete, fully interactive Streamlit web platform featuring seven specialized surveillance modules. In just a moment, I will demonstrate each of these live—including our interactive threshold tuning, 2D network visualizer, deep account investigation with SHAP attribution, and instant PDF dossier compilation."

---

### Slide 14: Limitations & Future Enhancements
- **Current Limitations:**
  - In-memory graph processing (NetworkX) requires migration to distributed graph databases for $>10^7$ nodes.
  - Synthetic data boundaries exhibit more regularity than adaptive human criminal syndicates.
  - Batch chronological evaluation vs real-time streaming payment rails.
- **Future Research Directions:**
  - *Graph Neural Networks (GNNs):* Inductive Relational Graph Convolutional Networks (RGCN).
  - *Federated Learning:* Cross-institutional money mule tracing without customer data exposure.
  - *Streaming Pipeline:* Integration with Apache Kafka and Apache Flink for sub-second payment scoring.
- **Suggested Visuals:** Roadmap timeline from current batch prototype to streaming federated enterprise deployment.
- **Speaker Notes:**
  > "We maintain strict scientific transparency regarding our project's limitations. While highly effective, our NetworkX graph runs in memory, and our pipeline processes chronological batches. Our future roadmap includes migrating to Graph Neural Networks, privacy-preserving federated learning across banking consortia, and Apache Kafka for real-time payment authorization scoring."

---

### Slide 15: Conclusion & Defense Summary
- **Core Engineering Achievements:**
  - Built an end-to-end, reproducible AML intelligence platform with zero data leakage.
  - Achieved **0.9959 PR-AUC**, **0.9744 F1**, and **0.00000 FPR** on out-of-time test data.
  - Fused tabular boosting, deep representations, and directed graph intelligence.
  - Solved black-box opacity through SHAP explainability and grounded GenAI briefs.
  - Verified with 23/23 passing automated tests and clean Git version control.
- **Academic Ethics:** Certified as an academic research prototype using synthetic AML data.
- **Thank You:** Open for Questions and Technical Defense.
- **Suggested Visuals:** Summary badge display: 'Zero Leakage Certified • 23/23 Tests Passed • 0.9959 PR-AUC'.
- **Speaker Notes:**
  > "In conclusion, MuleNetra AI bridges the critical gap between research-grade machine learning and operational financial crime compliance. It achieves exceptional detection performance with zero false alarms while providing full explainability and automated reporting. Thank you for your time and consideration, and I am now ready to answer your questions and demonstrate the platform."

---
*End of Presentation Outline — MuleNetra AI B.Tech Capstone Defense (2026)*
