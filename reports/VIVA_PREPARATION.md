# MuleTrace AI — Comprehensive B.Tech Final-Year Viva Preparation Guide

**Project Title:** MuleTrace AI — Hybrid AI-Powered Money Mule Network Detection and Financial Crime Investigation Platform  
**Target Degree:** Bachelor of Technology (B.Tech) in Computer Science / Artificial Intelligence / Data Science  
**Academic Year:** 2026  
**Audience:** B.Tech Student, Viva Defense Committee, Internal Guides, and External Examiners  

---

## 1. Project Overview & Problem Statement

### 1.1 Project Objective
Money mule networks represent one of the most critical conduits in financial crime, where recruitment networks (smurfs, unwitting victims, complicit intermediaries) are organized to launder illicit capital through rapid pass-through transfers, structuring, and circular wash trading.

**MuleTrace AI** is an advanced, locally runnable AI-powered financial intelligence and surveillance platform that:
1. Detects suspicious transactional anomalies using supervised machine learning and deep learning.
2. Uncovers multi-hop laundering topologies using directed graph intelligence.
3. Quantifies risk using a calibrated late-fusion hybrid scoring engine.
4. Explains individual model decisions using SHAP game-theoretic feature attribution.
5. Synthesizes evidence-grounded forensic briefs and PDF dossiers via a deterministic GenAI assistant.

### 1.2 The Research Gap
Traditional banking compliance relies on rule-based **Transaction Monitoring Systems (TMS)**:
- **High False-Positive Rates ($>95\%$):** Rule engines trigger alerts whenever a threshold is breached (e.g., amount $> \$10,000$), overwhelming human Financial Intelligence Units (FIUs).
- **Isolated Transaction View:** Rule engines evaluate individual transactions in isolation without contextual awareness of historical velocity, balance depletion, or multi-hop fund movement.
- **Topological Blindness:** Rule engines cannot detect graph-based money laundering patterns such as circular wash trading ($A \rightarrow B \rightarrow C \rightarrow A$) or fan-in/fan-out consolidation.
- **Lack of Explainability:** Many black-box deep learning proposals fail regulatory acceptance (FinCEN / EBA) because compliance officers cannot explain why an alert was flagged.

**MuleTrace AI solves this gap** by combining tabular tree ensembles (XGBoost + Random Forest), deep representation learning (PyTorch RiskNet), topological graph metrics (NetworkX), and post-hoc local explainability (SHAP).

---

## 2. End-to-End System Architecture

```
                                  [Raw Transactions / AMLSim]
                                              │
                                              ▼
                                 [Chronological Sorting & Split]
                                 ┌────────────┴────────────┐
                         80% Train Split            20% Test Split (Out-of-Time)
                                 │                         │
                                 ▼                         ▼
                        [Feature Engineering]      [Feature Engineering]
                      (22 Sequential Velocity)   (True Historical Counters)
                                 │                         │
            ┌────────────────────┼────────────────────┐    │
            ▼                    ▼                    ▼    ▼
     [XGBoost Champion]   [Random Forest]     [PyTorch RiskNet]
     (scale_pos_weight)  (balanced weights)  (BCE pos_weight)
            │                    │                    │
            └─────────┬──────────┘                    │
                      ▼                               │
              [Tree Ensemble] (40%)                   │ (30%)
                      │                               │
                      └──────────────┬────────────────┘
                                     │
                                     ├◄────── [NetworkX Directed Graph] (30%)
                                     ▼        (In/Out Degree, Cycles, Roles)
                          [Late-Fusion Risk Engine]
                                     │
                                     ▼
                        [Transaction & Account Profiles]
                                     │
         ┌──────────────┬────────────┼────────────┬──────────────┐
         ▼              ▼            ▼            ▼              ▼
    [Dashboard]    [Detection]   [Network]   [Account Deep]   [GenAI Assistant]
    (KPI & Vol)    (Sens. Cut)   (Plotly 2D) (SHAP Waterfall) (ReportLab PDF)
```

---

## 3. Dataset & Synthetic AML Topologies

### 3.1 Dataset Characteristics
- **Total Transactions:** 9,587 records across 500 accounts over a 60-day window ($1,440\text{ hours}$).
- **Prevalence:** 87 laundering transactions ($0.91\%$), representing extreme real-world class imbalance.
- **Academic Benchmark:** Model parameters benchmarked against IBM AMLSim and PaySim distributions.
- **Reproducibility:** Deterministic pseudo-random seeding (`seed=42`).

### 3.2 The Seven Generated Money Laundering Topologies
1. **Structuring / Smurfing:** Multiple deposits just below regulatory reporting thresholds (\$8,800 to \$9,950) into a target collector within compressed intervals.
2. **Fan-In (Collector Mule Aggregation):** Disparate satellite feeder accounts channeling capital into a single consolidation node.
3. **Rapid Pass-Through Movement:** High-velocity forward transfer where $95\%–99\%$ of incoming funds are disbursed within 2 to 4 hours.
4. **Fan-Out (Distributor Mule):** Central distributor dispersing aggregated sums to multiple downstream accounts.
5. **Circular Wash Trading:** Directed cyclic loops ($A \rightarrow B \rightarrow C \rightarrow D \rightarrow A$) executed to obscure original fund ownership.
6. **Multi-Hop Layering Chains:** Sequential linear hops ($A \rightarrow B \rightarrow C \rightarrow D \rightarrow E$) with fractional transaction fee deductions.
7. **Dormant Account Activation:** Accounts inactive for $>500$ hours suddenly executing high-value, high-velocity disbursements.

### 3.3 Anti-Cheating & Legitimacy Controls
To prevent models from relying on a trivial heuristic (such as "large amount = laundering"), the dataset incorporates **8% legitimate high-value business wire transfers** (\$12,000 to \$65,000). Benign amounts range from \$5.99 to \$64,919.11, ensuring the AI learns multi-variate velocity and structuring patterns rather than isolated dollar thresholds.

---

## 4. Feature Engineering (22 Domain Features)

Features are computed sequentially by timestamp to ensure strict zero-leakage compliance:

| Category | Features | Mathematical Intuition |
| :--- | :--- | :--- |
| **Transaction Level** | `amount`, `log_amount`, `is_structuring_range`, `is_round_amount`, `hour_of_day`, `day_of_week` | Captures structuring band (\$8k–\$10k), round criminal denominations (\$100, \$1k), and off-hours behavior. |
| **Balance Dynamics** | `balance_diff_sender`, `balance_diff_receiver`, `balance_depletion_sender` | Detects zero balance clearances ($\text{new\_balance} \le \$1.00$) following outgoing transfers. |
| **Sequential Velocity** | `sender_cum_count`, `sender_cum_amount`, `sender_mean_amount`, `sender_amount_ratio` | Cumulative historical volume and multiplier over historical sender average. Current transaction is subtracted to prevent self-leakage. |
| **Burst Time Intervals** | `time_delta_sender_hours`, `sender_flow_velocity`, `time_delta_receiver_hours` | Velocity of capital flight: $\frac{\text{Amount}}{\Delta t + 0.1\text{ hrs}}$. High values identify rapid pass-through spikes. |
| **Transaction Types** | `tx_type_TRANSFER`, `tx_type_PAYMENT`, `tx_type_CASH_OUT`, `tx_type_DEBIT` | One-hot encodings capturing wire transfer preferences for mule accounts. |

---

## 5. Machine Learning & Deep Learning Models

### 5.1 Champion Model: XGBoost Classifier
- **Hyperparameters:** `n_estimators=150`, `max_depth=5`, `learning_rate=0.06`, `subsample=0.85`, `colsample_bytree=0.85`, `gamma=1.0`.
- **Class Imbalance Strategy:** `scale_pos_weight = float(n_neg / n_pos) = 108.9`.
- **Performance:** **Precision: 1.0000**, **Recall: 0.8750**, **F1: 0.9333**, **PR-AUC: 0.9959**, **ROC-AUC: 0.9999**, **FPR: 0.00000**.
- **Role:** Primary champion model capturing non-linear feature interactions and structuring thresholds.

### 5.2 Baseline 1: Random Forest Classifier
- **Hyperparameters:** `n_estimators=120`, `max_depth=7`, `min_samples_split=4`.
- **Class Imbalance Strategy:** `class_weight='balanced'`.
- **Performance:** **Precision: 1.0000**, **Recall: 0.8750**, **F1: 0.9333**, **PR-AUC: 1.0000**, **ROC-AUC: 1.0000**, **FPR: 0.00000**.
- **Role:** Bagging tree baseline blended with XGBoost (60/40) for variance reduction.

### 5.3 Baseline 2: Logistic Regression (Linear Baseline)
- **Hyperparameters:** `class_weight='balanced'`, `max_iter=1000`.
- **Performance:** **Precision: 0.9487**, **Recall: 0.9250**, **F1: 0.9367**, **PR-AUC: 0.9536**, **ROC-AUC: 0.9854**, **FPR: 0.00106**.
- **Role:** Establishes the linear separability lower-bound.

### 5.4 Deep Learning: PyTorch Compact RiskNet
- **Architecture:** 4-layer fully connected MLP: $\text{Linear}(22 \rightarrow 64) \rightarrow \text{BatchNorm1d} \rightarrow \text{LeakyReLU}(0.1) \rightarrow \text{Dropout}(0.25) \rightarrow \text{Linear}(64 \rightarrow 32) \rightarrow \text{BatchNorm1d} \rightarrow \text{LeakyReLU}(0.1) \rightarrow \text{Dropout}(0.15) \rightarrow \text{Linear}(32 \rightarrow 16) \rightarrow \text{Linear}(16 \rightarrow 1)$.
- **Optimization:** AdamW optimizer (`lr=0.001`, `weight_decay=1e-4`), `CosineAnnealingLR` scheduler, 20 epochs.
- **Loss Function:** `nn.BCEWithLogitsLoss(pos_weight=pos_weight)`.
- **Performance:** **Precision: 0.9250**, **Recall: 0.9250**, **F1: 0.9250**, **PR-AUC: 0.9503**, **ROC-AUC: 0.9746**, **FPR: 0.00160**.
- **Role:** Extracts continuous latent representations to detect complex anomalies missed by tree splits.

---

## 6. Graph Intelligence Engine (NetworkX)

- **Graph Structure:** Directed Multigraph $G = (V, E)$ where nodes $V$ represent bank accounts and directed edges $E$ represent capital transfers with transaction weights and timestamps.
- **Mule Role Heuristics:**
  - **Collector (Fan-In Mule):** $\text{In-Degree} \ge 4$, $\text{Out-Degree} \le 2$ (Aggregates structuring smurf deposits).
  - **Distributor (Fan-Out Mule):** $\text{Out-Degree} \ge 4$, $\text{In-Degree} \le 2$ (Disburses capital to downstream nodes).
  - **Intermediary (Layering Mule):** $\text{In-Degree} \ge 2$, $\text{Out-Degree} \ge 2$, $0.82 \le \frac{\text{Outflow}}{\text{Inflow}} \le 1.18$ (Rapid pass-through with minimal retention).
  - **Exit Cash-Out Mule:** $\text{In-Degree} \ge 1$, $\text{Out-Degree} = 0$, $\text{Inflow} > \$12,000$ (Terminal withdrawal node).
- **Circular Cycle Detection:** Bounded-depth Depth-First Search (`max_cycle_length=5`) identifies wash-trading loops ($A \rightarrow B \rightarrow C \rightarrow D \rightarrow A$) while avoiding exponential recursion locks.
- **Temporal Leakage Prevention:** Built with `max_timestamp` filtering to ensure historical graph evaluations do not include future edges.

---

## 7. Hybrid Risk Scoring Formulation

### 7.1 Transaction Risk Formula
$$\text{Risk}_{\text{hybrid}}(\text{tx}) = 0.40 \cdot [0.60 \cdot P_{\text{XGBoost}} + 0.40 \cdot P_{\text{RandomForest}}] + 0.30 \cdot P_{\text{PyTorch}} + 0.30 \cdot \max(R_{\text{Graph}}(\text{sender}), R_{\text{Graph}}(\text{receiver}))$$
- Bounded in $[0.0, 1.0]$ via `np.clip()`.

### 7.2 Account Profile Risk Formula
$$\text{Risk}_{\text{account}} = 0.35 \cdot \text{PeakTxRisk} + 0.25 \cdot \text{MeanTxRisk} + 0.20 \cdot R_{\text{Graph}} + 0.20 \cdot \max(\text{PeakML}, \text{PeakDL})$$
- Tiers: **CRITICAL** ($\ge 0.75$), **HIGH** ($0.55–0.74$), **MEDIUM** ($0.35–0.54$), **LOW** ($< 0.35$).

---

## 8. Explainability & GenAI Assistant

### 8.1 SHAP (SHapley Additive exPlanations)
- **TreeExplainer:** Computes mathematically exact additive feature attributions based on cooperative game theory.
- **Attribution Property:**
  $$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i(x)$$
  where $\phi_0$ is the expected dataset base value and $\phi_i(x)$ is the marginal contribution of feature $i$.
- **Visuals:** Plotly horizontal global feature importances and local additive waterfall plots for individual target accounts.

### 8.2 GenAI Forensic Briefing Engine
- **Architecture:** 4-part structured intelligence brief:
  1. *Executive Incident Summary*
  2. *Factual Forensic Observations* (strictly grounded in verified database telemetry)
  3. *Investigative Hypotheses & Topology* (distinguished clearly from observations)
  4. *Recommended FIU Actions* (regulatory filing playbook)
- **Zero Hallucination Protocol:** Deterministic offline reasoning engine backed by concrete database queries, with optional Ollama (Llama 3.2) or Gemini API integration.
- **Export Engine:** Compiles official ReportLab PDF dossiers with simulated SHA-256 chain-of-custody hashes and FinCEN SAR JSON packages.

---

## 9. 25 Likely Viva Questions & Model Answers

### Category 1: Project Motivation & Domain Knowledge

#### Q1: What is a money mule, and why can't conventional banking rules catch them?
> **Answer:** A money mule is an individual whose bank account is used by criminal syndicates to receive and quickly transfer illicit funds, thereby laundering the money and breaking the audit trail. Traditional banking rule engines evaluate each transaction in isolation against simple static thresholds (e.g., amount $> \$10,000$). Money mules deliberately evade these rules through **structuring** (smurfing transfers below \$10,000), **rapid pass-through dispersal**, and **multi-hop layering chains**, which require sequential velocity and graph relationship intelligence to detect.

#### Q2: What is the difference between Placement, Layering, and Integration in AML?
> **Answer:** 
> - **Placement:** Introducing illicit cash into the legitimate financial system (e.g., structuring multiple small cash deposits).
> - **Layering:** Distance the funds from the criminal origin through a complex web of transactions, wire transfers, shell companies, and pass-through mule hops.
> - **Integration:** Returning the laundered funds to the criminal as seemingly legitimate business revenue or clean investment capital. MuleTrace AI focuses primarily on the **Placement and Layering** stages.

#### Q3: What is "Structuring" (or "Smurfing")?
> **Answer:** Structuring is the intentional practice of breaking down a large sum of money into multiple smaller transactions—typically just under regulatory reporting thresholds (e.g., \$8,800 to \$9,950 to evade the FinCEN \$10,000 Currency Transaction Report threshold). MuleTrace AI detects this via features like `is_structuring_range` and sequential transaction count grouping.

---

### Category 2: Machine Learning & Evaluation

#### Q4: Why did you use an out-of-time chronological train/test split instead of random k-fold cross-validation?
> **Answer:** Random splitting in time-series transaction data causes severe **data leakage (lookahead bias)**. If a model trains on future transactions and tests on past transactions, it unrealistically "remembers" account behavioral shifts that haven't occurred yet. Our chronological split (first 80% time window for training, subsequent 20% for testing) ensures the model is evaluated strictly on future unseen transactions, replicating real-world banking deployment.

#### Q5: How did you handle the severe class imbalance in your dataset (0.91% positive)?
> **Answer:** We handled class imbalance through algorithm-level loss weighting without distorting the underlying feature distribution:
> 1. In XGBoost, we set `scale_pos_weight = N_neg / N_pos = 108.9`, which scales the gradient of the minority positive class.
> 2. In Random Forest and Logistic Regression, we used `class_weight='balanced'`.
> 3. In PyTorch, we used `nn.BCEWithLogitsLoss(pos_weight=pos_weight)`.

#### Q6: Why is ROC-AUC alone misleading for fraud detection, and why did you prioritize PR-AUC?
> **Answer:** ROC-AUC plots True Positive Rate vs False Positive Rate. When the negative class is massive ($99.09\%$), the False Positive Rate ($\frac{FP}{FP + TN}$) remains tiny even when hundreds of false alarms occur, inflating the ROC-AUC score. PR-AUC (Precision-Recall AUC) directly plots Precision ($\frac{TP}{TP + FP}$) against Recall, giving an honest assessment of how many triggered alerts are actual fraud without being flattered by large true negative counts.

#### Q7: Why does Logistic Regression have a higher reported F1-score (0.9367) than XGBoost (0.9333) at threshold 0.50?
> **Answer:** Logistic Regression was trained with `class_weight='balanced'`, which scales the minority loss by $\sim 163\times$, aggressively pushing its uncalibrated decision probabilities higher. At the arbitrary 0.50 threshold, it catches 37/40 positives (92.5% recall) but incurs **2 false alarms** ($FPR = 0.00106$). XGBoost produces sharper, better-calibrated probabilities, catching 35/40 with **zero false alarms** ($Precision = 1.0000, FPR = 0.00000$). Because F1 is the harmonic mean, Logistic Regression's recall boost yields a slightly higher F1 at 0.50. However, XGBoost has a vastly superior **PR-AUC (0.9959 vs 0.9536)** and, when tuned to operational threshold 0.20, achieves **F1 = 0.9744** (95.0% recall, 100% precision, 0 false alarms).

#### Q8: What explicit criteria did you use to designate XGBoost as the "Champion Model"?
> **Answer:** We defined an explicit 4-part operational compliance criterion:
> 1. **PR-AUC Dominance:** 0.9959 (XGBoost) vs 0.9536 (Logistic Regression).
> 2. **False Positive Suppression:** In compliance, false alarms cost \$40–\$150 in analyst labor. XGBoost delivers $FPR = 0.00000$ at cutoffs $\ge 0.20$.
> 3. **Operational Threshold Tunability:** At cutoff 0.20, XGBoost achieves 95.0% recall and 100% precision ($F1 = 0.9744$).
> 4. **TreeSHAP Explainability:** Exact, regulatory-compliant feature attribution via TreeExplainer.

#### Q9: What was the threshold 0.80 inconsistency and how did you resolve it?
> **Answer:** In early drafting, threshold 0.80 was incorrectly listed with Recall=0.8750 and F1=0.9211. Re-running the actual model predictions revealed that at threshold 0.80, the model flags 30 True Positives, 0 False Positives, and 10 False Negatives. This yields $\text{Precision} = 1.0000$, $\text{Recall} = 0.7500$, and mathematically $F_1 = \frac{2 \times 1.0 \times 0.75}{1.75} = \mathbf{0.8571}$. The discrepancy was corrected across all documentation and artifacts.

---

### Category 3: Deep Learning & Neural Architectures

#### Q10: What is the architecture of your PyTorch neural network?
> **Answer:** PyTorch `MuleRiskNet` is a 4-layer fully connected Multi-Layer Perceptron (MLP) with dimensions $22 \rightarrow 64 \rightarrow 32 \rightarrow 16 \rightarrow 1$. It incorporates `BatchNorm1d` for internal covariate shift stabilization, `LeakyReLU(negative_slope=0.1)` to prevent dying neurons, and `Dropout` (0.25 and 0.15) to prevent overfitting on the imbalanced training split.

#### Q11: Why did you use `BCEWithLogitsLoss` instead of `BCELoss` with a Sigmoid layer in PyTorch?
> **Answer:** `nn.BCEWithLogitsLoss` combines a Sigmoid activation and Binary Cross Entropy in a single class using the **log-sum-exp trick**. This provides superior numerical stability compared to using a separate `nn.Sigmoid()` followed by `nn.BCELoss()`, which can suffer from catastrophic floating-point underflow or overflow when probabilities approach 0.0 or 1.0.

#### Q12: Why use both Deep Learning and Tree Ensembles? Isn't one enough?
> **Answer:** Tree ensembles (XGBoost/Random Forest) excel at discrete step-function boundaries (e.g., amounts between \$8,800 and \$9,950 or $\text{balance} \le \$1.00$). Deep neural networks excel at continuous latent representation learning across normalized velocity interactions. Fusing both via late-fusion combines the precise boundary cuts of trees with the smooth embedding manifolds of neural nets, yielding higher overall resilience against adversarial evasion.

---

### Category 4: Graph Intelligence & Topologies

#### Q13: How does your graph engine model transaction relationships?
> **Answer:** We construct a directed multigraph $G = (V, E)$ using NetworkX. Nodes $V$ represent bank account entities. Directed edges $(u, v) \in E$ represent fund transfers from sender $u$ to receiver $v$. Edge weights store cumulative transaction amounts, while attributes track transaction frequencies and timestamps.

#### Q14: How do you identify Collector and Intermediary mules using graph metrics?
> **Answer:**
> - **Collector Mules (Fan-In):** Detected via high in-degree ($\ge 4$) and low out-degree ($\le 2$), indicating consolidation of small satellite deposits.
> - **Intermediary Mules (Pass-Through):** Detected when in-degree and out-degree are balanced ($\ge 2$ each) and the net flow ratio ($\frac{\text{Outflow}}{\text{Inflow}}$) is near 1.0 ($0.82 \le \text{ratio} \le 1.18$), indicating rapid onward transmission with negligible capital retention.

#### Q15: Why did you implement bounded-depth DFS for circular cycle detection instead of NetworkX `simple_cycles`?
> **Answer:** NetworkX `simple_cycles()` uses Johnson's algorithm, which has $O((V + E)(c + 1))$ exponential time complexity. In a dense graph with thousands of edges, searching for unbounded cycles leads to an exponential recursion lock that freezes the system. In money laundering, circular wash loops are rarely longer than 3 to 5 hops ($A \rightarrow B \rightarrow C \rightarrow D \rightarrow A$). Our bounded-depth DFS restricts the search to cycles of length $\le 5$, completing the scan in milliseconds.

#### Q16: How did you ensure graph construction did not introduce temporal data leakage?
> **Answer:** In early iterations, building the graph over the entire dataset allowed transactions at Day 55 to create edges visible to transactions at Day 10. We remediated this by adding a `max_timestamp` filter to `GraphEngine.build_graph()`. When evaluating historical transactions, only edges timestamped prior to that cutoff are included in the graph.

---

### Category 5: Risk Fusion & Explainability

#### Q17: What is the exact formula for transaction-level hybrid risk?
> **Answer:**
> $$\text{Risk}_{\text{hybrid}}(\text{tx}) = 0.40 \cdot [0.60 \cdot P_{\text{XGBoost}} + 0.40 \cdot P_{\text{RandomForest}}] + 0.30 \cdot P_{\text{PyTorch}} + 0.30 \cdot \max(R_{\text{Graph}}(\text{sender}), R_{\text{Graph}}(\text{receiver}))$$
> All components are normalized between $[0.0, 1.0]$, and the final fused risk is clipped to $[0.0, 1.0]$.

#### Q18: What is SHAP, and how does it explain an individual transaction's risk score?
> **Answer:** SHAP (SHapley Additive exPlanations) is a game-theoretic approach that calculates the fair marginal contribution of each feature to the model's output compared to a baseline expectation:
> $$f(x) = \phi_0 + \sum_{i=1}^M \phi_i(x)$$
> In MuleTrace AI, TreeSHAP reveals exactly which features pushed an account's risk score up (e.g., $+0.42$ due to `is_structuring_range`, $+0.21$ due to `sender_flow_velocity`) or down (e.g., $-0.12$ due to high historical balance), satisfying regulatory explainability mandates.

#### Q19: How did you ensure that tests do not overwrite saved production models?
> **Answer:** We added a `save_artifacts: bool = True` parameter to `train_and_evaluate()` in both `MLModelEngine` and `DeepLearningEngine`. In pytest fixtures, `save_artifacts=False` is passed, ensuring that temporary test runs with small 500-row sample datasets never overwrite the production model weights (`xgboost_model.joblib`) or the audited `models/evaluation_summary.json`.

---

### Category 6: GenAI, Software Architecture & Academic Ethics

#### Q20: How does your GenAI Assistant generate investigation briefs without hallucinating facts?
> **Answer:** The GenAI Assistant uses a strictly **grounded, retrieval-constrained reasoning architecture**:
> 1. It queries actual SQLite transaction records, graph centralities, triggered heuristic rules, and SHAP feature attributions for the target account.
> 2. It compiles these verified facts into structured JSON context.
> 3. It utilizes a deterministic offline template engine as its primary baseline, ensuring 100% factually grounded text with zero hallucinations even without an API key or internet access.
> 4. It strictly segregates "Factual Forensic Observations" from "Investigative Hypotheses".

#### Q21: What regulatory formats does your platform support?
> **Answer:** MuleTrace AI generates:
> 1. Courtroom-ready ReportLab **PDF Investigation Dossiers** complete with case metadata, account summary tables, triggered rules, and cryptographic verification hash signatures.
> 2. Standardized machine-readable **FinCEN / FIU Suspicious Activity Report (SAR) JSON packages** containing structured subject entity profiles, violation codes, and audit timestamps.

#### Q22: What software architecture and design patterns did you employ?
> **Answer:** 
> - **Modular Service Architecture:** Clean separation of concerns between data loading, preprocessing, feature engineering, ML models, graph analysis, risk fusion, database ORM, and UI views.
> - **Repository / ORM Pattern:** SQLAlchemy for structured storage and queries.
> - **Factory / Strategy Pattern:** Swappable GenAI providers (Offline, Ollama, Gemini).
> - **Reactive UI Caching:** Streamlit `@st.cache_resource` ensuring single-load model inference across pages.

#### Q23: Why did your previous Git status command fail with exit code 1?
> **Answer:** The initial repository directory had not yet been initialized with `git init`. Once `git init` was executed and all project files were staged with `.gitignore` exclusion rules, the initial commit `c2e661c` succeeded. `git status` currently exits with code 0: `On branch master, nothing to commit, working tree clean`.

#### Q24: What are the main limitations of your project?
> **Answer:**
> 1. **Synthetic Data Bounds:** While benchmarked against AMLSim and PaySim, synthetic patterns have more regular behavioral boundaries than human laundering networks that actively adapt.
> 2. **In-Memory Graph Scalability:** NetworkX processes the graph in RAM. Scaling to hundreds of millions of bank transactions would require a distributed graph database like Neo4j.
> 3. **Batch vs Real-Time Streaming:** The prototype processes batch transaction windows. A production banking deployment would require an Apache Kafka / Apache Flink streaming pipeline.

#### Q25: What is the academic integrity disclaimer for MuleTrace AI?
> **Answer:** MuleTrace AI is an academic research prototype. The dataset is entirely synthetic, and all risk scores and role classifications indicate behavioral anomalies requiring human investigator review—they do not constitute confirmed legal proof of criminal activity. Generated SAR JSON payloads and PDF dossiers are academic simulation artifacts, not officially filed regulatory declarations.

---

*End of Viva Preparation Guide — MuleTrace AI Final Year Capstone Project (2026)*
