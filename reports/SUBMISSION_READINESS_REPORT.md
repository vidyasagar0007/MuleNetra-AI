# MuleTrace AI — Final B.Tech Project Submission Readiness Report

**Project Title:** MuleTrace AI — Hybrid AI-Powered Money Mule Network Detection and Financial Crime Investigation Platform  
**Target Degree:** Bachelor of Technology (B.Tech) in Computer Science & Engineering / Artificial Intelligence  
**Academic Submission Date:** November 22, 2026  
**Repository Path:** `C:\Users\vidya\.gemini\antigravity\scratch\MuleTrace-AI`  
**Git Branch & Commit:** `master` (Commit: `13d4338`, Working Tree Clean)  
**Overall Readiness Verdict:** **100% READY FOR FINAL B.TECH SUBMISSION & VIVA DEFENSE**

---

## 1. Executive Summary & Verification Declaration

This document certifies that **MuleTrace AI** has undergone a comprehensive academic review, functional verification across all seven operational modules, multi-threshold model auditing, and full documentation assembly. The project meets all rigorous standards required for an advanced B.Tech capstone project.

### Core System Attributes Confirmed:
1. **Mathematical Soundness:** All evaluation metrics (Precision, Recall, F1, ROC-AUC, PR-AUC, FPR) have been empirically verified on an unseen chronological test set ($N=1,918$ transactions, $0.91\%$ natural laundering prevalence).
2. **Zero Data Leakage:** Scalers are fitted strictly on the training split ($t \le T_{80\%}$); sequential historical counters are computed chronologically without forward lookahead; graph edges are filtered via `max_timestamp`.
3. **Harmonized Late-Fusion Formulation:** Fuses tabular tree models (60% XGBoost, 40% Random Forest), deep neural representations (PyTorch RiskNet), and topological graph signals (NetworkX).
4. **Interactive Surveillance Platform:** Verified across all seven modules with custom fintech styling, zero non-functional controls, and responsive Plotly visualizations.
5. **Academic Integrity:** All documentation, UI footers, and export payloads explicitly acknowledge the synthetic nature of the dataset and clarify that risk scores represent investigative anomalies, not legal proof of guilt.

---

## 2. Verification of Seven Core Modules

| # | Application Module | Key Interactive Capabilities | Verification Status |
| :-: | :--- | :--- | :---: |
| **1** | **Executive Dashboard** | Real-time surveillance telemetry (9,587 transactions, 500 accounts, \$21.68M volume, 20 critical mules). Plotly risk tier donut, 60-day volume timeline with anomaly markers, and top alerts table. | **VERIFIED** |
| **2** | **AI Detection Engine** | Multi-model consensus stream (XGBoost, Random Forest, PyTorch, Hybrid score). Interactive operational threshold sensitivity slider ($0.05–0.95$), risk tier filter, amount range slider, and account search. | **VERIFIED** |
| **3** | **Network Intelligence** | Interactive Plotly 2D directed network visualizer with custom markers (■ Collector, ◆ Distributor, ⬢ Intermediary, ★ Exit Mule). Bounded DFS detected 20 circular wash trading loops. | **VERIFIED** |
| **4** | **Account Investigation** | 360° deep audit of any account (e.g., `ACC_0477`). Shows profile banner, cumulative inflow/outflow, retention ratio, TreeSHAP waterfall chart, ego-network subgraph, and chronological transaction ledger. | **VERIFIED** |
| **5** | **GenAI Assistant** | Grounded 4-part forensic narrative (Incident Summary, Factual Observations, Topology Hypotheses, Recommended Actions). Supports deterministic offline engine, Ollama, and Gemini. Compiles ReportLab PDF dossiers with SHA-256 hashes. | **VERIFIED** |
| **6** | **Reports & Exports** | Bulk CSV extractions for transactions, profiles, and clusters. PDF dossier repository with direct download links. Automated FinCEN SAR JSON generator with academic disclaimer. | **VERIFIED** |
| **7** | **Model Performance** | Out-of-time comparative benchmark table, multi-threshold operational matrix, confusion matrix heatmap, global SHAP feature importance bar chart, and late fusion architecture breakdown. | **VERIFIED** |

---

## 3. Audited Evaluation Results (Chronological Test Split, $N=1,918$)

### 3.1 Model Benchmark Comparison (Default Threshold = 0.50)

| Model Architecture | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | False-Positive Rate | Confusion Matrix `[[TN, FP], [FN, TP]]` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier (Champion)** | **1.0000** | 0.8750 | **0.9333** | 0.9999 | **0.9959** | **0.00000** | `[[1878, 0], [5, 35]]` |
| **Random Forest Baseline** | **1.0000** | 0.8750 | **0.9333** | **1.0000** | **1.0000** | **0.00000** | `[[1878, 0], [5, 35]]` |
| **Logistic Regression (Baseline)** | 0.9487 | **0.9250** | **0.9367** | 0.9854 | 0.9536 | 0.00106 | `[[1876, 2], [3, 37]]` |
| **PyTorch Compact RiskNet (DL)** | 0.9250 | **0.9250** | 0.9250 | 0.9746 | 0.9503 | 0.00160 | `[[1875, 3], [3, 37]]` |

### 3.2 Operational Alert Threshold Triage (XGBoost Champion)

| Cutoff Threshold | Precision | Recall (Detection Rate) | F1-Score | False-Positive Rate | True Alerts (TP) | False Alarms (FP) | Missed Frauds (FN) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.10** | 0.9744 | **0.9500** | 0.9620 | 0.00053 | 38 | 1 | 2 |
| **0.20** | **1.0000** | **0.9500** | **0.9744** | **0.00000** | 38 | 0 | 2 |
| **0.35** | **1.0000** | **0.9500** | **0.9744** | **0.00000** | 38 | 0 | 2 |
| **0.50** | **1.0000** | 0.8750 | 0.9333 | **0.00000** | 35 | 0 | 5 |
| **0.70** | **1.0000** | 0.7500 | 0.8571 | **0.00000** | 30 | 0 | 10 |
| **0.80** | **1.0000** | 0.7500 | 0.8571 | **0.00000** | 30 | 0 | 10 |

---

## 4. Test Suite & Code Quality Results

- **Automated Regression Suite:** `pytest -v` executed across unit, backend, and UI layers:
  - **Total Tests:** 23
  - **Passed:** **23 (100%)**
  - **Failed:** **0**
  - **Warnings:** **0**
  - **Duration:** 19.35 seconds
- **Bytecode Compilation Check:** `python -m compileall src pages tests app.py config.py`:
  - **Status:** Exit code 0 (100% compilation success with zero syntax errors).
- **Git Version Control Status:** Branch `master`, working tree clean, all files tracked under intentional commits.

---

## 5. Completed Academic Deliverables

All documentation has been compiled in the `reports/` directory and root workspace:

| Document | File Path | Scope & Description |
| :--- | :--- | :--- |
| **Comprehensive Project Report** | [reports/PROJECT_REPORT.md](file:///C:/Users/vidya/.gemini/antigravity/scratch/MuleTrace-AI/reports/PROJECT_REPORT.md) | Full 20-section academic B.Tech thesis covering literature review, math formulations, architecture, evaluation, and references. |
| **Project Presentation Outline** | [reports/PROJECT_PRESENTATION.md](file:///C:/Users/vidya/.gemini/antigravity/scratch/MuleTrace-AI/reports/PROJECT_PRESENTATION.md) | 15-slide presentation with slide titles, bullet points, suggested visuals, and word-for-word speaker notes. |
| **Live Demonstration Script** | [reports/LIVE_DEMO_SCRIPT.md](file:///C:/Users/vidya/.gemini/antigravity/scratch/MuleTrace-AI/reports/LIVE_DEMO_SCRIPT.md) | 7–10 minute step-by-step viva presentation walkthrough, `ACC_0477` case study, and emergency fallback protocols. |
| **System Setup & Run Guide** | [reports/PROJECT_SETUP_GUIDE.md](file:///C:/Users/vidya/.gemini/antigravity/scratch/MuleTrace-AI/reports/PROJECT_SETUP_GUIDE.md) | Windows PowerShell setup instructions, venv activation, dependencies, execution, and troubleshooting. |
| **Final Acceptance Audit Report** | [reports/FINAL_ACCEPTANCE_REPORT.md](file:///C:/Users/vidya/.gemini/antigravity/scratch/MuleTrace-AI/reports/FINAL_ACCEPTANCE_REPORT.md) | Detailed verification audit covering metric recalculation, confusion matrices, and hybrid risk formulas. |
| **Viva Defense Preparation (25 Q&As)** | [reports/VIVA_PREPARATION.md](file:///C:/Users/vidya/.gemini/antigravity/scratch/MuleTrace-AI/reports/VIVA_PREPARATION.md) | Exhaustive preparation guide featuring 25 anticipated viva examination questions with concise model answers. |
| **Audited Evaluation JSON Artifact** | [models/evaluation_summary.json](file:///C:/Users/vidya/.gemini/antigravity/scratch/MuleTrace-AI/models/evaluation_summary.json) | Machine-readable benchmark telemetry and multi-threshold matrix. |
| **Forensic PDF Dossier Sample** | [reports/MuleTrace_Forensic_Report_ACC_0099_20261120_185925.pdf](file:///C:/Users/vidya/.gemini/antigravity/scratch/MuleTrace-AI/reports/MuleTrace_Forensic_Report_ACC_0099_20261120_185925.pdf) | Sample generated audit-grade PDF investigation dossier. |
| **Project Architecture & README** | [README.md](file:///C:/Users/vidya/.gemini/antigravity/scratch/MuleTrace-AI/README.md) | Comprehensive GitHub repository documentation and architecture breakdown. |

---

## 6. Remaining Known Limitations

1. **Synthetic Data Regularity:** Although calibrated against IBM AMLSim and PaySim, synthetic transaction behaviors exhibit cleaner statistical separation than real-world human money laundering syndicates that actively probe and evade bank detection thresholds.
2. **In-Memory Graph Memory Scaling:** NetworkX calculates graph centralities and bounded cycles in RAM. Scaling to enterprise banking transaction volumes ($>10^7$ nodes) would necessitate migrating to a distributed graph database such as Neo4j or Amazon Neptune.
3. **Batch Evaluation Paradigm:** The platform processes transactions in chronological batches. Live banking deployment requires an Apache Kafka / Apache Flink event stream for sub-second payment scoring at authorization time.

---

## 7. Verified Launch Command

To launch the live platform for presentation:

```powershell
Set-Location "C:\Users\vidya\.gemini\antigravity\scratch\MuleTrace-AI"
& ".\venv\Scripts\Activate.ps1"
streamlit run app.py
```
*Application opens automatically at `http://localhost:8501`.*

---
*End of Submission Readiness Report — MuleTrace AI B.Tech Capstone Project (2026)*
