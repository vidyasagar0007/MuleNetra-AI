# MuleNetra AI — Enterprise UI/UX Redesign Documentation

## Executive Summary

**MuleNetra AI** has undergone an end-to-end design transformation from a functional prototype into an institutional-grade financial intelligence, money mule network detection, and anti-money-laundering (AML) investigation platform.

This redesign establishes a cohesive, restrained visual system inspired by tier-1 institutional analytics, cyber threat intelligence, and defense-grade surveillance operations. All backend capabilities, ML/DL models, NetworkX graph intelligence, SHAP explainability calculations, and regulatory export engines were **100% preserved** with zero data fabrication or architectural regressions.

---

## 1. Design System & Visual Identity

### 1.1 Color Architecture
The interface uses a calibrated dark color palette engineered for long-duration investigator workflows:

| Token Name | Hex Code | Visual Application & Semantic Role |
| :--- | :---: | :--- |
| **Primary Canvas** | `#080D18` | Deep midnight navy background providing high contrast and minimal eye fatigue. |
| **Secondary Background** | `#0D1423` | Sidebar and nested component background, creating clear structural depth. |
| **Elevated Surface** | `#121C2E` | Card containers, interactive panels, chart canvases, and data grids. |
| **Higher Surface / Hover** | `#162238` | Interactive button states, elevated dropdown menus, and active table rows. |
| **Subtle Border** | `#23334B` | Structural dividers, card borders, and grid delineators. |
| **Border Hover** | `#354B6E` | Hover feedback boundary across cards and inputs. |
| **Electric Blue (Primary)** | `#4C8DFF` | Primary action buttons, active navigation indicator, and ML probability signals. |
| **Secondary Violet** | `#9270FF` | Deep learning indicators, graph intermediary nodes, and ensemble badges. |
| **Success Emerald** | `#27D69B` | Low-risk accounts, incoming capital flows, and positive health statuses. |
| **Warning Amber** | `#F5B942` | High-risk thresholds, distributor mules, and elevated queue alerts. |
| **Critical Red** | `#FF5B6E` | Critical risk tier, money mule collectors, wash-trading loops, and urgent alerts. |

### 1.2 Typography Hierarchy
The typography system pairs geometric clarity with technical precision:

- **Headings & Structural Titles (`Manrope`):** Modern geometric sans-serif with tight tracking (`letter-spacing: -0.025em`) conveying authority and institutional polish.
- **Body & Explanatory Prose (`Inter`):** Clean, neutral typeface optimized for legibility in dense compliance dossiers and investigative narratives.
- **Identities, Quantities & Indices (`JetBrains Mono`):** Monospace typeface for Account IDs (`ACC_XXXX`), transaction IDs (`TX_XXXX`), currency figures, calibrated risk probabilities, and mathematical late-fusion formulas.

---

## 2. Component System Architecture

### 2.1 Reusable UI Primitives
1. **`.kpi-container`:** Elevated surface card with a vertical indicator accent stripe (blue, red, amber, green, or violet), uppercase tracking label, bold monospace metric value, and contextual delta description.
2. **`.mt-card` / `.enterprise-card`:** Flat elevated container (`#121C2E`) with subtle border (`#23334B`), hover elevation, and responsive padding.
3. **Status Badges (`.badge-*`):** Subtle tinted backgrounds (`rgba(..., 0.14)`) with matching solid border and high-contrast text, avoiding harsh solid fills.
4. **Monospace Entity Pill (`.account-id`):** Rounded monospace badge highlighting target account numbers across tables, waterfall charts, and dossiers.
5. **Interactive Data Grids (`st.dataframe`):** Configured with custom column configurations (`TextColumn`, `DatetimeColumn`, `NumberColumn`) and monetary formatting.

---

## 3. Module-by-Module Redesign Specifications

### 3.1 Sidebar & Navigation Controller (`app.py`)
- **Branding Header:** Custom SVG logo featuring an interconnected hexagonal shield and directed transaction nodes, base64 encoded to guarantee zero raw text leaking.
- **Navigation Menu:** Redesigned Streamlit radio group styled into cohesive pill buttons with clean icons (`📊 Executive Dashboard`, `⚡ AI Detection Center`, `🕸️ Network Intelligence`, `🔍 Account Investigation`, `🤖 AI Investigation Assistant`, `📑 Reports & Exports`, `📈 Model Performance`).
- **Telemetry Cards:**
  - *Engine Health:* Displays real-time model status (XGBoost v3.2, PyTorch RiskNet, NetworkX DiGraph) and zero-leakage compliance.
  - *Environment Telemetry:* Summarizes dataset scale (10,000 transactions, 1,000 account nodes, SQLite persistence).
- **Streamlit Native Cleanliness:** Hidden duplicate native page menu via `.streamlit/config.toml` (`client.showSidebarNavigation = false`).

### 3.2 Executive Command Center (`pages/dashboard.py`)
- **Header:** Live surveillance status and zero-leakage benchmark pills.
- **6-KPI Telemetry Grid:** Arranged in two balanced rows of 3 spacious cards, completely eliminating label wrapping and cramped numbers.
- **Temporal Volume & Activity Chart:** Dual-axis visualization with navy volume bars (`#1E2E4A`) and vivid crimson anomaly line markers (`#FF5B6E`).
- **Risk Distribution Donut Chart:** High-ratio donut (`hole=0.68`) with institutional color mapping and centered legend.
- **Priority Investigation Queue:** Tabular triage list showing top high-risk entities with formatted cash values and risk scores.

### 3.3 AI Detection Center (`pages/detection.py`)
- **Surveillance Filter Toolbar:** Unified filter card featuring risk-tier dropdown, AML pattern selector, transaction amount range slider, and account search input.
- **Real-Time Filter Metrics:** Displays instantaneous record count, filtered volume, average hybrid risk index, and anomaly count.
- **Operational Threshold Sensitivity Slider:** Dynamic simulation allowing compliance officers to tune alert sensitivity (0.05 to 0.95) and view immediate queue volume impact.
- **Scored Transaction Stream Table:** Full tabular view of scored transactions with XGBoost, Random Forest, PyTorch, and Hybrid Risk scores.
- **Multi-Model Consensus & Divergence Matrix:** Plotly scatter plot contrasting tree ensemble anomaly probability against neural representation risk.

### 3.4 Mule Network Intelligence (`pages/network_analysis.py`)
- **Graph Telemetry Strip:** Summary of total nodes, collectors (fan-in), distributors (fan-out), intermediaries (relay), and exit mules (cash-out).
- **Interactive Directed Network Graph:** Large Plotly visualization displaying directed transaction links, color-coded node symbols (square collectors, diamond distributors, hexagon intermediaries, star exit mules, and circle peers).
- **Topological Cluster Analysis Table:** Weakly connected components ranked by flagged mule density and total volume.
- **Circular Wash-Trading Detection Table:** Identifies closed multi-hop transaction loops (e.g., $A \rightarrow B \rightarrow C \rightarrow A$) with path structure and severity classification.

### 3.5 Account Forensic Investigation (`pages/account_investigation.py`)
- **Target Entity Dossier Header:** Bold target profile banner with risk tier badge, topology classification, large calibrated risk score, and 4-column financial summary (Inflow, Outflow, Counterparties, Retention Ratio).
- **Explainable AI (XAI) Waterfall Chart:** Step-by-step risk attribution breakdown showing baseline probability, ML behavioral contribution, PyTorch neural representation, and graph centrality anomaly.
- **Triggered Forensic Rules:** Distinct alert chips detailing heuristic rule violations (e.g., structuring, rapid turnover).
- **Immediate Ego-Network Subgraph:** Dedicated 1-hop counterparty network visualization.
- **Audited Transaction History:** Chronologically sorted transaction ledger with directional indicators (`OUTGOING ↗`, `INCOMING ↙`).

### 3.6 GenAI Investigation Assistant (`pages/ai_assistant.py`)
- **Forensic Audit Setup:** Target selector and AI provider toggle (Deterministic Offline Fallback, Ollama Llama 3.2, or Gemini API).
- **Structured 4-Part Intelligence Brief:**
  1. *Executive Incident Summary* (Blue accent border)
  2. *Factual Forensic Observations* (Violet accent border)
  3. *Investigative Hypotheses & Topology* (Amber accent border)
  4. *Recommended Compliance Actions* (Emerald accent border)
- **Forensic PDF Dossier Compilation:** Direct ReportLab PDF dossier generation with chain-of-custody metadata, signature blocks, and instant download.
- **Analyst Case Disposition:** Interactive case management module saving priority, status, and compliance notes to SQLite.

### 3.7 Reports & Compliance Export Center (`pages/reports.py`)
- **Telemetry CSV Cards:** Three modular cards for downloading Scored Transactions, Account Risk Profiles, and Suspicious Network Clusters.
- **PDF Dossier Repository:** Table of compiled investigation reports with file size, timestamp, and selector for immediate re-download.
- **Automated Regulatory SAR Generator:** Generates standardized machine-readable FinCEN / FIU XML-JSON filing payloads with one-click download.

### 3.8 AI & Deep Learning Model Benchmarking (`pages/model_performance.py`)
- **Methodology Notice:** Out-of-time chronological validation protocol description.
- **Comparative Benchmark Table:** Performance metrics (Precision, Recall, F1, FPR, ROC-AUC, PR-AUC) for XGBoost, Random Forest, PyTorch RiskNet, and Logistic Regression.
- **Operational Selection Rationale Box:** Comprehensive explanation of why XGBoost is Champion over Logistic Regression.
- **Multi-Threshold Operational Performance Matrix:** Operational metrics from 0.10 to 0.80 cutoff.
- **Confusion Matrix Heatmap & SHAP Bar Chart:** Plotly visualizations matching the enterprise theme.
- **Late-Fusion Formula Banner:** Explains the hybrid risk fusion:
  $$R_{\text{hybrid}} = 0.40 \cdot P_{\text{TreeEnsemble}} + 0.30 \cdot P_{\text{PyTorchNet}} + 0.30 \cdot G_{\text{TopologyAnomaly}}$$

---

## 4. Quality Assurance & Regression Verification

- **Automated Tests:** `23 passed in 15.27s` (100% test pass rate across backend, core algorithms, explainability, graph intelligence, and UI page rendering).
- **Python Compilation:** `python -m compileall` executed with 0 syntax or runtime errors.
- **Browser Visual QA:** Executed via autonomous browser subagent inspecting all 7 pages on `http://localhost:8501`.
- **Zero Placeholder Code:** All data, graphs, and tables are populated by genuine application engines and models.
