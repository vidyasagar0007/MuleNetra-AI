# MuleNetra AI — Comprehensive System Setup & Execution Guide

**Operating System:** Windows 10 / Windows 11 (PowerShell)  
**Python Runtime:** Python 3.10 or 3.11 (64-bit)  
**Target Repository:** `C:\Users\vidya\.gemini\antigravity\scratch\MuleTrace-AI`  
**Application Port:** `http://localhost:8501`  

---

## 1. Prerequisites & Environment Check

Before proceeding, ensure Python is installed and accessible from PowerShell. Open a PowerShell terminal and verify:

```powershell
python --version
```
*Expected Output:* `Python 3.10.x` or `Python 3.11.x`

---

## 2. Directory Navigation & Virtual Environment Setup

### Step 2.1: Navigate to the Project Root
```powershell
Set-Location "C:\Users\vidya\.gemini\antigravity\scratch\MuleTrace-AI"
```

### Step 2.2: Create and Activate Python Virtual Environment
If creating a fresh virtual environment:
```powershell
# Create virtual environment in ./venv
python -m venv venv

# Set PowerShell execution policy if script execution is restricted
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```
*When activated, your PowerShell prompt will show `(venv)`.*

---

## 3. Dependency Installation

MuleNetra AI utilizes a locked `requirements.txt` containing all machine learning, deep learning, graph, and UI dependencies:

```powershell
# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install dependencies with PyTorch CPU wheel support
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### Core Dependencies Installed:
- `streamlit==1.67.0` (Reactive dark UI platform)
- `pandas>=2.1.0` & `numpy>=1.24.0` (Data manipulation & numerical arrays)
- `scikit-learn>=1.3.0` (Preprocessing, Logistic Regression, Random Forest, metrics)
- `xgboost>=2.0.0` (Gradient boosting champion model with scale_pos_weight)
- `torch>=2.1.0` (PyTorch deep learning neural network for CPU)
- `networkx>=3.1` (Directed multigraph topology and role heuristics)
- `shap>=0.42.0` (TreeSHAP game-theoretic explainability)
- `plotly>=5.16.0` (Interactive 2D network graphs, waterfalls, timelines)
- `sqlalchemy>=2.0.0` (Database ORM & SQLite persistence)
- `reportlab>=4.0.0` (Audit-grade forensic PDF compilation)
- `pytest>=7.4.0` (Automated verification test harness)

---

## 4. End-to-End Pipeline Execution (Data Generation & Model Training)

MuleNetra AI includes a standalone training script that generates the synthetic dataset, creates the SQLite database, performs the 80/20 chronological split, trains all ML/DL models, constructs the graph, and persists all model weights into `models/`:

```powershell
python src/train_pipeline.py
```

### What This Script Executes:
1. Ingests or generates the 9,587 transactions across 500 accounts.
2. Applies sequential 22-dimensional feature engineering.
3. Trains Logistic Regression baseline, Random Forest baseline, and XGBoost champion.
4. Trains the 4-layer PyTorch RiskNet across 20 epochs.
5. Constructs the NetworkX directed graph and extracts mule roles and circular cycles.
6. Evaluates all models strictly out-of-time on the unseen test split ($N=1,918$).
7. Saves all `.joblib` and `.pt` model weights and writes `models/evaluation_summary.json`.

---

## 5. Launching the Web Platform

To launch the interactive financial surveillance dashboard:

```powershell
streamlit run app.py
```

Once executed, Streamlit will output:
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
  Network URL: http://...
```
*Open `http://localhost:8501` in Google Chrome, Microsoft Edge, or Mozilla Firefox.*

---

## 6. Running the Automated Test Suite

To verify the complete platform integrity and ensure zero regressions across unit, backend, and UI layers:

```powershell
pytest -v
```

### Expected Output:
```
============================= test session starts =============================
platform win32 -- Python 3.11.0, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\vidya\.gemini\antigravity\scratch\MuleTrace-AI
collected 23 items

tests/test_backend.py::test_dataset_reproducibility_and_labels PASSED    [  4%]
tests/test_backend.py::test_chronological_splitting_no_leakage PASSED    [  8%]
tests/test_backend.py::test_feature_engineering_dimensions PASSED        [ 13%]
tests/test_backend.py::test_ml_models_and_shap PASSED                    [ 17%]
tests/test_backend.py::test_pytorch_deep_learning_network PASSED         [ 21%]
tests/test_backend.py::test_graph_engine_metrics_and_roles PASSED        [ 26%]
tests/test_backend.py::test_combined_risk_engine PASSED                  [ 30%]
tests/test_core.py::test_data_loader_generation PASSED                   [ 34%]
tests/test_core.py::test_preprocessing_and_splitting PASSED              [ 39%]
tests/test_core.py::test_feature_engineering PASSED                      [ 43%]
tests/test_core.py::test_ml_model_engine PASSED                          [ 47%]
tests/test_core.py::test_pytorch_deep_learning_engine PASSED             [ 52%]
tests/test_core.py::test_graph_intelligence_engine PASSED                [ 56%]
tests/test_core.py::test_hybrid_risk_scoring PASSED                      [ 60%]
tests/test_core.py::test_genai_assistant_offline_brief PASSED            [ 65%]
tests/test_core.py::test_pdf_report_generation PASSED                    [ 69%]
tests/test_ui.py::test_page_dashboard_render PASSED                      [ 73%]
tests/test_ui.py::test_page_detection_render PASSED                      [ 78%]
tests/test_ui.py::test_page_network_analysis_render PASSED               [ 82%]
tests/test_ui.py::test_page_account_investigation_render PASSED          [ 86%]
tests/test_ui.py::test_page_ai_assistant_render PASSED                   [ 91%]
tests/test_ui.py::test_page_reports_render PASSED                        [ 95%]
tests/test_ui.py::test_page_model_performance_render PASSED              [100%]

============================= 23 passed in 19.35s =============================
```

### Python Bytecode Compilation Verification:
```powershell
python -m compileall src pages tests app.py config.py
```
*Verifies 100% of source files compile with zero syntax errors.*

---

## 7. Troubleshooting & FAQ

### Issue 1: PowerShell Script Execution Policy Error
**Symptom:** `File .\venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system.`  
**Solution:** Run the following command in PowerShell:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1
```

### Issue 2: Port 8501 Already in Use
**Symptom:** `Port 8501 is already in use, trying to open next available port.`  
**Solution:** Specify a custom port or terminate the existing Streamlit process:
```powershell
# Option A: Run on alternative port
streamlit run app.py --server.port 8502

# Option B: Terminate existing Streamlit processes
Get-Process -Name python, streamlit -ErrorAction SilentlyContinue | Stop-Process -Force
streamlit run app.py
```

### Issue 3: GenAI Provider API Key Not Set
**Symptom:** Analyst wishes to run the GenAI Assistant without an external OpenAI/Gemini API key.  
**Solution:** In the GenAI Assistant page, keep the provider selection set to **"Deterministic Offline Engine"**. This engine requires zero internet access, zero API keys, and compiles 100% grounded investigation briefs and PDF reports using local database records.

---
*End of Setup Guide — MuleNetra AI B.Tech Capstone Project (2026)*
