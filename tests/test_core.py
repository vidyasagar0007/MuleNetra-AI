"""
MuleTrace AI - Core Automated Test Suite
Verifies data generation, feature engineering, ML models, PyTorch deep net,
graph heuristics, hybrid risk scoring, and report generation.
"""

import sys
import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.data_loader import generate_synthetic_aml_data
from src.preprocessing import clean_transaction_data, chronological_train_test_split
from src.feature_engineering import compute_features, get_feature_matrix, FEATURE_COLUMNS
from src.ml_model import MLModelEngine
from src.deep_learning import DeepLearningEngine, MuleRiskNet
from src.graph_engine import GraphEngine
from src.risk_engine import RiskScoringEngine
from src.genai_assistant import GenAIAssistant
from src.report_generator import ReportGenerator

@pytest.fixture(scope="session")
def sample_dataset():
    """Generate a reproducible small-scale test dataset."""
    df = generate_synthetic_aml_data(n_accounts=60, n_benign_tx=500, seed=42)
    return df

def test_data_loader_generation(sample_dataset):
    """Test that synthetic AML dataset adheres to contract and contains expected patterns."""
    df = sample_dataset
    assert not df.empty
    assert len(df) > 500
    assert "tx_id" in df.columns
    assert "sender_account" in df.columns
    assert "receiver_account" in df.columns
    assert "amount" in df.columns
    assert "is_laundering" in df.columns
    assert "pattern_type" in df.columns
    assert df["is_synthetic"].all()
    assert df["is_laundering"].sum() > 0
    # Verify presence of key AML patterns
    patterns = set(df["pattern_type"].unique())
    assert "NORMAL" in patterns
    assert "STRUCTURING" in patterns
    assert "FAN_IN_COLLECTOR" in patterns

def test_preprocessing_and_splitting(sample_dataset):
    """Test data cleaning and chronological leakage-safe splitting."""
    clean_df = clean_transaction_data(sample_dataset)
    assert not clean_df["amount"].isna().any()
    
    train_df, test_df = chronological_train_test_split(clean_df, test_size=0.2)
    assert len(train_df) + len(test_df) == len(clean_df)
    # Check that train max timestamp <= test min timestamp (strictly chronological)
    assert train_df["timestamp"].max() <= test_df["timestamp"].min()

def test_feature_engineering(sample_dataset):
    """Test domain feature extraction and vector dimensions."""
    X, y = get_feature_matrix(sample_dataset)
    assert X.shape[0] == len(sample_dataset)
    assert X.shape[1] == len(FEATURE_COLUMNS)
    assert not X.isna().any().any()
    assert (y.values == sample_dataset["is_laundering"].values).all()

def test_ml_model_engine(sample_dataset):
    """Test XGBoost and baseline ML model training, prediction, and metric computation."""
    engine = MLModelEngine()
    results = engine.train_and_evaluate(sample_dataset)
    
    assert "champion_xgboost" in results
    assert "baseline_logistic_regression" in results
    assert "feature_importance" in results
    
    xgb_res = results["champion_xgboost"]
    assert "f1_score" in xgb_res
    assert "roc_auc" in xgb_res
    assert 0.0 <= xgb_res["f1_score"] <= 1.0

    # Test risk prediction output
    preds = engine.predict_risk(sample_dataset.head(20))
    assert len(preds) == 20
    assert (preds >= 0.0).all() and (preds <= 1.0).all()

def test_pytorch_deep_learning_engine(sample_dataset):
    """Test compact PyTorch neural net architecture and inference."""
    dl_engine = DeepLearningEngine()
    results = dl_engine.train_and_evaluate(sample_dataset)
    
    assert "f1_score" in results
    assert "roc_auc" in results
    
    probs = dl_engine.predict_risk(sample_dataset.head(15))
    assert len(probs) == 15
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

def test_graph_intelligence_engine(sample_dataset):
    """Test NetworkX directed graph construction, metrics, and role heuristics."""
    ge = GraphEngine()
    ge.build_graph(sample_dataset)
    
    assert len(ge.graph.nodes()) > 0
    assert len(ge.graph.edges()) > 0
    assert len(ge.account_metrics) == len(ge.graph.nodes())
    
    # Test metric types
    for node, meta in list(ge.account_metrics.items())[:10]:
        assert "in_degree" in meta
        assert "out_degree" in meta
        assert "mule_role" in meta
        assert 0.0 <= meta["graph_risk_score"] <= 1.0

    # Test cycle search
    cycles = ge.detect_circular_patterns(max_cycle_length=5)
    assert isinstance(cycles, list)

def test_hybrid_risk_scoring(sample_dataset):
    """Test late-fusion risk combination and account profile aggregation."""
    engine = RiskScoringEngine()
    engine.initialize(sample_dataset)
    
    scored_tx = engine.score_transactions(sample_dataset)
    assert "hybrid_risk_score" in scored_tx.columns
    assert "risk_tier" in scored_tx.columns
    assert (scored_tx["hybrid_risk_score"] >= 0.0).all()
    assert (scored_tx["hybrid_risk_score"] <= 1.0).all()
    
    profiles = engine.compute_account_risk_profiles(scored_tx)
    assert not profiles.empty
    assert "risk_score" in profiles.columns
    assert "mule_role" in profiles.columns
    assert "risk_tier" in profiles.columns

def test_genai_assistant_offline_brief():
    """Test deterministic offline forensic briefing engine."""
    assistant = GenAIAssistant(provider="offline")
    meta = {
        "risk_score": 0.85,
        "risk_tier": "CRITICAL",
        "mule_role": "COLLECTOR",
        "total_inflow": 45000.0,
        "total_outflow": 43500.0,
        "in_degree": 6,
        "out_degree": 1,
        "flagged_indicators": ["Sub-threshold structuring detected"]
    }
    brief = assistant.generate_investigation_brief("ACC_TEST_01", meta, [], {})
    assert "executive_summary" in brief
    assert "factual_observations" in brief
    assert "hypotheses" in brief
    assert "recommended_actions" in brief
    assert "ACC_TEST_01" in brief["executive_summary"]

def test_pdf_report_generation(sample_dataset, tmp_path):
    """Test ReportLab PDF dossier generation."""
    rep_gen = ReportGenerator()
    rep_gen.reports_dir = tmp_path
    
    meta = {
        "account_id": "ACC_TEST_PDF",
        "risk_score": 0.88,
        "risk_tier": "CRITICAL",
        "mule_role": "COLLECTOR",
        "total_inflow": 55000.0,
        "total_outflow": 52000.0,
        "in_degree": 8,
        "out_degree": 1,
        "flagged_indicators": ["Structuring pattern"]
    }
    brief = {
        "executive_summary": "Test executive summary for PDF verification.",
        "factual_observations": "• **Finding**: Test observation line.",
        "hypotheses": "• **Hypothesis**: Test hypothesis line.",
        "recommended_actions": "1. **Action**: Test action step."
    }
    pdf_path = rep_gen.generate_account_pdf_report("ACC_TEST_PDF", meta, brief, sample_dataset.head(5))
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 1000
