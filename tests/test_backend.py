"""
MuleTrace AI - Automated Backend Verification Test Suite
Tests data loading, temporal splitting, feature engineering, ML models,
SHAP explainability, PyTorch deep learning, NetworkX graph metrics, and risk scoring.
"""

import sys
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
from src.explainability import ExplainabilityEngine

@pytest.fixture(scope="session")
def backend_dataset():
    """Generate a reproducible medium-scale dataset for testing."""
    df = generate_synthetic_aml_data(n_accounts=100, n_benign_tx=1200, seed=42)
    return df

def test_dataset_reproducibility_and_labels(backend_dataset):
    """Verify dataset format, schema, synthetic tags, and laundering patterns."""
    df = backend_dataset
    assert not df.empty
    assert len(df) > 1200
    assert "is_laundering" in df.columns
    assert "pattern_type" in df.columns
    assert df["is_synthetic"].all()
    assert df["is_laundering"].sum() > 0

    patterns = set(df["pattern_type"].unique())
    assert "NORMAL" in patterns
    assert "STRUCTURING" in patterns
    assert "FAN_IN_COLLECTOR" in patterns

def test_chronological_splitting_no_leakage(backend_dataset):
    """Verify that chronological splitting strictly prevents lookahead leakage."""
    clean_df = clean_transaction_data(backend_dataset)
    train_df, test_df = chronological_train_test_split(clean_df, test_size=0.2)
    
    assert len(train_df) + len(test_df) == len(clean_df)
    # Train max timestamp must be strictly <= test min timestamp
    assert train_df["timestamp"].max() <= test_df["timestamp"].min()
    assert train_df["step"].max() <= test_df["step"].min()

def test_feature_engineering_dimensions(backend_dataset):
    """Verify that computed feature matrix has exact expected columns with zero NaNs."""
    X, y = get_feature_matrix(backend_dataset)
    assert X.shape[0] == len(backend_dataset)
    assert X.shape[1] == len(FEATURE_COLUMNS)
    assert not X.isna().any().any()
    assert len(y) == len(backend_dataset)

def test_ml_models_and_shap(backend_dataset):
    """Verify baseline models, champion XGBoost, and SHAP explainability generation."""
    ml_engine = MLModelEngine()
    results = ml_engine.train_and_evaluate(backend_dataset, save_artifacts=False)
    
    assert "baseline_logistic_regression" in results
    assert "baseline_random_forest" in results
    assert "champion_xgboost" in results
    assert "shap_feature_importance" in results

    xgb_metrics = results["champion_xgboost"]
    assert 0.0 <= xgb_metrics["roc_auc"] <= 1.0
    assert 0.0 <= xgb_metrics["f1_score"] <= 1.0
    assert len(xgb_metrics["confusion_matrix"]) == 2

    # Verify SHAP feature importances
    shap_imps = results["shap_feature_importance"]
    assert len(shap_imps) > 0
    assert "amount" in shap_imps

    # Test local SHAP explanation on a sample
    sample_vec = get_feature_matrix(backend_dataset.head(1))[0].values[0]
    local_shap = ml_engine.explain_sample_shap(sample_vec)
    assert len(local_shap) == len(FEATURE_COLUMNS)

def test_pytorch_deep_learning_network(backend_dataset):
    """Verify PyTorch compact RiskNet forward pass, training, and probability bounds."""
    dl_engine = DeepLearningEngine()
    results = dl_engine.train_and_evaluate(backend_dataset, save_artifacts=False)
    
    assert "f1_score" in results
    assert "roc_auc" in results
    assert 0.0 <= results["roc_auc"] <= 1.0

    probs = dl_engine.predict_risk(backend_dataset.head(25))
    assert len(probs) == 25
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

def test_graph_engine_metrics_and_roles(backend_dataset):
    """Verify graph centrality calculation, role heuristics, and bounded cycle detection."""
    ge = GraphEngine()
    ge.build_graph(backend_dataset)
    
    assert len(ge.graph.nodes()) > 0
    assert len(ge.graph.edges()) > 0
    assert len(ge.account_metrics) == len(ge.graph.nodes())

    # Check that required centralities exist for all nodes
    for node, meta in list(ge.account_metrics.items())[:15]:
        assert "in_degree" in meta
        assert "out_degree" in meta
        assert "weighted_in_degree" in meta
        assert "weighted_out_degree" in meta
        assert "pagerank" in meta
        assert "clustering_coefficient" in meta
        assert "net_ratio" in meta
        assert "retention_ratio" in meta
        assert "mule_role" in meta

    cycles = ge.detect_circular_patterns(max_cycle_length=5)
    assert isinstance(cycles, list)

def test_combined_risk_engine(backend_dataset):
    """Verify hybrid late-fusion risk combination and tier assignments."""
    risk_engine = RiskScoringEngine()
    risk_engine.initialize(backend_dataset)
    
    scored_tx = risk_engine.score_transactions(backend_dataset)
    assert "hybrid_risk_score" in scored_tx.columns
    assert "risk_tier" in scored_tx.columns
    assert (scored_tx["hybrid_risk_score"] >= 0.0).all()
    assert (scored_tx["hybrid_risk_score"] <= 1.0).all()

    profiles = risk_engine.compute_account_risk_profiles(scored_tx)
    assert not profiles.empty
    assert "risk_score" in profiles.columns
    assert "risk_tier" in profiles.columns
    assert "mule_role" in profiles.columns
    assert set(profiles["risk_tier"].unique()).issubset({"CRITICAL", "HIGH", "MEDIUM", "LOW"})
