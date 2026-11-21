"""
MuleTrace AI - Complete End-to-End Training & Evaluation Pipeline
Executes data generation, chronological splitting, ML training, PyTorch training,
NetworkX graph analysis, SHAP explainability, and multi-modal risk scoring.
Saves all model checkpoints and outputs verified evaluation metrics.
"""

import sys
import os
import json
import time
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import MODELS_DIR, RANDOM_SEED, TEST_SIZE
from src.data_loader import get_or_create_dataset
from src.ml_model import MLModelEngine
from src.deep_learning import DeepLearningEngine
from src.graph_engine import GraphEngine
from src.risk_engine import RiskScoringEngine

def run_pipeline():
    start_time = time.time()
    print("=" * 80)
    print("MULETRACE AI - END-TO-END DATA & AI BACKEND TRAINING PIPELINE")
    print("=" * 80)

    # 1. Obtain / Generate Dataset
    print("\n[Stage 1/6] Loading & Generating Reproducible AML Dataset...")
    df = get_or_create_dataset(force_regenerate=True)
    n_total = len(df)
    n_pos = int(df["is_laundering"].sum())
    print(f"  -> Total transactions generated: {n_total:,}")
    print(f"  -> Total suspicious/laundering transactions: {n_pos} ({n_pos/n_total*100:.2f}%)")
    print(f"  -> Unique accounts involved: {len(set(df['sender_account']).union(set(df['receiver_account']))):,}")

    split_idx = int(n_total * (1 - TEST_SIZE))
    train_pos = int(df.iloc[:split_idx]["is_laundering"].sum())
    test_pos = int(df.iloc[split_idx:]["is_laundering"].sum())
    print(f"  -> Chronological Out-of-Time Split (80/20):")
    print(f"     * Train split (earlier 80%): {split_idx} txns, {train_pos} positives")
    print(f"     * Test split (later 20%):   {n_total - split_idx} txns, {test_pos} positives")

    # 2. Train Supervised ML Models (Baselines & Champion XGBoost)
    print("\n[Stage 2/6] Training Machine Learning Models & Computing SHAP...")
    ml_engine = MLModelEngine()
    ml_results = ml_engine.train_and_evaluate(df)
    
    print("  -> Baseline 1 (Logistic Regression):")
    lr_res = ml_results["baseline_logistic_regression"]
    print(f"     Precision: {lr_res['precision']:.4f} | Recall: {lr_res['recall']:.4f} | F1: {lr_res['f1_score']:.4f} | ROC-AUC: {lr_res['roc_auc']:.4f} | PR-AUC: {lr_res['pr_auc']:.4f} | FPR: {lr_res['false_positive_rate']:.5f}")

    print("  -> Baseline 2 (Random Forest Classifier):")
    rf_res = ml_results["baseline_random_forest"]
    print(f"     Precision: {rf_res['precision']:.4f} | Recall: {rf_res['recall']:.4f} | F1: {rf_res['f1_score']:.4f} | ROC-AUC: {rf_res['roc_auc']:.4f} | PR-AUC: {rf_res['pr_auc']:.4f} | FPR: {rf_res['false_positive_rate']:.5f}")

    print("  -> Champion Model (XGBoost Classifier with scale_pos_weight):")
    xgb_res = ml_results["champion_xgboost"]
    print(f"     Precision: {xgb_res['precision']:.4f} | Recall: {xgb_res['recall']:.4f} | F1: {xgb_res['f1_score']:.4f} | ROC-AUC: {xgb_res['roc_auc']:.4f} | PR-AUC: {xgb_res['pr_auc']:.4f} | FPR: {xgb_res['false_positive_rate']:.5f}")
    print(f"     Confusion Matrix: {xgb_res['confusion_matrix']}")

    print("  -> Top 5 Global SHAP Features:")
    for feat, val in list(ml_results["shap_feature_importance"].items())[:5]:
        print(f"     * {feat}: {val:.4f}")

    # 3. Train PyTorch Deep RiskNet
    print("\n[Stage 3/6] Training Compact PyTorch Deep RiskNet...")
    dl_engine = DeepLearningEngine()
    dl_results = dl_engine.train_and_evaluate(df)
    print("  -> PyTorch Deep RiskNet (Weighted BCE Loss):")
    print(f"     Precision: {dl_results['precision']:.4f} | Recall: {dl_results['recall']:.4f} | F1: {dl_results['f1_score']:.4f} | ROC-AUC: {dl_results['roc_auc']:.4f} | PR-AUC: {dl_results['pr_auc']:.4f} | FPR: {dl_results['false_positive_rate']:.5f}")
    print(f"     Confusion Matrix: {dl_results['confusion_matrix']}")

    # 4. Construct NetworkX Graph & Analyze Topology
    print("\n[Stage 4/6] Constructing NetworkX Graph & Extracting Topologies...")
    ge = GraphEngine()
    # Build graph with max_timestamp parameter supported for leakage prevention
    ge.build_graph(df)
    print(f"  -> Directed Graph Nodes: {len(ge.graph.nodes()):,}")
    print(f"  -> Directed Graph Edges: {len(ge.graph.edges()):,}")
    
    roles = [m["mule_role"] for m in ge.account_metrics.values()]
    from collections import Counter
    role_counts = Counter(roles)
    print(f"  -> Detected Mule Roles:")
    for role, count in role_counts.items():
        print(f"     * {role}: {count}")

    cycles = ge.detect_circular_patterns(max_cycle_length=5)
    print(f"  -> Circular Wash Cycles Detected: {len(cycles)}")
    clusters = ge.get_suspicious_clusters()
    print(f"  -> Suspicious Network Clusters: {len(clusters)}")

    # 5. Execute Hybrid Risk Scoring Engine
    print("\n[Stage 5/6] Executing Hybrid Multi-Modal Risk Scoring...")
    risk_engine = RiskScoringEngine()
    risk_engine.initialize(df)
    scored_tx = risk_engine.score_transactions(df)
    profiles = risk_engine.compute_account_risk_profiles(scored_tx)
    risk_engine.persist_account_profiles(profiles)
    
    tier_counts = profiles["risk_tier"].value_counts().to_dict()
    print("  -> Account Risk Tier Distribution:")
    for tier, cnt in tier_counts.items():
        print(f"     * {tier}: {cnt} accounts")

    # 6. Save Comprehensive Evaluation Summary
    print("\n[Stage 6/6] Persisting Models & Evaluation Summary JSON...")
    eval_summary = {
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "dataset_statistics": {
            "total_transactions": n_total,
            "total_accounts": len(profiles),
            "laundering_transactions": n_pos,
            "laundering_prevalence_pct": round(n_pos / n_total * 100, 3),
            "train_size": split_idx,
            "test_size": n_total - split_idx,
            "train_positives": train_pos,
            "test_positives": test_pos
        },
        "models_evaluation": {
            "baseline_logistic_regression": lr_res,
            "baseline_random_forest": rf_res,
            "champion_xgboost": xgb_res,
            "pytorch_deep_risknet": dl_results
        },
        "hybrid_fusion_architecture": {
            "formula": "Risk_hybrid = (0.40 * ML_prob) + (0.30 * DL_prob) + (0.30 * Graph_risk)",
            "ml_model_composition": "0.60 * XGBoost + 0.40 * Random Forest",
            "dl_model_composition": "PyTorch Compact RiskNet (BCE with pos_weight)",
            "graph_risk_composition": "NetworkX In/Out-degree anomaly, flow ratio, role heuristics"
        },
        "graph_statistics": {
            "nodes_count": len(ge.graph.nodes()),
            "edges_count": len(ge.graph.edges()),
            "mule_role_counts": dict(role_counts),
            "circular_cycles_detected": len(cycles),
            "suspicious_clusters_count": len(clusters)
        },
        "top_shap_features": list(ml_results["shap_feature_importance"].items())[:10]
    }

    summary_path = MODELS_DIR / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"  -> Evaluation summary saved to: {summary_path}")

    elapsed = time.time() - start_time
    print(f"\n[DONE] Pipeline execution completed successfully in {elapsed:.2f} seconds.")
    print("=" * 80)
    return eval_summary

if __name__ == "__main__":
    run_pipeline()
