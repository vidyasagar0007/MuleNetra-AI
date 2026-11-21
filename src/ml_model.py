"""
MuleTrace AI - Machine Learning Engine
Trains Random Forest and Logistic Regression baselines alongside an optimized
XGBoost champion model with class-imbalance handling, chronological evaluation,
multi-threshold operational benchmarking, and SHAP feature attribution.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import shap
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix
)

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import MODELS_DIR, RANDOM_SEED, TEST_SIZE
from src.preprocessing import clean_transaction_data
from src.feature_engineering import compute_features, get_feature_matrix, FEATURE_COLUMNS

class MLModelEngine:
    """
    Supervised Machine Learning Engine for AML Risk Classification.
    Provides Baseline Models (Logistic Regression, Random Forest),
    Final Model (XGBoost), multi-threshold evaluation, and SHAP explainability.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.model: Optional[XGBClassifier] = None
        self.rf_baseline: Optional[RandomForestClassifier] = None
        self.lr_baseline: Optional[LogisticRegression] = None
        self.shap_explainer = None
        self.evaluation_results: Dict[str, Any] = {}
        self.feature_names = FEATURE_COLUMNS
        self.is_trained = False

    def train_and_evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train ML models using chronological train-test split to prevent data leakage.
        Sequential features are computed strictly ordered by timestamp, ensuring
        every test record incorporates its genuine historical activity without forward leakage.
        """
        df_clean = clean_transaction_data(df)
        
        # Chronological sequential feature extraction across the historical stream
        full_feat = compute_features(df_clean)
        split_idx = int(len(full_feat) * (1 - TEST_SIZE))

        X_train_raw = full_feat.iloc[:split_idx][FEATURE_COLUMNS]
        y_train = full_feat.iloc[:split_idx]["is_laundering"].values
        X_test_raw = full_feat.iloc[split_idx:][FEATURE_COLUMNS]
        y_test = full_feat.iloc[split_idx:]["is_laundering"].values

        # Scaling: Fit exclusively on training split
        X_train = self.scaler.fit_transform(X_train_raw)
        X_test = self.scaler.transform(X_test_raw)

        # Class imbalance calculation for scale_pos_weight
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = float(n_neg / max(n_pos, 1))

        # -------------------------------------------------------------
        # 1. Baseline Model 1: Logistic Regression (Balanced)
        # -------------------------------------------------------------
        self.lr_baseline = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED
        )
        self.lr_baseline.fit(X_train, y_train)
        lr_probs = self.lr_baseline.predict_proba(X_test)[:, 1]

        # -------------------------------------------------------------
        # 2. Baseline Model 2: Random Forest Classifier
        # -------------------------------------------------------------
        self.rf_baseline = RandomForestClassifier(
            n_estimators=120,
            max_depth=7,
            class_weight="balanced",
            min_samples_split=4,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        self.rf_baseline.fit(X_train, y_train)
        rf_probs = self.rf_baseline.predict_proba(X_test)[:, 1]

        # -------------------------------------------------------------
        # 3. Final Model: XGBoost Classifier (Champion)
        # -------------------------------------------------------------
        self.model = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.06,
            scale_pos_weight=scale_pos_weight,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=1.0,
            eval_metric="logloss",
            random_state=RANDOM_SEED
        )
        self.model.fit(X_train, y_train)
        xgb_probs = self.model.predict_proba(X_test)[:, 1]

        # -------------------------------------------------------------
        # 4. Multi-Threshold Operational Evaluation
        # -------------------------------------------------------------
        def compute_multi_threshold_metrics(y_true, y_prob):
            has_pos = y_true.sum() > 0
            roc_auc = float(roc_auc_score(y_true, y_prob)) if has_pos and len(np.unique(y_true)) > 1 else 0.5
            pr_auc = float(average_precision_score(y_true, y_prob)) if has_pos else 0.0

            thresholds = [0.10, 0.20, 0.35, 0.50, 0.70, 0.85]
            th_results = {}
            for th in thresholds:
                preds = (y_prob >= th).astype(int)
                cm = confusion_matrix(y_true, preds).tolist()
                tn, fp = cm[0][0], cm[0][1]
                fn, tp = cm[1][0], cm[1][1]
                fpr = float(fp / max(fp + tn, 1))

                th_results[f"th_{th:.2f}"] = {
                    "threshold": th,
                    "precision": float(precision_score(y_true, preds, zero_division=0)),
                    "recall": float(recall_score(y_true, preds, zero_division=0)),
                    "f1_score": float(f1_score(y_true, preds, zero_division=0)),
                    "false_positive_rate": fpr,
                    "confusion_matrix": cm,
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "true_negatives": tn
                }

            # Standard 0.5 default metrics
            default_metrics = th_results["th_0.50"]
            return {
                "precision": default_metrics["precision"],
                "recall": default_metrics["recall"],
                "f1_score": default_metrics["f1_score"],
                "false_positive_rate": default_metrics["false_positive_rate"],
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "confusion_matrix": default_metrics["confusion_matrix"],
                "test_positive_count": int(y_true.sum()),
                "test_negative_count": int((y_true == 0).sum()),
                "test_total_count": int(len(y_true)),
                "threshold_breakdown": th_results
            }

        lr_eval = compute_multi_threshold_metrics(y_test, lr_probs)
        rf_eval = compute_multi_threshold_metrics(y_test, rf_probs)
        xgb_eval = compute_multi_threshold_metrics(y_test, xgb_probs)

        # -------------------------------------------------------------
        # 5. SHAP TreeExplainer & Global Attribution
        # -------------------------------------------------------------
        shap_feature_importance = {}
        try:
            self.shap_explainer = shap.TreeExplainer(self.model)
            sample_idx = np.random.choice(len(X_test), size=min(300, len(X_test)), replace=False)
            X_sample = X_test[sample_idx]
            shap_values = self.shap_explainer.shap_values(X_sample)
            
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            for idx, feat in enumerate(self.feature_names):
                shap_feature_importance[feat] = float(mean_abs_shap[idx])
            shap_feature_importance = dict(sorted(shap_feature_importance.items(), key=lambda item: item[1], reverse=True))
        except Exception as e:
            print(f"SHAP computation notice: {e}")
            importances = self.model.feature_importances_
            for feat, imp in zip(self.feature_names, importances):
                shap_feature_importance[feat] = float(imp)

        self.evaluation_results = {
            "baseline_logistic_regression": lr_eval,
            "baseline_random_forest": rf_eval,
            "champion_xgboost": xgb_eval,
            "shap_feature_importance": shap_feature_importance,
            "feature_importance": self.get_feature_importances()
        }

        self.is_trained = True
        self.save_model()
        return self.evaluation_results

    def get_feature_importances(self) -> Dict[str, float]:
        """Extract XGBoost feature importances."""
        if self.model is None:
            return {}
        importances = self.model.feature_importances_
        feat_dict = {
            feat: float(imp) 
            for feat, imp in zip(self.feature_names, importances)
        }
        return dict(sorted(feat_dict.items(), key=lambda item: item[1], reverse=True))

    def explain_sample_shap(self, X_row: np.ndarray) -> Dict[str, float]:
        """Compute local SHAP attribution values for a single transaction vector."""
        if self.shap_explainer is None:
            if not self.load_model():
                return {}
        if isinstance(X_row, pd.DataFrame):
            X_df = X_row
        else:
            if hasattr(X_row, "ndim") and X_row.ndim == 1:
                X_row = X_row.reshape(1, -1)
            X_df = pd.DataFrame(X_row, columns=self.feature_names)
        X_scaled = self.scaler.transform(X_df)
        shap_vals = self.shap_explainer.shap_values(X_scaled)[0]
        
        contribs = {feat: float(val) for feat, val in zip(self.feature_names, shap_vals)}
        return dict(sorted(contribs.items(), key=lambda item: abs(item[1]), reverse=True))

    def predict_risk(self, df: pd.DataFrame, use_ensemble: bool = True) -> np.ndarray:
        """
        Produce calibrated risk probabilities for given transactions.
        If use_ensemble=True, combines XGBoost (60%) and Random Forest (40%) for variance reduction.
        """
        if not self.is_trained:
            self.load_model()
            
        X_raw, _ = get_feature_matrix(df)
        X_scaled = self.scaler.transform(X_raw)
        
        xgb_p = self.model.predict_proba(X_scaled)[:, 1]
        if use_ensemble and self.rf_baseline is not None:
            rf_p = self.rf_baseline.predict_proba(X_scaled)[:, 1]
            return 0.60 * xgb_p + 0.40 * rf_p
        return xgb_p

    def save_model(self):
        """Save model, baselines, scaler, and SHAP explainer to disk."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, MODELS_DIR / "xgboost_model.joblib")
        joblib.dump(self.rf_baseline, MODELS_DIR / "random_forest_baseline.joblib")
        joblib.dump(self.lr_baseline, MODELS_DIR / "logistic_regression_baseline.joblib")
        joblib.dump(self.scaler, MODELS_DIR / "feature_scaler.joblib")
        joblib.dump(self.shap_explainer, MODELS_DIR / "shap_explainer.joblib")
        joblib.dump(self.evaluation_results, MODELS_DIR / "ml_eval_results.joblib")

    def load_model(self) -> bool:
        """Load trained models and scaler from disk."""
        model_path = MODELS_DIR / "xgboost_model.joblib"
        scaler_path = MODELS_DIR / "feature_scaler.joblib"
        eval_path = MODELS_DIR / "ml_eval_results.joblib"
        explainer_path = MODELS_DIR / "shap_explainer.joblib"

        if model_path.exists() and scaler_path.exists():
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            if (MODELS_DIR / "random_forest_baseline.joblib").exists():
                self.rf_baseline = joblib.load(MODELS_DIR / "random_forest_baseline.joblib")
            if (MODELS_DIR / "logistic_regression_baseline.joblib").exists():
                self.lr_baseline = joblib.load(MODELS_DIR / "logistic_regression_baseline.joblib")
            if explainer_path.exists():
                self.shap_explainer = joblib.load(explainer_path)
            if eval_path.exists():
                self.evaluation_results = joblib.load(eval_path)
            self.is_trained = True
            return True
        return False

if __name__ == "__main__":
    from src.data_loader import get_or_create_dataset
    df = get_or_create_dataset()
    engine = MLModelEngine()
    results = engine.train_and_evaluate(df)
    print("\n--- Model Evaluation Results (Default Threshold 0.50) ---")
    print("Baseline Logistic Regression:", results["baseline_logistic_regression"]["f1_score"])
    print("Baseline Random Forest:", results["baseline_random_forest"]["f1_score"])
    print("Champion XGBoost:", results["champion_xgboost"]["f1_score"])
    
    print("\n--- XGBoost Multi-Threshold Operational Matrix ---")
    for th_key, th_val in results["champion_xgboost"]["threshold_breakdown"].items():
        print(f"  {th_key}: Prec={th_val['precision']:.3f}, Rec={th_val['recall']:.3f}, F1={th_val['f1_score']:.3f}, FPR={th_val['false_positive_rate']:.5f}, TP={th_val['true_positives']}, FP={th_val['false_positives']}")
