"""
MuleTrace AI - Deep Learning Risk Engine (PyTorch)
Compact deep neural network for non-linear AML transaction risk classification
with class-imbalance weighted loss, chronological evaluation, and multi-threshold reporting.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix
)

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import (
    MODELS_DIR, RANDOM_SEED, TEST_SIZE,
    NN_LEARNING_RATE, NN_EPOCHS, NN_BATCH_SIZE
)
from src.preprocessing import clean_transaction_data
from src.feature_engineering import compute_features, get_feature_matrix, FEATURE_COLUMNS

class TransactionDataset(Dataset):
    """PyTorch Dataset wrapper for financial transaction feature vectors."""
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class MuleRiskNet(nn.Module):
    """
    Compact deep neural network tailored for tabular AML transaction risk classification.
    Uses LayerNorm/BatchNorm, LeakyReLU, and Dropout to prevent overfitting on imbalanced data.
    """
    def __init__(self, input_dim: int = len(FEATURE_COLUMNS), hidden_dim: int = 48):
        super(MuleRiskNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.25),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.15),
            
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.network(x)

class DeepLearningEngine:
    """
    Deep Learning manager for training, checkpointing, and inferring transaction risk.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MuleRiskNet(input_dim=len(FEATURE_COLUMNS)).to(self.device)
        self.scaler = StandardScaler()
        self.evaluation_results: Dict[str, Any] = {}
        self.is_trained = False

    def train_and_evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the compact PyTorch neural net on chronological train split and evaluate on test split.
        Sequential features preserve true account velocity without forward-looking leakage.
        """
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        df_clean = clean_transaction_data(df)
        full_feat = compute_features(df_clean)
        split_idx = int(len(full_feat) * (1 - TEST_SIZE))

        X_train_raw = full_feat.iloc[:split_idx][FEATURE_COLUMNS]
        y_train = full_feat.iloc[:split_idx]["is_laundering"].values
        X_test_raw = full_feat.iloc[split_idx:][FEATURE_COLUMNS]
        y_test = full_feat.iloc[split_idx:]["is_laundering"].values

        # Scaling: strictly on train data
        X_train = self.scaler.fit_transform(X_train_raw)
        X_test = self.scaler.transform(X_test_raw)

        # Handle class imbalance using pos_weight in BCEWithLogitsLoss
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(self.model.parameters(), lr=NN_LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NN_EPOCHS)

        train_dataset = TransactionDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=NN_BATCH_SIZE, shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(NN_EPOCHS):
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Evaluation on chronological test split
        self.model.eval()
        with torch.no_grad():
            test_x_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            test_logits = self.model(test_x_tensor)
            test_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()

        has_positives = y_test.sum() > 0
        roc_auc = float(roc_auc_score(y_test, test_probs)) if has_positives and len(np.unique(y_test)) > 1 else 0.5
        pr_auc = float(average_precision_score(y_test, test_probs)) if has_positives else 0.0

        thresholds = [0.10, 0.20, 0.35, 0.50, 0.70, 0.85]
        th_results = {}
        for th in thresholds:
            preds = (test_probs >= th).astype(int)
            cm = confusion_matrix(y_test, preds).tolist()
            tn, fp = cm[0][0], cm[0][1]
            fn, tp = cm[1][0], cm[1][1]
            fpr = float(fp / max(fp + tn, 1))

            th_results[f"th_{th:.2f}"] = {
                "threshold": th,
                "precision": float(precision_score(y_test, preds, zero_division=0)),
                "recall": float(recall_score(y_test, preds, zero_division=0)),
                "f1_score": float(f1_score(y_test, preds, zero_division=0)),
                "false_positive_rate": fpr,
                "confusion_matrix": cm,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn
            }

        default_metrics = th_results["th_0.50"]
        self.evaluation_results = {
            "precision": default_metrics["precision"],
            "recall": default_metrics["recall"],
            "f1_score": default_metrics["f1_score"],
            "false_positive_rate": default_metrics["false_positive_rate"],
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": default_metrics["confusion_matrix"],
            "test_positive_count": int(y_test.sum()),
            "test_negative_count": int((y_test == 0).sum()),
            "test_total_count": int(len(y_test)),
            "threshold_breakdown": th_results
        }

        self.is_trained = True
        self.save_model()
        return self.evaluation_results

    def predict_risk(self, df: pd.DataFrame) -> np.ndarray:
        """Produce risk probabilities for given transactions."""
        if not self.is_trained:
            self.load_model()

        X_raw, _ = get_feature_matrix(df)
        X_scaled = self.scaler.transform(X_raw)
        
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            logits = self.model(x_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
        return probs

    def save_model(self):
        """Save PyTorch weights, scaler, and evaluation results."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), MODELS_DIR / "pytorch_mule_net.pt")
        import joblib
        joblib.dump(self.scaler, MODELS_DIR / "pytorch_scaler.joblib")
        joblib.dump(self.evaluation_results, MODELS_DIR / "pytorch_eval_results.joblib")

    def load_model(self) -> bool:
        """Load PyTorch model weights and scaler from disk."""
        model_path = MODELS_DIR / "pytorch_mule_net.pt"
        scaler_path = MODELS_DIR / "pytorch_scaler.joblib"
        eval_path = MODELS_DIR / "pytorch_eval_results.joblib"

        if model_path.exists() and scaler_path.exists():
            import joblib
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.scaler = joblib.load(scaler_path)
            if eval_path.exists():
                self.evaluation_results = joblib.load(eval_path)
            self.is_trained = True
            return True
        return False

if __name__ == "__main__":
    from src.data_loader import get_or_create_dataset
    df = get_or_create_dataset()
    dl_engine = DeepLearningEngine()
    results = dl_engine.train_and_evaluate(df)
    print("PyTorch Compact RiskNet Evaluation Results (Default Threshold 0.50):")
    print(f"  Precision: {results['precision']:.3f}, Recall: {results['recall']:.3f}, F1: {results['f1_score']:.3f}, FPR: {results['false_positive_rate']:.5f}")
