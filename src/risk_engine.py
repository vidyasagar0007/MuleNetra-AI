"""
MuleNetra AI - Hybrid Risk Scoring Engine
Combines Supervised ML (Calibrated XGBoost & Random Forest Blend),
Deep Learning (PyTorch RiskNet), and Graph Centrality Heuristics into
unified, calibrated transaction and account-level risk scores.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import RISK_WEIGHTS, RISK_THRESHOLDS, TEST_SIZE
from src.ml_model import MLModelEngine
from src.deep_learning import DeepLearningEngine
from src.graph_engine import GraphEngine
from src.feature_engineering import get_feature_matrix
from src.database import get_db, AccountRiskRecord

class RiskScoringEngine:
    """
    Unified Risk Engine producing hybrid calibrated risk scores.
    Combines:
    1. Tabular Tree Ensemble: 60% XGBoost + 40% Random Forest
    2. Deep Learning: PyTorch Compact RiskNet
    3. Graph Intelligence: NetworkX topological anomaly & flow heuristics
    """
    def __init__(self):
        self.ml_engine = MLModelEngine()
        self.dl_engine = DeepLearningEngine()
        self.graph_engine = GraphEngine()
        self.is_initialized = False

    def initialize(self, df: pd.DataFrame, max_graph_timestamp: Optional[Any] = None):
        """
        Load or train ML and DL models, and construct transaction graph.
        If max_graph_timestamp is supplied, graph edges are constrained to avoid future leakage.
        """
        if not self.ml_engine.load_model():
            self.ml_engine.train_and_evaluate(df)
            
        if not self.dl_engine.load_model():
            self.dl_engine.train_and_evaluate(df)

        self.graph_engine.build_graph(df, max_timestamp=max_graph_timestamp)
        self.is_initialized = True

    def score_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ensemble risk probabilities for every transaction.
        Outputs individual model probabilities and calibrated hybrid score.
        """
        if not self.is_initialized:
            self.initialize(df)

        df = df.copy()
        
        # Individual model probabilities
        X_raw, _ = get_feature_matrix(df)
        X_scaled = self.ml_engine.scaler.transform(X_raw)
        
        xgb_probs = self.ml_engine.model.predict_proba(X_scaled)[:, 1]
        rf_probs = (
            self.ml_engine.rf_baseline.predict_proba(X_scaled)[:, 1]
            if self.ml_engine.rf_baseline is not None else xgb_probs
        )
        
        # Calibrated Tabular Tree Blend (60% XGBoost Champion + 40% Random Forest Baseline)
        ml_probs = (0.60 * xgb_probs) + (0.40 * rf_probs)
        
        # PyTorch Deep RiskNet Probability
        dl_probs = self.dl_engine.predict_risk(df)

        # Graph Edge Risk (Topological anomaly of sender/receiver)
        graph_edge_risks = []
        for _, row in df.iterrows():
            sender = str(row["sender_account"])
            receiver = str(row["receiver_account"])
            s_risk = self.graph_engine.account_metrics.get(sender, {}).get("graph_risk_score", 0.0)
            r_risk = self.graph_engine.account_metrics.get(receiver, {}).get("graph_risk_score", 0.0)
            graph_edge_risks.append(max(s_risk, r_risk))
            
        graph_edge_risks = np.array(graph_edge_risks)

        # Hybrid Late-Fusion Formulation:
        # Risk_hybrid = (0.40 * ML_prob) + (0.30 * DL_prob) + (0.30 * Graph_risk)
        w_ml = RISK_WEIGHTS.get("ml_model", 0.40)
        w_dl = RISK_WEIGHTS.get("dl_model", 0.30)
        w_gr = RISK_WEIGHTS.get("graph_heuristics", 0.30)

        hybrid_risk = (w_ml * ml_probs) + (w_dl * dl_probs) + (w_gr * graph_edge_risks)
        hybrid_risk = np.clip(hybrid_risk, 0.0, 1.0)

        df["xgb_risk_prob"] = xgb_probs
        df["rf_risk_prob"] = rf_probs
        df["ml_risk_prob"] = ml_probs
        df["dl_risk_prob"] = dl_probs
        df["graph_risk_prob"] = graph_edge_risks
        df["hybrid_risk_score"] = hybrid_risk
        
        # Risk Tier Tagging
        def assign_tier(score):
            if score >= RISK_THRESHOLDS["CRITICAL"]:
                return "CRITICAL"
            elif score >= RISK_THRESHOLDS["HIGH"]:
                return "HIGH"
            elif score >= RISK_THRESHOLDS["MEDIUM"]:
                return "MEDIUM"
            return "LOW"

        df["risk_tier"] = df["hybrid_risk_score"].apply(assign_tier)
        return df

    def compute_account_risk_profiles(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate transaction-level intelligence and graph centralities to build
        comprehensive account risk profiles.
        """
        account_profiles = []
        all_accounts = set(scored_df["sender_account"]).union(set(scored_df["receiver_account"]))

        for acc in all_accounts:
            sent_tx = scored_df[scored_df["sender_account"] == acc]
            recv_tx = scored_df[scored_df["receiver_account"] == acc]

            g_meta = self.graph_engine.account_metrics.get(acc, {})
            mule_role = g_meta.get("mule_role", "BENIGN")
            graph_risk = g_meta.get("graph_risk_score", 0.0)
            flag_reasons = list(g_meta.get("flag_reasons", []))

            associated_risks = []
            if len(sent_tx) > 0:
                associated_risks.extend(sent_tx["hybrid_risk_score"].tolist())
            if len(recv_tx) > 0:
                associated_risks.extend(recv_tx["hybrid_risk_score"].tolist())

            max_tx_risk = max(associated_risks) if associated_risks else 0.0
            mean_tx_risk = float(np.mean(associated_risks)) if associated_risks else 0.0

            ml_subscores = []
            dl_subscores = []
            if len(sent_tx) > 0:
                ml_subscores.extend(sent_tx["ml_risk_prob"].tolist())
                dl_subscores.extend(sent_tx["dl_risk_prob"].tolist())
            if len(recv_tx) > 0:
                ml_subscores.extend(recv_tx["ml_risk_prob"].tolist())
                dl_subscores.extend(recv_tx["dl_risk_prob"].tolist())

            ml_acc_risk = float(np.max(ml_subscores)) if ml_subscores else 0.0
            dl_acc_risk = float(np.max(dl_subscores)) if dl_subscores else 0.0

            # Aggregated Account Risk Score:
            # 35% Peak transaction risk + 25% Average risk + 20% Graph Topology + 20% Peak Model Prob
            account_risk = (
                (0.35 * max_tx_risk) +
                (0.25 * mean_tx_risk) +
                (0.20 * graph_risk) +
                (0.20 * max(ml_acc_risk, dl_acc_risk))
            )
            account_risk = float(np.clip(account_risk, 0.0, 1.0))

            if max_tx_risk >= 0.70:
                flag_reasons.append(f"High model anomaly detected on transaction (Peak: {max_tx_risk:.2f})")
            
            structuring_count = (
                ((sent_tx["amount"] >= 8000.0) & (sent_tx["amount"] < 10000.0)).sum() +
                ((recv_tx["amount"] >= 8000.0) & (recv_tx["amount"] < 10000.0)).sum()
            )
            if structuring_count >= 2:
                flag_reasons.append(f"Sub-threshold structuring detected ({structuring_count} transactions $8k-$10k)")

            if (sent_tx["new_balance_sender"] <= 1.0).any():
                flag_reasons.append("Account balance frequently cleared to zero after transfers")

            if account_risk >= RISK_THRESHOLDS["CRITICAL"]:
                tier = "CRITICAL"
            elif account_risk >= RISK_THRESHOLDS["HIGH"]:
                tier = "HIGH"
            elif account_risk >= RISK_THRESHOLDS["MEDIUM"]:
                tier = "MEDIUM"
            else:
                tier = "LOW"

            account_profiles.append({
                "account_id": acc,
                "risk_score": round(account_risk, 4),
                "risk_tier": tier,
                "ml_risk_score": round(ml_acc_risk, 4),
                "dl_risk_score": round(dl_acc_risk, 4),
                "graph_risk_score": round(graph_risk, 4),
                "mule_role": mule_role,
                "in_degree": int(g_meta.get("in_degree", 0)),
                "out_degree": int(g_meta.get("out_degree", 0)),
                "total_inflow": float(g_meta.get("inflow", 0.0)),
                "total_outflow": float(g_meta.get("outflow", 0.0)),
                "flagged_indicators": json.dumps(list(set(flag_reasons))),
                "flagged_count": len(set(flag_reasons))
            })

        profiles_df = pd.DataFrame(account_profiles)
        profiles_df = profiles_df.sort_values(by="risk_score", ascending=False).reset_index(drop=True)
        return profiles_df

    def persist_account_profiles(self, profiles_df: pd.DataFrame):
        """
        Store calculated profiles into SQLite for instant querying.
        """
        db = get_db()
        try:
            db.query(AccountRiskRecord).delete()
            for _, r in profiles_df.iterrows():
                rec = AccountRiskRecord(
                    account_id=r["account_id"],
                    risk_score=r["risk_score"],
                    risk_tier=r["risk_tier"],
                    ml_risk_score=r["ml_risk_score"],
                    dl_risk_score=r["dl_risk_score"],
                    graph_risk_score=r["graph_risk_score"],
                    mule_role=r["mule_role"],
                    in_degree=r["in_degree"],
                    out_degree=r["out_degree"],
                    total_inflow=r["total_inflow"],
                    total_outflow=r["total_outflow"],
                    flagged_indicators=r["flagged_indicators"]
                )
                db.add(rec)
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"Error persisting account profiles: {e}")
        finally:
            db.close()

if __name__ == "__main__":
    from src.data_loader import get_or_create_dataset
    df = get_or_create_dataset()
    engine = RiskScoringEngine()
    scored_tx = engine.score_transactions(df)
    print("Scored transactions sample:")
    print(scored_tx[["tx_id", "sender_account", "amount", "xgb_risk_prob", "rf_risk_prob", "ml_risk_prob", "dl_risk_prob", "hybrid_risk_score", "risk_tier"]].head())
    profiles = engine.compute_account_risk_profiles(scored_tx)
    print("\nTop 5 High-Risk Accounts:")
    print(profiles[["account_id", "risk_score", "risk_tier", "mule_role", "flagged_count"]].head(5))
    engine.persist_account_profiles(profiles)
    print("Persisted account profiles successfully.")
