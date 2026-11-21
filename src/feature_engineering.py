"""
MuleTrace AI - Feature Engineering Module
Builds leakage-safe transaction, temporal, account-velocity, and sequential flow dynamics features.
"""

import sys
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Standardized feature column vector for machine learning & deep learning models
FEATURE_COLUMNS = [
    "amount",
    "log_amount",
    "balance_diff_sender",
    "balance_diff_receiver",
    "sender_depleted",
    "is_structuring_range",
    "is_round_amount",
    "hour_of_day",
    "day_of_week",
    "sender_cum_count",
    "receiver_cum_count",
    "sender_cum_amount",
    "receiver_cum_amount",
    "sender_mean_amount",
    "sender_amount_ratio",
    "time_delta_sender_hours",
    "time_delta_receiver_hours",
    "sender_flow_velocity",
    "tx_type_TRANSFER",
    "tx_type_PAYMENT",
    "tx_type_CASH_OUT",
    "tx_type_DEBIT"
]

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-rich financial crime features strictly ordered by timestamp
    to ensure zero forward-looking data leakage.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    
    # 1. Direct Transaction-Level Attributes
    df["log_amount"] = np.log1p(np.maximum(df["amount"], 0.0))
    df["balance_diff_sender"] = np.abs(df["old_balance_sender"] - df["new_balance_sender"] - df["amount"])
    df["balance_diff_receiver"] = np.abs(df["new_balance_receiver"] - df["old_balance_receiver"] - df["amount"])
    df["sender_depleted"] = (df["new_balance_sender"] <= 1.0).astype(int)
    
    # Structuring heuristic: values between $8,000 and $9,999
    df["is_structuring_range"] = ((df["amount"] >= 8000.0) & (df["amount"] < 10000.0)).astype(int)
    
    # Clean rounded numbers often preferred by criminals ($100, $500, $1,000 multiples)
    df["is_round_amount"] = ((df["amount"] % 100 == 0) | (df["amount"] % 1000 == 0)).astype(int)
    
    # Temporal cyclical features
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # 2. Sequential Account Dynamics (Historical cumulative metrics without future lookahead)
    sender_groups = df.groupby("sender_account")
    df["sender_cum_count"] = sender_groups.cumcount()
    
    # Cumulative outgoing sum for sender (excluding current row to avoid leakage)
    cum_sum_sender = sender_groups["amount"].cumsum() - df["amount"]
    df["sender_cum_amount"] = cum_sum_sender
    df["sender_mean_amount"] = np.where(df["sender_cum_count"] > 0, cum_sum_sender / df["sender_cum_count"], df["amount"])
    df["sender_amount_ratio"] = df["amount"] / (df["sender_mean_amount"] + 1.0)
    
    # Time delta since previous transaction from same sender
    sender_prev_time = sender_groups["timestamp"].shift(1)
    df["time_delta_sender_hours"] = (df["timestamp"] - sender_prev_time).dt.total_seconds() / 3600.0
    df["time_delta_sender_hours"] = df["time_delta_sender_hours"].fillna(720.0)

    # Sender velocity: amount divided by hours elapsed (high value = rapid burst)
    df["sender_flow_velocity"] = df["amount"] / (df["time_delta_sender_hours"] + 0.1)

    # Receiver historical metrics
    receiver_groups = df.groupby("receiver_account")
    df["receiver_cum_count"] = receiver_groups.cumcount()
    cum_sum_receiver = receiver_groups["amount"].cumsum() - df["amount"]
    df["receiver_cum_amount"] = cum_sum_receiver

    receiver_prev_time = receiver_groups["timestamp"].shift(1)
    df["time_delta_receiver_hours"] = (df["timestamp"] - receiver_prev_time).dt.total_seconds() / 3600.0
    df["time_delta_receiver_hours"] = df["time_delta_receiver_hours"].fillna(720.0)

    # 3. Transaction Type One-Hot Encoding
    for tx_t in ["TRANSFER", "PAYMENT", "CASH_OUT", "DEBIT"]:
        df[f"tx_type_{tx_t}"] = (df["tx_type"] == tx_t).astype(int)

    return df

def get_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts analytical feature matrix X and target label y (if present).
    """
    df_feat = compute_features(df)
    
    for col in FEATURE_COLUMNS:
        if col not in df_feat.columns:
            df_feat[col] = 0.0
            
    X = df_feat[FEATURE_COLUMNS].fillna(0.0)
    y = df_feat["is_laundering"] if "is_laundering" in df_feat.columns else pd.Series(0, index=df_feat.index)
    return X, y
