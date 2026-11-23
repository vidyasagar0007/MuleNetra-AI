"""
MuleNetra AI - Preprocessing Module
Handles data sanitation, chronological train/test splitting,
and scaling with zero data leakage.
"""

import sys
from pathlib import Path
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import RANDOM_SEED, TEST_SIZE

def clean_transaction_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure correct data types, fill missing values, and sort chronologically.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    
    # Numeric sanitization
    numeric_cols = [
        "amount", "old_balance_sender", "new_balance_sender",
        "old_balance_receiver", "new_balance_receiver"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            
    # String sanitization
    for col in ["sender_account", "receiver_account", "tx_type"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            
    if "is_laundering" in df.columns:
        df["is_laundering"] = df["is_laundering"].astype(int)
        
    return df

def chronological_train_test_split(
    df: pd.DataFrame, 
    test_size: float = TEST_SIZE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset chronologically based on timestamp to prevent lookahead bias / target leakage.
    Earlier transactions are used for training, later transactions for testing.
    """
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    
    return train_df, test_df
