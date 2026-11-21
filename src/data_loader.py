"""
MuleTrace AI - Data Loading & Synthetic AML Transaction Generator
Generates realistic, reproducible financial crime datasets featuring structuring,
fan-in/fan-out, rapid pass-through, circular loops, layering, and dormant activations.
Distributes laundering patterns across the full timeline for realistic chronological evaluation.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import RAW_DATA_DIR, RANDOM_SEED
from src.database import save_transactions_to_db

def generate_synthetic_aml_data(
    n_accounts: int = 500,
    n_benign_tx: int = 9500,
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Generate a high-fidelity synthetic financial transaction dataset with realistic
    money mule topologies and benign baseline transactions distributed across a 60-day window.
    
    All generated records are explicitly tagged as synthetic.
    """
    np.random.seed(seed)
    
    account_ids = [f"ACC_{i:04d}" for i in range(1, n_accounts + 1)]
    start_time = datetime(2026, 1, 1, 8, 0, 0)
    total_hours = 60 * 24 # 60 days
    records = []
    tx_counter = 1

    # -------------------------------------------------------------
    # 1. Benign Baseline Transactions (Standard retail, P2P, payroll, high-value corporate)
    # -------------------------------------------------------------
    merchants = account_ids[:25]
    retail_users = account_ids[25:450]
    
    for _ in range(n_benign_tx):
        # 8% high-value legitimate business transactions ($12,000 - $65,000)
        # Teaches model that large amount alone does NOT equal money laundering
        is_high_value_business = (np.random.random() < 0.08)
        
        if is_high_value_business:
            sender, receiver = np.random.choice(merchants, size=2, replace=False)
            amount = float(np.round(np.random.uniform(12000.0, 65000.0), 2))
            tx_type = "TRANSFER"
            old_bal_sender = float(np.round(np.random.uniform(amount * 2.0, amount * 8.0), 2))
            new_bal_sender = float(np.round(old_bal_sender - amount, 2))
            old_bal_receiver = float(np.round(np.random.uniform(5000.0, 80000.0), 2))
            new_bal_receiver = float(np.round(old_bal_receiver + amount, 2))
        else:
            sender = np.random.choice(retail_users)
            receiver = np.random.choice(account_ids)
            while receiver == sender:
                receiver = np.random.choice(account_ids)
                
            tx_type = np.random.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"], p=[0.45, 0.35, 0.15, 0.05])
            # Log-normal distribution for typical retail spends ($10 - $3,500)
            amount = float(np.round(np.random.lognormal(mean=4.5, sigma=1.2) + 5.0, 2))
            amount = min(amount, 10000.0)
            
            old_bal_sender = float(np.round(np.random.uniform(amount * 1.1, amount * 10.0 + 2000), 2))
            new_bal_sender = float(np.round(old_bal_sender - amount, 2))
            old_bal_receiver = float(np.round(np.random.uniform(50.0, 5000.0), 2))
            new_bal_receiver = float(np.round(old_bal_receiver + amount, 2))

        delta_hours = np.random.uniform(0, total_hours)
        tx_time = start_time + timedelta(hours=delta_hours)
        step = int(delta_hours)

        records.append({
            "tx_id": f"TX_{tx_counter:07d}",
            "step": step,
            "timestamp": tx_time,
            "sender_account": sender,
            "receiver_account": receiver,
            "amount": amount,
            "tx_type": tx_type,
            "old_balance_sender": old_bal_sender,
            "new_balance_sender": new_bal_sender,
            "old_balance_receiver": old_bal_receiver,
            "new_balance_receiver": new_bal_receiver,
            "is_laundering": 0,
            "pattern_type": "NORMAL",
            "is_synthetic": True
        })
        tx_counter += 1

    # -------------------------------------------------------------
    # 2. Structuring / Smurfing Clusters (Distributed across Timeline)
    # Spread across early, middle, and late weeks
    # -------------------------------------------------------------
    structuring_targets = [
        ("ACC_0042", 150),  # Early (train period)
        ("ACC_0088", 450),  # Mid (train period)
        ("ACC_0150", 800),  # Mid-late (train period)
        ("ACC_0180", 1180), # Late (test period)
        ("ACC_0195", 1350)  # Late (test period)
    ]
    
    for collector, base_h in structuring_targets:
        smurfs = np.random.choice([acc for acc in retail_users if acc != collector], size=7, replace=False)
        for i, smurf in enumerate(smurfs):
            # Structuring amounts: just under $10k reporting limit ($8,800 to $9,950)
            amount = float(np.round(np.random.uniform(8800.0, 9950.0), 2))
            offset = i * 0.75 # Every 45 minutes
            tx_time = start_time + timedelta(hours=base_h + offset)
            records.append({
                "tx_id": f"TX_{tx_counter:07d}",
                "step": int(base_h + offset),
                "timestamp": tx_time,
                "sender_account": smurf,
                "receiver_account": collector,
                "amount": amount,
                "tx_type": "TRANSFER",
                "old_balance_sender": amount + 300.0,
                "new_balance_sender": 300.0,
                "old_balance_receiver": 1000.0 + (i * amount),
                "new_balance_receiver": 1000.0 + ((i + 1) * amount),
                "is_laundering": 1,
                "pattern_type": "STRUCTURING",
                "is_synthetic": True
            })
            tx_counter += 1

    # -------------------------------------------------------------
    # 3. Fan-In (Collector) & Fan-Out (Distributor) Networks
    # Instance 1: Train period (hour 350)
    # Instance 2: Test period (hour 1220)
    # -------------------------------------------------------------
    fan_networks = [
        ("ACC_0077", "ACC_0099", 350),   # Train period
        ("ACC_0477", "ACC_0499", 1220)   # Test period
    ]

    for hub_collector, hub_distributor, base_fan_h in fan_networks:
        feeders = np.random.choice([acc for acc in retail_users if acc not in [hub_collector, hub_distributor]], size=8, replace=False)
        total_aggregated = 0.0
        
        # 8 Inflow feeders
        for idx, feeder in enumerate(feeders):
            amt = float(np.round(np.random.uniform(4800.0, 8500.0), 2))
            total_aggregated += amt
            tx_time = start_time + timedelta(hours=base_fan_h + (idx * 1.2))
            records.append({
                "tx_id": f"TX_{tx_counter:07d}",
                "step": int(base_fan_h + (idx * 1.2)),
                "timestamp": tx_time,
                "sender_account": feeder,
                "receiver_account": hub_collector,
                "amount": amt,
                "tx_type": "TRANSFER",
                "old_balance_sender": amt + 150.0,
                "new_balance_sender": 150.0,
                "old_balance_receiver": 500.0 + (idx * amt),
                "new_balance_receiver": 500.0 + ((idx + 1) * amt),
                "is_laundering": 1,
                "pattern_type": "FAN_IN_COLLECTOR",
                "is_synthetic": True
            })
            tx_counter += 1

        # Intermediary Bridge Transfer (Rapid movement)
        bridge_amt = float(np.round(total_aggregated * 0.96, 2))
        records.append({
            "tx_id": f"TX_{tx_counter:07d}",
            "step": int(base_fan_h + 14),
            "timestamp": start_time + timedelta(hours=base_fan_h + 14),
            "sender_account": hub_collector,
            "receiver_account": hub_distributor,
            "amount": bridge_amt,
            "tx_type": "TRANSFER",
            "old_balance_sender": total_aggregated + 500.0,
            "new_balance_sender": total_aggregated - bridge_amt + 500.0,
            "old_balance_receiver": 1000.0,
            "new_balance_receiver": 1000.0 + bridge_amt,
            "is_laundering": 1,
            "pattern_type": "RAPID_MOVEMENT",
            "is_synthetic": True
        })
        tx_counter += 1

        # Fan-Out to Exit accounts
        exit_accounts = np.random.choice([acc for acc in retail_users if acc not in [hub_collector, hub_distributor]], size=8, replace=False)
        disburse_amt = float(np.round(bridge_amt / len(exit_accounts), 2))
        for idx, exit_acc in enumerate(exit_accounts):
            records.append({
                "tx_id": f"TX_{tx_counter:07d}",
                "step": int(base_fan_h + 18 + (idx * 0.5)),
                "timestamp": start_time + timedelta(hours=base_fan_h + 18 + (idx * 0.5)),
                "sender_account": hub_distributor,
                "receiver_account": exit_acc,
                "amount": disburse_amt,
                "tx_type": "TRANSFER",
                "old_balance_sender": bridge_amt - (idx * disburse_amt),
                "new_balance_sender": bridge_amt - ((idx + 1) * disburse_amt),
                "old_balance_receiver": 100.0,
                "new_balance_receiver": 100.0 + disburse_amt,
                "is_laundering": 1,
                "pattern_type": "FAN_OUT_DISTRIBUTOR",
                "is_synthetic": True
            })
            tx_counter += 1

    # -------------------------------------------------------------
    # 4. Circular Wash Trading Cycles
    # Loop 1: Train period (hour 600)
    # Loop 2: Test period (hour 1290)
    # -------------------------------------------------------------
    loops = [
        (["ACC_0301", "ACC_0302", "ACC_0303", "ACC_0304"], 600, 28000.0),
        (["ACC_0401", "ACC_0402", "ACC_0403", "ACC_0404"], 1290, 31500.0)
    ]
    for loop_nodes, base_loop_h, loop_amt in loops:
        for i in range(len(loop_nodes)):
            src = loop_nodes[i]
            dst = loop_nodes[(i + 1) % len(loop_nodes)]
            amt = float(np.round(loop_amt * (1 - (0.006 * i)), 2))
            records.append({
                "tx_id": f"TX_{tx_counter:07d}",
                "step": int(base_loop_h + (i * 2.0)),
                "timestamp": start_time + timedelta(hours=base_loop_h + (i * 2.0)),
                "sender_account": src,
                "receiver_account": dst,
                "amount": amt,
                "tx_type": "TRANSFER",
                "old_balance_sender": amt + 800.0,
                "new_balance_sender": 800.0,
                "old_balance_receiver": 500.0,
                "new_balance_receiver": 500.0 + amt,
                "is_laundering": 1,
                "pattern_type": "CIRCULAR_WASH",
                "is_synthetic": True
            })
            tx_counter += 1

    # -------------------------------------------------------------
    # 5. Multi-Hop Layering Chains
    # Chain 1: Train period (hour 720)
    # Chain 2: Test period (hour 1380)
    # -------------------------------------------------------------
    chains = [
        (["ACC_0320", "ACC_0321", "ACC_0322", "ACC_0323", "ACC_0324"], 720, 42000.0),
        (["ACC_0420", "ACC_0421", "ACC_0422", "ACC_0423", "ACC_0424"], 1380, 46000.0)
    ]
    for chain_nodes, base_chain_h, chain_amt in chains:
        for i in range(len(chain_nodes) - 1):
            src = chain_nodes[i]
            dst = chain_nodes[i + 1]
            amt = float(np.round(chain_amt * (1 - (0.015 * i)), 2))
            records.append({
                "tx_id": f"TX_{tx_counter:07d}",
                "step": int(base_chain_h + (i * 3.0)),
                "timestamp": start_time + timedelta(hours=base_chain_h + (i * 3.0)),
                "sender_account": src,
                "receiver_account": dst,
                "amount": amt,
                "tx_type": "TRANSFER",
                "old_balance_sender": amt + 400.0,
                "new_balance_sender": 400.0,
                "old_balance_receiver": 200.0,
                "new_balance_receiver": 200.0 + amt,
                "is_laundering": 1,
                "pattern_type": "LAYERING_CHAIN",
                "is_synthetic": True
            })
            tx_counter += 1

    # -------------------------------------------------------------
    # 6. Dormant Account Sudden Activations
    # Activation 1: Train period (hour 880)
    # Activation 2: Test period (hour 1410)
    # -------------------------------------------------------------
    dormants = [
        ("ACC_0399", "ACC_0388", 880, 52000.0),
        ("ACC_0490", "ACC_0485", 1410, 58000.0)
    ]
    for dormant_src, dormant_dst, dormant_h, dormant_amt in dormants:
        records.append({
            "tx_id": f"TX_{tx_counter:07d}",
            "step": dormant_h,
            "timestamp": start_time + timedelta(hours=dormant_h),
            "sender_account": dormant_src,
            "receiver_account": dormant_dst,
            "amount": dormant_amt,
            "tx_type": "TRANSFER",
            "old_balance_sender": dormant_amt + 1000.0,
            "new_balance_sender": 1000.0,
            "old_balance_receiver": 500.0,
            "new_balance_receiver": 500.0 + dormant_amt,
            "is_laundering": 1,
            "pattern_type": "DORMANT_ACTIVATION",
            "is_synthetic": True
        })
        tx_counter += 1

    # Convert to DataFrame and sort chronologically
    df = pd.DataFrame(records)
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    return df

def get_or_create_dataset(force_regenerate: bool = False) -> pd.DataFrame:
    """
    Load dataset from CSV if available, or generate and persist to raw CSV & SQLite.
    """
    raw_csv_path = RAW_DATA_DIR / "transactions_raw.csv"
    
    if not force_regenerate and raw_csv_path.exists():
        df = pd.read_csv(raw_csv_path, parse_dates=["timestamp"])
        return df
    
    # Generate new synthetic dataset
    df = generate_synthetic_aml_data()
    df.to_csv(raw_csv_path, index=False)
    
    # Also persist to SQLite database
    try:
        save_transactions_to_db(df)
    except Exception as e:
        print(f"Database insertion notice: {e}")
        
    return df

if __name__ == "__main__":
    df = get_or_create_dataset(force_regenerate=True)
    print(f"Dataset generated with {len(df)} transactions.")
    print(f"Total laundering transactions: {df['is_laundering'].sum()} ({df['is_laundering'].mean()*100:.2f}%)")
    print("Pattern distribution:")
    print(df['pattern_type'].value_counts())
    
    # Check chronological split distribution (80/20)
    split_idx = int(len(df) * 0.8)
    train_pos = df.iloc[:split_idx]['is_laundering'].sum()
    test_pos = df.iloc[split_idx:]['is_laundering'].sum()
    print(f"Train positive cases (earlier 80%): {train_pos}")
    print(f"Test positive cases (later 20%): {test_pos}")
