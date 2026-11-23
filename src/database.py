"""
MuleTrace AI - Database Engine & ORM Models
Provides SQLite connection via SQLAlchemy for persisting transactions,
account risk scores, alerts, and forensic investigation cases.
"""

import sys
from pathlib import Path
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Float, Integer, Boolean, DateTime, Text, ForeignKey, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
import pandas as pd

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import DATABASE_URL

Base = declarative_base()

class TransactionRecord(Base):
    """SQLAlchemy model for individual financial transactions."""
    __tablename__ = "transactions"

    tx_id = Column(String(64), primary_key=True, index=True)
    step = Column(Integer, index=True) # Step or hour
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    sender_account = Column(String(64), index=True, nullable=False)
    receiver_account = Column(String(64), index=True, nullable=False)
    amount = Column(Float, nullable=False)
    tx_type = Column(String(32), default="TRANSFER")
    old_balance_sender = Column(Float, default=0.0)
    new_balance_sender = Column(Float, default=0.0)
    old_balance_receiver = Column(Float, default=0.0)
    new_balance_receiver = Column(Float, default=0.0)
    is_laundering = Column(Integer, default=0, index=True) # Ground truth label
    pattern_type = Column(String(64), default="NORMAL") # AML topology pattern
    is_synthetic = Column(Boolean, default=True)

class AccountRiskRecord(Base):
    """SQLAlchemy model for account-level calculated risk profiles."""
    __tablename__ = "account_risk_profiles"

    account_id = Column(String(64), primary_key=True, index=True)
    risk_score = Column(Float, nullable=False, default=0.0)
    risk_tier = Column(String(32), default="LOW") # LOW, MEDIUM, HIGH, CRITICAL
    ml_risk_score = Column(Float, default=0.0)
    dl_risk_score = Column(Float, default=0.0)
    graph_risk_score = Column(Float, default=0.0)
    mule_role = Column(String(64), default="BENIGN") # COLLECTOR, INTERMEDIARY, EXIT, BENIGN
    degree_centrality = Column(Float, default=0.0)
    in_degree = Column(Integer, default=0)
    out_degree = Column(Integer, default=0)
    total_inflow = Column(Float, default=0.0)
    total_outflow = Column(Float, default=0.0)
    flagged_indicators = Column(Text, default="[]") # JSON list of detected rule triggers
    last_updated = Column(DateTime, default=datetime.utcnow)

class InvestigationCase(Base):
    """Forensic investigation cases opened by compliance analysts."""
    __tablename__ = "investigation_cases"

    case_id = Column(String(64), primary_key=True)
    account_id = Column(String(64), index=True, nullable=False)
    status = Column(String(32), default="UNDER_INVESTIGATION") # OPEN, UNDER_INVESTIGATION, ESCALATED, CLOSED
    priority = Column(String(32), default="HIGH")
    analyst_notes = Column(Text, default="")
    ai_summary = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

# Database Engine & Session Factory
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

def init_db():
    """Create all tables in the database if they do not exist."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get a database session context."""
    init_db()
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()

def save_transactions_to_db(df: pd.DataFrame):
    """Bulk insert or replace transactions dataframe into SQLite."""
    init_db()
    df.to_sql("transactions", con=engine, if_exists="replace", index=False)

def query_transactions(limit: int = 10000) -> pd.DataFrame:
    """Retrieve transactions as a pandas DataFrame."""
    init_db()
    query = f"SELECT * FROM transactions ORDER BY timestamp ASC LIMIT {limit}"
    return pd.read_sql(query, con=engine)
