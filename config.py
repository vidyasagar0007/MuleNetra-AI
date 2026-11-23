"""
MuleNetra AI - Central Configuration Module
Provides directory paths, model hyperparameters, graph heuristics,
risk scoring weights, and database connections.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
ASSETS_DIR = BASE_DIR / "assets"

# Ensure all primary directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, ASSETS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Database Configuration
db_file = (DATA_DIR / "muletrace.db").resolve()
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{db_file.as_posix()}")

# Machine Learning & Deep Learning Hyperparameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
CHRONOLOGICAL_SPLIT = True

# PyTorch Compact Risk NN Architecture
NN_INPUT_DIM = 14  # Computed features dimensionality
NN_HIDDEN_DIM = 32
NN_OUTPUT_DIM = 1
NN_LEARNING_RATE = 0.003
NN_BATCH_SIZE = 64
NN_EPOCHS = 15

# Risk Scoring Engine Weights (Total = 1.0)
RISK_WEIGHTS = {
    "ml_model": 0.40,        # XGBoost / Random Forest transaction probability
    "dl_model": 0.30,        # PyTorch compact neural net risk output
    "graph_heuristics": 0.30 # Graph degree, circularity, fan-in/fan-out anomaly
}

# Account Risk Classification Thresholds
RISK_THRESHOLDS = {
    "CRITICAL": 0.75,
    "HIGH": 0.55,
    "MEDIUM": 0.35,
    "LOW": 0.0
}

# Graph Heuristics Thresholds
GRAPH_CONFIG = {
    "fan_in_threshold": 4,       # Collector account minimum incoming transactions
    "fan_out_threshold": 4,      # Distributor account minimum outgoing transactions
    "structuring_amount_max": 9999.0, # Just below common $10k reporting ceiling
    "structuring_amount_min": 8000.0,
    "rapid_movement_window_hours": 4.0, # Rapid pass-through threshold
}

# GenAI & LLM Integration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "offline") # 'offline', 'ollama', 'gemini'
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# App Brand Info
APP_TITLE = "MuleNetra AI"
APP_TAGLINE = "See the Network. Detect the Risk."
APP_SUBTITLE = "Financial Intelligence Platform"
APP_DESCRIPTION = "AI-Powered Money Mule & Financial Network Intelligence Platform."
APP_VERSION = "1.0.0"
ACADEMIC_TAG = "B.Tech Final Year Capstone Project"
