"""
Configuration module for Zero-Day Attack Detection System.

Contains all configuration constants, model hyperparameters, dataset paths,
and feature selection parameters. Centralized configuration ensures easy
tuning and maintenance of the detection system.
"""

import os
from pathlib import Path

# =============================================================================
# Directory Configuration
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# =============================================================================
# Dataset Configuration
# =============================================================================
# NSL-KDD Dataset URLs
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/hassan06/nslkdd"
UNB_MIRROR_URL = "https://www.unb.ca/cic/datasets/inundation.html"

# Dataset file names
KDD_TRAIN_FILE = "KDDTrain+.csv"
KDD_TEST_FILE = "KDDTest+.csv"

# Dataset paths (can be overridden by download)
KDD_TRAIN_PATH = DATA_DIR / KDD_TRAIN_FILE
KDD_TEST_PATH = DATA_DIR / KDD_TEST_FILE

# =============================================================================
# NSL-KDD Dataset Column Names
# =============================================================================
NSL_KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

# Categorical columns that need one-hot encoding
CATEGORICAL_COLUMNS = ['protocol_type', 'service', 'flag']

# =============================================================================
# Feature Selection Configuration
# =============================================================================
# Number of top features to select using SelectKBest
SELECT_K_BEST_K = 15

# =============================================================================
# Model Hyperparameters
# =============================================================================

# Isolation Forest Parameters
ISOLATION_FOREST = {
    'n_estimators': 200,
    'contamination': 0.15,  # Expected proportion of anomalies
    'max_samples': 'auto',
    'max_features': 1.0,
    'bootstrap': False,
    'n_jobs': -1,
    'random_state': 42,
    'verbose': 0
}

# One-Class SVM Parameters
ONE_CLASS_SVM = {
    'kernel': 'rbf',
    'nu': 0.15,  # Upper bound on fraction of outliers
    'gamma': 'scale',
    'cache_size': 500,  # MB
    'verbose': False
}

# Local Outlier Factor Parameters
LOF = {
    'n_neighbors': 30,
    'contamination': 0.15,
    'novelty': True,  # Enable prediction on new data
    'n_jobs': -1
}

# =============================================================================
# Training Configuration
# =============================================================================
TRAIN_SPLIT_RANDOM_STATE = 42
TEST_SPLIT_RANDOM_STATE = 42

# =============================================================================
# Evaluation Configuration
# =============================================================================
# ROC curve resolution
ROC_NUM_POINTS = 100

# Confusion matrix display format
CONFUSION_MATRIX_LABELS = ['Benign', 'Attack']

# =============================================================================
# Visualization Configuration
# =============================================================================
# Figure size for plots
FIGURE_SIZE = (10, 8)

# DPI for saved figures
FIGURE_DPI = 150

# Color palette for visualizations
COLOR_PALETTE = {
    'benign': '#2ecc71',    # Green
    'attack': '#e74c3c',    # Red
    'model1': '#3498db',    # Blue
    'model2': '#9b59b6',    # Purple
    'model3': '#f39c12',    # Orange
    'background': '#ecf0f1' # Light gray
}

# =============================================================================
# Logging Configuration
# =============================================================================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# =============================================================================
# Output File Names
# =============================================================================
FEATURE_IMPORTANCE_FILE = OUTPUT_DIR / "feature_importance.png"
ANOMALY_DISTRIBUTION_FILE = OUTPUT_DIR / "anomaly_distribution.png"
CONFUSION_MATRIX_FILE = OUTPUT_DIR / "confusion_matrices.png"
ROC_CURVES_FILE = OUTPUT_DIR / "roc_curves.png"
MODEL_COMPARISON_FILE = OUTPUT_DIR / "model_comparison.png"

# =============================================================================
# Alternative Dataset Configuration (for future expansion)
# =============================================================================
# CIC-IDS2017 Dataset configuration (commented for future use)
# CIC_IDS_2017 = {
#     'url': 'https://www.unb.ca/cic/datasets/ids2017.html',
#     'files': ['Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
#               'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
#               'Friday-WorkingHours-Morning.pcap_ISCX.csv',
#               'Monday-WorkingHours.pcap_ISCX.csv',
#               'Thursday-WorkingHours-Afternoon-WebAttacks.pcap_ISCX.csv',
#               'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
#               'Tuesday-WorkingHours.pcap_ISCX.csv',
#               'Wednesday-WorkingHours.pcap_ISCX.csv'],
#     'benign_label': 'Benign'
# }

# Hugging Face dataset configuration (commented for future use)
# HUGGINGFACE_CONFIG = {
#     'dataset_name': 'cicids2017',
#     'repo_id': 'datasets/cicids2017',
#     'subset': None
# }
