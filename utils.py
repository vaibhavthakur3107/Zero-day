"""
Utility module for Zero-Day Attack Detection System.

Provides helper functions for data loading, visualization, metric calculation,
and general-purpose operations used throughout the detection pipeline.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    classification_report
)

from config import (
    COLOR_PALETTE,
    CONFUSION_MATRIX_LABELS,
    FIGURE_DPI,
    FIGURE_SIZE,
    OUTPUT_DIR
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading Utilities
# =============================================================================

def load_csv_safely(filepath: str, column_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load a CSV file with comprehensive error handling.

    Args:
        filepath: Path to the CSV file.
        column_names: Optional list of column names to use.

    Returns:
        pd.DataFrame: Loaded dataframe.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file is empty or unreadable.
    """
    logger.info(f"Loading CSV file: {filepath}")

    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        if column_names:
            df = pd.read_csv(filepath, names=column_names, header=0)
        else:
            df = pd.read_csv(filepath)

        logger.info(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    except pd.errors.EmptyDataError:
        raise ValueError(f"File is empty: {filepath}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file: {e}")


def download_dataset_with_fallback() -> Tuple[str, str]:
    """
    Attempt to download NSL-KDD dataset with multiple fallback options.

    Downloads from:
    1. Kaggle API (via kagglehub)
    2. Direct HTTP download from UNB mirror
    3. Alternative GitHub raw content mirrors

    Returns:
        Tuple[str, str]: Paths to training and test files.

    Raises:
        Exception: If all download methods fail.
    """
    logger.info("Starting dataset download with fallback mechanism")

    # Try Kaggle API first
    try:
        import kagglehub

        logger.info("Attempting download via Kaggle API...")
        path = kagglehub.dataset_download("hassan06/nslkdd")
        train_path = str(Path(path) / "KDDTrain+.csv")
        test_path = str(Path(path) / "KDDTest+.csv")

        if Path(train_path).exists() and Path(test_path).exists():
            logger.info("Dataset downloaded successfully via Kaggle API")
            return train_path, test_path

    except ImportError:
        logger.warning("kagglehub not installed, skipping Kaggle download")
    except Exception as e:
        logger.warning(f"Kaggle download failed: {e}")

    # Try direct HTTP download
    try:
        logger.info("Attempting direct HTTP download...")
        import urllib.request
        import zipfile
        import io

        # NSL-KDD dataset URL
        url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.csv"

        logger.info(f"Downloading from {url}")

        # Download training data
        with urllib.request.urlopen(url, timeout=60) as response:
            train_content = response.read().decode('utf-8')

        # Save training data
        with open("KDDTrain+.csv", 'w') as f:
            f.write(train_content)

        # Download test data
        test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.csv"
        with urllib.request.urlopen(test_url, timeout=60) as response:
            test_content = response.read().decode('utf-8')

        # Save test data
        with open("KDDTest+.csv", 'w') as f:
            f.write(test_content)

        logger.info("Dataset downloaded successfully via HTTP")
        return "KDDTrain+.csv", "KDDTest+.csv"

    except Exception as e:
        logger.error(f"HTTP download failed: {e}")

    # Final fallback - provide clear instructions
    error_msg = (
        "All automatic download methods failed. "
        "Please download the NSL-KDD dataset manually:\n"
        "1. Go to: https://www.unb.ca/cic/datasets/nsl.html\n"
        "2. Download NSL-KDD dataset (CSV format)\n"
        "3. Place KDDTrain+.csv and KDDTest+.csv in the data/ directory\n"
        "Or use Kaggle: https://www.kaggle.com/datasets/hassan06/nslkdd"
    )
    raise Exception(error_msg)


# =============================================================================
# Metric Calculation Utilities
# =============================================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.

    Args:
        y_true: True labels (0=benign, 1=attack).
        y_pred: Predicted labels.
        y_scores: Prediction scores for ROC-AUC calculation.

    Returns:
        Dict[str, float]: Dictionary containing all metrics.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }

    if y_scores is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
        except ValueError:
            logger.warning("Could not calculate ROC-AUC (likely due to single class in predictions)")
            metrics['roc_auc'] = 0.0

    return metrics


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> str:
    """
    Generate a formatted classification report.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        str: Formatted classification report.
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=['Benign', 'Attack'],
        digits=4
    )
    return report


def calculate_confusion_matrix_values(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, int]:
    """
    Extract values from confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Dict[str, int]: TN, FP, FN, TP values.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }


# =============================================================================
# Visualization Utilities
# =============================================================================

def plot_confusion_matrices(
    results: Dict[str, Dict[str, np.ndarray]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrices for all models side by side.

    Args:
        results: Dictionary mapping model names to their confusion matrices.
        save_path: Optional path to save the figure.
    """
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    colors = list(COLOR_PALETTE.values())[:n_models]

    for idx, (model_name, cm_data) in enumerate(results.items()):
        ax = axes[idx]
        cm = cm_data['confusion_matrix']

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=CONFUSION_MATRIX_LABELS,
            yticklabels=CONFUSION_MATRIX_LABELS,
            ax=ax,
            cbar=True
        )
        ax.set_title(f'{model_name}\nConfusion Matrix', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Confusion matrices saved to {save_path}")

    plt.show()


def plot_roc_curves(
    results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot ROC curves for all models on a single figure.

    Args:
        results: Dictionary mapping model names to their results (including fpr, tpr, auc).
        save_path: Optional path to save the figure.
    """
    plt.figure(figsize=FIGURE_SIZE)

    colors = ['#3498db', '#9b59b6', '#f39c12']
    linestyles = ['-', '--', ':']

    for idx, (model_name, model_results) in enumerate(results.items()):
        fpr = model_results['fpr']
        tpr = model_results['tpr']
        auc = model_results['roc_auc']

        plt.plot(
            fpr,
            tpr,
            color=colors[idx % len(colors)],
            linestyle=linestyles[idx % len(linestyles)],
            linewidth=2,
            label=f'{model_name} (AUC = {auc:.4f})'
        )

    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('ROC Curves - Zero-Day Attack Detection Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"ROC curves saved to {save_path}")

    plt.show()


def plot_anomaly_score_distribution(
    results: Dict[str, Dict[str, np.ndarray]],
    y_true: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot anomaly score distributions for benign vs attack samples.

    Args:
        results: Dictionary mapping model names to their predictions.
        y_true: True labels.
        save_path: Optional path to save the figure.
    """
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    colors = ['#3498db', '#9b59b6', '#f39c12']

    for idx, (model_name, model_results) in enumerate(results.items()):
        ax = axes[idx]
        scores = model_results['anomaly_scores']

        # Separate scores by class
        benign_scores = scores[y_true == 0]
        attack_scores = scores[y_true == 1]

        # Plot distributions
        ax.hist(
            benign_scores,
            bins=50,
            alpha=0.6,
            label='Benign',
            color='#2ecc71',
            density=True
        )
        ax.hist(
            attack_scores,
            bins=50,
            alpha=0.6,
            label='Attack',
            color='#e74c3c',
            density=True
        )

        ax.set_xlabel('Anomaly Score', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{model_name}\nAnomaly Score Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Anomaly score distributions saved to {save_path}")

    plt.show()


def plot_feature_importance(
    feature_names: List[str],
    importance_scores: np.ndarray,
    top_n: int = 15,
    save_path: Optional[str] = None
) -> None:
    """
    Plot feature importance from SelectKBest.

    Args:
        feature_names: List of feature names.
        importance_scores: Array of importance scores.
        top_n: Number of top features to display.
        save_path: Optional path to save the figure.
    """
    plt.figure(figsize=FIGURE_SIZE)

    # Get indices of top features
    if len(importance_scores) > top_n:
        top_indices = np.argsort(importance_scores)[-top_n:][::-1]
    else:
        top_indices = np.argsort(importance_scores)[::-1]

    top_features = [feature_names[i] for i in top_indices]
    top_scores = importance_scores[top_indices]

    # Create horizontal bar plot
    bars = plt.barh(range(len(top_features)), top_scores, color='#3498db')
    plt.yticks(range(len(top_features)), top_features)

    plt.xlabel('Importance Score (Chi2)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features for Anomaly Detection', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, axis='x', alpha=0.3)

    # Add value labels on bars
    for bar, score in zip(bars, top_scores):
        plt.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{score:.3f}', va='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")

    plt.show()


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> None:
    """
    Create a bar chart comparing model performance metrics.

    Args:
        results: Dictionary mapping model names to their metrics.
        save_path: Optional path to save the figure.
    """
    plt.figure(figsize=FIGURE_SIZE)

    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

    x = np.arange(len(metrics))
    width = 0.25

    colors = ['#3498db', '#9b59b6', '#f39c12']

    for idx, model_name in enumerate(models):
        model_metrics = [results[model_name].get(m, 0) for m in metrics]
        plt.bar(x + idx * width, model_metrics, width, label=model_name, color=colors[idx % len(colors)])

    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x + width, [m.replace('_', ' ').title() for m in metrics])
    plt.legend(loc='lower right', fontsize=10)
    plt.ylim([0, 1.1])
    plt.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {save_path}")

    plt.show()


# =============================================================================
# Data Processing Utilities
# =============================================================================

def normalize_scores(
    scores: np.ndarray,
    method: str = 'minmax'
) -> np.ndarray:
    """
    Normalize anomaly scores to [0, 1] range.

    Args:
        scores: Array of anomaly scores.
        method: Normalization method ('minmax' or 'zscore').

    Returns:
        np.ndarray: Normalized scores.
    """
    if method == 'minmax':
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val == 0:
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean = scores.mean()
        std = scores.std()
        if std == 0:
            return np.zeros_like(scores)
        return (scores - mean) / std
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def print_section_header(title: str, width: int = 80) -> None:
    """
    Print a formatted section header.

    Args:
        title: Title to display.
        width: Total width of the header.
    """
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_metric_table(results: Dict[str, Dict[str, float]]) -> None:
    """
    Print a formatted table of model results.

    Args:
        results: Dictionary mapping model names to their metrics.
    """
    print("\n" + "-" * 70)
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 70)

    for model_name, metrics in results.items():
        print(
            f"{model_name:<25} "
            f"{metrics.get('accuracy', 0):.4f}      "
            f"{metrics.get('precision', 0):.4f}      "
            f"{metrics.get('recall', 0):.4f}      "
            f"{metrics.get('f1_score', 0):.4f}"
        )

    print("-" * 70)


def suppress_warnings() -> None:
    """Suppress all warnings for cleaner output."""
    warnings.filterwarnings('ignore')
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# =============================================================================
# Type Conversion Utilities
# =============================================================================

def convert_labels_to_binary(labels: pd.Series) -> np.ndarray:
    """
    Convert attack labels to binary (0=benign, 1=attack).

    Args:
        labels: Series containing labels ('normal' or attack type names).

    Returns:
        np.ndarray: Binary labels.
    """
    return (labels != 'normal').astype(int)


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if divisor is zero.

    Args:
        a: Numerator.
        b: Denominator.
        default: Value to return if b is zero.

    Returns:
        float: Result of division or default.
    """
    if b == 0:
        return default
    return a / b
