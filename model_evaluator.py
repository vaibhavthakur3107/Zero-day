"""
Model evaluation module for Zero-Day Attack Detection System.

Provides comprehensive evaluation metrics, visualization generation,
and model comparison functionality for assessing anomaly detection performance.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
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
    classification_report,
    precision_recall_curve,
    average_precision_score
)

from config import (
    COLOR_PALETTE,
    CONFUSION_MATRIX_LABELS,
    FIGURE_DPI,
    FIGURE_SIZE,
    OUTPUT_DIR,
    ROC_NUM_POINTS
)

from utils import (
    calculate_metrics,
    get_classification_report,
    calculate_confusion_matrix_values,
    normalize_scores,
    print_section_header,
    print_metric_table
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluator for anomaly detection models.

    Provides methods for:
    - Computing evaluation metrics (precision, recall, F1, etc.)
    - Generating visualizations (confusion matrices, ROC curves, etc.)
    - Comparing multiple models side by side
    - Creating detailed performance reports
    """

    def __init__(self, output_dir: str = str(OUTPUT_DIR)):
        """
        Initialize the model evaluator.

        Args:
            output_dir: Directory to save output files.
        """
        self.output_dir = output_dir
        self.results: Dict[str, Dict[str, Any]] = {}

        logger.info("ModelEvaluator initialized")

    def evaluate_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate a single model and store results.

        Args:
            model_name: Name of the model.
            y_true: True labels (0=benign, 1=attack).
            y_pred: Predicted labels.
            y_scores: Anomaly scores (for ROC calculation).

        Returns:
            Dictionary containing all evaluation results.
        """
        logger.info(f"Evaluating model: {model_name}")

        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, y_scores)

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Calculate ROC curve and AUC
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = roc_auc_score(y_true, y_scores)
        except ValueError as e:
            logger.warning(f"Could not calculate ROC curve: {e}")
            fpr, tpr, thresholds = np.array([0, 1]), np.array([0, 1]), np.array([0, 1])
            roc_auc = 0.0

        # Calculate precision-recall curve
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
            avg_precision = average_precision_score(y_true, y_scores)
        except ValueError as e:
            logger.warning(f"Could not calculate PR curve: {e}")
            precision_curve, recall_curve = np.array([0]), np.array([0])
            avg_precision = 0.0

        # Store results
        results = {
            'model_name': model_name,
            'metrics': metrics,
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve,
            'avg_precision': avg_precision,
            'anomaly_scores': y_scores,
            'y_pred': y_pred,
            'y_true': y_true
        }

        self.results[model_name] = results

        logger.info(f"Evaluation completed for {model_name}")

        return results

    def get_summary_table(self) -> pd.DataFrame:
        """
        Create a summary DataFrame of all model results.

        Returns:
            DataFrame with model metrics comparison.
        """
        data = []
        for model_name, results in self.results.items():
            row = {
                'Model': model_name,
                'Accuracy': results['metrics'].get('accuracy', 0),
                'Precision': results['metrics'].get('precision', 0),
                'Recall': results['metrics'].get('recall', 0),
                'F1-Score': results['metrics'].get('f1_score', 0),
                'ROC-AUC': results.get('roc_auc', 0),
                'Avg Precision': results.get('avg_precision', 0)
            }
            data.append(row)

        return pd.DataFrame(data)

    def print_evaluation_report(self, model_name: str) -> None:
        """
        Print detailed evaluation report for a model.

        Args:
            model_name: Name of the model to report on.
        """
        if model_name not in self.results:
            logger.error(f"No results found for model: {model_name}")
            return

        results = self.results[model_name]
        metrics = results['metrics']
        cm = results['confusion_matrix']

        print_section_header(f"Evaluation Report: {model_name}")
        print(f"\nAccuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"Precision: {metrics.get('precision', 0):.4f}")
        print(f"Recall:    {metrics.get('recall', 0):.4f}")
        print(f"F1-Score:  {metrics.get('f1_score', 0):.4f}")
        print(f"ROC-AUC:   {results.get('roc_auc', 0):.4f}")

        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {(cm[0][0]):>6}")
        print(f"  False Positives: {(cm[0][1]):>6}")
        print(f"  False Negatives: {(cm[1][0]):>6}")
        print(f"  True Positives:  {(cm[1][1]):>6}")

        print(f"\nClassification Report:")
        report = get_classification_report(results['y_true'], results['y_pred'])
        print(report)

    def plot_all_visualizations(self, save: bool = True) -> None:
        """
        Generate all visualizations for all evaluated models.

        Args:
            save: Whether to save plots to files.
        """
        if not self.results:
            logger.warning("No results to visualize")
            return

        # Confusion matrices
        self.plot_confusion_matrices(save=save)

        # ROC curves
        self.plot_roc_curves(save=save)

        # Anomaly score distributions
        self.plot_anomaly_score_distributions(save=save)

        # Model comparison
        self.plot_model_comparison(save=save)

    def plot_confusion_matrices(self, save: bool = True) -> None:
        """
        Plot confusion matrices for all models.

        Args:
            save: Whether to save the plot.
        """
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

        if n_models == 1:
            axes = [axes]

        colors = ['#3498db', '#9b59b6', '#f39c12']

        for idx, (model_name, results) in enumerate(self.results.items()):
            ax = axes[idx]
            cm = results['confusion_matrix']

            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=CONFUSION_MATRIX_LABELS,
                yticklabels=CONFUSION_MATRIX_LABELS,
                ax=ax,
                cbar=True,
                annot_kws={'size': 12}
            )
            ax.set_title(f'{model_name}\nConfusion Matrix', fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=10)
            ax.set_ylabel('True Label', fontsize=10)

        plt.tight_layout()

        if save:
            save_path = f"{self.output_dir}/confusion_matrices.png"
            plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info(f"Confusion matrices saved to {save_path}")

        plt.show()
        plt.close()

    def plot_roc_curves(self, save: bool = True) -> None:
        """
        Plot ROC curves for all models on a single figure.

        Args:
            save: Whether to save the plot.
        """
        plt.figure(figsize=FIGURE_SIZE)

        colors = ['#3498db', '#9b59b6', '#f39c12']
        linestyles = ['-', '--', ':']

        for idx, (model_name, results) in enumerate(self.results.items()):
            fpr = results['fpr']
            tpr = results['tpr']
            auc = results['roc_auc']

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

        if save:
            save_path = f"{self.output_dir}/roc_curves.png"
            plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")

        plt.show()
        plt.close()

    def plot_anomaly_score_distributions(self, save: bool = True) -> None:
        """
        Plot anomaly score distributions for all models.

        Args:
            save: Whether to save the plot.
        """
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5))

        if n_models == 1:
            axes = [axes]

        for idx, (model_name, results) in enumerate(self.results.items()):
            ax = axes[idx]
            scores = results['anomaly_scores']
            y_true = results['y_true']

            # Separate scores by class
            benign_scores = scores[y_true == 0]
            attack_scores = scores[y_true == 1]

            # Normalize scores for better visualization
            benign_normalized = normalize_scores(benign_scores)
            attack_normalized = normalize_scores(attack_scores)

            # Plot distributions
            ax.hist(
                benign_normalized,
                bins=50,
                alpha=0.6,
                label='Benign',
                color='#2ecc71',
                density=True
            )
            ax.hist(
                attack_normalized,
                bins=50,
                alpha=0.6,
                label='Attack',
                color='#e74c3c',
                density=True
            )

            ax.set_xlabel('Normalized Anomaly Score', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title(f'{model_name}\nAnomaly Score Distribution', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            save_path = f"{self.output_dir}/anomaly_distribution.png"
            plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info(f"Anomaly score distributions saved to {save_path}")

        plt.show()
        plt.close()

    def plot_model_comparison(self, save: bool = True) -> None:
        """
        Create a bar chart comparing model performance metrics.

        Args:
            save: Whether to save the plot.
        """
        plt.figure(figsize=FIGURE_SIZE)

        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

        x = np.arange(len(metrics))
        width = 0.25

        colors = ['#3498db', '#9b59b6', '#f39c12']

        for idx, model_name in enumerate(models):
            model_metrics = [self.results[model_name]['metrics'].get(m, 0) for m in metrics]
            plt.bar(x + idx * width, model_metrics, width, label=model_name, color=colors[idx % len(colors)])

        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x + width, [m.replace('_', ' ').title() for m in metrics])
        plt.legend(loc='lower right', fontsize=10)
        plt.ylim([0, 1.1])
        plt.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()

        if save:
            save_path = f"{self.output_dir}/model_comparison.png"
            plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")

        plt.show()
        plt.close()

    def plot_feature_importance(
        self,
        feature_names: List[str],
        importance_scores: np.ndarray,
        top_n: int = 15,
        save: bool = True
    ) -> None:
        """
        Plot feature importance from SelectKBest.

        Args:
            feature_names: List of feature names.
            importance_scores: Array of importance scores.
            top_n: Number of top features to display.
            save: Whether to save the plot.
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

        if save:
            save_path = f"{self.output_dir}/feature_importance.png"
            plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")

        plt.show()
        plt.close()

    def generate_report(self) -> str:
        """
        Generate a comprehensive text report of all evaluations.

        Returns:
            Formatted report string.
        """
        lines = []
        lines.append("=" * 80)
        lines.append("ZERO-DAY ATTACK DETECTION - MODEL EVALUATION REPORT")
        lines.append("=" * 80)

        # Summary table
        lines.append("\n\nPERFORMANCE SUMMARY")
        lines.append("-" * 80)
        summary_df = self.get_summary_table()
        lines.append(summary_df.to_string(index=False))

        # Detailed results for each model
        lines.append("\n\nDETAILED RESULTS BY MODEL")
        lines.append("-" * 80)

        for model_name, results in self.results.items():
            lines.append(f"\n\n{model_name}")
            lines.append("~" * 40)

            metrics = results['metrics']
            cm = results['confusion_matrix']

            lines.append(f"Accuracy:    {metrics.get('accuracy', 0):.4f}")
            lines.append(f"Precision:   {metrics.get('precision', 0):.4f}")
            lines.append(f"Recall:      {metrics.get('recall', 0):.4f}")
            lines.append(f"F1-Score:    {metrics.get('f1_score', 0):.4f}")
            lines.append(f"ROC-AUC:     {results.get('roc_auc', 0):.4f}")
            lines.append(f"Avg Precision: {results.get('avg_precision', 0):.4f}")

            lines.append(f"\nConfusion Matrix:")
            lines.append(f"  TN: {cm[0][0]:>6}  FP: {cm[0][1]:>6}")
            lines.append(f"  FN: {cm[1][0]:>6}  TP: {cm[1][1]:>6}")

        lines.append("\n\n" + "=" * 80)

        return "\n".join(lines)

    def save_report(self, filepath: str) -> None:
        """
        Save the evaluation report to a text file.

        Args:
            filepath: Path to save the report.
        """
        report = self.generate_report()
        with open(filepath, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {filepath}")

    def save_summary_csv(self, filepath: str) -> None:
        """
        Save the summary table to a CSV file.

        Args:
            filepath: Path to save the CSV.
        """
        summary_df = self.get_summary_table()
        summary_df.to_csv(filepath, index=False)
        logger.info(f"Summary CSV saved to {filepath}")
