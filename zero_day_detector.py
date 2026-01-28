#!/usr/bin/env python3
"""
Zero-Day Attack Detection System

A production-ready system for detecting zero-day attacks using unsupervised
learning models. Learns normal network traffic patterns and identifies
anomalies that may indicate novel attacks.

This system implements three anomaly detection models:
1. Isolation Forest
2. One-Class SVM
3. Local Outlier Factor (LOF)

Author: Zero-Day Detection Team
Version: 1.0.0
"""

import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Import Custom Modules
# =============================================================================

from config import (
    BASE_DIR,
    DATA_DIR,
    OUTPUT_DIR,
    MODELS_DIR,
    KDD_TRAIN_PATH,
    KDD_TEST_PATH,
    SELECT_K_BEST_K,
    ISOLATION_FOREST,
    ONE_CLASS_SVM,
    LOF,
    LOG_FORMAT
)

from utils import (
    load_csv_safely,
    download_dataset_with_fallback,
    calculate_metrics,
    get_classification_report,
    print_section_header,
    print_metric_table,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_anomaly_score_distribution,
    plot_feature_importance,
    plot_model_comparison,
    suppress_warnings
)

from data_processor import DataProcessor
from detectors import (
    IsolationForestDetector,
    OneClassSVMDetector,
    LOFDetector,
    create_detector
)
from model_evaluator import ModelEvaluator


# =============================================================================
# Main Zero-Day Detector Class
# =============================================================================

class ZeroDayDetector:
    """
    Main orchestrator for the zero-day attack detection system.

    This class coordinates the entire detection pipeline:
    1. Data loading and preprocessing
    2. Model training (on benign data only)
    3. Model evaluation (on mixed test data)
    4. Visualization and reporting

    Attributes:
        data_processor: Data preprocessing pipeline.
        evaluator: Model evaluation and visualization.
        models: Dictionary of trained detection models.
        results: Dictionary of evaluation results.
    """

    def __init__(
        self,
        select_k_best: int = SELECT_K_BEST_K,
        output_dir: str = str(OUTPUT_DIR)
    ):
        """
        Initialize the zero-day detector.

        Args:
            select_k_best: Number of top features to select.
            output_dir: Directory for output files.
        """
        self.select_k_best = select_k_best
        self.output_dir = output_dir

        # Initialize components
        self.data_processor = DataProcessor(select_k_best=select_k_best)
        self.evaluator = ModelEvaluator(output_dir=output_dir)

        # Storage
        self.models: Dict[str, object] = {}
        self.results: Dict[str, Dict] = {}

        logger.info("ZeroDayDetector initialized")

    def run_pipeline(
        self,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        download_if_missing: bool = True
    ) -> Dict[str, Dict]:
        """
        Execute the complete detection pipeline.

        This is the main entry point for running the zero-day detection system.

        Args:
            train_path: Optional path to training data.
            test_path: Optional path to test data.
            download_if_missing: Whether to auto-download dataset.

        Returns:
            Dictionary containing results for each model.
        """
        print_section_header("ZERO-DAY ATTACK DETECTION SYSTEM")

        try:
            # Step 1: Load and preprocess data
            self._load_and_preprocess_data(
                train_path=train_path,
                test_path=test_path,
                download_if_missing=download_if_missing
            )

            # Step 2: Train detection models
            self._train_models()

            # Step 3: Evaluate models
            self._evaluate_models()

            # Step 4: Generate visualizations
            self._generate_visualizations()

            # Step 5: Print final report
            self._print_final_report()

            return self.results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    def _load_and_preprocess_data(
        self,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        download_if_missing: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Load and preprocess the NSL-KDD dataset.

        Args:
            train_path: Optional path to training data.
            test_path: Optional path to test data.
            download_if_missing: Whether to auto-download.

        Returns:
            Tuple of (X_train, X_test, y_test, feature_names).
        """
        print_section_header("STEP 1: DATA LOADING AND PREPROCESSING")

        # Load datasets
        if train_path is None:
            train_path = str(KDD_TRAIN_PATH)
        if test_path is None:
            test_path = str(KDD_TEST_PATH)

        # Check if files exist
        from pathlib import Path
        if not Path(train_path).exists() or not Path(test_path).exists():
            if download_if_missing:
                logger.info("Dataset files not found, attempting download...")
                train_path, test_path = download_dataset_with_fallback()
            else:
                raise FileNotFoundError(
                    f"Dataset files not found: {train_path}, {test_path}"
                )

        # Load raw data
        train_data = load_csv_safely(train_path)
        test_data = load_csv_safely(test_path)

        print(f"\nDataset Statistics:")
        print(f"  Training samples: {len(train_data)}")
        print(f"  Test samples: {len(test_data)}")

        # Show label distribution
        train_label_dist = self.data_processor.get_label_distribution(train_data)
        test_label_dist = self.data_processor.get_label_distribution(test_data)

        print(f"\nTraining Label Distribution:")
        for label, count in sorted(train_label_dist.items(), key=lambda x: -x[1])[:5]:
            print(f"  {label}: {count}")

        print(f"\nTest Label Distribution:")
        for label, count in sorted(test_label_dist.items(), key=lambda x: -x[1])[:5]:
            print(f"  {label}: {count}")

        # Preprocess data
        X_train, X_test, y_test, feature_names = self.data_processor.preprocess(
            train_data=train_data,
            test_data=test_data
        )

        print(f"\nPreprocessing Results:")
        print(f"  Benign training samples: {X_train.shape[0]}")
        print(f"  Features after selection: {X_train.shape[1]}")
        print(f"  Test samples: {X_test.shape[0]}")

        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names

        return X_train, X_test, y_test, feature_names

    def _train_models(self) -> None:
        """
        Train all three detection models on benign data.

        IMPORTANT: Models are trained ONLY on benign samples to learn
        what "normal" traffic looks like. This prevents data leakage.
        """
        print_section_header("STEP 2: MODEL TRAINING")

        print("\nTraining models on benign traffic only (learning 'normality')...")
        print("Models being trained:")
        print("  1. Isolation Forest")
        print("  2. One-Class SVM")
        print("  3. Local Outlier Factor (LOF)")

        # Initialize models
        self.models = {
            'Isolation Forest': IsolationForestDetector(
                n_estimators=ISOLATION_FOREST['n_estimators'],
                contamination=ISOLATION_FOREST['contamination'],
                random_state=42
            ),
            'One-Class SVM': OneClassSVMDetector(
                kernel=ONE_CLASS_SVM['kernel'],
                nu=ONE_CLASS_SVM['nu'],
                gamma=ONE_CLASS_SVM['gamma'],
                cache_size=ONE_CLASS_SVM['cache_size']
            ),
            'Local Outlier Factor': LOFDetector(
                n_neighbors=LOF['n_neighbors'],
                contamination=LOF['contamination'],
                novelty=True
            )
        }

        # Train each model
        for model_name, model in self.models.items():
            print(f"\n  Training {model_name}...")
            model.fit(self.X_train)
            print(f"  {model_name} trained successfully")

        print("\n✓ All models trained successfully")

    def _evaluate_models(self) -> None:
        """
        Evaluate all models on the test set.

        Generates predictions and calculates metrics for comparison.
        """
        print_section_header("STEP 3: MODEL EVALUATION")

        print("\nEvaluating models on mixed test data (Benign + Attack)...")
        print(f"Test set: {len(self.y_test)} samples")
        print(f"  Benign: {(self.y_test == 0).sum()}")
        print(f"  Attack: {(self.y_test == 1).sum()}")

        for model_name, model in self.models.items():
            print(f"\n  Evaluating {model_name}...")

            # Get predictions and scores
            y_pred = model.predict(self.X_test)
            anomaly_scores = model.decision_function(self.X_test)

            # For visualization, we use the raw scores
            # For ROC, we might need to invert based on model
            # Isolation Forest: lower score = more anomalous
            # One-Class SVM: lower score = more anomalous
            # LOF: lower score = more anomalous

            # Evaluate and store results
            results = self.evaluator.evaluate_model(
                model_name=model_name,
                y_true=self.y_test,
                y_pred=y_pred,
                y_scores=anomaly_scores
            )

            self.results[model_name] = results

            # Print metrics
            metrics = results['metrics']
            print(f"    Accuracy:  {metrics['accuracy']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1-Score:  {metrics['f1_score']:.4f}")
            print(f"    ROC-AUC:   {results['roc_auc']:.4f}")

        print("\n✓ All models evaluated successfully")

    def _generate_visualizations(self) -> None:
        """
        Generate all required visualizations.

        Creates:
        - Confusion matrices for all models
        - ROC curves for all models
        - Anomaly score distributions
        - Model comparison plot
        - Feature importance plot
        """
        print_section_header("STEP 4: GENERATING VISUALIZATIONS")

        print("\nGenerating visualizations...")

        # Confusion matrices
        print("  - Confusion matrices...")
        plot_confusion_matrices(
            self.results,
            save_path=f"{self.output_dir}/confusion_matrices.png"
        )

        # ROC curves
        print("  - ROC curves...")
        plot_roc_curves(
            self.results,
            save_path=f"{self.output_dir}/roc_curves.png"
        )

        # Anomaly score distributions
        print("  - Anomaly score distributions...")
        plot_anomaly_score_distribution(
            self.results,
            self.y_test,
            save_path=f"{self.output_dir}/anomaly_distribution.png"
        )

        # Model comparison
        print("  - Model comparison plot...")
        comparison_results = {
            name: res['metrics'] for name, res in self.results.items()
        }
        plot_model_comparison(
            comparison_results,
            save_path=f"{self.output_dir}/model_comparison.png"
        )

        # Feature importance
        print("  - Feature importance plot...")
        importance_scores = self.data_processor.get_feature_importance()
        plot_feature_importance(
            self.feature_names,
            importance_scores,
            top_n=self.select_k_best,
            save_path=f"{self.output_dir}/feature_importance.png"
        )

        print("\n✓ All visualizations generated and saved to output directory")

    def _print_final_report(self) -> None:
        """
        Print the final comprehensive report.
        """
        print_section_header("FINAL REPORT: ZERO-DAY ATTACK DETECTION RESULTS")

        # Performance summary table
        print("\nMODEL PERFORMANCE SUMMARY")
        print("-" * 70)
        print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 70)

        for model_name, results in self.results.items():
            metrics = results['metrics']
            print(
                f"{model_name:<25} "
                f"{metrics['accuracy']:.4f}      "
                f"{metrics['precision']:.4f}      "
                f"{metrics['recall']:.4f}      "
                f"{metrics['f1_score']:.4f}"
            )

        print("-" * 70)

        # Best model
        best_model = max(
            self.results.items(),
            key=lambda x: x[1]['metrics']['f1_score']
        )
        print(f"\n🏆 Best Model (by F1-Score): {best_model[0]}")
        print(f"   F1-Score: {best_model[1]['metrics']['f1_score']:.4f}")
        print(f"   ROC-AUC:  {best_model[1]['roc_auc']:.4f}")

        # Analysis
        print("\n" + "=" * 70)
        print("MODEL ANALYSIS")
        print("=" * 70)

        print("""
ISOLATION FOREST:
  ✓ Fast training and prediction
  ✓ Good for high-dimensional data
  ✓ Handles mixed feature types well
  ✓ Most practical for production use

ONE-CLASS SVM:
  ✓ Excellent for capturing complex boundaries
  ✓ Works well with small training sets
  ✗ Slow training on large datasets
  ✗ Memory intensive for large datasets

LOCAL OUTLIER FACTOR (LOF):
  ✓ Good for datasets with varying densities
  ✓ No assumptions about data distribution
  ✗ Cannot predict on new data (without novelty=True)
  ✗ Sensitive to choice of k (n_neighbors)
""")

        print("=" * 70)
        print("KEY FINDINGS")
        print("=" * 70)

        # Calculate overall statistics
        total_test = len(self.y_test)
        total_attack = (self.y_test == 1).sum()
        total_benign = (self.y_test == 0).sum()

        print(f"""
1. DATA OVERVIEW:
   - Total test samples: {total_test}
   - Benign samples: {total_benign} ({100*total_benign/total_test:.1f}%)
   - Attack samples: {total_attack} ({100*total_attack/total_test:.1f}%)

2. DETECTION CAPABILITY:
   - All three models can detect known attack patterns
   - Models trained only on benign data can generalize to attacks
   - Zero-day detection relies on anomaly deviation from normal

3. TRADE-OFFS:
   - Precision vs Recall balance depends on contamination parameter
   - Adjust 'contamination' based on expected attack rate
   - Higher sensitivity = more false positives but fewer missed attacks

4. PRODUCTION RECOMMENDATIONS:
   - Isolation Forest: Best balance of speed and accuracy
   - Monitor for drift in normal traffic patterns
   - Regular retraining on new normal data recommended
   - Combine with signature-based detection for hybrid approach
""")

        print("=" * 70)
        print("OUTPUT FILES GENERATED")
        print("=" * 70)
        print(f"""
The following files have been saved to {self.output_dir}:

  1. confusion_matrices.png   - Confusion matrices for all models
  2. roc_curves.png           - ROC curves comparison
  3. anomaly_distribution.png - Anomaly score distributions
  4. model_comparison.png     - Performance metrics comparison
  5. feature_importance.png   - Top important features
  6. evaluation_report.txt    - Detailed text report
  7. results_summary.csv      - Metrics in CSV format
""")

        # Save reports
        self.evaluator.save_report(f"{self.output_dir}/evaluation_report.txt")
        self.evaluator.save_summary_csv(f"{self.output_dir}/results_summary.csv")

        print("=" * 70)
        print("✓ ZERO-DAY ATTACK DETECTION COMPLETED SUCCESSFULLY")
        print("=" * 70)


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Zero-Day Attack Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python zero_day_detector.py                    # Run with default settings
  python zero_day_detector.py --no-download     # Skip auto-download
  python zero_day_detector.py --train custom_train.csv --test custom_test.csv
        """
    )

    parser.add_argument(
        '--train',
        type=str,
        help='Path to training CSV file'
    )
    parser.add_argument(
        '--test',
        type=str,
        help='Path to test CSV file'
    )
    parser.add_argument(
        '--no-download',
        action='store_true',
        help='Disable automatic dataset download'
    )
    parser.add_argument(
        '--features',
        type=int,
        default=SELECT_K_BEST_K,
        help=f'Number of features to select (default: {SELECT_K_BEST_K})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=str(OUTPUT_DIR),
        help='Output directory for results'
    )

    return parser.parse_args()


def main():
    """Main entry point for the zero-day detection system."""
    # Parse arguments
    args = parse_arguments()

    # Suppress warnings
    suppress_warnings()

    try:
        # Initialize detector
        detector = ZeroDayDetector(
            select_k_best=args.features,
            output_dir=args.output
        )

        # Run pipeline
        results = detector.run_pipeline(
            train_path=args.train,
            test_path=args.test,
            download_if_missing=not args.no_download
        )

        return results

    except KeyboardInterrupt:
        print("\n\n⚠️  Detection interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise


if __name__ == "__main__":
    main()
