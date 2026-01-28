"""
Data preprocessing module for Zero-Day Attack Detection System.

Handles all data loading, cleaning, encoding, scaling, and feature selection
operations. Implements the critical preprocessing pipeline to prevent data
leakage while preparing data for unsupervised anomaly detection models.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2

from config import (
    NSL_KDD_COLUMNS,
    CATEGORICAL_COLUMNS,
    SELECT_K_BEST_K,
    KDD_TRAIN_PATH,
    KDD_TEST_PATH,
    KDD_TRAIN_FILE,
    KDD_TEST_FILE
)

from utils import (
    load_csv_safely,
    convert_labels_to_binary,
    download_dataset_with_fallback
)

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Comprehensive data processor for network intrusion detection.

    This class handles the complete preprocessing pipeline:
    1. Dataset loading with automatic download fallback
    2. Data cleaning and validation
    3. One-hot encoding of categorical features
    4. Standard scaling of numerical features
    5. Feature selection using SelectKBest

    CRITICAL: Prevents data leakage by training on benign data only.
    """

    def __init__(
        self,
        select_k_best: int = SELECT_K_BEST_K,
        random_state: int = 42
    ):
        """
        Initialize the data processor.

        Args:
            select_k_best: Number of top features to select.
            random_state: Random state for reproducibility.
        """
        self.select_k_best = select_k_best
        self.random_state = random_state

        # Preprocessing components
        self.scaler: Optional[StandardScaler] = None
        self.selector: Optional[SelectKBest] = None
        self.feature_names: List[str] = []
        self.selected_feature_names: List[str] = []

        # Data storage
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.label_encoder: Optional[LabelEncoder] = None

        # Store classes for categorical encoding
        self.categorical_classes: Dict[str, np.ndarray] = {}

        logger.info("DataProcessor initialized")

    def load_dataset(
        self,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test datasets.

        Attempts automatic download if files don't exist.

        Args:
            train_path: Optional path to training file.
            test_path: Optional path to test file.

        Returns:
            Tuple of (training_dataframe, test_dataframe).

        Raises:
            Exception: If dataset cannot be loaded or downloaded.
        """
        logger.info("Loading NSL-KDD dataset...")

        # Use provided paths or default
        train_path = str(train_path) if train_path else str(KDD_TRAIN_PATH)
        test_path = str(test_path) if test_path else str(KDD_TEST_PATH)

        # Check if files exist, if not try to download
        from pathlib import Path

        if not Path(train_path).exists() or not Path(test_path).exists():
            logger.warning("Dataset files not found, attempting automatic download...")
            try:
                actual_train, actual_test = download_dataset_with_fallback()
                train_path = actual_train
                test_path = actual_test
            except Exception as e:
                logger.error(f"Automatic download failed: {e}")
                raise

        # Load datasets
        try:
            self.train_data = load_csv_safely(train_path, NSL_KDD_COLUMNS)
            self.test_data = load_csv_safely(test_path, NSL_KDD_COLUMNS)

            logger.info(f"Training data: {self.train_data.shape}")
            logger.info(f"Test data: {self.test_data.shape}")

            return self.train_data, self.test_data

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def preprocess(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        fit_selector: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline.

        This method performs:
        1. Separation of benign samples for training
        2. One-hot encoding of categorical features
        3. Standard scaling of all features
        4. Feature selection using SelectKBest

        CRITICAL: Training is done ONLY on benign samples to prevent
        data leakage and learn "normal" behavior.

        Args:
            train_data: Training dataframe.
            test_data: Test dataframe.
            fit_selector: Whether to fit the feature selector.

        Returns:
            Tuple of (X_train_benign, X_test, y_test, feature_names).
        """
        logger.info("Starting preprocessing pipeline...")

        # Step 1: Separate benign samples for training
        logger.info("Step 1: Separating benign training samples...")

        # Convert labels to binary (0=benign, 1=attack)
        y_train_all = convert_labels_to_binary(train_data['label'])
        y_test = convert_labels_to_binary(test_data['label'])

        # Get indices of benign samples
        benign_mask = y_train_all == 0
        X_train_benign_raw = train_data[benign_mask].drop(['label', 'difficulty'], axis=1)
        X_test_raw = test_data.drop(['label', 'difficulty'], axis=1)

        logger.info(f"Benign training samples: {len(X_train_benign_raw)}")
        logger.info(f"Total test samples: {len(X_test_raw)}")
        logger.info(f"Test set composition - Benign: {(y_test == 0).sum()}, Attack: {(y_test == 1).sum()}")

        # Step 2: One-hot encode categorical features
        logger.info("Step 2: One-hot encoding categorical features...")

        X_train_encoded, X_test_encoded, self.categorical_classes = self._encode_categorical(
            X_train_benign_raw,
            X_test_raw
        )

        self.feature_names = list(X_train_encoded.columns)
        logger.info(f"Features after encoding: {len(self.feature_names)}")

        # Step 3: Scale features
        logger.info("Step 3: Scaling features with StandardScaler...")

        X_train_scaled, X_test_scaled = self._scale_features(
            X_train_encoded.values,
            X_test_encoded.values,
            fit_scaler=fit_selector
        )

        # Step 4: Feature selection
        logger.info(f"Step 4: Selecting top {self.select_k_best} features...")

        X_train_selected, X_test_selected, self.selected_feature_names = self._select_features(
            X_train_scaled,
            X_test_scaled,
            y_train= np.zeros(len(X_train_scaled)),  # Dummy labels for chi2
            fit_selector=fit_selector
        )

        logger.info(f"Selected {len(self.selected_feature_names)} features")
        logger.info("Preprocessing pipeline completed successfully")

        return X_train_selected, X_test_selected, y_test, self.feature_names

    def _encode_categorical(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
        """
        One-hot encode categorical features.

        Args:
            X_train: Training features.
            X_test: Test features.

        Returns:
            Tuple of (encoded_train, encoded_test, categorical_classes).
        """
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()

        for col in CATEGORICAL_COLUMNS:
            if col in X_train.columns:
                # Get all unique classes from both train and test
                all_classes = pd.concat([X_train[col], X_test[col]]).unique()
                all_classes = sorted(all_classes)

                # Store for reference
                self.categorical_classes[col] = np.array(all_classes)

                # One-hot encode
                for cls in all_classes:
                    train_col_name = f"{col}_{cls}"
                    X_train_encoded[train_col_name] = (X_train[col] == cls).astype(int)

                    test_col_name = f"{col}_{cls}"
                    X_test_encoded[test_col_name] = (X_test[col] == cls).astype(int)

                # Drop original column
                X_train_encoded = X_train_encoded.drop(col, axis=1)
                X_test_encoded = X_test_encoded.drop(col, axis=1)

        return X_train_encoded, X_test_encoded, self.categorical_classes

    def _scale_features(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler.

        Args:
            X_train: Training features.
            X_test: Test features.
            fit_scaler: Whether to fit the scaler.

        Returns:
            Tuple of (scaled_train, scaled_test).
        """
        if fit_scaler:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = X_train

        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

    def _select_features(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        fit_selector: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Select top k features using SelectKBest with chi2.

        Args:
            X_train: Scaled training features.
            X_test: Scaled test features.
            y_train: Training labels.
            fit_selector: Whether to fit the selector.

        Returns:
            Tuple of (selected_train, selected_test, selected_feature_names).
        """
        # Chi2 requires non-negative values
        # Shift data to be non-negative
        X_train_positive = X_train - X_train.min() + 0.01
        X_test_positive = X_test - X_train.min() + 0.01

        if fit_selector:
            self.selector = SelectKBest(chi2, k=self.select_k_best)
            X_train_selected = self.selector.fit_transform(X_train_positive, y_train)
        else:
            X_train_selected = X_train_positive

        # Apply same selection to test data
        if self.selector is not None:
            X_test_selected = self.selector.transform(X_test_positive)

            # Get selected feature names
            selected_indices = self.selector.get_support(indices=True)
            selected_feature_names = [self.feature_names[i] for i in selected_indices]
        else:
            X_test_selected = X_test_positive
            selected_feature_names = self.feature_names[:self.select_k_best]

        return X_train_selected, X_test_selected, selected_feature_names

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores from SelectKBest.

        Returns:
            Array of importance scores for selected features.
        """
        if self.selector is None:
            raise RuntimeError("Selector not fitted. Run preprocess() first.")

        return self.selector.scores_

    def get_label_distribution(self, data: pd.DataFrame) -> Dict[str, int]:
        """
        Get distribution of labels in the dataset.

        Args:
            data: Dataframe with 'label' column.

        Returns:
            Dictionary mapping labels to counts.
        """
        return data['label'].value_counts().to_dict()

    def get_attack_types(self, data: pd.DataFrame) -> List[str]:
        """
        Get list of unique attack types (excluding 'normal').

        Args:
            data: Dataframe with 'label' column.

        Returns:
            List of unique attack type names.
        """
        attacks = data[data['label'] != 'normal']['label'].unique()
        return sorted(attacks.tolist())

    def prepare_full_test_data(self, test_data: pd.DataFrame) -> np.ndarray:
        """
        Prepare full test data for prediction (all features selected).

        This is used after training to prepare the complete test set.

        Args:
            test_data: Raw test dataframe.

        Returns:
            Preprocessed test data ready for prediction.
        """
        # Drop label and difficulty columns
        X_test = test_data.drop(['label', 'difficulty'], axis=1)

        # Encode categorical features
        for col in CATEGORICAL_COLUMNS:
            if col in X_test.columns:
                for cls in self.categorical_classes.get(col, []):
                    col_name = f"{col}_{cls}"
                    if col_name not in X_test.columns:
                        X_test[col_name] = 0
                    X_test[col_name] = X_test[col_name].astype(int)
                X_test = X_test.drop(col, axis=1)

        # Align columns with training data
        for col in self.feature_names:
            if col not in X_test.columns:
                X_test[col] = 0
        X_test = X_test[self.feature_names]

        # Scale
        X_test_scaled = self.scaler.transform(X_test.values)

        # Make non-negative for chi2-based selection
        X_test_positive = X_test_scaled - X_test_scaled.min() + 0.01

        # Select features
        X_test_selected = self.selector.transform(X_test_positive)

        return X_test_selected
